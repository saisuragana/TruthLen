from __future__ import annotations

import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import get_settings
from .detectors import DetectorUnavailable, classify
from .hive import classify_visual
from .provenance import inspect_provenance

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def media_kind(filename: str) -> str | None:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return None


def _image_analysis(path: Path, external_provider_consent: bool = False) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            image.verify()
        if width * height > 40_000_000:
            raise ValueError("This image exceeds the 40-megapixel analysis limit.")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("This image could not be decoded. Try a PNG, JPEG, or WebP file.") from exc
    settings = get_settings()
    if settings.hive_api_key and external_provider_consent:
        extension = path.suffix.lower()
        hive_supported = extension in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        try:
            if not hive_supported:
                raise DetectorUnavailable("Hive image detection accepts JPEG, PNG, WebP, or GIF; this file stays local.")
            signal = classify_visual(path, "image", settings.hive_api_key)
        except DetectorUnavailable as exc:
            signal = {"available": False, "modality": "image", "provider": "Hive cloud", "model": "Hive AI-Generated Image and Video Detection", "reason": str(exc)}
        signal["details"] = {**signal.get("details", {}), "width": width, "height": height}
        comparisons = []
        try:
            local = classify(raw, "image")
            local["provider"] = "Local model"
            comparisons.append(local)
        except DetectorUnavailable as exc:
            comparisons.append({"available": False, "modality": "image", "provider": "Local model", "model": settings.image_model_id, "reason": str(exc)})
        comparison_model = settings.image_comparison_model_id
        if comparison_model and comparison_model != settings.image_model_id:
            try:
                local = classify(raw, "image", model_override=comparison_model)
                local["provider"] = "Local comparison model"
                comparisons.append(local)
            except DetectorUnavailable as exc:
                comparisons.append({"available": False, "modality": "image", "provider": "Local comparison model", "model": comparison_model, "reason": str(exc)})
        signal["comparisons"] = comparisons
        return signal

    try:
        signal = classify(raw, "image")
        signal["details"] = {"width": width, "height": height}
        comparison_model = settings.image_comparison_model_id
        if comparison_model and comparison_model != signal.get("model"):
            try:
                signal["comparison"] = classify(raw, "image", model_override=comparison_model)
            except DetectorUnavailable as exc:
                signal["comparison"] = {"available": False, "modality": "image", "model": comparison_model, "reason": str(exc)}
        return signal
    except DetectorUnavailable as exc:
        return {"available": False, "modality": "image", "reason": str(exc), "details": {"width": width, "height": height}}


def _video_analysis(path: Path, external_provider_consent: bool = False) -> dict[str, Any]:
    settings = get_settings()
    if not settings.hive_api_key or not external_provider_consent:
        return _local_video_analysis(path)
    extension = path.suffix.lower()
    hive_supported = extension in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    try:
        if not hive_supported:
            raise DetectorUnavailable("Hive video detection accepts MP4, MOV, AVI, MKV, or WebM files.")
        signal = classify_visual(path, "video", settings.hive_api_key)
    except DetectorUnavailable as exc:
        signal = {"available": False, "modality": "video", "provider": "Hive cloud", "model": "Hive AI-Generated Image and Video Detection", "reason": str(exc)}
    try:
        local = _local_video_analysis(path)
    except (ValueError, DetectorUnavailable) as exc:
        local = {"available": False, "modality": "video", "provider": "Local frame model", "model": settings.image_model_id, "reason": str(exc)}
    local["provider"] = "Local frame model"
    signal["comparison"] = local
    return signal


def _local_video_analysis(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("Video support is missing. Install opencv-python-headless.") from exc

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("This video could not be opened. Check its codec and file integrity.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
        if frame_count <= 0:
            raise ValueError("No video frames were found in this file.")

        sample_count = min(frame_count, get_settings().max_video_frames)
        indices = sorted(set(round(i * (frame_count - 1) / max(1, sample_count - 1)) for i in range(sample_count)))
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            encoded, buffer = cv2.imencode(".jpg", frame)
            if not encoded:
                continue
            timestamp = index / fps if fps > 0 else 0.0
            try:
                signal = classify(buffer.tobytes(), "image")
                frame_result = {
                    "frame": index,
                    "timestamp_seconds": round(timestamp, 2),
                    "synthetic_score": signal["synthetic_score"],
                    "top_label": signal["top_label"],
                    "top_score": signal["top_score"],
                }
                if signal.get("manipulation_score") is not None:
                    frame_result["manipulation_score"] = signal["manipulation_score"]
                frames.append(frame_result)
            except DetectorUnavailable as exc:
                return {
                    "available": False,
                    "modality": "video",
                    "reason": str(exc),
                    "details": {"duration_seconds": round(duration, 2), "sampled_frames": len(frames)},
                }
        if not frames:
            raise ValueError("The video decoder could not extract any usable frames.")
        scores = [frame["synthetic_score"] for frame in frames]
        average = mean(scores)
        manipulation_scores = [frame["manipulation_score"] for frame in frames if "manipulation_score" in frame]
        average_manipulation = mean(manipulation_scores) if manipulation_scores else None
        variation = pstdev(scores) if len(scores) > 1 else 0.0
        highest_generated = sorted(frames, key=lambda item: item["synthetic_score"], reverse=True)[:3]
        highest_manipulation = sorted(
            (frame for frame in frames if "manipulation_score" in frame),
            key=lambda item: item["manipulation_score"],
            reverse=True,
        )[:3]
        result = {
            "available": True,
            "modality": "video",
            "model": get_settings().image_model_id,
            "synthetic_score": round(average, 4),
            "top_label": "frame-level mean",
            "top_score": round(average, 4),
            "uncertainty_note": "Frame scores vary; no temporal detector was run." if variation > 0.15 else "Frames were scored independently; no temporal detector was run.",
            "explanation": f"Analysed {len(frames)} sampled frames from a {duration:.1f}s clip using the image detector. AI-generation scores are averaged; deepfake/manipulation scores are reported separately. This does not assess motion or temporal consistency.",
            "details": {
                "duration_seconds": round(duration, 2),
                "sampled_frames": len(frames),
                "frame_score_stddev": round(variation, 4),
                "highest_ai_generated_timestamps": highest_generated,
                "highest_manipulation_timestamps": highest_manipulation,
            },
        }
        if average_manipulation is not None:
            result["manipulation_score"] = round(average_manipulation, 4)
        return result
    finally:
        capture.release()


def _audio_analysis(path: Path) -> dict[str, Any]:
    try:
        signal = classify(path.read_bytes(), "audio", path.name)
    except DetectorUnavailable as exc:
        return {"available": False, "modality": "audio", "reason": str(exc)}
    signal["explanation"] = (
        f"The audio classifier assigned {signal['synthetic_score']:.1%} to its synthetic/fake classes. "
        "No speaker identity or transcript was checked."
    )
    return signal


def _embedded_audio_analysis(path: Path) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {"available": False, "modality": "audio", "reason": "FFmpeg is unavailable, so the video's audio track could not be extracted."}
    try:
        extracted = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-t", "60", "-f", "wav", "pipe:1"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "modality": "audio", "reason": "The video's audio track could not be extracted."}
    if extracted.returncode != 0 or not extracted.stdout:
        return {"available": False, "modality": "audio", "reason": "No readable audio track was found in this video."}
    try:
        signal = classify(extracted.stdout, "audio", "embedded-audio.wav")
    except DetectorUnavailable as exc:
        return {"available": False, "modality": "audio", "reason": str(exc)}
    signal["explanation"] = (
        f"The video's first 60 seconds of audio were assigned {signal['synthetic_score']:.1%} to synthetic/fake classes. "
        "No speaker identity or transcript was checked."
    )
    return signal


def analyze_file(path: Path, original_name: str, external_provider_consent: bool = False) -> dict[str, Any]:
    kind = media_kind(original_name)
    if kind is None:
        raise ValueError("Unsupported format. Upload an image, video, or audio file listed in the README.")
    if kind == "image":
        modalities = [_image_analysis(path, external_provider_consent)]
    elif kind == "video":
        modalities = [_video_analysis(path, external_provider_consent), _embedded_audio_analysis(path)]
    else:
        modalities = [_audio_analysis(path)]

    from .fusion import fuse_scores

    expected_modalities = {"image": 1, "video": 2, "audio": 1}[kind]
    fusion = fuse_scores(modalities, expected_modalities=expected_modalities)
    comparisons = []
    for item in modalities:
        comparison = item.pop("comparison", None)
        if comparison is not None:
            comparisons.append(comparison)
        comparisons.extend(item.pop("comparisons", []))
    reasons = [
        item.get("explanation") or item.get("reason")
        for item in modalities
        if item.get("explanation") or item.get("reason")
    ]
    has_manipulation_score = any(item.get("manipulation_score") is not None for item in modalities)
    provenance = inspect_provenance(path)
    if has_manipulation_score:
        reasons.append(
            "Caution: manual checks found high deepfake/manipulation scores on camera photos and a camera video confirmed as genuine. Treat this class as experimental; it is not included in the AI-generated score."
        )
    plain_summary = _plain_language_summary(kind, fusion, modalities, comparisons, provenance)
    result = {
        "filename": original_name,
        "media_type": kind,
        "created_at": None,
        "fusion": fusion,
        "modalities": modalities,
        "provenance": provenance,
        "plain_summary": plain_summary,
        "explanations": reasons + [fusion["explanation"]],
        "notice": "Automated screening aid only. Scores are not proof of authenticity and have not been calibrated on TruthLen evaluation data."
        + (" Manual checks found high deepfake/manipulation scores on camera photos and video confirmed as genuine." if has_manipulation_score else ""),
    }
    if comparisons:
        result["comparisons"] = comparisons
    return result


def _plain_language_summary(
    kind: str,
    fusion: dict[str, Any],
    modalities: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> str:
    """Explain the structured output without generating or inventing evidence."""
    usable = [
        item for item in modalities
        if item.get("available") is True and isinstance(item.get("synthetic_score"), (int, float))
    ]
    score = fusion.get("synthetic_score")
    if not usable or score is None:
        summary = "TruthLen could not produce a usable detector score, so it cannot assess this file."
    elif len(usable) == 1:
        item = usable[0]
        score_label = "AI-generated" if item.get("modality") in {"image", "video"} else "synthetic/fake audio"
        summary = (
            f"TruthLen cannot determine whether this {kind} is AI-generated or camera-original. "
            f"Only one detector ({item.get('modality', 'media')}) produced a score; it assigned {float(item['synthetic_score']):.1%} "
            f"to its {score_label} class, but the overall result stays inconclusive because one "
            "uncalibrated detector is not enough to confirm either origin."
        )
    elif fusion.get("verdict") == "likely_synthetic":
        summary = (
            "The available checks leaned toward generated-media signals. This is a screening result, "
            "not proof that the file was made or altered by AI."
        )
    elif fusion.get("verdict") == "likely_authentic":
        summary = (
            "The available checks found fewer generated-media signals. That does not prove the file "
            "came directly from a camera or establish that it is authentic."
        )
    else:
        summary = "The available checks do not agree strongly enough to give a clear overall result."

    if comparisons:
        usable_comparisons = [
            item for item in comparisons
            if item.get("available") is True and isinstance(item.get("synthetic_score"), (int, float))
        ]
        primary = next((item for item in modalities if item.get("modality") == kind and item.get("available")), None)
        usable_comparisons = [item for item in usable_comparisons if item.get("modality") == kind]
        if primary and usable_comparisons:
            comparison = usable_comparisons[0]
            if abs(float(primary["synthetic_score"]) - float(comparison["synthetic_score"])) >= 0.3:
                summary += (
                    f" The independent {kind} comparison disagreed ({float(primary['synthetic_score']):.1%} "
                    f"versus {float(comparison['synthetic_score']):.1%}); neither model is decisive."
                )

    summary += f" Provenance check: {provenance.get('summary', 'not available')}"
    if kind in {"image", "video"}:
        summary += (
            " TruthLen has also observed false alarms on camera photos and video during manual checks, so treat "
            "a high model score cautiously."
        )
    return summary
