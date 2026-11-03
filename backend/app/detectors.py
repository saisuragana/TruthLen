from __future__ import annotations

import logging
import os
from functools import lru_cache
from io import BytesIO
from typing import Any

from PIL import Image

from .config import get_settings

logger = logging.getLogger("truthlen.detectors")
PE_TINY_MODEL_ID = "Deckard618/imdet/pe_tiny"


class DetectorUnavailable(RuntimeError):
    """Raised when a configured pretrained detector cannot return a usable score."""


@lru_cache(maxsize=4)
def _local_pipeline(task: str, model_id: str) -> Any:
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise DetectorUnavailable(
            "Local model packages are missing. Install backend/requirements-models.txt."
        ) from exc
    except Exception as exc:
        logger.exception("Could not import the local model runtime")
        raise DetectorUnavailable(
            "The local model runtime could not start. Check the Transformers and PyTorch installation."
        ) from exc
    try:
        requested_device = os.getenv("TRUTHLEN_DEVICE", "auto").lower()
        if requested_device == "auto":
            try:
                import torch

                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1
        elif requested_device == "cpu":
            device = -1
        elif requested_device == "cuda":
            device = 0
        elif requested_device.startswith("cuda:"):
            device = int(requested_device.split(":", 1)[1])
        else:
            device = -1
        return pipeline(task, model=model_id, device=device)
    except Exception as exc:
        logger.exception("Could not load pretrained model %s", model_id)
        raise DetectorUnavailable(
            "The configured model could not be loaded. Check model access, internet, and backend logs."
        ) from exc


def _predict(raw: bytes, modality: str, filename_hint: str | None = None) -> tuple[list[dict[str, Any]], str]:
    return _predict_with_model(raw, modality, filename_hint)


def _predict_local_audio(raw: bytes, model_id: str, filename_hint: str | None = None) -> Any:
    """Run local audio inference from a closed temporary file.

    Transformers reopens the path itself. Closing the writer before inference is
    required on Windows, where an open NamedTemporaryFile cannot be reopened.
    """
    import tempfile
    from pathlib import Path

    suffix = Path(filename_hint or "audio.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="truthlen-audio-") as temp_dir:
        temp_path = Path(temp_dir) / f"input{suffix}"
        temp_path.write_bytes(raw)
        pipeline = _local_pipeline("audio-classification", model_id)
        return pipeline(str(temp_path), top_k=10)


def _predict_with_model(
    raw: bytes,
    modality: str,
    filename_hint: str | None = None,
    model_override: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    settings = get_settings()
    endpoint = (settings.image_endpoint if modality == "image" else settings.audio_endpoint) if model_override is None else ""
    configured_model_id = settings.image_model_id if modality == "image" else settings.audio_model_id
    model_id = model_override or configured_model_id
    model = model_id
    task = "image-classification" if modality == "image" else "audio-classification"

    if endpoint or settings.inference_mode == "providers":
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise DetectorUnavailable("Install huggingface_hub to use hosted inference.") from exc
        model = endpoint or model_id
        try:
            client = InferenceClient(model=model, token=settings.hf_token or None, timeout=90)
            if modality == "image":
                output = client.image_classification(raw, top_k=10)
            else:
                output = client.audio_classification(raw, top_k=10)
            return _normalize_output(output), model
        except Exception as exc:
            logger.exception("Hosted inference failed for %s model %s", modality, model)
            raise DetectorUnavailable(
                "Hosted inference failed. Confirm the endpoint, provider support, token, and backend logs."
            ) from exc

    try:
        if modality == "image":
            image = Image.open(BytesIO(raw)).convert("RGB")
            output = _local_pipeline(task, model)(image)
        else:
            # The pipeline opens the path itself. _predict_local_audio closes the
            # writer before passing the file path, which is required on Windows.
            output = _predict_local_audio(raw, model, filename_hint)
        return _normalize_output(output), model
    except DetectorUnavailable:
        raise
    except Exception as exc:
        logger.exception("Local inference failed for %s model %s", modality, model)
        raise DetectorUnavailable(
            "Local inference failed. Confirm supported media format, model files, and backend logs."
        ) from exc


@lru_cache(maxsize=2)
def _onnx_session(model_path: str) -> Any:
    import onnxruntime

    return onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _predict_pe_tiny(raw: bytes) -> tuple[list[dict[str, Any]], str]:
    """Run the small binary ONNX comparison model locally on CPU."""
    try:
        import numpy as np
        from huggingface_hub import hf_hub_download
        from PIL import Image
        import onnxruntime  # noqa: F401 - surface an actionable missing-package message
    except ImportError as exc:
        raise DetectorUnavailable(
            "The local comparison model needs onnxruntime. Install backend/requirements-models.txt."
        ) from exc

    settings = get_settings()
    model_dir = settings.data_dir / "models" / "deckard_imdet"
    try:
        model_path = hf_hub_download(
            repo_id="Deckard618/imdet",
            filename="pe_tiny/model.onnx",
            local_dir=str(model_dir),
        )
        session = _onnx_session(str(model_path))
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left = (resized.width - 256) // 2
        top = (resized.height - 256) // 2
        cropped = resized.crop((left, top, left + 256, top + 256))
        pixels = np.asarray(cropped, dtype=np.float32) / 255.0
        tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
        input_name = session.get_inputs()[0].name
        logits = np.asarray(session.run(None, {input_name: tensor})[0]).reshape(-1)
        if logits.size != 2:
            raise ValueError("The comparison model did not return two class scores.")
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        # The model card specifies output index 0 = fake, 1 = real.
        output = [
            {"label": "fake", "score": float(probabilities[0])},
            {"label": "real", "score": float(probabilities[1])},
        ]
        return output, PE_TINY_MODEL_ID
    except DetectorUnavailable:
        raise
    except Exception as exc:
        logger.exception("Local comparison model failed")
        raise DetectorUnavailable(
            "The local comparison model could not run. Check the model download and backend logs."
        ) from exc


def _normalize_output(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, dict):
        output = [output]
    normalized = []
    for item in output or []:
        if isinstance(item, dict) and "label" in item and "score" in item:
            try:
                normalized.append({"label": str(item["label"]), "score": float(item["score"])})
            except (TypeError, ValueError):
                continue
    return sorted(normalized, key=lambda item: item["score"], reverse=True)


def _class_probabilities(
    output: list[dict[str, Any]], modality: str, model_id: str
) -> tuple[float, float | None]:
    if not output:
        raise DetectorUnavailable("The detector returned no class scores.")
    generated_total = 0.0
    manipulation_total = 0.0
    matched = False
    manipulation_matched = False
    for item in output:
        label = item["label"].strip().lower().replace(" ", "_").replace("-", "_")
        score = item["score"]
        # A negative class can contain the same keywords as a positive class
        # (for example, "not_ai_generated"). Recognize it before substring
        # matching so it is not accidentally counted as generated media.
        negated_class = label.startswith(("not_", "non_", "no_")) and any(
            token in label for token in ("ai", "fake", "synthetic", "generated")
        )
        if negated_class:
            matched = True
            continue
        if modality == "image" and (
            "deepfake" in label or "face_swap" in label or "face_manipulat" in label
        ):
            # Keep face manipulation distinct from whole-image AI generation.
            manipulation_total += score
            manipulation_matched = True
            matched = True
        elif label == "ai" or any(token in label for token in ("synthetic", "generated", "ai_generated")):
            generated_total += score
            matched = True
        elif "fake" in label:
            # Generic fake labels remain a synthetic signal (common for audio models).
            generated_total += score
            matched = True
        elif any(token in label for token in ("real", "authentic", "genuine", "natural", "human")):
            matched = True
        elif label.startswith("label_"):
            # Label order comes from the published model cards for these defaults.
            if modality == "image" and model_id == "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2":
                class_index = label.split("_", 1)[1]
                if class_index == "0":
                    generated_total += score
                    matched = True
                elif class_index == "1":
                    manipulation_total += score
                    manipulation_matched = True
                    matched = True
                elif class_index == "2":
                    matched = True
            elif modality == "audio" and model_id == "MelodyMachine/Deepfake-audio-detection-V2":
                class_index = label.split("_", 1)[1]
                if class_index == "0":
                    generated_total += score
                    matched = True
                elif class_index == "1":
                    matched = True
    if not matched:
        raise DetectorUnavailable("Class labels could not be mapped to generated or real media.")
    generated_score = max(0.0, min(1.0, generated_total))
    manipulation_score = max(0.0, min(1.0, manipulation_total)) if manipulation_matched else None
    return generated_score, manipulation_score


def _synthetic_probability(output: list[dict[str, Any]], modality: str, model_id: str) -> float:
    """Return only the AI-generated/generic-fake class score, excluding deepfakes."""
    return _class_probabilities(output, modality, model_id)[0]


def classify(
    raw: bytes,
    modality: str,
    filename_hint: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    try:
        if model_override is None:
            output, model_id = _predict(raw, modality, filename_hint)
        elif model_override == PE_TINY_MODEL_ID:
            output, model_id = _predict_pe_tiny(raw)
        else:
            output, model_id = _predict_with_model(raw, modality, filename_hint, model_override)
        score, manipulation_score = _class_probabilities(output, modality, model_id)
    except DetectorUnavailable:
        raise
    except Exception as exc:
        logger.exception("Unexpected %s detector failure", modality)
        raise DetectorUnavailable(
            f"Unexpected detector error ({type(exc).__name__}). Check the backend logs and model installation."
        ) from exc
    top = output[0] if output else {"label": "unknown", "score": 0}
    signal_name = "AI-generated" if modality == "image" else "synthetic/fake audio"
    result = {
        "available": True,
        "modality": modality,
        "model": model_id,
        "synthetic_score": round(score, 4),
        "top_label": top["label"],
        "top_score": round(top["score"], 4),
        "class_scores": [{"label": item["label"], "score": round(item["score"], 4)} for item in output],
        "explanation": f"The {signal_name} class score was {score:.1%}. {top['label']} was the highest-scoring class ({top['score']:.1%}). This is a model output, not forensic proof.",
    }
    if manipulation_score is not None:
        result["manipulation_score"] = round(manipulation_score, 4)
        result["explanation"] += f" The separate deepfake/manipulation class score was {manipulation_score:.1%}."
    return result
