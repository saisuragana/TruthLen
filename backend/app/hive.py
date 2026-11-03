from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

import requests

from .detectors import DetectorUnavailable

HIVE_SYNC_URL = "https://api.hivemoderation.com/api/v2/task/sync"
HIVE_MODEL = "Hive AI-Generated Image and Video Detection"


def classify_visual(path: Path, modality: str, api_key: str) -> dict[str, Any]:
    """Send a visual file to Hive and normalize its AI-generation head.

    This function is only called when the user configured a Hive key and
    accepted external processing in the upload form.
    """
    try:
        with path.open("rb") as media:
            response = requests.post(
                HIVE_SYNC_URL,
                headers={"Authorization": f"Token {api_key}", "Accept": "application/json"},
                data={
                    "user_id": "truthlen-user",
                    "post_id": uuid4().hex,
                    "models": '["ai_generated_media"]',
                },
                files={"media": (path.name, media)},
                timeout=(15, 240 if modality == "video" else 90),
            )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise DetectorUnavailable("Hive took too long to analyze this file. Try a shorter video or try again later.") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise DetectorUnavailable(
            f"Hive rejected the request (HTTP {status}). Check that the API key is active and includes AI-generated media detection."
        ) from exc
    except (requests.RequestException, ValueError) as exc:
        raise DetectorUnavailable("Could not reach Hive or read its response. Check the network and Hive API status.") from exc

    outputs = _extract_outputs(payload)
    frame_scores: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        classes = output.get("classes")
        if not isinstance(classes, list):
            continue
        scores = {
            entry.get("class"): _valid_score(entry.get("score"))
            for entry in classes
            if isinstance(entry, dict) and isinstance(entry.get("class"), str)
        }
        generated = scores.get("ai_generated")
        if generated is None:
            continue
        frame_scores.append({
            "timestamp_seconds": _valid_score(output.get("time")) or 0.0,
            "synthetic_score": generated,
            "manipulation_score": scores.get("deepfake"),
            "not_ai_generated": scores.get("not_ai_generated"),
            "classes": scores,
        })

    if not frame_scores:
        raise DetectorUnavailable("Hive returned no AI-generation score. Check that the key has this detector enabled.")

    synthetic = mean(item["synthetic_score"] for item in frame_scores)
    deepfake_values = [item["manipulation_score"] for item in frame_scores if item["manipulation_score"] is not None]
    top_sources: dict[str, float] = {}
    ignored = {"ai_generated", "not_ai_generated", "deepfake", "inconclusive", "inconclusive_video", "none"}
    for item in frame_scores:
        for name, score in item["classes"].items():
            if name not in ignored and score is not None:
                top_sources[name] = max(score, top_sources.get(name, 0.0))
    top_source = max(top_sources, key=top_sources.get) if top_sources else None
    result: dict[str, Any] = {
        "available": True,
        "modality": modality,
        "provider": "Hive cloud",
        "model": HIVE_MODEL,
        "synthetic_score": round(synthetic, 6),
        "top_label": "ai_generated" if synthetic >= 0.5 else "not_ai_generated",
        "top_score": round(max(synthetic, 1 - synthetic), 6),
        "class_scores": [
            {"label": "AI-generated", "score": round(synthetic, 6)},
            {"label": "Not AI-generated", "score": round(1 - synthetic, 6)},
        ],
        "explanation": (
            f"Hive's AI-generation detector assigned {synthetic:.1%} to the AI-generated class. "
            "This is an uncalibrated provider score, not proof of origin."
        ),
    }
    if top_source:
        result["class_scores"].append({"label": f"Highest generator-source class: {top_source}", "score": round(top_sources[top_source], 6)})
    if deepfake_values:
        result["manipulation_score"] = round(mean(deepfake_values), 6)
    if top_source:
        result["details"] = {"highest_generator_source": top_source, "source_score": round(top_sources[top_source], 6)}
    if modality == "video":
        highest = sorted(frame_scores, key=lambda item: item["synthetic_score"], reverse=True)[:3]
        result["details"] = {
            **result.get("details", {}),
            "sampled_frames": len(frame_scores),
            "highest_ai_generated_timestamps": [
                {"timestamp_seconds": round(item["timestamp_seconds"], 2), "synthetic_score": round(item["synthetic_score"], 6)}
                for item in highest
            ],
        }
        result["uncertainty_note"] = "Hive scored video frames; the reported average does not assess motion or temporal consistency."
        result["explanation"] = (
            f"Hive scored {len(frame_scores)} video frame(s); their AI-generation scores average {synthetic:.1%}. "
            "This does not assess motion or temporal consistency."
        )
    return result


def _extract_outputs(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    outputs = payload.get("output")
    if isinstance(outputs, list):
        return [item for item in outputs if isinstance(item, dict)]
    status = payload.get("status")
    if isinstance(status, list):
        found = []
        for entry in status:
            if not isinstance(entry, dict):
                continue
            response = entry.get("response")
            if isinstance(response, dict) and isinstance(response.get("output"), list):
                found.extend(item for item in response["output"] if isinstance(item, dict))
        return found
    return []


def _valid_score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None
