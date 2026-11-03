from __future__ import annotations

from statistics import mean, pstdev
from typing import Any


def fuse_scores(signals: list[dict[str, Any]], expected_modalities: int = 3) -> dict[str, Any]:
    """Combine detector scores with an explicit coverage and disagreement penalty.

    The returned confidence is a heuristic evidence-strength estimate. It is not a
    calibrated probability and should not be presented as one.
    """
    usable = [
        item for item in signals
        if item.get("available") is True
        and isinstance(item.get("synthetic_score"), (int, float))
        and 0 <= float(item["synthetic_score"]) <= 1
    ]
    if not usable:
        return {
            "synthetic_score": None,
            "verdict": "unavailable",
            "confidence": 0.0,
            "uncertainty": 1.0,
            "coverage": 0.0,
            "explanation": "No configured detector produced a usable generated-media score for this file.",
        }

    scores = [float(item["synthetic_score"]) for item in usable]
    score = mean(scores)
    disagreement = min(1.0, pstdev(scores) * 2) if len(scores) > 1 else 0.0
    margin = abs(score - 0.5) * 2
    coverage = min(1.0, len(usable) / max(1, expected_modalities))
    # A lone detector is weak evidence even when its class score is extreme.
    # Keep that visible in the evidence index while reporting modality coverage
    # against only the media checks that apply to this upload.
    detector_consensus = 1 / 3 if len(usable) == 1 else 1.0
    confidence = max(0.0, min(1.0, margin * coverage * (1 - disagreement) * detector_consensus))
    uncertainty = 1 - confidence

    if len(usable) < 2:
        verdict = "inconclusive"
    elif score >= 0.65:
        verdict = "likely_synthetic"
    elif score <= 0.35:
        verdict = "likely_authentic"
    else:
        verdict = "inconclusive"

    names = ", ".join(item.get("modality", "media") for item in usable)
    caveats = []
    if coverage < 1:
        caveats.append(f"only {len(usable)} of {expected_modalities} modalities contributed")
    if len(usable) == 1:
        caveats.append("one detector alone is insufficient for an overall verdict")
    if disagreement >= 0.35:
        caveats.append("available detectors disagree")
    caveat_text = f"; {', and '.join(caveats)}" if caveats else ""
    return {
        "synthetic_score": round(score, 4),
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "uncertainty": round(uncertainty, 4),
        "coverage": round(coverage, 4),
        "disagreement": round(disagreement, 4),
        "explanation": f"Combined {names} generated-media scores{caveat_text}. Evidence strength is a heuristic index weighted by score margin, applicable-modality coverage, detector consensus, and disagreement; it is not a calibrated probability.",
    }
