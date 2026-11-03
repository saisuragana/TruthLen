#!/usr/bin/env python3
"""Local Image Detector Evaluation Script for TruthLen.

Reads a CSV listing image paths and known labels ('real' or 'ai'), runs the
existing local image detectors (primary and/or local comparison), and reports:
  - Confusion matrix (TP, TN, FP, FN)
  - Key classification metrics (Accuracy, Precision, Recall, Specificity, FPR, FNR, F1)
  - Detailed lists of False Positives and False Negatives
  - Optional JSON and CSV export

Usage:
  python scripts/evaluate_images.py path/to/dataset.csv
  python scripts/evaluate_images.py path/to/dataset.csv --threshold 0.5 --detector all
  python scripts/evaluate_images.py path/to/dataset.csv --output eval_summary.json --output-csv eval_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure backend package is in python search path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import get_settings
from app.detectors import PE_TINY_MODEL_ID, DetectorUnavailable, classify
from app.evaluation import (
    DetectorEvaluation,
    compute_metrics,
    evaluate_detector,
    load_labeled_csv,
)


def _print_separator(char: str = "=", width: int = 76) -> None:
    print(char * width)


def _format_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def _format_rate(val: float) -> str:
    return f"{val:.4f}"


def format_evaluation_report(eval_res: DetectorEvaluation) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"Detector: {eval_res.detector_name}")
    lines.append(f"Model ID: {eval_res.model_id}")
    lines.append(f"Decision Threshold: synthetic_score >= {eval_res.threshold:.2f} -> AI")
    lines.append("-" * 76)

    m = eval_res.metrics
    lines.append("Confusion Matrix:")
    lines.append(f"  True Positives  (AI correctly flagged):     {m.tp:5d}")
    lines.append(f"  True Negatives  (Real correctly identified): {m.tn:5d}")
    lines.append(f"  False Positives (Real misclassified as AI):  {m.fp:5d}  <-- False Alarms")
    lines.append(f"  False Negatives (AI missed as Real):         {m.fn:5d}  <-- Missed Detections")
    lines.append(f"  Total Evaluated:                             {m.total:5d}")
    lines.append("")

    lines.append("Performance Metrics:")
    lines.append(f"  Accuracy:                 {_format_pct(m.accuracy):>8s}  ({m.tp + m.tn}/{m.total})")
    lines.append(f"  Precision (PPV):          {_format_pct(m.precision):>8s}  (TP / [TP + FP])")
    lines.append(f"  Recall / TPR:             {_format_pct(m.recall):>8s}  (TP / [TP + FN])")
    lines.append(f"  Specificity / TNR:        {_format_pct(m.specificity):>8s}  (TN / [TN + FP])")
    lines.append(f"  False Positive Rate (FPR):{_format_pct(m.false_positive_rate):>8s}  (FP / [FP + TN])")
    lines.append(f"  False Negative Rate (FNR):{_format_pct(m.false_negative_rate):>8s}  (FN / [FN + TP])")
    lines.append(f"  F1 Score:                 {_format_rate(m.f1_score):>8s}")
    lines.append("")

    # Score distributions
    real_scores = [s.predicted_score for s in eval_res.all_samples if s.true_label == "real"]
    ai_scores = [s.predicted_score for s in eval_res.all_samples if s.true_label == "ai"]
    if real_scores:
        lines.append(f"  Avg score on REAL images: {sum(real_scores) / len(real_scores):.4f} (n={len(real_scores)})")
    if ai_scores:
        lines.append(f"  Avg score on AI images:   {sum(ai_scores) / len(ai_scores):.4f} (n={len(ai_scores)})")
    lines.append("")

    # False Positives Breakdown
    if eval_res.false_positives:
        lines.append(f"False Positives ({len(eval_res.false_positives)} real images misclassified as AI):")
        for idx, fp in enumerate(eval_res.false_positives, start=1):
            manip_txt = f", deepfake={fp.manipulation_score:.1%}" if fp.manipulation_score is not None else ""
            lines.append(
                f"  [{idx:2d}] {fp.path} -> synthetic_score={fp.predicted_score:.1%}{manip_txt} (top: {fp.top_label} {fp.top_score:.1%})"
            )
        lines.append("")
    else:
        lines.append("False Positives: None (0 real images misclassified as AI)")
        lines.append("")

    # False Negatives Breakdown
    if eval_res.false_negatives:
        lines.append(f"False Negatives ({len(eval_res.false_negatives)} AI images missed as real):")
        for idx, fn in enumerate(eval_res.false_negatives, start=1):
            manip_txt = f", deepfake={fn.manipulation_score:.1%}" if fn.manipulation_score is not None else ""
            lines.append(
                f"  [{idx:2d}] {fn.path} -> synthetic_score={fn.predicted_score:.1%}{manip_txt} (top: {fn.top_label} {fn.top_score:.1%})"
            )
        lines.append("")
    else:
        lines.append("False Negatives: None (0 AI images missed as real)")
        lines.append("")

    # Errors
    if eval_res.errors:
        lines.append(f"Unprocessed / Errored Images ({len(eval_res.errors)}):")
        for err in eval_res.errors:
            lines.append(f"  - {err.get('path')}: {err.get('error')}")
        lines.append("")

    return "\n".join(lines)


def export_predictions_csv(csv_output_path: Path, results: dict[str, DetectorEvaluation]) -> None:
    """Export per-image prediction details across all evaluated detectors to a CSV file."""
    # Group samples by image path
    images: dict[str, dict[str, Any]] = {}
    detector_names = list(results.keys())

    for det_name, det_eval in results.items():
        for s in det_eval.all_samples:
            if s.path not in images:
                images[s.path] = {"path": s.path, "true_label": s.true_label}
            images[s.path][f"{det_name}_score"] = round(s.predicted_score, 4)
            images[s.path][f"{det_name}_pred"] = s.predicted_label
            images[s.path][f"{det_name}_error"] = s.error_type or "OK"

    fieldnames = ["path", "true_label"]
    for det_name in detector_names:
        fieldnames.extend([f"{det_name}_score", f"{det_name}_pred", f"{det_name}_error"])

    with open(csv_output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in images.values():
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TruthLen local image detectors against a labeled CSV dataset and report metrics."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to CSV file with image paths and labels ('real' or 'ai').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying synthetic_score as 'ai' (default: 0.5).",
    )
    parser.add_argument(
        "--detector",
        choices=["all", "primary", "comparison"],
        default="all",
        help="Which local detector to run: 'primary', 'comparison', or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override TRUTHLEN_DEVICE for this evaluation run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save evaluation results in JSON format.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save per-image predictions to CSV.",
    )

    args = parser.parse_args()

    if args.device:
        os.environ["TRUTHLEN_DEVICE"] = args.device

    # Ensure local inference only
    os.environ["TRUTHLEN_INFERENCE_MODE"] = "local"
    # Never call cloud endpoints
    os.environ["TRUTHLEN_IMAGE_ENDPOINT"] = ""

    settings = get_settings()

    _print_separator("=")
    print(" TruthLen Local Image Detector Evaluation")
    _print_separator("=")
    print(f"Dataset CSV:          {args.csv_path}")
    print(f"Classification Rule:  synthetic_score >= {args.threshold:.2f} => 'ai', else => 'real'")
    print(f"Primary Model:        {settings.image_model_id}")
    print(f"Comparison Model:     {settings.image_comparison_model_id or PE_TINY_MODEL_ID}")

    try:
        records = load_labeled_csv(args.csv_path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset CSV: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("\n[ERROR] No valid rows found in the provided CSV file.", file=sys.stderr)
        return 1

    real_count = sum(1 for _, label in records if label == "real")
    ai_count = sum(1 for _, label in records if label == "ai")
    print(f"Loaded Samples:       {len(records)} total ({real_count} real, {ai_count} AI)")
    _print_separator("=")

    detectors_to_run: list[tuple[str, str, Any]] = []

    if args.detector in {"all", "primary"}:
        def primary_classify(raw: bytes) -> dict[str, Any]:
            return classify(raw, modality="image", model_override=None)

        detectors_to_run.append(("Primary Image Detector", settings.image_model_id, primary_classify))

    if args.detector in {"all", "comparison"}:
        comp_model_id = settings.image_comparison_model_id or PE_TINY_MODEL_ID

        def comparison_classify(raw: bytes) -> dict[str, Any]:
            return classify(raw, modality="image", model_override=comp_model_id)

        detectors_to_run.append(("Local Comparison Detector", comp_model_id, comparison_classify))

    results: dict[str, DetectorEvaluation] = {}

    for det_name, model_id, fn in detectors_to_run:
        print(f"\nEvaluating: {det_name} ({model_id})...")
        eval_result = evaluate_detector(
            detector_name=det_name,
            model_id=model_id,
            classify_fn=fn,
            records=records,
            threshold=args.threshold,
        )
        results[det_name] = eval_result
        print(format_evaluation_report(eval_result))
        _print_separator("-")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "dataset_csv": str(args.csv_path),
            "threshold": args.threshold,
            "detectors": {name: res.to_dict() for name, res in results.items()},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nSaved JSON evaluation summary to: {args.output}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        export_predictions_csv(args.output_csv, results)
        print(f"Saved predictions CSV to:          {args.output_csv}")

    _print_separator("=")
    print(" Evaluation Completed.")
    _print_separator("=")

    return 0


if __name__ == "__main__":
    sys.exit(main())

