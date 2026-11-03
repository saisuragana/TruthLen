from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger("truthlen.evaluation")

LABEL_MAPPING_REAL = {
    "real",
    "authentic",
    "genuine",
    "camera",
    "natural",
    "human",
    "photo",
    "0",
    "false",
}
LABEL_MAPPING_AI = {
    "ai",
    "fake",
    "synthetic",
    "generated",
    "ai_generated",
    "deepfake",
    "1",
    "true",
}


def normalize_label(label: str) -> str:
    """Normalize ground-truth label string to 'real' or 'ai'.

    Accepts common aliases:
    - Real: 'real', 'authentic', 'genuine', 'camera', 'natural', 'human', 'photo', '0', 'false'
    - AI: 'ai', 'fake', 'synthetic', 'generated', 'ai_generated', 'deepfake', '1', 'true'
    """
    clean = str(label).strip().lower()
    if clean in LABEL_MAPPING_REAL:
        return "real"
    if clean in LABEL_MAPPING_AI:
        return "ai"
    raise ValueError(
        f"Unrecognized label '{label}'. Expected a real or AI label (e.g. 'real', 'ai', 'synthetic')."
    )


@dataclass
class BinaryMetrics:
    total: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
        }


@dataclass
class EvaluationSample:
    path: str
    true_label: str  # 'real' or 'ai'
    predicted_score: float  # synthetic_score in [0.0, 1.0]
    predicted_label: str  # 'real' or 'ai'
    is_correct: bool
    error_type: str | None  # None, 'FP', or 'FN'
    model_id: str
    manipulation_score: float | None = None
    top_label: str = ""
    top_score: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectorEvaluation:
    detector_name: str
    model_id: str
    threshold: float
    metrics: BinaryMetrics
    false_positives: list[EvaluationSample] = field(default_factory=list)
    false_negatives: list[EvaluationSample] = field(default_factory=list)
    all_samples: list[EvaluationSample] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "metrics": self.metrics.to_dict(),
            "false_positives": [sample.to_dict() for sample in self.false_positives],
            "false_negatives": [sample.to_dict() for sample in self.false_negatives],
            "sample_count": len(self.all_samples),
            "error_count": len(self.errors),
            "errors": self.errors,
        }


def classify_score(score: float, threshold: float = 0.5) -> str:
    """Map a synthetic score in [0, 1] to 'ai' or 'real' using a decision threshold."""
    return "ai" if score >= threshold else "real"


def determine_error_type(true_label: str, predicted_label: str) -> str | None:
    """Return 'FP' for false positive (real flagged as AI), 'FN' for false negative (AI missed), or None."""
    if true_label == "real" and predicted_label == "ai":
        return "FP"
    if true_label == "ai" and predicted_label == "real":
        return "FN"
    return None


def compute_metrics(samples: Sequence[EvaluationSample]) -> BinaryMetrics:
    """Compute binary classification metrics with zero-division protection.

    Positive class: 'ai'
    Negative class: 'real'
    """
    total = len(samples)
    tp = sum(1 for s in samples if s.true_label == "ai" and s.predicted_label == "ai")
    tn = sum(1 for s in samples if s.true_label == "real" and s.predicted_label == "real")
    fp = sum(1 for s in samples if s.true_label == "real" and s.predicted_label == "ai")
    fn = sum(1 for s in samples if s.true_label == "ai" and s.predicted_label == "real")

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return BinaryMetrics(
        total=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
    )


def load_labeled_csv(csv_path: str | Path) -> list[tuple[Path, str]]:
    """Parse a CSV containing image paths and known labels ('real' or 'ai').

    Supports:
    - Named header columns: (path/image_path/file/filename) and (label/ground_truth/class)
    - Unlabeled 2-column CSV (col 0 = path, col 1 = label)
    - Resolves relative image paths relative to the CSV's parent directory, falling back to CWD.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation CSV file not found: {path}")

    records: list[tuple[Path, str]] = []
    csv_dir = path.parent

    path_headers = {"path", "image_path", "image", "filepath", "file", "filename", "img"}
    label_headers = {"label", "ground_truth", "true_label", "class", "actual", "target", "category"}

    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        return records

    first_row = [cell.strip().lower() for cell in rows[0]]
    has_header = False
    path_idx = 0
    label_idx = 1

    for idx, col in enumerate(first_row):
        if col in path_headers:
            path_idx = idx
            has_header = True
        elif col in label_headers:
            label_idx = idx
            has_header = True

    data_rows = rows[1:] if has_header else rows

    for row_num, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) <= max(path_idx, label_idx):
            logger.warning("Skipping row %d in %s: insufficient columns (%s)", row_num, path.name, row)
            continue

        raw_path = row[path_idx].strip()
        raw_label = row[label_idx].strip()

        if not raw_path or not raw_label:
            continue

        try:
            normalized_label = normalize_label(raw_label)
        except ValueError as exc:
            logger.warning("Skipping row %d in %s: %s", row_num, path.name, exc)
            continue

        candidate_path = Path(raw_path)
        if candidate_path.is_absolute():
            resolved_path = candidate_path
        else:
            relative_to_csv = csv_dir / candidate_path
            resolved_path = relative_to_csv if relative_to_csv.exists() else Path.cwd() / candidate_path

        records.append((resolved_path, normalized_label))

    return records


def evaluate_detector(
    detector_name: str,
    model_id: str,
    classify_fn: Callable[[bytes], dict[str, Any]],
    records: list[tuple[Path, str]],
    threshold: float = 0.5,
) -> DetectorEvaluation:
    """Evaluate one detector function against a set of labeled image paths."""
    samples: list[EvaluationSample] = []
    false_positives: list[EvaluationSample] = []
    false_negatives: list[EvaluationSample] = []
    errors: list[dict[str, str]] = []

    for img_path, true_label in records:
        if not img_path.is_file():
            errors.append({
                "path": str(img_path),
                "true_label": true_label,
                "error": "File does not exist or is not a readable file.",
            })
            continue

        try:
            raw_bytes = img_path.read_bytes()
            result = classify_fn(raw_bytes)
        except Exception as exc:
            logger.warning("Detector error on %s: %s", img_path.name, exc)
            errors.append({
                "path": str(img_path),
                "true_label": true_label,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        score = float(result.get("synthetic_score", 0.0))
        predicted_label = classify_score(score, threshold=threshold)
        is_correct = (predicted_label == true_label)
        error_type = determine_error_type(true_label, predicted_label)

        sample = EvaluationSample(
            path=str(img_path),
            true_label=true_label,
            predicted_score=score,
            predicted_label=predicted_label,
            is_correct=is_correct,
            error_type=error_type,
            model_id=result.get("model", model_id),
            manipulation_score=result.get("manipulation_score"),
            top_label=result.get("top_label", ""),
            top_score=result.get("top_score", 0.0),
            explanation=result.get("explanation", ""),
        )
        samples.append(sample)

        if error_type == "FP":
            false_positives.append(sample)
        elif error_type == "FN":
            false_negatives.append(sample)

    metrics = compute_metrics(samples)

    return DetectorEvaluation(
        detector_name=detector_name,
        model_id=model_id,
        threshold=threshold,
        metrics=metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
        all_samples=samples,
        errors=errors,
    )

