import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from backend.app.evaluation import (
    BinaryMetrics,
    EvaluationSample,
    classify_score,
    compute_metrics,
    determine_error_type,
    evaluate_detector,
    load_labeled_csv,
    normalize_label,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_normalize_label_variants(self):
        # Real labels
        for alias in ["real", "REAL", "authentic", "genuine", "camera", "natural", "human", "photo", "0", "false"]:
            self.assertEqual(normalize_label(alias), "real", f"Failed for alias: {alias}")

        # AI labels
        for alias in ["ai", "AI", "fake", "synthetic", "generated", "ai_generated", "deepfake", "1", "true"]:
            self.assertEqual(normalize_label(alias), "ai", f"Failed for alias: {alias}")

        # Invalid labels
        with self.assertRaises(ValueError):
            normalize_label("unknown")
        with self.assertRaises(ValueError):
            normalize_label("")
        with self.assertRaises(ValueError):
            normalize_label("maybe_ai")

    def test_determine_error_type(self):
        self.assertIsNone(determine_error_type("real", "real"))
        self.assertIsNone(determine_error_type("ai", "ai"))
        self.assertEqual(determine_error_type("real", "ai"), "FP")
        self.assertEqual(determine_error_type("ai", "real"), "FN")

    def test_classify_score_thresholds(self):
        # Default 0.5 threshold
        self.assertEqual(classify_score(0.0), "real")
        self.assertEqual(classify_score(0.499), "real")
        self.assertEqual(classify_score(0.500), "ai")
        self.assertEqual(classify_score(0.999), "ai")

        # Custom threshold
        self.assertEqual(classify_score(0.60, threshold=0.65), "real")
        self.assertEqual(classify_score(0.65, threshold=0.65), "ai")
        self.assertEqual(classify_score(0.70, threshold=0.65), "ai")

    def test_compute_metrics_perfect_predictions(self):
        samples = [
            EvaluationSample("ai1.png", "ai", 0.9, "ai", True, None, "test"),
            EvaluationSample("ai2.png", "ai", 0.8, "ai", True, None, "test"),
            EvaluationSample("real1.png", "real", 0.1, "real", True, None, "test"),
            EvaluationSample("real2.png", "real", 0.2, "real", True, None, "test"),
        ]
        m = compute_metrics(samples)
        self.assertEqual(m.total, 4)
        self.assertEqual(m.tp, 2)
        self.assertEqual(m.tn, 2)
        self.assertEqual(m.fp, 0)
        self.assertEqual(m.fn, 0)
        self.assertEqual(m.accuracy, 1.0)
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 1.0)
        self.assertEqual(m.specificity, 1.0)
        self.assertEqual(m.f1_score, 1.0)
        self.assertEqual(m.false_positive_rate, 0.0)
        self.assertEqual(m.false_negative_rate, 0.0)

    def test_compute_metrics_inverted_predictions(self):
        samples = [
            EvaluationSample("ai1.png", "ai", 0.1, "real", False, "FN", "test"),
            EvaluationSample("ai2.png", "ai", 0.2, "real", False, "FN", "test"),
            EvaluationSample("real1.png", "real", 0.8, "ai", False, "FP", "test"),
            EvaluationSample("real2.png", "real", 0.9, "ai", False, "FP", "test"),
        ]
        m = compute_metrics(samples)
        self.assertEqual(m.total, 4)
        self.assertEqual(m.tp, 0)
        self.assertEqual(m.tn, 0)
        self.assertEqual(m.fp, 2)
        self.assertEqual(m.fn, 2)
        self.assertEqual(m.accuracy, 0.0)
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.specificity, 0.0)
        self.assertEqual(m.f1_score, 0.0)
        self.assertEqual(m.false_positive_rate, 1.0)
        self.assertEqual(m.false_negative_rate, 1.0)

    def test_compute_metrics_mixed_matrix(self):
        # 6 TP, 4 TN, 2 FP, 3 FN -> Total = 15
        samples = []
        for i in range(6):
            samples.append(EvaluationSample(f"tp_{i}.png", "ai", 0.8, "ai", True, None, "test"))
        for i in range(4):
            samples.append(EvaluationSample(f"tn_{i}.png", "real", 0.2, "real", True, None, "test"))
        for i in range(2):
            samples.append(EvaluationSample(f"fp_{i}.png", "real", 0.7, "ai", False, "FP", "test"))
        for i in range(3):
            samples.append(EvaluationSample(f"fn_{i}.png", "ai", 0.3, "real", False, "FN", "test"))

        m = compute_metrics(samples)
        self.assertEqual(m.total, 15)
        self.assertEqual(m.tp, 6)
        self.assertEqual(m.tn, 4)
        self.assertEqual(m.fp, 2)
        self.assertEqual(m.fn, 3)

        self.assertAlmostEqual(m.accuracy, 10 / 15)
        self.assertAlmostEqual(m.precision, 6 / 8)
        self.assertAlmostEqual(m.recall, 6 / 9)
        self.assertAlmostEqual(m.specificity, 4 / 6)
        self.assertAlmostEqual(m.false_positive_rate, 2 / 6)
        self.assertAlmostEqual(m.false_negative_rate, 3 / 9)

        prec = 6 / 8
        rec = 6 / 9
        expected_f1 = 2 * (prec * rec) / (prec + rec)
        self.assertAlmostEqual(m.f1_score, expected_f1)

        dict_res = m.to_dict()
        self.assertEqual(dict_res["total"], 15)
        self.assertEqual(dict_res["tp"], 6)
        self.assertEqual(dict_res["fp"], 2)

    def test_compute_metrics_zero_division_safety(self):
        # Empty samples
        empty_m = compute_metrics([])
        self.assertEqual(empty_m.total, 0)
        self.assertEqual(empty_m.accuracy, 0.0)
        self.assertEqual(empty_m.precision, 0.0)
        self.assertEqual(empty_m.recall, 0.0)
        self.assertEqual(empty_m.f1_score, 0.0)

        # Dataset has only REAL images (no AI)
        real_only = [
            EvaluationSample("real1.png", "real", 0.1, "real", True, None, "test"),
            EvaluationSample("real2.png", "real", 0.2, "real", True, None, "test"),
        ]
        m_real = compute_metrics(real_only)
        self.assertEqual(m_real.total, 2)
        self.assertEqual(m_real.tp, 0)
        self.assertEqual(m_real.tn, 2)
        self.assertEqual(m_real.precision, 0.0)  # TP / (TP+FP) = 0/0 -> 0.0
        self.assertEqual(m_real.recall, 0.0)     # TP / (TP+FN) = 0/0 -> 0.0
        self.assertEqual(m_real.specificity, 1.0)
        self.assertEqual(m_real.false_positive_rate, 0.0)

        # Dataset has only AI images (no Real)
        ai_only = [
            EvaluationSample("ai1.png", "ai", 0.9, "ai", True, None, "test"),
            EvaluationSample("ai2.png", "ai", 0.4, "real", False, "FN", "test"),
        ]
        m_ai = compute_metrics(ai_only)
        self.assertEqual(m_ai.total, 2)
        self.assertEqual(m_ai.tp, 1)
        self.assertEqual(m_ai.fn, 1)
        self.assertEqual(m_ai.specificity, 0.0)  # TN / (TN+FP) = 0/0 -> 0.0
        self.assertEqual(m_ai.false_positive_rate, 0.0)


class CsvLoadingTests(unittest.TestCase):
    def test_load_csv_with_headers(self):
        csv_content = "image_path,label\nphoto1.jpg,real\nphoto2.png,ai\nphoto3.webp,authentic\nphoto4.bmp,synthetic\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "data.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            records = load_labeled_csv(csv_path)
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0][1], "real")
            self.assertEqual(records[1][1], "ai")
            self.assertEqual(records[2][1], "real")
            self.assertEqual(records[3][1], "ai")

    def test_load_csv_headerless(self):
        csv_content = "photo1.jpg,real\nphoto2.png,ai\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "headerless.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            records = load_labeled_csv(csv_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0][1], "real")
            self.assertEqual(records[1][1], "ai")

    def test_load_csv_handles_empty_lines_and_spaces(self):
        csv_content = "filepath , ground_truth\n\n  img_a.jpg  ,  REAL  \n\n  img_b.png ,  AI  \n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "spaced.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            records = load_labeled_csv(csv_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0][1], "real")
            self.assertEqual(records[1][1], "ai")

    def test_missing_csv_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_labeled_csv("nonexistent_path_truthlen_test.csv")


class DetectorEvaluationRunnerTests(unittest.TestCase):
    def test_evaluate_detector_tracks_fp_and_fn(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            img_real_ok = dir_path / "real_ok.png"
            img_real_fp = dir_path / "real_fp.png"
            img_ai_ok = dir_path / "ai_ok.png"
            img_ai_fn = dir_path / "ai_fn.png"

            img_real_ok.write_bytes(b"dummy")
            img_real_fp.write_bytes(b"dummy")
            img_ai_ok.write_bytes(b"dummy")
            img_ai_fn.write_bytes(b"dummy")

            records = [
                (img_real_ok, "real"),
                (img_real_fp, "real"),
                (img_ai_ok, "ai"),
                (img_ai_fn, "ai"),
            ]

            def mock_classify(raw: bytes):
                # We identify which file by inspecting caller context or score mapping
                # Mock return based on caller
                return {"synthetic_score": 0.5, "model": "test-model"}

            scores = {
                str(img_real_ok): 0.1,  # Real -> Real (TN)
                str(img_real_fp): 0.9,  # Real -> AI   (FP)
                str(img_ai_ok): 0.85,  # AI   -> AI   (TP)
                str(img_ai_fn): 0.15,  # AI   -> Real (FN)
            }

            def scored_classify(img_p: Path):
                return {
                    "synthetic_score": scores[str(img_p)],
                    "model": "mock-detector",
                    "top_label": "ai" if scores[str(img_p)] >= 0.5 else "real",
                    "top_score": max(scores[str(img_p)], 1 - scores[str(img_p)]),
                    "explanation": "mock explanation",
                }

            # Wrap classify_fn to map via reading
            class MockClassifier:
                def __init__(self, scores_dict):
                    self.scores_dict = scores_dict
                    self.call_idx = 0

                def __call__(self, raw: bytes):
                    # Sequential return based on records order
                    expected_p = records[self.call_idx][0]
                    self.call_idx += 1
                    return scored_classify(expected_p)

            result = evaluate_detector(
                detector_name="Mock Detector",
                model_id="mock-detector",
                classify_fn=MockClassifier(scores),
                records=records,
                threshold=0.5,
            )

            self.assertEqual(result.metrics.tp, 1)
            self.assertEqual(result.metrics.tn, 1)
            self.assertEqual(result.metrics.fp, 1)
            self.assertEqual(result.metrics.fn, 1)
            self.assertEqual(len(result.false_positives), 1)
            self.assertEqual(result.false_positives[0].path, str(img_real_fp))
            self.assertEqual(len(result.false_negatives), 1)
            self.assertEqual(result.false_negatives[0].path, str(img_ai_fn))
            self.assertEqual(len(result.errors), 0)

    def test_evaluate_detector_handles_missing_file_gracefully(self):
        missing_p = Path("nonexistent_image_12345.png")
        records = [(missing_p, "real")]

        result = evaluate_detector(
            detector_name="Mock",
            model_id="mock",
            classify_fn=lambda raw: {"synthetic_score": 0.1},
            records=records,
            threshold=0.5,
        )

        self.assertEqual(len(result.errors), 1)
        self.assertIn("does not exist", result.errors[0]["error"])
        self.assertEqual(result.metrics.total, 0)

    def test_evaluate_detector_handles_detector_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img = Path(tmp_dir) / "test.png"
            img.write_bytes(b"dummy")
            records = [(img, "real")]

            def failing_classify(raw: bytes):
                from backend.app.detectors import DetectorUnavailable
                raise DetectorUnavailable("Model weights not available")

            result = evaluate_detector(
                detector_name="Failing Detector",
                model_id="failing-model",
                classify_fn=failing_classify,
                records=records,
                threshold=0.5,
            )

            self.assertEqual(len(result.errors), 1)
            self.assertIn("DetectorUnavailable", result.errors[0]["error"])
            self.assertEqual(result.metrics.total, 0)


if __name__ == "__main__":
    unittest.main()
