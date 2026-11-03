import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from backend.app.detectors import DetectorUnavailable, _class_probabilities, _predict, _synthetic_probability, classify
from backend.app.fusion import fuse_scores


class FusionTests(unittest.TestCase):
    def test_no_detector_score_is_unavailable(self):
        result = fuse_scores([{"modality": "image", "available": False}])
        self.assertIsNone(result["synthetic_score"])
        self.assertEqual(result["verdict"], "unavailable")
        self.assertEqual(result["confidence"], 0.0)

    def test_single_available_modality_is_fused_and_coverage_is_partial(self):
        result = fuse_scores([{"modality": "image", "available": True, "synthetic_score": 0.9}])
        self.assertEqual(result["synthetic_score"], 0.9)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertAlmostEqual(result["coverage"], 1 / 3, places=3)
        self.assertIn("not a calibrated probability", result["explanation"])

    def test_disagreement_reduces_evidence_strength(self):
        aligned = fuse_scores([
            {"modality": "image", "available": True, "synthetic_score": 0.9},
            {"modality": "audio", "available": True, "synthetic_score": 0.85},
        ])
        split = fuse_scores([
            {"modality": "image", "available": True, "synthetic_score": 0.9},
            {"modality": "audio", "available": True, "synthetic_score": 0.1},
        ])
        self.assertGreater(aligned["confidence"], split["confidence"])
        self.assertEqual(split["verdict"], "inconclusive")

    def test_invalid_scores_are_not_used(self):
        result = fuse_scores([
            {"modality": "image", "available": True, "synthetic_score": 1.7},
            {"modality": "audio", "available": True, "synthetic_score": 0.2},
        ])
        self.assertEqual(result["synthetic_score"], 0.2)
        self.assertAlmostEqual(result["coverage"], 1 / 3, places=3)

    def test_image_model_keeps_ai_generation_and_deepfake_scores_separate(self):
        generated_score, manipulation_score = _class_probabilities(
            [
                {"label": "AI", "score": 0.4},
                {"label": "Deepfake", "score": 0.3},
                {"label": "Real", "score": 0.3},
            ],
            "image",
            "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2",
        )
        self.assertAlmostEqual(generated_score, 0.4)
        self.assertAlmostEqual(manipulation_score, 0.3)
        self.assertAlmostEqual(
            _synthetic_probability(
                [{"label": "fake", "score": 0.7}, {"label": "real", "score": 0.3}],
                "audio",
                "example/audio-model",
            ),
            0.7,
        )

    def test_unexpected_runtime_errors_become_actionable_unavailable_errors(self):
        with patch("backend.app.detectors._predict", side_effect=RuntimeError("broken runtime")), patch(
            "backend.app.detectors.logger.exception"
        ):
            with self.assertRaises(DetectorUnavailable) as raised:
                classify(b"media", "image")
        self.assertIn("RuntimeError", str(raised.exception))
        self.assertIn("backend logs", str(raised.exception))

    def test_local_inference_uses_the_configured_model_id(self):
        settings = SimpleNamespace(
            image_endpoint="",
            audio_endpoint="",
            image_model_id="example/image-model",
            audio_model_id="example/audio-model",
            inference_mode="local",
            hf_token="",
        )
        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
        pipeline = Mock(return_value=[{"label": "real", "score": 1.0}])
        with patch("backend.app.detectors.get_settings", return_value=settings), patch(
            "backend.app.detectors._local_pipeline", return_value=pipeline
        ) as load_pipeline:
            output, model = _predict(image_bytes.getvalue(), "image")
        load_pipeline.assert_called_once_with("image-classification", "example/image-model")
        self.assertEqual(model, "example/image-model")
        self.assertEqual(output[0]["label"], "real")


if __name__ == "__main__":
    unittest.main()
