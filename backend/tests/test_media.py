import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.app.media import analyze_file, media_kind


class MediaKindTests(unittest.TestCase):
    def test_supported_media_extensions(self):
        self.assertEqual(media_kind("photo.WEBP"), "image")
        self.assertEqual(media_kind("clip.mp4"), "video")
        self.assertEqual(media_kind("voice.FLAC"), "audio")

    def test_unknown_extension_is_rejected(self):
        self.assertIsNone(media_kind("archive.zip"))


class ImageComparisonTests(unittest.TestCase):
    def test_comparison_model_is_reported_separately_and_not_fused(self):
        def classify_sample(_raw, _modality, filename_hint=None, model_override=None):
            score = 0.8 if model_override else 0.2
            result = {
                "available": True,
                "modality": "image",
                "model": model_override or "primary",
                "synthetic_score": score,
                "top_label": "fake" if score > 0.5 else "real",
                "top_score": max(score, 1 - score),
                "class_scores": [],
                "explanation": "test output",
            }
            if model_override is None:
                result["manipulation_score"] = 0.9
            return result

        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "photo.jpg"
            Image.new("RGB", (16, 12), "white").save(image_path)
            with patch(
                "backend.app.media.get_settings",
                return_value=SimpleNamespace(image_comparison_model_id="comparison", hive_api_key=""),
            ), patch("backend.app.media.classify", side_effect=classify_sample):
                result = analyze_file(image_path, "photo.jpg")

        self.assertEqual(result["modalities"][0]["synthetic_score"], 0.2)
        self.assertAlmostEqual(result["fusion"]["synthetic_score"], 0.2)
        self.assertAlmostEqual(result["fusion"]["coverage"], 1.0, places=3)
        self.assertEqual(result["comparisons"][0]["synthetic_score"], 0.8)
        self.assertTrue(any("camera video confirmed as genuine" in item for item in result["explanations"]))
        self.assertIn("false alarm", result["plain_summary"])


if __name__ == "__main__":
    unittest.main()
