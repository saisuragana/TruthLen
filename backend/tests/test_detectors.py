import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.detectors import _predict_local_audio


class LocalAudioInferenceTests(unittest.TestCase):
    def test_audio_file_is_closed_and_readable_when_pipeline_opens_it(self):
        audio_bytes = b"test wav payload"
        observed_paths = []

        def pipeline_factory(task, model_id):
            self.assertEqual(task, "audio-classification")
            self.assertEqual(model_id, "test-audio-model")

            def classify(path, top_k):
                audio_path = Path(path)
                observed_paths.append(audio_path)
                self.assertEqual(top_k, 10)
                self.assertTrue(audio_path.is_file())
                self.assertEqual(audio_path.read_bytes(), audio_bytes)
                return [{"label": "fake", "score": 0.9}]

            return classify

        with patch("backend.app.detectors._local_pipeline", side_effect=pipeline_factory):
            result = _predict_local_audio(audio_bytes, "test-audio-model", "embedded-audio.wav")

        self.assertEqual(result, [{"label": "fake", "score": 0.9}])
        self.assertEqual(len(observed_paths), 1)
        self.assertFalse(observed_paths[0].exists())


if __name__ == "__main__":
    unittest.main()
