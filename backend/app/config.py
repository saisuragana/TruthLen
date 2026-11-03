from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int
    image_model_id: str
    image_comparison_model_id: str
    audio_model_id: str
    image_endpoint: str
    audio_endpoint: str
    hf_token: str
    hive_api_key: str
    inference_mode: str
    queue_backend: str
    redis_url: str
    max_video_frames: int
    retain_uploads: bool


def get_settings() -> Settings:
    return Settings(
        data_dir=Path(os.getenv("TRUTHLEN_DATA_DIR", "./data")).expanduser(),
        max_upload_bytes=int(os.getenv("TRUTHLEN_MAX_UPLOAD_MB", "100")) * 1024 * 1024,
        image_model_id=os.getenv(
            "TRUTHLEN_IMAGE_MODEL", "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"
        ),
        image_comparison_model_id=os.getenv(
            "TRUTHLEN_IMAGE_COMPARISON_MODEL", "Deckard618/imdet/pe_tiny"
        ).strip(),
        audio_model_id=os.getenv(
            "TRUTHLEN_AUDIO_MODEL", "MelodyMachine/Deepfake-audio-detection-V2"
        ),
        image_endpoint=os.getenv("TRUTHLEN_IMAGE_ENDPOINT", "").strip(),
        audio_endpoint=os.getenv("TRUTHLEN_AUDIO_ENDPOINT", "").strip(),
        hf_token=os.getenv("HF_TOKEN", "").strip(),
        hive_api_key=os.getenv("HIVE_API_KEY", "").strip(),
        inference_mode=os.getenv("TRUTHLEN_INFERENCE_MODE", "local").strip().lower(),
        queue_backend=os.getenv("TRUTHLEN_QUEUE_BACKEND", "inline").strip().lower(),
        redis_url=os.getenv("TRUTHLEN_REDIS_URL", "redis://localhost:6379/0"),
        max_video_frames=max(1, min(24, int(os.getenv("TRUTHLEN_MAX_VIDEO_FRAMES", "12")))),
        retain_uploads=os.getenv("TRUTHLEN_RETAIN_UPLOADS", "false").lower() == "true",
    )
