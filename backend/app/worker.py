from __future__ import annotations

import logging
from pathlib import Path

from .config import get_settings
from .media import analyze_file
from .store import get_job, init_store, update_job

logger = logging.getLogger("truthlen.worker")


def process_job(job_id: str) -> None:
    init_store()
    job = get_job(job_id)
    if not job:
        logger.error("Queued job %s was not found", job_id)
        return
    path = Path(job["file_path"])
    update_job(job_id, "processing")
    try:
        result = analyze_file(path, job["original_name"], external_provider_consent=bool(job.get("external_provider_consent", False)))
        result["created_at"] = job["created_at"]
        update_job(job_id, "completed", result=result)
    except ValueError as exc:
        logger.info("Analysis input rejected for job %s: %s", job_id, exc)
        update_job(job_id, "failed", error=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed for job %s", job_id)
        update_job(
            job_id,
            "failed",
            error=f"Analysis failed ({type(exc).__name__}). Check the backend log for details.",
        )
    finally:
        if not get_settings().retain_uploads:
            path.unlink(missing_ok=True)
