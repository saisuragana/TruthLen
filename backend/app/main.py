from __future__ import annotations

import json
import logging
import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .media import ALLOWED_EXTENSIONS, media_kind
from .store import create_job, get_job, init_store, list_jobs, update_job
from .worker import process_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthlen.api")
settings = get_settings()
app = FastAPI(title="TruthLen API", version="0.1.0", description="Multimodal synthetic-media screening")


@app.on_event("startup")
def startup() -> None:
    init_store()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "queue_backend": settings.queue_backend,
        "inference_mode": settings.inference_mode,
        "readiness_note": "This endpoint confirms that the API is running. Model weights and inference are checked when a file is analyzed.",
        "models": {
            "image": {"model": settings.image_endpoint or settings.image_model_id, "configured": bool(settings.image_endpoint or settings.image_model_id)},
            "image_comparison": {"model": settings.image_comparison_model_id or None, "configured": bool(settings.image_comparison_model_id)},
            "audio": {"model": settings.audio_endpoint or settings.audio_model_id, "configured": bool(settings.audio_endpoint or settings.audio_model_id)},
            "hive": {"model": "Hive AI-Generated Image and Video Detection", "configured": bool(settings.hive_api_key)},
            "video": {"method": "sampled frames scored with image detector", "temporal_model": False},
        },
        "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
        "upload_policy": {
            "deleted_after_analysis": not settings.retain_uploads,
            "external_visual_check_available": bool(settings.hive_api_key),
        },
    }


@app.get("/api/jobs")
def history() -> list[dict]:
    return [_public_job(job) for job in list_jobs()]


@app.post("/api/jobs", status_code=202)
async def submit_job(
    background_tasks: BackgroundTasks,
    upload: UploadFile = File(...),
    external_provider_consent: bool = Form(False),
) -> JSONResponse:
    original_name = (upload.filename or "upload").replace("\\", "/").split("/")[-1]
    extension = Path(original_name).suffix.lower()
    kind = media_kind(original_name)
    if extension not in ALLOWED_EXTENSIONS or kind is None:
        raise HTTPException(status_code=415, detail="Unsupported file type. See accepted formats on the upload page.")

    job_id = str(uuid4())
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{job_id}{extension}"
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds the configured upload limit.")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    media_type = upload.content_type or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    create_job(job_id, original_name, media_type, str(destination.resolve()), external_provider_consent)
    if settings.queue_backend == "redis":
        try:
            from redis import Redis
            from rq import Queue

            Queue("truthlen", connection=Redis.from_url(settings.redis_url)).enqueue("app.worker.process_job", job_id)
        except Exception as exc:
            logger.exception("Could not enqueue job %s", job_id)
            update_job(job_id, "failed", error="The analysis queue is unavailable. Check Redis and the worker.")
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail="The analysis queue is unavailable.") from exc
    else:
        background_tasks.add_task(process_job, job_id)

    return JSONResponse(status_code=202, content={"id": job_id, "status": "queued", "media_type": kind})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return _public_job(job)


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str) -> Response:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    if job["status"] != "completed" or not job["result"]:
        raise HTTPException(status_code=409, detail="The report is available after analysis completes.")
    payload = json.dumps({"product": "TruthLen", "job_id": job_id, **job["result"]}, indent=2)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="truthlen-report-{job_id[:8]}.json"'},
    )


def _public_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "filename": job["original_name"],
        "media_type": media_kind(job["original_name"]) or "unknown",
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
