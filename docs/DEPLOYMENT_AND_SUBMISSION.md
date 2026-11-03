# Deployment and submission guide

## Demo deployment

1. On a Linux machine with Docker Compose, clone the repository and copy `.env.example` to `.env`.
2. Set `HF_TOKEN` only if needed, verify the model IDs, then run `docker compose up --build -d`.
3. Open port 8000 only to the demo audience, or put the app behind an HTTPS reverse proxy. Do not expose an unauthenticated upload endpoint as a public service.
4. Keep the named `truthlen-data` and `model-cache` volumes. Confirm the first image and audio model loads before the demo; model downloads can take time.
   The optional comparison detector runs locally and downloads its small ONNX weights into the data volume on the first image check.
5. Open `/api/health` and submit one sample for each modality. Use `/docs` if the dashboard is unavailable.

To enable the optional Hive image/video comparison, set `HIVE_API_KEY` in `.env` and restart the Compose services. The uploader must check the external-processing consent box to send a file to Hive; leaving it clear uses local analysis. Review the published [Hive pricing](https://thehive.ai/models/hive/ai-generated-and-deepfake-content-detection) and [retention FAQ](https://docs.thehive.ai/docs/frequently-asked-questions-faq) before enabling it. Audio remains local.

The Compose deployment runs a Redis queue and a worker. Additional workers can be started with `docker compose up --scale worker=2 -d`; each worker loads model weights into its own process. The included SQLite database and Docker volume are single-host demo infrastructure, not a multi-host production data layer.

## Windows local demo without Docker

From PowerShell opened in the repository root, install dependencies once if they are not already present:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt -r backend/requirements-models.txt
```

Start the API and dashboard:

```powershell
$env:PYTHONPATH = "backend"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

Keep that PowerShell window open and browse to `http://127.0.0.1:8000`. This local mode processes jobs in the API process; it does not use the Redis/RQ distributed queue. Install FFmpeg and add it to `PATH` to extract embedded audio from videos. Without FFmpeg, video frame analysis may still run, but its audio result will be unavailable.

For a public launch, add authentication, rate limiting, HTTPS, monitoring, object storage, and a retention policy before opening uploads broadly. Keep secrets in the deployment provider's secret manager or an untracked `.env` file.

## Hackathon demo path

1. Explain the flow: upload → queued job → per-modality detectors → score fusion → downloadable report.
2. Show a real sample, wait for the model result, and point to the contributing modality, class scores, confidence estimate, and uncertainty.
3. Show a video report with sampled frame timestamps and, when available, embedded audio results.
4. State that video temporal analysis is not implemented, confidence is not calibrated, and no model training/fine-tuning has been performed.
5. If a detector is unavailable, show its unavailable reason and continue with the modalities that did return evidence.

## Submission package

- Repository URL: `https://github.com/saisuragana/TruthLen`
- Project title: **TruthLen — Multimodal AI-Generated Media Detection**
- Include the README, implementation plan, model limitations, and a short screen recording or screenshots of the working dashboard.
- Submit before **November 4, 2026, 8:00 PM IST**. Reserve time before the deadline for upload, links, and a final check; do not plan to finish at the deadline minute.
