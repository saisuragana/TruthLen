# TruthLen

**Multimodal synthetic-media screening for images, video, and audio.** TruthLen combines pretrained detector outputs, reports which modalities contributed, and explains the limits of each result. Image AI-generation scores and face-deepfake scores are reported separately.

> TruthLen is a screening aid. Its scores are not proof of authenticity, and the bundled model candidates have not been independently validated on a TruthLen benchmark.

## What works in this repository

- Upload PNG, JPEG, WebP, BMP, MP4, MOV, AVI, MKV, WebM, WAV, MP3, M4A, FLAC, OGG, or AAC files through the dashboard.
- Queue analyses as jobs and poll job status. SQLite persists job history; Docker Compose enables Redis/RQ workers for background processing.
- Image classification uses a configurable pretrained image model. The default candidate emits AI, deepfake, and real classes.
- Video is sampled at up to 12 evenly spaced frames. Each frame uses the image detector and the report includes the highest-scored timestamps. FFmpeg also extracts up to 60 seconds of embedded audio for the audio detector.
- Standalone and extracted audio use a configurable pretrained audio classifier.
- Fusion reports available model scores, modality coverage, disagreement, a heuristic evidence-strength index, and explanations. A single available modality produces an **Inconclusive** overall verdict; two or more are required for a directional verdict.
- The dashboard gives a plain-language overall reading, then keeps per-check scores, model names, and frame timestamps in expandable details. It also shows the optional second-model comparison, recent checks, and a downloadable JSON report.
- Optional Hive cloud detection adds a separate provider check for images and videos. Its score is not fused with the local model score; the upload form requires explicit consent first. Audio remains local.

The video path currently has no temporal model: it does not evaluate motion consistency. If model access, model dependencies, or inference is unavailable, the result is marked unavailable and no substitute score is generated.

## Run with Docker Compose

1. Copy `.env.example` to `.env` and review model IDs and upload limits. Keep `HF_TOKEN` empty for local inference unless a model requires access.
2. Start the app, Redis, and worker:

   ```sh
   docker compose up --build
   ```

3. Open [http://localhost:8000](http://localhost:8000). The first model request may download several hundred megabytes of model weights; allow it time to finish.

Compose persists job history and model cache in named volumes. Uploaded media is deleted after its analysis by default. Set `TRUTHLEN_RETAIN_UPLOADS=true` only if you need to keep it for a controlled demo.

## Run locally

Python 3.10+ is required. FFmpeg is required to extract audio tracks from video; frame sampling can run without it. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-models.txt
```

Local inference selects CUDA when PyTorch detects an available GPU; otherwise it uses CPU. Set `TRUTHLEN_DEVICE=cpu`, `cuda`, or `cuda:N` to override.

Then start the API and dashboard:

```sh
# Windows PowerShell
$env:PYTHONPATH = "backend"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload

# macOS/Linux
PYTHONPATH=backend uvicorn app.main:app --app-dir backend --reload
```

The local default uses a single-process background task. For queued processing, start Redis, set `TRUTHLEN_QUEUE_BACKEND=redis`, then run `rq worker --url redis://localhost:6379/0 truthlen` with `PYTHONPATH=backend`.

## Unit tests

From the repository root, run `python -m unittest discover -s backend/tests -v`. The small test suite covers score fusion and media type handling; pretrained model inference still needs separate end-to-end samples and is listed in the release checklist.

## Hosted inference endpoints

For an existing Hugging Face Inference Endpoint, set `HF_TOKEN` and either `TRUTHLEN_IMAGE_ENDPOINT` or `TRUTHLEN_AUDIO_ENDPOINT` to its URL. Endpoint inference takes precedence over local inference for that modality. The audio candidate below is not currently deployed by an Inference Provider, so hosted inference requires your own endpoint; otherwise use local Transformers inference.

## Optional Hive image/video second opinion

Hive provides a separate AI-generated image/video detector, with per-frame output for video. It can help compare model outputs, but it is another fallible model—not a guarantee of correctness. When `HIVE_API_KEY` is configured and the uploader checks consent for a supported image/video, TruthLen sends the file to Hive. Hive's score is then the primary image/video score and the existing local model is displayed separately as a comparison; the two are not fused. If consent is left clear, analysis remains local. Audio analysis remains local. If Hive is unavailable after consent, its result is shown as unavailable rather than being silently replaced by the local score.

1. Get a **Hive V2 project key** with Hive's [AI-Generated Image and Video Detection](https://docs.thehive.ai/docs/ai-image-and-video-detection) model enabled. Hive documents V2 as its Enterprise API; the self-serve Playground uses V3 keys, which are not accepted by this V2 adapter. Check Hive's [API version FAQ](https://docs.thehive.ai/docs/frequently-asked-questions-faq) before creating a key. If you only have a V3 key, use Hive's Playground for a manual second opinion; this repository needs the V3 Playground's API sample to be integrated separately.
2. Add `HIVE_API_KEY=your_key` to the untracked `.env` file in the repository root. Do not put the key in a screenshot, public issue, or Git commit.
3. For local Windows use, restart the API after saving `.env`. For Docker Compose, rebuild/restart with `docker compose up --build -d` so the service receives the key.
4. Open `/api/health` and check `models.hive.configured`. Select an image or video. Check the Hive consent box to send it to Hive, or leave it clear to use local analysis only.

Hive says it retains customer data for 14 days by default (retention can be changed by request). The published price for the model is $6 per 1,000 image requests and $6 per 1,000 video frames; confirm current terms and your account's billing before using it. See [Hive's FAQ](https://docs.thehive.ai/docs/frequently-asked-questions-faq) and [model page](https://thehive.ai/models/hive/ai-generated-and-deepfake-content-detection).

## API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | App, queue, model, and upload configuration |
| `POST /api/jobs` | Submit one multipart file; returns a job ID |
| `GET /api/jobs/{id}` | Poll status and retrieve a completed report payload |
| `GET /api/jobs` | Recent analyses |
| `GET /api/jobs/{id}/report` | Download a completed JSON report |

FastAPI's interactive API reference is at `/docs`.

## Model choices and limits

- **Image and sampled video frames:** [`prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2`](https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2), an Apache-2.0 image classifier. Its card describes AI/deepfake/real classes and reports its own evaluation results. Those results are not TruthLen's and are not repeated as a product accuracy claim.
- **Image comparison model:** [`Deckard618/imdet` `pe_tiny`](https://huggingface.co/Deckard618/imdet), a 12.5 MB ONNX fake/real candidate that runs locally on CPU. It is shown separately and is not fused with the primary result. Its reported face-dataset performance comes from its model card and has not been reproduced by TruthLen. First use downloads the model weights into `data/models`.
- **Provenance check:** optional C2PA Content Credentials inspection runs locally when the optional SDK is installed. It reports whether a credential was found and the SDK's validation state. Credentials describe a recorded provenance chain; they do not prove that the depicted scene is true or decide whether a file is AI-generated. No credential is common and is not evidence that media is fake.
- **Audio:** [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2), an Apache-2.0 audio classifier. Its card has limited detail about intended use and evaluation data; the displayed score must be treated cautiously. The current pipeline does not first verify that a clip contains speech, so music, silence, and background noise can make the result misleading. It is not deployed to Hugging Face's shared inference providers at the time this implementation was prepared.
- **Video:** The earlier candidate [`Naman712/Deep-fake-detection`](https://huggingface.co/Naman712/Deep-fake-detection) requires accepting gated access conditions. It is not bundled as the default. TruthLen instead reports sampled-frame evidence and clearly states that temporal analysis is absent.

Model files are downloaded at runtime and are not committed. TruthLen has not trained or fine-tuned any model. Manual smoke tests found known-real camera photos scoring 99.98–100% on the primary model's deepfake class; one user-confirmed camera photo scored AI-generated 0%, deepfake/manipulation 100%, while the independent comparison model scored it 87.74% real. Another user-confirmed camera photo scored 98% AI-generated, while a known Gemini-generated image scored 100% AI-generated. A camera video confirmed real scored 83.39% on the primary model's deepfake class. The primary model also scored a known Gemini-generated image 99.96% AI-generated, but the comparison model called it 82.05% real. These few examples are not a benchmark; they show that both models can be wrong and the primary model's high AI score does not reliably separate these real and AI examples. Deepfake scores are reported separately and excluded from the AI-generated score. The local comparison model is not fused with the primary score. Hive has not been exercised with a TruthLen API key yet, so its integration still needs a real-account smoke check. A single detector always produces an **Inconclusive** overall verdict, even if its uncalibrated class score is extreme. Coverage counts only the modalities relevant to the uploaded media type. The evidence-strength index also applies a single-detector penalty and is not calibrated. Do not use results as the sole basis for high-impact decisions.

## Project plan and submission status

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for FT-02 coverage, the time-boxed plan to the **November 4, 2026, 8:00 PM IST** deadline, team split, and remaining validation work. See [docs/MODELS_AND_LIMITATIONS.md](docs/MODELS_AND_LIMITATIONS.md) for model provenance and known constraints.

For a public hackathon demo URL using an eligible Azure for Students subscription, follow [docs/AZURE_STUDENT_DEPLOYMENT.md](docs/AZURE_STUDENT_DEPLOYMENT.md). The guide uses Azure Container Apps with a maximum of one replica and scale-to-zero enabled.

## License

This repository is MIT licensed. Each downloaded model has its own license and terms; verify those before redistribution or deployment.
