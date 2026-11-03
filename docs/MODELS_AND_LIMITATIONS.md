# Model integrations and limitations

## Image and sampled video frames

Default model: `prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2`.

The Hugging Face model card presents three classes: AI-generated, deepfake, and real. TruthLen now uses only the AI-generated class for the image generation score. It reports the deepfake/manipulation class separately and does not add it to the AI-generated score. This distinction matters because a face-level deepfake score is not the same claim as an entirely AI-generated image. The model card reports its own evaluation numbers, but TruthLen has not reproduced those metrics and does not claim them as product performance.

An optional local comparison uses [`Deckard618/imdet` `pe_tiny`](https://huggingface.co/Deckard618/imdet), a small ONNX fake/real classifier. It is run on still images, displayed separately, and excluded from score fusion so two image models are not mistaken for two independent modalities. Its model card reports 94.6% balanced accuracy on a face benchmark and notes false positives and missed recent generators; TruthLen has not reproduced those results. The weights download to `data/models` on first use. Install `onnxruntime` through `backend/requirements-models.txt` to enable it.

## Optional Hive cloud detector

When `HIVE_API_KEY` is configured, TruthLen can submit supported images and videos to Hive's [AI-generated image and video detector](https://docs.thehive.ai/docs/ai-image-and-video-detection). The dashboard requires an explicit consent checkbox before an image/video is sent to Hive; leaving it clear keeps analysis local. For an upload where consent is given, Hive's output is the primary image/video signal and TruthLen's current local image/frame detector is shown separately; the local output does not enter score fusion in this mode. A Hive error is shown as unavailable and does not fall back to a local score as if it came from Hive. The hosted score is still an uncalibrated model output, can be wrong, and is not proof of origin. Video frame scores are averaged for display; temporal consistency is not analyzed. Audio continues to use the local audio model.

The current adapter targets Hive's V2 task API. Hive describes V2 as an Enterprise API and its self-serve Playground as V3; a V3 key will not work with this V2 adapter. No Hive request has been made from TruthLen because no API key was configured, so the V2 integration still requires an account-level smoke check. Follow the Playground's API tab for a V3 key/sample and update the adapter before using that key.

Hive states a default 14-day customer-data retention period in its [FAQ](https://docs.thehive.ai/docs/frequently-asked-questions-faq). The model page currently lists $6 per 1,000 image requests and $6 per 1,000 video frames; users should confirm the current billing terms before enabling it. Do not submit private media unless the uploader is authorized and agrees to external processing.

## Audio

Default model: `MelodyMachine/Deepfake-audio-detection-V2`.

The model card lists labels `fake` and `real`, and offers a Transformers pipeline example. Its model card gives little detail on intended use or evaluation data; its reported score is self-reported and not independently reproduced here. The card says the model is not deployed to shared Hugging Face Inference Providers, so local Transformers inference or a separately deployed endpoint is needed.

The current pipeline does not run voice activity detection or confirm that speech is present before scoring. It sends the uploaded audio, or up to the first 60 seconds extracted from a video, to the speech classifier. Music, silence, effects, multiple speakers, language, and recording quality may affect the output; do not interpret a result for non-speech audio as evidence about AI voice generation.

## Video

TruthLen samples up to 12 evenly spaced frames and applies the image classifier independently. For videos with readable audio, FFmpeg extracts mono audio at 16 kHz, capped at 60 seconds, for audio inference. Frame-score variation and high-scoring timestamps are reported.

This design does **not** evaluate temporal consistency, face identity, lip synchronization, or scene transitions. A static image classifier applied to frames is not equivalent to a video deepfake detector. The suggested `Naman712/Deep-fake-detection` checkpoint is gated behind access conditions and is not assumed available.

## Score and confidence semantics

- For images and sampled video frames, the fused generation score uses the AI-generated class only; a deepfake/manipulation class is reported separately. For audio, the fused score uses the model's fake/synthetic class.
- The second image model is a comparison output only. It does not add to coverage, confidence, or the fused verdict.
- Fusion averages the available detector scores. Coverage uses only applicable modalities: one for images, two for video (sampled frames and extractable audio), and one for audio. Missing applicable modalities reduce coverage; disagreement contributes to the evidence-strength index.
- The `confidence` API field is displayed as a heuristic evidence-strength index calculated from `score margin × applicable-modality coverage × detector-consensus factor × (1 − disagreement)`. A single available detector receives a 1/3 consensus factor. It is not calibrated against a TruthLen validation set.
- A result with only one available detector is always **Inconclusive**. Directional verdict bands (`0.35`, `0.65`) apply only when at least two modality results are available; they remain provisional, not statistically validated decision thresholds.
- Manual smoke testing with user-confirmed examples found: a Gemini-generated fantasy image received 99.96–100% in the primary model's AI class, while the comparison model assigned it 17.95% fake; another user-confirmed camera photo received 98% in the primary model's AI class; known-real camera photos received 99.98–100% in the primary model's deepfake class; one camera photo received 0% AI-generated and 100% deepfake/manipulation from the primary model while the comparison model assigned it 87.74% real; and a camera video confirmed as genuine received an 8.29% average AI-generated frame score but an 83.39% average deepfake/manipulation class score. In one direct comparison, the primary model gave a known-real camera photo 98% AI-generated and a known Gemini image 100% AI-generated. These are a few smoke-test examples, not a benchmark. The class split prevents the deepfake output from being presented as the AI-generated score, but the primary model has a demonstrated false-positive problem and the comparison model missed a known AI image. Neither should be trusted as a final detector.
- A lower AI-generation/fake-audio score does not establish origin or prove that media is authentic.

## Provenance and plain-language summaries

When `c2pa-python` is installed, TruthLen reads embedded C2PA Content Credentials locally and reports the SDK validation state. A missing credential is common, including on genuine camera media; it is not a synthetic-media signal. A valid or trusted credential supports a signed provenance record under the SDK trust policy, not the truth of the depicted scene. Unsupported formats and SDK failures are reported as unavailable rather than treated as fake. The plain-language summary is deterministic text derived from detector scores, their availability, comparison disagreement, and the provenance status. It is not an additional AI model and cannot override the underlying evidence.

## Operational limits

- Default local inference requires optional Torch/Transformers packages and a first-use model download. CPU inference can be slow.
- Hugging Face endpoints require working credentials and a deployed endpoint. Hosted inference sends media to that provider; use only when the team's privacy policy allows it.
- Hive inference also sends the selected image/video to a third party, requires a Hive model-enabled API key, and may incur charges. Consent is collected in the upload form; the local-only path remains available when the key is unset or consent is declined.
- Redis/RQ mode provides a durable queue, but SQLite and a shared local Docker volume remain single-host infrastructure. Public production needs authentication, rate limits, object storage, retention policy, monitoring, and horizontally scaled workers.
- `GET /api/health` is a liveness/configuration response; it does not load or prove that model weights can run. Model access and inference are only known after an analysis request. The upload API has no authentication and is intended for a local or access-controlled demo, not an exposed public service.
- Upload size defaults to 100 MB; video sampling and audio extraction are capped to keep work bounded.
- No model has been trained or fine-tuned by the TruthLen team in this repository.
