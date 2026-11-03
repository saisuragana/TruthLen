# TruthLen implementation plan

**Brief:** FT-02, multimodal AI-generated media detection  
**Submission deadline:** November 4, 2026, 8:00 PM IST  
**Current state:** dashboard and local API run on Windows; model evaluation, video/audio smoke checks, and submission packaging remain.

## Requirement coverage

| FT-02 requirement | Current implementation | Remaining work |
| --- | --- | --- |
| Image detection | Local pretrained classifier and optional Hive cloud AI-generated/deepfake detector; scores are compared separately, with consent before external processing. Image AI-generation and face-deepfake scores remain distinct. | Obtain an authorized Hive key and smoke-check its API response; compare labeled examples without claiming accuracy |
| Video detection | Local mode samples up to 12 frames; optional Hive mode sends the full video for its frame-level AI-generation output. Embedded audio extraction is attempted when FFmpeg is available. | Test a short video with and without audio using the authorized service key; FFmpeg is missing from the current Windows PATH |
| Audio detection | Configurable pretrained audio classifier for uploaded or extracted audio | Run standalone audio samples and check codecs, clip lengths, model labels, and FFmpeg-dependent extraction |
| Per-modality detectors | Separate image, video-frame, and audio outputs with availability and explanation fields | Compare supported model candidates and lock model IDs before demo |
| Score fusion | Mean available AI-generation/fake-audio score; image deepfake/manipulation classes remain separate. Coverage, disagreement, and a coverage-bounded heuristic evidence index. One available detector yields an inconclusive verdict; directional verdicts require at least two. | Evaluate provisional thresholds on labeled samples; do not describe the index as calibrated confidence |
| Confidence and explanations | Heuristic evidence-strength estimate plus class scores, model names, detector reasons, and video timestamps | Add visual heatmap only if it can be generated from an explainable model without implying pixel-level proof |
| Large-volume processing | Upload limits, SQLite job records, and optional Redis/RQ worker deployment. The current Windows demo uses the API's inline background task, not the distributed queue. | Docker is unavailable on the current Windows machine; validate Compose/Redis/RQ on a Docker-capable host and load-test before claiming large-volume readiness |
| Dashboard and reports | Responsive upload/results/history dashboard and JSON download | Run an end-to-end demo in the final deployment environment |

## Time-boxed execution order

1. **Integration pass:** keep the local Python server running, verify `/api/health`, re-run unit checks after the fusion change, and test one labeled sample per modality.
2. **Environment pass:** install or configure FFmpeg for video audio extraction; do not spend deadline time installing Docker on the demo machine unless Compose deployment is needed.
3. **Evidence pass:** record model IDs, known-real/known-generated samples, false positives, unavailable modalities, and observed outcomes. This small set is a smoke check, not an accuracy benchmark.
4. **Submission pass:** update screenshots/demo after the latest changes, verify README instructions, commit and push reviewed code, then submit before 8:00 PM IST on Nov 4.

If an inference model cannot be accessed or loaded, keep that modality visibly unavailable and demo the remaining working paths. Do not replace it with a random score or claim that the full multimodal model ran.

## Suggested split for a three-person team

- **Project lead:** integrate backend, model loading, fusion, and final demo run.
- **Teammate 2:** prepare a small labeled evaluation set, verify format/model access, record confusion matrix and failure cases.
- **Teammate 3:** polish dashboard and submission story, verify deployment steps, capture demo screenshots/video.

Keep model credentials out of GitHub. Do not commit uploaded sample media, datasets, or downloaded model weights.

## Release checklist

- [ ] Start from a fresh clone using the README instructions.
- [ ] Confirm `/api/health`, one image, one video, one standalone audio, and one downloadable report.
- [ ] Confirm one-modality results are inconclusive and explain the uncalibrated raw model score.
- [ ] Confirm that a video with audio produces both video-frame and audio entries where FFmpeg can extract the track.
- [ ] Record known limitations, test conditions, and any model failures.
- [ ] Check that job status and recent history survive an API restart when using Compose.
- [ ] Remove demo uploads and secrets; leave uploads deleted by default.
- [ ] Submit before **2026-11-04 20:00 IST**.
