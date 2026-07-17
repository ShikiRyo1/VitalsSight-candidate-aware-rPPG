# Multimodal assistant QA report

Date: 2026-07-17
Scope: local engineering conformance for the VitalsSight research console

## Validated configuration

- Text orchestration: `qwen3:4b` through Ollama.
- Image intake: `qwen3-vl:4b-instruct` through Ollama.
- Speech-to-text: `faster-whisper small`, CPU int8.
- UI/API: Streamlit and FastAPI on localhost.
- Validation host: Ryzen 7 5800HS, 42 GB RAM, CPU inference.

## Results

- Python suite: 85/85 tests passed.
- Dedicated browser/API suite: 33/33 checks passed.
- Browser diagnostics: zero console errors, zero page errors, zero HTTP 5xx responses.
- Desktop and 390 x 844 mobile viewports: no horizontal overflow.
- Latest image-grounded chat: `provider=ollama`, `model=qwen3:4b`, `degraded=false`.

The browser run exercised sidebar collapse and restoration; a synthetic microphone recording; local transcription; editable transcript display; recording deletion; independent transcript discard; image upload; Qwen3-VL summary; image-context chat; desktop and mobile layout; and media-retention disclosures.

## Safety and privacy checks

- Raw audio temporary files were deleted after decoding, including failure paths.
- Images were dimension-checked before full decode, EXIF-normalized, resized, and re-encoded without metadata.
- Image focus questions and model output were screened for prompt injection, emergency, diagnosis, treatment, and policy-bypass text.
- OCR remained visible only in the intake panel for human review and was omitted from the main model evidence prompt.
- Media context was explicitly `authoritative=false` and `retained=false`.
- Assistant audit rows stored media type, context identifier, SHA-256, and retention flag, not raw audio, image bytes, OCR, transcript, or temporary path.
- The answer post-validator rejected any image-derived numeric measurement not present in authoritative tool evidence.

## Observed local latency

- Qwen3-VL screenshot analysis: approximately 17-23 seconds in exercised hot runs.
- Faster-whisper for a 5.2-second English recording: approximately 3-11 seconds across cold and hot API runs.
- End-to-end latency depends on model residency, CPU contention, media complexity, and prompt length; no real-time claim is made.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node scripts\validate_multimodal_assistant.mjs `
  http://127.0.0.1:8502 http://127.0.0.1:8011 `
  docs\assets\product-console-overview.png `
  output\multimodal_acceptance_20260717 `
  runtime\multimodal_acceptance_voice.wav
```

The checked-in browser harness accepts any authorized local audio fixture; the synthetic WAV used for this run is not part of the public repository.

## Claim boundary

This report establishes finite local software conformance on exercised paths. It is not clinical validation, diagnostic evidence, prospective usability evidence, a production security certification, or a medical-device performance claim.
