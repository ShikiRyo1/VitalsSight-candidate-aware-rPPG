# VitalsSight multimodal assistant

## Product flow

```text
typed question ------------------------------+
                                                |
microphone -> faster-whisper -> editable text --+--> input guard -> case/report/RAG tools
                                                |                    |
image -> metadata removal -> Qwen3-VL JSON -----+                    v
        -> injection and clinical-boundary filter             evidence-bounded answer
```

The multimodal layer changes how users provide a question or supporting workflow context. It does not replace the video measurement pipeline, candidate selector, risk policy, release/review/retake gate, report service, or human confirmation contract.

## Voice contract

1. Streamlit records a 16 kHz WAV through `st.audio_input`.
2. The service enforces byte, extension, and 120-second duration limits.
3. `faster-whisper small` runs locally with CPU int8 by default.
4. The temporary decoder file is deleted in `finally` whether transcription succeeds or fails.
5. A clear transcript is sent directly to read-only chat and remains visible as the user turn. An uncertain transcript is shown in an editable field and requires confirmation before sending.
6. The next chat turn receives text plus a hash-bound `audio_transcript` context; no audio bytes are sent to Qwen or written to the audit database.

The `clear`/`uncertain` label is an acoustic heuristic, not a word-error-rate estimate. Names, abbreviations, negation, drug names, and clinical terminology require manual verification. Clinical questions remain outside the assistant boundary even when spoken.

## Image contract

1. Accept JPEG, PNG, or WebP up to 8 MiB.
2. Decode with Pillow under a pixel limit, apply EXIF orientation, convert to RGB, resize to at most 1600 px, and re-encode without metadata.
3. Send only the normalized bytes to `qwen3-vl:4b-instruct` through Ollama's local base64 image API.
4. Request a compact JSON object: `summary`, `visible_text`, `workflow_relevance`, and `safety_flags`.
5. Validate locally with Pydantic. Empty, malformed, oversized, or boundary-crossing output activates a technical-only fallback.
6. Treat image text and the optional focus question as untrusted data. Prompt-disclosure, policy-bypass, diagnosis, treatment, emergency, and forged output-state text is removed before chat composition.
7. Keep OCR visible in the intake panel for human review, but exclude it from the main model's evidence prompt; numeric OCR therefore cannot become a system measurement.
8. Attach only a hash-bound `image` context to the next turn. The context is explicitly non-authoritative.

Images may help explain a screen, identify a capture-quality issue visible to a human, or route the user to a report/review/capture workspace. They cannot establish identity, infer demographic attributes, estimate HR or another vital sign, diagnose, or decide release/review/retake.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/assistant/multimodal/health` | Independent speech and image capability status |
| `POST` | `/api/v1/assistant/transcribe` | Multipart audio to editable local transcript |
| `POST` | `/api/v1/assistant/voice-chat` | Multipart audio to local transcript and evidence-bounded reply in one call |
| `POST` | `/api/v1/assistant/analyze-image` | Multipart image to bounded visual context |
| `POST` | `/api/v1/assistant/image-chat` | Privacy-normalized image analysis and evidence-bounded reply in one call |
| `POST` | `/api/v1/assistant/workflows/video` | Consented video assessment, output state, assistant explanation, and inline report contract |
| `POST` | `/api/v1/assistant/chat` | Text question with zero to two sanitized `media_contexts` |

Every media response states `raw_audio_retained=false` or `raw_image_retained=false`. Audit rows store only context identifier, media kind, SHA-256 digest, and the same retention flag.

## Local acceptance evidence

On the July 2026 Windows validation workstation (Ryzen 7 5800HS, 42 GB RAM, CPU inference):

- Qwen3-VL 4B instruct correctly summarized the VitalsSight overview screenshot after the JSON-length fix; a hot run took about 17.4 seconds.
- A boundary image containing prompt-injection text and a forged `RELEASE 88 BPM` line was reduced to a neutral exclusion message with `media_prompt_injection` and no visible text forwarded.
- Faster-whisper small transcribed an English 4.9-second synthetic recording in about 9.3 seconds and a Chinese 7.6-second synthetic recording in about 5.9 seconds.
- The full automated suite passed 85 tests after multimodal integration.
- The dedicated browser/API acceptance passed 33/33 checks, including synthetic microphone capture, editable transcription, recording cleanup, image upload, model-backed visual analysis, image-grounded Qwen chat, sidebar recovery, mobile layout, and zero console/page/HTTP 5xx errors.

These values are local engineering observations, not latency guarantees, clinical validation, speaker-independent ASR accuracy claims, or medical-device performance claims.
