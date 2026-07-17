# Local assistant setup

## Supported local profile

- Python 3.10 to 3.12 using the repository `.venv`.
- Ollama bound to `127.0.0.1:11434`.
- `qwen3:4b` for CPU or low-resource local use.
- `qwen3:8b` for a GPU-backed or latency-tolerant profile.
- `qwen3-vl:4b-instruct` for bounded image understanding.
- `faster-whisper small` with CPU int8 for local Chinese/English speech transcription.
- Existing Streamlit and FastAPI services on localhost.

The 4B and 8B profiles use the same tools, evidence contract, post-validation, and deterministic fallback. The model changes language quality and latency, not measurement or gate behavior.

## Install and verify

```powershell
.\.venv\Scripts\python.exe scripts\setup_local_assistant.py --model qwen3:4b
```

The command invokes `ollama pull`, then verifies that the configured model appears in the local Ollama registry.
The configured tag must exist exactly in `ollama list` (with the normal `:latest` alias accepted where applicable). A similarly prefixed tag is not treated as the requested model.

Install the multimodal dependencies and cache both sidecars:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-multimodal.txt
.\.venv\Scripts\python.exe scripts\setup_multimodal_assistant.py `
  --vision-model qwen3-vl:4b-instruct --asr-model small
```

Use the explicit `-instruct` Qwen3-VL tag. The default `qwen3-vl:4b` tag is a thinking profile and is not the supported low-latency structured-intake configuration.

## Start the complete product

Read-only assistant mode is the default:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vitalssight_with_assistant.ps1
```

For a controlled reviewer test that includes explicit-confirmation review updates:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vitalssight_with_assistant.ps1 -EnableReviewActions
```

For browser QA or a controlled pilot, isolate both persistent state and uploaded fixtures:

```powershell
$run = Get-Date -Format "yyyyMMdd_HHmmss"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vitalssight_with_assistant.ps1 `
  -UiPort 8502 -ApiPort 8011 `
  -DbPath "output\controlled_pilot\$run\state.db" `
  -UploadDir "output\controlled_pilot\$run\uploads"
```

The script starts:

- Streamlit UI: `http://127.0.0.1:8501`
- FastAPI and API documentation: `http://127.0.0.1:8010/docs`
- Local Qwen through Ollama: `http://127.0.0.1:11434`
- Local image sidecar: `qwen3-vl:4b-instruct` through the same Ollama endpoint
- Local speech sidecar: `faster-whisper small` loaded in the API/UI process

Stop only the API and UI processes started by the launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_vitalssight.ps1
```

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `VITALSSIGHT_ASSISTANT_MODEL` | `qwen3:4b` | Ollama model name |
| `VITALSSIGHT_ASSISTANT_VISION_MODEL` | `qwen3-vl:4b-instruct` | Ollama image-intake model |
| `VITALSSIGHT_ASSISTANT_ASR_MODEL` | `small` | Faster-whisper speech model |
| `VITALSSIGHT_ASSISTANT_ASR_DEVICE` | `cpu` | Speech inference device |
| `VITALSSIGHT_ASSISTANT_ASR_COMPUTE_TYPE` | `int8` | Speech inference precision |
| `VITALSSIGHT_ASSISTANT_ASR_CACHE` | `runtime/models/whisper` | Local speech-model cache |
| `VITALSSIGHT_ASSISTANT_MAX_AUDIO_SECONDS` | `120` | Maximum accepted recording duration |
| `VITALSSIGHT_ASSISTANT_MAX_AUDIO_BYTES` | `26214400` | Maximum transient audio payload |
| `VITALSSIGHT_ASSISTANT_MAX_IMAGE_BYTES` | `8388608` | Maximum transient image payload |
| `VITALSSIGHT_OLLAMA_URL` | `http://127.0.0.1:11434` | Local provider endpoint |
| `VITALSSIGHT_ASSISTANT_TIMEOUT` | `120` | Provider timeout in seconds |
| `VITALSSIGHT_ASSISTANT_MODEL_TOOL_ROUTING` | `true` | Enable optional model-selected read-only tools |
| `VITALSSIGHT_ASSISTANT_ACTIONS_ENABLED` | `false` | Allow reviewer/admin pending-action preparation |
| `VITALSSIGHT_DB_PATH` | `runtime/vitalsight_console.db` | Shared evidence and assistant-audit database |
| `VITALSSIGHT_UPLOAD_DIR` | `runtime/uploads` | Temporary authorized-video upload directory |

The launcher also accepts `-Model`, `-VisionModel`, `-AsrModel`, `-UiPort`, `-ApiPort`, `-DbPath`, `-UploadDir`, `-EnableReviewActions`, `-SkipModelCheck`, `-RequireModel`, `-SkipMultimodalCheck`, and `-RequireMultimodal`. The require switches make startup fail instead of degrading when an exact model tag or sidecar is unavailable.

## Verify the REST contract

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/v1/assistant/health
```

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/v1/assistant/multimodal/health
```

```powershell
$body = @{
  message = "Why is this case under review?"
  case_id = "demo_motion_conflict"
  role = "operator"
  language = "en"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8010/api/v1/assistant/chat `
  -Method Post -ContentType "application/json" -Body $body
```

The interactive API documentation includes multipart examples for `POST /api/v1/assistant/transcribe` and `POST /api/v1/assistant/analyze-image`. Both endpoints return a sanitized `context` object that may be supplied in `media_contexts` on the next `/chat` request. Raw bytes are never included in that request or audit record.

## Troubleshooting

- `status=degraded`: Ollama or the configured model is unavailable; deterministic guidance remains active.
- Model not installed: run `ollama list`, then rerun `setup_local_assistant.py` without `--skip-pull`.
- Slow CPU response: use `qwen3:4b`, keep model tool routing enabled only when needed, or deploy `qwen3:8b` on a GPU server. On the validation workstation, exercised CPU responses were approximately 15-51 seconds; this is a local observation, not a performance guarantee or a real-time claim.
- Slow image analysis: confirm the explicit `qwen3-vl:4b-instruct` tag. A hot local screenshot analysis took approximately 17 seconds on the validation workstation; model load, image complexity, and CPU contention can increase this substantially.
- Empty Qwen3-VL response: check that the thinking tag was not selected. Run `ollama list` and set `VITALSSIGHT_ASSISTANT_VISION_MODEL=qwen3-vl:4b-instruct`.
- Speech unavailable: install `requirements-multimodal.txt`, run the multimodal setup script, and verify the `/multimodal/health` response. Faster-whisper uses PyAV and does not require a separate system FFmpeg installation.
- Uncertain transcript: edit the transcript before sending. The quality label is an acoustic heuristic, not a transcription-accuracy guarantee.
- Port in use: stop the existing service or pass unused `-UiPort` and `-ApiPort` values.
- Review action unavailable: start with `-EnableReviewActions`, select reviewer or administrator, and enable “Prepare review updates”.
- Confirmation denied: check that actions are enabled and that the pending token has not expired or already been used.
- Startup failure: inspect `runtime/logs/assistant_api.stderr.log` and `runtime/logs/assistant_ui.stderr.log`; the launcher terminates its listeners when either health check fails.

Do not expose Ollama, Streamlit, or FastAPI directly to a public network without production authentication, authorization, TLS, rate limiting, monitoring, and a separate security review.
