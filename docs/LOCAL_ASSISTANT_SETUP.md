# Local assistant setup

## Supported local profile

- Python 3.10 to 3.12 using the repository `.venv`.
- Ollama bound to `127.0.0.1:11434`.
- `qwen3:4b` for CPU or low-resource local use.
- `qwen3:8b` for a GPU-backed or latency-tolerant profile.
- Existing Streamlit and FastAPI services on localhost.

The 4B and 8B profiles use the same tools, evidence contract, post-validation, and deterministic fallback. The model changes language quality and latency, not measurement or gate behavior.

## Install and verify

```powershell
.\.venv\Scripts\python.exe scripts\setup_local_assistant.py --model qwen3:4b
```

The command invokes `ollama pull`, then verifies that the configured model appears in the local Ollama registry.
The configured tag must exist exactly in `ollama list` (with the normal `:latest` alias accepted where applicable). A similarly prefixed tag is not treated as the requested model.

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

Stop only the API and UI processes started by the launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_vitalssight.ps1
```

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `VITALSSIGHT_ASSISTANT_MODEL` | `qwen3:4b` | Ollama model name |
| `VITALSSIGHT_OLLAMA_URL` | `http://127.0.0.1:11434` | Local provider endpoint |
| `VITALSSIGHT_ASSISTANT_TIMEOUT` | `120` | Provider timeout in seconds |
| `VITALSSIGHT_ASSISTANT_MODEL_TOOL_ROUTING` | `true` | Enable optional model-selected read-only tools |
| `VITALSSIGHT_ASSISTANT_ACTIONS_ENABLED` | `false` | Allow reviewer/admin pending-action preparation |
| `VITALSSIGHT_DB_PATH` | `runtime/vitalsight_console.db` | Shared evidence and assistant-audit database |
| `VITALSSIGHT_UPLOAD_DIR` | `runtime/uploads` | Temporary authorized-video upload directory |

The launcher also accepts `-Model`, `-UiPort`, `-ApiPort`, `-DbPath`, `-UploadDir`, `-EnableReviewActions`, `-SkipModelCheck`, and `-RequireModel`. `-SkipModelCheck` is for deliberate degraded-mode testing; `-RequireModel` makes startup fail instead of degrading when Ollama or the exact model tag is unavailable.

## Verify the REST contract

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/v1/assistant/health
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

## Troubleshooting

- `status=degraded`: Ollama or the configured model is unavailable; deterministic guidance remains active.
- Model not installed: run `ollama list`, then rerun `setup_local_assistant.py` without `--skip-pull`.
- Slow CPU response: use `qwen3:4b`, keep model tool routing enabled only when needed, or deploy `qwen3:8b` on a GPU server. On the validation workstation, exercised CPU responses were approximately 15-51 seconds; this is a local observation, not a performance guarantee or a real-time claim.
- Port in use: stop the existing service or pass unused `-UiPort` and `-ApiPort` values.
- Review action unavailable: start with `-EnableReviewActions`, select reviewer or administrator, and enable “Prepare review updates”.
- Confirmation denied: check that actions are enabled and that the pending token has not expired or already been used.
- Startup failure: inspect `runtime/logs/assistant_api.stderr.log` and `runtime/logs/assistant_ui.stderr.log`; the launcher terminates its listeners when either health check fails.

Do not expose Ollama, Streamlit, or FastAPI directly to a public network without production authentication, authorization, TLS, rate limiting, monitoring, and a separate security review.
