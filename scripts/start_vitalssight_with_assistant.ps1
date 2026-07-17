param(
    [string]$Model = "qwen3.6:35b",
    [string]$VisionModel = "qwen3-vl:4b-instruct",
    [string]$AsrModel = "small",
    [int]$UiPort = 8501,
    [int]$ApiPort = 8010,
    [string]$DbPath = "",
    [string]$UploadDir = "",
    [switch]$EnableReviewActions,
    [switch]$SkipModelCheck,
    [switch]$RequireModel,
    [switch]$SkipMultimodalCheck,
    [switch]$RequireMultimodal
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Runtime = Join-Path $Project "runtime"
$Logs = Join-Path $Runtime "logs"
$PidFile = Join-Path $Runtime "vitalsight_services.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The repository virtual environment is missing: $Python"
}
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Test-LocalPort([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Get-ListenerPid([int]$Port) {
    $Listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($Listener) { return [int]$Listener.OwningProcess }
    return $null
}

function Stop-Listener([int]$Port) {
    $ListenerPid = Get-ListenerPid $Port
    if ($ListenerPid) {
        Stop-Process -Id $ListenerPid -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-LocalPort 11434)) {
    $OllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    if ($OllamaCommand) {
        Start-Process -FilePath $OllamaCommand.Source -ArgumentList "serve" -WindowStyle Hidden | Out-Null
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Milliseconds 500
            if (Test-LocalPort 11434) { break }
        }
    }
}

if (-not $SkipModelCheck -and (Test-LocalPort 11434)) {
    & $Python (Join-Path $PSScriptRoot "setup_local_assistant.py") --model $Model --base-url "http://127.0.0.1:11434" --skip-pull
    if ($LASTEXITCODE -ne 0) {
        if ($RequireModel) { throw "The configured model is not ready" }
        Write-Warning "The configured model is not ready. VitalsSight will start with deterministic evidence guidance."
    }
} elseif (-not (Test-LocalPort 11434)) {
    if ($RequireModel) { throw "Ollama did not start on 127.0.0.1:11434" }
    Write-Warning "Ollama is unavailable. VitalsSight will start with deterministic evidence guidance."
}

if (-not $SkipMultimodalCheck -and (Test-LocalPort 11434)) {
    & $Python (Join-Path $PSScriptRoot "setup_multimodal_assistant.py") `
        --vision-model $VisionModel --asr-model $AsrModel --base-url "http://127.0.0.1:11434" `
        --skip-vision-pull --skip-asr-download
    if ($LASTEXITCODE -ne 0) {
        if ($RequireMultimodal) { throw "One or more multimodal sidecars are not ready" }
        Write-Warning "One or more multimodal sidecars are unavailable. Typed questions and available fallbacks will remain operational."
    }
}

if (Test-LocalPort $UiPort) { throw "UI port $UiPort is already in use" }
if (Test-LocalPort $ApiPort) { throw "API port $ApiPort is already in use" }

$env:VITALSSIGHT_ASSISTANT_MODEL = $Model
$env:VITALSSIGHT_ASSISTANT_VISION_MODEL = $VisionModel
$env:VITALSSIGHT_ASSISTANT_ASR_MODEL = $AsrModel
$env:VITALSSIGHT_OLLAMA_URL = "http://127.0.0.1:11434"
$env:VITALSSIGHT_ASSISTANT_ACTIONS_ENABLED = if ($EnableReviewActions) { "true" } else { "false" }
$ResolvedDbPath = if ($DbPath) { [System.IO.Path]::GetFullPath($DbPath) } else { Join-Path $Runtime "vitalsight_console.db" }
$ResolvedUploadDir = if ($UploadDir) { [System.IO.Path]::GetFullPath($UploadDir) } else { Join-Path $Runtime "uploads" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ResolvedDbPath) | Out-Null
New-Item -ItemType Directory -Force -Path $ResolvedUploadDir | Out-Null
$env:VITALSSIGHT_DB_PATH = $ResolvedDbPath
$env:VITALSSIGHT_UPLOAD_DIR = $ResolvedUploadDir
$AuthMode = if ($env:VITALSSIGHT_AUTH_MODE) { $env:VITALSSIGHT_AUTH_MODE.ToLowerInvariant() } else { "disabled" }
if ($AuthMode -notin @("disabled", "required")) {
    throw "VITALSSIGHT_AUTH_MODE must be disabled or required"
}
if ($AuthMode -eq "disabled") {
    Write-Warning "VitalsSight is starting with the local development identity. Do not use this mode for a controlled multi-user trial."
}

$ApiLauncher = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "uvicorn", "app.api_server:app", "--host", "127.0.0.1", "--port", "$ApiPort"
) -WorkingDirectory $Project -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $Logs "assistant_api.stdout.log") `
    -RedirectStandardError (Join-Path $Logs "assistant_api.stderr.log")

$UiLauncher = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", "$UiPort", "--server.headless", "true"
) -WorkingDirectory $Project -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $Logs "assistant_ui.stdout.log") `
    -RedirectStandardError (Join-Path $Logs "assistant_ui.stderr.log")

$ApiHealthy = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/v1/assistant/health" -TimeoutSec 2
        if ($Health.status) {
            $ApiHealthy = $true
            break
        }
    } catch { }
}

$UiHealthy = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $UiResponse = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$UiPort" -TimeoutSec 2
        if ($UiResponse.StatusCode -eq 200) {
            $UiHealthy = $true
            break
        }
    } catch { }
}

if (-not $ApiHealthy -or -not $UiHealthy) {
    Stop-Listener $ApiPort
    Stop-Listener $UiPort
    throw "VitalsSight failed startup health checks (API=$ApiHealthy, UI=$UiHealthy). Inspect runtime/logs."
}

$ApiPid = Get-ListenerPid $ApiPort
$UiPid = Get-ListenerPid $UiPort
if (-not $ApiPid -or -not $UiPid) {
    Stop-Listener $ApiPort
    Stop-Listener $UiPort
    throw "VitalsSight listeners disappeared after startup health checks."
}

@{
    api_pid = $ApiPid
    ui_pid = $UiPid
    api_port = $ApiPort
    ui_port = $UiPort
    api_url = "http://127.0.0.1:$ApiPort"
    ui_url = "http://127.0.0.1:$UiPort"
    model = $Model
    vision_model = $VisionModel
    asr_model = $AsrModel
    actions_enabled = [bool]$EnableReviewActions
    auth_mode = $AuthMode
    db_path = $ResolvedDbPath
    upload_dir = $ResolvedUploadDir
    started_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding UTF8

Write-Output "VitalsSight UI: http://127.0.0.1:$UiPort"
Write-Output "VitalsSight API: http://127.0.0.1:$ApiPort/docs"
Write-Output "Assistant model: $Model"
Write-Output "Vision model: $VisionModel"
Write-Output "Speech model: $AsrModel"
Write-Output "Review actions enabled: $([bool]$EnableReviewActions)"
Write-Output "Authentication mode: $AuthMode"
