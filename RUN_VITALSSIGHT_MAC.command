#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$HOME/Library/Logs/VitalsSight"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/macos-launcher.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "VitalsSight macOS launcher"
echo "Repository: $ROOT_DIR"
echo "Log: $LOG_FILE"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is intended for macOS."
  exit 2
fi

choose_python() {
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo
  echo "Python 3.10-3.12 was not found. Opening the official Python macOS download page."
  open "https://www.python.org/downloads/macos/"
  echo "Install Python, then double-click RUN_VITALSSIGHT_MAC.command again."
  read -r -p "Press Return to close..." _
  exit 3
fi

VENV_DIR="$ROOT_DIR/.venv-macos"
STAMP_FILE="$VENV_DIR/.vitalssight_requirements.sha256"
REQ_HASH="$(shasum -a 256 requirements-core.txt | awk '{print $1}')"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating the isolated VitalsSight environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ ! -f "$STAMP_FILE" ]] || [[ "$(cat "$STAMP_FILE")" != "$REQ_HASH" ]]; then
  echo "Installing verified core dependencies. The first run may take several minutes..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r requirements-core.txt
  printf '%s' "$REQ_HASH" > "$STAMP_FILE"
fi

echo "Preparing the pinned face-landmark runtime asset..."
"$VENV_DIR/bin/python" scripts/setup_runtime_assets.py

echo "Starting VitalsSight on this Mac..."
echo "The browser will open automatically. Keep this Terminal window open while using VitalsSight."
exec "$VENV_DIR/bin/python" scripts/run_vitalssight_local.py "$@"
