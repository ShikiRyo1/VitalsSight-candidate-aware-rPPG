from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.assistant.provider import OllamaProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify the local VitalsSight assistant model.")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--skip-pull", action="store_true")
    args = parser.parse_args()

    executable = shutil.which("ollama")
    if not executable:
        raise SystemExit("Ollama is not installed or is not on PATH: https://ollama.com/download")
    if not args.skip_pull:
        subprocess.run([executable, "pull", args.model], check=True)
    os.environ["VITALSSIGHT_ASSISTANT_MODEL"] = args.model
    os.environ["VITALSSIGHT_OLLAMA_URL"] = args.base_url
    status = OllamaProvider(base_url=args.base_url, model=args.model).status()
    payload = {
        "provider": status.provider,
        "model": status.model,
        "available": status.available,
        "details": status.details,
        "next": "Run scripts/start_vitalssight_with_assistant.ps1",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status.available else 1)


if __name__ == "__main__":
    main()
