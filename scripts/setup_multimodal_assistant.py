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

from src.assistant.multimodal import FasterWhisperTranscriber
from src.assistant.provider import OllamaVisionProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify VitalsSight image and speech sidecars.")
    parser.add_argument("--vision-model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--skip-vision-pull", action="store_true")
    parser.add_argument("--skip-asr-download", action="store_true")
    args = parser.parse_args()

    executable = shutil.which("ollama")
    if not executable:
        raise SystemExit("Ollama is not installed or is not on PATH: https://ollama.com/download")
    if not args.skip_vision_pull:
        subprocess.run([executable, "pull", args.vision_model], check=True)

    os.environ["VITALSSIGHT_ASSISTANT_VISION_MODEL"] = args.vision_model
    os.environ["VITALSSIGHT_OLLAMA_URL"] = args.base_url
    os.environ["VITALSSIGHT_ASSISTANT_ASR_MODEL"] = args.asr_model
    vision = OllamaVisionProvider(base_url=args.base_url, model=args.vision_model).status()
    speech = FasterWhisperTranscriber(model_name=args.asr_model)
    speech_status = speech.status() if args.skip_asr_download else speech.warmup()
    payload = {
        "vision": {
            "provider": vision.provider,
            "model": vision.model,
            "available": vision.available,
            "details": vision.details,
        },
        "speech": {
            "provider": speech_status.provider,
            "model": speech_status.model,
            "available": speech_status.available,
            "details": speech_status.details,
        },
        "raw_media_policy": "transient_processing_only",
        "next": "Run scripts/start_vitalssight_with_assistant.ps1",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if vision.available and speech_status.available else 1)


if __name__ == "__main__":
    main()
