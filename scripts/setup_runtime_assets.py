from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.face_mesh_roi import (
    DEFAULT_FACE_LANDMARKER_MODEL,
    FACE_LANDMARKER_MODEL_SHA256,
    FACE_LANDMARKER_MODEL_URL,
    face_landmarker_model_sha256,
)


def sha256_file(path: Path) -> str:
    return face_landmarker_model_sha256(path)


def install_model(target: Path, *, source: Path | None = None, force: bool = False) -> dict[str, object]:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not force and sha256_file(target) == FACE_LANDMARKER_MODEL_SHA256:
        return {"status": "already_installed", "path": str(target), "sha256": FACE_LANDMARKER_MODEL_SHA256}

    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        if source is not None:
            shutil.copyfile(source.expanduser().resolve(), temporary)
            origin = str(source.expanduser().resolve())
        else:
            request = urllib.request.Request(FACE_LANDMARKER_MODEL_URL, headers={"User-Agent": "VitalsSight-runtime-setup/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            origin = FACE_LANDMARKER_MODEL_URL
        observed = sha256_file(temporary)
        if observed != FACE_LANDMARKER_MODEL_SHA256:
            raise ValueError(f"Face Landmarker SHA256 mismatch: expected {FACE_LANDMARKER_MODEL_SHA256}, observed {observed}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "installed",
        "path": str(target),
        "sha256": FACE_LANDMARKER_MODEL_SHA256,
        "source": origin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the pinned MediaPipe runtime model used by VitalsSight.")
    parser.add_argument("--target", type=Path, default=DEFAULT_FACE_LANDMARKER_MODEL)
    parser.add_argument("--source", type=Path, help="Optional authorized local copy for offline installation.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = install_model(args.target, source=args.source, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
