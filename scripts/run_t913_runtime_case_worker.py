from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t905_full_pipeline_runtime_profile import (  # noqa: E402
    AdultHRMVPConfig,
    VideoCase,
    json_safe,
    run_case,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--no-mediapipe", action="store_true")
    args = parser.parse_args()

    cfg = AdultHRMVPConfig(
        seconds=args.seconds,
        window_sec=args.seconds,
        step_sec=max(1.0, args.seconds / 2.0),
        frame_stride=args.frame_stride,
        min_window_sec=max(4.0, min(args.seconds - 1.0, 6.0)),
        use_mediapipe=not args.no_mediapipe,
    )
    case = VideoCase(args.dataset, args.case_id, Path(args.video_path), args.note)
    row = run_case(case, cfg=cfg)
    write_json(Path(args.out_json), row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
