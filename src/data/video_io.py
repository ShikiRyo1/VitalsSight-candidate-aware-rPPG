from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import pandas as pd


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def load_sample_index(path: str | Path) -> pd.DataFrame:
    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError(f"Sample index not found: {index_path}")
    return pd.read_csv(index_path, encoding="utf-8-sig")


def get_video_metadata(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0:
        # Valid AVI streams can omit the index used by CAP_PROP_FRAME_COUNT.
        # Count decodable frames so downstream pipelines do not process one frame.
        while cap.grab():
            frame_count += 1
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(video_path, fps, frame_count, width, height, duration)


def iter_video_frames(
    path: str | Path,
    *,
    max_frames: int | None = None,
    sample_every: int = 1,
    convert_rgb: bool = True,
) -> Iterator[tuple[int, object]]:
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    yielded = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                if convert_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def select_sample(index: pd.DataFrame, sample_id: str | None = None, dataset: str | None = None) -> pd.Series:
    data = index
    if sample_id:
        data = data[data["sample_id"] == sample_id]
    if dataset:
        data = data[data["dataset"] == dataset]
    usable = data[data["video_path"].fillna("") != ""]
    if usable.empty:
        raise ValueError("No directly readable video_path sample matched the request.")
    return usable.iloc[0]
