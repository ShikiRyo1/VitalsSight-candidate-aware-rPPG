from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .face_mesh_roi import FaceRegionMask, extract_region_rgb_features, face_region_masks_from_landmarks, mentor_aligned_face_rois


@dataclass(frozen=True)
class FaceROITimeSeriesConfig:
    fps: float = 30.0
    frame_stride: int = 1
    max_frames: int | None = None
    per_frame_landmarks: bool = False


def extract_face_roi_timeseries_from_frames(
    frames: Iterable[np.ndarray],
    *,
    config: FaceROITimeSeriesConfig | None = None,
    landmarks_by_frame: Sequence[np.ndarray | None] | None = None,
) -> pd.DataFrame:
    """Extract per-frame mentor-aligned face ROI color features.

    If ``per_frame_landmarks`` is false, the first accepted frame defines the
    masks and later frames reuse those masks. This is fast and deterministic for
    smoke tests or stabilized face crops. For moving raw videos, pass
    ``per_frame_landmarks=True`` with a landmarks sequence so each frame can
    build its own ROI masks.
    """

    cfg = config or FaceROITimeSeriesConfig()
    if cfg.frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if cfg.fps <= 0:
        raise ValueError("fps must be positive")

    rows: list[dict[str, float | int | str]] = []
    static_regions: list[FaceRegionMask] | None = None
    accepted = 0
    for frame_index, frame in enumerate(frames):
        if frame_index % cfg.frame_stride != 0:
            continue
        if cfg.max_frames is not None and accepted >= cfg.max_frames:
            break
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frames must be RGB arrays with shape (height, width, 3)")

        landmarks = None
        if landmarks_by_frame is not None and frame_index < len(landmarks_by_frame):
            landmarks = landmarks_by_frame[frame_index]

        if cfg.per_frame_landmarks:
            regions = face_region_masks_from_landmarks(frame, landmarks) if landmarks is not None else mentor_aligned_face_rois(frame)
        else:
            if static_regions is None:
                static_regions = face_region_masks_from_landmarks(frame, landmarks) if landmarks is not None else mentor_aligned_face_rois(frame)
            regions = static_regions

        feature_row = extract_region_rgb_features(frame, regions)
        timestamp_s = frame_index / cfg.fps
        for region in regions:
            prefix = region.name
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "region": region.name,
                    "interpretive_label": region.interpretive_label,
                    "mean_r": float(feature_row.get(f"{prefix}_mean_r", np.nan)),
                    "mean_g": float(feature_row.get(f"{prefix}_mean_g", np.nan)),
                    "mean_b": float(feature_row.get(f"{prefix}_mean_b", np.nan)),
                    "lab_l": float(feature_row.get(f"{prefix}_lab_l", np.nan)),
                    "lab_a": float(feature_row.get(f"{prefix}_lab_a", np.nan)),
                    "lab_b": float(feature_row.get(f"{prefix}_lab_b", np.nan)),
                    "area_px": float(region.area_px),
                    "coverage": float(region.coverage),
                }
            )
        accepted += 1

    return pd.DataFrame(rows)


def iter_video_rgb_frames(video_path: str | Path) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_face_roi_timeseries_from_video(
    video_path: str | Path,
    *,
    config: FaceROITimeSeriesConfig | None = None,
    landmarks_by_frame: Sequence[np.ndarray | None] | None = None,
) -> pd.DataFrame:
    cfg = config or FaceROITimeSeriesConfig()
    return extract_face_roi_timeseries_from_frames(
        iter_video_rgb_frames(video_path),
        config=cfg,
        landmarks_by_frame=landmarks_by_frame,
    )


def extract_face_roi_timeseries_from_video_stream(
    video_path: str | Path,
    *,
    detector: object,
    config: FaceROITimeSeriesConfig | None = None,
    start_frame: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Streaming real-video ROI extraction using a reusable detector.

    This avoids loading a large video into memory. ``detector`` must expose a
    ``detect(frame_rgb) -> landmarks | None`` method, such as
    ``MediaPipeFaceLandmarkDetector``.
    """

    cfg = config or FaceROITimeSeriesConfig()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or cfg.fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: list[dict[str, float | int | str]] = []
    read_frames = 0
    accepted_frames = 0
    detected_frames = 0
    frame_index = start_frame
    try:
        while True:
            if cfg.max_frames is not None and accepted_frames >= cfg.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            read_frames += 1
            if (frame_index - start_frame) % cfg.frame_stride != 0:
                frame_index += 1
                continue
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(frame)
            if landmarks is None:
                frame_index += 1
                accepted_frames += 1
                continue
            detected_frames += 1
            regions = face_region_masks_from_landmarks(frame, landmarks)
            feature_row = extract_region_rgb_features(frame, regions)
            timestamp_s = frame_index / source_fps if source_fps > 0 else frame_index / cfg.fps
            for region in regions:
                prefix = region.name
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": timestamp_s,
                        "region": region.name,
                        "interpretive_label": region.interpretive_label,
                        "mean_r": float(feature_row.get(f"{prefix}_mean_r", np.nan)),
                        "mean_g": float(feature_row.get(f"{prefix}_mean_g", np.nan)),
                        "mean_b": float(feature_row.get(f"{prefix}_mean_b", np.nan)),
                        "lab_l": float(feature_row.get(f"{prefix}_lab_l", np.nan)),
                        "lab_a": float(feature_row.get(f"{prefix}_lab_a", np.nan)),
                        "lab_b": float(feature_row.get(f"{prefix}_lab_b", np.nan)),
                        "area_px": float(region.area_px),
                        "coverage": float(region.coverage),
                    }
                )
            accepted_frames += 1
            frame_index += 1
    finally:
        cap.release()

    meta = {
        "source_fps": source_fps,
        "read_frames": float(read_frames),
        "accepted_frames": float(accepted_frames),
        "detected_frames": float(detected_frames),
        "detection_rate": float(detected_frames / max(1, accepted_frames)),
    }
    return pd.DataFrame(rows), meta
