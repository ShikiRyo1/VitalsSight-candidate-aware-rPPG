from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.data.video_io import get_video_metadata, iter_video_frames
from src.signal.estimate import RateEstimate, estimate_rr_adult, estimate_rr_infant
from src.vision.body_roi import body_aware_respiration_rois, candidate_respiration_rois, resize_roi_gray
from src.vision.roi import ROI


@dataclass(frozen=True)
class RespirationSignal:
    method: str
    roi: ROI
    values: np.ndarray
    fps: float


@dataclass(frozen=True)
class RespirationPrediction:
    method: str
    roi: ROI
    estimate: RateEstimate
    frames_used: int


@dataclass(frozen=True)
class WindowRatePrediction:
    method: str
    roi: ROI
    estimate: RateEstimate
    start_sec: float
    end_sec: float
    frame_start: int
    frame_end: int


def first_frame(path: str | Path) -> np.ndarray:
    for _, frame in iter_video_frames(path, max_frames=1, convert_rgb=True):
        return frame
    raise RuntimeError(f"No frames could be read from video: {path}")


def motion_energy_signals(
    video_path: str | Path,
    rois: list[ROI],
    *,
    max_frames: int | None = None,
    sample_every: int = 1,
    max_side: int = 120,
) -> list[RespirationSignal]:
    meta = get_video_metadata(video_path)
    fps = meta.fps / sample_every
    traces: dict[str, list[float]] = {roi.name: [] for roi in rois}
    previous: dict[str, np.ndarray | None] = {roi.name: None for roi in rois}
    roi_lookup = {roi.name: roi for roi in rois}

    for _, frame in iter_video_frames(video_path, max_frames=max_frames, sample_every=sample_every, convert_rgb=True):
        for roi in rois:
            gray = resize_roi_gray(frame, roi, max_side=max_side)
            prev = previous[roi.name]
            value = 0.0 if prev is None or prev.shape != gray.shape else float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32))))
            traces[roi.name].append(value)
            previous[roi.name] = gray

    return [
        RespirationSignal(method="motion_energy", roi=roi_lookup[name], values=np.asarray(values, dtype=float), fps=fps)
        for name, values in traces.items()
        if values
    ]


def optical_flow_signals(
    video_path: str | Path,
    roi: ROI,
    *,
    max_frames: int | None = None,
    sample_every: int = 1,
    max_side: int = 120,
) -> dict[str, RespirationSignal]:
    meta = get_video_metadata(video_path)
    fps = meta.fps / sample_every
    prev_gray: np.ndarray | None = None
    y_values: list[float] = []
    mag_values: list[float] = []

    for _, frame in iter_video_frames(video_path, max_frames=max_frames, sample_every=sample_every, convert_rgb=True):
        gray = resize_roi_gray(frame, roi, max_side=max_side)
        if prev_gray is None or prev_gray.shape != gray.shape:
            y_values.append(0.0)
            mag_values.append(0.0)
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        dy = flow[..., 1]
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        y_values.append(float(np.mean(dy)))
        mag_values.append(float(np.mean(mag)))
        prev_gray = gray

    return {
        "optical_flow_y": RespirationSignal("optical_flow_y", roi, np.asarray(y_values, dtype=float), fps),
        "optical_flow_mag": RespirationSignal("optical_flow_mag", roi, np.asarray(mag_values, dtype=float), fps),
    }


def estimate_respiration(signal: RespirationSignal, *, infant: bool = False) -> RateEstimate:
    if infant:
        return estimate_rr_infant(signal.values, signal.fps)
    return estimate_rr_adult(signal.values, signal.fps)


def best_signal_by_confidence(signals: list[RespirationSignal], *, infant: bool = False) -> tuple[RespirationSignal, RateEstimate]:
    if not signals:
        raise ValueError("No respiration signals provided.")
    scored: list[tuple[float, RespirationSignal, RateEstimate]] = []
    for signal in signals:
        estimate = estimate_respiration(signal, infant=infant)
        score = estimate.confidence if np.isfinite(estimate.bpm) else -1.0
        scored.append((score, signal, estimate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]


def window_slices(length: int, fps: float, *, window_sec: float, step_sec: float) -> list[tuple[float, float, int, int]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    window = max(8, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))
    if length < window:
        return [(0.0, length / fps, 0, length)]
    slices: list[tuple[float, float, int, int]] = []
    start = 0
    while start + window <= length:
        end = start + window
        slices.append((start / fps, end / fps, start, end))
        start += step
    return slices


def estimate_signal_windows(
    signal: RespirationSignal,
    *,
    infant: bool = False,
    window_sec: float = 30.0,
    step_sec: float = 10.0,
    method_name: str | None = None,
) -> list[WindowRatePrediction]:
    predictions: list[WindowRatePrediction] = []
    for start_sec, end_sec, start, end in window_slices(len(signal.values), signal.fps, window_sec=window_sec, step_sec=step_sec):
        values = signal.values[start:end]
        estimate = estimate_respiration(RespirationSignal(signal.method, signal.roi, values, signal.fps), infant=infant)
        predictions.append(
            WindowRatePrediction(
                method=method_name or signal.method,
                roi=signal.roi,
                estimate=estimate,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_start=start,
                frame_end=end,
            )
        )
    return predictions


def best_window_predictions(
    signals: list[RespirationSignal],
    *,
    infant: bool = False,
    window_sec: float = 30.0,
    step_sec: float = 10.0,
    method_name: str = "motion_energy_best_window",
    roi_scores: dict[str, float] | None = None,
) -> list[WindowRatePrediction]:
    if not signals:
        return []
    slices = window_slices(min(len(signal.values) for signal in signals), signals[0].fps, window_sec=window_sec, step_sec=step_sec)
    outputs: list[WindowRatePrediction] = []
    roi_scores = roi_scores or {}
    for start_sec, end_sec, start, end in slices:
        candidates: list[tuple[float, RespirationSignal, RateEstimate]] = []
        for signal in signals:
            values = signal.values[start:end]
            estimate = estimate_respiration(RespirationSignal(signal.method, signal.roi, values, signal.fps), infant=infant)
            quality = roi_scores.get(signal.roi.name, 1.0)
            confidence = estimate.confidence if np.isfinite(estimate.bpm) else -1.0
            candidates.append((confidence * (0.5 + 0.5 * quality), signal, estimate))
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, chosen_signal, estimate = candidates[0]
        outputs.append(
            WindowRatePrediction(
                method=method_name,
                roi=chosen_signal.roi,
                estimate=estimate,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_start=start,
                frame_end=end,
            )
        )
    return outputs


def run_respiration_baselines_for_video(
    video_path: str | Path,
    *,
    view: str = "",
    infant: bool = False,
    seconds: float | None = None,
    sample_every: int = 1,
    max_rois: int | None = None,
    max_side: int = 120,
    roi_strategy: str = "all",
) -> tuple[list[RespirationPrediction], list[ROI]]:
    meta = get_video_metadata(video_path)
    max_frames = int(min(meta.frame_count, meta.fps * seconds)) if seconds else None
    frame = first_frame(video_path)
    if roi_strategy == "body_aware":
        rois = body_aware_respiration_rois(frame, view=view, infant=infant, max_rois=max_rois)
    else:
        rois = candidate_respiration_rois(frame, view=view, infant=infant)
        if max_rois is not None:
            rois = rois[:max_rois]
    motion_signals = motion_energy_signals(
        video_path,
        rois,
        max_frames=max_frames,
        sample_every=sample_every,
        max_side=max_side,
    )
    best_motion, best_motion_estimate = best_signal_by_confidence(motion_signals, infant=infant)

    predictions = [
        RespirationPrediction(
            method="motion_energy_best_roi",
            roi=best_motion.roi,
            estimate=best_motion_estimate,
            frames_used=len(best_motion.values),
        )
    ]

    center = motion_signals[0]
    predictions.append(
        RespirationPrediction(
            method="motion_energy_semantic_roi",
            roi=center.roi,
            estimate=estimate_respiration(center, infant=infant),
            frames_used=len(center.values),
        )
    )

    flow_signals = optical_flow_signals(
        video_path,
        best_motion.roi,
        max_frames=max_frames,
        sample_every=sample_every,
        max_side=max_side,
    )
    for signal in flow_signals.values():
        predictions.append(
            RespirationPrediction(
                method=signal.method,
                roi=signal.roi,
                estimate=estimate_respiration(signal, infant=infant),
                frames_used=len(signal.values),
            )
        )

    return predictions, rois
