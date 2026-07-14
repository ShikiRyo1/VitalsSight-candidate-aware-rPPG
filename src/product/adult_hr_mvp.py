from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.baselines.traditional_rppg import METHODS
from src.data.video_io import get_video_metadata, iter_video_frames
from src.selection.roi_evidence import build_roi_candidate_clusters, select_roi_supported_clusters_v2
from src.signal.estimate import estimate_hr
from src.vision.face_mesh_roi import (
    FACE_LANDMARKER_MODEL_SHA256,
    FaceRegionMask,
    MediaPipeFaceLandmarkDetector,
    TCM_INTERPRETIVE_LABELS,
    draw_face_region_masks,
    extract_region_rgb_features,
    mentor_aligned_face_rois,
)
from src.vision.face_roi_timeseries import (
    FaceROITimeSeriesConfig,
    extract_face_roi_timeseries_from_video_stream,
)
from src.vision.roi import face_like_rois


@dataclass(frozen=True)
class AdultHRMVPConfig:
    seconds: float = 30.0
    window_sec: float = 20.0
    step_sec: float = 10.0
    frame_stride: int = 1
    min_window_sec: float = 8.0
    min_detection_rate: float = 0.50
    min_candidates: int = 6
    use_mediapipe: bool = True
    fallback_detection_rate: float | None = None
    start_sec: float = 0.0


@dataclass(frozen=True)
class AdultHRMVPResult:
    windows: pd.DataFrame
    candidates: pd.DataFrame
    clusters: pd.DataFrame
    roi_timeseries: pd.DataFrame
    preview_rgb: np.ndarray | None
    metadata: dict[str, object]


def detector_is_release_eligible(detector: MediaPipeFaceLandmarkDetector) -> bool:
    return bool(
        detector.available
        and detector.backend == "mediapipe_face_landmarker_task"
        and detector.model_integrity_status == "verified_pinned_sha256"
        and detector.model_sha256 == FACE_LANDMARKER_MODEL_SHA256
    )


def run_adult_hr_video(video_path: str | Path, *, config: AdultHRMVPConfig | None = None) -> AdultHRMVPResult:
    """Run the product-facing adult HR MVP pipeline on one RGB face video.

    This is intentionally label-free and suitable for product/demo use:
    it outputs a released HR only when the multi-ROI, multi-method candidate
    evidence passes the frozen v2 selector gate. Otherwise it returns a review
    decision with candidate evidence for clinician/caregiver/operator review.
    """

    cfg = config or AdultHRMVPConfig()
    if cfg.frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if cfg.window_sec <= 0 or cfg.step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")

    path = Path(video_path)
    meta = get_video_metadata(path)
    source_fps = float(meta.fps or 30.0)
    start_frame = int(max(0.0, cfg.start_sec) * source_fps)
    max_source_frames = int(min(max(1, meta.frame_count - start_frame), max(1.0, cfg.seconds) * source_fps))
    max_frames = max(1, (max_source_frames + cfg.frame_stride - 1) // cfg.frame_stride)
    analysis_fps = source_fps / cfg.frame_stride

    extraction_cfg = FaceROITimeSeriesConfig(
        fps=source_fps,
        frame_stride=cfg.frame_stride,
        max_frames=max_frames,
        per_frame_landmarks=cfg.use_mediapipe,
    )
    detector_meta: dict[str, object] = {}
    release_eligible_detector = False
    if cfg.use_mediapipe:
        with MediaPipeFaceLandmarkDetector() as detector:
            if detector.available:
                release_eligible_detector = detector_is_release_eligible(detector)
                roi_ts, detector_meta = extract_face_roi_timeseries_from_video_stream(
                    path,
                    detector=detector,
                    config=extraction_cfg,
                    start_frame=start_frame,
                )
                detector_meta["detector_backend"] = detector.backend
                detector_meta["detector_model_path"] = detector.model_path
                detector_meta["detector_model_sha256"] = detector.model_sha256
                detector_meta["detector_model_integrity"] = detector.model_integrity_status
                detector_meta["release_eligible_detector"] = release_eligible_detector
            else:
                roi_ts = extract_static_face_like_roi_timeseries(
                    path,
                    fps=source_fps,
                    frame_stride=cfg.frame_stride,
                    max_frames=max_frames,
                    start_frame=start_frame,
                )
                detector_meta = {
                    "source_fps": source_fps,
                    "fallback_static_roi": 1.0,
                    "fallback_reason": detector.backend,
                    "detector_backend": "static_roi_fallback",
                    "detector_model_path": detector.model_path,
                    "detector_model_sha256": detector.model_sha256,
                    "detector_model_integrity": detector.model_integrity_status,
                    "detector_initialization_error": detector.initialization_error,
                    "release_eligible_detector": False,
                    "detection_rate": float(cfg.fallback_detection_rate or 0.0),
                }
    else:
        roi_ts = extract_static_face_like_roi_timeseries(
            path,
            fps=source_fps,
            frame_stride=cfg.frame_stride,
            max_frames=max_frames,
            start_frame=start_frame,
        )
        detector_meta = {
            "source_fps": source_fps,
            "fallback_static_roi": 1.0,
            "detector_backend": "static_roi_requested",
            "detection_rate": float(cfg.fallback_detection_rate if cfg.fallback_detection_rate is not None else 1.0),
            "release_eligible_detector": False,
        }

    route_failures: list[dict[str, object]] = []
    candidates = candidate_table_from_roi_timeseries_windows(
        roi_ts,
        sample_id=path.stem,
        fps=analysis_fps,
        window_sec=cfg.window_sec,
        step_sec=cfg.step_sec,
        min_window_sec=cfg.min_window_sec,
        route_failures=route_failures,
    )
    clusters = build_roi_candidate_clusters(candidates) if not candidates.empty else pd.DataFrame()
    selected = select_roi_supported_clusters_v2(candidates) if not candidates.empty else pd.DataFrame()
    windows = build_release_windows(
        candidates,
        selected,
        cfg=cfg,
        detection_rate=float(detector_meta.get("detection_rate", 0.0)),
        release_eligible_detector=release_eligible_detector,
    )
    preview = first_face_preview(path, start_frame=start_frame, use_mediapipe=cfg.use_mediapipe)
    metadata = {
        "video_path": str(path),
        "video_fps": source_fps,
        "analysis_fps": analysis_fps,
        "frame_count": int(meta.frame_count),
        "duration_sec": float(meta.duration_sec),
        "max_source_frames": max_source_frames,
        "max_analysis_frames": max_frames,
        "start_frame": start_frame,
        "config": asdict(cfg),
        "detector_meta": detector_meta,
        "n_roi_rows": int(len(roi_ts)),
        "n_candidates": int(len(candidates)),
        "n_clusters": int(len(clusters)),
        "n_windows": int(len(windows)),
        "route_failure_count": len(route_failures),
        "route_failures": route_failures,
    }
    return AdultHRMVPResult(windows=windows, candidates=candidates, clusters=clusters, roi_timeseries=roi_ts, preview_rgb=preview, metadata=metadata)


def candidate_table_from_roi_timeseries_windows(
    roi_ts: pd.DataFrame,
    *,
    sample_id: str,
    fps: float,
    window_sec: float,
    step_sec: float,
    min_window_sec: float = 8.0,
    route_failures: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    if roi_ts.empty or fps <= 0:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    min_t = float(pd.to_numeric(roi_ts["timestamp_s"], errors="coerce").min())
    max_t = float(pd.to_numeric(roi_ts["timestamp_s"], errors="coerce").max())
    if not np.isfinite(min_t) or not np.isfinite(max_t) or max_t <= min_t:
        return pd.DataFrame()
    window_id = 0
    start = min_t
    while start + min_window_sec <= max_t + 1e-6:
        end = min(start + window_sec, max_t + 1.0 / max(fps, 1e-6))
        window = roi_ts[(roi_ts["timestamp_s"] >= start) & (roi_ts["timestamp_s"] < end)].copy()
        rows.extend(
            _candidate_rows_for_window(
                window,
                sample_id=f"{sample_id}_w{window_id:03d}",
                fps=fps,
                window_id=window_id,
                start_sec=start,
                end_sec=end,
                route_failures=route_failures,
            )
        )
        window_id += 1
        start += step_sec
        if start >= max_t:
            break
    return pd.DataFrame(rows)


def _candidate_rows_for_window(
    window: pd.DataFrame,
    *,
    sample_id: str,
    fps: float,
    window_id: int,
    start_sec: float,
    end_sec: float,
    route_failures: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region, group in window.groupby("region", sort=True):
        rgb = group.sort_values("frame_index")[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
        if len(rgb) < max(64, int(fps * 8)):
            continue
        for method_name, method_fn in METHODS.items():
            try:
                signal_values = method_fn(rgb)
                estimate = estimate_hr(signal_values, fps)
            except Exception as error:
                if route_failures is not None:
                    route_failures.append(
                        {
                            "window_id": int(window_id),
                            "region": str(region),
                            "method": str(method_name),
                            "error_type": type(error).__name__,
                            "error_message": str(error)[:240],
                        }
                    )
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "window_id": int(window_id),
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "region": region,
                    "method": method_name,
                    "candidate_bpm": estimate.bpm,
                    "peak_hz": estimate.peak_hz,
                    "confidence": estimate.confidence,
                    "power": estimate.band_power,
                    "total_power": estimate.total_power,
                }
            )
    return rows


def build_release_windows(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    cfg: AdultHRMVPConfig,
    detection_rate: float,
    release_eligible_detector: bool = True,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            [
                {
                    "window_id": 0,
                    "product_hr_bpm": np.nan,
                    "accepted": False,
                    "decision": "review",
                    "refusal_reason": "no_candidate_generated",
                    "detection_rate": detection_rate,
                }
            ]
        )
    peaks = (
        candidates.sort_values(["sample_id", "power", "confidence"], ascending=[True, False, False])
        .groupby("sample_id", as_index=False, sort=True)
        .head(1)
    )
    selected_by_sample = {str(row["sample_id"]): row for _, row in selected.iterrows()} if not selected.empty else {}
    rows: list[dict[str, object]] = []
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        peak = peaks[peaks["sample_id"].astype(str).eq(str(sample_id))].head(1)
        peak_row = peak.iloc[0] if not peak.empty else None
        chosen = selected_by_sample.get(str(sample_id))
        if chosen is None:
            product_hr = np.nan
            accepted = False
            reason = "no_selected_cluster"
            score = np.nan
            roi_support = 0
            method_support = 0
            gate = 0
        else:
            product_hr = float(chosen.get("cluster_bpm", np.nan))
            score = float(chosen.get("roi_evidence_v2_score", np.nan))
            roi_support = int(chosen.get("roi_support", 0))
            method_support = int(chosen.get("method_support", 0))
            gate = int(chosen.get("passes_roi_evidence_v2_gate", 0))
            accepted = bool(
                gate
                and release_eligible_detector
                and detection_rate >= cfg.min_detection_rate
                and group.shape[0] >= cfg.min_candidates
                and np.isfinite(product_hr)
                and 45.0 <= product_hr <= 180.0
            )
            reason = "accepted" if accepted else _refusal_reason(
                gate=gate,
                detection_rate=detection_rate,
                candidate_count=group.shape[0],
                cfg=cfg,
                product_hr=product_hr,
                release_eligible_detector=release_eligible_detector,
            )
        rows.append(
            {
                "sample_id": sample_id,
                "window_id": int(group["window_id"].iloc[0]) if "window_id" in group.columns else len(rows),
                "start_sec": float(group["start_sec"].iloc[0]) if "start_sec" in group.columns else np.nan,
                "end_sec": float(group["end_sec"].iloc[0]) if "end_sec" in group.columns else np.nan,
                "product_hr_bpm": product_hr if accepted else np.nan,
                "candidate_hr_bpm": product_hr,
                "accepted": accepted,
                "decision": "release" if accepted else "review",
                "refusal_reason": reason,
                "roi_evidence_v2_score": score,
                "roi_support": roi_support,
                "method_support": method_support,
                "passes_roi_evidence_v2_gate": gate,
                "detection_rate": detection_rate,
                "release_eligible_detector": release_eligible_detector,
                "candidate_count": int(group.shape[0]),
                "max_power_candidate_bpm": float(peak_row["candidate_bpm"]) if peak_row is not None else np.nan,
                "max_power_region": str(peak_row["region"]) if peak_row is not None else "",
                "max_power_method": str(peak_row["method"]) if peak_row is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _refusal_reason(
    *,
    gate: int,
    detection_rate: float,
    candidate_count: int,
    cfg: AdultHRMVPConfig,
    product_hr: float,
    release_eligible_detector: bool = True,
) -> str:
    if not release_eligible_detector:
        return "face_landmark_backend_not_release_eligible"
    if detection_rate < cfg.min_detection_rate:
        return "low_face_detection_rate"
    if candidate_count < cfg.min_candidates:
        return "too_few_candidates"
    if not np.isfinite(product_hr) or product_hr < 45.0 or product_hr > 180.0:
        return "outside_adult_physiology_range"
    if not gate:
        return "insufficient_multi_roi_candidate_support"
    return "review_required"


def first_face_preview(video_path: Path, *, start_frame: int = 0, use_mediapipe: bool = True) -> np.ndarray | None:
    for frame_index, frame in iter_video_frames(video_path, max_frames=max(1, start_frame + 1), sample_every=1, convert_rgb=True):
        if frame_index < start_frame:
            continue
        landmarks = None
        if use_mediapipe:
            with MediaPipeFaceLandmarkDetector() as detector:
                if detector.available:
                    landmarks = detector.detect(frame)
                    regions = mentor_aligned_face_rois(frame, landmarks=landmarks)
                else:
                    regions = fallback_face_region_masks(frame)
        else:
            regions = fallback_face_region_masks(frame)
        preview_bgr = draw_face_region_masks(frame, regions)
        return cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    return None


def extract_static_face_like_roi_timeseries(
    video_path: str | Path,
    *,
    fps: float,
    frame_stride: int,
    max_frames: int,
    start_frame: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    static_regions: list[FaceRegionMask] | None = None
    accepted = 0
    for frame_index, frame in iter_video_frames(video_path, max_frames=start_frame + max_frames * frame_stride, sample_every=1, convert_rgb=True):
        if frame_index < start_frame:
            continue
        if (frame_index - start_frame) % frame_stride != 0:
            continue
        if accepted >= max_frames:
            break
        if static_regions is None:
            static_regions = fallback_face_region_masks(frame)
        feature_row = extract_region_rgb_features(frame, static_regions)
        timestamp_s = frame_index / fps if fps > 0 else float(accepted)
        for region in static_regions:
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


def fallback_face_region_masks(frame: np.ndarray) -> list[FaceRegionMask]:
    height, width = frame.shape[:2]
    masks: list[FaceRegionMask] = []
    for roi in face_like_rois(frame):
        clamped = roi.clamp(width, height)
        mask = np.zeros((height, width), dtype=bool)
        mask[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w] = True
        masks.append(
            FaceRegionMask(
                name=clamped.name,
                roi=clamped,
                mask=mask,
                landmark_indices=(),
                interpretive_label=TCM_INTERPRETIVE_LABELS.get(clamped.name, "fallback_face_zone"),
                area_px=int(mask.sum()),
                coverage=float(mask.sum() / max(1, width * height)),
            )
        )
    return masks
