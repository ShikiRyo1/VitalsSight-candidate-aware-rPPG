from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.video_io import get_video_metadata
from src.product.adult_hr_mvp import (
    AdultHRMVPConfig,
    build_release_windows,
    candidate_table_from_roi_timeseries_windows,
    fallback_face_region_masks,
)
from src.selection.roi_evidence import build_roi_candidate_clusters, select_roi_supported_clusters_v2
from src.vision.face_mesh_roi import MediaPipeFaceLandmarkDetector, extract_region_rgb_features, face_region_masks_from_landmarks


EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
TASK_ID = "T905"

CASE_CSV = EXP / "t905_full_pipeline_runtime_case_metrics.csv"
STAGE_CSV = EXP / "t905_full_pipeline_runtime_stage_metrics.csv"
CLAIM_CSV = EXP / "t905_full_pipeline_runtime_claim_gate.csv"
SUMMARY_JSON = EXP / "t905_full_pipeline_runtime_summary.json"
DOC_MD = DOCS / "t905_full_pipeline_runtime_profile.md"


@dataclass(frozen=True)
class VideoCase:
    dataset: str
    case_id: str
    video_path: Path
    note: str


def pct(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=float), q))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    lines = [
        "| " + " | ".join(str(c) for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in show.columns) + " |")
    return "\n".join(lines)


def discover_video_cases() -> list[VideoCase]:
    roots = [
        Path("/root/autodl-tmp/datasets"),
        Path("H:/科研/contactless_vitals_project/datasets"),
        Path("H:/科研/datasets"),
        ROOT / "datasets",
    ]
    candidates: list[VideoCase] = []
    for root in roots:
        if not root.exists():
            continue
        fixed = [
            (
                "MCD-rPPG",
                "mcd_iriun_before",
                root / "adult" / "MCD-rPPG" / "video" / "8555_IriunWebcam_before.avi",
                "small MCD IriunWebcam before-exercise stress sample",
            ),
            (
                "MCD-rPPG",
                "mcd_iriun_after",
                root / "adult" / "MCD-rPPG" / "video" / "1181_IriunWebcam_after.avi",
                "small MCD IriunWebcam after-exercise stress sample",
            ),
        ]
        if os.environ.get("T905_INCLUDE_LARGE_VIDEOS") == "1":
            fixed.extend(
                [
                    (
                        "UBFC-rPPG",
                        "ubfc_subject4",
                        root / "adult" / "UBFC-rPPG" / "kaggle_extracted" / "subject4" / "vid.avi",
                        "UBFC-rPPG standard RGB face video, first segment only",
                    ),
                    (
                        "UBFC-Phys-S1-S14",
                        "ubfc_phys_s2_t1",
                        root / "adult" / "UBFC-Phys-S1-S14" / "extracted" / "s2" / "s2" / "vid_s2_T1.avi",
                        "UBFC-Phys real RGB face video, first segment only",
                    ),
                ]
            )
        for dataset, case_id, path, note in fixed:
            if path.exists():
                candidates.append(VideoCase(dataset, case_id, path, note))
        if candidates:
            break
    if len(candidates) < 2:
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.avi"))[:6]:
                candidates.append(VideoCase("auto_discovered", path.stem, path, "fallback discovered AVI"))
            if candidates:
                break
    return candidates[:4]


def decode_sampled_frames(video_path: Path, *, start_frame: int, max_frames: int, frame_stride: int) -> tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    read_frames = 0
    frame_index = start_frame
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            read_frames += 1
            if (frame_index - start_frame) % frame_stride == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frame_index += 1
    finally:
        cap.release()
    return frames, read_frames


def roi_timeseries_from_frames(
    frames: list[np.ndarray],
    landmarks_by_frame: list[np.ndarray | None],
    *,
    fps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        landmarks = landmarks_by_frame[frame_index] if frame_index < len(landmarks_by_frame) else None
        if landmarks is not None:
            try:
                regions = face_region_masks_from_landmarks(frame, landmarks)
            except Exception:
                regions = []
        else:
            regions = []
        if not regions:
            regions = fallback_face_region_masks(frame)
        feature_row = extract_region_rgb_features(frame, regions)
        timestamp_s = frame_index / fps if fps > 0 else float(frame_index)
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
    return pd.DataFrame(rows)


def stage_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def run_case(case: VideoCase, *, cfg: AdultHRMVPConfig) -> dict[str, Any]:
    stage: dict[str, float] = {}

    t0 = time.perf_counter()
    meta = get_video_metadata(case.video_path)
    stage["metadata_ms"] = stage_ms(t0)

    source_fps = float(meta.fps or 30.0)
    start_frame = int(max(0.0, cfg.start_sec) * source_fps)
    max_frames = int(max(1, cfg.seconds) * source_fps / max(1, cfg.frame_stride))
    analysis_fps = source_fps / max(1, cfg.frame_stride)

    t0 = time.perf_counter()
    frames, read_frames = decode_sampled_frames(case.video_path, start_frame=start_frame, max_frames=max_frames, frame_stride=cfg.frame_stride)
    stage["video_decode_and_sampling_ms"] = stage_ms(t0)

    t0 = time.perf_counter()
    landmarks: list[np.ndarray | None] = []
    detector_backend = "mediapipe"
    with MediaPipeFaceLandmarkDetector() as detector:
        if detector.mp is None:
            detector_backend = "fallback_static_roi_no_mediapipe"
            landmarks = [None for _ in frames]
        else:
            for frame in frames:
                landmarks.append(detector.detect(frame))
    stage["face_landmark_detection_ms"] = stage_ms(t0)
    detected_frames = sum(1 for item in landmarks if item is not None)
    detection_rate = float(detected_frames / max(1, len(frames)))

    t0 = time.perf_counter()
    roi_ts = roi_timeseries_from_frames(frames, landmarks, fps=analysis_fps)
    stage["roi_feature_extraction_ms"] = stage_ms(t0)

    t0 = time.perf_counter()
    candidates = candidate_table_from_roi_timeseries_windows(
        roi_ts,
        sample_id=case.case_id,
        fps=analysis_fps,
        window_sec=cfg.window_sec,
        step_sec=cfg.step_sec,
        min_window_sec=cfg.min_window_sec,
    )
    stage["candidate_generation_ms"] = stage_ms(t0)

    t0 = time.perf_counter()
    clusters = build_roi_candidate_clusters(candidates) if not candidates.empty else pd.DataFrame()
    selected = select_roi_supported_clusters_v2(candidates) if not candidates.empty else pd.DataFrame()
    stage["selector_cluster_scoring_ms"] = stage_ms(t0)

    t0 = time.perf_counter()
    windows = build_release_windows(candidates, selected, cfg=cfg, detection_rate=detection_rate)
    stage["release_gate_ms"] = stage_ms(t0)

    t0 = time.perf_counter()
    packet = {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "n_windows": int(len(windows)),
        "n_candidates": int(len(candidates)),
        "decision_counts": windows["decision"].value_counts().to_dict() if "decision" in windows.columns else {},
        "warning": "research runtime profile; not clinical deployment latency",
    }
    json.dumps(packet, ensure_ascii=False)
    stage["api_packet_serialization_ms"] = stage_ms(t0)

    total_ms = sum(stage.values())
    row: dict[str, Any] = {
        "task_id": TASK_ID,
        "dataset": case.dataset,
        "case_id": case.case_id,
        "video_path": str(case.video_path),
        "note": case.note,
        "source_fps": source_fps,
        "analysis_fps": analysis_fps,
        "frame_stride": cfg.frame_stride,
        "configured_seconds": cfg.seconds,
        "video_frame_count": int(meta.frame_count) if int(meta.frame_count) > 0 else pd.NA,
        "video_duration_sec": float(meta.duration_sec) if float(meta.duration_sec) > 0 else pd.NA,
        "read_frames": read_frames,
        "sampled_frames": len(frames),
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
        "detector_backend": detector_backend,
        "n_roi_rows": int(len(roi_ts)),
        "n_candidates": int(len(candidates)),
        "n_clusters": int(len(clusters)),
        "n_selected_clusters": int(len(selected)),
        "n_windows": int(len(windows)),
        "release_count": int(windows["decision"].eq("release").sum()) if "decision" in windows.columns else 0,
        "review_count": int(windows["decision"].eq("review").sum()) if "decision" in windows.columns else int(len(windows)),
        "total_measured_ms": total_ms,
    }
    row.update(stage)
    return row


def aggregate_stage_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    stage_cols = [c for c in cases.columns if c.endswith("_ms") and c not in {"total_measured_ms"}]
    rows: list[dict[str, Any]] = []
    for col in stage_cols:
        values = pd.to_numeric(cases[col], errors="coerce").dropna().astype(float).tolist()
        rows.append(
            {
                "task_id": TASK_ID,
                "stage": col.removesuffix("_ms"),
                "n_cases": len(values),
                "mean_ms": float(np.mean(values)) if values else math.nan,
                "p50_ms": pct(values, 0.50),
                "p95_ms": pct(values, 0.95),
                "p99_ms": pct(values, 0.99),
                "min_ms": min(values) if values else math.nan,
                "max_ms": max(values) if values else math.nan,
                "stage_scope": stage_scope(col),
            }
        )
    total_values = pd.to_numeric(cases["total_measured_ms"], errors="coerce").dropna().astype(float).tolist()
    rows.append(
        {
            "task_id": TASK_ID,
            "stage": "total_measured_pipeline",
            "n_cases": len(total_values),
            "mean_ms": float(np.mean(total_values)) if total_values else math.nan,
            "p50_ms": pct(total_values, 0.50),
            "p95_ms": pct(total_values, 0.95),
            "p99_ms": pct(total_values, 0.99),
            "min_ms": min(total_values) if total_values else math.nan,
            "max_ms": max(total_values) if total_values else math.nan,
            "stage_scope": "Sum of measured online stages for short-window product MVP profile.",
        }
    )
    rows.append(
        {
            "task_id": TASK_ID,
            "stage": "deep_backbone_inference",
            "n_cases": 0,
            "mean_ms": math.nan,
            "p50_ms": math.nan,
            "p95_ms": math.nan,
            "p99_ms": math.nan,
            "min_ms": math.nan,
            "max_ms": math.nan,
            "stage_scope": "Not invoked in this MVP runtime profile; deep/backbone candidates are treated as precomputed evidence in the manuscript experiments.",
        }
    )
    return pd.DataFrame(rows)


def stage_scope(stage_col: str) -> str:
    mapping = {
        "metadata_ms": "OpenCV/video metadata query.",
        "video_decode_and_sampling_ms": "Read and RGB-convert sampled real-video frames from disk.",
        "face_landmark_detection_ms": "MediaPipe face landmark detection on sampled frames; fallback recorded if unavailable.",
        "roi_feature_extraction_ms": "Build face ROI masks and extract per-region RGB/Lab features.",
        "candidate_generation_ms": "Generate ROI/method HR candidates and spectral features.",
        "selector_cluster_scoring_ms": "Build candidate clusters and score multi-ROI evidence.",
        "release_gate_ms": "Apply product release/review gate to selected clusters.",
        "api_packet_serialization_ms": "Serialize a product-style API packet.",
    }
    return mapping.get(stage_col, "")


def environment_summary() -> dict[str, Any]:
    out: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        out["opencv"] = cv2.__version__
    except Exception:
        out["opencv"] = "unknown"
    try:
        import mediapipe as mp  # type: ignore

        out["mediapipe"] = mp.__version__
    except Exception as exc:
        out["mediapipe"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        out["gpu"] = result.stdout.strip() if result.returncode == 0 else "nvidia-smi unavailable"
    except Exception as exc:
        out["gpu"] = f"nvidia-smi unavailable: {exc}"
    return out


def build_claim_gate(cases: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame:
    measured = not cases.empty
    face_measured = measured and "face_landmark_detection_ms" in cases.columns
    total_p95 = float(stage.loc[stage["stage"].eq("total_measured_pipeline"), "p95_ms"].iloc[0]) if not stage.empty else math.nan
    detection_rate_min = float(pd.to_numeric(cases.get("detection_rate", pd.Series(dtype=float)), errors="coerce").min()) if measured else math.nan
    return pd.DataFrame(
        [
            {
                "task_id": TASK_ID,
                "gate": "real_video_pipeline_runtime_measured",
                "passed": bool(measured),
                "evidence": f"cases={len(cases)}; total_p95_ms={total_p95:.3f}" if measured else "No video cases discovered.",
                "manuscript_instruction": "Can report as short-window online MVP runtime profile, not as optimized production latency.",
            },
            {
                "task_id": TASK_ID,
                "gate": "face_detection_runtime_measured",
                "passed": bool(face_measured),
                "evidence": f"min_detection_rate={detection_rate_min:.3f}; stage included face_landmark_detection_ms" if face_measured else "Face detection stage missing.",
                "manuscript_instruction": "Report detection rate with runtime so latency is not separated from detection quality.",
            },
            {
                "task_id": TASK_ID,
                "gate": "deep_backbone_online_runtime_measured",
                "passed": False,
                "evidence": "Deep/backbone inference is not invoked in this product-MVP runtime script.",
                "manuscript_instruction": "Do not claim online deep-backbone latency; say deep-route evidence was precomputed in current experiments unless a separate deep inference benchmark is run.",
            },
        ]
    )


def build_doc(cases: pd.DataFrame, stage: pd.DataFrame, gate: pd.DataFrame, summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# T905 Full Pipeline Runtime Profile",
            "",
            f"Generated: {summary['generated_at']}",
            "",
            "## Conclusion",
            "",
            "- This profile measures a real-video, short-window product MVP path: video sampling, face landmarks, ROI features, candidate generation, selector scoring, release gate, and API serialization.",
            "- It does not measure online deep-backbone inference. The manuscript must not use this table to claim deep-model runtime.",
            "- Runtime should be reported as engineering feasibility evidence, not as clinical deployment validation.",
            "",
            "## Stage Metrics",
            "",
            markdown_table(stage),
            "",
            "## Case Metrics",
            "",
            markdown_table(cases),
            "",
            "## Claim Gates",
            "",
            markdown_table(gate),
            "",
            "## Manuscript Use",
            "",
            "Use this in Methods/Appendix as an end-to-end runtime profile for the current MVP implementation. The main text can mention that online policy overhead is small relative to face/ROI/candidate extraction, but it must retain the boundary that deep/backbone inference was not timed here.",
            "",
        ]
    )


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    cfg = AdultHRMVPConfig(seconds=8.0, window_sec=8.0, step_sec=4.0, frame_stride=3, min_window_sec=6.0, use_mediapipe=True)
    cases = discover_video_cases()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case in cases:
        try:
            print(f"[{TASK_ID}] running {case.dataset}/{case.case_id}: {case.video_path}", flush=True)
            rows.append(run_case(case, cfg=cfg))
            print(f"[{TASK_ID}] completed {case.case_id}", flush=True)
        except Exception as exc:
            errors.append({"case_id": case.case_id, "dataset": case.dataset, "video_path": str(case.video_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{TASK_ID}] error {case.case_id}: {type(exc).__name__}: {exc}", flush=True)

    case_df = pd.DataFrame(rows)
    if case_df.empty:
        stage_df = pd.DataFrame()
        gate_df = build_claim_gate(case_df, stage_df)
    else:
        stage_df = aggregate_stage_metrics(case_df)
        gate_df = build_claim_gate(case_df, stage_df)

    case_df.to_csv(CASE_CSV, index=False, encoding="utf-8-sig")
    stage_df.to_csv(STAGE_CSV, index=False, encoding="utf-8-sig")
    gate_df.to_csv(CLAIM_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "n_cases_discovered": len(cases),
        "n_cases_completed": int(len(case_df)),
        "errors": errors,
        "environment": environment_summary(),
        "decision": "runtime_profile_completed" if not case_df.empty else "runtime_profile_failed_no_cases",
        "claim_boundary": "Short-window online MVP profile; deep/backbone inference is not measured and must not be claimed.",
        "outputs": {
            "case_metrics": str(CASE_CSV.relative_to(ROOT)),
            "stage_metrics": str(STAGE_CSV.relative_to(ROOT)),
            "claim_gate": str(CLAIM_CSV.relative_to(ROOT)),
            "doc": str(DOC_MD.relative_to(ROOT)),
        },
    }
    if not stage_df.empty:
        total = stage_df[stage_df["stage"].eq("total_measured_pipeline")]
        if not total.empty:
            summary["total_mean_ms"] = float(total["mean_ms"].iloc[0])
            summary["total_p50_ms"] = float(total["p50_ms"].iloc[0])
            summary["total_p95_ms"] = float(total["p95_ms"].iloc[0])
    write_json(SUMMARY_JSON, summary)
    DOC_MD.write_text(build_doc(case_df, stage_df, gate_df, summary), encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
