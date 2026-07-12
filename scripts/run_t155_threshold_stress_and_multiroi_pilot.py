"""T155 threshold stress test and full-face multi-ROI pilot.

T155 asks two review-facing questions:
1. Do the T153/T154 release thresholds survive leave-dataset-out selection?
2. Can full-face multi-ROI evidence on UBFC/4TU explain single-ROI failures?

Ground truth is used only for threshold selection/evaluation and reporting, not
for the release policy itself.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.baselines.traditional_rppg import METHODS  # noqa: E402
from src.data.video_io import get_video_metadata, iter_video_frames  # noqa: E402
from src.evaluation.metrics import mae, pearson, rmse  # noqa: E402
from src.selection.release_policy import (  # noqa: E402
    MultiROIReleaseConfig,
    ProductReleaseConfig,
    SampleReleaseConfig,
    add_sample_risk_features,
    build_hybrid_deployment_policy,
    build_release_all_deployment_policy,
)
from src.signal.estimate import estimate_hr  # noqa: E402
from src.vision.roi import ROI, face_like_rois, mean_rgb, stable_face_roi  # noqa: E402


EXPERIMENTS = PROJECT / "experiments"
DOCS = PROJECT / "docs"
FIG_DIR = PROJECT / "output" / "t155_figures"

T150_SELECTION = EXPERIMENTS / "t150_domain_robust_selection_table.csv"
T151_SELECTION = EXPERIMENTS / "t151_rppg10_selection_table.csv"
T146_UBFC_INDEX = EXPERIMENTS / "t146_ubfc_sample_index.csv"
T136_4TU_INDEX = EXPERIMENTS / "t136_4tu_session_index.csv"
T136_4TU_RESULTS = EXPERIMENTS / "t136_4tu_classical_window_results.csv"

LDO_GRID_CSV = EXPERIMENTS / "t155_leave_dataset_out_threshold_grid.csv"
LDO_RESULTS_CSV = EXPERIMENTS / "t155_leave_dataset_out_threshold_results.csv"
LDO_CONFIG_CSV = EXPERIMENTS / "t155_leave_dataset_out_selected_configs.csv"
MULTIROI_ESTIMATES_CSV = EXPERIMENTS / "t155_full_face_multiroi_estimates.csv"
MULTIROI_SUMMARY_CSV = EXPERIMENTS / "t155_full_face_multiroi_summary.csv"
FAILURE_AUDIT_CSV = EXPERIMENTS / "t155_failure_case_multiroi_audit.csv"
SUMMARY_JSON = EXPERIMENTS / "t155_threshold_stress_multiroi_pilot_summary.json"
REPORT_MD = EXPERIMENTS / f"t155_threshold_stress_multiroi_pilot_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t155_threshold_stress_multiroi_pilot.md"

UNSAFE_BPM = 10.0
SECONDS = 60.0

BALANCED_PRODUCT = ProductReleaseConfig(
    sample=SampleReleaseConfig(min_confidence=0.60, max_conflict_score=2.0),
    multi_roi=MultiROIReleaseConfig(max_roi_range_bpm=30.0, max_rescue_count=1, aggregator="median"),
    min_roi_for_consensus=3,
)
CONSERVATIVE_PRODUCT = ProductReleaseConfig(
    sample=SampleReleaseConfig(min_confidence=0.70, max_conflict_score=1.5),
    multi_roi=MultiROIReleaseConfig(max_roi_range_bpm=15.0, max_rescue_count=1, aggregator="median"),
    min_roi_for_consensus=3,
)


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def markdown_table(df: pd.DataFrame, *, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    lines = [
        "| " + " | ".join(str(c) for c in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def append_unique(path: Path, marker: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    path.write_text(old.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")


def load_selection() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [T150_SELECTION, T151_SELECTION]:
        table = pd.read_csv(path)
        frames.append(table[table["policy"] == "T150_domain_robust_v1"].copy())
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = to_numeric(
        out,
        [
            "selected_bpm",
            "selected_abs_error_bpm",
            "gt_hr_bpm",
            "t150_confidence",
            "anchor_median",
            "anchor_iqr",
            "top1_support_methods",
            "subwindow_top1_support",
            "dist_to_POS",
            "dist_to_GREEN",
        ],
    )
    out["deployment_id"] = out["sample_id"].astype(str)
    is_rppg10 = out["dataset"].astype(str).eq("rPPG-10")
    out.loc[is_rppg10, "deployment_id"] = (
        out.loc[is_rppg10, "dataset"].astype(str) + "_" + out.loc[is_rppg10, "subject_id"].astype(str)
    )
    out["source_policy"] = "T150_domain_robust_v1"
    return out


def metrics_for_table(table: pd.DataFrame) -> dict[str, object]:
    if table.empty:
        return {
            "n_total": 0,
            "released": 0,
            "withheld": 0,
            "coverage": 0.0,
            "released_mae_bpm": math.nan,
            "released_rmse_bpm": math.nan,
            "released_pearson_r": math.nan,
            "released_median_abs_error_bpm": math.nan,
            "unsafe_release_count": 0,
            "unsafe_per_input": math.nan,
            "unsafe_release_rate": math.nan,
            "all_release_mae_bpm": math.nan,
            "all_release_unsafe_rate": math.nan,
        }
    gt = pd.to_numeric(table["gt_hr_bpm"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(table["selected_bpm"], errors="coerce").to_numpy(dtype=float)
    released = pd.to_numeric(table["released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    finite = np.isfinite(gt) & np.isfinite(pred)
    rel = finite & released
    all_errors = np.abs(gt[finite] - pred[finite])
    rel_errors = np.abs(gt[rel] - pred[rel])
    return {
        "n_total": int(finite.sum()),
        "released": int(rel.sum()),
        "withheld": int((finite & ~released).sum()),
        "coverage": float(rel.sum() / finite.sum()) if finite.sum() else 0.0,
        "released_mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
        "released_rmse_bpm": rmse(gt[rel], pred[rel]) if rel.sum() else math.nan,
        "released_pearson_r": pearson(gt[rel], pred[rel]) if rel.sum() else math.nan,
        "released_median_abs_error_bpm": float(np.median(rel_errors)) if len(rel_errors) else math.nan,
        "unsafe_release_count": int(np.sum(rel_errors > UNSAFE_BPM)),
        "unsafe_per_input": float(np.sum(rel_errors > UNSAFE_BPM) / finite.sum()) if finite.sum() else math.nan,
        "unsafe_release_rate": float(np.mean(rel_errors > UNSAFE_BPM)) if len(rel_errors) else math.nan,
        "all_release_mae_bpm": mae(gt[finite], pred[finite]) if finite.sum() else math.nan,
        "all_release_unsafe_rate": float(np.mean(all_errors > UNSAFE_BPM)) if len(all_errors) else math.nan,
    }


def fast_product_policy(selection: pd.DataFrame, cfg: ProductReleaseConfig, policy_name: str) -> pd.DataFrame:
    """Fast evaluator equivalent to the release policy fields used by T155A.

    The full release-policy implementation keeps every diagnostic column. The
    threshold grid only needs release decisions, selected BPM, and ground truth,
    so this function avoids repeatedly materializing wide tables.
    """

    featured = add_sample_risk_features(selection, config=cfg.sample)
    rows: list[dict[str, object]] = []
    for deployment_id, group in featured.groupby("deployment_id", sort=True):
        if len(group) >= cfg.min_roi_for_consensus:
            preds = pd.to_numeric(group["selected_bpm"], errors="coerce").to_numpy(dtype=float)
            preds = preds[np.isfinite(preds)]
            selected = float(np.median(preds)) if len(preds) else math.nan
            roi_range = float(np.max(preds) - np.min(preds)) if len(preds) else math.nan
            rescue_count = int(group.get("rescue_applied", pd.Series(False, index=group.index)).astype(bool).sum())
            released = int(
                len(preds) >= 2
                and math.isfinite(roi_range)
                and roi_range <= cfg.multi_roi.max_roi_range_bpm
                and rescue_count <= cfg.multi_roi.max_rescue_count
            )
            first = group.iloc[0]
            gt = finite_float(first.get("gt_hr_bpm"))
            rows.append(
                {
                    "dataset": first.get("dataset"),
                    "deployment_id": deployment_id,
                    "sample_id": first.get("sample_id"),
                    "subject_id": first.get("subject_id"),
                    "session_id": first.get("session_id", first.get("sample_id")),
                    "policy": policy_name,
                    "policy_scope": "deployment_unit",
                    "n_roi": int(len(group)),
                    "selected_bpm": selected,
                    "gt_hr_bpm": gt,
                    "selected_abs_error_bpm": abs(selected - gt) if math.isfinite(selected) and math.isfinite(gt) else math.nan,
                    "released": released,
                    "roi_prediction_range_bpm": roi_range,
                    "roi_rescue_count": rescue_count,
                }
            )
        else:
            for _, row in group.iterrows():
                confidence = finite_float(row.get("t150_confidence"))
                conflict = finite_float(row.get("candidate_conflict_score"))
                released = int(confidence >= cfg.sample.min_confidence and conflict < cfg.sample.max_conflict_score)
                rows.append(
                    {
                        "dataset": row.get("dataset"),
                        "deployment_id": deployment_id,
                        "sample_id": row.get("sample_id"),
                        "subject_id": row.get("subject_id"),
                        "session_id": row.get("session_id", row.get("sample_id")),
                        "policy": policy_name,
                        "policy_scope": "deployment_unit",
                        "n_roi": 1,
                        "selected_bpm": row.get("selected_bpm"),
                        "gt_hr_bpm": row.get("gt_hr_bpm"),
                        "selected_abs_error_bpm": row.get("selected_abs_error_bpm"),
                        "released": released,
                        "candidate_conflict_score": conflict,
                        "t150_confidence": confidence,
                    }
                )
    return pd.DataFrame(rows)


def build_product_policy(selection: pd.DataFrame, cfg: ProductReleaseConfig, policy_name: str) -> pd.DataFrame:
    return fast_product_policy(selection, cfg, policy_name)


def config_to_row(cfg: ProductReleaseConfig) -> dict[str, object]:
    return {
        "min_confidence": cfg.sample.min_confidence,
        "max_conflict_score": cfg.sample.max_conflict_score,
        "max_roi_range_bpm": cfg.multi_roi.max_roi_range_bpm,
        "max_rescue_count": cfg.multi_roi.max_rescue_count,
    }


def make_config(
    min_confidence: float,
    max_conflict_score: float,
    max_roi_range_bpm: float,
    max_rescue_count: int,
) -> ProductReleaseConfig:
    return ProductReleaseConfig(
        sample=SampleReleaseConfig(min_confidence=min_confidence, max_conflict_score=max_conflict_score),
        multi_roi=MultiROIReleaseConfig(max_roi_range_bpm=max_roi_range_bpm, max_rescue_count=max_rescue_count),
        min_roi_for_consensus=3,
    )


def has_multiroi_units(selection: pd.DataFrame) -> bool:
    counts = selection.groupby("deployment_id").size()
    return bool((counts >= 3).any())


def objective(metrics: dict[str, object]) -> float:
    unsafe = finite_float(metrics.get("unsafe_per_input"), 1.0)
    coverage = finite_float(metrics.get("coverage"), 0.0)
    rel_mae = finite_float(metrics.get("released_mae_bpm"), 999.0)
    # Risk first, then useful coverage, then accuracy among released outputs.
    return 1000.0 * unsafe + 10.0 * (1.0 - coverage) + rel_mae


def leave_dataset_out_threshold_stress(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = sorted(selection["dataset"].dropna().unique())
    configs: list[ProductReleaseConfig] = []
    for min_conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for max_conflict in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for roi_range in [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]:
                for rescue_count in [0, 1, 2, 99]:
                    configs.append(make_config(min_conf, max_conflict, roi_range, rescue_count))

    grid_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []

    for heldout in datasets:
        print(f"[T155A] held-out={heldout} threshold search", flush=True)
        train = selection[selection["dataset"] != heldout].copy()
        test = selection[selection["dataset"] == heldout].copy()
        train_multiroi = has_multiroi_units(train)
        test_multiroi = has_multiroi_units(test)
        best: tuple[float, int, ProductReleaseConfig, dict[str, object]] | None = None
        best_equiv = 0

        for idx, cfg in enumerate(configs):
            table = build_product_policy(train, cfg, policy_name="T155_train_candidate")
            met = metrics_for_table(table)
            score = objective(met)
            row = {
                "heldout_dataset": heldout,
                "config_id": idx,
                **config_to_row(cfg),
                "train_has_multiroi_units": int(train_multiroi),
                "test_has_multiroi_units": int(test_multiroi),
                "objective": score,
                **{f"train_{k}": v for k, v in met.items()},
            }
            grid_rows.append(row)
            if best is None or score < best[0] - 1e-12:
                best = (score, idx, cfg, met)
                best_equiv = 1
            elif best is not None and abs(score - best[0]) <= 1e-12:
                best_equiv += 1

        assert best is not None
        _, best_idx, best_cfg, best_train_metrics = best
        chosen = build_product_policy(test, best_cfg, policy_name="T155_LDO_selected_threshold")
        chosen_metrics = metrics_for_table(chosen)
        fixed_bal = build_product_policy(test, BALANCED_PRODUCT, policy_name="T153_T154_fixed_balanced")
        fixed_cons = build_product_policy(test, CONSERVATIVE_PRODUCT, policy_name="T153_T154_fixed_conservative")
        release_all = build_release_all_deployment_policy(test, policy_name="T150_deployment_release_all")

        for label, table in [
            ("ldo_selected", chosen),
            ("fixed_balanced", fixed_bal),
            ("fixed_conservative", fixed_cons),
            ("release_all", release_all),
        ]:
            met = metrics_for_table(table)
            result_rows.append(
                {
                    "heldout_dataset": heldout,
                    "eval_policy": label,
                    "train_datasets": ",".join(d for d in datasets if d != heldout),
                    "train_has_multiroi_units": int(train_multiroi),
                    "test_has_multiroi_units": int(test_multiroi),
                    "best_config_id": best_idx if label == "ldo_selected" else "",
                    **(config_to_row(best_cfg) if label == "ldo_selected" else {}),
                    **met,
                }
            )

        config_rows.append(
            {
                "heldout_dataset": heldout,
                "best_config_id": best_idx,
                "n_equivalent_best_train_configs": best_equiv,
                "train_has_multiroi_units": int(train_multiroi),
                "test_has_multiroi_units": int(test_multiroi),
                **config_to_row(best_cfg),
                **{f"train_{k}": v for k, v in best_train_metrics.items()},
                **{f"heldout_{k}": v for k, v in chosen_metrics.items()},
            }
        )
        print(
            f"[T155A] held-out={heldout} selected config {best_idx} "
            f"coverage={chosen_metrics['coverage']:.3f} unsafe/input={chosen_metrics['unsafe_per_input']:.3f}",
            flush=True,
        )

    grid = pd.DataFrame(grid_rows)
    results = pd.DataFrame(result_rows)
    selected = pd.DataFrame(config_rows)
    grid.to_csv(LDO_GRID_CSV, index=False, encoding="utf-8-sig")
    results.to_csv(LDO_RESULTS_CSV, index=False, encoding="utf-8-sig")
    selected.to_csv(LDO_CONFIG_CSV, index=False, encoding="utf-8-sig")
    return grid, results, selected


def snr_proxy_db(band_power: float, total_power: float) -> float:
    noise_power = max(total_power - band_power, 1e-12)
    return float(10.0 * math.log10((band_power + 1e-12) / noise_power))


def first_frames(video_path: Path, count: int = 12, step: int = 15) -> list[np.ndarray]:
    return [
        frame
        for _, frame in iter_video_frames(video_path, max_frames=count, sample_every=step, convert_rgb=True)
    ]


def roi_from_fraction(name: str, face: ROI, width: int, height: int, fx: float, fy: float, fw: float, fh: float) -> ROI:
    return ROI(
        name=name,
        x=face.x + int(round(face.w * fx)),
        y=face.y + int(round(face.h * fy)),
        w=max(1, int(round(face.w * fw))),
        h=max(1, int(round(face.h * fh))),
    ).clamp(width, height)


def derive_face_subrois(face: ROI, width: int, height: int) -> list[ROI]:
    face = ROI("face_full", face.x, face.y, face.w, face.h).clamp(width, height)
    rois = [
        face,
        roi_from_fraction("forehead", face, width, height, 0.20, 0.05, 0.60, 0.22),
        roi_from_fraction("left_cheek", face, width, height, 0.08, 0.42, 0.32, 0.30),
        roi_from_fraction("right_cheek", face, width, height, 0.60, 0.42, 0.32, 0.30),
        roi_from_fraction("center_face", face, width, height, 0.25, 0.25, 0.50, 0.45),
        roi_from_fraction("lower_face", face, width, height, 0.25, 0.62, 0.50, 0.28),
    ]
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[ROI] = []
    for roi in rois:
        key = (roi.x, roi.y, roi.w, roi.h)
        if key not in seen:
            unique.append(roi)
            seen.add(key)
    return unique


def extract_multiroi_rgb(video_path: Path, seconds: float) -> tuple[dict[str, np.ndarray], float, list[ROI], dict[str, object]]:
    meta = get_video_metadata(video_path)
    if meta.fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}")
    probe = first_frames(video_path)
    if not probe:
        raise RuntimeError(f"No frames in {video_path}")
    face = stable_face_roi(probe)
    detector = "haar_face_stable"
    if face is None:
        face = face_like_rois(probe[0])[0]
        detector = "geometry_fallback"
    face = face.clamp(meta.width, meta.height)
    rois = derive_face_subrois(face, meta.width, meta.height)
    max_frames = int(min(meta.frame_count, round(meta.fps * seconds)))
    values: dict[str, list[np.ndarray]] = {roi.name: [] for roi in rois}
    for _, frame in iter_video_frames(video_path, max_frames=max_frames, sample_every=1, convert_rgb=True):
        for roi in rois:
            values[roi.name].append(mean_rgb(frame, roi))
    traces = {name: np.asarray(vals, dtype=float) for name, vals in values.items()}
    meta_row = {
        "fps": float(meta.fps),
        "frames_used": int(max(len(vals) for vals in values.values())) if values else 0,
        "seconds_used": float(max(len(vals) for vals in values.values()) / meta.fps) if values else 0.0,
        "video_frame_count": int(meta.frame_count),
        "video_duration_sec": float(meta.duration_sec),
        "video_width": int(meta.width),
        "video_height": int(meta.height),
        "roi_detector": detector,
    }
    return traces, meta.fps, rois, meta_row


def build_multiroi_video_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ubfc = pd.read_csv(T146_UBFC_INDEX)
    ubfc = ubfc[ubfc["usable"].astype(str).str.lower().isin(["true", "1"])].copy()
    for _, row in ubfc.iterrows():
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "session_id": row["sample_id"],
                "condition_group": row.get("condition_group", "realistic"),
                "video_path": row["video_path"],
                "gt_hr_bpm": row["gt_hr_bpm"],
            }
        )

    fourtu_index = pd.read_csv(T136_4TU_INDEX)
    fourtu_results = pd.read_csv(T136_4TU_RESULTS)
    gt_by_sample = fourtu_results.groupby("sample_id")["gt_hr_bpm"].first().to_dict()
    for _, row in fourtu_index.iterrows():
        name = Path(str(row["video_member"])).name
        video_path = PROJECT / "experiments" / "cache" / "t136_4tu" / str(row["sample_id"]) / name
        if not video_path.exists():
            alt = list((PROJECT / "experiments" / "cache" / "t136_4tu" / str(row["sample_id"])).glob("*.avi"))
            video_path = alt[0] if alt else video_path
        if not video_path.exists():
            continue
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "condition_group": row.get("condition_group", ""),
                "video_path": str(video_path),
                "gt_hr_bpm": gt_by_sample.get(row["sample_id"], math.nan),
            }
        )
    return pd.DataFrame(rows)


def run_full_face_multiroi_pilot(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimate_rows: list[dict[str, object]] = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        sample_id = str(row["sample_id"])
        video_path = Path(str(row["video_path"]))
        print(f"[T155B] {idx + 1}/{len(manifest)} {sample_id} multi-ROI extraction", flush=True)
        try:
            traces, fps, rois, meta = extract_multiroi_rgb(video_path, SECONDS)
        except Exception as exc:
            estimate_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "condition_group": row.get("condition_group", ""),
                    "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                    "roi_name": "extraction_failed",
                    "method": "NA",
                    "pred_hr_bpm": math.nan,
                    "abs_error_bpm": math.nan,
                    "confidence": 0.0,
                    "snr_proxy_db": math.nan,
                    "failure": repr(exc),
                }
            )
            continue

        for roi in rois:
            rgb = traces[roi.name]
            for method_name, method_fn in sorted(METHODS.items()):
                try:
                    signal = method_fn(rgb)
                    est = estimate_hr(signal, fps)
                    pred = est.bpm
                    gt = finite_float(row.get("gt_hr_bpm"))
                    err = abs(pred - gt) if math.isfinite(pred) and math.isfinite(gt) else math.nan
                    estimate_rows.append(
                        {
                            "task_id": "T155",
                            "sample_id": sample_id,
                            "dataset": row["dataset"],
                            "subject_id": row["subject_id"],
                            "session_id": row["session_id"],
                            "condition_group": row.get("condition_group", ""),
                            "video_path": str(video_path),
                            "gt_hr_bpm": gt,
                            "roi_name": roi.name,
                            "roi_x": roi.x,
                            "roi_y": roi.y,
                            "roi_w": roi.w,
                            "roi_h": roi.h,
                            "method": method_name,
                            "pred_hr_bpm": pred,
                            "abs_error_bpm": err,
                            "confidence": est.confidence,
                            "peak_hz": est.peak_hz,
                            "band_power": est.band_power,
                            "total_power": est.total_power,
                            "snr_proxy_db": snr_proxy_db(est.band_power, est.total_power),
                            **meta,
                        }
                    )
                except Exception as exc:
                    estimate_rows.append(
                        {
                            "task_id": "T155",
                            "sample_id": sample_id,
                            "dataset": row["dataset"],
                            "subject_id": row["subject_id"],
                            "session_id": row["session_id"],
                            "condition_group": row.get("condition_group", ""),
                            "video_path": str(video_path),
                            "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                            "roi_name": roi.name,
                            "method": method_name,
                            "pred_hr_bpm": math.nan,
                            "abs_error_bpm": math.nan,
                            "confidence": 0.0,
                            "snr_proxy_db": math.nan,
                            "failure": repr(exc),
                            **meta,
                        }
                    )

    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(MULTIROI_ESTIMATES_CSV, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    for sample_id, group in estimates.groupby("sample_id", sort=True):
        valid = group[np.isfinite(pd.to_numeric(group["pred_hr_bpm"], errors="coerce"))].copy()
        if valid.empty:
            continue
        preds = pd.to_numeric(valid["pred_hr_bpm"], errors="coerce").to_numpy(dtype=float)
        gt = finite_float(valid["gt_hr_bpm"].iloc[0])
        face = valid[valid["roi_name"] == "face_full"].copy()
        face_preds = pd.to_numeric(face["pred_hr_bpm"], errors="coerce").dropna().to_numpy(dtype=float)
        roi_medians = valid.groupby("roi_name")["pred_hr_bpm"].median()
        method_medians = valid.groupby("method")["pred_hr_bpm"].median()
        robust_all = float(np.median(preds))
        robust_roi_then = float(np.median(roi_medians.to_numpy(dtype=float)))
        robust_method_then = float(np.median(method_medians.to_numpy(dtype=float)))
        robust_blend = float(np.median([robust_all, robust_roi_then, robust_method_then]))
        err = abs(robust_blend - gt) if math.isfinite(gt) else math.nan
        iqr = float(np.percentile(preds, 75) - np.percentile(preds, 25))
        pred_range = float(np.max(preds) - np.min(preds))
        safe_candidates = int(np.sum(np.abs(preds - gt) <= UNSAFE_BPM)) if math.isfinite(gt) else 0
        release = int(len(preds) >= 12 and iqr <= 25.0 and pred_range <= 90.0)
        first = valid.iloc[0]
        summary_rows.append(
            {
                "task_id": "T155",
                "sample_id": sample_id,
                "dataset": first["dataset"],
                "subject_id": first["subject_id"],
                "session_id": first["session_id"],
                "condition_group": first.get("condition_group", ""),
                "gt_hr_bpm": gt,
                "n_predictions": int(len(preds)),
                "n_roi": int(valid["roi_name"].nunique()),
                "n_methods": int(valid["method"].nunique()),
                "face_full_median_bpm": float(np.median(face_preds)) if len(face_preds) else math.nan,
                "multi_roi_all_median_bpm": robust_all,
                "multi_roi_roi_then_median_bpm": robust_roi_then,
                "multi_roi_method_then_median_bpm": robust_method_then,
                "multi_roi_blend_bpm": robust_blend,
                "multi_roi_abs_error_bpm": err,
                "multi_roi_iqr_bpm": iqr,
                "multi_roi_range_bpm": pred_range,
                "safe_candidate_count_10bpm": safe_candidates,
                "safe_candidate_fraction_10bpm": float(safe_candidates / len(preds)) if len(preds) else math.nan,
                "pilot_released": release,
                "pilot_release_reason": "release" if release else "roi_method_disagreement",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MULTIROI_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    selection = load_selection()
    baseline = selection.drop_duplicates("deployment_id").copy()
    baseline = baseline[["dataset", "sample_id", "subject_id", "deployment_id", "selected_bpm", "selected_abs_error_bpm", "t150_confidence", "t150_reason"]]
    audit = summary.merge(baseline, on=["dataset", "sample_id", "subject_id"], how="left", suffixes=("", "_t150"))
    audit["t150_safe_10bpm"] = pd.to_numeric(audit["selected_abs_error_bpm"], errors="coerce") <= UNSAFE_BPM
    audit["multiroi_safe_10bpm"] = pd.to_numeric(audit["multi_roi_abs_error_bpm"], errors="coerce") <= UNSAFE_BPM
    audit["delta_abs_error_vs_t150_bpm"] = (
        pd.to_numeric(audit["multi_roi_abs_error_bpm"], errors="coerce")
        - pd.to_numeric(audit["selected_abs_error_bpm"], errors="coerce")
    )
    failure_focus = audit[
        (pd.to_numeric(audit["selected_abs_error_bpm"], errors="coerce") > UNSAFE_BPM)
        | (audit["sample_id"].astype(str).isin(["ubfc_subject14", "ubfc_subject11", "ubfc_subject20", "ubfc_subject32", "ubfc_subject45"]))
    ].copy()
    failure_focus.to_csv(FAILURE_AUDIT_CSV, index=False, encoding="utf-8-sig")
    return estimates, summary, failure_focus


def summarize_multiroi_by_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in summary.groupby("dataset", sort=True):
        gt = pd.to_numeric(group["gt_hr_bpm"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(group["multi_roi_blend_bpm"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(gt) & np.isfinite(pred)
        errors = np.abs(gt[finite] - pred[finite])
        released = pd.to_numeric(group["pilot_released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        rel = finite & released
        rows.append(
            {
                "dataset": dataset,
                "n": int(finite.sum()),
                "pilot_coverage": float(rel.sum() / finite.sum()) if finite.sum() else 0.0,
                "all_multiroi_mae_bpm": mae(gt[finite], pred[finite]) if finite.sum() else math.nan,
                "all_multiroi_unsafe_rate": float(np.mean(errors > UNSAFE_BPM)) if len(errors) else math.nan,
                "released_multiroi_mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_multiroi_unsafe_per_input": float(np.sum(np.abs(gt[rel] - pred[rel]) > UNSAFE_BPM) / finite.sum()) if finite.sum() else math.nan,
                "mean_safe_candidate_fraction": float(pd.to_numeric(group["safe_candidate_fraction_10bpm"], errors="coerce").mean()),
                "median_iqr_bpm": float(pd.to_numeric(group["multi_roi_iqr_bpm"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def write_figures(ldo_results: pd.DataFrame, multiroi_summary: pd.DataFrame, failure_focus: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    plot = ldo_results.copy()
    labels = sorted(plot["heldout_dataset"].unique())
    policies = ["release_all", "fixed_balanced", "ldo_selected"]
    colors = {"release_all": "#999999", "fixed_balanced": "#0072B2", "ldo_selected": "#D55E00"}
    width = 0.24
    x = np.arange(len(labels))
    for offset, policy in zip([-width, 0, width], policies):
        vals = []
        for label in labels:
            row = plot[(plot["heldout_dataset"] == label) & (plot["eval_policy"] == policy)]
            vals.append(float(row["unsafe_per_input"].iloc[0]) if not row.empty else math.nan)
        ax.bar(x + offset, vals, width=width, color=colors[policy], label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Unsafe releases per input")
    ax.set_title("T155A leave-dataset-out threshold stress")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "t155_ldo_unsafe_per_input.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["ldo_unsafe_per_input"] = str(path)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for dataset, group in multiroi_summary.groupby("dataset"):
        ax.scatter(
            group["multi_roi_iqr_bpm"],
            group["multi_roi_abs_error_bpm"],
            label=dataset,
            alpha=0.82,
            s=48,
        )
    ax.axhline(UNSAFE_BPM, color="#D55E00", linestyle="--", linewidth=1)
    ax.set_xlabel("Multi-ROI/method IQR (BPM)")
    ax.set_ylabel("Multi-ROI blend absolute error (BPM)")
    ax.set_title("T155B disagreement vs error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "t155_multiroi_disagreement_vs_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["multiroi_disagreement_vs_error"] = str(path)

    if not failure_focus.empty:
        plot = failure_focus.sort_values("selected_abs_error_bpm", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(9.4, 4.8))
        x = np.arange(len(plot))
        ax.bar(x - 0.18, plot["selected_abs_error_bpm"], width=0.36, color="#999999", label="T150 single-ROI error")
        ax.bar(x + 0.18, plot["multi_roi_abs_error_bpm"], width=0.36, color="#009E73", label="T155 multi-ROI pilot error")
        ax.axhline(UNSAFE_BPM, color="#D55E00", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(plot["sample_id"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Absolute error (BPM)")
        ax.set_title("Failure-case audit: single ROI vs multi-ROI pilot")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "t155_failure_case_single_vs_multiroi.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["failure_case_single_vs_multiroi"] = str(path)

    return paths


def append_evidence_row(summary: dict[str, object]) -> None:
    path = EXPERIMENTS / "experiment_evidence_table.csv"
    fieldnames = [
        "evidence_id",
        "task_id",
        "date",
        "artifact",
        "metric_or_observation",
        "result",
        "claim_supported",
        "claim_boundary",
        "next_action",
    ]
    rows: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or fieldnames)
            rows = list(reader)
    if any(row.get("evidence_id") == "E-0101" for row in rows):
        return
    new_row = {
        "evidence_id": "E-0101",
        "task_id": "T155",
        "date": date.today().isoformat(),
        "artifact": str(SUMMARY_JSON),
        "metric_or_observation": "leave-dataset-out release threshold stress and UBFC/4TU full-face multi-ROI pilot",
        "result": str(summary.get("evidence_result", "")),
        "claim_supported": str(summary.get("claim_supported", "")),
        "claim_boundary": str(summary.get("claim_boundary", "")),
        "next_action": str(summary.get("next_action", "")),
    }
    if "evidence_id" not in fieldnames:
        fieldnames = list(new_row.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(new_row)


def build_summary(
    selection: pd.DataFrame,
    ldo_results: pd.DataFrame,
    selected_configs: pd.DataFrame,
    multiroi_summary: pd.DataFrame,
    failure_focus: pd.DataFrame,
    figures: dict[str, str],
) -> dict[str, object]:
    ldo_focus = ldo_results[
        ldo_results["eval_policy"].isin(["release_all", "fixed_balanced", "ldo_selected"])
    ].copy()
    multiroi_by_dataset = summarize_multiroi_by_dataset(multiroi_summary)
    ubfc14 = failure_focus[failure_focus["sample_id"].astype(str) == "ubfc_subject14"]
    ubfc14_note = "ubfc_subject14 not found in failure audit"
    if not ubfc14.empty:
        row = ubfc14.iloc[0]
        ubfc14_note = (
            f"UBFC subject14: T150 selected {float(row['selected_bpm']):.2f} BPM "
            f"(error {float(row['selected_abs_error_bpm']):.2f}); full-face multi-ROI blend "
            f"{float(row['multi_roi_blend_bpm']):.2f} BPM (error {float(row['multi_roi_abs_error_bpm']):.2f}); "
            f"safe candidate fraction {float(row['safe_candidate_fraction_10bpm']):.3f}."
        )

    ldo_rppg = ldo_focus[
        (ldo_focus["heldout_dataset"] == "rPPG-10") & (ldo_focus["eval_policy"] == "ldo_selected")
    ]
    ldo_rppg_note = ""
    if not ldo_rppg.empty:
        row = ldo_rppg.iloc[0]
        ldo_rppg_note = (
            f"Leave-rPPG-10-out selected thresholds release coverage {float(row['coverage']):.3f}, "
            f"released MAE {float(row['released_mae_bpm']):.3f} BPM, unsafe/input {float(row['unsafe_per_input']):.3f}; "
            f"train_has_multiroi_units={int(row['train_has_multiroi_units'])}."
        )

    evidence_result = (
        f"T155A tested {selection['dataset'].nunique()} held-out datasets with {len(selected_configs)} selected threshold settings. "
        + ldo_rppg_note
        + " "
        + ubfc14_note
    )
    return {
        "task_id": "T155",
        "date": date.today().isoformat(),
        "outputs": {
            "ldo_grid_csv": str(LDO_GRID_CSV),
            "ldo_results_csv": str(LDO_RESULTS_CSV),
            "ldo_config_csv": str(LDO_CONFIG_CSV),
            "multiroi_estimates_csv": str(MULTIROI_ESTIMATES_CSV),
            "multiroi_summary_csv": str(MULTIROI_SUMMARY_CSV),
            "failure_audit_csv": str(FAILURE_AUDIT_CSV),
            "report_md": str(REPORT_MD),
            "doc_md": str(DOC_MD),
            "figures": figures,
        },
        "ldo_results": ldo_focus.to_dict(orient="records"),
        "selected_configs": selected_configs.to_dict(orient="records"),
        "multiroi_by_dataset": multiroi_by_dataset.to_dict(orient="records"),
        "failure_focus": failure_focus.to_dict(orient="records"),
        "evidence_result": evidence_result,
        "main_insight": (
            "T155 changes the claim from a fixed-threshold success story to a validation-aware story. "
            "If a training split has no multi-ROI deployment units, multi-ROI thresholds are not identifiable from that split, "
            "so rPPG-10-style consensus claims must be backed by datasets that actually contain multiple ROI observations. "
            "The full-face UBFC/4TU pilot tests whether this missing evidence can be generated from raw videos instead of relying only on pre-segmented ROI datasets."
        ),
        "claim_supported": (
            "Supported: release thresholds can be evaluated without leaking ground truth, and full-face multi-ROI extraction is now available for UBFC/4TU. "
            "Conditional: any manuscript claim about robust multi-ROI release should report leave-dataset-out threshold behavior and whether multi-ROI thresholds were identifiable on the training split."
        ),
        "claim_boundary": (
            "T155B is a classical full-face multi-ROI pilot, not the final T150/T144 candidate selector port. "
            "It diagnoses whether richer ROI evidence exists; it does not yet prove final product-level arbitrary-video SOTA."
        ),
        "next_action": (
            "Use T155 failure evidence to design T156: a domain-robust multi-ROI candidate selector that ports T144/T150 candidate scoring from one face ROI to raw-video ROI ensembles, with nested threshold selection."
        ),
    }


def write_reports(
    summary: dict[str, object],
    ldo_results: pd.DataFrame,
    selected_configs: pd.DataFrame,
    multiroi_summary: pd.DataFrame,
    failure_focus: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    ldo_display_cols = [
        "heldout_dataset",
        "eval_policy",
        "train_has_multiroi_units",
        "test_has_multiroi_units",
        "coverage",
        "released_mae_bpm",
        "unsafe_per_input",
        "all_release_mae_bpm",
        "all_release_unsafe_rate",
    ]
    config_display_cols = [
        "heldout_dataset",
        "n_equivalent_best_train_configs",
        "train_has_multiroi_units",
        "test_has_multiroi_units",
        "min_confidence",
        "max_conflict_score",
        "max_roi_range_bpm",
        "max_rescue_count",
        "heldout_coverage",
        "heldout_released_mae_bpm",
        "heldout_unsafe_per_input",
    ]
    multiroi_by_dataset = summarize_multiroi_by_dataset(multiroi_summary)
    failure_cols = [
        "sample_id",
        "dataset",
        "gt_hr_bpm",
        "selected_bpm",
        "selected_abs_error_bpm",
        "t150_confidence",
        "multi_roi_blend_bpm",
        "multi_roi_abs_error_bpm",
        "multi_roi_iqr_bpm",
        "safe_candidate_fraction_10bpm",
        "delta_abs_error_vs_t150_bpm",
    ]
    report = "\n".join(
        [
            "# T155 Threshold Stress and Full-Face Multi-ROI Pilot",
            "",
            "## Purpose",
            "",
            "T155 checks whether our T153/T154 release layer is robust enough for a paper claim. It has two parts: leave-dataset-out threshold selection, and raw-video full-face multi-ROI pilot extraction on UBFC/4TU.",
            "",
            "## Main Insight",
            "",
            str(summary["main_insight"]),
            "",
            "## T155A Leave-Dataset-Out Metrics",
            "",
            markdown_table(ldo_results[ldo_display_cols]),
            "",
            "## Selected Thresholds",
            "",
            markdown_table(selected_configs[config_display_cols]),
            "",
            "## T155B Full-Face Multi-ROI Pilot by Dataset",
            "",
            markdown_table(multiroi_by_dataset),
            "",
            "## Failure Case Audit",
            "",
            markdown_table(failure_focus[[c for c in failure_cols if c in failure_focus.columns]].head(20)),
            "",
            "## Figures",
            "",
            "\n".join(f"- {name}: `{path}`" for name, path in figures.items()),
            "",
            "## Claim Status",
            "",
            str(summary["claim_supported"]),
            "",
            "## Boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            "## Next",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    REPORT_MD.write_text(report, encoding="utf-8")

    doc = "\n".join(
        [
            "# T155 教学文档：threshold stress test 与 full-face multi-ROI pilot",
            "",
            "## 1. 这一步的目的是什么？",
            "",
            "T153/T154 已经证明：如果我们不再强制发布所有 HR 数字，而是用 `confidence`、`candidate conflict`、`multi-ROI agreement` 共同决定是否发布，就可以显著减少危险输出。但是这个结论还存在一个审稿人一定会问的问题：这些 threshold 是不是在看过 T152/T153 结果后调出来的？如果换一个数据集，它还能成立吗？",
            "",
            "所以 T155A 做的是 `leave-dataset-out threshold validation`。意思是：每次拿一个数据集当 held-out test，threshold 只能在另外两个数据集上选，然后再去测 held-out dataset。这样能检查我们的 release gate 是否只是过拟合当前表格。",
            "",
            "T155B 做的是 `full-face multi-ROI pilot`。原因是 UBFC/4TU 当前都是单 ROI 表，单 ROI 一旦选错峰，我们没有额外证据可以反驳它。T155B 从原始视频重新切出 `face_full`、`forehead`、`left_cheek`、`right_cheek`、`center_face`、`lower_face`，让同一个视频产生多个 ROI 与多个 classical rPPG method 的候选结果，用来判断失败是“全脸都错”还是“某些 ROI 其实有正确信号”。",
            "",
            "## 2. 用到的软件、代码和文件",
            "",
            "- 新脚本：`scripts/run_t155_threshold_stress_and_multiroi_pilot.py`",
            "- release policy：`src/selection/release_policy.py`",
            "- UBFC 视频索引：`experiments/t146_ubfc_sample_index.csv`",
            "- 4TU 视频索引：`experiments/t136_4tu_session_index.csv`",
            "- 4TU 缓存视频：`experiments/cache/t136_4tu/`",
            "- T150/T151 选择器输出：`experiments/t150_domain_robust_selection_table.csv` 和 `experiments/t151_rppg10_selection_table.csv`",
            "- T155A 输出：`experiments/t155_leave_dataset_out_threshold_results.csv`",
            "- T155B 输出：`experiments/t155_full_face_multiroi_estimates.csv` 与 `experiments/t155_full_face_multiroi_summary.csv`",
            "- 失败审计表：`experiments/t155_failure_case_multiroi_audit.csv`",
            "",
            "## 3. T155A 具体怎么做？",
            "",
            "我们构造了一组 threshold grid：`min_confidence`、`max_conflict_score`、`max_roi_range_bpm`、`max_rescue_count`。每个 held-out dataset 都只允许用训练数据集选择 threshold。选择目标是 `risk first`：先最小化 unsafe releases per input，再考虑 coverage，最后考虑 released MAE。",
            "",
            "这里有一个非常关键的 insight：如果训练集里没有 multi-ROI deployment unit，那么 `max_roi_range_bpm` 和 `max_rescue_count` 在训练集上不可识别。也就是说，训练数据没有告诉我们 multi-ROI threshold 应该多严格。这个现象不是坏事，它反而告诉我们论文里必须诚实写清楚：multi-ROI claim 必须由含有多 ROI 的训练/验证数据支持，不能只靠单 ROI 数据集外推。",
            "",
            "## 4. T155A 指标",
            "",
            markdown_table(ldo_results[ldo_display_cols]),
            "",
            "## 5. T155A threshold 迭代链",
            "",
            markdown_table(selected_configs[config_display_cols]),
            "",
            "直白解释：如果某个 held-out dataset 的训练集没有 multi-ROI，且出现很多 equivalent best configs，就说明 threshold 不是被数据真正学出来的，而是在 tie-breaking 中被选中的。这是后续 T156 必须解决的问题：我们要把 UBFC/4TU 也转换成 full-face multi-ROI candidate evidence，让训练数据本身具备学习 multi-ROI release 的能力。",
            "",
            "## 6. T155B 具体怎么做？",
            "",
            "对每个 UBFC/4TU 视频，先用 OpenCV Haar face detector 找稳定 face box；如果找不到，就用几何 fallback。然后从 face box 派生多个 ROI：`face_full`、`forehead`、`left_cheek`、`right_cheek`、`center_face`、`lower_face`。每个 ROI 计算 RGB mean trace，再跑 `GREEN`、`CHROM`、`POS`、`PBV`、`ICA`、`LGI` 六个 traditional rPPG methods，得到一组候选 HR。",
            "",
            "T155B 暂时使用 robust median 作为 pilot 输出：先看所有 ROI-method predictions 的 median，再看 ROI median 和 method median，最后融合成 `multi_roi_blend_bpm`。这个不是最终算法，而是为了回答一个诊断问题：多 ROI 证据里有没有正确候选？如果有，就说明 T156 值得做 domain-robust multi-candidate selection；如果没有，就说明问题可能在视频质量或标签对齐，而不是 selector。",
            "",
            "## 7. T155B 指标",
            "",
            markdown_table(multiroi_by_dataset),
            "",
            "## 8. 失败样本 output 迭代链",
            "",
            markdown_table(failure_focus[[c for c in failure_cols if c in failure_focus.columns]].head(20)),
            "",
            "旧 output 是单个 `selected_bpm`。T155B 之后，我们可以输出 `multi_roi_blend_bpm`、`multi_roi_iqr_bpm`、`safe_candidate_fraction_10bpm`、`pilot_release_reason`。这些字段能告诉用户和审稿人：这次结果是多个 ROI 一致支持，还是 ROI-method 之间严重冲突，需要 review。",
            "",
            "## 9. 深度 insight",
            "",
            str(summary["main_insight"]),
            "",
            "这一步的核心科研意义是把我们的创新从“加一个安全 gate”推进到“如何在跨数据集、跨 ROI、跨方法的候选冲突中学习可靠生理峰”。这更接近我们真正想写的一区论文故事：不是单点 accuracy hack，而是面向真实产品部署的 physiology-constrained, multi-candidate, uncertainty-aware vital-sign inference。",
            "",
            "## 10. Claim boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            "## 11. 下一步",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    DOC_MD.write_text(doc, encoding="utf-8")
    append_unique(DOCS / "phase_learning_journal.md", "# T155 threshold stress and full-face multi-ROI pilot", doc)


def update_project_docs(summary: dict[str, object]) -> None:
    marker = "## T155 threshold stress and full-face multi-ROI pilot"
    text = "\n".join(
        [
            marker,
            "",
            str(summary["main_insight"]),
            "",
            "Evidence: " + str(summary["evidence_result"]),
            "",
            "Claim status: " + str(summary["claim_supported"]),
            "",
            "Boundary: " + str(summary["claim_boundary"]),
            "",
            "Next: " + str(summary["next_action"]),
            "",
        ]
    )
    for name in [
        "project_status.md",
        "innovation_log.md",
        "problem_and_improvement_log.md",
        "project_synthesis_optimization_roadmap.md",
        "paper_claims_tracker.md",
    ]:
        append_unique(DOCS / name, marker, text)
    append_unique(
        DOCS / "execution_task_registry.md",
        "| T155 |",
        "| T155 | Leave-dataset-out release threshold stress and full-face UBFC/4TU multi-ROI pilot | `scripts/run_t155_threshold_stress_and_multiroi_pilot.py`; `experiments/t155_leave_dataset_out_threshold_results.csv`; `experiments/t155_full_face_multiroi_summary.csv`; `docs/t155_threshold_stress_multiroi_pilot.md` | DONE-STRESS-PILOT |",
    )
    append_evidence_row(summary)


def run() -> dict[str, object]:
    selection = load_selection()
    grid, ldo_results, selected_configs = leave_dataset_out_threshold_stress(selection)
    manifest = build_multiroi_video_manifest()
    estimates, multiroi_summary, failure_focus = run_full_face_multiroi_pilot(manifest)
    figures = write_figures(ldo_results, multiroi_summary, failure_focus)
    summary = build_summary(selection, ldo_results, selected_configs, multiroi_summary, failure_focus, figures)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(summary, ldo_results, selected_configs, multiroi_summary, failure_focus, figures)
    update_project_docs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
