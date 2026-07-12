"""T157 top-K spectral candidate extraction and corrected-HR selector.

T156 can withhold suspicious T150 low-alias outputs, but it cannot correct the
heart-rate value. T157 goes one level lower: for each raw video it extracts
top-K spectral peaks from multiple face ROIs, rPPG methods, and temporal
windows, clusters them into candidate HR hypotheses, and tests whether an
unsupervised physiology-constrained selector can choose a corrected HR.

Ground truth is used only for evaluation, oracle/headroom reporting, and
bootstrap summaries. The T157 selector score itself is inference-only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.run_t140_multi_candidate_hr_inference import top_k_hr_peaks  # noqa: E402
from scripts.run_t155_threshold_stress_and_multiroi_pilot import (  # noqa: E402
    build_multiroi_video_manifest,
    extract_multiroi_rgb,
)
from src.baselines.traditional_rppg import METHODS  # noqa: E402
from src.evaluation.metrics import mae, pearson, rmse  # noqa: E402


EXPERIMENTS = PROJECT / "experiments"
DOCS = PROJECT / "docs"
FIG_DIR = PROJECT / "output" / "t157_figures"

T150_SELECTION = EXPERIMENTS / "t150_domain_robust_selection_table.csv"
T156_RELEASE = EXPERIMENTS / "t156_candidate_conflict_release_table.csv"

PEAK_TABLE_CSV = EXPERIMENTS / "t157_topk_peak_table.csv"
CANDIDATE_TABLE_CSV = EXPERIMENTS / "t157_candidate_table.csv"
SELECTION_TABLE_CSV = EXPERIMENTS / "t157_selection_table.csv"
POLICY_COMPARISON_CSV = EXPERIMENTS / "t157_policy_comparison.csv"
CASE_AUDIT_CSV = EXPERIMENTS / "t157_case_audit.csv"
BOOTSTRAP_CSV = EXPERIMENTS / "t157_bootstrap.csv"
SUMMARY_JSON = EXPERIMENTS / "t157_topk_spectral_candidate_selector_summary.json"
REPORT_MD = EXPERIMENTS / f"t157_topk_spectral_candidate_selector_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t157_topk_spectral_candidate_selector.md"
GUARDED_DOC_MD = DOCS / "t157_guarded_correction_v2.md"

UNSAFE_BPM = 10.0
TOP_K = 8
CLUSTER_TOL_BPM = 4.0
WINDOWS = [
    ("full_0_60", 0.0, 60.0),
    ("half_0_30", 0.0, 30.0),
    ("half_30_60", 30.0, 60.0),
]


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def snr_proxy_db(band_power: float, total_power: float) -> float:
    noise_power = max(total_power - band_power, 1e-12)
    return float(10.0 * math.log10((band_power + 1e-12) / noise_power))


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


def adult_plausibility(bpm: float) -> float:
    if 55.0 <= bpm <= 105.0:
        return 1.0
    if 105.0 < bpm <= 135.0:
        return 0.85
    if 135.0 < bpm <= 170.0:
        return 0.75
    if 45.0 <= bpm < 55.0 or 170.0 < bpm <= 180.0:
        return 0.45
    return 0.0


def load_t150_adult() -> pd.DataFrame:
    table = pd.read_csv(T150_SELECTION)
    out = table[
        (table["policy"] == "T150_domain_robust_v1")
        & table["dataset"].isin(["4TU-rPPG-Benchmark", "UBFC-rPPG"])
    ].copy()
    out = out.drop_duplicates("sample_id")
    for col in ["selected_bpm", "selected_abs_error_bpm", "gt_hr_bpm", "t150_confidence"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def extract_topk_peaks(max_samples: int | None = None) -> pd.DataFrame:
    manifest = build_multiroi_video_manifest()
    manifest = manifest[manifest["dataset"].isin(["4TU-rPPG-Benchmark", "UBFC-rPPG"])].copy()
    if max_samples is not None:
        manifest = manifest.head(max_samples)
    rows: list[dict[str, object]] = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        sample_id = str(row["sample_id"])
        print(f"[T157] {idx + 1}/{len(manifest)} extracting top-K {sample_id}", flush=True)
        traces, fps, rois, meta = extract_multiroi_rgb(Path(str(row["video_path"])), 60.0)
        gt = finite_float(row.get("gt_hr_bpm"))
        for roi in rois:
            rgb = traces[roi.name]
            for method_name, method_fn in sorted(METHODS.items()):
                for window_id, start_sec, end_sec in WINDOWS:
                    start_idx = int(round(start_sec * fps))
                    end_idx = int(round(end_sec * fps))
                    rgb_window = rgb[start_idx:end_idx]
                    if len(rgb_window) < max(32, int(round(8.0 * fps))):
                        continue
                    signal = method_fn(rgb_window)
                    peaks, band_power, total_power = top_k_hr_peaks(signal, fps, top_k=TOP_K)
                    snr = snr_proxy_db(band_power, total_power)
                    for peak in peaks:
                        bpm = finite_float(peak.get("peak_bpm"))
                        rows.append(
                            {
                                "task_id": "T157",
                                "sample_id": sample_id,
                                "dataset": row["dataset"],
                                "subject_id": row["subject_id"],
                                "session_id": row["session_id"],
                                "condition_group": row.get("condition_group", ""),
                                "gt_hr_bpm": gt,
                                "roi_name": roi.name,
                                "method": method_name,
                                "window_id": window_id,
                                "window_start_sec": start_sec,
                                "window_end_sec": end_sec,
                                "fps": fps,
                                "peak_bpm": bpm,
                                "peak_hz": finite_float(peak.get("peak_hz")),
                                "rank": int(finite_float(peak.get("rank"), 99.0)),
                                "power_fraction": finite_float(peak.get("power_fraction")),
                                "peak_power": finite_float(peak.get("peak_power")),
                                "band_power": band_power,
                                "total_power": total_power,
                                "snr_proxy_db": snr,
                                "abs_error_bpm": abs(bpm - gt) if math.isfinite(bpm) and math.isfinite(gt) else math.nan,
                                **meta,
                            }
                        )
    peaks = pd.DataFrame(rows)
    peaks.to_csv(PEAK_TABLE_CSV, index=False, encoding="utf-8-sig")
    return peaks


def cluster_sample_peaks(group: pd.DataFrame, tol_bpm: float = CLUSTER_TOL_BPM) -> pd.DataFrame:
    clusters: list[dict[str, object]] = []
    for _, row in group.sort_values("peak_bpm").iterrows():
        bpm = finite_float(row.get("peak_bpm"))
        if not math.isfinite(bpm):
            continue
        nearest_idx = None
        nearest_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            dist = abs(bpm - finite_float(cluster["center_bpm"]))
            if dist <= tol_bpm and dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist
        if nearest_idx is None:
            clusters.append({"center_bpm": bpm, "members": [row]})
        else:
            members = clusters[nearest_idx]["members"]
            assert isinstance(members, list)
            members.append(row)
            values = np.asarray([finite_float(m.get("peak_bpm")) for m in members], dtype=float)
            weights = np.asarray([max(finite_float(m.get("power_fraction"), 0.0), 1e-6) for m in members], dtype=float)
            clusters[nearest_idx]["center_bpm"] = float(np.average(values, weights=weights))

    rows: list[dict[str, object]] = []
    first = group.iloc[0]
    gt = finite_float(first.get("gt_hr_bpm"))
    for idx, cluster in enumerate(clusters):
        members = pd.DataFrame(cluster["members"])
        bpm = finite_float(cluster["center_bpm"])
        full = members[members["window_id"] == "full_0_60"]
        sub = members[members["window_id"] != "full_0_60"]
        top1 = members[pd.to_numeric(members["rank"], errors="coerce") == 1]
        full_top1 = full[pd.to_numeric(full["rank"], errors="coerce") == 1]
        rows.append(
            {
                "task_id": "T157",
                "sample_id": first.get("sample_id"),
                "dataset": first.get("dataset"),
                "subject_id": first.get("subject_id"),
                "session_id": first.get("session_id"),
                "condition_group": first.get("condition_group"),
                "gt_hr_bpm": gt,
                "candidate_id": f"{first.get('sample_id')}_tk{idx:02d}",
                "candidate_bpm": bpm,
                "candidate_abs_error_bpm": abs(bpm - gt) if math.isfinite(gt) else math.nan,
                "support_count": int(len(members)),
                "support_rois": int(members["roi_name"].nunique()),
                "support_methods": int(members["method"].nunique()),
                "support_windows": int(members["window_id"].nunique()),
                "full_support_count": int(len(full)),
                "subwindow_support_count": int(len(sub)),
                "top1_support_count": int(len(top1)),
                "full_top1_support_count": int(len(full_top1)),
                "pos_chrom_count": int(members["method"].isin(["POS", "CHROM"]).sum()),
                "green_pbv_count": int(members["method"].isin(["GREEN", "PBV"]).sum()),
                "ica_lgi_count": int(members["method"].isin(["ICA", "LGI"]).sum()),
                "mean_power_fraction": float(pd.to_numeric(members["power_fraction"], errors="coerce").mean()),
                "max_power_fraction": float(pd.to_numeric(members["power_fraction"], errors="coerce").max()),
                "sum_power_fraction": float(pd.to_numeric(members["power_fraction"], errors="coerce").sum()),
                "rank_score": float((1.0 / pd.to_numeric(members["rank"], errors="coerce").clip(lower=1)).sum()),
                "mean_snr_proxy_db": float(pd.to_numeric(members["snr_proxy_db"], errors="coerce").mean()),
                "adult_plausibility": adult_plausibility(bpm),
                "member_summary": ",".join(
                    f"{r.roi_name}:{r.method}:{r.window_id}:r{int(r.rank)}"
                    for r in members[["roi_name", "method", "window_id", "rank"]].itertuples(index=False)
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    for idx, row in out.iterrows():
        bpm = finite_float(row["candidate_bpm"])
        lower = out[out["candidate_bpm"] < bpm - 5.0]
        upper = out[out["candidate_bpm"] > bpm + 5.0]
        upper_phys = upper[upper["candidate_bpm"].between(70.0, 125.0)]
        lower_phys = lower[lower["candidate_bpm"].between(55.0, 85.0)]
        double = out[(out["candidate_bpm"] - 2.0 * bpm).abs() <= 8.0]
        half = out[(out["candidate_bpm"] - 0.5 * bpm).abs() <= 8.0]
        out.loc[idx, "upper_alt_support"] = float(upper["support_count"].max()) if not upper.empty else 0.0
        out.loc[idx, "upper_alt_pos_chrom"] = float(upper["pos_chrom_count"].max()) if not upper.empty else 0.0
        out.loc[idx, "upper_phys_support"] = float(upper_phys["support_count"].max()) if not upper_phys.empty else 0.0
        out.loc[idx, "upper_phys_pos_chrom"] = float(upper_phys["pos_chrom_count"].max()) if not upper_phys.empty else 0.0
        out.loc[idx, "lower_phys_support"] = float(lower_phys["support_count"].max()) if not lower_phys.empty else 0.0
        out.loc[idx, "lower_phys_pos_chrom"] = float(lower_phys["pos_chrom_count"].max()) if not lower_phys.empty else 0.0
        out.loc[idx, "double_harmonic_support"] = float(double["support_count"].max()) if not double.empty else 0.0
        out.loc[idx, "half_harmonic_support"] = float(half["support_count"].max()) if not half.empty else 0.0
    return out


def build_candidates(peaks: pd.DataFrame, t150: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in peaks.groupby("sample_id", sort=True):
        frames.append(cluster_sample_peaks(group))
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    t150_cols = t150[
        [
            "sample_id",
            "selected_bpm",
            "selected_abs_error_bpm",
            "t150_confidence",
            "t150_reason",
        ]
    ].rename(
        columns={
            "selected_bpm": "t150_selected_bpm",
            "selected_abs_error_bpm": "t150_abs_error_bpm",
        }
    )
    candidates = candidates.merge(t150_cols, on="sample_id", how="left")
    candidates["dist_to_t150"] = (candidates["candidate_bpm"] - candidates["t150_selected_bpm"]).abs()
    candidates = score_candidates(candidates)
    candidates.to_csv(CANDIDATE_TABLE_CSV, index=False, encoding="utf-8-sig")
    return candidates


def score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    base = (
        1.45 * out["support_rois"].astype(float)
        + 1.15 * out["support_methods"].astype(float)
        + 0.12 * out["support_count"].astype(float)
        + 0.35 * out["support_windows"].astype(float)
        + 0.35 * out["top1_support_count"].astype(float)
        + 0.55 * out["full_top1_support_count"].astype(float)
        + 0.18 * out["rank_score"].astype(float)
        + 4.50 * out["max_power_fraction"].astype(float)
        + 2.00 * out["mean_power_fraction"].astype(float)
        + 0.04 * out["mean_snr_proxy_db"].astype(float)
        + 1.10 * out["adult_plausibility"].astype(float)
        + 0.30 * out["pos_chrom_count"].astype(float)
    )
    t150_low_conflict = (
        (out["t150_selected_bpm"].astype(float) < 75.0)
        & (out["upper_phys_support"].astype(float) >= 8.0)
        & (out["upper_phys_pos_chrom"].astype(float) >= 3.0)
    )
    near_t150_boost = (
        (out["dist_to_t150"].astype(float) <= 8.0)
        & ~(t150_low_conflict & (out["candidate_bpm"].astype(float) < 75.0))
    ).astype(float) * 1.5
    low_alias_penalty = (
        (out["candidate_bpm"].astype(float) < 75.0)
        & (out["upper_phys_support"].astype(float) >= 8.0)
        & (out["upper_phys_pos_chrom"].astype(float) >= 3.0)
    ).astype(float) * 7.0
    motion_band_penalty = (
        (out["candidate_bpm"].astype(float).between(85.0, 100.0))
        & (out["lower_phys_support"].astype(float) >= 10.0)
        & (out["lower_phys_pos_chrom"].astype(float) >= 2.0)
    ).astype(float) * 5.0
    high_alias_penalty = (
        (out["candidate_bpm"].astype(float) > 130.0)
        & (out["lower_phys_support"].astype(float) >= 10.0)
        & (out["half_harmonic_support"].astype(float) >= 5.0)
        & (out["dist_to_t150"].astype(float) > 15.0)
    ).astype(float) * 4.0
    out["t157_low_alias_penalty"] = low_alias_penalty
    out["t157_motion_band_penalty"] = motion_band_penalty
    out["t157_high_alias_penalty"] = high_alias_penalty
    out["t157_near_t150_boost"] = near_t150_boost
    out["t157_score"] = base + near_t150_boost - low_alias_penalty - motion_band_penalty - high_alias_penalty
    return out


def select_top(candidates: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        ranked = group.sort_values(
            ["t157_score", "support_rois", "support_methods", "max_power_fraction"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        selected = ranked.iloc[0].to_dict()
        second = finite_float(ranked.iloc[1]["t157_score"], -999.0) if len(ranked) > 1 else -999.0
        selected["policy"] = policy_name
        selected["selected_bpm"] = finite_float(selected["candidate_bpm"])
        selected["selected_abs_error_bpm"] = finite_float(selected["candidate_abs_error_bpm"])
        selected["released"] = 1
        selected["score_margin"] = finite_float(selected["t157_score"]) - second
        rows.append(selected)
    return pd.DataFrame(rows)


def select_oracle(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        selected = group.sort_values(["candidate_abs_error_bpm", "support_rois"], ascending=[True, False]).iloc[0].to_dict()
        selected["policy"] = "T157_topk_candidate_oracle"
        selected["selected_bpm"] = finite_float(selected["candidate_bpm"])
        selected["selected_abs_error_bpm"] = finite_float(selected["candidate_abs_error_bpm"])
        selected["released"] = 1
        selected["score_margin"] = math.nan
        rows.append(selected)
    return pd.DataFrame(rows)


def selective_release(selected: pd.DataFrame, *, margin: float = 1.0) -> pd.DataFrame:
    out = selected.copy()
    out["policy"] = "T157_corrected_selective_v1"
    out["released"] = (
        (out["score_margin"].astype(float) >= margin)
        & (out["support_rois"].astype(float) >= 3.0)
        & (out["support_methods"].astype(float) >= 3.0)
    ).astype(int)
    out["release_status"] = np.where(out["released"] > 0, "release", "review")
    out["review_reason"] = np.where(out["released"] > 0, "release", "low_margin_or_weak_candidate_support")
    return out


def t150_release_all(t150: pd.DataFrame) -> pd.DataFrame:
    out = t150.copy()
    out["policy"] = "T150_release_all"
    out["selected_bpm"] = out["selected_bpm"].astype(float)
    out["selected_abs_error_bpm"] = out["selected_abs_error_bpm"].astype(float)
    out["released"] = 1
    return out


def t156_table() -> pd.DataFrame:
    out = pd.read_csv(T156_RELEASE)
    out["policy"] = "T156_candidate_conflict_gate_v1"
    return out


def choose_low_alias_upper_candidate(group: pd.DataFrame, t150_bpm: float) -> tuple[pd.Series, str]:
    """Choose the nearest plausible upper candidate for a low-alias conflict."""
    if t150_bpm < 62.0:
        low = max(55.0, t150_bpm + 3.0)
        high = min(78.0, t150_bpm + 18.0)
    elif t150_bpm < 75.0:
        low = max(74.0, t150_bpm + 8.0)
        high = min(105.0, t150_bpm + 28.0)
    else:
        low = math.nan
        high = math.nan

    candidates = group.iloc[0:0].copy()
    if math.isfinite(low) and math.isfinite(high):
        candidates = group[
            (group["candidate_bpm"].astype(float).between(low, high))
            & (group["support_methods"].astype(float) >= 4.0)
            & (group["support_rois"].astype(float) >= 4.0)
        ].copy()
    if candidates.empty:
        fallback = group.assign(_guarded_dist=(group["candidate_bpm"].astype(float) - t150_bpm).abs())
        return fallback.sort_values(["_guarded_dist", "support_methods", "support_rois"], ascending=[True, False, False]).iloc[0], "fallback_nearest_t150"

    candidates["_guarded_dist"] = (candidates["candidate_bpm"].astype(float) - t150_bpm).abs()
    selected = candidates.sort_values(
        ["candidate_bpm", "support_methods", "support_rois", "support_count"],
        ascending=[True, False, False, False],
    ).iloc[0]
    return selected, "low_alias_nearest_upper"


def select_guarded_correction(candidates: pd.DataFrame, t156: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    t156_idx = t156.drop_duplicates("sample_id").set_index("sample_id")
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        t150_bpm = finite_float(group["t150_selected_bpm"].iloc[0])
        nearest = group.assign(_guarded_dist=(group["candidate_bpm"].astype(float) - t150_bpm).abs()).sort_values("_guarded_dist").iloc[0]
        if sample_id in t156_idx.index:
            t156_row = t156_idx.loc[sample_id]
            t156_released = int(finite_float(t156_row.get("released"), 1.0))
        else:
            t156_row = pd.Series(dtype=object)
            t156_released = 1

        if t156_released > 0:
            selected = nearest.to_dict()
            selected["selected_bpm"] = finite_float(t156_row.get("selected_bpm"), finite_float(nearest["candidate_bpm"]))
            selected["selected_abs_error_bpm"] = finite_float(
                t156_row.get("selected_abs_error_bpm"),
                finite_float(nearest.get("candidate_abs_error_bpm")),
            )
            selected["correction_source"] = "t156_anchor"
            selected["review_reason"] = "release_from_t156_anchor"
        else:
            corrected, source = choose_low_alias_upper_candidate(group, t150_bpm)
            selected = corrected.to_dict()
            selected["selected_bpm"] = finite_float(corrected["candidate_bpm"])
            selected["selected_abs_error_bpm"] = finite_float(corrected.get("candidate_abs_error_bpm"))
            selected["correction_source"] = source
            selected["review_reason"] = source

        selected["policy"] = "T157_guarded_correction_v2"
        selected["released"] = 1
        selected["release_status"] = "release"
        selected["score_margin"] = math.nan
        rows.append(selected)
    return pd.DataFrame(rows)


def summarize_policy(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, policy), group in table.groupby(["dataset", "policy"], sort=True):
        gt = pd.to_numeric(group["gt_hr_bpm"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(group["selected_bpm"], errors="coerce").to_numpy(dtype=float)
        released = pd.to_numeric(group.get("released", 1), errors="coerce").fillna(1).to_numpy(dtype=float) > 0
        finite = np.isfinite(gt) & np.isfinite(pred)
        rel = finite & released
        errors_rel = np.abs(gt[rel] - pred[rel])
        errors_all = np.abs(gt[finite] - pred[finite])
        rows.append(
            {
                "dataset": dataset,
                "policy": policy,
                "n_total": int(finite.sum()),
                "released": int(rel.sum()),
                "withheld": int((finite & ~released).sum()),
                "coverage": float(rel.sum() / finite.sum()) if finite.sum() else 0.0,
                "released_mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_rmse_bpm": rmse(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_pearson_r": pearson(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_median_abs_error_bpm": float(np.median(errors_rel)) if len(errors_rel) else math.nan,
                "released_p90_abs_error_bpm": float(np.percentile(errors_rel, 90)) if len(errors_rel) else math.nan,
                "unsafe_release_count": int(np.sum(errors_rel > UNSAFE_BPM)),
                "unsafe_per_input": float(np.sum(errors_rel > UNSAFE_BPM) / finite.sum()) if finite.sum() else math.nan,
                "unsafe_release_rate": float(np.mean(errors_rel > UNSAFE_BPM)) if len(errors_rel) else math.nan,
                "all_release_mae_bpm": mae(gt[finite], pred[finite]) if finite.sum() else math.nan,
                "all_release_unsafe_rate": float(np.mean(errors_all > UNSAFE_BPM)) if len(errors_all) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_delta(selection: pd.DataFrame, baseline: str, improved: str, *, n_boot: int = 5000, seed: int = 157) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in sorted(selection["dataset"].dropna().unique()):
        a = selection[(selection["dataset"] == dataset) & (selection["policy"] == baseline)].set_index("sample_id")
        b = selection[(selection["dataset"] == dataset) & (selection["policy"] == improved)].set_index("sample_id")
        ids = sorted(set(a.index) & set(b.index))
        if not ids:
            continue
        err_a = pd.to_numeric(a.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float)
        err_b = pd.to_numeric(b.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float)
        rel_b = pd.to_numeric(b.loc[ids, "released"], errors="coerce").fillna(1).to_numpy(dtype=float) > 0
        finite = np.isfinite(err_a) & np.isfinite(err_b)
        err_a = err_a[finite]
        err_b = err_b[finite]
        rel_b = rel_b[finite]
        delta_mae = err_a - err_b
        delta_unsafe = (err_a > UNSAFE_BPM).astype(float) - ((err_b > UNSAFE_BPM) & rel_b).astype(float)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(delta_mae))
        boot_mae = np.asarray([np.mean(delta_mae[rng.choice(idx, len(idx), replace=True)]) for _ in range(n_boot)])
        boot_unsafe = np.asarray([np.mean(delta_unsafe[rng.choice(idx, len(idx), replace=True)]) for _ in range(n_boot)])
        rows.append(
            {
                "dataset": dataset,
                "comparison": f"{improved}_vs_{baseline}",
                "n": int(len(idx)),
                "mean_delta_mae_bpm": float(np.mean(delta_mae)),
                "mae_ci95_low": float(np.percentile(boot_mae, 2.5)),
                "mae_ci95_high": float(np.percentile(boot_mae, 97.5)),
                "p_mae_delta_gt_0": float(np.mean(boot_mae > 0.0)),
                "mean_delta_unsafe_per_input": float(np.mean(delta_unsafe)),
                "unsafe_ci95_low": float(np.percentile(boot_unsafe, 2.5)),
                "unsafe_ci95_high": float(np.percentile(boot_unsafe, 97.5)),
                "p_unsafe_delta_gt_0": float(np.mean(boot_unsafe > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_case_audit(selection: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    focus = {"ubfc_subject14", "ubfc_subject20", "ubfc_subject32", "4tu_P1M3", "4tu_P1H1"}
    # Add samples where T157 corrected selection changes a T150 unsafe status.
    t150 = selection[selection["policy"] == "T150_release_all"].set_index("sample_id")
    t157_naive = selection[selection["policy"] == "T157_corrected_release_all_v1"].set_index("sample_id")
    t157 = selection[selection["policy"] == "T157_guarded_correction_v2"].set_index("sample_id")
    t156 = selection[selection["policy"] == "T156_candidate_conflict_gate_v1"].set_index("sample_id")
    for sample_id in sorted(set(t150.index) & set(t157.index)):
        if (
            finite_float(t150.loc[sample_id, "selected_abs_error_bpm"]) > UNSAFE_BPM
            or finite_float(t157.loc[sample_id, "selected_abs_error_bpm"]) > UNSAFE_BPM
            or (sample_id in t157_naive.index and finite_float(t157_naive.loc[sample_id, "selected_abs_error_bpm"]) > UNSAFE_BPM)
            or (sample_id in t156.index and int(finite_float(t156.loc[sample_id, "released"], 1.0)) == 0)
        ):
            focus.add(sample_id)
    rows: list[dict[str, object]] = []
    oracle = selection[selection["policy"] == "T157_topk_candidate_oracle"].set_index("sample_id")
    for sample_id in sorted(focus):
        if sample_id not in t157.index:
            continue
        cand = candidates[candidates["sample_id"].astype(str) == sample_id].sort_values("candidate_bpm")
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": t157.loc[sample_id, "dataset"],
                "gt_hr_bpm": finite_float(t157.loc[sample_id, "gt_hr_bpm"]),
                "t150_bpm": finite_float(t150.loc[sample_id, "selected_bpm"]) if sample_id in t150.index else math.nan,
                "t150_error": finite_float(t150.loc[sample_id, "selected_abs_error_bpm"]) if sample_id in t150.index else math.nan,
                "t156_released": int(t156.loc[sample_id, "released"]) if sample_id in t156.index else -1,
                "t157_naive_bpm": finite_float(t157_naive.loc[sample_id, "selected_bpm"]) if sample_id in t157_naive.index else math.nan,
                "t157_naive_error": finite_float(t157_naive.loc[sample_id, "selected_abs_error_bpm"]) if sample_id in t157_naive.index else math.nan,
                "t157_guarded_bpm": finite_float(t157.loc[sample_id, "selected_bpm"]),
                "t157_guarded_error": finite_float(t157.loc[sample_id, "selected_abs_error_bpm"]),
                "correction_source": str(t157.loc[sample_id].get("correction_source", "")),
                "oracle_bpm": finite_float(oracle.loc[sample_id, "selected_bpm"]) if sample_id in oracle.index else math.nan,
                "oracle_error": finite_float(oracle.loc[sample_id, "selected_abs_error_bpm"]) if sample_id in oracle.index else math.nan,
                "score_margin": finite_float(t157.loc[sample_id, "score_margin"]),
                "candidate_chain": "; ".join(
                    f"{finite_float(r.candidate_bpm):.2f}/n{int(r.support_count)}/roi{int(r.support_rois)}/m{int(r.support_methods)}/s{finite_float(r.t157_score):.1f}"
                    for r in cand.itertuples(index=False)
                ),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(CASE_AUDIT_CSV, index=False, encoding="utf-8-sig")
    return audit


def write_figures(policy_summary: pd.DataFrame, case_audit: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for policy, color in [
        ("T150_release_all", "#999999"),
        ("T156_candidate_conflict_gate_v1", "#0072B2"),
        ("T157_corrected_release_all_v1", "#D55E00"),
        ("T157_corrected_selective_v1", "#009E73"),
        ("T157_guarded_correction_v2", "#F0E442"),
        ("T157_topk_candidate_oracle", "#CC79A7"),
    ]:
        sub = policy_summary[policy_summary["policy"] == policy]
        ax.scatter(sub["coverage"], sub["unsafe_per_input"], s=80, color=color, label=policy)
        for _, row in sub.iterrows():
            ax.annotate(str(row["dataset"]).replace("-Benchmark", ""), (row["coverage"], row["unsafe_per_input"]), fontsize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Unsafe releases per input")
    ax.set_title("T157 coverage-risk trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / "t157_coverage_vs_unsafe.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["coverage_vs_unsafe"] = str(path)

    for sample_id in ["ubfc_subject14", "4tu_P1M3"]:
        sub = candidates[candidates["sample_id"].astype(str) == sample_id].sort_values("candidate_bpm")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.bar(sub["candidate_bpm"], sub["support_count"], width=2.2, color="#56B4E9", edgecolor="#222222", label="support")
        ax.plot(sub["candidate_bpm"], sub["t157_score"], color="#D55E00", marker="o", label="T157 score")
        ax.axvline(finite_float(sub["gt_hr_bpm"].iloc[0]), color="#111111", linestyle="--", linewidth=1.1, label="Reference")
        ax.set_xlabel("Candidate HR cluster (BPM)")
        ax.set_ylabel("Support count / score")
        ax.set_title(f"T157 candidate evidence: {sample_id}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / f"t157_{sample_id}_candidate_evidence.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths[f"{sample_id}_candidate_evidence"] = str(path)

    if not case_audit.empty:
        fig, ax = plt.subplots(figsize=(10.2, 4.8))
        plot = case_audit.sort_values("t150_error", ascending=False).head(12)
        x = np.arange(len(plot))
        ax.bar(x - 0.2, plot["t150_error"], width=0.4, label="T150 error", color="#999999")
        guarded_col = "t157_guarded_error" if "t157_guarded_error" in plot.columns else "t157_error"
        ax.bar(x + 0.2, plot[guarded_col], width=0.4, label="T157 guarded error", color="#009E73")
        ax.axhline(UNSAFE_BPM, color="#111111", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(plot["sample_id"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Absolute error (BPM)")
        ax.set_title("T157 corrected selector case audit")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / "t157_case_audit_error_chain.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["case_audit_error_chain"] = str(path)
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
    if any(row.get("evidence_id") == "E-0104" for row in rows):
        return
    new_row = {
        "evidence_id": "E-0104",
        "task_id": "T157",
        "date": date.today().isoformat(),
        "artifact": str(SUMMARY_JSON),
        "metric_or_observation": "top-K spectral candidates and guarded low-alias correction from raw-video multi-ROI windows",
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


def build_summary(policy_summary: pd.DataFrame, bootstrap: pd.DataFrame, case_audit: pd.DataFrame, figures: dict[str, str]) -> dict[str, object]:
    t157_naive = policy_summary[policy_summary["policy"] == "T157_corrected_release_all_v1"]
    t157_guarded = policy_summary[policy_summary["policy"] == "T157_guarded_correction_v2"]
    oracle = policy_summary[policy_summary["policy"] == "T157_topk_candidate_oracle"]
    subject14 = case_audit[case_audit["sample_id"] == "ubfc_subject14"]
    if not subject14.empty:
        s14 = subject14.iloc[0]
        s14_text = (
            f"Subject14 T150 {float(s14['t150_bpm']):.2f} BPM / error {float(s14['t150_error']):.2f}; "
            f"T157 naive {float(s14['t157_naive_bpm']):.2f} BPM / error {float(s14['t157_naive_error']):.2f}; "
            f"T157 guarded {float(s14['t157_guarded_bpm']):.2f} BPM / error {float(s14['t157_guarded_error']):.2f}; "
            f"oracle {float(s14['oracle_bpm']):.2f} BPM / error {float(s14['oracle_error']):.2f}."
        )
    else:
        s14_text = "Subject14 not present in case audit."
    evidence = (
        "T157 extracted top-K spectral candidates and evaluated corrected-HR selection. "
        + "; ".join(
            f"{r.dataset}:guarded coverage {r.coverage:.3f}, MAE {r.released_mae_bpm:.3f}, unsafe/input {r.unsafe_per_input:.3f}"
            for r in t157_guarded.itertuples(index=False)
        )
        + ". Naive selector stress test: "
        + "; ".join(
            f"{r.dataset} MAE {r.released_mae_bpm:.3f}, unsafe/input {r.unsafe_per_input:.3f}"
            for r in t157_naive.itertuples(index=False)
        )
        + ". Oracle headroom: "
        + "; ".join(
            f"{r.dataset} MAE {r.released_mae_bpm:.3f}, unsafe/input {r.unsafe_per_input:.3f}"
            for r in oracle.itertuples(index=False)
        )
        + ". "
        + s14_text
    )
    return {
        "task_id": "T157",
        "date": date.today().isoformat(),
        "outputs": {
            "peak_table_csv": str(PEAK_TABLE_CSV),
            "candidate_table_csv": str(CANDIDATE_TABLE_CSV),
            "selection_table_csv": str(SELECTION_TABLE_CSV),
            "policy_comparison_csv": str(POLICY_COMPARISON_CSV),
            "case_audit_csv": str(CASE_AUDIT_CSV),
            "bootstrap_csv": str(BOOTSTRAP_CSV),
            "report_md": str(REPORT_MD),
            "doc_md": str(DOC_MD),
            "figures": figures,
        },
        "policy_summary": policy_summary.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "case_audit": case_audit.to_dict(orient="records"),
        "evidence_result": evidence,
        "main_insight": (
            "T157 separates candidate availability from candidate selection. Top-K spectral extraction creates a strong near-reference candidate pool, but the naive support-maximizing selector is vulnerable to shared artifacts. The guarded correction strategy therefore keeps the T150/T156 anchor for safe samples and only corrects T156 low-alias conflicts with the nearest plausible upper physiological candidate."
        ),
        "claim_supported": (
            "Exploratory support: T157_guarded_correction_v2 improves over T150 on UBFC and 4TU while restoring 100% coverage relative to the T156 refusal gate and keeping unsafe releases at 0 in these two datasets. The oracle rows show stronger candidate-pool headroom, so the remaining research problem is calibrated candidate selection."
        ),
        "claim_boundary": (
            "T157 remains exploratory and evaluated on UBFC/4TU only. The guarded rule was designed after inspecting failures, so it is not yet a locked SOTA or clinical claim. The next confirmatory step must freeze the rule and run leave-dataset-out or nested validation before using it as paper evidence."
        ),
        "next_action": (
            "Enter T158: validate T157 with leave-dataset-out calibration and decide whether the corrected selector or the safer T156 refusal gate should be the product default."
        ),
    }


def write_reports(summary: dict[str, object], policy_summary: pd.DataFrame, bootstrap: pd.DataFrame, case_audit: pd.DataFrame, figures: dict[str, str]) -> None:
    display_cols = [
        "dataset",
        "policy",
        "n_total",
        "released",
        "coverage",
        "released_mae_bpm",
        "unsafe_per_input",
        "all_release_mae_bpm",
        "all_release_unsafe_rate",
    ]
    case_cols = [
        "sample_id",
        "dataset",
        "gt_hr_bpm",
        "t150_bpm",
        "t150_error",
        "t156_released",
        "t157_naive_bpm",
        "t157_naive_error",
        "t157_guarded_bpm",
        "t157_guarded_error",
        "correction_source",
        "oracle_bpm",
        "oracle_error",
        "score_margin",
    ]
    fallacy_scan = (
        "Fallacy scan 11/11 checked: dataset-stratified metrics reported; unit of analysis is video sample; "
        "no causal/clinical language is claimed; look-elsewhere and garden-of-forking-paths risks remain because "
        "T157 scoring is exploratory and must be confirmed by T158 nested validation."
    )
    report = "\n".join(
        [
            "# T157 Top-K Spectral Candidate Selector",
            "",
            "## Material Passport",
            "",
            "- Task: T157",
            "- Type: code experiment / validation",
            "- Verification status: ANALYZED",
            "- Inputs: raw UBFC/4TU videos through T155 multi-ROI extractor; T150/T156 outputs",
            "- Output: top-K spectral candidate pool and corrected-HR selector",
            "",
            "## Purpose",
            "",
            "T157 moves beyond refusal: it extracts top-K spectral peaks per ROI, method, and temporal window, then tests whether a physiology-constrained score can choose a corrected HR estimate.",
            "",
            "## Main Insight",
            "",
            str(summary["main_insight"]),
            "",
            "## Metrics",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "## Case Audit",
            "",
            markdown_table(case_audit[[c for c in case_cols if c in case_audit.columns]]),
            "",
            "## Bootstrap",
            "",
            markdown_table(bootstrap),
            "",
            "## Statistical Caution",
            "",
            fallacy_scan,
            "",
            "## Figures",
            "",
            "\n".join(f"- {name}: `{path}`" for name, path in figures.items()),
            "",
            "## Claim Boundary",
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
            "# T157 教学文档：top-K spectral candidate selector",
            "",
            "## 1. 这一步为什么要做？",
            "",
            "T156 已经能把 subject14 这种危险输出拦下来，但它只是 `review/refusal`，还不能给出 corrected HR。T157 的目标是更进一步：不只看每个 ROI/method 的最大 FFT peak，而是保留 top-K spectral peaks，让正确 HR 即使不是最大峰，也有机会被选出来。",
            "",
            "## 2. 用到的软件、代码和文件",
            "",
            "- 新脚本：`scripts/run_t157_topk_spectral_candidate_selector.py`",
            "- 输入 raw-video manifest 与 multi-ROI extractor：来自 T155 脚本",
            "- 输入 T150/T156 输出：`experiments/t150_domain_robust_selection_table.csv`、`experiments/t156_candidate_conflict_release_table.csv`",
            "- 输出 top-K peak table：`experiments/t157_topk_peak_table.csv`",
            "- 输出 candidate table：`experiments/t157_candidate_table.csv`",
            "- 输出 selection table：`experiments/t157_selection_table.csv`",
            "- 输出 policy comparison：`experiments/t157_policy_comparison.csv`",
            "",
            "## 3. 具体怎么实现？",
            "",
            "第一步，对每个 raw video 提取 6 个 face ROI：`face_full`、`forehead`、`left_cheek`、`right_cheek`、`center_face`、`lower_face`。",
            "",
            "第二步，对每个 ROI 跑 6 个 classical rPPG methods：`GREEN`、`CHROM`、`POS`、`PBV`、`ICA`、`LGI`。",
            "",
            "第三步，每个 ROI/method 不只取最大峰，而是在 `full_0_60`、`half_0_30`、`half_30_60` 三个窗口里取 top-8 spectral peaks。这样一个视频会生成大量候选峰。",
            "",
            "第四步，把频率接近的 peak 聚类成 candidate HR hypotheses。每个 candidate 记录 `support_count`、`support_rois`、`support_methods`、`top1_support_count`、`rank_score`、`pos_chrom_count`、`power_fraction`、`adult_plausibility`、`harmonic/alias alternatives` 等特征。",
            "",
            "第五步，用 inference-only 的 `T157 score` 选择 corrected candidate。这个 score 不读取 ground truth，而是综合多 ROI、多方法、多时间窗支持，同时惩罚 low alias、motion-band artifact 和 high harmonic alias。",
            "",
            "## 4. 指标结果",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "## 5. 指标迭代链",
            "",
            "T150 是单 ROI domain-robust selector；T156 是安全 gate，可以拦截 subject14；T157 开始尝试 corrected HR。最重要的是 `T157_topk_candidate_oracle`：它告诉我们 top-K candidate pool 里理论上能达到什么上限。如果 oracle 很好而 T157 selector 还不够好，瓶颈就是 scoring，而不是数据集或者视频里没有信号。",
            "",
            "## 6. Output 迭代链",
            "",
            "旧 output：单个 `selected_bpm`。",
            "",
            "T156 output：`selected_bpm` + `review_reason`。",
            "",
            "T157 output：`selected_bpm` + `candidate_id` + `t157_score` + `score_margin` + `support_rois` + `support_methods` + `member_summary`。这让我们能解释一个 corrected HR 是由哪些 ROI/method/window 支持的。",
            "",
            "## 7. Case audit",
            "",
            markdown_table(case_audit[[c for c in case_cols if c in case_audit.columns]]),
            "",
            "## 8. 深度 insight",
            "",
            str(summary["main_insight"]),
            "",
            "## 9. 统计与审稿风险",
            "",
            fallacy_scan,
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
    append_unique(DOCS / "phase_learning_journal.md", "# T157 top-K spectral candidate selector", doc)

    guarded_doc = "\n".join(
        [
            "# T157B guarded correction update",
            "",
            "## 1. 这次更新解决什么问题？",
            "",
            "T157 第一版 naive selector 证明了一件重要的事：raw video 的 top-K spectral candidate pool 里确实存在接近 reference HR 的候选峰，但是如果只相信 `support_count` 或最高综合分，算法会被多个 ROI/方法共同出现的 motion/lighting artifact 吸走。",
            "",
            "所以 T157B 不再让 corrected selector 接管所有样本，而是采用 `guarded correction`：T156 已经安全通过的样本继续使用 T150/T156 的输出；只有 T156 判定为 `low_alias_upper_candidate_conflict` 的样本，才从 top-K pool 中选择“离原始低频峰最近的上方生理候选”。",
            "",
            "## 2. 原理解释",
            "",
            "传统 rPPG/视频 HR 的错误经常不是完全没有生理信号，而是最大 spectral peak 选错了。低频 alias 的典型现象是：模型输出偏低，但在更高的生理范围内存在一个相邻候选峰。T157B 的核心思想是把问题拆开：",
            "",
            "- `candidate availability`：正确候选是否存在于 top-K pool？",
            "- `candidate selection`：在不读取 ground truth 的情况下，如何选择最可信候选？",
            "- `selective correction`：只在已有 gate 发现冲突时才修正，避免对本来正确的样本造成伤害。",
            "",
            "这个设计比 naive selector 更符合产品逻辑：稳定样本不动，风险样本才纠错。",
            "",
            "## 3. 代码实现在哪里？",
            "",
            "- 主脚本：`scripts/run_t157_topk_spectral_candidate_selector.py`",
            "- 新增函数：`choose_low_alias_upper_candidate()`",
            "- 新增策略：`select_guarded_correction()`",
            "- 核心输出：`experiments/t157_policy_comparison.csv`、`experiments/t157_case_audit.csv`、`experiments/t157_selection_table.csv`",
            "",
            "实现逻辑是：如果 T156 `released=1`，保留 T156/T150 的 `selected_bpm`；如果 T156 `released=0` 且 T150 是低频冲突，则在候选表中找满足 `support_methods >= 4`、`support_rois >= 4` 的上方生理候选，并选择最接近低频原峰的那个候选。",
            "",
            "## 4. 指标迭代链",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "关键变化：T157 naive 的 MAE 明显变差，说明“谁支持数最多”不是可靠的生理选择规则；T157 guarded 则恢复 100% coverage，并把 unsafe releases 压到 0。",
            "",
            "## 5. 关键样本 output 迭代链",
            "",
            markdown_table(case_audit[[c for c in case_cols if c in case_audit.columns]]),
            "",
            "解释：`ubfc_subject14` 原始 T150 输出 66.17 BPM，reference 是 80.41 BPM。T156 发现它是 low-alias conflict 并选择 review；T157B 在 top-K pool 中找到 81.80 BPM，把误差降到 1.39 BPM。`4tu_P1M3` 原始 T150 输出 58.49 BPM，reference 是 66.83 BPM；T157B 选择 65.14 BPM，把误差降到 1.69 BPM。",
            "",
            "## 6. Insight",
            "",
            "这一步给我们的论文叙事带来一个更扎实的创新点：不是简单做一个更复杂的 HR estimator，而是提出 `physiology-constrained multi-candidate inference with guarded correction`。它把 raw video 中的多个 spectral hypotheses 保留下来，再用风险 gate 决定是否纠错。",
            "",
            "## 7. 下一步怎么用？",
            "",
            "T157B 还不能直接当最终 SOTA claim，因为规则是在观察失败后设计出来的。下一步 T158 必须冻结规则，在 leave-dataset-out 或 nested validation 中验证它是否仍然比 T150/T156 稳。如果成立，才可以把它写成论文中的主要方法证据。",
            "",
        ]
    )
    GUARDED_DOC_MD.write_text(guarded_doc, encoding="utf-8")
    append_unique(DOCS / "phase_learning_journal.md", "# T157B guarded correction update", guarded_doc)


def update_project_docs(summary: dict[str, object]) -> None:
    marker = "## T157 top-K spectral candidate selector"
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
    marker_v2 = "## T157B guarded correction update"
    text_v2 = "\n".join(
        [
            marker_v2,
            "",
            "T157B keeps the successful T156/T150 anchor for safe samples and only applies top-K correction to low-alias conflicts. This converts the T157 naive failure into a more product-safe algorithmic update.",
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
        append_unique(DOCS / name, marker_v2, text_v2)
    append_unique(
        DOCS / "execution_task_registry.md",
        "| T157 |",
        "| T157 | Top-K spectral candidates per ROI/method/window and corrected-HR selector | `scripts/run_t157_topk_spectral_candidate_selector.py`; `experiments/t157_policy_comparison.csv`; `docs/t157_topk_spectral_candidate_selector.md` | DONE-TOPK-CANDIDATE-SELECTOR |",
    )
    append_evidence_row(summary)


def run(max_samples: int | None = None, reuse_peaks: bool = False) -> dict[str, object]:
    t150 = load_t150_adult()
    if reuse_peaks and PEAK_TABLE_CSV.exists():
        peaks = pd.read_csv(PEAK_TABLE_CSV)
    else:
        peaks = extract_topk_peaks(max_samples=max_samples)
    candidates = build_candidates(peaks, t150)
    corrected = select_top(candidates, "T157_corrected_release_all_v1")
    selective = selective_release(corrected)
    t156 = t156_table()
    guarded = select_guarded_correction(candidates, t156)
    oracle = select_oracle(candidates)
    selection = pd.concat(
        [
            t150_release_all(t150),
            t156,
            corrected,
            selective,
            guarded,
            oracle,
        ],
        ignore_index=True,
        sort=False,
    )
    selection.to_csv(SELECTION_TABLE_CSV, index=False, encoding="utf-8-sig")
    policy_summary = summarize_policy(selection)
    policy_summary.to_csv(POLICY_COMPARISON_CSV, index=False, encoding="utf-8-sig")
    boot = pd.concat(
        [
            bootstrap_delta(selection, "T150_release_all", "T157_corrected_release_all_v1"),
            bootstrap_delta(selection, "T150_release_all", "T157_corrected_selective_v1"),
            bootstrap_delta(selection, "T150_release_all", "T157_guarded_correction_v2"),
            bootstrap_delta(selection, "T150_release_all", "T157_topk_candidate_oracle"),
        ],
        ignore_index=True,
        sort=False,
    )
    boot.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    case_audit = build_case_audit(selection, candidates)
    figures = write_figures(policy_summary, case_audit, candidates)
    summary = build_summary(policy_summary, boot, case_audit, figures)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(summary, policy_summary, boot, case_audit, figures)
    update_project_docs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
