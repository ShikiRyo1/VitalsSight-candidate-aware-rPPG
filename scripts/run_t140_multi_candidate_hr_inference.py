"""T140 adult HR multi-candidate harmonic/alias inference.

T139 showed that simple method-disagreement gating is not enough: several
traditional rPPG methods can agree on the same wrong frequency. T140 therefore
keeps top-K spectral candidates and evaluates whether physiology-aware
candidate reasoning can recover hidden HR candidates or route them to review.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.run_t136_4tu_adult_hr_benchmark import (  # noqa: E402
    discover_sessions,
    extract_rgb_trace,
    snr_proxy_db,
)
from src.baselines.traditional_rppg import METHODS  # noqa: E402
from src.data.archive_io import extract_zip_member, read_zip_text  # noqa: E402
from src.data.labels_4tu import parse_rr_intervals  # noqa: E402
from src.evaluation.metrics import mae, pearson, rmse  # noqa: E402
from src.signal.filters import as_clean_array, bandpass, zscore  # noqa: E402


EXPERIMENTS = PROJECT / "experiments"
DOCS = PROJECT / "docs"
FIG_DIR = PROJECT / "output" / "t140_figures"
CACHE_DIR = EXPERIMENTS / "cache" / "t136_4tu"

T136_RESULTS = EXPERIMENTS / "t136_4tu_classical_window_results.csv"
PEAK_TABLE_CSV = EXPERIMENTS / "t140_peak_table.csv"
CANDIDATE_TABLE_CSV = EXPERIMENTS / "t140_candidate_table.csv"
SELECTION_TABLE_CSV = EXPERIMENTS / "t140_selection_table.csv"
POLICY_SUMMARY_CSV = EXPERIMENTS / "t140_policy_summary.csv"
FAILURE_AUDIT_CSV = EXPERIMENTS / "t140_failure_case_audit.csv"
LOSO_SUMMARY_CSV = EXPERIMENTS / "t140_loso_summary.csv"
SUMMARY_JSON = EXPERIMENTS / "t140_multi_candidate_hr_summary.json"
REPORT_MD = EXPERIMENTS / f"t140_multi_candidate_hr_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t140_multi_candidate_hr_inference.md"

HR_MIN_BPM = 45.0
HR_MAX_BPM = 180.0
UNSAFE_BPM = 10.0
TOP_K = 10
CLUSTER_TOL_BPM = 7.5
METHOD_ORDER = ["CHROM", "GREEN", "ICA", "LGI", "PBV", "POS"]
SUBWINDOWS = [
    ("full_0_60", 0.0, 60.0),
    ("half_0_30", 0.0, 30.0),
    ("half_30_60", 30.0, 60.0),
    ("third_0_20", 0.0, 20.0),
    ("third_20_40", 20.0, 40.0),
    ("third_40_60", 40.0, 60.0),
]


@dataclass(frozen=True)
class ReleasePolicy:
    name: str
    score_col: str
    min_margin: float
    min_score: float
    max_high_artifact: float
    min_support: int


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_k_hr_peaks(values: np.ndarray, fps: float, *, top_k: int = TOP_K) -> tuple[list[dict[str, float]], float, float]:
    arr = zscore(as_clean_array(values))
    low_hz = HR_MIN_BPM / 60.0
    high_hz = HR_MAX_BPM / 60.0
    filtered = bandpass(arr, fps, low_hz, high_hz)
    nperseg = min(len(filtered), max(32, int(fps * 16)))
    freqs, power = signal.welch(filtered, fs=fps, nperseg=nperseg)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return [], 0.0, float(np.sum(power))
    band_freqs = freqs[mask]
    band_power_values = power[mask]
    peaks, _ = signal.find_peaks(band_power_values)
    ranked: list[int] = []
    if len(peaks):
        ranked.extend(list(peaks[np.argsort(band_power_values[peaks])[::-1]]))
    ranked.extend([int(i) for i in np.argsort(band_power_values)[::-1]])
    unique_ranked: list[int] = []
    seen: set[int] = set()
    for idx in ranked:
        if idx not in seen:
            unique_ranked.append(idx)
            seen.add(idx)
        if len(unique_ranked) >= top_k:
            break
    band_power = float(np.sum(band_power_values) + 1e-12)
    total_power = float(np.sum(power) + 1e-12)
    rows = [
        {
            "peak_bpm": float(band_freqs[idx] * 60.0),
            "peak_hz": float(band_freqs[idx]),
            "power_fraction": float(band_power_values[idx] / band_power),
            "peak_power": float(band_power_values[idx]),
            "rank": float(rank),
        }
        for rank, idx in enumerate(unique_ranked, start=1)
    ]
    return rows, band_power, total_power


def adult_plausibility_score(bpm: float) -> float:
    if 55.0 <= bpm <= 105.0:
        return 1.0
    if 105.0 < bpm <= 135.0:
        return 0.85
    if 135.0 < bpm <= 165.0:
        return 0.75
    if 45.0 <= bpm < 55.0 or 165.0 < bpm <= 180.0:
        return 0.50
    return 0.0


def cluster_peak_rows(peak_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sorted_rows = sorted(peak_rows, key=lambda row: safe_float(row["peak_bpm"]))
    clusters: list[dict[str, object]] = []
    for row in sorted_rows:
        bpm = safe_float(row["peak_bpm"])
        if not math.isfinite(bpm):
            continue
        nearest_idx = None
        nearest_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            dist = abs(bpm - safe_float(cluster["center_bpm"]))
            if dist <= CLUSTER_TOL_BPM and dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist
        if nearest_idx is None:
            clusters.append({"members": [row], "center_bpm": bpm})
        else:
            members = clusters[nearest_idx]["members"]
            assert isinstance(members, list)
            members.append(row)
            weights = np.asarray([safe_float(m["power_fraction"]) for m in members], dtype=float)
            values = np.asarray([safe_float(m["peak_bpm"]) for m in members], dtype=float)
            weights = np.clip(weights, 1e-6, None)
            clusters[nearest_idx]["center_bpm"] = float(np.average(values, weights=weights))
    return clusters


def method_set(members: list[dict[str, object]]) -> set[str]:
    return {str(member["method"]) for member in members}


def build_candidate_features(
    sample_meta: dict[str, object],
    peak_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    clusters = cluster_peak_rows(peak_rows)
    rows: list[dict[str, object]] = []
    gt = safe_float(sample_meta["gt_hr_bpm"])
    for idx, cluster in enumerate(clusters):
        members = cluster["members"]
        assert isinstance(members, list)
        bpm = safe_float(cluster["center_bpm"])
        full_members = [m for m in members if m["window_id"] == "full_0_60"]
        sub_members = [m for m in members if m["window_id"] != "full_0_60"]
        full_methods = method_set(full_members)
        all_methods = method_set(members)
        top1_full_methods = method_set([m for m in full_members if int(safe_float(m["rank"])) == 1])
        top1_sub_pairs = {
            (str(m["window_id"]), str(m["method"]))
            for m in sub_members
            if int(safe_float(m["rank"])) == 1
        }
        sub_pairs = {(str(m["window_id"]), str(m["method"])) for m in sub_members}
        ranks = [safe_float(m["rank"]) for m in full_members] or [safe_float(m["rank"]) for m in members]
        power_full = float(sum(safe_float(m["power_fraction"]) for m in full_members))
        power_all = float(sum(safe_float(m["power_fraction"]) for m in members))
        power_max = float(max([safe_float(m["power_fraction"]) for m in full_members] or [0.0]))
        row = {
            **sample_meta,
            "task_id": "T140",
            "candidate_id": f"{sample_meta['sample_id']}_c{idx:02d}",
            "candidate_bpm": bpm,
            "abs_error_bpm": abs(bpm - gt) if math.isfinite(gt) else math.nan,
            "is_safe_10bpm": int(abs(bpm - gt) <= UNSAFE_BPM) if math.isfinite(gt) else 0,
            "support_methods": len(full_methods),
            "all_support_methods": len(all_methods),
            "top1_support_methods": len(top1_full_methods),
            "subwindow_support": len(sub_pairs),
            "subwindow_top1_support": len(top1_sub_pairs),
            "power_sum_full": power_full,
            "power_sum_all": power_all,
            "power_max_full": power_max,
            "median_rank_full": float(np.median(ranks)) if ranks else math.nan,
            "rank_score_full": float(sum(1.0 / max(1.0, safe_float(m["rank"])) for m in full_members)),
            "has_green_support": int("GREEN" in full_methods),
            "has_pos_support": int("POS" in full_methods),
            "has_pbv_support": int("PBV" in full_methods),
            "has_chrom_support": int("CHROM" in full_methods),
            "adult_plausibility": adult_plausibility_score(bpm),
        }
        rows.append(row)

    for row in rows:
        bpm = safe_float(row["candidate_bpm"])
        lower_rows = [
            other
            for other in rows
            if 55.0 <= safe_float(other["candidate_bpm"]) <= 100.0
            and safe_float(other["candidate_bpm"]) < bpm - 15.0
        ]
        row["half_harmonic_support"] = max(
            [safe_float(other["support_methods"]) for other in rows if abs(safe_float(other["candidate_bpm"]) - bpm / 2.0) <= 10.0]
            or [0.0]
        )
        row["double_harmonic_support"] = max(
            [safe_float(other["support_methods"]) for other in rows if abs(safe_float(other["candidate_bpm"]) - bpm * 2.0) <= 12.5]
            or [0.0]
        )
        row["nearby_high_support"] = max(
            [
                safe_float(other["support_methods"])
                for other in rows
                if other is not row and abs(safe_float(other["candidate_bpm"]) - bpm) <= 20.0
            ]
            or [0.0]
        )
        row["lower_alternative_support"] = max([safe_float(other["support_methods"]) for other in lower_rows] or [0.0])
        row["lower_alternative_power"] = max([safe_float(other["power_sum_full"]) for other in lower_rows] or [0.0])
        row["lower_alternative_power_ratio"] = safe_float(row["lower_alternative_power"]) / max(
            safe_float(row["power_sum_full"]), 1e-8
        )
        high_artifact = (
            bpm >= 105.0
            and safe_float(row["lower_alternative_support"]) >= max(3.0, safe_float(row["support_methods"]) - 1.0)
            and safe_float(row["lower_alternative_power_ratio"]) >= 0.20
        )
        row["high_frequency_artifact_suspicion"] = int(high_artifact)
        row["support_power_score"] = (
            2.0 * safe_float(row["support_methods"])
            + 0.8 * safe_float(row["top1_support_methods"])
            + 0.10 * safe_float(row["subwindow_support"])
            + 4.0 * safe_float(row["power_sum_full"])
        )
        row["physiology_candidate_score"] = (
            2.0 * safe_float(row["support_methods"])
            + 0.70 * safe_float(row["top1_support_methods"])
            + 0.12 * safe_float(row["subwindow_support"])
            + 3.5 * safe_float(row["power_sum_full"])
            + 1.5 * safe_float(row["adult_plausibility"])
            + 0.70 * safe_float(row["has_pos_support"])
            + 0.45 * safe_float(row["has_pbv_support"])
            + 0.25 * safe_float(row["has_green_support"])
            + 0.45 * safe_float(row["double_harmonic_support"])
            + 0.25 * safe_float(row["nearby_high_support"])
            - 5.0 * safe_float(row["high_frequency_artifact_suspicion"])
        )
    return rows


def load_green_baseline() -> pd.DataFrame:
    if not T136_RESULTS.exists():
        raise FileNotFoundError(f"Missing T136 results: {T136_RESULTS}")
    df = pd.read_csv(T136_RESULTS)
    green = df[df["method"] == "GREEN"].copy()
    return green.rename(
        columns={
            "pred_hr_bpm": "selected_bpm",
            "abs_error_bpm": "selected_abs_error_bpm",
        }
    )


def extract_sample(record) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rr_text = read_zip_text(record.archive_path, record.rr_member)
    gt_hr = parse_rr_intervals(rr_text).mean_hr(0.0, 60.0)
    video_path = extract_zip_member(record.archive_path, record.video_member, CACHE_DIR / record.sample_id)
    rgb_trace, fps, roi, meta = extract_rgb_trace(video_path, seconds=60.0)
    sample_meta = {
        "sample_id": record.sample_id,
        "dataset": record.dataset,
        "session_id": record.session_id,
        "subject_id": record.subject_id,
        "split": record.split,
        "condition_group": record.condition_group,
        "gt_hr_bpm": gt_hr,
        "fps": fps,
        "roi_name": roi.name,
        "frames_used": int(meta["frames_used"]),
    }
    peak_rows: list[dict[str, object]] = []
    for method_name, method_fn in sorted(METHODS.items()):
        for window_id, start_sec, end_sec in SUBWINDOWS:
            start_idx = int(round(start_sec * fps))
            end_idx = int(round(end_sec * fps))
            rgb_window = rgb_trace[start_idx:end_idx]
            if len(rgb_window) < max(32, int(8 * fps)):
                continue
            values = method_fn(rgb_window)
            peaks, band_power, total_power = top_k_hr_peaks(values, fps)
            snr = snr_proxy_db(band_power, total_power)
            for peak in peaks:
                peak_rows.append(
                    {
                        **sample_meta,
                        "window_id": window_id,
                        "window_start_sec": start_sec,
                        "window_end_sec": end_sec,
                        "method": method_name,
                        "rank": int(peak["rank"]),
                        "peak_bpm": peak["peak_bpm"],
                        "peak_hz": peak["peak_hz"],
                        "power_fraction": peak["power_fraction"],
                        "peak_power": peak["peak_power"],
                        "band_power": band_power,
                        "total_power": total_power,
                        "snr_proxy_db": snr,
                    }
                )
    candidates = build_candidate_features(sample_meta, peak_rows)
    return peak_rows, candidates


def select_by_score(candidates: pd.DataFrame, score_col: str, policy_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        ranked = group.sort_values([score_col, "support_methods", "power_sum_full"], ascending=[False, False, False])
        selected = ranked.iloc[0].to_dict()
        second_score = safe_float(ranked.iloc[1][score_col]) if len(ranked) > 1 else -math.inf
        selected["policy"] = policy_name
        selected["score_col"] = score_col
        selected["selected_bpm"] = safe_float(selected["candidate_bpm"])
        selected["selected_abs_error_bpm"] = safe_float(selected["abs_error_bpm"])
        selected["score_margin"] = safe_float(selected[score_col]) - second_score
        selected["released"] = 1
        rows.append(selected)
    return pd.DataFrame(rows)


def candidate_oracle(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        selected = group.sort_values(["abs_error_bpm", "support_methods"], ascending=[True, False]).iloc[0].to_dict()
        selected["policy"] = "candidate_oracle_topK"
        selected["score_col"] = "oracle_abs_error"
        selected["selected_bpm"] = safe_float(selected["candidate_bpm"])
        selected["selected_abs_error_bpm"] = safe_float(selected["abs_error_bpm"])
        selected["score_margin"] = math.nan
        selected["released"] = 1
        rows.append(selected)
    return pd.DataFrame(rows)


def add_supervised_scores(candidates: pd.DataFrame, train_mask: np.ndarray, out_col: str) -> pd.DataFrame:
    feature_cols = [
        "candidate_bpm",
        "support_methods",
        "top1_support_methods",
        "subwindow_support",
        "subwindow_top1_support",
        "power_sum_full",
        "power_sum_all",
        "power_max_full",
        "median_rank_full",
        "rank_score_full",
        "has_green_support",
        "has_pos_support",
        "has_pbv_support",
        "adult_plausibility",
        "half_harmonic_support",
        "double_harmonic_support",
        "nearby_high_support",
        "lower_alternative_support",
        "lower_alternative_power_ratio",
        "high_frequency_artifact_suspicion",
        "support_power_score",
        "physiology_candidate_score",
    ]
    out = candidates.copy()
    x_train = out.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = out.loc[train_mask, "is_safe_10bpm"].astype(int)
    x_all = out.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(set(y_train)) < 2:
        out[out_col] = out["physiology_candidate_score"]
        return out
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=140,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    out[out_col] = model.predict_proba(x_all)[:, 1]
    return out


def release_mask_for_policy(selected: pd.DataFrame, policy: ReleasePolicy) -> np.ndarray:
    return (
        (selected["score_margin"].to_numpy(dtype=float) >= policy.min_margin)
        & (selected[policy.score_col].to_numpy(dtype=float) >= policy.min_score)
        & (selected["high_frequency_artifact_suspicion"].to_numpy(dtype=float) <= policy.max_high_artifact)
        & (selected["support_methods"].to_numpy(dtype=float) >= policy.min_support)
    )


def summarize_selection(selected: pd.DataFrame, policy_name: str, release_mask: np.ndarray | None = None) -> dict[str, object]:
    mask = np.ones(len(selected), dtype=bool) if release_mask is None else np.asarray(release_mask, dtype=bool)
    finite = np.isfinite(selected["gt_hr_bpm"].to_numpy(dtype=float)) & np.isfinite(
        selected["selected_bpm"].to_numpy(dtype=float)
    )
    mask = mask & finite
    n = int(finite.sum())
    released = int(mask.sum())
    withheld = n - released
    if released:
        y_true = selected["gt_hr_bpm"].to_numpy(dtype=float)[mask]
        y_pred = selected["selected_bpm"].to_numpy(dtype=float)[mask]
        abs_err = np.abs(y_true - y_pred)
        unsafe_count = int(np.sum(abs_err > UNSAFE_BPM))
        p90 = float(np.percentile(abs_err, 90))
    else:
        y_true = np.asarray([], dtype=float)
        y_pred = np.asarray([], dtype=float)
        abs_err = np.asarray([], dtype=float)
        unsafe_count = 0
        p90 = math.nan
    return {
        "policy": policy_name,
        "n_total": n,
        "released": released,
        "withheld": withheld,
        "coverage": released / n if n else 0.0,
        "mae_bpm": mae(y_true, y_pred),
        "rmse_bpm": rmse(y_true, y_pred),
        "pearson_r": pearson(y_true, y_pred),
        "median_abs_error_bpm": float(np.median(abs_err)) if released else math.nan,
        "p90_abs_error_bpm": p90,
        "unsafe_released_count": unsafe_count,
        "unsafe_release_rate": unsafe_count / released if released else 0.0,
        "unsafe_rate_over_all": unsafe_count / n if n else 0.0,
    }


def choose_release_policy(selected: pd.DataFrame, train_mask: np.ndarray, score_col: str) -> ReleasePolicy:
    rows: list[dict[str, object]] = []
    score_values = selected.loc[train_mask, score_col].to_numpy(dtype=float)
    margin_values = selected.loc[train_mask, "score_margin"].to_numpy(dtype=float)
    score_grid = [float(np.nanpercentile(score_values, q)) for q in [0, 10, 20, 30, 40, 50]]
    margin_grid = [0.0, 0.25, 0.50, 1.00, 1.50, 2.00]
    for min_score in sorted(set(score_grid)):
        for min_margin in margin_grid:
            for max_artifact in [0.0, 1.0]:
                for min_support in [1, 2, 3, 4]:
                    policy = ReleasePolicy(
                        name="grid_candidate_release",
                        score_col=score_col,
                        min_margin=min_margin,
                        min_score=min_score,
                        max_high_artifact=max_artifact,
                        min_support=min_support,
                    )
                    release = release_mask_for_policy(selected, policy)
                    summary = summarize_selection(
                        selected.loc[train_mask].reset_index(drop=True),
                        "train_grid",
                        release[train_mask],
                    )
                    rows.append({**summary, **policy.__dict__})
    grid = pd.DataFrame(rows)
    feasible = grid[(grid["released"] > 0) & (grid["coverage"] >= 0.40) & (grid["unsafe_release_rate"] <= 0.10)]
    if feasible.empty:
        feasible = grid[grid["released"] > 0].copy()
    feasible = feasible.copy()
    feasible["utility"] = (
        3.0 * feasible["unsafe_release_rate"]
        + feasible["mae_bpm"] / 30.0
        + 0.5 * (1.0 - feasible["coverage"])
    )
    chosen = feasible.sort_values(["utility", "unsafe_release_rate", "mae_bpm", "coverage"], ascending=[True, True, True, False]).iloc[0]
    return ReleasePolicy(
        name=f"T140_selective_{score_col}",
        score_col=score_col,
        min_margin=float(chosen["min_margin"]),
        min_score=float(chosen["min_score"]),
        max_high_artifact=float(chosen["max_high_artifact"]),
        min_support=int(chosen["min_support"]),
    )


def bootstrap_delta(
    selected_a: pd.DataFrame,
    selected_b: pd.DataFrame,
    *,
    n_boot: int = 5000,
    seed: int = 140,
) -> dict[str, float]:
    a = selected_a.set_index("sample_id")
    b = selected_b.set_index("sample_id")
    ids = sorted(set(a.index) & set(b.index))
    if len(ids) < 2:
        return {"delta_mae_bpm": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_improvement": math.nan}
    y = a.loc[ids, "gt_hr_bpm"].to_numpy(dtype=float)
    pred_a = a.loc[ids, "selected_bpm"].to_numpy(dtype=float)
    pred_b = b.loc[ids, "selected_bpm"].to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y = y[finite]
    pred_a = pred_a[finite]
    pred_b = pred_b[finite]
    if len(y) < 2:
        return {"delta_mae_bpm": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_improvement": math.nan}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        deltas[i] = np.mean(np.abs(y[sample] - pred_a[sample])) - np.mean(np.abs(y[sample] - pred_b[sample]))
    return {
        "delta_mae_bpm": float(np.mean(np.abs(y - pred_a)) - np.mean(np.abs(y - pred_b))),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_improvement": float(np.mean(deltas > 0.0)),
    }


def build_loso(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in sorted(candidates["subject_id"].unique()):
        train_mask = candidates["subject_id"].to_numpy() != subject
        test_mask_samples = None
        scored = add_supervised_scores(candidates, train_mask, "loso_supervised_score")
        selected = select_by_score(scored, "loso_supervised_score", f"T140_loso_supervised_holdout_{subject}")
        test_mask_samples = selected["subject_id"].to_numpy() == subject
        rows.append(
            {
                **summarize_selection(
                    selected.loc[test_mask_samples].reset_index(drop=True),
                    f"T140_loso_supervised_holdout_{subject}",
                ),
                "held_out_subject": subject,
            }
        )
    return pd.DataFrame(rows)


def write_figures(policy_summary: pd.DataFrame, selected: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    ordered = policy_summary.sort_values("mae_bpm")
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(ordered["policy"], ordered["mae_bpm"], color="#0072B2", alpha=0.85)
    ax1.set_ylabel("MAE (BPM)")
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(ordered["policy"], ordered["unsafe_rate_over_all"], color="#D55E00", marker="o", linewidth=2)
    ax2.set_ylabel("Unsafe rate over all")
    ax1.set_title("T140 multi-candidate HR policy comparison")
    fig.tight_layout()
    path = FIG_DIR / "t140_policy_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["policy_comparison"] = str(path)

    oracle = candidates.sort_values("abs_error_bpm").groupby("sample_id", as_index=False).first()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(oracle["gt_hr_bpm"], oracle["candidate_bpm"], color="#009E73", edgecolor="black", linewidth=0.3, s=56)
    lo = min(float(oracle["gt_hr_bpm"].min()), float(oracle["candidate_bpm"].min())) - 5
    hi = max(float(oracle["gt_hr_bpm"].max()), float(oracle["candidate_bpm"].max())) + 5
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#111111", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference HR (BPM)")
    ax.set_ylabel("Best top-K candidate (BPM)")
    ax.set_title("T140 candidate-oracle headroom")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "t140_candidate_oracle_headroom.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["candidate_oracle_headroom"] = str(path)

    p3lc1 = candidates[candidates["session_id"] == "P3LC1"].copy()
    if not p3lc1.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        plot_df = p3lc1.sort_values("candidate_bpm")
        ax.scatter(
            plot_df["candidate_bpm"],
            plot_df["physiology_candidate_score"],
            s=80 + 40 * plot_df["support_methods"],
            c=plot_df["abs_error_bpm"],
            cmap="viridis_r",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.axvline(float(plot_df["gt_hr_bpm"].iloc[0]), color="#111111", linestyle="--", linewidth=1, label="reference")
        ax.set_xlabel("Candidate HR (BPM)")
        ax.set_ylabel("Physiology candidate score")
        ax.set_title("P3LC1: dominant artifact vs hidden physiological candidates")
        ax.legend(frameon=False)
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Absolute error (BPM)")
        fig.tight_layout()
        path = FIG_DIR / "t140_p3lc1_candidate_scores.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["p3lc1_candidate_scores"] = str(path)
    return paths


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    display = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    for col in display.columns:
        if display[col].dtype.kind in "fc":
            display[col] = display[col].map(lambda v: "" if not math.isfinite(float(v)) else f"{float(v):.3f}")
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *body])


def write_report(
    summary: dict[str, object],
    policy_summary: pd.DataFrame,
    failure_audit: pd.DataFrame,
    loso: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    policy_table = markdown_table(
        policy_summary,
        [
            "policy",
            "n_total",
            "released",
            "coverage",
            "mae_bpm",
            "rmse_bpm",
            "pearson_r",
            "p90_abs_error_bpm",
            "unsafe_release_rate",
            "unsafe_rate_over_all",
        ],
    )
    audit_table = markdown_table(
        failure_audit,
        [
            "session_id",
            "condition_group",
            "gt_hr_bpm",
            "green_bpm",
            "oracle_bpm",
            "physiology_bpm",
            "supervised_bpm",
            "physiology_error",
            "supervised_error",
            "decision_note",
        ],
    )
    loso_table = markdown_table(
        loso,
        ["held_out_subject", "released", "coverage", "mae_bpm", "unsafe_release_rate", "unsafe_rate_over_all"],
    )
    text = f"""# T140 Adult HR Multi-Candidate Harmonic/Alias Inference

Date: {date.today().isoformat()}

## Purpose

T140 tests the new project baseline:

```text
physiology-constrained multi-candidate vital-sign inference
```

The motivation is the T139 negative result. A reliability gate based on method disagreement can fail when several estimators agree on the same wrong-source frequency. T140 therefore preserves top-K spectral peaks and evaluates whether candidate-level reasoning can recover hidden physiological peaks or route risky outputs to review.

## What Was Implemented

```text
scripts/run_t140_multi_candidate_hr_inference.py
```

The script reuses the T136 4TU parser and traditional rPPG methods. For every 4TU session, it extracts top-K HR-band peaks from GREEN, CHROM, POS, PBV, ICA, and LGI across the full 60-second window and 30/20-second subwindows. It clusters peaks into candidate HR values, computes candidate features, and evaluates several selection policies.

## Main Metrics

{policy_table}

## P3LC1 and Failure-Case Audit

{audit_table}

## Leave-One-Subject-Out Supervised Candidate Check

{loso_table}

## Bootstrap vs GREEN

```text
{json.dumps(summary.get("bootstrap_vs_green", {}), indent=2)}
```

## Figures

```text
{json.dumps(figures, indent=2)}
```

## Interpretation

T140 is designed to answer two questions:

1. Does the correct adult HR candidate often exist in the top-K spectrum?
2. Can a candidate-selection layer recover it better than a single top-1 peak or simple method consensus?

The candidate-oracle row answers the first question. The mechanistic and supervised candidate rows answer the second question. If the oracle is strong but the learned/mechanistic selector is weak, the next optimization should focus on candidate scoring, not data acquisition.

## Boundary

This is internal 4TU adult HR evidence. It does not prove adult video HR SOTA, adult video RR SOTA, or clinical readiness. It is a measured algorithmic step toward the paper-level claim.
"""
    REPORT_MD.write_text(text, encoding="utf-8")
    DOC_MD.write_text(text, encoding="utf-8")


def run() -> dict[str, object]:
    records = discover_sessions()
    all_peaks: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        print(f"[T140] {idx}/{len(records)} {record.session_id} extracting top-K candidates", flush=True)
        peaks, candidates = extract_sample(record)
        all_peaks.extend(peaks)
        all_candidates.extend(candidates)

    save_csv(PEAK_TABLE_CSV, all_peaks)
    save_csv(CANDIDATE_TABLE_CSV, all_candidates)
    candidates = pd.DataFrame(all_candidates)
    green = load_green_baseline()
    green_selected = green[
        [
            "sample_id",
            "dataset",
            "session_id",
            "subject_id",
            "split",
            "condition_group",
            "gt_hr_bpm",
            "selected_bpm",
            "selected_abs_error_bpm",
        ]
    ].copy()
    green_selected["policy"] = "GREEN_release_all"
    green_selected["released"] = 1

    oracle = candidate_oracle(candidates)
    max_power = select_by_score(candidates, "power_sum_full", "max_power_cluster")
    support_power = select_by_score(candidates, "support_power_score", "support_power_cluster")
    physiology = select_by_score(candidates, "physiology_candidate_score", "physiology_candidate_score")

    non_heldout_candidate_mask = candidates["split"].to_numpy() != "held_out_test"
    supervised_candidates = add_supervised_scores(candidates, non_heldout_candidate_mask, "supervised_candidate_score")
    supervised = select_by_score(supervised_candidates, "supervised_candidate_score", "supervised_candidate_scorer")

    train_sample_mask = physiology["split"].to_numpy() != "held_out_test"
    selective_policy = choose_release_policy(physiology, train_sample_mask, "physiology_candidate_score")
    selective_release = release_mask_for_policy(physiology, selective_policy)
    physiology_selective = physiology.copy()
    physiology_selective["released"] = selective_release.astype(int)
    physiology_selective["policy"] = selective_policy.name

    selected_tables = [green_selected, oracle, max_power, support_power, physiology, supervised, physiology_selective]
    selection = pd.concat(selected_tables, ignore_index=True, sort=False)
    selection.to_csv(SELECTION_TABLE_CSV, index=False, encoding="utf-8-sig")

    policy_rows = [
        summarize_selection(green_selected, "GREEN_release_all"),
        summarize_selection(oracle, "candidate_oracle_topK"),
        summarize_selection(max_power, "max_power_cluster"),
        summarize_selection(support_power, "support_power_cluster"),
        summarize_selection(physiology, "physiology_candidate_score"),
        summarize_selection(supervised, "supervised_candidate_scorer"),
        summarize_selection(physiology, selective_policy.name, selective_release),
    ]
    policy_summary = pd.DataFrame(policy_rows)
    policy_summary.to_csv(POLICY_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    by_policy = {name: df.set_index("sample_id") for name, df in selection.groupby("policy")}
    audit_rows: list[dict[str, object]] = []
    sample_ids = sorted(green_selected["sample_id"].unique())
    for sample_id in sample_ids:
        green_row = by_policy["GREEN_release_all"].loc[sample_id]
        oracle_row = by_policy["candidate_oracle_topK"].loc[sample_id]
        phys_row = by_policy["physiology_candidate_score"].loc[sample_id]
        sup_row = by_policy["supervised_candidate_scorer"].loc[sample_id]
        note = "ok"
        if safe_float(green_row["selected_abs_error_bpm"]) > UNSAFE_BPM and safe_float(phys_row["selected_abs_error_bpm"]) <= UNSAFE_BPM:
            note = "physiology_score_recovers_green_failure"
        elif safe_float(oracle_row["selected_abs_error_bpm"]) <= UNSAFE_BPM and safe_float(phys_row["selected_abs_error_bpm"]) > UNSAFE_BPM:
            note = "recoverable_candidate_missed_by_physiology_score"
        elif safe_float(oracle_row["selected_abs_error_bpm"]) > UNSAFE_BPM:
            note = "topK_oracle_still_unsafe"
        audit_rows.append(
            {
                "sample_id": sample_id,
                "session_id": green_row["session_id"],
                "subject_id": green_row["subject_id"],
                "split": green_row["split"],
                "condition_group": green_row["condition_group"],
                "gt_hr_bpm": safe_float(green_row["gt_hr_bpm"]),
                "green_bpm": safe_float(green_row["selected_bpm"]),
                "green_error": safe_float(green_row["selected_abs_error_bpm"]),
                "oracle_bpm": safe_float(oracle_row["selected_bpm"]),
                "oracle_error": safe_float(oracle_row["selected_abs_error_bpm"]),
                "physiology_bpm": safe_float(phys_row["selected_bpm"]),
                "physiology_error": safe_float(phys_row["selected_abs_error_bpm"]),
                "supervised_bpm": safe_float(sup_row["selected_bpm"]),
                "supervised_error": safe_float(sup_row["selected_abs_error_bpm"]),
                "decision_note": note,
            }
        )
    failure_audit = pd.DataFrame(audit_rows)
    failure_audit.to_csv(FAILURE_AUDIT_CSV, index=False, encoding="utf-8-sig")

    loso = build_loso(candidates)
    loso.to_csv(LOSO_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    figures = write_figures(policy_summary, selection, candidates)

    summary = {
        "task": "T140",
        "date": date.today().isoformat(),
        "n_sessions": len(records),
        "n_peak_rows": len(all_peaks),
        "n_candidates": len(all_candidates),
        "selected_release_policy": selective_policy.__dict__,
        "policy_summary": policy_rows,
        "p3lc1_audit": failure_audit[failure_audit["session_id"] == "P3LC1"].to_dict(orient="records"),
        "candidate_oracle_safe_rate": float((oracle["selected_abs_error_bpm"].to_numpy(dtype=float) <= UNSAFE_BPM).mean()),
        "bootstrap_vs_green": {
            "physiology_candidate_score": bootstrap_delta(green_selected, physiology),
            "supervised_candidate_scorer": bootstrap_delta(green_selected, supervised),
            "candidate_oracle_topK": bootstrap_delta(green_selected, oracle),
        },
        "outputs": {
            "peak_table": str(PEAK_TABLE_CSV),
            "candidate_table": str(CANDIDATE_TABLE_CSV),
            "selection_table": str(SELECTION_TABLE_CSV),
            "policy_summary": str(POLICY_SUMMARY_CSV),
            "failure_audit": str(FAILURE_AUDIT_CSV),
            "loso_summary": str(LOSO_SUMMARY_CSV),
            "report": str(REPORT_MD),
            "doc": str(DOC_MD),
            "figures": figures,
        },
        "claim_boundary": "Internal 4TU adult HR multi-candidate evidence only; not adult SOTA or clinical validation.",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, policy_summary, failure_audit, loso, figures)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()

