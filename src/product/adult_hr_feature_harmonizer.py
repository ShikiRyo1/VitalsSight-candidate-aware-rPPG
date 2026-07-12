from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "candidate_bpm",
    "support_count",
    "support_methods",
    "top1_support_count",
    "rank_score",
    "sum_power_fraction",
    "mean_power_fraction",
    "max_power_fraction",
    "mean_snr_proxy_db",
    "adult_plausibility",
    "window_candidate_count",
    "window_max_support_methods",
    "window_max_top1_support",
    "window_max_power_fraction",
    "higher_candidate_count",
    "higher_max_support_methods",
    "higher_max_power_fraction",
    "lower_candidate_count",
    "lower_max_support_methods",
    "harmonic_lower_support_methods",
    "harmonic_upper_support_methods",
    "high_harmonic_anchor",
    "low_alias_risk",
    "support_fraction_of_window_max",
    "top1_fraction_of_window_max",
    "power_fraction_of_window_max",
    "bpm_minus_window_median",
    "bpm_abs_minus_window_median",
]


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    out = pd.to_numeric(numerator, errors="coerce") / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def choose_group_columns(frame: pd.DataFrame, preferred: Iterable[str] | None = None) -> list[str]:
    if preferred:
        cols = [col for col in preferred if col in frame.columns]
        if cols:
            return cols
    if "window_key" in frame.columns:
        return ["window_key"]
    if "sample_id" in frame.columns:
        return ["sample_id"]
    return ["_candidate_window"]


def _support_methods(frame: pd.DataFrame) -> pd.Series:
    if "support_methods" in frame.columns:
        return numeric_series(frame, "support_methods")
    if "methods" in frame.columns:
        return frame["methods"].fillna("").astype(str).map(
            lambda value: 0.0 if not value else float(len([item for item in value.split(",") if item.strip()]))
        )
    return numeric_series(frame, "support_count")


def harmonize_candidate_features(
    candidates: pd.DataFrame,
    *,
    group_columns: Iterable[str] | None = None,
    harmonic_tolerance: float = 0.18,
) -> pd.DataFrame:
    """Return candidates with the frozen T324/T327 feature schema.

    The training-side selector learned from window-level candidate clusters. Live
    product code may produce a smaller table, so this function reconstructs the
    same context features without using labels or ground truth.
    """

    if candidates.empty:
        out = candidates.copy()
        for col in FEATURE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    out = candidates.copy()
    if "_candidate_window" not in out.columns:
        out["_candidate_window"] = "single_window"

    group_cols = choose_group_columns(out, group_columns)

    out["candidate_bpm"] = numeric_series(out, "candidate_bpm", math.nan)
    out["support_count"] = numeric_series(out, "support_count")
    out["support_methods"] = _support_methods(out)
    out["top1_support_count"] = numeric_series(out, "top1_support_count")
    out["rank_score"] = numeric_series(out, "rank_score")
    out["sum_power_fraction"] = numeric_series(out, "sum_power_fraction")
    out["mean_power_fraction"] = numeric_series(out, "mean_power_fraction")
    out["max_power_fraction"] = numeric_series(out, "max_power_fraction")
    out["mean_snr_proxy_db"] = numeric_series(out, "mean_snr_proxy_db")
    out["adult_plausibility"] = [
        adult_plausibility(finite_float(value)) for value in out["candidate_bpm"]
    ]

    context_rows: list[dict[str, float | int]] = []
    for _, group in out.groupby(group_cols, sort=False, dropna=False):
        bpms = pd.to_numeric(group["candidate_bpm"], errors="coerce")
        support = pd.to_numeric(group["support_methods"], errors="coerce").fillna(0.0)
        power = pd.to_numeric(group["sum_power_fraction"], errors="coerce").fillna(0.0)
        top1 = pd.to_numeric(group["top1_support_count"], errors="coerce").fillna(0.0)
        median_bpm = float(bpms.median()) if bpms.notna().any() else math.nan

        for idx, row in group.iterrows():
            bpm = finite_float(row.get("candidate_bpm"))
            if not math.isfinite(bpm):
                context_rows.append({"_idx": idx})
                continue

            bpm_ratio_den = bpms.replace(0.0, np.nan)
            higher = group[(bpms > bpm + 8.0) & (bpms <= min(180.0, bpm + 80.0))]
            lower = group[(bpms < bpm - 8.0) & (bpms >= max(45.0, bpm - 80.0))]
            harmonic_lower = group[
                (bpms < bpm - 6.0)
                & (
                    ((bpm / bpm_ratio_den) - 1.50).abs().le(harmonic_tolerance)
                    | ((bpm / bpm_ratio_den) - 1.75).abs().le(harmonic_tolerance)
                    | ((bpm / bpm_ratio_den) - 2.00).abs().le(harmonic_tolerance)
                )
            ]
            harmonic_upper = group[
                (bpms > bpm + 6.0)
                & (
                    ((bpms / max(bpm, 1e-6)) - 1.50).abs().le(harmonic_tolerance)
                    | ((bpms / max(bpm, 1e-6)) - 1.75).abs().le(harmonic_tolerance)
                    | ((bpms / max(bpm, 1e-6)) - 2.00).abs().le(harmonic_tolerance)
                )
            ]

            context_rows.append(
                {
                    "_idx": idx,
                    "window_candidate_count": int(len(group)),
                    "window_max_support_methods": float(support.max()) if len(support) else 0.0,
                    "window_max_top1_support": float(top1.max()) if len(top1) else 0.0,
                    "window_max_power_fraction": float(power.max()) if len(power) else 0.0,
                    "higher_candidate_count": int(len(higher)),
                    "higher_max_support_methods": float(numeric_series(higher, "support_methods").max()) if not higher.empty else 0.0,
                    "higher_max_power_fraction": float(numeric_series(higher, "sum_power_fraction").max()) if not higher.empty else 0.0,
                    "lower_candidate_count": int(len(lower)),
                    "lower_max_support_methods": float(numeric_series(lower, "support_methods").max()) if not lower.empty else 0.0,
                    "harmonic_lower_support_methods": float(numeric_series(harmonic_lower, "support_methods").max()) if not harmonic_lower.empty else 0.0,
                    "harmonic_upper_support_methods": float(numeric_series(harmonic_upper, "support_methods").max()) if not harmonic_upper.empty else 0.0,
                    "bpm_minus_window_median": bpm - median_bpm if math.isfinite(median_bpm) else 0.0,
                    "bpm_abs_minus_window_median": abs(bpm - median_bpm) if math.isfinite(median_bpm) else 0.0,
                }
            )

    context = pd.DataFrame(context_rows).set_index("_idx")
    out = out.join(context, how="left", rsuffix="_context")

    out["high_harmonic_anchor"] = (
        (pd.to_numeric(out["candidate_bpm"], errors="coerce") >= 95.0)
        & (pd.to_numeric(out["candidate_bpm"], errors="coerce") <= 145.0)
        & (pd.to_numeric(out["support_methods"], errors="coerce").fillna(0.0) >= 2.0)
        & (pd.to_numeric(out["harmonic_lower_support_methods"], errors="coerce").fillna(0.0) >= 2.0)
    ).astype(float)
    out["low_alias_risk"] = (
        (pd.to_numeric(out["candidate_bpm"], errors="coerce") < 75.0)
        & (pd.to_numeric(out["harmonic_upper_support_methods"], errors="coerce").fillna(0.0) >= 2.0)
        & ~(
            (pd.to_numeric(out["candidate_bpm"], errors="coerce") >= 55.0)
            & (pd.to_numeric(out["top1_support_count"], errors="coerce").fillna(0.0) >= 3.0)
            & (pd.to_numeric(out["rank_score"], errors="coerce").fillna(0.0) >= 3.0)
        )
    ).astype(float)

    out["support_fraction_of_window_max"] = safe_div(out["support_methods"], out["window_max_support_methods"])
    out["top1_fraction_of_window_max"] = safe_div(out["top1_support_count"], out["window_max_top1_support"])
    out["power_fraction_of_window_max"] = safe_div(out["sum_power_fraction"], out["window_max_power_fraction"])

    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.drop(columns=["_candidate_window"], errors="ignore")


def validate_feature_schema(frame: pd.DataFrame, feature_columns: Iterable[str] = FEATURE_COLUMNS) -> dict[str, object]:
    features = list(feature_columns)
    missing = [col for col in features if col not in frame.columns]
    non_numeric = [
        col
        for col in features
        if col in frame.columns and not pd.api.types.is_numeric_dtype(pd.to_numeric(frame[col], errors="coerce"))
    ]
    nan_counts = {
        col: int(pd.to_numeric(frame[col], errors="coerce").isna().sum())
        for col in features
        if col in frame.columns
    }
    return {
        "n_rows": int(len(frame)),
        "n_features": int(len(features)),
        "missing_features": missing,
        "non_numeric_features": non_numeric,
        "nan_counts": nan_counts,
        "schema_pass": not missing and not non_numeric and all(value == 0 for value in nan_counts.values()),
    }
