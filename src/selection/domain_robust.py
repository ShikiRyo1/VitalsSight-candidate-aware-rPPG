"""Domain-robust multi-candidate vital-sign selection.

The selector in this module is intentionally inference-only: it does not read
ground-truth columns such as ``gt_hr_bpm`` or ``abs_error_bpm`` while deciding
which candidate to select. Those columns may exist in experiment tables, but are
used only by downstream evaluation scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainRobustSelectionConfig:
    """Thresholds for the first domain-robust selector version."""

    base_score_col: str = "t144_fixed_score"
    pos_anchor_tolerance_bpm: float = 15.0
    low_lock_bpm: float = 70.0
    same_band_max_bpm: float = 90.0
    rescue_min_bpm: float = 85.0
    rescue_max_bpm: float = 135.0
    lower_neighbor_min_bpm: float = 105.0
    lower_neighbor_max_bpm: float = 125.0
    high_candidate_cut_bpm: float = 128.0


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _selected_row(row: pd.Series, policy_name: str, reason: str, confidence: float) -> dict[str, object]:
    out = row.to_dict()
    out["policy"] = policy_name
    out["score_col"] = "domain_robust_score"
    out["selected_bpm"] = _finite_float(row.get("candidate_bpm"))
    if "abs_error_bpm" in row.index:
        out["selected_abs_error_bpm"] = _finite_float(row.get("abs_error_bpm"))
    out["released"] = 1
    out["t150_reason"] = reason
    out["t150_confidence"] = confidence
    return out


def _confidence(row: pd.Series, reason: str) -> float:
    support = min(_finite_float(row.get("support_methods"), 0.0) / 6.0, 1.0)
    top1 = min(_finite_float(row.get("top1_support_methods"), 0.0) / 6.0, 1.0)
    temporal = min(_finite_float(row.get("subwindow_top1_support"), 0.0) / 24.0, 1.0)
    power = min(math.log1p(max(_finite_float(row.get("power_sum_full"), 0.0), 0.0)) / math.log(4.0), 1.0)
    penalty = 0.10 * _finite_float(row.get("motion_artifact_penalty"), 0.0)
    penalty += 0.08 * _finite_float(row.get("high_frequency_artifact_suspicion"), 0.0)
    reason_bonus = {
        "t144_base": 0.02,
        "same_band_temporal_rescue": 0.04,
        "mid_tie_rescue": 0.05,
        "pos_harmonic_rescue": 0.06,
        "extreme_low_harmonic_rescue": 0.03,
    }.get(reason, 0.0)
    return float(np.clip(0.20 + 0.22 * support + 0.20 * top1 + 0.20 * temporal + 0.16 * power + reason_bonus - penalty, 0.0, 1.0))


def select_domain_robust_candidates(
    candidates: pd.DataFrame,
    *,
    policy_name: str = "T150_domain_robust_v1",
    config: DomainRobustSelectionConfig | None = None,
) -> pd.DataFrame:
    """Select one candidate per sample with domain-robust rescue rules.

    Required inference columns include ``candidate_bpm``, ``support_methods``,
    ``top1_support_methods``, ``subwindow_support``,
    ``subwindow_top1_support``, ``power_sum_full``, ``half_harmonic_support``,
    ``motion_artifact_penalty``, ``mid_rescue_boost``, ``anchor_POS``, and the
    configured base score column.
    """

    cfg = config or DomainRobustSelectionConfig()
    rows: list[dict[str, object]] = []
    for _, group in candidates.groupby("sample_id", sort=True):
        g = group.copy()
        ranked = g.sort_values(
            [cfg.base_score_col, "support_methods", "power_sum_full"],
            ascending=[False, False, False],
        )
        selected = ranked.iloc[0].copy()
        reason = "t144_base"
        selected_bpm = _finite_float(selected.get("candidate_bpm"))
        selected_score = _finite_float(selected.get(cfg.base_score_col), 0.0)
        pos_anchor = _finite_float(g["anchor_POS"].iloc[0]) if "anchor_POS" in g.columns else math.nan

        if selected_bpm < cfg.same_band_max_bpm:
            same_band = g[
                g["candidate_bpm"].between(55.0, cfg.same_band_max_bpm)
                & (g[cfg.base_score_col] >= selected_score - 2.0)
                & (g["subwindow_top1_support"] >= _finite_float(selected.get("subwindow_top1_support"), 0.0) + 8.0)
                & (g["support_methods"] >= 5)
            ].copy()
            if not same_band.empty:
                same_band["same_band_score"] = (
                    same_band[cfg.base_score_col]
                    + 0.25 * same_band["subwindow_top1_support"]
                    + same_band["top1_support_methods"]
                    - 0.05 * (same_band["candidate_bpm"] - selected_bpm).abs()
                )
                candidate = same_band.sort_values(
                    ["same_band_score", cfg.base_score_col],
                    ascending=[False, False],
                ).iloc[0]
                if _finite_float(candidate.get("candidate_bpm")) > selected_bpm + 4.0:
                    selected = candidate.copy()
                    reason = "same_band_temporal_rescue"
                    selected_bpm = _finite_float(selected.get("candidate_bpm"))
                    selected_score = _finite_float(selected.get(cfg.base_score_col), 0.0)

        if selected_bpm < cfg.same_band_max_bpm:
            mids = g[
                g["candidate_bpm"].between(95.0, 110.0)
                & (g["mid_rescue_boost"] == 1)
                & (g[cfg.base_score_col] >= selected_score - 3.0)
                & (g["support_methods"] >= 4)
            ].copy()
            if not mids.empty:
                mids["mid_score"] = (
                    mids[cfg.base_score_col]
                    + 0.10 * mids["subwindow_support"]
                    - 0.25 * (mids["candidate_bpm"] - 102.0).abs()
                )
                selected = mids.sort_values(["mid_score", cfg.base_score_col], ascending=[False, False]).iloc[0].copy()
                reason = "mid_tie_rescue"
                selected_bpm = _finite_float(selected.get("candidate_bpm"))
                selected_score = _finite_float(selected.get(cfg.base_score_col), 0.0)

        if selected_bpm < cfg.low_lock_bpm and math.isfinite(pos_anchor):
            rescue = g[
                g["candidate_bpm"].between(cfg.rescue_min_bpm, cfg.rescue_max_bpm)
                & ((g["candidate_bpm"] - pos_anchor).abs() <= cfg.pos_anchor_tolerance_bpm)
                & (g["support_methods"] >= 4)
                & (
                    (g["top1_support_methods"] >= 1)
                    | (g["subwindow_top1_support"] >= 4)
                    | (g["power_sum_full"] >= 0.25)
                )
            ].copy()
            if not rescue.empty:
                harmonic_ok = rescue["half_harmonic_support"] >= 1
                pos_motion_ok = (
                    rescue["candidate_bpm"].between(85.0, 95.0)
                    & rescue["top1_support_methods"].between(1, 3)
                    & (rescue["subwindow_top1_support"] >= 10)
                    & (rescue["power_sum_full"] <= 1.5)
                )
                rescue = rescue[harmonic_ok | pos_motion_ok].copy()
            if not rescue.empty:
                lower_neighbor = rescue[
                    rescue["candidate_bpm"].between(cfg.lower_neighbor_min_bpm, cfg.lower_neighbor_max_bpm)
                ].copy()
                if not lower_neighbor.empty and rescue["candidate_bpm"].max() > cfg.high_candidate_cut_bpm:
                    rescue = lower_neighbor
                rescue["dist_pos"] = (rescue["candidate_bpm"] - pos_anchor).abs()
                rescue["dist_double"] = (rescue["candidate_bpm"] - 2.0 * selected_bpm).abs()
                rescue["rescue_score"] = (
                    -0.55 * rescue["dist_pos"]
                    -0.75 * rescue["dist_double"]
                    + 0.80 * rescue["top1_support_methods"]
                    + 0.05 * rescue["subwindow_top1_support"]
                    + 0.80 * np.log1p(rescue["power_sum_full"].clip(lower=0.0))
                    + 0.20 * rescue["support_methods"]
                )
                candidate = rescue.sort_values(
                    ["rescue_score", cfg.base_score_col, "power_sum_full"],
                    ascending=[False, False, False],
                ).iloc[0]
                if _finite_float(candidate.get("rescue_score"), -math.inf) > -25.0:
                    selected = candidate.copy()
                    reason = "pos_harmonic_rescue"
                    selected_bpm = _finite_float(selected.get("candidate_bpm"))
                    selected_score = _finite_float(selected.get(cfg.base_score_col), 0.0)

        if selected_bpm < cfg.low_lock_bpm:
            extreme = g[
                g["candidate_bpm"].between(90.0, 110.0)
                & (g["support_methods"] >= 5)
                & (g["subwindow_support"] >= 25)
                & (g["half_harmonic_support"] >= 5)
                & (g["power_sum_full"] >= 0.10)
            ].copy()
            if not extreme.empty:
                extreme["extreme_score"] = (
                    extreme["support_methods"]
                    + 0.10 * extreme["subwindow_support"]
                    + 0.05 * extreme["subwindow_top1_support"]
                    + np.log1p(extreme["power_sum_full"].clip(lower=0.0))
                    - 0.10 * (extreme["candidate_bpm"] - 2.0 * selected_bpm).abs()
                )
                selected = extreme.sort_values(
                    ["extreme_score", cfg.base_score_col],
                    ascending=[False, False],
                ).iloc[0].copy()
                reason = "extreme_low_harmonic_rescue"

        rows.append(_selected_row(selected, policy_name, reason, _confidence(selected, reason)))
    return pd.DataFrame(rows)

