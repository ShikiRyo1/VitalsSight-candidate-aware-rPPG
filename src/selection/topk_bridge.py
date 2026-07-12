from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TopKBridgeConfig:
    """Inference-only bridge from an anchor estimate to a top-K candidate pool."""

    min_support_methods: int = 4
    min_support_rois: int = 4
    high_hr_anchor_bpm: float = 120.0
    low_alias_threshold_bpm: float = 75.0
    nearest_anchor_tolerance_bpm: float = 8.0
    enable_upper_band_rescue: bool = False
    upper_band_low_anchor_min_bpm: float = 75.0
    upper_band_low_anchor_max_bpm: float = 90.0
    upper_band_low_candidate_min_bpm: float = 92.0
    upper_band_low_candidate_max_bpm: float = 116.0
    upper_band_low_min_delta_bpm: float = 10.0
    upper_band_low_score_slack: float = 2.25
    upper_band_normal_anchor_min_bpm: float = 90.0
    upper_band_normal_anchor_max_bpm: float = 101.0
    upper_band_normal_min_delta_bpm: float = 4.0
    upper_band_normal_max_delta_bpm: float = 12.0
    upper_band_normal_score_slack: float = 6.0
    upper_band_min_support_count: int = 8


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def candidate_score(row: pd.Series) -> float:
    """Return the selector score from either historical or live candidate tables."""

    score = finite_float(row.get("t157_score"))
    if math.isfinite(score):
        return score
    return finite_float(row.get("score"))


def choose_upper_band_rescue_candidate(
    candidates: pd.DataFrame,
    anchor_bpm: float,
    anchor_candidate_score: float,
    *,
    config: TopKBridgeConfig | None = None,
) -> tuple[pd.Series | None, str]:
    """Choose a constrained upper-band rescue candidate without labels.

    This targets a failure mode found in T221: a low or low-normal anchor may be
    preserved even when the candidate chain contains a strongly supported upper
    physiological hypothesis. The rule is deliberately narrow and disabled by
    default so prior experiments remain reproducible.
    """

    cfg = config or TopKBridgeConfig()
    if not cfg.enable_upper_band_rescue or candidates.empty:
        return None, "upper_band_rescue_disabled"
    if not math.isfinite(anchor_bpm) or not math.isfinite(anchor_candidate_score):
        return None, "upper_band_missing_anchor_score"

    work = candidates.copy()
    work["candidate_bpm"] = pd.to_numeric(work["candidate_bpm"], errors="coerce")
    work["_bridge_score"] = work.apply(candidate_score, axis=1)
    for col in ["support_count", "support_rois", "support_methods"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = 0.0

    if cfg.upper_band_low_anchor_min_bpm <= anchor_bpm <= cfg.upper_band_low_anchor_max_bpm:
        viable = work[
            work["candidate_bpm"].between(cfg.upper_band_low_candidate_min_bpm, cfg.upper_band_low_candidate_max_bpm)
            & ((work["candidate_bpm"] - anchor_bpm) >= cfg.upper_band_low_min_delta_bpm)
            & (work["support_methods"] >= cfg.min_support_methods + 1)
            & (work["support_rois"] >= max(3, cfg.min_support_rois - 1))
            & (work["support_count"] >= max(10, cfg.upper_band_min_support_count))
            & (work["_bridge_score"] >= anchor_candidate_score - cfg.upper_band_low_score_slack)
        ].copy()
    elif cfg.upper_band_normal_anchor_min_bpm < anchor_bpm <= cfg.upper_band_normal_anchor_max_bpm:
        viable = work[
            work["candidate_bpm"].between(anchor_bpm + cfg.upper_band_normal_min_delta_bpm, min(116.0, anchor_bpm + cfg.upper_band_normal_max_delta_bpm))
            & (work["support_methods"] >= cfg.min_support_methods + 1)
            & (work["support_rois"] >= max(3, cfg.min_support_rois - 1))
            & (work["support_count"] >= cfg.upper_band_min_support_count)
            & (work["_bridge_score"] >= anchor_candidate_score - cfg.upper_band_normal_score_slack)
        ].copy()
    else:
        viable = work.iloc[0:0].copy()

    if viable.empty:
        return None, "upper_band_no_viable_candidate"
    viable["_upper_dist_to_anchor"] = (viable["candidate_bpm"] - anchor_bpm).abs()
    selected = viable.sort_values(
        ["_bridge_score", "support_count", "support_rois", "support_methods", "_upper_dist_to_anchor"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return selected, "upper_band_rescue_v1"


def choose_low_alias_upper_candidate(
    candidates: pd.DataFrame,
    anchor_bpm: float,
    *,
    config: TopKBridgeConfig | None = None,
) -> tuple[pd.Series, str]:
    """Choose a plausible upper candidate when the anchor is a low alias.

    The bounds are intentionally local to the anchor value, so this does not use
    labels or dataset identity. It asks whether a nearby upper physiological
    hypothesis has enough ROI/method support to replace the low anchor.
    """

    cfg = config or TopKBridgeConfig()
    if anchor_bpm < 62.0:
        low = max(55.0, anchor_bpm + 3.0)
        high = min(78.0, anchor_bpm + 18.0)
    elif anchor_bpm < cfg.low_alias_threshold_bpm:
        low = max(74.0, anchor_bpm + 8.0)
        high = min(105.0, anchor_bpm + 28.0)
    else:
        low = math.nan
        high = math.nan

    viable = candidates.iloc[0:0].copy()
    if math.isfinite(low) and math.isfinite(high):
        viable = candidates[
            pd.to_numeric(candidates["candidate_bpm"], errors="coerce").between(low, high)
            & (pd.to_numeric(candidates["support_methods"], errors="coerce") >= cfg.min_support_methods)
            & (pd.to_numeric(candidates["support_rois"], errors="coerce") >= cfg.min_support_rois)
        ].copy()
    if viable.empty:
        fallback = candidates.assign(_bridge_dist_to_anchor=(pd.to_numeric(candidates["candidate_bpm"], errors="coerce") - anchor_bpm).abs())
        return fallback.sort_values(["_bridge_dist_to_anchor", "support_methods", "support_rois"], ascending=[True, False, False]).iloc[0], "fallback_nearest_anchor"

    viable["_bridge_dist_to_anchor"] = (pd.to_numeric(viable["candidate_bpm"], errors="coerce") - anchor_bpm).abs()
    selected = viable.sort_values(["candidate_bpm", "support_methods", "support_rois", "support_count"], ascending=[True, False, False, False]).iloc[0]
    return selected, "low_alias_upper_rescue"


def choose_topk_bridge_candidate(
    candidates: pd.DataFrame,
    *,
    anchor_bpm: float,
    anchor_released: bool,
    anchor_review_reason: str = "",
    anchor_selected_bpm: float | None = None,
    config: TopKBridgeConfig | None = None,
) -> dict[str, Any]:
    """Select a final HR from a top-K candidate pool without reading labels.

    The bridge protects safe anchors, including high-HR anchors, and only uses
    the top-K pool for constrained rescue when the upstream gate reports a
    conflict. This avoids the common failure where the largest support cluster
    is an artifact shared across many ROI/method streams.
    """

    cfg = config or TopKBridgeConfig()
    if candidates.empty:
        return {
            "selected_bpm": math.nan,
            "candidate_bpm": math.nan,
            "candidate_id": "",
            "correction_source": "empty_candidate_pool",
            "release_status": "review",
            "released": 0,
        }

    work = candidates.copy()
    work["candidate_bpm"] = pd.to_numeric(work["candidate_bpm"], errors="coerce")
    work = work[np.isfinite(work["candidate_bpm"])]
    if work.empty:
        return {
            "selected_bpm": math.nan,
            "candidate_bpm": math.nan,
            "candidate_id": "",
            "correction_source": "empty_finite_candidate_pool",
            "release_status": "review",
            "released": 0,
        }

    anchor = finite_float(anchor_selected_bpm, anchor_bpm)
    nearest = work.assign(_bridge_dist_to_anchor=(work["candidate_bpm"] - anchor).abs()).sort_values(
        ["_bridge_dist_to_anchor", "support_methods", "support_rois", "support_count"],
        ascending=[True, False, False, False],
    ).iloc[0]

    if anchor_released:
        if anchor < cfg.high_hr_anchor_bpm and cfg.enable_upper_band_rescue:
            rescued, rescue_source = choose_upper_band_rescue_candidate(work, anchor, candidate_score(nearest), config=cfg)
            if rescued is not None:
                selected = rescued.to_dict()
                selected["selected_bpm"] = finite_float(rescued.get("candidate_bpm"))
                selected["correction_source"] = rescue_source
                selected["release_status"] = "release"
                selected["released"] = 1
                selected["bridge_dist_to_anchor"] = abs(finite_float(rescued.get("candidate_bpm")) - anchor)
                return selected
        source = "anchor_preserved"
        if anchor >= cfg.high_hr_anchor_bpm:
            source = "high_hr_anchor_preserved"
        elif finite_float(nearest.get("_bridge_dist_to_anchor")) <= cfg.nearest_anchor_tolerance_bpm:
            source = "anchor_nearest_topk_consistent"
        selected = nearest.to_dict()
        selected["selected_bpm"] = anchor
        selected["correction_source"] = source
        selected["release_status"] = "release"
        selected["released"] = 1
        selected["bridge_dist_to_anchor"] = finite_float(nearest.get("_bridge_dist_to_anchor"))
        return selected

    reason = str(anchor_review_reason or "").lower()
    if "low_alias" in reason or anchor < cfg.low_alias_threshold_bpm:
        corrected, source = choose_low_alias_upper_candidate(work, anchor, config=cfg)
        selected = corrected.to_dict()
        selected["selected_bpm"] = finite_float(corrected["candidate_bpm"])
        selected["correction_source"] = source
        selected["release_status"] = "release"
        selected["released"] = 1
        selected["bridge_dist_to_anchor"] = finite_float(corrected.get("_bridge_dist_to_anchor"))
        return selected

    selected = nearest.to_dict()
    selected["selected_bpm"] = anchor
    selected["correction_source"] = "reviewed_anchor_fallback"
    selected["release_status"] = "release"
    selected["released"] = 1
    selected["bridge_dist_to_anchor"] = finite_float(nearest.get("_bridge_dist_to_anchor"))
    return selected


def select_topk_bridge(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    config: TopKBridgeConfig | None = None,
    policy_name: str = "T216_topk_bridge_v1",
) -> pd.DataFrame:
    cfg = config or TopKBridgeConfig()
    anchor_index = anchors.drop_duplicates("sample_id").set_index("sample_id") if not anchors.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        if sample_id in anchor_index.index:
            anchor_row = anchor_index.loc[sample_id]
            anchor_bpm = finite_float(anchor_row.get("selected_bpm"), finite_float(group["t150_selected_bpm"].iloc[0]))
            anchor_released = int(finite_float(anchor_row.get("released"), 1.0)) > 0
            review_reason = str(anchor_row.get("review_reason", ""))
        else:
            anchor_bpm = finite_float(group["t150_selected_bpm"].iloc[0])
            anchor_released = True
            review_reason = ""
        selected = choose_topk_bridge_candidate(
            group,
            anchor_bpm=anchor_bpm,
            anchor_released=anchor_released,
            anchor_review_reason=review_reason,
            anchor_selected_bpm=anchor_bpm,
            config=cfg,
        )
        selected["policy"] = policy_name
        selected["sample_id"] = sample_id
        selected["bridge_anchor_bpm"] = anchor_bpm
        selected["bridge_anchor_released"] = int(anchor_released)
        selected["bridge_anchor_review_reason"] = review_reason
        rows.append(selected)
    return pd.DataFrame(rows)
