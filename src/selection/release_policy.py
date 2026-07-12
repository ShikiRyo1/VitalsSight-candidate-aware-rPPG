"""Selective-release policies for contactless vital-sign estimates.

The functions in this module are inference-only. They may consume model outputs,
candidate-support features, anchors, ROI names, and selector reasons, but they
must not read ground-truth columns when deciding whether to release an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleReleaseConfig:
    """Single-ROI/sample selective-release thresholds."""

    min_confidence: float = 0.60
    max_conflict_score: float = 2.0
    low_confidence_cut: float = 0.55
    far_anchor_bpm: float = 20.0
    low_anchor_bpm: float = 80.0
    anchor_iqr_max_bpm: float = 10.0
    high_against_low_anchor_bpm: float = 90.0
    weak_top1_support_max: float = 1.0
    weak_subwindow_top1_max: float = 4.0


@dataclass(frozen=True)
class MultiROIReleaseConfig:
    """Multi-ROI consensus thresholds for one input video/subject."""

    max_roi_range_bpm: float = 30.0
    max_rescue_count: int = 1
    aggregator: str = "median"


@dataclass(frozen=True)
class ProductReleaseConfig:
    """Hybrid product policy: multi-ROI if available, otherwise sample gate."""

    sample: SampleReleaseConfig = field(default_factory=SampleReleaseConfig)
    multi_roi: MultiROIReleaseConfig = field(default_factory=MultiROIReleaseConfig)
    min_roi_for_consensus: int = 3


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_sample_risk_features(
    selection: pd.DataFrame,
    *,
    config: SampleReleaseConfig | None = None,
) -> pd.DataFrame:
    """Add non-leaking candidate-conflict features to selected estimates."""

    cfg = config or SampleReleaseConfig()
    out = _to_numeric(
        selection,
        [
            "selected_bpm",
            "t150_confidence",
            "anchor_median",
            "anchor_iqr",
            "top1_support_methods",
            "subwindow_top1_support",
            "dist_to_POS",
            "dist_to_GREEN",
            "selected_abs_error_bpm",
            "gt_hr_bpm",
        ],
    )
    out["t150_reason"] = out.get("t150_reason", "").fillna("")
    out["rescue_applied"] = out["t150_reason"] != "t144_base"
    out["selected_far_from_anchor_median_bpm"] = (out["selected_bpm"] - out["anchor_median"]).abs()
    out["selected_far_from_anchor"] = out["selected_far_from_anchor_median_bpm"] > cfg.far_anchor_bpm
    out["anchor_low_consensus"] = (
        (out["anchor_median"] < cfg.low_anchor_bpm)
        & (out["anchor_iqr"] <= cfg.anchor_iqr_max_bpm)
    )
    out["selected_high_against_low_anchor"] = (
        (out["selected_bpm"] >= cfg.high_against_low_anchor_bpm)
        & out["anchor_low_consensus"]
        & out["selected_far_from_anchor"]
    )
    out["low_confidence"] = out["t150_confidence"] < cfg.low_confidence_cut
    out["weak_temporal_selected"] = (
        (out["top1_support_methods"] <= cfg.weak_top1_support_max)
        & (out["subwindow_top1_support"] <= cfg.weak_subwindow_top1_max)
    )
    out["candidate_conflict_score"] = (
        0.8 * out["rescue_applied"].astype(float)
        + 1.4 * out["selected_high_against_low_anchor"].astype(float)
        + 0.7 * out["low_confidence"].astype(float)
        + 0.5 * out["selected_far_from_anchor"].astype(float)
        + 0.4 * out["weak_temporal_selected"].astype(float)
    )
    return out


def apply_sample_release_gate(
    selection: pd.DataFrame,
    *,
    policy_name: str,
    config: SampleReleaseConfig | None = None,
) -> pd.DataFrame:
    """Apply single-sample confidence/conflict gate."""

    cfg = config or SampleReleaseConfig()
    out = add_sample_risk_features(selection, config=cfg)
    out["released"] = (
        (out["t150_confidence"] >= cfg.min_confidence)
        & (out["candidate_conflict_score"] < cfg.max_conflict_score)
    ).astype(int)
    reasons: list[str] = []
    for _, row in out.iterrows():
        if int(row["released"]) == 1:
            reasons.append("release")
            continue
        flags: list[str] = []
        if _finite_float(row.get("t150_confidence")) < cfg.min_confidence:
            flags.append("low_confidence")
        if _finite_float(row.get("candidate_conflict_score")) >= cfg.max_conflict_score:
            flags.append("candidate_conflict")
        if bool(row.get("selected_high_against_low_anchor")):
            flags.append("high_against_low_anchor")
        if bool(row.get("weak_temporal_selected")):
            flags.append("weak_temporal_support")
        reasons.append(";".join(flags) if flags else "review")
    out["release_status"] = np.where(out["released"] == 1, "release", "review")
    out["review_reason"] = reasons
    out["policy"] = policy_name
    out["policy_scope"] = "sample"
    return out


def multi_roi_consensus_row(
    group: pd.DataFrame,
    *,
    policy_name: str,
    config: MultiROIReleaseConfig | None = None,
) -> dict[str, object]:
    """Aggregate multiple ROI predictions for one deployment unit."""

    cfg = config or MultiROIReleaseConfig()
    g = group.copy()
    preds = pd.to_numeric(g["selected_bpm"], errors="coerce").to_numpy(dtype=float)
    preds = preds[np.isfinite(preds)]
    if len(preds) == 0:
        selected = math.nan
        roi_range = math.nan
        roi_iqr = math.nan
    else:
        selected = float(np.median(preds) if cfg.aggregator == "median" else np.mean(preds))
        roi_range = float(np.max(preds) - np.min(preds))
        roi_iqr = float(np.percentile(preds, 75) - np.percentile(preds, 25))

    rescue_count = int((g.get("rescue_applied", pd.Series(False, index=g.index)).astype(bool)).sum())
    low_conf_count = int((g.get("low_confidence", pd.Series(False, index=g.index)).astype(bool)).sum())
    weak_count = int((g.get("weak_temporal_selected", pd.Series(False, index=g.index)).astype(bool)).sum())
    mean_conf = float(pd.to_numeric(g.get("t150_confidence"), errors="coerce").mean())
    released = bool(
        len(preds) >= 2
        and math.isfinite(roi_range)
        and roi_range <= cfg.max_roi_range_bpm
        and rescue_count <= cfg.max_rescue_count
    )
    if released:
        reason = "multi_roi_consensus"
    elif not math.isfinite(roi_range):
        reason = "no_valid_roi_prediction"
    elif roi_range > cfg.max_roi_range_bpm:
        reason = "roi_disagreement"
    elif rescue_count > cfg.max_rescue_count:
        reason = "too_many_rescue_rois"
    else:
        reason = "review"

    first = g.iloc[0]
    gt = _finite_float(first.get("gt_hr_bpm"))
    abs_error = abs(selected - gt) if math.isfinite(selected) and math.isfinite(gt) else math.nan
    return {
        "task_id": "T153_T154",
        "dataset": first.get("dataset"),
        "deployment_id": first.get("deployment_id"),
        "subject_id": first.get("subject_id"),
        "session_id": first.get("session_id", first.get("sample_id")),
        "roi_name": "multi_roi_consensus",
        "n_roi": int(len(g)),
        "policy": policy_name,
        "policy_scope": "deployment_unit",
        "selected_bpm": selected,
        "gt_hr_bpm": gt,
        "selected_abs_error_bpm": abs_error,
        "released": int(released),
        "release_status": "release" if released else "review",
        "review_reason": reason,
        "roi_prediction_range_bpm": roi_range,
        "roi_prediction_iqr_bpm": roi_iqr,
        "roi_mean_confidence": mean_conf,
        "roi_min_confidence": float(pd.to_numeric(g.get("t150_confidence"), errors="coerce").min()),
        "roi_rescue_count": rescue_count,
        "roi_low_confidence_count": low_conf_count,
        "roi_weak_temporal_count": weak_count,
        "roi_predictions": ",".join(f"{value:.3f}" for value in preds),
    }


def build_hybrid_deployment_policy(
    selection: pd.DataFrame,
    *,
    policy_name: str,
    config: ProductReleaseConfig | None = None,
) -> pd.DataFrame:
    """Build product-level release rows.

    If a deployment unit has enough ROI rows, use multi-ROI consensus. Otherwise
    apply the sample-level gate and keep one row per input sample.
    """

    cfg = config or ProductReleaseConfig()
    rows: list[dict[str, object]] = []
    featured = add_sample_risk_features(selection, config=cfg.sample)
    for _, group in featured.groupby("deployment_id", sort=True):
        if len(group) >= cfg.min_roi_for_consensus:
            rows.append(multi_roi_consensus_row(group, policy_name=policy_name, config=cfg.multi_roi))
        else:
            gated = apply_sample_release_gate(group, policy_name=policy_name, config=cfg.sample)
            for _, row in gated.iterrows():
                out = row.to_dict()
                out["deployment_id"] = row.get("deployment_id")
                out["policy_scope"] = "deployment_unit"
                out["n_roi"] = 1
                out["roi_prediction_range_bpm"] = 0.0
                out["roi_prediction_iqr_bpm"] = 0.0
                out["roi_predictions"] = f"{_finite_float(row.get('selected_bpm')):.3f}"
                rows.append(out)
    return pd.DataFrame(rows)


def build_release_all_deployment_policy(
    selection: pd.DataFrame,
    *,
    policy_name: str = "T150_deployment_release_all",
    min_roi_for_consensus: int = 3,
) -> pd.DataFrame:
    """Deployment-level release-all baseline.

    Multi-ROI units use the median prediction; single-ROI units release the
    original selected estimate.
    """

    featured = add_sample_risk_features(selection)
    rows: list[dict[str, object]] = []
    for _, group in featured.groupby("deployment_id", sort=True):
        if len(group) >= min_roi_for_consensus:
            cfg = MultiROIReleaseConfig(max_roi_range_bpm=math.inf, max_rescue_count=10**9)
            rows.append(multi_roi_consensus_row(group, policy_name=policy_name, config=cfg))
        else:
            row = group.iloc[0].copy()
            out = row.to_dict()
            out["policy"] = policy_name
            out["policy_scope"] = "deployment_unit"
            out["deployment_id"] = row.get("deployment_id")
            out["n_roi"] = 1
            out["released"] = 1
            out["release_status"] = "release"
            out["review_reason"] = "release_all"
            out["roi_prediction_range_bpm"] = 0.0
            out["roi_prediction_iqr_bpm"] = 0.0
            out["roi_predictions"] = f"{_finite_float(row.get('selected_bpm')):.3f}"
            rows.append(out)
    return pd.DataFrame(rows)
