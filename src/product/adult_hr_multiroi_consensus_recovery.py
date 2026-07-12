from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd


RECOVERY_MODE = "multiroi_consensus_recovery"
RECOVERY_LABEL = "Label-free multi-ROI consensus recovery after safety review"
RECOVERY_BOUNDARY = (
    "Replay screen for recovering reviewed windows with multi-ROI agreement. "
    "Requires locked validation before becoming the default live product policy."
)

DEFAULT_RECOVERY_POLICY = {
    "min_group_size": 3,
    "min_candidate_hr_bpm": 55.0,
    "max_group_range_bpm": 12.0,
    "max_abs_to_group_median_bpm": 2.0,
}


def _safe_load_json(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)


def derive_multiroi_group_key(sample_id: object) -> str:
    """Approximate live multi-ROI grouping from replay sample ids."""

    text = str(sample_id)
    text = re.sub(r"_(cheek1|cheek2|forehead|left_cheek|right_cheek|nose|chin)$", "", text)
    text = re.sub(r"_w\d+$", "", text)
    return text


def add_multiroi_consensus_features(
    product: pd.DataFrame,
    policy: dict[str, float] | None = None,
) -> pd.DataFrame:
    policy = {**DEFAULT_RECOVERY_POLICY, **(policy or {})}
    features = product[["sample_id", "candidate_hr_bpm"]].copy()
    features["candidate_hr_bpm"] = pd.to_numeric(features["candidate_hr_bpm"], errors="coerce")
    features["consensus_group_key"] = features["sample_id"].map(derive_multiroi_group_key)

    grouped = (
        features.groupby("consensus_group_key", sort=False)["candidate_hr_bpm"]
        .agg(["count", "median", "min", "max", "std"])
        .rename(
            columns={
                "count": "consensus_group_size",
                "median": "consensus_group_median_bpm",
                "min": "consensus_group_min_bpm",
                "max": "consensus_group_max_bpm",
                "std": "consensus_group_std_bpm",
            }
        )
        .reset_index()
    )
    grouped["consensus_group_range_bpm"] = grouped["consensus_group_max_bpm"] - grouped["consensus_group_min_bpm"]
    features = features.merge(grouped, on="consensus_group_key", how="left")
    features["abs_to_consensus_median_bpm"] = (
        features["candidate_hr_bpm"] - features["consensus_group_median_bpm"]
    ).abs()
    features["multiroi_consensus_confident"] = (
        features["consensus_group_size"].ge(float(policy["min_group_size"]))
        & features["candidate_hr_bpm"].ge(float(policy["min_candidate_hr_bpm"]))
        & features["consensus_group_range_bpm"].le(float(policy["max_group_range_bpm"]))
        & features["abs_to_consensus_median_bpm"].le(float(policy["max_abs_to_group_median_bpm"]))
    )
    return features


def apply_multiroi_consensus_recovery(
    product: pd.DataFrame,
    policy: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = {**DEFAULT_RECOVERY_POLICY, **(policy or {})}
    recovered = product.copy()
    features = add_multiroi_consensus_features(recovered, policy)
    if recovered.empty:
        return recovered, features

    was_review = recovered["policy_branch"].astype(str).eq("review_retest")
    can_recover = was_review & features["multiroi_consensus_confident"].to_numpy()

    recovered["pre_recovery_policy_branch"] = recovered.get("policy_branch", "")
    recovered["pre_recovery_decision"] = recovered.get("decision", "")
    recovered["pre_recovery_released"] = recovered.get("released", 0)
    recovered["pre_recovery_product_hr_bpm"] = recovered.get("product_hr_bpm", np.nan)

    recovered.loc[can_recover, "policy_branch"] = RECOVERY_MODE
    recovered.loc[can_recover, "product_mode"] = RECOVERY_MODE
    recovered.loc[can_recover, "product_mode_label"] = RECOVERY_LABEL
    recovered.loc[can_recover, "decision"] = "release"
    recovered.loc[can_recover, "released"] = 1
    recovered.loc[can_recover, "product_hr_bpm"] = pd.to_numeric(
        recovered.loc[can_recover, "candidate_hr_bpm"], errors="coerce"
    )
    recovered.loc[can_recover, "review_reason"] = ""
    recovered.loc[can_recover, "claim_boundary"] = RECOVERY_BOUNDARY

    evidence_rows: list[str] = []
    for idx, row in recovered.iterrows():
        original = _safe_load_json(row.get("evidence_json", ""))
        feat = features.loc[idx].to_dict()
        evidence_rows.append(
            _json_dumps(
                {
                    "product_mode": row.get("product_mode", ""),
                    "policy_branch": row.get("policy_branch", ""),
                    "decision": row.get("decision", ""),
                    "recovered_by_t355": bool(can_recover.loc[idx]),
                    "recovery_policy": policy,
                    "multiroi_consensus_features": {
                        key: feat.get(key)
                        for key in [
                            "consensus_group_key",
                            "consensus_group_size",
                            "consensus_group_median_bpm",
                            "consensus_group_range_bpm",
                            "abs_to_consensus_median_bpm",
                            "multiroi_consensus_confident",
                        ]
                    },
                    "claim_boundary": RECOVERY_BOUNDARY if bool(can_recover.loc[idx]) else row.get("claim_boundary", ""),
                    "original_t354_evidence": original,
                }
            )
        )
    recovered["evidence_json"] = evidence_rows
    return recovered, features


def summarize_product(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["dataset", "policy_branch"]
    rows: list[dict[str, Any]] = []
    for keys, group in product.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        released = pd.to_numeric(group["released"], errors="coerce").fillna(0).astype(int).gt(0)
        err = pd.to_numeric(group.get("eval_abs_error_bpm", pd.Series(np.nan, index=group.index)), errors="coerce")
        released_err = err[released & np.isfinite(err)]
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_rows": int(len(group)),
                "coverage": float(released.mean()) if len(group) else math.nan,
                "released_mae_bpm": float(released_err.mean()) if len(released_err) else math.nan,
                "released_unsafe_gt10bpm_rate": float((released_err > 10.0).mean()) if len(released_err) else math.nan,
                "published_unsafe_per_input": float(((err > 10.0) & released).sum() / len(group)) if len(group) else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
