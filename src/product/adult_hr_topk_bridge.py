from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def candidate_chain_summary(candidates: pd.DataFrame, sample_id: str, *, top_n: int = 8) -> list[dict[str, Any]]:
    sub = candidates[candidates["sample_id"].astype(str).eq(str(sample_id))].copy()
    if sub.empty:
        return []
    score_col = "t157_score" if "t157_score" in sub.columns else "support_count"
    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
    sub["candidate_bpm"] = pd.to_numeric(sub["candidate_bpm"], errors="coerce")
    sub = sub.sort_values([score_col, "support_rois", "support_methods"], ascending=[False, False, False]).head(top_n)
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "candidate_bpm": finite_float(row.get("candidate_bpm")),
                "score": finite_float(row.get(score_col)),
                "support_count": int(finite_float(row.get("support_count"), 0.0)),
                "support_rois": int(finite_float(row.get("support_rois"), 0.0)),
                "support_methods": int(finite_float(row.get("support_methods"), 0.0)),
                "pos_chrom_count": int(finite_float(row.get("pos_chrom_count"), 0.0)),
            }
        )
    return rows


def selected_reliability_features(row: pd.Series) -> dict[str, Any]:
    """Extract label-free support features for downstream reliability guards."""

    support_count = int(finite_float(row.get("support_count"), 0.0))
    support_rois = int(finite_float(row.get("support_rois"), 0.0))
    support_methods = int(finite_float(row.get("support_methods"), 0.0))
    top1_support_count = int(finite_float(row.get("top1_support_count"), 0.0))
    pos_chrom_count = int(finite_float(row.get("pos_chrom_count"), 0.0))
    score = finite_float(row.get("t157_score"), finite_float(row.get("support_count")))
    anchor_bpm = finite_float(row.get("bridge_anchor_bpm"), finite_float(row.get("t150_selected_bpm")))
    candidate_bpm = finite_float(row.get("candidate_bpm"))
    dist_to_anchor = finite_float(row.get("bridge_dist_to_anchor"), abs(candidate_bpm - anchor_bpm))
    return {
        "selected_support_count": support_count,
        "selected_support_rois": support_rois,
        "selected_support_methods": support_methods,
        "selected_top1_support_count": top1_support_count,
        "selected_pos_chrom_count": pos_chrom_count,
        "selected_green_pbv_count": int(finite_float(row.get("green_pbv_count"), 0.0)),
        "selected_ica_lgi_count": int(finite_float(row.get("ica_lgi_count"), 0.0)),
        "selected_score": score,
        "selected_mean_power_fraction": finite_float(row.get("mean_power_fraction")),
        "selected_max_power_fraction": finite_float(row.get("max_power_fraction")),
        "selected_sum_power_fraction": finite_float(row.get("sum_power_fraction")),
        "selected_rank_score": finite_float(row.get("rank_score")),
        "selected_mean_snr_proxy_db": finite_float(row.get("mean_snr_proxy_db")),
        "selected_adult_plausibility": finite_float(row.get("adult_plausibility")),
        "selected_dist_to_anchor_bpm": dist_to_anchor,
        "selected_support_sum": support_rois + support_methods + top1_support_count,
        "selected_methods": str(row.get("methods", "")),
        "selected_regions": str(row.get("regions", "")),
    }


def build_adult_hr_bridge_product_table(selection: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Convert T216-style bridge selection into a product/backend response table.

    The returned table is label-free by default: ground truth and errors are
    intentionally not required. If evaluation columns exist in the input, they
    are passed through only with `eval_` prefixes for research dashboards.
    """

    rows: list[dict[str, Any]] = []
    for _, row in selection.iterrows():
        sample_id = str(row.get("sample_id", ""))
        released = int(finite_float(row.get("released"), 0.0)) > 0
        product_hr = finite_float(row.get("selected_bpm"))
        chain = candidate_chain_summary(candidates, sample_id)
        reliability = selected_reliability_features(row)
        bridge_source = str(row.get("correction_source", ""))
        decision = "release" if released and math.isfinite(product_hr) else "review"
        if bridge_source in {"empty_candidate_pool", "empty_finite_candidate_pool"}:
            decision = "review"
        evidence = {
            "bridge_source": bridge_source,
            "anchor_bpm": finite_float(row.get("bridge_anchor_bpm")),
            "anchor_released": int(finite_float(row.get("bridge_anchor_released"), 0.0)),
            "anchor_review_reason": str(row.get("bridge_anchor_review_reason", "")),
            "selected_candidate_bpm": finite_float(row.get("candidate_bpm")),
            "selected_candidate_id": str(row.get("candidate_id", "")),
            "selected_reliability_features": reliability,
            "candidate_chain": chain,
        }
        out: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset": row.get("dataset", ""),
            "subject_id": row.get("subject_id", ""),
            "session_id": row.get("session_id", ""),
            "condition_group": row.get("condition_group", ""),
            "decision": decision,
            "released": int(decision == "release"),
            "product_hr_bpm": product_hr if decision == "release" else math.nan,
            "candidate_hr_bpm": finite_float(row.get("candidate_bpm")),
            "bridge_source": bridge_source,
            "bridge_anchor_bpm": finite_float(row.get("bridge_anchor_bpm")),
            **reliability,
            "evidence_json": json.dumps(evidence, ensure_ascii=False),
        }
        if "gt_hr_bpm" in row:
            out["eval_gt_hr_bpm"] = finite_float(row.get("gt_hr_bpm"))
        if "selected_abs_error_bpm" in row:
            out["eval_abs_error_bpm"] = finite_float(row.get("selected_abs_error_bpm"))
        rows.append(out)
    return pd.DataFrame(rows)
