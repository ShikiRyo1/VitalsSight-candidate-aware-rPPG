from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.product.adult_hr_unified_policy import summarize_unified_policy


EXPERIMENTAL_MODE = "experimental_frozen_recovery"
EXPERIMENTAL_MODE_LABEL = "Experimental adult HR policy: label-free router + frozen multi-ROI recovery"
EXPERIMENTAL_CLAIM_BOUNDARY = (
    "Experimental replay/API product mode. It combines T354 label-free safety routing with T356 frozen "
    "multi-ROI recovery, but remains research evidence and is not a clinical or final SOTA claim."
)


BRANCH_DISPLAY_NAMES = {
    "topk_bridge": "Top-k bridge release",
    "source_aware_repair": "Source-aware repair release",
    "multiroi_consensus_recovery": "Multi-ROI consensus recovery",
    "review_retest": "Review/retest",
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


def standardize_experimental_frozen_recovery_product(product: pd.DataFrame) -> pd.DataFrame:
    """Expose the frozen T356 replay table as one product/API mode."""

    out = product.copy()
    if out.empty:
        return out
    out["source_product_mode"] = out.get("product_mode", "")
    out["product_mode"] = EXPERIMENTAL_MODE
    out["product_mode_label"] = EXPERIMENTAL_MODE_LABEL
    out["claim_boundary"] = EXPERIMENTAL_CLAIM_BOUNDARY
    out["branch_display_name"] = out["policy_branch"].astype(str).map(BRANCH_DISPLAY_NAMES).fillna(out["policy_branch"].astype(str))

    released = pd.to_numeric(out.get("released", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    out["released"] = released
    out["decision"] = np.where(released.gt(0), "release", "review")
    out["product_hr_bpm"] = pd.to_numeric(out.get("product_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce").where(
        released.gt(0),
        np.nan,
    )

    evidence_rows: list[str] = []
    for _, row in out.iterrows():
        original = _safe_load_json(row.get("evidence_json", ""))
        evidence_rows.append(
            _json_dumps(
                {
                    "product_mode": EXPERIMENTAL_MODE,
                    "product_mode_label": EXPERIMENTAL_MODE_LABEL,
                    "policy_branch": row.get("policy_branch", ""),
                    "branch_display_name": row.get("branch_display_name", ""),
                    "decision": row.get("decision", ""),
                    "released": int(row.get("released", 0)),
                    "claim_boundary": EXPERIMENTAL_CLAIM_BOUNDARY,
                    "source_product_mode": row.get("source_product_mode", ""),
                    "original_t356_evidence": original,
                }
            )
        )
    out["evidence_json"] = evidence_rows

    first_cols = [
        "sample_id",
        "dataset",
        "analysis_split",
        "product_mode",
        "product_mode_label",
        "policy_branch",
        "branch_display_name",
        "decision",
        "released",
        "product_hr_bpm",
        "candidate_hr_bpm",
        "source",
        "source_kind",
        "window_candidate_count",
        "repair_probability",
        "claim_boundary",
        "review_reason",
        "eval_gt_hr_bpm",
        "eval_abs_error_bpm",
        "evidence_json",
    ]
    return out[[col for col in first_cols if col in out.columns] + [col for col in out.columns if col not in first_cols]].copy()


def summarize_experimental_product(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    return summarize_unified_policy(product, group_cols or ["dataset", "policy_branch"])


def build_experimental_api_examples(product: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for branch in ["topk_bridge", "source_aware_repair", "multiroi_consensus_recovery", "review_retest"]:
        rows = product[product["policy_branch"].astype(str).eq(branch)].copy()
        if rows.empty:
            continue
        row = rows.sort_values(["dataset", "sample_id"]).iloc[0].to_dict()
        examples.append(
            {
                "example_type": branch,
                "sample_id": row.get("sample_id", ""),
                "dataset_for_eval_only": row.get("dataset", ""),
                "product_mode": row.get("product_mode", ""),
                "policy_branch": row.get("policy_branch", ""),
                "branch_display_name": row.get("branch_display_name", ""),
                "decision": row.get("decision", ""),
                "released": int(row.get("released", 0)),
                "product_hr_bpm": None if pd.isna(row.get("product_hr_bpm")) else float(row.get("product_hr_bpm")),
                "candidate_hr_bpm": None if pd.isna(row.get("candidate_hr_bpm")) else float(row.get("candidate_hr_bpm")),
                "review_reason": "" if pd.isna(row.get("review_reason")) else str(row.get("review_reason", "")),
                "evidence": _safe_load_json(row.get("evidence_json", "")),
            }
        )
    return examples
