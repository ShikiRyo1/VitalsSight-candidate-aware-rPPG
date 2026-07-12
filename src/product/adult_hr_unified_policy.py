from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


UNIFIED_MODE = "unified_adult_hr"
UNIFIED_MODE_LABEL = "Unified adult HR policy: top-k bridge + source-aware repair"
UNIFIED_CLAIM_BOUNDARY = (
    "Unified replay/product contract; combines validated branches but remains research/demo evidence, not clinical-grade monitoring."
)


def _as_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_json_loads(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)


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


def standardize_topk_bridge_product_for_unified(bridge: pd.DataFrame) -> pd.DataFrame:
    """Convert the T217 top-k bridge table into the unified product contract."""

    if bridge.empty:
        return pd.DataFrame()
    out = bridge.copy()
    released = pd.to_numeric(out.get("released", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    product = pd.DataFrame(
        {
            "sample_id": out.get("sample_id", pd.Series("", index=out.index)).astype(str),
            "dataset": out.get("dataset", pd.Series("", index=out.index)).astype(str),
            "analysis_split": out.get("analysis_split", pd.Series("topk_bridge_replay", index=out.index)).astype(str),
            "product_mode": UNIFIED_MODE,
            "product_mode_label": UNIFIED_MODE_LABEL,
            "policy_branch": "topk_bridge",
            "branch_source_task": "T217",
            "decision": out.get("decision", pd.Series("review", index=out.index)).astype(str),
            "released": released,
            "product_hr_bpm": pd.to_numeric(out.get("product_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce").where(released.gt(0), np.nan),
            "candidate_hr_bpm": pd.to_numeric(out.get("candidate_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce"),
            "source": out.get("bridge_source", pd.Series("topk_bridge", index=out.index)).astype(str),
            "source_kind": "topk_bridge",
            "window_candidate_count": np.nan,
            "claim_boundary": UNIFIED_CLAIM_BOUNDARY,
            "artifact_name": "t217_topk_bridge_product_table",
            "bridge_anchor_bpm": pd.to_numeric(out.get("bridge_anchor_bpm", pd.Series(np.nan, index=out.index)), errors="coerce"),
            "repair_probability": np.nan,
            "eval_gt_hr_bpm": pd.to_numeric(out.get("eval_gt_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce"),
            "eval_abs_error_bpm": pd.to_numeric(out.get("eval_abs_error_bpm", pd.Series(np.nan, index=out.index)), errors="coerce"),
        }
    )
    evidence = []
    for pos, (_, row) in enumerate(product.iterrows()):
        original = _safe_json_loads(out.iloc[pos].get("evidence_json", ""))
        evidence.append(
            _json_dumps(
                {
                    "product_mode": UNIFIED_MODE,
                    "policy_branch": "topk_bridge",
                    "branch_source_task": "T217",
                    "bridge_source": row["source"],
                    "bridge_anchor_bpm": _as_float(row.get("bridge_anchor_bpm")),
                    "candidate_hr_bpm": _as_float(row.get("candidate_hr_bpm")),
                    "claim_boundary": UNIFIED_CLAIM_BOUNDARY,
                    "original_bridge_evidence": original,
                }
            )
        )
    product["evidence_json"] = evidence
    return product


def standardize_source_aware_product_for_unified(source_aware: pd.DataFrame) -> pd.DataFrame:
    """Convert T348 source-aware rows into the unified product contract."""

    if source_aware.empty:
        return pd.DataFrame()
    out = source_aware.copy()
    released = pd.to_numeric(out.get("released", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    out["product_mode"] = UNIFIED_MODE
    out["product_mode_label"] = UNIFIED_MODE_LABEL
    out["policy_branch"] = "source_aware_repair"
    out["branch_source_task"] = "T348_T345"
    out["released"] = released
    out["product_hr_bpm"] = pd.to_numeric(out.get("product_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce").where(released.gt(0), np.nan)
    out["candidate_hr_bpm"] = pd.to_numeric(out.get("candidate_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["repair_probability"] = pd.to_numeric(out.get("repair_probability", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["claim_boundary"] = UNIFIED_CLAIM_BOUNDARY

    evidence = []
    for _, row in out.iterrows():
        original = _safe_json_loads(row.get("evidence_json", ""))
        evidence.append(
            _json_dumps(
                {
                    "product_mode": UNIFIED_MODE,
                    "policy_branch": "source_aware_repair",
                    "branch_source_task": "T348_T345",
                    "source": row.get("source", ""),
                    "source_kind": row.get("source_kind", ""),
                    "window_candidate_count": _as_float(row.get("window_candidate_count")),
                    "repair_probability": _as_float(row.get("repair_probability")),
                    "claim_boundary": UNIFIED_CLAIM_BOUNDARY,
                    "original_source_aware_evidence": original,
                }
            )
        )
    out["evidence_json"] = evidence
    return out


def build_unified_adult_hr_policy_table(
    source_aware_product_table: pd.DataFrame,
    topk_bridge_product_table: pd.DataFrame,
    *,
    bridge_datasets: set[str] | None = None,
) -> pd.DataFrame:
    """Build a branch-unified adult HR product table for replay/product QA.

    T217 bridge rows are used for datasets where the bridge has already been
    validated. T348/T345 source-aware repair rows are used for all other
    datasets. This function is a replay contract, not a final live label-free
    branch router.
    """

    bridge_datasets = bridge_datasets or {"4TU-rPPG-Benchmark", "UBFC-rPPG"}
    source = source_aware_product_table.copy()
    source = source[source.get("product_mode", "") == "source_aware_repair"].copy()
    source = source[~source["dataset"].isin(bridge_datasets)].copy()
    source_part = standardize_source_aware_product_for_unified(source)

    bridge = topk_bridge_product_table.copy()
    bridge = bridge[bridge["dataset"].isin(bridge_datasets)].copy()
    bridge_part = standardize_topk_bridge_product_for_unified(bridge)

    product = pd.concat([bridge_part, source_part], ignore_index=True, sort=False)
    first_cols = [
        "sample_id",
        "dataset",
        "analysis_split",
        "product_mode",
        "product_mode_label",
        "policy_branch",
        "branch_source_task",
        "decision",
        "released",
        "product_hr_bpm",
        "candidate_hr_bpm",
        "source",
        "source_kind",
        "window_candidate_count",
        "repair_probability",
        "bridge_anchor_bpm",
        "claim_boundary",
        "artifact_name",
        "eval_gt_hr_bpm",
        "eval_abs_error_bpm",
        "evidence_json",
    ]
    cols = [col for col in first_cols if col in product.columns] + [col for col in product.columns if col not in first_cols]
    return product[cols].copy()


def summarize_unified_policy(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if product.empty:
        return pd.DataFrame()
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
