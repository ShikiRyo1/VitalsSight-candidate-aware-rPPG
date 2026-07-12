from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


TRUSTED_BRIDGE_SOURCES = {
    "anchor_nearest_topk_consistent",
    "high_hr_anchor_preserved",
    "low_alias_upper_rescue",
}

ROUTER_MODE = "label_free_branch_router"
ROUTER_LABEL = "Label-free branch router: bridge evidence + source-aware repair confidence"
ROUTER_BOUNDARY = (
    "Label-free replay gate based on candidate-regime evidence. Safety-first; not a final coverage-optimized live router."
)


def _safe_load_json(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


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


def extract_label_free_routing_features(product: pd.DataFrame) -> pd.DataFrame:
    """Extract routing features that do not use dataset labels or ground truth."""

    rows: list[dict[str, Any]] = []
    for _, row in product.iterrows():
        evidence = _safe_load_json(row.get("evidence_json", ""))
        bridge_evidence = evidence.get("original_bridge_evidence", {})
        if not isinstance(bridge_evidence, dict):
            bridge_evidence = {}
        chain = bridge_evidence.get("candidate_chain", [])
        if not isinstance(chain, list):
            chain = []
        max_support_count = 0.0
        max_support_methods = 0.0
        max_support_rois = 0.0
        for item in chain:
            if not isinstance(item, dict):
                continue
            max_support_count = max(max_support_count, _finite_float(item.get("support_count"), 0.0))
            max_support_methods = max(max_support_methods, _finite_float(item.get("support_methods"), 0.0))
            max_support_rois = max(max_support_rois, _finite_float(item.get("support_rois"), 0.0))

        source = str(row.get("source", ""))
        source_kind = str(row.get("source_kind", ""))
        bridge_source = str(bridge_evidence.get("bridge_source") or source)
        has_bridge_evidence = bool(bridge_evidence) or source in TRUSTED_BRIDGE_SOURCES or source_kind == "topk_bridge"
        bridge_anchor_released = int(_finite_float(bridge_evidence.get("anchor_released"), 1.0 if has_bridge_evidence else 0.0))
        # Low-alias rescue intentionally rejects the low-frequency anchor and
        # releases a higher candidate supported by the candidate chain.
        anchor_ok = bridge_anchor_released == 1 or bridge_source == "low_alias_upper_rescue"
        bridge_confident = (
            has_bridge_evidence
            and bridge_source in TRUSTED_BRIDGE_SOURCES
            and anchor_ok
        )

        repair_probability = _finite_float(row.get("repair_probability"))
        window_candidate_count = _finite_float(row.get("window_candidate_count"))
        source_aware_source = "source_aware" in source or source_kind == "source_aware_candidate_repair"
        repair_confident = (
            source_aware_source
            and math.isfinite(window_candidate_count)
            and window_candidate_count >= 18.0
            and (not math.isfinite(repair_probability) or repair_probability >= 0.78)
        )

        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "has_bridge_evidence": has_bridge_evidence,
                "bridge_source": bridge_source,
                "bridge_anchor_released": bridge_anchor_released,
                "bridge_max_support_count": max_support_count,
                "bridge_max_support_methods": max_support_methods,
                "bridge_max_support_rois": max_support_rois,
                "bridge_confident": bridge_confident,
                "window_candidate_count": window_candidate_count,
                "repair_probability": repair_probability,
                "source_kind": source_kind,
                "source_aware_source": source_aware_source,
                "repair_confident": repair_confident,
            }
        )
    return pd.DataFrame(rows)


def apply_label_free_branch_router(product: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a safety-first label-free release/review router."""

    routed = product.copy()
    features = extract_label_free_routing_features(routed)
    if routed.empty:
        return routed, features

    route_branch = np.where(
        features["bridge_confident"].to_numpy(),
        "topk_bridge",
        np.where(features["repair_confident"].to_numpy(), "source_aware_repair", "review_retest"),
    )
    release_mask = route_branch != "review_retest"

    routed["pre_router_policy_branch"] = routed.get("policy_branch", "")
    routed["policy_branch"] = route_branch
    routed["product_mode"] = ROUTER_MODE
    routed["product_mode_label"] = ROUTER_LABEL
    routed["claim_boundary"] = ROUTER_BOUNDARY
    routed["released"] = release_mask.astype(int)
    routed["decision"] = np.where(release_mask, "release", "review")
    candidate_hr = pd.to_numeric(routed.get("candidate_hr_bpm", pd.Series(np.nan, index=routed.index)), errors="coerce")
    product_hr = pd.to_numeric(routed.get("product_hr_bpm", pd.Series(np.nan, index=routed.index)), errors="coerce")
    routed["candidate_hr_bpm"] = candidate_hr.where(candidate_hr.notna(), product_hr)
    routed["product_hr_bpm"] = pd.to_numeric(routed["product_hr_bpm"], errors="coerce").where(release_mask, np.nan)
    routed["review_reason"] = np.where(
        release_mask,
        "",
        "review_no_label_free_bridge_or_repair_confidence",
    )

    evidence_rows: list[str] = []
    for idx, row in routed.iterrows():
        feat = features.loc[idx].to_dict()
        original = _safe_load_json(row.get("evidence_json", ""))
        evidence_rows.append(
            _json_dumps(
                {
                    "product_mode": ROUTER_MODE,
                    "policy_branch": row.get("policy_branch", ""),
                    "decision": row.get("decision", ""),
                    "routing_features": {
                        key: feat.get(key)
                        for key in [
                            "has_bridge_evidence",
                            "bridge_source",
                            "bridge_anchor_released",
                            "bridge_max_support_count",
                            "bridge_max_support_methods",
                            "bridge_max_support_rois",
                            "bridge_confident",
                            "window_candidate_count",
                            "repair_probability",
                            "source_kind",
                            "source_aware_source",
                            "repair_confident",
                        ]
                    },
                    "review_reason": row.get("review_reason", ""),
                    "claim_boundary": ROUTER_BOUNDARY,
                    "original_unified_evidence": original,
                }
            )
        )
    routed["evidence_json"] = evidence_rows
    return routed, features


def summarize_router(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
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
