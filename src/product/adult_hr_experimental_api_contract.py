from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from src.product.adult_hr_experimental_policy import EXPERIMENTAL_CLAIM_BOUNDARY, EXPERIMENTAL_MODE, EXPERIMENTAL_MODE_LABEL


API_VERSION = "t359.experimental_adult_hr.v1"

RESEARCH_WARNINGS = [
    "Research MVP only; not a diagnostic or clinical monitoring device.",
    "Automatic release is allowed only when the frozen policy releases the row.",
    "Review/retest rows must not display a final HR number.",
    "Ground-truth/evaluation labels are excluded from product responses.",
]


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_float(value: object) -> float | None:
    out = finite_float(value)
    return out if math.isfinite(out) else None


def safe_json_load(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def strict_json_text(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)


def build_endpoint_manifest() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "product_mode": EXPERIMENTAL_MODE,
        "product_mode_label": EXPERIMENTAL_MODE_LABEL,
        "claim_boundary": EXPERIMENTAL_CLAIM_BOUNDARY,
        "endpoints": [
            {
                "method": "GET",
                "path": "/api/v1/adult-hr/experimental/mode",
                "purpose": "Return product-mode metadata, warnings, and claim boundary.",
                "response_schema": ["api_version", "product_mode", "product_mode_label", "claim_boundary", "warnings"],
            },
            {
                "method": "GET",
                "path": "/api/v1/adult-hr/experimental/summary",
                "purpose": "Return frozen replay metrics and branch-level product counts.",
                "response_schema": ["api_version", "overall", "branches", "dashboard_qa"],
            },
            {
                "method": "GET",
                "path": "/api/v1/adult-hr/experimental/samples/{sample_id}",
                "purpose": "Return one product decision without evaluation labels.",
                "response_schema": [
                    "api_version",
                    "sample_id",
                    "decision",
                    "released",
                    "product_hr_bpm",
                    "policy_branch",
                    "evidence_summary",
                    "claim_boundary",
                    "warnings",
                ],
            },
            {
                "method": "GET",
                "path": "/api/v1/adult-hr/experimental/examples",
                "purpose": "Return representative release/review examples for smoke tests and demos.",
                "response_schema": ["api_version", "examples"],
            },
        ],
        "forbidden_product_response_fields": [
            "eval_gt_hr_bpm",
            "eval_abs_error_bpm",
            "gt_hr_bpm",
            "ground_truth",
            "label",
        ],
    }


def product_mode_response() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "product_mode": EXPERIMENTAL_MODE,
        "product_mode_label": EXPERIMENTAL_MODE_LABEL,
        "claim_boundary": EXPERIMENTAL_CLAIM_BOUNDARY,
        "warnings": RESEARCH_WARNINGS,
    }


def branch_records(branch_summary: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in branch_summary.iterrows():
        records.append(
            {
                "policy_branch": str(row.get("policy_branch", "")),
                "n_rows": int(finite_float(row.get("n_rows"), 0.0)),
                "coverage": json_float(row.get("coverage")),
                "released_mae_bpm": json_float(row.get("released_mae_bpm")),
                "published_unsafe_per_input": json_float(row.get("published_unsafe_per_input")),
            }
        )
    return records


def summary_response(summary: dict[str, Any], branch_summary: pd.DataFrame, dashboard_qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "product_mode": EXPERIMENTAL_MODE,
        "overall": {
            "coverage": json_float(summary.get("overall", {}).get("coverage")),
            "released_mae_bpm": json_float(summary.get("overall", {}).get("released_mae_bpm")),
            "published_unsafe_per_input": json_float(summary.get("overall", {}).get("published_unsafe_per_input")),
            "n_rows": int(finite_float(summary.get("n_rows"), 0.0)),
        },
        "branches": branch_records(branch_summary),
        "dashboard_qa": {
            "task_id": dashboard_qa.get("task_id", "T358"),
            "all_qa_passed": bool(dashboard_qa.get("all_qa_passed", False)),
            "n_checks": int(finite_float(dashboard_qa.get("n_checks"), 0.0)),
            "screenshot": dashboard_qa.get("screenshot", ""),
        },
        "claim_boundary": EXPERIMENTAL_CLAIM_BOUNDARY,
        "warnings": RESEARCH_WARNINGS,
    }


def _nested(path: list[str], data: dict[str, Any]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def evidence_summary(row: dict[str, Any]) -> dict[str, Any]:
    evidence = safe_json_load(row.get("evidence_json", ""))
    original_t356 = evidence.get("original_t356_evidence", {}) if isinstance(evidence.get("original_t356_evidence"), dict) else {}
    original_t354 = original_t356.get("original_t354_evidence", {}) if isinstance(original_t356.get("original_t354_evidence"), dict) else {}
    routing = original_t354.get("routing_features", {}) if isinstance(original_t354.get("routing_features"), dict) else {}
    consensus = original_t356.get("multiroi_consensus_features", {}) if isinstance(original_t356.get("multiroi_consensus_features"), dict) else {}
    unified = original_t354.get("original_unified_evidence", {}) if isinstance(original_t354.get("original_unified_evidence"), dict) else {}
    bridge_evidence = unified.get("original_bridge_evidence", {}) if isinstance(unified.get("original_bridge_evidence"), dict) else {}
    candidate_chain = bridge_evidence.get("candidate_chain", []) if isinstance(bridge_evidence.get("candidate_chain"), list) else []

    return {
        "policy_branch": row.get("policy_branch", ""),
        "branch_display_name": row.get("branch_display_name", ""),
        "source_kind": row.get("source_kind", ""),
        "source": row.get("source", ""),
        "review_reason": "" if pd.isna(row.get("review_reason")) else str(row.get("review_reason", "")),
        "routing": {
            "has_bridge_evidence": routing.get("has_bridge_evidence"),
            "bridge_confident": routing.get("bridge_confident"),
            "repair_confident": routing.get("repair_confident"),
            "source_aware_source": routing.get("source_aware_source"),
            "window_candidate_count": json_float(routing.get("window_candidate_count")),
            "repair_probability": json_float(routing.get("repair_probability")),
        },
        "multiroi_consensus": {
            "group_size": json_float(consensus.get("consensus_group_size")),
            "group_range_bpm": json_float(consensus.get("consensus_group_range_bpm")),
            "abs_to_group_median_bpm": json_float(consensus.get("abs_to_consensus_median_bpm")),
            "confident": consensus.get("multiroi_consensus_confident"),
            "recovered_by_t355": original_t356.get("recovered_by_t355"),
        },
        "top_candidate_chain_preview": [
            {
                "candidate_id": item.get("candidate_id"),
                "candidate_bpm": json_float(item.get("candidate_bpm")),
                "support_count": item.get("support_count"),
                "support_rois": item.get("support_rois"),
                "support_methods": item.get("support_methods"),
            }
            for item in candidate_chain[:5]
            if isinstance(item, dict)
        ],
    }


def sample_response(row: dict[str, Any]) -> dict[str, Any]:
    released = int(finite_float(row.get("released"), 0.0))
    decision = "release" if released else "review"
    product_hr = json_float(row.get("product_hr_bpm")) if released else None
    review_reason = "" if pd.isna(row.get("review_reason")) else str(row.get("review_reason", ""))
    user_message = (
        f"Experimental HR estimate released: {product_hr:.2f} BPM."
        if product_hr is not None
        else f"HR withheld for review/retest: {review_reason or 'policy did not release this window'}."
    )
    return {
        "api_version": API_VERSION,
        "sample_id": str(row.get("sample_id", "")),
        "product_mode": EXPERIMENTAL_MODE,
        "decision": decision,
        "released": released,
        "product_hr_bpm": product_hr,
        "policy_branch": str(row.get("policy_branch", "")),
        "branch_display_name": str(row.get("branch_display_name", "")),
        "review_reason": review_reason,
        "user_message": user_message,
        "evidence_summary": evidence_summary(row),
        "claim_boundary": EXPERIMENTAL_CLAIM_BOUNDARY,
        "warnings": RESEARCH_WARNINGS,
    }


def representative_sample_rows(product: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in ["topk_bridge", "source_aware_repair", "multiroi_consensus_recovery", "review_retest"]:
        subset = product[product["policy_branch"].astype(str).eq(branch)].copy()
        if subset.empty:
            continue
        rows.append(subset.sort_values(["dataset", "sample_id"]).iloc[0].to_dict())
    return rows


def examples_response(product: pd.DataFrame) -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "product_mode": EXPERIMENTAL_MODE,
        "examples": [sample_response(row) for row in representative_sample_rows(product)],
    }


def forbidden_fields_present(value: Any, forbidden: set[str] | None = None) -> list[str]:
    forbidden = forbidden or set(build_endpoint_manifest()["forbidden_product_response_fields"])
    found: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, val in item.items():
                if str(key) in forbidden:
                    found.add(str(key))
                walk(val)
        elif isinstance(item, list):
            for val in item:
                walk(val)

    walk(value)
    return sorted(found)
