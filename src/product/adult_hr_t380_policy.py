from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


T380_PRODUCT_MODE = "t380_uncertainty_aware_experimental_policy"
T380_PRODUCT_MODE_LABEL = "T380 uncertainty-aware adult HR policy: source-aware repair review + candidate rescue"
T380_CLAIM_BOUNDARY = (
    "Experimental replay/API product mode. It integrates T380/T381 MCD source-aware uncertainty review with prior "
    "top-k bridge, multi-ROI recovery, and T376 upper-tail rescue behavior. It is research evidence, not clinical-grade monitoring or final SOTA."
)
T380_API_VERSION = "t382.t380_uncertainty_adult_hr.v1"

T380_WARNINGS = [
    "Research MVP only; not a diagnostic or clinical monitoring device.",
    "Release means the experimental policy allowed an HR estimate; review means the product must withhold HR.",
    "MCD source-aware repair can be routed to review when uncertainty risk is high.",
    "Ground-truth/evaluation labels are excluded from product-facing API examples.",
]

BRANCH_DISPLAY_NAMES = {
    "topk_bridge": "Top-k bridge release",
    "source_aware_repair": "Source-aware repair release",
    "t376_projection_upper_tail_filtered_rescue": "T376 upper-tail filtered rescue",
    "mcd_source_aware_uncertainty_review": "MCD source-aware uncertainty review",
    "multiroi_consensus_recovery": "Multi-ROI consensus recovery",
    "review_retest": "Review/retest",
}


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_float(value: object) -> float | None:
    out = finite_float(value)
    return out if math.isfinite(out) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def strict_json_text(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)


def normalize_branch(branch: object, released: int) -> str:
    text = "" if pd.isna(branch) else str(branch)
    if not text:
        return "review_retest" if released <= 0 else "source_aware_repair"
    return text


def standardize_t380_product_table(product: pd.DataFrame) -> pd.DataFrame:
    out = product.copy()
    if out.empty:
        return out
    released = pd.to_numeric(out.get("released", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    out["source_product_mode"] = out.get("product_mode", "")
    out["product_mode"] = T380_PRODUCT_MODE
    out["product_mode_label"] = T380_PRODUCT_MODE_LABEL
    out["released"] = released
    out["decision"] = np.where(released.gt(0), "release", "review")
    out["product_hr_bpm"] = pd.to_numeric(out.get("product_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce").where(released.gt(0), np.nan)
    out["policy_branch"] = [normalize_branch(branch, int(rel)) for branch, rel in zip(out.get("policy_branch", pd.Series("", index=out.index)), released)]
    out["branch_display_name"] = out["policy_branch"].map(BRANCH_DISPLAY_NAMES).fillna(out["policy_branch"].astype(str))
    out["claim_boundary"] = T380_CLAIM_BOUNDARY
    out["source_task_chain"] = "T357->T376->T379->T380->T381->T382"

    evidence_rows: list[str] = []
    for _, row in out.iterrows():
        original_evidence: dict[str, Any] = {}
        raw = row.get("evidence_json", "")
        if isinstance(raw, str) and raw.strip():
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    original_evidence = loaded
            except Exception:
                original_evidence = {}
        branch = str(row.get("policy_branch", ""))
        evidence_rows.append(
            strict_json_text(
                {
                    "product_mode": T380_PRODUCT_MODE,
                    "policy_branch": branch,
                    "branch_display_name": row.get("branch_display_name", ""),
                    "decision": row.get("decision", ""),
                    "released": int(row.get("released", 0)),
                    "review_reason": "" if pd.isna(row.get("review_reason")) else str(row.get("review_reason", "")),
                    "source_task_chain": "T357->T376->T379->T380->T381->T382",
                    "uncertainty_policy_logic": {
                        "mcd_uncertainty_review": branch == "mcd_source_aware_uncertainty_review",
                        "t376_upper_tail_rescue": branch == "t376_projection_upper_tail_filtered_rescue",
                        "source_aware_release": branch == "source_aware_repair",
                        "topk_bridge_release": branch == "topk_bridge",
                        "multiroi_consensus_recovery": branch == "multiroi_consensus_recovery",
                    },
                    "claim_boundary": T380_CLAIM_BOUNDARY,
                    "original_replay_evidence_present": bool(original_evidence),
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
        "review_reason",
        "claim_boundary",
        "source_task_chain",
        "eval_gt_hr_bpm",
        "eval_abs_error_bpm",
        "evidence_json",
    ]
    return out[[col for col in first_cols if col in out.columns] + [col for col in out.columns if col not in first_cols]].copy()


def summarize_t380_product(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if product.empty:
        return pd.DataFrame()
    group_cols = group_cols or ["policy_branch"]
    rows: list[dict[str, Any]] = []
    for keys, group in product.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        released = pd.to_numeric(group.get("released", pd.Series(0, index=group.index)), errors="coerce").fillna(0).astype(int).gt(0)
        err = pd.to_numeric(group.get("eval_abs_error_bpm", pd.Series(np.nan, index=group.index)), errors="coerce")
        released_err = err[released & err.notna()]
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_rows": int(len(group)),
                "released_rows": int(released.sum()),
                "review_rows": int(len(group) - released.sum()),
                "coverage": float(released.mean()) if len(group) else math.nan,
                "released_mae_bpm": float(released_err.mean()) if len(released_err) else math.nan,
                "published_unsafe_per_input": float(((err > 10.0) & released).sum() / len(group)) if len(group) else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_endpoint_manifest() -> dict[str, Any]:
    return {
        "api_version": T380_API_VERSION,
        "product_mode": T380_PRODUCT_MODE,
        "product_mode_label": T380_PRODUCT_MODE_LABEL,
        "claim_boundary": T380_CLAIM_BOUNDARY,
        "endpoints": [
            {"method": "GET", "path": "/api/v1/adult-hr/t380/mode", "purpose": "Return experimental T380 mode metadata and warnings."},
            {"method": "GET", "path": "/api/v1/adult-hr/t380/summary", "purpose": "Return replay summary and supporting audit boundary."},
            {"method": "GET", "path": "/api/v1/adult-hr/t380/samples/{sample_id}", "purpose": "Return one product decision without evaluation labels."},
            {"method": "GET", "path": "/api/v1/adult-hr/t380/examples", "purpose": "Return representative branch examples for QA/demo."},
        ],
        "forbidden_product_response_fields": ["eval_gt_hr_bpm", "eval_abs_error_bpm", "gt_hr_bpm", "ground_truth", "label"],
    }


def mode_response() -> dict[str, Any]:
    return {
        "api_version": T380_API_VERSION,
        "product_mode": T380_PRODUCT_MODE,
        "product_mode_label": T380_PRODUCT_MODE_LABEL,
        "claim_boundary": T380_CLAIM_BOUNDARY,
        "warnings": T380_WARNINGS,
    }


def branch_records(branch_summary: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in branch_summary.iterrows():
        records.append(
            {
                "policy_branch": str(row.get("policy_branch", "")),
                "n_rows": int(finite_float(row.get("n_rows"), 0.0)),
                "released_rows": int(finite_float(row.get("released_rows"), 0.0)),
                "review_rows": int(finite_float(row.get("review_rows"), 0.0)),
                "coverage": json_float(row.get("coverage")),
                "released_mae_bpm": json_float(row.get("released_mae_bpm")),
                "published_unsafe_per_input": json_float(row.get("published_unsafe_per_input")),
            }
        )
    return records


def summary_response(overall: dict[str, Any], branch_summary: pd.DataFrame, audit_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_version": T380_API_VERSION,
        "product_mode": T380_PRODUCT_MODE,
        "overall": {
            "n_rows": int(finite_float(overall.get("n_rows"), 0.0)),
            "coverage": json_float(overall.get("coverage")),
            "released_mae_bpm": json_float(overall.get("released_mae_bpm")),
            "published_unsafe_per_input": json_float(overall.get("published_unsafe_per_input")),
        },
        "branches": branch_records(branch_summary),
        "supporting_audit": {
            "source_task": "T381",
            "decision": audit_summary.get("decision", ""),
            "main_insight": audit_summary.get("main_insight", ""),
            "claim_boundary": audit_summary.get("claim_boundary", ""),
        },
        "claim_boundary": T380_CLAIM_BOUNDARY,
        "warnings": T380_WARNINGS,
    }


def sample_response(row: dict[str, Any]) -> dict[str, Any]:
    released = int(finite_float(row.get("released"), 0.0))
    product_hr = json_float(row.get("product_hr_bpm")) if released else None
    review_reason = "" if pd.isna(row.get("review_reason")) else str(row.get("review_reason", ""))
    evidence: dict[str, Any] = {}
    raw = row.get("evidence_json", "")
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                evidence = loaded
        except Exception:
            evidence = {}
    user_message = (
        f"Experimental HR estimate released: {product_hr:.2f} BPM."
        if product_hr is not None
        else f"HR withheld for review/retest: {review_reason or 'policy did not release this window'}."
    )
    return {
        "api_version": T380_API_VERSION,
        "sample_id": str(row.get("sample_id", "")),
        "product_mode": T380_PRODUCT_MODE,
        "decision": "release" if released else "review",
        "released": released,
        "product_hr_bpm": product_hr,
        "policy_branch": str(row.get("policy_branch", "")),
        "branch_display_name": str(row.get("branch_display_name", "")),
        "review_reason": review_reason,
        "user_message": user_message,
        "evidence_summary": {
            "policy_branch": evidence.get("policy_branch", row.get("policy_branch", "")),
            "source_task_chain": evidence.get("source_task_chain", "T357->T376->T379->T380->T381->T382"),
            "uncertainty_policy_logic": evidence.get("uncertainty_policy_logic", {}),
        },
        "claim_boundary": T380_CLAIM_BOUNDARY,
        "warnings": T380_WARNINGS,
    }


def representative_examples(product: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    branch_order = [
        "topk_bridge",
        "source_aware_repair",
        "t376_projection_upper_tail_filtered_rescue",
        "mcd_source_aware_uncertainty_review",
        "multiroi_consensus_recovery",
        "review_retest",
    ]
    for branch in branch_order:
        rows = product[product["policy_branch"].astype(str).eq(branch)].copy()
        if rows.empty:
            continue
        examples.append(sample_response(rows.sort_values(["dataset", "sample_id"]).iloc[0].to_dict()))
    return examples


def examples_response(product: pd.DataFrame) -> dict[str, Any]:
    return {"api_version": T380_API_VERSION, "product_mode": T380_PRODUCT_MODE, "examples": representative_examples(product)}


def forbidden_fields_present(value: Any, forbidden: set[str] | None = None) -> list[str]:
    forbidden = forbidden or set(build_endpoint_manifest()["forbidden_product_response_fields"])
    found: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in forbidden:
                    found.add(key)
                walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(value)
    return sorted(found)
