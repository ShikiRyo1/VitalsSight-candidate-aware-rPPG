from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


T388_PRODUCT_MODE = "t388_dlcn_dataset_agnostic_candidate_selector"
T388_PRODUCT_MODE_LABEL = "T388 DLCN dataset-agnostic multi-candidate selector"
T388_API_VERSION = "t389.t388_candidate_selector.v1"
T388_THRESHOLD = 0.46
T388_POLICY = "gbr_selector_val_threshold_cov60"
T388_SOURCE_TASK_CHAIN = "T318->T387->T388->T389"
T388_CLAIM_BOUNDARY = (
    "Research product module for frozen DLCN candidate-pool replay. It demonstrates dataset-agnostic candidate "
    "selection over precomputed top-k rPPG candidates, not clinical monitoring, final SOTA, or broad cross-dataset deployment."
)
T388_WARNINGS = [
    "Research MVP only; not a diagnostic or clinical monitoring device.",
    "The module consumes a frozen candidate-probability table from T388 and does not retrain or retune thresholds at runtime.",
    "Release means the selected candidate probability passed the validation-selected threshold; review means HR is withheld.",
    "Ground-truth/evaluation labels are excluded from product-facing API examples.",
]
T388_FORBIDDEN_PRODUCT_FIELDS = ["gt_hr_bpm", "chosen_abs_error_bpm", "abs_error_bpm", "safe_label", "ground_truth", "label"]


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def strict_json_text(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)


def standardize_t388_product_table(predictions: pd.DataFrame, policy: str = T388_POLICY) -> pd.DataFrame:
    rows = predictions[predictions["policy"].astype(str).eq(policy)].copy()
    if rows.empty:
        return rows
    released = pd.to_numeric(rows.get("released", pd.Series(0, index=rows.index)), errors="coerce").fillna(0).astype(int)
    prob = pd.to_numeric(rows.get("t388_safe_probability", pd.Series(np.nan, index=rows.index)), errors="coerce")
    chosen = pd.to_numeric(rows.get("chosen_bpm", rows.get("candidate_bpm", pd.Series(np.nan, index=rows.index))), errors="coerce")
    rows["product_mode"] = T388_PRODUCT_MODE
    rows["product_mode_label"] = T388_PRODUCT_MODE_LABEL
    rows["api_version"] = T388_API_VERSION
    rows["released"] = released
    rows["decision"] = np.where(released.gt(0), "release", "review")
    rows["product_hr_bpm"] = chosen.where(released.gt(0), np.nan)
    rows["product_safe_probability"] = prob
    rows["release_threshold"] = T388_THRESHOLD
    rows["policy_branch"] = np.where(released.gt(0), "t388_candidate_selector_release", "t388_candidate_selector_review")
    rows["review_reason"] = np.where(released.gt(0), "", "t388_safe_probability_below_validation_threshold")
    rows["claim_boundary"] = T388_CLAIM_BOUNDARY
    rows["source_task_chain"] = T388_SOURCE_TASK_CHAIN

    evidence_rows: list[str] = []
    for _, row in rows.iterrows():
        evidence_rows.append(
            strict_json_text(
                {
                    "product_mode": T388_PRODUCT_MODE,
                    "policy": policy,
                    "policy_branch": row.get("policy_branch", ""),
                    "decision": row.get("decision", ""),
                    "released": int(row.get("released", 0)),
                    "product_safe_probability": json_float(row.get("product_safe_probability")),
                    "release_threshold": T388_THRESHOLD,
                    "candidate_window_id": row.get("candidate_window_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "source_task_chain": T388_SOURCE_TASK_CHAIN,
                    "claim_boundary": T388_CLAIM_BOUNDARY,
                }
            )
        )
    rows["evidence_json"] = evidence_rows

    first_cols = [
        "sample_id",
        "candidate_window_id",
        "candidate_id",
        "dataset",
        "subject_id",
        "trial_id",
        "split",
        "window_idx",
        "window_start_sec",
        "window_end_sec",
        "product_mode",
        "product_mode_label",
        "api_version",
        "policy",
        "policy_branch",
        "decision",
        "released",
        "product_hr_bpm",
        "product_safe_probability",
        "release_threshold",
        "review_reason",
        "claim_boundary",
        "source_task_chain",
        "gt_hr_bpm",
        "chosen_abs_error_bpm",
        "evidence_json",
    ]
    return rows[[col for col in first_cols if col in rows.columns] + [col for col in rows.columns if col not in first_cols]].copy()


def summarize_t388_product(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if product.empty:
        return pd.DataFrame()
    group_cols = group_cols or ["split"]
    rows: list[dict[str, Any]] = []
    for keys, group in product.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        released = pd.to_numeric(group.get("released", pd.Series(0, index=group.index)), errors="coerce").fillna(0).astype(int).gt(0)
        err = pd.to_numeric(group.get("chosen_abs_error_bpm", pd.Series(np.nan, index=group.index)), errors="coerce")
        released_err = err[released & err.notna()]
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_windows": int(len(group)),
                "released_windows": int(released.sum()),
                "review_windows": int(len(group) - released.sum()),
                "coverage": float(released.mean()) if len(group) else math.nan,
                "released_mae_bpm": float(released_err.mean()) if len(released_err) else math.nan,
                "unsafe_input_rate": float(((err > 10.0) & released).sum() / len(group)) if len(group) else math.nan,
                "unsafe_released_rate": float((released_err > 10.0).mean()) if len(released_err) else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_endpoint_manifest() -> dict[str, Any]:
    return {
        "api_version": T388_API_VERSION,
        "product_mode": T388_PRODUCT_MODE,
        "product_mode_label": T388_PRODUCT_MODE_LABEL,
        "policy": T388_POLICY,
        "release_threshold": T388_THRESHOLD,
        "claim_boundary": T388_CLAIM_BOUNDARY,
        "endpoints": [
            {"method": "GET", "path": "/api/v1/adult-hr/t388/mode", "purpose": "Return T388 candidate-selector mode metadata and warnings."},
            {"method": "GET", "path": "/api/v1/adult-hr/t388/summary", "purpose": "Return replay summary and claim boundary."},
            {"method": "GET", "path": "/api/v1/adult-hr/t388/windows/{candidate_window_id}", "purpose": "Return one frozen candidate-selector decision without evaluation labels."},
            {"method": "GET", "path": "/api/v1/adult-hr/t388/examples", "purpose": "Return representative release/review examples for demo and QA."},
        ],
        "forbidden_product_response_fields": T388_FORBIDDEN_PRODUCT_FIELDS,
    }


def mode_response() -> dict[str, Any]:
    return {
        "api_version": T388_API_VERSION,
        "product_mode": T388_PRODUCT_MODE,
        "product_mode_label": T388_PRODUCT_MODE_LABEL,
        "policy": T388_POLICY,
        "release_threshold": T388_THRESHOLD,
        "claim_boundary": T388_CLAIM_BOUNDARY,
        "warnings": T388_WARNINGS,
    }


def summary_response(overall: dict[str, Any], split_summary: pd.DataFrame, source_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_version": T388_API_VERSION,
        "product_mode": T388_PRODUCT_MODE,
        "overall": {
            "n_windows": int(finite_float(overall.get("n_windows"), 0)),
            "released_windows": int(finite_float(overall.get("released_windows"), 0)),
            "review_windows": int(finite_float(overall.get("review_windows"), 0)),
            "coverage": json_float(overall.get("coverage")),
            "released_mae_bpm": json_float(overall.get("released_mae_bpm")),
            "unsafe_input_rate": json_float(overall.get("unsafe_input_rate")),
        },
        "split_summary": split_summary.to_dict(orient="records"),
        "supporting_audit": {
            "source_task": "T388",
            "decision": source_summary.get("decision", ""),
            "main_insight": source_summary.get("main_insight", ""),
            "claim_boundary": source_summary.get("claim_boundary", ""),
        },
        "claim_boundary": T388_CLAIM_BOUNDARY,
        "warnings": T388_WARNINGS,
    }


def sample_response(row: dict[str, Any]) -> dict[str, Any]:
    released = int(finite_float(row.get("released"), 0))
    product_hr = json_float(row.get("product_hr_bpm")) if released else None
    probability = json_float(row.get("product_safe_probability"))
    review_reason = "" if pd.isna(row.get("review_reason")) else str(row.get("review_reason", ""))
    user_message = (
        f"Experimental candidate-selected HR estimate released: {product_hr:.2f} BPM."
        if product_hr is not None
        else f"HR withheld for review/retest: {review_reason or 'candidate-selector confidence below threshold'}."
    )
    return {
        "api_version": T388_API_VERSION,
        "candidate_window_id": str(row.get("candidate_window_id", "")),
        "sample_id": str(row.get("sample_id", "")),
        "dataset": str(row.get("dataset", "")),
        "split": str(row.get("split", "")),
        "product_mode": T388_PRODUCT_MODE,
        "decision": "release" if released else "review",
        "released": released,
        "product_hr_bpm": product_hr,
        "product_safe_probability": probability,
        "release_threshold": T388_THRESHOLD,
        "policy_branch": str(row.get("policy_branch", "")),
        "review_reason": review_reason,
        "user_message": user_message,
        "evidence_summary": {
            "source_task_chain": T388_SOURCE_TASK_CHAIN,
            "candidate_id": str(row.get("candidate_id", "")),
            "threshold_rule": "release if product_safe_probability >= 0.46",
        },
        "claim_boundary": T388_CLAIM_BOUNDARY,
        "warnings": T388_WARNINGS,
    }


def representative_examples(product: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    release = product[product["released"].astype(int).eq(1)].sort_values(["split", "product_safe_probability"], ascending=[True, False])
    review = product[product["released"].astype(int).eq(0)].sort_values(["split", "product_safe_probability"], ascending=[True, True])
    if not release.empty:
        examples.append(sample_response(release.iloc[0].to_dict()))
    if not review.empty:
        examples.append(sample_response(review.iloc[0].to_dict()))
    return examples


def examples_response(product: pd.DataFrame) -> dict[str, Any]:
    return {"api_version": T388_API_VERSION, "product_mode": T388_PRODUCT_MODE, "examples": representative_examples(product)}


def forbidden_fields_present(value: Any, forbidden: set[str] | None = None) -> list[str]:
    forbidden = forbidden or set(T388_FORBIDDEN_PRODUCT_FIELDS)
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
