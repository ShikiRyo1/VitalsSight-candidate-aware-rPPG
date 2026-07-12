from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.product.adult_hr_source_aware_repair import (
    build_t345_repair_router_product,
    load_t345_repair_artifact,
)


PROJECT = Path(__file__).resolve().parents[2]

MODE_LABELS = {
    "balanced": "T339 balanced candidate-count router",
    "high_caution": "T341 high-caution review gate",
    "source_aware_repair": "T345/T347 source-aware repair router",
}

MODE_CLAIM_BOUNDARIES = {
    "balanced": "balanced release/review baseline; safer than global selector but still unsafe on MCD-like windows",
    "high_caution": "review-heavy safety mode; lowers unsafe release but sacrifices MCD coverage",
    "source_aware_repair": "candidate-evidence repair branch; supported on locked replay, not a clinical device",
}


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _source_kind(source: object) -> str:
    text = str(source)
    if "t345_source_aware_candidate_repair" in text:
        return "source_aware_candidate_repair"
    if "review" in text:
        return "review_gate"
    if "t338" in text:
        return "domain_selector"
    if "t330" in text:
        return "base_selector"
    return text or "unknown"


def standardize_adult_hr_router_product(
    frame: pd.DataFrame,
    *,
    mode: str,
    mode_label: str | None = None,
    artifact_name: str = "",
) -> pd.DataFrame:
    """Convert router/replay predictions into one product/API contract.

    Ground-truth columns are kept only with `eval_` prefixes. The deployable
    fields are `product_hr_bpm`, `decision`, `released`, and `evidence_json`.
    """

    label = mode_label or MODE_LABELS.get(mode, mode)
    out = frame.copy()
    if out.empty:
        return out

    released = pd.to_numeric(out.get("released", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    selected = pd.to_numeric(out.get("selected_hr_bpm", pd.Series(np.nan, index=out.index)), errors="coerce")
    decision = out.get("decision", pd.Series("", index=out.index)).astype(str)
    decision = decision.where(decision.ne(""), np.where(released.gt(0), "release", "review"))

    product = pd.DataFrame(
        {
            "sample_id": out.get("sample_id", pd.Series("", index=out.index)).astype(str),
            "dataset": out.get("dataset", pd.Series("", index=out.index)).astype(str),
            "analysis_split": out.get("analysis_split", pd.Series("", index=out.index)).astype(str),
            "product_mode": mode,
            "product_mode_label": label,
            "decision": decision,
            "released": released,
            "product_hr_bpm": selected.where(released.gt(0), np.nan),
            "candidate_hr_bpm": selected,
            "source": out.get("source", pd.Series("", index=out.index)).astype(str),
            "source_kind": out.get("source", pd.Series("", index=out.index)).map(_source_kind),
            "window_candidate_count": pd.to_numeric(out.get("window_candidate_count", pd.Series(np.nan, index=out.index)), errors="coerce"),
            "claim_boundary": MODE_CLAIM_BOUNDARIES.get(mode, ""),
            "artifact_name": artifact_name,
        }
    )
    if "repair_probability" in out.columns:
        product["repair_probability"] = pd.to_numeric(out["repair_probability"], errors="coerce")
    elif "t345_repair_probability" in out.columns:
        product["repair_probability"] = pd.to_numeric(out["t345_repair_probability"], errors="coerce")
    else:
        product["repair_probability"] = np.nan
    for source_col, target_col in [
        ("gt_hr_bpm", "eval_gt_hr_bpm"),
        ("abs_error_bpm", "eval_abs_error_bpm"),
        ("low_review_max", "low_review_max"),
        ("high_t338_min", "high_t338_min"),
        ("t338_probability", "t338_probability"),
        ("t338_support_methods", "t338_support_methods"),
        ("t330_t338_gap_bpm", "t330_t338_gap_bpm"),
    ]:
        if source_col in out.columns:
            product[target_col] = out[source_col]

    evidence_rows: list[str] = []
    for _, row in product.iterrows():
        evidence_rows.append(
            json.dumps(
                {
                    "product_mode": row["product_mode"],
                    "mode_label": row["product_mode_label"],
                    "source": row["source"],
                    "source_kind": row["source_kind"],
                    "window_candidate_count": finite_float(row.get("window_candidate_count")),
                    "repair_probability": finite_float(row.get("repair_probability")),
                    "claim_boundary": row["claim_boundary"],
                    "artifact_name": artifact_name,
                },
                ensure_ascii=False,
            )
        )
    product["evidence_json"] = evidence_rows
    return product


def summarize_product_modes(product: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if product.empty:
        return pd.DataFrame()
    for (split, dataset, mode), group in product.groupby(["analysis_split", "dataset", "product_mode"], sort=True, dropna=False):
        released = pd.to_numeric(group["released"], errors="coerce").fillna(0).astype(int).gt(0)
        err = pd.to_numeric(group.get("eval_abs_error_bpm", pd.Series(np.nan, index=group.index)), errors="coerce")
        released_err = err[released & np.isfinite(err)]
        rows.append(
            {
                "analysis_split": split,
                "dataset": dataset,
                "product_mode": mode,
                "product_mode_label": MODE_LABELS.get(str(mode), str(mode)),
                "n_windows": int(len(group)),
                "coverage": float(released.mean()) if len(group) else math.nan,
                "review_rate": float((~released).mean()) if len(group) else math.nan,
                "released_mae_bpm": float(released_err.mean()) if len(released_err) else math.nan,
                "released_unsafe_gt10bpm_rate": float((released_err > 10.0).mean()) if len(released_err) else math.nan,
                "published_unsafe_per_input": float(((err > 10.0) & released).sum() / len(group)) if len(group) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_source_aware_repair_product_contract(
    candidates: pd.DataFrame,
    balanced_router_predictions: pd.DataFrame,
    *,
    artifact: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_artifact = artifact or load_t345_repair_artifact()
    scored, selected, routed = build_t345_repair_router_product(
        candidates,
        balanced_router_predictions,
        artifact=model_artifact,
    )
    product = standardize_adult_hr_router_product(
        routed,
        mode="source_aware_repair",
        artifact_name=str(model_artifact.get("artifact_name", "t345_source_aware_candidate_repair")),
    )
    return scored, selected, product


def build_mode_bundle(
    *,
    balanced_predictions: pd.DataFrame,
    high_caution_predictions: pd.DataFrame,
    source_aware_product: pd.DataFrame,
    modes: Iterable[str] = ("balanced", "high_caution", "source_aware_repair"),
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    requested = set(modes)
    if "balanced" in requested:
        parts.append(standardize_adult_hr_router_product(balanced_predictions, mode="balanced"))
    if "high_caution" in requested:
        parts.append(standardize_adult_hr_router_product(high_caution_predictions, mode="high_caution"))
    if "source_aware_repair" in requested:
        parts.append(source_aware_product.copy())
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
