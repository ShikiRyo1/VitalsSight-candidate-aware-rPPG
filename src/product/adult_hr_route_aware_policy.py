from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


UNSAFE_BPM_ERROR = 10.0


@dataclass(frozen=True)
class RouteAwarePolicyConfig:
    """Configuration for the research-MVP adult HR route-aware policy."""

    default_branch: str = "t498_context_aware_default"
    rescue_branch: str = "t504_mediapipe_t3_dual_range_rescue"
    review_branch: str = "review_or_deep_roi_escalation"
    unsafe_bpm_error: float = UNSAFE_BPM_ERROR


def _as_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _decision_row(
    *,
    condition_id: str,
    dataset: str,
    branch: str,
    source_task: str,
    source_reason: str,
    reference_bpm: float,
    released_bpm: float,
    decision: str,
    review_reason: str,
    config: RouteAwarePolicyConfig,
) -> dict[str, object]:
    err = abs(released_bpm - reference_bpm) if math.isfinite(released_bpm) and math.isfinite(reference_bpm) else math.nan
    released = int(decision == "release" and math.isfinite(released_bpm))
    return {
        "condition_id": condition_id,
        "dataset": dataset,
        "policy_branch": branch,
        "source_task": source_task,
        "source_reason": source_reason,
        "decision": decision,
        "released": released,
        "product_hr_bpm": released_bpm if released else math.nan,
        "reference_bpm_for_eval_only": reference_bpm,
        "absolute_error_bpm_for_eval_only": err,
        "unsafe_release_gt10_for_eval_only": bool(released and math.isfinite(err) and err > config.unsafe_bpm_error),
        "review_reason": "" if released else review_reason,
        "claim_boundary": "Research MVP output; not a clinical monitoring or diagnostic decision.",
    }


def build_route_aware_adult_hr_policy_table(
    t498_decisions: pd.DataFrame,
    t504_decisions: pd.DataFrame,
    *,
    config: RouteAwarePolicyConfig | None = None,
) -> pd.DataFrame:
    """Build a route-aware release/review product table.

    Policy:
    - T498 remains the global default.
    - T504 is used only as a rescue when T498 refuses and T504 safely releases
      a MediaPipe/T3 selected-route condition.
    - MR/low-light rows in T504 remain review/deep-ROI escalation, never forced
      release.
    """

    cfg = config or RouteAwarePolicyConfig()
    t498 = t498_decisions.copy()
    t504 = t504_decisions.copy()
    rows: list[dict[str, object]] = []
    t504_by_condition = {str(row["condition_id"]): row for _, row in t504.iterrows()}

    for _, row in t498.iterrows():
        condition = str(row["condition_id"])
        reference = _as_float(row.get("reference_bpm"))
        dataset = str(row.get("dataset", "UBFC-Phys-S1-S14"))
        default_policy = str(row.get("policy", "refuse"))
        default_bpm = _as_float(row.get("released_bpm"))
        default_reason = str(row.get("reason", ""))

        if default_policy == "release" and math.isfinite(default_bpm):
            rows.append(
                _decision_row(
                    condition_id=condition,
                    dataset=dataset,
                    branch=cfg.default_branch,
                    source_task="T498",
                    source_reason=default_reason,
                    reference_bpm=reference,
                    released_bpm=default_bpm,
                    decision="release",
                    review_reason="",
                    config=cfg,
                )
            )
            continue

        rescue = t504_by_condition.get(condition)
        rescue_policy = str(rescue.get("policy", "")) if rescue is not None else ""
        rescue_bpm = _as_float(rescue.get("released_bpm")) if rescue is not None else math.nan
        rescue_error = _as_float(rescue.get("absolute_error_bpm")) if rescue is not None else math.nan
        rescue_safe = bool(rescue_policy == "release" and math.isfinite(rescue_bpm) and (not math.isfinite(rescue_error) or rescue_error <= cfg.unsafe_bpm_error))
        if rescue_safe:
            rows.append(
                _decision_row(
                    condition_id=condition,
                    dataset=dataset,
                    branch=cfg.rescue_branch,
                    source_task="T504",
                    source_reason=str(rescue.get("reason", "")),
                    reference_bpm=reference,
                    released_bpm=rescue_bpm,
                    decision="release",
                    review_reason="",
                    config=cfg,
                )
            )
        else:
            rows.append(
                _decision_row(
                    condition_id=condition,
                    dataset=dataset,
                    branch=cfg.review_branch,
                    source_task="T498",
                    source_reason=default_reason,
                    reference_bpm=reference,
                    released_bpm=math.nan,
                    decision="review",
                    review_reason=f"default_refuse:{default_reason}",
                    config=cfg,
                )
            )

    known_conditions = {str(row["condition_id"]) for _, row in t498.iterrows()}
    for _, row in t504.iterrows():
        condition = str(row["condition_id"])
        if condition in known_conditions:
            continue
        reference = _as_float(row.get("reference_bpm"))
        dataset = str(row.get("dataset", ""))
        policy = str(row.get("policy", "refuse"))
        bpm = _as_float(row.get("released_bpm"))
        if policy == "release" and math.isfinite(bpm):
            rows.append(
                _decision_row(
                    condition_id=condition,
                    dataset=dataset,
                    branch=cfg.rescue_branch,
                    source_task="T504",
                    source_reason=str(row.get("reason", "")),
                    reference_bpm=reference,
                    released_bpm=bpm,
                    decision="release",
                    review_reason="",
                    config=cfg,
                )
            )
        else:
            rows.append(
                _decision_row(
                    condition_id=condition,
                    dataset=dataset,
                    branch=cfg.review_branch,
                    source_task="T504",
                    source_reason=str(row.get("reason", "")),
                    reference_bpm=reference,
                    released_bpm=math.nan,
                    decision="review",
                    review_reason="low_light_or_route_not_validated:deep_roi_or_retest_needed",
                    config=cfg,
                )
            )

    return pd.DataFrame(rows)


def summarize_route_aware_policy(product: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["dataset", "policy_branch"]
    rows = []
    for keys, group in product.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        item = {col: value for col, value in zip(group_cols, key_values)}
        released = pd.to_numeric(group["released"], errors="coerce").fillna(0).astype(int).gt(0)
        err = pd.to_numeric(group["absolute_error_bpm_for_eval_only"], errors="coerce")
        rel_err = err[released & np.isfinite(err)]
        item.update(
            {
                "n": int(len(group)),
                "released": int(released.sum()),
                "reviewed": int(len(group) - released.sum()),
                "coverage": float(released.mean()) if len(group) else math.nan,
                "released_mae_bpm": float(rel_err.mean()) if len(rel_err) else math.nan,
                "released_median_ae_bpm": float(rel_err.median()) if len(rel_err) else math.nan,
                "released_unsafe_gt10_rate": float((rel_err > UNSAFE_BPM_ERROR).mean()) if len(rel_err) else 0.0,
                "unsafe_per_input": float(((err > UNSAFE_BPM_ERROR) & released).sum() / len(group)) if len(group) else math.nan,
            }
        )
        rows.append(item)
    if group_cols != ["ALL"]:
        released = pd.to_numeric(product["released"], errors="coerce").fillna(0).astype(int).gt(0)
        err = pd.to_numeric(product["absolute_error_bpm_for_eval_only"], errors="coerce")
        rel_err = err[released & np.isfinite(err)]
        rows.append(
            {
                **{col: "ALL" for col in group_cols},
                "n": int(len(product)),
                "released": int(released.sum()),
                "reviewed": int(len(product) - released.sum()),
                "coverage": float(released.mean()) if len(product) else math.nan,
                "released_mae_bpm": float(rel_err.mean()) if len(rel_err) else math.nan,
                "released_median_ae_bpm": float(rel_err.median()) if len(rel_err) else math.nan,
                "released_unsafe_gt10_rate": float((rel_err > UNSAFE_BPM_ERROR).mean()) if len(rel_err) else 0.0,
                "unsafe_per_input": float(((err > UNSAFE_BPM_ERROR) & released).sum() / len(product)) if len(product) else math.nan,
            }
        )
    return pd.DataFrame(rows)
