from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass(frozen=True)
class DisagreementReviewConfig:
    """Product release rule for base/auxiliary HR disagreement.

    The default values are the validation-selected T328 rule:
    review when the high-HR-aware auxiliary estimate is >=100 BPM, the stable
    base estimate is <=100 BPM, and the upward disagreement is at least 10 BPM.
    """

    policy_name: str = "t328_upward_disagreement_review"
    aux_high_min_bpm: float = 100.0
    base_max_bpm: float = 100.0
    upward_gap_min_bpm: float = 10.0
    abs_gap_min_bpm: float = 10.0
    aux_probability_min: float = 0.0
    mode: str = "upward_conflict"


def parse_evidence(raw: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def apply_disagreement_review_gate(
    table: pd.DataFrame,
    *,
    config: DisagreementReviewConfig | None = None,
    base_bpm_col: str = "base_hr_bpm",
    aux_bpm_col: str = "aux_hr_bpm",
    aux_probability_col: str = "aux_probability",
    output_bpm_col: str = "product_hr_bpm",
) -> pd.DataFrame:
    """Apply the T328-style release/review gate to a product table.

    The function does not compute the base or auxiliary estimate. It only
    converts their disagreement into a product-safe release decision.
    """

    if table.empty:
        return table.copy()
    cfg = config or DisagreementReviewConfig()
    required = {base_bpm_col, aux_bpm_col}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing disagreement gate columns: {sorted(missing)}")

    out = table.copy()
    base = pd.to_numeric(out[base_bpm_col], errors="coerce")
    aux = pd.to_numeric(out[aux_bpm_col], errors="coerce")
    if aux_probability_col in out.columns:
        aux_prob = pd.to_numeric(out[aux_probability_col], errors="coerce").fillna(0.0)
    else:
        aux_prob = pd.Series(1.0, index=out.index)

    upward_gap = aux - base
    mode = str(cfg.mode or "upward_conflict").strip().lower()
    if mode in {"upward", "upward_conflict", "t328"}:
        review = (
            aux.ge(cfg.aux_high_min_bpm)
            & base.le(cfg.base_max_bpm)
            & upward_gap.ge(cfg.upward_gap_min_bpm)
            & aux_prob.ge(cfg.aux_probability_min)
        )
        review_reason = "aux_high_base_normal_upward_conflict"
        pass_reason = "passed_or_not_upward_conflict"
    elif mode in {"absolute_gap", "symmetric_abs_gap", "t333"}:
        review = upward_gap.abs().ge(cfg.abs_gap_min_bpm) & aux_prob.ge(cfg.aux_probability_min)
        review_reason = "base_aux_absolute_gap_conflict"
        pass_reason = "passed_or_not_absolute_gap_conflict"
    else:
        raise ValueError(f"Unknown disagreement review mode: {cfg.mode}")

    out["disagreement_review_policy"] = cfg.policy_name
    out["base_hr_bpm"] = base
    out["aux_hr_bpm"] = aux
    out["aux_probability"] = aux_prob
    out["upward_disagreement_bpm"] = upward_gap
    out["disagreement_review_triggered"] = review.astype(int)
    out["disagreement_review_reason"] = np.where(
        review,
        review_reason,
        pass_reason,
    )

    if "decision" not in out.columns:
        out["decision"] = "release"
    if "released" not in out.columns:
        out["released"] = 1
    if output_bpm_col not in out.columns:
        out[output_bpm_col] = base

    out.loc[review, "decision"] = "review"
    out.loc[review, "released"] = 0
    out.loc[review, output_bpm_col] = math.nan

    evidence_values: list[str] = []
    for idx, row in out.iterrows():
        evidence = parse_evidence(row.get("evidence_json", "{}"))
        evidence["disagreement_review_gate"] = {
            "config": asdict(cfg),
            "triggered": bool(review.loc[idx]),
            "reason": str(row.get("disagreement_review_reason", "")),
            "base_hr_bpm": finite_float(row.get("base_hr_bpm")),
            "aux_hr_bpm": finite_float(row.get("aux_hr_bpm")),
            "aux_probability": finite_float(row.get("aux_probability")),
            "upward_disagreement_bpm": finite_float(row.get("upward_disagreement_bpm")),
        }
        evidence_values.append(json.dumps(evidence, ensure_ascii=False))
    out["evidence_json"] = evidence_values
    return out
