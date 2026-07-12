from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CandidateQuality:
    score: float
    confidence_score: float
    roi_score: float
    temporal_score: float
    validation_score: float
    agreement_score: float
    group_stability_score: float
    method_prior: float


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def method_reliability_prior(method: str) -> float:
    """A fixed first-pass prior derived from Phase D engineering insight.

    This is intentionally conservative and documented as a first-pass product
    policy, not a learned paper-grade calibration model.
    """
    method = str(method)
    if "optical_flow_y_body_aware_half_validated" in method:
        return 0.92
    if "motion_energy_semantic_half_validated" in method:
        return 0.68
    if "motion_energy_body_aware_half_validated" in method:
        return 0.64
    if "motion_energy_best_confidence_half_validated" in method:
        return 0.58
    if "depth_right_chest" in method:
        return 0.82
    if "depth_right_abdomen" in method:
        return 0.76
    if "depth_mean" in method:
        return 0.72
    if "depth_" in method:
        return 0.68
    if "green_" in method or "ir_" in method:
        return 0.52
    return 0.60


def group_prediction_context(predictions: list[float]) -> dict[str, float]:
    values = np.asarray([p for p in predictions if math.isfinite(p)], dtype=float)
    if values.size == 0:
        return {"median": math.nan, "spread": math.inf, "stability_score": 0.0}
    median = float(np.median(values))
    if values.size == 1:
        spread = 0.0
    else:
        spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    return {
        "median": median,
        "spread": spread,
        "stability_score": clamp01(1.0 - spread / 30.0),
    }


def candidate_quality(
    row: Mapping[str, object],
    *,
    group_median: float,
    group_stability_score: float,
    use_method_prior: bool = True,
) -> CandidateQuality:
    confidence = finite_float(row.get("confidence"), 0.0)
    raw_confidence = finite_float(row.get("raw_confidence"), confidence)
    confidence_score = clamp01(max(confidence, raw_confidence) / 0.45)

    roi_quality = finite_float(row.get("roi_quality"), 0.55)
    roi_score = clamp01(roi_quality)

    pred = finite_float(row.get("pred_rr_bpm"), finite_float(row.get("validated_rr_bpm"), math.nan))
    previous = finite_float(row.get("previous_bpm"), math.nan)
    if math.isfinite(previous) and math.isfinite(pred):
        temporal_score = clamp01(1.0 - abs(pred - previous) / 24.0)
    else:
        temporal_score = 0.62

    decision = str(row.get("decision", ""))
    half_ratio = finite_float(row.get("half_power_ratio"), 0.0)
    if decision.startswith("half_"):
        validation_score = clamp01(half_ratio / 0.55)
    elif decision in {"raw_kept", "raw_half_out_of_band", ""}:
        validation_score = 0.70
    else:
        validation_score = 0.45

    if math.isfinite(pred) and math.isfinite(group_median):
        agreement_score = clamp01(1.0 - abs(pred - group_median) / 30.0)
    else:
        agreement_score = 0.50

    prior = method_reliability_prior(str(row.get("method", ""))) if use_method_prior else 0.60
    base = (
        0.22 * confidence_score
        + 0.16 * roi_score
        + 0.18 * temporal_score
        + 0.18 * validation_score
        + 0.16 * agreement_score
        + 0.10 * group_stability_score
    )
    score = 0.78 * base + 0.22 * prior if use_method_prior else base
    return CandidateQuality(
        score=clamp01(score),
        confidence_score=confidence_score,
        roi_score=roi_score,
        temporal_score=temporal_score,
        validation_score=validation_score,
        agreement_score=agreement_score,
        group_stability_score=group_stability_score,
        method_prior=prior,
    )

