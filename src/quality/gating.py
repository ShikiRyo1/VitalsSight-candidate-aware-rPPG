from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GatedPrediction:
    pred_rr_bpm: float
    accepted: bool
    quality_score: float
    refusal_reason: str
    selected_method: str
    candidate_count: int
    disagreement_bpm: float


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = [
        (float(value), max(0.0, float(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(float(value)) and math.isfinite(float(weight))
    ]
    if not pairs:
        return math.nan
    pairs.sort(key=lambda item: item[0])
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return float(np.median([value for value, _ in pairs]))
    threshold = total / 2.0
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= threshold:
            return value
    return pairs[-1][0]


def quality_gate(
    *,
    pred_rr_bpm: float,
    quality_score: float,
    selected_method: str,
    candidate_count: int,
    disagreement_bpm: float,
    min_quality: float,
    max_disagreement_bpm: float | None = None,
    min_rr_bpm: float = 15.0,
    max_rr_bpm: float = 80.0,
) -> GatedPrediction:
    if not math.isfinite(pred_rr_bpm):
        return GatedPrediction(pred_rr_bpm, False, quality_score, "nonfinite_prediction", selected_method, candidate_count, disagreement_bpm)
    if pred_rr_bpm < min_rr_bpm or pred_rr_bpm > max_rr_bpm:
        return GatedPrediction(pred_rr_bpm, False, quality_score, "outside_physiological_range", selected_method, candidate_count, disagreement_bpm)
    if quality_score < min_quality:
        return GatedPrediction(pred_rr_bpm, False, quality_score, "low_quality_score", selected_method, candidate_count, disagreement_bpm)
    if max_disagreement_bpm is not None and disagreement_bpm > max_disagreement_bpm:
        return GatedPrediction(pred_rr_bpm, False, quality_score, "high_candidate_disagreement", selected_method, candidate_count, disagreement_bpm)
    return GatedPrediction(pred_rr_bpm, True, quality_score, "accepted", selected_method, candidate_count, disagreement_bpm)

