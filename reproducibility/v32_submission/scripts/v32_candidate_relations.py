"""Prediction-only candidate-relation features frozen for V32 inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


RELATION_COLUMNS = (
    "pool_n_candidates",
    "pool_hr_mad",
    "pool_hr_iqr",
    "candidate_robust_z",
    "candidate_hr_rank_pct",
    "nearest_candidate_gap_bpm",
    "route_agreement3_frac",
    "route_agreement5_frac",
    "route_agreement10_frac",
    "deep_agreement5_frac",
    "deep_agreement10_frac",
    "classical_agreement5_frac",
    "classical_agreement10_frac",
    "cross_family_agreement5",
    "cross_family_agreement10",
    "local_support5_sum",
    "local_support10_sum",
    "half_branch_route_frac",
    "double_branch_route_frac",
    "agreement_minus_harmonic",
)


def _fraction_near(hr: np.ndarray, mask: np.ndarray, value: float, tolerance: float) -> float:
    denominator = int(mask.sum())
    if denominator == 0:
        return 0.0
    return float((np.abs(hr[mask] - value) <= tolerance).sum() / denominator)


def add_prediction_only_relations(pool: pd.DataFrame) -> pd.DataFrame:
    """Add within-window relations without reading a reference or target field."""

    required = {
        "sample_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise RuntimeError(f"candidate pool lacks relation inputs: {missing}")
    parts: list[pd.DataFrame] = []
    for _, raw in pool.groupby("sample_id", sort=False):
        group = raw.copy()
        hr = pd.to_numeric(group["candidate_hr_bpm"], errors="coerce").to_numpy(float)
        if not np.isfinite(hr).all():
            raise FloatingPointError("non-finite candidate HR in relation calculation")
        support_source = group.get("support_count", pd.Series(0.0, index=group.index))
        support = pd.to_numeric(support_source, errors="coerce").fillna(0.0).to_numpy(float)
        deep = group["source_type"].astype(str).eq("deep").to_numpy(bool)
        classical = group["source_type"].astype(str).eq("classical").to_numpy(bool)
        route = group["candidate_model"].astype(str).to_numpy(object)
        family = group["candidate_family"].astype(str).to_numpy(object)
        median = float(np.nanmedian(hr))
        mad = float(np.nanmedian(np.abs(hr - median)))
        q25, q75 = np.nanquantile(hr, [0.25, 0.75])
        denominator = max(1.4826 * mad, 1.0)
        unique_routes = np.unique(route)
        family_count = max(len(set(map(str, family))), 1)
        route_count = max(len(unique_routes), 1)
        rows: list[dict[str, float]] = []
        for index, value in enumerate(hr):
            delta = np.abs(hr - value)
            nonself = np.arange(len(hr)) != index
            nearest = float(np.nanmin(delta[nonself])) if nonself.any() else 0.0
            route_distance = np.asarray(
                [np.nanmin(delta[route == route_name]) for route_name in unique_routes],
                dtype=float,
            )
            near5 = delta <= 5.0
            near10 = delta <= 10.0
            family5 = set(map(str, family[near5]))
            family10 = set(map(str, family[near10]))
            half = np.abs(hr - value / 2.0) <= max(3.0, value * 0.04)
            double = np.abs(hr - value * 2.0) <= max(3.0, value * 0.08)
            rows.append(
                {
                    "pool_n_candidates": float(len(hr)),
                    "pool_hr_mad": mad,
                    "pool_hr_iqr": float(q75 - q25),
                    "candidate_robust_z": float(abs(value - median) / denominator),
                    "candidate_hr_rank_pct": float((hr <= value).mean()),
                    "nearest_candidate_gap_bpm": nearest,
                    "route_agreement3_frac": float((route_distance <= 3.0).mean()),
                    "route_agreement5_frac": float((route_distance <= 5.0).mean()),
                    "route_agreement10_frac": float((route_distance <= 10.0).mean()),
                    "deep_agreement5_frac": _fraction_near(hr, deep, value, 5.0),
                    "deep_agreement10_frac": _fraction_near(hr, deep, value, 10.0),
                    "classical_agreement5_frac": _fraction_near(hr, classical, value, 5.0),
                    "classical_agreement10_frac": _fraction_near(hr, classical, value, 10.0),
                    "cross_family_agreement5": float(len(family5) / family_count),
                    "cross_family_agreement10": float(len(family10) / family_count),
                    "local_support5_sum": float(support[near5].sum()),
                    "local_support10_sum": float(support[near10].sum()),
                    "half_branch_route_frac": float(len(set(route[half])) / route_count),
                    "double_branch_route_frac": float(len(set(route[double])) / route_count),
                    "agreement_minus_harmonic": float(
                        (route_distance <= 5.0).mean()
                        - len(set(route[half | double])) / route_count
                    ),
                }
            )
        relations = pd.DataFrame(rows, index=group.index)
        for column in relations.columns:
            group[column] = relations[column]
        parts.append(group)
    output = pd.concat(parts, ignore_index=True, sort=False)
    if output.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError("relation calculation changed candidate identity")
    return output

