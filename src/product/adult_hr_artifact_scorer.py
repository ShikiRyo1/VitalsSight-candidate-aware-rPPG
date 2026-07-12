from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.product.adult_hr_disagreement_review import (
    DisagreementReviewConfig,
    apply_disagreement_review_gate,
)
from src.product.adult_hr_feature_harmonizer import FEATURE_COLUMNS, harmonize_candidate_features


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = PROJECT / "models" / "t330_disagreement_candidate_scorers.pkl"


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_t330_disagreement_artifact(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    feature_columns = list(artifact.get("feature_columns", []))
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("T330 artifact feature columns do not match product harmonizer feature columns.")
    return artifact


def ensure_sklearn_predict_proba_compat(model: Any) -> None:
    """Patch small sklearn pickle-version gaps before calling predict_proba."""
    estimators = [model]
    if hasattr(model, "steps"):
        estimators.extend(step for _, step in getattr(model, "steps", []) if step is not None)
    if hasattr(model, "named_steps"):
        estimators.extend(step for step in getattr(model, "named_steps", {}).values() if step is not None)
    for estimator in estimators:
        if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(estimator, "multi_class"):
            setattr(estimator, "multi_class", "auto")


def class_one_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    ensure_sklearn_predict_proba_compat(model)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("model"), "classes_", None)
        if classes is not None:
            classes = list(classes)
            if 1 in classes:
                return proba[:, classes.index(1)]
        return proba[:, -1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-score))
    return np.asarray(model.predict(features), dtype=float)


def logit_probability(values: np.ndarray) -> np.ndarray:
    p = np.clip(values.astype(float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def transition_penalty(prev_bpm: np.ndarray, cur_bpm: np.ndarray, decode: dict[str, Any]) -> np.ndarray:
    jump = np.abs(prev_bpm[:, None] - cur_bpm[None, :])
    free = float(decode.get("transition_free_bpm", 0.0))
    sigma = max(float(decode.get("transition_sigma_bpm", 1.0)), 1e-6)
    excess = np.maximum(0.0, jump - free)
    quad = (excess / sigma) ** 2
    return float(decode.get("transition_weight", 1.0)) * quad + float(decode.get("jump_linear_weight", 0.0)) * excess


def parse_sequence_window(sample_id: object, window_idx: object | None = None) -> tuple[str, int]:
    text = str(sample_id)
    if window_idx is not None and math.isfinite(finite_float(window_idx)):
        idx = int(finite_float(window_idx))
        sequence_id = re.sub(r"_w\d+.*$", "", text) or text
        return sequence_id, idx
    match = re.search(r"^(?P<prefix>.+?)_w(?P<idx>\d+)", text)
    if match:
        return match.group("prefix"), int(match.group("idx"))
    return text, 0


def add_sequence_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sequence_ids: list[str] = []
    window_indices: list[int] = []
    for _, row in out.iterrows():
        sequence_id, window_idx = parse_sequence_window(row.get("sample_id", "sample"), row.get("window_idx", row.get("window_id")))
        sequence_ids.append(sequence_id)
        window_indices.append(window_idx)
    out["sequence_id"] = sequence_ids
    out["artifact_window_idx"] = window_indices
    return out


def viterbi_positions(windows: list[pd.DataFrame], probability_col: str, decode: dict[str, Any]) -> list[int]:
    scores: list[np.ndarray] = []
    bpms: list[np.ndarray] = []
    for group in windows:
        raw = logit_probability(
            pd.to_numeric(group[probability_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        raw = raw - np.nanmax(raw)
        scores.append(float(decode.get("score_scale", 1.0)) * raw)
        bpms.append(pd.to_numeric(group["candidate_bpm"], errors="coerce").to_numpy(dtype=float))

    dp = scores[0].copy()
    backptrs: list[np.ndarray] = []
    for step in range(1, len(windows)):
        penalty = transition_penalty(bpms[step - 1], bpms[step], decode)
        total = dp[:, None] + scores[step][None, :] - penalty
        back = np.nanargmax(total, axis=0)
        dp = total[back, np.arange(total.shape[1])]
        backptrs.append(back)
    chosen = [int(np.nanargmax(dp))]
    for back in reversed(backptrs):
        chosen.append(int(back[chosen[-1]]))
    chosen = list(reversed(chosen))
    return [int(group.index[idx]) for group, idx in zip(windows, chosen)]


def decode_candidate_sequence(frame: pd.DataFrame, probability_col: str, decode: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = add_sequence_columns(frame)
    if str(decode.get("decode", "single_window")) == "single_window":
        selected = (
            out.sort_values(
                ["sample_id", probability_col, "support_methods", "top1_support_count", "rank_score"],
                ascending=[True, False, False, False, False],
            )
            .groupby("sample_id", sort=False, dropna=False)
            .head(1)
            .copy()
        )
        return selected

    positions: list[int] = []
    sorted_df = out.sort_values(["sequence_id", "artifact_window_idx", "candidate_bpm"]).copy()
    for _, sample in sorted_df.groupby("sequence_id", sort=False, dropna=False):
        windows = [group.copy() for _, group in sample.groupby("artifact_window_idx", sort=True, dropna=False)]
        if windows:
            positions.extend(viterbi_positions(windows, probability_col, decode))
    return out.loc[positions].copy()


def score_live_candidates_with_t330_artifact(
    candidates: pd.DataFrame,
    *,
    artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    model_artifact = artifact or load_t330_disagreement_artifact()
    feature_columns = list(model_artifact["feature_columns"])
    scored = harmonize_candidate_features(candidates)
    features = scored[feature_columns].astype(float)
    scored["t330_base_close5_probability"] = class_one_probability(model_artifact["base"]["model"], features)
    scored["t330_aux_close5_probability"] = class_one_probability(model_artifact["auxiliary"]["model"], features)
    return scored


def build_disagreement_review_product_from_candidates(
    candidates: pd.DataFrame,
    *,
    artifact: dict[str, Any] | None = None,
    review_config: DisagreementReviewConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    model_artifact = artifact or load_t330_disagreement_artifact()
    scored = score_live_candidates_with_t330_artifact(candidates, artifact=model_artifact)
    base_selected = decode_candidate_sequence(
        scored,
        "t330_base_close5_probability",
        model_artifact["base"]["decode"],
    )
    aux_selected = decode_candidate_sequence(
        scored,
        "t330_aux_close5_probability",
        model_artifact["auxiliary"]["decode"],
    )

    base_cols = {
        "candidate_id": "base_candidate_id",
        "candidate_bpm": "base_hr_bpm",
        "t330_base_close5_probability": "base_probability",
        "support_methods": "base_support_methods",
        "top1_support_count": "base_top1_support_count",
        "high_harmonic_anchor": "base_high_harmonic_anchor",
        "low_alias_risk": "base_low_alias_risk",
    }
    aux_cols = {
        "candidate_id": "aux_candidate_id",
        "candidate_bpm": "aux_hr_bpm",
        "t330_aux_close5_probability": "aux_probability",
        "support_methods": "aux_support_methods",
        "top1_support_count": "aux_top1_support_count",
        "high_harmonic_anchor": "aux_high_harmonic_anchor",
        "low_alias_risk": "aux_low_alias_risk",
    }
    base = base_selected[["sample_id", *base_cols.keys()]].rename(columns=base_cols)
    aux = aux_selected[["sample_id", *aux_cols.keys()]].rename(columns=aux_cols)
    product_input = base.merge(aux, on="sample_id", how="outer")
    product_input["product_hr_bpm"] = pd.to_numeric(product_input["base_hr_bpm"], errors="coerce")
    product_input["decision"] = np.where(product_input["product_hr_bpm"].notna(), "release", "review")
    product_input["released"] = np.where(product_input["product_hr_bpm"].notna(), 1, 0)
    product_input["evidence_json"] = [
        json.dumps(
            {
                "t330_scorer": {
                    "base_policy": model_artifact["base"]["decode"]["policy"],
                    "auxiliary_policy": model_artifact["auxiliary"]["decode"]["policy"],
                    "feature_schema_n": len(model_artifact["feature_columns"]),
                }
            },
            ensure_ascii=False,
        )
        for _ in range(len(product_input))
    ]
    product = apply_disagreement_review_gate(
        product_input,
        config=review_config or DisagreementReviewConfig(),
        base_bpm_col="base_hr_bpm",
        aux_bpm_col="aux_hr_bpm",
        aux_probability_col="aux_probability",
        output_bpm_col="product_hr_bpm",
    )
    return scored, base_selected, aux_selected, product
