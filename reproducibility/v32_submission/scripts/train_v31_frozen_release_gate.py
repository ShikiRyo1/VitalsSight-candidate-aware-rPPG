#!/usr/bin/env python3
"""Train and freeze a development-only evidence-linked release/review gate.

The gate is intentionally separate from candidate selection. It estimates a
cross-fitted, uncalibrated safe-within-10 score for the already selected,
source-identified candidate using the same 43 prediction-time node features as
the frozen selector. The threshold is a label-independent 20% coverage anchor
computed from development OOF risk scores. It is not a safety probability or a
clinical release guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import brier_score_loss, roc_auc_score

import screen_v31_tree_rankers as candidate


TASK_ID = "V31_FROZEN_DEVELOPMENT_RELEASE_GATE"
SEEDS = (704, 1704, 2704)
UNSAFE_THRESHOLD_BPM = 10.0
TARGET_DEVELOPMENT_COVERAGE = 0.20
CONFIDENCE = 0.95


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--nested-selected-predictions", type=Path, required=True)
    parser.add_argument("--outer-subject-splits", type=Path, required=True)
    parser.add_argument("--selector-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def cp_upper(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    if n <= 0:
        return math.nan
    if k >= n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def load_splits(path: Path, expected_subjects: set[str]) -> dict[int, tuple[set[str], set[str]]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"outer_fold", "train_subjects", "test_subjects"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"outer split ledger missing columns: {missing}")
    splits: dict[int, tuple[set[str], set[str]]] = {}
    all_test: list[str] = []
    for row in frame.itertuples(index=False):
        fold = int(row.outer_fold)
        train = {value for value in str(row.train_subjects).split(";") if value}
        test = {value for value in str(row.test_subjects).split(";") if value}
        if fold in splits or not train or not test or train & test:
            raise RuntimeError(f"invalid outer split {fold}")
        if train | test != expected_subjects:
            raise RuntimeError(f"outer split subject coverage mismatch: {fold}")
        splits[fold] = (train, test)
        all_test.extend(sorted(test))
    if sorted(all_test) != sorted(expected_subjects):
        raise RuntimeError("outer test subjects are not a one-time partition")
    return splits


def threshold_row(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    released = frame[frame["research_risk_score"].le(float(threshold))].copy()
    unsafe = released["abs_error_bpm"].gt(UNSAFE_THRESHOLD_BPM)
    participant_any = (
        released.assign(_unsafe=unsafe)
        .groupby("subject_std", sort=False)["_unsafe"]
        .any()
        .astype(bool)
    )
    participant_mae = released.groupby("subject_std")["abs_error_bpm"].mean()
    all_participant_mae = frame.groupby("subject_std")["abs_error_bpm"].mean()
    return {
        "risk_threshold": float(threshold),
        "released_windows": int(len(released)),
        "window_coverage": float(len(released) / max(len(frame), 1)),
        "released_participants": int(participant_any.size),
        "participant_coverage": float(participant_any.size / frame["subject_std"].nunique()),
        "released_window_mae_bpm": float(released["abs_error_bpm"].mean()) if len(released) else math.nan,
        "released_participant_equal_mae_bpm": float(participant_mae.mean()) if len(participant_mae) else math.nan,
        "all_participant_equal_mae_bpm": float(all_participant_mae.mean()),
        "unsafe_windows_gt10": int(unsafe.sum()),
        "unsafe_window_rate_gt10": float(unsafe.mean()) if len(released) else math.nan,
        "unsafe_window_upper95_gt10": cp_upper(int(unsafe.sum()), int(len(released))),
        "participants_with_any_unsafe_release_gt10": int(participant_any.sum()),
        "unsafe_participant_rate_gt10": float(participant_any.mean()) if len(participant_any) else math.nan,
        "unsafe_participant_upper95_gt10": cp_upper(
            int(participant_any.sum()), int(len(participant_any))
        ),
    }


def positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    if 1 not in model.classes_:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    selector_contract = json.loads(args.selector_contract.read_text(encoding="utf-8"))
    if selector_contract.get("status") != "FROZEN_DEVELOPMENT_SELECTOR":
        raise RuntimeError("selector contract is not frozen")
    if selector_contract.get("relation_features_enabled") is not False:
        raise RuntimeError("release gate must follow the frozen no-relation selector")
    feature_columns = list(selector_contract.get("feature_columns", []))
    if len(feature_columns) != 43:
        raise RuntimeError("release-gate feature contract must contain 43 columns")

    pool = pd.read_csv(args.input_pool, low_memory=False)
    required_pool = {
        "sample_id",
        "subject_std",
        "candidate_id",
        "candidate_hr_bpm",
        "candidate_abs_error",
        "source_type",
        "candidate_family",
        "candidate_model",
    }
    missing = sorted(required_pool - set(pool.columns))
    if missing:
        raise RuntimeError(f"development pool missing columns: {missing}")
    if pool["subject_std"].astype(str).nunique() != 42 or pool["sample_id"].astype(str).nunique() != 439:
        raise RuntimeError("development pool topology drifted")
    if pool.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError("duplicate development candidate identity")
    candidate_error = pd.to_numeric(pool["candidate_abs_error"], errors="coerce")
    if not np.isfinite(candidate_error).all():
        raise RuntimeError("development candidate errors are incomplete")
    pool = pool.copy()
    pool["candidate_abs_error"] = candidate_error
    pool["_pool_row"] = np.arange(len(pool), dtype=int)

    selected = pd.read_csv(args.nested_selected_predictions, low_memory=False)
    required_selected = {
        "outer_fold",
        "sample_id",
        "subject_std",
        "selected_candidate_id",
        "pred_hr_bpm",
        "abs_error_bpm",
    }
    missing = sorted(required_selected - set(selected.columns))
    if missing:
        raise RuntimeError(f"nested selected predictions missing columns: {missing}")
    if len(selected) != 439 or selected["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("nested selected prediction coverage drifted")
    selected = selected.merge(
        pool[
            [
                "sample_id",
                "subject_std",
                "candidate_id",
                "candidate_hr_bpm",
                "candidate_abs_error",
                "_pool_row",
            ]
        ],
        left_on=["sample_id", "subject_std", "selected_candidate_id"],
        right_on=["sample_id", "subject_std", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["_pool_row"].isna().any():
        raise RuntimeError("nested selected candidate is absent from the development pool")
    if not np.allclose(
        pd.to_numeric(selected["pred_hr_bpm"], errors="coerce"),
        pd.to_numeric(selected["candidate_hr_bpm"], errors="coerce"),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("nested selected HR is not the retained candidate value")
    if not np.allclose(
        pd.to_numeric(selected["abs_error_bpm"], errors="coerce"),
        pd.to_numeric(selected["candidate_abs_error"], errors="coerce"),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("nested selected error does not match the retained candidate")

    subjects = set(pool["subject_std"].astype(str).unique())
    splits = load_splits(args.outer_subject_splits, subjects)
    features, observed_columns = candidate.feature_frame(pool, relation=False)
    if set(observed_columns) != set(feature_columns):
        raise RuntimeError("release-gate features differ from the selector contract")
    features = features.reindex(columns=feature_columns)

    oof_seed_rows: list[dict[str, Any]] = []
    for fold, (train_subjects, test_subjects) in sorted(splits.items()):
        fold_selected = selected[selected["outer_fold"].astype(int).eq(fold)].copy()
        if set(fold_selected["subject_std"].astype(str).unique()) != test_subjects:
            raise RuntimeError(f"selected prediction/test-subject mismatch in fold {fold}")
        train_mask = pool["subject_std"].astype(str).isin(train_subjects)
        train_pool = pool.loc[train_mask]
        train_features = features.loc[train_mask]
        medians = train_features.median(numeric_only=True).reindex(feature_columns).fillna(0.0)
        x_train = train_features.fillna(medians).fillna(0.0).to_numpy(float)
        y_train = train_pool["candidate_abs_error"].le(UNSAFE_THRESHOLD_BPM).astype(int).to_numpy()
        weights = candidate.sample_weights(train_pool)
        selected_rows = fold_selected["_pool_row"].astype(int).to_numpy()
        x_test = features.iloc[selected_rows].fillna(medians).fillna(0.0).to_numpy(float)
        for seed in SEEDS:
            model = candidate.make_classifier(int(seed), 3, int(args.n_estimators))
            model.set_params(n_jobs=int(args.n_jobs))
            model.fit(x_train, y_train, sample_weight=weights)
            score = positive_probability(model, x_test)
            if not np.isfinite(score).all():
                raise RuntimeError(f"non-finite OOF release score in fold {fold}, seed {seed}")
            for row, value in zip(fold_selected.itertuples(index=False), score, strict=True):
                oof_seed_rows.append(
                    {
                        "outer_fold": fold,
                        "seed": int(seed),
                        "sample_id": str(row.sample_id),
                        "subject_std": str(row.subject_std),
                        "selected_candidate_id": str(row.selected_candidate_id),
                        "pred_hr_bpm": float(row.pred_hr_bpm),
                        "abs_error_bpm": float(row.abs_error_bpm),
                        "research_safe10_score": float(value),
                    }
                )

    oof_per_seed = pd.DataFrame(oof_seed_rows)
    identity = [
        "outer_fold",
        "sample_id",
        "subject_std",
        "selected_candidate_id",
        "pred_hr_bpm",
        "abs_error_bpm",
    ]
    oof = (
        oof_per_seed.groupby(identity, as_index=False)
        .agg(
            research_safe10_score=("research_safe10_score", "median"),
            research_safe10_score_mean=("research_safe10_score", "mean"),
            research_safe10_score_std=("research_safe10_score", "std"),
            n_seed_scores=("seed", "nunique"),
        )
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(oof) != 439 or not oof["n_seed_scores"].eq(3).all():
        raise RuntimeError("OOF release-score coverage is incomplete")
    oof["research_risk_score"] = 1.0 - oof["research_safe10_score"]
    ordered_risk = np.sort(oof["research_risk_score"].to_numpy(float))
    threshold_index = max(
        0,
        min(len(ordered_risk) - 1, int(math.ceil(TARGET_DEVELOPMENT_COVERAGE * len(oof))) - 1),
    )
    threshold = float(ordered_risk[threshold_index])
    oof["proposed_state"] = np.where(
        oof["research_risk_score"].le(threshold), "PROPOSED_RELEASE", "PROPOSED_REVIEW"
    )
    threshold_curve = pd.DataFrame(
        [threshold_row(oof, value) for value in np.sort(oof["research_risk_score"].unique())]
    )
    selected_operating_point = threshold_row(oof, threshold)
    safe10 = oof["abs_error_bpm"].le(UNSAFE_THRESHOLD_BPM).astype(int)
    oof_validation = {
        "roc_auc_window_descriptive": float(
            roc_auc_score(safe10, oof["research_safe10_score"])
        ),
        "brier_window_descriptive": float(
            brier_score_loss(safe10, oof["research_safe10_score"])
        ),
        "selected_operating_point": selected_operating_point,
        "threshold_sensitivity": {
            f"released_unsafe_rate_gt{int(value)}": float(
                oof.loc[oof["proposed_state"].eq("PROPOSED_RELEASE"), "abs_error_bpm"].gt(value).mean()
            )
            for value in (5.0, 8.0, 10.0, 15.0)
        },
        "interpretation": (
            "Cross-fitted development evidence only. The score is uncalibrated; the 20% "
            "coverage anchor is frozen without using OOF outcome labels for threshold selection."
        ),
    }

    oof_seed_path = args.output_dir / "development_oof_release_scores_per_seed.csv"
    oof_path = args.output_dir / "development_oof_release_scores.csv"
    curve_path = args.output_dir / "development_oof_release_threshold_curve.csv"
    oof_per_seed.to_csv(oof_seed_path, index=False)
    oof.to_csv(oof_path, index=False)
    threshold_curve.to_csv(curve_path, index=False)

    full_medians = features.median(numeric_only=True).reindex(feature_columns).fillna(0.0)
    x_full = features.fillna(full_medians).fillna(0.0).to_numpy(float)
    y_full = pool["candidate_abs_error"].le(UNSAFE_THRESHOLD_BPM).astype(int).to_numpy()
    weights_full = candidate.sample_weights(pool)
    models: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = candidate.make_classifier(int(seed), 3, int(args.n_estimators))
        model.set_params(n_jobs=int(args.n_jobs))
        model.fit(x_full, y_full, sample_weight=weights_full)
        probe = positive_probability(model, x_full[: min(32, len(x_full))])
        artifact = {
            "task_id": TASK_ID,
            "model": model,
            "seed": int(seed),
            "objective": "candidate_safe_within_10_bpm_score",
            "feature_columns": feature_columns,
            "feature_medians": {key: float(full_medians[key]) for key in feature_columns},
            "relation_features_enabled": False,
            "n_estimators": int(args.n_estimators),
            "min_samples_leaf": 3,
            "max_features": 0.75,
            "unsafe_threshold_bpm": UNSAFE_THRESHOLD_BPM,
            "research_score_not_probability": True,
            "external_outcomes_accessed": False,
        }
        path = args.output_dir / f"release_gate_seed{seed}.joblib"
        joblib.dump(artifact, path, compress=3)
        loaded = joblib.load(path)
        loaded_probe = positive_probability(loaded["model"], x_full[: min(32, len(x_full))])
        maximum_difference = float(np.max(np.abs(probe - loaded_probe)))
        if not np.allclose(probe, loaded_probe, atol=1e-12, rtol=1e-12):
            raise RuntimeError(f"release-gate serialization mismatch for seed {seed}")
        models.append(
            {
                "seed": int(seed),
                **file_record(path),
                "probe_prediction_sha256": hashlib.sha256(
                    np.asarray(loaded_probe, dtype="<f8").tobytes()
                ).hexdigest().upper(),
                "serialization_roundtrip_max_abs_diff": maximum_difference,
                "serialization_roundtrip_tolerance": 1e-12,
            }
        )

    contract = {
        "task_id": TASK_ID,
        "schema_version": "1.0",
        "generated_at_utc": utc_now(),
        "status": "FROZEN_DEVELOPMENT_RELEASE_GATE",
        "architecture": "ExtraTreesClassifier",
        "architecture_selection_rule": (
            "Reuse the frozen node-only selector feature contract, leaf size and three seeds; "
            "no release-gate architecture screen was performed."
        ),
        "objective": "candidate_safe_within_10_bpm_score",
        "unsafe_threshold_bpm": UNSAFE_THRESHOLD_BPM,
        "threshold_selection_rule": (
            "Empirical 20th percentile of cross-fitted development research risk scores; "
            "OOF outcome labels are not consulted when selecting the threshold."
        ),
        "target_development_window_coverage": TARGET_DEVELOPMENT_COVERAGE,
        "proposed_release_threshold": threshold,
        "seeds": list(SEEDS),
        "n_estimators": int(args.n_estimators),
        "training_participants": 42,
        "training_windows": 439,
        "training_candidates": int(len(pool)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "relation_features_enabled": False,
        "models": models,
        "inputs": {
            "development_pool": file_record(args.input_pool),
            "nested_selected_predictions": file_record(args.nested_selected_predictions),
            "outer_subject_splits": file_record(args.outer_subject_splits),
            "selector_contract": file_record(args.selector_contract),
        },
        "development_oof_outputs": {
            oof_seed_path.name: file_record(oof_seed_path),
            oof_path.name: file_record(oof_path),
            curve_path.name: file_record(curve_path),
        },
        "development_oof_validation": oof_validation,
        "research_score_not_probability": True,
        "no_calibrated_safety_claim": True,
        "external_outcomes_accessed": False,
        "claim_boundary": (
            "The gate proposes release or review separately from the selected HR. It is a "
            "development-frozen research score, not a calibrated safety probability, clinical "
            "guarantee or authorization for autonomous use."
        ),
    }
    write_json(args.output_dir / "release_gate_freeze_contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
