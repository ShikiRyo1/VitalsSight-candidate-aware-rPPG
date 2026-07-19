#!/usr/bin/env python3
"""Nested development audit of a causal source-preserving candidate path.

Each window retains all observed route-specific candidates.  A fixed two-branch
ExtraTrees emission model estimates candidate error and <=5-BPM plausibility
from intrinsic and within-window relation features.  A causal dynamic program
then selects the minimum-cost candidate at each time using only current/past
windows.  The temporal penalty is selected inside participant-disjoint inner
folds.  External data never enter architecture or penalty selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_v31_matched_stacking_baselines as stacking  # noqa: E402
import screen_v31_tree_rankers as candidate  # noqa: E402


TASK_ID = "V32_CAUSAL_SOURCE_PRESERVING_CANDIDATE_PATH"
SEEDS = (704, 1704, 2704)
METHOD = "VitalsSight_v32_causal_candidate_path"
INDEPENDENT = "VitalsSight_v32_independent_emission"
PRIMARY_COMPARATOR = "VitalsSight_candidate_aware"
STACKER_COMPARATOR = "matched_extra_trees_stacker"
WINDOW_RE = re.compile(r"^(?P<sequence>.+)_w(?P<window>\d+)$")


@dataclass(frozen=True)
class Protocol:
    outer_split_seed: int = 310718
    inner_split_seed: int = 610718
    outer_folds: int = 3
    inner_folds: int = 3
    seeds: tuple[int, ...] = SEEDS
    n_estimators: int = 160
    min_samples_leaf: int = 7
    max_features: float = 0.75
    intrinsic_relation_blend: float = 0.5
    unsafe5_penalty: float = 5.0
    transition_lambdas: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40)
    max_transition_bpm: float = 40.0
    bootstrap_draws: int = 10000
    role: str = "development_only_nested_participant_disjoint"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_regressor(protocol: Protocol, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=protocol.n_estimators,
        min_samples_leaf=protocol.min_samples_leaf,
        max_features=protocol.max_features,
        random_state=seed,
        n_jobs=-1,
    )


def make_classifier(protocol: Protocol, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=protocol.n_estimators,
        min_samples_leaf=protocol.min_samples_leaf,
        max_features=protocol.max_features,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def positive_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, classes.index(1)]


def score_emissions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    protocol: Protocol,
    seed: int,
) -> pd.DataFrame:
    error, _, safe5 = candidate.target_values(train)
    weights = candidate.sample_weights(train)
    alpha = protocol.intrinsic_relation_blend

    node_train, node_test, node_columns = candidate.align_features(train, test, relation=False)
    relation_train, relation_test, relation_columns = candidate.align_features(
        train, test, relation=True
    )
    if any(
        token in column.lower()
        for column in (*node_columns, *relation_columns)
        for token in candidate.FORBIDDEN_TOKENS
    ):
        raise RuntimeError("forbidden target-derived feature entered emission model")

    node_reg = make_regressor(protocol, seed)
    relation_reg = make_regressor(protocol, seed + 100_000)
    node_safe = make_classifier(protocol, seed + 200_000)
    relation_safe = make_classifier(protocol, seed + 300_000)
    target_error = np.clip(error, 0.0, 20.0)

    node_reg.fit(node_train, target_error, sample_weight=weights)
    relation_reg.fit(relation_train, target_error, sample_weight=weights)
    node_safe.fit(node_train, safe5, sample_weight=weights)
    relation_safe.fit(relation_train, safe5, sample_weight=weights)

    predicted_error = (
        (1.0 - alpha) * node_reg.predict(node_test)
        + alpha * relation_reg.predict(relation_test)
    )
    safe_probability = (
        (1.0 - alpha) * positive_probability(node_safe, node_test)
        + alpha * positive_probability(relation_safe, relation_test)
    )
    emission = predicted_error + protocol.unsafe5_penalty * (1.0 - safe_probability)
    if not np.isfinite(emission).all():
        raise FloatingPointError("non-finite candidate emission score")

    keep = [
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    scored = test[keep].copy().reset_index(drop=True)
    scored["predicted_error_bpm"] = predicted_error
    scored["predicted_safe5_probability"] = safe_probability
    scored["emission_score"] = emission
    scored["seed"] = seed
    return scored


def add_sequence_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sequences: list[str] = []
    windows: list[int] = []
    for sample_id in result["sample_id"].astype(str):
        match = WINDOW_RE.match(sample_id)
        if match is None:
            raise RuntimeError(f"sample_id lacks frozen sequence/window identity: {sample_id}")
        sequences.append(match.group("sequence"))
        windows.append(int(match.group("window")))
    result["sequence_id"] = sequences
    result["window_index"] = windows
    identity = result[["sample_id", "sequence_id", "window_index"]].drop_duplicates()
    if identity.duplicated(["sequence_id", "window_index"]).any():
        raise RuntimeError("sequence/window identity is not unique")
    return result


def ensemble_emissions(per_seed: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    identity = [
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    result = (
        per_seed.groupby(identity, as_index=False)
        .agg(
            emission_score=("emission_score", "median"),
            emission_score_mean=("emission_score", "mean"),
            predicted_error_bpm=("predicted_error_bpm", "median"),
            predicted_safe5_probability=("predicted_safe5_probability", "median"),
            n_seed_scores=("seed", "nunique"),
        )
    )
    if not result["n_seed_scores"].eq(expected_seeds).all():
        raise RuntimeError("candidate emission seed coverage mismatch")
    return add_sequence_identity(result)


def select_causal_path(
    scored: pd.DataFrame,
    transition_lambda: float,
    max_transition_bpm: float,
    method: str,
) -> pd.DataFrame:
    numeric = scored[["candidate_hr_bpm", "emission_score"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise FloatingPointError("non-finite candidate HR or emission score")
    selections: list[pd.Series] = []
    for sequence_id, sequence in scored.groupby("sequence_id", sort=True):
        windows = sorted(sequence["window_index"].unique().tolist())
        previous_cost: np.ndarray | None = None
        previous_hr: np.ndarray | None = None
        previous_window: int | None = None
        for window_index in windows:
            candidates = sequence[sequence["window_index"].eq(window_index)].copy()
            candidates = candidates.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
            emission = candidates["emission_score"].to_numpy(float)
            hr = candidates["candidate_hr_bpm"].to_numpy(float)
            contiguous = previous_window is not None and window_index == previous_window + 1
            if previous_cost is None or previous_hr is None or not contiguous:
                cost = emission.copy()
            else:
                transition = np.minimum(
                    np.abs(previous_hr[:, None] - hr[None, :]), max_transition_bpm
                )
                cost = emission + np.min(
                    previous_cost[:, None] + transition_lambda * transition,
                    axis=0,
                )
            cost = cost - float(np.min(cost))
            selected_index = int(np.argmin(cost))
            row = candidates.iloc[selected_index].copy()
            row["path_cost_normalized"] = float(cost[selected_index])
            row["transition_lambda"] = float(transition_lambda)
            selections.append(row)
            previous_cost = cost
            previous_hr = hr
            previous_window = window_index

    selected = pd.DataFrame(selections)
    result = selected[
        [
            "sample_id",
            "subject_std",
            "sequence_id",
            "window_index",
            "gt_hr_bpm",
            "candidate_hr_bpm",
            "candidate_id",
            "source_type",
            "candidate_family",
            "candidate_model",
            "emission_score",
            "predicted_error_bpm",
            "predicted_safe5_probability",
            "transition_lambda",
            "path_cost_normalized",
        ]
    ].rename(
        columns={
            "candidate_hr_bpm": "pred_hr_bpm",
            "candidate_id": "selected_candidate_id",
            "source_type": "selected_source_type",
            "candidate_family": "selected_candidate_family",
            "candidate_model": "selected_candidate_model",
        }
    )
    result.insert(0, "method", method)
    result["abs_error_bpm"] = (result["pred_hr_bpm"] - result["gt_hr_bpm"]).abs()
    if result["sample_id"].duplicated().any():
        raise RuntimeError("causal selector emitted duplicate windows")
    return result.sort_values(["sequence_id", "window_index"]).reset_index(drop=True)


def participant_equal_mae(predictions: pd.DataFrame) -> float:
    return float(predictions.groupby("subject_std")["abs_error_bpm"].mean().mean())


def participant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, method_frame in predictions.groupby("method", sort=False):
        for participant, frame in method_frame.groupby("subject_std", sort=False):
            error = frame["abs_error_bpm"].to_numpy(float)
            rows.append(
                {
                    "method": method,
                    "subject_std": participant,
                    "n_windows": len(frame),
                    "mae_bpm": float(np.mean(error)),
                    "within5": float(np.mean(error <= 5.0)),
                    "within10": float(np.mean(error <= 10.0)),
                    "unsafe_gt10": float(np.mean(error > 10.0)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(predictions: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, frame in predictions.groupby("method", sort=False):
        error = frame["abs_error_bpm"].to_numpy(float)
        participant = participants[participants["method"].eq(method)]
        rows.append(
            {
                "method": method,
                "n_windows": len(frame),
                "n_subjects": frame["subject_std"].nunique(),
                "window_mae_bpm": float(np.mean(error)),
                "window_rmse_bpm": float(np.sqrt(np.mean(np.square(error)))),
                "within5": float(np.mean(error <= 5.0)),
                "within10": float(np.mean(error <= 10.0)),
                "unsafe_gt10": float(np.mean(error > 10.0)),
                "participant_equal_mae_bpm": float(participant["mae_bpm"].mean()),
                "participant_equal_within10": float(participant["within10"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("participant_equal_mae_bpm")


def paired_bootstrap(
    participants: pd.DataFrame,
    method_a: str,
    method_b: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    wide = participants.pivot(index="subject_std", columns="method", values="mae_bpm")
    pair = wide[[method_a, method_b]].dropna()
    delta = (pair[method_a] - pair[method_b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[take].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(delta)))
    permuted = np.abs((signs * delta).mean(axis=1))
    observed = abs(float(delta.mean()))
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_subjects": len(delta),
        "mean_delta_a_minus_b_bpm": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "paired_sign_flip_p_plus_one": float(
            (np.count_nonzero(permuted >= observed - 1e-15) + 1) / (draws + 1)
        ),
        "a_lower_mae_supported": bool(np.quantile(means, 0.975) < 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--matched-stacking-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    protocol = Protocol(n_estimators=args.n_estimators, bootstrap_draws=args.bootstrap_draws)

    pool = pd.read_csv(args.input_pool, low_memory=False)
    if pool["subject_std"].astype(str).nunique() != 42:
        raise RuntimeError("development participant identity mismatch")
    if pool["sample_id"].astype(str).nunique() != 439:
        raise RuntimeError("development window identity mismatch")
    if (pool.groupby("sample_id")["gt_hr_bpm"].nunique() != 1).any():
        raise RuntimeError("candidate-independent reference mismatch")
    if pool.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError("candidate identity is not unique within a window")

    subjects = sorted(pool["subject_std"].astype(str).unique())
    outer_splits = candidate.fixed_folds(subjects, protocol.outer_folds, protocol.outer_split_seed)
    split_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    outer_score_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    independent_rows: list[pd.DataFrame] = []

    for outer_index, (outer_train_subjects, outer_test_subjects) in enumerate(
        outer_splits, start=1
    ):
        outer_train = pool[pool["subject_std"].astype(str).isin(outer_train_subjects)].reset_index(drop=True)
        outer_test = pool[pool["subject_std"].astype(str).isin(outer_test_subjects)].reset_index(drop=True)
        split_rows.append(
            {
                "outer_fold": outer_index,
                "train_subjects": ";".join(outer_train_subjects),
                "test_subjects": ";".join(outer_test_subjects),
                "n_train_subjects": len(outer_train_subjects),
                "n_test_subjects": len(outer_test_subjects),
            }
        )
        inner_splits = candidate.fixed_folds(
            outer_train_subjects,
            protocol.inner_folds,
            protocol.inner_split_seed + outer_index,
        )
        inner_lambda_scores = {value: [] for value in protocol.transition_lambdas}
        for inner_index, (inner_train_subjects, inner_val_subjects) in enumerate(
            inner_splits, start=1
        ):
            inner_train = outer_train[
                outer_train["subject_std"].astype(str).isin(inner_train_subjects)
            ].reset_index(drop=True)
            inner_val = outer_train[
                outer_train["subject_std"].astype(str).isin(inner_val_subjects)
            ].reset_index(drop=True)
            per_seed = [
                score_emissions(
                    inner_train,
                    inner_val,
                    protocol,
                    seed + outer_index * 10_000 + inner_index * 100,
                )
                for seed in protocol.seeds
            ]
            inner_scores = ensemble_emissions(pd.concat(per_seed, ignore_index=True), len(protocol.seeds))
            for transition_lambda in protocol.transition_lambdas:
                prediction = select_causal_path(
                    inner_scores,
                    transition_lambda,
                    protocol.max_transition_bpm,
                    METHOD,
                )
                score = participant_equal_mae(prediction)
                inner_lambda_scores[transition_lambda].append(score)
                inner_rows.append(
                    {
                        "outer_fold": outer_index,
                        "inner_fold": inner_index,
                        "transition_lambda": transition_lambda,
                        "participant_equal_mae_bpm": score,
                    }
                )
        candidates = [
            (float(np.mean(values)), transition_lambda)
            for transition_lambda, values in inner_lambda_scores.items()
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected_inner_mae, selected_lambda = candidates[0]
        lambda_rows.append(
            {
                "outer_fold": outer_index,
                "selected_transition_lambda": selected_lambda,
                "inner_mean_participant_mae_bpm": selected_inner_mae,
            }
        )

        outer_per_seed = [
            score_emissions(
                outer_train,
                outer_test,
                protocol,
                seed + outer_index * 10_000,
            )
            for seed in protocol.seeds
        ]
        outer_scores = ensemble_emissions(
            pd.concat(outer_per_seed, ignore_index=True), len(protocol.seeds)
        )
        outer_scores["outer_fold"] = outer_index
        outer_score_rows.append(outer_scores)
        temporal = select_causal_path(
            outer_scores,
            selected_lambda,
            protocol.max_transition_bpm,
            METHOD,
        )
        temporal["outer_fold"] = outer_index
        prediction_rows.append(temporal)
        independent = select_causal_path(
            outer_scores,
            0.0,
            protocol.max_transition_bpm,
            INDEPENDENT,
        )
        independent["outer_fold"] = outer_index
        independent_rows.append(independent)
        print(
            f"outer={outer_index} lambda={selected_lambda:.3f} "
            f"inner_mae={selected_inner_mae:.4f}",
            flush=True,
        )

    split_frame = pd.DataFrame(split_rows)
    stack_splits = pd.read_csv(args.matched_stacking_dir / "outer_subject_splits.csv")
    compare = ["outer_fold", "train_subjects", "test_subjects"]
    if not split_frame[compare].equals(stack_splits[compare]):
        raise RuntimeError("candidate-path and comparator outer participant folds differ")

    temporal = pd.concat(prediction_rows, ignore_index=True)
    independent = pd.concat(independent_rows, ignore_index=True)
    if temporal["sample_id"].nunique() != 439 or independent["sample_id"].nunique() != 439:
        raise RuntimeError("outer predictions do not cover the canonical 439 windows")
    stack = pd.read_csv(
        args.matched_stacking_dir / "matched_stacking_evaluation_predictions.csv",
        low_memory=False,
    )
    comparators = [
        PRIMARY_COMPARATOR,
        STACKER_COMPARATOR,
        "matched_ridge_stacker",
        "deep_candidate_median",
        *stacking.ROUTES,
    ]
    stack = stack[stack["method"].astype(str).isin(comparators)].copy()
    keep = ["method", "sample_id", "subject_std", "gt_hr_bpm", "pred_hr_bpm", "abs_error_bpm"]
    evaluation = pd.concat(
        [temporal[keep], independent[keep], stack[keep]], ignore_index=True, sort=False
    )
    for method, frame in evaluation.groupby("method"):
        if frame["sample_id"].nunique() != 439:
            raise RuntimeError(f"method does not cover 439 windows: {method}")

    participant = participant_metrics(evaluation)
    metrics = aggregate_metrics(evaluation, participant)
    comparisons = [
        paired_bootstrap(
            participant,
            METHOD,
            comparator,
            protocol.bootstrap_draws,
            protocol.outer_split_seed + index,
        )
        for index, comparator in enumerate([INDEPENDENT, *comparators])
    ]

    split_frame.to_csv(args.output_dir / "outer_subject_splits.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(args.output_dir / "inner_lambda_screen.csv", index=False)
    pd.DataFrame(lambda_rows).to_csv(args.output_dir / "selected_transition_lambdas.csv", index=False)
    pd.concat(outer_score_rows, ignore_index=True).to_csv(
        args.output_dir / "outer_candidate_emission_scores.csv", index=False
    )
    temporal.to_csv(args.output_dir / "outer_temporal_predictions.csv", index=False)
    independent.to_csv(args.output_dir / "outer_independent_predictions.csv", index=False)
    evaluation.to_csv(args.output_dir / "outer_predictions_with_comparators.csv", index=False)
    participant.to_csv(args.output_dir / "participant_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    write_json(args.output_dir / "paired_participant_inference.json", comparisons)

    metric_map = metrics.set_index("method").to_dict("index")
    primary = next(item for item in comparisons if item["method_b"] == PRIMARY_COMPARATOR)
    temporal_effect = next(item for item in comparisons if item["method_b"] == INDEPENDENT)
    summary = {
        "task": TASK_ID,
        "generated_at_utc": utc_now(),
        "protocol": asdict(protocol),
        "input_pool": str(args.input_pool.resolve()),
        "input_sha256": sha256(args.input_pool),
        "matched_stacking_dir": str(args.matched_stacking_dir.resolve()),
        "n_subjects": 42,
        "n_windows": 439,
        "n_candidates": len(pool),
        "external_outcomes_accessed": False,
        "development_only": True,
        "method_metrics": metric_map[METHOD],
        "independent_emission_metrics": metric_map[INDEPENDENT],
        "primary_comparator_metrics": metric_map[PRIMARY_COMPARATOR],
        "causal_path_minus_independent": temporal_effect,
        "causal_path_minus_primary_comparator": primary,
        "selected_transition_lambdas": lambda_rows,
        "frozen_transition_lambda_candidate": float(
            np.median([row["selected_transition_lambda"] for row in lambda_rows])
        ),
        "source_files": {
            "run_v32_causal_candidate_path.py": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "screen_v31_tree_rankers.py": {
                "path": str((TOOLS / "screen_v31_tree_rankers.py").resolve()),
                "sha256": sha256(TOOLS / "screen_v31_tree_rankers.py"),
            },
            "run_v31_matched_stacking_baselines.py": {
                "path": str((TOOLS / "run_v31_matched_stacking_baselines.py").resolve()),
                "sha256": sha256(TOOLS / "run_v31_matched_stacking_baselines.py"),
            },
        },
        "reentry_gate": {
            "finite_outputs": bool(np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()),
            "internal_primary_ci_below_zero": bool(primary["ci95_high"] < 0.0),
            "internal_primary_sign_flip_p_below_0_05": bool(
                primary["paired_sign_flip_p_plus_one"] < 0.05
            ),
            "temporal_increment_ci_below_zero": bool(temporal_effect["ci95_high"] < 0.0),
            "temporal_increment_sign_flip_p_below_0_05": bool(
                temporal_effect["paired_sign_flip_p_plus_one"] < 0.05
            ),
            "external_confirmation_required": True,
        },
        "claim_boundary": (
            "Development evidence can select and freeze the V32 causal candidate-path architecture. "
            "It cannot establish transportability, calibrated safety or clinical utility."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    products = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    write_json(
        args.output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in products
            ],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
