#!/usr/bin/env python3
"""Strict nested comparison of source-preserving selection and direct stacking.

Every architecture decision is made inside each outer-training partition.  The
candidate selector and direct stackers use the same participant folds, route
predictions, prediction-only quality features, and development windows.  The
selector aggregates within-window candidate ranks across seeds and then emits
one observed candidate, so its reported HR retains a source identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import run_v31_matched_stacking_baselines as stacking
import screen_v31_tree_rankers as candidate


TASK_ID = "V31_NESTED_CANDIDATE_VS_STACKING"
PRIMARY_COMPARATOR = "matched_extra_trees_stacker"


@dataclass(frozen=True)
class Protocol:
    split_seed: int = 310718
    inner_split_seed: int = 510718
    outer_folds: int = 3
    inner_folds: int = 3
    seeds: tuple[int, ...] = (704, 1704, 2704)
    n_estimators: int = 260
    bootstrap_draws: int = 10000
    role: str = "development_only_strict_nested_participant_disjoint"


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


def make_regressor(seed: int, leaf: int, trees: int, jobs: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.75,
        random_state=seed,
        n_jobs=jobs,
    )


def make_classifier(seed: int, leaf: int, trees: int, jobs: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.75,
        class_weight="balanced",
        random_state=seed,
        n_jobs=jobs,
    )


def positive_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    if 1 not in model.classes_:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def score_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: dict[str, Any],
    seed: int,
    trees: int,
    jobs: int,
) -> pd.DataFrame:
    if bool(spec["relation"]):
        raise RuntimeError("this confirmatory control permits node-only features")
    x_train, x_test, _ = candidate.align_features(train, test, relation=False)
    error, rank, safe5 = candidate.target_values(train)
    weights = candidate.sample_weights(train)
    objective = str(spec["objective"])
    leaf = int(spec["leaf"])

    if objective == "error20":
        model = make_regressor(seed, leaf, trees, jobs)
        model.fit(x_train, np.clip(error, 0.0, 20.0), sample_weight=weights)
        score = model.predict(x_test)
        safe_probability = np.exp(-np.clip(score, 0.0, None) / 5.0)
    elif objective == "rank":
        model = make_regressor(seed, leaf, trees, jobs)
        model.fit(x_train, rank, sample_weight=weights)
        score = model.predict(x_test)
        safe_probability = 1.0 - np.clip(score, 0.0, 1.0)
    elif objective == "safe5":
        model = make_classifier(seed, leaf, trees, jobs)
        model.fit(x_train, safe5, sample_weight=weights)
        safe_probability = positive_probability(model, x_test)
        score = 1.0 - safe_probability
    elif objective == "hybrid":
        ranker = make_regressor(seed, leaf, trees, jobs)
        classifier = make_classifier(seed + 100_000, leaf, trees, jobs)
        ranker.fit(x_train, rank, sample_weight=weights)
        classifier.fit(x_train, safe5, sample_weight=weights)
        predicted_rank = np.clip(ranker.predict(x_test), 0.0, 1.0)
        safe_probability = positive_probability(classifier, x_test)
        score = predicted_rank + 0.5 * (1.0 - safe_probability)
    else:
        raise ValueError(objective)

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
    scored["selection_score"] = np.asarray(score, dtype=float)
    scored["predicted_safe5_probability"] = np.asarray(safe_probability, dtype=float)
    scored["within_window_rank"] = scored.groupby("sample_id")["selection_score"].rank(
        method="average", pct=True
    )
    return scored


def selected_prediction(scored: pd.DataFrame, method: str) -> pd.DataFrame:
    ordered = scored.sort_values(
        ["sample_id", "selection_score", "candidate_id"], kind="mergesort"
    )
    selected = ordered.groupby("sample_id", sort=False, as_index=False).head(1).copy()
    result = selected[
        [
            "sample_id",
            "subject_std",
            "gt_hr_bpm",
            "candidate_hr_bpm",
            "candidate_id",
            "source_type",
            "candidate_family",
            "candidate_model",
            "selection_score",
            "predicted_safe5_probability",
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
    return result


def participant_equal_mae(predictions: pd.DataFrame) -> float:
    return float(predictions.groupby("subject_std")["abs_error_bpm"].mean().mean())


def aggregate_seed_scores(scored: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "outer_fold",
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    aggregated = (
        scored.groupby(identity, as_index=False)
        .agg(
            ensemble_rank=("within_window_rank", "median"),
            ensemble_rank_mean=("within_window_rank", "mean"),
            n_seed_scores=("seed", "nunique"),
        )
    )
    if not aggregated["n_seed_scores"].eq(scored["seed"].nunique()).all():
        raise RuntimeError("candidate seed-score coverage mismatch")
    ordered = aggregated.sort_values(
        ["sample_id", "ensemble_rank", "ensemble_rank_mean", "candidate_id"],
        kind="mergesort",
    )
    selected = ordered.groupby("sample_id", sort=False, as_index=False).head(1).copy()
    result = selected.rename(
        columns={
            "candidate_hr_bpm": "pred_hr_bpm",
            "candidate_id": "selected_candidate_id",
            "source_type": "selected_source_type",
            "candidate_family": "selected_candidate_family",
            "candidate_model": "selected_candidate_model",
        }
    )
    result.insert(0, "method", "VitalsSight_candidate_aware_nested")
    result["abs_error_bpm"] = (result["pred_hr_bpm"] - result["gt_hr_bpm"]).abs()
    return result


def paired_bootstrap(
    participant: pd.DataFrame,
    method_a: str,
    method_b: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    wide = participant.pivot(index="subject_std", columns="method", values="mae_bpm")
    paired = wide[[method_a, method_b]].dropna()
    delta = (paired[method_a] - paired[method_b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        take = rng.integers(0, len(delta), len(delta))
        samples[index] = float(delta[take].mean())
    tail = min(int((samples >= 0.0).sum()), int((samples <= 0.0).sum()))
    p_value = min(1.0, 2.0 * (tail + 1.0) / (draws + 1.0))
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_subjects": int(len(delta)),
        "mean_delta_a_minus_b_bpm": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_two_sided_bootstrap_plus_one": float(p_value),
        "a_better_supported": bool(np.quantile(samples, 0.975) < 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--matched-stacking-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    protocol = Protocol(
        n_estimators=args.n_estimators,
        bootstrap_draws=args.bootstrap_draws,
    )
    pool = pd.read_csv(args.input_pool, low_memory=False)
    required = {
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_id",
        "candidate_hr_bpm",
        "candidate_abs_error",
        "source_type",
        "candidate_family",
        "candidate_model",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise RuntimeError(f"input missing required columns: {missing}")
    if pool["subject_std"].astype(str).nunique() != 42:
        raise RuntimeError("development participant identity mismatch")
    if pool["sample_id"].astype(str).nunique() != 439:
        raise RuntimeError("development window identity mismatch")
    if (pool.groupby("sample_id")["gt_hr_bpm"].nunique() != 1).any():
        raise RuntimeError("candidate-independent reference mismatch")

    subjects = sorted(pool["subject_std"].astype(str).unique())
    outer_splits = candidate.fixed_folds(subjects, protocol.outer_folds, protocol.split_seed)
    specs = [spec for spec in candidate.model_specs() if not bool(spec["relation"])]
    split_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    outer_scores: list[pd.DataFrame] = []
    seed_predictions: list[pd.DataFrame] = []

    for outer_index, (outer_train_subjects, outer_test_subjects) in enumerate(
        outer_splits, start=1
    ):
        outer_train = pool[
            pool["subject_std"].astype(str).isin(outer_train_subjects)
        ].reset_index(drop=True)
        outer_test = pool[
            pool["subject_std"].astype(str).isin(outer_test_subjects)
        ].reset_index(drop=True)
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
        for seed in protocol.seeds:
            config_scores = []
            for spec in specs:
                inner_scores = []
                for inner_index, (inner_train_subjects, inner_val_subjects) in enumerate(
                    inner_splits, start=1
                ):
                    inner_train = outer_train[
                        outer_train["subject_std"].astype(str).isin(inner_train_subjects)
                    ].reset_index(drop=True)
                    inner_val = outer_train[
                        outer_train["subject_std"].astype(str).isin(inner_val_subjects)
                    ].reset_index(drop=True)
                    scored = score_candidates(
                        inner_train,
                        inner_val,
                        spec,
                        seed + outer_index * 10000 + inner_index * 100,
                        protocol.n_estimators,
                        args.n_jobs,
                    )
                    prediction = selected_prediction(scored, str(spec["name"]))
                    score = participant_equal_mae(prediction)
                    inner_scores.append(score)
                    screen_rows.append(
                        {
                            "outer_fold": outer_index,
                            "seed": seed,
                            "config": spec["name"],
                            "inner_fold": inner_index,
                            "participant_equal_mae_bpm": score,
                        }
                    )
                config_scores.append((float(np.mean(inner_scores)), str(spec["name"]), spec))
            config_scores.sort(key=lambda value: (value[0], value[1]))
            selected_score, _, selected_spec = config_scores[0]
            selection_rows.append(
                {
                    "outer_fold": outer_index,
                    "seed": seed,
                    "selected_config": selected_spec["name"],
                    "inner_mean_participant_mae_bpm": selected_score,
                }
            )
            scored = score_candidates(
                outer_train,
                outer_test,
                selected_spec,
                seed + outer_index * 10000,
                protocol.n_estimators,
                args.n_jobs,
            )
            scored["outer_fold"] = outer_index
            scored["seed"] = seed
            scored["selected_config"] = selected_spec["name"]
            outer_scores.append(scored)
            per_seed = selected_prediction(scored, "VitalsSight_candidate_aware_nested")
            per_seed["outer_fold"] = outer_index
            per_seed["seed"] = seed
            per_seed["selected_config"] = selected_spec["name"]
            seed_predictions.append(per_seed)
        print(f"outer={outer_index} complete", flush=True)

    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(args.output_dir / "outer_subject_splits.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "inner_candidate_screen.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output_dir / "selected_configs.csv", index=False)
    score_frame = pd.concat(outer_scores, ignore_index=True)
    score_frame.to_csv(args.output_dir / "candidate_scores_per_seed.csv", index=False)
    per_seed_frame = pd.concat(seed_predictions, ignore_index=True)
    per_seed_frame.to_csv(args.output_dir / "candidate_predictions_per_seed.csv", index=False)
    nested = aggregate_seed_scores(score_frame)
    nested.to_csv(args.output_dir / "nested_candidate_predictions.csv", index=False)

    stack_split_path = args.matched_stacking_dir / "outer_subject_splits.csv"
    stack_splits = pd.read_csv(stack_split_path)
    compare_columns = ["outer_fold", "train_subjects", "test_subjects"]
    if not split_frame[compare_columns].equals(stack_splits[compare_columns]):
        raise RuntimeError("candidate and stacking outer participant folds differ")
    stack_eval_path = args.matched_stacking_dir / "matched_stacking_evaluation_predictions.csv"
    stack_eval = pd.read_csv(stack_eval_path, low_memory=False)
    stack_eval = stack_eval[
        stack_eval["method"].astype(str).ne("VitalsSight_candidate_aware")
    ].copy()
    if set(nested["sample_id"].astype(str)) != set(stack_eval["sample_id"].astype(str)):
        raise RuntimeError("candidate and stacking window identity mismatch")
    evaluation = pd.concat([nested, stack_eval], ignore_index=True, sort=False)
    evaluation.to_csv(args.output_dir / "nested_candidate_vs_stacking_predictions.csv", index=False)
    metrics, participant = stacking.metric_tables(evaluation)
    metrics.to_csv(args.output_dir / "nested_candidate_vs_stacking_metrics.csv", index=False)
    participant.to_csv(
        args.output_dir / "nested_candidate_vs_stacking_participant_metrics.csv", index=False
    )

    method_a = "VitalsSight_candidate_aware_nested"
    comparators = [
        PRIMARY_COMPARATOR,
        "matched_ridge_stacker",
        "deep_candidate_median",
        *stacking.ROUTES,
    ]
    comparisons = [
        paired_bootstrap(
            participant,
            method_a,
            comparator,
            protocol.bootstrap_draws,
            protocol.split_seed + index,
        )
        for index, comparator in enumerate(comparators)
    ]
    write_json(args.output_dir / "paired_participant_bootstrap.json", comparisons)

    observed = pool[
        [
            "sample_id",
            "candidate_id",
            "candidate_hr_bpm",
            "source_type",
            "candidate_model",
        ]
    ].drop_duplicates(["sample_id", "candidate_id"])
    observed_check = nested.merge(
        observed,
        left_on=["sample_id", "selected_candidate_id"],
        right_on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_observed"),
    )
    observed_match = bool(
        len(observed_check) == len(nested)
        and observed_check["candidate_id"].notna().all()
        and np.allclose(
            observed_check["pred_hr_bpm"].to_numpy(float),
            observed_check["candidate_hr_bpm"].to_numpy(float),
        )
        and observed_check["selected_source_type"].astype(str).eq(
            observed_check["source_type"].astype(str)
        ).all()
        and observed_check["selected_candidate_model"].astype(str).eq(
            observed_check["candidate_model"].astype(str)
        ).all()
    )
    feature_frame, feature_names = candidate.feature_frame(pool, relation=False)
    del feature_frame
    forbidden_features = [
        name
        for name in feature_names
        if any(token in name.lower() for token in candidate.FORBIDDEN_TOKENS)
    ]
    audit = {
        "passed": True,
        "n_subjects": int(nested["subject_std"].nunique()),
        "n_windows": int(nested["sample_id"].nunique()),
        "one_prediction_per_window": bool(not nested["sample_id"].duplicated().any()),
        "selected_candidate_identity_complete": bool(
            nested[
                [
                    "selected_candidate_id",
                    "selected_source_type",
                    "selected_candidate_model",
                ]
            ].notna().all().all()
        ),
        "all_selected_values_are_observed_candidates": observed_match,
        "candidate_seed_score_coverage_complete": bool(
            score_frame.groupby(["sample_id", "candidate_id"])["seed"]
            .nunique()
            .eq(len(protocol.seeds))
            .all()
        ),
        "outer_split_matches_stacking": True,
        "architecture_selection_uses_outer_train_only": True,
        "candidate_feature_count": len(feature_names),
        "forbidden_inference_features": forbidden_features,
        "external_outcomes_accessed": False,
    }
    if audit["n_subjects"] != 42 or audit["n_windows"] != 439:
        raise RuntimeError("nested candidate coverage audit failed")
    if (
        not audit["one_prediction_per_window"]
        or not audit["selected_candidate_identity_complete"]
        or not audit["all_selected_values_are_observed_candidates"]
        or not audit["candidate_seed_score_coverage_complete"]
        or audit["forbidden_inference_features"]
    ):
        raise RuntimeError("source-preserving prediction audit failed")
    write_json(args.output_dir / "integrity_audit.json", audit)

    summary = {
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "protocol": asdict(protocol),
        "input_pool": str(args.input_pool),
        "input_pool_sha256": sha256(args.input_pool),
        "matched_stacking_dir": str(args.matched_stacking_dir),
        "matched_stacking_summary_sha256": sha256(args.matched_stacking_dir / "summary.json"),
        "primary_comparator": PRIMARY_COMPARATOR,
        "candidate_config_count": len(specs),
        "candidate_seed_aggregation": "median within-window rank then observed-candidate selection",
        "metrics": metrics.to_dict("records"),
        "comparisons": comparisons,
        "integrity_audit": audit,
        "claim_boundary": (
            "Development-only strict nested participant-disjoint evidence. It tests whether "
            "source-preserving candidate selection adds value beyond direct route stacking; "
            "it does not establish external transport or clinical safety."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
