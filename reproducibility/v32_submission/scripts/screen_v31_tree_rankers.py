#!/usr/bin/env python3
"""Development-only screen for candidate-aware tree rankers.

All variants use participant-disjoint folds and prediction-time features only.
The screen is explicitly non-confirmatory and preserves every attempted model.
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


SEEDS = (704, 1704, 2704)
THRESHOLDS = (5.0, 8.0, 10.0, 15.0)

NODE_NUMERIC = [
    "candidate_hr_bpm",
    "adult_plausibility",
    "pred_peak_power_fraction",
    "pred_spectral_entropy",
    "pred_top2_power_ratio",
    "pred_autocorr_peak",
    "pred_signal_std",
    "pred_spectral_snr_db",
    "is_deep",
    "is_classical",
]

RELATION_NUMERIC = [
    "support_count",
    "full_support_count",
    "subwindow_support_count",
    "top1_support_count",
    "full_top1_support_count",
    "pos_chrom_count",
    "green_pbv_count",
    "ica_lgi_count",
    "mean_power_fraction",
    "max_power_fraction",
    "sum_power_fraction",
    "upper_alt_support",
    "upper_alt_pos_chrom",
    "upper_phys_support",
    "upper_phys_pos_chrom",
    "lower_phys_support",
    "lower_phys_pos_chrom",
    "double_harmonic_support",
    "half_harmonic_support",
    "dist_to_group_median_hr",
    "dist_to_group_mean_hr",
    "agreement5_frac",
    "agreement10_frac",
    "agreement20_frac",
    "support_rank_pct",
    "harmonic_risk",
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
]

FORBIDDEN_TOKENS = (
    "gt_hr",
    "reference_hr",
    "abs_error",
    "unsafe",
    "macc",
    "rank_score",
    "deep_snr",
    "mean_snr_proxy",
    "dist_to_t150",
    "t150_",
    "t157_",
)


@dataclass(frozen=True)
class Protocol:
    split_seed: int = 310718
    n_folds: int = 3
    seeds: tuple[int, ...] = SEEDS
    n_estimators: int = 260
    bootstrap_draws: int = 10000
    role: str = "development_only_non_confirmatory"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_folds(subjects: list[str], n_folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    rng = np.random.default_rng(seed)
    values = np.asarray(sorted(subjects), dtype=object)
    rng.shuffle(values)
    chunks = np.array_split(values, n_folds)
    folds = []
    for chunk in chunks:
        test = sorted(map(str, chunk.tolist()))
        train = sorted(subject for subject in subjects if subject not in set(test))
        folds.append((train, test))
    return folds


def model_specs() -> list[dict[str, Any]]:
    specs = []
    for relation in (False, True):
        prefix = "relation" if relation else "node_only"
        for leaf in (3, 7, 12):
            for objective in ("error20", "rank", "safe5", "hybrid"):
                specs.append(
                    {
                        "name": f"{prefix}_{objective}_leaf{leaf}",
                        "relation": relation,
                        "objective": objective,
                        "leaf": leaf,
                    }
                )
    return specs


def feature_frame(pool: pd.DataFrame, relation: bool) -> tuple[pd.DataFrame, list[str]]:
    numeric = NODE_NUMERIC + (RELATION_NUMERIC if relation else [])
    out = pd.DataFrame(index=pool.index)
    for column in numeric:
        values = pd.to_numeric(pool[column], errors="coerce") if column in pool else np.nan
        if isinstance(values, pd.Series):
            values = values.replace([np.inf, -np.inf], np.nan)
        out[column] = values
        out[f"missing__{column}"] = values.isna().astype(float)
    categorical = pd.get_dummies(
        pool[["source_type", "candidate_family", "candidate_model"]].astype(str),
        prefix=["source", "family", "route"],
        dtype=float,
    )
    out = pd.concat([out, categorical], axis=1).replace([np.inf, -np.inf], np.nan)
    forbidden = [
        column for column in out.columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"forbidden inference features: {forbidden}")
    return out, sorted(out.columns)


def align_features(
    train: pd.DataFrame, test: pd.DataFrame, relation: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_train, train_columns = feature_frame(train, relation)
    x_test, test_columns = feature_frame(test, relation)
    columns = sorted(set(train_columns) | set(test_columns))
    x_train = x_train.reindex(columns=columns)
    x_test = x_test.reindex(columns=columns)
    medians = x_train.median(numeric_only=True).reindex(columns).fillna(0.0)
    return (
        x_train.fillna(medians).fillna(0.0).to_numpy(float),
        x_test.fillna(medians).fillna(0.0).to_numpy(float),
        columns,
    )


def sample_weights(pool: pd.DataFrame) -> np.ndarray:
    group_size = pool.groupby("sample_id")["sample_id"].transform("size").to_numpy(float)
    participant_windows = pool.groupby("subject_std")["sample_id"].transform("nunique").to_numpy(float)
    return 1.0 / np.clip(group_size * participant_windows, 1.0, None)


def target_values(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error = pd.to_numeric(pool["candidate_abs_error"], errors="coerce").fillna(100.0)
    rank = pool.assign(_error=error).groupby("sample_id")["_error"].rank(method="average", pct=True)
    safe5 = (error <= 5.0).astype(int)
    return error.to_numpy(float), rank.to_numpy(float), safe5.to_numpy(int)


def make_regressor(seed: int, leaf: int, trees: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.75,
        random_state=seed,
        n_jobs=-1,
    )


def make_classifier(seed: int, leaf: int, trees: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.75,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def positive_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    if 1 not in model.classes_:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: dict[str, Any],
    seed: int,
    trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, x_test, columns = align_features(train, test, bool(spec["relation"]))
    error, rank, safe5 = target_values(train)
    weights = sample_weights(train)
    objective = str(spec["objective"])
    leaf = int(spec["leaf"])
    importance_parts = []

    if objective == "error20":
        model = make_regressor(seed, leaf, trees)
        model.fit(x_train, np.clip(error, 0.0, 20.0), sample_weight=weights)
        score = model.predict(x_test)
        safe_probability = np.exp(-np.clip(score, 0.0, None) / 5.0)
        importance_parts.append(model.feature_importances_)
    elif objective == "rank":
        model = make_regressor(seed, leaf, trees)
        model.fit(x_train, rank, sample_weight=weights)
        score = model.predict(x_test)
        safe_probability = 1.0 - np.clip(score, 0.0, 1.0)
        importance_parts.append(model.feature_importances_)
    elif objective == "safe5":
        classifier = make_classifier(seed, leaf, trees)
        classifier.fit(x_train, safe5, sample_weight=weights)
        safe_probability = positive_probability(classifier, x_test)
        score = 1.0 - safe_probability
        importance_parts.append(classifier.feature_importances_)
    elif objective == "hybrid":
        ranker = make_regressor(seed, leaf, trees)
        classifier = make_classifier(seed + 100_000, leaf, trees)
        ranker.fit(x_train, rank, sample_weight=weights)
        classifier.fit(x_train, safe5, sample_weight=weights)
        predicted_rank = np.clip(ranker.predict(x_test), 0.0, 1.0)
        safe_probability = positive_probability(classifier, x_test)
        score = predicted_rank + 0.5 * (1.0 - safe_probability)
        importance_parts.extend([ranker.feature_importances_, classifier.feature_importances_])
    else:
        raise ValueError(objective)

    candidates = test.copy().reset_index(drop=True)
    candidates["selection_score"] = score
    candidates["predicted_safe5_probability"] = safe_probability
    selected = candidates.loc[candidates.groupby("sample_id")["selection_score"].idxmin()].copy()
    predictions = pd.DataFrame(
        {
            "method": spec["name"],
            "sample_id": selected["sample_id"].astype(str),
            "subject_std": selected["subject_std"].astype(str),
            "gt_hr_bpm": pd.to_numeric(selected["gt_hr_bpm"], errors="coerce"),
            "pred_hr_bpm": pd.to_numeric(selected["candidate_hr_bpm"], errors="coerce"),
            "selected_candidate_id": selected["candidate_id"].astype(str),
            "selected_source_type": selected["source_type"].astype(str),
            "selected_candidate_model": selected["candidate_model"].astype(str),
            "selection_score": pd.to_numeric(selected["selection_score"], errors="coerce"),
            "predicted_safe5_probability": pd.to_numeric(
                selected["predicted_safe5_probability"], errors="coerce"
            ),
        }
    )
    predictions["abs_error_bpm"] = (
        predictions["pred_hr_bpm"] - predictions["gt_hr_bpm"]
    ).abs()
    importance = np.mean(np.stack(importance_parts), axis=0)
    importance_frame = pd.DataFrame(
        {"method": spec["name"], "feature": columns, "importance": importance}
    ).sort_values("importance", ascending=False)
    return predictions, importance_frame


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_rows = []
    method_rows = []
    for method, group in predictions.groupby("method", sort=False):
        for subject, subject_group in group.groupby("subject_std", sort=False):
            error = subject_group["abs_error_bpm"].to_numpy(float)
            participant_rows.append(
                {
                    "method": method,
                    "subject_std": subject,
                    "n_windows": len(subject_group),
                    "mae_bpm": float(error.mean()),
                    "rmse_bpm": float(np.sqrt(np.mean(np.square(error)))),
                    "within_10bpm": float((error <= 10.0).mean()),
                }
            )
        error = group["abs_error_bpm"].to_numpy(float)
        method_rows.append(
            {
                "method": method,
                "n_windows": len(group),
                "n_subjects": group["subject_std"].nunique(),
                "window_mae_bpm": float(error.mean()),
                "window_rmse_bpm": float(np.sqrt(np.mean(np.square(error)))),
                "median_abs_error_bpm": float(np.median(error)),
                "p90_abs_error_bpm": float(np.quantile(error, 0.9)),
                **{f"within_{int(value)}bpm": float((error <= value).mean()) for value in THRESHOLDS},
            }
        )
    participant = pd.DataFrame(participant_rows)
    subject_summary = participant.groupby("method", as_index=False).agg(
        participant_equal_mae_bpm=("mae_bpm", "mean"),
        participant_equal_rmse_bpm=("rmse_bpm", "mean"),
        participant_equal_within10=("within_10bpm", "mean"),
    )
    methods = pd.DataFrame(method_rows).merge(subject_summary, on="method")
    return methods.sort_values("participant_equal_mae_bpm"), participant


def paired_bootstrap(
    participant: pd.DataFrame,
    relation_method: str,
    node_method: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    wide = participant.pivot(index="subject_std", columns="method", values="mae_bpm")
    paired = wide[[relation_method, node_method]].dropna()
    delta = (paired[relation_method] - paired[node_method]).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(delta), len(delta))
        samples[draw] = float(delta[indices].mean())
    return {
        "relation_method": relation_method,
        "node_method": node_method,
        "n_subjects": len(delta),
        "mean_delta_relation_minus_node_bpm": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_two_sided_bootstrap": float(2.0 * min((samples >= 0).mean(), (samples <= 0).mean())),
        "relation_supported": bool(np.quantile(samples, 0.975) < 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    protocol = Protocol(
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        bootstrap_draws=args.bootstrap_draws,
    )

    pool = pd.read_csv(args.input_pool, low_memory=False)
    required = {
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_hr_bpm",
        "candidate_abs_error",
        "source_type",
        "candidate_family",
        "candidate_model",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise RuntimeError(f"input missing required columns: {missing}")
    if pool["subject_std"].astype(str).nunique() != 42 or pool["sample_id"].astype(str).nunique() != 439:
        raise RuntimeError("development-pool identity mismatch")
    if (pool.groupby("sample_id")["gt_hr_bpm"].nunique() != 1).any():
        raise RuntimeError("candidate-independent reference mismatch")

    subjects = sorted(pool["subject_std"].astype(str).unique())
    folds = fixed_folds(subjects, protocol.n_folds, protocol.split_seed)
    split_rows = []
    predictions = []
    importances = []
    specs = model_specs()
    for fold_index, (train_subjects, test_subjects) in enumerate(folds, start=1):
        train = pool[pool["subject_std"].astype(str).isin(train_subjects)].reset_index(drop=True)
        test = pool[pool["subject_std"].astype(str).isin(test_subjects)].reset_index(drop=True)
        split_rows.append(
            {
                "fold": fold_index,
                "train_subjects": ";".join(train_subjects),
                "test_subjects": ";".join(test_subjects),
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
            }
        )
        for seed in protocol.seeds:
            for spec in specs:
                prediction, importance = fit_predict(
                    train, test, spec, seed, protocol.n_estimators
                )
                prediction["fold"] = fold_index
                prediction["seed"] = seed
                predictions.append(prediction)
                importance["fold"] = fold_index
                importance["seed"] = seed
                importances.append(importance)

    pd.DataFrame(split_rows).to_csv(args.output_dir / "subject_splits.csv", index=False)
    per_seed = pd.concat(predictions, ignore_index=True)
    per_seed.to_csv(args.output_dir / "predictions_per_seed.csv", index=False)
    ensemble = (
        per_seed.groupby(["method", "sample_id", "subject_std", "gt_hr_bpm"], as_index=False)
        .agg(
            pred_hr_bpm=("pred_hr_bpm", "median"),
            n_seed_predictions=("pred_hr_bpm", "size"),
        )
    )
    ensemble["abs_error_bpm"] = (ensemble["pred_hr_bpm"] - ensemble["gt_hr_bpm"]).abs()
    ensemble.to_csv(args.output_dir / "predictions_seed_ensemble.csv", index=False)
    methods, participants = metric_tables(ensemble)
    if not methods["n_windows"].eq(439).all():
        raise RuntimeError("non-canonical window count in method screen")
    methods.to_csv(args.output_dir / "model_screen_metrics.csv", index=False)
    participants.to_csv(args.output_dir / "participant_metrics.csv", index=False)

    importance = pd.concat(importances, ignore_index=True)
    importance.to_csv(args.output_dir / "feature_importance_per_fold_seed.csv", index=False)
    (
        importance.groupby(["method", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["method", "importance"], ascending=[True, False])
        .to_csv(args.output_dir / "feature_importance_summary.csv", index=False)
    )

    pair_rows = []
    for leaf in (3, 7, 12):
        for objective in ("error20", "rank", "safe5", "hybrid"):
            pair_rows.append(
                paired_bootstrap(
                    participants,
                    f"relation_{objective}_leaf{leaf}",
                    f"node_only_{objective}_leaf{leaf}",
                    protocol.bootstrap_draws,
                    protocol.split_seed + leaf * 100 + len(objective),
                )
            )
    pair_table = pd.DataFrame(pair_rows).sort_values("mean_delta_relation_minus_node_bpm")
    pair_table.to_csv(args.output_dir / "relation_vs_node_bootstrap.csv", index=False)

    best_relation = methods[methods["method"].str.startswith("relation_")].iloc[0].to_dict()
    matched_node = str(best_relation["method"]).replace("relation_", "node_only_", 1)
    matched_pair = pair_table[pair_table["relation_method"].eq(best_relation["method"])]
    summary = {
        "task": "V31_TREE_RANKER_DEVELOPMENT_SCREEN",
        "generated_at_utc": utc_now(),
        "protocol": asdict(protocol),
        "input_pool": str(args.input_pool),
        "input_sha256": sha256(args.input_pool),
        "subjects": int(pool["subject_std"].nunique()),
        "windows": int(pool["sample_id"].nunique()),
        "candidates": int(len(pool)),
        "attempted_variants": len(specs),
        "all_variants_preserved": True,
        "external_outcomes_accessed": False,
        "best_development_method": methods.iloc[0].to_dict(),
        "best_relation_method": best_relation,
        "matched_node_method": matched_node,
        "matched_relation_result": matched_pair.to_dict("records"),
        "claim_boundary": (
            "Development-only architecture screening. It cannot establish a confirmatory "
            "relation contribution or external generalization."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
