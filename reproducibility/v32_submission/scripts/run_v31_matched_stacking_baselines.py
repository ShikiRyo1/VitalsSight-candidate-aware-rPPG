#!/usr/bin/env python3
"""Matched-label-budget stacking controls for VitalsSight candidate selection.

The controls receive the same five route predictions and prediction-only route
quality features under the same participant folds. They directly regress HR and
do not select or preserve a candidate identity. This isolates whether a
source-preserving candidate representation adds value beyond ordinary stacking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TASK_ID = "V31_MATCHED_STACKING_BASELINES"
ROUTES = ("DeepPhys", "EfficientPhys", "TSCAN", "PhysFormer", "RhythmFormer")
QUALITY = (
    "pred_peak_power_fraction",
    "pred_spectral_entropy",
    "pred_top2_power_ratio",
    "pred_autocorr_peak",
    "pred_signal_std",
    "pred_spectral_snr_db",
)
FORBIDDEN_FEATURE_TOKENS = (
    "gt_hr",
    "reference_hr",
    "abs_error",
    "unsafe",
    "macc",
    "label",
)


@dataclass(frozen=True)
class Protocol:
    split_seed: int = 310718
    inner_split_seed: int = 510718
    outer_folds: int = 3
    inner_folds: int = 3
    seeds: tuple[int, ...] = (704, 1704, 2704)
    n_estimators: int = 260
    bootstrap_draws: int = 10000
    role: str = "development_only_participant_disjoint_matched_label_budget"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixed_folds(subjects: list[str], count: int, seed: int) -> list[tuple[list[str], list[str]]]:
    values = np.asarray(sorted(subjects), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    chunks = np.array_split(values, count)
    all_subjects = set(map(str, values.tolist()))
    return [
        (sorted(all_subjects - set(map(str, chunk.tolist()))), sorted(map(str, chunk.tolist())))
        for chunk in chunks
    ]


def build_feature_table(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    deep = pool[pool["source_type"].astype(str).eq("deep")].copy()
    keys = ["sample_id", "subject_std", "gt_hr_bpm"]
    if deep.duplicated(["sample_id", "candidate_model"]).any():
        raise RuntimeError("duplicate deep route within a window")
    if set(deep["candidate_model"].astype(str).unique()) != set(ROUTES):
        raise RuntimeError("deep route identity mismatch")

    pieces = []
    value_columns = ["candidate_hr_bpm", *QUALITY]
    for route in ROUTES:
        route_rows = deep[deep["candidate_model"].astype(str).eq(route)][keys + value_columns].copy()
        rename = {
            column: f"{route}__{'hr_bpm' if column == 'candidate_hr_bpm' else column}"
            for column in value_columns
        }
        pieces.append(route_rows.rename(columns=rename))
    table = pieces[0]
    for piece in pieces[1:]:
        table = table.merge(piece, on=keys, how="inner", validate="one_to_one")
    if table["sample_id"].nunique() != pool["sample_id"].nunique():
        raise RuntimeError("fixed-route stack does not cover every development window")

    hr_columns = [f"{route}__hr_bpm" for route in ROUTES]
    route_hr = table[hr_columns].apply(pd.to_numeric, errors="coerce")
    table["route_hr_mean"] = route_hr.mean(axis=1)
    table["route_hr_median"] = route_hr.median(axis=1)
    table["route_hr_std"] = route_hr.std(axis=1, ddof=0)
    table["route_hr_min"] = route_hr.min(axis=1)
    table["route_hr_max"] = route_hr.max(axis=1)
    table["route_hr_range"] = table["route_hr_max"] - table["route_hr_min"]
    for quality in QUALITY:
        columns = [f"{route}__{quality}" for route in ROUTES]
        values = table[columns].apply(pd.to_numeric, errors="coerce")
        table[f"route_quality_mean__{quality}"] = values.mean(axis=1)
        table[f"route_quality_std__{quality}"] = values.std(axis=1, ddof=0)

    features = sorted(column for column in table.columns if column not in keys)
    forbidden = [
        feature
        for feature in features
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"forbidden inference features: {forbidden}")
    return table, features


def model_specs() -> list[dict[str, Any]]:
    specs = [
        {"name": f"ridge_alpha{alpha:g}", "family": "ridge", "alpha": alpha}
        for alpha in (0.1, 1.0, 10.0, 100.0)
    ]
    specs.extend(
        {
            "name": f"extra_trees_leaf{leaf}_mf{int(max_features * 100)}",
            "family": "extra_trees",
            "leaf": leaf,
            "max_features": max_features,
        }
        for leaf in (3, 7, 12)
        for max_features in (0.75, 1.0)
    )
    return specs


def participant_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("subject_std")["sample_id"].transform("count").to_numpy(float)
    return 1.0 / np.clip(counts, 1.0, None)


def prepare_xy(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train_x = train[features].apply(pd.to_numeric, errors="coerce")
    test_x = test[features].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median().fillna(0.0)
    x_train = train_x.fillna(medians).fillna(0.0).to_numpy(float)
    x_test = test_x.fillna(medians).fillna(0.0).to_numpy(float)
    y_train = pd.to_numeric(train["gt_hr_bpm"], errors="coerce").to_numpy(float)
    if not np.isfinite(y_train).all():
        raise RuntimeError("non-finite training targets")
    return x_train, x_test, y_train, medians.to_dict()


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    spec: dict[str, Any],
    seed: int,
    trees: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train, x_test, y_train, medians = prepare_xy(train, test, features)
    weights = participant_sample_weights(train)
    if spec["family"] == "ridge":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = Ridge(alpha=float(spec["alpha"]))
        model.fit(x_train_scaled, y_train, sample_weight=weights)
        prediction = model.predict(x_test_scaled)
        artifact = {
            "family": "ridge",
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "medians": medians,
        }
    elif spec["family"] == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=int(spec["leaf"]),
            max_features=float(spec["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        prediction = model.predict(x_test)
        artifact = {
            "family": "extra_trees",
            "feature_importance": model.feature_importances_.tolist(),
            "medians": medians,
        }
    else:
        raise ValueError(spec["family"])
    return np.clip(prediction, 40.0, 200.0), artifact


def participant_equal_mae(frame: pd.DataFrame) -> float:
    return float(frame.groupby("subject_std")["abs_error_bpm"].mean().mean())


def prediction_frame(method: str, test: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    result = test[["sample_id", "subject_std", "gt_hr_bpm"]].copy()
    result.insert(0, "method", method)
    result["pred_hr_bpm"] = values
    result["abs_error_bpm"] = (
        pd.to_numeric(result["pred_hr_bpm"], errors="coerce")
        - pd.to_numeric(result["gt_hr_bpm"], errors="coerce")
    ).abs()
    return result


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = (
        predictions.groupby(["method", "subject_std"], as_index=False)
        .agg(
            n_windows=("sample_id", "count"),
            mae_bpm=("abs_error_bpm", "mean"),
            rmse_bpm=("abs_error_bpm", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            within10=("abs_error_bpm", lambda values: float((np.asarray(values) <= 10.0).mean())),
        )
    )
    rows = []
    for method, group in predictions.groupby("method", sort=False):
        error = pd.to_numeric(group["abs_error_bpm"], errors="coerce").to_numpy(float)
        subject = participants[participants["method"].eq(method)]
        rows.append(
            {
                "method": method,
                "n_windows": int(len(group)),
                "n_subjects": int(group["subject_std"].nunique()),
                "window_mae_bpm": float(error.mean()),
                "window_rmse_bpm": float(np.sqrt(np.mean(np.square(error)))),
                "within5": float((error <= 5.0).mean()),
                "within10": float((error <= 10.0).mean()),
                "unsafe_gt10": float((error > 10.0).mean()),
                "p90_abs_error_bpm": float(np.quantile(error, 0.9)),
                "participant_equal_mae_bpm": float(subject["mae_bpm"].mean()),
                "participant_equal_rmse_bpm": float(subject["rmse_bpm"].mean()),
                "participant_equal_within10": float(subject["within10"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("participant_equal_mae_bpm"), participants


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
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_subjects": int(len(delta)),
        "mean_delta_a_minus_b_bpm": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_two_sided_bootstrap": float(
            2.0 * min((samples >= 0.0).mean(), (samples <= 0.0).mean())
        ),
        "a_better_supported": bool(np.quantile(samples, 0.975) < 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--candidate-method", default="node_only_error20_leaf3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    protocol = Protocol(n_estimators=args.n_estimators, bootstrap_draws=args.bootstrap_draws)
    pool = pd.read_csv(args.input_pool, low_memory=False)
    table, features = build_feature_table(pool)
    if table["subject_std"].nunique() != 42 or table["sample_id"].nunique() != 439:
        raise RuntimeError("development identity mismatch")
    table.to_csv(args.output_dir / "matched_stacking_feature_table.csv", index=False)
    write_json(
        args.output_dir / "feature_contract.json",
        {
            "features": features,
            "routes": list(ROUTES),
            "prediction_only_quality": list(QUALITY),
            "forbidden_tokens": list(FORBIDDEN_FEATURE_TOKENS),
            "reference_fields_are_targets_only": True,
            "candidate_identity_preserved_in_output": False,
            "passed": True,
        },
    )

    subjects = sorted(table["subject_std"].astype(str).unique())
    outer_splits = fixed_folds(subjects, protocol.outer_folds, protocol.split_seed)
    specs = model_specs()
    predictions = []
    screen_rows = []
    selections = []
    split_rows = []
    artifacts: dict[str, Any] = {}
    for outer_index, (outer_train_subjects, outer_test_subjects) in enumerate(outer_splits, start=1):
        outer_train = table[table["subject_std"].astype(str).isin(outer_train_subjects)].copy()
        outer_test = table[table["subject_std"].astype(str).isin(outer_test_subjects)].copy()
        split_rows.append(
            {
                "outer_fold": outer_index,
                "train_subjects": ";".join(outer_train_subjects),
                "test_subjects": ";".join(outer_test_subjects),
            }
        )
        inner_splits = fixed_folds(
            outer_train_subjects, protocol.inner_folds, protocol.inner_split_seed + outer_index
        )
        for seed in protocol.seeds:
            for family in ("ridge", "extra_trees"):
                family_specs = [spec for spec in specs if spec["family"] == family]
                scores = []
                for spec in family_specs:
                    fold_scores = []
                    for inner_index, (inner_train_subjects, inner_val_subjects) in enumerate(
                        inner_splits, start=1
                    ):
                        inner_train = outer_train[
                            outer_train["subject_std"].astype(str).isin(inner_train_subjects)
                        ].copy()
                        inner_val = outer_train[
                            outer_train["subject_std"].astype(str).isin(inner_val_subjects)
                        ].copy()
                        values, _ = fit_predict(
                            inner_train,
                            inner_val,
                            features,
                            spec,
                            seed + outer_index * 10000 + inner_index * 100,
                            protocol.n_estimators,
                        )
                        pred = prediction_frame(spec["name"], inner_val, values)
                        score = participant_equal_mae(pred)
                        fold_scores.append(score)
                        screen_rows.append(
                            {
                                "outer_fold": outer_index,
                                "seed": seed,
                                "family": family,
                                "config": spec["name"],
                                "inner_fold": inner_index,
                                "participant_equal_mae_bpm": score,
                            }
                        )
                    scores.append((float(np.mean(fold_scores)), spec["name"], spec))
                scores.sort(key=lambda value: (value[0], value[1]))
                selected_score, _, selected = scores[0]
                selections.append(
                    {
                        "outer_fold": outer_index,
                        "seed": seed,
                        "family": family,
                        "selected_config": selected["name"],
                        "inner_mean_participant_mae_bpm": selected_score,
                    }
                )
                values, artifact = fit_predict(
                    outer_train,
                    outer_test,
                    features,
                    selected,
                    seed + outer_index * 10000,
                    protocol.n_estimators,
                )
                method = f"matched_{family}_stacker"
                pred = prediction_frame(method, outer_test, values)
                pred["outer_fold"] = outer_index
                pred["seed"] = seed
                pred["selected_config"] = selected["name"]
                predictions.append(pred)
                artifacts[f"outer{outer_index}_seed{seed}_{family}"] = {
                    "spec": selected,
                    "artifact": artifact,
                }
        print(f"outer={outer_index} complete", flush=True)

    pd.DataFrame(split_rows).to_csv(args.output_dir / "outer_subject_splits.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "inner_model_screen.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output_dir / "selected_configs.csv", index=False)
    write_json(args.output_dir / "fitted_model_artifacts.json", artifacts)
    per_seed = pd.concat(predictions, ignore_index=True)
    per_seed.to_csv(args.output_dir / "stacking_predictions_per_seed.csv", index=False)
    ensemble = (
        per_seed.groupby(["method", "sample_id", "subject_std", "gt_hr_bpm"], as_index=False)
        .agg(pred_hr_bpm=("pred_hr_bpm", "median"), n_seed_predictions=("pred_hr_bpm", "size"))
    )
    ensemble["abs_error_bpm"] = (ensemble["pred_hr_bpm"] - ensemble["gt_hr_bpm"]).abs()

    candidate_all = pd.read_csv(args.candidate_predictions, low_memory=False)
    candidate = candidate_all[candidate_all["method"].astype(str).eq(args.candidate_method)].copy()
    if candidate["sample_id"].nunique() != 439:
        raise RuntimeError("candidate comparator does not cover 439 windows")
    candidate = candidate[
        ["sample_id", "subject_std", "gt_hr_bpm", "pred_hr_bpm", "abs_error_bpm"]
    ].copy()
    candidate.insert(0, "method", "VitalsSight_candidate_aware")

    deep = pool[pool["source_type"].astype(str).eq("deep")].copy()
    static = []
    for route in ROUTES:
        route_rows = deep[deep["candidate_model"].astype(str).eq(route)][
            ["sample_id", "subject_std", "gt_hr_bpm", "candidate_hr_bpm"]
        ].copy()
        route_rows = route_rows.rename(columns={"candidate_hr_bpm": "pred_hr_bpm"})
        route_rows.insert(0, "method", route)
        route_rows["abs_error_bpm"] = (route_rows["pred_hr_bpm"] - route_rows["gt_hr_bpm"]).abs()
        static.append(route_rows)
    median = deep.groupby(["sample_id", "subject_std", "gt_hr_bpm"], as_index=False)[
        "candidate_hr_bpm"
    ].median().rename(columns={"candidate_hr_bpm": "pred_hr_bpm"})
    median.insert(0, "method", "deep_candidate_median")
    median["abs_error_bpm"] = (median["pred_hr_bpm"] - median["gt_hr_bpm"]).abs()
    evaluation = pd.concat([ensemble, candidate, median, *static], ignore_index=True, sort=False)
    evaluation.to_csv(args.output_dir / "matched_stacking_evaluation_predictions.csv", index=False)
    metrics, participant = metric_tables(evaluation)
    metrics.to_csv(args.output_dir / "matched_stacking_metrics.csv", index=False)
    participant.to_csv(args.output_dir / "matched_stacking_participant_metrics.csv", index=False)

    comparisons = []
    comparators = [
        "matched_extra_trees_stacker",
        "matched_ridge_stacker",
        "deep_candidate_median",
        *ROUTES,
    ]
    for index, comparator in enumerate(comparators):
        comparisons.append(
            paired_bootstrap(
                participant,
                "VitalsSight_candidate_aware",
                comparator,
                protocol.bootstrap_draws,
                protocol.split_seed + index,
            )
        )
    write_json(args.output_dir / "paired_participant_bootstrap.json", comparisons)
    summary = {
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "protocol": asdict(protocol),
        "input_pool": str(args.input_pool),
        "input_pool_sha256": sha256(args.input_pool),
        "candidate_predictions": str(args.candidate_predictions),
        "candidate_predictions_sha256": sha256(args.candidate_predictions),
        "candidate_method": args.candidate_method,
        "n_features": len(features),
        "n_subjects": int(table["subject_std"].nunique()),
        "n_windows": int(table["sample_id"].nunique()),
        "metrics": metrics.to_dict("records"),
        "comparisons": comparisons,
        "external_outcomes_accessed": False,
        "claim_boundary": (
            "Development-only matched-label-budget evidence. The stackers use fixed route columns "
            "and prediction-only quality but do not preserve a selected candidate identity."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
