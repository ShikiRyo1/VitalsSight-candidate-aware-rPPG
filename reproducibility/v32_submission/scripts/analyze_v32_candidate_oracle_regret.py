#!/usr/bin/env python3
"""Diagnose whether VitalsSight errors arise from the candidate pool or selector.

This is a development-only audit. Oracle values use reference HR and therefore
must never enter inference or be reported as deployable performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def participant_equal(frame: pd.DataFrame, error_col: str) -> float:
    return float(frame.groupby("subject_std", sort=False)[error_col].mean().mean())


def bootstrap_participant_gap(
    participant: pd.DataFrame, draws: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    gaps = participant["selector_minus_oracle_mae_bpm"].to_numpy(float)
    sampled = rng.choice(gaps, size=(draws, len(gaps)), replace=True).mean(axis=1)
    return {
        "mean_gap_bpm": float(gaps.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", required=True, type=Path)
    parser.add_argument("--selector-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=320719)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    scores = pd.read_csv(args.candidate_scores)
    selected = pd.read_csv(args.selector_predictions)
    required_scores = {
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
        "seed",
    }
    required_selected = {
        "sample_id",
        "subject_std",
        "gt_hr_bpm",
        "selected_candidate_id",
        "pred_hr_bpm",
        "selected_source_type",
        "selected_candidate_family",
        "selected_candidate_model",
    }
    missing_scores = sorted(required_scores - set(scores.columns))
    missing_selected = sorted(required_selected - set(selected.columns))
    if missing_scores or missing_selected:
        raise ValueError(
            f"Missing columns: scores={missing_scores}, selected={missing_selected}"
        )

    identity_cols = [
        "sample_id",
        "candidate_id",
        "subject_std",
        "gt_hr_bpm",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    candidates = scores[identity_cols].drop_duplicates().copy()
    per_candidate_counts = scores.groupby(["sample_id", "candidate_id"])["seed"].nunique()
    candidates["candidate_abs_error_bpm"] = (
        candidates["candidate_hr_bpm"] - candidates["gt_hr_bpm"]
    ).abs()

    sample_group = candidates.groupby("sample_id", sort=False)
    sample_oracle = sample_group.agg(
        subject_std=("subject_std", "first"),
        gt_hr_bpm=("gt_hr_bpm", "first"),
        candidate_count=("candidate_id", "nunique"),
        oracle_abs_error_bpm=("candidate_abs_error_bpm", "min"),
    ).reset_index()

    best_rows = candidates.loc[
        candidates.groupby("sample_id")["candidate_abs_error_bpm"].idxmin(),
        [
            "sample_id",
            "candidate_id",
            "candidate_hr_bpm",
            "source_type",
            "candidate_family",
            "candidate_model",
        ],
    ].rename(
        columns={
            "candidate_id": "oracle_candidate_id",
            "candidate_hr_bpm": "oracle_hr_bpm",
            "source_type": "oracle_source_type",
            "candidate_family": "oracle_candidate_family",
            "candidate_model": "oracle_candidate_model",
        }
    )
    sample_oracle = sample_oracle.merge(best_rows, on="sample_id", how="left", validate="one_to_one")

    selected_one = selected.copy()
    if selected_one["sample_id"].duplicated().any():
        raise ValueError("Selector prediction file must contain one row per sample_id")
    selected_one["selector_abs_error_bpm"] = (
        selected_one["pred_hr_bpm"] - selected_one["gt_hr_bpm"]
    ).abs()
    window = selected_one.merge(
        sample_oracle,
        on=["sample_id", "subject_std", "gt_hr_bpm"],
        how="left",
        validate="one_to_one",
    )
    window["selector_regret_bpm"] = (
        window["selector_abs_error_bpm"] - window["oracle_abs_error_bpm"]
    )
    window["selected_is_oracle_tie"] = np.isclose(
        window["selector_abs_error_bpm"], window["oracle_abs_error_bpm"], atol=1e-9
    )
    for threshold in (5, 10, 15):
        window[f"oracle_within{threshold}"] = window["oracle_abs_error_bpm"] <= threshold
        window[f"selector_within{threshold}"] = window["selector_abs_error_bpm"] <= threshold

    window["failure_class_5bpm"] = np.select(
        [
            window["selector_abs_error_bpm"] <= 5,
            (window["selector_abs_error_bpm"] > 5) & (window["oracle_abs_error_bpm"] <= 5),
            window["oracle_abs_error_bpm"] > 5,
        ],
        ["selector_success", "selector_error_reducible", "candidate_pool_failure"],
        default="unclassified",
    )
    window["failure_class_10bpm"] = np.select(
        [
            window["selector_abs_error_bpm"] <= 10,
            (window["selector_abs_error_bpm"] > 10) & (window["oracle_abs_error_bpm"] <= 10),
            window["oracle_abs_error_bpm"] > 10,
        ],
        ["selector_success", "selector_error_reducible", "candidate_pool_failure"],
        default="unclassified",
    )

    participant = window.groupby("subject_std", sort=False).agg(
        n_windows=("sample_id", "size"),
        selector_mae_bpm=("selector_abs_error_bpm", "mean"),
        oracle_mae_bpm=("oracle_abs_error_bpm", "mean"),
        selector_within5=("selector_within5", "mean"),
        oracle_within5=("oracle_within5", "mean"),
        selector_within10=("selector_within10", "mean"),
        oracle_within10=("oracle_within10", "mean"),
        selected_oracle_tie_rate=("selected_is_oracle_tie", "mean"),
    ).reset_index()
    participant["selector_minus_oracle_mae_bpm"] = (
        participant["selector_mae_bpm"] - participant["oracle_mae_bpm"]
    )

    failure_rows: list[dict[str, object]] = []
    for threshold in (5, 10):
        counts = window[f"failure_class_{threshold}bpm"].value_counts()
        for label in ("selector_success", "selector_error_reducible", "candidate_pool_failure"):
            count = int(counts.get(label, 0))
            failure_rows.append(
                {
                    "threshold_bpm": threshold,
                    "failure_class": label,
                    "n_windows": count,
                    "fraction": count / len(window),
                }
            )
    failure = pd.DataFrame(failure_rows)

    selected_model = window.groupby(
        ["selected_source_type", "selected_candidate_model"], dropna=False, sort=False
    ).agg(
        n_windows=("sample_id", "size"),
        mae_bpm=("selector_abs_error_bpm", "mean"),
        within5=("selector_within5", "mean"),
        within10=("selector_within10", "mean"),
        mean_regret_bpm=("selector_regret_bpm", "mean"),
    ).reset_index()
    selected_model["fraction_selected"] = selected_model["n_windows"] / len(window)

    oracle_model = window.groupby(
        ["oracle_source_type", "oracle_candidate_model"], dropna=False, sort=False
    ).size().rename("n_oracle_windows").reset_index()
    oracle_model["fraction_oracle"] = oracle_model["n_oracle_windows"] / len(window)

    summary = {
        "task_id": "V32_DEVELOPMENT_CANDIDATE_ORACLE_REGRET",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development_only_diagnostic_reference_hr_never_enters_inference",
        "inputs": {
            "candidate_scores": str(args.candidate_scores.resolve()),
            "candidate_scores_sha256": sha256(args.candidate_scores),
            "selector_predictions": str(args.selector_predictions.resolve()),
            "selector_predictions_sha256": sha256(args.selector_predictions),
        },
        "n_windows": int(len(window)),
        "n_participants": int(window["subject_std"].nunique()),
        "candidate_count": {
            "min": int(window["candidate_count"].min()),
            "median": float(window["candidate_count"].median()),
            "max": int(window["candidate_count"].max()),
        },
        "selector": {
            "window_mae_bpm": float(window["selector_abs_error_bpm"].mean()),
            "participant_equal_mae_bpm": participant_equal(window, "selector_abs_error_bpm"),
            "within5": float(window["selector_within5"].mean()),
            "within10": float(window["selector_within10"].mean()),
        },
        "candidate_oracle_non_deployable": {
            "window_mae_bpm": float(window["oracle_abs_error_bpm"].mean()),
            "participant_equal_mae_bpm": participant_equal(window, "oracle_abs_error_bpm"),
            "within5": float(window["oracle_within5"].mean()),
            "within10": float(window["oracle_within10"].mean()),
        },
        "selector_regret": {
            "window_mean_bpm": float(window["selector_regret_bpm"].mean()),
            "window_median_bpm": float(window["selector_regret_bpm"].median()),
            "selected_oracle_tie_rate": float(window["selected_is_oracle_tie"].mean()),
            "participant_bootstrap": bootstrap_participant_gap(
                participant, args.bootstrap_draws, args.bootstrap_seed
            ),
        },
        "failure_decomposition_5bpm": {
            row["failure_class"]: {
                "n_windows": int(row["n_windows"]),
                "fraction": float(row["fraction"]),
            }
            for row in failure_rows
            if row["threshold_bpm"] == 5
        },
        "failure_decomposition_10bpm": {
            row["failure_class"]: {
                "n_windows": int(row["n_windows"]),
                "fraction": float(row["fraction"]),
            }
            for row in failure_rows
            if row["threshold_bpm"] == 10
        },
        "interpretation_gate": {
            "selector_is_primary_bottleneck_if_reducible_failures_exceed_pool_failures": bool(
                (
                    (window["failure_class_5bpm"] == "selector_error_reducible").sum()
                    > (window["failure_class_5bpm"] == "candidate_pool_failure").sum()
                )
            ),
            "oracle_is_not_a_reportable_model": True,
            "external_outcomes_accessed": False,
        },
    }

    integrity = {
        "passed": bool(
            len(window) == selected_one["sample_id"].nunique()
            and window["oracle_abs_error_bpm"].notna().all()
            and (window["selector_regret_bpm"] >= -1e-8).all()
            and window["selected_candidate_id"].isin(candidates["candidate_id"]).all()
            and per_candidate_counts.min() == per_candidate_counts.max() == 3
        ),
        "one_selector_prediction_per_window": bool(
            not selected_one["sample_id"].duplicated().any()
        ),
        "all_selector_windows_have_candidates": bool(
            window["oracle_abs_error_bpm"].notna().all()
        ),
        "all_selected_ids_exist_in_pool": bool(
            window["selected_candidate_id"].isin(candidates["candidate_id"]).all()
        ),
        "selector_regret_nonnegative": bool((window["selector_regret_bpm"] >= -1e-8).all()),
        "seed_score_count_per_candidate": int(per_candidate_counts.min()),
        "reference_hr_used_only_for_retrospective_diagnostic": True,
    }
    if not integrity["passed"]:
        raise ValueError(f"Integrity audit failed: {integrity}")

    window.to_csv(args.output_dir / "window_oracle_regret.csv", index=False)
    participant.to_csv(args.output_dir / "participant_oracle_regret.csv", index=False)
    failure.to_csv(args.output_dir / "failure_decomposition.csv", index=False)
    selected_model.to_csv(args.output_dir / "selected_model_metrics.csv", index=False)
    oracle_model.to_csv(args.output_dir / "oracle_model_frequency.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "integrity": integrity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
