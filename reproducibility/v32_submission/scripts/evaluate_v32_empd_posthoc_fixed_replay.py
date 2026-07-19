#!/usr/bin/env python3
"""Evaluate frozen V32 predictions on the already unsealed EMPD cohort.

This is a post hoc fixed-model replay. It performs no fitting, threshold search,
feature selection, exclusion selection or endpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "V32_EMPD_POSTHOC_FIXED_MODEL_REPLAY"
PRIMARY_METHOD = "VitalsSight_v32_causal_candidate_path"
COMPARATORS = (
    "VitalsSight_frozen_candidate_aware",
    "matched_extra_trees_stacker_frozen",
    "single_route_TSCAN_frozen",
)
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 520719
SIGN_FLIP_DRAWS = 100000
SIGN_FLIP_SEED = 620719


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_prediction_manifest(predictions: Path, manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    records = {str(record["path"]): record for record in manifest.get("files", [])}
    require(predictions.name in records, "prediction file is absent from inference manifest")
    record = records[predictions.name]
    require(predictions.stat().st_size == int(record["bytes"]), "prediction size mismatch")
    require(sha256(predictions) == str(record["sha256"]).upper(), "prediction hash mismatch")


def participant_metrics(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, group in frame.groupby("subject_std", sort=True):
        error = group["abs_error_bpm"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "subject_std": str(subject),
                "windows": int(len(group)),
                "mae_bpm": float(np.mean(error)),
                "within5": float(np.mean(error <= 5.0)),
                "within10": float(np.mean(error <= 10.0)),
                "unsafe_gt10": float(np.mean(error > 10.0)),
                "rmse_bpm": float(np.sqrt(np.mean(np.square(error)))),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(participants: pd.DataFrame, windows: int) -> dict[str, Any]:
    return {
        "method": str(participants["method"].iloc[0]),
        "participants": int(len(participants)),
        "windows": int(windows),
        "participant_equal_mae_bpm": float(participants["mae_bpm"].mean()),
        "participant_equal_rmse_bpm": float(participants["rmse_bpm"].mean()),
        "participant_equal_within5": float(participants["within5"].mean()),
        "participant_equal_within10": float(participants["within10"].mean()),
        "participant_equal_unsafe_gt10": float(participants["unsafe_gt10"].mean()),
    }


def paired_inference(differences: np.ndarray, bootstrap_seed: int, sign_seed: int) -> dict[str, Any]:
    require(np.isfinite(differences).all(), "non-finite participant difference")
    n = len(differences)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    sampled = bootstrap_rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    bootstrap_means = differences[sampled].mean(axis=1)
    observed = float(differences.mean())
    sign_rng = np.random.default_rng(sign_seed)
    batch = 10000
    extreme = 0
    completed = 0
    while completed < SIGN_FLIP_DRAWS:
        current = min(batch, SIGN_FLIP_DRAWS - completed)
        signs = sign_rng.choice(np.array([-1.0, 1.0]), size=(current, n))
        null_means = (signs * differences).mean(axis=1)
        extreme += int(np.sum(np.abs(null_means) >= abs(observed) - 1e-15))
        completed += current
    return {
        "n_participants": int(n),
        "mean_delta_primary_minus_comparator_bpm": observed,
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "paired_sign_flip_p_plus_one": float((extreme + 1) / (SIGN_FLIP_DRAWS + 1)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "sign_flip_draws": SIGN_FLIP_DRAWS,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["paired_sign_flip_p_plus_one"])
    adjusted = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * rows[index]["paired_sign_flip_p_plus_one"])
        adjusted = max(adjusted, candidate)
        rows[index]["p_holm_three_posthoc_contrasts"] = adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--reference-windows", type=Path, required=True)
    parser.add_argument("--existing-participant-metrics", type=Path, required=True)
    parser.add_argument("--existing-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    verify_prediction_manifest(args.predictions, args.prediction_manifest)
    existing_summary = read_json(args.existing_summary)
    require(existing_summary.get("status") == "EXTERNAL_RESULT_FROZEN", "existing EMPD result is not frozen")
    require(existing_summary.get("external_outcomes_accessed") is True, "EMPD outcome-access state is inconsistent")

    predictions = pd.read_csv(args.predictions, low_memory=False)
    reference = pd.read_csv(args.reference_windows, low_memory=False)
    require(predictions["method"].astype(str).eq(PRIMARY_METHOD).all(), "unexpected V32 method identity")
    require(predictions["sample_id"].is_unique, "duplicate V32 sample identity")
    require(reference["sample_id"].is_unique, "duplicate reference sample identity")
    require(reference["reference_status"].astype(str).eq("evaluable").all(), "non-evaluable reference window")
    require(set(predictions["sample_id"]) == set(reference["sample_id"]), "prediction/reference identity mismatch")
    joined = predictions.merge(
        reference[["sample_id", "subject_std", "recording_id", "condition", "gender", "window_index", "reference_hr_bpm"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    require(joined["subject_std"].astype(str).eq(joined["subject_std_reference"].astype(str)).all(), "participant mismatch")
    require(pd.to_numeric(joined["pred_hr_bpm"], errors="coerce").notna().all(), "non-finite V32 HR")
    require(pd.to_numeric(joined["reference_hr_bpm"], errors="coerce").notna().all(), "non-finite reference HR")
    joined["abs_error_bpm"] = (joined["pred_hr_bpm"] - joined["reference_hr_bpm"]).abs()
    v32_participants = participant_metrics(joined, PRIMARY_METHOD)
    v32_aggregate = aggregate_metrics(v32_participants, len(joined))

    existing = pd.read_csv(args.existing_participant_metrics, low_memory=False)
    require(set(COMPARATORS).issubset(set(existing["method"].astype(str))), "missing comparator participant metrics")
    contrasts: list[dict[str, Any]] = []
    v32_index = v32_participants.set_index("subject_std")
    for comparator_index, comparator in enumerate(COMPARATORS):
        comparison = existing[existing["method"].astype(str).eq(comparator)].copy()
        comparison["subject_std"] = comparison["subject_std"].astype(str)
        comparison = comparison.set_index("subject_std")
        require(set(v32_index.index) == set(comparison.index), f"participant mismatch: {comparator}")
        ordered = sorted(v32_index.index)
        differences = (
            v32_index.loc[ordered, "mae_bpm"].to_numpy(float)
            - comparison.loc[ordered, "mae_bpm"].to_numpy(float)
        )
        row = {
            "method_a": PRIMARY_METHOD,
            "method_b": comparator,
            **paired_inference(
                differences,
                BOOTSTRAP_SEED + comparator_index,
                SIGN_FLIP_SEED + comparator_index,
            ),
        }
        contrasts.append(row)
    holm_adjust(contrasts)

    joined.to_csv(args.output_dir / "v32_empd_predictions_with_reference.csv", index=False)
    v32_participants.to_csv(args.output_dir / "v32_empd_participant_metrics.csv", index=False)
    pd.DataFrame([v32_aggregate]).to_csv(args.output_dir / "v32_empd_method_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.output_dir / "v32_empd_paired_contrasts.csv", index=False)
    summary = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "POSTHOC_FIXED_REPLAY_FROZEN",
        "scientific_role": "exploratory_external_replay_only",
        "external_outcomes_were_known_before_v32_design": True,
        "no_fitting_or_tuning_on_empd": True,
        "prediction_input": {
            "path": str(args.predictions.resolve()),
            "bytes": args.predictions.stat().st_size,
            "sha256": sha256(args.predictions),
            "manifest_sha256": sha256(args.prediction_manifest),
        },
        "reference_input": {
            "path": str(args.reference_windows.resolve()),
            "bytes": args.reference_windows.stat().st_size,
            "sha256": sha256(args.reference_windows),
        },
        "participants": int(v32_participants["subject_std"].nunique()),
        "windows": int(len(joined)),
        "method_metrics": v32_aggregate,
        "paired_contrasts": contrasts,
        "selected_source_counts": joined["selected_candidate_model"].value_counts().to_dict(),
        "claim_boundary": (
            "EMPD outcomes had already been viewed before V32 was designed. This fixed-model replay may be "
            "reported only as exploratory consistency evidence and cannot replace the prespecified V31 external test."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    products = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    write_json(
        args.output_dir / "manifest.json",
        {
            "task_id": TASK_ID,
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in products
            ],
        },
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
