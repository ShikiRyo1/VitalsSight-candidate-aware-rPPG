#!/usr/bin/env python3
"""Independently audit the frozen V32 EMPD post hoc replay.

This script intentionally does not import the inference or evaluation modules.
It recomputes participant-equal endpoints from the frozen prediction and
reference ledgers, then repeats paired inference with independent random seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_METHOD = "VitalsSight_v32_causal_candidate_path"
COMPARATORS = (
    "VitalsSight_frozen_candidate_aware",
    "matched_extra_trees_stacker_frozen",
    "single_route_TSCAN_frozen",
)
BOOTSTRAP_DRAWS = 20000
SIGN_FLIP_DRAWS = 200000
BOOTSTRAP_SEED = 920719
SIGN_FLIP_SEED = 930719


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


def participant_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, group in frame.groupby("subject_std", sort=True):
        errors = group["abs_error_bpm"].to_numpy(float)
        rows.append(
            {
                "subject_std": str(subject),
                "windows": int(len(group)),
                "mae_bpm": float(errors.mean()),
                "rmse_bpm": float(np.sqrt(np.square(errors).mean())),
                "within5": float((errors <= 5.0).mean()),
                "within10": float((errors <= 10.0).mean()),
                "unsafe_gt10": float((errors > 10.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_audit(differences: np.ndarray, index: int) -> dict[str, float | int]:
    require(len(differences) == 83, "paired contrast must contain 83 participants")
    require(np.isfinite(differences).all(), "paired differences contain non-finite values")
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED + index)
    sampled = bootstrap_rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    bootstrap_means = differences[sampled].mean(axis=1)
    observed = float(differences.mean())

    sign_rng = np.random.default_rng(SIGN_FLIP_SEED + index)
    extreme = 0
    completed = 0
    batch_size = 10000
    while completed < SIGN_FLIP_DRAWS:
        current = min(batch_size, SIGN_FLIP_DRAWS - completed)
        signs = sign_rng.choice(np.array([-1.0, 1.0]), size=(current, len(differences)))
        null_means = (signs * differences).mean(axis=1)
        extreme += int((np.abs(null_means) >= abs(observed) - 1e-15).sum())
        completed += current
    return {
        "n_participants": int(len(differences)),
        "mean_delta_primary_minus_comparator_bpm": observed,
        "independent_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "independent_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "independent_sign_flip_p_plus_one": float((extreme + 1) / (SIGN_FLIP_DRAWS + 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparator-participants", type=Path, required=True)
    parser.add_argument("--reported-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    reported = read_json(args.reported_summary)
    require(reported.get("status") == "POSTHOC_FIXED_REPLAY_FROZEN", "reported replay is not frozen")
    require(reported.get("scientific_role") == "exploratory_external_replay_only", "scientific role changed")
    require(reported.get("external_outcomes_were_known_before_v32_design") is True, "outcome chronology is missing")
    require(reported.get("no_fitting_or_tuning_on_empd") is True, "no-tuning boundary is missing")

    predictions = pd.read_csv(args.predictions, low_memory=False)
    reference = pd.read_csv(args.reference, low_memory=False)
    require(len(predictions) == 2120 and predictions["sample_id"].is_unique, "prediction identity/count mismatch")
    require(len(reference) == 2120 and reference["sample_id"].is_unique, "reference identity/count mismatch")
    require(predictions["method"].astype(str).eq(PRIMARY_METHOD).all(), "unexpected method in prediction ledger")
    require(set(predictions["sample_id"]) == set(reference["sample_id"]), "prediction/reference sample mismatch")

    joined = predictions.merge(
        reference[["sample_id", "subject_std", "reference_hr_bpm"]],
        on="sample_id",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    require(
        joined["subject_std"].astype(str).eq(joined["subject_std_reference"].astype(str)).all(),
        "participant identity mismatch",
    )
    joined["abs_error_bpm"] = (
        pd.to_numeric(joined["pred_hr_bpm"], errors="raise")
        - pd.to_numeric(joined["reference_hr_bpm"], errors="raise")
    ).abs()
    participants = participant_endpoints(joined)
    require(len(participants) == 83, "participant count mismatch")

    recomputed = {
        "participants": int(len(participants)),
        "windows": int(len(joined)),
        "participant_equal_mae_bpm": float(participants["mae_bpm"].mean()),
        "participant_equal_rmse_bpm": float(participants["rmse_bpm"].mean()),
        "participant_equal_within5": float(participants["within5"].mean()),
        "participant_equal_within10": float(participants["within10"].mean()),
        "participant_equal_unsafe_gt10": float(participants["unsafe_gt10"].mean()),
    }
    reported_metrics = reported["method_metrics"]
    for key, value in recomputed.items():
        require(abs(float(value) - float(reported_metrics[key])) <= 1e-12, f"reported metric mismatch: {key}")

    comparator_frame = pd.read_csv(args.comparator_participants, low_memory=False)
    primary = participants.set_index("subject_std")
    reported_contrasts = {row["method_b"]: row for row in reported["paired_contrasts"]}
    contrast_rows: list[dict[str, Any]] = []
    for index, comparator in enumerate(COMPARATORS):
        current = comparator_frame[comparator_frame["method"].astype(str).eq(comparator)].copy()
        current["subject_std"] = current["subject_std"].astype(str)
        current = current.set_index("subject_std")
        require(set(primary.index) == set(current.index), f"participant mismatch: {comparator}")
        subjects = sorted(primary.index)
        differences = primary.loc[subjects, "mae_bpm"].to_numpy(float) - current.loc[subjects, "mae_bpm"].to_numpy(float)
        audit = paired_audit(differences, index)
        reported_row = reported_contrasts[comparator]
        require(
            abs(audit["mean_delta_primary_minus_comparator_bpm"] - reported_row["mean_delta_primary_minus_comparator_bpm"]) <= 1e-12,
            f"reported paired mean mismatch: {comparator}",
        )
        require(audit["independent_ci95_high"] < 0.0, f"independent confidence interval crosses zero: {comparator}")
        require(audit["independent_sign_flip_p_plus_one"] < 0.05, f"independent sign-flip test is not below 0.05: {comparator}")
        contrast_rows.append({"method_b": comparator, **audit})

    participants.to_csv(args.output_dir / "independent_participant_metrics.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(args.output_dir / "independent_paired_audit.csv", index=False)
    audit_summary = {
        "task_id": "V32_EMPD_POSTHOC_FIXED_REPLAY_INDEPENDENT_AUDIT",
        "status": "PASS",
        "scientific_role_preserved": "exploratory_external_replay_only",
        "prediction_sha256": sha256(args.predictions),
        "reference_sha256": sha256(args.reference),
        "reported_summary_sha256": sha256(args.reported_summary),
        "recomputed_metrics": recomputed,
        "independent_contrasts": contrast_rows,
        "checks": [
            "2120 one-to-one prediction/reference identities",
            "83 participant-equal endpoint recomputation",
            "reported point estimates reproduced to 1e-12",
            "independent bootstrap intervals remain below zero",
            "independent sign-flip p-values remain below 0.05",
            "post hoc exploratory claim boundary remains explicit",
        ],
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest = {
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ]
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
