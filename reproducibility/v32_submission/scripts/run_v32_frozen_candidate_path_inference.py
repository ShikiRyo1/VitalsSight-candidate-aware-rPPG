#!/usr/bin/env python3
"""Run the frozen V32 selector on an outcome-free external candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_v32_causal_candidate_path as v32  # noqa: E402
import screen_v31_tree_rankers as candidate  # noqa: E402
from v32_candidate_relations import add_prediction_only_relations  # noqa: E402


TASK_ID = "V32_FROZEN_CAUSAL_CANDIDATE_PATH_EXTERNAL_INFERENCE"
FORBIDDEN_INPUT_TOKENS = (
    "gt_hr",
    "reference",
    "ground_truth",
    "candidate_abs_error",
    "unsafe",
    "biopac",
    "ecg",
    "empatica",
    "bvp",
    "target_hr",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_freeze(freeze_dir: Path) -> dict[str, Any]:
    manifest = read_json(freeze_dir / "manifest.json")
    for record in manifest["files"]:
        path = freeze_dir / str(record["path"])
        require(path.is_file(), f"freeze artifact missing: {path}")
        require(path.stat().st_size == int(record["bytes"]), f"freeze size mismatch: {path}")
        require(sha256(path) == str(record["sha256"]).upper(), f"freeze hash mismatch: {path}")
    contract = read_json(freeze_dir / "freeze_contract.json")
    require(
        contract.get("status") == "FROZEN_BEFORE_EXTERNAL_PARTICIPANT_ARCHIVE_ACCESS",
        "freeze status is invalid",
    )
    require(contract.get("external_outcomes_accessed") is False, "freeze reports outcome access")
    return contract


def external_features(
    pool: pd.DataFrame,
    relation: bool,
    contract: dict[str, Any],
) -> np.ndarray:
    prepared = pool.copy()
    numeric_columns = list(candidate.NODE_NUMERIC)
    if relation:
        numeric_columns.extend(candidate.RELATION_NUMERIC)
    for column in numeric_columns:
        if column not in prepared.columns:
            prepared[column] = pd.Series(np.nan, index=prepared.index, dtype=float)
    frame, _ = candidate.feature_frame(prepared, relation)
    columns = [str(value) for value in contract["columns"]]
    medians = pd.Series(contract["training_medians"], dtype=float).reindex(columns).fillna(0.0)
    values = frame.reindex(columns=columns).fillna(medians).fillna(0.0).to_numpy(float)
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite external model features")
    return values


def select_unlabeled_path(
    scored: pd.DataFrame,
    transition_lambda: float,
    max_transition_bpm: float,
) -> pd.DataFrame:
    numeric = scored[["candidate_hr_bpm", "emission_score"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise FloatingPointError("non-finite external candidate score")
    selections: list[pd.Series] = []
    for _, sequence in scored.groupby("sequence_id", sort=True):
        previous_cost: np.ndarray | None = None
        previous_hr: np.ndarray | None = None
        previous_window: int | None = None
        for window_index in sorted(sequence["window_index"].unique().tolist()):
            choices = sequence[sequence["window_index"].eq(window_index)].copy()
            choices = choices.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
            emission = choices["emission_score"].to_numpy(float)
            hr = choices["candidate_hr_bpm"].to_numpy(float)
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
            cost -= float(np.min(cost))
            row = choices.iloc[int(np.argmin(cost))].copy()
            row["path_cost_normalized"] = float(np.min(cost))
            row["transition_lambda"] = transition_lambda
            selections.append(row)
            previous_cost, previous_hr, previous_window = cost, hr, window_index
    selected = pd.DataFrame(selections)
    output = selected[
        [
            "sample_id",
            "subject_std",
            "sequence_id",
            "window_index",
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
    output.insert(0, "method", v32.METHOD)
    require(output["sample_id"].is_unique, "external selector emitted duplicate windows")
    return output.sort_values(["sequence_id", "window_index"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-pool-audit", type=Path)
    parser.add_argument("--relation-parity-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    freeze = verify_freeze(args.freeze_dir)
    parity = read_json(args.relation_parity_summary)
    require(parity.get("passed") is True, "relation-feature parity did not pass")
    relation_module = TOOLS / "v32_candidate_relations.py"
    require(
        sha256(relation_module) == str(parity["portable_relation_module_sha256"]).upper(),
        "relation module hash differs from parity audit",
    )

    pool = pd.read_csv(args.candidate_pool, low_memory=False)
    forbidden = sorted(
        column
        for column in pool.columns
        if column != "reference_status"
        if any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)
    )
    require(not forbidden, f"external outcome-derived columns are forbidden: {forbidden}")
    if "reference_status" in pool.columns:
        status = pool["reference_status"]
        explicit_sealed = status.fillna("").astype(str).eq("sealed_not_accessed")
        blank = status.isna() | status.fillna("").astype(str).str.strip().eq("")
        require(
            (explicit_sealed | blank).all(),
            "external candidate pool contains a non-sealed reference status",
        )
        if bool(blank.any()):
            require(
                args.candidate_pool_audit is not None,
                "blank reference status requires the frozen candidate-pool audit",
            )
            pool_audit = read_json(args.candidate_pool_audit)
            require(pool_audit.get("passed") is True, "candidate-pool audit did not pass")
            require(
                pool_audit.get("reference_outcomes_accessed") is False,
                "candidate-pool audit reports reference-outcome access",
            )
            require(
                pool_audit.get("populated_outcome_fields") == [],
                "candidate-pool audit reports populated outcome fields",
            )
            require(
                str(pool_audit.get("candidate_pool_sha256", "")).upper()
                == sha256(args.candidate_pool),
                "candidate-pool audit hash mismatch",
            )
            expected_blank = int(pool_audit.get("classical_candidates", -1))
            expected_explicit = int(pool_audit.get("deep_candidates", -1))
            require(int(blank.sum()) == expected_blank, "blank status count mismatch")
            require(
                int(explicit_sealed.sum()) == expected_explicit,
                "explicit sealed status count mismatch",
            )
    required = {
        "sample_id",
        "subject_std",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    }
    require(not (required - set(pool.columns)), f"candidate pool missing: {sorted(required-set(pool.columns))}")
    require(not pool.duplicated(["sample_id", "candidate_id"]).any(), "duplicate external candidate identity")
    pool = add_prediction_only_relations(pool)

    feature_contract = read_json(args.freeze_dir / "feature_contract.json")
    node = external_features(pool, False, feature_contract["node"])
    relation = external_features(pool, True, feature_contract["relation"])
    alpha = float(freeze["protocol"]["intrinsic_relation_blend"])
    unsafe_penalty = float(freeze["protocol"]["unsafe5_penalty"])
    seed_rows: list[pd.DataFrame] = []
    artifacts = {(int(row["seed"]), str(row["role"])): row for row in freeze["model_artifacts"]}
    for seed in [int(value) for value in freeze["protocol"]["seeds"]]:
        models = {}
        for role in (
            "node_regressor",
            "relation_regressor",
            "node_safe5_classifier",
            "relation_safe5_classifier",
        ):
            record = artifacts[(seed, role)]
            path = args.freeze_dir / str(record["path"])
            require(sha256(path) == str(record["sha256"]).upper(), f"model hash drift: {path}")
            models[role] = joblib.load(path)
        predicted_error = (
            (1.0 - alpha) * models["node_regressor"].predict(node)
            + alpha * models["relation_regressor"].predict(relation)
        )
        safe_probability = (
            (1.0 - alpha) * v32.positive_probability(models["node_safe5_classifier"], node)
            + alpha * v32.positive_probability(models["relation_safe5_classifier"], relation)
        )
        emission = predicted_error + unsafe_penalty * (1.0 - safe_probability)
        require(np.isfinite(emission).all(), "non-finite external emission")
        frame = pool[
            [
                "sample_id",
                "subject_std",
                "candidate_id",
                "candidate_hr_bpm",
                "source_type",
                "candidate_family",
                "candidate_model",
            ]
        ].copy()
        frame["seed"] = seed
        frame["predicted_error_bpm"] = predicted_error
        frame["predicted_safe5_probability"] = safe_probability
        frame["emission_score"] = emission
        seed_rows.append(frame)
    per_seed = pd.concat(seed_rows, ignore_index=True)
    identity = [
        "sample_id",
        "subject_std",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    ensemble = per_seed.groupby(identity, as_index=False).agg(
        emission_score=("emission_score", "median"),
        predicted_error_bpm=("predicted_error_bpm", "median"),
        predicted_safe5_probability=("predicted_safe5_probability", "median"),
        n_seed_scores=("seed", "nunique"),
    )
    require(
        ensemble["n_seed_scores"].eq(len(freeze["protocol"]["seeds"])).all(),
        "external seed coverage mismatch",
    )
    ensemble = v32.add_sequence_identity(ensemble)
    predictions = select_unlabeled_path(
        ensemble,
        float(freeze["frozen_transition_lambda"]),
        float(freeze["protocol"]["max_transition_bpm"]),
    )
    per_seed.to_csv(args.output_dir / "external_candidate_scores_per_seed.csv", index=False)
    ensemble.to_csv(args.output_dir / "external_candidate_scores_ensemble.csv", index=False)
    predictions.to_csv(args.output_dir / "external_frozen_predictions.csv", index=False)
    summary = {
        "task": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREDICTIONS_FROZEN_WITHOUT_EXTERNAL_OUTCOMES",
        "external_outcomes_accessed": False,
        "participants": int(predictions["subject_std"].astype(str).nunique()),
        "windows": int(len(predictions)),
        "candidates": int(len(ensemble)),
        "candidate_pool_sha256": sha256(args.candidate_pool),
        "freeze_contract_sha256": sha256(args.freeze_dir / "freeze_contract.json"),
        "relation_parity_summary_sha256": sha256(args.relation_parity_summary),
        "candidate_pool_audit_sha256": (
            sha256(args.candidate_pool_audit) if args.candidate_pool_audit else None
        ),
        "relation_module_sha256": sha256(relation_module),
        "frozen_transition_lambda": float(freeze["frozen_transition_lambda"]),
        "selected_source_counts": predictions["selected_candidate_model"].value_counts().to_dict(),
        "claim_boundary": "No performance metric can be computed until the separately sealed external reference is opened once.",
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
