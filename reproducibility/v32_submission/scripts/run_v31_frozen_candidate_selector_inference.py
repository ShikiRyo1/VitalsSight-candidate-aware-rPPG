#!/usr/bin/env python3
"""Apply the frozen three-seed candidate selector without reference outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import screen_v31_tree_rankers as candidate


TASK_ID = "V31_FROZEN_CANDIDATE_SELECTOR_INFERENCE"
LABEL_TOKENS = ("gt_hr", "reference_hr", "abs_error", "unsafe_candidate", "macc")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def populated(series: pd.Series) -> bool:
    values = series.astype("string").str.strip().str.lower()
    return bool((values.notna() & ~values.isin(["", "nan", "none", "<na>"])).any())


def resolve_contract_artifact(directory: Path, recorded_path: object) -> Path:
    """Resolve a frozen artifact after a Windows-to-POSIX workspace move."""
    raw = str(recorded_path)
    direct = Path(raw)
    if direct.is_file():
        return direct
    basename = Path(raw.replace("\\", "/")).name
    if not basename or basename in {".", ".."}:
        raise RuntimeError(f"invalid artifact path in frozen contract: {raw!r}")
    relocated = directory / basename
    if not relocated.is_file():
        raise FileNotFoundError(relocated)
    return relocated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--release-gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-cohort", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    contract_path = args.selector_dir / "selector_freeze_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_DEVELOPMENT_SELECTOR":
        raise RuntimeError("selector contract is not frozen")
    if contract.get("relation_features_enabled") is not False:
        raise RuntimeError("unvalidated relation features are not permitted")
    if contract.get("external_outcomes_accessed") is not False:
        raise RuntimeError("selector contract reports external outcome access")
    gate_contract_path = args.release_gate_dir / "release_gate_freeze_contract.json"
    gate_contract = json.loads(gate_contract_path.read_text(encoding="utf-8"))
    if gate_contract.get("status") != "FROZEN_DEVELOPMENT_RELEASE_GATE":
        raise RuntimeError("release gate contract is not frozen")
    if gate_contract.get("relation_features_enabled") is not False:
        raise RuntimeError("release gate contains unvalidated relation features")
    if gate_contract.get("research_score_not_probability") is not True:
        raise RuntimeError("release gate research-score boundary is missing")
    if gate_contract.get("no_calibrated_safety_claim") is not True:
        raise RuntimeError("release gate safety-claim boundary is missing")
    release_threshold = float(gate_contract.get("proposed_release_threshold", np.nan))
    if not np.isfinite(release_threshold) or not 0.0 <= release_threshold <= 1.0:
        raise RuntimeError("release gate threshold is invalid")

    pool = pd.read_csv(args.candidate_pool, low_memory=False)
    required = {
        "sample_id",
        "subject_std",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    }
    missing = sorted(required - set(pool.columns))
    if missing:
        raise RuntimeError(f"candidate pool missing columns: {missing}")
    label_columns = [
        column for column in pool.columns if any(token in column.lower() for token in LABEL_TOKENS)
    ]
    populated_labels = [column for column in label_columns if populated(pool[column])]
    if populated_labels:
        raise RuntimeError(f"candidate pool contains populated outcome fields: {populated_labels}")
    if pool.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError("duplicate sample/candidate identity")
    if not np.isfinite(pd.to_numeric(pool["candidate_hr_bpm"], errors="coerce")).all():
        raise RuntimeError("non-finite candidate HR")

    observed_participants = int(pool["subject_std"].astype(str).nunique())
    if not args.allow_incomplete_cohort and observed_participants != 83:
        raise RuntimeError(
            f"external candidate pool is incomplete: {observed_participants} participants, expected 83"
        )

    base_features, _ = candidate.feature_frame(pool, relation=False)
    if np.isinf(base_features.to_numpy(dtype=float, na_value=np.nan)).any():
        raise RuntimeError("infinite selector feature survived missing-value normalization")
    scored_parts: list[pd.DataFrame] = []
    gate_scored_parts: list[pd.DataFrame] = []
    model_audit: list[dict[str, Any]] = []
    gate_model_audit: list[dict[str, Any]] = []
    expected_models = contract.get("models", [])
    if len(expected_models) != 3:
        raise RuntimeError("frozen selector must contain exactly three seed models")
    for model_row in expected_models:
        path = resolve_contract_artifact(args.selector_dir, model_row["path"])
        actual_hash = sha256(path)
        if actual_hash != str(model_row["sha256"]).upper():
            raise RuntimeError(f"selector hash mismatch: {path}")
        artifact = joblib.load(path)
        columns = list(artifact["feature_columns"])
        forbidden_features = [
            column
            for column in columns
            if any(token in column.lower() for token in candidate.FORBIDDEN_TOKENS)
        ]
        if forbidden_features:
            raise RuntimeError(f"frozen model contains forbidden features: {forbidden_features}")
        medians = pd.Series(artifact["feature_medians"], dtype=float).reindex(columns).fillna(0.0)
        x = base_features.reindex(columns=columns).fillna(medians).fillna(0.0).to_numpy(float)
        score = np.asarray(artifact["model"].predict(x), dtype=float)
        if not np.isfinite(score).all():
            raise RuntimeError(f"non-finite selector scores from {path.name}")
        part = pool[
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
        part["seed"] = int(artifact["seed"])
        part["selection_score"] = score
        part["within_window_rank"] = part.groupby("sample_id")["selection_score"].rank(
            method="average", pct=True
        )
        scored_parts.append(part)
        model_audit.append(
            {
                "seed": int(artifact["seed"]),
                "path": str(path.resolve()),
                "sha256": actual_hash,
                "feature_count": len(columns),
                "relation_features_enabled": bool(artifact["relation_features_enabled"]),
            }
        )

    expected_gate_models = gate_contract.get("models", [])
    if len(expected_gate_models) != 3:
        raise RuntimeError("frozen release gate must contain exactly three seed models")
    for model_row in expected_gate_models:
        path = resolve_contract_artifact(args.release_gate_dir, model_row["path"])
        actual_hash = sha256(path)
        if actual_hash != str(model_row["sha256"]).upper():
            raise RuntimeError(f"release gate hash mismatch: {path}")
        artifact = joblib.load(path)
        if artifact.get("relation_features_enabled") is not False:
            raise RuntimeError(f"release gate model contains relation features: {path.name}")
        if artifact.get("research_score_not_probability") is not True:
            raise RuntimeError(f"release gate model boundary missing: {path.name}")
        columns = list(artifact["feature_columns"])
        if columns != list(gate_contract.get("feature_columns", [])):
            raise RuntimeError(f"release gate feature contract mismatch: {path.name}")
        medians = pd.Series(artifact["feature_medians"], dtype=float).reindex(columns).fillna(0.0)
        x = base_features.reindex(columns=columns).fillna(medians).fillna(0.0).to_numpy(float)
        classifier = artifact["model"]
        if 1 in classifier.classes_:
            safe_score = classifier.predict_proba(x)[:, list(classifier.classes_).index(1)]
        else:
            safe_score = np.zeros(len(x), dtype=float)
        if not np.isfinite(safe_score).all() or np.any((safe_score < 0.0) | (safe_score > 1.0)):
            raise RuntimeError(f"invalid release research scores from {path.name}")
        part = pool[
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
        part["seed"] = int(artifact["seed"])
        part["research_safe10_score"] = np.asarray(safe_score, dtype=float)
        gate_scored_parts.append(part)
        gate_model_audit.append(
            {
                "seed": int(artifact["seed"]),
                "path": str(path.resolve()),
                "sha256": actual_hash,
                "feature_count": len(columns),
                "research_score_not_probability": True,
            }
        )

    scored = pd.concat(scored_parts, ignore_index=True)
    identity = [
        "sample_id",
        "subject_std",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    aggregate = (
        scored.groupby(identity, as_index=False)
        .agg(
            ensemble_rank=("within_window_rank", "median"),
            ensemble_rank_mean=("within_window_rank", "mean"),
            rank_std=("within_window_rank", "std"),
            n_seed_scores=("seed", "nunique"),
        )
    )
    if not aggregate["n_seed_scores"].eq(3).all():
        raise RuntimeError("incomplete seed-score coverage")
    ordered = aggregate.sort_values(
        ["sample_id", "ensemble_rank", "ensemble_rank_mean", "candidate_id"],
        kind="mergesort",
    )
    selected = ordered.groupby("sample_id", sort=False, as_index=False).head(1).copy()
    gate_identity = [
        "sample_id",
        "subject_std",
        "candidate_id",
        "candidate_hr_bpm",
        "source_type",
        "candidate_family",
        "candidate_model",
    ]
    gate_scores = pd.concat(gate_scored_parts, ignore_index=True)
    gate_aggregate = (
        gate_scores.groupby(gate_identity, as_index=False)
        .agg(
            research_safe10_score=("research_safe10_score", "median"),
            research_safe10_score_mean=("research_safe10_score", "mean"),
            research_safe10_score_std=("research_safe10_score", "std"),
            n_gate_seed_scores=("seed", "nunique"),
        )
    )
    if not gate_aggregate["n_gate_seed_scores"].eq(3).all():
        raise RuntimeError("incomplete release-gate seed-score coverage")
    selected = selected.merge(
        gate_aggregate,
        on=gate_identity,
        how="left",
        validate="one_to_one",
    )
    if selected["research_safe10_score"].isna().any():
        raise RuntimeError("selected candidates are missing release-gate evidence")
    selected["research_risk_score"] = 1.0 - selected["research_safe10_score"]
    selected["proposed_release_threshold"] = release_threshold
    selected["proposed_state"] = np.where(
        selected["research_risk_score"].le(release_threshold),
        "PROPOSED_RELEASE",
        "PROPOSED_REVIEW",
    )
    pool_summary = (
        aggregate.groupby("sample_id", as_index=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            source_count=("source_type", "nunique"),
            candidate_hr_min=("candidate_hr_bpm", "min"),
            candidate_hr_max=("candidate_hr_bpm", "max"),
            median_seed_rank_std=("rank_std", "median"),
        )
    )
    second = ordered.groupby("sample_id", sort=False).nth(1).reset_index()[
        ["sample_id", "ensemble_rank"]
    ].rename(columns={"ensemble_rank": "second_ensemble_rank"})
    selected = selected.merge(pool_summary, on="sample_id", how="left", validate="one_to_one")
    selected = selected.merge(second, on="sample_id", how="left", validate="one_to_one")
    selected["rank_margin"] = selected["second_ensemble_rank"] - selected["ensemble_rank"]
    selected.insert(0, "method", "VitalsSight_frozen_candidate_aware")
    selected = selected.rename(
        columns={
            "candidate_id": "selected_candidate_id",
            "candidate_hr_bpm": "pred_hr_bpm",
            "source_type": "selected_source_type",
            "candidate_family": "selected_candidate_family",
            "candidate_model": "selected_candidate_model",
        }
    )
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("more than one selected candidate per sample")

    per_seed_path = args.output_dir / "candidate_scores_per_seed.csv"
    ensemble_path = args.output_dir / "candidate_scores_ensemble.csv"
    selected_path = args.output_dir / "external_unlabeled_selected_predictions.csv"
    scored.to_csv(per_seed_path, index=False)
    aggregate.to_csv(ensemble_path, index=False)
    selected.to_csv(selected_path, index=False)
    audit: dict[str, Any] = {
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "passed": True,
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_sha256": sha256(args.candidate_pool),
        "selector_contract": str(contract_path.resolve()),
        "selector_contract_sha256": sha256(contract_path),
        "models": model_audit,
        "release_gate_contract": str(gate_contract_path.resolve()),
        "release_gate_contract_sha256": sha256(gate_contract_path),
        "release_gate_models": gate_model_audit,
        "proposed_release_threshold": release_threshold,
        "proposed_release_windows": int(selected["proposed_state"].eq("PROPOSED_RELEASE").sum()),
        "proposed_review_windows": int(selected["proposed_state"].eq("PROPOSED_REVIEW").sum()),
        "unlabeled_proposed_release_coverage": float(
            selected["proposed_state"].eq("PROPOSED_RELEASE").mean()
        ),
        "research_score_not_probability": True,
        "no_calibrated_safety_claim": True,
        "participants": observed_participants,
        "samples": int(selected["sample_id"].nunique()),
        "candidates": int(len(pool)),
        "one_observed_candidate_per_sample": True,
        "selected_source_identity_complete": bool(
            selected[
                ["selected_candidate_id", "selected_source_type", "selected_candidate_model"]
            ].notna().all().all()
        ),
        "populated_outcome_fields": populated_labels,
        "external_outcomes_accessed": False,
        "reference_outcomes_accessed": False,
        "cohort_complete_required": not args.allow_incomplete_cohort,
        "outputs": {
            per_seed_path.name: file_record(per_seed_path),
            ensemble_path.name: file_record(ensemble_path),
            selected_path.name: file_record(selected_path),
        },
        "claim_boundary": (
            "This file contains label-sealed predictions and evidence only. It is not an "
            "external performance result until the one-time authorized unseal."
        ),
    }
    write_json(args.output_dir / "inference_audit.json", audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
