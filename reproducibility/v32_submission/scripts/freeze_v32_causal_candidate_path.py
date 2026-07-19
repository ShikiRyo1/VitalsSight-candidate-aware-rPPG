#!/usr/bin/env python3
"""Freeze the V32 candidate-path selector before external outcome access.

The four emission models are fitted on the complete development cohort for
each predeclared seed.  Feature columns, training medians, temporal penalty,
software versions, input hashes and code hashes are serialized as one
immutable inference contract.  No external data path is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_v32_causal_candidate_path as v32  # noqa: E402
import screen_v31_tree_rankers as candidate  # noqa: E402


TASK_ID = "V32_CAUSAL_CANDIDATE_PATH_PREEXTERNAL_FREEZE"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_protocol(summary: dict[str, Any]) -> v32.Protocol:
    source = dict(summary["protocol"])
    source["seeds"] = tuple(int(value) for value in source["seeds"])
    source["transition_lambdas"] = tuple(float(value) for value in source["transition_lambdas"])
    return v32.Protocol(**source)


def feature_contract(pool: pd.DataFrame, relation: bool) -> tuple[np.ndarray, dict[str, Any]]:
    frame, columns = candidate.feature_frame(pool, relation)
    columns = sorted(columns)
    frame = frame.reindex(columns=columns)
    medians = frame.median(numeric_only=True).reindex(columns).fillna(0.0)
    values = frame.fillna(medians).fillna(0.0).to_numpy(float)
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite frozen feature matrix")
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in candidate.FORBIDDEN_TOKENS)
    ]
    require(not forbidden, f"forbidden target-derived features: {forbidden}")
    return values, {
        "relation_features": relation,
        "columns": columns,
        "training_medians": {column: float(medians[column]) for column in columns},
        "forbidden_tokens": list(candidate.FORBIDDEN_TOKENS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--nested-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir()

    summary_path = args.nested_result_dir / "summary.json"
    summary = read_json(summary_path)
    require(summary.get("task") == v32.TASK_ID, "wrong nested-development task")
    require(summary.get("external_outcomes_accessed") is False, "nested run reports external access")
    require(summary.get("development_only") is True, "nested run role is not development-only")
    require(int(summary.get("n_subjects", 0)) == 42, "development participant count mismatch")
    require(int(summary.get("n_windows", 0)) == 439, "development window count mismatch")
    primary = summary["causal_path_minus_primary_comparator"]
    require(primary.get("method_b") == v32.PRIMARY_COMPARATOR, "wrong primary comparator")
    require(float(primary["ci95_high"]) < 0.0, "primary nested CI does not support freezing")
    require(float(primary["paired_sign_flip_p_plus_one"]) < 0.05, "primary sign-flip test does not support freezing")

    for name, record in summary.get("source_files", {}).items():
        path = Path(str(record["path"]))
        require(path.is_file(), f"nested source file missing: {name}")
        require(sha256(path) == str(record["sha256"]).upper(), f"nested source hash drift: {name}")

    pool = pd.read_csv(args.input_pool, low_memory=False)
    require(sha256(args.input_pool) == str(summary["input_sha256"]).upper(), "development input hash mismatch")
    require(pool["subject_std"].astype(str).nunique() == 42, "pool participant count mismatch")
    require(pool["sample_id"].astype(str).nunique() == 439, "pool window count mismatch")
    require(not pool.duplicated(["sample_id", "candidate_id"]).any(), "duplicate candidate identity")

    protocol = build_protocol(summary)
    transition_lambda = float(summary["frozen_transition_lambda_candidate"])
    selected_lambdas = [
        float(row["selected_transition_lambda"])
        for row in summary["selected_transition_lambdas"]
    ]
    require(transition_lambda == float(np.median(selected_lambdas)), "temporal penalty rule mismatch")

    node_values, node_contract = feature_contract(pool, False)
    relation_values, relation_contract = feature_contract(pool, True)
    error, _, safe5 = candidate.target_values(pool)
    weights = candidate.sample_weights(pool)
    clipped_error = np.clip(error, 0.0, 20.0)

    contracts = {"node": node_contract, "relation": relation_contract}
    write_json(args.output_dir / "feature_contract.json", contracts)
    artifacts: list[dict[str, Any]] = []
    prediction_checks: list[dict[str, Any]] = []
    for seed in protocol.seeds:
        specifications = [
            ("node_regressor", v32.make_regressor(protocol, seed), node_values, clipped_error),
            ("relation_regressor", v32.make_regressor(protocol, seed + 100_000), relation_values, clipped_error),
            ("node_safe5_classifier", v32.make_classifier(protocol, seed + 200_000), node_values, safe5),
            ("relation_safe5_classifier", v32.make_classifier(protocol, seed + 300_000), relation_values, safe5),
        ]
        for name, model, values, target in specifications:
            model.fit(values, target, sample_weight=weights)
            path = model_dir / f"seed_{seed}_{name}.joblib"
            joblib.dump(model, path, compress=3)
            if "classifier" in name:
                prediction = v32.positive_probability(model, values)
            else:
                prediction = model.predict(values)
            require(np.isfinite(prediction).all(), f"non-finite frozen model check: {name}")
            artifacts.append(
                {
                    "seed": seed,
                    "role": name,
                    "path": str(path.relative_to(args.output_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
            prediction_checks.append(
                {
                    "seed": seed,
                    "role": name,
                    "minimum": float(np.min(prediction)),
                    "maximum": float(np.max(prediction)),
                    "finite": True,
                }
            )

    source_paths = [
        Path(__file__).resolve(),
        TOOLS / "run_v32_causal_candidate_path.py",
        TOOLS / "screen_v31_tree_rankers.py",
    ]
    freeze = {
        "task": TASK_ID,
        "generated_at_utc": utc_now(),
        "status": "FROZEN_BEFORE_EXTERNAL_PARTICIPANT_ARCHIVE_ACCESS",
        "external_outcomes_accessed": False,
        "external_participant_archives_accessed": False,
        "development_pool": {
            "path": str(args.input_pool.resolve()),
            "bytes": args.input_pool.stat().st_size,
            "sha256": sha256(args.input_pool),
            "participants": 42,
            "windows": 439,
            "candidates": int(len(pool)),
        },
        "nested_development_result": {
            "path": str(summary_path.resolve()),
            "sha256": sha256(summary_path),
            "primary_comparator": v32.PRIMARY_COMPARATOR,
            "participant_equal_delta_bpm": float(primary["mean_delta_a_minus_b_bpm"]),
            "ci95": [float(primary["ci95_low"]), float(primary["ci95_high"])],
            "paired_sign_flip_p_plus_one": float(primary["paired_sign_flip_p_plus_one"]),
        },
        "protocol": asdict(protocol),
        "frozen_transition_lambda": transition_lambda,
        "transition_lambda_selection_rule": "median of the three outer-fold inner-selected penalties",
        "model_artifacts": artifacts,
        "prediction_range_checks": prediction_checks,
        "source_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "claim_boundary": (
            "This artifact fixes candidate emission, relation features and causal transition "
            "penalty before external participant archives or outcomes are accessed."
        ),
    }
    write_json(args.output_dir / "freeze_contract.json", freeze)
    products = sorted(path for path in args.output_dir.rglob("*") if path.is_file())
    write_json(
        args.output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "generated_at_utc": utc_now(),
            "files": [
                {
                    "path": str(path.relative_to(args.output_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in products
            ],
        },
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

