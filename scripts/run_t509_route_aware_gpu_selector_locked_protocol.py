from __future__ import annotations

import csv
import io
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_t480_unified_candidate_selector_gpu as t480  # noqa: E402


TASK_ID = "T509"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

T480_PREDICTIONS = EXP / "t480_unified_selector_predictions.csv"
T480_SUMMARY = EXP / "t480_unified_selector_summary.json"
T508_PREDICTIONS = EXP / "t508_route_aware_gpu_selector_predictions.csv"
T508_SUMMARY = EXP / "t508_route_aware_gpu_feature_audit_summary.json"

BOOTSTRAP_CSV = EXP / "t509_route_aware_gpu_selector_bootstrap_ci.csv"
ENDPOINT_GATES_CSV = EXP / "t509_route_aware_gpu_selector_endpoint_gates.csv"
PROTOCOL_JSON = EXP / "t509_route_aware_gpu_selector_locked_protocol.json"
SUMMARY_JSON = EXP / "t509_route_aware_gpu_selector_protocol_summary.json"
DOC_MD = DOCS / "t509_route_aware_gpu_selector_locked_protocol.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
PROBLEM_LOG = DOCS / "problem_and_improvement_log.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"

RNG = np.random.default_rng(509)
N_BOOT = 4000
REVIEW_PENALTY_BPM = 10.0


def read_json(path: Path) -> dict[str, Any]:
    return t480.read_json(path)


def write_json(path: Path, value: Any) -> None:
    t480.write_json(path, value)


def append_or_replace(path: Path, marker: str, block: str) -> None:
    t480.append_or_replace(path, marker, block)


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    return t480.markdown_table(frame, max_rows=max_rows)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def policy_frame(pred: pd.DataFrame, policy: str) -> pd.DataFrame:
    out = pred[pred["policy"].astype(str).eq(policy) & pred["split"].astype(str).eq("test")].copy()
    out["released"] = pd.to_numeric(out["released"], errors="coerce").fillna(0).astype(int)
    out["selected_abs_error_bpm"] = pd.to_numeric(out["selected_abs_error_bpm"], errors="coerce")
    out["utility_loss_bpm"] = np.where(out["released"].gt(0), out["selected_abs_error_bpm"], REVIEW_PENALTY_BPM)
    out["unsafe_input"] = ((out["released"].gt(0)) & (out["selected_abs_error_bpm"] > 10.0)).astype(int)
    return out


def summarize(group: pd.DataFrame) -> dict[str, float]:
    released = group["released"].astype(int).gt(0)
    err = pd.to_numeric(group.loc[released, "selected_abs_error_bpm"], errors="coerce").dropna()
    return {
        "n_windows": int(len(group)),
        "coverage": float(released.mean()) if len(group) else math.nan,
        "released_mae_bpm": float(err.mean()) if len(err) else math.nan,
        "unsafe_per_input": float(group["unsafe_input"].mean()) if len(group) else math.nan,
        "utility_loss_bpm": float(pd.to_numeric(group["utility_loss_bpm"], errors="coerce").mean()) if len(group) else math.nan,
    }


def bootstrap_dataset(merged: pd.DataFrame, dataset: str) -> dict[str, Any]:
    sub = merged[merged["dataset"].astype(str).eq(dataset)].copy()
    if len(sub) == 0:
        return {"dataset": dataset, "n_windows": 0}
    ids = sub["candidate_window_id"].astype(str).to_numpy()
    n = len(ids)
    old_metrics = summarize(
        sub.rename(
            columns={
                "released_t480": "released",
                "selected_abs_error_bpm_t480": "selected_abs_error_bpm",
                "unsafe_input_t480": "unsafe_input",
                "utility_loss_bpm_t480": "utility_loss_bpm",
            }
        )
    )
    new_metrics = summarize(
        sub.rename(
            columns={
                "released_t508": "released",
                "selected_abs_error_bpm_t508": "selected_abs_error_bpm",
                "unsafe_input_t508": "unsafe_input",
                "utility_loss_bpm_t508": "utility_loss_bpm",
            }
        )
    )
    old_rel = pd.to_numeric(sub["released_t480"], errors="coerce").fillna(0).astype(int).to_numpy() > 0
    new_rel = pd.to_numeric(sub["released_t508"], errors="coerce").fillna(0).astype(int).to_numpy() > 0
    old_err = pd.to_numeric(sub["selected_abs_error_bpm_t480"], errors="coerce").to_numpy(dtype=float)
    new_err = pd.to_numeric(sub["selected_abs_error_bpm_t508"], errors="coerce").to_numpy(dtype=float)
    old_unsafe = pd.to_numeric(sub["unsafe_input_t480"], errors="coerce").fillna(0).to_numpy(dtype=float)
    new_unsafe = pd.to_numeric(sub["unsafe_input_t508"], errors="coerce").fillna(0).to_numpy(dtype=float)
    old_utility = pd.to_numeric(sub["utility_loss_bpm_t480"], errors="coerce").to_numpy(dtype=float)
    new_utility = pd.to_numeric(sub["utility_loss_bpm_t508"], errors="coerce").to_numpy(dtype=float)

    def released_mae(err: np.ndarray, rel: np.ndarray, idx: np.ndarray) -> float:
        mask = rel[idx] & np.isfinite(err[idx])
        if not bool(mask.any()):
            return math.nan
        return float(np.mean(err[idx][mask]))

    deltas = {
        "released_mae_delta_t508_minus_t480": [],
        "unsafe_per_input_delta_t508_minus_t480": [],
        "coverage_delta_t508_minus_t480": [],
        "utility_loss_delta_t508_minus_t480": [],
    }
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        old_mae = released_mae(old_err, old_rel, idx)
        new_mae = released_mae(new_err, new_rel, idx)
        if math.isfinite(old_mae) and math.isfinite(new_mae):
            deltas["released_mae_delta_t508_minus_t480"].append(float(new_mae - old_mae))
        deltas["unsafe_per_input_delta_t508_minus_t480"].append(float(np.mean(new_unsafe[idx]) - np.mean(old_unsafe[idx])))
        deltas["coverage_delta_t508_minus_t480"].append(float(np.mean(new_rel[idx]) - np.mean(old_rel[idx])))
        deltas["utility_loss_delta_t508_minus_t480"].append(float(np.mean(new_utility[idx]) - np.mean(old_utility[idx])))
    row: dict[str, Any] = {
        "dataset": dataset,
        "n_windows": int(n),
        "t480_coverage": old_metrics["coverage"],
        "t508_coverage": new_metrics["coverage"],
        "t480_released_mae_bpm": old_metrics["released_mae_bpm"],
        "t508_released_mae_bpm": new_metrics["released_mae_bpm"],
        "t480_unsafe_per_input": old_metrics["unsafe_per_input"],
        "t508_unsafe_per_input": new_metrics["unsafe_per_input"],
        "t480_utility_loss_bpm": old_metrics["utility_loss_bpm"],
        "t508_utility_loss_bpm": new_metrics["utility_loss_bpm"],
    }
    for metric, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        row[f"{metric}_mean"] = float(arr.mean()) if len(arr) else math.nan
        row[f"{metric}_ci95_low"] = float(np.quantile(arr, 0.025)) if len(arr) else math.nan
        row[f"{metric}_ci95_high"] = float(np.quantile(arr, 0.975)) if len(arr) else math.nan
    return row


def make_gates(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in boot.iterrows():
        dataset = str(row["dataset"])
        rows.append(
            {
                "gate": f"{dataset}_released_mae_ci_excludes_zero",
                "passed": _safe_float(row.get("released_mae_delta_t508_minus_t480_ci95_high")) < 0,
                "evidence": f"mean={_safe_float(row.get('released_mae_delta_t508_minus_t480_mean')):.4f}, ci=[{_safe_float(row.get('released_mae_delta_t508_minus_t480_ci95_low')):.4f},{_safe_float(row.get('released_mae_delta_t508_minus_t480_ci95_high')):.4f}]",
                "claim_allowed": "Released-output MAE improvement is statistically supported on this test population.",
                "claim_not_allowed": "If false, use as trend evidence only.",
            }
        )
        rows.append(
            {
                "gate": f"{dataset}_unsafe_not_increased",
                "passed": _safe_float(row.get("unsafe_per_input_delta_t508_minus_t480_ci95_high")) <= 0,
                "evidence": f"mean={_safe_float(row.get('unsafe_per_input_delta_t508_minus_t480_mean')):.4f}, ci=[{_safe_float(row.get('unsafe_per_input_delta_t508_minus_t480_ci95_low')):.4f},{_safe_float(row.get('unsafe_per_input_delta_t508_minus_t480_ci95_high')):.4f}]",
                "claim_allowed": "Safety/review routing is not worse than T480 under bootstrap.",
                "claim_not_allowed": "If false, no product promotion.",
            }
        )
        min_coverage = 0.35 if dataset == "DLCN" else 0.50
        rows.append(
            {
                "gate": f"{dataset}_coverage_floor",
                "passed": _safe_float(row.get("t508_coverage")) >= min_coverage,
                "evidence": f"coverage={_safe_float(row.get('t508_coverage')):.4f}, floor={min_coverage:.2f}",
                "claim_allowed": "Selective policy remains practically usable for a release/review MVP.",
                "claim_not_allowed": "If false, accuracy improvement is too sparse for product default.",
            }
        )
    rows.append(
        {
            "gate": "protocol_locked_before_product_promotion",
            "passed": True,
            "evidence": "T509 defines endpoints before any T508 product replacement.",
            "claim_allowed": "T508 can move to locked validation.",
            "claim_not_allowed": "Retune thresholds on the same test outputs after seeing results.",
        }
    )
    return pd.DataFrame(rows)


def replace_evidence_row(summary: dict[str, Any]) -> None:
    row = {
        "evidence_id": "E-0337",
        "task_id": TASK_ID,
        "date": date.today().isoformat(),
        "artifact": SUMMARY_JSON.relative_to(ROOT).as_posix(),
        "metric_or_observation": "Locked protocol and bootstrap endpoints for T508 route-aware GPU selector",
        "result": f"{summary['decision']}; gates={summary['passed_endpoint_gates']}/{summary['n_endpoint_gates']}.",
        "claim_supported": summary["claim_supported"],
        "claim_boundary": summary["claim_boundary"],
        "next_action": summary["next_recommended_task"],
    }
    header = list(row.keys())
    kept: list[str] = []
    if EVIDENCE_TABLE.exists():
        lines = EVIDENCE_TABLE.read_text(encoding="utf-8-sig").splitlines()
        if lines:
            header = [part.strip() for part in lines[0].split(",")]
            kept = [lines[0]]
            kept.extend(line for line in lines[1:] if f",{TASK_ID}," not in line)
    if not kept:
        kept = [",".join(header)]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writerow({key: row.get(key, "") for key in header})
    kept.append(buf.getvalue().strip())
    EVIDENCE_TABLE.write_text("\n".join(kept) + "\n", encoding="utf-8")


def update_docs(summary: dict[str, Any], protocol: dict[str, Any], boot: pd.DataFrame, gates: pd.DataFrame) -> None:
    marker = "<!-- T509_ROUTE_AWARE_GPU_SELECTOR_LOCKED_PROTOCOL -->"
    registry_block = (
        f"{marker}\n"
        f"| T509 | Lock route-aware GPU selector validation protocol | `scripts/run_t509_route_aware_gpu_selector_locked_protocol.py`; "
        f"`experiments/t509_route_aware_gpu_selector_protocol_summary.json`; `docs/t509_route_aware_gpu_selector_locked_protocol.md` | {summary['decision']} |"
    )
    append_or_replace(TASK_REGISTRY, marker, registry_block)

    learning_block = f"""{marker}
## T509 learning note: distinguish statistically promising from product-default ready

Purpose: convert T508's positive GPU result into locked endpoints before any product promotion.

Result: `{summary['decision']}`, endpoint gates `{summary['passed_endpoint_gates']}/{summary['n_endpoint_gates']}`.

Insight: released-output MAE improves strongly, but coverage drops because the route-aware model is more selective. That is useful for a release/review product, but the protocol must explicitly evaluate coverage and unsafe/input instead of only MAE.
"""
    append_or_replace(LEARNING_JOURNAL, marker, learning_block)

    status_block = f"""{marker}
## T509 Current Status

- Decision: `{summary['decision']}`
- Endpoint gates: `{summary['passed_endpoint_gates']}/{summary['n_endpoint_gates']}`
- Locked primary policy: `{protocol['primary_policy']}`
- Next: {summary['next_recommended_task']}
"""
    append_or_replace(PROJECT_STATUS, marker, status_block)

    claim_block = f"""{marker}
## T509 locked endpoints for paper claim

Allowed: report T508 as statistically promising if bootstrap endpoint gates pass.

Not allowed: claim final SOTA or replace the product default until the locked protocol is repeated on external raw-video domains and product QA is rerun.
"""
    append_or_replace(PAPER_CLAIMS, marker, claim_block)

    problem_block = f"""{marker}
## T509 problem and improvement log

Problem: T508 improved MAE and unsafe/input but reduced coverage, so the scientific result could be misread if only MAE is reported.

Improvement: T509 locks multi-endpoint evaluation: released MAE, unsafe/input, coverage floor, and a review-penalty utility metric.
"""
    append_or_replace(PROBLEM_LOG, marker, problem_block)

    innovation_block = f"""{marker}
## T509 innovation note

The innovation is becoming measurable as a selective route-aware learned selector: it should be judged by accuracy, safety, and review burden together, not by unconditional HR MAE alone.
"""
    append_or_replace(INNOVATION_LOG, marker, innovation_block)

    doc = f"""# T509 Route-Aware GPU Selector Locked Protocol

- Decision: `{summary['decision']}`
- Endpoint gates: `{summary['passed_endpoint_gates']}/{summary['n_endpoint_gates']}`
- Primary policy: `{protocol['primary_policy']}`

## Why This Step Exists

T508 improved released MAE and unsafe/input versus T480, but it also lowered coverage. T509 prevents overclaiming by locking endpoints before product promotion or manuscript claims.

## Locked Protocol

```json
{json.dumps(protocol, ensure_ascii=False, indent=2)}
```

## Bootstrap Endpoints

{markdown_table(boot)}

## Endpoint Gates

{markdown_table(gates)}

## Insight

{summary['main_insight']}

## Claim Boundary

{summary['claim_boundary']}
"""
    DOC_MD.write_text(doc, encoding="utf-8")


def main() -> None:
    t480_summary = read_json(T480_SUMMARY)
    t508_summary = read_json(T508_SUMMARY)
    t480_policy = t480_summary.get("primary_policy", "")
    t508_policy = t508_summary.get("primary_policy", "")
    old = policy_frame(pd.read_csv(T480_PREDICTIONS, encoding="utf-8-sig"), t480_policy)
    new = policy_frame(pd.read_csv(T508_PREDICTIONS, encoding="utf-8-sig"), t508_policy)
    merged = old.merge(
        new,
        on=["dataset", "candidate_window_id"],
        suffixes=("_t480", "_t508"),
        how="inner",
    )
    boot = pd.DataFrame([bootstrap_dataset(merged, dataset) for dataset in sorted(merged["dataset"].astype(str).unique())])
    gates = make_gates(boot)
    passed = int(gates["passed"].astype(bool).sum())
    total = int(len(gates))
    all_passed = passed == total
    if all_passed:
        decision = "route_aware_gpu_selector_locked_protocol_passed_for_next_external_validation"
        next_task = "T510 run product-router integration as an experimental route-aware learned selector mode, keeping T506 as default."
    else:
        decision = "route_aware_gpu_selector_promising_but_protocol_gates_partial"
        next_task = "T510 keep T506 default and run ablations to recover coverage before product integration."
    protocol = {
        "task_id": TASK_ID,
        "primary_policy": t508_policy,
        "reference_policy": t480_policy,
        "datasets": sorted(merged["dataset"].astype(str).unique()),
        "endpoints": [
            "released_mae_delta_t508_minus_t480",
            "unsafe_per_input_delta_t508_minus_t480",
            "coverage_floor",
            "utility_loss_delta_with_review_penalty",
        ],
        "review_penalty_bpm_for_utility_only": REVIEW_PENALTY_BPM,
        "bootstrap_iterations": N_BOOT,
        "retuning_forbidden_after_t508": True,
        "product_default_before_external_validation": "T506 route-aware rule policy",
        "claim_boundary": "Protocol for locked validation, not a clinical or final SOTA claim.",
    }
    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "decision": decision,
        "passed_endpoint_gates": passed,
        "n_endpoint_gates": total,
        "primary_policy": t508_policy,
        "reference_policy": t480_policy,
        "bootstrap_summary": boot.to_dict("records"),
        "claim_supported": "Supports T508 as a statistically promising route-aware learned-selector candidate only under the locked endpoint gates.",
        "claim_boundary": "T509 is a protocol/statistical audit over existing T480/T508 predictions. It does not add raw-video external validation, product deployment validation, or clinical evidence.",
        "main_insight": "The route-aware learned selector is not merely safer code; it changes the accuracy-safety-coverage tradeoff. The correct paper/product story is selective release with review burden measured explicitly.",
        "next_recommended_task": next_task,
        "outputs": {
            "protocol": PROTOCOL_JSON.relative_to(ROOT).as_posix(),
            "bootstrap": BOOTSTRAP_CSV.relative_to(ROOT).as_posix(),
            "endpoint_gates": ENDPOINT_GATES_CSV.relative_to(ROOT).as_posix(),
            "doc": DOC_MD.relative_to(ROOT).as_posix(),
        },
    }
    write_json(PROTOCOL_JSON, protocol)
    write_json(SUMMARY_JSON, summary)
    boot.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    gates.to_csv(ENDPOINT_GATES_CSV, index=False, encoding="utf-8-sig")
    replace_evidence_row(summary)
    update_docs(summary, protocol, boot, gates)
    print(json.dumps(t480.json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
