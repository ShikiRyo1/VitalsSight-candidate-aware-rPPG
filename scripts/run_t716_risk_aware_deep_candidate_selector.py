from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_t712_mcd_deep_augmented_gate as base  # noqa: E402


TASK_ID = "T716"
UNSAFE_BPM = 10.0

base.TASK_ID = TASK_ID
base.T618_MCD_DEEP_PREDS = base.EXP / "t713_mcd_deep_candidate_diagnostic_predictions.csv"

base.OUT_POOL = base.EXP / "t716_risk_aware_deep_candidate_pool.csv"
base.OUT_PREDS = base.EXP / "t716_risk_aware_selector_predictions.csv"
base.OUT_METRICS = base.EXP / "t716_risk_aware_selector_metrics.csv"
base.OUT_BASELINES = base.EXP / "t716_risk_aware_baseline_metrics.csv"
base.OUT_RELEASE = base.EXP / "t716_risk_aware_release_gate.csv"
base.OUT_CLAIM = base.EXP / "t716_risk_aware_claim_gate.csv"
base.OUT_FAILURE = base.EXP / "t716_risk_aware_failure_taxonomy.csv"
base.OUT_SUMMARY = base.EXP / "t716_risk_aware_selector_gate_summary.json"
base.OUT_MD = base.DOCS / "t716_risk_aware_deep_candidate_selector.md"


def risk_aware_selector_loss(
    scores: torch.Tensor,
    group,
    *,
    temp: float = 2.5,
    err_scale: float = 20.0,
    unsafe_weight: float = 0.55,
    barrier_weight: float = 1.75,
) -> torch.Tensor:
    errors = torch.nan_to_num(group.errors, nan=100.0, posinf=100.0, neginf=100.0)
    barrier = torch.relu(errors - UNSAFE_BPM)
    adjusted_errors = errors + barrier_weight * barrier
    q = F.softmax(-adjusted_errors / temp, dim=0).detach()
    logp = F.log_softmax(scores, dim=0)
    p = F.softmax(scores, dim=0)
    kl = F.kl_div(logp, q, reduction="batchmean")
    expected_error = torch.sum(p * errors) / err_scale
    unsafe_risk = torch.sigmoid((errors - UNSAFE_BPM) / 2.0)
    expected_unsafe = torch.sum(p * unsafe_risk)
    best_idx = torch.argmin(adjusted_errors)
    margin = torch.relu(1.0 - scores[best_idx] + scores).mean()
    harmonic_risk = torch.tensor(pd.to_numeric(group.meta["harmonic_risk"], errors="coerce").fillna(0.0).to_numpy(np.float32), device=scores.device)
    harmonic_penalty = torch.sum(p * torch.clamp(harmonic_risk, min=0.0)) * 0.02
    return kl + expected_error + unsafe_weight * expected_unsafe + 0.10 * margin + harmonic_penalty


base.selector_loss = risk_aware_selector_loss


def write_report(summary, base_metrics, metrics, claim, fail) -> None:
    lines = [
        "# T716 Risk-Aware Deep-Candidate Selector",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T714 showed that full MCD deep-candidate coverage can sharply reduce MAE, but unsafe release remains too high. T716 changes the selector objective from pure error imitation to physiology/risk-constrained candidate learning by adding an explicit soft barrier above the 10 BPM unsafe boundary.",
        "",
        "## Selector Objective",
        "",
        "`L = KL(p || q_adjusted) + E_p[|HR_i-y|]/20 + 0.55 E_p[sigmoid((|HR_i-y|-10)/2)] + margin + harmonic_penalty`",
        "",
        "The adjusted target distribution penalizes candidates beyond the unsafe boundary more strongly than ordinary MAE. This is a diagnostic implementation of the planned risk-aware selector contribution.",
        "",
        "## Baselines",
        "",
        base.markdown_table(base_metrics),
        "",
        "## Selector Metrics",
        "",
        base.markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        base.markdown_table(claim),
        "",
        "## Failure Taxonomy",
        "",
        base.markdown_table(fail),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "T716 still uses T713 diagnostic deep predictions. If it passes or nearly passes, the next paper-safe step is fold-safe/OOF deep prediction generation and the same risk-aware selector under a locked protocol.",
    ]
    base.OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.write_report = write_report


if __name__ == "__main__":
    raise SystemExit(base.main())
