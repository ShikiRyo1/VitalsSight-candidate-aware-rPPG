from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"


def read_json(name: str) -> dict[str, Any]:
    path = EXP / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(name: str) -> list[dict[str, str]]:
    path = EXP / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def latest_epoch_from_log() -> dict[str, Any]:
    rows = read_csv_rows("t587_dlcn_end_to_end_epoch_log.csv")
    if not rows:
        return {}
    best_val: float | None = None
    latest: dict[str, str] = rows[-1]
    for row in rows:
        try:
            val = float(row.get("val_loss", "nan"))
        except ValueError:
            continue
        if best_val is None or val < best_val:
            best_val = val
    try:
        epoch = int(float(latest.get("epoch", "0")))
    except ValueError:
        epoch = None
    return {
        "current_epoch": epoch,
        "best_val_loss": best_val,
        "last_epoch": latest,
        "n_epoch_rows": len(rows),
        "source": "t587_dlcn_end_to_end_epoch_log.csv",
    }


def first_present(*values: Any, default: str = "missing") -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return str(value)
    return default


def lookup(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    for row in rows:
        if row.get("claim_dimension") == key:
            return row
    return {}


def fmt_delta(before: Any, after: Any, unit: str = "") -> str:
    if before is None or after is None:
        return "missing"
    try:
        before_f = float(before)
        after_f = float(after)
        delta = after_f - before_f
        return f"{before_f:.4f}->{after_f:.4f}{unit} (delta {delta:+.4f}{unit})"
    except (TypeError, ValueError):
        return f"{before}->{after}{unit}"


def metric_from_dict(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")

    t158 = read_json("t158_guarded_correction_validation_summary.json")
    t160 = read_json("t160_physio_consistency_rescue_summary.json")
    t480 = read_json("t480_unified_selector_summary.json")
    t508 = read_json("t508_route_aware_gpu_feature_audit_summary.json")
    t521 = read_json("t521_route_level_locked_statistical_validation_summary.json")
    t522 = read_json("t522_route_specific_ablation_negative_control_summary.json")
    t579 = read_json("t579_t577_deep_external_claim_gate_summary.json")
    t591 = read_json("t591_dlcn_condition_gate_summary.json")
    t618 = read_json("t618_mcd_cached_clip_training_summary.json")
    t619 = read_json("t619_mcd_cached_source_shift_bootstrap_gate_summary.json")
    t590_rows = read_csv_rows("t590_live_evidence_chain_matrix.csv")
    t605_rows = read_csv_rows("t605_t448_goal_gap_audit.csv")
    t587 = read_json("t587_dlcn_end_to_end_training_status.json")
    t587_epoch_log = latest_epoch_from_log()
    t587_current_epoch = t587_epoch_log.get("current_epoch", t587.get("current_epoch"))
    t587_best_val_loss = t587_epoch_log.get("best_val_loss", t587.get("best_val_loss"))

    m_dlcn = lookup(t590_rows, "DLCN_dynamic_illumination")
    m_mcd = lookup(t590_rows, "MCD_highHR_source_shift")
    m_fair = lookup(t590_rows, "fairness_lowlight_subgroup_claims")
    m_final = lookup(t590_rows, "final_SOTA_Q1_claim")

    rows: list[dict[str, str]] = [
        {
            "claim_id": "C1_candidate_pool_error_model",
            "mathematical_claim": "r_theta(c)=E[|b(c)-y| | x(c)] can rank multiple ROI/method/window/peak candidates by expected physiological error rather than by max FFT power alone.",
            "prior_pain_point": "Traditional rPPG and several deep baselines can select harmonic, alias, motion, or lighting peaks when the target frequency is not the largest peak.",
            "current_support": "T448/T603 lock the candidate-risk formulation; T158/T160 show guarded candidate correction and rescue evidence on existing candidate tables.",
            "key_metrics": first_present(t158.get("claim_boundary"), t160.get("claim_boundary")),
            "evidence_strength": "PARTIAL",
            "remaining_gap": "Need final raw-video prediction rows connected to candidate-risk decisions, not only feature-table replay.",
            "next_experiment": "After T587/T618 predictions, rebuild candidate packets and compare risk-ranked selector vs max-peak, POS, CHROM, and deep direct regressors.",
        },
        {
            "claim_id": "C2_release_review_risk_control",
            "mathematical_claim": "A selective policy pi(c)=release iff risk<=tau, margin>=gamma, and quality>=q_min can minimize unsafe released estimates under a coverage constraint.",
            "prior_pain_point": "Existing systems often always output a vital sign even under low SNR or candidate conflict, creating unsafe confident errors.",
            "current_support": "T158/T160/T521/T522 support route-aware guarded release as useful, especially unsafe-release reduction, but not final global superiority.",
            "key_metrics": "; ".join(
                [
                    f"T521={first_present(t521.get('decision'), t521.get('passed_gates'))}",
                    f"T522={first_present(t522.get('decision'), t522.get('passed_gates'))}",
                ]
            ),
            "evidence_strength": "MODERATE_BUT_NOT_FINAL",
            "remaining_gap": "Need paired CI/bootstrap on final raw-video outputs and worst-subgroup unsafe-release/coverage gates.",
            "next_experiment": "Run T594/T619 bootstrap gates and threshold sweeps over tau/gamma/q_min; report coverage, MAE, unsafe/input, unsafe/released.",
        },
        {
            "claim_id": "C3_route_aware_beats_domain_blind",
            "mathematical_claim": "Adding route/domain/ROI/quality evidence reduces candidate-risk under distribution shift compared with a domain-blind selector.",
            "prior_pain_point": "A single global selector can collapse under source, lighting, camera, or activity shifts because reliability differs by route and condition.",
            "current_support": "T480 vs T508 suggests route-aware features reduce MAE and unsafe rate on DLCN/UBFC while trading coverage.",
            "key_metrics": first_present(
                t508.get("headline"),
                t508.get("claim_boundary"),
                t480.get("claim_boundary"),
                default="T480/T508 available; see CSV outputs",
            ),
            "evidence_strength": "PARTIAL",
            "remaining_gap": "Need route-aware improvements reproduced on raw-video deep predictions and compared with deep baselines under same split.",
            "next_experiment": "Run final route-aware selector on T587/T618/T570 outputs; add ablation without route/domain/quality features.",
        },
        {
            "claim_id": "C4_low_light_dynamic_illumination",
            "mathematical_claim": "Condition-aware candidate risk should reduce low-light/dynamic-light failure by treating illumination drift as an explicit risk variable.",
            "prior_pain_point": "Low-light and illumination changes distort color time series and can dominate the spectral peak.",
            "current_support": f"T587 running on DLCN; current_epoch={t587_current_epoch}; best_val_loss={t587_best_val_loss}; T591={first_present(t591.get('decision'), t591.get('claim_boundary'))}",
            "key_metrics": first_present(m_dlcn.get("evidence"), t591.get("metric")),
            "evidence_strength": "RUNNING_NOT_CLOSED",
            "remaining_gap": "T587 must complete; T591/T594 condition and bootstrap gates must be rerun from final predictions.",
            "next_experiment": "Let T587 finish 30 epochs, then run DLCN condition MAE/unsafe/coverage by illumination condition and bootstrap confidence intervals.",
        },
        {
            "claim_id": "C5_skin_fairness_boundary",
            "mathematical_claim": "Fairness can only be claimed if subgroup worst-case risk, coverage, and unsafe-release gaps are bounded with confidence intervals.",
            "prior_pain_point": "Public rPPG datasets and models can show demographic imbalance and subgroup performance gaps.",
            "current_support": "Current CMU/fairness replay supports measurable boundary audit and review-route design, not solved fairness.",
            "key_metrics": first_present(m_fair.get("evidence"), "fairness gate not closed"),
            "evidence_strength": "NOT_MET",
            "remaining_gap": "Worst-subgroup gate still fails or is not strong enough for solved-fairness wording.",
            "next_experiment": "Run subgroup risk-control threshold sweeps; target a narrower claim: reduced unsafe auto-release under subgroup-aware review policy.",
        },
        {
            "claim_id": "C6_high_hr_source_shift_low_snr",
            "mathematical_claim": "Route-aware candidate selection should distinguish true high-HR/post-exercise trajectories from source-shift and low-SNR artifacts.",
            "prior_pain_point": "High heart rate, motion, exercise recovery, camera shift, and low SNR can generate wrong-but-plausible peak candidates.",
            "current_support": first_present(m_mcd.get("evidence"), "MCD source-shift branch pending"),
            "key_metrics": first_present(m_mcd.get("gate_status"), "pending"),
            "evidence_strength": "NOT_MET",
            "remaining_gap": f"MCD T618/T619 full raw-video cached training/evaluation status: T618={t618.get('decision', 'missing')}; T619={t619.get('decision', 'missing')}.",
            "next_experiment": "Finish MCD cached full raw-video source-shift/high-HR/low-SNR training, then route-level bootstrap and subgroup gates.",
        },
        {
            "claim_id": "C7_synthetic_pretraining_transfer",
            "mathematical_claim": "Synthetic pretraining is useful only if it improves real-dataset risk and robustness after transfer.",
            "prior_pain_point": "Synthetic signals can improve representation coverage but can also overfit to simulator artifacts.",
            "current_support": "SCAMPS is still downloading; no synthetic-to-real training conclusion yet.",
            "key_metrics": "SCAMPS not yet in training",
            "evidence_strength": "PENDING",
            "remaining_gap": "Download completion, integrity check, pretraining, real-data fine-tune, and transfer ablation are not complete.",
            "next_experiment": "After SCAMPS completes, run pretrain -> fine-tune -> no-pretrain ablation on DLCN/MCD/UBFC.",
        },
        {
            "claim_id": "C8_product_evidence_binding",
            "mathematical_claim": "A market-facing product must bind model uncertainty and release/review policy into explicit user workflows, not just display a number.",
            "prior_pain_point": "Many monitoring dashboards hide uncertainty and provide weak evidence for why a value is safe to act on.",
            "current_support": "UI shell exists but user-reported functional gaps remain; final model/policy outputs are not fully wired into every workflow.",
            "key_metrics": "product QA not final",
            "evidence_strength": "NOT_MET",
            "remaining_gap": "Need complete upload/scan/review/report/API/bilingual/theme/focus-case interactions and backend evidence integration.",
            "next_experiment": "After policy outputs stabilize, run product E2E tests and repair every non-clickable or stale workspace/module path.",
        },
        {
            "claim_id": "C9_q1_manuscript_evidence_chain",
            "mathematical_claim": "A manuscript claim is admissible only when claim, theory, dataset use, protocol, metric, statistics, ablation, and reviewer-risk boundary align.",
            "prior_pain_point": "Overbroad rPPG claims are vulnerable if robustness evidence is only anecdotal or split protocols are inconsistent.",
            "current_support": first_present(m_final.get("evidence"), t579.get("claim_boundary"), "final gate not closed"),
            "key_metrics": first_present(m_final.get("gate_status"), t579.get("decision"), "not ready"),
            "evidence_strength": "NOT_MET",
            "remaining_gap": "Final Q1/SOTA/manuscript wording cannot be promoted until the downstream gates close.",
            "next_experiment": "Refresh T590 after every final experiment; generate paper figures/tables only from locked outputs.",
        },
    ]

    out_csv = EXP / "t606_selective_candidate_risk_evidence_table.csv"
    out_json = EXP / "t606_selective_candidate_risk_evidence_summary.json"
    out_md = DOCS / "t606_selective_candidate_risk_evidence_table.md"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["evidence_strength"]] = counts.get(row["evidence_strength"], 0) + 1
    unmet = sum(v for k, v in counts.items() if k in {"NOT_MET", "PENDING", "RUNNING_NOT_CLOSED"})
    summary = {
        "task_id": "T606",
        "generated_at": generated_at,
        "decision": "core_algorithm_claim_is_selective_candidate_risk_control_not_explainability",
        "claim_boundary": "Theoretical contribution is now explicit, but final T448/Q1 evidence remains incomplete until T587/T618/SCAMPS/subgroup/product gates pass.",
        "evidence_strength_counts": counts,
        "n_claim_rows": len(rows),
        "n_unclosed_rows": unmet,
        "active_training_context": {
            "t587_current_epoch": t587_current_epoch,
            "t587_best_val_loss": t587_best_val_loss,
            "t587_stage": t587.get("stage"),
            "t587_epoch_log": t587_epoch_log,
        },
        "outputs": {
            "csv": str(out_csv),
            "json": str(out_json),
            "doc": str(out_md),
        },
        "next_actions": [
            "Continue T587 to completion and rerun T591/T594.",
            "Run/finish MCD T618/T619 full raw-video cached branch.",
            "Finish SCAMPS download, integrity check, pretrain/fine-tune/ablation.",
            "Run candidate-risk ablations: no ROI, no route, no temporal, no harmonic/alias, no quality, no uncertainty.",
            "Repair product E2E functionality after final policy outputs stabilize.",
            "Refresh manuscript claim gates only from locked metrics.",
        ],
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# T606 Selective Candidate-Risk Evidence Table",
        "",
        f"Generated: {generated_at}",
        "",
        "Core correction: the manuscript/product contribution is not merely explainability. The central algorithmic claim is physiology-constrained selective candidate-risk control: generate multiple plausible HR candidates, estimate their risk using physiological and domain evidence, and release only when the selected candidate is safe enough.",
        "",
        f"Claim boundary: {summary['claim_boundary']}",
        "",
        "## Evidence Rows",
        "",
        "| Claim | Evidence Strength | Mathematical Claim | Current Support | Remaining Gap | Next Experiment |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["claim_id"],
                    row["evidence_strength"],
                    row["mathematical_claim"].replace("|", "/"),
                    row["current_support"].replace("|", "/"),
                    row["remaining_gap"].replace("|", "/"),
                    row["next_experiment"].replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Immediate Execution Logic",
            "",
            "1. Do not claim SOTA, solved fairness, solved low-light, or final Q1 readiness yet.",
            "2. Use T587/T618/SCAMPS outputs to convert the current candidate-risk theory into raw-video evidence.",
            "3. Promote only narrow claims that pass subgroup, bootstrap, ablation, and product workflow gates.",
            "4. If a dimension remains weak, keep it as a review/reject boundary rather than a solved claim.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
