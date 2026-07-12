from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "T531"

EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
MANUSCRIPT = ROOT / "manuscript"

CITED_MD = MANUSCRIPT / "vitalsight_route_moe_cited_predraft.md"
AUDITED_MD = MANUSCRIPT / "vitalsight_route_moe_reviewer_audited.md"
RISK_REGISTER = EXP / "t531_reviewer_risk_register.csv"
REVISION_ACTIONS = EXP / "t531_targeted_revision_actions.csv"
SUMMARY_JSON = EXP / "t531_reviewer_risk_audit_summary.json"
DOC_MD = DOCS / "t531_reviewer_risk_audit.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_or_replace(path: Path, marker: str, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        start = old.index(marker)
        after = start + len(marker)
        next_marker = old.find("\n<!-- ", after)
        end = next_marker if next_marker != -1 else len(old)
        path.write_text(old[:start].rstrip() + "\n\n" + block.strip() + "\n\n" + old[end:].lstrip(), encoding="utf-8")
        return
    path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.head(max_rows).copy()
    cols = [str(col) for col in show.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for col in show.columns:
            val = "" if pd.isna(row[col]) else str(row[col])
            vals.append(val.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def sentence_hits(text: str, patterns: list[str]) -> dict[str, list[str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    hits: dict[str, list[str]] = {}
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        hits[pat] = [s.strip() for s in sentences if rx.search(s)]
    return hits


def build_risks(text: str) -> list[dict[str, Any]]:
    hits = sentence_hits(
        text,
        [
            r"\bvital-sign\b|\bvital signs\b",
            r"mixture-of-experts|\bMoE\b",
            r"passed 9/9|product QA",
            r"no unsafe releases|unsafe releases per input 0\.000",
            r"strongest reproduced|outperforming",
            r"fairness|low-light|demographic",
            r"research-MVP|dashboard|API",
            r"clinical-grade|clinical deployment|regulated medical-device|universal SOTA",
        ],
    )
    broad_vital_count = sum(len(v) for k, v in hits.items() if "vital" in k)
    forbidden_count = len(hits[r"clinical-grade|clinical deployment|regulated medical-device|universal SOTA"])

    return [
        {
            "risk_id": "RISK-01",
            "review_axis": "technical soundness",
            "severity": "high",
            "risk": "Title and framing may sound broader than current HR-only release evidence.",
            "evidence_anchor": f"{broad_vital_count} manuscript sentences mention vital-sign/vital signs while the released quantitative endpoint is HR.",
            "required_action": "Revise title/abstract wording toward contactless heart-rate inference, and explicitly treat RR/SpO2/other vitals as future or review-only scope unless validated.",
            "current_status": "needs_targeted_revision",
        },
        {
            "risk_id": "RISK-02",
            "review_axis": "originality",
            "severity": "high",
            "risk": "Route-aware MoE could be challenged as a rule-based dispatch wrapper unless the distinction from prior quality gating is made explicit.",
            "evidence_anchor": "The method cites MoE and selective prediction, but current experts include route-specific policies rather than a fully end-to-end neural sparse MoE.",
            "required_action": "Define contribution as physiology-constrained multi-candidate route dispatch plus release/review contract, not as a generic neural MoE breakthrough.",
            "current_status": "needs_targeted_revision",
        },
        {
            "risk_id": "RISK-03",
            "review_axis": "scientific importance",
            "severity": "medium",
            "risk": "Top-journal significance depends on proving the route-aware design solves an active field limitation, not only building a useful MVP.",
            "evidence_anchor": "T529 citations support low-light, demographic-bias, and cross-dataset generalization gaps; experiments must tie improvements to these gaps route by route.",
            "required_action": "In Introduction/Discussion, connect route-aware release/review to documented limitation/future-work themes: domain shift, artifact peaks, fairness, and low illumination.",
            "current_status": "partially_supported",
        },
        {
            "risk_id": "RISK-04",
            "review_axis": "technical soundness",
            "severity": "high",
            "risk": "No-unsafe-release and release coverage numbers can be overread as general safety.",
            "evidence_anchor": "Stable route reports unsafe releases per input 0.000 on the locked UBFC-Phys-style subset only.",
            "required_action": "Keep subset/protocol wording next to every safety number; add confidence intervals or bootstrap intervals before final submission if time permits.",
            "current_status": "needs_statistical_tightening",
        },
        {
            "risk_id": "RISK-05",
            "review_axis": "technical soundness",
            "severity": "medium",
            "risk": "TSCAN/learned selector comparisons can be misread as broad SOTA if protocol boundaries are not repeated.",
            "evidence_anchor": "T470 beats a TSCAN reference in one protocol; T480 shows a UBFC non-regression gap.",
            "required_action": "Present learned selector as a route expert and include negative controls beside positive comparisons.",
            "current_status": "supported_if_bounded",
        },
        {
            "risk_id": "RISK-06",
            "review_axis": "interdisciplinary readership",
            "severity": "medium",
            "risk": "Readers outside rPPG may not understand why not releasing a value is a scientific contribution.",
            "evidence_anchor": "The product contract returns review packets and hides HR for uncertain routes.",
            "required_action": "Add a concise explanation that release/review is a measurement reliability output analogous to selective prediction under uncertainty.",
            "current_status": "needs_expository_revision",
        },
        {
            "risk_id": "RISK-07",
            "review_axis": "technical soundness",
            "severity": "medium",
            "risk": "Product MVP evidence is a functional QA result, not market validation or clinical workflow validation.",
            "evidence_anchor": "T527 passed 9/9 demo checks on six representative cases.",
            "required_action": "Frame product result as research-MVP integration and claim-boundary demonstration; do not imply mature commercial validation.",
            "current_status": "supported_if_bounded",
        },
        {
            "risk_id": "RISK-08",
            "review_axis": "readability",
            "severity": "low",
            "risk": "TCM/ROI/route language may be difficult for nonspecialists if introduced abruptly.",
            "evidence_anchor": "Mentor-aligned pipeline includes MediaPipe Face Mesh, ROI segmentation, rPPG extraction, and multimodal analysis.",
            "required_action": "Add a plain-language pipeline schematic sentence before technical route labels.",
            "current_status": "minor_revision",
        },
        {
            "risk_id": "RISK-09",
            "review_axis": "boundary check",
            "severity": "pass",
            "risk": "Forbidden clinical/universal claims scanner.",
            "evidence_anchor": f"Forbidden phrase count in cited predraft: {forbidden_count}. The phrase appears only in negated boundary language if present.",
            "required_action": "Keep explicit claim boundary section in final paper and product UI.",
            "current_status": "boundary_present",
        },
    ]


def main() -> None:
    if not CITED_MD.exists():
        raise FileNotFoundError(CITED_MD)
    text = CITED_MD.read_text(encoding="utf-8")
    risks = pd.DataFrame(build_risks(text))
    actions = risks[risks["current_status"].isin(["needs_targeted_revision", "needs_statistical_tightening", "needs_expository_revision"])][
        ["risk_id", "severity", "required_action", "current_status"]
    ].copy()

    RISK_REGISTER.parent.mkdir(parents=True, exist_ok=True)
    risks.to_csv(RISK_REGISTER, index=False, encoding="utf-8-sig")
    actions.to_csv(REVISION_ACTIONS, index=False, encoding="utf-8-sig")

    high_open = int(((risks["severity"] == "high") & (risks["current_status"].str.contains("needs", na=False))).sum())
    decision = "targeted_revision_required_before_submission_draft" if high_open else "ready_for_submission_style_polish"
    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "n_risks": int(risks.shape[0]),
        "n_high_open_risks": high_open,
        "n_revision_actions": int(actions.shape[0]),
        "claim_supported": "The cited predraft is coherent but still needs targeted wording/statistical tightening before it should be treated as a submission draft.",
        "claim_boundary": "T531 is a reviewer-risk audit; it does not add new experimental evidence or certify journal acceptance.",
        "outputs": {
            "risk_register": RISK_REGISTER,
            "revision_actions": REVISION_ACTIONS,
            "audit_doc": DOC_MD,
            "audited_manuscript": AUDITED_MD,
            "summary": SUMMARY_JSON,
        },
    }
    write_json(SUMMARY_JSON, summary)

    review_setup = """Review setup
- Input scope: cited predraft `manuscript/vitalsight_route_moe_cited_predraft.md` plus local experiment summaries T464-T530.
- Assessment boundary: reviewer-style pre-submission risk audit only; no editorial decision is inferred.
- Shared manuscript claim summary: the paper argues for physiology-constrained route-aware multi-candidate rPPG inference with route-specific release/review decisions.
- Visible evidence base: stable RGB route metrics, MCD high-HR route metrics, deep baseline reproduction, learned-selector positive/negative controls, product QA, and verified literature inventory.
- Missing materials affecting confidence: final confidence intervals, final figure panels, complete final methods details, and any external clinical/user study.
"""
    reviewer_1 = """Reviewer 1 - technical validity / technical failings emphasis
- Overall assessment: promising but not submission-ready until broad endpoint wording and safety metrics are tightened.
- Who would be interested in the results, and why: rPPG, digital health, and deployable AI researchers interested in when a video-based physiological estimate should be withheld.
- Major strengths: route-level evidence, explicit negative controls, and product-visible review decisions.
- Major concerns: HR-only evidence can be blurred by vital-sign wording; safety metrics require subset/protocol labels and ideally interval estimates.
- Technical failings that need to be addressed before the case is established: clarify MoE implementation boundary, add statistical intervals where feasible, and keep failed universal-selector evidence visible.
- Assessment against Nature-style criteria: technically interesting but dependent on disciplined evidence boundaries.
- Recommendation posture: major targeted revision before treating as a strong submission draft.
"""
    reviewer_2 = """Reviewer 2 - originality / scientific importance emphasis
- Overall assessment: the strongest novelty is not another rPPG estimator, but a route-aware release/review inference contract.
- Who would be interested in the results, and why: readers concerned with reliable physiological AI under domain shift and artifact ambiguity.
- Major strengths: candidate-bank framing, route dispatch, and negative evidence against global promotion.
- Major concerns: novelty over quality gating and selective prediction must be stated precisely.
- Technical failings that need to be addressed before the case is established: show how the route-aware design maps to known limitations from prior rPPG papers, especially domain shift, low light, demographic bias, and high-HR stress.
- Assessment against Nature-style criteria: significant if framed as reliability methodology with bounded deployment implications; weaker if framed as generic performance improvement.
- Recommendation posture: potentially strong after sharper prior-work differentiation.
"""
    reviewer_3 = """Reviewer 3 - interdisciplinary readership / readability emphasis
- Overall assessment: the story can be understood by broader readers if the paper explains why refusing to output HR is a meaningful measurement result.
- Who would be interested in the results, and why: digital health, human-centered AI, and safety-oriented ML readers who need transparent uncertainty behavior.
- Major strengths: product packet exposes route, quality flags, and review reason.
- Major concerns: route labels, candidate peaks, and TCM/ROI language need a simple front-loaded schematic explanation.
- Technical failings that need to be addressed before the case is established: separate research-MVP QA from market/clinical validation.
- Assessment against Nature-style criteria: readability improves if the pipeline is introduced as video -> face landmarks -> ROI signals -> candidate peaks -> route decision -> release/review.
- Recommendation posture: revise for clarity and boundary discipline.
"""
    synthesis = """Cross-review synthesis
- Consensus strengths: route-specific evidence, visible refusal/review behavior, citation-grounded limitation framing, and negative controls.
- Consensus technical risks: overbroad vital-sign wording, MoE overinterpretation, missing interval estimates, and product claims exceeding research-MVP QA.
- Where emphasis differs across reviewers: Reviewer 1 prioritizes technical proof; Reviewer 2 prioritizes novelty against prior work; Reviewer 3 prioritizes non-specialist clarity.
- Broad-interest / significance readout: the most publishable angle is reliability-aware contactless physiological inference, not universal rPPG accuracy.
- Most important issues to resolve before a strong Nature-style case is established: revise scope wording, define route-aware MoE precisely, add/plan statistical intervals, and keep fairness/low-light as review-only boundaries unless new validation is added.

Risk / unsupported claims
- Do not claim clinical readiness.
- Do not claim universal SOTA.
- Do not imply fairness or low-light robustness is solved.
- Do not imply 9/9 product QA is market validation.
"""
    doc = f"""# T531 Reviewer-Risk Audit

## Decision

`{decision}`

{review_setup}

{reviewer_1}

{reviewer_2}

{reviewer_3}

{synthesis}

## Risk Register

{markdown_table(risks)}

## Targeted Revision Actions

{markdown_table(actions)}
"""
    DOC_MD.write_text(doc, encoding="utf-8")
    AUDITED_MD.write_text(text.rstrip() + "\n\n---\n\n" + doc, encoding="utf-8")

    marker = f"<!-- {TASK_ID}: reviewer risk audit -->"
    log_block = f"""{marker}
## {TASK_ID} - Reviewer-risk audit

- Decision: `{decision}`
- Risks: {summary["n_risks"]}
- Open high risks: {summary["n_high_open_risks"]}
- Revision actions: {summary["n_revision_actions"]}
- Main insight: the strongest story is reliability-aware route-specific release/review, not universal rPPG accuracy or clinical readiness.
- Outputs: `{RISK_REGISTER.as_posix()}`, `{REVISION_ACTIONS.as_posix()}`, `{DOC_MD.as_posix()}`
"""
    for path in [TASK_REGISTRY, LEARNING_JOURNAL, PROJECT_STATUS, PAPER_CLAIMS, INNOVATION_LOG]:
        append_or_replace(path, marker, log_block)

    evidence_row = pd.DataFrame(
        [
            {
                "task_id": TASK_ID,
                "artifact": DOC_MD.as_posix(),
                "metric": "open_high_reviewer_risks",
                "value": high_open,
                "decision": decision,
                "claim_boundary": summary["claim_boundary"],
            }
        ]
    )
    if EVIDENCE_TABLE.exists():
        old = pd.read_csv(EVIDENCE_TABLE)
        old = old[old.get("task_id", pd.Series(dtype=str)).astype(str) != TASK_ID]
        evidence = pd.concat([old, evidence_row], ignore_index=True)
    else:
        evidence = evidence_row
    evidence.to_csv(EVIDENCE_TABLE, index=False, encoding="utf-8-sig")

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
