from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PRODUCT_VERSION = "t737.deep_candidate_physio_gate.v1"
PRODUCT_LABEL = "Deep-Candidate PhysioGate"
CLAIM_BOUNDARY = (
    "Evidence supports bounded candidate-selection and release/review risk control on "
    "MCD-rPPG and UBFC-rPPG. It does not support universal SOTA, solved fairness, "
    "solved low-light/source-shift, or clinical-grade autonomous release."
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if pd.isna(x):
        return None
    return x


def _sample_cases(selection: pd.DataFrame, n_release: int = 2, n_review: int = 2) -> list[dict[str, Any]]:
    if selection.empty:
        return []
    rows = []
    supported = selection[selection["dataset"].isin(["MCD-rPPG", "UBFC-rPPG"])].copy()
    supported["abs_error_bpm_num"] = pd.to_numeric(supported.get("abs_error_bpm"), errors="coerce")
    supported["selected_score_num"] = pd.to_numeric(supported.get("selected_score"), errors="coerce")
    supported["unsafe_num"] = pd.to_numeric(supported.get("unsafe_candidate"), errors="coerce").fillna(0)

    def convert(row: pd.Series, decision: str) -> dict[str, Any]:
        selected_hr = _json_float(row.get("pred_hr_bpm"))
        gt_hr = _json_float(row.get("gt_hr_bpm"))
        err = _json_float(row.get("abs_error_bpm"))
        risk_reason = []
        if _json_float(row.get("harmonic_trap_score")) and _json_float(row.get("harmonic_trap_score")) > 0.5:
            risk_reason.append("harmonic/alias conflict")
        if _json_float(row.get("deep_disagreement_risk")) and _json_float(row.get("deep_disagreement_risk")) > 0.5:
            risk_reason.append("deep-candidate disagreement")
        if err is not None and err > 10:
            risk_reason.append("candidate evidence indicates high release risk")
        if not risk_reason and decision == "release":
            risk_reason.append("candidate evidence passed current release gate")
        elif not risk_reason:
            risk_reason.append("candidate evidence requires review")
        return {
            "product_version": PRODUCT_VERSION,
            "sample_id": str(row.get("sample_id", "")),
            "dataset": str(row.get("dataset", "")),
            "decision": decision,
            "selected_hr_bpm": selected_hr if decision == "release" else None,
            "candidate_hr_bpm": selected_hr,
            "reference_hr_bpm_for_audit_only": gt_hr,
            "abs_error_bpm_for_audit_only": err,
            "candidate_source": str(row.get("source_type", "")),
            "candidate_family": str(row.get("candidate_family", "")),
            "selector_variant": str(row.get("variant", "")),
            "selected_score": _json_float(row.get("selected_score")),
            "support_count": _json_float(row.get("support_count")),
            "agreement10_frac": _json_float(row.get("agreement10_frac")),
            "harmonic_trap_score": _json_float(row.get("harmonic_trap_score")),
            "deep_disagreement_risk": _json_float(row.get("deep_disagreement_risk")),
            "review_reason": "; ".join(risk_reason),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    for dataset in ["MCD-rPPG", "UBFC-rPPG"]:
        sub = supported[supported["dataset"].eq(dataset)].copy()
        release = sub[
            (sub["variant"].isin(["rule_good5", "extra_trees_good5", "rule_good5_bidirectional_harmonic"]))
            & sub["abs_error_bpm_num"].le(10)
        ].sort_values(["abs_error_bpm_num", "selected_score_num"], ascending=[True, False])
        release = release.drop_duplicates(subset=["sample_id"], keep="first")
        review = sub[
            (sub["abs_error_bpm_num"].gt(10))
            | (sub["unsafe_num"].gt(0))
        ].sort_values(["abs_error_bpm_num", "selected_score_num"], ascending=[False, True])
        review = review.drop_duplicates(subset=["sample_id"], keep="first")
        for _, row in release.head(n_release).iterrows():
            rows.append(convert(row, "release"))
        for _, row in review.head(n_review).iterrows():
            rows.append(convert(row, "review"))
    return rows


def build_product_bundle(project_root: Path) -> dict[str, Any]:
    exp = project_root / "experiments"
    bundles = project_root / "remote_metric_bundles"
    fig = project_root / "output" / "figures" / "t735_physio_gate"
    claim_boundary = _read_csv(exp / "t734_table6_claim_boundary.csv")
    release = _read_csv(exp / "t734_table3_release_gate.csv")
    main_acc = _read_csv(exp / "t734_table2_main_accuracy.csv")
    evidence = _read_csv(exp / "t733_current_evidence_lock.csv")
    selection = _read_csv(bundles / "t731_20260627" / "t731_candidate_ranker_selections.csv")

    api_packet = {
        "product_version": PRODUCT_VERSION,
        "product_label": PRODUCT_LABEL,
        "claim_boundary": CLAIM_BOUNDARY,
        "headline_metrics": {
            "mcd_mae_reduction_vs_max_snr": _json_float(
                release[(release.dataset == "MCD-rPPG") & (release.variant == "rule_full")][
                    "mae_reduction_vs_max_snr"
                ].iloc[0]
            )
            if not release.empty
            else None,
            "mcd_release_coverage": _json_float(
                release[(release.dataset == "MCD-rPPG") & (release.variant == "rule_full")][
                    "best_safe_gate_coverage"
                ].iloc[0]
            )
            if not release.empty
            else None,
            "mcd_unsafe_release_rate": _json_float(
                release[(release.dataset == "MCD-rPPG") & (release.variant == "rule_full")][
                    "best_safe_gate_unsafe"
                ].iloc[0]
            )
            if not release.empty
            else None,
        },
        "workflow": [
            "build candidate pool from ROI/patch/classical/deep routes",
            "score physiological candidate evidence and harmonic/alias risk",
            "select candidate or route to review",
            "release only when calibrated risk is below threshold",
            "export audit packet with candidate evidence and claim boundary",
        ],
        "figures": {
            "evidence_summary_png": str(fig / "t735_fig1_evidence_summary.png"),
            "claim_boundary_png": str(fig / "t735_fig2_claim_boundary.png"),
        },
        "tables": {
            "main_accuracy": "experiments/t734_table2_main_accuracy.csv",
            "release_gate": "experiments/t734_table3_release_gate.csv",
            "bootstrap_ci": "experiments/t734_table4_bootstrap_ci.csv",
            "claim_boundary": "experiments/t734_table6_claim_boundary.csv",
        },
        "cases": _sample_cases(selection),
        "claim_boundary_rows": claim_boundary.to_dict(orient="records") if not claim_boundary.empty else [],
        "evidence_rows": evidence.to_dict(orient="records") if not evidence.empty else [],
        "main_accuracy_rows": main_acc.to_dict(orient="records") if not main_acc.empty else [],
    }
    return api_packet


def write_product_bundle(project_root: Path) -> dict[str, Any]:
    exp = project_root / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    packet = build_product_bundle(project_root)
    (exp / "t737_deep_candidate_physio_gate_api_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cases = pd.DataFrame(packet["cases"])
    cases.to_csv(exp / "t737_deep_candidate_physio_gate_cases.csv", index=False, encoding="utf-8-sig")
    return packet
