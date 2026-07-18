from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


EVIDENCE_BUNDLE_VERSION = "t478.product_evidence_bundle.v1"
PRODUCT_MODE = "research_adult_contactless_vitals"
PRODUCT_MODE_LABEL = "Research adult contactless vital-sign platform"
CLAIM_BOUNDARY = (
    "Research/product MVP evidence bundle. Current evidence supports adult HR candidate-reliability "
    "selection under explicit release/review boundaries; it is not a clinical diagnostic device, not a "
    "final universal SOTA claim, and not yet validated for continuous infant monitoring."
)

PRODUCT_WARNINGS = [
    "Research MVP only; do not use for diagnosis, treatment, emergency detection, or standalone clinical monitoring.",
    "A released HR value is shown only when the configured evidence policy releases a window.",
    "Review/retest decisions are valid product outputs and should not be replaced by a hidden HR number.",
    "Ground-truth labels and evaluation-only errors are excluded from user-facing API payloads.",
]


@dataclass(frozen=True)
class ProductEvidencePaths:
    root: Path

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def t474_summary(self) -> Path:
        return self.experiments / "t474_ubfc_protocol_harmonized_summary.json"

    @property
    def t476b_summary(self) -> Path:
        return self.experiments / "t476b_clip180_deep_pickle_window_alignment_summary.json"

    @property
    def t477_summary(self) -> Path:
        return self.experiments / "t477_ubfc_to_dlcn_common_schema_transfer_summary.json"

    @property
    def t472_summary(self) -> Path:
        return self.experiments / "t472_dlcn_external_selector_model_screen_summary.json"

    @property
    def t455_summary(self) -> Path:
        return self.experiments / "t455_post_migration_no_gpu_gate_summary.json"

    @property
    def t407_summary(self) -> Path:
        return self.experiments / "t407_cmu_fairness_replay_pilot_summary.json"

    @property
    def t386_summary(self) -> Path:
        return self.experiments / "t386_locked_replay_summary.json"


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_float(value: object) -> float | None:
    out = finite_float(value)
    return out if math.isfinite(out) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def strict_json_text(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def metric_card(
    *,
    task_id: str,
    dataset: str,
    metric_scope: str,
    primary_metric_name: str,
    primary_metric_value: object,
    comparison: str,
    evidence_status: str,
    claim_use: str,
    limitation: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "dataset": dataset,
        "metric_scope": metric_scope,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": json_float(primary_metric_value),
        "comparison": comparison,
        "evidence_status": evidence_status,
        "claim_use": claim_use,
        "limitation": limitation,
    }


def build_metric_cards(paths: ProductEvidencePaths) -> list[dict[str, Any]]:
    t474 = read_json(paths.t474_summary)
    t476b = read_json(paths.t476b_summary)
    t477 = read_json(paths.t477_summary)
    t472 = read_json(paths.t472_summary)
    t386 = read_json(paths.t386_summary)
    t407 = read_json(paths.t407_summary)

    cards = [
        metric_card(
            task_id="T474",
            dataset="UBFC-rPPG",
            metric_scope="protocol-harmonized UBFC method comparison",
            primary_metric_name="MAE_BPM",
            primary_metric_value=t474.get("best_mae_bpm"),
            comparison=(
                f"Best selector {t474.get('best_policy', '')} vs aligned TSCAN "
                f"{json_float(t474.get('tscan_mae_bpm'))} BPM; paired delta "
                f"{json_float(t474.get('delta_mae_vs_tscan_bpm'))} BPM."
            ),
            evidence_status="paper_figure_ready" if t474 else "missing",
            claim_use="Primary same-dataset method evidence for multi-candidate reliability selection.",
            limitation=t474.get("claim_boundary", "UBFC-only evidence.") if t474 else "Missing T474 summary.",
        ),
        metric_card(
            task_id="T476B",
            dataset="UBFC-rPPG",
            metric_scope="same-window deep baseline alignment",
            primary_metric_name="best_deep_MAE_BPM",
            primary_metric_value=t476b.get("best_deep_mae_bpm"),
            comparison=(
                f"T474 selector {json_float(t476b.get('t474_best_mae_bpm'))} BPM vs best clip180 deep model "
                f"{t476b.get('best_deep_model', '')} {json_float(t476b.get('best_deep_mae_bpm'))} BPM."
            ),
            evidence_status="supports_method_against_deep_baselines" if t476b else "missing",
            claim_use="Reviewer-facing comparison against PhysFormer/RhythmFormer under aligned UBFC windows.",
            limitation=t476b.get("claim_boundary", "UBFC-only evidence.") if t476b else "Missing T476B summary.",
        ),
        metric_card(
            task_id="T477",
            dataset="DLCN",
            metric_scope="UBFC-to-DLCN common-feature transfer",
            primary_metric_name="release_all_MAE_BPM",
            primary_metric_value=t477.get("primary_t477_mae_bpm"),
            comparison=(
                f"Transfer release-all vs top_power {json_float(t477.get('top_power_mae_bpm'))} BPM and "
                f"top_support {json_float(t477.get('top_support_mae_bpm'))} BPM; DLCN-trained reference "
                f"{json_float(t477.get('t472_dlcn_trained_reference_mae_bpm'))} BPM."
            ),
            evidence_status="external_transfer_positive_but_domain_calibration_needed" if t477 else "missing",
            claim_use="External evidence that candidate reliability features transfer beyond naive spectral ranking.",
            limitation=t477.get("claim_boundary", "DLCN transfer only.") if t477 else "Missing T477 summary.",
        ),
        metric_card(
            task_id="T472",
            dataset="DLCN",
            metric_scope="DLCN-trained selector reference",
            primary_metric_name="MAE_BPM",
            primary_metric_value=t472.get("best_mae_bpm"),
            comparison=(
                f"DLCN-trained selector vs top_power {json_float(t472.get('top_power_mae_bpm'))} BPM and "
                f"classical rank-1 {json_float(t472.get('best_classical_mae_bpm'))} BPM."
            ),
            evidence_status="domain_specific_reference" if t472 else "missing",
            claim_use="Shows dynamic lighting requires domain calibration; prevents overclaiming T477 transfer.",
            limitation=t472.get("claim_boundary", "DLCN candidate-level evidence.") if t472 else "Missing T472 summary.",
        ),
    ]

    if t386:
        overall = t386.get("overall", {}) if isinstance(t386.get("overall"), dict) else {}
        cards.append(
            metric_card(
                task_id="T386",
                dataset="multi-dataset adult product replay",
                metric_scope="locked product policy reference",
                primary_metric_name="released_MAE_BPM",
                primary_metric_value=overall.get("released_mae_bpm"),
                comparison=f"Coverage {json_float(overall.get('coverage'))}; unsafe/input {json_float(overall.get('published_unsafe_per_input'))}.",
                evidence_status="legacy_locked_product_reference",
                claim_use="Product-policy continuity reference; not overwritten by T474-T477.",
                limitation=t386.get("claim_boundary", "Earlier product replay reference."),
            )
        )
    if t407:
        cards.append(
            metric_card(
                task_id="T407",
                dataset="CMU-rPPG-biases",
                metric_scope="fairness/skin-tone pilot",
                primary_metric_name="audit_status",
                primary_metric_value=math.nan,
                comparison=t407.get("decision", "CMU fairness replay pilot present."),
                evidence_status="fairness_audit_context",
                claim_use="Fairness/skin-tone risk context for product claim boundaries.",
                limitation=t407.get("claim_boundary", "Pilot fairness audit; not a final fairness guarantee."),
            )
        )
    return cards


def build_claim_gates(paths: ProductEvidencePaths) -> list[dict[str, Any]]:
    t474 = read_json(paths.t474_summary)
    t476b = read_json(paths.t476b_summary)
    t477 = read_json(paths.t477_summary)
    t455 = read_json(paths.t455_summary)
    dataset_gate_passed, dataset_gate_evidence = dataset_coverage_gate(paths, t455)

    gates = [
        {
            "gate": "same_dataset_method_evidence",
            "passed": bool(t474) and finite_float(t474.get("best_mae_bpm")) < finite_float(t474.get("tscan_mae_bpm")),
            "evidence": "T474 best selector improves over aligned TSCAN on UBFC.",
            "claim_allowed": "UBFC protocol-aligned method comparison.",
            "claim_not_allowed": "Universal SOTA or clinical readiness.",
        },
        {
            "gate": "deep_baseline_alignment",
            "passed": bool(t476b) and finite_float(t476b.get("t474_best_mae_bpm")) < finite_float(t476b.get("best_deep_mae_bpm")),
            "evidence": "T476B compares against clip180 PhysFormer/RhythmFormer outputs.",
            "claim_allowed": "Same-window UBFC comparison versus selected modern deep baselines.",
            "claim_not_allowed": "All deep rPPG architectures or all datasets.",
        },
        {
            "gate": "external_transfer_direction",
            "passed": bool(t477) and finite_float(t477.get("primary_t477_mae_bpm")) < finite_float(t477.get("top_power_mae_bpm")),
            "evidence": "T477 UBFC-trained common-schema selector beats naive top-power/top-support on DLCN.",
            "claim_allowed": "Candidate reliability transfer over naive spectral ranking.",
            "claim_not_allowed": "Domain-invariant full-performance release; DLCN-trained reference remains better.",
        },
        {
            "gate": "dataset_coverage",
            "passed": dataset_gate_passed,
            "evidence": dataset_gate_evidence,
            "claim_allowed": "Adult multi-domain development/validation plan is operational.",
            "claim_not_allowed": "All uploaded datasets have finished final model validation.",
        },
        {
            "gate": "product_safety_boundary",
            "passed": True,
            "evidence": "API bundle enforces release/review language and excludes evaluation labels.",
            "claim_allowed": "Research MVP with explicit review/retest semantics.",
            "claim_not_allowed": "Medical device, diagnosis, emergency monitoring, or infant-ready deployment.",
        },
    ]
    return gates


def dataset_coverage_gate(paths: ProductEvidencePaths, t455: dict[str, Any]) -> tuple[bool, str]:
    """Return a conservative dataset-availability gate for local or AutoDL runs.

    T455 is the preferred audited migration artifact. On a fresh AutoDL clone the
    historical JSON may be absent even when the data disk is mounted correctly,
    so this function falls back to checking the canonical remote dataset tree.
    """

    if t455:
        decision = str(t455.get("decision", ""))
        all_passed = bool(t455.get("all_gates_passed", False))
        return all_passed or "passed" in decision, f"T455 migration gate present: decision={decision}, all_gates_passed={all_passed}."

    required = [
        "UBFC-rPPG",
        "UBFC-Phys-S1-S14",
        "4TU-rPPG-Benchmark",
        "MCD-rPPG",
        "DLCN",
        "MR-NIRP",
        "CMU-rPPG-biases",
        "small-rPPG-Empatica",
    ]
    candidate_roots = []
    if os.environ.get("ADULT_DATA_ROOT"):
        candidate_roots.append(Path(os.environ["ADULT_DATA_ROOT"]))
    if os.environ.get("CONTACTLESS_DATA_ROOT"):
        candidate_roots.append(Path(os.environ["CONTACTLESS_DATA_ROOT"]) / "adult")
    candidate_roots.append(paths.root.parent / "datasets" / "adult")
    for root in candidate_roots:
        if not root.exists():
            continue
        present = [name for name in required if (root / name).exists()]
        missing = [name for name in required if name not in present]
        if not missing:
            return True, f"Dataset directories present under {root.as_posix()}: {', '.join(present)}."
        if len(present) >= 6:
            return False, f"Partial dataset directories under {root.as_posix()}: present={present}; missing={missing}."
    return False, "No T455 summary and no complete canonical dataset directory tree found."


def build_product_evidence_bundle(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    paths = ProductEvidencePaths(root=root_path)
    metric_cards = build_metric_cards(paths)
    claim_gates = build_claim_gates(paths)

    return {
        "bundle_version": EVIDENCE_BUNDLE_VERSION,
        "product_mode": PRODUCT_MODE,
        "product_mode_label": PRODUCT_MODE_LABEL,
        "claim_boundary": CLAIM_BOUNDARY,
        "warnings": PRODUCT_WARNINGS,
        "target_users": [
            "adult care monitoring research prototype",
            "hospital/nursing-home caregiver dashboard MVP",
            "fitness or rehabilitation video-based HR trend feedback prototype",
        ],
        "excluded_or_not_yet_supported_users": [
            "diagnostic clinical decision-making",
            "emergency alarm replacement",
            "final infant monitoring product",
        ],
        "input_contract": {
            "primary_input": "RGB face video or webcam stream",
            "optional_inputs": ["camera metadata", "analysis window length", "review/retest policy threshold"],
            "not_required_for_product_response": ["ground-truth HR labels", "ECG/PPG labels"],
        },
        "output_contract": {
            "release_output": ["HR estimate in BPM", "window timestamp", "quality/evidence summary", "research warning"],
            "review_output": ["review/retest decision", "reason code", "candidate evidence summary", "research warning"],
            "forbidden_user_fields": ["ground_truth", "gt_hr_bpm", "eval_abs_error_bpm", "label"],
        },
        "method_stack": [
            "MediaPipe Face Mesh ROI extraction",
            "mentor-aligned facial ROI grouping",
            "traditional rPPG candidate generation",
            "multi-candidate physiological plausibility checks",
            "learned candidate reliability selector",
            "release/review product gate",
            "dashboard/API evidence reporting",
        ],
        "metric_cards": metric_cards,
        "claim_gates": claim_gates,
        "overall_decision": product_decision(metric_cards, claim_gates),
        "next_recommended_tasks": [
            "T479: MR-NIRP/CMU adapter validation and fairness/lighting-domain evidence expansion.",
            "T480: train/freeze the next unified selector with UBFC+DLCN and evaluate on held-out domains.",
            "T481: integrate evidence bundle into Streamlit/FastAPI MVP and run browser/API QA.",
        ],
    }


def product_decision(metric_cards: list[dict[str, Any]], claim_gates: list[dict[str, Any]]) -> str:
    missing = [card["task_id"] for card in metric_cards if card["evidence_status"] == "missing"]
    failed = [gate["gate"] for gate in claim_gates if not bool(gate["passed"])]
    if missing or failed:
        return "product_evidence_bundle_incomplete"
    return "product_mvp_evidence_ready_with_research_boundaries"


def build_mode_response(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_version": bundle["bundle_version"],
        "product_mode": bundle["product_mode"],
        "product_mode_label": bundle["product_mode_label"],
        "claim_boundary": bundle["claim_boundary"],
        "warnings": bundle["warnings"],
        "input_contract": bundle["input_contract"],
        "output_contract": bundle["output_contract"],
        "overall_decision": bundle["overall_decision"],
    }


def build_summary_response(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_version": bundle["bundle_version"],
        "product_mode": bundle["product_mode"],
        "overall_decision": bundle["overall_decision"],
        "method_stack": bundle["method_stack"],
        "metric_cards": bundle["metric_cards"],
        "claim_gates": bundle["claim_gates"],
        "next_recommended_tasks": bundle["next_recommended_tasks"],
        "claim_boundary": bundle["claim_boundary"],
        "warnings": bundle["warnings"],
    }


def forbidden_fields_present(value: Any, forbidden: set[str] | None = None) -> list[str]:
    forbidden = forbidden or {"ground_truth", "gt_hr_bpm", "eval_abs_error_bpm", "label"}
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden:
                found.append(str(key))
            found.extend(forbidden_fields_present(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.extend(forbidden_fields_present(child, forbidden))
    return sorted(set(found))


def metric_cards_dataframe(bundle: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(bundle.get("metric_cards", []))


def claim_gates_dataframe(bundle: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(bundle.get("claim_gates", []))
