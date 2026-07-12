from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BUNDLE_VERSION = "t686.dense_patch_review_first.v1"
PRODUCT_MODE = "adult_dense_patch_review_first_hr"
PRODUCT_MODE_LABEL = "Adult dense-patch rPPG with review-first release gate"
CLAIM_BOUNDARY = (
    "Current evidence supports dense face-patch candidate headroom and temporal semantic-ROI "
    "candidate selection for adult rPPG HR estimation. External-domain automatic release is not solved; "
    "the product policy is review-first and not a clinical diagnostic or emergency-alert system."
)

USER_FORBIDDEN_FIELDS = {
    "reference_hr_bpm",
    "ground_truth",
    "gt_hr_bpm",
    "candidate_abs_error",
    "selected_abs_error_bpm",
    "oracle_abs_error_bpm",
    "unsafe_candidate",
    "unsafe_selected",
    "unsafe_released",
    "absolute_error_bpm_for_eval_only",
    "unsafe_release_gt10_for_eval_only",
}


@dataclass(frozen=True)
class DensePatchEvidencePaths:
    root: Path

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def t681_candidates(self) -> Path:
        return self.experiments / "t681_ubfc_rppg_candidate_table.csv"

    @property
    def t682_decisions(self) -> Path:
        return self.experiments / "t682_external_release_gate_refit_decisions.csv"

    @property
    def t682_summary(self) -> Path:
        return self.experiments / "t682_external_release_gate_refit_summary.json"

    @property
    def t683_cases(self) -> Path:
        return self.experiments / "t683_external_failure_taxonomy_cases.csv"

    @property
    def t683_summary(self) -> Path:
        return self.experiments / "t683_external_failure_taxonomy_summary.json"

    @property
    def t684_modules(self) -> Path:
        return self.experiments / "t684_ablation_module_contribution_table.csv"

    @property
    def t684_product(self) -> Path:
        return self.experiments / "t684_product_policy_evidence_table.csv"

    @property
    def t684_claims(self) -> Path:
        return self.experiments / "t684_claim_evidence_matrix.csv"

    @property
    def t685_summary(self) -> Path:
        return self.experiments / "t685_manuscript_product_figure_pack_summary.json"


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def json_float(value: Any) -> float | None:
    out = finite_float(value)
    return out if math.isfinite(out) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    out = str(value)
    return fallback if out.lower() in {"nan", "none"} else out


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _top_competing_candidates(candidates: pd.DataFrame, clip_id: str, selected: pd.Series, n: int = 5) -> list[dict[str, Any]]:
    if candidates.empty or "clip_id" not in candidates.columns:
        return []
    subset = candidates[candidates["clip_id"].astype(str).eq(str(clip_id))].copy()
    if subset.empty:
        return []
    subset["candidate_snr_db_num"] = pd.to_numeric(subset.get("candidate_snr_db"), errors="coerce")
    subset["candidate_peak_support_num"] = pd.to_numeric(subset.get("candidate_peak_support"), errors="coerce")
    subset = subset.sort_values(["candidate_snr_db_num", "candidate_peak_support_num"], ascending=False).head(n)
    records: list[dict[str, Any]] = []
    selected_key = (
        _text(selected.get("region")),
        _text(selected.get("method")),
        round(finite_float(selected.get("candidate_hr_bpm")), 6),
    )
    for rank, (_, row) in enumerate(subset.iterrows(), start=1):
        key = (_text(row.get("region")), _text(row.get("method")), round(finite_float(row.get("candidate_hr_bpm")), 6))
        records.append(
            {
                "rank": rank,
                "candidate_hr_bpm": json_float(row.get("candidate_hr_bpm")),
                "region": _text(row.get("region")),
                "patch_family": _text(row.get("patch_family")),
                "method": _text(row.get("method")),
                "snr_db": json_float(row.get("candidate_snr_db")),
                "peak_support": json_float(row.get("candidate_peak_support")),
                "coverage_mean": json_float(row.get("coverage_mean")),
                "is_selected": key == selected_key,
            }
        )
    return records


def _risk_reason(row: pd.Series, taxonomy: pd.Series | None) -> str:
    decision = _text(row.get("release_decision"), "review").lower()
    if decision == "release":
        return "selected candidate passed review-first release thresholds"
    mode = _text(taxonomy.get("primary_failure_mode") if taxonomy is not None else "", "")
    if mode == "safe_selection":
        return "candidate is plausible but external-domain release confidence is insufficient"
    if mode:
        return f"review required: {mode.replace('_', ' ')}"
    return "review required by external-domain release gate"


def _risk_factors(row: pd.Series, taxonomy: pd.Series | None) -> dict[str, Any]:
    return {
        "snr_db": json_float(row.get("candidate_snr_db")),
        "peak_support": json_float(row.get("candidate_peak_support")),
        "median_consistency": json_float(row.get("median_consistency")),
        "coverage_mean": json_float(row.get("coverage_mean")),
        "selected_score": json_float(row.get("selected_score")),
        "failure_mode": _text(taxonomy.get("primary_failure_mode") if taxonomy is not None else ""),
        "review_first_policy": True,
    }


def build_dense_patch_product_cases(paths: DensePatchEvidencePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = read_csv(paths.t682_decisions)
    taxonomy = read_csv(paths.t683_cases)
    candidates = read_csv(paths.t681_candidates)
    if decisions.empty:
        raise FileNotFoundError(f"Missing or empty decision table: {paths.t682_decisions}")

    tax_by_clip = {str(row["clip_id"]): row for _, row in taxonomy.iterrows()} if not taxonomy.empty and "clip_id" in taxonomy.columns else {}
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(decisions.iterrows(), start=1):
        clip_id = _text(row.get("clip_id"), f"case_{idx:03d}")
        tax = tax_by_clip.get(clip_id)
        decision = _text(row.get("release_decision"), "review").lower()
        selected_hr = json_float(row.get("candidate_hr_bpm"))
        product_hr = selected_hr if decision == "release" else None
        room = f"DP-{idx:03d}"
        competitors = _top_competing_candidates(candidates, clip_id, row)
        risk_factors = _risk_factors(row, tax)
        failure_mode = risk_factors["failure_mode"] or "release_threshold_review"
        review_reason = _risk_reason(row, tax)
        case = {
            "case_id": f"t686_{clip_id}",
            "dataset": _text(row.get("dataset"), "UBFC-rPPG"),
            "sample_id": clip_id,
            "room": room,
            "route_id": "dense_patch_temporal_semantic_roi_review_first",
            "selected_policy": "T682 review-first external release gate",
            "decision": "release" if decision == "release" else "review",
            "hr_bpm": product_hr,
            "confidence_proxy_for_demo": json_float(row.get("selected_score")),
            "review_reason": "" if decision == "release" else review_reason,
            "required_input": "adult RGB face video, Face Mesh dense patches, candidate peaks, temporal consistency",
            "quality_flags": failure_mode,
            "claim_boundary": CLAIM_BOUNDARY,
            "product_warning": "Research use only; not clinical diagnosis or emergency monitoring.",
            "selected_candidate_hr_bpm": selected_hr,
            "selected_region": _text(row.get("region")),
            "selected_patch_family": _text(row.get("patch_family")),
            "selected_method": _text(row.get("method")),
            "selected_snr_db": json_float(row.get("candidate_snr_db")),
            "selected_peak_support": json_float(row.get("candidate_peak_support")),
            "selected_median_consistency": json_float(row.get("median_consistency")),
            "selected_score": json_float(row.get("selected_score")),
            "release_policy": "release only when external-domain safety thresholds pass; otherwise review",
            "failure_mode": failure_mode,
            "risk_factors_json": strict_json_text(risk_factors),
            "competing_candidates_json": strict_json_text(competitors),
            "evidence_summary": (
                f"{_text(row.get('region'))}/{_text(row.get('method'))}, "
                f"SNR={json_float(row.get('candidate_snr_db'))}, "
                f"support={json_float(row.get('candidate_peak_support'))}, "
                f"consistency={json_float(row.get('median_consistency'))}"
            ),
        }
        rows.append(case)
        audit = dict(case)
        audit.update(
            {
                "reference_hr_bpm": json_float(row.get("reference_hr_bpm")),
                "selected_abs_error_bpm": json_float(row.get("candidate_abs_error")),
                "unsafe_candidate": _bool(row.get("unsafe_candidate")),
                "unsafe_selected": _bool(tax.get("unsafe_selected")) if tax is not None else None,
                "unsafe_released": _bool(tax.get("unsafe_released")) if tax is not None else None,
                "oracle_hr_bpm": json_float(tax.get("oracle_hr_bpm")) if tax is not None else None,
                "oracle_abs_error_bpm": json_float(tax.get("oracle_abs_error_bpm")) if tax is not None else None,
                "oracle_region": _text(tax.get("oracle_region")) if tax is not None else "",
                "oracle_method": _text(tax.get("oracle_method")) if tax is not None else "",
            }
        )
        audit_rows.append(audit)
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def build_route_metrics(paths: DensePatchEvidencePaths) -> pd.DataFrame:
    modules = read_csv(paths.t684_modules)
    t682 = read_json(paths.t682_summary)
    t683 = read_json(paths.t683_summary)
    rows: list[dict[str, Any]] = []
    for _, row in modules.iterrows():
        rows.append(
            {
                "route_id": _text(row.get("module")),
                "dataset": _text(row.get("dataset")),
                "baseline": _text(row.get("baseline")),
                "module_output": _text(row.get("module_output")),
                "coverage": json_float(row.get("coverage")),
                "mae_released_bpm": json_float(row.get("module_mae_bpm")),
                "baseline_mae_bpm": json_float(row.get("baseline_mae_bpm")),
                "relative_mae_reduction": json_float(row.get("relative_mae_reduction")),
                "unsafe_per_input": json_float(row.get("unsafe_release_rate")),
                "gate_status": _text(row.get("gate_status")),
                "evidence_interpretation": _text(row.get("evidence_interpretation")),
            }
        )
    chosen = t682.get("chosen_safe_policy", {}) if isinstance(t682.get("chosen_safe_policy"), dict) else {}
    if chosen:
        rows.append(
            {
                "route_id": "external_review_first_policy",
                "dataset": "UBFC-rPPG",
                "baseline": "temporal selector all predictions",
                "module_output": "selective external release",
                "coverage": json_float(chosen.get("coverage")),
                "mae_released_bpm": json_float(chosen.get("released_mae_bpm")),
                "baseline_mae_bpm": json_float(t682.get("all_mae_bpm")),
                "relative_mae_reduction": None,
                "unsafe_per_input": json_float(chosen.get("unsafe_release_rate")),
                "gate_status": "review_first_required",
                "evidence_interpretation": "external release is safe only at limited coverage; review-first product policy is required",
            }
        )
    rows.append(
        {
            "route_id": "failure_taxonomy_review_catch",
            "dataset": "UBFC-rPPG",
            "baseline": "unsafe selected cases",
            "module_output": "review-first caught unsafe cases",
            "coverage": json_float(t683.get("unsafe_caught_by_review_cases")) / json_float(t683.get("n_cases")) if json_float(t683.get("n_cases")) else None,
            "mae_released_bpm": None,
            "baseline_mae_bpm": None,
            "relative_mae_reduction": None,
            "unsafe_per_input": json_float(t683.get("unsafe_released_cases")) / json_float(t683.get("n_cases")) if json_float(t683.get("n_cases")) else None,
            "gate_status": "failure_taxonomy_supported",
            "evidence_interpretation": "most unsafe selected cases are caught by review, but one threshold leak remains",
        }
    )
    return pd.DataFrame(rows)


def build_product_policy(paths: DensePatchEvidencePaths) -> pd.DataFrame:
    product = read_csv(paths.t684_product)
    claims = read_csv(paths.t684_claims)
    rows: list[dict[str, Any]] = []
    for _, row in product.iterrows():
        rows.append(
            {
                "policy_id": _text(row.get("product_component")).lower().replace("/", "_").replace(" ", "_"),
                "product_component": _text(row.get("product_component")),
                "status": _text(row.get("status")),
                "metric": _text(row.get("metric")),
                "product_wording": _text(row.get("product_wording")),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    for _, row in claims.iterrows():
        if _text(row.get("status")) == "not_supported":
            rows.append(
                {
                    "policy_id": "claim_boundary_" + str(len(rows) + 1),
                    "product_component": "claim boundary",
                    "status": "not_supported",
                    "metric": _text(row.get("claim")),
                    "product_wording": _text(row.get("boundary")),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def sanitize_for_user(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_user(v) for k, v in value.items() if str(k) not in USER_FORBIDDEN_FIELDS}
    if isinstance(value, list):
        return [sanitize_for_user(v) for v in value]
    return json_safe(value)


def forbidden_fields_present(value: Any, forbidden: set[str] | None = None) -> list[str]:
    forbidden = forbidden or USER_FORBIDDEN_FIELDS
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


def build_api_examples(cases: pd.DataFrame, route_metrics: pd.DataFrame, policy: pd.DataFrame) -> dict[str, Any]:
    examples = []
    for _, row in cases.head(8).iterrows():
        record = row.to_dict()
        examples.append(sanitize_for_user(record))
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "product_mode": PRODUCT_MODE,
        "product_mode_label": PRODUCT_MODE_LABEL,
        "claim_boundary": CLAIM_BOUNDARY,
        "warnings": [
            "Research use only.",
            "Review rows intentionally do not publish a final HR value.",
            "External-domain automatic release is not claimed solved.",
        ],
        "examples": examples,
        "summary": {
            "n_cases": int(len(cases)),
            "n_release": int(cases["decision"].eq("release").sum()) if "decision" in cases.columns else 0,
            "n_review": int(cases["decision"].ne("release").sum()) if "decision" in cases.columns else 0,
            "route_rows": int(len(route_metrics)),
            "policy_rows": int(len(policy)),
        },
    }
    return sanitize_for_user(payload)


def build_qa_summary(api_examples: dict[str, Any], cases: pd.DataFrame, route_metrics: pd.DataFrame, policy: pd.DataFrame, paths: DensePatchEvidencePaths) -> dict[str, Any]:
    checks = [
        {
            "check": "product_cases_nonempty",
            "passed": not cases.empty,
            "detail": f"{len(cases)} product cases",
        },
        {
            "check": "route_metrics_nonempty",
            "passed": not route_metrics.empty,
            "detail": f"{len(route_metrics)} route rows",
        },
        {
            "check": "policy_nonempty",
            "passed": not policy.empty,
            "detail": f"{len(policy)} policy rows",
        },
        {
            "check": "api_no_eval_or_ground_truth_fields",
            "passed": len(forbidden_fields_present(api_examples)) == 0,
            "detail": ",".join(forbidden_fields_present(api_examples)) or "clean",
        },
        {
            "check": "review_rows_hide_product_hr",
            "passed": bool(cases[cases["decision"].ne("release")]["hr_bpm"].isna().all()) if "hr_bpm" in cases.columns and not cases.empty else False,
            "detail": "review rows keep hr_bpm empty",
        },
        {
            "check": "figures_available",
            "passed": paths.t685_summary.exists(),
            "detail": str(paths.t685_summary.relative_to(paths.root)) if paths.t685_summary.exists() else "missing",
        },
    ]
    return {
        "bundle_version": BUNDLE_VERSION,
        "n_checks": len(checks),
        "passed_checks": sum(1 for item in checks if item["passed"]),
        "all_qa_passed": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "claim_boundary": CLAIM_BOUNDARY,
    }
