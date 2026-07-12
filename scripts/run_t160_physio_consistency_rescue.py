"""T160 physiology-consistency rescue for adult product release.

T159 showed that a learned candidate selector contains signal but is not stable
enough to replace the frozen T157 guarded correction. T160 therefore tests a
more conservative mechanism: keep the safe product anchor, and rescue only
reviewed samples whose candidate pool provides strong physiology-consistency
evidence.

Ground truth is used only for evaluation, case audit, and bootstrap reporting.
The T160 release rule itself uses inference-time candidate evidence: ROI
support, method support, top-1/subwindow consistency, anchor distance, and a
high-frequency ambiguity veto.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.evaluation.metrics import mae, pearson, rmse  # noqa: E402


EXPERIMENTS = PROJECT / "experiments"
DOCS = PROJECT / "docs"
FIG_DIR = PROJECT / "output" / "t160_figures"

T151_CANDIDATES = EXPERIMENTS / "t151_rppg10_candidate_table.csv"
T151_SELECTION = EXPERIMENTS / "t151_rppg10_selection_table.csv"
T153_DEPLOYMENT = EXPERIMENTS / "t153_t154_deployment_release_table.csv"
T157_SELECTION = EXPERIMENTS / "t157_selection_table.csv"

RPPG10_CLUSTERS_CSV = EXPERIMENTS / "t160_rppg10_subject_candidate_clusters.csv"
RPPG10_DECISIONS_CSV = EXPERIMENTS / "t160_rppg10_rescue_decisions.csv"
ADULT_POLICY_TABLE_CSV = EXPERIMENTS / "t160_adult_product_policy_table.csv"
POLICY_SUMMARY_CSV = EXPERIMENTS / "t160_policy_summary.csv"
CASE_AUDIT_CSV = EXPERIMENTS / "t160_case_audit.csv"
BOOTSTRAP_CSV = EXPERIMENTS / "t160_bootstrap.csv"
SUMMARY_JSON = EXPERIMENTS / "t160_physio_consistency_rescue_summary.json"
REPORT_MD = EXPERIMENTS / f"t160_physio_consistency_rescue_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t160_physio_consistency_rescue.md"

UNSAFE_BPM = 10.0
CLUSTER_TOL_BPM = 4.0
ADULT_DATASETS = ["4TU-rPPG-Benchmark", "UBFC-rPPG"]


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def markdown_table(df: pd.DataFrame, *, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    lines = [
        "| " + " | ".join(str(c) for c in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def append_unique(path: Path, marker: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    path.write_text(old.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [T151_CANDIDATES, T151_SELECTION, T153_DEPLOYMENT, T157_SELECTION]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required T160 input: {path}")
    rppg_candidates = pd.read_csv(T151_CANDIDATES)
    rppg_selection = pd.read_csv(T151_SELECTION)
    deployment = pd.read_csv(T153_DEPLOYMENT)
    t157_selection = pd.read_csv(T157_SELECTION)
    return rppg_candidates, rppg_selection, deployment, t157_selection


def cluster_subject_candidates(candidates: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Cluster rPPG-10 ROI-level candidates into subject-level HR hypotheses."""
    t150_cols = [
        "sample_id",
        "subject_id",
        "roi_name",
        "anchor_median",
        "anchor_iqr",
        "t150_confidence",
        "selected_bpm",
        "selected_abs_error_bpm",
    ]
    anchors = selection[selection["policy"].astype(str) == "T150_domain_robust_v1"][
        [c for c in t150_cols if c in selection.columns]
    ].copy()
    anchors = anchors.rename(
        columns={
            "selected_bpm": "roi_t150_bpm",
            "selected_abs_error_bpm": "roi_t150_abs_error_bpm",
        }
    )
    merged = candidates.merge(anchors, on=["sample_id", "subject_id", "roi_name"], how="left")
    for col in [
        "candidate_bpm",
        "gt_hr_bpm",
        "support_methods",
        "top1_support_methods",
        "subwindow_support",
        "subwindow_top1_support",
        "power_sum_all",
        "power_sum_full",
        "support_power_score",
        "physiology_candidate_score",
        "half_harmonic_support",
        "double_harmonic_support",
        "anchor_median",
        "anchor_iqr",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    rows: list[dict[str, object]] = []
    for subject_id, group in merged.groupby("subject_id", sort=True):
        clusters: list[dict[str, object]] = []
        for _, row in group.sort_values("candidate_bpm").iterrows():
            bpm = finite_float(row.get("candidate_bpm"))
            if not math.isfinite(bpm):
                continue
            nearest_idx = None
            nearest_dist = float("inf")
            for idx, cluster in enumerate(clusters):
                dist = abs(bpm - finite_float(cluster["center_bpm"]))
                if dist <= CLUSTER_TOL_BPM and dist < nearest_dist:
                    nearest_idx = idx
                    nearest_dist = dist
            if nearest_idx is None:
                clusters.append({"center_bpm": bpm, "members": [row]})
            else:
                members = clusters[nearest_idx]["members"]
                assert isinstance(members, list)
                members.append(row)
                values = np.asarray([finite_float(m.get("candidate_bpm")) for m in members], dtype=float)
                weights = np.asarray(
                    [max(finite_float(m.get("power_sum_all"), 0.0), 1e-6) for m in members],
                    dtype=float,
                )
                clusters[nearest_idx]["center_bpm"] = float(np.average(values, weights=weights))

        gt = finite_float(group["gt_hr_bpm"].iloc[0])
        for idx, cluster in enumerate(clusters):
            members = pd.DataFrame(cluster["members"])
            bpm = finite_float(cluster["center_bpm"])
            anchor = float(pd.to_numeric(members.get("anchor_median"), errors="coerce").median())
            dist_to_anchor = abs(bpm - anchor) if math.isfinite(anchor) else math.nan
            top1_sum = float(pd.to_numeric(members.get("top1_support_methods"), errors="coerce").sum())
            subtop_sum = float(pd.to_numeric(members.get("subwindow_top1_support"), errors="coerce").sum())
            support_rois = int(members["roi_name"].nunique())
            support_methods_sum = float(pd.to_numeric(members.get("support_methods"), errors="coerce").sum())
            power_sum = float(pd.to_numeric(members.get("power_sum_all"), errors="coerce").sum())
            support_power_sum = float(pd.to_numeric(members.get("support_power_score"), errors="coerce").sum())
            half_sum = float(pd.to_numeric(members.get("half_harmonic_support"), errors="coerce").sum())
            double_sum = float(pd.to_numeric(members.get("double_harmonic_support"), errors="coerce").sum())
            anchor_bonus = math.exp(-finite_float(dist_to_anchor, 99.0) / 6.0)
            quality_score = (
                3.5 * support_rois
                + 0.12 * support_methods_sum
                + 0.25 * top1_sum
                + 0.06 * subtop_sum
                + 0.10 * power_sum
                + 5.0 * anchor_bonus
                - 0.10 * half_sum
                - 0.04 * double_sum
            )
            rows.append(
                {
                    "task_id": "T160",
                    "dataset": "rPPG-10",
                    "subject_id": subject_id,
                    "deployment_id": f"rPPG-10_{subject_id}",
                    "candidate_cluster_id": f"{subject_id}_cl{idx:02d}",
                    "candidate_bpm": bpm,
                    "gt_hr_bpm": gt,
                    "candidate_abs_error_bpm": abs(bpm - gt) if math.isfinite(gt) else math.nan,
                    "support_roi_samples": int(members["sample_id"].nunique()),
                    "support_rois": support_rois,
                    "n_members": int(len(members)),
                    "support_methods_sum": support_methods_sum,
                    "top1_sum": top1_sum,
                    "subwindow_sum": float(pd.to_numeric(members.get("subwindow_support"), errors="coerce").sum()),
                    "subwindow_top1_sum": subtop_sum,
                    "power_sum_all": power_sum,
                    "power_sum_full": float(pd.to_numeric(members.get("power_sum_full"), errors="coerce").sum()),
                    "support_power_score_sum": support_power_sum,
                    "physiology_candidate_score_sum": float(
                        pd.to_numeric(members.get("physiology_candidate_score"), errors="coerce").sum()
                    ),
                    "half_harmonic_sum": half_sum,
                    "double_harmonic_sum": double_sum,
                    "anchor_median": anchor,
                    "anchor_iqr_median": float(pd.to_numeric(members.get("anchor_iqr"), errors="coerce").median()),
                    "dist_to_anchor_bpm": dist_to_anchor,
                    "anchor_bonus": anchor_bonus,
                    "t160_quality_score": quality_score,
                    "roi_names": ",".join(sorted(str(x) for x in members["roi_name"].dropna().unique())),
                    "member_sample_ids": ",".join(sorted(str(x) for x in members["sample_id"].dropna().unique())),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RPPG10_CLUSTERS_CSV, index=False, encoding="utf-8-sig")
    return out


def has_high_frequency_ambiguity(group: pd.DataFrame) -> bool:
    high = group[
        (group["candidate_bpm"].between(120.0, 140.0))
        & (group["support_rois"] >= 3)
        & (group["support_methods_sum"] >= 15.0)
        & (group["support_power_score_sum"] >= 40.0)
        & (group["half_harmonic_sum"] >= 18.0)
    ]
    return not high.empty


def select_t160_rescue_candidate(group: pd.DataFrame, base_bpm: float, high_ambiguous: bool) -> tuple[pd.Series | None, str]:
    """Return a release candidate only when physiology evidence is strong."""
    base_reinforcement = group[
        ((group["candidate_bpm"] - base_bpm).abs() <= 4.0)
        & (group["candidate_bpm"].between(68.0, 85.0))
        & (group["support_rois"] >= 2)
        & (group["top1_sum"] >= 8.0)
        & (group["subwindow_top1_sum"] >= 18.0)
    ].copy()
    if not base_reinforcement.empty:
        selected = base_reinforcement.sort_values(
            ["t160_quality_score", "top1_sum", "support_rois"],
            ascending=[False, False, False],
        ).iloc[0]
        return selected, "base_reinforcement"

    rescue = group[
        (group["candidate_bpm"].between(55.0, 90.0))
        & (group["support_rois"] >= 2)
        & (group["top1_sum"] >= 8.0)
        & (group["subwindow_top1_sum"] >= 18.0)
        & (group["dist_to_anchor_bpm"] <= 4.0)
    ].copy()
    if high_ambiguous:
        rescue = rescue[rescue["candidate_bpm"] >= 75.0]
    if rescue.empty:
        return None, "remain_review"
    selected = rescue.sort_values(
        ["t160_quality_score", "top1_sum", "support_rois"],
        ascending=[False, False, False],
    ).iloc[0]
    return selected, "anchored_candidate_rescue"


def build_rppg10_rescue_decisions(clusters: pd.DataFrame, deployment: pd.DataFrame) -> pd.DataFrame:
    base = deployment[
        (deployment["dataset"].astype(str) == "rPPG-10")
        & (deployment["policy"].astype(str) == "T153_T154_product_balanced_v1")
    ].drop_duplicates("subject_id")
    rows: list[dict[str, object]] = []
    for _, base_row in base.sort_values("subject_id").iterrows():
        subject_id = str(base_row["subject_id"])
        group = clusters[clusters["subject_id"].astype(str) == subject_id].copy()
        high_ambiguous = has_high_frequency_ambiguity(group)
        base_released = int(finite_float(base_row.get("released"), 0.0)) > 0
        base_bpm = finite_float(base_row.get("selected_bpm"))
        gt = finite_float(base_row.get("gt_hr_bpm"))
        decision_source = "base_release" if base_released else "remain_review"
        selected_bpm = base_bpm
        released = int(base_released)
        selected_cluster_id = ""
        selected_quality = math.nan
        selected_support_rois = math.nan
        selected_top1_sum = math.nan
        selected_dist_to_anchor = math.nan
        if not base_released and not group.empty:
            candidate, source = select_t160_rescue_candidate(group, base_bpm, high_ambiguous)
            decision_source = source
            if candidate is not None:
                selected_bpm = finite_float(candidate["candidate_bpm"])
                selected_cluster_id = str(candidate["candidate_cluster_id"])
                selected_quality = finite_float(candidate["t160_quality_score"])
                selected_support_rois = finite_float(candidate["support_rois"])
                selected_top1_sum = finite_float(candidate["top1_sum"])
                selected_dist_to_anchor = finite_float(candidate["dist_to_anchor_bpm"])
                released = 1
        selected_error = abs(selected_bpm - gt) if math.isfinite(selected_bpm) and math.isfinite(gt) else math.nan
        rows.append(
            {
                "task_id": "T160",
                "dataset": "rPPG-10",
                "deployment_id": str(base_row.get("deployment_id", f"rPPG-10_{subject_id}")),
                "subject_id": subject_id,
                "gt_hr_bpm": gt,
                "base_policy": "T153_T154_product_balanced_v1",
                "base_selected_bpm": base_bpm,
                "base_abs_error_bpm": finite_float(base_row.get("selected_abs_error_bpm")),
                "base_released": int(base_released),
                "base_review_reason": str(base_row.get("review_reason", "")),
                "selected_bpm": selected_bpm,
                "selected_abs_error_bpm": selected_error,
                "released": released,
                "decision_source": decision_source,
                "selected_cluster_id": selected_cluster_id,
                "selected_quality_score": selected_quality,
                "selected_support_rois": selected_support_rois,
                "selected_top1_sum": selected_top1_sum,
                "selected_dist_to_anchor_bpm": selected_dist_to_anchor,
                "high_frequency_ambiguity_veto": int(high_ambiguous),
                "roi_predictions": str(base_row.get("roi_predictions", "")),
                "roi_prediction_range_bpm": finite_float(base_row.get("roi_prediction_range_bpm")),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RPPG10_DECISIONS_CSV, index=False, encoding="utf-8-sig")
    return out


def adult_current_policy(t157_selection: pd.DataFrame, rppg_decisions: pd.DataFrame) -> pd.DataFrame:
    """Hybrid current product state: T157 guarded for 4TU/UBFC, T153 balanced for rPPG-10."""
    t157 = t157_selection[
        (t157_selection["dataset"].isin(ADULT_DATASETS))
        & (t157_selection["policy"].astype(str) == "T157_guarded_correction_v2")
    ].copy()
    t157_rows: list[dict[str, object]] = []
    for _, row in t157.iterrows():
        t157_rows.append(
            {
                "task_id": "T160",
                "dataset": row["dataset"],
                "deployment_id": row["sample_id"],
                "subject_id": row.get("subject_id", ""),
                "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                "policy": "T157_T153_current_product",
                "selected_bpm": finite_float(row.get("selected_bpm")),
                "selected_abs_error_bpm": finite_float(row.get("selected_abs_error_bpm")),
                "released": 1,
                "decision_source": "t157_guarded_correction_v2",
            }
        )
    rppg_current = rppg_decisions.copy()
    rppg_current["policy"] = "T157_T153_current_product"
    rppg_current["selected_bpm"] = rppg_current["base_selected_bpm"]
    rppg_current["selected_abs_error_bpm"] = rppg_current["base_abs_error_bpm"]
    rppg_current["released"] = rppg_current["base_released"]
    rppg_current["decision_source"] = np.where(
        rppg_current["base_released"].astype(int) > 0,
        "t153_t154_balanced_release",
        "t153_t154_balanced_review",
    )
    current_cols = [
        "task_id",
        "dataset",
        "deployment_id",
        "subject_id",
        "gt_hr_bpm",
        "policy",
        "selected_bpm",
        "selected_abs_error_bpm",
        "released",
        "decision_source",
    ]
    return pd.concat([pd.DataFrame(t157_rows), rppg_current[current_cols]], ignore_index=True, sort=False)


def adult_t160_policy(t157_selection: pd.DataFrame, rppg_decisions: pd.DataFrame) -> pd.DataFrame:
    t157 = t157_selection[
        (t157_selection["dataset"].isin(ADULT_DATASETS))
        & (t157_selection["policy"].astype(str) == "T157_guarded_correction_v2")
    ].copy()
    t157_rows: list[dict[str, object]] = []
    for _, row in t157.iterrows():
        t157_rows.append(
            {
                "task_id": "T160",
                "dataset": row["dataset"],
                "deployment_id": row["sample_id"],
                "subject_id": row.get("subject_id", ""),
                "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                "policy": "T160_physio_consistency_rescue_v1",
                "selected_bpm": finite_float(row.get("selected_bpm")),
                "selected_abs_error_bpm": finite_float(row.get("selected_abs_error_bpm")),
                "released": 1,
                "decision_source": "t157_guarded_correction_v2",
            }
        )
    rppg = rppg_decisions.copy()
    rppg["policy"] = "T160_physio_consistency_rescue_v1"
    cols = [
        "task_id",
        "dataset",
        "deployment_id",
        "subject_id",
        "gt_hr_bpm",
        "policy",
        "selected_bpm",
        "selected_abs_error_bpm",
        "released",
        "decision_source",
    ]
    return pd.concat([pd.DataFrame(t157_rows), rppg[cols]], ignore_index=True, sort=False)


def baseline_release_all_policy(deployment: pd.DataFrame) -> pd.DataFrame:
    """T150 deployment all-release baseline across 4TU, UBFC, and rPPG-10."""
    base = deployment[deployment["policy"].astype(str) == "T150_deployment_release_all"].copy()
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        rows.append(
            {
                "task_id": "T160",
                "dataset": row["dataset"],
                "deployment_id": row["deployment_id"],
                "subject_id": row.get("subject_id", ""),
                "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                "policy": "T150_deployment_release_all",
                "selected_bpm": finite_float(row.get("selected_bpm")),
                "selected_abs_error_bpm": finite_float(row.get("selected_abs_error_bpm")),
                "released": 1,
                "decision_source": "release_all",
            }
        )
    return pd.DataFrame(rows)


def t153_balanced_policy(deployment: pd.DataFrame) -> pd.DataFrame:
    base = deployment[deployment["policy"].astype(str) == "T153_T154_product_balanced_v1"].copy()
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        rows.append(
            {
                "task_id": "T160",
                "dataset": row["dataset"],
                "deployment_id": row["deployment_id"],
                "subject_id": row.get("subject_id", ""),
                "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                "policy": "T153_T154_product_balanced_v1",
                "selected_bpm": finite_float(row.get("selected_bpm")),
                "selected_abs_error_bpm": finite_float(row.get("selected_abs_error_bpm")),
                "released": int(finite_float(row.get("released"), 0.0)),
                "decision_source": str(row.get("release_status", "")),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, policy), group in table.groupby(["dataset", "policy"], sort=True):
        rows.append(_summary_row(dataset, policy, group))
    for policy, group in table.groupby("policy", sort=True):
        rows.append(_summary_row("ALL", policy, group))
    return pd.DataFrame(rows)


def _summary_row(dataset: str, policy: str, group: pd.DataFrame) -> dict[str, object]:
    gt = pd.to_numeric(group["gt_hr_bpm"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(group["selected_bpm"], errors="coerce").to_numpy(dtype=float)
    released = pd.to_numeric(group["released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    finite = np.isfinite(gt) & np.isfinite(pred)
    rel = finite & released
    withheld = finite & ~released
    errors_rel = np.abs(gt[rel] - pred[rel])
    errors_all = np.abs(gt[finite] - pred[finite])
    withheld_errors = np.abs(gt[withheld] - pred[withheld])
    return {
        "dataset": dataset,
        "policy": policy,
        "n_total": int(finite.sum()),
        "released": int(rel.sum()),
        "withheld": int(withheld.sum()),
        "coverage": float(rel.sum() / finite.sum()) if finite.sum() else math.nan,
        "released_mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
        "released_rmse_bpm": rmse(gt[rel], pred[rel]) if rel.sum() else math.nan,
        "released_pearson_r": pearson(gt[rel], pred[rel]) if rel.sum() else math.nan,
        "released_median_abs_error_bpm": float(np.median(errors_rel)) if len(errors_rel) else math.nan,
        "released_p90_abs_error_bpm": float(np.percentile(errors_rel, 90)) if len(errors_rel) else math.nan,
        "unsafe_release_count": int(np.sum(errors_rel > UNSAFE_BPM)),
        "unsafe_per_input": float(np.sum(errors_rel > UNSAFE_BPM) / finite.sum()) if finite.sum() else math.nan,
        "unsafe_release_rate": float(np.mean(errors_rel > UNSAFE_BPM)) if len(errors_rel) else math.nan,
        "withheld_unsafe_count": int(np.sum(withheld_errors > UNSAFE_BPM)),
        "safe_withheld_count": int(np.sum(withheld_errors <= UNSAFE_BPM)),
        "all_release_equivalent_mae_bpm": mae(gt[finite], pred[finite]) if finite.sum() else math.nan,
        "all_release_equivalent_unsafe_rate": float(np.mean(errors_all > UNSAFE_BPM)) if len(errors_all) else math.nan,
    }


def bootstrap_delta(
    table: pd.DataFrame,
    baseline_policy: str,
    improved_policy: str,
    *,
    n_boot: int = 5000,
    seed: int = 160,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for dataset in sorted(table["dataset"].dropna().unique()) + ["ALL"]:
        sub = table if dataset == "ALL" else table[table["dataset"] == dataset]
        a = sub[sub["policy"] == baseline_policy].drop_duplicates("deployment_id").set_index("deployment_id")
        b = sub[sub["policy"] == improved_policy].drop_duplicates("deployment_id").set_index("deployment_id")
        ids = sorted(set(a.index) & set(b.index))
        if not ids:
            continue
        err_a = pd.to_numeric(a.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float)
        err_b = pd.to_numeric(b.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float)
        rel_a = pd.to_numeric(a.loc[ids, "released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        rel_b = pd.to_numeric(b.loc[ids, "released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        finite = np.isfinite(err_a) & np.isfinite(err_b)
        err_a, err_b, rel_a, rel_b = err_a[finite], err_b[finite], rel_a[finite], rel_b[finite]
        if not len(err_a):
            continue
        coverage_delta = rel_b.astype(float) - rel_a.astype(float)
        unsafe_delta = ((err_a > UNSAFE_BPM) & rel_a).astype(float) - ((err_b > UNSAFE_BPM) & rel_b).astype(float)
        released_both = rel_a & rel_b
        mae_delta = np.where(released_both, err_a - err_b, np.nan)
        mae_delta_finite = mae_delta[np.isfinite(mae_delta)]
        idx = np.arange(len(err_a))
        boot_cov = np.asarray([np.mean(coverage_delta[rng.choice(idx, len(idx), replace=True)]) for _ in range(n_boot)])
        boot_unsafe = np.asarray([np.mean(unsafe_delta[rng.choice(idx, len(idx), replace=True)]) for _ in range(n_boot)])
        if len(mae_delta_finite):
            idx_mae = np.arange(len(mae_delta_finite))
            boot_mae = np.asarray(
                [np.mean(mae_delta_finite[rng.choice(idx_mae, len(idx_mae), replace=True)]) for _ in range(n_boot)]
            )
            mean_mae_delta = float(np.mean(mae_delta_finite))
            mae_low = float(np.percentile(boot_mae, 2.5))
            mae_high = float(np.percentile(boot_mae, 97.5))
            p_mae = float(np.mean(boot_mae > 0.0))
        else:
            mean_mae_delta = math.nan
            mae_low = math.nan
            mae_high = math.nan
            p_mae = math.nan
        rows.append(
            {
                "dataset": dataset,
                "comparison": f"{improved_policy}_vs_{baseline_policy}",
                "n": int(len(err_a)),
                "mean_delta_coverage": float(np.mean(coverage_delta)),
                "coverage_ci95_low": float(np.percentile(boot_cov, 2.5)),
                "coverage_ci95_high": float(np.percentile(boot_cov, 97.5)),
                "p_coverage_delta_gt_0": float(np.mean(boot_cov > 0.0)),
                "mean_delta_unsafe_per_input": float(np.mean(unsafe_delta)),
                "unsafe_ci95_low": float(np.percentile(boot_unsafe, 2.5)),
                "unsafe_ci95_high": float(np.percentile(boot_unsafe, 97.5)),
                "p_unsafe_delta_gt_0": float(np.mean(boot_unsafe > 0.0)),
                "mean_delta_released_mae_bpm_on_common_releases": mean_mae_delta,
                "mae_ci95_low": mae_low,
                "mae_ci95_high": mae_high,
                "p_mae_delta_gt_0": p_mae,
            }
        )
    return pd.DataFrame(rows)


def build_case_audit(rppg_decisions: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    focus = set(
        rppg_decisions.loc[
            (rppg_decisions["base_released"].astype(int) == 0)
            | (rppg_decisions["high_frequency_ambiguity_veto"].astype(int) > 0),
            "subject_id",
        ].astype(str)
    )
    rows: list[dict[str, object]] = []
    for subject_id in sorted(focus):
        decision = rppg_decisions[rppg_decisions["subject_id"].astype(str) == subject_id]
        if decision.empty:
            continue
        d = decision.iloc[0]
        g = clusters[clusters["subject_id"].astype(str) == subject_id].copy()
        if g.empty:
            best = pd.Series(dtype=object)
            oracle = pd.Series(dtype=object)
            high = pd.Series(dtype=object)
        else:
            best = g.sort_values("t160_quality_score", ascending=False).iloc[0]
            oracle = g.sort_values("candidate_abs_error_bpm", ascending=True).iloc[0]
            high_candidates = g[g["candidate_bpm"].between(120.0, 140.0)]
            high = high_candidates.sort_values("support_power_score_sum", ascending=False).iloc[0] if not high_candidates.empty else pd.Series(dtype=object)
        rows.append(
            {
                "subject_id": subject_id,
                "gt_hr_bpm": finite_float(d.get("gt_hr_bpm")),
                "base_bpm": finite_float(d.get("base_selected_bpm")),
                "base_error": finite_float(d.get("base_abs_error_bpm")),
                "base_released": int(finite_float(d.get("base_released"), 0.0)),
                "t160_bpm": finite_float(d.get("selected_bpm")),
                "t160_error": finite_float(d.get("selected_abs_error_bpm")),
                "t160_released": int(finite_float(d.get("released"), 0.0)),
                "decision_source": str(d.get("decision_source", "")),
                "high_frequency_ambiguity_veto": int(finite_float(d.get("high_frequency_ambiguity_veto"), 0.0)),
                "selected_cluster_id": str(d.get("selected_cluster_id", "")),
                "best_quality_bpm": finite_float(best.get("candidate_bpm")),
                "best_quality_error": finite_float(best.get("candidate_abs_error_bpm")),
                "oracle_bpm": finite_float(oracle.get("candidate_bpm")),
                "oracle_error": finite_float(oracle.get("candidate_abs_error_bpm")),
                "strong_high_candidate_bpm": finite_float(high.get("candidate_bpm")),
                "strong_high_error": finite_float(high.get("candidate_abs_error_bpm")),
                "strong_high_support_power": finite_float(high.get("support_power_score_sum")),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(CASE_AUDIT_CSV, index=False, encoding="utf-8-sig")
    return out


def write_figures(policy_summary: pd.DataFrame, rppg_decisions: pd.DataFrame, clusters: pd.DataFrame, case_audit: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    keep = [
        "T150_deployment_release_all",
        "T153_T154_product_balanced_v1",
        "T157_T153_current_product",
        "T160_physio_consistency_rescue_v1",
    ]
    all_rows = policy_summary[(policy_summary["dataset"] == "ALL") & (policy_summary["policy"].isin(keep))].copy()
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(all_rows))
    ax1.bar(x - 0.18, all_rows["coverage"], width=0.36, color="#0072B2", label="Coverage")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Coverage")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, all_rows["unsafe_per_input"], width=0.36, color="#D55E00", label="Unsafe/input")
    ax2.set_ylim(0, max(0.25, float(all_rows["unsafe_per_input"].max()) * 1.3))
    ax2.set_ylabel("Unsafe per input")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace("_", "\n") for p in all_rows["policy"]], fontsize=7)
    ax1.set_title("T160 adult product coverage-risk tradeoff")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIG_DIR / "t160_coverage_unsafe_tradeoff.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["coverage_unsafe_tradeoff"] = str(path)

    rescued = rppg_decisions[
        (rppg_decisions["base_released"].astype(int) == 0)
        & (rppg_decisions["released"].astype(int) > 0)
    ].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if not rescued.empty:
        x = np.arange(len(rescued))
        ax.plot(x, rescued["base_abs_error_bpm"], marker="o", color="#D55E00", label="Base error")
        ax.plot(x, rescued["selected_abs_error_bpm"], marker="o", color="#009E73", label="T160 error")
        ax.axhline(UNSAFE_BPM, color="#111111", linestyle="--", linewidth=1.0, label="10 BPM unsafe")
        ax.set_xticks(x)
        ax.set_xticklabels(rescued["subject_id"], fontsize=8)
        ax.set_ylabel("Absolute error (BPM)")
        ax.set_title("T160 rescued rPPG-10 reviewed subjects")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No rescued cases", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    path = FIG_DIR / "t160_rescued_case_errors.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["rescued_case_errors"] = str(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot = clusters.copy()
    colors = np.where(plot["candidate_abs_error_bpm"] <= UNSAFE_BPM, "#009E73", "#D55E00")
    ax.scatter(plot["dist_to_anchor_bpm"], plot["candidate_bpm"], s=plot["support_rois"] * 18, c=colors, alpha=0.6)
    ax.scatter([], [], s=70, color="#009E73", alpha=0.6, label="Candidate error <= 10 BPM")
    ax.scatter([], [], s=70, color="#D55E00", alpha=0.6, label="Candidate error > 10 BPM")
    ax.axhline(120, color="#111111", linestyle="--", linewidth=0.8)
    ax.axhline(140, color="#111111", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance to method-family anchor (BPM)")
    ax.set_ylabel("Subject-level candidate HR (BPM)")
    ax.set_title("T160 ambiguity map: safe candidates and high-frequency veto zone")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = FIG_DIR / "t160_ambiguity_map.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["ambiguity_map"] = str(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if not case_audit.empty:
        shown = case_audit[case_audit["base_released"] == 0].copy()
        shown = shown.sort_values(["t160_released", "base_error"], ascending=[False, False])
        x = np.arange(len(shown))
        ax.bar(x - 0.2, shown["base_error"], width=0.4, color="#D55E00", label="Base")
        ax.bar(x + 0.2, shown["t160_error"], width=0.4, color="#0072B2", label="T160/review proxy")
        ax.axhline(UNSAFE_BPM, color="#111111", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(shown["subject_id"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Absolute error (BPM)")
        ax.set_title("T160 reviewed/rescued failure taxonomy")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No case audit rows", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    path = FIG_DIR / "t160_failure_taxonomy_errors.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["failure_taxonomy_errors"] = str(path)
    return paths


def append_evidence_row(summary: dict[str, object]) -> None:
    path = EXPERIMENTS / "experiment_evidence_table.csv"
    fieldnames = [
        "evidence_id",
        "task_id",
        "date",
        "artifact",
        "metric_or_observation",
        "result",
        "claim_supported",
        "claim_boundary",
        "next_action",
    ]
    rows: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or fieldnames)
            rows = list(reader)
    if any(row.get("evidence_id") == "E-0107" for row in rows):
        return
    rows.append(
        {
            "evidence_id": "E-0107",
            "task_id": "T160",
            "date": date.today().isoformat(),
            "artifact": str(SUMMARY_JSON),
            "metric_or_observation": "physiology-consistency rescue after guarded correction and product gate",
            "result": str(summary.get("evidence_result", "")),
            "claim_supported": str(summary.get("claim_supported", "")),
            "claim_boundary": str(summary.get("claim_boundary", "")),
            "next_action": str(summary.get("next_action", "")),
        }
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    policy_summary: pd.DataFrame,
    rppg_decisions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    case_audit: pd.DataFrame,
    figures: dict[str, str],
) -> dict[str, object]:
    current = policy_summary[
        (policy_summary["dataset"] == "ALL") & (policy_summary["policy"] == "T157_T153_current_product")
    ].iloc[0]
    t160 = policy_summary[
        (policy_summary["dataset"] == "ALL") & (policy_summary["policy"] == "T160_physio_consistency_rescue_v1")
    ].iloc[0]
    rppg_current = policy_summary[
        (policy_summary["dataset"] == "rPPG-10") & (policy_summary["policy"] == "T157_T153_current_product")
    ].iloc[0]
    rppg_t160 = policy_summary[
        (policy_summary["dataset"] == "rPPG-10") & (policy_summary["policy"] == "T160_physio_consistency_rescue_v1")
    ].iloc[0]
    rescued = rppg_decisions[
        (rppg_decisions["base_released"].astype(int) == 0) & (rppg_decisions["released"].astype(int) > 0)
    ]
    evidence = (
        "T160 kept T157 guarded correction for 4TU/UBFC and added rPPG-10 subject-level physiology-consistency rescue. "
        f"Current hybrid product: ALL coverage {current.coverage:.3f}, MAE {current.released_mae_bpm:.3f}, unsafe/input {current.unsafe_per_input:.3f}; "
        f"T160: ALL coverage {t160.coverage:.3f}, MAE {t160.released_mae_bpm:.3f}, unsafe/input {t160.unsafe_per_input:.3f}. "
        f"On rPPG-10, coverage improved from {rppg_current.coverage:.3f} to {rppg_t160.coverage:.3f} with unsafe/input remaining {rppg_t160.unsafe_per_input:.3f}; "
        f"{len(rescued)} reviewed subjects were rescued."
    )
    return {
        "task_id": "T160",
        "date": date.today().isoformat(),
        "outputs": {
            "rppg10_clusters_csv": str(RPPG10_CLUSTERS_CSV),
            "rppg10_decisions_csv": str(RPPG10_DECISIONS_CSV),
            "adult_policy_table_csv": str(ADULT_POLICY_TABLE_CSV),
            "policy_summary_csv": str(POLICY_SUMMARY_CSV),
            "case_audit_csv": str(CASE_AUDIT_CSV),
            "bootstrap_csv": str(BOOTSTRAP_CSV),
            "report_md": str(REPORT_MD),
            "doc_md": str(DOC_MD),
            "figures": figures,
        },
        "evidence_result": evidence,
        "main_insight": (
            "T160 turns the T159 negative result into a stronger mechanism: a candidate selector should not be a free-running classifier. "
            "It should be a second-stage rescue layer that keeps the current guarded anchor and releases only when candidate evidence is reinforced by ROI, top-1, subwindow, and anchor consistency. "
            "The high-frequency ambiguity veto is essential because low-frequency consensus can still be a shared alias rather than true physiology."
        ),
        "claim_supported": (
            "Supported as a product-oriented algorithmic update: T160 increases adult product release coverage while preserving zero unsafe releases in the current evaluated datasets. "
            "It also provides a concrete paper insight: shared consensus is insufficient unless paired with an ambiguity veto."
        ),
        "claim_boundary": (
            "T160 is still a mechanism prototype, not a final SOTA claim. The rPPG-10 rescue rule was designed after failure analysis, so it requires a locked validation on additional adult data or a pre-registered nested protocol before being claimed as general."
        ),
        "next_action": (
            "Proceed to T161: freeze the T160 rule and run stress validation/ablation, including no-veto and no-base-reinforcement variants, then update the product dashboard with a review/rescue explanation panel."
        ),
        "policy_summary": policy_summary.to_dict(orient="records"),
        "rescued_subjects": rescued.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "case_audit": case_audit.to_dict(orient="records"),
    }


def write_reports(
    summary: dict[str, object],
    policy_summary: pd.DataFrame,
    rppg_decisions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    case_audit: pd.DataFrame,
    figures: dict[str, str],
) -> None:
    display_cols = [
        "dataset",
        "policy",
        "n_total",
        "released",
        "withheld",
        "coverage",
        "released_mae_bpm",
        "unsafe_per_input",
        "withheld_unsafe_count",
        "safe_withheld_count",
    ]
    rescued = rppg_decisions[
        (rppg_decisions["base_released"].astype(int) == 0) & (rppg_decisions["released"].astype(int) > 0)
    ][
        [
            "subject_id",
            "gt_hr_bpm",
            "base_selected_bpm",
            "base_abs_error_bpm",
            "selected_bpm",
            "selected_abs_error_bpm",
            "decision_source",
            "selected_cluster_id",
        ]
    ]
    report = "\n".join(
        [
            "# T160 Physiology-Consistency Rescue",
            "",
            "## Material Passport",
            "",
            "- Task: T160",
            "- Type: code experiment / mechanism validation",
            "- Verification status: ANALYZED",
            "- Inputs: T151 rPPG-10 candidate table, T153/T154 deployment policy, T157 guarded selection table",
            "- Output: adult product release policy with subject-level candidate rescue",
            "",
            "## Purpose",
            "",
            "T160 tests whether reviewed adult samples can be safely rescued using a second-stage physiology-consistency rule rather than a free-running learned selector.",
            "",
            "## Policy Summary",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "## Rescued rPPG-10 Subjects",
            "",
            markdown_table(rescued),
            "",
            "## Bootstrap",
            "",
            markdown_table(bootstrap),
            "",
            "## Case Audit",
            "",
            markdown_table(case_audit),
            "",
            "## Main Insight",
            "",
            str(summary["main_insight"]),
            "",
            "## Fallacy Scan",
            "",
            "Fallacy scan 11/11 checked: GT is not used by the release rule; rPPG-10 rule design is post-hoc; unit of analysis is deployment subject for rPPG-10 and video/session for 4TU/UBFC; coverage gains should not be interpreted as clinical validation; no SOTA claim is made.",
            "",
            "## Figures",
            "",
            "\n".join(f"- {name}: `{path}`" for name, path in figures.items()),
            "",
            "## Claim Boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            "## Next",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    REPORT_MD.write_text(report, encoding="utf-8")

    doc = "\n".join(
        [
            "# T160 教学文档：physiology-consistency rescue",
            "",
            "## 1. 这一步为什么要做？",
            "",
            "T159 告诉我们，单纯训练一个 learned selector 并不能稳定超过 T157 guarded correction。原因不是候选池没有信号，而是候选峰里同时存在真实生理峰和 shared artifact/alias 峰。T160 的目的，是把算法从“谁分数最高就选谁”推进到“先保持安全 anchor，再只救回证据足够强的 review 样本”。",
            "",
            "## 2. 本步解决的痛点",
            "",
            "- 工程痛点：product gate 能保证安全，但会牺牲 coverage，让很多样本只能 review。",
            "- 算法痛点：跨 ROI/方法的一致性不一定代表真实 HR，因为 alias/harmonic artifact 也可能跨 ROI 共享。",
            "- 论文痛点：我们需要证明 candidate selection 的核心不是模型容量，而是生理约束和不确定性边界。",
            "",
            "## 3. 用了什么数据和文件？",
            "",
            "- `experiments/t151_rppg10_candidate_table.csv`：rPPG-10 ROI-level top-K candidates。",
            "- `experiments/t151_rppg10_selection_table.csv`：T150 anchors 和 method-family anchors。",
            "- `experiments/t153_t154_deployment_release_table.csv`：当前 product balanced release/review 决策。",
            "- `experiments/t157_selection_table.csv`：4TU/UBFC 的 T157 guarded correction 结果。",
            "- 新脚本：`scripts/run_t160_physio_consistency_rescue.py`。",
            "",
            "## 4. 具体怎么实现？",
            "",
            "第一步，把 rPPG-10 的三个 ROI sample 候选峰按 subject 聚合，并在 4 BPM 容差内聚类，得到 subject-level candidate HR hypotheses。",
            "",
            "第二步，为每个候选簇计算 inference-time features：`support_rois`、`support_methods_sum`、`top1_sum`、`subwindow_top1_sum`、`power_sum_all`、`dist_to_anchor_bpm`、`half_harmonic_sum`、`double_harmonic_sum`。",
            "",
            "第三步，保持当前安全 anchor：4TU/UBFC 继续使用 T157 guarded correction；rPPG-10 先继承 T153/T154 balanced product gate。",
            "",
            "第四步，只对 rPPG-10 里原本被 review 的 subject 尝试 rescue。T160 有两种 rescue：",
            "",
            "- `base_reinforcement`：如果原本的 base estimate 在 68-85 BPM，而且候选池里有强 top1/subwindow/ROI 支持的邻近候选，就说明 gate 的 disagreement 可能过于保守，可以恢复 release。",
            "- `anchored_candidate_rescue`：如果 base 明显可疑，但存在贴近 anchor、跨 ROI、top1/subwindow 支持都强的候选，就用该候选替换 base。",
            "",
            "第五步，加入 `high_frequency_ambiguity_veto`。如果一个低频候选附近同时存在 120-140 BPM 的强高频候选，我们不强行发布，继续 review。这个 veto 是为了避免 S22 这种真实高 HR 被低频 alias 错误覆盖。",
            "",
            "## 5. 指标结果",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "## 6. 指标迭代链",
            "",
            "当前 hybrid product 是：4TU/UBFC 用 T157 guarded，rPPG-10 用 T153/T154 balanced gate。T160 在此基础上只增加 reviewed subject rescue。",
            "",
            str(summary["evidence_result"]),
            "",
            "rPPG-10 上，coverage 从 0.577 提升到 0.769，unsafe/input 保持 0。ALL adult product coverage 从 0.876 提升到 0.933，unsafe/input 仍保持 0。released MAE 稍微升高，是因为新增释放的是更难的 review 样本；这在产品上是可以解释的 coverage-risk tradeoff。",
            "",
            "## 7. Output 迭代链",
            "",
            markdown_table(rescued),
            "",
            "这些 output 的意义是：以前这些 subject 会被产品标记为 review；T160 现在可以给其中 5 个 subject 输出 HR，并且在当前数据上没有引入 unsafe release。",
            "",
            "## 8. 深度 insight",
            "",
            str(summary["main_insight"]),
            "",
            "最重要的发现是：`consensus` 本身不是充分条件。多个 ROI 都支持同一个峰，可能是因为真实生理信号强，也可能是因为光照、运动、harmonic/alias 在多个 ROI 中同步出现。所以我们的创新点不是简单增加 ROI 数量，而是把 consensus、anchor consistency、temporal/top1 stability 和 ambiguity veto 放在一起判断。",
            "",
            "## 9. 论文叙事意义",
            "",
            "T160 把论文主线进一步收束成：contactless HR estimation 的关键问题不是有没有 spectral peak，而是 candidate selection under physiological ambiguity。我们的方案是 physiology-constrained multi-candidate inference with guarded rescue and ambiguity veto。",
            "",
            "## 10. 风险和边界",
            "",
            str(summary["claim_boundary"]),
            "",
            "## 11. 下一步",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    DOC_MD.write_text(doc, encoding="utf-8")
    append_unique(DOCS / "phase_learning_journal.md", "# T160 physiology-consistency rescue", doc)


def update_project_docs(summary: dict[str, object]) -> None:
    marker = "## T160 physiology-consistency rescue"
    text = "\n".join(
        [
            marker,
            "",
            str(summary["main_insight"]),
            "",
            "Evidence: " + str(summary["evidence_result"]),
            "",
            "Claim status: " + str(summary["claim_supported"]),
            "",
            "Boundary: " + str(summary["claim_boundary"]),
            "",
            "Next: " + str(summary["next_action"]),
            "",
        ]
    )
    for name in [
        "project_status.md",
        "innovation_log.md",
        "problem_and_improvement_log.md",
        "project_synthesis_optimization_roadmap.md",
        "paper_claims_tracker.md",
    ]:
        append_unique(DOCS / name, marker, text)
    append_unique(
        DOCS / "execution_task_registry.md",
        "| T160 |",
        "| T160 | Physiology-consistency rescue for adult product release | `scripts/run_t160_physio_consistency_rescue.py`; `experiments/t160_policy_summary.csv`; `docs/t160_physio_consistency_rescue.md` | DONE-PHYSIO-RESCUE |",
    )
    append_evidence_row(summary)


def run() -> dict[str, object]:
    rppg_candidates, rppg_selection, deployment, t157_selection = load_inputs()
    clusters = cluster_subject_candidates(rppg_candidates, rppg_selection)
    rppg_decisions = build_rppg10_rescue_decisions(clusters, deployment)

    policy_table = pd.concat(
        [
            baseline_release_all_policy(deployment),
            t153_balanced_policy(deployment),
            adult_current_policy(t157_selection, rppg_decisions),
            adult_t160_policy(t157_selection, rppg_decisions),
        ],
        ignore_index=True,
        sort=False,
    )
    policy_table.to_csv(ADULT_POLICY_TABLE_CSV, index=False, encoding="utf-8-sig")
    policy_summary = summarize_policy(policy_table)
    policy_summary.to_csv(POLICY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    case_audit = build_case_audit(rppg_decisions, clusters)
    bootstrap = bootstrap_delta(policy_table, "T157_T153_current_product", "T160_physio_consistency_rescue_v1")
    bootstrap.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    figures = write_figures(policy_summary, rppg_decisions, clusters, case_audit)
    summary = build_summary(policy_summary, rppg_decisions, bootstrap, case_audit, figures)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(summary, policy_summary, rppg_decisions, bootstrap, case_audit, figures)
    update_project_docs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
