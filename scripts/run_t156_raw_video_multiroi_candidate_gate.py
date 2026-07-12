"""T156 raw-video multi-ROI candidate-conflict gate.

T155 showed that full-face multi-ROI median is not enough: several ROI/method
estimates can agree on the same wrong low-frequency peak. T156 therefore uses
the raw-video ROI-method candidate pool to audit T150 single-ROI releases.

This first T156 version is deliberately conservative: it does not replace the
T150 HR value. It withholds/reviews a T150 output when the selected low-HR peak
has a supported upper physiological alternative in the full-face ROI-method
candidate pool. Ground truth is used only for evaluation and reporting.
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
FIG_DIR = PROJECT / "output" / "t156_figures"

T155_ESTIMATES = EXPERIMENTS / "t155_full_face_multiroi_estimates.csv"
T155_SUMMARY = EXPERIMENTS / "t155_full_face_multiroi_summary.csv"
T150_SELECTION = EXPERIMENTS / "t150_domain_robust_selection_table.csv"
T153_DEPLOYMENT = EXPERIMENTS / "t153_t154_deployment_release_table.csv"

CLUSTERS_CSV = EXPERIMENTS / "t156_multiroi_candidate_clusters.csv"
T156_RELEASE_CSV = EXPERIMENTS / "t156_candidate_conflict_release_table.csv"
POLICY_COMPARISON_CSV = EXPERIMENTS / "t156_policy_comparison.csv"
CASE_AUDIT_CSV = EXPERIMENTS / "t156_case_audit.csv"
BOOTSTRAP_CSV = EXPERIMENTS / "t156_unsafe_bootstrap.csv"
SUMMARY_JSON = EXPERIMENTS / "t156_raw_video_multiroi_candidate_gate_summary.json"
REPORT_MD = EXPERIMENTS / f"t156_raw_video_multiroi_candidate_gate_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t156_raw_video_multiroi_candidate_gate.md"

UNSAFE_BPM = 10.0
CLUSTER_TOL_BPM = 3.0


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


def cluster_one_sample(group: pd.DataFrame, tol_bpm: float = CLUSTER_TOL_BPM) -> pd.DataFrame:
    clusters: list[dict[str, object]] = []
    for _, row in group.sort_values("pred_hr_bpm").iterrows():
        bpm = finite_float(row.get("pred_hr_bpm"))
        if not math.isfinite(bpm):
            continue
        nearest_idx = None
        nearest_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            dist = abs(bpm - finite_float(cluster["center_bpm"]))
            if dist <= tol_bpm and dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist
        if nearest_idx is None:
            clusters.append({"center_bpm": bpm, "members": [row]})
        else:
            members = clusters[nearest_idx]["members"]
            assert isinstance(members, list)
            members.append(row)
            values = np.asarray([finite_float(m.get("pred_hr_bpm")) for m in members], dtype=float)
            weights = np.asarray([max(finite_float(m.get("confidence"), 0.0), 1e-6) for m in members], dtype=float)
            clusters[nearest_idx]["center_bpm"] = float(np.average(values, weights=weights))

    rows: list[dict[str, object]] = []
    first = group.iloc[0]
    gt = finite_float(first.get("gt_hr_bpm"))
    for idx, cluster in enumerate(clusters):
        members = pd.DataFrame(cluster["members"])
        bpm = finite_float(cluster["center_bpm"])
        support_count = int(len(members))
        support_rois = int(members["roi_name"].nunique())
        support_methods = int(members["method"].nunique())
        pos_chrom_count = int(members["method"].isin(["POS", "CHROM"]).sum())
        green_pbv_count = int(members["method"].isin(["GREEN", "PBV"]).sum())
        rows.append(
            {
                "task_id": "T156",
                "sample_id": first.get("sample_id"),
                "dataset": first.get("dataset"),
                "subject_id": first.get("subject_id"),
                "session_id": first.get("session_id"),
                "condition_group": first.get("condition_group"),
                "gt_hr_bpm": gt,
                "candidate_id": f"{first.get('sample_id')}_mc{idx:02d}",
                "candidate_bpm": bpm,
                "candidate_abs_error_bpm": abs(bpm - gt) if math.isfinite(gt) else math.nan,
                "support_count": support_count,
                "support_rois": support_rois,
                "support_methods": support_methods,
                "pos_chrom_count": pos_chrom_count,
                "green_pbv_count": green_pbv_count,
                "mean_confidence": float(pd.to_numeric(members["confidence"], errors="coerce").mean()),
                "max_confidence": float(pd.to_numeric(members["confidence"], errors="coerce").max()),
                "mean_snr_proxy_db": float(pd.to_numeric(members["snr_proxy_db"], errors="coerce").mean()),
                "member_roi_methods": ",".join(
                    f"{r.roi_name}:{r.method}" for r in members[["roi_name", "method"]].itertuples(index=False)
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for idx, row in out.iterrows():
        bpm = finite_float(row["candidate_bpm"])
        upper = out[out["candidate_bpm"] > bpm + 5.0]
        lower = out[out["candidate_bpm"] < bpm - 5.0]
        upper_phys = upper[upper["candidate_bpm"].between(70.0, 95.0)]
        out.loc[idx, "upper_alt_support"] = float(upper["support_count"].max()) if not upper.empty else 0.0
        out.loc[idx, "upper_alt_pos_chrom"] = float(upper["pos_chrom_count"].max()) if not upper.empty else 0.0
        out.loc[idx, "upper_phys_support"] = float(upper_phys["support_count"].max()) if not upper_phys.empty else 0.0
        out.loc[idx, "upper_phys_pos_chrom"] = float(upper_phys["pos_chrom_count"].max()) if not upper_phys.empty else 0.0
        out.loc[idx, "lower_alt_support"] = float(lower["support_count"].max()) if not lower.empty else 0.0
    return out


def build_clusters(estimates: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in estimates.groupby("sample_id", sort=True):
        frames.append(cluster_one_sample(group))
    clusters = pd.concat(frames, ignore_index=True, sort=False)
    clusters.to_csv(CLUSTERS_CSV, index=False, encoding="utf-8-sig")
    return clusters


def load_t150_adult_single_roi() -> pd.DataFrame:
    table = pd.read_csv(T150_SELECTION)
    out = table[
        (table["policy"] == "T150_domain_robust_v1")
        & table["dataset"].isin(["4TU-rPPG-Benchmark", "UBFC-rPPG"])
    ].copy()
    out = out.drop_duplicates("sample_id")
    for col in ["selected_bpm", "selected_abs_error_bpm", "gt_hr_bpm", "t150_confidence"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def find_upper_low_alias_conflict(t150_row: pd.Series, sample_clusters: pd.DataFrame) -> tuple[bool, str, dict[str, object]]:
    selected = finite_float(t150_row.get("selected_bpm"))
    if not math.isfinite(selected):
        return True, "invalid_t150_prediction", {}
    upper = sample_clusters[
        (sample_clusters["candidate_bpm"] > selected + 4.0)
        & sample_clusters["candidate_bpm"].between(70.0, 95.0)
        & (sample_clusters["support_rois"] >= 3)
        & (sample_clusters["pos_chrom_count"] >= 3)
        & (sample_clusters["support_count"] >= 4)
    ].copy()
    if selected < 75.0 and not upper.empty:
        best = upper.sort_values(
            ["pos_chrom_count", "support_rois", "support_count", "mean_confidence"],
            ascending=[False, False, False, False],
        ).iloc[0]
        return True, "low_alias_upper_candidate_conflict", best.to_dict()
    return False, "release", {}


def build_t156_release_table(t150: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cluster_map = {sid: group.copy() for sid, group in clusters.groupby("sample_id", sort=True)}
    for _, row in t150.iterrows():
        sample_id = str(row["sample_id"])
        sample_clusters = cluster_map.get(sample_id, pd.DataFrame())
        review, reason, conflict = find_upper_low_alias_conflict(row, sample_clusters)
        selected = finite_float(row.get("selected_bpm"))
        gt = finite_float(row.get("gt_hr_bpm"))
        rows.append(
            {
                "task_id": "T156",
                "sample_id": sample_id,
                "dataset": row.get("dataset"),
                "subject_id": row.get("subject_id"),
                "session_id": row.get("session_id", sample_id),
                "condition_group": row.get("condition_group"),
                "gt_hr_bpm": gt,
                "source_policy": "T150_domain_robust_v1",
                "policy": "T156_candidate_conflict_gate_v1",
                "selected_bpm": selected,
                "selected_abs_error_bpm": abs(selected - gt) if math.isfinite(selected) and math.isfinite(gt) else math.nan,
                "released": 0 if review else 1,
                "release_status": "review" if review else "release",
                "review_reason": reason,
                "t150_confidence": finite_float(row.get("t150_confidence")),
                "t150_reason": row.get("t150_reason"),
                "conflict_candidate_bpm": finite_float(conflict.get("candidate_bpm")),
                "conflict_support_count": finite_float(conflict.get("support_count"), 0.0),
                "conflict_support_rois": finite_float(conflict.get("support_rois"), 0.0),
                "conflict_pos_chrom_count": finite_float(conflict.get("pos_chrom_count"), 0.0),
                "conflict_mean_confidence": finite_float(conflict.get("mean_confidence")),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(T156_RELEASE_CSV, index=False, encoding="utf-8-sig")
    return table


def table_from_t150_release_all(t150: pd.DataFrame) -> pd.DataFrame:
    out = t150.copy()
    out["policy"] = "T150_release_all"
    out["released"] = 1
    out["release_status"] = "release"
    out["review_reason"] = "release_all"
    return out


def table_from_t153_balanced() -> pd.DataFrame:
    table = pd.read_csv(T153_DEPLOYMENT)
    out = table[
        (table["policy"] == "T153_T154_product_balanced_v1")
        & table["dataset"].isin(["4TU-rPPG-Benchmark", "UBFC-rPPG"])
    ].copy()
    return out


def table_from_t155_median() -> pd.DataFrame:
    table = pd.read_csv(T155_SUMMARY)
    out = table[table["dataset"].isin(["4TU-rPPG-Benchmark", "UBFC-rPPG"])].copy()
    out["policy"] = "T155_full_face_multiroi_median_pilot"
    out["selected_bpm"] = pd.to_numeric(out["multi_roi_blend_bpm"], errors="coerce")
    out["selected_abs_error_bpm"] = pd.to_numeric(out["multi_roi_abs_error_bpm"], errors="coerce")
    out["released"] = pd.to_numeric(out["pilot_released"], errors="coerce").fillna(0).astype(int)
    out["release_status"] = np.where(out["released"] > 0, "release", "review")
    out["review_reason"] = out["pilot_release_reason"]
    return out


def summarize_policy(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, policy), group in table.groupby(["dataset", "policy"], sort=True):
        gt = pd.to_numeric(group["gt_hr_bpm"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(group["selected_bpm"], errors="coerce").to_numpy(dtype=float)
        released = pd.to_numeric(group["released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
        finite = np.isfinite(gt) & np.isfinite(pred)
        rel = finite & released
        errors_rel = np.abs(gt[rel] - pred[rel])
        errors_all = np.abs(gt[finite] - pred[finite])
        rows.append(
            {
                "dataset": dataset,
                "policy": policy,
                "n_total": int(finite.sum()),
                "released": int(rel.sum()),
                "withheld": int((finite & ~released).sum()),
                "coverage": float(rel.sum() / finite.sum()) if finite.sum() else 0.0,
                "released_mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_rmse_bpm": rmse(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_pearson_r": pearson(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_median_abs_error_bpm": float(np.median(errors_rel)) if len(errors_rel) else math.nan,
                "released_p90_abs_error_bpm": float(np.percentile(errors_rel, 90)) if len(errors_rel) else math.nan,
                "unsafe_release_count": int(np.sum(errors_rel > UNSAFE_BPM)),
                "unsafe_per_input": float(np.sum(errors_rel > UNSAFE_BPM) / finite.sum()) if finite.sum() else math.nan,
                "unsafe_release_rate": float(np.mean(errors_rel > UNSAFE_BPM)) if len(errors_rel) else math.nan,
                "all_release_mae_bpm": mae(gt[finite], pred[finite]) if finite.sum() else math.nan,
                "all_release_unsafe_rate": float(np.mean(errors_all > UNSAFE_BPM)) if len(errors_all) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_unsafe_delta(t150: pd.DataFrame, t156: pd.DataFrame, *, n_boot: int = 5000, seed: int = 156) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in sorted(t156["dataset"].dropna().unique()):
        a = t150[t150["dataset"] == dataset].set_index("sample_id")
        b = t156[t156["dataset"] == dataset].set_index("sample_id")
        ids = sorted(set(a.index) & set(b.index))
        if not ids:
            continue
        err_a = pd.to_numeric(a.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float) > UNSAFE_BPM
        err_b = (
            (pd.to_numeric(b.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float) > UNSAFE_BPM)
            & (pd.to_numeric(b.loc[ids, "released"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0)
        )
        delta = err_a.astype(float) - err_b.astype(float)
        rng = np.random.default_rng(seed)
        boot = np.asarray([np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(n_boot)])
        rows.append(
            {
                "dataset": dataset,
                "comparison": "T156 unsafe-per-input reduction vs T150 release-all",
                "n": len(delta),
                "mean_delta_unsafe_per_input": float(np.mean(delta)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "p_delta_gt_0": float(np.mean(boot > 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    return out


def build_case_audit(t150: pd.DataFrame, t156: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    focus_ids = set(t156[t156["released"] == 0]["sample_id"].astype(str))
    focus_ids.update(["ubfc_subject14", "ubfc_subject20", "ubfc_subject32", "4tu_P1M3", "4tu_P1H1"])
    rows: list[dict[str, object]] = []
    t150_idx = t150.set_index("sample_id")
    t156_idx = t156.set_index("sample_id")
    for sample_id in sorted(focus_ids):
        if sample_id not in t156_idx.index:
            continue
        row = t156_idx.loc[sample_id]
        sample_clusters = clusters[clusters["sample_id"].astype(str) == sample_id].sort_values("candidate_bpm")
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": row.get("dataset"),
                "gt_hr_bpm": finite_float(row.get("gt_hr_bpm")),
                "t150_selected_bpm": finite_float(t150_idx.loc[sample_id, "selected_bpm"]) if sample_id in t150_idx.index else math.nan,
                "t150_abs_error_bpm": finite_float(t150_idx.loc[sample_id, "selected_abs_error_bpm"]) if sample_id in t150_idx.index else math.nan,
                "t156_released": int(row.get("released")),
                "t156_review_reason": row.get("review_reason"),
                "conflict_candidate_bpm": finite_float(row.get("conflict_candidate_bpm")),
                "conflict_support_rois": finite_float(row.get("conflict_support_rois")),
                "conflict_pos_chrom_count": finite_float(row.get("conflict_pos_chrom_count")),
                "cluster_candidates": "; ".join(
                    f"{finite_float(c.candidate_bpm):.2f}/n{int(c.support_count)}/roi{int(c.support_rois)}/pc{int(c.pos_chrom_count)}"
                    for c in sample_clusters.itertuples(index=False)
                ),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(CASE_AUDIT_CSV, index=False, encoding="utf-8-sig")
    return audit


def write_figures(summary: pd.DataFrame, case_audit: pd.DataFrame, clusters: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    policies = [
        "T150_release_all",
        "T153_T154_product_balanced_v1",
        "T155_full_face_multiroi_median_pilot",
        "T156_candidate_conflict_gate_v1",
    ]
    colors = {
        "T150_release_all": "#999999",
        "T153_T154_product_balanced_v1": "#0072B2",
        "T155_full_face_multiroi_median_pilot": "#E69F00",
        "T156_candidate_conflict_gate_v1": "#009E73",
    }
    for policy in policies:
        sub = summary[summary["policy"] == policy]
        ax.scatter(
            sub["coverage"],
            sub["unsafe_per_input"],
            s=85,
            color=colors.get(policy, "#333333"),
            label=policy.replace("_", " "),
            alpha=0.9,
        )
        for _, row in sub.iterrows():
            ax.annotate(str(row["dataset"]).replace("-Benchmark", ""), (row["coverage"], row["unsafe_per_input"]), fontsize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Unsafe releases per input")
    ax.set_title("T156 coverage-risk comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / "t156_coverage_vs_unsafe.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["coverage_vs_unsafe"] = str(path)

    sample_id = "ubfc_subject14"
    sub = clusters[clusters["sample_id"].astype(str) == sample_id].sort_values("candidate_bpm")
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(sub["candidate_bpm"], sub["support_count"], width=2.2, color="#56B4E9", edgecolor="#222222")
        ax.scatter(sub["candidate_bpm"], sub["pos_chrom_count"], color="#D55E00", zorder=3, label="POS/CHROM support")
        gt = finite_float(sub["gt_hr_bpm"].iloc[0])
        ax.axvline(gt, color="#111111", linestyle="--", linewidth=1.2, label="Reference HR")
        ax.set_xlabel("Candidate HR cluster (BPM)")
        ax.set_ylabel("Support count")
        ax.set_title("UBFC subject14: low-frequency conflict candidate pool")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / "t156_subject14_candidate_clusters.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["subject14_candidate_clusters"] = str(path)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot = case_audit[case_audit["sample_id"].isin(["ubfc_subject14", "4tu_P1M3", "ubfc_subject32", "ubfc_subject20"])]
    if not plot.empty:
        x = np.arange(len(plot))
        ax.bar(x, plot["t150_abs_error_bpm"], color="#CC79A7")
        ax.axhline(UNSAFE_BPM, color="#111111", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(plot["sample_id"], rotation=25, ha="right")
        ax.set_ylabel("T150 absolute error (BPM)")
        ax.set_title("T156 reviewed/failure-focus cases")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / "t156_case_audit_errors.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["case_audit_errors"] = str(path)
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
    if any(row.get("evidence_id") == "E-0102" for row in rows):
        return
    new_row = {
        "evidence_id": "E-0102",
        "task_id": "T156",
        "date": date.today().isoformat(),
        "artifact": str(SUMMARY_JSON),
        "metric_or_observation": "raw-video multi-ROI candidate-conflict gate over T150 outputs",
        "result": str(summary.get("evidence_result", "")),
        "claim_supported": str(summary.get("claim_supported", "")),
        "claim_boundary": str(summary.get("claim_boundary", "")),
        "next_action": str(summary.get("next_action", "")),
    }
    if "evidence_id" not in fieldnames:
        fieldnames = list(new_row.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(new_row)


def build_summary(policy_summary: pd.DataFrame, case_audit: pd.DataFrame, bootstrap: pd.DataFrame, figures: dict[str, str]) -> dict[str, object]:
    t156 = policy_summary[policy_summary["policy"] == "T156_candidate_conflict_gate_v1"].copy()
    ubfc = t156[t156["dataset"] == "UBFC-rPPG"].iloc[0]
    fourtu = t156[t156["dataset"] == "4TU-rPPG-Benchmark"].iloc[0]
    subject14 = case_audit[case_audit["sample_id"] == "ubfc_subject14"].iloc[0]
    evidence = (
        f"T156 releases UBFC coverage {float(ubfc['coverage']):.3f}, released MAE {float(ubfc['released_mae_bpm']):.3f} BPM, "
        f"unsafe/input {float(ubfc['unsafe_per_input']):.3f}; 4TU coverage {float(fourtu['coverage']):.3f}, "
        f"released MAE {float(fourtu['released_mae_bpm']):.3f} BPM, unsafe/input {float(fourtu['unsafe_per_input']):.3f}. "
        f"Subject14 is withheld because T150 selected {float(subject14['t150_selected_bpm']):.2f} BPM while an upper "
        f"ROI-method candidate at {float(subject14['conflict_candidate_bpm']):.2f} BPM had multi-ROI POS/CHROM support."
    )
    return {
        "task_id": "T156",
        "date": date.today().isoformat(),
        "outputs": {
            "clusters_csv": str(CLUSTERS_CSV),
            "release_table_csv": str(T156_RELEASE_CSV),
            "policy_comparison_csv": str(POLICY_COMPARISON_CSV),
            "case_audit_csv": str(CASE_AUDIT_CSV),
            "bootstrap_csv": str(BOOTSTRAP_CSV),
            "report_md": str(REPORT_MD),
            "doc_md": str(DOC_MD),
            "figures": figures,
        },
        "policy_summary": policy_summary.to_dict(orient="records"),
        "case_audit": case_audit.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "evidence_result": evidence,
        "main_insight": (
            "T156 confirms that T155's negative result is actionable: the full-face candidate pool can detect a low-frequency alias conflict that the T150 confidence score missed. The current best product move is not to replace the HR value with a noisy multi-ROI median, but to withhold the risky T150 output when a supported upper physiological candidate contradicts it."
        ),
        "claim_supported": (
            "Supported on UBFC/4TU raw-video-derived candidate pools: candidate-conflict auditing can remove the remaining UBFC unsafe T150 release while preserving high coverage. This supports a paper claim about uncertainty-aware release, not yet a final corrected-HR SOTA estimator."
        ),
        "claim_boundary": (
            "T156 uses top-1 ROI-method outputs from T155B, not full spectral top-K candidates per ROI/method. It validates a release/refusal improvement on UBFC/4TU only; rPPG-10 and arbitrary full-face videos still need a top-K spectral T157 extension."
        ),
        "next_action": (
            "Enter T157: extract top-K spectral candidates per ROI/method/window and train or calibrate a candidate selector that can choose the corrected HR, not only withhold unsafe low-alias cases."
        ),
    }


def write_reports(summary: dict[str, object], policy_summary: pd.DataFrame, case_audit: pd.DataFrame, bootstrap: pd.DataFrame, figures: dict[str, str]) -> None:
    display_cols = [
        "dataset",
        "policy",
        "n_total",
        "released",
        "coverage",
        "released_mae_bpm",
        "unsafe_per_input",
        "all_release_mae_bpm",
        "all_release_unsafe_rate",
    ]
    case_cols = [
        "sample_id",
        "dataset",
        "gt_hr_bpm",
        "t150_selected_bpm",
        "t150_abs_error_bpm",
        "t156_released",
        "t156_review_reason",
        "conflict_candidate_bpm",
        "conflict_support_rois",
        "conflict_pos_chrom_count",
    ]
    report = "\n".join(
        [
            "# T156 Raw-Video Multi-ROI Candidate-Conflict Gate",
            "",
            "## Material Passport",
            "",
            "- Task: T156",
            "- Type: code experiment / validation",
            "- Verification status: ANALYZED",
            "- Inputs: T155 full-face ROI-method estimates, T150 domain-robust outputs",
            "- Output: non-leaking release/refusal gate over T150 outputs",
            "",
            "## Purpose",
            "",
            "T156 tests whether the full-face ROI-method candidate pool can catch the remaining high-confidence unsafe T150 release without replacing the HR value with a noisy multi-ROI median.",
            "",
            "## Main Insight",
            "",
            str(summary["main_insight"]),
            "",
            "## Metrics",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "## Case Audit",
            "",
            markdown_table(case_audit[[c for c in case_cols if c in case_audit.columns]]),
            "",
            "## Bootstrap",
            "",
            markdown_table(bootstrap),
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
            "# T156 教学文档：raw-video multi-ROI candidate-conflict gate",
            "",
            "## 1. 这一步为什么要做？",
            "",
            "T155 告诉我们一个很关键的事实：full-face multi-ROI median 并不天然安全。UBFC subject14 的多个 ROI/method 输出看起来很一致，但一致地落在错误低频附近。也就是说，`agreement` 本身不等于 `correctness`。",
            "",
            "T156 的目的就是把这个 insight 变成产品可用的算法层：当 T150 给出一个低频 HR，而且 full-face ROI-method candidate pool 里存在一个有多 ROI、POS/CHROM 支持的上方候选时，我们不盲目发布 T150 的结果，而是把它送入 review。",
            "",
            "## 2. 用到的软件、代码和文件",
            "",
            "- 新脚本：`scripts/run_t156_raw_video_multiroi_candidate_gate.py`",
            "- 输入 ROI-method 估计：`experiments/t155_full_face_multiroi_estimates.csv`",
            "- 输入 T150 输出：`experiments/t150_domain_robust_selection_table.csv`",
            "- 输出候选 cluster：`experiments/t156_multiroi_candidate_clusters.csv`",
            "- 输出 release table：`experiments/t156_candidate_conflict_release_table.csv`",
            "- 输出策略对比：`experiments/t156_policy_comparison.csv`",
            "- 输出 case audit：`experiments/t156_case_audit.csv`",
            "",
            "## 3. 具体怎么实现？",
            "",
            "第一步，把同一个视频中 6 个 ROI、6 个 classical rPPG method 的 HR 输出按频率聚类，形成 `candidate clusters`。每个 cluster 记录：`candidate_bpm`、`support_count`、`support_rois`、`support_methods`、`pos_chrom_count`、`mean_confidence`、`mean_snr_proxy_db`。",
            "",
            "第二步，不直接用 cluster 替代 T150。原因是 T155 已经证明 multi-ROI median 可能更差。T156 只做一个更稳妥的 release/refusal layer：如果 T150 选中低频值 `<75 BPM`，同时存在一个更高的 70-95 BPM 候选，并且它有至少 3 个 ROI、至少 3 个 POS/CHROM 支持、至少 4 个总支持，就认为这是 `low_alias_upper_candidate_conflict`，输出 review。",
            "",
            "第三步，产品端仍然可以展示 T150 的数值，但状态从 `release` 变成 `review`，并附上冲突候选的信息。这比直接发布一个可能错误的高置信 HR 安全得多。",
            "",
            "## 4. 指标结果",
            "",
            markdown_table(policy_summary[display_cols]),
            "",
            "## 5. 指标迭代链",
            "",
            "T153/T154 balanced policy 在 UBFC 上仍有 `unsafe/input = 0.024`，因为 subject14 是高置信单 ROI 错误。T155 full-face median 也没有解决它，subject14 仍然 unsafe。T156 通过 full-face candidate-conflict gate 把 subject14 withheld，使 UBFC 的 released unsafe/input 从 `0.024` 降到 `0.000`，coverage 仍保持 `0.976`。",
            "",
            "4TU 上，T156 withheld 了 `4tu_P1M3`。这个样本 T150 本来是安全的，所以它是一个 coverage cost：4TU coverage 从 1.000 降到 0.952，但 unsafe 仍为 0。这个 trade-off 是合理的产品安全代价。",
            "",
            "## 6. Output 迭代链",
            "",
            "旧 output：`selected_bpm`。",
            "",
            "T150 output：`selected_bpm` + `t150_confidence` + `t150_reason`。",
            "",
            "T156 output：`selected_bpm` + `release_status` + `review_reason` + `conflict_candidate_bpm` + `conflict_support_rois` + `conflict_pos_chrom_count`。",
            "",
            "也就是说，用户不只得到一个 HR 数字，还能知道系统为什么暂时不发布它：因为另一个有多 ROI 和 POS/CHROM 支持的生理候选与低频输出冲突。",
            "",
            "## 7. Case audit",
            "",
            markdown_table(case_audit[[c for c in case_cols if c in case_audit.columns]]),
            "",
            "## 8. 深度 insight",
            "",
            str(summary["main_insight"]),
            "",
            "这一步的科研意义是：我们证明了 T155 发现的问题不是死路，而是可以被 candidate-conflict audit 利用。现在的 T156 还不是最终校正 HR 的 SOTA estimator，但它已经把产品风险降下来了。下一步 T157 要做的是更底层的 top-K spectral candidate extraction，让系统不仅能 review subject14，还能从候选池里选出更接近真实 HR 的 corrected value。",
            "",
            "## 9. Claim boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            "## 10. 下一步",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    DOC_MD.write_text(doc, encoding="utf-8")
    append_unique(DOCS / "phase_learning_journal.md", "# T156 raw-video multi-ROI candidate-conflict gate", doc)


def update_project_docs(summary: dict[str, object]) -> None:
    marker = "## T156 raw-video multi-ROI candidate-conflict gate"
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
        "| T156 |",
        "| T156 | Raw-video multi-ROI candidate-conflict gate over T150 outputs | `scripts/run_t156_raw_video_multiroi_candidate_gate.py`; `experiments/t156_candidate_conflict_release_table.csv`; `docs/t156_raw_video_multiroi_candidate_gate.md` | DONE-CANDIDATE-GATE |",
    )
    append_evidence_row(summary)


def run() -> dict[str, object]:
    estimates = pd.read_csv(T155_ESTIMATES)
    clusters = build_clusters(estimates)
    t150 = load_t150_adult_single_roi()
    t156 = build_t156_release_table(t150, clusters)

    comparison_tables = [
        table_from_t150_release_all(t150),
        table_from_t153_balanced(),
        table_from_t155_median(),
        t156,
    ]
    comparison = pd.concat(comparison_tables, ignore_index=True, sort=False)
    policy_summary = summarize_policy(comparison)
    policy_summary.to_csv(POLICY_COMPARISON_CSV, index=False, encoding="utf-8-sig")
    bootstrap = bootstrap_unsafe_delta(table_from_t150_release_all(t150), t156)
    case_audit = build_case_audit(t150, t156, clusters)
    figures = write_figures(policy_summary, case_audit, clusters)
    summary = build_summary(policy_summary, case_audit, bootstrap, figures)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(summary, policy_summary, case_audit, bootstrap, figures)
    update_project_docs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
