from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "T901"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

T759_DECISIONS = EXP / "t759_targeted_external_rescue_decisions.csv"
T760_DECISIONS = EXP / "t760_original_domain_revalidation_decisions.csv"
T160_RPPG10 = EXP / "t160_rppg10_rescue_decisions.csv"
T730_POOL = EXP / "t730_harmonic_aware_candidate_pool.csv"
T675_PHYS_SELECTIONS = EXP / "t675_ubfc_phys_loso_subject_selector_selections.csv"

OUT_INPUT = EXP / "t901_subject_cluster_input_audit.csv"
OUT_SUBJECT = EXP / "t901_subject_level_metrics.csv"
OUT_BOOT = EXP / "t901_subject_cluster_bootstrap.csv"
OUT_SUMMARY = EXP / "t901_subject_cluster_summary.csv"
OUT_THRESH = EXP / "t901_threshold_sensitivity_subject.csv"
OUT_CLAIM = EXP / "t901_claim_gate.csv"
OUT_JSON = EXP / "t901_subject_cluster_bootstrap_summary.json"
OUT_MD = DOCS / "t901_subject_cluster_bootstrap.md"

SEEDS = [2024, 2025, 2026]
BOOT = 3000
THRESHOLDS = [5.0, 8.0, 10.0, 15.0]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def build_max_snr_baseline(pool: pd.DataFrame) -> pd.DataFrame:
    work = pool.copy()
    work["mean_snr_proxy_db"] = num(work, "mean_snr_proxy_db", -np.inf)
    work["candidate_abs_error"] = num(work, "candidate_abs_error")
    rows = (
        work.sort_values(["dataset", "sample_id", "mean_snr_proxy_db"], ascending=[True, True, False])
        .groupby(["dataset", "sample_id"], as_index=False)
        .first()
    )
    return rows[
        [
            "dataset",
            "sample_id",
            "subject_std",
            "candidate_hr_bpm",
            "gt_hr_bpm",
            "candidate_abs_error",
            "mean_snr_proxy_db",
        ]
    ].rename(
        columns={
            "subject_std": "baseline_subject_std",
            "candidate_hr_bpm": "baseline_hr_bpm",
            "candidate_abs_error": "baseline_abs_error_bpm",
            "mean_snr_proxy_db": "baseline_snr_proxy_db",
        }
    )


def build_phys_dense_baseline() -> pd.DataFrame:
    phys = read_csv(T675_PHYS_SELECTIONS)
    dense = phys[phys["variant"].astype(str).eq("dense_best_snr")].copy()
    if dense.empty:
        raise RuntimeError("Missing UBFC-Phys dense_best_snr rows.")
    return pd.DataFrame(
        {
            "dataset": "UBFC-Phys-S1-S14",
            "sample_id": dense["clip_id"].astype(str),
            "baseline_subject_std": dense["subject"].astype(str),
            "baseline_hr_bpm": num(dense, "candidate_hr_bpm"),
            "gt_hr_bpm": num(dense, "reference_hr_bpm"),
            "baseline_abs_error_bpm": num(dense, "candidate_abs_error"),
            "baseline_snr_proxy_db": num(dense, "candidate_snr_db"),
        }
    )


def load_method_decisions() -> pd.DataFrame:
    t759 = read_csv(T759_DECISIONS)
    t760 = read_csv(T760_DECISIONS)
    first_three = pd.concat([t759, t760], ignore_index=True, sort=False)
    first_three_out = pd.DataFrame(
        {
            "dataset": first_three["dataset"].astype(str),
            "sample_id": first_three["sample_id"].astype(str),
            "subject_std": first_three["subject_std"].astype(str),
            "gt_hr_bpm": num(first_three, "gt_hr_bpm"),
            "selected_hr_bpm": num(first_three, "pred_hr_bpm"),
            "selected_abs_error_bpm": num(first_three, "abs_error_bpm"),
            "release_decision": first_three["release_decision"].astype(str),
            "release_risk": num(first_three, "release_risk"),
            "decision_source": first_three.get("rescue_strategy", first_three.get("strategy", "unknown")).astype(str),
        }
    )

    r10 = read_csv(T160_RPPG10)
    r10_out = pd.DataFrame(
        {
            "dataset": "rPPG-10",
            "sample_id": r10["deployment_id"].astype(str),
            "subject_std": r10["subject_id"].astype(str),
            "gt_hr_bpm": num(r10, "gt_hr_bpm"),
            "selected_hr_bpm": num(r10, "selected_bpm"),
            "selected_abs_error_bpm": num(r10, "selected_abs_error_bpm"),
            "release_decision": np.where(num(r10, "released", 0.0).astype(int).eq(1), "release", "review"),
            "release_risk": np.where(num(r10, "released", 0.0).astype(int).eq(1), 0.0, 1.0),
            "decision_source": r10["decision_source"].astype(str),
        }
    )
    return pd.concat([first_three_out, r10_out], ignore_index=True, sort=False)


def attach_baselines(decisions: pd.DataFrame) -> pd.DataFrame:
    pool = read_csv(T730_POOL)
    max_snr = build_max_snr_baseline(pool)
    phys_dense = build_phys_dense_baseline()

    # Use dense_best_snr for UBFC-Phys because the T759 branch was calibrated
    # against the T675 dense baseline; use max-SNR candidate rows elsewhere.
    base_parts = [
        max_snr[max_snr["dataset"].astype(str).isin(["MCD-rPPG", "UBFC-rPPG"])].copy(),
        phys_dense,
    ]
    baseline = pd.concat(base_parts, ignore_index=True, sort=False)
    first_three = decisions[decisions["dataset"].astype(str).isin(["MCD-rPPG", "UBFC-Phys-S1-S14", "UBFC-rPPG"])].copy()
    merged = first_three.merge(
        baseline[
            [
                "dataset",
                "sample_id",
                "baseline_subject_std",
                "baseline_hr_bpm",
                "baseline_abs_error_bpm",
                "baseline_snr_proxy_db",
            ]
        ],
        on=["dataset", "sample_id"],
        how="left",
    )

    r10 = read_csv(T160_RPPG10)
    r10_base = pd.DataFrame(
        {
            "dataset": "rPPG-10",
            "sample_id": r10["deployment_id"].astype(str),
            "baseline_subject_std": r10["subject_id"].astype(str),
            "baseline_hr_bpm": num(r10, "base_selected_bpm"),
            "baseline_abs_error_bpm": num(r10, "base_abs_error_bpm"),
            "baseline_snr_proxy_db": np.nan,
        }
    )
    r10_decisions = decisions[decisions["dataset"].astype(str).eq("rPPG-10")].copy()
    r10_merged = r10_decisions.merge(r10_base, on=["dataset", "sample_id"], how="left")
    out = pd.concat([merged, r10_merged], ignore_index=True, sort=False)
    out["subject_std"] = np.where(
        out["subject_std"].astype(str).isin(["", "nan", "None"]),
        out["baseline_subject_std"].astype(str),
        out["subject_std"].astype(str),
    )
    out["baseline_abs_error_bpm"] = num(out, "baseline_abs_error_bpm")
    out["selected_abs_error_bpm"] = num(out, "selected_abs_error_bpm")
    out["baseline_missing"] = out["baseline_abs_error_bpm"].isna()
    return out


def subject_metrics(input_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, subject), g in input_df.groupby(["dataset", "subject_std"], sort=False):
        released = g[g["release_decision"].astype(str).eq("release")]
        rel_err = num(released, "selected_abs_error_bpm")
        row: dict[str, Any] = {
            "dataset": dataset,
            "subject_std": subject,
            "n_samples": int(len(g)),
            "n_released": int(len(released)),
            "baseline_mae_bpm": float(num(g, "baseline_abs_error_bpm").mean()),
            "method_mae_bpm": float(num(g, "selected_abs_error_bpm").mean()),
            "release_coverage": float(len(released) / len(g)) if len(g) else math.nan,
            "released_mae_bpm": float(rel_err.mean()) if len(released) else math.nan,
        }
        row["mae_improvement_bpm"] = row["baseline_mae_bpm"] - row["method_mae_bpm"]
        row["mae_reduction_vs_baseline"] = (
            1.0 - row["method_mae_bpm"] / row["baseline_mae_bpm"]
            if row["baseline_mae_bpm"] > 0
            else math.nan
        )
        for threshold in THRESHOLDS:
            row[f"unsafe_release_gt{int(threshold)}bpm"] = (
                float((rel_err > threshold).mean()) if len(released) else math.nan
            )
            row[f"all_unsafe_gt{int(threshold)}bpm"] = float((num(g, "selected_abs_error_bpm") > threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_subject_rows(subj: pd.DataFrame, threshold: float = 10.0) -> dict[str, float]:
    released = subj[subj["n_released"] > 0].copy()
    baseline_mae = float(subj["baseline_mae_bpm"].mean())
    method_mae = float(subj["method_mae_bpm"].mean())
    released_mae = float(released["released_mae_bpm"].mean()) if len(released) else math.nan
    unsafe_col = f"unsafe_release_gt{int(threshold)}bpm"
    unsafe = float(released[unsafe_col].mean()) if len(released) else math.nan
    coverage = float(subj["release_coverage"].mean()) if len(subj) else math.nan
    return {
        "n_subjects": float(len(subj)),
        "baseline_mae_bpm": baseline_mae,
        "method_mae_bpm": method_mae,
        "mae_improvement_bpm": baseline_mae - method_mae,
        "mae_reduction_vs_baseline": 1.0 - method_mae / baseline_mae if baseline_mae > 0 else math.nan,
        "release_coverage": coverage,
        "released_mae_bpm": released_mae,
        "unsafe_release_rate": unsafe,
    }


def ci(arr: np.ndarray) -> tuple[float, float]:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_dataset(dataset: str, subj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = aggregate_subject_rows(subj, threshold=10.0)
    boot_rows: list[dict[str, Any]] = []
    subj = subj.reset_index(drop=True)
    n_subjects = len(subj)
    if n_subjects == 0:
        raise RuntimeError(f"No subjects for {dataset}")
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for b in range(BOOT):
            idx = rng.integers(0, n_subjects, n_subjects)
            sample = subj.iloc[idx]
            row = {"dataset": dataset, "seed": seed, "bootstrap_id": b}
            row.update(aggregate_subject_rows(sample, threshold=10.0))
            boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    summary: dict[str, Any] = {
        "task_id": TASK_ID,
        "dataset": dataset,
        "unit": "subject_equal_weight",
        "n_subjects": int(n_subjects),
        "n_samples": int(subj["n_samples"].sum()),
    }
    for metric in [
        "baseline_mae_bpm",
        "method_mae_bpm",
        "mae_improvement_bpm",
        "mae_reduction_vs_baseline",
        "release_coverage",
        "released_mae_bpm",
        "unsafe_release_rate",
    ]:
        vals = pd.to_numeric(boot[metric], errors="coerce").to_numpy(float)
        lo, hi = ci(vals)
        summary[f"{metric}_point"] = float(point[metric])
        summary[f"{metric}_ci95_low"] = lo
        summary[f"{metric}_ci95_high"] = hi
    imp = pd.to_numeric(boot["mae_improvement_bpm"], errors="coerce").to_numpy(float)
    summary["prob_improvement_gt_0"] = float(np.mean(imp > 0.0))
    summary["improvement_ci_excludes_zero"] = bool(summary["mae_improvement_bpm_ci95_low"] > 0.0)
    summary["point_gate_pass"] = bool(
        summary["mae_reduction_vs_baseline_point"] >= 0.20
        and summary["release_coverage_point"] >= 0.40
        and np.isfinite(summary["unsafe_release_rate_point"])
        and summary["unsafe_release_rate_point"] <= 0.10
    )
    summary["statistical_gate_pass"] = bool(summary["point_gate_pass"] and summary["improvement_ci_excludes_zero"])
    return boot, pd.DataFrame([summary])


def threshold_table(subj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, g in subj.groupby("dataset", sort=False):
        for threshold in THRESHOLDS:
            released = g[g["n_released"] > 0].copy()
            rows.append(
                {
                    "task_id": TASK_ID,
                    "dataset": dataset,
                    "threshold_bpm": threshold,
                    "unit": "subject_equal_weight",
                    "n_subjects": int(g["subject_std"].nunique()),
                    "release_coverage": float(g["release_coverage"].mean()),
                    "released_mae_bpm": float(released["released_mae_bpm"].mean()) if len(released) else math.nan,
                    "unsafe_release_rate": float(released[f"unsafe_release_gt{int(threshold)}bpm"].mean())
                    if len(released)
                    else math.nan,
                    "all_window_unsafe_rate": float(g[f"all_unsafe_gt{int(threshold)}bpm"].mean()),
                }
            )
    return pd.DataFrame(rows)


def claim_gate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        n_subjects = int(r["n_subjects"])
        statistical = bool(r["statistical_gate_pass"])
        point = bool(r["point_gate_pass"])
        if statistical and n_subjects >= 20:
            use = "main_text_primary"
            reason = "subject-cluster bootstrap supported and subject count is sufficient for a primary dataset."
        elif point and n_subjects >= 5:
            use = "appendix_or_boundary"
            reason = "point gate passes, but subject count or CI strength is not enough for a strong main claim."
        else:
            use = "boundary_only"
            reason = "subject-level statistical gate is not passed."
        rows.append(
            {
                "task_id": TASK_ID,
                "dataset": r["dataset"],
                "n_subjects": n_subjects,
                "point_gate_pass": point,
                "improvement_ci_excludes_zero": bool(r["improvement_ci_excludes_zero"]),
                "statistical_gate_pass": statistical,
                "manuscript_use": use,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in show.columns) + " |")
    return "\n".join(lines)


def write_report(generated_at: str, summary: pd.DataFrame, thresh: pd.DataFrame, claim: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# T901 Subject-Cluster Bootstrap and Threshold Sensitivity",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Purpose",
        "",
        "T901 repairs the pseudo-replication risk in the previous four-dataset table. T764 resampled rows for MCD-rPPG, UBFC-Phys-S1-S14, and UBFC-rPPG, whereas T901 aggregates each dataset by subject and then bootstraps subjects with equal subject weight.",
        "",
        "## Input Audit",
        "",
        f"- Missing baseline rows: `{audit['missing_baseline_rows']}`.",
        f"- Datasets: `{', '.join(audit['datasets'])}`.",
        f"- Bootstrap seeds: `{', '.join(map(str, SEEDS))}`; draws per seed: `{BOOT}`.",
        "",
        "## Subject-Level Summary",
        "",
        md_table(
            summary,
            [
                "dataset",
                "unit",
                "n_subjects",
                "n_samples",
                "baseline_mae_bpm_point",
                "method_mae_bpm_point",
                "mae_improvement_bpm_ci95_low",
                "mae_improvement_bpm_ci95_high",
                "release_coverage_point",
                "released_mae_bpm_point",
                "unsafe_release_rate_point",
                "statistical_gate_pass",
            ],
        ),
        "",
        "## Threshold Sensitivity",
        "",
        md_table(
            thresh,
            [
                "dataset",
                "threshold_bpm",
                "release_coverage",
                "released_mae_bpm",
                "unsafe_release_rate",
                "all_window_unsafe_rate",
            ],
        ),
        "",
        "## Manuscript Use",
        "",
        md_table(claim, ["dataset", "n_subjects", "statistical_gate_pass", "manuscript_use", "reason"]),
        "",
        "## Interpretation",
        "",
        "Use T901 as the manuscript-facing statistical table. Datasets with small subject counts can still support mechanism and boundary statements, but they should not be phrased as definitive population-level validation. T764 remains useful as a row-level stability check, but it should be cited as supplementary because it does not remove within-subject dependence.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    generated_at = now()
    decisions = load_method_decisions()
    inputs = attach_baselines(decisions)
    missing = inputs["baseline_abs_error_bpm"].isna()
    if bool(missing.any()):
        bad = inputs.loc[missing, ["dataset", "sample_id"]].head(20).to_dict("records")
        raise RuntimeError(f"Missing baseline errors for {int(missing.sum())} rows: {bad}")

    subj = subject_metrics(inputs)
    boot_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for dataset, g in subj.groupby("dataset", sort=False):
        boot, summ = bootstrap_dataset(str(dataset), g.copy())
        boot_parts.append(boot)
        summary_parts.append(summ)
    boot_all = pd.concat(boot_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    thresh = threshold_table(subj)
    claim = claim_gate(summary)
    audit = {
        "task_id": TASK_ID,
        "generated_at": generated_at,
        "datasets": sorted(inputs["dataset"].astype(str).unique().tolist()),
        "input_rows": int(len(inputs)),
        "subject_rows": int(len(subj)),
        "missing_baseline_rows": int(missing.sum()),
        "seeds": SEEDS,
        "bootstrap_per_seed": BOOT,
        "outputs": {
            "input_audit": str(OUT_INPUT.relative_to(ROOT)),
            "subject_metrics": str(OUT_SUBJECT.relative_to(ROOT)),
            "bootstrap": str(OUT_BOOT.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
            "threshold_sensitivity": str(OUT_THRESH.relative_to(ROOT)),
            "claim_gate": str(OUT_CLAIM.relative_to(ROOT)),
            "doc": str(OUT_MD.relative_to(ROOT)),
        },
    }

    inputs.to_csv(OUT_INPUT, index=False, encoding="utf-8-sig")
    subj.to_csv(OUT_SUBJECT, index=False, encoding="utf-8-sig")
    boot_all.to_csv(OUT_BOOT, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    thresh.to_csv(OUT_THRESH, index=False, encoding="utf-8-sig")
    claim.to_csv(OUT_CLAIM, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(generated_at, summary, thresh, claim, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
