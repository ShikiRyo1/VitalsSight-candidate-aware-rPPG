from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "T732"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

IN_CANDIDATES = EXP / "t731_candidate_ranker_candidate_predictions.csv"
IN_SELECTIONS = EXP / "t731_candidate_ranker_selections.csv"
IN_BASELINES = EXP / "t730_harmonic_aware_baseline_metrics.csv"
IN_POOL = EXP / "t730_harmonic_aware_candidate_pool.csv"

OUT_ABLATION_SELECTIONS = EXP / "t732_t731_ablation_selections.csv"
OUT_ABLATION_METRICS = EXP / "t732_t731_ablation_metrics.csv"
OUT_RELEASE = EXP / "t732_t731_ablation_release_gate.csv"
OUT_BOOTSTRAP = EXP / "t732_t731_bootstrap_ci.csv"
OUT_CLAIM = EXP / "t732_t731_ablation_claim_gate.csv"
OUT_SUMMARY = EXP / "t732_t731_bootstrap_ablation_summary.json"
OUT_MD = DOCS / "t732_t731_bootstrap_ablation.md"

UNSAFE_BPM = 10.0
SEED = 732
BOOT = 3000


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    return "\n".join(lines)


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    snr = numeric(out, "snr_rank_pct")
    support = numeric(out, "support_rank_pct")
    agree10 = numeric(out, "agreement10_frac")
    agree20 = numeric(out, "agreement20_frac")
    adult = numeric(out, "adult_plausibility")
    trap = numeric(out, "harmonic_trap_score")
    deep_disagree = numeric(out, "deep_disagreement_risk")
    alias = numeric(out, "alias_band_risk")
    out["rule_full"] = 0.24 * snr + 0.20 * support + 0.24 * agree10 + 0.12 * agree20 + 0.08 * adult - 0.12 * trap - 0.06 * deep_disagree
    out["rule_no_harmonic_trap"] = 0.24 * snr + 0.20 * support + 0.24 * agree10 + 0.12 * agree20 + 0.08 * adult - 0.06 * deep_disagree
    out["rule_no_agreement"] = 0.34 * snr + 0.30 * support + 0.10 * adult - 0.16 * trap - 0.06 * deep_disagree
    out["rule_snr_support_only"] = 0.56 * snr + 0.44 * support
    out["rule_trap_heavy"] = 0.20 * snr + 0.18 * support + 0.22 * agree10 + 0.12 * agree20 + 0.08 * adult - 0.24 * trap - 0.10 * alias - 0.06 * deep_disagree
    return out


def select_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, sample_id), g in df.groupby(["dataset", "sample_id"], sort=False):
        row = g.sort_values([score_col, "mean_snr_proxy_db"], ascending=[False, False]).iloc[0].copy()
        row["variant"] = score_col
        row["selected_score"] = float(row[score_col])
        row["pred_hr_bpm"] = float(row["candidate_hr_bpm"])
        row["abs_error_bpm"] = abs(float(row["candidate_hr_bpm"]) - float(row["gt_hr_bpm"]))
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


def build_max_snr_preds(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, sample_id), g in pool.groupby(["dataset", "sample_id"], sort=False):
        row = g.sort_values("mean_snr_proxy_db", ascending=False).iloc[0].copy()
        row["variant"] = "max_snr"
        row["selected_score"] = float(row.get("mean_snr_proxy_db", 0.0))
        row["pred_hr_bpm"] = float(row["candidate_hr_bpm"])
        row["abs_error_bpm"] = abs(float(row["candidate_hr_bpm"]) - float(row["gt_hr_bpm"]))
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


def metric_rows(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, variant), g in selections.groupby(["dataset", "variant"], sort=False):
        err = numeric(g, "abs_error_bpm")
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_windows": int(len(g)),
                "mae_bpm": float(err.mean()),
                "rmse_bpm": float(np.sqrt(np.mean(np.square(err)))),
                "median_abs_error_bpm": float(err.median()),
                "p90_abs_error_bpm": float(err.quantile(0.90)),
                "unsafe_gt10bpm_rate": float((err > UNSAFE_BPM).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mae_bpm"])


def release_gate(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, variant), g in selections.groupby(["dataset", "variant"], sort=False):
        h = g.copy()
        h["release_risk"] = (
            1.0
            - numeric(h, "selected_score")
            + 0.65 * numeric(h, "harmonic_trap_score")
            + 0.25 * numeric(h, "alias_band_risk")
            + 0.18 * numeric(h, "deep_disagreement_risk")
            - 0.30 * numeric(h, "agreement10_frac")
        )
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["release_risk"].quantile(q))
            rel = h[h["release_risk"] <= tau]
            if rel.empty:
                continue
            err = numeric(rel, "abs_error_bpm")
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "threshold_quantile": float(q),
                    "risk_threshold": tau,
                    "coverage": float(len(rel) / len(h)),
                    "released_mae_bpm": float(err.mean()),
                    "unsafe_release_rate": float((err > UNSAFE_BPM).mean()),
                    "n_released": int(len(rel)),
                    "n_total": int(len(h)),
                    "gate_pass_unsafe10": bool((err > UNSAFE_BPM).mean() <= 0.10),
                }
            )
    return pd.DataFrame(rows)


def claim_gate(metrics: pd.DataFrame, release: pd.DataFrame, baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        base = baseline_metrics[(baseline_metrics["dataset"].astype(str) == dataset) & (baseline_metrics["method"].astype(str) == "max_snr")]
        base_mae = float(base["mae_bpm"].iloc[0]) if not base.empty else math.nan
        safe = release[(release["dataset"].astype(str) == dataset) & (release["variant"].astype(str) == variant) & (release["gate_pass_unsafe10"])]
        best = safe.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        best_cov = float(best["coverage"].iloc[0]) if not best.empty else 0.0
        best_unsafe = float(best["unsafe_release_rate"].iloc[0]) if not best.empty else math.nan
        best_mae = float(best["released_mae_bpm"].iloc[0]) if not best.empty else math.nan
        reduction = 1.0 - float(row["mae_bpm"]) / base_mae if math.isfinite(base_mae) and base_mae > 0 else math.nan
        released_reduction = 1.0 - best_mae / base_mae if math.isfinite(base_mae) and math.isfinite(best_mae) and base_mae > 0 else math.nan
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "mae_bpm": float(row["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(row["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_max_snr": reduction,
                "best_safe_gate_coverage": best_cov,
                "best_safe_gate_unsafe": best_unsafe,
                "best_safe_gate_released_mae_bpm": best_mae,
                "released_mae_reduction_vs_max_snr": released_reduction,
                "pass_dataset_gate": bool(math.isfinite(released_reduction) and released_reduction >= 0.20 and best_cov >= 0.40 and best_unsafe <= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mae_bpm"])


def bootstrap_ci(selections: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ctrl = control[["dataset", "sample_id", "abs_error_bpm"]].rename(columns={"abs_error_bpm": "control_abs_error_bpm"})
    rows = []
    for (dataset, variant), g in selections.groupby(["dataset", "variant"], sort=False):
        merged = g[["dataset", "sample_id", "abs_error_bpm"]].merge(ctrl, on=["dataset", "sample_id"], how="inner")
        if merged.empty:
            continue
        diff = (pd.to_numeric(merged["abs_error_bpm"], errors="coerce") - pd.to_numeric(merged["control_abs_error_bpm"], errors="coerce")).to_numpy(float)
        diff = diff[np.isfinite(diff)]
        if len(diff) == 0:
            continue
        draws = []
        n = len(diff)
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            draws.append(float(np.mean(diff[idx])))
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "control": "max_snr",
                "n_pairs": int(n),
                "mean_delta_mae_bpm": float(np.mean(diff)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "prob_delta_lt_0": float(np.mean(np.asarray(draws) < 0.0)),
                "improvement_supported": bool(np.quantile(draws, 0.975) < 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mean_delta_mae_bpm"])


def write_report(summary: dict[str, Any], metrics: pd.DataFrame, claim: pd.DataFrame, boot: pd.DataFrame) -> None:
    lines = [
        "# T732 Bootstrap and Ablation for T731",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T731 found that the interpretable rule selector, not the tree ranker, is the current passing route on MCD. T732 tests whether this route is statistically supported against max-SNR and whether agreement/harmonic-trap terms are necessary.",
        "",
        "## Ablation Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        markdown_table(claim),
        "",
        "## Bootstrap vs Max-SNR",
        "",
        markdown_table(boot),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "This is still public-dataset selector evidence, not a universal clinical claim. The paper-safe wording is risk-controlled candidate selection improves release safety on MCD and transfers to UBFC-rPPG under the tested candidate-table protocol.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not IN_CANDIDATES.exists():
        raise FileNotFoundError(IN_CANDIDATES)
    if not IN_POOL.exists():
        raise FileNotFoundError(IN_POOL)
    candidates = pd.read_csv(IN_CANDIDATES, low_memory=False)
    pool = pd.read_csv(IN_POOL, low_memory=False)
    baselines = pd.read_csv(IN_BASELINES) if IN_BASELINES.exists() else pd.DataFrame()
    candidates = add_rule_scores(candidates)
    pool = add_rule_scores(pool)

    variants = ["rule_full", "rule_no_harmonic_trap", "rule_no_agreement", "rule_snr_support_only", "rule_trap_heavy"]
    selected_parts = [select_by_score(candidates, v) for v in variants]
    selections = pd.concat(selected_parts, ignore_index=True, sort=False)
    max_snr = build_max_snr_preds(pool)
    metrics = metric_rows(selections)
    release = release_gate(selections)
    claim = claim_gate(metrics, release, baselines)
    boot = bootstrap_ci(selections, max_snr)

    selections.to_csv(OUT_ABLATION_SELECTIONS, index=False, encoding="utf-8")
    metrics.to_csv(OUT_ABLATION_METRICS, index=False, encoding="utf-8")
    release.to_csv(OUT_RELEASE, index=False, encoding="utf-8")
    claim.to_csv(OUT_CLAIM, index=False, encoding="utf-8")
    boot.to_csv(OUT_BOOTSTRAP, index=False, encoding="utf-8")

    pass_count = int(claim["pass_dataset_gate"].sum()) if "pass_dataset_gate" in claim.columns else 0
    supported = boot[(boot["dataset"].astype(str) == "MCD-rPPG") & (boot["variant"].astype(str) == "rule_full")]["improvement_supported"]
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "decision": "bootstrap_supported_prepare_external_locked_bundle" if (not supported.empty and bool(supported.iloc[0]) and pass_count >= 2) else "bootstrap_partial_requires_more_validation",
        "pass_count": pass_count,
        "mcd_rule_full_bootstrap_supported": bool(supported.iloc[0]) if not supported.empty else False,
        "outputs": {
            "selections": str(OUT_ABLATION_SELECTIONS),
            "metrics": str(OUT_ABLATION_METRICS),
            "release": str(OUT_RELEASE),
            "claim": str(OUT_CLAIM),
            "bootstrap": str(OUT_BOOTSTRAP),
            "doc": str(OUT_MD),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, metrics, claim, boot)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
