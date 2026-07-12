from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ID = "T755"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

IN_SELECTIONS = EXP / "t732_t731_ablation_selections.csv"
IN_POOL = EXP / "t730_harmonic_aware_candidate_pool.csv"
IN_CLAIM = EXP / "t732_t731_ablation_claim_gate.csv"
OUT_BOOT = EXP / "t755_rule_policy_multiseed_bootstrap_ci.csv"
OUT_AGG = EXP / "t755_rule_policy_multiseed_bootstrap_aggregate.csv"
OUT_SUMMARY = EXP / "t755_rule_policy_multiseed_bootstrap_summary.json"
OUT_MD = DOCS / "t755_rule_policy_multiseed_bootstrap_cn.md"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    lines = [
        "| " + " | ".join(str(c).replace("|", "\\|") for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in show.columns) + " |")
    return "\n".join(lines)


def build_max_snr(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, sample_id), g in pool.groupby(["dataset", "sample_id"], sort=False):
        row = g.sort_values("mean_snr_proxy_db", ascending=False).iloc[0]
        pred = float(row["candidate_hr_bpm"])
        gt = float(row["gt_hr_bpm"])
        rows.append({"dataset": dataset, "sample_id": sample_id, "control_abs_error_bpm": abs(pred - gt)})
    return pd.DataFrame(rows)


def bootstrap_once(selections: pd.DataFrame, control: pd.DataFrame, *, seed: int, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ctrl = control[["dataset", "sample_id", "control_abs_error_bpm"]]
    for (dataset, variant), g in selections.groupby(["dataset", "variant"], sort=False):
        merged = g[["dataset", "sample_id", "abs_error_bpm"]].merge(ctrl, on=["dataset", "sample_id"], how="inner")
        if merged.empty:
            continue
        diff = pd.to_numeric(merged["abs_error_bpm"], errors="coerce").to_numpy(float) - pd.to_numeric(merged["control_abs_error_bpm"], errors="coerce").to_numpy(float)
        diff = diff[np.isfinite(diff)]
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(diff), len(diff))
            draws.append(float(np.mean(diff[idx])))
        rows.append(
            {
                "bootstrap_seed": seed,
                "dataset": dataset,
                "variant": variant,
                "control": "max_snr",
                "n_pairs": int(len(diff)),
                "mean_delta_mae_bpm": float(np.mean(diff)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "prob_delta_lt_0": float(np.mean(np.asarray(draws) < 0.0)),
                "improvement_supported": bool(np.quantile(draws, 0.975) < 0.0),
            }
        )
    return pd.DataFrame(rows)


def aggregate(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, variant), g in boot.groupby(["dataset", "variant"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_bootstrap_seeds": int(g["bootstrap_seed"].nunique()),
                "all_bootstrap_seeds_supported": bool(g["improvement_supported"].astype(bool).all()),
                "mean_delta_mae_bpm": float(g["mean_delta_mae_bpm"].mean()),
                "ci95_low_worst": float(g["ci95_low"].max()),
                "ci95_high_worst": float(g["ci95_high"].max()),
                "prob_delta_lt_0_min": float(g["prob_delta_lt_0"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mean_delta_mae_bpm"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("704,1704,2704"))
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    if not IN_SELECTIONS.exists():
        raise FileNotFoundError(IN_SELECTIONS)
    if not IN_POOL.exists():
        raise FileNotFoundError(IN_POOL)
    selections = pd.read_csv(IN_SELECTIONS, low_memory=False)
    pool = pd.read_csv(IN_POOL, low_memory=False)
    claim = pd.read_csv(IN_CLAIM) if IN_CLAIM.exists() else pd.DataFrame()
    control = build_max_snr(pool)
    boot = pd.concat([bootstrap_once(selections, control, seed=seed, n_boot=args.n_boot) for seed in args.seeds], ignore_index=True, sort=False)
    agg = aggregate(boot)
    boot.to_csv(OUT_BOOT, index=False, encoding="utf-8")
    agg.to_csv(OUT_AGG, index=False, encoding="utf-8")
    supported = agg[agg["all_bootstrap_seeds_supported"]]
    gate_supported = claim[claim["pass_dataset_gate"].astype(bool)] if not claim.empty else pd.DataFrame()
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "bootstrap_seeds": args.seeds,
        "n_boot": args.n_boot,
        "all_seed_supported_rows": int(len(supported)),
        "gate_supported_rows": int(len(gate_supported)),
        "decision": "rule_policy_bootstrap_stable" if len(supported) >= 4 else "rule_policy_bootstrap_partial",
        "outputs": {"bootstrap": str(OUT_BOOT), "aggregate": str(OUT_AGG), "doc": str(OUT_MD)},
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    view_cols = ["dataset", "variant", "n_bootstrap_seeds", "all_bootstrap_seeds_supported", "mean_delta_mae_bpm", "ci95_low_worst", "ci95_high_worst", "prob_delta_lt_0_min"]
    lines = [
        "# T755 Rule Policy Three-Seed Bootstrap",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T753 showed that neuralizing the selector does not improve external release gates. T755 therefore strengthens the current best supported deterministic policy from T731/T732 by repeating paired bootstrap CI with seeds 704/1704/2704.",
        "",
        "## Aggregate Bootstrap Stability",
        "",
        markdown_table(agg[view_cols]),
        "",
        "## T732 Claim Gate Reference",
        "",
        markdown_table(claim),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Interpretation",
        "",
        "Rows with `all_bootstrap_seeds_supported=True` have a 95% bootstrap CI whose upper bound remains below zero for every bootstrap seed, meaning the MAE reduction versus max-SNR is stable to resampling randomness. This strengthens the statistical evidence for deterministic candidate/risk-control policy variants without pretending UBFC-Phys is solved.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
