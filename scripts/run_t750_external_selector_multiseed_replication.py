from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "T750"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t705_external_selector_validation import (  # noqa: E402
    MCD_CANDIDATES,
    T704_POOL,
    UBFC_PHYS_CANDIDATES,
    apply_models,
    baseline_predictions,
    build_unified_candidate_pool,
    metric_row,
    normalize_external_table,
    set_seed,
    train_final_models,
)


EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
UNSAFE_BPM = 10.0

OUT_POOL = EXP / "t750_external_multiseed_candidate_pool.csv"
OUT_PREDS = EXP / "t750_external_multiseed_selector_predictions.csv"
OUT_METRICS = EXP / "t750_external_multiseed_selector_metrics.csv"
OUT_BASELINES = EXP / "t750_external_multiseed_baseline_metrics.csv"
OUT_RELEASE = EXP / "t750_external_multiseed_release_gate.csv"
OUT_CLAIM = EXP / "t750_external_multiseed_claim_gate.csv"
OUT_SUMMARY = EXP / "t750_external_selector_multiseed_replication_summary.json"
OUT_MD = DOCS / "t750_external_selector_multiseed_replication_cn.md"


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
        "| " + " | ".join(str(c).replace("|", "\\|") for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in show.columns) + " |")
    return "\n".join(lines)


def release_gate_by_seed(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, dataset, selector), g in preds.groupby(["seed", "dataset", "selector"], sort=False):
        h = g.copy()
        denom = np.log(pd.to_numeric(h["n_candidates"], errors="coerce").clip(lower=2))
        risk = (
            (1.0 - pd.to_numeric(h["max_probability"], errors="coerce").fillna(0.0))
            + pd.to_numeric(h["entropy"], errors="coerce").fillna(0.0) / denom
            + 0.15 * pd.to_numeric(h["selected_harmonic_risk"], errors="coerce").fillna(0.0)
            - 0.20 * pd.to_numeric(h["selected_agreement10_frac"], errors="coerce").fillna(0.0)
        )
        h["risk_score"] = risk.replace([np.inf, -np.inf], np.nan).fillna(999.0)
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["risk_score"].quantile(q))
            released = h[h["risk_score"] <= tau]
            if released.empty:
                continue
            err = pd.to_numeric(released["abs_error_bpm"], errors="coerce")
            rows.append(
                {
                    "seed": int(seed),
                    "dataset": dataset,
                    "selector": selector,
                    "threshold_quantile": float(q),
                    "risk_threshold": tau,
                    "coverage": float(len(released) / len(h)),
                    "released_mae_bpm": float(err.mean()),
                    "unsafe_release_rate": float((err > UNSAFE_BPM).mean()),
                    "n_released": int(len(released)),
                    "n_total": int(len(h)),
                    "gate_pass_unsafe10": bool((err > UNSAFE_BPM).mean() <= 0.10),
                }
            )
    return pd.DataFrame(rows)


def per_seed_claim(metrics: pd.DataFrame, baseline_metrics: pd.DataFrame, release: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        seed = int(row["seed"])
        dataset = str(row["dataset"])
        selector = str(row["method"])
        base = baseline_metrics[
            (baseline_metrics["dataset"].astype(str) == dataset)
            & (baseline_metrics["method"].astype(str) == "external_max_snr")
        ]
        base_mae = float(base["mae_bpm"].iloc[0]) if not base.empty else math.nan
        reduction = 1.0 - float(row["mae_bpm"]) / base_mae if math.isfinite(base_mae) and base_mae > 0 else math.nan
        safe = release[
            (release["seed"] == seed)
            & (release["dataset"].astype(str) == dataset)
            & (release["selector"].astype(str) == selector)
            & (release["gate_pass_unsafe10"])
        ]
        best_safe = safe.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        coverage = float(best_safe["coverage"].iloc[0]) if not best_safe.empty else 0.0
        rows.append(
            {
                "seed": seed,
                "dataset": dataset,
                "selector": selector,
                "mae_bpm": float(row["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(row["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_external_max_snr": reduction,
                "best_safe_gate_coverage": coverage,
                "best_safe_gate_unsafe": float(best_safe["unsafe_release_rate"].iloc[0]) if not best_safe.empty else math.nan,
                "seed_external_gate_pass": bool(reduction >= 0.20 and coverage >= 0.40),
            }
        )
    return pd.DataFrame(rows)


def aggregate_claim(seed_claim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, selector), g in seed_claim.groupby(["dataset", "selector"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "selector": selector,
                "n_seeds": int(g["seed"].nunique()),
                "mae_mean": float(g["mae_bpm"].mean()),
                "mae_std": float(g["mae_bpm"].std(ddof=1)) if len(g) > 1 else 0.0,
                "unsafe_gt10bpm_rate_mean": float(g["unsafe_gt10bpm_rate"].mean()),
                "mae_reduction_vs_external_max_snr_mean": float(g["mae_reduction_vs_external_max_snr"].mean()),
                "best_safe_gate_coverage_mean": float(g["best_safe_gate_coverage"].mean()),
                "best_safe_gate_coverage_min": float(g["best_safe_gate_coverage"].min()),
                "seed_pass_count": int(g["seed_external_gate_pass"].sum()),
                "external_multiseed_gate_pass": bool(
                    int(g["seed"].nunique()) >= 3
                    and int(g["seed_external_gate_pass"].sum()) >= 2
                    and float(g["mae_reduction_vs_external_max_snr"].mean()) >= 0.20
                    and float(g["best_safe_gate_coverage"].min()) >= 0.40
                ),
                "strict_all_seed_pass": bool(g["seed_external_gate_pass"].all()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mae_mean"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="704,1704,2704")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if T704_POOL.exists():
        train_pool = pd.read_csv(T704_POOL)
    else:
        train_pool, _ = build_unified_candidate_pool()

    external_parts = [
        normalize_external_table(MCD_CANDIDATES, "MCD-rPPG"),
        normalize_external_table(UBFC_PHYS_CANDIDATES, "UBFC-Phys-S1-S14"),
    ]
    external_pool = pd.concat(external_parts, ignore_index=True, sort=False)
    external_pool.to_csv(OUT_POOL, index=False, encoding="utf-8")

    _, baseline_metrics = baseline_predictions(external_pool)
    baseline_metrics.to_csv(OUT_BASELINES, index=False, encoding="utf-8")

    all_preds: list[pd.DataFrame] = []
    for seed in seeds:
        set_seed(seed)
        print(f"[{TASK_ID}] seed={seed} training external replication selectors", flush=True)
        models, feature_cols, std = train_final_models(train_pool, epochs=args.epochs, lr=args.lr)
        pred = apply_models(models, feature_cols, std, external_pool)
        pred["seed"] = seed
        all_preds.append(pred)

    preds = pd.concat(all_preds, ignore_index=True, sort=False)
    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8")

    metrics = pd.DataFrame(
        [
            {"seed": int(seed), "dataset": ds, **metric_row(selector, g.rename(columns={"selector": "method"}))}
            for (seed, ds, selector), g in preds.groupby(["seed", "dataset", "selector"], sort=False)
        ]
    ).sort_values(["dataset", "method", "seed"])
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8")

    release = release_gate_by_seed(preds)
    release.to_csv(OUT_RELEASE, index=False, encoding="utf-8")

    seed_claim = per_seed_claim(metrics, baseline_metrics, release)
    agg = aggregate_claim(seed_claim)
    seed_claim["row_type"] = "per_seed"
    agg["row_type"] = "aggregate"
    claim = pd.concat([seed_claim, agg], ignore_index=True, sort=False)
    claim.to_csv(OUT_CLAIM, index=False, encoding="utf-8")

    pass_rows = agg[agg["external_multiseed_gate_pass"] == True]  # noqa: E712
    datasets_with_pass = sorted(pass_rows["dataset"].astype(str).unique().tolist())
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "datasets": sorted(external_pool["dataset"].astype(str).unique().tolist()),
        "n_external_windows": int(external_pool["sample_id"].nunique()),
        "n_candidate_rows": int(len(external_pool)),
        "datasets_with_pass": datasets_with_pass,
        "n_aggregate_pass_rows": int(len(pass_rows)),
        "decision": (
            "external_multiseed_replication_supports_main_claim"
            if len(datasets_with_pass) >= 2
            else "external_multiseed_replication_partial_requires_calibration_or_boundary_wording"
        ),
        "outputs": {
            "pool": str(OUT_POOL),
            "predictions": str(OUT_PREDS),
            "metrics": str(OUT_METRICS),
            "baselines": str(OUT_BASELINES),
            "release": str(OUT_RELEASE),
            "claim": str(OUT_CLAIM),
            "doc": str(OUT_MD),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# T750 外部数据集三 seed 复现实验",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 目的",
        "",
        "用与 T742 一致的三组随机种子，在两个外部数据集 MCD-rPPG 与 UBFC-Phys-S1-S14 上复现 deep-candidate PhysioGate selector 的跨数据集效果。",
        "",
        "## Baseline",
        "",
        markdown_table(baseline_metrics),
        "",
        "## 三 seed 指标",
        "",
        markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        markdown_table(agg),
        "",
        "## 结论",
        "",
        summary["decision"],
        "",
        "## 证据边界",
        "",
        "该实验验证外部候选表上的 selector transfer。若 gate 未通过，下一步应做 domain calibration 或候选池质量诊断，而不是直接扩大论文 claim。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
