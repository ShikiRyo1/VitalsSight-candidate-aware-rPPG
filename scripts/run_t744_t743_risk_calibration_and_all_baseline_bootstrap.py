from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "T744"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

PRED = EXP / "t743_extended_selector_predictions.csv"
BASE = EXP / "t743_extended_deep_baseline_predictions.csv"

OUT_BOOT = EXP / "t744_all_baseline_bootstrap_ci.csv"
OUT_CAL_PREDS = EXP / "t744_oof_risk_calibrated_predictions.csv"
OUT_RISK = EXP / "t744_risk_calibration_metrics.csv"
OUT_SWEEP = EXP / "t744_risk_coverage_sweep.csv"
OUT_GATE = EXP / "t744_claim_gate.csv"
OUT_SUMMARY = EXP / "t744_risk_calibration_summary.json"
OUT_MD = DOCS / "t744_risk_calibration_and_all_baseline_bootstrap_results.md"

UNSAFE_BPM = 10.0


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    vals, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, c in enumerate(counts):
            if c > 1:
                idx = np.where(inv == k)[0]
                ranks[idx] = ranks[idx].mean()
    rank_pos = ranks[y == 1].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def make_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["low_confidence"] = 1.0 - pd.to_numeric(df["max_probability"], errors="coerce").fillna(0.0)
    out["entropy"] = pd.to_numeric(df["entropy"], errors="coerce").fillna(0.0)
    out["harmonic_risk"] = pd.to_numeric(df["selected_harmonic_risk"], errors="coerce").fillna(0.0)
    out["disagreement10"] = 1.0 - pd.to_numeric(df["selected_agreement10_frac"], errors="coerce").fillna(0.0)
    out["candidate_count"] = pd.to_numeric(df["n_candidates"], errors="coerce").fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_logistic_numpy(x: np.ndarray, y: np.ndarray, *, steps: int = 800, lr: float = 0.05, l2: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    z = (x - mu) / sd
    z = np.c_[np.ones(len(z)), z]
    w = np.zeros(z.shape[1], dtype=float)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(steps):
        p = sigmoid(z @ w)
        grad = (z.T @ ((p - y) * weights)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w, mu, sd


def predict_logistic_numpy(x: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (x - mu) / sd
    z = np.c_[np.ones(len(z)), z]
    return sigmoid(z @ w)


def oof_calibrate(pred: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    for (variant, seed), g in pred.groupby(["variant_id", "seed"], sort=False):
        g = g.copy()
        x_all = make_risk_features(g)
        y_all = (pd.to_numeric(g["abs_error_bpm"], errors="coerce") > UNSAFE_BPM).astype(int).to_numpy()
        g["risk_probability_only"] = 1.0 - pd.to_numeric(g["max_probability"], errors="coerce").fillna(0.0)
        g["risk_probability_entropy"] = g["risk_probability_only"] + pd.to_numeric(g["entropy"], errors="coerce").fillna(0.0) / np.log(pd.to_numeric(g["n_candidates"], errors="coerce").fillna(2.0).clip(lower=2.0))
        g["risk_harmonic_only"] = pd.to_numeric(g["selected_harmonic_risk"], errors="coerce").fillna(0.0)
        g["risk_calibrated_oof"] = np.nan
        for fold in sorted(g["fold_idx"].dropna().unique()):
            test_mask = g["fold_idx"].eq(fold).to_numpy()
            train_mask = ~test_mask
            if train_mask.sum() < 10 or y_all[train_mask].sum() == 0 or y_all[train_mask].sum() == train_mask.sum():
                g.loc[test_mask, "risk_calibrated_oof"] = g.loc[test_mask, "risk_probability_entropy"]
                continue
            w, mu, sd = fit_logistic_numpy(x_all.loc[train_mask].to_numpy(float), y_all[train_mask].astype(float))
            g.loc[test_mask, "risk_calibrated_oof"] = predict_logistic_numpy(x_all.loc[test_mask].to_numpy(float), w, mu, sd)
        all_rows.append(g)
    return pd.concat(all_rows, ignore_index=True, sort=False)


def risk_metrics(cal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    risk_cols = ["risk_probability_only", "risk_probability_entropy", "risk_harmonic_only", "risk_calibrated_oof"]
    for (variant, seed), g in cal.groupby(["variant_id", "seed"], sort=False):
        y = (pd.to_numeric(g["abs_error_bpm"], errors="coerce") > UNSAFE_BPM).astype(int).to_numpy()
        for col in risk_cols:
            s = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            rows.append(
                {
                    "variant_id": variant,
                    "seed": int(seed),
                    "risk_mode": col.replace("risk_", ""),
                    "unsafe_rate": float(y.mean()),
                    "risk_auroc_for_unsafe_gt10": roc_auc(y, s),
                    "mean_risk_safe": float(np.nanmean(s[y == 0])),
                    "mean_risk_unsafe": float(np.nanmean(s[y == 1])) if y.sum() else math.nan,
                }
            )
    return pd.DataFrame(rows)


def coverage_sweep(cal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    risk_cols = ["risk_probability_only", "risk_probability_entropy", "risk_harmonic_only", "risk_calibrated_oof"]
    quantiles = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    for (variant, seed), g in cal.groupby(["variant_id", "seed"], sort=False):
        err = pd.to_numeric(g["abs_error_bpm"], errors="coerce")
        for col in risk_cols:
            risk = pd.to_numeric(g[col], errors="coerce")
            for q in quantiles:
                thr = float(risk.quantile(q))
                keep = risk <= thr
                if keep.sum() == 0:
                    continue
                rows.append(
                    {
                        "variant_id": variant,
                        "seed": int(seed),
                        "risk_mode": col.replace("risk_", ""),
                        "threshold_quantile": q,
                        "risk_threshold": thr,
                        "coverage": float(keep.mean()),
                        "released_mae_bpm": float(err[keep].mean()),
                        "released_rmse_bpm": float(np.sqrt(np.mean(np.square(err[keep])))),
                        "unsafe_release_rate": float((err[keep] > UNSAFE_BPM).mean()),
                        "n_released": int(keep.sum()),
                        "n_total": int(len(g)),
                        "gate_pass_unsafe10": bool((err[keep] > UNSAFE_BPM).mean() <= 0.10),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_all_baselines(pred: pd.DataFrame, base: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    base_by = {name: g.set_index("sample_id") for name, g in base.groupby("method")}
    for (variant, selector_seed), g in pred.groupby(["variant_id", "seed"], sort=False):
        g = g.set_index("sample_id")
        for baseline, b in base_by.items():
            common = g.index.intersection(b.index)
            if len(common) == 0:
                continue
            sel_err = pd.to_numeric(g.loc[common, "abs_error_bpm"], errors="coerce").to_numpy(float)
            base_err = pd.to_numeric(b.loc[common, "abs_error_bpm"], errors="coerce").to_numpy(float)
            delta = sel_err - base_err
            draws = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(delta), len(delta))
                draws.append(float(np.nanmean(delta[idx])))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append(
                {
                    "variant_id": variant,
                    "seed": int(selector_seed),
                    "baseline": baseline,
                    "n_pairs": int(len(delta)),
                    "mean_delta_mae_bpm": float(np.nanmean(delta)),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "improvement_supported": bool(hi < 0),
                    "p_delta_lt_0_boot": float(np.mean(np.array(draws) < 0)),
                }
            )
    return pd.DataFrame(rows)


def claim_gate(risk: pd.DataFrame, sweep: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, g in risk.groupby("variant_id", sort=False):
        cal = g[g["risk_mode"] == "calibrated_oof"]
        harm = g[g["risk_mode"] == "harmonic_only"]
        best_risk = g.groupby("risk_mode")["risk_auroc_for_unsafe_gt10"].mean().sort_values(ascending=False)
        sw = sweep[(sweep["variant_id"] == variant) & (sweep["risk_mode"] == "calibrated_oof") & (sweep["gate_pass_unsafe10"])]
        sw_best = sw.groupby("threshold_quantile").mean(numeric_only=True).reset_index().sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        b = boot[(boot["variant_id"] == variant) & (boot["baseline"].astype(str).str.startswith("deep_"))]
        all_deep_supported = bool(b.groupby("baseline")["improvement_supported"].all().all()) if not b.empty else False
        rows.append(
            {
                "variant_id": variant,
                "calibrated_auc_mean": float(cal["risk_auroc_for_unsafe_gt10"].mean()) if not cal.empty else math.nan,
                "harmonic_auc_mean": float(harm["risk_auroc_for_unsafe_gt10"].mean()) if not harm.empty else math.nan,
                "best_risk_mode": str(best_risk.index[0]) if not best_risk.empty else "",
                "best_risk_auc_mean": float(best_risk.iloc[0]) if not best_risk.empty else math.nan,
                "best_calibrated_coverage_unsafe10": float(sw_best["coverage"].iloc[0]) if not sw_best.empty else 0.0,
                "best_calibrated_released_mae": float(sw_best["released_mae_bpm"].iloc[0]) if not sw_best.empty else math.nan,
                "bootstrap_all_deep_supported": all_deep_supported,
                "pass_risk_calibration_gate": bool(
                    (not sw_best.empty and float(sw_best["coverage"].iloc[0]) >= 0.50)
                    and (not cal.empty and float(cal["risk_auroc_for_unsafe_gt10"].mean()) >= 0.60)
                    and all_deep_supported
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], gate: pd.DataFrame, risk: pd.DataFrame, sweep: pd.DataFrame, boot: pd.DataFrame) -> None:
    best_sweep = (
        sweep[sweep["gate_pass_unsafe10"]]
        .sort_values(["variant_id", "risk_mode", "coverage", "released_mae_bpm"], ascending=[True, True, False, True])
        .groupby(["variant_id", "risk_mode"])
        .head(1)
    )
    lines = [
        "# T744 Risk Calibration and All-Baseline Bootstrap",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Gate",
        "",
        markdown_table(gate),
        "",
        "## Risk AUROC",
        "",
        markdown_table(risk.groupby(["variant_id", "risk_mode"]).mean(numeric_only=True).reset_index().sort_values(["variant_id", "risk_auroc_for_unsafe_gt10"], ascending=[True, False])),
        "",
        "## Best Safe Coverage by Risk Mode",
        "",
        markdown_table(best_sweep[["variant_id", "seed", "risk_mode", "threshold_quantile", "coverage", "released_mae_bpm", "unsafe_release_rate"]].head(40)),
        "",
        "## Bootstrap vs All Baselines",
        "",
        markdown_table(boot.groupby(["variant_id", "baseline"]).mean(numeric_only=True).reset_index().sort_values(["variant_id", "mean_delta_mae_bpm"]).head(60)),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=744)
    args = parser.parse_args()

    if not PRED.exists():
        raise FileNotFoundError(PRED)
    if not BASE.exists():
        raise FileNotFoundError(BASE)

    pred = pd.read_csv(PRED)
    base = pd.read_csv(BASE)
    cal = oof_calibrate(pred)
    cal.to_csv(OUT_CAL_PREDS, index=False, encoding="utf-8")
    risk = risk_metrics(cal)
    risk.to_csv(OUT_RISK, index=False, encoding="utf-8")
    sweep = coverage_sweep(cal)
    sweep.to_csv(OUT_SWEEP, index=False, encoding="utf-8")
    boot = bootstrap_all_baselines(pred, base, args.bootstrap, args.seed)
    boot.to_csv(OUT_BOOT, index=False, encoding="utf-8")
    gate = claim_gate(risk, sweep, boot)
    gate.to_csv(OUT_GATE, index=False, encoding="utf-8")
    pass_gate = bool(gate["pass_risk_calibration_gate"].any()) if not gate.empty else False
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "n_predictions": int(len(pred)),
        "n_baseline_predictions": int(len(base)),
        "n_bootstrap": int(args.bootstrap),
        "pass_risk_calibration_gate": pass_gate,
        "decision": (
            "PASS: risk calibration and all-baseline bootstrap support a stronger release/review claim."
            if pass_gate
            else "NOT PASSED: selector superiority is supported, but risk calibration remains a boundary; do not overclaim release/review risk discrimination."
        ),
        "outputs": {
            "calibrated_predictions": str(OUT_CAL_PREDS),
            "risk_metrics": str(OUT_RISK),
            "risk_coverage_sweep": str(OUT_SWEEP),
            "all_baseline_bootstrap": str(OUT_BOOT),
            "claim_gate": str(OUT_GATE),
            "report": str(OUT_MD),
        },
    }
    write_json(OUT_SUMMARY, summary)
    write_report(summary, gate, risk, sweep, boot)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
