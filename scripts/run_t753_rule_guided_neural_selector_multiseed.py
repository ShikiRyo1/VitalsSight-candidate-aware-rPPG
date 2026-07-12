from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

TASK_ID = "T753"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t704_deep_candidate_physio_gate import (
    GraphSelector,
    GroupBatch,
    MLPSelector,
    SetAttentionSelector,
    make_groups,
)
from scripts.run_t731_candidate_ranker_harmonic_rescue import (
    CATEGORICAL,
    NUMERIC,
    add_relative_rule_features,
)


EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"

IN_POOL = EXP / "t730_harmonic_aware_candidate_pool.csv"
IN_BASELINES = EXP / "t730_harmonic_aware_baseline_metrics.csv"

OUT_CLAIM = EXP / "t753_rule_guided_neural_multiseed_claim_gate.csv"
OUT_AGG = EXP / "t753_rule_guided_neural_aggregate_claim_gate.csv"
OUT_METRICS = EXP / "t753_rule_guided_neural_multiseed_metrics.csv"
OUT_RELEASE = EXP / "t753_rule_guided_neural_multiseed_release_gate.csv"
OUT_PREDS = EXP / "t753_rule_guided_neural_multiseed_predictions.csv"
OUT_BOOT = EXP / "t753_rule_guided_neural_bootstrap_ci.csv"
OUT_SUMMARY = EXP / "t753_rule_guided_neural_selector_multiseed_summary.json"
OUT_MD = DOCS / "t753_rule_guided_neural_selector_multiseed_cn.md"

GOOD_BPM = 5.0
UNSAFE_BPM = 10.0


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


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


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def feature_frame(pool: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = add_relative_rule_features(pool.copy())
    base_numeric = list(dict.fromkeys(NUMERIC + ["observable_rule_score"]))
    for col in base_numeric:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "unknown"
    cat = pd.get_dummies(df[CATEGORICAL].astype(str), prefix=CATEGORICAL)
    features = pd.concat([df[base_numeric], cat], axis=1).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    if feature_cols is None:
        feature_cols = list(features.columns)
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0
    return features[feature_cols].astype(float), feature_cols


def stratified_domain_folds(pool: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[set[str], set[str]]]:
    rng = np.random.default_rng(seed)
    keys = pool[["dataset", "subject_std"]].copy()
    keys["dataset"] = keys["dataset"].astype(str)
    keys["subject_std"] = keys["subject_std"].astype(str)
    keys["group_key"] = keys["dataset"] + "::" + keys["subject_std"]
    keys = keys[["dataset", "group_key"]].drop_duplicates()
    fold_tests = [set() for _ in range(n_splits)]
    all_keys = set(keys["group_key"].astype(str))
    for _, g in keys.groupby("dataset"):
        ds_keys = g["group_key"].astype(str).tolist()
        rng.shuffle(ds_keys)
        chunks = np.array_split(np.asarray(ds_keys, dtype=object), n_splits)
        for idx, chunk in enumerate(chunks):
            fold_tests[idx].update(str(x) for x in chunk)
    return [(all_keys - test, test) for test in fold_tests]


def guided_selector_loss(
    scores: torch.Tensor,
    group: GroupBatch,
    *,
    alpha_rule: float = 0.25,
    temp_error: float = 2.5,
    temp_rule: float = 0.18,
    err_scale: float = 20.0,
) -> torch.Tensor:
    errors = torch.nan_to_num(group.errors, nan=100.0, posinf=100.0, neginf=100.0)
    rule_np = pd.to_numeric(group.meta["observable_rule_score"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    rule = torch.tensor(rule_np, dtype=torch.float32, device=scores.device)
    q_error = F.softmax(-errors / temp_error, dim=0).detach()
    q_rule = F.softmax((rule - rule.mean()) / max(temp_rule, 1e-3), dim=0).detach()
    q = ((1.0 - alpha_rule) * q_error + alpha_rule * q_rule).detach()
    q = q / torch.clamp(q.sum(), min=1e-8)

    logp = F.log_softmax(scores, dim=0)
    p = F.softmax(scores, dim=0)
    kl = F.kl_div(logp, q, reduction="batchmean")
    expected_error = torch.sum(p * errors) / err_scale

    best_idx = torch.argmin(errors)
    margin = torch.relu(1.0 - scores[best_idx] + scores).mean()

    harmonic = torch.tensor(pd.to_numeric(group.meta.get("harmonic_trap_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32), device=scores.device)
    alias = torch.tensor(pd.to_numeric(group.meta.get("alias_band_risk", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32), device=scores.device)
    deep_disagree = torch.tensor(pd.to_numeric(group.meta.get("deep_disagreement_risk", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32), device=scores.device)
    agreement = torch.tensor(pd.to_numeric(group.meta.get("agreement10_frac", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32), device=scores.device)
    risk_penalty = torch.sum(p * (0.08 * harmonic + 0.04 * alias + 0.03 * deep_disagree - 0.04 * agreement))
    distill = F.kl_div(logp, q_rule, reduction="batchmean")
    return kl + expected_error + 0.08 * margin + risk_penalty + 0.05 * distill


def train_one_model(name: str, model: nn.Module, groups: list[GroupBatch], *, epochs: int, lr: float) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        random.shuffle(groups)
        total = 0.0
        for group in groups:
            opt.zero_grad(set_to_none=True)
            loss = guided_selector_loss(model(group), group)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu())
        if epoch in {0, epochs - 1}:
            print(f"[{TASK_ID}] {name} epoch={epoch + 1}/{epochs} loss={total / max(1, len(groups)):.4f}", flush=True)
    return model


def predict_model(name: str, model: nn.Module, groups: list[GroupBatch]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for group in groups:
            scores = model(group)
            probs = F.softmax(scores, dim=0)
            idx = int(torch.argmax(probs).item())
            entropy = float((-(probs * torch.log(probs + 1e-12)).sum()).detach().cpu())
            meta = group.meta.iloc[idx]
            pred_hr = float(group.hrs[idx].detach().cpu())
            err = abs(pred_hr - group.gt)
            rows.append(
                {
                    "selector": name,
                    "sample_id": group.sample_id,
                    "dataset": str(meta["dataset"]),
                    "subject_std": group.subject,
                    "gt_hr_bpm": group.gt,
                    "pred_hr_bpm": pred_hr,
                    "abs_error_bpm": err,
                    "unsafe_gt10bpm": bool(err > UNSAFE_BPM),
                    "selected_candidate_id": meta["candidate_id"],
                    "selected_source_type": meta["source_type"],
                    "selected_source_name": meta["source_name"],
                    "selected_candidate_family": meta["candidate_family"],
                    "selected_candidate_model": meta["candidate_model"],
                    "raw_score": float(scores[idx].detach().cpu()),
                    "max_probability": float(probs[idx].detach().cpu()),
                    "entropy": entropy,
                    "n_candidates": int(len(group.meta)),
                    "selected_observable_rule_score": float(meta.get("observable_rule_score", 0.0)),
                    "selected_harmonic_trap_score": float(meta.get("harmonic_trap_score", 0.0)),
                    "selected_alias_band_risk": float(meta.get("alias_band_risk", 0.0)),
                    "selected_deep_disagreement_risk": float(meta.get("deep_disagreement_risk", 0.0)),
                    "selected_agreement10_frac": float(meta.get("agreement10_frac", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def run_seed(pool: pd.DataFrame, *, seed: int, epochs: int, lr: float, n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool2 = add_relative_rule_features(pool.copy()).reset_index(drop=True)
    features, feature_cols = feature_frame(pool2)
    pool2["group_key"] = pool2["dataset"].astype(str) + "::" + pool2["subject_std"].astype(str)
    folds = stratified_domain_folds(pool2, n_splits, seed)
    constructors = {
        "mlp_rule_guided": lambda: MLPSelector(len(feature_cols)),
        "set_attention_rule_guided": lambda: SetAttentionSelector(len(feature_cols)),
        "graph_rule_guided": lambda: GraphSelector(len(feature_cols)),
    }
    pred_parts: list[pd.DataFrame] = []
    print(f"[{TASK_ID}] seed={seed} device={device} feature_dim={len(feature_cols)} folds={len(folds)}", flush=True)
    for fold_idx, (train_keys, test_keys) in enumerate(folds, start=1):
        train_mask = pool2["group_key"].isin(train_keys)
        test_mask = pool2["group_key"].isin(test_keys)
        train_pool = pool2[train_mask].reset_index(drop=True)
        test_pool = pool2[test_mask].reset_index(drop=True)
        train_features = features.loc[train_mask].reset_index(drop=True)
        test_features = features.loc[test_mask].reset_index(drop=True)
        _, std = make_groups(train_pool, train_features, feature_cols, device)
        train_groups, _ = make_groups(train_pool, train_features, feature_cols, device, standardize=std)
        test_groups, _ = make_groups(test_pool, test_features, feature_cols, device, standardize=std)
        print(f"[{TASK_ID}] seed={seed} fold={fold_idx}/{len(folds)} train_windows={len(train_groups)} test_windows={len(test_groups)}", flush=True)
        for name, ctor in constructors.items():
            model = ctor().to(device)
            train_one_model(name, model, train_groups, epochs=epochs, lr=lr)
            pred = predict_model(name, model, test_groups)
            pred["seed"] = seed
            pred["fold"] = fold_idx
            pred_parts.append(pred)
    preds = pd.concat(pred_parts, ignore_index=True, sort=False)
    metrics = build_metrics(preds)
    return preds, metrics


def build_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, dataset, selector), g in preds.groupby(["seed", "dataset", "selector"], sort=False):
        err = pd.to_numeric(g["abs_error_bpm"], errors="coerce")
        pred = pd.to_numeric(g["pred_hr_bpm"], errors="coerce").to_numpy(float)
        gt = pd.to_numeric(g["gt_hr_bpm"], errors="coerce").to_numpy(float)
        finite = np.isfinite(pred) & np.isfinite(gt)
        corr = float(np.corrcoef(gt[finite], pred[finite])[0, 1]) if finite.sum() > 1 and np.std(pred[finite]) > 1e-8 and np.std(gt[finite]) > 1e-8 else math.nan
        rows.append(
            {
                "seed": int(seed),
                "dataset": dataset,
                "selector": selector,
                "n_windows": int(len(g)),
                "coverage": 1.0,
                "mae_bpm": float(err.mean()),
                "rmse_bpm": float(np.sqrt(np.mean(np.square(err)))),
                "median_abs_error_bpm": float(err.median()),
                "p90_abs_error_bpm": float(err.quantile(0.90)),
                "unsafe_gt10bpm_rate": float((err > UNSAFE_BPM).mean()),
                "pearson_r": corr,
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "dataset", "mae_bpm"])


def build_max_snr_preds(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pool2 = add_relative_rule_features(pool.copy())
    for (dataset, sample_id), g in pool2.groupby(["dataset", "sample_id"], sort=False):
        row = g.sort_values("mean_snr_proxy_db", ascending=False).iloc[0]
        pred = float(row["candidate_hr_bpm"])
        gt = float(row["gt_hr_bpm"])
        rows.append({"dataset": dataset, "sample_id": sample_id, "baseline_abs_error_bpm": abs(pred - gt)})
    return pd.DataFrame(rows)


def release_gate(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, dataset, selector), g in preds.groupby(["seed", "dataset", "selector"], sort=False):
        h = g.copy()
        entropy_norm = pd.to_numeric(h["entropy"], errors="coerce").fillna(0.0) / np.log(pd.to_numeric(h["n_candidates"], errors="coerce").clip(lower=2))
        h["release_risk"] = (
            1.0
            - pd.to_numeric(h["max_probability"], errors="coerce").fillna(0.0)
            + entropy_norm
            + 0.65 * pd.to_numeric(h["selected_harmonic_trap_score"], errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(h["selected_alias_band_risk"], errors="coerce").fillna(0.0)
            + 0.18 * pd.to_numeric(h["selected_deep_disagreement_risk"], errors="coerce").fillna(0.0)
            - 0.30 * pd.to_numeric(h["selected_agreement10_frac"], errors="coerce").fillna(0.0)
            - 0.15 * pd.to_numeric(h["selected_observable_rule_score"], errors="coerce").fillna(0.0)
        )
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["release_risk"].quantile(q))
            released = h[h["release_risk"] <= tau]
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


def claim_gate(metrics: pd.DataFrame, release: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, m in metrics.iterrows():
        seed = int(m["seed"])
        dataset = str(m["dataset"])
        selector = str(m["selector"])
        base = baselines[(baselines["dataset"].astype(str) == dataset) & (baselines["method"].astype(str) == "max_snr")]
        base_mae = float(base["mae_bpm"].iloc[0]) if not base.empty else math.nan
        safe = release[(release["seed"].astype(int) == seed) & (release["dataset"].astype(str) == dataset) & (release["selector"].astype(str) == selector) & (release["gate_pass_unsafe10"])]
        best = safe.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        best_cov = float(best["coverage"].iloc[0]) if not best.empty else 0.0
        best_unsafe = float(best["unsafe_release_rate"].iloc[0]) if not best.empty else math.nan
        best_mae = float(best["released_mae_bpm"].iloc[0]) if not best.empty else math.nan
        reduction = 1.0 - float(m["mae_bpm"]) / base_mae if math.isfinite(base_mae) and base_mae > 0 else math.nan
        released_reduction = 1.0 - best_mae / base_mae if math.isfinite(base_mae) and math.isfinite(best_mae) and base_mae > 0 else math.nan
        rows.append(
            {
                "seed": seed,
                "dataset": dataset,
                "selector": selector,
                "mae_bpm": float(m["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(m["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_max_snr": reduction,
                "best_safe_gate_coverage": best_cov,
                "best_safe_gate_unsafe": best_unsafe,
                "best_safe_gate_released_mae_bpm": best_mae,
                "released_mae_reduction_vs_max_snr": released_reduction,
                "pass_dataset_gate": bool(math.isfinite(released_reduction) and released_reduction >= 0.20 and best_cov >= 0.40 and best_unsafe <= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "dataset", "mae_bpm"])


def aggregate_claim(claim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, selector), g in claim.groupby(["dataset", "selector"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "selector": selector,
            "n_seeds": int(g["seed"].nunique()),
            "seed_pass_count": int(g["pass_dataset_gate"].fillna(False).astype(bool).sum()),
            "strict_all_seed_pass": bool(g["pass_dataset_gate"].fillna(False).astype(bool).all()),
        }
        for col in [
            "mae_bpm",
            "unsafe_gt10bpm_rate",
            "mae_reduction_vs_max_snr",
            "best_safe_gate_coverage",
            "best_safe_gate_unsafe",
            "best_safe_gate_released_mae_bpm",
            "released_mae_reduction_vs_max_snr",
        ]:
            values = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() >= 2 else 0.0
            row[f"{col}_min"] = float(values.min()) if values.notna().any() else math.nan
            row[f"{col}_max"] = float(values.max()) if values.notna().any() else math.nan
        row["aggregate_gate_pass"] = bool(
            math.isfinite(row.get("released_mae_reduction_vs_max_snr_mean", math.nan))
            and math.isfinite(row.get("best_safe_gate_coverage_mean", math.nan))
            and math.isfinite(row.get("best_safe_gate_unsafe_mean", math.nan))
            and row["released_mae_reduction_vs_max_snr_mean"] >= 0.20
            and row["best_safe_gate_coverage_mean"] >= 0.40
            and row["best_safe_gate_unsafe_mean"] <= 0.10
            and row["seed_pass_count"] >= 2
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "aggregate_gate_pass", "released_mae_reduction_vs_max_snr_mean"], ascending=[True, False, False])


def bootstrap_ci(preds: pd.DataFrame, baseline: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (dataset, selector), g in preds.groupby(["dataset", "selector"], sort=False):
        merged = g[["sample_id", "abs_error_bpm"]].merge(baseline[baseline["dataset"].astype(str) == str(dataset)], on="sample_id", how="inner")
        if merged.empty:
            continue
        diff = pd.to_numeric(merged["abs_error_bpm"], errors="coerce").to_numpy(float) - pd.to_numeric(merged["baseline_abs_error_bpm"], errors="coerce").to_numpy(float)
        diff = diff[np.isfinite(diff)]
        if len(diff) == 0:
            continue
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(diff), len(diff))
            draws.append(float(np.mean(diff[idx])))
        rows.append(
            {
                "dataset": dataset,
                "selector": selector,
                "control": "max_snr",
                "n_pairs": int(len(diff)),
                "mean_delta_mae_bpm": float(np.mean(diff)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "prob_delta_lt_0": float(np.mean(np.asarray(draws) < 0.0)),
                "improvement_supported": bool(np.quantile(draws, 0.975) < 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mean_delta_mae_bpm"])


def write_report(summary: dict[str, Any], agg: pd.DataFrame, claim: pd.DataFrame, boot: pd.DataFrame) -> None:
    view_cols = [
        "dataset",
        "selector",
        "n_seeds",
        "seed_pass_count",
        "aggregate_gate_pass",
        "mae_bpm_mean",
        "mae_bpm_std",
        "mae_reduction_vs_max_snr_mean",
        "best_safe_gate_coverage_mean",
        "best_safe_gate_unsafe_mean",
        "released_mae_reduction_vs_max_snr_mean",
    ]
    lines = [
        "# T753 Rule-Guided Neural Selector Multiseed",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T752 showed that a naked protocol-fixed domain-balanced neural selector transfers poorly to MCD and UBFC-Phys. T753 adds the T731/T732 harmonic-aware rule score as both an input feature and a soft distillation target, while keeping the error-supervised candidate selection objective. The goal is to test whether a physiology/rule-guided neural selector improves external release/review coverage without increasing unsafe release.",
        "",
        "## Aggregate Claim Gate",
        "",
        markdown_table(agg[[c for c in view_cols if c in agg.columns]]),
        "",
        "## Seed-Level Claim Gate",
        "",
        markdown_table(claim),
        "",
        "## Bootstrap CI vs Max-SNR",
        "",
        markdown_table(boot),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "This task only supports the main paper claim if at least MCD-rPPG plus one external dataset pass the pre-specified aggregate release gate. Otherwise it is evidence that rule-guided neural selection is insufficient and a stronger risk calibration or bounded claim is required.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("704,1704,2704"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if not IN_POOL.exists():
        raise FileNotFoundError(IN_POOL)
    if not IN_BASELINES.exists():
        raise FileNotFoundError(IN_BASELINES)
    EXP.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(IN_POOL, low_memory=False)
    baselines = pd.read_csv(IN_BASELINES)
    all_preds: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for seed in args.seeds:
        preds, metrics = run_seed(pool, seed=seed, epochs=args.epochs, lr=args.lr, n_splits=args.n_splits)
        all_preds.append(preds)
        all_metrics.append(metrics)
    preds = pd.concat(all_preds, ignore_index=True, sort=False)
    metrics = pd.concat(all_metrics, ignore_index=True, sort=False)
    release = release_gate(preds)
    claim = claim_gate(metrics, release, baselines)
    agg = aggregate_claim(claim)
    baseline_preds = build_max_snr_preds(pool)
    boot = bootstrap_ci(preds, baseline_preds, n_boot=args.bootstrap, seed=args.seeds[0])

    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8")
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8")
    release.to_csv(OUT_RELEASE, index=False, encoding="utf-8")
    claim.to_csv(OUT_CLAIM, index=False, encoding="utf-8")
    agg.to_csv(OUT_AGG, index=False, encoding="utf-8")
    boot.to_csv(OUT_BOOT, index=False, encoding="utf-8")

    external = agg[agg["dataset"].astype(str).ne("UBFC-rPPG")]
    passed_external = external[external["aggregate_gate_pass"]]
    decision = (
        "rule_guided_neural_selector_supported_continue_product_paper_update"
        if passed_external["dataset"].nunique() >= 2
        else "rule_guided_neural_selector_partial_requires_risk_gate_or_bounded_claim"
    )
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "n_splits": args.n_splits,
        "n_windows": int(pool["sample_id"].nunique()),
        "n_candidates": int(len(pool)),
        "datasets": sorted(pool["dataset"].astype(str).unique()),
        "external_datasets_passing_aggregate_gate": sorted(passed_external["dataset"].astype(str).unique()) if not passed_external.empty else [],
        "decision": decision,
        "outputs": {
            "predictions": str(OUT_PREDS),
            "metrics": str(OUT_METRICS),
            "release": str(OUT_RELEASE),
            "claim": str(OUT_CLAIM),
            "aggregate": str(OUT_AGG),
            "bootstrap": str(OUT_BOOT),
            "doc": str(OUT_MD),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, agg, claim, boot)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
