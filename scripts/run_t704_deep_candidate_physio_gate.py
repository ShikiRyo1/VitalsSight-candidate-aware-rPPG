from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


TASK_ID = "T704"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"

CLASSICAL_CANDIDATES = EXP / "t467_ubfc_protocol_window_candidate_table.csv"
DEEP_WINDOW_METRICS = EXP / "t475_deep_baseline_window_metrics.csv"
PREVIOUS_COMPARISON = EXP / "t474_ubfc_protocol_harmonized_comparison_table.csv"

OUT_POOL = EXP / "t704_unified_candidate_pool.csv"
OUT_AUDIT = EXP / "t704_matched_window_audit.csv"
OUT_BASELINES = EXP / "t704_baseline_metrics.csv"
OUT_SELECTOR_METRICS = EXP / "t704_selector_comparison_metrics.csv"
OUT_SELECTOR_PREDS = EXP / "t704_selector_predictions.csv"
OUT_RELEASE = EXP / "t704_release_gate_risk_coverage.csv"
OUT_ABLATION = EXP / "t704_ablation_metrics.csv"
OUT_BOOTSTRAP = EXP / "t704_bootstrap_ci.csv"
OUT_CLAIM_GATE = EXP / "t704_claim_gate.csv"
OUT_SUMMARY = EXP / "t704_deep_candidate_physio_gate_summary.json"
OUT_MD = DOCS / "t704_deep_candidate_physio_gate_results.md"

UNSAFE_BPM = 10.0
SEED = 704


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


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


def standard_subject(subject_id: Any, sample_id: str) -> str:
    text = str(subject_id)
    if text and text != "nan":
        if text.startswith("subject"):
            suffix = text.replace("subject", "")
            try:
                return f"subject{int(suffix)}"
            except Exception:
                return text
        return text
    return str(sample_id).split("_")[0]


def build_unified_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CLASSICAL_CANDIDATES.exists():
        raise FileNotFoundError(CLASSICAL_CANDIDATES)
    if not DEEP_WINDOW_METRICS.exists():
        raise FileNotFoundError(DEEP_WINDOW_METRICS)

    classical_raw = pd.read_csv(CLASSICAL_CANDIDATES)
    deep_raw = pd.read_csv(DEEP_WINDOW_METRICS)
    deep_raw = deep_raw[
        (deep_raw["dataset"].astype(str) == "UBFC-rPPG")
        & (pd.to_numeric(deep_raw["clip_len"], errors="coerce") == 180)
        & (deep_raw["status"].astype(str) == "ok")
    ].copy()

    gt_classical = (
        classical_raw[["sample_id", "gt_hr_bpm"]]
        .drop_duplicates()
        .rename(columns={"gt_hr_bpm": "gt_classical"})
    )
    gt_deep = (
        deep_raw[["sample_id", "gt_hr_bpm"]]
        .drop_duplicates()
        .rename(columns={"gt_hr_bpm": "gt_deep"})
    )
    audit = gt_classical.merge(gt_deep, on="sample_id", how="outer")
    audit["gt_delta_bpm"] = (audit["gt_classical"] - audit["gt_deep"]).abs()
    audit["matched"] = audit["gt_classical"].notna() & audit["gt_deep"].notna()
    audit["label_consistent_1bpm"] = audit["matched"] & (audit["gt_delta_bpm"] <= 1.0)
    audit["label_consistent_3bpm"] = audit["matched"] & (audit["gt_delta_bpm"] <= 3.0)

    matched_ids = set(audit.loc[audit["label_consistent_1bpm"], "sample_id"].astype(str))
    if len(matched_ids) < 100:
        # T467 and T475 should be protocol aligned. If strict 1 BPM is too harsh,
        # fall back to matched windows and record the actual delta for review.
        matched_ids = set(audit.loc[audit["matched"], "sample_id"].astype(str))

    classical = classical_raw[classical_raw["sample_id"].astype(str).isin(matched_ids)].copy()
    classical["source_type"] = "classical"
    classical["source_name"] = "candidate_pool"
    classical["candidate_hr_bpm"] = pd.to_numeric(classical["candidate_bpm"], errors="coerce")
    classical["reference_hr_bpm"] = pd.to_numeric(classical["gt_hr_bpm"], errors="coerce")
    classical["candidate_abs_error"] = pd.to_numeric(classical["candidate_abs_error_bpm"], errors="coerce")
    classical["subject_std"] = [standard_subject(s, sid) for s, sid in zip(classical["subject_id"], classical["sample_id"])]
    classical["candidate_family"] = "regional_peak"
    classical["candidate_model"] = classical["candidate_id"].astype(str).str.extract(r"(tk\d+)", expand=False).fillna("peak")
    classical["deep_snr"] = np.nan
    classical["deep_macc"] = np.nan

    deep = deep_raw[deep_raw["sample_id"].astype(str).isin(matched_ids)].copy()
    deep["source_type"] = "deep"
    deep["source_name"] = deep["model"].astype(str)
    deep["candidate_id"] = deep["sample_id"].astype(str) + "_deep_" + deep["model"].astype(str)
    deep["candidate_hr_bpm"] = pd.to_numeric(deep["pred_hr_bpm"], errors="coerce")
    deep["reference_hr_bpm"] = pd.to_numeric(deep["gt_hr_bpm"], errors="coerce")
    deep["candidate_abs_error"] = pd.to_numeric(deep["abs_error_bpm"], errors="coerce")
    deep["subject_std"] = [standard_subject(s, sid) for s, sid in zip(deep["subject_id"], deep["sample_id"])]
    deep["candidate_family"] = "deep_backbone"
    deep["candidate_model"] = deep["model"].astype(str)
    deep["deep_snr"] = pd.to_numeric(deep["snr"], errors="coerce")
    deep["deep_macc"] = pd.to_numeric(deep["macc"], errors="coerce")

    # Align columns by creating missing feature columns on deep rows.
    common_feature_defaults = {
        "support_count": 1.0,
        "support_rois": "",
        "support_methods": "",
        "support_windows": "",
        "full_support_count": 0.0,
        "subwindow_support_count": 0.0,
        "top1_support_count": 0.0,
        "full_top1_support_count": 0.0,
        "pos_chrom_count": 0.0,
        "green_pbv_count": 0.0,
        "ica_lgi_count": 0.0,
        "mean_power_fraction": 0.0,
        "max_power_fraction": 0.0,
        "sum_power_fraction": 0.0,
        "rank_score": 0.0,
        "mean_snr_proxy_db": np.nan,
        "adult_plausibility": 1.0,
        "upper_alt_support": 0.0,
        "upper_alt_pos_chrom": 0.0,
        "upper_phys_support": 0.0,
        "upper_phys_pos_chrom": 0.0,
        "lower_phys_support": 0.0,
        "lower_phys_pos_chrom": 0.0,
        "double_harmonic_support": 0.0,
        "half_harmonic_support": 0.0,
        "t150_selected_bpm": np.nan,
        "t150_abs_error_bpm": np.nan,
        "t150_confidence": 0.0,
        "dist_to_t150": np.nan,
        "t157_low_alias_penalty": 0.0,
        "t157_motion_band_penalty": 0.0,
        "t157_high_alias_penalty": 0.0,
        "t157_near_t150_boost": 0.0,
        "t157_score": 0.0,
    }
    for col, default in common_feature_defaults.items():
        if col not in classical.columns:
            classical[col] = default
        if col not in deep.columns:
            deep[col] = default
    deep["mean_snr_proxy_db"] = pd.to_numeric(deep["deep_snr"], errors="coerce")
    deep["rank_score"] = pd.to_numeric(deep["deep_macc"], errors="coerce").fillna(0.0)

    keep_cols = [
        "sample_id",
        "dataset",
        "subject_std",
        "gt_hr_bpm",
        "reference_hr_bpm",
        "candidate_id",
        "candidate_hr_bpm",
        "candidate_abs_error",
        "source_type",
        "source_name",
        "candidate_family",
        "candidate_model",
        "support_count",
        "full_support_count",
        "subwindow_support_count",
        "top1_support_count",
        "full_top1_support_count",
        "pos_chrom_count",
        "green_pbv_count",
        "ica_lgi_count",
        "mean_power_fraction",
        "max_power_fraction",
        "sum_power_fraction",
        "rank_score",
        "mean_snr_proxy_db",
        "adult_plausibility",
        "upper_alt_support",
        "upper_alt_pos_chrom",
        "upper_phys_support",
        "upper_phys_pos_chrom",
        "lower_phys_support",
        "lower_phys_pos_chrom",
        "double_harmonic_support",
        "half_harmonic_support",
        "t150_selected_bpm",
        "t150_confidence",
        "dist_to_t150",
        "t157_low_alias_penalty",
        "t157_motion_band_penalty",
        "t157_high_alias_penalty",
        "t157_near_t150_boost",
        "t157_score",
        "deep_snr",
        "deep_macc",
    ]
    pool = pd.concat([classical[keep_cols], deep[keep_cols]], ignore_index=True, sort=False)
    pool["gt_hr_bpm"] = pd.to_numeric(pool["reference_hr_bpm"], errors="coerce")
    pool["candidate_abs_error"] = (pool["candidate_hr_bpm"] - pool["gt_hr_bpm"]).abs()
    pool["unsafe_candidate"] = pool["candidate_abs_error"] > UNSAFE_BPM
    pool = add_relation_features(pool)
    pool.to_csv(OUT_POOL, index=False, encoding="utf-8")
    audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8")
    return pool, audit


def add_relation_features(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, g in pool.groupby("sample_id", sort=False):
        h = g.copy()
        hr = pd.to_numeric(h["candidate_hr_bpm"], errors="coerce")
        snr = pd.to_numeric(h["mean_snr_proxy_db"], errors="coerce")
        support = pd.to_numeric(h["support_count"], errors="coerce").fillna(0.0)
        median_hr = float(hr.median())
        mean_hr = float(hr.mean())
        top_snr_hr = float(h.loc[snr.fillna(-999).idxmax(), "candidate_hr_bpm"]) if len(h) else math.nan
        h["dist_to_group_median_hr"] = (hr - median_hr).abs()
        h["dist_to_group_mean_hr"] = (hr - mean_hr).abs()
        h["dist_to_top_snr_hr"] = (hr - top_snr_hr).abs() if math.isfinite(top_snr_hr) else 0.0
        h["agreement5_frac"] = [float(((hr - x).abs() <= 5.0).mean()) for x in hr]
        h["agreement10_frac"] = [float(((hr - x).abs() <= 10.0).mean()) for x in hr]
        h["agreement20_frac"] = [float(((hr - x).abs() <= 20.0).mean()) for x in hr]
        h["snr_rank_pct"] = snr.rank(pct=True).fillna(0.0)
        h["support_rank_pct"] = support.rank(pct=True).fillna(0.0)
        harmonic = (
            pd.to_numeric(h["double_harmonic_support"], errors="coerce").fillna(0.0)
            + pd.to_numeric(h["half_harmonic_support"], errors="coerce").fillna(0.0)
            + pd.to_numeric(h["t157_low_alias_penalty"], errors="coerce").fillna(0.0)
            + pd.to_numeric(h["t157_high_alias_penalty"], errors="coerce").fillna(0.0)
        )
        h["harmonic_risk"] = harmonic
        h["is_deep"] = (h["source_type"].astype(str) == "deep").astype(float)
        h["is_classical"] = (h["source_type"].astype(str) == "classical").astype(float)
        rows.append(h)
    return pd.concat(rows, ignore_index=True, sort=False)


def metric_row(name: str, pred: pd.DataFrame) -> dict[str, Any]:
    err = pd.to_numeric(pred["abs_error_bpm"], errors="coerce")
    ok = err.notna()
    gt = pd.to_numeric(pred.loc[ok, "gt_hr_bpm"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(pred.loc[ok, "pred_hr_bpm"], errors="coerce").to_numpy(float)
    finite = np.isfinite(gt) & np.isfinite(y)
    if finite.any() and np.std(gt[finite]) > 1e-8 and np.std(y[finite]) > 1e-8:
        corr = float(np.corrcoef(gt[finite], y[finite])[0, 1])
    else:
        corr = math.nan
    return {
        "method": name,
        "n_windows": int(ok.sum()),
        "coverage": float(ok.mean()),
        "mae_bpm": float(err[ok].mean()) if ok.any() else math.nan,
        "rmse_bpm": float(np.sqrt(np.mean(np.square(err[ok])))) if ok.any() else math.nan,
        "median_abs_error_bpm": float(err[ok].median()) if ok.any() else math.nan,
        "p90_abs_error_bpm": float(err[ok].quantile(0.90)) if ok.any() else math.nan,
        "unsafe_gt10bpm_rate": float((err[ok] > UNSAFE_BPM).mean()) if ok.any() else math.nan,
        "pearson_r": corr,
    }


def build_baselines(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict[str, Any]] = []
    for sample_id, g in pool.groupby("sample_id", sort=False):
        gt = float(g["gt_hr_bpm"].iloc[0])
        classical = g[g["source_type"] == "classical"].copy()
        if not classical.empty:
            top_snr = classical.loc[pd.to_numeric(classical["mean_snr_proxy_db"], errors="coerce").fillna(-999).idxmax()]
            oracle = classical.loc[pd.to_numeric(classical["candidate_abs_error"], errors="coerce").idxmin()]
            for name, row in [("classical_max_snr", top_snr), ("classical_oracle", oracle)]:
                pred_rows.append(
                    {
                        "method": name,
                        "sample_id": sample_id,
                        "subject_std": row["subject_std"],
                        "gt_hr_bpm": gt,
                        "pred_hr_bpm": float(row["candidate_hr_bpm"]),
                        "abs_error_bpm": abs(float(row["candidate_hr_bpm"]) - gt),
                        "selected_source_type": row["source_type"],
                        "selected_source_name": row["source_name"],
                        "selected_candidate_id": row["candidate_id"],
                    }
                )
        for model, dg in g[g["source_type"] == "deep"].groupby("source_name"):
            row = dg.iloc[0]
            pred_rows.append(
                {
                    "method": f"deep_{model}",
                    "sample_id": sample_id,
                    "subject_std": row["subject_std"],
                    "gt_hr_bpm": gt,
                    "pred_hr_bpm": float(row["candidate_hr_bpm"]),
                    "abs_error_bpm": abs(float(row["candidate_hr_bpm"]) - gt),
                    "selected_source_type": row["source_type"],
                    "selected_source_name": row["source_name"],
                    "selected_candidate_id": row["candidate_id"],
                }
            )
        oracle = g.loc[pd.to_numeric(g["candidate_abs_error"], errors="coerce").idxmin()]
        pred_rows.append(
            {
                "method": "unified_oracle",
                "sample_id": sample_id,
                "subject_std": oracle["subject_std"],
                "gt_hr_bpm": gt,
                "pred_hr_bpm": float(oracle["candidate_hr_bpm"]),
                "abs_error_bpm": abs(float(oracle["candidate_hr_bpm"]) - gt),
                "selected_source_type": oracle["source_type"],
                "selected_source_name": oracle["source_name"],
                "selected_candidate_id": oracle["candidate_id"],
            }
        )
    preds = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame([metric_row(name, p) for name, p in preds.groupby("method")]).sort_values("mae_bpm")
    metrics.to_csv(OUT_BASELINES, index=False, encoding="utf-8")
    return preds, metrics


FEATURE_NUMERIC = [
    "candidate_hr_bpm",
    "support_count",
    "full_support_count",
    "subwindow_support_count",
    "top1_support_count",
    "full_top1_support_count",
    "pos_chrom_count",
    "green_pbv_count",
    "ica_lgi_count",
    "mean_power_fraction",
    "max_power_fraction",
    "sum_power_fraction",
    "rank_score",
    "mean_snr_proxy_db",
    "adult_plausibility",
    "upper_alt_support",
    "upper_alt_pos_chrom",
    "upper_phys_support",
    "upper_phys_pos_chrom",
    "lower_phys_support",
    "lower_phys_pos_chrom",
    "double_harmonic_support",
    "half_harmonic_support",
    "t150_selected_bpm",
    "t150_confidence",
    "dist_to_t150",
    "t157_low_alias_penalty",
    "t157_motion_band_penalty",
    "t157_high_alias_penalty",
    "t157_near_t150_boost",
    "t157_score",
    "deep_snr",
    "deep_macc",
    "dist_to_group_median_hr",
    "dist_to_group_mean_hr",
    "dist_to_top_snr_hr",
    "agreement5_frac",
    "agreement10_frac",
    "agreement20_frac",
    "snr_rank_pct",
    "support_rank_pct",
    "harmonic_risk",
    "is_deep",
    "is_classical",
]


@dataclass
class GroupBatch:
    sample_id: str
    subject: str
    x: torch.Tensor
    errors: torch.Tensor
    hrs: torch.Tensor
    gt: float
    meta: pd.DataFrame
    adj: torch.Tensor | None


def feature_frame(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = pool.copy()
    for col in FEATURE_NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat = pd.get_dummies(df[["source_type", "source_name", "candidate_family", "candidate_model"]].astype(str), prefix=["src", "name", "family", "model"])
    features = pd.concat([df[FEATURE_NUMERIC], cat], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    feature_cols = list(features.columns)
    return features.astype(float), feature_cols


def make_groups(pool: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str], device: torch.device, *, standardize: dict[str, np.ndarray] | None = None) -> tuple[list[GroupBatch], dict[str, np.ndarray]]:
    x_all = features[feature_cols].to_numpy(np.float32)
    if standardize is None:
        mean = np.nanmean(x_all, axis=0)
        std = np.nanstd(x_all, axis=0)
        std[std < 1e-6] = 1.0
        standardize = {"mean": mean, "std": std}
    x_all = (x_all - standardize["mean"]) / standardize["std"]
    groups: list[GroupBatch] = []
    pool2 = pool.copy().reset_index(drop=True)
    pool2["_row"] = np.arange(len(pool2))
    for sample_id, g in pool2.groupby("sample_id", sort=False):
        idx = g["_row"].to_numpy(int)
        x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
        err = torch.tensor(pd.to_numeric(g["candidate_abs_error"], errors="coerce").to_numpy(np.float32), device=device)
        hr = torch.tensor(pd.to_numeric(g["candidate_hr_bpm"], errors="coerce").to_numpy(np.float32), device=device)
        gt = float(g["gt_hr_bpm"].iloc[0])
        adj = build_adjacency(g).to(device)
        groups.append(GroupBatch(str(sample_id), str(g["subject_std"].iloc[0]), x, err, hr, gt, g.copy(), adj))
    return groups, standardize


def build_adjacency(g: pd.DataFrame) -> torch.Tensor:
    hr = pd.to_numeric(g["candidate_hr_bpm"], errors="coerce").to_numpy(float)
    n = len(g)
    adj = np.eye(n, dtype=np.float32)
    src = g["source_type"].astype(str).to_numpy()
    name = g["source_name"].astype(str).to_numpy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            close = abs(hr[i] - hr[j]) <= 5.0
            same_src = src[i] == src[j]
            same_name = name[i] == name[j]
            harmonic = False
            if hr[i] > 1e-6 and hr[j] > 1e-6:
                ratio = hr[i] / hr[j]
                harmonic = abs(ratio - 2.0) <= 0.08 or abs(ratio - 0.5) <= 0.04
            if close or same_src or same_name or harmonic:
                adj[i, j] = 1.0
    denom = adj.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return torch.tensor(adj / denom, dtype=torch.float32)


class MLPSelector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.08), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, group: GroupBatch) -> torch.Tensor:
        return self.net(group.x).squeeze(-1)


class SetAttentionSelector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, group: GroupBatch) -> torch.Tensor:
        h = self.embed(group.x).unsqueeze(0)
        h2, _ = self.attn(h, h, h, need_weights=False)
        return self.out((h + h2).squeeze(0)).squeeze(-1)


class GraphSelector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.self1 = nn.Linear(hidden, hidden)
        self.neigh1 = nn.Linear(hidden, hidden)
        self.self2 = nn.Linear(hidden, hidden)
        self.neigh2 = nn.Linear(hidden, hidden)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, group: GroupBatch) -> torch.Tensor:
        h = F.relu(self.inp(group.x))
        adj = group.adj
        assert adj is not None
        n1 = adj @ h
        h = F.relu(self.self1(h) + self.neigh1(n1))
        n2 = adj @ h
        h = F.relu(self.self2(h) + self.neigh2(n2))
        return self.out(h).squeeze(-1)


def selector_loss(scores: torch.Tensor, group: GroupBatch, *, temp: float = 2.5, err_scale: float = 20.0) -> torch.Tensor:
    errors = torch.nan_to_num(group.errors, nan=100.0, posinf=100.0, neginf=100.0)
    q = F.softmax(-errors / temp, dim=0).detach()
    logp = F.log_softmax(scores, dim=0)
    p = F.softmax(scores, dim=0)
    kl = F.kl_div(logp, q, reduction="batchmean")
    expected_error = torch.sum(p * errors) / err_scale
    best_idx = torch.argmin(errors)
    margin = torch.relu(1.0 - scores[best_idx] + scores).mean()
    harmonic_risk = torch.tensor(pd.to_numeric(group.meta["harmonic_risk"], errors="coerce").fillna(0.0).to_numpy(np.float32), device=scores.device)
    harmonic_penalty = torch.sum(p * torch.clamp(harmonic_risk, min=0.0)) * 0.02
    return kl + expected_error + 0.10 * margin + harmonic_penalty


def train_one_model(name: str, model: nn.Module, train_groups: list[GroupBatch], *, epochs: int, lr: float) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        random.shuffle(train_groups)
        total = 0.0
        for group in train_groups:
            opt.zero_grad(set_to_none=True)
            scores = model(group)
            loss = selector_loss(scores, group)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu())
        if epoch in {0, epochs - 1}:
            print(f"[{TASK_ID}] {name} epoch={epoch + 1}/{epochs} loss={total / max(1, len(train_groups)):.4f}", flush=True)
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
                    "subject_std": group.subject,
                    "gt_hr_bpm": group.gt,
                    "pred_hr_bpm": pred_hr,
                    "abs_error_bpm": err,
                    "unsafe_gt10bpm": bool(err > UNSAFE_BPM),
                    "selected_candidate_id": meta["candidate_id"],
                    "selected_source_type": meta["source_type"],
                    "selected_source_name": meta["source_name"],
                    "selected_candidate_family": meta["candidate_family"],
                    "max_probability": float(probs[idx].detach().cpu()),
                    "entropy": entropy,
                    "n_candidates": int(len(group.meta)),
                    "selected_harmonic_risk": safe_float(meta.get("harmonic_risk", 0.0)),
                    "selected_agreement10_frac": safe_float(meta.get("agreement10_frac", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def subject_folds(subjects: list[str], *, mode: str, n_splits: int, seed: int) -> list[tuple[list[str], list[str]]]:
    subjects = list(subjects)
    if mode == "loso":
        return [([s for s in subjects if s != held], [held]) for held in subjects]
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    if mode == "holdout":
        n_test = max(1, int(round(len(shuffled) * 0.20)))
        test = sorted(shuffled[:n_test])
        train = sorted(shuffled[n_test:])
        return [(train, test)]
    if mode == "groupk":
        n_splits = max(2, min(n_splits, len(shuffled)))
        chunks = [list(x) for x in np.array_split(np.asarray(shuffled, dtype=object), n_splits)]
        folds: list[tuple[list[str], list[str]]] = []
        for chunk in chunks:
            test = sorted(str(x) for x in chunk)
            train = sorted(s for s in subjects if s not in test)
            folds.append((train, test))
        return folds
    raise ValueError(f"Unsupported cv mode: {mode}")


def run_group_selectors(
    pool: pd.DataFrame,
    *,
    epochs: int,
    lr: float,
    seed: int,
    cv_mode: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    features, feature_cols = feature_frame(pool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = sorted(pool["subject_std"].astype(str).unique())
    folds = subject_folds(subjects, mode=cv_mode, n_splits=n_splits, seed=seed)
    all_pred: list[pd.DataFrame] = []
    in_dim = len(feature_cols)
    print(f"[{TASK_ID}] device={device} subjects={len(subjects)} feature_dim={in_dim} cv_mode={cv_mode} folds={len(folds)}", flush=True)
    for fold_idx, (train_subjects, test_subjects) in enumerate(folds, start=1):
        train_set = set(train_subjects)
        test_set = set(test_subjects)
        train_mask = pool["subject_std"].astype(str).isin(train_set)
        test_mask = pool["subject_std"].astype(str).isin(test_set)
        train_pool = pool[train_mask].reset_index(drop=True)
        test_pool = pool[test_mask].reset_index(drop=True)
        train_features = features.loc[train_mask].reset_index(drop=True)
        test_features = features.loc[test_mask].reset_index(drop=True)
        _, std = make_groups(train_pool, train_features, feature_cols, device)
        train_groups, _ = make_groups(train_pool, train_features, feature_cols, device, standardize=std)
        test_groups, _ = make_groups(test_pool, test_features, feature_cols, device, standardize=std)
        constructors = {
            "mlp_relation": lambda: MLPSelector(in_dim),
            "set_attention": lambda: SetAttentionSelector(in_dim),
            "graph_selector": lambda: GraphSelector(in_dim),
        }
        fold_label = ",".join(test_subjects[:5]) + ("..." if len(test_subjects) > 5 else "")
        print(f"[{TASK_ID}] fold {fold_idx}/{len(folds)} test_subjects={fold_label} train_groups={len(train_groups)} test_groups={len(test_groups)}", flush=True)
        for name, ctor in constructors.items():
            model = ctor().to(device)
            model = train_one_model(name, model, train_groups, epochs=epochs, lr=lr)
            pred = predict_model(name, model, test_groups)
            pred["fold_subject"] = fold_label
            all_pred.append(pred)
    preds = pd.concat(all_pred, ignore_index=True, sort=False)
    metrics = pd.DataFrame([metric_row(name, g.rename(columns={"selector": "method"})) for name, g in preds.groupby("selector")])
    return preds, metrics.sort_values("mae_bpm")


def release_gate(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selector, g in preds.groupby("selector"):
        h = g.copy()
        risk = (
            (1.0 - pd.to_numeric(h["max_probability"], errors="coerce").fillna(0.0))
            + pd.to_numeric(h["entropy"], errors="coerce").fillna(0.0) / np.log(pd.to_numeric(h["n_candidates"], errors="coerce").clip(lower=2))
            + 0.15 * pd.to_numeric(h["selected_harmonic_risk"], errors="coerce").fillna(0.0)
            - 0.20 * pd.to_numeric(h["selected_agreement10_frac"], errors="coerce").fillna(0.0)
        )
        h["risk_score"] = risk
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["risk_score"].quantile(q))
            released = h[h["risk_score"] <= tau]
            if released.empty:
                continue
            err = pd.to_numeric(released["abs_error_bpm"], errors="coerce")
            rows.append(
                {
                    "selector": selector,
                    "threshold_quantile": float(q),
                    "risk_threshold": tau,
                    "coverage": float(len(released) / len(h)),
                    "released_mae_bpm": float(err.mean()),
                    "released_rmse_bpm": float(np.sqrt(np.mean(np.square(err)))),
                    "unsafe_release_rate": float((err > UNSAFE_BPM).mean()),
                    "n_released": int(len(released)),
                    "n_total": int(len(h)),
                    "gate_pass_unsafe10": bool((err > UNSAFE_BPM).mean() <= 0.10),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["selector", "threshold_quantile"])


def bootstrap_ci(preds: pd.DataFrame, baseline_preds: pd.DataFrame, baseline_name: str = "deep_TSCAN", n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = baseline_preds[baseline_preds["method"] == baseline_name][["sample_id", "abs_error_bpm"]].rename(columns={"abs_error_bpm": "baseline_error"})
    rows: list[dict[str, Any]] = []
    for selector, g in preds.groupby("selector"):
        merged = g[["sample_id", "abs_error_bpm"]].merge(base, on="sample_id", how="inner")
        if merged.empty:
            continue
        diff = (merged["abs_error_bpm"] - merged["baseline_error"]).to_numpy(float)
        boot = []
        n = len(diff)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boot.append(float(np.mean(diff[idx])))
        rows.append(
            {
                "selector": selector,
                "baseline": baseline_name,
                "n_pairs": int(n),
                "mean_delta_mae_bpm": float(np.mean(diff)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "improvement_supported": bool(np.quantile(boot, 0.975) < 0.0),
            }
        )
    return pd.DataFrame(rows)


def ablation_table(pool: pd.DataFrame, preds: pd.DataFrame, baseline_preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(metric_row("classical_max_snr", baseline_preds[baseline_preds["method"] == "classical_max_snr"].rename(columns={"method": "selector"})))
    rows.append(metric_row("deep_TSCAN", baseline_preds[baseline_preds["method"] == "deep_TSCAN"].rename(columns={"method": "selector"})))
    for selector, g in preds.groupby("selector"):
        rows.append(metric_row(selector, g.rename(columns={"selector": "method"})))
        no_deep = g[g["selected_source_type"] != "deep"]
        rows.append(metric_row(selector + "_selected_non_deep_only", no_deep.rename(columns={"selector": "method"})))
    return pd.DataFrame(rows)


def claim_gate(selector_metrics: pd.DataFrame, baselines: pd.DataFrame, release: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    base_top = baselines[baselines["method"] == "classical_max_snr"].iloc[0]
    tscan = baselines[baselines["method"] == "deep_TSCAN"].iloc[0] if (baselines["method"] == "deep_TSCAN").any() else None
    rows: list[dict[str, Any]] = []
    for _, m in selector_metrics.iterrows():
        selector = str(m["method"])
        rel = release[(release["selector"] == selector) & (release["gate_pass_unsafe10"])]
        best_rel = rel.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        boot_row = boot[boot["selector"] == selector].head(1)
        reduction_vs_top = 1.0 - float(m["mae_bpm"]) / float(base_top["mae_bpm"])
        reduction_vs_tscan = math.nan if tscan is None else 1.0 - float(m["mae_bpm"]) / float(tscan["mae_bpm"])
        rows.append(
            {
                "selector": selector,
                "mae_bpm": float(m["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(m["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_classical_max_snr": reduction_vs_top,
                "mae_reduction_vs_deep_tscan": reduction_vs_tscan,
                "best_safe_gate_coverage": float(best_rel["coverage"].iloc[0]) if not best_rel.empty else 0.0,
                "best_safe_gate_unsafe": float(best_rel["unsafe_release_rate"].iloc[0]) if not best_rel.empty else math.nan,
                "bootstrap_vs_tscan_supported": bool(boot_row["improvement_supported"].iloc[0]) if not boot_row.empty else False,
                "pass_main_gate": bool(
                    reduction_vs_top >= 0.20
                    and float(m["unsafe_gt10bpm_rate"]) <= 0.10
                    and (not best_rel.empty)
                    and float(best_rel["coverage"].iloc[0]) >= 0.40
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_bpm")


def write_report(summary: dict[str, Any], baselines: pd.DataFrame, selector_metrics: pd.DataFrame, release: pd.DataFrame, boot: pd.DataFrame, gate: pd.DataFrame) -> None:
    best_selector = summary.get("best_selector", "")
    lines = [
        "# T704 Deep-Candidate PhysioGate Results",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T704 tests whether adding deep backbone predictions into the same-window candidate pool, then selecting with a physiology-constrained risk-aware selector, improves contactless HR release decisions over both classical max-SNR selection and deep-backbone-alone baselines.",
        "",
        "## Baselines",
        "",
        markdown_table(baselines[["method", "n_windows", "mae_bpm", "rmse_bpm", "unsafe_gt10bpm_rate", "pearson_r"]].head(20)),
        "",
        "## Selector Comparison",
        "",
        markdown_table(selector_metrics[["method", "n_windows", "mae_bpm", "rmse_bpm", "unsafe_gt10bpm_rate", "pearson_r"]]),
        "",
        "## Risk Gate",
        "",
        markdown_table(release.sort_values(["selector", "gate_pass_unsafe10", "coverage"], ascending=[True, False, False]).groupby("selector").head(3)),
        "",
        "## Bootstrap CI vs Deep TSCAN",
        "",
        markdown_table(boot),
        "",
        "## Claim Gate",
        "",
        markdown_table(gate),
        "",
        "## Interim Decision",
        "",
        f"Best selector by MAE: `{best_selector}`.",
        "",
        summary.get("decision", ""),
        "",
        "## Claim Boundary",
        "",
        "This result can support a bounded decision-layer claim only if the claim gate passes. It does not by itself prove universal SOTA, clinical-grade monitoring, solved fairness, or solved low-light/NIR robustness.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-mode", choices=["groupk", "holdout", "loso"], default="groupk")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    LOGS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    pool, audit = build_unified_candidate_pool()
    baseline_preds, baseline_metrics = build_baselines(pool)
    preds, selector_metrics = run_group_selectors(
        pool,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        cv_mode=args.cv_mode,
        n_splits=args.n_splits,
    )
    release = release_gate(preds)
    boot = bootstrap_ci(preds, baseline_preds, n_boot=args.bootstrap)
    ablation = ablation_table(pool, preds, baseline_preds)
    gate = claim_gate(selector_metrics, baseline_metrics, release, boot)

    preds.to_csv(OUT_SELECTOR_PREDS, index=False, encoding="utf-8")
    selector_metrics.to_csv(OUT_SELECTOR_METRICS, index=False, encoding="utf-8")
    release.to_csv(OUT_RELEASE, index=False, encoding="utf-8")
    boot.to_csv(OUT_BOOTSTRAP, index=False, encoding="utf-8")
    ablation.to_csv(OUT_ABLATION, index=False, encoding="utf-8")
    gate.to_csv(OUT_CLAIM_GATE, index=False, encoding="utf-8")

    best = selector_metrics.sort_values("mae_bpm").iloc[0]
    passed = bool(gate["pass_main_gate"].any())
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "decision": "main_gate_passed_continue_external_validation" if passed else "main_gate_not_yet_passed_analyze_failure_before_external_validation",
        "best_selector": str(best["method"]),
        "best_selector_mae_bpm": float(best["mae_bpm"]),
        "best_selector_unsafe_gt10bpm_rate": float(best["unsafe_gt10bpm_rate"]),
        "n_candidates": int(len(pool)),
        "n_windows": int(pool["sample_id"].nunique()),
        "n_subjects": int(pool["subject_std"].nunique()),
        "n_deep_candidates": int((pool["source_type"] == "deep").sum()),
        "n_classical_candidates": int((pool["source_type"] == "classical").sum()),
        "cv_mode": args.cv_mode,
        "n_splits": args.n_splits,
        "matched_window_audit": {
            "n_rows": int(len(audit)),
            "n_matched": int(audit["matched"].sum()),
            "label_consistent_1bpm": int(audit["label_consistent_1bpm"].sum()),
            "label_consistent_3bpm": int(audit["label_consistent_3bpm"].sum()),
            "mean_gt_delta_bpm": float(pd.to_numeric(audit["gt_delta_bpm"], errors="coerce").mean()),
            "max_gt_delta_bpm": float(pd.to_numeric(audit["gt_delta_bpm"], errors="coerce").max()),
        },
        "outputs": {
            "pool": str(OUT_POOL),
            "audit": str(OUT_AUDIT),
            "baselines": str(OUT_BASELINES),
            "selector_metrics": str(OUT_SELECTOR_METRICS),
            "selector_predictions": str(OUT_SELECTOR_PREDS),
            "release_gate": str(OUT_RELEASE),
            "bootstrap": str(OUT_BOOTSTRAP),
            "claim_gate": str(OUT_CLAIM_GATE),
            "doc": str(OUT_MD),
        },
    }
    write_json(OUT_SUMMARY, summary)
    write_report(summary, baseline_metrics, selector_metrics, release, boot, gate)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
