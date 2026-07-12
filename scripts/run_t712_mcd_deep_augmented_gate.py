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
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK_ID = "T712"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t704_deep_candidate_physio_gate import (  # noqa: E402
    FEATURE_NUMERIC,
    GraphSelector,
    MLPSelector,
    SetAttentionSelector,
    add_relation_features,
    make_groups,
    metric_row,
    predict_model,
    selector_loss,
)


EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

T704_POOL = EXP / "t704_unified_candidate_pool.csv"
T705_EXTERNAL_POOL = EXP / "t705_external_candidate_pool.csv"
T618_MCD_DEEP_PREDS = EXP / "t618_mcd_cached_clip_predictions.csv"

OUT_POOL = EXP / "t712_mcd_deep_augmented_candidate_pool.csv"
OUT_PREDS = EXP / "t712_mcd_deep_augmented_selector_predictions.csv"
OUT_METRICS = EXP / "t712_mcd_deep_augmented_selector_metrics.csv"
OUT_BASELINES = EXP / "t712_mcd_deep_augmented_baseline_metrics.csv"
OUT_RELEASE = EXP / "t712_mcd_deep_augmented_release_gate.csv"
OUT_CLAIM = EXP / "t712_mcd_deep_augmented_claim_gate.csv"
OUT_FAILURE = EXP / "t712_mcd_deep_augmented_failure_taxonomy.csv"
OUT_SUMMARY = EXP / "t712_mcd_deep_augmented_selector_gate_summary.json"
OUT_MD = DOCS / "t712_mcd_deep_augmented_gate.md"

UNSAFE_BPM = 10.0
SEED = 706


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def feature_frame_domain(pool: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = pool.copy()
    for col in FEATURE_NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat_cols = ["dataset", "source_type", "source_name", "candidate_family", "candidate_model"]
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "unknown"
    cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
    extra = pd.DataFrame(
        {
            "log_n_candidates": np.log1p(df.groupby("sample_id")["sample_id"].transform("count").astype(float)),
            "candidate_hr_centered_by_window": pd.to_numeric(df["candidate_hr_bpm"], errors="coerce")
            - df.groupby("sample_id")["candidate_hr_bpm"].transform("median").astype(float),
        }
    )
    features = pd.concat([df[FEATURE_NUMERIC], extra, cat], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    if feature_cols is None:
        feature_cols = list(features.columns)
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0
    return features[feature_cols].astype(float), feature_cols


def prefilter_candidates(pool: pd.DataFrame, max_candidates_per_window: int) -> pd.DataFrame:
    if max_candidates_per_window <= 0:
        return pool
    parts: list[pd.DataFrame] = []
    for _, g in pool.groupby("sample_id", sort=False):
        if len(g) <= max_candidates_per_window:
            parts.append(g)
            continue
        h = g.copy()
        deep = h[h["source_type"].astype(str) == "deep"]
        rest = h[h["source_type"].astype(str) != "deep"].copy()
        remaining = max(0, max_candidates_per_window - len(deep))
        if remaining > 0 and not rest.empty:
            snr = pd.to_numeric(rest["mean_snr_proxy_db"], errors="coerce").fillna(pd.to_numeric(rest["mean_snr_proxy_db"], errors="coerce").median())
            support = pd.to_numeric(rest["support_count"], errors="coerce").fillna(0.0)
            rank_score = pd.to_numeric(rest["rank_score"], errors="coerce").fillna(0.0)
            agreement = pd.to_numeric(rest["agreement10_frac"], errors="coerce").fillna(0.0)
            harmonic = pd.to_numeric(rest["harmonic_risk"], errors="coerce").fillna(0.0)
            # Non-leaking prefilter: rank by signal quality, support and agreement;
            # never use ground truth or candidate error.
            rest["_prefilter_score"] = (
                snr.rank(pct=True)
                + support.rank(pct=True)
                + rank_score.rank(pct=True)
                + agreement.rank(pct=True)
                - 0.5 * harmonic.rank(pct=True)
            )
            rest["_hr_bin_5bpm"] = (pd.to_numeric(rest["candidate_hr_bpm"], errors="coerce") // 5.0).astype("Int64")
            diverse = (
                rest.sort_values("_prefilter_score", ascending=False)
                .groupby("_hr_bin_5bpm", dropna=True)
                .head(1)
                .sort_values("_prefilter_score", ascending=False)
                .head(remaining)
            )
            if len(diverse) < remaining:
                fill = rest[~rest.index.isin(diverse.index)].sort_values("_prefilter_score", ascending=False).head(remaining - len(diverse))
                diverse = pd.concat([diverse, fill], ignore_index=False, sort=False)
            rest = diverse.drop(columns=["_prefilter_score", "_hr_bin_5bpm"])
        else:
            rest = rest.head(0)
        parts.append(pd.concat([deep, rest], ignore_index=True, sort=False))
    return pd.concat(parts, ignore_index=True, sort=False)


def mcd_deep_candidates(existing_pool: pd.DataFrame) -> pd.DataFrame:
    if not T618_MCD_DEEP_PREDS.exists():
        return pd.DataFrame()
    preds = pd.read_csv(T618_MCD_DEEP_PREDS)
    if preds.empty or "clip_id" not in preds.columns:
        return pd.DataFrame()
    wanted = set(existing_pool.loc[existing_pool["dataset"].astype(str) == "MCD-rPPG", "sample_id"].astype(str))
    preds = preds[preds["clip_id"].astype(str).isin(wanted)].copy()
    if preds.empty:
        return pd.DataFrame()
    rows = pd.DataFrame(
        {
            "sample_id": preds["clip_id"].astype(str),
            "dataset": "MCD-rPPG",
            "subject_std": preds["subject"].astype(str),
            "gt_hr_bpm": pd.to_numeric(preds["hr_true"], errors="coerce"),
            "reference_hr_bpm": pd.to_numeric(preds["hr_true"], errors="coerce"),
            "candidate_id": preds["clip_id"].astype(str) + "_deep_T618_MCD_E2E",
            "candidate_hr_bpm": pd.to_numeric(preds["hr_pred"], errors="coerce"),
            "source_type": "deep",
            "source_name": "T618_MCD_E2E",
            "candidate_family": "deep_backbone",
            "candidate_model": "t618_mcd_cached_small_video_regressor",
            "deep_snr": pd.to_numeric(preds.get("fft_snr_db", np.nan), errors="coerce"),
            "mean_snr_proxy_db": pd.to_numeric(preds.get("fft_snr_db", np.nan), errors="coerce"),
        }
    )
    rows["candidate_abs_error"] = (rows["candidate_hr_bpm"] - rows["gt_hr_bpm"]).abs()
    rows["unsafe_candidate"] = rows["candidate_abs_error"] > UNSAFE_BPM
    defaults = {
        "support_count": 1.0,
        "full_support_count": 1.0,
        "subwindow_support_count": 1.0,
        "top1_support_count": 1.0,
        "full_top1_support_count": 1.0,
        "pos_chrom_count": 0.0,
        "green_pbv_count": 0.0,
        "ica_lgi_count": 0.0,
        "mean_power_fraction": 0.0,
        "max_power_fraction": 0.0,
        "sum_power_fraction": 0.0,
        "rank_score": 0.0,
        "adult_plausibility": 1.0,
        "upper_alt_support": 0.0,
        "upper_alt_pos_chrom": 0.0,
        "upper_phys_support": 0.0,
        "upper_phys_pos_chrom": 0.0,
        "lower_phys_support": 0.0,
        "lower_phys_pos_chrom": 0.0,
        "double_harmonic_support": 0.0,
        "half_harmonic_support": 0.0,
        "t150_confidence": 0.0,
        "t157_low_alias_penalty": 0.0,
        "t157_motion_band_penalty": 0.0,
        "t157_high_alias_penalty": 0.0,
        "t157_near_t150_boost": 0.0,
        "t157_score": 0.0,
        "deep_macc": np.nan,
    }
    for col, value in defaults.items():
        rows[col] = value
    return rows


def load_combined_pool(max_candidates_per_window: int) -> pd.DataFrame:
    if not T704_POOL.exists():
        raise FileNotFoundError(T704_POOL)
    if not T705_EXTERNAL_POOL.exists():
        raise FileNotFoundError(T705_EXTERNAL_POOL)
    ubfc = pd.read_csv(T704_POOL)
    ext = pd.read_csv(T705_EXTERNAL_POOL)
    keep = sorted(set(ubfc.columns).union(ext.columns))
    for df in (ubfc, ext):
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
    pool = pd.concat([ubfc[keep], ext[keep]], ignore_index=True, sort=False)
    pool["dataset"] = pool["dataset"].astype(str)
    pool.loc[pool["dataset"].isin({"nan", "None", ""}), "dataset"] = "UBFC-rPPG"
    deep_mcd = mcd_deep_candidates(pool)
    if not deep_mcd.empty:
        for col in sorted(set(pool.columns).union(deep_mcd.columns)):
            if col not in pool.columns:
                pool[col] = np.nan
            if col not in deep_mcd.columns:
                deep_mcd[col] = np.nan
        pool = pd.concat([pool, deep_mcd[pool.columns]], ignore_index=True, sort=False)
    pool["candidate_abs_error"] = (pd.to_numeric(pool["candidate_hr_bpm"], errors="coerce") - pd.to_numeric(pool["gt_hr_bpm"], errors="coerce")).abs()
    pool["unsafe_candidate"] = pool["candidate_abs_error"] > UNSAFE_BPM
    pool = add_relation_features(pool)
    before = len(pool)
    pool = prefilter_candidates(pool, max_candidates_per_window)
    pool["prefilter_max_candidates_per_window"] = max_candidates_per_window
    pool["prefilter_rows_before"] = before
    pool["prefilter_rows_after"] = len(pool)
    pool.to_csv(OUT_POOL, index=False, encoding="utf-8")
    return pool


def stratified_domain_folds(pool: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[set[str], set[str]]]:
    rng = np.random.default_rng(seed)
    fold_train: list[set[str]] = [set() for _ in range(n_splits)]
    fold_test: list[set[str]] = [set() for _ in range(n_splits)]
    all_groups = set()
    key_df = pool[["dataset", "subject_std"]].copy()
    key_df["dataset"] = key_df["dataset"].astype(str)
    key_df["subject_std"] = key_df["subject_std"].astype(str)
    key_df["group_key"] = key_df["dataset"].astype(str) + "::" + key_df["subject_std"].astype(str)
    key_df = key_df[["dataset", "group_key"]].drop_duplicates()
    for dataset, g in key_df.groupby("dataset"):
        keys = g["group_key"].astype(str).tolist()
        rng.shuffle(keys)
        chunks = [list(x) for x in np.array_split(np.asarray(keys, dtype=object), n_splits)]
        for i, chunk in enumerate(chunks):
            fold_test[i].update(str(x) for x in chunk)
        all_groups.update(keys)
    for i in range(n_splits):
        fold_train[i] = all_groups - fold_test[i]
    return list(zip(fold_train, fold_test))


def baseline_metrics(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for sample_id, g in pool.groupby("sample_id", sort=False):
        gt = float(g["gt_hr_bpm"].iloc[0])
        dataset = str(g["dataset"].iloc[0])
        top = g.loc[pd.to_numeric(g["mean_snr_proxy_db"], errors="coerce").fillna(-999).idxmax()]
        oracle = g.loc[pd.to_numeric(g["candidate_abs_error"], errors="coerce").idxmin()]
        for method, row in [("max_snr", top), ("oracle", oracle)]:
            pred = float(row["candidate_hr_bpm"])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "sample_id": sample_id,
                    "subject_std": row["subject_std"],
                    "gt_hr_bpm": gt,
                    "pred_hr_bpm": pred,
                    "abs_error_bpm": abs(pred - gt),
                }
            )
    preds = pd.DataFrame(rows)
    metrics = pd.DataFrame([{"dataset": ds, **metric_row(method, g)} for (ds, method), g in preds.groupby(["dataset", "method"])])
    return preds, metrics.sort_values(["dataset", "mae_bpm"])


def dataset_weights(groups: list[Any]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for group in groups:
        dataset = str(group.meta["dataset"].iloc[0])
        counts[dataset] = counts.get(dataset, 0) + 1
    if not counts:
        return {}
    total = sum(counts.values())
    n_domains = len(counts)
    return {dataset: float(total / (n_domains * count)) for dataset, count in counts.items()}


def train_one_model_domain_balanced(
    name: str,
    model: nn.Module,
    train_groups: list[Any],
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    weights = dataset_weights(train_groups)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        random.shuffle(train_groups)
        total = 0.0
        total_w = 0.0
        for group in train_groups:
            dataset = str(group.meta["dataset"].iloc[0])
            w = float(weights.get(dataset, 1.0))
            opt.zero_grad(set_to_none=True)
            scores = model(group)
            loss = selector_loss(scores, group) * w
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu())
            total_w += w
        if epoch in {0, epochs - 1}:
            print(f"[{TASK_ID}] {name} epoch={epoch + 1}/{epochs} weighted_loss={total / max(1e-9, total_w):.4f}", flush=True)
    return model


RISK_NUMERIC = [
    "max_probability",
    "entropy",
    "n_candidates",
    "pred_hr_bpm",
    "selected_harmonic_risk",
    "selected_agreement10_frac",
]
RISK_CATEGORICAL = [
    "dataset",
    "selected_source_type",
    "selected_source_name",
    "selected_candidate_family",
]


def heuristic_risk(pred: pd.DataFrame) -> pd.Series:
    n = pd.to_numeric(pred["n_candidates"], errors="coerce").clip(lower=2)
    risk = (
        (1.0 - pd.to_numeric(pred["max_probability"], errors="coerce").fillna(0.0))
        + pd.to_numeric(pred["entropy"], errors="coerce").fillna(0.0) / np.log(n)
        + 0.20 * pd.to_numeric(pred["selected_harmonic_risk"], errors="coerce").fillna(0.0)
        - 0.25 * pd.to_numeric(pred["selected_agreement10_frac"], errors="coerce").fillna(0.0)
    )
    return risk.astype(float)


def add_calibrated_risk(train_pred: pd.DataFrame, test_pred: pd.DataFrame) -> pd.DataFrame:
    train = train_pred.copy()
    test = test_pred.copy()
    y = (pd.to_numeric(train["abs_error_bpm"], errors="coerce") > UNSAFE_BPM).astype(int)
    for col in RISK_NUMERIC:
        if col not in train.columns:
            train[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan
    for col in RISK_CATEGORICAL:
        if col not in train.columns:
            train[col] = "unknown"
        if col not in test.columns:
            test[col] = "unknown"
    if y.nunique() < 2 or len(train) < 20:
        test["calibrated_unsafe_risk"] = heuristic_risk(test)
        test["risk_calibrator"] = "heuristic_fallback"
        return test
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), RISK_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), RISK_CATEGORICAL),
        ]
    )
    clf = Pipeline(
        [
            ("pre", pre),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    try:
        clf.fit(train[RISK_NUMERIC + RISK_CATEGORICAL], y)
        test["calibrated_unsafe_risk"] = clf.predict_proba(test[RISK_NUMERIC + RISK_CATEGORICAL])[:, 1]
        test["risk_calibrator"] = "logistic_fold_calibrator"
    except Exception:
        test["calibrated_unsafe_risk"] = heuristic_risk(test)
        test["risk_calibrator"] = "heuristic_exception_fallback"
    return test


def run_cv(pool: pd.DataFrame, *, epochs: int, lr: float, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, feature_cols = feature_frame_domain(pool)
    pool2 = pool.copy().reset_index(drop=True)
    pool2["group_key"] = pool2["dataset"].astype(str) + "::" + pool2["subject_std"].astype(str)
    folds = stratified_domain_folds(pool2, n_splits, seed)
    constructors = {
        "mlp_relation_domain": lambda: MLPSelector(len(feature_cols)),
        "set_attention_domain": lambda: SetAttentionSelector(len(feature_cols)),
        "graph_selector_domain": lambda: GraphSelector(len(feature_cols)),
    }
    all_preds: list[pd.DataFrame] = []
    print(f"[{TASK_ID}] device={device} feature_dim={len(feature_cols)} folds={len(folds)}", flush=True)
    for fold_idx, (train_groups_keys, test_groups_keys) in enumerate(folds, start=1):
        train_mask = pool2["group_key"].isin(train_groups_keys)
        test_mask = pool2["group_key"].isin(test_groups_keys)
        train_pool = pool2[train_mask].reset_index(drop=True)
        test_pool = pool2[test_mask].reset_index(drop=True)
        train_features = features.loc[train_mask].reset_index(drop=True)
        test_features = features.loc[test_mask].reset_index(drop=True)
        _, std = make_groups(train_pool, train_features, feature_cols, device)
        train_batches, _ = make_groups(train_pool, train_features, feature_cols, device, standardize=std)
        test_batches, _ = make_groups(test_pool, test_features, feature_cols, device, standardize=std)
        print(f"[{TASK_ID}] fold={fold_idx}/{len(folds)} train_windows={len(train_batches)} test_windows={len(test_batches)}", flush=True)
        sample_to_ds = test_pool.drop_duplicates("sample_id").set_index("sample_id")["dataset"].to_dict()
        for name, ctor in constructors.items():
            model = ctor().to(device)
            train_one_model_domain_balanced(name, model, train_batches, epochs=epochs, lr=lr)
            train_pred = predict_model(name, model, train_batches)
            train_pred["dataset"] = train_pred["sample_id"].map(train_pool.drop_duplicates("sample_id").set_index("sample_id")["dataset"].to_dict())
            pred = predict_model(name, model, test_batches)
            pred["dataset"] = pred["sample_id"].map(sample_to_ds)
            pred = add_calibrated_risk(train_pred, pred)
            pred["fold"] = fold_idx
            all_preds.append(pred)
    preds = pd.concat(all_preds, ignore_index=True, sort=False)
    metrics = pd.DataFrame([{"dataset": ds, **metric_row(selector, g.rename(columns={"selector": "method"}))} for (ds, selector), g in preds.groupby(["dataset", "selector"])])
    return preds, metrics.sort_values(["dataset", "mae_bpm"])


def release_gate(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, selector), g in preds.groupby(["dataset", "selector"], sort=False):
        h = g.copy()
        if "calibrated_unsafe_risk" in h.columns:
            h["risk_score"] = pd.to_numeric(h["calibrated_unsafe_risk"], errors="coerce").fillna(1.0)
        else:
            h["risk_score"] = heuristic_risk(h)
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["risk_score"].quantile(q))
            released = h[h["risk_score"] <= tau]
            if released.empty:
                continue
            err = pd.to_numeric(released["abs_error_bpm"], errors="coerce")
            rows.append(
                {
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


def claim_gate(metrics: pd.DataFrame, base: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, m in metrics.iterrows():
        dataset = str(m["dataset"])
        selector = str(m["method"])
        b = base[(base["dataset"].astype(str) == dataset) & (base["method"] == "max_snr")]
        base_mae = float(b["mae_bpm"].iloc[0]) if not b.empty else math.nan
        reduction = 1.0 - float(m["mae_bpm"]) / base_mae if math.isfinite(base_mae) and base_mae > 0 else math.nan
        safe = gate[(gate["dataset"].astype(str) == dataset) & (gate["selector"].astype(str) == selector) & (gate["gate_pass_unsafe10"])]
        best_safe = safe.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        best_coverage = float(best_safe["coverage"].iloc[0]) if not best_safe.empty else 0.0
        best_unsafe = float(best_safe["unsafe_release_rate"].iloc[0]) if not best_safe.empty else math.nan
        best_released_mae = float(best_safe["released_mae_bpm"].iloc[0]) if not best_safe.empty else math.nan
        released_reduction = 1.0 - best_released_mae / base_mae if math.isfinite(best_released_mae) and math.isfinite(base_mae) and base_mae > 0 else math.nan
        rows.append(
            {
                "dataset": dataset,
                "selector": selector,
                "mae_bpm": float(m["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(m["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_max_snr": reduction,
                "best_safe_gate_coverage": best_coverage,
                "best_safe_gate_unsafe": best_unsafe,
                "best_safe_gate_released_mae_bpm": best_released_mae,
                "released_mae_reduction_vs_max_snr": released_reduction,
                "pass_dataset_gate": bool(math.isfinite(released_reduction) and released_reduction >= 0.20 and best_coverage >= 0.40 and best_unsafe <= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mae_bpm"])


def failure_taxonomy(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, selector), g in preds.groupby(["dataset", "selector"]):
        unsafe = g[pd.to_numeric(g["abs_error_bpm"], errors="coerce") > UNSAFE_BPM]
        rows.append(
            {
                "dataset": dataset,
                "selector": selector,
                "n_windows": int(len(g)),
                "n_unsafe": int(len(unsafe)),
                "unsafe_rate": float(len(unsafe) / max(1, len(g))),
                "mean_unsafe_error_bpm": float(unsafe["abs_error_bpm"].mean()) if not unsafe.empty else 0.0,
                "top_selected_source": str(g["selected_source_name"].mode().iloc[0]) if not g.empty else "",
                "top_unsafe_source": str(unsafe["selected_source_name"].mode().iloc[0]) if not unsafe.empty else "",
                "low_conf_unsafe_frac": float((pd.to_numeric(unsafe["max_probability"], errors="coerce") < 0.5).mean()) if not unsafe.empty else 0.0,
                "high_conf_unsafe_frac": float((pd.to_numeric(unsafe["max_probability"], errors="coerce") >= 0.8).mean()) if not unsafe.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], base: pd.DataFrame, metrics: pd.DataFrame, claim: pd.DataFrame, fail: pd.DataFrame) -> None:
    lines = [
        "# T712 MCD Deep-Augmented Protocol-Fixed Selector and Gate",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T705 showed that a UBFC-only selector does not zero-shot transfer to external candidate distributions. T711 fixed the split protocol but still failed on external domains because the external candidate pool was mostly classical. T712 adds available MCD raw-video deep predictions as additional candidates, then re-tests the protocol-fixed domain-balanced selector and calibrated release gate.",
        "",
        "## Baselines",
        "",
        markdown_table(base),
        "",
        "## Selector Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        markdown_table(claim),
        "",
        "## Failure Taxonomy",
        "",
        markdown_table(fail),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "This supports a candidate arbitration and release/review claim only when the dataset-level release gate reaches the pre-specified unsafe-release and coverage thresholds. It does not prove universal zero-shot robustness.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-candidates-per-window", type=int, default=64)
    args = parser.parse_args()
    set_seed(args.seed)

    pool = load_combined_pool(args.max_candidates_per_window)
    _, base = baseline_metrics(pool)
    preds, metrics = run_cv(pool, epochs=args.epochs, lr=args.lr, n_splits=args.n_splits, seed=args.seed)
    gate = release_gate(preds)
    claim = claim_gate(metrics, base, gate)
    fail = failure_taxonomy(preds)

    preds.to_csv(OUT_PREDS, index=False, encoding="utf-8")
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8")
    base.to_csv(OUT_BASELINES, index=False, encoding="utf-8")
    gate.to_csv(OUT_RELEASE, index=False, encoding="utf-8")
    claim.to_csv(OUT_CLAIM, index=False, encoding="utf-8")
    fail.to_csv(OUT_FAILURE, index=False, encoding="utf-8")

    pass_count = int(claim["pass_dataset_gate"].sum())
    n_dataset_selector = int(len(claim))
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "decision": "mcd_deep_augmented_gate_supported_continue_ablation_and_paper_update" if pass_count >= 4 else "mcd_deep_augmented_gate_partial_requires_more_feature_or_gate_work",
        "pass_count": pass_count,
        "n_dataset_selector": n_dataset_selector,
        "datasets": sorted(pool["dataset"].astype(str).unique()),
        "n_windows": int(pool["sample_id"].nunique()),
        "n_candidates": int(len(pool)),
        "max_candidates_per_window": args.max_candidates_per_window,
        "outputs": {
            "pool": str(OUT_POOL),
            "predictions": str(OUT_PREDS),
            "metrics": str(OUT_METRICS),
            "baselines": str(OUT_BASELINES),
            "release": str(OUT_RELEASE),
            "claim": str(OUT_CLAIM),
            "failure": str(OUT_FAILURE),
            "doc": str(OUT_MD),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, base, metrics, claim, fail)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
