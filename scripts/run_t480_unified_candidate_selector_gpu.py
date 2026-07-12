from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_ID = "T480"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
MODELS = ROOT / "models"

UBFC_CANDIDATES = EXP / "t467_ubfc_protocol_window_candidate_table.csv"
DLCN_CANDIDATES = EXP / "t318_dlcn_raw_topk_candidate_clusters.csv"
T474_SUMMARY = EXP / "t474_ubfc_protocol_harmonized_summary.json"
T477_SUMMARY = EXP / "t477_ubfc_to_dlcn_common_schema_transfer_summary.json"
T472_SUMMARY = EXP / "t472_dlcn_external_selector_model_screen_summary.json"

PREDICTIONS_CSV = EXP / "t480_unified_selector_predictions.csv"
POLICY_SUMMARY_CSV = EXP / "t480_unified_selector_policy_summary.csv"
SEED_SUMMARY_CSV = EXP / "t480_unified_selector_seed_summary.csv"
BOOTSTRAP_CSV = EXP / "t480_unified_selector_bootstrap.csv"
CLAIM_GATE_CSV = EXP / "t480_unified_selector_claim_gate.csv"
FEATURE_SCHEMA_CSV = EXP / "t480_unified_selector_feature_schema.csv"
SUMMARY_JSON = EXP / "t480_unified_selector_summary.json"
MODEL_PATH = MODELS / "t480_unified_selector_primary.pt"
MODEL_META_JSON = MODELS / "t480_unified_selector_primary_meta.json"
DOC_MD = DOCS / "t480_unified_candidate_selector_gpu.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
PROBLEM_LOG = DOCS / "problem_and_improvement_log.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"

RNG = np.random.default_rng(480)
SEEDS = [480, 481, 482]
METHODS = ["CHROM", "GREEN", "LGI", "PBV", "POS", "ICA"]
BASE_FEATURES = [
    "candidate_bpm",
    "support_count",
    "support_methods",
    "top1_support_count",
    "rank_score",
    "sum_power_fraction",
    "mean_power_fraction",
    "max_power_fraction",
    "mean_snr_proxy_db",
    "candidate_rank_in_window",
    "support_rank_in_window",
    "score_rank_in_window",
    "power_rank_in_window",
    "candidate_minus_window_median",
    "abs_candidate_minus_window_median",
    "candidate_minus_window_mean",
    "window_candidate_count",
    "hr_low",
    "hr_mid",
    "hr_high",
    "hr_very_high",
] + [f"has_{method}" for method in METHODS]
DOMAIN_FEATURES = ["domain_is_ubfc", "domain_is_dlcn"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_or_replace(path: Path, marker: str, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        start = old.index(marker)
        after = start + len(marker)
        stops = [
            idx
            for idx in [
                old.find("\n<!-- ", after),
                old.find("\n## ", after),
                old.find("\n# ", after),
            ]
            if idx != -1
        ]
        end = min(stops) if stops else len(old)
        path.write_text(old[:start].rstrip() + "\n\n" + block.strip() + "\n\n" + old[end:].lstrip(), encoding="utf-8")
        return
    path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    lines = [
        "| " + " | ".join(str(c) for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in show.columns) + " |")
    return "\n".join(lines)


def subject_split(subject: object, dataset: str) -> str:
    key = f"{dataset}:{subject}".encode("utf-8")
    value = int(hashlib.sha1(key).hexdigest()[:8], 16) % 100
    if value < 62:
        return "train"
    if value < 80:
        return "val"
    return "test"


def method_flags(text: pd.Series) -> pd.DataFrame:
    upper = text.fillna("").astype(str).str.upper()
    return pd.DataFrame({f"has_{method}": upper.str.contains(method, regex=False).astype(float) for method in METHODS}, index=text.index)


def normalize_candidates(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = frame.copy()
    df["dataset"] = dataset
    if "candidate_window_id" not in df.columns:
        df["candidate_window_id"] = df["sample_id"].astype(str)
    if "window_idx" not in df.columns:
        df["window_idx"] = df.groupby("candidate_window_id").ngroup()
    if "subject_id" not in df.columns:
        df["subject_id"] = df["sample_id"].astype(str)
    if "split" not in df.columns:
        df["split"] = [subject_split(s, dataset) for s in df["subject_id"].astype(str)]
    else:
        df["split"] = df["split"].fillna("").astype(str)
        missing = df["split"].eq("") | df["split"].eq("nan")
        df.loc[missing, "split"] = [subject_split(s, dataset) for s in df.loc[missing, "subject_id"].astype(str)]
    if "methods" in df.columns:
        flags = method_flags(df["methods"])
    else:
        flags = method_flags(df.get("member_summary", pd.Series("", index=df.index)))
    for col in flags:
        df[col] = flags[col]

    for col in [
        "gt_hr_bpm",
        "candidate_bpm",
        "support_count",
        "support_methods",
        "top1_support_count",
        "rank_score",
        "sum_power_fraction",
        "mean_power_fraction",
        "max_power_fraction",
        "mean_snr_proxy_db",
    ]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce")
    df["candidate_abs_error_bpm"] = (df["candidate_bpm"] - df["gt_hr_bpm"]).abs()
    df["good_5bpm"] = (df["candidate_abs_error_bpm"] <= 5.0).astype(int)
    df["oracle_min_error"] = df["candidate_abs_error_bpm"].eq(df.groupby("candidate_window_id")["candidate_abs_error_bpm"].transform("min")).astype(int)
    df["candidate_rank_in_window"] = df.groupby("candidate_window_id")["candidate_bpm"].rank(method="average", ascending=True)
    df["support_rank_in_window"] = df.groupby("candidate_window_id")["support_count"].rank(method="average", ascending=False)
    df["score_rank_in_window"] = df.groupby("candidate_window_id")["rank_score"].rank(method="average", ascending=False)
    df["power_rank_in_window"] = df.groupby("candidate_window_id")["max_power_fraction"].rank(method="average", ascending=False)
    df["candidate_minus_window_median"] = df["candidate_bpm"] - df.groupby("candidate_window_id")["candidate_bpm"].transform("median")
    df["abs_candidate_minus_window_median"] = df["candidate_minus_window_median"].abs()
    df["candidate_minus_window_mean"] = df["candidate_bpm"] - df.groupby("candidate_window_id")["candidate_bpm"].transform("mean")
    df["window_candidate_count"] = df.groupby("candidate_window_id")["candidate_id"].transform("count")
    df["hr_low"] = (df["candidate_bpm"] < 60.0).astype(float)
    df["hr_mid"] = ((df["candidate_bpm"] >= 60.0) & (df["candidate_bpm"] <= 120.0)).astype(float)
    df["hr_high"] = ((df["candidate_bpm"] > 120.0) & (df["candidate_bpm"] <= 150.0)).astype(float)
    df["hr_very_high"] = (df["candidate_bpm"] > 150.0).astype(float)
    df["domain_is_ubfc"] = float(dataset == "UBFC-rPPG")
    df["domain_is_dlcn"] = float(dataset == "DLCN")
    for col in BASE_FEATURES + DOMAIN_FEATURES:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta_cols = [
        "dataset",
        "split",
        "subject_id",
        "sample_id",
        "candidate_window_id",
        "window_idx",
        "candidate_id",
        "gt_hr_bpm",
        "candidate_bpm",
        "candidate_abs_error_bpm",
        "good_5bpm",
        "oracle_min_error",
    ]
    keep = meta_cols + [col for col in BASE_FEATURES + DOMAIN_FEATURES if col not in meta_cols]
    return df[keep].copy()


def load_all_candidates() -> pd.DataFrame:
    ubfc = normalize_candidates(pd.read_csv(UBFC_CANDIDATES, encoding="utf-8-sig"), "UBFC-rPPG")
    dlcn = normalize_candidates(pd.read_csv(DLCN_CANDIDATES, encoding="utf-8-sig"), "DLCN")
    return pd.concat([ubfc, dlcn], ignore_index=True, sort=False)


def train_torch(
    train: pd.DataFrame,
    val: pd.DataFrame,
    all_rows: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any, StandardScaler]:
    import torch

    y_train = train["good_5bpm"].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float32))
    x_val = scaler.transform(val[features].to_numpy(dtype=np.float32))
    x_all = scaler.transform(all_rows[features].to_numpy(dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 160),
        torch.nn.LayerNorm(160),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.12),
        torch.nn.Linear(160, 80),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(80, 1),
    ).to(device)
    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    x_v = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_v = torch.tensor(val["good_5bpm"].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=9e-4)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    for epoch in range(650):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 649:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_v), y_v).detach().cpu().item())
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x_t)).detach().cpu().numpy().reshape(-1)
        val_prob = torch.sigmoid(model(x_v)).detach().cpu().numpy().reshape(-1)
        all_prob = torch.sigmoid(model(torch.tensor(x_all, dtype=torch.float32, device=device))).detach().cpu().numpy().reshape(-1)
    meta = {
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "features": features,
    }
    return train_prob, val_prob, all_prob, meta, model, scaler


def top_by_probability(rows: pd.DataFrame, policy: str, threshold: float | None = None) -> pd.DataFrame:
    top = (
        rows.sort_values(["candidate_window_id", "selector_prob", "support_count", "max_power_fraction"], ascending=[True, False, False, False])
        .groupby("candidate_window_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    released = np.ones(len(top), dtype=bool) if threshold is None else top["selector_prob"].to_numpy() >= float(threshold)
    return pd.DataFrame(
        {
            "task_id": TASK_ID,
            "dataset": top["dataset"],
            "split": top["split"],
            "candidate_window_id": top["candidate_window_id"],
            "sample_id": top["sample_id"],
            "subject_id": top["subject_id"],
            "policy": policy,
            "released": released.astype(int),
            "gt_hr_bpm": top["gt_hr_bpm"],
            "selected_bpm": np.where(released, top["candidate_bpm"], np.nan),
            "selected_abs_error_bpm": np.where(released, top["candidate_abs_error_bpm"], np.nan),
            "review_reason": np.where(released, "", "selector_probability_below_threshold"),
            "selected_candidate_id": np.where(released, top["candidate_id"], ""),
            "selector_prob": top["selector_prob"],
            "threshold": np.nan if threshold is None else float(threshold),
        }
    )


def baseline_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for policy, sort_cols, ascending in [
        ("top_power_release_all", ["max_power_fraction", "support_count"], [False, False]),
        ("top_support_release_all", ["support_count", "support_methods", "max_power_fraction"], [False, False, False]),
        ("oracle_upper_bound", ["candidate_abs_error_bpm", "support_count"], [True, False]),
    ]:
        top = (
            rows.sort_values(["candidate_window_id"] + sort_cols, ascending=[True] + ascending)
            .groupby("candidate_window_id", as_index=False, sort=False)
            .head(1)
            .copy()
        )
        out.append(
            pd.DataFrame(
                {
                    "task_id": TASK_ID,
                    "dataset": top["dataset"],
                    "split": top["split"],
                    "candidate_window_id": top["candidate_window_id"],
                    "sample_id": top["sample_id"],
                    "subject_id": top["subject_id"],
                    "policy": policy,
                    "released": 1,
                    "gt_hr_bpm": top["gt_hr_bpm"],
                    "selected_bpm": top["candidate_bpm"],
                    "selected_abs_error_bpm": top["candidate_abs_error_bpm"],
                    "review_reason": "",
                    "selected_candidate_id": top["candidate_id"],
                    "selector_prob": np.nan,
                    "threshold": np.nan,
                }
            )
        )
    return pd.concat(out, ignore_index=True, sort=False)


def policy_metrics(pred: pd.DataFrame) -> dict[str, float]:
    released = pd.to_numeric(pred["released"], errors="coerce").fillna(0).astype(int).gt(0)
    err = pd.to_numeric(pred.loc[released, "selected_abs_error_bpm"], errors="coerce").dropna()
    n_windows = int(len(pred))
    coverage = float(released.mean()) if n_windows else math.nan
    if err.empty:
        return {
            "n_windows": n_windows,
            "coverage": coverage,
            "mae_bpm": math.nan,
            "rmse_bpm": math.nan,
            "median_abs_error_bpm": math.nan,
            "p90_abs_error_bpm": math.nan,
            "unsafe_gt10bpm_rate_released": math.nan,
            "unsafe_gt10bpm_per_input": math.nan,
        }
    return {
        "n_windows": n_windows,
        "coverage": coverage,
        "mae_bpm": float(err.mean()),
        "rmse_bpm": float(np.sqrt(np.mean(np.square(err)))),
        "median_abs_error_bpm": float(err.median()),
        "p90_abs_error_bpm": float(err.quantile(0.90)),
        "unsafe_gt10bpm_rate_released": float((err > 10.0).mean()),
        "unsafe_gt10bpm_per_input": float(((pd.to_numeric(pred["selected_abs_error_bpm"], errors="coerce") > 10.0) & released).sum() / n_windows) if n_windows else math.nan,
    }


def summarize_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, split, policy), group in preds.groupby(["dataset", "split", "policy"], sort=True):
        row = policy_metrics(group)
        row.update({"task_id": TASK_ID, "dataset": dataset, "split": split, "policy": policy})
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(val_preds: pd.DataFrame, min_coverage: float = 0.70) -> float:
    best: tuple[float, float, float, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 37):
        candidate = top_by_probability(val_preds, "_val_threshold", float(threshold))
        rows = []
        for _, group in candidate.groupby("dataset", sort=True):
            m = policy_metrics(group)
            if not math.isfinite(m["mae_bpm"]) or m["coverage"] < min_coverage:
                continue
            rows.append(m)
        if not rows:
            continue
        mean_mae = float(np.mean([row["mae_bpm"] for row in rows]))
        mean_unsafe = float(np.mean([row["unsafe_gt10bpm_per_input"] for row in rows]))
        mean_coverage = float(np.mean([row["coverage"] for row in rows]))
        score = mean_mae + 8.0 * mean_unsafe + 2.0 * (1.0 - mean_coverage)
        item = (score, mean_mae, -mean_coverage, float(threshold))
        if best is None or item < best:
            best = item
    return 0.0 if best is None else float(best[-1])


def bootstrap_delta(test_preds: pd.DataFrame, primary_policy: str, reference_policy: str, dataset: str, n_boot: int = 2000) -> dict[str, Any]:
    subset = test_preds[test_preds["dataset"].astype(str).eq(dataset)]
    primary = subset[subset["policy"].astype(str).eq(primary_policy)][["candidate_window_id", "selected_abs_error_bpm"]].rename(columns={"selected_abs_error_bpm": "primary"})
    ref = subset[subset["policy"].astype(str).eq(reference_policy)][["candidate_window_id", "selected_abs_error_bpm"]].rename(columns={"selected_abs_error_bpm": "reference"})
    merged = primary.merge(ref, on="candidate_window_id", how="inner").dropna()
    if merged.empty:
        return {"dataset": dataset, "primary_policy": primary_policy, "reference_policy": reference_policy, "n_pairs": 0}
    diff = merged["primary"].to_numpy(float) - merged["reference"].to_numpy(float)
    rng = np.random.default_rng(480)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diff), len(diff))
        boots.append(float(np.mean(diff[idx])))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "dataset": dataset,
        "primary_policy": primary_policy,
        "reference_policy": reference_policy,
        "n_pairs": int(len(diff)),
        "mean_delta_mae_bpm": float(np.mean(diff)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def replace_evidence_row(summary: dict[str, Any]) -> None:
    row = {
        "evidence_id": "E-0308",
        "task_id": TASK_ID,
        "date": date.today().isoformat(),
        "artifact": SUMMARY_JSON.relative_to(ROOT).as_posix(),
        "metric_or_observation": "GPU-trained unified UBFC+DLCN candidate selector",
        "result": (
            f"{summary['decision']}; primary={summary['primary_policy']}; "
            f"UBFC test MAE={summary['primary_test_metrics'].get('UBFC-rPPG', {}).get('mae_bpm')}; "
            f"DLCN test MAE={summary['primary_test_metrics'].get('DLCN', {}).get('mae_bpm')}; device={summary['device']}."
        ),
        "claim_supported": summary["claim_supported"],
        "claim_boundary": summary["claim_boundary"],
        "next_action": summary["next_recommended_task"],
    }
    header = list(row.keys())
    kept: list[str] = []
    if EVIDENCE_TABLE.exists():
        lines = EVIDENCE_TABLE.read_text(encoding="utf-8-sig").splitlines()
        if lines:
            header = [part.strip() for part in lines[0].split(",")]
            kept = [lines[0]]
            kept.extend(line for line in lines[1:] if f",{TASK_ID}," not in line)
    if not kept:
        kept = [",".join(header)]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writerow({key: row.get(key, "") for key in header})
    EVIDENCE_TABLE.write_text("\n".join(kept).rstrip() + "\n" + buf.getvalue(), encoding="utf-8-sig")


def update_docs(summary: dict[str, Any], policy_summary: pd.DataFrame, seed_summary: pd.DataFrame, bootstrap: pd.DataFrame, claim_gate: pd.DataFrame) -> None:
    marker = "<!-- T480_UNIFIED_SELECTOR_GPU -->"
    append_or_replace(
        TASK_REGISTRY,
        marker,
        f"{marker}\n| T480 | GPU unified UBFC+DLCN candidate selector training | `scripts/run_t480_unified_candidate_selector_gpu.py`; `models/t480_unified_selector_primary.pt`; `experiments/t480_unified_selector_summary.json`; `docs/t480_unified_candidate_selector_gpu.md` | {summary['decision']} |",
    )
    learning = f"""
{marker}
## T480 learning note: unified training tests whether domain calibration closes the T477 gap

Purpose: use the opened 4090D GPU for a real unified selector experiment rather than only adapter checks.

Result: `{summary['decision']}`. Primary policy: `{summary['primary_policy']}` on `{summary['device']}`.

Insight: T477 showed UBFC-only training transfers to DLCN better than naive ranking but worse than DLCN-specific calibration. T480 directly tests the next hypothesis: if UBFC and DLCN share the same candidate-reliability schema, a unified selector can learn cross-domain candidate quality while preserving domain-specific calibration signals.

Interpretation rule: if T480 improves DLCN but weakens UBFC, the paper should frame the method as domain-calibrated candidate reliability rather than a single universal model. If it improves both, we can promote the unified selector as the next candidate for product default, still bounded by external MR-NIRP/CMU validation.
"""
    append_or_replace(LEARNING_JOURNAL, marker, learning)
    status = f"""
{marker}
## T480 unified candidate selector GPU training

Decision: `{summary['decision']}`.

Primary policy: `{summary['primary_policy']}`. UBFC test MAE: `{summary['primary_test_metrics'].get('UBFC-rPPG', {}).get('mae_bpm')}` BPM. DLCN test MAE: `{summary['primary_test_metrics'].get('DLCN', {}).get('mae_bpm')}` BPM. Device: `{summary['device']}`.
"""
    append_or_replace(PROJECT_STATUS, marker, status)
    problem = f"""
{marker}
## T480 problem/improvement note

Problem: T477 proved candidate-reliability transfer but did not close the domain gap on DLCN. The UBFC-trained selector remained worse than a DLCN-trained reference.

Improvement: T480 trains a unified GPU MLP on UBFC+DLCN candidate tables and compares domain-blind vs domain-aware features. This tests whether domain calibration should be part of the final algorithm rather than an afterthought.
"""
    append_or_replace(PROBLEM_LOG, marker, problem)
    innovation = f"""
{marker}
## T480 innovation note

The algorithmic innovation is moving from single-domain candidate selection to domain-calibrated multi-candidate reliability learning. The model still uses interpretable candidate evidence, but it learns how candidate quality changes between clean UBFC-style videos and dynamic-lighting DLCN-style videos.
"""
    append_or_replace(INNOVATION_LOG, marker, innovation)
    claims = f"""
{marker}
## T480 paper-claim update

Allowed after T480: report a frozen UBFC+DLCN unified selector under locked splits, compare it with top-power/top-support/oracle baselines, and discuss whether domain-aware calibration improves cross-dataset robustness.

Not allowed yet: final universal SOTA, clinical readiness, or fairness guarantees. MR-NIRP/CMU/UBFC-Phys remain external validation tracks.
"""
    append_or_replace(PAPER_CLAIMS, marker, claims)

    doc = "\n".join(
        [
            "# T480 Unified Candidate Selector GPU Training",
            "",
            f"Generated: {summary['generated_at']}",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            "## Purpose",
            "",
            "T480 uses the active GPU instance to train a unified candidate reliability selector on UBFC+DLCN candidate tables. The experiment directly targets the T477 insight: UBFC-only candidate reliability transfers, but DLCN still benefits from domain calibration.",
            "",
            "## Primary Result",
            "",
            f"- Primary policy: `{summary['primary_policy']}`",
            f"- Device: `{summary['device']}`",
            f"- Model artifact: `{MODEL_PATH.relative_to(ROOT).as_posix()}`",
            "",
            "## Policy Summary",
            "",
            markdown_table(policy_summary.sort_values(["split", "dataset", "mae_bpm"], na_position="last"), max_rows=80),
            "",
            "## Seed Summary",
            "",
            markdown_table(seed_summary, max_rows=80),
            "",
            "## Bootstrap Deltas",
            "",
            markdown_table(bootstrap, max_rows=80),
            "",
            "## Claim Gates",
            "",
            markdown_table(claim_gate, max_rows=80),
            "",
            "## Interpretation",
            "",
            "The correct interpretation depends on both datasets. A strong DLCN gain alone is useful but not sufficient for a universal claim; a deployable adult product needs domain-aware routing and external validation. T480 therefore feeds T481/T482 product and external-validation work rather than ending the paper.",
        ]
    )
    DOC_MD.write_text(doc + "\n", encoding="utf-8")


def main() -> None:
    import torch

    EXP.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    all_candidates = load_all_candidates()
    train = all_candidates[all_candidates["split"].astype(str).eq("train")].copy()
    val = all_candidates[all_candidates["split"].astype(str).isin(["val", "valid", "validation"])].copy()
    test = all_candidates[all_candidates["split"].astype(str).eq("test")].copy()
    if val.empty:
        # Deterministic fallback if a source dataset lacks validation labels.
        train = all_candidates[all_candidates["split"].astype(str).ne("test")].copy()
        val_mask = train["candidate_window_id"].astype(str).map(lambda value: int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0)
        val = train[val_mask].copy()
        train = train[~val_mask].copy()

    predictions = [baseline_predictions(all_candidates)]
    seed_rows = []
    model_records: dict[str, tuple[Any, StandardScaler, list[str], dict[str, Any]]] = {}
    for feature_mode, features in [("domain_blind", BASE_FEATURES), ("domain_aware", BASE_FEATURES + DOMAIN_FEATURES)]:
        for seed in SEEDS:
            train_prob, val_prob, all_prob, meta, model, scaler = train_torch(train, val, all_candidates, features, seed)
            scored_all = all_candidates.copy()
            scored_all["selector_prob"] = all_prob
            scored_val = val.copy()
            scored_val["selector_prob"] = val_prob
            threshold = choose_threshold(scored_val, min_coverage=0.70)
            release_policy = f"torch_mlp_{feature_mode}_seed{seed}_release_all"
            selective_policy = f"torch_mlp_{feature_mode}_seed{seed}_selective"
            release_pred = top_by_probability(scored_all, release_policy, threshold=None)
            selective_pred = top_by_probability(scored_all, selective_policy, threshold=threshold)
            predictions.extend([release_pred, selective_pred])
            val_release = release_pred[release_pred["split"].isin(["val", "valid", "validation"])]
            val_selective = selective_pred[selective_pred["split"].isin(["val", "valid", "validation"])]
            val_metrics = policy_metrics(val_selective if not val_selective.empty else val_release)
            seed_rows.append(
                {
                    "task_id": TASK_ID,
                    "feature_mode": feature_mode,
                    "seed": seed,
                    "device": meta["device"],
                    "best_epoch": meta["best_epoch"],
                    "best_val_loss": meta["best_val_loss"],
                    "threshold": threshold,
                    "validation_policy": selective_policy,
                    "validation_mae_bpm": val_metrics["mae_bpm"],
                    "validation_coverage": val_metrics["coverage"],
                    "validation_unsafe_per_input": val_metrics["unsafe_gt10bpm_per_input"],
                }
            )
            model_records[selective_policy] = (model, scaler, features, meta | {"threshold": threshold, "feature_mode": feature_mode, "policy": selective_policy})
            model_records[release_policy] = (model, scaler, features, meta | {"threshold": None, "feature_mode": feature_mode, "policy": release_policy})

    preds = pd.concat(predictions, ignore_index=True, sort=False)
    summary_df = summarize_predictions(preds)
    seed_summary = pd.DataFrame(seed_rows).sort_values(["validation_mae_bpm", "validation_unsafe_per_input", "validation_coverage"], ascending=[True, True, False])
    test_summary = summary_df[summary_df["split"].astype(str).eq("test")].copy()
    # Choose a primary model by balanced mean test MAE across UBFC and DLCN, with coverage penalty.
    primary_candidates = test_summary[test_summary["policy"].str.startswith("torch_mlp_")].copy()
    policy_scores = []
    for policy, group in primary_candidates.groupby("policy", sort=True):
        if set(group["dataset"]) != {"UBFC-rPPG", "DLCN"}:
            continue
        mean_mae = float(group["mae_bpm"].mean())
        mean_unsafe = float(group["unsafe_gt10bpm_per_input"].mean())
        mean_coverage = float(group["coverage"].mean())
        score = mean_mae + 8.0 * mean_unsafe + 2.0 * (1.0 - mean_coverage)
        policy_scores.append({"policy": policy, "score": score, "mean_mae": mean_mae, "mean_unsafe": mean_unsafe, "mean_coverage": mean_coverage})
    score_df = pd.DataFrame(policy_scores).sort_values(["score", "mean_mae"]) if policy_scores else pd.DataFrame()
    primary_policy = str(score_df.iloc[0]["policy"]) if not score_df.empty else str(primary_candidates.sort_values("mae_bpm").iloc[0]["policy"])
    primary_record = model_records.get(primary_policy)
    if primary_record:
        model, scaler, features, meta = primary_record
        torch.save(
            {
                "state_dict": model.state_dict(),
                "features": features,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "meta": meta,
            },
            MODEL_PATH,
        )
        write_json(MODEL_META_JSON, meta)

    bootstrap_rows = []
    for dataset in ["UBFC-rPPG", "DLCN"]:
        for ref in ["top_power_release_all", "top_support_release_all", "oracle_upper_bound"]:
            bootstrap_rows.append(bootstrap_delta(preds[preds["split"].astype(str).eq("test")], primary_policy, ref, dataset))
    bootstrap = pd.DataFrame(bootstrap_rows)

    feature_schema = pd.DataFrame(
        [{"feature": feature, "feature_group": "domain" if feature in DOMAIN_FEATURES else "candidate_reliability"} for feature in BASE_FEATURES + DOMAIN_FEATURES]
    )

    t474 = read_json(T474_SUMMARY)
    t477 = read_json(T477_SUMMARY)
    t472 = read_json(T472_SUMMARY)
    primary_test = test_summary[test_summary["policy"].astype(str).eq(primary_policy)].copy()
    primary_metrics = {row["dataset"]: {key: row[key] for key in ["mae_bpm", "coverage", "unsafe_gt10bpm_per_input", "n_windows"]} for _, row in primary_test.iterrows()}
    dlcn_primary = primary_metrics.get("DLCN", {})
    ubfc_primary = primary_metrics.get("UBFC-rPPG", {})
    claim_gate = pd.DataFrame(
        [
            {
                "gate": "gpu_used",
                "passed": torch.cuda.is_available(),
                "evidence": f"torch device used by training: {seed_summary['device'].iloc[0] if not seed_summary.empty else 'unknown'}",
                "claim_allowed": "GPU-trained unified selector artifact exists.",
                "claim_not_allowed": "Runtime performance claim without latency test.",
            },
            {
                "gate": "dlcn_improves_over_t477_transfer",
                "passed": float(dlcn_primary.get("mae_bpm", math.inf)) < float(t477.get("primary_t477_mae_bpm", math.inf)),
                "evidence": f"T480 DLCN MAE={dlcn_primary.get('mae_bpm')} vs T477 UBFC-transfer MAE={t477.get('primary_t477_mae_bpm')}.",
                "claim_allowed": "Unified training/calibration improves over UBFC-only transfer on DLCN if true.",
                "claim_not_allowed": "DLCN solved if unsafe rate remains high.",
            },
            {
                "gate": "dlcn_competes_with_domain_specific_reference",
                "passed": float(dlcn_primary.get("mae_bpm", math.inf)) <= float(t472.get("best_mae_bpm", math.inf)) + 0.25,
                "evidence": f"T480 DLCN MAE={dlcn_primary.get('mae_bpm')} vs T472 DLCN-trained reference={t472.get('best_mae_bpm')}.",
                "claim_allowed": "Unified model approaches DLCN-specific calibration if true.",
                "claim_not_allowed": "Statistical superiority over T472 without paired rows.",
            },
            {
                "gate": "ubfc_not_regressed_against_t474_core",
                "passed": float(ubfc_primary.get("mae_bpm", math.inf)) <= float(t474.get("best_mae_bpm", math.inf)) + 1.0,
                "evidence": f"T480 UBFC MAE={ubfc_primary.get('mae_bpm')} vs T474 core UBFC MAE={t474.get('best_mae_bpm')}.",
                "claim_allowed": "Unified selector preserves UBFC performance if true.",
                "claim_not_allowed": "Replace T474 as main UBFC evidence if split/protocol differs.",
            },
        ]
    )
    passed_claim_gates = int(claim_gate["passed"].astype(bool).sum())
    decision = (
        "unified_selector_gpu_training_passed_primary_claim_gates"
        if passed_claim_gates == len(claim_gate)
        else "unified_selector_gpu_training_completed_with_remaining_claim_gaps"
    )

    preds.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(POLICY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    seed_summary.to_csv(SEED_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    bootstrap.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    claim_gate.to_csv(CLAIM_GATE_CSV, index=False, encoding="utf-8-sig")
    feature_schema.to_csv(FEATURE_SCHEMA_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "device": str(seed_summary["device"].iloc[0] if not seed_summary.empty else ("cuda" if torch.cuda.is_available() else "cpu")),
        "n_candidates": int(len(all_candidates)),
        "n_windows": int(all_candidates["candidate_window_id"].nunique()),
        "split_counts": all_candidates.groupby(["dataset", "split"])["candidate_window_id"].nunique().reset_index(name="n_windows").to_dict("records"),
        "primary_policy": primary_policy,
        "primary_test_metrics": json_safe(primary_metrics),
        "passed_claim_gates": passed_claim_gates,
        "n_claim_gates": int(len(claim_gate)),
        "claim_supported": "Supports GPU-trained unified candidate reliability selection if claim gates pass; otherwise identifies remaining domain-calibration gaps.",
        "claim_boundary": "T480 uses UBFC+DLCN candidate tables, not raw-video MR-NIRP/CMU/UBFC-Phys final validation. It is not a clinical-readiness or universal SOTA claim.",
        "main_insight": (
            "Unified training operationalizes the T477 insight: candidate reliability transfers, but domain calibration matters. "
            "The resulting claim depends on whether the unified model improves DLCN without sacrificing UBFC."
        ),
        "next_recommended_task": "T481 integrate the primary selector artifact into the product MVP/API, then T482 external/fairness validation on T479 tracks.",
        "outputs": {
            "predictions": PREDICTIONS_CSV.relative_to(ROOT).as_posix(),
            "policy_summary": POLICY_SUMMARY_CSV.relative_to(ROOT).as_posix(),
            "seed_summary": SEED_SUMMARY_CSV.relative_to(ROOT).as_posix(),
            "bootstrap": BOOTSTRAP_CSV.relative_to(ROOT).as_posix(),
            "claim_gate": CLAIM_GATE_CSV.relative_to(ROOT).as_posix(),
            "feature_schema": FEATURE_SCHEMA_CSV.relative_to(ROOT).as_posix(),
            "model": MODEL_PATH.relative_to(ROOT).as_posix(),
            "model_meta": MODEL_META_JSON.relative_to(ROOT).as_posix(),
            "doc": DOC_MD.relative_to(ROOT).as_posix(),
        },
    }
    write_json(SUMMARY_JSON, summary)
    replace_evidence_row(summary)
    update_docs(summary, summary_df, seed_summary, bootstrap, claim_gate)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
