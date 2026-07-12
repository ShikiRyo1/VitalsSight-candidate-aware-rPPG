from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


TASK_ID = "T730"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_t726_t724_oof_deep_candidate_selector as t726  # noqa: E402


base = t726.base
UNSAFE_BPM = 10.0

base.TASK_ID = TASK_ID
base.OUT_POOL = EXP / "t730_harmonic_aware_candidate_pool.csv"
base.OUT_PREDS = EXP / "t730_harmonic_aware_selector_predictions.csv"
base.OUT_METRICS = EXP / "t730_harmonic_aware_selector_metrics.csv"
base.OUT_BASELINES = EXP / "t730_harmonic_aware_baseline_metrics.csv"
base.OUT_RELEASE = EXP / "t730_harmonic_aware_release_gate.csv"
base.OUT_CLAIM = EXP / "t730_harmonic_aware_claim_gate.csv"
base.OUT_FAILURE = EXP / "t730_harmonic_aware_failure_taxonomy.csv"
base.OUT_SUMMARY = EXP / "t730_harmonic_aware_selector_gate_summary.json"
base.OUT_MD = DOCS / "t730_harmonic_aware_selector.md"


EXTRA_FEATURES = [
    "hr_rank_pct",
    "hr_percentile_center_distance",
    "half_neighbor_count",
    "double_neighbor_count",
    "half_neighbor_support_frac",
    "double_neighbor_support_frac",
    "harmonic_neighbor_count",
    "harmonic_neighbor_frac",
    "snr_agreement_conflict",
    "support_agreement_conflict",
    "alias_band_risk",
    "hr_boundary_risk",
    "deep_median_distance",
    "deep_min_distance",
    "deep_disagreement_risk",
    "harmonic_trap_score",
]

for _col in EXTRA_FEATURES:
    if _col not in base.FEATURE_NUMERIC:
        base.FEATURE_NUMERIC.append(_col)


_original_add_relation_features = base.add_relation_features
_original_predict_model = base.predict_model
_original_failure_taxonomy = base.failure_taxonomy


def _safe_numeric(series: pd.Series, fill: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().any():
        out = out.fillna(float(out.median()))
    return out.fillna(fill)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def harmonic_aware_relation_features(pool: pd.DataFrame) -> pd.DataFrame:
    """Add label-free features for harmonic/alias traps.

    The key failure after T726/T727 was not absence of a usable candidate. The
    oracle candidate often existed, but high-SNR harmonic/alias candidates near
    60/120/130 BPM won. These features expose that situation to the selector
    without using ground truth at inference time.
    """

    out = _original_add_relation_features(pool)
    parts: list[pd.DataFrame] = []
    for _, g in out.groupby("sample_id", sort=False):
        h = g.copy()
        hr = _safe_numeric(h["candidate_hr_bpm"], np.nan).to_numpy(float)
        support = _safe_numeric(h.get("support_count", pd.Series(0.0, index=h.index))).to_numpy(float)
        snr_rank = _safe_numeric(h.get("snr_rank_pct", pd.Series(0.0, index=h.index))).to_numpy(float)
        support_rank = _safe_numeric(h.get("support_rank_pct", pd.Series(0.0, index=h.index))).to_numpy(float)
        agree10 = _safe_numeric(h.get("agreement10_frac", pd.Series(0.0, index=h.index))).to_numpy(float)

        n = max(1, len(h))
        finite = np.isfinite(hr)
        if finite.any():
            ranks = pd.Series(hr).rank(pct=True).to_numpy(float)
            med = float(np.nanmedian(hr))
            deep_hr = pd.to_numeric(h.loc[h["source_type"].astype(str).eq("deep"), "candidate_hr_bpm"], errors="coerce").dropna().to_numpy(float)
            if len(deep_hr):
                deep_med = float(np.median(deep_hr))
            else:
                deep_med = med
        else:
            ranks = np.zeros(len(h), dtype=float)
            med = 0.0
            deep_med = 0.0
            deep_hr = np.asarray([], dtype=float)

        half_counts: list[int] = []
        double_counts: list[int] = []
        half_support: list[float] = []
        double_support: list[float] = []
        deep_min_distance: list[float] = []

        for i, value in enumerate(hr):
            if not math.isfinite(float(value)) or value <= 0:
                half_counts.append(0)
                double_counts.append(0)
                half_support.append(0.0)
                double_support.append(0.0)
                deep_min_distance.append(0.0)
                continue
            half_mask = np.isfinite(hr) & (np.abs(hr - value / 2.0) <= max(3.0, 0.04 * value))
            double_mask = np.isfinite(hr) & (np.abs(hr - value * 2.0) <= max(3.0, 0.08 * value))
            half_mask[i] = False
            double_mask[i] = False
            half_counts.append(int(half_mask.sum()))
            double_counts.append(int(double_mask.sum()))
            half_support.append(float(np.nansum(support[half_mask]) / max(1.0, np.nansum(support))))
            double_support.append(float(np.nansum(support[double_mask]) / max(1.0, np.nansum(support))))
            if len(deep_hr):
                deep_min_distance.append(float(np.nanmin(np.abs(deep_hr - value))))
            else:
                deep_min_distance.append(abs(float(value) - deep_med))

        h["hr_rank_pct"] = ranks
        h["hr_percentile_center_distance"] = np.abs(ranks - 0.5)
        h["half_neighbor_count"] = half_counts
        h["double_neighbor_count"] = double_counts
        h["half_neighbor_support_frac"] = half_support
        h["double_neighbor_support_frac"] = double_support
        h["harmonic_neighbor_count"] = h["half_neighbor_count"] + h["double_neighbor_count"]
        h["harmonic_neighbor_frac"] = h["harmonic_neighbor_count"] / max(1, n - 1)
        h["snr_agreement_conflict"] = np.maximum(0.0, snr_rank - agree10)
        h["support_agreement_conflict"] = np.maximum(0.0, support_rank - agree10)
        # These bands were repeated MCD false-peak regions. The model receives
        # them as soft risk features, not as a hard rejection rule, because 60
        # BPM can also be a true resting HR.
        h["alias_band_risk"] = (
            ((np.abs(hr - 60.0) <= 3.0) | (np.abs(hr - 120.0) <= 5.0) | (np.abs(hr - 130.0) <= 5.0))
            & (agree10 < 0.35)
        ).astype(float)
        h["hr_boundary_risk"] = ((hr < 45.0) | (hr > 180.0)).astype(float)
        h["deep_median_distance"] = np.abs(hr - deep_med)
        h["deep_min_distance"] = deep_min_distance
        h["deep_disagreement_risk"] = np.clip(np.asarray(deep_min_distance, dtype=float) / 40.0, 0.0, 2.0)
        base_harm = _safe_numeric(h.get("harmonic_risk", pd.Series(0.0, index=h.index))).to_numpy(float)
        h["harmonic_trap_score"] = (
            0.35 * np.clip(base_harm / 20.0, 0.0, 2.0)
            + 0.25 * h["snr_agreement_conflict"].to_numpy(float)
            + 0.15 * h["support_agreement_conflict"].to_numpy(float)
            + 0.10 * h["alias_band_risk"].to_numpy(float)
            + 0.10 * h["harmonic_neighbor_frac"].to_numpy(float)
            + 0.05 * h["deep_disagreement_risk"].to_numpy(float)
        )
        parts.append(h)
    return pd.concat(parts, ignore_index=True, sort=False)


def t730_selector_loss(
    scores: torch.Tensor,
    group: Any,
    *,
    temp: float = 2.0,
    err_scale: float = 18.0,
    unsafe_weight: float = 0.85,
    barrier_weight: float = 2.75,
    harmonic_weight: float = 0.18,
) -> torch.Tensor:
    errors = torch.nan_to_num(group.errors, nan=100.0, posinf=100.0, neginf=100.0)
    unsafe_barrier = torch.relu(errors - UNSAFE_BPM)
    adjusted_errors = errors + barrier_weight * unsafe_barrier
    q = F.softmax(-adjusted_errors / temp, dim=0).detach()
    logp = F.log_softmax(scores, dim=0)
    p = F.softmax(scores, dim=0)
    kl = F.kl_div(logp, q, reduction="batchmean")
    expected_error = torch.sum(p * errors) / err_scale
    expected_unsafe = torch.sum(p * torch.sigmoid((errors - UNSAFE_BPM) / 2.0))
    best_idx = torch.argmin(adjusted_errors)
    margin = torch.relu(1.25 - scores[best_idx] + scores).mean()

    meta = group.meta
    trap_np = pd.to_numeric(meta.get("harmonic_trap_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    harmonic_np = pd.to_numeric(meta.get("harmonic_risk", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    agree_np = pd.to_numeric(meta.get("agreement10_frac", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    trap = torch.tensor(trap_np, device=scores.device)
    harmonic = torch.tensor(np.clip(harmonic_np / 20.0, 0.0, 2.0), device=scores.device)
    agreement = torch.tensor(agree_np, device=scores.device)
    harmonic_penalty = torch.sum(p * (trap + 0.35 * harmonic - 0.15 * agreement))

    safe_mask = errors <= UNSAFE_BPM
    unsafe_mask = errors > (UNSAFE_BPM + 10.0)
    if bool(safe_mask.any()) and bool(unsafe_mask.any()):
        best_safe_score = scores[safe_mask].max()
        worst_unsafe_score = scores[unsafe_mask].max()
        unsafe_rank_margin = torch.relu(1.5 - best_safe_score + worst_unsafe_score)
    else:
        unsafe_rank_margin = torch.tensor(0.0, device=scores.device)

    return (
        kl
        + expected_error
        + unsafe_weight * expected_unsafe
        + 0.12 * margin
        + harmonic_weight * harmonic_penalty
        + 0.20 * unsafe_rank_margin
    )


def t730_predict_model(name: str, model: torch.nn.Module, groups: list[Any]) -> pd.DataFrame:
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
            row = {
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
                "selected_harmonic_risk": _safe_float(meta.get("harmonic_risk", 0.0)),
                "selected_agreement10_frac": _safe_float(meta.get("agreement10_frac", 0.0)),
                "selected_harmonic_trap_score": _safe_float(meta.get("harmonic_trap_score", 0.0)),
                "selected_alias_band_risk": _safe_float(meta.get("alias_band_risk", 0.0)),
                "selected_deep_disagreement_risk": _safe_float(meta.get("deep_disagreement_risk", 0.0)),
                "selected_snr_agreement_conflict": _safe_float(meta.get("snr_agreement_conflict", 0.0)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def t730_heuristic_risk(pred: pd.DataFrame) -> pd.Series:
    n = pd.to_numeric(pred["n_candidates"], errors="coerce").clip(lower=2)
    return (
        (1.0 - pd.to_numeric(pred["max_probability"], errors="coerce").fillna(0.0))
        + pd.to_numeric(pred["entropy"], errors="coerce").fillna(0.0) / np.log(n)
        + 0.22 * pd.to_numeric(pred["selected_harmonic_risk"], errors="coerce").fillna(0.0)
        + 0.90 * pd.to_numeric(pred.get("selected_harmonic_trap_score", 0.0), errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(pred.get("selected_alias_band_risk", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(pred.get("selected_deep_disagreement_risk", 0.0), errors="coerce").fillna(0.0)
        - 0.35 * pd.to_numeric(pred["selected_agreement10_frac"], errors="coerce").fillna(0.0)
    ).astype(float)


def t730_failure_taxonomy(preds: pd.DataFrame) -> pd.DataFrame:
    fail = _original_failure_taxonomy(preds)
    extra_rows: list[dict[str, Any]] = []
    for (dataset, selector), g in preds.groupby(["dataset", "selector"], sort=False):
        unsafe = g[pd.to_numeric(g["abs_error_bpm"], errors="coerce") > UNSAFE_BPM]
        extra_rows.append(
            {
                "dataset": dataset,
                "selector": selector,
                "n_windows": int(len(g)),
                "unsafe_rate": float(len(unsafe) / max(1, len(g))),
                "mean_selected_harmonic_trap_score": float(pd.to_numeric(g.get("selected_harmonic_trap_score", 0.0), errors="coerce").fillna(0.0).mean()),
                "unsafe_alias_band_frac": float(pd.to_numeric(unsafe.get("selected_alias_band_risk", 0.0), errors="coerce").fillna(0.0).mean()) if not unsafe.empty else 0.0,
                "unsafe_deep_disagreement_mean": float(pd.to_numeric(unsafe.get("selected_deep_disagreement_risk", 0.0), errors="coerce").fillna(0.0).mean()) if not unsafe.empty else 0.0,
            }
        )
    return fail.merge(pd.DataFrame(extra_rows), on=["dataset", "selector", "n_windows", "unsafe_rate"], how="left")


def write_report(summary: dict[str, Any], base_metrics: pd.DataFrame, metrics: pd.DataFrame, claim: pd.DataFrame, fail: pd.DataFrame) -> None:
    lines = [
        "# T730 Harmonic-Aware Deep-Candidate Selector",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T724-T727 showed that fold-safe deep candidates alone do not solve MCD and that a calibrated gate cannot rescue already-wrong selections. T730 therefore targets the observed bottleneck directly: harmonic/alias false peaks that look strong by SNR/support but are inconsistent with candidate relations.",
        "",
        "## Method Change",
        "",
        "- Keep the same candidate pool as T726, including T724 out-of-fold deep candidates.",
        "- Add label-free harmonic/alias relation features such as half/double neighbors, SNR-agreement conflict, alias-band risk, deep disagreement risk, and a combined `harmonic_trap_score`.",
        "- Replace the selector loss with a stronger physiology-risk objective: adjusted unsafe target distribution, expected unsafe penalty, harmonic-trap penalty, and safe-vs-unsafe ranking margin.",
        "",
        "## Baselines",
        "",
        base.markdown_table(base_metrics),
        "",
        "## Selector Metrics",
        "",
        base.markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        base.markdown_table(claim),
        "",
        "## Failure Taxonomy",
        "",
        base.markdown_table(fail),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "T730 can strengthen the main paper claim only if MCD reaches safe released coverage and at least one external dataset remains supported. If MCD still fails, the result narrows the failure to missing route/domain calibration rather than absence of candidates.",
    ]
    base.OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.add_relation_features = harmonic_aware_relation_features
base.selector_loss = t730_selector_loss
base.predict_model = t730_predict_model
base.heuristic_risk = t730_heuristic_risk
base.failure_taxonomy = t730_failure_taxonomy
base.write_report = write_report


def main() -> int:
    rc = t726.main()
    if base.OUT_SUMMARY.exists():
        summary = json.loads(base.OUT_SUMMARY.read_text(encoding="utf-8"))
        claim = pd.read_csv(base.OUT_CLAIM)
        mcd = claim[claim["dataset"].astype(str).eq("MCD-rPPG")].copy()
        summary["t730_added_features"] = EXTRA_FEATURES
        summary["t730_mcd_best_safe_coverage"] = float(mcd["best_safe_gate_coverage"].max()) if not mcd.empty else 0.0
        summary["t730_mcd_best_released_mae"] = float(mcd["best_safe_gate_released_mae_bpm"].min()) if not mcd["best_safe_gate_released_mae_bpm"].dropna().empty else math.nan
        summary["claim_boundary"] = (
            "T730 is a harmonic-aware selector refit on the T726 pool. "
            "It is valid only as a selector/gate evidence update; it does not prove universal SOTA."
        )
        base.OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
