from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ROIEvidenceConfig:
    tolerance_bpm: float = 3.0
    min_roi_support: int = 2
    min_method_support: int = 2
    release_min_score: float = 0.55
    anchor_method: str = "POS"
    pos_consensus_min_roi: int = 3
    pos_rescue_min_score: float = 0.65
    pos_consensus_bonus: float = 0.18
    low_artifact_review_bpm: float = 70.0
    high_sparse_anchor_penalty_bpm: float = 125.0
    low_no_anchor_penalty: float = 0.25
    high_sparse_anchor_penalty: float = 0.12
    region_weight: float = 0.35
    method_weight: float = 0.25
    power_weight: float = 0.25
    agreement_weight: float = 0.15


def build_roi_candidate_clusters(
    candidates: pd.DataFrame,
    *,
    config: ROIEvidenceConfig | None = None,
    sample_col: str = "sample_id",
    bpm_col: str = "candidate_bpm",
    region_col: str = "region",
    method_col: str = "method",
    power_col: str = "power",
) -> pd.DataFrame:
    """Cluster candidate peaks by BPM and summarize multi-ROI support.

    The function is label-free: it never reads ground-truth or error columns.
    It only asks whether multiple regions and methods independently point to a
    similar physiological frequency.
    """

    cfg = config or ROIEvidenceConfig()
    required = {sample_col, bpm_col, region_col, method_col}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for sample_id, group in candidates.groupby(sample_col, sort=True):
        valid = group[np.isfinite(pd.to_numeric(group[bpm_col], errors="coerce"))].copy()
        if valid.empty:
            continue
        valid[bpm_col] = pd.to_numeric(valid[bpm_col], errors="coerce")
        if power_col not in valid.columns:
            valid[power_col] = 1.0
        valid[power_col] = pd.to_numeric(valid[power_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        valid = valid.sort_values(bpm_col)

        clusters: list[pd.DataFrame] = []
        current: list[int] = []
        center = None
        for idx, row in valid.iterrows():
            bpm = float(row[bpm_col])
            if center is None or abs(bpm - center) <= cfg.tolerance_bpm:
                current.append(idx)
                center = float(valid.loc[current, bpm_col].median())
            else:
                clusters.append(valid.loc[current].copy())
                current = [idx]
                center = bpm
        if current:
            clusters.append(valid.loc[current].copy())

        for cluster_id, cluster in enumerate(clusters):
            power = cluster[power_col].to_numpy(dtype=float)
            bpm = cluster[bpm_col].to_numpy(dtype=float)
            weights = power + 1e-6
            weighted_bpm = float(np.average(bpm, weights=weights))
            n_regions = int(cluster[region_col].nunique())
            n_methods = int(cluster[method_col].nunique())
            anchor = cluster[cluster[method_col].astype(str).eq(cfg.anchor_method)]
            anchor_roi_support = int(anchor[region_col].nunique())
            anchor_rows = int(len(anchor))
            bpm_std = float(np.std(bpm, ddof=0))
            power_sum = float(power.sum())
            score = _roi_evidence_score(
                n_regions=n_regions,
                n_methods=n_methods,
                power_sum=power_sum,
                bpm_std=bpm_std,
                cfg=cfg,
            )
            low_without_anchor = weighted_bpm < cfg.low_artifact_review_bpm and anchor_roi_support < cfg.pos_consensus_min_roi
            high_sparse_anchor = weighted_bpm > cfg.high_sparse_anchor_penalty_bpm and anchor_roi_support < cfg.pos_consensus_min_roi
            v2_score = float(
                np.clip(
                    score
                    + cfg.pos_consensus_bonus * min(anchor_roi_support / max(1, cfg.pos_consensus_min_roi + 1), 1.0)
                    - (cfg.low_no_anchor_penalty if low_without_anchor else 0.0)
                    - (cfg.high_sparse_anchor_penalty if high_sparse_anchor else 0.0),
                    0.0,
                    1.0,
                )
            )
            v2_core_gate = (
                n_regions >= cfg.min_roi_support
                and n_methods >= cfg.min_method_support
                and v2_score >= cfg.release_min_score
                and not low_without_anchor
            )
            v2_pos_rescue_gate = (
                anchor_roi_support >= cfg.pos_consensus_min_roi
                and n_regions >= cfg.pos_consensus_min_roi
                and v2_score >= cfg.pos_rescue_min_score
            )
            rows.append(
                {
                    sample_col: sample_id,
                    "cluster_id": cluster_id,
                    "cluster_bpm": weighted_bpm,
                    "cluster_bpm_median": float(np.median(bpm)),
                    "cluster_bpm_std": bpm_std,
                    "roi_support": n_regions,
                    "method_support": n_methods,
                    "support_rows": int(len(cluster)),
                    "power_sum": power_sum,
                    "regions": ",".join(sorted(map(str, cluster[region_col].dropna().unique()))),
                    "methods": ",".join(sorted(map(str, cluster[method_col].dropna().unique()))),
                    "anchor_method": cfg.anchor_method,
                    "anchor_roi_support": anchor_roi_support,
                    "anchor_rows": anchor_rows,
                    "low_without_anchor": int(low_without_anchor),
                    "high_sparse_anchor": int(high_sparse_anchor),
                    "roi_evidence_score": score,
                    "roi_evidence_v2_score": v2_score,
                    "passes_roi_evidence_gate": int(
                        n_regions >= cfg.min_roi_support and n_methods >= cfg.min_method_support and score >= cfg.release_min_score
                    ),
                    "passes_roi_evidence_v2_gate": int(v2_core_gate or v2_pos_rescue_gate),
                }
            )
    return pd.DataFrame(rows)


def select_roi_supported_clusters(
    candidates: pd.DataFrame,
    *,
    config: ROIEvidenceConfig | None = None,
    sample_col: str = "sample_id",
) -> pd.DataFrame:
    clusters = build_roi_candidate_clusters(candidates, config=config, sample_col=sample_col)
    if clusters.empty:
        return clusters
    ranked = clusters.sort_values(
        [sample_col, "passes_roi_evidence_gate", "roi_evidence_score", "roi_support", "method_support"],
        ascending=[True, False, False, False, False],
    )
    return ranked.groupby(sample_col, as_index=False, sort=True).head(1).reset_index(drop=True)


def select_roi_supported_clusters_v2(
    candidates: pd.DataFrame,
    *,
    config: ROIEvidenceConfig | None = None,
    sample_col: str = "sample_id",
) -> pd.DataFrame:
    clusters = build_roi_candidate_clusters(candidates, config=config, sample_col=sample_col)
    if clusters.empty:
        return clusters
    ranked = clusters.sort_values(
        [
            sample_col,
            "passes_roi_evidence_v2_gate",
            "roi_evidence_v2_score",
            "anchor_roi_support",
            "roi_support",
            "method_support",
        ],
        ascending=[True, False, False, False, False, False],
    )
    return ranked.groupby(sample_col, as_index=False, sort=True).head(1).reset_index(drop=True)


def _roi_evidence_score(*, n_regions: int, n_methods: int, power_sum: float, bpm_std: float, cfg: ROIEvidenceConfig) -> float:
    region_score = min(n_regions / max(1, cfg.min_roi_support + 2), 1.0)
    method_score = min(n_methods / max(1, cfg.min_method_support + 2), 1.0)
    power_score = min(np.log1p(max(power_sum, 0.0)) / np.log(16.0), 1.0)
    agreement_score = 1.0 - min(bpm_std / max(cfg.tolerance_bpm, 1e-6), 1.0)
    score = (
        cfg.region_weight * region_score
        + cfg.method_weight * method_score
        + cfg.power_weight * power_score
        + cfg.agreement_weight * agreement_score
    )
    return float(np.clip(score, 0.0, 1.0))
