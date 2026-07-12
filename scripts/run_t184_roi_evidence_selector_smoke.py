from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.selection.roi_evidence import build_roi_candidate_clusters, select_roi_supported_clusters


EXPERIMENTS = ROOT / "experiments"
DOCS = ROOT / "docs"


def synthetic_candidates() -> pd.DataFrame:
    rows = []
    for region in ["forehead", "left_cheek", "right_cheek", "nose_tip"]:
        for method, offset, power in [("GREEN", 0.0, 1.4), ("POS", 0.8, 1.1), ("CHROM", -0.6, 1.0)]:
            rows.append({"sample_id": "S_true_multi_roi", "region": region, "method": method, "candidate_bpm": 80.0 + offset, "power": power})
    rows.append({"sample_id": "S_true_multi_roi", "region": "chin", "method": "GREEN", "candidate_bpm": 121.0, "power": 5.5})
    rows.append({"sample_id": "S_true_multi_roi", "region": "chin", "method": "POS", "candidate_bpm": 123.0, "power": 4.2})

    for method, bpm, power in [("GREEN", 62.0, 1.0), ("POS", 63.5, 0.8), ("CHROM", 119.0, 3.5), ("PBV", 121.0, 2.8)]:
        rows.append({"sample_id": "S_ambiguous_review", "region": "forehead", "method": method, "candidate_bpm": bpm, "power": power})
    rows.append({"sample_id": "S_ambiguous_review", "region": "left_cheek", "method": "GREEN", "candidate_bpm": 65.0, "power": 0.7})

    for region in ["forehead", "left_cheek"]:
        rows.append({"sample_id": "S_borderline", "region": region, "method": "GREEN", "candidate_bpm": 95.0, "power": 1.2})
        rows.append({"sample_id": "S_borderline", "region": region, "method": "POS", "candidate_bpm": 96.5, "power": 0.9})
    return pd.DataFrame(rows)


def main() -> None:
    EXPERIMENTS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    candidates = synthetic_candidates()
    clusters = build_roi_candidate_clusters(candidates)
    selected = select_roi_supported_clusters(candidates)
    candidates.to_csv(EXPERIMENTS / "t184_roi_evidence_synthetic_candidates.csv", index=False)
    clusters.to_csv(EXPERIMENTS / "t184_roi_evidence_clusters.csv", index=False)
    selected.to_csv(EXPERIMENTS / "t184_roi_evidence_selected.csv", index=False)
    expected = {
        "S_true_multi_roi": "around_80_multi_roi",
        "S_ambiguous_review": "low_confidence_or_single_region_dominant",
        "S_borderline": "around_95_two_roi_two_method",
    }
    summary = {
        "task": "T184 ROI-evidence selector smoke test",
        "date": date.today().isoformat(),
        "n_candidates": int(len(candidates)),
        "n_clusters": int(len(clusters)),
        "n_selected": int(len(selected)),
        "selected": selected[["sample_id", "cluster_bpm", "roi_support", "method_support", "roi_evidence_score", "passes_roi_evidence_gate"]].to_dict("records"),
        "expected_behavior": expected,
        "decision": "pass"
        if (
            selected.loc[selected["sample_id"].eq("S_true_multi_roi"), "cluster_bpm"].between(78, 82).any()
            and selected.loc[selected["sample_id"].eq("S_ambiguous_review"), "passes_roi_evidence_gate"].eq(0).any()
            and selected.loc[selected["sample_id"].eq("S_borderline"), "passes_roi_evidence_gate"].eq(1).any()
        )
        else "review",
    }
    (EXPERIMENTS / "t184_roi_evidence_selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# T184 ROI-Evidence Candidate Selector Smoke Test

## 目的
T184 检查一个关键算法想法：当一个高功率 spectral peak 只出现在单一区域，而另一个中等功率 peak 被多个 ROI 和多个传统方法共同支持时，selector 应该更信后者。这个模块是 T182/T183 ROI layer 与 T144/T150/T160 multi-candidate selection 之间的桥。

## 结果
- synthetic candidates：{len(candidates)}
- candidate clusters：{len(clusters)}
- selected clusters：{len(selected)}
- 输出：`experiments/t184_roi_evidence_clusters.csv`、`experiments/t184_roi_evidence_selected.csv`

## Insight
这个 smoke test 支持我们的理论方向：rPPG 错误不一定来自“没有信号”，而经常来自“选错信号源或选错峰”。ROI evidence score 把 candidate peak 的可信度从单一功率峰转成多区域、多方法、频率一致性的组合证据。后续真实实验需要证明它能在 UBFC/DLCN/MCD 等 domain shift 数据上降低 unsafe release 或 MAE。
"""
    (DOCS / "t184_roi_evidence_selector_smoke.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
