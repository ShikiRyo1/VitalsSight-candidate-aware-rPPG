from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.selection.release_policy import apply_sample_release_gate
from src.selection.roi_evidence import select_roi_supported_clusters_v2


def main() -> None:
    candidates = pd.DataFrame(
        [
            {"sample_id": "stable", "candidate_bpm": 72.0, "region": "forehead", "method": "POS", "power": 1.0},
            {"sample_id": "stable", "candidate_bpm": 73.0, "region": "left_cheek", "method": "CHROM", "power": 0.9},
            {"sample_id": "stable", "candidate_bpm": 72.5, "region": "right_cheek", "method": "POS", "power": 0.8},
            {"sample_id": "conflict", "candidate_bpm": 60.0, "region": "forehead", "method": "POS", "power": 1.0},
            {"sample_id": "conflict", "candidate_bpm": 112.0, "region": "left_cheek", "method": "CHROM", "power": 1.0},
        ]
    )
    selected = select_roi_supported_clusters_v2(candidates)

    release_input = pd.DataFrame(
        [
            {
                "sample_id": "stable",
                "selected_bpm": 72.5,
                "t150_confidence": 0.86,
                "anchor_median": 72.0,
                "anchor_iqr": 2.0,
                "top1_support_methods": 3,
                "subwindow_top1_support": 8,
                "t150_reason": "candidate_agreement",
            },
            {
                "sample_id": "conflict",
                "selected_bpm": 112.0,
                "t150_confidence": 0.42,
                "anchor_median": 61.0,
                "anchor_iqr": 4.0,
                "top1_support_methods": 1,
                "subwindow_top1_support": 2,
                "t150_reason": "candidate_conflict",
            },
        ]
    )
    decisions = apply_sample_release_gate(release_input, policy_name="public_demo")
    output = {
        "selected_candidates": selected[["sample_id", "cluster_bpm", "roi_evidence_v2_score"]].to_dict("records"),
        "output_contract": decisions[["sample_id", "selected_bpm", "release_status", "review_reason"]].to_dict("records"),
        "boundary": "Synthetic demonstration only; no reference HR is used by selection or release.",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
