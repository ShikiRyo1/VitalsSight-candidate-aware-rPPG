from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from src.selection.release_policy import apply_sample_release_gate
from src.selection.roi_evidence import build_roi_candidate_clusters


def test_roi_candidate_selection_is_label_free() -> None:
    base = pd.DataFrame(
        [
            {"sample_id": "s1", "candidate_bpm": 72.0, "region": "forehead", "method": "POS", "power": 1.0},
            {"sample_id": "s1", "candidate_bpm": 73.0, "region": "left_cheek", "method": "CHROM", "power": 0.8},
            {"sample_id": "s1", "candidate_bpm": 72.5, "region": "right_cheek", "method": "POS", "power": 0.9},
        ]
    )
    first = base.assign(gt_hr_bpm=72.0, candidate_abs_error=0.5)
    second = base.assign(gt_hr_bpm=140.0, candidate_abs_error=68.0)
    assert_frame_equal(build_roi_candidate_clusters(first), build_roi_candidate_clusters(second))


def test_release_decision_does_not_depend_on_reference_fields() -> None:
    row = {
        "sample_id": "s1",
        "selected_bpm": 72.0,
        "t150_confidence": 0.85,
        "anchor_median": 72.0,
        "anchor_iqr": 2.0,
        "top1_support_methods": 3,
        "subwindow_top1_support": 8,
        "t150_reason": "candidate_agreement",
    }
    first = apply_sample_release_gate(
        pd.DataFrame([{**row, "gt_hr_bpm": 72.0, "selected_abs_error_bpm": 0.0}]),
        policy_name="test",
    )
    second = apply_sample_release_gate(
        pd.DataFrame([{**row, "gt_hr_bpm": 140.0, "selected_abs_error_bpm": 68.0}]),
        policy_name="test",
    )
    assert first.loc[0, "release_status"] == second.loc[0, "release_status"] == "release"
    assert first.loc[0, "review_reason"] == second.loc[0, "review_reason"]

