# Input schemas

The CSV files in this directory contain headers only. They describe the minimum fields consumed by the copied scripts and contain no observations.

## Internal candidate scores

`internal_candidate_scores_header.csv` records one candidate per window and seed. `candidate_id` must uniquely identify a candidate within a `sample_id`; every candidate in the frozen analysis had three seed scores. The oracle diagnostic uses `gt_hr_bpm` only retrospectively.

## Internal selected predictions

`internal_selector_predictions_header.csv` contains exactly one selected candidate per window. `selected_candidate_id` must exist in the matching candidate set.

## Internal matched prediction ledger

`internal_prediction_ledger_header.csv` contains one row per method and window. At minimum the table builder consumes `method`, `sample_id`, `subject_std` and `abs_error_bpm`. The complete executed ledger also retained prediction and reference values for authorized local auditing.

The companion summary JSON passed to `build_v32_internal_core_evidence.py` must contain:

```json
{
  "causal_path_minus_primary_comparator": {
    "mean_delta_a_minus_b_bpm": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "paired_sign_flip_p_plus_one": 1.0
  },
  "causal_path_minus_independent": {
    "mean_delta_a_minus_b_bpm": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "paired_sign_flip_p_plus_one": 1.0
  }
}
```

The zeros above are schema placeholders, not study results.

## EMPD inputs

- `empd_frozen_predictions_header.csv`: outcome-free one-row-per-window frozen V32 predictions.
- `empd_reference_windows_header.csv`: provider-controlled reference ledger. It is never distributed in this package.
- `empd_comparator_participant_metrics_header.csv`: provider-controlled participant-equal comparator endpoints.

The prediction manifest is a JSON object with a `files` array. The record whose `path` equals the prediction filename must provide integer `bytes` and uppercase hexadecimal `sha256` fields.

The existing frozen V31 summary must have `status` equal to `EXTERNAL_RESULT_FROZEN` and `external_outcomes_accessed` equal to `true`. The independent audit consumes the `summary.json` emitted by the primary V32 evaluator.

