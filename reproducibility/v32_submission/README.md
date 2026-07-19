# VitalsSight V32 submission reproducibility package

This directory contains the public scientific code and data contracts needed to fit, freeze and run the V32 source-preserving candidate path on authorized candidate ledgers, recompute the internal aggregate tables and candidate-oracle diagnostic, and reproduce or independently audit the frozen EMPD aggregate contrasts when the data provider's reference files are available locally.

The scientific programs in `scripts/` are byte-identical copies of the frozen manuscript-workspace programs. No fitting, inference, threshold search or analysis logic was rewritten for this package.

## What is included

- `scripts/analyze_v32_candidate_oracle_regret.py`: development-only candidate-pool/oracle diagnostic. Reference HR is used retrospectively and never enters inference.
- `scripts/screen_v31_tree_rankers.py`: participant-disjoint candidate-ranker screening with explicit forbidden-feature checks.
- `scripts/run_v31_matched_stacking_baselines.py`: matched ridge and ExtraTrees direct-combination controls using the five deep-route prediction stack.
- `scripts/run_v31_nested_candidate_vs_stacking.py`: nested participant-disjoint source-preserving selector and matched comparator evaluation.
- `scripts/run_v32_causal_candidate_path.py`: nested V32 candidate-emission fitting, transition-penalty selection and causal source-preserving path inference.
- `scripts/freeze_v32_causal_candidate_path.py`: fit and serialize the development-selected V32 path without accessing external outcomes.
- `scripts/run_v32_frozen_candidate_path_inference.py`: run the frozen candidate path on an outcome-free candidate pool with relation-parity and forbidden-field checks.
- `scripts/v32_candidate_relations.py`: prediction-only relation-feature construction used by frozen inference.
- `scripts/train_v31_frozen_release_gate.py`: exact development-only fitting program for the separately frozen three-seed ExtraTrees release/review research score.
- `scripts/run_v31_frozen_candidate_selector_inference.py`: exact pre-outcome selector and release-gate inference program used for the prospectively frozen EMPD run.
- `scripts/build_v32_internal_core_evidence.py`: participant-equal internal endpoints, participant bootstrap intervals, paired effects, Table 2 source data, and Figure 2 exports.
- `scripts/evaluate_v32_empd_posthoc_fixed_replay.py`: fixed-model EMPD replay evaluation from an outcome-free frozen prediction ledger plus authorized reference and comparator files.
- `scripts/audit_v32_empd_posthoc_fixed_replay.py`: independent EMPD endpoint and paired-inference audit using independent random seeds.
- `frozen_aggregates/`: exact aggregate-only outputs used to check the manuscript numbers and claim boundaries.
- `schemas/`: header-only input templates. They contain no observations, subject identifiers, reference values, or provider labels.
- `contracts/v32_internal_training_contract.json`: path-free machine-readable denominator, split, seed, selector, transition and claim-boundary contract.
- `contracts/v31_release_gate_contract_public.json`: path-free threshold rule, feature allow-list, script hashes and frozen gate-model hashes; model binaries and participant data are not redistributed.
- `expected_headline_metrics.json`: machine-readable expected point estimates and inferential roles.
- `SHA256SUMS.txt`: hashes for every packaged file except the checksum file itself.

## What is deliberately excluded

This package contains no video, image, waveform, dataset archive, participant-level EMPD table, window-level EMPD reference HR, source identifier, timestamp, condition or demographic value, model weight, credential, key, token, authorization artifact, or one-time-unseal record. It also excludes the frozen prediction CSV because its sample and participant identities remain tied to provider-controlled records.

The EMPD scripts are therefore executable only after an authorized user supplies the provider-controlled inputs described below. Their generated participant and joined window outputs are local audit artifacts and must not be redistributed without a separate rights review.

## Environment

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The aggregate builders require no GPU. Candidate fitting can run on CPU; runtime depends on the authorized ledger size. The frozen inference scripts do not require provider outcomes.

## Package verification

Run both the hash/metric verifier and Python byte-compilation check from this directory:

```powershell
python scripts\verify_package.py
python -c "import pathlib, py_compile; [py_compile.compile(str(p), doraise=True) for p in pathlib.Path('scripts').glob('*.py')]"
```

`verify_package.py` checks all packaged hashes and the central manuscript values to a tolerance of `1e-12`.

## Internal V32 analyses

### 0. Comparator and candidate-path fitting contract

`contracts/v32_internal_training_contract.json` fixes the participant folds, seeds, model settings, transition grid and feature guards used for the reported internal path. The methods do not receive identical representations: the source-preserving selectors operate on the full 8,096-candidate ledger, ridge and ExtraTrees operate on a 53-field stack derived from five deep-route predictions for the same windows, and TS-CAN is a single-route control. The matched stackers are therefore direct-combination controls; the primary mechanism contrast is V32 versus the earlier source-preserving selector.

The executable sequence on authorized internal ledgers is:

```powershell
python scripts\screen_v31_tree_rankers.py --help
python scripts\run_v31_matched_stacking_baselines.py --help
python scripts\run_v31_nested_candidate_vs_stacking.py --help
python scripts\run_v32_causal_candidate_path.py --help
python scripts\freeze_v32_causal_candidate_path.py --help
python scripts\run_v32_frozen_candidate_path_inference.py --help
python scripts\train_v31_frozen_release_gate.py --help
python scripts\run_v31_frozen_candidate_selector_inference.py --help
```

Each program requires explicit input and output paths and refuses label-derived inference fields. Dataset records and provider-controlled ledgers are deliberately not included in this repository.

### 1. Candidate-pool and oracle diagnostic

Use the frozen candidate-score ledger and the one-row-per-window selected prediction ledger. The required columns are listed in `schemas/internal_candidate_scores_header.csv` and `schemas/internal_selector_predictions_header.csv`.

```powershell
python scripts\analyze_v32_candidate_oracle_regret.py `
  --candidate-scores <authorized_candidate_scores.csv> `
  --selector-predictions <authorized_selected_predictions.csv> `
  --output-dir <new_oracle_audit_directory> `
  --bootstrap-draws 10000 `
  --bootstrap-seed 320719
```

The output directory must not already exist. Oracle performance is a non-deployable development diagnostic; it quantifies candidate-pool headroom and is not a model result.

### 2. Internal aggregate table, effects and Figure 2

Use the matched participant-disjoint prediction ledger and its frozen summary JSON. The ledger requires the methods named in the script, one row per method-window pair, and the fields listed in `schemas/internal_prediction_ledger_header.csv`. The summary object must contain `causal_path_minus_primary_comparator` and `causal_path_minus_independent`, each with the keys documented in `schemas/README.md`.

```powershell
python scripts\build_v32_internal_core_evidence.py `
  --prediction-ledger <authorized_internal_prediction_ledger.csv> `
  --summary <authorized_internal_summary.json> `
  --output-dir <new_internal_evidence_directory>
```

The script enforces 42 participants for every matched method and refuses to overwrite an existing output directory.

## EMPD frozen analyses

### 3. Primary fixed replay evaluation

The prediction ledger must have been frozen before reference outcomes were opened. Its hash must match the supplied prediction manifest. Reference and comparator files remain provider-controlled and are not distributed here.

```powershell
python scripts\evaluate_v32_empd_posthoc_fixed_replay.py `
  --predictions <authorized_outcome_free_frozen_predictions.csv> `
  --prediction-manifest <authorized_prediction_manifest.json> `
  --reference-windows <provider_controlled_reference_windows.csv> `
  --existing-participant-metrics <authorized_frozen_comparator_participants.csv> `
  --existing-summary <authorized_v31_frozen_external_summary.json> `
  --output-dir <new_empd_replay_directory>
```

Important: this script writes `v32_empd_predictions_with_reference.csv` and a participant-level metric table. Those outputs are local restricted audit products, not files for a public repository.

### 4. Independent replay audit

```powershell
python scripts\audit_v32_empd_posthoc_fixed_replay.py `
  --predictions <authorized_outcome_free_frozen_predictions.csv> `
  --reference <provider_controlled_reference_windows.csv> `
  --comparator-participants <authorized_frozen_comparator_participants.csv> `
  --reported-summary <local_replay_output\summary.json> `
  --output-dir <new_independent_audit_directory>
```

The audit re-derives 83 participant-equal endpoints and paired effects with independent bootstrap and sign-flip seeds. It intentionally does not import the primary evaluator.

## Expected results and interpretation

The central internal V32 path has participant-equal MAE `1.8354543521954239` BPM across 42 participants and 439 windows. Its paired MAE difference from the prior within-study source-preserving selector is `-0.8533296130952379` BPM (95% bootstrap interval `-1.7202972808441555` to `-0.248826856737013`; paired sign-flip `P=0.0029997000299970002`). The temporal increment versus independent emission remains exploratory because its sign-flip `P=0.0993900609939006`.

The prospectively frozen V31 EMPD result is the external primary evidence: participant-equal MAE `4.2166107612267245` BPM across 83 participants and 2,120 windows. It improves on frozen TS-CAN by `-1.4970820783132535` BPM after Holm adjustment (`P=0.00007999920000799993`) but does not differ from matched ExtraTrees (`-0.1275212003015192` BPM; `P=0.35422645773542266`).

The V32 EMPD replay is post hoc consistency evidence only because EMPD outcomes had already been viewed before V32 was designed. Its MAE is `3.952243633625411` BPM. It cannot replace the prespecified V31 external test or support a new untouched-external claim.

The release/review score is not a calibrated safety probability. Proposed release covered 42 of 2,120 EMPD windows and did not establish clinical safety, autonomous release, or clinical utility.

The exact output-state rule is documented in `contracts/v31_release_gate_contract_public.json`. Its threshold was fixed at `0.0`, the empirical 20th percentile of cross-fitted development research-risk scores, before EMPD reference outcomes were accessed. The JSON records the three model hashes and the hashes of the exact fitting and frozen-inference scripts, while excluding local paths and provider-controlled model binaries.

## Reproducibility boundary

The public artifacts allow number-level verification and provide exact analysis code and input contracts. They do not make provider-controlled participant records public. Full outcome-level recomputation requires lawful access to the original datasets and locally retained frozen ledgers whose hashes are recorded in the aggregate summaries.
