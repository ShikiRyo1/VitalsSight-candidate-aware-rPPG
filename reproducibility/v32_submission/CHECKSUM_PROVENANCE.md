# Checksum provenance: original package and public LF checkout

## Why there are two manifests

`SHA256SUMS.txt` is the unchanged historical manifest. Seventeen files under `frozen_aggregates/` were hashed with Windows CRLF line endings before Git normalized their committed contents to LF under the repository's `.gitattributes`. All 17 original hashes can be reproduced from the committed files by reconstructing CRLF line endings alone. No numerical or textual content change is needed.

`SHA256SUMS.public-lf.txt` is the separate exact-byte manifest for the public checkout as updated on 6 September 2026. It covers the original packaged file list, the retained historical manifest and this note. It excludes itself to avoid a circular digest. Its verifier entry records the deliberately updated public-package verifier, not the old verifier's historical hash.

The public verifier first checks every current byte against the public-LF manifest. It then independently checks each historical aggregate against its original digest, allowing only the documented LF-to-CRLF reconstruction for that provenance comparison. A numerical, textual or missing-file change fails. An unexpected line-ending change in the current checkout also fails its exact-byte checksum; normalization is not a general-purpose fallback.

## Preserved records and scope

- Original `SHA256SUMS.txt`, `VALIDATION_REPORT.json`, contracts, expected metrics, aggregate CSV/JSON files and scientific analysis programs are unchanged.
- `scripts/verify_package.py` is a maintained verification wrapper. Its original hash remains in the historical manifest; its current hash is recorded in the public-LF manifest. It is not claimed to be byte-identical to its historical version.
- The 13 scientific fitting, inference and analysis programs recorded in the historical validation report retain their original bytes. The historical report's `PASS` describes that original report, not a new full experiment rerun.
- `tests/test_package_checksums.py` at the repository root provides synthetic temporary-file checks for exact bytes, CRLF handling, missing files and real content changes.
- This check verifies packaged contracts and aggregate values. Provider-controlled datasets, frozen prediction ledgers and model binaries remain absent; a passing result is not end-to-end scientific reproduction.

Run from the repository root:

```bash
python -m unittest discover -s tests -p test_package_checksums.py -v
python reproducibility/v32_submission/scripts/verify_package.py
```
