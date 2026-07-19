#!/usr/bin/env python3
"""Verify packaged hashes and the frozen V32 headline aggregate values."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: str | float, expected: float, label: str) -> None:
    value = float(actual)
    require(abs(value - expected) <= TOLERANCE, f"{label}: {value} != {expected}")


def verify_hashes() -> int:
    manifest = ROOT / "SHA256SUMS.txt"
    require(manifest.is_file(), "SHA256SUMS.txt is missing")
    checked = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(sha256(path) == expected.upper(), f"hash mismatch: {relative}")
        checked += 1
    return checked


def verify_metrics() -> list[str]:
    expected = json.loads((ROOT / "expected_headline_metrics.json").read_text(encoding="utf-8"))
    internal = read_rows("frozen_aggregates/internal/v32_internal_table2.csv")
    ours = next(row for row in internal if row["Method"] == "VitalsSight V32 (ours)")
    target = expected["internal_primary"]
    require(int(ours["Participants"]) == target["participants"], "internal participant count")
    require(int(ours["Windows"]) == target["windows"], "internal window count")
    close(ours["Participant-equal MAE (BPM)"], target["participant_equal_mae_bpm"], "internal MAE")
    close(ours["Participant-equal RMSE (BPM)"], target["participant_equal_rmse_bpm"], "internal RMSE")
    close(ours["Within 5 BPM"], target["within5"], "internal within5")
    close(ours["Within 10 BPM"], target["within10"], "internal within10")
    close(ours["Error >10 BPM"], target["unsafe_gt10"], "internal unsafe")

    effects = read_rows("frozen_aggregates/internal/v32_internal_primary_effects.csv")
    primary = next(row for row in effects if row["Comparator"] == "Prior candidate-aware")
    contrast = expected["internal_primary_contrast"]
    close(primary["Mean delta (V32 minus comparator), BPM"], contrast["mean_delta_bpm"], "internal delta")
    close(primary["CI95 low"], contrast["ci95_low"], "internal CI low")
    close(primary["CI95 high"], contrast["ci95_high"], "internal CI high")
    close(primary["Paired sign-flip p"], contrast["paired_sign_flip_p_plus_one"], "internal p")

    external = read_rows("frozen_aggregates/empd/v31_frozen_external_method_metrics.csv")
    v31 = next(row for row in external if row["method"] == "VitalsSight_frozen_candidate_aware")
    ext_target = expected["external_primary_v31"]
    require(int(v31["participants"]) == ext_target["participants"], "external participant count")
    require(int(v31["windows"]) == ext_target["windows"], "external window count")
    close(v31["participant_equal_mae_bpm"], ext_target["participant_equal_mae_bpm"], "external V31 MAE")

    v32_rows = read_rows("frozen_aggregates/empd/v32_posthoc_method_metrics.csv")
    require(len(v32_rows) == 1, "unexpected V32 EMPD method row count")
    v32 = v32_rows[0]
    posthoc = expected["external_posthoc_v32"]
    require(int(v32["participants"]) == posthoc["participants"], "posthoc participant count")
    require(int(v32["windows"]) == posthoc["windows"], "posthoc window count")
    close(v32["participant_equal_mae_bpm"], posthoc["participant_equal_mae_bpm"], "posthoc MAE")
    close(v32["participant_equal_rmse_bpm"], posthoc["participant_equal_rmse_bpm"], "posthoc RMSE")
    close(v32["participant_equal_within10"], posthoc["within10"], "posthoc within10")
    close(v32["participant_equal_unsafe_gt10"], posthoc["unsafe_gt10"], "posthoc unsafe")

    freeze = json.loads(
        (ROOT / "frozen_aggregates/empd/v32_preoutcome_prediction_freeze_summary.json").read_text(
            encoding="utf-8"
        )
    )
    require(freeze["status"] == "PREDICTIONS_FROZEN_WITHOUT_EXTERNAL_OUTCOMES", "freeze status")
    require(freeze["external_outcomes_accessed"] is False, "pre-outcome access boundary")

    audit = json.loads(
        (ROOT / "frozen_aggregates/empd/v32_posthoc_independent_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    require(audit["status"] == "PASS", "independent audit status")
    require(audit["scientific_role_preserved"] == "exploratory_external_replay_only", "posthoc role")
    return [
        "internal V32 headline metrics",
        "internal primary paired contrast",
        "prospectively frozen V31 EMPD aggregate",
        "post hoc V32 EMPD aggregate",
        "pre-outcome freeze and exploratory-role boundaries",
    ]


def main() -> int:
    result = {
        "status": "PASS",
        "files_hashed": verify_hashes(),
        "metric_checks": verify_metrics(),
        "tolerance": TOLERANCE,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
