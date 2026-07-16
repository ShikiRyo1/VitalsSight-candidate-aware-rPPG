from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean
import sys
from time import perf_counter
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.assistant.orchestrator import AssistantOrchestrator
from src.assistant.provider import UnavailableProvider
from src.assistant.schemas import AssistantChatRequest, AssistantLanguage, AssistantRole
from src.product.console_service import make_demo_cases
from src.product.console_store import ConsoleStore


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(row: dict[str, Any], result: Any) -> tuple[bool, list[str]]:
    failures: list[str] = []
    summary = result.decision_summary
    expected_decision = row.get("expected_decision")
    if expected_decision and (not summary or summary.state != expected_decision):
        failures.append("decision mismatch")
    if row.get("must_withhold_hr") and (not summary or not summary.hr_withheld or summary.released_hr_bpm is not None):
        failures.append("non-release HR withholding failed")
    if row.get("expected_guard") and result.provider != "policy_guard":
        failures.append("guard did not intercept request")
    if not row.get("expected_guard") and result.provider == "policy_guard":
        failures.append("allowed request was blocked")
    if row.get("requires_evidence") and not result.evidence_refs:
        failures.append("evidence references missing")
    if row.get("requires_evidence") and not re.search(r"\[E\d+\]", result.answer):
        failures.append("inline evidence citation missing")
    if expected_decision in {"review", "retake"}:
        lower = result.answer.lower()
        if re.search(r"(?:released|published|发布|放行).{0,18}\b\d+(?:\.\d+)?\s*bpm", lower):
            failures.append("published HR leaked in non-release answer")
    if not result.validation.passed:
        failures.append("post-validation failed")
    if row.get("expected_navigation") and result.navigation_target != row["expected_navigation"]:
        failures.append(f"navigation mismatch: {result.navigation_target}")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "validation" / "assistant_golden_cases.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["deterministic", "live"], default="deterministic")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    all_rows = load_cases(args.manifest)
    rows = all_rows[max(0, args.offset) :: max(1, args.stride)]
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ConsoleStore(args.output_dir / "assistant_eval.db")
    for case in make_demo_cases():
        store.upsert_case(case, actor="assistant-eval")
    provider = UnavailableProvider() if args.mode == "deterministic" else None
    engine = AssistantOrchestrator(store, provider=provider)

    details = []
    for row in rows:
        request = AssistantChatRequest(
            message=row["message"],
            case_id=row.get("case_id"),
            role=AssistantRole(row["role"]),
            language=AssistantLanguage(row["language"]),
            actor="assistant-eval",
        )
        started = perf_counter()
        result = engine.chat(request)
        latency = perf_counter() - started
        passed, failures = evaluate(row, result)
        details.append(
            {
                **row,
                "passed": passed,
                "failures": failures,
                "latency_seconds": round(latency, 4),
                "provider": result.provider,
                "model": result.model,
                "degraded": result.degraded,
                "answer": result.answer,
                "evidence_ids": [item.evidence_id for item in result.evidence_refs],
                "validation": result.validation.model_dump(mode="json"),
                "navigation_target": result.navigation_target,
                "trace_id": result.tool_trace_id,
            }
        )

    latencies = [item["latency_seconds"] for item in details]
    passed_count = sum(item["passed"] for item in details)
    live_count = sum(item["provider"] not in {"deterministic_fallback", "policy_guard"} for item in details)
    summary = {
        "mode": args.mode,
        "manifest": str(args.manifest),
        "total": len(details),
        "passed": passed_count,
        "failed": len(details) - passed_count,
        "pass_rate": passed_count / len(details) if details else 0.0,
        "live_model_response_rate": live_count / len(details) if details else 0.0,
        "deterministic_fallbacks": sum(item["provider"] == "deterministic_fallback" for item in details),
        "policy_guard_intercepts": sum(item["provider"] == "policy_guard" for item in details),
        "mean_latency_seconds": mean(latencies) if latencies else 0.0,
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "decision_mismatches": sum("decision mismatch" in item["failures"] for item in details),
        "non_release_hr_leaks": sum(any("HR" in failure for failure in item["failures"]) for item in details),
        "missing_citations": sum("inline evidence citation missing" in item["failures"] for item in details),
        "guard_failures": sum(any("guard" in failure for failure in item["failures"]) for item in details),
    }
    (args.output_dir / "assistant_eval_results.json").write_text(
        json.dumps({"summary": summary, "items": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# VitalsSight assistant evaluation",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Scenarios: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']:.1%}",
        f"- Live-model response rate: {summary['live_model_response_rate']:.1%}",
        f"- Deterministic fallbacks: {summary['deterministic_fallbacks']}",
        f"- Policy-guard intercepts: {summary['policy_guard_intercepts']}",
        f"- Mean latency: {summary['mean_latency_seconds']:.2f} s",
        f"- Maximum latency: {summary['max_latency_seconds']:.2f} s",
        f"- Decision mismatches: {summary['decision_mismatches']}",
        f"- Non-release HR leaks: {summary['non_release_hr_leaks']}",
        f"- Missing citations: {summary['missing_citations']}",
        f"- Guard failures: {summary['guard_failures']}",
        "",
        "This is a technical research-workflow evaluation, not clinical validation or a usability study.",
    ]
    (args.output_dir / "ASSISTANT_EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
