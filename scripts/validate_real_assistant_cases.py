from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.guardrails import causal_alignment_error


def request_json(url: str, *, payload: dict[str, Any] | None = None, timeout: int = 420) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def uploaded_cases(api_url: str) -> dict[str, dict[str, Any]]:
    registry = request_json(f"{api_url}/api/v1/cases")
    selected: dict[str, dict[str, Any]] = {}
    for item in registry.get("items") or []:
        case_id = str(item.get("case_id") or "")
        decision = str(item.get("decision") or "")
        if not case_id.startswith("upload_") or decision not in {"release", "review", "retake"}:
            continue
        full = request_json(f"{api_url}/api/v1/cases/{case_id}")
        current = selected.get(decision)
        if current is None or str(full.get("created_at") or "") > str(current.get("created_at") or ""):
            selected[decision] = full
    missing = {"release", "review", "retake"} - set(selected)
    if missing:
        raise RuntimeError(f"Missing uploaded real-data states: {sorted(missing)}")
    return selected


def run_scenario(
    api_url: str,
    *,
    name: str,
    case: dict[str, Any],
    language: str,
    question: str,
    expected_model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = request_json(
        f"{api_url}/api/v1/assistant/chat",
        payload={
            "message": question,
            "case_id": case["case_id"],
            "role": "operator",
            "language": language,
            "actor": "real-data-assistant-acceptance",
        },
    )
    elapsed = time.perf_counter() - started
    report = request_json(f"{api_url}/api/v1/cases/{case['case_id']}/report?format=json&language=en")
    answer = str(response.get("answer") or "")
    expected_state = str(case["decision"])
    checks: list[dict[str, Any]] = []

    def check(label: str, passed: bool, detail: Any = "") -> None:
        checks.append({"label": label, "passed": bool(passed), "detail": detail})

    check("provider is Ollama", response.get("provider") == "ollama", response.get("provider"))
    check("configured model answered", response.get("model") == expected_model, response.get("model"))
    check("answer is not degraded", response.get("degraded") is False, response.get("validation"))
    check("post-validation passed", (response.get("validation") or {}).get("passed") is True)
    check("no fallback reason", not (response.get("validation") or {}).get("fallback_reason"))
    check("recorded state preserved", (response.get("decision_summary") or {}).get("state") == expected_state)
    check("answer cites supplied evidence", bool(re.search(r"\[E\d+\]", answer)))
    check("answer is substantive", len(answer) >= 180, len(answer))
    check(
        "causal statements use only triggered evidence",
        causal_alignment_error(answer, [{"action_plan": report.get("action_plan") or {}}]) is None,
        causal_alignment_error(answer, [{"action_plan": report.get("action_plan") or {}}]),
    )

    if expected_state == "release":
        released = float(case["released_hr_bpm"])
        formatted = f"{released:.1f} BPM"
        check("release exposes finite HR", (response.get("decision_summary") or {}).get("released_hr_bpm") == released)
        check("prose uses one-decimal released HR", formatted.lower() in answer.lower(), formatted)
        check("prose does not expose raw stored precision", str(released) not in answer, str(released))
    else:
        check("non-release HR is withheld in contract", (response.get("decision_summary") or {}).get("released_hr_bpm") is None)
        check("non-release answer contains no BPM value", not re.search(r"-?\d+(?:\.\d+)?\s*BPM", answer, re.IGNORECASE))
        withholding_terms = r"withheld|not published|not released|不发布|不输出|暂不发布|保持隐藏|保留不报"
        check("non-release answer states HR withholding", bool(re.search(withholding_terms, answer, re.IGNORECASE)))

    if expected_state == "review":
        if language == "zh":
            check("Chinese review answer is Chinese", len(re.findall(r"[\u4e00-\u9fff]", answer)) >= 30)
            check("Chinese answer names competing tracks", "候选" in answer and ("轨迹" in answer or "跨窗口" in answer))
        else:
            check("review answer names competing tracks", bool(re.search(r"competing.*track|track.*competing", answer, re.IGNORECASE)))
    if expected_state == "retake":
        check("retake answer identifies duration", bool(re.search(r"duration|recording (?:time|length)", answer, re.IGNORECASE)))
        check("retake answer gives the recorded minimum", bool(re.search(r"at least\s+8\s+seconds|8\s*s(?:econds)?", answer, re.IGNORECASE)))

    return {
        "name": name,
        "case_id": case["case_id"],
        "display_id": case.get("display_id"),
        "source_name": case.get("source_name"),
        "expected_state": expected_state,
        "language": language,
        "elapsed_seconds": round(elapsed, 3),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "response": response,
        "action_plan": report.get("action_plan"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Real-data assistant acceptance",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- API: `{payload['api_url']}`",
        f"- Expected model: `{payload['expected_model']}`",
        f"- Result: **{'PASS' if payload['passed'] else 'FAIL'}**",
        "",
        "| Scenario | Case | State | Language | Provider | Time (s) | Result |",
        "|---|---|---|---|---|---:|---|",
    ]
    for item in payload["scenarios"]:
        response = item["response"]
        lines.append(
            f"| {item['name']} | `{item['display_id']}` | {item['expected_state']} | {item['language']} | "
            f"{response.get('provider')} / {response.get('model')} | {item['elapsed_seconds']:.3f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Checks", ""])
    for item in payload["scenarios"]:
        lines.append(f"### {item['name']}")
        for check in item["checks"]:
            suffix = f" - {check['detail']}" if check.get("detail") not in (None, "", False) else ""
            lines.append(f"- {'PASS' if check['passed'] else 'FAIL'}: {check['label']}{suffix}")
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "This is a finite product-workflow acceptance on authorized local fixtures. It is not independent clinical validation, diagnostic validation, or proof that no defect exists outside the exercised scope.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local assistant against uploaded real release/review/retake cases.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8011")
    parser.add_argument("--expected-model", default="qwen3.6:35b")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    api_url = args.api_url.rstrip("/")
    cases = uploaded_cases(api_url)
    scenarios = [
        run_scenario(
            api_url,
            name="real release explanation",
            case=cases["release"],
            language="en",
            question="For this exact case, state the output and released HR, explain which evidence drove the state and which checks did not, then give the permitted verification step.",
            expected_model=args.expected_model,
        ),
        run_scenario(
            api_url,
            name="real review explanation",
            case=cases["review"],
            language="en",
            question="For this exact case, identify only the checks that caused review, distinguish passing checks from causes, state the HR publication status, and give the permitted next action.",
            expected_model=args.expected_model,
        ),
        run_scenario(
            api_url,
            name="real review explanation in Chinese",
            case=cases["review"],
            language="zh",
            question="请只说明这个真实案例中真正触发人工复核的检查项，明确哪些通过项不是原因，说明心率是否发布，并给出证据允许的下一步操作。",
            expected_model=args.expected_model,
        ),
        run_scenario(
            api_url,
            name="real duration-only retake explanation",
            case=cases["retake"],
            language="en",
            question="Explain exactly why this video requires retake, which acquisition checks passed and therefore are not causes, whether HR is published, and the only recorded correction step.",
            expected_model=args.expected_model,
        ),
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "api_url": api_url,
        "expected_model": args.expected_model,
        "passed": all(item["passed"] for item in scenarios),
        "scenarios": scenarios,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "real_assistant_acceptance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "REAL_ASSISTANT_ACCEPTANCE.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "scenarios": len(scenarios), "output_dir": str(args.output_dir)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
