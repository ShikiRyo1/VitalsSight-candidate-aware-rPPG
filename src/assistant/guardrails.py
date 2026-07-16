from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from src.product.console_service import CLAIM_BOUNDARY


@dataclass(frozen=True)
class InputInspection:
    allowed: bool
    category: str
    message_en: str
    message_zh: str


INJECTION_PATTERNS = (
    r"ignore (all |the )?(previous|system|developer) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"bypass (the )?(safety|policy|guardrail)",
    r"忽略.{0,8}(之前|系统|开发者).{0,8}(指令|提示)",
    r"泄露.{0,8}(系统|开发者).{0,8}(提示词|指令)",
    r"绕过.{0,8}(安全|策略|限制)",
)

CLINICAL_PATTERNS = (
    r"\bdiagnos(e|is|tic)\b",
    r"\btreat(ment|ing)?\b",
    r"\bprescri(be|ption)\b",
    r"should i take",
    r"是不是.{0,5}(心脏病|疾病|感染)",
    r"帮我诊断|诊断我|治疗方案|开什么药|应该吃什么药",
)

SAFE_NEGATED_CLINICAL_PATTERNS = (
    r"\b(?:cannot|can't|will not|won't|does not|doesn't|do not|don't)\s+diagnose(?:,\s*prescribe)?(?:,?\s*(?:or|and)\s+recommend\s+treatment)\b",
    r"\b(?:cannot|can't|will not|won't|does not|doesn't|do not|don't)\s+(?:provide\s+)?(?:a\s+)?(?:diagnos(?:e|is)|prescri(?:be|ption)|recommend\s+treatment|treat(?:ment)?)\b",
    r"\bnot\s+(?:a\s+)?(?:diagnosis|treatment recommendation|prescription)\b",
    r"(?:不能|不会|不可以|无法).{0,10}(?:诊断|开药|处方|治疗建议|推荐治疗)",
    r"(?:不构成|并非).{0,8}(?:诊断|治疗建议|处方)",
)

EMERGENCY_PATTERNS = (
    r"chest pain|cannot breathe|unconscious|medical emergency",
    r"胸痛|无法呼吸|失去意识|急救|生命危险",
)


def inspect_input(message: str) -> InputInspection:
    normalized = " ".join(message.strip().split())
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS):
        return InputInspection(
            False,
            "prompt_injection",
            "I cannot override the evidence contract, reveal internal prompts, or bypass the tool policy. I can explain a case using recorded evidence.",
            "我不能覆盖证据契约、泄露内部提示词或绕过工具策略；可以继续依据已记录证据解释案例。",
        )
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in EMERGENCY_PATTERNS):
        return InputInspection(
            False,
            "emergency_request",
            "VitalsSight is a retrospective research workflow and cannot provide emergency guidance. Contact local emergency services or a qualified clinician now.",
            "VitalsSight 仅用于回顾性研究流程，不能提供急救判断。请立即联系当地急救服务或合格医务人员。",
        )
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in CLINICAL_PATTERNS):
        return InputInspection(
            False,
            "clinical_advice_request",
            "I cannot diagnose, prescribe, or recommend treatment. I can explain the recorded signal-quality evidence, output state, and research-workflow next step.",
            "我不能进行诊断、开药或提出治疗建议；可以解释已记录的信号质量证据、输出状态和研究流程下一步。",
        )
    return InputInspection(True, "allowed", "", "")


def collect_allowed_measurements(tool_results: list[dict[str, Any]]) -> dict[str, set[float]]:
    allowed: dict[str, set[float]] = {"bpm": set(), "fps": set(), "%": set()}

    def add(unit: str, value: float) -> None:
        for precision in (1, 2, 3, 4):
            allowed[unit].add(round(float(value), precision))

    trusted_numeric_text = {"excerpt", "verification", "because", "rationale", "expected_outcome"}

    def visit(value: Any, *, field: str = "") -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            normalized_field = field.lower()
            if normalized_field.endswith("_bpm") or normalized_field in {"bpm", "heart_rate"}:
                add("bpm", float(value))
            elif normalized_field.endswith("_fps") or normalized_field in {"fps", "frame_rate"}:
                add("fps", float(value))
            elif normalized_field.endswith(("_fraction", "_score", "_risk", "_coverage")):
                numeric = float(value)
                if 0.0 <= numeric <= 1.0:
                    add("%", numeric * 100.0)
            elif normalized_field.endswith(("_percent", "_percentage", "_pct")):
                add("%", float(value))
            return
        if isinstance(value, str) and field in trusted_numeric_text:
            for number, unit in re.findall(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)\s*(BPM|fps|%)", value, re.IGNORECASE):
                add(unit.lower(), float(number))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, field=str(key))
        elif isinstance(value, list):
            for child in value:
                visit(child, field=field)

    visit(tool_results)
    return allowed


def contains_disallowed_clinical_advice(answer: str) -> bool:
    """Allow explicit boundary disclaimers while rejecting affirmative advice."""

    for sentence in re.split(r"[.!?;。！？；]+", answer.lower()):
        matches = [
            match
            for pattern in CLINICAL_PATTERNS
            for match in [re.search(pattern, sentence, flags=re.IGNORECASE)]
            if match is not None
        ]
        if not matches:
            continue

        scrubbed = sentence
        for pattern in SAFE_NEGATED_CLINICAL_PATTERNS:
            scrubbed = re.sub(pattern, "", scrubbed, flags=re.IGNORECASE)
        if not any(re.search(pattern, scrubbed, flags=re.IGNORECASE) for pattern in CLINICAL_PATTERNS):
            continue

        first_clinical = min(match.start() for match in matches)
        prefix = sentence[:first_clinical]
        denial = re.search(
            r"(?:\bcannot\b|\bcan't\b|\bwill not\b|\bwon't\b|\bdoes not\b|\bdoesn't\b|\bdo not\b|\bdon't\b|不能|不会|不可以|无法|不构成|并非)",
            prefix,
            flags=re.IGNORECASE,
        )
        if denial and not re.search(r"\b(?:but|however|yet)\b|但是|但|不过|然而", sentence[denial.end() :]):
            continue
        return True
    return False


def validate_generated_answer(
    answer: str,
    *,
    case: dict[str, Any] | None,
    evidence_ids: set[str],
    tool_results: list[dict[str, Any]],
) -> tuple[bool, list[str], str | None]:
    checks: list[str] = []
    normalized = answer.strip()
    if not normalized:
        return False, checks, "empty model answer"
    checks.append("non-empty answer")

    lower = normalized.lower()
    if contains_disallowed_clinical_advice(normalized):
        return False, checks, "model answer crossed the clinical-advice boundary"
    checks.append("clinical-advice boundary preserved")

    cited = set(re.findall(r"\[(E\d+)\]", normalized))
    if evidence_ids and (not cited or not cited.issubset(evidence_ids)):
        return False, checks, "model answer did not cite only supplied evidence identifiers"
    checks.append("evidence identifiers valid")

    if case:
        decision = str(case.get("decision") or "")
        wrong_states = {"release", "review", "retake"} - {decision}
        explicit_pattern = r"(?:decision|state|状态|结论)\s*(?:is|为|：|:)\s*({})".format("|".join(sorted(wrong_states)))
        if re.search(explicit_pattern, lower, flags=re.IGNORECASE):
            return False, checks, "model answer contradicted the recorded output state"
        checks.append("recorded output state preserved")
        if decision != "release" and re.search(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?\s*bpm", lower):
            return False, checks, "model answer included BPM in a non-release state"
        checks.append("non-release HR withholding preserved")

    allowed = collect_allowed_measurements(tool_results)
    measured = [
        (float(value), unit.lower())
        for value, unit in re.findall(
            r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)\s*(BPM|fps|%)",
            normalized,
            flags=re.IGNORECASE,
        )
    ]
    for value, unit in measured:
        candidates = {round(value, precision) for precision in (1, 2, 3, 4)}
        if not candidates & allowed[unit]:
            return False, checks, f"unsupported numeric claim: {value} {unit}"
    checks.append("numeric claims grounded in tool output")
    return True, checks, None


def safe_boundary(language: str) -> str:
    if language.lower().startswith("zh"):
        return "仅用于回顾性研究工作流；不构成诊断、急救告警、医疗器械决策或经验证的临床自主放行。"
    return CLAIM_BOUNDARY


def compact_json(value: Any, *, limit: int = 12000) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return encoded if len(encoded) <= limit else encoded[:limit] + "..."
