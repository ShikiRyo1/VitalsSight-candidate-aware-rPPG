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


SIGNAL_ALIASES = {
    "illumination": ("illumination", "lighting", "light level", "光照", "亮度", "补光"),
    "motion": ("motion", "movement", "stability", "运动", "动作", "稳定性"),
    "duration": ("duration", "recording time", "video length", "时长", "录制时间", "视频长度"),
    "frame rate": ("frame rate", "fps", "帧率"),
    "resolution": ("resolution", "分辨率"),
    "face visibility": ("face visibility", "face coverage", "人脸可见", "面部可见", "人脸覆盖"),
    "candidate count": ("candidate count", "number of candidates", "候选数量", "候选数"),
    "candidate agreement": ("candidate agreement", "cross-route agreement", "候选一致", "跨路线一致"),
    "candidate construction": ("candidate construction", "candidate stage", "候选构建", "候选阶段"),
    "competing candidate tracks": ("competing candidate tracks", "competing tracks", "竞争候选轨迹", "候选轨迹冲突"),
    "harmonic ambiguity": ("harmonic ambiguity", "harmonic risk", "谐波歧义", "谐波风险"),
    "window consistency": ("window consistency", "window-level consistency", "窗口一致性"),
    "selector support": ("selector support", "selection support", "选择器支持", "选择支持", "选择置信"),
    "face-landmark backend": ("face-landmark backend", "landmark backend", "人脸关键点后端", "关键点后端"),
    "candidate route omissions": ("candidate route omissions", "route omissions", "候选路线缺失", "路线缺失"),
}


def _signal_aliases(row: dict[str, Any]) -> tuple[str, ...]:
    signal = str(row.get("signal") or "").strip().lower()
    source_field = str(row.get("source_field") or "").strip().lower()
    aliases: list[str] = []
    if signal:
        aliases.append(signal)
        aliases.extend(SIGNAL_ALIASES.get(signal, ()))
    field_tail = source_field.rsplit(".", 1)[-1].replace("_", " ")
    if field_tail and field_tail not in {"checks", "decision", "claim boundary"}:
        aliases.append(field_tail)
    for key, values in SIGNAL_ALIASES.items():
        if key in signal or key.replace(" ", "_") in source_field:
            aliases.extend(values)
    return tuple(dict.fromkeys(item.lower() for item in aliases if len(item.strip()) >= 2))


def _action_plan_rows(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result in tool_results:
        plans: list[dict[str, Any]] = []
        if isinstance(result.get("action_plan"), dict):
            plans.append(result["action_plan"])
        if isinstance(result.get("evidence"), list):
            plans.append({"evidence": result["evidence"]})
        for plan in plans:
            for item in plan.get("evidence") or []:
                if not isinstance(item, dict):
                    continue
                key = (str(item.get("source_field") or item.get("signal") or ""), str(item.get("status") or ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(item)
    return rows


def _safe_noncausal_statement(clause: str, alias: str) -> bool:
    escaped = re.escape(alias)
    safe_after = (
        r"within (?:the )?target|passed|met (?:the )?(?:documented )?target|"
        r"did not (?:cause|trigger|fail)|(?:was|is) not (?:a )?(?:cause|driver|failure)|"
        r"not causal|no corrective action|does not require correction|"
        r"目标内|已通过|达标|符合.{0,12}(?:目标|阈值)|未触发|不是.{0,8}(?:原因|驱动项)|"
        r"并非.{0,8}(?:原因|驱动项)|无需.{0,8}(?:纠正|改善|调整)"
    )
    safe_before = (
        r"within (?:the )?target|passed|not caused by|not triggered by|"
        r"目标内|已通过|达标|不是由|并非由|未由"
    )
    return bool(
        re.search(rf"{escaped}.{{0,55}}(?:{safe_after})", clause, flags=re.IGNORECASE)
        or re.search(rf"(?:{safe_before}).{{0,45}}{escaped}", clause, flags=re.IGNORECASE)
    )


def causal_alignment_error(answer: str, tool_results: list[dict[str, Any]]) -> str | None:
    """Reject causal or corrective claims about checks that did not trigger the stored action plan."""

    rows = _action_plan_rows(tool_results)
    non_drivers = [
        row
        for row in rows
        if str(row.get("status") or "").strip().lower() not in {"triggered", "warning"}
    ]
    if not non_drivers:
        return None

    clauses = [
        item.strip()
        for item in re.split(
            r"[.!?;。！？；\n]+|\b(?:but|while|whereas|although)\b|(?:但是|但|而|同时|且)",
            answer.lower(),
            flags=re.IGNORECASE,
        )
        if item.strip()
    ]
    causal_terms = (
        r"because|due to|caus(?:e|ed|ing)|trigger(?:ed|ing)?|contribut(?:e|ed|ing)|"
        r"driver|fail(?:ed|ure|ing)?|insufficient|extremely low|too low|too high|"
        r"below (?:the )?(?:target|threshold)|above (?:the )?(?:target|threshold)|"
        r"did not meet|outside (?:the )?target|needs? correction|should be corrected|"
        r"improve|increase|decrease|reduce|adjust|fix|"
        r"因为|由于|导致|触发|驱动|失败|不足|过低|极低|过高|低于.{0,10}(?:目标|阈值)|"
        r"高于.{0,10}(?:目标|阈值)|未达到|不达标|需要.{0,10}(?:纠正|改善|提高|降低|调整)|"
        r"应当.{0,10}(?:纠正|改善|提高|降低|调整)|改善|提高|降低|纠正|修复|调整"
    )
    for row in non_drivers:
        for alias in _signal_aliases(row):
            escaped = re.escape(alias)
            for clause in clauses:
                if not re.search(escaped, clause, flags=re.IGNORECASE):
                    continue
                if _safe_noncausal_statement(clause, alias):
                    continue
                if re.search(rf"(?:{causal_terms}).{{0,55}}{escaped}|{escaped}.{{0,55}}(?:{causal_terms})", clause, flags=re.IGNORECASE):
                    return str(row.get("signal") or row.get("source_field") or alias)
    return None


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

    causal_error = causal_alignment_error(normalized, tool_results)
    if causal_error:
        return False, checks, f"model answer treated a non-driver as causal: {causal_error}"
    checks.append("causal attribution matches triggered evidence")

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
