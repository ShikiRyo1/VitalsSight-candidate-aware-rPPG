from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.assistant.provider import ChatProvider, UnavailableProvider


PROHIBITED_CLINICAL_TERMS = (
    "diagnose",
    "diagnosis",
    "diagnostic conclusion",
    "treatment",
    "prescribe",
    "prescription",
    "clinical clearance",
    "medically safe",
    "确诊",
    "诊断为",
    "治疗方案",
    "处方",
    "临床放行",
)


class ReportNarrativeDraft(BaseModel):
    direct_summary: str = Field(min_length=1, max_length=1200)
    evidence_explanation: str = Field(min_length=1, max_length=2200)
    action_guidance: str = Field(min_length=1, max_length=1600)
    limitations: str = Field(min_length=1, max_length=1600)
    evidence_ids: list[str] = Field(min_length=1, max_length=30)


@dataclass(frozen=True)
class NarrativeEvidence:
    evidence_id: str
    label: str
    value: str
    source: str
    causal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "label": self.label,
            "value": self.value,
            "source": self.source,
            "causal": self.causal,
        }


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def build_narrative_evidence(payload: dict[str, Any]) -> list[NarrativeEvidence]:
    case = payload.get("case") or {}
    action_plan = payload.get("action_plan") or {}
    evidence: list[NarrativeEvidence] = []

    def add(label: str, value: Any, source: str, *, causal: bool = False) -> str:
        item = NarrativeEvidence(
            evidence_id=f"E{len(evidence) + 1}",
            label=_text(label),
            value=_text(value),
            source=source,
            causal=causal,
        )
        evidence.append(item)
        return item.evidence_id

    decision = _text(case.get("decision") or "review")
    add("Output state", decision, "case.decision")
    if decision == "release":
        add("Released HR", f"{case.get('released_hr_bpm')} BPM", "case.released_hr_bpm")
    else:
        add("HR publication status", "withheld", "case.released_hr_bpm")
    add("Acquisition gate", (case.get("preflight") or {}).get("overall") or "not recorded", "case.preflight.overall")
    add("Policy version", case.get("policy_version") or "not recorded", "case.policy_version")
    add("Model version", case.get("model_version") or "not recorded", "case.model_version")

    review_reason = _text(case.get("review_reason"))
    if review_reason:
        add("Recorded state reason", review_reason, "case.review_reason", causal=decision != "release")
    for item in action_plan.get("evidence") or []:
        status = _text(item.get("status")).lower()
        causal = status not in {
            "within target",
            "supports release",
            "pass",
            "passed",
            "not evaluated",
        }
        add(
            item.get("signal") or "Evidence signal",
            f"{item.get('observed')} | {item.get('status')} | {item.get('reason')}",
            _text(item.get("source_field") or "action_plan.evidence"),
            causal=causal,
        )
    steps = action_plan.get("steps") or []
    for item in steps[:4]:
        add(
            f"Recommended action {item.get('step') or ''}".strip(),
            f"{item.get('action')} | verify: {item.get('verification')}",
            _text(item.get("source_field") or "action_plan.steps"),
            causal=True,
        )
    add("Evidence boundary", payload.get("claim_boundary") or "", "claim_boundary")
    longitudinal = payload.get("longitudinal") or {}
    if longitudinal:
        add(
            "Longitudinal state counts",
            json.dumps(longitudinal.get("state_counts") or {}, ensure_ascii=False, sort_keys=True),
            "longitudinal.state_counts",
        )
    return evidence


def _citation_ids(text: str) -> set[str]:
    cited: set[str] = set()
    for block in re.findall(r"\[([^\]]+)\]", text):
        cited.update(re.findall(r"\bE\d+\b", block))
    return cited


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if item.strip()]


def _number_tokens(text: str) -> set[str]:
    stripped = re.sub(r"\[[^\]]+\]", "", text)
    return {match.group(0).lstrip("+") for match in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", stripped)}


def validate_narrative(
    draft: ReportNarrativeDraft,
    *,
    payload: dict[str, Any],
    catalog: list[NarrativeEvidence],
) -> list[str]:
    errors: list[str] = []
    catalog_by_id = {item.evidence_id: item for item in catalog}
    sections = {
        "direct_summary": draft.direct_summary,
        "evidence_explanation": draft.evidence_explanation,
        "action_guidance": draft.action_guidance,
        "limitations": draft.limitations,
    }
    all_text = "\n".join(sections.values())
    all_citations = _citation_ids(all_text)
    declared = set(draft.evidence_ids)
    invalid = sorted((all_citations | declared) - set(catalog_by_id))
    if invalid:
        errors.append(f"unknown evidence identifiers: {', '.join(invalid)}")
    if not all_citations:
        errors.append("no inline evidence citations")
    if not all_citations.issubset(declared):
        errors.append("inline citations are not fully declared in evidence_ids")
    for section, text in sections.items():
        for sentence in _sentences(text):
            if not _citation_ids(sentence):
                errors.append(f"uncited sentence in {section}")
                break

    allowed_numbers: set[str] = set()
    for item in catalog:
        allowed_numbers.update(_number_tokens(f"{item.label} {item.value} {item.source}"))
    invented = sorted(_number_tokens(all_text) - allowed_numbers)
    if invented:
        errors.append(f"unsupported numeric tokens: {', '.join(invented)}")

    lower = all_text.lower()
    clinical_scan = lower
    for allowed_boundary in (
        "not a diagnosis",
        "not diagnosis",
        "does not diagnose",
        "not a treatment",
        "not treatment",
        "does not prescribe",
        "不构成临床诊断",
        "不构成诊断",
        "不是诊断",
        "不提供治疗",
        "不构成治疗建议",
        "不构成临床结论",
    ):
        clinical_scan = clinical_scan.replace(allowed_boundary, "")
    prohibited = [term for term in PROHIBITED_CLINICAL_TERMS if term.lower() in clinical_scan]
    if prohibited:
        errors.append(f"prohibited clinical wording: {', '.join(prohibited)}")

    case = payload.get("case") or {}
    decision = str(case.get("decision") or "review")
    if decision != "release":
        candidate = case.get("selected_candidate_hr_bpm")
        if isinstance(candidate, (int, float)):
            candidate_tokens = {
                f"{float(candidate):g}",
                f"{float(candidate):.1f}",
                f"{float(candidate):.2f}",
            }
            if any(re.search(rf"\b{re.escape(token)}\s*BPM\b", all_text, flags=re.I) for token in candidate_tokens):
                errors.append("withheld candidate HR appears in narrative")
        causal_ids = {item.evidence_id for item in catalog if item.causal}
        if not (_citation_ids(draft.evidence_explanation) & causal_ids):
            errors.append("non-release explanation does not cite a causal evidence item")
    return errors


def _fallback_narrative(
    payload: dict[str, Any],
    catalog: list[NarrativeEvidence],
    *,
    language: str,
) -> ReportNarrativeDraft:
    by_source = {item.source: item for item in catalog}
    case = payload.get("case") or {}
    decision = str(case.get("decision") or "review")
    state = by_source["case.decision"]
    publication = by_source["case.released_hr_bpm"]
    boundary = by_source["claim_boundary"]
    causal = next((item for item in catalog if item.causal), state)
    action = next((item for item in catalog if item.label.startswith("Recommended action")), causal)
    compact = lambda value: re.sub(r"[.!?。！？]+\s*", "; ", value).strip(" ;")
    if language == "zh":
        if decision == "release":
            direct = f"当前证据流程已发布心率结果 {publication.value} [{state.evidence_id}, {publication.evidence_id}]。"
        elif decision == "retake":
            direct = f"本次采集未发布心率结果，当前输出为重新采集 [{state.evidence_id}, {publication.evidence_id}]。"
        else:
            direct = f"本次分析未发布心率结果，当前输出为人工复核 [{state.evidence_id}, {publication.evidence_id}]。"
        explanation = f"该状态由已记录的证据与策略条件共同形成：{compact(causal.value)} [{causal.evidence_id}]。"
        guidance = f"建议按报告中的可验证步骤处理：{compact(action.value)} [{action.evidence_id}]。"
        limitations = f"本说明仅解释研究工作流证据，不构成临床结论：{compact(boundary.value)} [{boundary.evidence_id}]。"
    else:
        if decision == "release":
            direct = f"The evidence workflow released an HR output of {publication.value} [{state.evidence_id}, {publication.evidence_id}]."
        elif decision == "retake":
            direct = f"No HR output was released; the current state requests a new acquisition [{state.evidence_id}, {publication.evidence_id}]."
        else:
            direct = f"No HR output was released; the current state requires human review [{state.evidence_id}, {publication.evidence_id}]."
        explanation = f"The recorded evidence and policy conditions support this state: {compact(causal.value)} [{causal.evidence_id}]."
        guidance = f"Follow the report's verifiable action: {compact(action.value)} [{action.evidence_id}]."
        limitations = f"This explanation is limited to the research workflow evidence: {compact(boundary.value)} [{boundary.evidence_id}]."
    used = sorted(
        _citation_ids("\n".join((direct, explanation, guidance, limitations))),
        key=lambda value: int(value[1:]),
    )
    return ReportNarrativeDraft(
        direct_summary=direct,
        evidence_explanation=explanation,
        action_guidance=guidance,
        limitations=limitations,
        evidence_ids=used,
    )


class EvidenceBoundedReportNarrator:
    def __init__(self, provider: ChatProvider | None = None) -> None:
        self.provider = provider or UnavailableProvider()

    def generate(self, payload: dict[str, Any], *, language: str = "en") -> dict[str, Any]:
        language = "zh" if language.lower().startswith("zh") else "en"
        catalog = build_narrative_evidence(payload)
        status = self.provider.status()
        fallback_reason = ""
        provider_errors: list[str] = []
        draft: ReportNarrativeDraft | None = None
        if status.available:
            prompt = (
                "Write a concise VitalsSight report explanation using only the supplied evidence catalog. "
                "Do not infer a diagnosis, treatment, safety guarantee, or unlisted number. Every sentence must "
                "end with one or more inline evidence citations such as [E1] or [E1, E2]. For review or retake, "
                "never state a candidate or withheld HR value. Passing checks are context, not failure causes. "
                f"Respond in {'Chinese' if language == 'zh' else 'English'} and match the JSON schema exactly.\n\n"
                f"OUTPUT STATE: {(payload.get('case') or {}).get('decision')}\n"
                f"EVIDENCE CATALOG:\n{json.dumps([item.to_dict() for item in catalog], ensure_ascii=False)}"
            )
            try:
                reply = self.provider.chat(
                    [
                        {"role": "system", "content": "You explain deterministic research workflow evidence without changing it."},
                        {"role": "user", "content": prompt},
                    ],
                    response_schema=ReportNarrativeDraft.model_json_schema(),
                )
                draft = ReportNarrativeDraft.model_validate_json(reply.content)
                provider_errors = validate_narrative(draft, payload=payload, catalog=catalog)
                if provider_errors:
                    fallback_reason = "provider output failed evidence validation"
                    draft = None
            except (RuntimeError, ValidationError, json.JSONDecodeError, ValueError) as error:
                fallback_reason = f"provider output unavailable: {type(error).__name__}"
        else:
            fallback_reason = status.details or "model provider unavailable"

        mode = "model" if draft is not None else "deterministic_fallback"
        if draft is None:
            draft = _fallback_narrative(payload, catalog, language=language)
        final_errors = validate_narrative(draft, payload=payload, catalog=catalog)
        if final_errors:
            raise ValueError(f"The deterministic report narrative failed validation: {final_errors}")
        return {
            **draft.model_dump(),
            "status": "draft",
            "mode": mode,
            "provider": status.provider,
            "model": status.model,
            "language": language,
            "generated_at": datetime.now(UTC).isoformat(),
            "catalog": [item.to_dict() for item in catalog],
            "validation": {
                "passed": True,
                "checks": [
                    "inline citations resolved",
                    "numeric claims grounded",
                    "non-release HR withheld",
                    "clinical wording boundary enforced",
                ],
                "provider_errors": provider_errors,
                "fallback_reason": fallback_reason,
            },
        }
