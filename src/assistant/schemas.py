from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AssistantRole(str, Enum):
    operator = "operator"
    reviewer = "reviewer"
    clinician = "clinician"
    admin = "admin"


class AssistantLanguage(str, Enum):
    zh = "zh"
    en = "en"


class AssistantMediaKind(str, Enum):
    audio_transcript = "audio_transcript"
    image = "image"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    case_id: str | None = Field(default=None, max_length=160)
    role: AssistantRole = AssistantRole.operator
    language: AssistantLanguage = AssistantLanguage.zh
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    actor: str = Field(default="research-user", min_length=1, max_length=120)
    conversation_id: str | None = Field(default=None, max_length=160)
    allow_action_proposals: bool = False
    media_contexts: list["AssistantMediaContext"] = Field(default_factory=list, max_length=2)

    @field_validator("message", "actor")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class AssistantMediaContext(BaseModel):
    """Sanitized, non-authoritative context derived from transient user media."""

    context_id: str = Field(pattern=r"^media_[a-f0-9]{16,64}$")
    kind: AssistantMediaKind
    source_label: str = Field(min_length=1, max_length=180)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    summary: str = Field(min_length=1, max_length=2600)
    visible_text: str = Field(default="", max_length=1800)
    model: str = Field(default="deterministic-media-intake", max_length=160)
    safety_flags: list[str] = Field(default_factory=list, max_length=10)
    authoritative: Literal[False] = False
    retained: Literal[False] = False

    @field_validator("source_label", "summary", "visible_text", "model")
    @classmethod
    def normalize_media_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class EvidenceReference(BaseModel):
    evidence_id: str
    label: str
    source: str
    value: str = ""
    kind: Literal["case", "report", "policy", "knowledge", "review", "system", "media"]


class DecisionSummary(BaseModel):
    state: Literal["release", "review", "retake"]
    released_hr_bpm: float | None = None
    acquisition_gate: str
    policy_version: str
    hr_withheld: bool


class RecommendedAction(BaseModel):
    label: str
    rationale: str
    verification: str = ""
    source_field: str = ""
    navigation_target: str | None = None
    requires_confirmation: bool = False


class PendingAction(BaseModel):
    token: str
    action_type: Literal["review_update"]
    summary: str
    expires_at: str
    requires_confirmation: bool = True


class AssistantValidation(BaseModel):
    passed: bool
    checks: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class AssistantChatResponse(BaseModel):
    answer: str
    conversation_id: str
    case_id: str | None = None
    role: AssistantRole
    language: AssistantLanguage
    provider: str
    model: str
    degraded: bool
    decision_summary: DecisionSummary | None = None
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    warning_or_boundary: str
    navigation_target: str | None = None
    pending_action: PendingAction | None = None
    tool_trace_id: str
    validation: AssistantValidation


class AssistantHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    provider: str
    model: str
    model_available: bool
    fallback_available: bool = True
    knowledge_chunks: int
    actions_enabled: bool
    claim_boundary: str
    details: str = ""


class MultimodalCapability(BaseModel):
    available: bool
    provider: str
    model: str
    details: str


class AssistantMultimodalHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    image: MultimodalCapability
    speech: MultimodalCapability
    raw_media_retained: Literal[False] = False
    claim_boundary: str


class AudioTranscriptionResponse(BaseModel):
    transcript: str = Field(min_length=1, max_length=4000)
    detected_language: str = Field(min_length=1, max_length=24)
    duration_seconds: float = Field(ge=0)
    quality: Literal["clear", "uncertain"]
    context: AssistantMediaContext
    raw_audio_retained: Literal[False] = False
    warning_or_boundary: str


class ImageAnalysisResponse(BaseModel):
    summary: str = Field(min_length=1, max_length=2600)
    visible_text: str = Field(default="", max_length=1800)
    workflow_relevance: str = Field(default="", max_length=1600)
    safety_flags: list[str] = Field(default_factory=list, max_length=10)
    technical_checks: dict[str, str] = Field(default_factory=dict)
    context: AssistantMediaContext
    degraded: bool = False
    raw_image_retained: Literal[False] = False
    warning_or_boundary: str


class AssistantConfirmRequest(BaseModel):
    token: str = Field(min_length=16, max_length=160)
    actor: str = Field(default="research-user", min_length=1, max_length=120)


class AssistantConfirmResponse(BaseModel):
    status: Literal["confirmed", "rejected", "expired"]
    action_type: str
    case_id: str
    message: str


class ProviderDraft(BaseModel):
    direct_answer: str = Field(min_length=1, max_length=1800)
    evidence_explanation: str = Field(default="", max_length=2400)
    next_step: str = Field(default="", max_length=1400)
    used_evidence_ids: list[str] = Field(default_factory=list, max_length=20)

    @property
    def answer(self) -> str:
        return "\n\n".join(
            section.strip()
            for section in (self.direct_answer, self.evidence_explanation, self.next_step)
            if section.strip()
        )
