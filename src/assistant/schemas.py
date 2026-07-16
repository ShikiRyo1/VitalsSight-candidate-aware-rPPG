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

    @field_validator("message", "actor")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class EvidenceReference(BaseModel):
    evidence_id: str
    label: str
    source: str
    value: str = ""
    kind: Literal["case", "report", "policy", "knowledge", "review", "system"]


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


class AssistantConfirmRequest(BaseModel):
    token: str = Field(min_length=16, max_length=160)
    actor: str = Field(default="research-user", min_length=1, max_length=120)


class AssistantConfirmResponse(BaseModel):
    status: Literal["confirmed", "rejected", "expired"]
    action_type: str
    case_id: str
    message: str


class ProviderDraft(BaseModel):
    answer: str = Field(min_length=1, max_length=5000)
    used_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
