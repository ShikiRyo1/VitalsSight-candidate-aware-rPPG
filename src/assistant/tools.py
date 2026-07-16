from __future__ import annotations

import os
from typing import Any

from src.assistant.audit import AssistantAuditStore
from src.assistant.retrieval import KnowledgeIndex
from src.assistant.schemas import AssistantRole, PendingAction
from src.product.console_service import (
    build_action_plan,
    build_attribution,
    build_report_payload,
    case_quality_snapshot,
    ensure_output_contract,
    sanitize_report_value,
)
from src.product.console_store import ConsoleStore


class ToolExecutionError(ValueError):
    pass


def _bounded_text(value: Any, *, limit: int = 600) -> str:
    return str(value or "").strip()[:limit]


class AssistantTools:
    """Whitelisted access to VitalsSight evidence; no direct model-to-store access."""

    def __init__(
        self,
        store: ConsoleStore,
        knowledge: KnowledgeIndex,
        audit_store: AssistantAuditStore,
        *,
        actions_enabled: bool | None = None,
    ) -> None:
        self.store = store
        self.knowledge = knowledge
        self.audit_store = audit_store
        configured = os.getenv("VITALSSIGHT_ASSISTANT_ACTIONS_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.actions_enabled = configured if actions_enabled is None else bool(actions_enabled)

    def specs(self, *, role: AssistantRole, allow_action_proposals: bool) -> list[dict[str, Any]]:
        tools = [
            self._spec(
                "list_cases",
                "List de-identified case summaries, optionally filtered by output state.",
                {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["release", "review", "retake", "all"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "additionalProperties": False,
                },
            ),
            self._spec(
                "get_case",
                "Read the evidence-bounded summary, quality, candidate, attribution and next action for one case.",
                {
                    "type": "object",
                    "properties": {"case_id": {"type": "string", "minLength": 1, "maxLength": 160}},
                    "required": ["case_id"],
                    "additionalProperties": False,
                },
            ),
            self._spec(
                "get_report_summary",
                "Read the structured evidence report summary for one case.",
                {
                    "type": "object",
                    "properties": {"case_id": {"type": "string", "minLength": 1, "maxLength": 160}},
                    "required": ["case_id"],
                    "additionalProperties": False,
                },
            ),
            self._spec(
                "get_review",
                "Read the current review record for a review or retake case.",
                {
                    "type": "object",
                    "properties": {"case_id": {"type": "string", "minLength": 1, "maxLength": 160}},
                    "required": ["case_id"],
                    "additionalProperties": False,
                },
            ),
            self._spec(
                "validate_output_contract",
                "Verify that release publishes finite HR and non-release states withhold HR.",
                {
                    "type": "object",
                    "properties": {"case_id": {"type": "string", "minLength": 1, "maxLength": 160}},
                    "required": ["case_id"],
                    "additionalProperties": False,
                },
            ),
            self._spec(
                "search_help",
                "Search the versioned, content-hashed local workflow, quality, report, privacy and troubleshooting guidance.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 6},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ]
        if self.actions_enabled and allow_action_proposals and role in {AssistantRole.reviewer, AssistantRole.admin}:
            tools.append(
                self._spec(
                    "prepare_review_update",
                    "Prepare, but do not execute, a review update. A human must confirm the returned token.",
                    {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string", "minLength": 1, "maxLength": 160},
                            "status": {"type": "string", "enum": ["open", "in_review", "waiting_retake", "closed"]},
                            "priority": {"type": "string", "enum": ["urgent", "high", "routine", "low"]},
                            "assignee": {"type": "string", "maxLength": 120},
                            "note": {"type": "string", "maxLength": 1000},
                            "resolution": {"type": "string", "maxLength": 600},
                        },
                        "required": ["case_id", "status", "priority"],
                        "additionalProperties": False,
                    },
                )
            )
        return tools

    @staticmethod
    def _spec(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        role: AssistantRole,
        allow_action_proposals: bool,
        actor: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        allowed = {item["function"]["name"] for item in self.specs(role=role, allow_action_proposals=allow_action_proposals)}
        if name not in allowed:
            raise ToolExecutionError(f"Tool is not allowed for this role and mode: {name}")
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise ToolExecutionError(f"Tool is not implemented: {name}")
        result = handler(arguments, actor=actor, conversation_id=conversation_id)
        return sanitize_report_value(result)

    def _case(self, case_id: str) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        if not case:
            raise ToolExecutionError(f"Case not found: {case_id}")
        return ensure_output_contract(case)

    def _list_cases(self, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        decision = str(arguments.get("decision") or "all").lower()
        if decision not in {"all", "release", "review", "retake"}:
            raise ToolExecutionError("decision must be release, review, retake, or all")
        limit = max(1, min(int(arguments.get("limit") or 8), 20))
        rows = []
        for case in self.store.list_cases():
            if decision != "all" and case.get("decision") != decision:
                continue
            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "display_id": case.get("display_id"),
                    "decision": case.get("decision"),
                    "priority": case.get("priority"),
                    "created_at": case.get("created_at"),
                    "released_hr_bpm": case.get("released_hr_bpm") if case.get("decision") == "release" else None,
                }
            )
            if len(rows) >= limit:
                break
        return {"count": len(rows), "items": rows, "decision_filter": decision}

    def _get_case(self, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        case = self._case(_bounded_text(arguments.get("case_id"), limit=160))
        plan = build_action_plan(case)
        attribution = build_attribution(case)
        runtime = case.get("runtime_metadata") or {}
        return {
            "case": {
                "case_id": case["case_id"],
                "display_id": case.get("display_id"),
                "decision": case["decision"],
                "priority": case.get("priority"),
                "released_hr_bpm": case.get("released_hr_bpm") if case["decision"] == "release" else None,
                "hr_withheld": case["decision"] != "release",
                "quality_score": case.get("quality_score"),
                "face_coverage": case.get("face_coverage"),
                "illumination_score": case.get("illumination_score"),
                "motion_score": case.get("motion_score"),
                "candidate_count": case.get("candidate_count"),
                "agreement_fraction": case.get("agreement_fraction"),
                "harmonic_risk": case.get("harmonic_risk"),
                "review_reason": case.get("review_reason"),
                "recommended_action": case.get("recommended_action"),
                "policy_version": case.get("policy_version"),
                "model_version": case.get("model_version"),
                "quality": case_quality_snapshot(case),
                "runtime": {
                    "detector_backend": runtime.get("detector_backend"),
                    "detector_model_integrity": runtime.get("detector_model_integrity"),
                    "route_failure_count": runtime.get("route_failure_count"),
                },
                "candidates": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "candidate_bpm": item.get("candidate_bpm"),
                        "method": item.get("method"),
                        "region": item.get("region"),
                        "support": item.get("support"),
                        "score": item.get("score"),
                    }
                    for item in (case.get("candidates") or [])[:12]
                ],
            },
            "action_plan": plan,
            "attribution": {
                "primary_review_drivers": attribution.get("primary_review_drivers", []),
                "primary_release_support": attribution.get("primary_release_support", []),
                "boundary": attribution.get("boundary"),
            },
        }

    def _get_report_summary(self, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        case_id = _bounded_text(arguments.get("case_id"), limit=160)
        case = self._case(case_id)
        review = next((item for item in self.store.list_reviews() if item["case_id"] == case_id), None)
        payload = build_report_payload(case, review=review, audit_events=self.store.audit_events(case_id))
        return {
            "report_version": payload["report_version"],
            "generated_at": payload["generated_at"],
            "case": {
                "case_id": case_id,
                "decision": case["decision"],
                "released_hr_bpm": case.get("released_hr_bpm") if case["decision"] == "release" else None,
                "hr_withheld": case["decision"] != "release",
                "policy_version": case.get("policy_version"),
            },
            "interpretation": {
                "headline": payload["action_plan"].get("headline"),
                "rationale": payload["action_plan"].get("rationale"),
                "recommendation": payload["action_plan"].get("recommendation"),
                "expected_outcome": payload["action_plan"].get("expected_outcome"),
            },
            "evidence": payload["action_plan"].get("evidence", []),
            "steps": payload["action_plan"].get("steps", []),
            "review": {
                key: (review or {}).get(key)
                for key in ("status", "priority", "assignee", "resolution", "updated_at")
            },
            "claim_boundary": payload["claim_boundary"],
        }

    def _get_review(self, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        case_id = _bounded_text(arguments.get("case_id"), limit=160)
        self._case(case_id)
        review = next((item for item in self.store.list_reviews() if item["case_id"] == case_id), None)
        if not review:
            return {"case_id": case_id, "status": "not_applicable", "message": "Release cases do not have an open review record."}
        return {
            "case_id": case_id,
            "status": review.get("status"),
            "priority": review.get("priority"),
            "assignee": review.get("assignee"),
            "note": review.get("note"),
            "resolution": review.get("resolution"),
            "updated_at": review.get("updated_at"),
        }

    def _validate_output_contract(self, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        case = self._case(_bounded_text(arguments.get("case_id"), limit=160))
        decision = str(case["decision"])
        released = case.get("released_hr_bpm")
        checks = {
            "supported_state": decision in {"release", "review", "retake"},
            "release_has_finite_hr": decision != "release" or isinstance(released, (int, float)),
            "non_release_withholds_hr": decision == "release" or released is None,
            "policy_version_present": bool(case.get("policy_version")),
        }
        return {"case_id": case["case_id"], "decision": decision, "checks": checks, "passed": all(checks.values())}

    def _search_help(self, arguments: dict[str, Any], **_: Any) -> dict[str, Any]:
        query = _bounded_text(arguments.get("query"), limit=500)
        if not query:
            raise ToolExecutionError("query is required")
        limit = max(1, min(int(arguments.get("limit") or 4), 6))
        return {"query": query, "items": self.knowledge.search(query, limit=limit)}

    def _prepare_review_update(
        self,
        arguments: dict[str, Any],
        *,
        actor: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        case_id = _bounded_text(arguments.get("case_id"), limit=160)
        case = self._case(case_id)
        if case["decision"] == "release":
            raise ToolExecutionError("A release case has no review record to update")
        payload = {
            "status": _bounded_text(arguments.get("status"), limit=40),
            "priority": _bounded_text(arguments.get("priority"), limit=40),
            "assignee": _bounded_text(arguments.get("assignee"), limit=120),
            "note": _bounded_text(arguments.get("note"), limit=1000),
            "resolution": _bounded_text(arguments.get("resolution"), limit=600),
        }
        if payload["status"] not in {"open", "in_review", "waiting_retake", "closed"}:
            raise ToolExecutionError("Unsupported review status")
        if payload["priority"] not in {"urgent", "high", "routine", "low"}:
            raise ToolExecutionError("Unsupported review priority")
        pending: PendingAction = self.audit_store.prepare_review_update(
            case_id=case_id,
            actor=actor,
            conversation_id=conversation_id,
            payload=payload,
        )
        return {"pending_action": pending.model_dump(mode="json"), "executed": False}
