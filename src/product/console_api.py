from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.assistant import (
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantConfirmRequest,
    AssistantConfirmResponse,
    AssistantHealthResponse,
    AssistantMultimodalHealthResponse,
    AudioTranscriptionResponse,
    ImageAnalysisResponse,
    MultimodalAssistantService,
    AssistantOrchestrator,
)
from src.assistant.multimodal import MediaProcessingError
from src.assistant.provider import ChatProvider
from src.product.console_service import (
    CLAIM_BOUNDARY,
    REPORT_VERSION,
    build_report_payload,
    build_report_pdf,
    case_from_preflight,
    case_from_runtime_failure,
    make_demo_cases,
    preflight_from_decode_error,
    run_uploaded_video,
    sanitize_report_value,
    video_preflight,
)
from src.product.build_identity import path_fingerprint, source_build_identity
from src.product.auth import AuthError, AuthSettings, IdentityResolver, require_roles
from src.product.console_store import ConsoleStore, ScopedConsoleStore
from src.product.identity import (
    IdentityContext,
    ROLE_AUDITOR,
    ROLE_OPERATOR,
    ROLE_ORG_ADMIN,
    ROLE_PARTICIPANT,
    ROLE_RESEARCHER,
    ROLE_REVIEWER,
    ROLE_SERVICE,
)
from src.product.reporting import (
    build_fhir_bundle,
    build_longitudinal_context,
    enrich_report_payload,
    report_for_audience,
    report_content_sha256,
    report_version_sha256,
)
from src.product.report_narrative import EvidenceBoundedReportNarrator


class ReviewUpdate(BaseModel):
    status: str = Field(pattern="^(open|in_review|waiting_retake|closed)$")
    priority: str = Field(pattern="^(urgent|high|routine|low)$")
    assignee: str = ""
    note: str = ""
    resolution: str = ""
    actor: str = "api-user"


class ParticipantCreate(BaseModel):
    pseudonym: str = Field(min_length=1, max_length=96)
    study_id: str = Field(default="", max_length=96)
    external_reference_hash: str = Field(default="", max_length=128)


class ConsentCreate(BaseModel):
    purpose: str = Field(pattern="^(workflow_validation|algorithm_evaluation|research_demo)$")
    document_version: str = Field(min_length=1, max_length=96)
    details: dict[str, Any] = Field(default_factory=dict)


class ReportVersionCreate(BaseModel):
    audience: str = Field(default="reviewer", pattern="^(participant|operator|reviewer|research)$")
    language: str = Field(default="en", pattern="^(en|zh)$")
    supersedes_report_id: str = Field(default="", max_length=160)
    generate_narrative: bool = True


def default_db_path() -> Path:
    configured = os.getenv("VITALSSIGHT_DB_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "runtime" / "vitalsight_console.db"


def create_app(
    db_path: str | Path | None = None,
    *,
    seed_demo: bool = True,
    assistant_provider: ChatProvider | None = None,
    assistant_actions_enabled: bool | None = None,
    multimodal_service: MultimodalAssistantService | None = None,
    auth_settings: AuthSettings | None = None,
) -> FastAPI:
    resolved_db_path = Path(db_path or default_db_path())
    store = ConsoleStore(resolved_db_path)
    assistant = AssistantOrchestrator(
        store,
        db_path=resolved_db_path,
        provider=assistant_provider,
        actions_enabled=assistant_actions_enabled,
    )
    multimodal = multimodal_service or MultimodalAssistantService()
    report_narrator = EvidenceBoundedReportNarrator(assistant.provider)
    auth = IdentityResolver(auth_settings or AuthSettings.from_env())
    upload_dir = Path(os.getenv("VITALSSIGHT_UPLOAD_DIR", resolved_db_path.parent / "uploads" / "api"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_upload_bytes = int(os.getenv("VITALSSIGHT_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
    if seed_demo and not store.list_cases():
        for case in make_demo_cases():
            store.upsert_case(case, actor="demo-seed")

    app = FastAPI(
        title="VitalsSight Evidence API",
        version="1.1.0",
        description=(
            "Research product API for candidate-aware adult HR evidence, review workflow, report export, "
            "and transient voice/image assistant intake. It does not provide clinical decisions."
        ),
    )
    app.state.store = store
    app.state.assistant = assistant
    app.state.multimodal = multimodal
    app.state.report_narrator = report_narrator
    app.state.auth = auth
    bearer = HTTPBearer(auto_error=False)

    def resolve_identity(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> IdentityContext:
        authorization = (
            f"{credentials.scheme} {credentials.credentials}" if credentials is not None else None
        )
        try:
            identity = auth.resolve(authorization, headers=request.headers)
            store.ensure_identity(identity)
            return identity
        except AuthError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    def allow_roles(*roles: str):
        def dependency(identity: IdentityContext = Depends(resolve_identity)) -> IdentityContext:
            try:
                return require_roles(identity, *roles)
            except AuthError as error:
                raise HTTPException(status_code=error.status_code, detail=error.detail) from error

        return dependency

    read_identity = allow_roles(
        ROLE_PARTICIPANT,
        ROLE_OPERATOR,
        ROLE_REVIEWER,
        ROLE_RESEARCHER,
        ROLE_AUDITOR,
        ROLE_ORG_ADMIN,
        ROLE_SERVICE,
    )
    capture_identity = allow_roles(
        ROLE_PARTICIPANT,
        ROLE_OPERATOR,
        ROLE_REVIEWER,
        ROLE_ORG_ADMIN,
        ROLE_SERVICE,
    )
    reviewer_identity = allow_roles(ROLE_REVIEWER, ROLE_ORG_ADMIN)
    operator_identity = allow_roles(ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ORG_ADMIN)
    audit_identity = allow_roles(ROLE_AUDITOR, ROLE_ORG_ADMIN)
    review_read_identity = allow_roles(
        ROLE_OPERATOR,
        ROLE_REVIEWER,
        ROLE_RESEARCHER,
        ROLE_AUDITOR,
        ROLE_ORG_ADMIN,
    )
    participant_admin_identity = allow_roles(ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ORG_ADMIN)
    consent_identity = allow_roles(ROLE_PARTICIPANT, ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ORG_ADMIN)

    def scoped(identity: IdentityContext) -> ScopedConsoleStore:
        return ScopedConsoleStore(store, identity)

    def scoped_assistant(identity: IdentityContext) -> AssistantOrchestrator:
        return AssistantOrchestrator(
            scoped(identity),
            db_path=resolved_db_path,
            provider=assistant.provider,
            actions_enabled=assistant.tools.actions_enabled,
        )

    def participant_for_identity(
        tenant_store: ScopedConsoleStore,
        identity: IdentityContext,
        participant_id: str,
    ) -> dict[str, Any]:
        resolved_id = participant_id.strip()
        if identity.primary_role == ROLE_PARTICIPANT:
            if not identity.participant_id:
                raise HTTPException(
                    status_code=403,
                    detail="The participant token is not linked to a participant record",
                )
            if resolved_id and resolved_id != identity.participant_id:
                raise HTTPException(status_code=403, detail="Participant access is limited to the linked record")
            resolved_id = identity.participant_id
        if not resolved_id:
            raise HTTPException(status_code=422, detail="participant_id is required")
        participant = tenant_store.get_participant(resolved_id)
        if participant is None:
            raise HTTPException(status_code=404, detail="Participant not found")
        return participant

    def assert_report_audience(identity: IdentityContext, audience: str) -> None:
        if identity.primary_role == ROLE_PARTICIPANT and audience != "participant":
            raise HTTPException(status_code=403, detail="Participants may request participant reports only")
        if identity.primary_role == ROLE_RESEARCHER and audience not in {"research", "operator"}:
            raise HTTPException(status_code=403, detail="The requested report audience is not permitted")

    def governed_report(
        tenant_store: ScopedConsoleStore,
        identity: IdentityContext,
        case_id: str,
        *,
        audience: str,
        language: str,
    ) -> dict[str, Any]:
        assert_report_audience(identity, audience)
        case = tenant_store.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        review = next(
            (item for item in tenant_store.list_reviews() if item["case_id"] == case_id),
            None,
        )
        payload = build_report_payload(
            case,
            review=review,
            audit_events=tenant_store.audit_events(case_id),
        )
        payload = report_for_audience(payload, audience=audience)
        participant_id = str(case.get("participant_id") or identity.participant_id or "")
        participant = tenant_store.get_participant(participant_id) if participant_id else None
        consent = (
            tenant_store.active_consent(
                participant_id=participant_id,
                purpose=str(case.get("purpose") or "workflow_validation"),
            )
            if participant_id
            else None
        )
        related_cases = (
            tenant_store.list_cases(participant_id=participant_id)
            if participant_id
            else [case]
        )
        enriched = enrich_report_payload(
            payload,
            organization_id=identity.organization_id,
            audience=audience,
            language=language,
            participant=participant,
            consent=consent,
            longitudinal=build_longitudinal_context(related_cases, current_case_id=case_id),
        )
        return sanitize_report_value(enriched)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return sanitize_report_value({
            "status": "ok",
            "service": "vitalssight-console",
            "report_version": REPORT_VERSION,
            "build": source_build_identity(),
            "storage": {
                "upload_dir_fingerprint": path_fingerprint(upload_dir),
                "raw_video_policy": "delete_after_analysis",
            },
            "auth": auth.settings.public_dict(),
            "claim_boundary": CLAIM_BOUNDARY,
        })

    @app.get("/api/v1/cases")
    def list_cases(identity: IdentityContext = Depends(read_identity)) -> dict[str, Any]:
        tenant_store = scoped(identity)
        cases = tenant_store.list_cases()
        tenant_store.log_access(action="case.list", resource_type="case")
        return sanitize_report_value({"count": len(cases), "items": cases, "claim_boundary": CLAIM_BOUNDARY})

    @app.get("/api/v1/organization/context")
    def organization_context(identity: IdentityContext = Depends(read_identity)) -> dict[str, Any]:
        tenant_store = scoped(identity)
        tenant_store.log_access(action="organization.context", resource_type="organization")
        return sanitize_report_value(
            {
                "organization": store.organization(identity.organization_id),
                "identity": identity.to_audit_dict(),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    @app.get("/api/v1/organization/memberships")
    def organization_memberships(
        identity: IdentityContext = Depends(audit_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        items = tenant_store.memberships()
        tenant_store.log_access(action="membership.list", resource_type="membership")
        return sanitize_report_value({"count": len(items), "items": items})

    @app.get("/api/v1/organization/access-events")
    def organization_access_events(
        limit: int = 200,
        identity: IdentityContext = Depends(audit_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        items = tenant_store.access_events(limit=limit)
        tenant_store.log_access(action="access-event.list", resource_type="access-event")
        return sanitize_report_value({"count": len(items), "items": items})

    @app.get("/api/v1/participants")
    def list_participants(
        identity: IdentityContext = Depends(participant_admin_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        items = tenant_store.list_participants()
        tenant_store.log_access(action="participant.list", resource_type="participant")
        return sanitize_report_value({"count": len(items), "items": items})

    @app.post("/api/v1/participants", status_code=201)
    def create_participant(
        participant: ParticipantCreate,
        identity: IdentityContext = Depends(participant_admin_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        try:
            item = tenant_store.upsert_participant(**participant.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        tenant_store.log_access(
            action="participant.create",
            resource_type="participant",
            resource_id=item["participant_id"],
        )
        return sanitize_report_value({"item": item})

    @app.get("/api/v1/participants/{participant_id}")
    def get_participant(
        participant_id: str,
        identity: IdentityContext = Depends(consent_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        item = participant_for_identity(tenant_store, identity, participant_id)
        tenant_store.log_access(
            action="participant.read",
            resource_type="participant",
            resource_id=item["participant_id"],
        )
        return sanitize_report_value({"item": item})

    @app.get("/api/v1/participants/{participant_id}/consents")
    def list_consents(
        participant_id: str,
        identity: IdentityContext = Depends(consent_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        item = participant_for_identity(tenant_store, identity, participant_id)
        consents = tenant_store.list_consents(participant_id=item["participant_id"])
        tenant_store.log_access(
            action="consent.list",
            resource_type="participant",
            resource_id=item["participant_id"],
        )
        return sanitize_report_value({"count": len(consents), "items": consents})

    @app.post("/api/v1/participants/{participant_id}/consents", status_code=201)
    def record_consent(
        participant_id: str,
        consent: ConsentCreate,
        identity: IdentityContext = Depends(consent_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        item = participant_for_identity(tenant_store, identity, participant_id)
        try:
            recorded = tenant_store.record_consent(
                participant_id=item["participant_id"],
                **consent.model_dump(),
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        tenant_store.log_access(
            action="consent.record",
            resource_type="consent",
            resource_id=recorded["consent_id"],
        )
        return sanitize_report_value({"item": recorded})

    @app.post("/api/v1/participants/{participant_id}/consents/{consent_id}/withdraw")
    def withdraw_consent(
        participant_id: str,
        consent_id: str,
        identity: IdentityContext = Depends(consent_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        participant_for_identity(tenant_store, identity, participant_id)
        consent = next(
            (
                item
                for item in tenant_store.list_consents(participant_id=participant_id)
                if item["consent_id"] == consent_id
            ),
            None,
        )
        if consent is None:
            raise HTTPException(status_code=404, detail="Consent not found")
        try:
            tenant_store.withdraw_consent(consent_id)
        except KeyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "withdrawn", "consent_id": consent_id}

    @app.get("/api/v1/assistant/health", response_model=AssistantHealthResponse)
    def assistant_health(
        identity: IdentityContext = Depends(read_identity),
    ) -> AssistantHealthResponse:
        return assistant.health()

    @app.get("/api/v1/assistant/multimodal/health", response_model=AssistantMultimodalHealthResponse)
    def assistant_multimodal_health(
        identity: IdentityContext = Depends(read_identity),
    ) -> AssistantMultimodalHealthResponse:
        return multimodal.health()

    @app.post("/api/v1/assistant/transcribe", response_model=AudioTranscriptionResponse)
    def assistant_transcribe(
        file: UploadFile = File(...),
        language: str = Form("auto"),
        identity: IdentityContext = Depends(capture_identity),
    ) -> AudioTranscriptionResponse:
        try:
            if language not in {"auto", "zh", "en"}:
                raise HTTPException(status_code=422, detail="language must be auto, zh, or en")
            data = file.file.read(multimodal.max_audio_bytes + 1)
            if len(data) > multimodal.max_audio_bytes:
                raise HTTPException(status_code=413, detail="Audio exceeds the configured upload limit")
            response = multimodal.transcribe_audio(
                data,
                filename=file.filename or "voice.wav",
                content_type=file.content_type or "audio/wav",
                language=None if language == "auto" else language,
            )
            scoped(identity).log_access(
                action="assistant.transcribe",
                resource_type="transient_audio",
                details={"raw_media_retained": False},
            )
            return response
        except MediaProcessingError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        finally:
            file.file.close()

    @app.post("/api/v1/assistant/analyze-image", response_model=ImageAnalysisResponse)
    def assistant_analyze_image(
        file: UploadFile = File(...),
        question: str = Form(""),
        language: str = Form("zh"),
        identity: IdentityContext = Depends(capture_identity),
    ) -> ImageAnalysisResponse:
        try:
            if language not in {"zh", "en"}:
                raise HTTPException(status_code=422, detail="language must be zh or en")
            data = file.file.read(multimodal.max_image_bytes + 1)
            if len(data) > multimodal.max_image_bytes:
                raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit")
            response = multimodal.analyze_image(
                data,
                filename=file.filename or "image.png",
                content_type=file.content_type or "application/octet-stream",
                question=question,
                language=language,
            )
            scoped(identity).log_access(
                action="assistant.analyze_image",
                resource_type="transient_image",
                details={"raw_media_retained": False},
            )
            return response
        except MediaProcessingError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        finally:
            file.file.close()

    @app.post("/api/v1/assistant/chat", response_model=AssistantChatResponse)
    def assistant_chat(
        request: AssistantChatRequest,
        identity: IdentityContext = Depends(read_identity),
    ) -> AssistantChatResponse:
        secured_request = request.model_copy(update={"actor": identity.actor})
        response = scoped_assistant(identity).chat(secured_request)
        scoped(identity).log_access(
            action="assistant.chat",
            resource_type="case" if request.case_id else "knowledge",
            resource_id=request.case_id or "",
            details={"trace_id": response.tool_trace_id},
        )
        return response

    @app.post("/api/v1/assistant/confirm", response_model=AssistantConfirmResponse)
    def assistant_confirm(
        request: AssistantConfirmRequest,
        identity: IdentityContext = Depends(reviewer_identity),
    ) -> AssistantConfirmResponse:
        try:
            return scoped_assistant(identity).confirm(request.token, actor=identity.actor)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/assistant/reject", response_model=AssistantConfirmResponse)
    def assistant_reject(
        request: AssistantConfirmRequest,
        identity: IdentityContext = Depends(reviewer_identity),
    ) -> AssistantConfirmResponse:
        try:
            return scoped_assistant(identity).reject(request.token, actor=identity.actor)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/assessments/video", status_code=201)
    def assess_video(
        file: UploadFile = File(...),
        consent_recorded: bool = Form(...),
        purpose: str = Form("workflow_validation"),
        retention_policy: str = Form("delete_after_analysis"),
        participant_id: str = Form(""),
        study_id: str = Form(""),
        consent_document_version: str = Form(""),
        identity: IdentityContext = Depends(capture_identity),
    ) -> dict[str, Any]:
        if not consent_recorded:
            file.file.close()
            raise HTTPException(status_code=422, detail="Explicit research-processing consent is required")
        if purpose not in {"workflow_validation", "algorithm_evaluation", "research_demo"}:
            file.file.close()
            raise HTTPException(status_code=422, detail="Unsupported research purpose")
        if retention_policy != "delete_after_analysis":
            file.file.close()
            raise HTTPException(
                status_code=422,
                detail="The upload API accepts delete_after_analysis only; raw video is never retained",
            )

        tenant_store = scoped(identity)
        linked_participant: dict[str, Any] | None = None
        if identity.auth_mode != "disabled":
            linked_participant = participant_for_identity(
                tenant_store,
                identity,
                participant_id,
            )
            active_consent = tenant_store.active_consent(
                participant_id=linked_participant["participant_id"],
                purpose=purpose,
            )
            if active_consent is None:
                file.file.close()
                raise HTTPException(
                    status_code=422,
                    detail="An active, versioned consent record is required for this purpose",
                )
            consent_document_version = active_consent["document_version"]
            study_id = str(linked_participant.get("study_id") or study_id)
        elif participant_id:
            linked_participant = tenant_store.get_participant(participant_id)
            if linked_participant is None:
                file.file.close()
                raise HTTPException(status_code=404, detail="Participant not found")
            study_id = str(linked_participant.get("study_id") or study_id)

        original_name = Path(file.filename or "upload.bin").name
        safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in original_name)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            file.file.close()
            raise HTTPException(status_code=415, detail="Supported video types: mp4, mov, avi, mkv, m4v")

        upload_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = upload_dir / f"{uuid4().hex}_{safe_name}"
        size = 0
        try:
            with temporary_path.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Video exceeds the configured upload limit")
                    target.write(chunk)
            if size == 0:
                raise HTTPException(status_code=422, detail="Uploaded video is empty")

            try:
                preflight = video_preflight(temporary_path)
            except Exception as error:
                preflight = preflight_from_decode_error(original_name, error)

            if preflight["overall"] == "fail":
                case = case_from_preflight(
                    preflight,
                    purpose=purpose,
                    retention_policy=retention_policy,
                )
            else:
                try:
                    case = run_uploaded_video(
                        temporary_path,
                        purpose=purpose,
                        retention_policy=retention_policy,
                        preflight=preflight,
                    )
                except Exception as error:
                    case = case_from_runtime_failure(
                        preflight,
                        error,
                        purpose=purpose,
                        retention_policy=retention_policy,
                    )
            case["source_name"] = safe_name
            case["participant_id"] = (
                str(linked_participant["participant_id"]) if linked_participant else ""
            )
            case["study_id"] = study_id.strip()
            case["consent"] = {
                "recorded": True,
                "document_version": consent_document_version.strip(),
                "purpose": purpose,
            }
            case = sanitize_report_value(case)
            tenant_store.upsert_case(case)
            tenant_store.log_access(
                action="assessment.create",
                resource_type="case",
                resource_id=case["case_id"],
                details={
                    "decision": case["decision"],
                    "raw_video_retained": False,
                    "participant_id": case["participant_id"],
                },
            )
            return sanitize_report_value(
                {"item": case, "raw_video_retained": False, "claim_boundary": CLAIM_BOUNDARY}
            )
        finally:
            file.file.close()
            if temporary_path.exists():
                temporary_path.unlink()

    @app.get("/api/v1/cases/{case_id}")
    def get_case(
        case_id: str,
        identity: IdentityContext = Depends(read_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        case = tenant_store.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        tenant_store.log_access(action="case.read", resource_type="case", resource_id=case_id)
        return sanitize_report_value(case)

    @app.get("/api/v1/reviews")
    def list_reviews(
        include_closed: bool = True,
        identity: IdentityContext = Depends(review_read_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        reviews = tenant_store.list_reviews(include_closed=include_closed)
        tenant_store.log_access(action="review.list", resource_type="review")
        return sanitize_report_value({"count": len(reviews), "items": reviews, "claim_boundary": CLAIM_BOUNDARY})

    @app.put("/api/v1/reviews/{case_id}")
    def update_review(
        case_id: str,
        update: ReviewUpdate,
        identity: IdentityContext = Depends(reviewer_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        try:
            tenant_store.update_review(case_id, **update.model_dump())
        except KeyError as error:
            raise HTTPException(status_code=404, detail=sanitize_report_value(str(error))) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=sanitize_report_value(str(error))) from error
        review = next((item for item in tenant_store.list_reviews() if item["case_id"] == case_id), None)
        tenant_store.log_access(action="review.update", resource_type="review", resource_id=case_id)
        return sanitize_report_value({"item": review, "claim_boundary": CLAIM_BOUNDARY})

    @app.get("/api/v1/cases/{case_id}/report")
    def get_report(
        case_id: str,
        format: str = "json",
        language: str = "en",
        audience: str = "",
        identity: IdentityContext = Depends(read_identity),
    ) -> Any:
        tenant_store = scoped(identity)
        resolved_language = language.lower()
        if resolved_language not in {"en", "zh"}:
            raise HTTPException(status_code=422, detail="language must be en or zh")
        resolved_audience = audience.lower() or (
            "participant" if identity.primary_role == ROLE_PARTICIPANT else "operator"
        )
        if resolved_audience not in {"participant", "operator", "reviewer", "research"}:
            raise HTTPException(status_code=422, detail="Unsupported report audience")
        payload = governed_report(
            tenant_store,
            identity,
            case_id,
            audience=resolved_audience,
            language=resolved_language,
        )
        if format.lower() == "pdf":
            pdf = build_report_pdf(payload, language=resolved_language)
            tenant_store.log_access(
                action="report.export.pdf",
                resource_type="case",
                resource_id=case_id,
            )
            return Response(
                content=pdf,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{case_id}.pdf"'},
            )
        if format.lower() == "fhir":
            bundle = build_fhir_bundle(payload)
            tenant_store.log_access(
                action="report.export.fhir",
                resource_type="case",
                resource_id=case_id,
            )
            return Response(
                content=json.dumps(bundle, ensure_ascii=False, allow_nan=False),
                media_type="application/fhir+json",
                headers={"Content-Disposition": f'attachment; filename="{case_id}.fhir.json"'},
            )
        if format.lower() != "json":
            raise HTTPException(status_code=422, detail="format must be json, pdf, or fhir")
        tenant_store.log_access(
            action="report.read",
            resource_type="case",
            resource_id=case_id,
        )
        return sanitize_report_value(payload)

    @app.post("/api/v1/cases/{case_id}/report-versions", status_code=201)
    def create_report_version(
        case_id: str,
        request: ReportVersionCreate,
        identity: IdentityContext = Depends(operator_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        payload = governed_report(
            tenant_store,
            identity,
            case_id,
            audience=request.audience,
            language=request.language,
        )
        payload["governance"]["approval_status"] = "draft"
        narrative = (
            report_narrator.generate(payload, language=request.language)
            if request.generate_narrative
            else {}
        )
        payload["governance"]["narrative_status"] = (
            "draft" if narrative else "not_generated"
        )
        evidence_hash = report_content_sha256(payload)
        payload["governance"]["content_sha256"] = evidence_hash
        report_hash = report_version_sha256(payload, narrative)
        try:
            item = tenant_store.save_report_version(
                case_id=case_id,
                report_sha256=report_hash,
                audience=request.audience,
                language=request.language,
                payload=payload,
                narrative=narrative,
                supersedes_report_id=request.supersedes_report_id,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        tenant_store.log_access(
            action="report-version.create",
            resource_type="report",
            resource_id=item["report_id"],
            details={
                "case_id": case_id,
                "content_sha256": evidence_hash,
                "report_version_sha256": report_hash,
            },
        )
        return sanitize_report_value({"item": item})

    @app.get("/api/v1/cases/{case_id}/report-versions")
    def list_report_versions(
        case_id: str,
        identity: IdentityContext = Depends(read_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        if not tenant_store.get_case(case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        items = tenant_store.list_report_versions(case_id)
        if identity.primary_role == ROLE_PARTICIPANT:
            items = [item for item in items if item["audience"] == "participant"]
        tenant_store.log_access(
            action="report-version.list",
            resource_type="case",
            resource_id=case_id,
        )
        return sanitize_report_value({"count": len(items), "items": items})

    @app.post("/api/v1/report-versions/{report_id}/approve")
    def approve_report_version(
        report_id: str,
        identity: IdentityContext = Depends(reviewer_identity),
    ) -> dict[str, Any]:
        tenant_store = scoped(identity)
        try:
            item = tenant_store.approve_report_version(report_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        tenant_store.log_access(
            action="report-version.approve",
            resource_type="report",
            resource_id=report_id,
        )
        return sanitize_report_value({"item": item})

    return app


app = create_app()
