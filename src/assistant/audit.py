from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4

from src.assistant.schemas import AssistantConfirmResponse, PendingAction
from src.product.console_store import ConsoleStore


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def content_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AssistantAuditStore:
    """Privacy-minimizing assistant audit and explicit action confirmation store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assistant_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    case_id TEXT,
                    actor TEXT NOT NULL,
                    role TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message_sha256 TEXT NOT NULL,
                    response_sha256 TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS assistant_pending_actions (
                    token TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    confirmed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_assistant_trace ON assistant_events(trace_id);
                CREATE INDEX IF NOT EXISTS idx_assistant_pending_status ON assistant_pending_actions(status, expires_at);
                """
            )

    def log_chat(
        self,
        *,
        trace_id: str,
        conversation_id: str,
        case_id: str | None,
        actor: str,
        role: str,
        provider: str,
        event_type: str,
        message: str,
        response: str,
        details: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_events(
                    trace_id, conversation_id, case_id, actor, role, provider, event_type,
                    message_sha256, response_sha256, details_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    conversation_id,
                    case_id,
                    actor,
                    role,
                    provider,
                    event_type,
                    content_digest(message),
                    content_digest(response),
                    json.dumps(details, ensure_ascii=False, allow_nan=False),
                    utc_now(),
                ),
            )

    def prepare_review_update(
        self,
        *,
        case_id: str,
        actor: str,
        conversation_id: str,
        payload: dict[str, Any],
        ttl_minutes: int = 15,
    ) -> PendingAction:
        token = uuid4().hex + uuid4().hex
        expires_at = (datetime.now(UTC) + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_pending_actions(
                    token, action_type, case_id, actor, conversation_id, payload_json,
                    status, created_at, expires_at
                ) VALUES(?, 'review_update', ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    token,
                    case_id,
                    actor,
                    conversation_id,
                    json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    utc_now(),
                    expires_at,
                ),
            )
        summary = f"Update review {case_id} to {payload['status']} with {payload['priority']} priority"
        return PendingAction(
            token=token,
            action_type="review_update",
            summary=summary,
            expires_at=expires_at,
        )

    def reject(self, token: str, *, actor: str) -> AssistantConfirmResponse:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT action_type, case_id, actor, status, expires_at FROM assistant_pending_actions WHERE token = ?",
                (token,),
            ).fetchone()
            if not row:
                raise KeyError("Pending assistant action was not found")
            if str(row["actor"]) != actor:
                raise PermissionError("Only the actor who prepared this action may reject it")
            if row["status"] != "pending":
                raise ValueError(f"Assistant action is already {row['status']}")
            if datetime.fromisoformat(str(row["expires_at"])) <= datetime.now(UTC):
                connection.execute(
                    "UPDATE assistant_pending_actions SET status = 'expired' WHERE token = ? AND status = 'pending'",
                    (token,),
                )
                return AssistantConfirmResponse(
                    status="expired",
                    action_type=str(row["action_type"]),
                    case_id=str(row["case_id"]),
                    message="The confirmation window expired; no review data changed.",
                )
            updated = connection.execute(
                "UPDATE assistant_pending_actions SET status = 'rejected', confirmed_at = ? "
                "WHERE token = ? AND status = 'pending'",
                (utc_now(), token),
            )
            if updated.rowcount != 1:
                raise ValueError("Assistant action changed before it could be rejected")
        return AssistantConfirmResponse(
            status="rejected",
            action_type=str(row["action_type"]),
            case_id=str(row["case_id"]),
            message=f"Action rejected by {actor}; no review data changed.",
        )

    def confirm(self, token: str, *, actor: str, console_store: ConsoleStore) -> AssistantConfirmResponse:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assistant_pending_actions WHERE token = ?",
                (token,),
            ).fetchone()
            if not row:
                raise KeyError("Pending assistant action was not found")
            if str(row["actor"]) != actor:
                raise PermissionError("Only the actor who prepared this action may confirm it")
            if row["status"] != "pending":
                raise ValueError(f"Assistant action is already {row['status']}")
            expires_at = datetime.fromisoformat(str(row["expires_at"]))
            if expires_at <= datetime.now(UTC):
                connection.execute(
                    "UPDATE assistant_pending_actions SET status = 'expired' WHERE token = ?",
                    (token,),
                )
                return AssistantConfirmResponse(
                    status="expired",
                    action_type=str(row["action_type"]),
                    case_id=str(row["case_id"]),
                    message="The confirmation window expired; no review data changed.",
                )
            payload = json.loads(row["payload_json"])
            claimed = connection.execute(
                "UPDATE assistant_pending_actions SET status = 'executing' "
                "WHERE token = ? AND status = 'pending'",
                (token,),
            )
            if claimed.rowcount != 1:
                raise ValueError("Assistant action changed before it could be confirmed")

        if row["action_type"] != "review_update":
            raise ValueError(f"Unsupported assistant action: {row['action_type']}")
        try:
            console_store.update_review(
                str(row["case_id"]),
                status=str(payload["status"]),
                priority=str(payload["priority"]),
                assignee=str(payload.get("assignee") or ""),
                note=str(payload.get("note") or ""),
                resolution=str(payload.get("resolution") or ""),
                actor=actor,
            )
        except Exception:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE assistant_pending_actions SET status = 'failed', confirmed_at = ? "
                    "WHERE token = ? AND status = 'executing'",
                    (utc_now(), token),
                )
            raise
        with self._connect() as connection:
            completed = connection.execute(
                "UPDATE assistant_pending_actions SET status = 'confirmed', confirmed_at = ? "
                "WHERE token = ? AND status = 'executing'",
                (utc_now(), token),
            )
            if completed.rowcount != 1:
                raise RuntimeError("Review changed, but the assistant confirmation audit could not be finalized")
        return AssistantConfirmResponse(
            status="confirmed",
            action_type="review_update",
            case_id=str(row["case_id"]),
            message="The explicitly confirmed review update was saved and added to the audit trail.",
        )

    def events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM assistant_events ORDER BY event_id DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result
