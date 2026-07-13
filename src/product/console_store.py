from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ConsoleStore:
    """Small SQLite store for product cases, reviews, and audit events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    display_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    case_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    assignee TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    resolution TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_cases_updated ON cases(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_events(case_id, created_at DESC);
                """
            )

    def upsert_case(self, payload: dict[str, Any], *, actor: str = "system") -> None:
        now = utc_now()
        case_id = str(payload["case_id"])
        display_id = str(payload.get("display_id", case_id))
        decision = str(payload.get("decision", "review"))
        priority = str(payload.get("priority", "routine"))
        created_at = str(payload.get("created_at", now))
        normalized = dict(payload)
        normalized["created_at"] = created_at
        normalized["updated_at"] = now
        encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO cases(case_id, display_id, decision, priority, created_at, updated_at, payload_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    display_id = excluded.display_id,
                    decision = excluded.decision,
                    priority = excluded.priority,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (case_id, display_id, decision, priority, created_at, now, encoded),
            )
            if decision in {"review", "retake"}:
                connection.execute(
                    """
                    INSERT INTO reviews(case_id, status, priority, created_at, updated_at)
                    VALUES(?, 'open', ?, ?, ?)
                    ON CONFLICT(case_id) DO UPDATE SET
                        priority = excluded.priority,
                        updated_at = excluded.updated_at
                    """,
                    (case_id, priority, now, now),
                )
            event_type = "case.updated" if exists else "case.created"
            self._insert_event(connection, case_id, event_type, actor, {"decision": decision})

    def list_cases(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM cases ORDER BY updated_at DESC, display_id"
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list_reviews(self, *, include_closed: bool = True) -> list[dict[str, Any]]:
        where = "" if include_closed else "WHERE r.status != 'closed'"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT r.*, c.display_id, c.decision, c.payload_json
                FROM reviews r
                JOIN cases c ON c.case_id = r.case_id
                {where}
                ORDER BY
                    CASE r.priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'routine' THEN 2 ELSE 3 END,
                    r.updated_at DESC
                """
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["case"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def update_review(
        self,
        case_id: str,
        *,
        status: str,
        priority: str,
        assignee: str,
        note: str,
        resolution: str,
        actor: str,
    ) -> None:
        if status not in {"open", "in_review", "waiting_retake", "closed"}:
            raise ValueError(f"Unsupported review status: {status}")
        if priority not in {"urgent", "high", "routine", "low"}:
            raise ValueError(f"Unsupported review priority: {priority}")
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reviews WHERE case_id = ?", (case_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Review not found for case: {case_id}")
            connection.execute(
                """
                UPDATE reviews
                SET status = ?, priority = ?, assignee = ?, note = ?, resolution = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (status, priority, assignee.strip(), note.strip(), resolution.strip(), now, case_id),
            )
            self._insert_event(
                connection,
                case_id,
                "review.updated",
                actor,
                {
                    "status": status,
                    "priority": priority,
                    "assignee": assignee.strip(),
                    "resolution": resolution.strip(),
                },
            )

    def audit_events(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, case_id, event_type, actor, details_json, created_at
                FROM audit_events
                WHERE case_id = ?
                ORDER BY event_id DESC
                """,
                (case_id,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            events.append(event)
        return events

    def log_event(
        self,
        case_id: str,
        event_type: str,
        *,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            self._insert_event(connection, case_id, event_type, actor, details or {})

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        case_id: str,
        event_type: str,
        actor: str,
        details: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events(case_id, event_type, actor, details_json, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                case_id,
                event_type,
                actor,
                json.dumps(details, ensure_ascii=False, allow_nan=False),
                utc_now(),
            ),
        )
