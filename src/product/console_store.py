from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator
from uuid import uuid4

from src.product.identity import (
    DEFAULT_ORGANIZATION_ID,
    DEFAULT_ORGANIZATION_NAME,
    DEFAULT_USER_ID,
    IdentityContext,
    LOCAL_ADMIN_ROLES,
    normalize_identifier,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class ConsoleStore:
    """SQLite evidence store with backward-compatible tenant-aware migrations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    organization_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL DEFAULT '',
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memberships (
                    organization_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(organization_id, user_id, role),
                    FOREIGN KEY(organization_id) REFERENCES organizations(organization_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    pseudonym TEXT NOT NULL,
                    study_id TEXT NOT NULL DEFAULT '',
                    external_reference_hash TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(organization_id, pseudonym, study_id),
                    FOREIGN KEY(organization_id) REFERENCES organizations(organization_id)
                );

                CREATE TABLE IF NOT EXISTS consents (
                    consent_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    document_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recorded_by TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    withdrawn_at TEXT NOT NULL DEFAULT '',
                    details_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(organization_id) REFERENCES organizations(organization_id),
                    FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
                );

                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL DEFAULT 'local-research',
                    participant_id TEXT NOT NULL DEFAULT '',
                    study_id TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT 'system',
                    display_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    case_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL DEFAULT 'local-research',
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
                    organization_id TEXT NOT NULL DEFAULT 'local-research',
                    actor_user_id TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS report_versions (
                    report_id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    audience TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    narrative_json TEXT NOT NULL DEFAULT '{}',
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_by TEXT NOT NULL DEFAULT '',
                    approved_at TEXT NOT NULL DEFAULT '',
                    supersedes_report_id TEXT NOT NULL DEFAULT '',
                    UNIQUE(organization_id, case_id, report_sha256, audience, language),
                    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS access_events (
                    access_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    organization_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

            # Existing local databases predate tenant metadata. Additive migrations
            # keep their evidence intact while assigning it to the local workspace.
            self._ensure_column(
                connection,
                "cases",
                "organization_id",
                f"TEXT NOT NULL DEFAULT '{DEFAULT_ORGANIZATION_ID}'",
            )
            self._ensure_column(connection, "cases", "participant_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "cases", "study_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "cases", "created_by", "TEXT NOT NULL DEFAULT 'system'")
            self._ensure_column(
                connection,
                "reviews",
                "organization_id",
                f"TEXT NOT NULL DEFAULT '{DEFAULT_ORGANIZATION_ID}'",
            )
            self._ensure_column(
                connection,
                "audit_events",
                "organization_id",
                f"TEXT NOT NULL DEFAULT '{DEFAULT_ORGANIZATION_ID}'",
            )
            self._ensure_column(connection, "audit_events", "actor_user_id", "TEXT NOT NULL DEFAULT ''")

            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_cases_org_updated
                    ON cases(organization_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cases_org_participant
                    ON cases(organization_id, participant_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_reviews_org_status
                    ON reviews(organization_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_org_case
                    ON audit_events(organization_id, case_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_report_versions_case
                    ON report_versions(organization_id, case_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_access_events_org_created
                    ON access_events(organization_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_consents_participant
                    ON consents(organization_id, participant_id, recorded_at DESC);
                """
            )
            now = utc_now()
            connection.execute(
                """
                INSERT INTO organizations(organization_id, name, status, settings_json, created_at, updated_at)
                VALUES(?, ?, 'active', '{}', ?, ?)
                ON CONFLICT(organization_id) DO NOTHING
                """,
                (DEFAULT_ORGANIZATION_ID, DEFAULT_ORGANIZATION_NAME, now, now),
            )
            connection.execute(
                """
                INSERT INTO users(user_id, subject, email, display_name, status, created_at, updated_at)
                VALUES(?, ?, '', 'Research operator', 'active', ?, ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (DEFAULT_USER_ID, f"local:{DEFAULT_USER_ID}", now, now),
            )
            for role in sorted(LOCAL_ADMIN_ROLES):
                connection.execute(
                    """
                    INSERT INTO memberships(organization_id, user_id, role, status, created_at, updated_at)
                    VALUES(?, ?, ?, 'active', ?, ?)
                    ON CONFLICT(organization_id, user_id, role) DO NOTHING
                    """,
                    (DEFAULT_ORGANIZATION_ID, DEFAULT_USER_ID, role, now, now),
                )

    def ensure_identity(self, identity: IdentityContext, *, organization_name: str = "") -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO organizations(organization_id, name, status, settings_json, created_at, updated_at)
                VALUES(?, ?, 'active', '{}', ?, ?)
                ON CONFLICT(organization_id) DO UPDATE SET
                    name = CASE WHEN excluded.name != '' THEN excluded.name ELSE organizations.name END,
                    updated_at = excluded.updated_at
                """,
                (
                    identity.organization_id,
                    organization_name.strip() or identity.organization_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO users(user_id, subject, email, display_name, status, created_at, updated_at)
                VALUES(?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    subject = excluded.subject,
                    email = excluded.email,
                    display_name = excluded.display_name,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (
                    identity.user_id,
                    identity.subject,
                    identity.email,
                    identity.display_name,
                    now,
                    now,
                ),
            )
            for role in sorted(identity.roles):
                connection.execute(
                    """
                    INSERT INTO memberships(organization_id, user_id, role, status, created_at, updated_at)
                    VALUES(?, ?, ?, 'active', ?, ?)
                    ON CONFLICT(organization_id, user_id, role) DO UPDATE SET
                        status = 'active', updated_at = excluded.updated_at
                    """,
                    (identity.organization_id, identity.user_id, role, now, now),
                )

    def organization(self, organization_id: str = DEFAULT_ORGANIZATION_ID) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM organizations WHERE organization_id = ?",
                (organization_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["settings"] = json.loads(item.pop("settings_json"))
        return item

    def memberships(self, organization_id: str = DEFAULT_ORGANIZATION_ID) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.organization_id, m.user_id, m.role, m.status,
                       u.email, u.display_name, u.subject, m.created_at, m.updated_at
                FROM memberships m
                JOIN users u ON u.user_id = m.user_id
                WHERE m.organization_id = ?
                ORDER BY u.display_name, m.role
                """,
                (organization_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_participant(
        self,
        *,
        organization_id: str,
        pseudonym: str,
        study_id: str = "",
        external_reference_hash: str = "",
        created_by: str,
        participant_id: str = "",
    ) -> dict[str, Any]:
        normalized_org = normalize_identifier(organization_id, fallback=DEFAULT_ORGANIZATION_ID)
        clean_pseudonym = pseudonym.strip()
        if not clean_pseudonym:
            raise ValueError("Participant pseudonym is required")
        resolved_id = participant_id.strip() or f"pt_{uuid4().hex}"
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO participants(
                    participant_id, organization_id, pseudonym, study_id,
                    external_reference_hash, status, created_by, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, 'active', ?, ?, ?)
                ON CONFLICT(organization_id, pseudonym, study_id) DO UPDATE SET
                    external_reference_hash = excluded.external_reference_hash,
                    status = 'active', updated_at = excluded.updated_at
                """,
                (
                    resolved_id,
                    normalized_org,
                    clean_pseudonym,
                    study_id.strip(),
                    external_reference_hash.strip(),
                    created_by,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM participants
                WHERE organization_id = ? AND pseudonym = ? AND study_id = ?
                """,
                (normalized_org, clean_pseudonym, study_id.strip()),
            ).fetchone()
        return dict(row)

    def list_participants(self, organization_id: str = DEFAULT_ORGANIZATION_ID) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM participants
                WHERE organization_id = ? AND status != 'deleted'
                ORDER BY updated_at DESC, pseudonym
                """,
                (organization_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_participant(
        self,
        participant_id: str,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM participants
                WHERE participant_id = ? AND organization_id = ? AND status != 'deleted'
                """,
                (participant_id, organization_id),
            ).fetchone()
        return dict(row) if row else None

    def record_consent(
        self,
        *,
        organization_id: str,
        participant_id: str,
        purpose: str,
        document_version: str,
        recorded_by: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.get_participant(participant_id, organization_id=organization_id):
            raise KeyError("Participant not found in this organization")
        now = utc_now()
        consent_id = f"consent_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE consents SET status = 'superseded'
                WHERE organization_id = ? AND participant_id = ? AND purpose = ? AND status = 'active'
                """,
                (organization_id, participant_id, purpose),
            )
            connection.execute(
                """
                INSERT INTO consents(
                    consent_id, organization_id, participant_id, purpose,
                    document_version, status, recorded_by, recorded_at, details_json
                ) VALUES(?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    consent_id,
                    organization_id,
                    participant_id,
                    purpose,
                    document_version.strip(),
                    recorded_by,
                    now,
                    _json(details or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM consents WHERE consent_id = ?",
                (consent_id,),
            ).fetchone()
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json"))
        return item

    def active_consent(
        self,
        *,
        organization_id: str,
        participant_id: str,
        purpose: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM consents
                WHERE organization_id = ? AND participant_id = ? AND purpose = ? AND status = 'active'
                ORDER BY recorded_at DESC LIMIT 1
                """,
                (organization_id, participant_id, purpose),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json"))
        return item

    def list_consents(
        self,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        participant_id: str = "",
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM consents WHERE organization_id = ?"
        parameters: list[Any] = [organization_id]
        if participant_id:
            query += " AND participant_id = ?"
            parameters.append(participant_id)
        query += " ORDER BY recorded_at DESC, consent_id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def withdraw_consent(
        self,
        consent_id: str,
        *,
        organization_id: str,
        actor: str,
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE consents
                SET status = 'withdrawn', withdrawn_at = ?
                WHERE consent_id = ? AND organization_id = ? AND status = 'active'
                """,
                (now, consent_id, organization_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("Active consent not found in this organization")
        self.log_access(
            organization_id=organization_id,
            user_id=actor,
            action="consent.withdraw",
            resource_type="consent",
            resource_id=consent_id,
            outcome="success",
        )

    def upsert_case(
        self,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        actor_user_id: str = "",
        participant_id: str = "",
        study_id: str = "",
    ) -> None:
        now = utc_now()
        normalized_org = normalize_identifier(organization_id, fallback=DEFAULT_ORGANIZATION_ID)
        case_id = str(payload["case_id"])
        display_id = str(payload.get("display_id", case_id))
        decision = str(payload.get("decision", "review"))
        priority = str(payload.get("priority", "routine"))
        created_at = str(payload.get("created_at", now))
        normalized = dict(payload)
        normalized["organization_id"] = normalized_org
        normalized["participant_id"] = participant_id or str(payload.get("participant_id") or "")
        normalized["study_id"] = study_id or str(payload.get("study_id") or "")
        normalized["created_by"] = str(payload.get("created_by") or actor_user_id or actor)
        normalized["created_at"] = created_at
        normalized["updated_at"] = now
        encoded = _json(normalized)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT organization_id FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if existing and existing["organization_id"] != normalized_org:
                raise PermissionError("Case identifier belongs to another organization")
            connection.execute(
                """
                INSERT INTO cases(
                    case_id, organization_id, participant_id, study_id, created_by,
                    display_id, decision, priority, created_at, updated_at, payload_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    participant_id = excluded.participant_id,
                    study_id = excluded.study_id,
                    display_id = excluded.display_id,
                    decision = excluded.decision,
                    priority = excluded.priority,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    case_id,
                    normalized_org,
                    normalized["participant_id"],
                    normalized["study_id"],
                    normalized["created_by"],
                    display_id,
                    decision,
                    priority,
                    created_at,
                    now,
                    encoded,
                ),
            )
            if decision in {"review", "retake"}:
                connection.execute(
                    """
                    INSERT INTO reviews(case_id, organization_id, status, priority, created_at, updated_at)
                    VALUES(?, ?, 'open', ?, ?, ?)
                    ON CONFLICT(case_id) DO UPDATE SET
                        priority = excluded.priority,
                        updated_at = excluded.updated_at
                    """,
                    (case_id, normalized_org, priority, now, now),
                )
            event_type = "case.updated" if existing else "case.created"
            self._insert_event(
                connection,
                case_id,
                event_type,
                actor,
                {"decision": decision},
                organization_id=normalized_org,
                actor_user_id=actor_user_id,
            )

    def list_cases(
        self,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        participant_id: str = "",
        study_id: str = "",
    ) -> list[dict[str, Any]]:
        clauses = ["organization_id = ?"]
        params: list[Any] = [organization_id]
        if participant_id:
            clauses.append("participant_id = ?")
            params.append(participant_id)
        if study_id:
            clauses.append("study_id = ?")
            params.append(study_id)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT payload_json FROM cases
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, display_id
                """,
                params,
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_case(
        self,
        case_id: str,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        participant_id: str = "",
    ) -> dict[str, Any] | None:
        clauses = ["case_id = ?", "organization_id = ?"]
        params: list[Any] = [case_id, organization_id]
        if participant_id:
            clauses.append("participant_id = ?")
            params.append(participant_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT payload_json FROM cases WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list_reviews(
        self,
        *,
        include_closed: bool = True,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
    ) -> list[dict[str, Any]]:
        closed = "" if include_closed else "AND r.status != 'closed'"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT r.*, c.display_id, c.decision, c.payload_json
                FROM reviews r
                JOIN cases c ON c.case_id = r.case_id
                WHERE r.organization_id = ? {closed}
                ORDER BY
                    CASE r.priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'routine' THEN 2 ELSE 3 END,
                    r.updated_at DESC
                """,
                (organization_id,),
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
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        actor_user_id: str = "",
    ) -> None:
        if status not in {"open", "in_review", "waiting_retake", "closed"}:
            raise ValueError(f"Unsupported review status: {status}")
        if priority not in {"urgent", "high", "routine", "low"}:
            raise ValueError(f"Unsupported review priority: {priority}")
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reviews WHERE case_id = ? AND organization_id = ?",
                (case_id, organization_id),
            ).fetchone()
            if not row:
                raise KeyError(f"Review not found for case: {case_id}")
            connection.execute(
                """
                UPDATE reviews
                SET status = ?, priority = ?, assignee = ?, note = ?, resolution = ?, updated_at = ?
                WHERE case_id = ? AND organization_id = ?
                """,
                (
                    status,
                    priority,
                    assignee.strip(),
                    note.strip(),
                    resolution.strip(),
                    now,
                    case_id,
                    organization_id,
                ),
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
                organization_id=organization_id,
                actor_user_id=actor_user_id,
            )

    def audit_events(
        self,
        case_id: str,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, case_id, organization_id, actor_user_id,
                       event_type, actor, details_json, created_at
                FROM audit_events
                WHERE case_id = ? AND organization_id = ?
                ORDER BY event_id DESC
                """,
                (case_id, organization_id),
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
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        actor_user_id: str = "",
    ) -> None:
        with self._connect() as connection:
            case = connection.execute(
                "SELECT 1 FROM cases WHERE case_id = ? AND organization_id = ?",
                (case_id, organization_id),
            ).fetchone()
            if not case:
                raise KeyError(f"Case not found: {case_id}")
            self._insert_event(
                connection,
                case_id,
                event_type,
                actor,
                details or {},
                organization_id=organization_id,
                actor_user_id=actor_user_id,
            )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        case_id: str,
        event_type: str,
        actor: str,
        details: dict[str, Any],
        *,
        organization_id: str,
        actor_user_id: str = "",
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events(
                case_id, organization_id, actor_user_id,
                event_type, actor, details_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                organization_id,
                actor_user_id,
                event_type,
                actor,
                _json(details),
                utc_now(),
            ),
        )

    def save_report_version(
        self,
        *,
        organization_id: str,
        case_id: str,
        report_sha256: str,
        audience: str,
        language: str,
        payload: dict[str, Any],
        narrative: dict[str, Any] | None,
        created_by: str,
        supersedes_report_id: str = "",
    ) -> dict[str, Any]:
        if not self.get_case(case_id, organization_id=organization_id):
            raise KeyError("Case not found in this organization")
        report_id = f"report_{uuid4().hex}"
        now = utc_now()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM report_versions
                WHERE organization_id = ? AND case_id = ? AND report_sha256 = ?
                      AND audience = ? AND language = ?
                """,
                (organization_id, case_id, report_sha256, audience, language),
            ).fetchone()
            if existing:
                item = dict(existing)
                item["payload"] = json.loads(item.pop("payload_json"))
                item["narrative"] = json.loads(item.pop("narrative_json"))
                return item
            connection.execute(
                """
                INSERT INTO report_versions(
                    report_id, organization_id, case_id, report_sha256,
                    audience, language, status, payload_json, narrative_json,
                    created_by, created_at, supersedes_report_id
                ) VALUES(?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    organization_id,
                    case_id,
                    report_sha256,
                    audience,
                    language,
                    _json(payload),
                    _json(narrative or {}),
                    created_by,
                    now,
                    supersedes_report_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM report_versions WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        item["narrative"] = json.loads(item.pop("narrative_json"))
        return item

    def list_report_versions(
        self,
        case_id: str,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM report_versions
                WHERE case_id = ? AND organization_id = ?
                ORDER BY created_at DESC, report_id DESC
                """,
                (case_id, organization_id),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            item["narrative"] = json.loads(item.pop("narrative_json"))
            result.append(item)
        return result

    def approve_report_version(
        self,
        report_id: str,
        *,
        organization_id: str,
        approved_by: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, case_id FROM report_versions
                WHERE report_id = ? AND organization_id = ?
                """,
                (report_id, organization_id),
            ).fetchone()
            if not row:
                raise KeyError("Report version not found in this organization")
            if row["status"] != "draft":
                raise ValueError("Only a draft report can be approved")
            connection.execute(
                """
                UPDATE report_versions
                SET status = 'approved', approved_by = ?, approved_at = ?
                WHERE report_id = ? AND organization_id = ? AND status = 'draft'
                """,
                (approved_by, now, report_id, organization_id),
            )
            self._insert_event(
                connection,
                str(row["case_id"]),
                "report.approved",
                approved_by,
                {"report_id": report_id},
                organization_id=organization_id,
                actor_user_id=approved_by,
            )
        return next(
            item
            for item in self.list_report_versions(str(row["case_id"]), organization_id=organization_id)
            if item["report_id"] == report_id
        )

    def log_access(
        self,
        *,
        organization_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO access_events(
                    organization_id, user_id, action, resource_type,
                    resource_id, outcome, details_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    organization_id,
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    outcome,
                    _json(details or {}),
                    utc_now(),
                ),
            )

    def access_events(
        self,
        *,
        organization_id: str = DEFAULT_ORGANIZATION_ID,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM access_events
                WHERE organization_id = ?
                ORDER BY access_event_id DESC LIMIT ?
                """,
                (organization_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result


class ScopedConsoleStore:
    """Organization- and participant-bound view used by the UI and assistant."""

    def __init__(self, store: ConsoleStore, identity: IdentityContext) -> None:
        self.base = store
        self.identity = identity
        self.path = store.path
        self.base.ensure_identity(identity)

    @property
    def _participant_scope(self) -> str:
        return self.identity.participant_id if self.identity.primary_role == "participant" else ""

    def list_cases(self, *, participant_id: str = "") -> list[dict[str, Any]]:
        participant_scope = self._participant_scope or participant_id
        return self.base.list_cases(
            organization_id=self.identity.organization_id,
            participant_id=participant_scope,
        )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        return self.base.get_case(
            case_id,
            organization_id=self.identity.organization_id,
            participant_id=self._participant_scope,
        )

    def upsert_case(self, payload: dict[str, Any], *, actor: str = "") -> None:
        self.base.upsert_case(
            payload,
            actor=self.identity.actor,
            organization_id=self.identity.organization_id,
            actor_user_id=self.identity.user_id,
            participant_id=str(payload.get("participant_id") or self.identity.participant_id or ""),
            study_id=str(payload.get("study_id") or ""),
        )

    def list_reviews(self, *, include_closed: bool = True) -> list[dict[str, Any]]:
        if self.identity.primary_role == "participant":
            return []
        return self.base.list_reviews(
            include_closed=include_closed,
            organization_id=self.identity.organization_id,
        )

    def update_review(self, case_id: str, **values: Any) -> None:
        values["actor"] = self.identity.actor
        values["actor_user_id"] = self.identity.user_id
        values["organization_id"] = self.identity.organization_id
        self.base.update_review(case_id, **values)

    def audit_events(self, case_id: str) -> list[dict[str, Any]]:
        if not self.get_case(case_id):
            return []
        return self.base.audit_events(case_id, organization_id=self.identity.organization_id)

    def log_event(
        self,
        case_id: str,
        event_type: str,
        *,
        actor: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.base.log_event(
            case_id,
            event_type,
            actor=self.identity.actor,
            details=details,
            organization_id=self.identity.organization_id,
            actor_user_id=self.identity.user_id,
        )

    def list_participants(self) -> list[dict[str, Any]]:
        if self.identity.primary_role == "participant":
            item = self.get_participant(self.identity.participant_id)
            return [item] if item else []
        return self.base.list_participants(self.identity.organization_id)

    def get_participant(self, participant_id: str) -> dict[str, Any] | None:
        if self.identity.primary_role == "participant" and participant_id != self.identity.participant_id:
            return None
        return self.base.get_participant(
            participant_id,
            organization_id=self.identity.organization_id,
        )

    def upsert_participant(self, **values: Any) -> dict[str, Any]:
        return self.base.upsert_participant(
            organization_id=self.identity.organization_id,
            created_by=self.identity.user_id,
            **values,
        )

    def record_consent(self, **values: Any) -> dict[str, Any]:
        participant_id = str(values.get("participant_id") or "")
        if self.identity.primary_role == "participant" and participant_id != self.identity.participant_id:
            raise PermissionError("Participant access is limited to the linked record")
        return self.base.record_consent(
            organization_id=self.identity.organization_id,
            recorded_by=self.identity.user_id,
            **values,
        )

    def active_consent(self, *, participant_id: str, purpose: str) -> dict[str, Any] | None:
        return self.base.active_consent(
            organization_id=self.identity.organization_id,
            participant_id=participant_id,
            purpose=purpose,
        )

    def list_consents(self, *, participant_id: str = "") -> list[dict[str, Any]]:
        if self.identity.primary_role == "participant":
            participant_id = self.identity.participant_id
        return self.base.list_consents(
            organization_id=self.identity.organization_id,
            participant_id=participant_id,
        )

    def withdraw_consent(self, consent_id: str) -> None:
        if self.identity.primary_role == "participant":
            consent = next(
                (item for item in self.list_consents() if item["consent_id"] == consent_id),
                None,
            )
            if consent is None:
                raise PermissionError("Participant access is limited to the linked record")
        self.base.withdraw_consent(
            consent_id,
            organization_id=self.identity.organization_id,
            actor=self.identity.user_id,
        )

    def save_report_version(self, **values: Any) -> dict[str, Any]:
        return self.base.save_report_version(
            organization_id=self.identity.organization_id,
            created_by=self.identity.user_id,
            **values,
        )

    def list_report_versions(self, case_id: str) -> list[dict[str, Any]]:
        if not self.get_case(case_id):
            return []
        return self.base.list_report_versions(
            case_id,
            organization_id=self.identity.organization_id,
        )

    def approve_report_version(self, report_id: str) -> dict[str, Any]:
        return self.base.approve_report_version(
            report_id,
            organization_id=self.identity.organization_id,
            approved_by=self.identity.user_id,
        )

    def memberships(self) -> list[dict[str, Any]]:
        return self.base.memberships(self.identity.organization_id)

    def access_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        return self.base.access_events(
            organization_id=self.identity.organization_id,
            limit=limit,
        )

    def log_access(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str = "",
        outcome: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.base.log_access(
            organization_id=self.identity.organization_id,
            user_id=self.identity.user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            details=details,
        )
