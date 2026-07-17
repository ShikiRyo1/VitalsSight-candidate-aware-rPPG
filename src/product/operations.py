from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from src.product.build_identity import path_fingerprint


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_sqlite_backup(
    database_path: str | Path,
    output_dir: str | Path,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create and verify an online SQLite backup without copying raw media."""

    source = Path(database_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"VitalsSight database was not found: {source}")
    destination_root = Path(output_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    timestamp = (created_at or datetime.now(UTC)).astimezone(UTC)
    stem = f"vitalssight-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    destination = destination_root / f"{stem}.sqlite"
    temporary = destination.with_suffix(".sqlite.partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Backup destination already exists: {destination}")

    try:
        with closing(sqlite3.connect(source)) as source_connection:
            with closing(sqlite3.connect(temporary)) as backup_connection:
                source_connection.backup(backup_connection)
                check = backup_connection.execute("PRAGMA quick_check").fetchone()
                if not check or str(check[0]).lower() != "ok":
                    raise RuntimeError(f"SQLite backup integrity check failed: {check}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "schema": "vitalssight.sqlite-backup.v1",
        "created_at": timestamp.isoformat(),
        "database_path_fingerprint": path_fingerprint(source),
        "backup_file": destination.name,
        "backup_sha256": sha256_file(destination),
        "backup_bytes": destination.stat().st_size,
        "integrity_check": "ok",
        "raw_media_included": False,
    }
    manifest_path = destination_root / f"{stem}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {**manifest, "backup_path": str(destination), "manifest_path": str(manifest_path)}


def prune_sqlite_backups(output_dir: str | Path, *, keep: int) -> list[str]:
    """Keep the newest exact VitalsSight backup pairs in one declared directory."""

    if keep < 1:
        raise ValueError("keep must be at least 1")
    root = Path(output_dir).resolve()
    backups = sorted(root.glob("vitalssight-????????T??????Z.sqlite"), reverse=True)
    removed: list[str] = []
    for backup in backups[keep:]:
        manifest = backup.with_suffix(".json")
        backup.unlink()
        manifest.unlink(missing_ok=True)
        removed.append(backup.name)
    return removed
