from datetime import UTC, datetime
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from src.product.operations import create_sqlite_backup, prune_sqlite_backups


def _database(path: Path, value: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE evidence(value TEXT NOT NULL)")
        connection.execute("INSERT INTO evidence(value) VALUES(?)", (value,))


def test_online_backup_is_verified_and_contains_no_raw_media(tmp_path: Path) -> None:
    source = tmp_path / "state.db"
    _database(source, "evidence")
    result = create_sqlite_backup(
        source,
        tmp_path / "backups",
        created_at=datetime(2026, 7, 17, 10, 0, tzinfo=UTC),
    )

    assert result["integrity_check"] == "ok"
    assert result["raw_media_included"] is False
    assert len(result["backup_sha256"]) == 64
    with sqlite3.connect(result["backup_path"]) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchone()[0] == "evidence"


def test_backup_refuses_to_overwrite_and_prunes_only_exact_pairs(tmp_path: Path) -> None:
    source = tmp_path / "state.db"
    output = tmp_path / "backups"
    _database(source, "evidence")
    first_time = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
    create_sqlite_backup(source, output, created_at=first_time)
    with pytest.raises(FileExistsError):
        create_sqlite_backup(source, output, created_at=first_time)
    create_sqlite_backup(
        source,
        output,
        created_at=datetime(2026, 7, 17, 11, 0, tzinfo=UTC),
    )
    unrelated = output / "keep-me.sqlite"
    unrelated.write_bytes(b"not a VitalsSight backup")

    removed = prune_sqlite_backups(output, keep=1)

    assert removed == ["vitalssight-20260717T100000Z.sqlite"]
    assert unrelated.exists()


def test_backup_cli_can_run_directly_from_repository_root() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/backup_controlled_trial_state.py", "--help"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "integrity-checked VitalsSight SQLite evidence backup" in result.stdout
