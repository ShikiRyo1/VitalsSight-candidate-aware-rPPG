from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def path_fingerprint(path: str | Path) -> str:
    """Return a non-reversible identifier for a configured local path."""

    normalized = Path(path).resolve().as_posix().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@lru_cache(maxsize=1)
def source_build_identity() -> dict[str, Any]:
    """Return source identity for binding a running service to validation evidence."""

    try:
        commit = _git_output("rev-parse", "HEAD")
        tree = _git_output("rev-parse", "HEAD^{tree}")
        dirty = bool(_git_output("status", "--porcelain"))
        source = "git"
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = os.environ.get("VITALSSIGHT_BUILD_COMMIT", "unknown")
        tree = os.environ.get("VITALSSIGHT_BUILD_TREE", "unknown")
        dirty = os.environ.get("VITALSSIGHT_BUILD_DIRTY", "unknown").lower() not in {"0", "false", "clean"}
        source = "environment"
    return {
        "commit": commit,
        "tree": tree,
        "dirty": dirty,
        "source": source,
    }
