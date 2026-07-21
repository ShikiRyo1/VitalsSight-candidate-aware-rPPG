from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    env = os.environ.get("CONTACTLESS_DATA_ROOT")
    if env:
        return Path(env)

    autodl = Path("/root/autodl-tmp/datasets")
    if autodl.exists():
        return autodl

    return project_root().parent / "\u6570\u636e\u96c6"


def adult_data_root() -> Path:
    env = os.environ.get("ADULT_DATA_ROOT")
    if env:
        return Path(env)
    return default_data_root() / "adult"


def dataset_root(env_name: str, fallback: Path) -> Path:
    env = os.environ.get(env_name)
    return Path(env) if env else fallback
