from __future__ import annotations

import zipfile
from pathlib import Path


def extract_zip_member(archive_path: str | Path, member: str, output_dir: str | Path) -> Path:
    archive = Path(archive_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(member).name
    if target.exists() and target.stat().st_size > 0:
        return target
    with zipfile.ZipFile(archive) as z:
        with z.open(member) as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return target


def read_zip_text(archive_path: str | Path, member: str, encoding: str = "utf-8") -> str:
    with zipfile.ZipFile(archive_path) as z:
        return z.read(member).decode(encoding, errors="replace")
