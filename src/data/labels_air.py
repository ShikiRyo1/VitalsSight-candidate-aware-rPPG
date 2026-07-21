from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import zipfile

import h5py
import numpy as np

from src.signal.estimate import RateEstimate, estimate_rr_infant


@dataclass(frozen=True)
class AIRRespirationLabel:
    archive_path: Path
    member: str
    filename: str
    respiration: np.ndarray
    impulse: np.ndarray | None = None


def read_air_label(archive_path: str | Path, gt_member: str) -> AIRRespirationLabel:
    archive = Path(archive_path)
    with zipfile.ZipFile(archive) as z:
        payload = z.read(gt_member)

    with h5py.File(io.BytesIO(payload), "r") as f:
        respiration = np.asarray(f["respiration"][()], dtype=float).reshape(-1)
        impulse = np.asarray(f["impulse"][()], dtype=float).reshape(-1) if "impulse" in f else None
        raw_filename = f["filename"][()] if "filename" in f else b""

    if isinstance(raw_filename, bytes):
        filename = raw_filename.decode("utf-8", errors="replace")
    elif hasattr(raw_filename, "item"):
        value = raw_filename.item()
        filename = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    else:
        filename = str(raw_filename)

    return AIRRespirationLabel(
        archive_path=archive,
        member=gt_member,
        filename=filename,
        respiration=respiration,
        impulse=impulse,
    )


def estimate_air_reference_rr(label: AIRRespirationLabel, fps: float) -> RateEstimate:
    return estimate_rr_infant(label.respiration, fps)
