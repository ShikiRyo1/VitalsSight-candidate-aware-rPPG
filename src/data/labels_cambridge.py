from __future__ import annotations

from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd


def read_cambridge_csv_member(archive_path: str | Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as z:
        data = z.read(member)
    frame = pd.read_csv(io.BytesIO(data))
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def cambridge_subjects(archive_path: str | Path) -> list[str]:
    subjects: set[str] = set()
    with zipfile.ZipFile(archive_path) as z:
        for item in z.infolist():
            parts = item.filename.split("/")
            if len(parts) > 2 and parts[0] == "dataset" and parts[1].startswith("mk"):
                subjects.add(parts[1])
    return sorted(subjects)


def cambridge_reference_member(archive_path: str | Path, subject: str) -> str | None:
    candidates = [
        f"dataset/{subject}/Impedance respiratory rate/Impedance_{subject}",
        f"dataset/{subject}/Ventilator respiratory rate/Ventilator_rate_{subject}",
    ]
    with zipfile.ZipFile(archive_path) as z:
        names = set(z.namelist())
    for member in candidates:
        if member in names:
            return member
    return None


def cambridge_camera_member(archive_path: str | Path, subject: str) -> str | None:
    member = f"dataset/{subject}/RGB-D camera video data/Camera_{subject}"
    with zipfile.ZipFile(archive_path) as z:
        return member if member in set(z.namelist()) else None


def mean_reference_bpm(reference: pd.DataFrame) -> float:
    numeric_cols = [col for col in reference.columns if col.lower() != "time (s)"]
    if not numeric_cols:
        return float("nan")
    values = pd.to_numeric(reference[numeric_cols[0]], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def uniform_signal_from_camera(camera: pd.DataFrame, column: str, *, max_fps: float = 60.0) -> tuple[np.ndarray, float]:
    if "Time (s)" not in camera.columns:
        raise ValueError("Cambridge camera data does not include a Time (s) column.")
    if column not in camera.columns:
        raise ValueError(f"Column not found in Cambridge camera data: {column}")

    time = pd.to_numeric(camera["Time (s)"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(camera[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(time) & np.isfinite(values) & (time >= 0.0)
    time = time[mask]
    values = values[mask]
    if len(time) < 8:
        raise ValueError("Not enough valid Cambridge camera samples.")

    order = np.argsort(time)
    time = time[order]
    values = values[order]
    diffs = np.diff(time)
    diffs = diffs[(diffs > 1e-3) & (diffs < 1.0)]
    if len(diffs) == 0:
        raise ValueError("Cambridge camera timestamps are not increasing.")
    fps = float(min(1.0 / np.median(diffs), max_fps))
    uniform_time = np.arange(time[0], time[-1], 1.0 / fps)
    max_points = int(max_fps * max(1.0, time[-1] - time[0] + 1.0))
    if len(uniform_time) > max_points:
        uniform_time = np.linspace(time[0], time[-1], max_points)
        fps = float((len(uniform_time) - 1) / max(1e-6, time[-1] - time[0]))
    uniform_values = np.interp(uniform_time, time, values)
    return uniform_values, fps
