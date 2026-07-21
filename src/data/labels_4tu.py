from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RRIntervals:
    timestamps_sec: np.ndarray
    intervals_sec: np.ndarray

    @property
    def hr_bpm(self) -> np.ndarray:
        return 60.0 / self.intervals_sec

    def mean_hr(self, start_sec: float = 0.0, end_sec: float | None = None) -> float:
        mask = self.timestamps_sec >= start_sec
        if end_sec is not None:
            mask &= self.timestamps_sec <= end_sec
        values = self.hr_bpm[mask]
        if values.size == 0:
            values = self.hr_bpm
        return float(np.mean(values)) if values.size else float("nan")


def parse_rr_intervals(text: str) -> RRIntervals:
    timestamps: list[float] = []
    intervals: list[float] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
            interval = float(parts[1])
        except ValueError:
            continue
        if interval > 0:
            timestamps.append(ts)
            intervals.append(interval)
    return RRIntervals(np.asarray(timestamps, dtype=float), np.asarray(intervals, dtype=float))
