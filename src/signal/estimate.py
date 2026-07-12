from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .filters import as_clean_array, bandpass, zscore


@dataclass(frozen=True)
class RateEstimate:
    bpm: float
    peak_hz: float
    confidence: float
    band_power: float
    total_power: float


def estimate_rate_fft(
    values: np.ndarray | list[float],
    fps: float,
    *,
    min_bpm: float,
    max_bpm: float,
) -> RateEstimate:
    arr = zscore(as_clean_array(values))
    low_hz = min_bpm / 60.0
    high_hz = max_bpm / 60.0
    filtered = bandpass(arr, fps, low_hz, high_hz)
    nperseg = min(len(filtered), max(32, int(fps * 16)))
    freqs, power = signal.welch(filtered, fs=fps, nperseg=nperseg)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return RateEstimate(float("nan"), float("nan"), 0.0, 0.0, float(np.sum(power)))
    band_freqs = freqs[mask]
    band_power_values = power[mask]
    peak_idx = int(np.argmax(band_power_values))
    peak_hz = float(band_freqs[peak_idx])
    band_power = float(np.sum(band_power_values))
    total_power = float(np.sum(power) + 1e-12)
    confidence = float(band_power_values[peak_idx] / (band_power + 1e-12))
    return RateEstimate(peak_hz * 60.0, peak_hz, confidence, band_power, total_power)


def top_k_rate_fft(
    values: np.ndarray | list[float],
    fps: float,
    *,
    min_bpm: float,
    max_bpm: float,
    top_k: int = 5,
) -> tuple[list[dict[str, float]], float, float]:
    arr = zscore(as_clean_array(values))
    low_hz = min_bpm / 60.0
    high_hz = max_bpm / 60.0
    filtered = bandpass(arr, fps, low_hz, high_hz)
    nperseg = min(len(filtered), max(32, int(fps * 16)))
    freqs, power = signal.welch(filtered, fs=fps, nperseg=nperseg)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return [], 0.0, float(np.sum(power))
    band_freqs = freqs[mask]
    band_power_values = power[mask]
    peaks, _ = signal.find_peaks(band_power_values)
    ranked: list[int] = []
    if len(peaks):
        ranked.extend(list(peaks[np.argsort(band_power_values[peaks])[::-1]]))
    ranked.extend([int(i) for i in np.argsort(band_power_values)[::-1]])
    unique_ranked: list[int] = []
    seen: set[int] = set()
    for idx in ranked:
        if idx not in seen:
            unique_ranked.append(idx)
            seen.add(idx)
        if len(unique_ranked) >= top_k:
            break
    band_power = float(np.sum(band_power_values) + 1e-12)
    total_power = float(np.sum(power) + 1e-12)
    return (
        [
            {
                "peak_bpm": float(band_freqs[idx] * 60.0),
                "peak_hz": float(band_freqs[idx]),
                "power_fraction": float(band_power_values[idx] / band_power),
                "peak_power": float(band_power_values[idx]),
                "rank": float(rank),
            }
            for rank, idx in enumerate(unique_ranked, start=1)
        ],
        band_power,
        total_power,
    )


def estimate_hr(values: np.ndarray | list[float], fps: float) -> RateEstimate:
    return estimate_rate_fft(values, fps, min_bpm=45.0, max_bpm=180.0)


def estimate_rr_adult(values: np.ndarray | list[float], fps: float) -> RateEstimate:
    return estimate_rate_fft(values, fps, min_bpm=6.0, max_bpm=42.0)


def estimate_rr_infant(values: np.ndarray | list[float], fps: float) -> RateEstimate:
    return estimate_rate_fft(values, fps, min_bpm=15.0, max_bpm=80.0)
