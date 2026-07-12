from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from src.signal.estimate import RateEstimate, estimate_rate_fft
from src.signal.filters import as_clean_array, bandpass, zscore


@dataclass(frozen=True)
class HalfRateValidation:
    estimate: RateEstimate
    raw_estimate: RateEstimate
    decision: str
    half_bpm: float
    half_power_ratio: float
    previous_bpm: float | None


def spectral_support_ratio(
    values: np.ndarray | list[float],
    fps: float,
    *,
    target_bpm: float,
    min_bpm: float,
    max_bpm: float,
) -> float:
    """Return local spectral support at target_bpm relative to the strongest band peak."""
    if fps <= 0 or not np.isfinite(target_bpm):
        return 0.0

    arr = zscore(as_clean_array(values))
    if len(arr) < 8:
        return 0.0

    low_hz = min_bpm / 60.0
    high_hz = max_bpm / 60.0
    target_hz = target_bpm / 60.0
    if target_hz < low_hz or target_hz > high_hz:
        return 0.0

    try:
        filtered = bandpass(arr, fps, low_hz, high_hz)
    except ValueError:
        return 0.0

    nperseg = min(len(filtered), max(32, int(fps * 16)))
    freqs, power = signal.welch(filtered, fs=fps, nperseg=nperseg)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return 0.0

    band_freqs = freqs[mask]
    band_power = power[mask]
    peak_power = float(np.max(band_power) + 1e-12)
    idx = int(np.argmin(np.abs(band_freqs - target_hz)))
    left = max(0, idx - 1)
    right = min(len(band_power), idx + 2)
    local_power = float(np.max(band_power[left:right]))
    return float(local_power / peak_power)


def estimate_rr_half_rate_validated(
    values: np.ndarray | list[float],
    fps: float,
    *,
    min_bpm: float = 15.0,
    max_bpm: float = 80.0,
    previous_bpm: float | None = None,
    high_raw_bpm: float = 37.5,
    half_max_bpm: float = 42.0,
    min_half_support: float = 0.02,
    temporal_margin_bpm: float = 6.0,
) -> HalfRateValidation:
    """Estimate RR, then conservatively suppress likely two-times frequency errors.

    Motion-energy respiration traces can fire once on inhalation and once on exhalation,
    creating a dominant spectral peak near 2x the real respiratory rate. This validator
    only halves the estimate when the half-rate candidate is physiologically plausible
    and has either temporal or spectral support.
    """
    raw = estimate_rate_fft(values, fps, min_bpm=min_bpm, max_bpm=max_bpm)
    if not np.isfinite(raw.bpm):
        return HalfRateValidation(raw, raw, "raw_nonfinite", float("nan"), 0.0, previous_bpm)

    half_bpm = raw.bpm / 2.0
    if half_bpm < min_bpm or half_bpm > max_bpm:
        return HalfRateValidation(raw, raw, "raw_half_out_of_band", half_bpm, 0.0, previous_bpm)

    half_ratio = spectral_support_ratio(
        values,
        fps,
        target_bpm=half_bpm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    choose_half = False
    decision = "raw_kept"
    if previous_bpm is not None and np.isfinite(previous_bpm):
        raw_jump = abs(raw.bpm - previous_bpm)
        half_jump = abs(half_bpm - previous_bpm)
        if half_jump + temporal_margin_bpm < raw_jump and half_ratio >= min_half_support:
            choose_half = True
            decision = "half_temporal"

    if not choose_half and raw.bpm >= high_raw_bpm and half_bpm <= half_max_bpm and half_ratio >= min_half_support:
        choose_half = True
        decision = "half_high_raw"

    if not choose_half:
        return HalfRateValidation(raw, raw, decision, half_bpm, half_ratio, previous_bpm)

    validated = RateEstimate(
        bpm=float(half_bpm),
        peak_hz=float(raw.peak_hz / 2.0),
        confidence=float(raw.confidence * min(1.0, max(0.25, half_ratio))),
        band_power=raw.band_power,
        total_power=raw.total_power,
    )
    return HalfRateValidation(validated, raw, decision, half_bpm, half_ratio, previous_bpm)
