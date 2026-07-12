from __future__ import annotations

import numpy as np
from scipy import signal


def as_clean_array(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if np.isnan(arr).any():
        finite = np.isfinite(arr)
        if finite.any():
            arr = np.interp(np.arange(len(arr)), np.flatnonzero(finite), arr[finite])
        else:
            arr = np.zeros_like(arr)
    return arr


def zscore(values: np.ndarray | list[float], eps: float = 1e-8) -> np.ndarray:
    arr = as_clean_array(values)
    return (arr - np.mean(arr)) / (np.std(arr) + eps)


def detrend(values: np.ndarray | list[float]) -> np.ndarray:
    arr = as_clean_array(values)
    if len(arr) < 3:
        return arr - np.mean(arr)
    return signal.detrend(arr)


def bandpass(values: np.ndarray | list[float], fps: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    arr = detrend(values)
    if fps <= 0:
        raise ValueError("fps must be positive")
    nyquist = fps / 2.0
    high = min(high_hz, nyquist * 0.95)
    low = max(low_hz, 1e-4)
    if low >= high:
        raise ValueError(f"Invalid bandpass range: low={low}, high={high}, fps={fps}")
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass")
    if len(arr) <= max(len(a), len(b)) * 3:
        return signal.lfilter(b, a, arr)
    return signal.filtfilt(b, a, arr)


def moving_average(values: np.ndarray | list[float], window: int = 5) -> np.ndarray:
    arr = as_clean_array(values)
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")
