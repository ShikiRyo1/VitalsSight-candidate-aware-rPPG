from __future__ import annotations

import math

import numpy as np


def _paired(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(true), len(pred))
    true = true[:n]
    pred = pred[:n]
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def mae(y_true, y_pred) -> float:
    true, pred = _paired(y_true, y_pred)
    if len(true) == 0:
        return math.nan
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true, pred = _paired(y_true, y_pred)
    if len(true) == 0:
        return math.nan
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def pearson(y_true, y_pred) -> float:
    true, pred = _paired(y_true, y_pred)
    if len(true) < 2 or np.std(true) == 0 or np.std(pred) == 0:
        return math.nan
    return float(np.corrcoef(true, pred)[0, 1])


def coverage(predictions) -> float:
    arr = np.asarray(predictions, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def metric_summary(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "coverage": coverage(y_pred),
    }
