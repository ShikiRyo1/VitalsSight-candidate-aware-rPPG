from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA

from src.signal.filters import zscore


def _standardize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("rgb must have shape (n_frames, 3)")
    means = np.mean(arr, axis=0, keepdims=True)
    return arr / (means + 1e-8) - 1.0


def green(rgb: np.ndarray) -> np.ndarray:
    return zscore(np.asarray(rgb, dtype=float)[:, 1])


def chrom(rgb: np.ndarray) -> np.ndarray:
    c = _standardize_rgb(rgb)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = np.std(x) / (np.std(y) + 1e-8)
    return zscore(x - alpha * y)


def pos(rgb: np.ndarray) -> np.ndarray:
    c = _standardize_rgb(rgb)
    h1 = c[:, 1] - c[:, 2]
    h2 = -2.0 * c[:, 0] + c[:, 1] + c[:, 2]
    alpha = np.std(h1) / (np.std(h2) + 1e-8)
    return zscore(h1 + alpha * h2)


def pbv(rgb: np.ndarray) -> np.ndarray:
    c = _standardize_rgb(rgb)
    # Blood-volume pulse signature from the PBV family of methods. This simple
    # projection is intentionally kept as a baseline, not our proposed method.
    signature = np.asarray([0.33, 0.77, 0.53], dtype=float)
    signature = signature / np.linalg.norm(signature)
    projection = c @ signature
    return zscore(projection)


def ica(rgb: np.ndarray) -> np.ndarray:
    c = _standardize_rgb(rgb)
    if c.shape[0] < 10:
        return green(rgb)
    model = FastICA(n_components=3, whiten="unit-variance", random_state=7, max_iter=500, tol=1e-3)
    try:
        sources = model.fit_transform(c)
    except Exception:
        return green(rgb)
    # Choose the component with the largest temporal variance after normalization.
    idx = int(np.argmax(np.std(sources, axis=0)))
    return zscore(sources[:, idx])


def lgi(rgb: np.ndarray) -> np.ndarray:
    c = _standardize_rgb(rgb).T.reshape(1, 3, -1)
    if c.shape[-1] < 10:
        return green(rgb)
    try:
        u, _, _ = np.linalg.svd(c, full_matrices=False)
        skin_axis = u[:, :, 0][:, :, np.newaxis]
        projection = np.eye(3)[np.newaxis, :, :] - np.matmul(skin_axis, np.swapaxes(skin_axis, 1, 2))
        projected = np.matmul(projection, c)
    except Exception:
        return green(rgb)
    return zscore(projected[:, 1, :].reshape(-1))


METHODS = {
    "GREEN": green,
    "CHROM": chrom,
    "POS": pos,
    "PBV": pbv,
    "ICA": ica,
    "LGI": lgi,
}
