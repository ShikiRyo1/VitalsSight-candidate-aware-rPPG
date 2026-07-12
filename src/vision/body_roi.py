from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .roi import ROI, crop


@dataclass(frozen=True)
class ROIGridConfig:
    rows: int = 3
    cols: int = 3
    margin_x: float = 0.12
    margin_y: float = 0.12


@dataclass(frozen=True)
class ROIQuality:
    roi: ROI
    score: float
    body_overlap: float
    center_prior: float
    texture_score: float
    color_score: float
    border_penalty: float


def respiratory_body_rois(frame: np.ndarray, *, view: str = "", infant: bool = False) -> list[ROI]:
    height, width = frame.shape[:2]
    view = (view or "").lower()

    if infant or view == "infant":
        rois = [
            ROI("infant_torso", int(width * 0.18), int(height * 0.20), int(width * 0.64), int(height * 0.55)),
            ROI("infant_chest", int(width * 0.22), int(height * 0.18), int(width * 0.56), int(height * 0.28)),
            ROI("infant_abdomen", int(width * 0.22), int(height * 0.42), int(width * 0.56), int(height * 0.30)),
        ]
    elif view == "side":
        rois = [
            ROI("side_torso", int(width * 0.22), int(height * 0.18), int(width * 0.56), int(height * 0.62)),
            ROI("side_chest", int(width * 0.25), int(height * 0.18), int(width * 0.50), int(height * 0.30)),
            ROI("side_abdomen", int(width * 0.25), int(height * 0.44), int(width * 0.50), int(height * 0.30)),
        ]
    elif view == "lying":
        rois = [
            ROI("lying_torso", int(width * 0.15), int(height * 0.25), int(width * 0.70), int(height * 0.48)),
            ROI("lying_chest", int(width * 0.18), int(height * 0.25), int(width * 0.32), int(height * 0.45)),
            ROI("lying_abdomen", int(width * 0.48), int(height * 0.25), int(width * 0.34), int(height * 0.45)),
        ]
    else:
        rois = [
            ROI("front_torso", int(width * 0.24), int(height * 0.18), int(width * 0.52), int(height * 0.62)),
            ROI("front_chest", int(width * 0.27), int(height * 0.20), int(width * 0.46), int(height * 0.28)),
            ROI("front_abdomen", int(width * 0.27), int(height * 0.48), int(width * 0.46), int(height * 0.28)),
        ]

    return [roi.clamp(width, height) for roi in rois]


def grid_body_rois(frame: np.ndarray, *, config: ROIGridConfig | None = None, prefix: str = "grid") -> list[ROI]:
    cfg = config or ROIGridConfig()
    height, width = frame.shape[:2]
    x0 = int(width * cfg.margin_x)
    y0 = int(height * cfg.margin_y)
    usable_w = max(1, int(width * (1.0 - 2 * cfg.margin_x)))
    usable_h = max(1, int(height * (1.0 - 2 * cfg.margin_y)))
    cell_w = max(1, usable_w // cfg.cols)
    cell_h = max(1, usable_h // cfg.rows)

    rois: list[ROI] = []
    for row in range(cfg.rows):
        for col in range(cfg.cols):
            rois.append(
                ROI(
                    f"{prefix}_{row}_{col}",
                    x0 + col * cell_w,
                    y0 + row * cell_h,
                    cell_w,
                    cell_h,
                ).clamp(width, height)
            )
    return rois


def candidate_respiration_rois(frame: np.ndarray, *, view: str = "", infant: bool = False) -> list[ROI]:
    semantic = respiratory_body_rois(frame, view=view, infant=infant)
    grid = grid_body_rois(frame, prefix="motion_grid")
    seen: set[tuple[int, int, int, int]] = set()
    rois: list[ROI] = []
    for roi in [*semantic, *grid]:
        key = (roi.x, roi.y, roi.w, roi.h)
        if key not in seen:
            rois.append(roi)
            seen.add(key)
    return rois


def body_aware_roi_scores(frame: np.ndarray, *, view: str = "", infant: bool = False) -> list[ROIQuality]:
    height, width = frame.shape[:2]
    rois = candidate_respiration_rois(frame, view=view, infant=infant)
    body_priors = respiratory_body_rois(frame, view=view, infant=infant)
    max_dist = float(np.hypot(width / 2.0, height / 2.0) + 1e-6)
    scores: list[ROIQuality] = []

    for roi in rois:
        region = crop(frame, roi)
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        body_overlap = max((_overlap_fraction(roi, prior) for prior in body_priors), default=0.0)
        roi_cx = roi.x + roi.w / 2.0
        roi_cy = roi.y + roi.h / 2.0
        center_prior = 1.0 - min(float(np.hypot(roi_cx - width / 2.0, roi_cy - height / 2.0) / max_dist), 1.0)
        texture_score = float(np.clip(np.std(gray) / 64.0, 0.0, 1.0))
        color_score = float(np.clip(np.mean(np.std(region.reshape(-1, 3), axis=0)) / 64.0, 0.0, 1.0))
        border_penalty = _border_penalty(roi, width, height)
        brightness = float(np.mean(gray))
        brightness_penalty = 0.15 if brightness < 20.0 or brightness > 235.0 else 0.0
        score = (
            0.45 * body_overlap
            + 0.25 * center_prior
            + 0.20 * texture_score
            + 0.10 * color_score
            - 0.15 * border_penalty
            - brightness_penalty
        )
        scores.append(
            ROIQuality(
                roi=roi,
                score=float(np.clip(score, 0.0, 1.0)),
                body_overlap=body_overlap,
                center_prior=center_prior,
                texture_score=texture_score,
                color_score=color_score,
                border_penalty=border_penalty,
            )
        )

    return sorted(scores, key=lambda item: item.score, reverse=True)


def body_aware_respiration_rois(
    frame: np.ndarray,
    *,
    view: str = "",
    infant: bool = False,
    max_rois: int | None = None,
    min_score: float = 0.20,
) -> list[ROI]:
    scores = body_aware_roi_scores(frame, view=view, infant=infant)
    selected = [item.roi for item in scores if item.score >= min_score]
    if not selected:
        selected = respiratory_body_rois(frame, view=view, infant=infant)
    return selected[:max_rois] if max_rois is not None else selected


def roi_score_lookup(frame: np.ndarray, *, view: str = "", infant: bool = False) -> dict[str, ROIQuality]:
    return {item.roi.name: item for item in body_aware_roi_scores(frame, view=view, infant=infant)}


def draw_rois(frame: np.ndarray, rois: list[ROI], *, selected: ROI | None = None) -> np.ndarray:
    image = frame.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for roi in rois:
        color = (0, 180, 255)
        thickness = 1
        if selected is not None and roi.name == selected.name:
            color = (0, 255, 0)
            thickness = 2
        cv2.rectangle(image, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, thickness)
        cv2.putText(image, roi.name, (roi.x, max(12, roi.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return image


def resize_roi_gray(frame: np.ndarray, roi: ROI, *, max_side: int = 160) -> np.ndarray:
    region = crop(frame, roi)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return gray
    scale = max_side / longest
    return cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _overlap_fraction(a: ROI, b: ROI) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap = float((x2 - x1) * (y2 - y1))
    return overlap / float(max(1, a.w * a.h))


def _border_penalty(roi: ROI, width: int, height: int) -> float:
    margins = [
        roi.x / max(1, width),
        roi.y / max(1, height),
        (width - (roi.x + roi.w)) / max(1, width),
        (height - (roi.y + roi.h)) / max(1, height),
    ]
    nearest = min(margins)
    if nearest >= 0.08:
        return 0.0
    return float(np.clip((0.08 - nearest) / 0.08, 0.0, 1.0))
