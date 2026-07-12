from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    name: str
    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> "ROI":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return ROI(self.name, x, y, w, h)


def crop(frame: np.ndarray, roi: ROI) -> np.ndarray:
    h, w = frame.shape[:2]
    r = roi.clamp(w, h)
    return frame[r.y : r.y + r.h, r.x : r.x + r.w]


def center_body_roi(frame: np.ndarray, *, view: str = "", name: str = "body_center") -> ROI:
    height, width = frame.shape[:2]
    view = (view or "").lower()
    if view == "side":
        x, y, w, h = int(width * 0.25), int(height * 0.22), int(width * 0.50), int(height * 0.58)
    elif view == "lying":
        x, y, w, h = int(width * 0.18), int(height * 0.28), int(width * 0.64), int(height * 0.42)
    else:
        x, y, w, h = int(width * 0.25), int(height * 0.20), int(width * 0.50), int(height * 0.60)
    return ROI(name, x, y, w, h).clamp(width, height)


def face_like_rois(frame: np.ndarray) -> list[ROI]:
    """Fast fallback ROIs before MediaPipe integration.

    This intentionally avoids pretending to be a robust detector. It gives us
    a deterministic smoke-test ROI so the signal pipeline can run end to end.
    """
    height, width = frame.shape[:2]
    face = ROI("face_center", int(width * 0.30), int(height * 0.12), int(width * 0.40), int(height * 0.45)).clamp(width, height)
    forehead = ROI("forehead", face.x + int(face.w * 0.20), face.y, int(face.w * 0.60), int(face.h * 0.20)).clamp(width, height)
    left_cheek = ROI("left_cheek", face.x + int(face.w * 0.10), face.y + int(face.h * 0.42), int(face.w * 0.28), int(face.h * 0.28)).clamp(width, height)
    right_cheek = ROI("right_cheek", face.x + int(face.w * 0.62), face.y + int(face.h * 0.42), int(face.w * 0.28), int(face.h * 0.28)).clamp(width, height)
    return [face, forehead, left_cheek, right_cheek]


def detect_face_roi(frame: np.ndarray) -> ROI | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cascade_path = _opencv_safe_cascade_path()
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return ROI("haar_face", int(x), int(y), int(w), int(h)).clamp(frame.shape[1], frame.shape[0])


def _opencv_safe_cascade_path() -> str:
    source = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    target = Path(tempfile.gettempdir()) / "contactless_vitals_haarcascade_frontalface_default.xml"
    if not target.exists():
        shutil.copyfile(source, target)
    return str(target)


def stable_face_roi(frames: list[np.ndarray]) -> ROI | None:
    detections = [detect_face_roi(frame) for frame in frames]
    detections = [roi for roi in detections if roi is not None]
    if not detections:
        return None
    xs = np.asarray([r.x for r in detections])
    ys = np.asarray([r.y for r in detections])
    ws = np.asarray([r.w for r in detections])
    hs = np.asarray([r.h for r in detections])
    return ROI(
        "haar_face_stable",
        int(np.median(xs)),
        int(np.median(ys)),
        int(np.median(ws)),
        int(np.median(hs)),
    )


def mean_rgb(frame: np.ndarray, roi: ROI) -> np.ndarray:
    region = crop(frame, roi)
    if region.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return region.reshape(-1, region.shape[-1]).mean(axis=0)


def motion_energy(prev_frame: np.ndarray | None, frame: np.ndarray, roi: ROI) -> float:
    if prev_frame is None:
        return 0.0
    current = crop(frame, roi)
    previous = crop(prev_frame, roi)
    if current.shape != previous.shape or current.size == 0:
        return 0.0
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    return float(np.mean(np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))))
