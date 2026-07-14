from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from .roi import ROI, crop, face_like_rois


LANDMARK_GROUPS: Mapping[str, tuple[int, ...]] = {
    "forehead": (10, 109, 67, 103, 332, 284, 295),
    "left_cheek": (234, 227, 116, 117, 118, 119, 120),
    "right_cheek": (454, 447, 345, 346, 347, 348, 349),
    "nasal_bridge": (168, 193, 194, 195, 197, 4),
    "nose_tip": (1, 2, 98, 327, 278),
    "chin": (152, 148, 149, 150, 151, 175, 176),
}


TCM_INTERPRETIVE_LABELS: Mapping[str, str] = {
    "forehead": "heart_zone",
    "left_cheek": "liver_zone",
    "right_cheek": "lung_zone",
    "nasal_bridge": "liver_zone",
    "nose_tip": "spleen_zone",
    "chin": "kidney_zone",
}


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_FACE_LANDMARKER_MODEL = (
    Path(__file__).resolve().parents[2] / "runtime" / "models" / "face_landmarker.task"
)


def resolve_face_landmarker_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve an explicit, environment-provided, or locally installed model asset."""

    if model_path:
        return Path(model_path).expanduser().resolve()
    configured = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_TASK")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (
        DEFAULT_FACE_LANDMARKER_MODEL,
        Path(__file__).resolve().parents[2] / "third_party" / "mediapipe" / "face_landmarker.task",
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())


def face_landmarker_model_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_face_landmarker_model(path: str | Path) -> str:
    """Return the model hash only when the runtime asset matches the pinned bundle."""

    model = Path(path)
    if not model.is_file():
        raise FileNotFoundError(f"Face Landmarker model is missing: {model.name}")
    observed = face_landmarker_model_sha256(model)
    if observed != FACE_LANDMARKER_MODEL_SHA256:
        raise ValueError(
            "Face Landmarker SHA256 mismatch: "
            f"expected {FACE_LANDMARKER_MODEL_SHA256}, observed {observed}"
        )
    return observed


@dataclass(frozen=True)
class FaceRegionMask:
    name: str
    roi: ROI
    mask: np.ndarray
    landmark_indices: tuple[int, ...]
    interpretive_label: str
    area_px: int
    coverage: float


class MediaPipeFaceLandmarkDetector:
    """Reusable MediaPipe face-landmark detector.

    Newer MediaPipe builds expose the Tasks API, while older builds expose
    ``mp.solutions.face_mesh``. This wrapper supports both and avoids creating a
    new model object for every video frame.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except Exception:
            mp = None
        self.mp = mp
        self.model_path = str(resolve_face_landmarker_model_path(model_path))
        self.model_sha256: str | None = None
        self.model_integrity_status = "not_checked"
        self.initialization_error = ""
        self._mesh = None
        self._landmarker = None
        if self.mp is None:
            return
        if hasattr(self.mp, "solutions"):
            self.model_integrity_status = "not_applicable_builtin_face_mesh"
            self._mesh = self.mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
        else:
            try:
                self.model_sha256 = validate_face_landmarker_model(self.model_path)
                self.model_integrity_status = "verified_pinned_sha256"
                model_buffer = Path(self.model_path).read_bytes()
                base_options = self.mp.tasks.BaseOptions(model_asset_buffer=model_buffer)  # type: ignore[attr-defined]
                options = self.mp.tasks.vision.FaceLandmarkerOptions(  # type: ignore[attr-defined]
                    base_options=base_options,
                    running_mode=self.mp.tasks.vision.RunningMode.IMAGE,  # type: ignore[attr-defined]
                    num_faces=1,
                )
                self._landmarker = self.mp.tasks.vision.FaceLandmarker.create_from_options(options)  # type: ignore[attr-defined]
            except Exception as error:
                self.initialization_error = f"{type(error).__name__}: {str(error)[:240]}"
                self.model_integrity_status = "failed"
                self._landmarker = None

    @property
    def available(self) -> bool:
        return self._mesh is not None or self._landmarker is not None

    @property
    def backend(self) -> str:
        if self._mesh is not None:
            return "mediapipe_face_mesh"
        if self._landmarker is not None:
            return "mediapipe_face_landmarker_task"
        if self.mp is None:
            return "mediapipe_package_unavailable"
        if self.model_integrity_status == "failed":
            return "mediapipe_model_integrity_failed"
        return "mediapipe_model_unavailable"

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        if self.mp is None:
            return None
        if self._mesh is not None:
            result = self._mesh.process(frame)
            if not result.multi_face_landmarks:
                return None
            height, width = frame.shape[:2]
            return normalized_landmarks_to_pixels(result.multi_face_landmarks[0].landmark, width, height)
        if self._landmarker is not None:
            try:
                image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))  # type: ignore[attr-defined]
                result = self._landmarker.detect(image)
            except Exception:
                return None
            if not getattr(result, "face_landmarks", None):
                return None
            height, width = frame.shape[:2]
            return normalized_landmarks_to_pixels(result.face_landmarks[0], width, height)
        return None

    def close(self) -> None:
        for obj in (self._mesh, self._landmarker):
            if obj is not None and hasattr(obj, "close"):
                obj.close()

    def __enter__(self) -> "MediaPipeFaceLandmarkDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def normalized_landmarks_to_pixels(landmarks: Sequence[object], width: int, height: int) -> np.ndarray:
    """Convert MediaPipe-style normalized landmarks to pixel coordinates.

    Each landmark may be a MediaPipe object with ``x``/``y`` attributes, a
    mapping with ``x``/``y`` keys, or a two-item sequence.
    """

    points: list[tuple[float, float]] = []
    for item in landmarks:
        if hasattr(item, "x") and hasattr(item, "y"):
            x = float(getattr(item, "x")) * width
            y = float(getattr(item, "y")) * height
        elif isinstance(item, Mapping):
            x = float(item["x"]) * width
            y = float(item["y"]) * height
        else:
            values = list(item)  # type: ignore[arg-type]
            x = float(values[0]) * width
            y = float(values[1]) * height
        points.append((x, y))
    return np.asarray(points, dtype=np.float32)


def face_region_masks_from_landmarks(
    frame: np.ndarray,
    landmarks: Sequence[object] | np.ndarray,
    *,
    groups: Mapping[str, Iterable[int]] | None = None,
    dilation_px: int = 3,
) -> list[FaceRegionMask]:
    height, width = frame.shape[:2]
    if isinstance(landmarks, np.ndarray):
        points = landmarks.astype(np.float32)
    else:
        points = normalized_landmarks_to_pixels(landmarks, width, height)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("landmarks must have shape (n_landmarks, 2+) or MediaPipe-style x/y objects")

    masks: list[FaceRegionMask] = []
    for name, raw_indices in (groups or LANDMARK_GROUPS).items():
        indices = tuple(int(idx) for idx in raw_indices)
        if not indices or max(indices) >= len(points):
            continue
        region_points = points[list(indices), :2]
        mask = _convex_hull_mask(region_points, width, height, dilation_px=dilation_px)
        area = int(mask.sum())
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        roi = ROI(name, int(x), int(y), int(w), int(h)).clamp(width, height)
        masks.append(
            FaceRegionMask(
                name=name,
                roi=roi,
                mask=mask,
                landmark_indices=indices,
                interpretive_label=TCM_INTERPRETIVE_LABELS.get(name, "face_zone"),
                area_px=area,
                coverage=float(area / max(1, width * height)),
            )
        )
    return masks


def detect_mediapipe_face_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """Detect one face mesh with MediaPipe if the optional dependency exists."""

    with MediaPipeFaceLandmarkDetector() as detector:
        return detector.detect(frame)


def mentor_aligned_face_rois(frame: np.ndarray, *, landmarks: np.ndarray | None = None) -> list[FaceRegionMask]:
    """Return six mentor-aligned facial ROI masks or a deterministic fallback."""

    points = landmarks if landmarks is not None else detect_mediapipe_face_landmarks(frame)
    if points is not None:
        masks = face_region_masks_from_landmarks(frame, points)
        if masks:
            return masks
    return [_roi_to_mask(frame, roi, label=TCM_INTERPRETIVE_LABELS.get(roi.name, "fallback_face_zone")) for roi in face_like_rois(frame)]


def dense_face_patch_masks_from_landmarks(
    frame: np.ndarray,
    landmarks: Sequence[object] | np.ndarray,
    *,
    tile_px: int = 20,
    max_grid_patches: int = 62,
    landmark_patch_counts: Sequence[int] = (50,),
    patch_px: int = 24,
    include_semantic_regions: bool = True,
    min_patch_area_px: int = 25,
) -> list[FaceRegionMask]:
    """Build dense face-patch candidates from MediaPipe landmarks.

    This supports the T664A dense-patch hypothesis: instead of committing to a
    small fixed ROI set, generate many local candidates and let downstream
    quality/selector logic decide which patches are usable.
    """

    if tile_px < 4:
        raise ValueError("tile_px must be >= 4")
    if patch_px < 4:
        raise ValueError("patch_px must be >= 4")

    height, width = frame.shape[:2]
    points = landmarks.astype(np.float32) if isinstance(landmarks, np.ndarray) else normalized_landmarks_to_pixels(landmarks, width, height)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("landmarks must have shape (n_landmarks, 2+) or MediaPipe-style x/y objects")

    face_mask = _convex_hull_mask(points[:, :2], width, height, dilation_px=0)
    x0, y0, w0, h0 = cv2.boundingRect(face_mask.astype(np.uint8))

    regions: list[FaceRegionMask] = []
    if include_semantic_regions:
        regions.extend(face_region_masks_from_landmarks(frame, points))

    grid_regions: list[FaceRegionMask] = []
    idx = 0
    for y in range(y0, y0 + h0, tile_px):
        for x in range(x0, x0 + w0, tile_px):
            rect = np.zeros((height, width), dtype=np.uint8)
            rect[y : min(height, y + tile_px), x : min(width, x + tile_px)] = 1
            mask = (rect & face_mask).astype(np.uint8)
            area = int(mask.sum())
            if area < min_patch_area_px:
                continue
            bx, by, bw, bh = cv2.boundingRect(mask)
            grid_regions.append(
                FaceRegionMask(
                    name=f"grid_{idx:03d}",
                    roi=ROI(f"grid_{idx:03d}", int(bx), int(by), int(bw), int(bh)).clamp(width, height),
                    mask=mask,
                    landmark_indices=(),
                    interpretive_label="dense_grid_patch",
                    area_px=area,
                    coverage=float(area / max(1, width * height)),
                )
            )
            idx += 1
    if max_grid_patches > 0 and len(grid_regions) > max_grid_patches:
        step = len(grid_regions) / max_grid_patches
        grid_regions = [grid_regions[min(len(grid_regions) - 1, int(i * step))] for i in range(max_grid_patches)]
    regions.extend(grid_regions)

    for count in landmark_patch_counts:
        if count <= 0:
            continue
        selected = np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int)
        half = patch_px // 2
        for local_idx, lm_idx in enumerate(selected):
            cx, cy = points[lm_idx, :2]
            x = int(round(cx)) - half
            y = int(round(cy)) - half
            rect = np.zeros((height, width), dtype=np.uint8)
            rect[max(0, y) : min(height, y + patch_px), max(0, x) : min(width, x + patch_px)] = 1
            mask = (rect & face_mask).astype(np.uint8)
            area = int(mask.sum())
            if area < min_patch_area_px:
                continue
            bx, by, bw, bh = cv2.boundingRect(mask)
            name = f"lm{count}_{local_idx:03d}"
            regions.append(
                FaceRegionMask(
                    name=name,
                    roi=ROI(name, int(bx), int(by), int(bw), int(bh)).clamp(width, height),
                    mask=mask,
                    landmark_indices=(int(lm_idx),),
                    interpretive_label="landmark_patch",
                    area_px=area,
                    coverage=float(area / max(1, width * height)),
                )
            )

    # Preserve deterministic order and avoid accidental duplicate names.
    unique: dict[str, FaceRegionMask] = {}
    for region in regions:
        unique.setdefault(region.name, region)
    return list(unique.values())


def masked_mean_rgb(frame: np.ndarray, region: FaceRegionMask) -> np.ndarray:
    if region.mask.shape != frame.shape[:2]:
        raise ValueError("mask shape must match frame height/width")
    pixels = frame[region.mask.astype(bool)]
    if pixels.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return pixels.reshape(-1, frame.shape[-1]).mean(axis=0).astype(float)


def extract_region_rgb_features(frame: np.ndarray, regions: Sequence[FaceRegionMask]) -> dict[str, float]:
    features: dict[str, float] = {}
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB) if frame.ndim == 3 and frame.shape[2] == 3 else None
    for region in regions:
        mean = masked_mean_rgb(frame, region)
        features[f"{region.name}_mean_r"] = float(mean[0])
        features[f"{region.name}_mean_g"] = float(mean[1])
        features[f"{region.name}_mean_b"] = float(mean[2])
        features[f"{region.name}_area_px"] = float(region.area_px)
        features[f"{region.name}_coverage"] = float(region.coverage)
        if lab is not None:
            lab_pixels = lab[region.mask.astype(bool)]
            if lab_pixels.size:
                lab_mean = lab_pixels.reshape(-1, 3).mean(axis=0)
                features[f"{region.name}_lab_l"] = float(lab_mean[0])
                features[f"{region.name}_lab_a"] = float(lab_mean[1])
                features[f"{region.name}_lab_b"] = float(lab_mean[2])
    return features


def draw_face_region_masks(frame: np.ndarray, regions: Sequence[FaceRegionMask]) -> np.ndarray:
    image = frame.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    palette = [
        (20, 120, 220),
        (40, 170, 90),
        (220, 110, 30),
        (180, 90, 200),
        (60, 170, 200),
        (180, 180, 40),
    ]
    overlay = image.copy()
    for idx, region in enumerate(regions):
        color = palette[idx % len(palette)]
        overlay[region.mask.astype(bool)] = color
        cv2.rectangle(image, (region.roi.x, region.roi.y), (region.roi.x + region.roi.w, region.roi.y + region.roi.h), color, 1)
        cv2.putText(image, region.name, (region.roi.x, max(12, region.roi.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0)


def _convex_hull_mask(points: np.ndarray, width: int, height: int, *, dilation_px: int) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    hull = cv2.convexHull(clipped.astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def _roi_to_mask(frame: np.ndarray, roi: ROI, *, label: str) -> FaceRegionMask:
    height, width = frame.shape[:2]
    clamped = roi.clamp(width, height)
    mask = np.zeros((height, width), dtype=bool)
    mask[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w] = True
    return FaceRegionMask(
        name=clamped.name,
        roi=clamped,
        mask=mask,
        landmark_indices=(),
        interpretive_label=label,
        area_px=int(mask.sum()),
        coverage=float(mask.sum() / max(1, width * height)),
    )
