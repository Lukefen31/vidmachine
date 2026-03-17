"""
YOLOv8 (Ultralytics) subject detection for automatic 9:16 reframing.

Uses ultralytics YOLO("yolov8n.pt") — the model auto-downloads on first use.
Requires: ultralytics, opencv-python.

If ultralytics is not installed, all public functions raise ImportError with
a clear install instruction.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    _YOLO = None  # type: ignore[assignment,misc]

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


def _require_deps() -> None:
    missing = []
    if not _ULTRALYTICS_AVAILABLE:
        missing.append("ultralytics")
    if not _CV2_AVAILABLE:
        missing.append("opencv-python")
    if missing:
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# YOLO class name index (COCO dataset — yolov8n.pt trained on COCO)
# ---------------------------------------------------------------------------

_COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model() -> "_YOLO":
    return _YOLO("yolov8n.pt")


def _extract_frames(
    video_path: str,
    sample_interval: float,
) -> list[tuple[float, "np.ndarray"]]:
    """
    Open video with OpenCV and extract one frame every sample_interval seconds.
    Returns list of (timestamp_seconds, frame_bgr_array).
    """
    cap = _cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps: float = cap.get(_cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    frame_step: int = max(1, int(fps * sample_interval))

    frames: list[tuple[float, np.ndarray]] = []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frames.append((t, frame))
        frame_idx += frame_step

    cap.release()
    return frames


def _bbox_to_center(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """Convert xyxy bbox to (cx, cy, w, h)."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h


def _rolling_average(values: list[float], window: int) -> list[float]:
    """Apply a symmetric rolling average to a 1-D list of floats."""
    if len(values) <= 1:
        return list(values)
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(float(np.mean(values[lo:hi])))
    return smoothed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_subjects_in_video(
    video_path: str,
    sample_interval: float = 0.5,
    confidence_threshold: float = 0.3,
    target_classes: list[str] | None = None,
) -> list[dict]:
    """
    Sample frames from the video at sample_interval seconds, run YOLOv8
    detection on each frame, and return per-frame detections.

    Returns list of dicts:
        {t, x_center, y_center, width, height, class_name, confidence}

    target_classes: if None, detect all COCO classes.
                    Common useful values: ["person", "car", "bird", "sports ball"]
    """
    _require_deps()

    model = _load_model()
    frames = _extract_frames(video_path, sample_interval)

    # Normalise target class names to lower-case for comparison
    target_set: set[str] | None = (
        {c.lower() for c in target_classes} if target_classes else None
    )

    detections: list[dict] = []

    for t, frame in frames:
        try:
            results = model(frame, verbose=False)
        except Exception as exc:
            logger.warning("YOLO inference failed at t=%.2f: %s", t, exc)
            continue

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                class_name = (
                    _COCO_CLASSES[cls_id]
                    if cls_id < len(_COCO_CLASSES)
                    else str(cls_id)
                )
                if target_set is not None and class_name.lower() not in target_set:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx, cy, bw, bh = _bbox_to_center(x1, y1, x2, y2)

                detections.append({
                    "t": t,
                    "x_center": cx,
                    "y_center": cy,
                    "width": bw,
                    "height": bh,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                })

    logger.info("detect_subjects_in_video: %d detections from %d frames", len(detections), len(frames))
    return detections


def compute_reframe_coords(
    detections: list[dict],
    source_width: int,
    source_height: int,
    target_aspect: float = 9 / 16,
    smoothing_window: int = 10,
) -> list[dict]:
    """
    Given per-frame subject detections, compute smooth crop keyframes for 9:16 output.

    Algorithm:
      1. For each timestep, pick the highest-confidence detection.
      2. Compute a crop box of fixed output aspect ratio centred on the subject,
         scaled so the subject fits with some padding.
      3. Apply temporal smoothing (rolling average) on crop x/y to prevent jitter.
      4. Clamp crop coordinates to frame boundaries.
      5. If no detections exist at all, fall back to a single centre-crop keyframe.

    Returns list of {t, x, y, w, h} dicts (sorted by t).
    """
    # Target crop size: full height, width determined by 9:16 aspect
    crop_h = source_height
    crop_w = int(crop_h * target_aspect)  # e.g. 1080 * (9/16) = 607.5 → 607

    if not detections:
        # Centre crop fallback
        cx = (source_width - crop_w) // 2
        cy = 0
        return [{"t": 0.0, "x": cx, "y": cy, "w": crop_w, "h": crop_h}]

    # Group detections by timestamp, keep highest-confidence per frame
    by_time: dict[float, dict] = {}
    for det in detections:
        t = det["t"]
        if t not in by_time or det["confidence"] > by_time[t]["confidence"]:
            by_time[t] = det

    sorted_times = sorted(by_time.keys())

    raw_cx: list[float] = []
    raw_cy: list[float] = []

    for t in sorted_times:
        det = by_time[t]
        # Centre crop on the subject's centre
        cx = det["x_center"] - crop_w / 2
        cy = det["y_center"] - crop_h / 2
        raw_cx.append(cx)
        raw_cy.append(cy)

    # Temporal smoothing
    smooth_cx = _rolling_average(raw_cx, smoothing_window)
    smooth_cy = _rolling_average(raw_cy, smoothing_window)

    keyframes: list[dict] = []
    for i, t in enumerate(sorted_times):
        # Clamp to frame boundaries
        x = int(max(0, min(smooth_cx[i], source_width - crop_w)))
        y = int(max(0, min(smooth_cy[i], source_height - crop_h)))
        keyframes.append({"t": t, "x": x, "y": y, "w": crop_w, "h": crop_h})

    return keyframes


def generate_reframe_keyframes(
    video_path: str,
    source_width: int = 1920,
    source_height: int = 1080,
) -> list[dict]:
    """
    Convenience function: detect subjects → compute smooth crop → return keyframes
    ready to be stored in blueprint.reframe_9x16.keyframes.

    Prioritises "person" detections; falls back to all classes if no persons found.
    Returns list of {t, x, y, w, h} dicts.
    """
    _require_deps()

    logger.info("generate_reframe_keyframes: running YOLO detection on %s", video_path)

    # First try person-only detection
    detections = detect_subjects_in_video(
        video_path,
        sample_interval=0.5,
        confidence_threshold=0.3,
        target_classes=["person"],
    )

    # If no persons found, try all classes
    if not detections:
        logger.info("No persons detected; falling back to all-class detection")
        detections = detect_subjects_in_video(
            video_path,
            sample_interval=0.5,
            confidence_threshold=0.3,
            target_classes=None,
        )

    keyframes = compute_reframe_coords(
        detections,
        source_width=source_width,
        source_height=source_height,
        target_aspect=9 / 16,
        smoothing_window=10,
    )

    logger.info("generate_reframe_keyframes: produced %d keyframes", len(keyframes))
    return keyframes
