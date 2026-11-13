
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MaskRecord:
    frame_index: int
    mask: Optional[np.ndarray]
    bbox: Optional[tuple]
    confidence: float
    detection_index: int
    status: str


def calculate_iou(box_a, box_b):
    """Calculate IoU between two xyxy bounding boxes."""
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def calculate_center_distance(box_a, box_b, frame_shape):
    """Return normalized center distance."""
    if box_a is None or box_b is None:
        return 1.0

    h, w = frame_shape[:2]

    center_a = np.array([
        (box_a[0] + box_a[2]) / 2,
        (box_a[1] + box_a[3]) / 2
    ])

    center_b = np.array([
        (box_b[0] + box_b[2]) / 2,
        (box_b[1] + box_b[3]) / 2
    ])

    diagonal = np.sqrt(w ** 2 + h ** 2)

    return np.linalg.norm(center_a - center_b) / diagonal


def calculate_area_similarity(box_a, box_b):
    """Compare bounding box areas."""
    if box_a is None or box_b is None:
        return 0.0

    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    return min(area_a, area_b) / max(area_a, area_b)


def extract_cat_detections(
    result,
    frame_shape,
    cat_class_id=15,
    min_confidence=0.25,
):
    """
    Extract cat masks and bounding boxes from one Ultralytics Results object.

    Requires prediction with retina_masks=True so masks are aligned
    with the original frame dimensions.
    """
    detections = []

    if result.boxes is None or result.masks is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    h, w = frame_shape[:2]

    for i, (box, cls_id, conf) in enumerate(
        zip(boxes, classes, confidences)
    ):
        if cls_id != cat_class_id:
            continue

        if conf < min_confidence:
            continue

        mask = masks[i]

        # Ensure mask dimensions match the original frame.
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        binary_mask = (mask > 0).astype(np.uint8)

        # Remove empty or extremely small masks.
        mask_area = int(binary_mask.sum())

        if mask_area == 0:
            continue

        bbox = tuple(
            np.round(box).astype(int).tolist()
        )

        detections.append({
            "mask": binary_mask,
            "bbox": bbox,
            "confidence": float(conf),
            "detection_index": i,
            "area": mask_area,
        })

    return detections


def select_target_cat(
    detections,
    frame_shape,
    previous_bbox=None,
):
    """
    Select one cat detection.

    First frame:
        Select the detection with the highest area-confidence score.

    Subsequent frames:
        Select based on spatial and size continuity.
    """
    if not detections:
        return None

    if previous_bbox is None:
        return max(
            detections,
            key=lambda d: d["area"] * d["confidence"],
        )

    scored = []

    for detection in detections:
        bbox = detection["bbox"]

        iou = calculate_iou(previous_bbox, bbox)

        distance = calculate_center_distance(
            previous_bbox,
            bbox,
            frame_shape,
        )

        area_similarity = calculate_area_similarity(
            previous_bbox,
            bbox,
        )

        # A higher score indicates stronger temporal consistency.
        score = (
            0.50 * iou
            + 0.30 * max(0.0, 1.0 - distance)
            + 0.20 * area_similarity
        )

        scored.append((score, detection))

    best_score, best_detection = max(
        scored,
        key=lambda x: x[0],
    )

    # Avoid selecting a completely unrelated detection.
    if best_score < 0.15:
        return None

    return best_detection


def clean_mask(mask, kernel_size=5):
    """Remove small noise and fill small gaps."""
    if mask is None:
        return None

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    cleaned = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
    )

    return cleaned


def process_frame(
    result,
    frame,
    frame_index,
    previous_bbox=None,
    cat_class_id=15,
    min_confidence=0.25,
):
    """
    Process one frame and return its selected cat mask.
    """
    detections = extract_cat_detections(
        result=result,
        frame_shape=frame.shape,
        cat_class_id=cat_class_id,
        min_confidence=min_confidence,
    )

    selected = select_target_cat(
        detections=detections,
        frame_shape=frame.shape,
        previous_bbox=previous_bbox,
    )

    if selected is None:
        return MaskRecord(
            frame_index=frame_index,
            mask=None,
            bbox=None,
            confidence=0.0,
            detection_index=-1,
            status="MISSING",
        )

    mask = clean_mask(selected["mask"])

    return MaskRecord(
        frame_index=frame_index,
        mask=mask,
        bbox=selected["bbox"],
        confidence=selected["confidence"],
        detection_index=selected["detection_index"],
        status="MATCHED",
    )
