
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class ROICandidate:
    candidate_id: int
    bbox: tuple
    crop: np.ndarray
    mask: np.ndarray
    occupancy: float
    mask_area: int
    center_normalized: tuple


def get_mask_bbox(mask):
    """Return xyxy bounding box of a binary mask."""
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def extract_roi_candidates(
    frame: np.ndarray,
    mask: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    min_occupancy: float = 0.65,
    min_mask_area: int = 300,
    top_k: int = 8,
) -> List[ROICandidate]:
    """
    Divide the cat mask bounding box into grid cells.

    Return candidate regions with sufficient cat-mask occupancy.
    Each ROI retains its corresponding binary mask.
    """
    if mask is None:
        return []

    if frame.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "Frame and mask dimensions must match."
        )

    binary_mask = (mask > 0).astype(np.uint8)

    bbox = get_mask_bbox(binary_mask)

    if bbox is None:
        return []

    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    if bbox_width <= 0 or bbox_height <= 0:
        return []

    cell_width = bbox_width / grid_cols
    cell_height = bbox_height / grid_rows

    candidates = []

    candidate_id = 0

    for row in range(grid_rows):
        for col in range(grid_cols):

            rx1 = int(x1 + col * cell_width)
            ry1 = int(y1 + row * cell_height)

            rx2 = int(x1 + (col + 1) * cell_width)
            ry2 = int(y1 + (row + 1) * cell_height)

            rx2 = min(rx2, x2)
            ry2 = min(ry2, y2)

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            roi_mask = binary_mask[ry1:ry2, rx1:rx2]

            cell_area = roi_mask.size
            mask_area = int(roi_mask.sum())

            if cell_area == 0 or mask_area < min_mask_area:
                continue

            occupancy = mask_area / cell_area

            if occupancy < min_occupancy:
                continue

            roi_frame = frame[ry1:ry2, rx1:rx2].copy()

            # Keep only pixels belonging to the cat mask.
            masked_crop = cv2.bitwise_and(
                roi_frame,
                roi_frame,
                mask=roi_mask,
            )

            center_x = (rx1 + rx2) / 2
            center_y = (ry1 + ry2) / 2

            center_normalized = (
                (center_x - x1) / bbox_width,
                (center_y - y1) / bbox_height,
            )

            candidates.append(
                ROICandidate(
                    candidate_id=candidate_id,
                    bbox=(rx1, ry1, rx2, ry2),
                    crop=masked_crop,
                    mask=roi_mask.copy(),
                    occupancy=float(occupancy),
                    mask_area=mask_area,
                    center_normalized=center_normalized,
                )
            )

            candidate_id += 1

    # Prefer regions with high mask occupancy.
    candidates.sort(
        key=lambda item: (
            item.occupancy,
            item.mask_area,
        ),
        reverse=True,
    )

    return candidates[:top_k]


def draw_roi_candidates(
    frame: np.ndarray,
    candidates: List[ROICandidate],
) -> np.ndarray:
    """Visualize ROI candidate boxes and IDs."""
    output = frame.copy()

    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"ROI {candidate.candidate_id} "
            f"({candidate.occupancy:.2f})"
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return output
