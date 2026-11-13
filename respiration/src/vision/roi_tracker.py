
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import linear_sum_assignment

import numpy as np


@dataclass
class ROITrack:
    track_id: int
    bbox: np.ndarray
    last_frame: int
    missed_frames: int = 0


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _center_distance(
    box_a: np.ndarray,
    box_b: np.ndarray,
    frame_diag: float,
) -> float:
    center_a = np.array([
        (box_a[0] + box_a[2]) / 2,
        (box_a[1] + box_a[3]) / 2,
    ])

    center_b = np.array([
        (box_b[0] + box_b[2]) / 2,
        (box_b[1] + box_b[3]) / 2,
    ])

    return np.linalg.norm(center_a - center_b) / max(frame_diag, 1.0)


def _area_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    return max(area_a, area_b) / min(area_a, area_b)


def _matching_cost(box_a, box_b, frame_diag):
    iou = _iou(box_a, box_b)
    distance = _center_distance(box_a, box_b, frame_diag)
    ratio = _area_ratio(box_a, box_b)

    if distance > 0.35 and iou < 0.05:
        return np.inf

    if ratio > 4.0:
        return np.inf

    normalized_distance = min(distance / 0.35, 1.0)
    normalized_area_change = min(
        abs(np.log(ratio)) / np.log(4.0),
        1.0,
    )

    return (
        0.55 * (1.0 - iou)
        + 0.30 * normalized_distance
        + 0.15 * normalized_area_change
    )


class ROITracker:
    def __init__(
        self,
        max_missing=2,
        max_cost=0.75,
    ):
        self.max_missing = max_missing
        self.max_cost = max_cost

        self.tracks = {}
        self.next_track_id = 0

    @staticmethod
    def _valid_bbox(bbox):
        x1, y1, x2, y2 = bbox

        return (
            x1 >= 0
            and y1 >= 0
            and x2 > x1
            and y2 > y1
        )

    def update(self, bboxes, frame_shape, frame_idx):
        height, width = frame_shape
        frame_diag = np.hypot(height, width)

        track_ids = np.full(
            len(bboxes), -1, dtype=np.int32
        )

        valid_indices = [
            i for i, box in enumerate(bboxes)
            if self._valid_bbox(box)
        ]

        active_tracks = list(self.tracks.values())

        matched_tracks = set()
        matched_rois = set()

        if active_tracks and valid_indices:
            cost_matrix = np.full(
                (len(active_tracks), len(valid_indices)),
                np.inf,
                dtype=np.float64,
            )

            for i, track in enumerate(active_tracks):
                for j, roi_idx in enumerate(valid_indices):
                    cost_matrix[i, j] = _matching_cost(
                        track.bbox,
                        bboxes[roi_idx],
                        frame_diag,
                    )

            # linear_sum_assignment cannot safely process
            # a matrix containing only infeasible assignments.
            safe_cost = np.where(
                np.isfinite(cost_matrix),
                cost_matrix,
                1e6,
            )

            row_indices, col_indices = linear_sum_assignment(
                safe_cost
            )

            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]

                if not np.isfinite(cost) or cost > self.max_cost:
                    continue

                track = active_tracks[row]
                roi_idx = valid_indices[col]

                track.bbox = bboxes[roi_idx].copy()
                track.last_frame = frame_idx
                track.missed_frames = 0

                track_ids[roi_idx] = track.track_id

                matched_tracks.add(track.track_id)
                matched_rois.add(roi_idx)

        for track in active_tracks:
            if track.track_id not in matched_tracks:
                track.missed_frames += 1

        # Create tracks for unmatched valid ROIs.
        for roi_idx in valid_indices:
            if roi_idx in matched_rois:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = ROITrack(
                track_id=track_id,
                bbox=bboxes[roi_idx].copy(),
                last_frame=frame_idx,
            )

            track_ids[roi_idx] = track_id

        # Remove expired tracks.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missing
        ]

        for track_id in expired:
            del self.tracks[track_id]

        return track_ids

def track_roi_sequence(
    bboxes: np.ndarray,
    frame_shape: tuple[int, int],
    max_missing: int = 2,
) -> np.ndarray:
    """
    Args:
        bboxes: (T, K, 4)

    Returns:
        track_ids: (T, K)
    """
    tracker = ROITracker(max_missing=max_missing)

    num_frames, num_rois, _ = bboxes.shape

    result = np.full(
        (num_frames, num_rois),
        -1,
        dtype=np.int32,
    )

    for frame_idx in range(num_frames):
        result[frame_idx] = tracker.update(
            bboxes[frame_idx],
            frame_shape,
            frame_idx,
        )

    return result
