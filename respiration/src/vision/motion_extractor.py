
import numpy as np


FEATURE_NAMES = [
    "mean_flow_x",
    "mean_flow_y",
    "median_flow_x",
    "median_flow_y",
    "mean_magnitude",
    "median_magnitude",
    "flow_std_x",
    "flow_std_y",
    "magnitude_std",
    "valid_pixels",
]


def extract_roi_motion(
    flow: np.ndarray,
    bbox: np.ndarray,
    cat_mask: np.ndarray | None = None,
) -> dict:
    """
    Extract motion features inside one ROI.

    Args:
        flow: (H, W, 2), compensated optical flow
        bbox: (x1, y1, x2, y2)
        cat_mask: (H, W), True = cat region

    Returns:
        Dictionary of motion features.
    """
    height, width = flow.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return _empty_features()

    roi_flow = flow[y1:y2, x1:x2]

    if roi_flow.size == 0:
        return _empty_features()

    if cat_mask is not None:
        roi_mask = cat_mask[y1:y2, x1:x2].astype(bool)
    else:
        roi_mask = np.ones(
            roi_flow.shape[:2],
            dtype=bool,
        )

    vectors = roi_flow[roi_mask]

    if len(vectors) < 10:
        return _empty_features(valid_pixels=len(vectors))

    flow_x = vectors[:, 0]
    flow_y = vectors[:, 1]

    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)

    return {
        "mean_flow_x": float(np.mean(flow_x)),
        "mean_flow_y": float(np.mean(flow_y)),
        "median_flow_x": float(np.median(flow_x)),
        "median_flow_y": float(np.median(flow_y)),
        "mean_magnitude": float(np.mean(magnitude)),
        "median_magnitude": float(np.median(magnitude)),
        "flow_std_x": float(np.std(flow_x)),
        "flow_std_y": float(np.std(flow_y)),
        "magnitude_std": float(np.std(magnitude)),
        "valid_pixels": int(len(vectors)),
    }


def _empty_features(valid_pixels: int = 0) -> dict:
    result = {
        name: np.nan
        for name in FEATURE_NAMES
    }

    result["valid_pixels"] = valid_pixels

    return result


class MotionExtractor:
    def extract_frame(
        self,
        flow: np.ndarray,
        bboxes: np.ndarray,
        track_ids: np.ndarray,
        cat_mask: np.ndarray | None = None,
    ) -> list[dict]:
        results = []

        for roi_idx, bbox in enumerate(bboxes):
            if bbox[0] < 0:
                continue

            features = extract_roi_motion(
                flow=flow,
                bbox=bbox,
                cat_mask=cat_mask,
            )

            results.append({
                "roi_slot": roi_idx,
                "track_id": int(track_ids[roi_idx]),
                **features,
            })

        return results
