
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OpticalFlowResult:
    flow: np.ndarray
    compensated_flow: np.ndarray
    global_motion_x: float
    global_motion_y: float


class FarnebackOpticalFlow:
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        self.params = {
            "pyr_scale": pyr_scale,
            "levels": levels,
            "winsize": winsize,
            "iterations": iterations,
            "poly_n": poly_n,
            "poly_sigma": poly_sigma,
            "flags": 0,
        }

    @staticmethod
    def to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def calculate(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        cat_mask: np.ndarray | None = None,
    ) -> OpticalFlowResult:
        prev_gray = self.to_gray(prev_frame)
        curr_gray = self.to_gray(curr_frame)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            **self.params,
        )

        global_x, global_y = self.estimate_global_motion(
            flow,
            cat_mask,
        )

        compensated = flow.copy()
        compensated[..., 0] -= global_x
        compensated[..., 1] -= global_y

        return OpticalFlowResult(
            flow=flow,
            compensated_flow=compensated,
            global_motion_x=global_x,
            global_motion_y=global_y,
        )

    @staticmethod
    def estimate_global_motion(
        flow: np.ndarray,
        cat_mask: np.ndarray | None,
    ) -> tuple[float, float]:
        """
        Estimate camera/global motion using background flow.

        cat_mask:
            True = cat region
            False = background
        """
        height, width = flow.shape[:2]

        if cat_mask is None:
            background = np.ones(
                (height, width),
                dtype=bool,
            )
        else:
            cat_mask = cat_mask.astype(bool)

            # Dilate cat mask to reduce boundary contamination.
            kernel = np.ones((21, 21), np.uint8)

            expanded = cv2.dilate(
                cat_mask.astype(np.uint8),
                kernel,
                iterations=1,
            )

            background = expanded == 0

        # Subsample for computational efficiency.
        sampled_flow = flow[::4, ::4]
        sampled_background = background[::4, ::4]

        vectors = sampled_flow[sampled_background]

        if len(vectors) < 100:
            return 0.0, 0.0

        median_flow = np.median(vectors, axis=0)

        return float(median_flow[0]), float(median_flow[1])


def draw_flow_visualization(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 24,
) -> np.ndarray:
    output = frame.copy()

    height, width = flow.shape[:2]

    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            dx, dy = flow[y, x]

            end_x = int(x + dx * 3)
            end_y = int(y + dy * 3)

            cv2.line(
                output,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),
                1,
            )

    return output
