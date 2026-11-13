
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.vision.roi_tracker import track_roi_sequence
from src.vision.optical_flow import (
    FarnebackOpticalFlow,
    draw_flow_visualization,
)
from src.vision.motion_extractor import MotionExtractor


def load_roi_sequence(path: Path):
    data = np.load(path)

    required = {
        "bboxes",
        "occupancies",
        "counts",
        "fps",
    }

    missing = required - set(data.files)

    if missing:
        raise ValueError(
            f"Missing keys in ROI NPZ: {missing}"
        )

    bboxes = data["bboxes"].astype(np.int32)
    occupancies = data["occupancies"]
    counts = data["counts"]
    fps = float(np.asarray(data["fps"]).item())

    if bboxes.ndim != 3 or bboxes.shape[-1] != 4:
        raise ValueError(
            f"Invalid bboxes shape: {bboxes.shape}"
        )

    return bboxes, occupancies, counts, fps


def load_mask_frame(
    mask_file,
    frame_idx: int,
    height: int,
    width: int,
    bytes_per_frame: int,
):
    mask_file.seek(frame_idx * bytes_per_frame)

    packed = np.frombuffer(
        mask_file.read(bytes_per_frame),
        dtype=np.uint8,
    )

    if len(packed) != bytes_per_frame:
        raise ValueError(
            f"Mask data is incomplete at frame {frame_idx}"
        )

    unpacked = np.unpackbits(
        packed,
        bitorder="little",
        count=height * width,
    )

    return unpacked.reshape(height, width).astype(bool)


def draw_roi_overlay(
    frame: np.ndarray,
    bboxes: np.ndarray,
    track_ids: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    output = frame.copy()

    for roi_idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)

        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue

        track_id = int(track_ids[roi_idx])

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"ROI{roi_idx} T{track_id}"

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        output,
        f"Frame: {frame_idx}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return output


def run_optical_flow(
    video_path: Path,
    segmentation_dir: Path,
    output_dir: Path,
    device: str = "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_path = segmentation_dir / "roi_sequence.npz"
    mask_path = segmentation_dir / "mask_sequence.bin"

    if not roi_path.exists():
        raise FileNotFoundError(roi_path)

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    bboxes, occupancies, counts, roi_fps = (
        load_roi_sequence(roi_path)
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_roi_frames, num_rois, _ = bboxes.shape

    if video_frames != num_roi_frames:
        raise ValueError(
            "Video/ROI frame count mismatch: "
            f"video={video_frames}, roi={num_roi_frames}"
        )

    if abs(fps - roi_fps) > 0.1:
        print(
            f"[WARNING] FPS mismatch: "
            f"video={fps:.3f}, ROI={roi_fps:.3f}"
        )

    bytes_per_frame = (height * width + 7) // 8

    expected_mask_size = (
        num_roi_frames * bytes_per_frame
    )

    actual_mask_size = mask_path.stat().st_size

    if actual_mask_size != expected_mask_size:
        raise ValueError(
            "Mask file size mismatch: "
            f"expected={expected_mask_size}, "
            f"actual={actual_mask_size}"
        )

    # Track ROI candidates across frames.
    track_ids = track_roi_sequence(
        bboxes=bboxes,
        frame_shape=(height, width),
        max_missing=2,
    )

    tracking_rows = []

    for frame_idx in range(num_roi_frames):
        for roi_idx in range(num_rois):
            bbox = bboxes[frame_idx, roi_idx]

            valid = (
                bbox[0] >= 0
                and bbox[1] >= 0
                and bbox[2] > bbox[0]
                and bbox[3] > bbox[1]
            )

            tracking_rows.append({
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "roi_slot": roi_idx,
                "track_id": int(track_ids[frame_idx, roi_idx]),
                "valid": valid,
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
                "occupancy": float(
                    occupancies[frame_idx, roi_idx]
                ),
            })

    tracking_df = pd.DataFrame(tracking_rows)
    tracking_df.to_csv(
        output_dir / "roi_tracking.csv",
        index=False,
    )

    flow_calculator = FarnebackOpticalFlow()
    motion_extractor = MotionExtractor()

    preview_path = output_dir / "optical_flow_preview.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(preview_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create preview: {preview_path}"
        )

    motion_rows = []
    global_motion_rows = []

    previous_frame = None

    try:
        with open(mask_path, "rb") as mask_file:
            for frame_idx in range(num_roi_frames):
                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError(
                        f"Cannot read frame {frame_idx}"
                    )

                cat_mask = load_mask_frame(
                    mask_file=mask_file,
                    frame_idx=frame_idx,
                    height=height,
                    width=width,
                    bytes_per_frame=bytes_per_frame,
                )

                current_bboxes = bboxes[frame_idx]
                current_track_ids = track_ids[frame_idx]

                if previous_frame is None:
                    preview = draw_roi_overlay(
                        frame,
                        current_bboxes,
                        current_track_ids,
                        frame_idx,
                    )

                    writer.write(preview)
                    previous_frame = frame
                    continue

                flow_result = flow_calculator.calculate(
                    prev_frame=previous_frame,
                    curr_frame=frame,
                    cat_mask=cat_mask,
                )

                compensated_flow = (
                    flow_result.compensated_flow
                )

                frame_features = motion_extractor.extract_frame(
                    flow=compensated_flow,
                    bboxes=current_bboxes,
                    track_ids=current_track_ids,
                    cat_mask=cat_mask,
                )

                timestamp = frame_idx / fps

                for feature in frame_features:
                    motion_rows.append({
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp,
                        **feature,
                        "global_motion_x": (
                            flow_result.global_motion_x
                        ),
                        "global_motion_y": (
                            flow_result.global_motion_y
                        ),
                    })

                global_motion_rows.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp,
                    "global_motion_x": (
                        flow_result.global_motion_x
                    ),
                    "global_motion_y": (
                        flow_result.global_motion_y
                    ),
                })

                flow_preview = draw_flow_visualization(
                    frame,
                    compensated_flow,
                    step=32,
                )

                preview = draw_roi_overlay(
                    flow_preview,
                    current_bboxes,
                    current_track_ids,
                    frame_idx,
                )

                writer.write(preview)

                previous_frame = frame

                if frame_idx % 100 == 0:
                    print(
                        f"[Optical Flow] "
                        f"{frame_idx}/{num_roi_frames}"
                    )

    finally:
        cap.release()
        writer.release()

    motion_df = pd.DataFrame(motion_rows)
    motion_df.to_csv(
        output_dir / "roi_motion_signals.csv",
        index=False,
    )

    global_df = pd.DataFrame(global_motion_rows)
    global_df.to_csv(
        output_dir / "global_motion.csv",
        index=False,
    )

    metadata = {
        "video": str(video_path),
        "segmentation_dir": str(segmentation_dir),
        "frame_count": num_roi_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "roi_count": num_rois,
        "flow_method": "Farneback",
        "global_motion_compensation": "background_median",
        "motion_feature_count": 10,
        "processed_flow_frames": max(0, num_roi_frames - 1),
        "device": device,
    }

    with open(
        output_dir / "optical_flow_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("\n=== Optical Flow Summary ===")
    print(f"Video: {video_path.name}")
    print(f"Frames: {num_roi_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"ROI slots: {num_rois}")
    print(f"Motion rows: {len(motion_df)}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    args = parser.parse_args()

    video_stem = args.video.stem

    segmentation_dir = (
        args.segmentation_dir
        or Path("outputs/segmentation") / video_stem
    )

    output_dir = (
        args.output_dir
        or Path("outputs/optical_flow") / video_stem
    )

    run_optical_flow(
        video_path=args.video,
        segmentation_dir=segmentation_dir,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
