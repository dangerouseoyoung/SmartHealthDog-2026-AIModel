
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.vision.mask_processor import process_frame
from src.vision.roi_extractor import (
    extract_roi_candidates,
    draw_roi_candidates,
)


def overlay_mask(frame, mask, alpha=0.35):
    """Overlay segmentation mask on the original frame."""
    if mask is None:
        return frame.copy()

    output = frame.copy()
    color = np.zeros_like(frame)
    color[:, :] = (0, 255, 0)

    binary = mask.astype(bool)

    output[binary] = (
        output[binary] * (1 - alpha)
        + color[binary] * alpha
    ).astype(np.uint8)

    return output


def process_video(
    video_path,
    model_path,
    output_dir,
    device=None,
    confidence=0.25,
    cat_class_id=15,
    grid_rows=4,
    grid_cols=4,
    min_occupancy=0.65,
    top_k=8,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_binary_path = output_dir / "mask_sequence.bin"
    mask_metadata_path = output_dir / "mask_metadata.json"
    roi_sequence_path = output_dir / "roi_sequence.npz"
    metrics_path = output_dir / "frame_metrics.csv"
    preview_path = output_dir / "segmentation_preview.mp4"

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # Load model
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

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
            f"Cannot create output video: {preview_path}"
        )

    # A binary mask uses one bit per pixel.
    packed_size = (height * width + 7) // 8

    roi_bboxes = []
    roi_occupancies = []
    roi_counts = []
    frame_metrics = []

    previous_bbox = None
    frame_index = 0

    matched_count = 0
    missing_count = 0

    try:
        with open(mask_binary_path, "wb") as mask_file:

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        "Video frame dimensions changed."
                    )

                inference_start = time.perf_counter()

                predict_kwargs = {
                    "classes": [cat_class_id],
                    "conf": confidence,
                    "retina_masks": True,
                    "verbose": False,
                }

                if device is not None:
                    predict_kwargs["device"] = device

                result = model.predict(
                    frame,
                    **predict_kwargs,
                )[0]

                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000

                # 1. Select and process cat mask
                record = process_frame(
                    result=result,
                    frame=frame,
                    frame_index=frame_index,
                    previous_bbox=previous_bbox,
                    cat_class_id=cat_class_id,
                    min_confidence=confidence,
                )

                if record.status == "MATCHED":
                    matched_count += 1
                    previous_bbox = record.bbox
                    mask = record.mask
                else:
                    missing_count += 1
                    mask = np.zeros(
                        (height, width),
                        dtype=np.uint8,
                    )

                # 2. Extract ROI candidates
                candidates = []

                if record.status == "MATCHED":
                    candidates = extract_roi_candidates(
                        frame=frame,
                        mask=mask,
                        grid_rows=grid_rows,
                        grid_cols=grid_cols,
                        min_occupancy=min_occupancy,
                        top_k=top_k,
                    )

                # 3. Store packed mask immediately.
                # This avoids keeping every full-resolution mask in RAM.
                packed_mask = np.packbits(
                    mask.reshape(-1),
                    bitorder="little",
                )

                if packed_mask.size != packed_size:
                    raise RuntimeError(
                        "Unexpected packed mask size."
                    )

                mask_file.write(packed_mask.tobytes())

                # 4. Store fixed-size ROI metadata
                frame_bboxes = np.full(
                    (top_k, 4),
                    -1,
                    dtype=np.int32,
                )

                frame_occupancies = np.full(
                    (top_k,),
                    np.nan,
                    dtype=np.float32,
                )

                for i, candidate in enumerate(
                    candidates[:top_k]
                ):
                    frame_bboxes[i] = candidate.bbox
                    frame_occupancies[i] = candidate.occupancy

                roi_bboxes.append(frame_bboxes)
                roi_occupancies.append(frame_occupancies)
                roi_counts.append(len(candidates))

                # 5. Create visualization
                visualized = overlay_mask(
                    frame,
                    mask,
                )

                visualized = draw_roi_candidates(
                    visualized,
                    candidates,
                )

                if record.bbox is not None:
                    x1, y1, x2, y2 = record.bbox

                    cv2.rectangle(
                        visualized,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2,
                    )

                status_text = (
                    f"{record.status} | "
                    f"Confidence: {record.confidence:.3f} | "
                    f"ROIs: {len(candidates)}"
                )

                cv2.putText(
                    visualized,
                    status_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(visualized)

                # 6. Record frame metrics
                mask_area = (
                    int(mask.sum())
                    if mask is not None
                    else 0
                )

                frame_metrics.append({
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "status": record.status,
                    "confidence": record.confidence,
                    "bbox": (
                        str(record.bbox)
                        if record.bbox is not None
                        else ""
                    ),
                    "mask_area": mask_area,
                    "roi_count": len(candidates),
                    "inference_ms": inference_ms,
                })

                frame_index += 1

                if frame_index % 100 == 0:
                    print(
                        f"Processed: {frame_index}/{frame_count}"
                    )

    finally:
        cap.release()
        writer.release()

    # Save frame metrics
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "status",
        "confidence",
        "bbox",
        "mask_area",
        "roi_count",
        "inference_ms",
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        csv_writer.writeheader()
        csv_writer.writerows(frame_metrics)

    # Save ROI sequence
    if roi_bboxes:
        np.savez_compressed(
            roi_sequence_path,
            bboxes=np.stack(roi_bboxes),
            occupancies=np.stack(roi_occupancies),
            counts=np.asarray(roi_counts, dtype=np.int32),
            fps=np.float32(fps),
        )
    else:
        np.savez_compressed(
            roi_sequence_path,
            bboxes=np.empty((0, top_k, 4), dtype=np.int32),
            occupancies=np.empty((0, top_k), dtype=np.float32),
            counts=np.empty((0,), dtype=np.int32),
            fps=np.float32(fps),
        )

    # Save mask sequence metadata
    metadata = {
        "video_file": video_path.name,
        "frame_count": frame_index,
        "width": width,
        "height": height,
        "fps": fps,
        "mask_format": "uint1_packed",
        "bitorder": "little",
        "packed_bytes_per_frame": packed_size,
        "mask_file": mask_binary_path.name,
    }

    with open(
        mask_metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # Summary
    matched_rate = (
        matched_count / frame_index * 100
        if frame_index > 0
        else 0
    )

    print("\n===== Segmentation Summary =====")
    print(f"Video: {video_path.name}")
    print(f"Frames processed: {frame_index}")
    print(f"FPS: {fps:.2f}")
    print(f"Resolution: {width}x{height}")
    print(f"Matched frames: {matched_count}")
    print(f"Missing frames: {missing_count}")
    print(f"Detection availability: {matched_rate:.2f}%")
    print(f"Preview: {preview_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Mask sequence: {mask_binary_path}")
    print(f"ROI sequence: {roi_sequence_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s-seg.pt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    output_dir = (
        Path(args.output)
        if args.output
        else Path("outputs/segmentation") / video_path.stem
    )

    process_video(
        video_path=video_path,
        model_path=args.model,
        output_dir=output_dir,
        device=args.device,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
