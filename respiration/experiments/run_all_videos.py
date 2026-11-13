
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import cv2


def get_video_fps(video_path: Path) -> float:
    """Read FPS from the input video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS: {video_path}")

    return float(fps)


def run_stage(
    name: str,
    command: list[str],
    log_path: Path,
    force: bool = False,
) -> str:
    """Run one pipeline stage and save its stdout/stderr."""
    logging.info("[%s] Starting", name)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        logging.error(
            "[%s] Failed with return code %s. Log: %s",
            name,
            result.returncode,
            log_path,
        )
        return "FAILED"

    logging.info("[%s] Completed", name)
    return "SUCCESS"


def get_result(video_path: Path, result_path: Path) -> dict:
    """Load the final measurement result."""
    if not result_path.exists():
        return {
            "status": "measurement_failed",
            "respiratory_rate": None,
            "reason_code": "RESULT_FILE_NOT_FOUND",
        }

    try:
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logging.exception("Cannot read result for %s: %s", video_path.name, exc)
        return {
            "status": "measurement_failed",
            "respiratory_rate": None,
            "reason_code": "RESULT_FILE_INVALID",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete respiratory-rate pipeline for all videos."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "yolo11s-seg.pt",
    )
    parser.add_argument(
        "--seg-device",
        default="0",
        help="YOLO inference device, e.g. 0 or cpu",
    )
    parser.add_argument(
        "--flow-device",
        default="cpu",
        help="Optical Flow runner device metadata",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if output files already exist.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    model_path = args.model.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    # Search all CAT folders recursively.
    videos = sorted(
        path for path in data_dir.rglob("*.mp4")
        if path.parent.name.startswith("CAT")
    )   

    if not videos:
        raise RuntimeError(f"No MP4 videos found under: {data_dir}")

    logging.info("Discovered %d videos", len(videos))

    output_root = PROJECT_ROOT / "outputs"
    log_root = output_root / "batch_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    batch_results = []

    for index, video_path in enumerate(videos, start=1):
        video_path = video_path.resolve()
        stem = video_path.stem

        logging.info("=" * 70)
        logging.info(
            "[%d/%d] Processing %s",
            index,
            len(videos),
            video_path.name,
        )

        # Per-video output directories.
        seg_dir = output_root / "segmentation" / stem
        flow_dir = output_root / "optical_flow" / stem
        signal_dir = output_root / "signal_processing" / stem
        analysis_dir = output_root / "respiratory_analysis" / stem

        for directory in (seg_dir, flow_dir, signal_dir, analysis_dir):
            directory.mkdir(parents=True, exist_ok=True)

        video_log_dir = log_root / stem
        video_log_dir.mkdir(parents=True, exist_ok=True)

        stage_status = {
            "segmentation": "NOT_STARTED",
            "optical_flow": "NOT_STARTED",
            "signal_processing": "NOT_STARTED",
            "respiratory_analysis": "NOT_STARTED",
        }
        
        fps = ""

        try:
            fps = get_video_fps(video_path)

            # -------------------------------------------------
            # 1. Video Segmentation
            # -------------------------------------------------
            segmentation_ready = (
                (seg_dir / "roi_sequence.npz").exists()
                and (seg_dir / "mask_sequence.bin").exists()
                and (seg_dir / "mask_metadata.json").exists()
            )

            if segmentation_ready and not args.force:
                stage_status["segmentation"] = "SKIPPED"
                logging.info("[Segmentation] Existing outputs found.")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "src.vision.video_segmentation_runner",
                    "--video",
                    str(video_path),
                    "--model",
                    str(model_path),
                    "--device",
                    args.seg_device,
                    "--output",
                    str(seg_dir),
                ]

                stage_status["segmentation"] = run_stage(
                    "Segmentation",
                    command,
                    video_log_dir / "segmentation.log",
                )

            if stage_status["segmentation"] == "FAILED":
                raise RuntimeError("Segmentation failed")

            # -------------------------------------------------
            # 2. Optical Flow + ROI Tracking
            # -------------------------------------------------
            flow_ready = (
                (flow_dir / "roi_tracking.csv").exists()
                and (flow_dir / "roi_motion_signals.csv").exists()
            )

            if flow_ready and not args.force:
                stage_status["optical_flow"] = "SKIPPED"
                logging.info("[Optical Flow] Existing outputs found.")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "src.vision.optical_flow_runner",
                    "--video",
                    str(video_path),
                    "--segmentation-dir",
                    str(seg_dir),
                    "--output-dir",
                    str(flow_dir),
                    "--device",
                    args.flow_device,
                ]

                stage_status["optical_flow"] = run_stage(
                    "Optical Flow",
                    command,
                    video_log_dir / "optical_flow.log",
                )

            if stage_status["optical_flow"] == "FAILED":
                raise RuntimeError("Optical Flow failed")

            # -------------------------------------------------
            # 3. Signal Processing
            # -------------------------------------------------
            processed_signal_path = signal_dir / "processed_signals.csv"

            if processed_signal_path.exists() and not args.force:
                stage_status["signal_processing"] = "SKIPPED"
                logging.info("[Signal Processing] Existing output found.")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "src.signal.signal_processor",
                    "--input",
                    str(flow_dir / "roi_motion_signals.csv"),
                    "--output-dir",
                    str(signal_dir),
                    "--fps",
                    str(fps),
                ]

                stage_status["signal_processing"] = run_stage(
                    "Signal Processing",
                    command,
                    video_log_dir / "signal_processing.log",
                )

            if stage_status["signal_processing"] == "FAILED":
                raise RuntimeError("Signal Processing failed")

            if not processed_signal_path.exists():
                raise RuntimeError("processed_signals.csv not found")

            # -------------------------------------------------
            # 4. Respiratory Analysis
            # -------------------------------------------------
            measurement_path = analysis_dir / "measurement_result.json"

            if measurement_path.exists() and not args.force:
                stage_status["respiratory_analysis"] = "SKIPPED"
                logging.info("[Analyzer] Existing result found.")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "src.signal.respiratory_analyzer",
                    "--input",
                    str(processed_signal_path),
                    "--tracking",
                    str(flow_dir / "roi_tracking.csv"),
                    "--output-dir",
                    str(analysis_dir),
                    "--fps",
                    str(fps),
                ]

                stage_status["respiratory_analysis"] = run_stage(
                    "Respiratory Analyzer",
                    command,
                    video_log_dir / "respiratory_analyzer.log",
                )

            if stage_status["respiratory_analysis"] == "FAILED":
                raise RuntimeError("Respiratory Analyzer failed")

            result = get_result(video_path, measurement_path)

        except Exception as exc:
            logging.exception("Pipeline failed for %s", video_path.name)

            result = {
                "status": "pipeline_failed",
                "respiratory_rate": None,
                "reason_code": str(exc),
            }

        # -------------------------------------------------
        # 5. Collect per-video results
        # -------------------------------------------------
        video_id_match = re.search(r"_(\d+)$", stem)

        batch_results.append({
            "pet_id": video_path.parent.name,
            "video_id": (
                video_id_match.group(1)
                if video_id_match
                else ""
            ),
            "video_name": video_path.name,
            "fps": locals().get("fps", ""),
            "status": result.get("status"),
            "respiratory_rate": result.get("respiratory_rate"),
            "reason_code": result.get("reason_code"),
            "segmentation": stage_status["segmentation"],
            "optical_flow": stage_status["optical_flow"],
            "signal_processing": stage_status["signal_processing"],
            "respiratory_analysis": stage_status["respiratory_analysis"],
        })

        logging.info(
            "Result: status=%s, BPM=%s, reason=%s",
            result.get("status"),
            result.get("respiratory_rate"),
            result.get("reason_code"),
        )

    # -----------------------------------------------------
    # 6. Save batch summary
    # -----------------------------------------------------
    summary_path = output_root / "batch_analysis_summary.csv"

    fieldnames = [
        "pet_id",
        "video_id",
        "video_name",
        "fps",
        "status",
        "respiratory_rate",
        "reason_code",
        "segmentation",
        "optical_flow",
        "signal_processing",
        "respiratory_analysis",
    ]

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_results)

    success_count = sum(
        row["status"] == "success"
        for row in batch_results
    )
    failed_count = len(batch_results) - success_count

    logging.info("=" * 70)
    logging.info("Batch processing completed.")
    logging.info("Total videos: %d", len(batch_results))
    logging.info("Successful measurements: %d", success_count)
    logging.info("Failed / unmeasurable: %d", failed_count)
    logging.info("Summary: %s", summary_path)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    main()
