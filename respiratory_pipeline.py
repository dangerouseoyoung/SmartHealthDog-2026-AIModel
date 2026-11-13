
# respiratory_pipeline.py

import json
import logging
import subprocess
import sys

from pathlib import Path


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RESPIRATION_ROOT = PROJECT_ROOT / "respiration"

SEGMENTATION_MODULE = "src.vision.video_segmentation_runner"
OPTICAL_FLOW_MODULE = "src.vision.optical_flow_runner"
SIGNAL_MODULE = "src.signal.signal_processor"
ANALYZER_MODULE = "src.signal.respiratory_analyzer"


class RespiratoryPipelineError(Exception):
    """AI 파이프라인 실행 오류."""


class RespiratoryMeasurementError(Exception):
    """신뢰할 수 있는 호흡수 측정 결과가 없는 경우."""


def _run_command(
    stage_name: str,
    command: list[str],
    log_dir: Path,
    timeout: int,
) -> None:
    """분석 단계를 실행하고 로그를 파일로 저장한다."""

    log_path = log_dir / f"{stage_name}.log"

    logger.info("Starting stage: %s", stage_name)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                cwd=str(RESPIRATION_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
                text=True,
            )

    except subprocess.TimeoutExpired as exc:
        raise RespiratoryPipelineError(
            f"{stage_name} stage timed out"
        ) from exc

    except subprocess.CalledProcessError as exc:
        raise RespiratoryPipelineError(
            f"{stage_name} stage failed. Log: {log_path}"
        ) from exc

    logger.info("Completed stage: %s", stage_name)


def _validate_result(result_path: Path) -> int:
    """최종 분석 결과를 확인하고 BPM을 반환한다."""

    if not result_path.is_file():
        raise RespiratoryPipelineError(
            f"Result file not found: {result_path}"
        )

    try:
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise RespiratoryPipelineError(
            "Cannot read measurement_result.json"
        ) from exc

    if result.get("status") != "success":
        reason_code = (
            result.get("reason_code")
            or "MEASUREMENT_FAILED"
        )
        raise RespiratoryMeasurementError(reason_code)

    respiratory_rate = result.get("respiratory_rate")

    if (
        not isinstance(respiratory_rate, int)
        or isinstance(respiratory_rate, bool)
        or respiratory_rate <= 0
    ):
        raise RespiratoryPipelineError(
            "Invalid respiratory_rate"
        )

    return respiratory_rate


def run_respiratory_pipeline(
    video_path: str | Path,
    work_dir: str | Path,
    model_path: str | Path,
    device: str = "0",
    stage_timeout: int = 900,
) -> dict:
    """
    영상에서 호흡수를 추정한다.

    Returns:
        {
            "status": "success",
            "respiratory_rate": 25,
            "reason_code": None
        }
    """

    video_path = Path(video_path).resolve()
    work_dir = Path(work_dir).resolve()
    model_path = Path(model_path).resolve()

    if not video_path.is_file():
        raise RespiratoryPipelineError(
            f"Video not found: {video_path}"
        )

    if not RESPIRATION_ROOT.is_dir():
        raise RespiratoryPipelineError(
            f"Respiration directory not found: {RESPIRATION_ROOT}"
        )

    if not model_path.is_file():
        raise RespiratoryPipelineError(
            f"YOLO model not found: {model_path}"
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    segmentation_dir = work_dir / "segmentation"
    optical_flow_dir = work_dir / "optical_flow"
    signal_dir = work_dir / "signal_processing"
    analysis_dir = work_dir / "respiratory_analysis"

    for directory in (
        segmentation_dir,
        optical_flow_dir,
        signal_dir,
        analysis_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # FPS 확인
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RespiratoryPipelineError(
            "Cannot open input video"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps <= 0:
        raise RespiratoryPipelineError(
            f"Invalid video FPS: {fps}"
        )

    # ----------------------------------------------
    # 1. YOLO Segmentation
    # ----------------------------------------------

    segmentation_command = [
        python,
        "-m",
        SEGMENTATION_MODULE,
        "--video", str(video_path),
        "--model", str(model_path),
        "--device", str(device),
        "--output", str(segmentation_dir),
    ]

    _run_command(
        "segmentation",
        segmentation_command,
        log_dir,
        stage_timeout,
    )

    # ----------------------------------------------
    # 2. Optical Flow
    # ----------------------------------------------

    optical_flow_command = [
        python,
        "-m",
        OPTICAL_FLOW_MODULE,
        "--video", str(video_path),
        "--segmentation-dir", str(segmentation_dir),
        "--output-dir", str(optical_flow_dir),
        "--device", "cpu",
    ]

    _run_command(
        "optical_flow",
        optical_flow_command,
        log_dir,
        stage_timeout,
    )

    tracking_path = optical_flow_dir / "roi_tracking.csv"
    motion_path = optical_flow_dir / "roi_motion_signals.csv"

    if not tracking_path.is_file():
        raise RespiratoryPipelineError(
            "roi_tracking.csv was not generated"
        )

    if not motion_path.is_file():
        raise RespiratoryPipelineError(
            "roi_motion_signals.csv was not generated"
        )

    # ----------------------------------------------
    # 3. Signal Processing
    # ----------------------------------------------

    signal_command = [
        python,
        "-m",
        SIGNAL_MODULE,
        "--input", str(motion_path),
        "--output-dir", str(signal_dir),
        "--fps", str(fps),
    ]

    _run_command(
        "signal_processing",
        signal_command,
        log_dir,
        stage_timeout,
    )

    processed_signal_path = signal_dir / "processed_signals.csv"

    if not processed_signal_path.is_file():
        raise RespiratoryPipelineError(
            "processed_signals.csv was not generated"
        )

    # ----------------------------------------------
    # 4. Respiratory Analysis
    # ----------------------------------------------

    analysis_command = [
        python,
        "-m",
        ANALYZER_MODULE,
        "--input", str(processed_signal_path),
        "--tracking", str(tracking_path),
        "--output-dir", str(analysis_dir),
        "--fps", str(fps),
    ]

    _run_command(
        "respiratory_analysis",
        analysis_command,
        log_dir,
        stage_timeout,
    )

    # ----------------------------------------------
    # 5. Final Result
    # ----------------------------------------------

    result_path = analysis_dir / "measurement_result.json"

    respiratory_rate = _validate_result(result_path)

    logger.info(
        "Respiratory analysis completed: %d BPM",
        respiratory_rate,
    )

    return {
        "status": "success",
        "respiratory_rate": respiratory_rate,
        "reason_code": None,
    }
