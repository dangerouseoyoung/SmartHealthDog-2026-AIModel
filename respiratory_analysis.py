# respiratory_analysis.py

import logging
import os
import tempfile
import time

from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from celery_app import app

from respiratory_pipeline import (
    run_respiratory_pipeline,
    RespiratoryMeasurementError,
    RespiratoryPipelineError,
)


# --------------------------------------------------
# Environment
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


BACKEND_USERNAME = os.getenv("AI_MODEL_SERVICE_EMAIL")
BACKEND_PASSWORD = os.getenv("AI_MODEL_SERVICE_PASSWORD")

BACKEND_LOGIN_ENDPOINT = os.getenv(
    "AI_MODEL_SERVICE_LOGIN_ENDPOINT"
)

BACKEND_RESPIRATION_ENDPOINT = os.getenv(
    "AI_MODEL_SERVICE_RESPIRATION_UPDATE_ENDPOINT"
)

BACKEND_STATUS_ENDPOINT = os.getenv(
    "AI_MODEL_SERVICE_RESPIRATION_STATUS_UPDATE_ENDPOINT"
)


RESPIRATION_MODEL_PATH = Path(
    os.getenv(
        "RESPIRATION_MODEL_PATH",
        str(
            PROJECT_ROOT
            / "respiration"
            / "weights"
            / "yolo11s-seg.pt"
        ),
    )
).resolve()

RESPIRATION_DEVICE = os.getenv(
    "RESPIRATION_DEVICE",
    "0",
)

RESPIRATION_VIDEO_TIMEOUT_CONNECT = int(
    os.getenv("RESPIRATION_VIDEO_TIMEOUT_CONNECT", "10")
)

RESPIRATION_VIDEO_TIMEOUT = int(
    os.getenv("RESPIRATION_VIDEO_TIMEOUT", "180")
)

RESPIRATION_MAX_VIDEO_SIZE_MB = int(
    os.getenv("RESPIRATION_MAX_VIDEO_SIZE_MB", "100")
)

RESPIRATION_STAGE_TIMEOUT = int(
    os.getenv("RESPIRATION_STAGE_TIMEOUT", "900")
)

BACKEND_REQUEST_TIMEOUT = (
    int(os.getenv("BACKEND_CONNECT_TIMEOUT", "10")),
    int(os.getenv("BACKEND_READ_TIMEOUT", "30")),
)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")

# --------------------------------------------------
# Environment Validation
# --------------------------------------------------

def validate_environment() -> None:

    required = {
        "AI_MODEL_SERVICE_EMAIL": BACKEND_USERNAME,
        "AI_MODEL_SERVICE_PASSWORD": BACKEND_PASSWORD,
        "AI_MODEL_SERVICE_LOGIN_ENDPOINT": BACKEND_LOGIN_ENDPOINT,
        "AI_MODEL_SERVICE_RESPIRATION_UPDATE_ENDPOINT":
            BACKEND_RESPIRATION_ENDPOINT,
        "AI_MODEL_SERVICE_STATUS_UPDATE_ENDPOINT":
            BACKEND_STATUS_ENDPOINT,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    if not RESPIRATION_MODEL_PATH.is_file():
        raise RuntimeError(
            f"YOLO model not found: {RESPIRATION_MODEL_PATH}"
        )


# --------------------------------------------------
# Backend Authentication
# --------------------------------------------------

def get_auth_token() -> str:

    response = requests.post(
        BACKEND_LOGIN_ENDPOINT,
        json={
            "email": BACKEND_USERNAME,
            "password": BACKEND_PASSWORD,
        },
        timeout=BACKEND_REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    token = response.json().get("accessToken")

    if not token:
        raise RuntimeError(
            "Backend login response does not contain accessToken"
        )

    return token


def get_auth_headers(token: str) -> dict:

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# --------------------------------------------------
# Video Download
# --------------------------------------------------

def download_video(
    video_url: str,
    destination: Path,
) -> Path:
    """S3 영상을 스트리밍 방식으로 다운로드한다."""

    parsed_url = urlparse(video_url)

    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("Invalid video URL")

    max_bytes = RESPIRATION_MAX_VIDEO_SIZE_MB * 1024 * 1024
    max_retries = 3
    chunk_size = 1024 * 1024

    for attempt in range(max_retries):
        downloaded_bytes = 0

        try:
            with requests.get(
                video_url,
                stream=True,
                timeout=(
                    RESPIRATION_VIDEO_TIMEOUT_CONNECT,
                    RESPIRATION_VIDEO_TIMEOUT,
                ),
            ) as response:

                response.raise_for_status()

                content_length = response.headers.get("Content-Length")

                if content_length:
                    if int(content_length) > max_bytes:
                        raise ValueError(
                            "Video exceeds maximum file size"
                        )

                with destination.open("wb") as file:
                    for chunk in response.iter_content(
                        chunk_size=chunk_size
                    ):
                        if not chunk:
                            continue

                        downloaded_bytes += len(chunk)

                        if downloaded_bytes > max_bytes:
                            raise ValueError(
                                "Video exceeds maximum file size"
                            )

                        file.write(chunk)

            if downloaded_bytes == 0:
                raise ValueError("Downloaded video is empty")

            logger.info(
                "Video downloaded: %.2f MB",
                downloaded_bytes / (1024 * 1024),
            )

            return destination

        except requests.RequestException:
            if destination.exists():
                destination.unlink()

            logger.warning(
                "Video download failed: attempt=%d/%d",
                attempt + 1,
                max_retries,
                exc_info=True,
            )

            if attempt == max_retries - 1:
                raise

            time.sleep(2 ** attempt)

        except Exception:
            if destination.exists():
                destination.unlink()
            raise

    raise RuntimeError("Video download failed")

# --------------------------------------------------
# Backend API
# --------------------------------------------------

def submit_respiratory_result(
    submission_id: str,
    respiratory_rate: int,
    token: str,
) -> None:

    endpoint = BACKEND_RESPIRATION_ENDPOINT.format(
        id=submission_id
    )

    response = requests.patch(
        endpoint,
        json={
            "respiratoryRate": respiratory_rate,
        },
        headers=get_auth_headers(token),
        timeout=BACKEND_REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    logger.info(
        "Respiratory result submitted: id=%s, rate=%d",
        submission_id,
        respiratory_rate,
    )


def submit_status_update(
    submission_id: str,
    status: str,
    token: str,
    failure_reason: str | None = None,
) -> None:

    endpoint = BACKEND_STATUS_ENDPOINT.format(
        id=submission_id
    )

    payload = {
        "status": status,
    }

    if failure_reason:
        payload["failureReason"] = failure_reason

    response = requests.patch(
        endpoint,
        json=payload,
        headers=get_auth_headers(token),
        timeout=BACKEND_REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    logger.info(
        "Status updated: id=%s, status=%s",
        submission_id,
        status,
    )


# --------------------------------------------------
# Failure Handling
# --------------------------------------------------

def map_failure_reason(exc: Exception) -> str:

    if isinstance(exc, TimeoutError):
        return "TIMEOUT"

    if isinstance(exc, ValueError):
        return "INVALID_INPUT"

    if isinstance(exc, RespiratoryMeasurementError):
        return "MEASUREMENT_FAILED"

    if isinstance(exc, RespiratoryPipelineError):
        return "INFERENCE_ERROR"

    return "INFERENCE_ERROR"


def update_failure_status(
    submission_id: str,
    token: str,
    failure_reason: str,
) -> None:

    try:
        submit_status_update(
            submission_id=submission_id,
            status="FAILED",
            token=token,
            failure_reason=failure_reason,
        )

    except requests.RequestException:
        logger.exception(
            "Failed to update FAILED status: id=%s",
            submission_id,
        )


# --------------------------------------------------
# Main Task Logic
# --------------------------------------------------

def execute_respiratory_analysis(
    video_url: str,
    submission_id: str,
) -> dict:

    token = None

    try:
        validate_environment()

        # 1. Backend 인증
        token = get_auth_token()
        
        # 2. 임시 디렉터리 생성
        temp_dir = tempfile.mkdtemp(
            prefix="respiration_"
        )

        temp_path = Path(temp_dir)
        video_path = temp_path / "input_video.mp4"

        try:
            # 3. 영상 다운로드
            download_video(
                video_url=video_url,
                destination=video_path,
            )

            # 4. AI 분석
            result = run_respiratory_pipeline(
                video_path=video_path,
                work_dir=temp_path / "analysis",
                model_path=RESPIRATION_MODEL_PATH,
                device=RESPIRATION_DEVICE,
                stage_timeout=RESPIRATION_STAGE_TIMEOUT,
            )

            respiratory_rate = result["respiratory_rate"]

            # 5. 결과 PATCH
            submit_respiratory_result(
                submission_id=submission_id,
                respiratory_rate=respiratory_rate,
                token=token,
            )

            # 6. 완료 상태 업데이트
            submit_status_update(
                submission_id=submission_id,
                status="COMPLETED",
                token=token,
            )

            logger.info(
                "Respiratory analysis completed. "
                "Debug directory will be removed: %s",
                temp_path,
            )

            # 분석 성공 시 임시 디렉터리 삭제
            shutil.rmtree(
                temp_path,
                ignore_errors=True,
            )

            return {
                "status": "success",
                "respiratory_rate": respiratory_rate,
            }

        except Exception as exc:

            # 분석 실패 시 임시 디렉터리 유지
            logger.exception(
                "Respiratory analysis failed: id=%s",
                submission_id,
            )

            logger.error(
                "Debug files preserved at: %s",
                temp_path,
            )

            # 인증 성공 이후에만 상태 업데이트 가능
            if token:
                failure_reason = map_failure_reason(exc)

                update_failure_status(
                    submission_id=submission_id,
                    token=token,
                    failure_reason=failure_reason,
                )

                return {
                    "status": "failed",
                    "reason_code": failure_reason,
                    "debug_directory": str(temp_path),
                }

            return {
                "status": "failed",
                "reason_code": "AUTHENTICATION_FAILED",
                "debug_directory": str(temp_path),
            }
            
        """
        # 2. 임시 디렉터리 생성
        with tempfile.TemporaryDirectory(
            prefix="respiration_"
        ) as temp_dir:

            temp_path = Path(temp_dir)
            video_path = temp_path / "input_video.mp4"

            # 3. 영상 다운로드
            download_video(
                video_url=video_url,
                destination=video_path,
            )

            # 4. AI 분석
            result = run_respiratory_pipeline(
                video_path=video_path,
                work_dir=temp_path / "analysis",
                model_path=RESPIRATION_MODEL_PATH,
                device=RESPIRATION_DEVICE,
                stage_timeout=RESPIRATION_STAGE_TIMEOUT,
            )

            respiratory_rate = result["respiratory_rate"]

            # 5. 결과 PATCH
            submit_respiratory_result(
                submission_id=submission_id,
                respiratory_rate=respiratory_rate,
                token=token,
            )

            # 6. 완료 상태 업데이트
            submit_status_update(
                submission_id=submission_id,
                status="COMPLETED",
                token=token,
            )

            return {
                "status": "success",
                "respiratory_rate": respiratory_rate,
            }
            """

    except Exception as exc:

        logger.exception(
            "Respiratory analysis failed: id=%s",
            submission_id,
        )

        # 인증 성공 이후에만 상태 업데이트 가능
        if token:
            failure_reason = map_failure_reason(exc)

            update_failure_status(
                submission_id=submission_id,
                token=token,
                failure_reason=failure_reason,
            )

            return {
                "status": "failed",
                "reason_code": failure_reason,
            }

        return {
            "status": "failed",
            "reason_code": "AUTHENTICATION_FAILED",
        }


# --------------------------------------------------
# Celery Task
# --------------------------------------------------

@app.task(
    name="smart_health_dog_disease_detection.predict_respiration_rate",
    bind=True,
    max_retries=0,
    queue="respiration",
)
def predict_respiratory_rate(
    self,
    video_url: str,
    submission_id: str,
):
    logger.info(
        "Respiratory task started: id=%s",
        submission_id,
    )

    return execute_respiratory_analysis(
        video_url=video_url,
        submission_id=submission_id,
    )
