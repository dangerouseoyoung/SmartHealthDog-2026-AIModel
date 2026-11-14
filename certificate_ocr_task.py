
import logging
import os

import requests
from celery_app import app

from google_vision_ocr import extract_document
from ocr.health_check_parser import HealthCheckParser

from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


def get_ocr_auth_token() -> str | None:
    """백엔드 로그인 후 JWT 토큰 발급"""

    login_url = os.getenv("AI_MODEL_SERVICE_LOGIN_ENDPOINT")
    email = os.getenv("AI_MODEL_SERVICE_EMAIL")
    password = os.getenv("AI_MODEL_SERVICE_PASSWORD")

    if not all([login_url, email, password]):
        logger.error("OCR backend authentication configuration missing")
        return None

    try:
        response = requests.post(
            login_url,
            json={
                "email": email,
                "password": password
            },
            timeout=10
        )

        response.raise_for_status()

        token = response.json().get("accessToken")

        if not token:
            logger.error("Access token missing in login response")
            return None

        return token

    except requests.RequestException:
        logger.error("OCR backend authentication request failed")
        return None
    except (ValueError, TypeError):
        logger.error("Invalid OCR backend authentication response")
        return None

def process_certificate(image_bytes: bytes) -> dict:
    """이미지 OCR 후 진단서 항목별 데이터 추출"""

    ocr_result = extract_document(image_bytes)

    if not ocr_result["success"]:
        return {
            "success": False,
            "fields": None,
            "error": ocr_result["error"]
        }

    parser = HealthCheckParser()
    fields = parser.parse(ocr_result)

    return {
        "success": True,
        "fields": fields,
        "error": None
    }


def send_certificate_result(submission_id, token, result):
    """OCR 결과를 Backend에 PATCH 전송"""

    endpoint = os.getenv(
        "AI_MODEL_SERVICE_CERTIFICATE_RESULT_ENDPOINT"
    )

    if not endpoint:
        raise RuntimeError(
            "OCR result endpoint is not configured"
        )

    url = endpoint.format(id=submission_id)

    response = requests.patch(
        url,
        json=result,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        timeout=30
    )

    response.raise_for_status()
    return response.status_code


def send_ocr_failure_status(
    submission_id: str,
    token: str,
    failure_reason: str
):
    """OCR 실패 상태를 Backend에 전송"""

    endpoint = os.getenv(
        "AI_MODEL_SERVICE_CERTIFICATE_STATUS_ENDPOINT"
    )

    if not endpoint:
        raise RuntimeError(
            "Certificate status endpoint is not configured"
        )

    url = endpoint.format(id=submission_id)

    payload = {
        "status": "FAILED",
        "failureReason": failure_reason
    }

    response = requests.patch(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        timeout=30
    )

    response.raise_for_status()
    return response.status_code


@app.task(name="predict_certificate_ocr")
def predict_certificate_ocr(image_url: str, submission_id: str):
    """Celery OCR 작업 및 실패 상태 처리"""

    token = get_ocr_auth_token()

    if not token:
        logger.error("OCR backend authentication failed")
        raise RuntimeError("Backend authentication failed")

    try:
        # 1. 이미지 다운로드
        response = requests.get(
            image_url,
            timeout=(10, 30)
        )
        response.raise_for_status()

        # 2. OCR 및 필드 추출
        result = process_certificate(response.content)

        # 3. OCR 실패 확인
        if not result["success"]:
            error_code = result["error"]

            if error_code == "OCR_TIMEOUT":
                reason = "TIMEOUT"
            elif error_code in (
                "INVALID_IMAGE",
                "UNSUPPORTED_FORMAT",
                "IMAGE_TOO_LARGE"
            ):
                reason = "INVALID_INPUT"
            else:
                reason = "INFERENCE_ERROR"

            send_ocr_failure_status(
                submission_id,
                token,
                reason
            )

            return {
                "status": "FAILED",
                "error": error_code
            }

        # 4. 정상 결과 전송
        send_certificate_result(
            submission_id=submission_id,
            token=token,
            result=result
        )

        return result

    except requests.exceptions.Timeout:
        reason = "TIMEOUT"

    except requests.exceptions.RequestException:
        reason = "INFERENCE_ERROR"

    except Exception:
        reason = "INFERENCE_ERROR"

    # 5. 예외 발생 시 실패 상태 전달
    logger.error(
        "Certificate OCR failed: submission_id=%s, reason=%s",
        submission_id,
        reason
    )

    send_ocr_failure_status(
        submission_id,
        token,
        reason
    )

    return {
        "status": "FAILED",
        "error": reason
    }

