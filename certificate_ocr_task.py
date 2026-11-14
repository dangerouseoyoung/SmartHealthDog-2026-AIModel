
import logging
import os

import requests
from celery_app import app

from google_vision_ocr import extract_document
from ocr.health_check_parser import HealthCheckParser


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


@app.task(name="predict_certificate_ocr")
def predict_certificate_ocr(image_url: str, submission_id: str):
    """Celery OCR 작업"""

    # 1. 백엔드 JWT 인증
    token = get_ocr_auth_token()

    if not token:
        raise RuntimeError("Backend authentication failed")

    # 2. 이미지 다운로드
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    # 3. OCR 실행 및 필드별 데이터 추출
    result = process_certificate(response.content)

    # 4. 백엔드로 결과 전달
    send_certificate_result(
        submission_id=submission_id,
        token=token,
        result=result
    )

    return result

