# ocr/certificate_ocr_task.py
"""진단서 OCR Celery 태스크.

Backend(CeleryTaskSender) → Redis `ocr` 큐 → predict_certificate_ocr(image_url, submission_id)
→ Google Vision DOCUMENT_TEXT_DETECTION → 17개 필드 추출
→ PATCH /api/submissions/{id}/certificate/result

- 콜백 본문: success, rawText, confidence, fields(17개 camelCase), error
- rawText / 블록 좌표는 민감정보이므로 로그로 남기지 않고 Celery 결과 백엔드에도 저장하지 않는다.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from celery_app import app
from ocr.google_vision_ocr import extract_document
from ocr.health_check_parser import HealthCheckParser

# --------------------------------------------------
# Environment
# --------------------------------------------------

OCR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OCR_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

OCR_QUEUE_NAME = "ocr"
OCR_TASK_NAME = "predict_certificate_ocr"

# 서비스 계정 키: 환경변수가 없으면 ocr/ 에 둔 키 파일을 사용한다 (Git 커밋 금지).
DEFAULT_GOOGLE_CREDENTIALS = OCR_ROOT / "handy-buffer-514715-n6-7c18fba5679c.json"
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and DEFAULT_GOOGLE_CREDENTIALS.is_file():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(DEFAULT_GOOGLE_CREDENTIALS)

BACKEND_USERNAME = os.getenv("AI_MODEL_SERVICE_EMAIL")
BACKEND_PASSWORD = os.getenv("AI_MODEL_SERVICE_PASSWORD")
BACKEND_LOGIN_ENDPOINT = os.getenv("AI_MODEL_SERVICE_LOGIN_ENDPOINT")
BACKEND_CERTIFICATE_RESULT_ENDPOINT = os.getenv("AI_MODEL_SERVICE_CERTIFICATE_RESULT_ENDPOINT")

BACKEND_REQUEST_TIMEOUT = (
    int(os.getenv("BACKEND_CONNECT_TIMEOUT", "10")),
    int(os.getenv("BACKEND_READ_TIMEOUT", "30")),
)
IMAGE_DOWNLOAD_TIMEOUT = (10, int(os.getenv("OCR_IMAGE_DOWNLOAD_TIMEOUT", "60")))

# Python parser key -> Java CertificateOcrFieldsDto camelCase field (17개).
FIELD_TO_BACKEND = {
    "species": "speciesText",
    "breed": "breed",
    "animal_name": "animalName",
    "registration_number": "registrationNumber",
    "gender": "genderText",
    "age": "ageText",
    "coat_color": "coatColor",
    "features": "features",
    "disease_name": "diseaseName",
    "onset_date": "onsetDateText",
    "diagnosis_date": "diagnosedDateText",
    "symptoms": "symptoms",
    "treatment": "treatment",
    "hospitalization_discharge_date": "hospitalizationDischargeDateText",
    "prognosis": "prognosis",
    "notes": "remarks",
    "document_date": "documentDateText",
}


# --------------------------------------------------
# Backend
# --------------------------------------------------

def get_auth_token() -> str | None:
    """AI_MODEL_SERVICE 계정으로 로그인해 JWT accessToken 을 받는다."""
    if not all([BACKEND_LOGIN_ENDPOINT, BACKEND_USERNAME, BACKEND_PASSWORD]):
        logger.error("OCR backend authentication configuration missing")
        return None
    try:
        response = requests.post(
            BACKEND_LOGIN_ENDPOINT,
            json={"email": BACKEND_USERNAME, "password": BACKEND_PASSWORD},
            timeout=BACKEND_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        token = response.json().get("accessToken")
    except requests.RequestException:
        logger.error("OCR backend authentication request failed")
        return None
    except (ValueError, TypeError, AttributeError):
        logger.error("Invalid OCR backend authentication response")
        return None

    if not token:
        logger.error("Access token missing in login response")
        return None
    return token


def send_certificate_result(submission_id: str, token: str, result: dict) -> int:
    """OCR 결과를 Backend 에 PATCH 전송. 2xx 여도 내부적으로 FAILED 처리될 수 있다."""
    if not BACKEND_CERTIFICATE_RESULT_ENDPOINT:
        raise RuntimeError("AI_MODEL_SERVICE_CERTIFICATE_RESULT_ENDPOINT is not configured")

    response = requests.patch(
        BACKEND_CERTIFICATE_RESULT_ENDPOINT.format(id=submission_id),
        json=result,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=BACKEND_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.status_code


# --------------------------------------------------
# OCR
# --------------------------------------------------

def failure_result(error: str) -> dict:
    return {"success": False, "fields": None, "error": error}


def _average_confidence(blocks: list[dict]) -> float:
    """Vision 문단 신뢰도 평균 (0..1). 값을 임의로 높이지 않는다."""
    values = []
    for block in blocks:
        value = block.get("confidence")
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0:
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else 0.0


def process_certificate(image_bytes: bytes) -> dict:
    """이미지 OCR 후 진단서 항목별 데이터 추출 → 백엔드 콜백 본문."""
    ocr_result = extract_document(image_bytes)
    if not ocr_result["success"]:
        return failure_result(ocr_result["error"])

    parsed = HealthCheckParser().parse(ocr_result)
    fields = {java_key: parsed.get(py_key) for py_key, java_key in FIELD_TO_BACKEND.items()}

    return {
        "success": True,
        "rawText": ocr_result["rawText"],
        "confidence": _average_confidence(ocr_result.get("blocks", [])),
        "fields": fields,
        "error": None,
    }


def download_image(image_url: str) -> bytes:
    response = requests.get(image_url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    return response.content


def execute_certificate_ocr(image_url: str, submission_id: str) -> dict:
    # 1. 백엔드 JWT 인증 (실패 시 콜백 불가)
    token = get_auth_token()
    if not token:
        return {"status": "failed", "error": "AUTHENTICATION_FAILED"}

    # 2. 이미지 다운로드 → 3. OCR 및 필드 추출
    try:
        image_bytes = download_image(image_url)
    except requests.Timeout:
        logger.warning("Certificate image download timeout: id=%s", submission_id)
        result = failure_result("OCR_TIMEOUT")  # 백엔드에서 재시도 대상
    except requests.RequestException:
        logger.warning("Certificate image download failed: id=%s", submission_id)
        result = failure_result("INVALID_IMAGE")
    else:
        try:
            result = process_certificate(image_bytes)
        except Exception:
            logger.exception("Certificate OCR parsing failed: id=%s", submission_id)
            result = failure_result("OCR_SERVICE_ERROR")

    # 4. 백엔드로 결과 전달
    try:
        status_code = send_certificate_result(submission_id, token, result)
    except (requests.RequestException, RuntimeError) as exc:
        logger.error("Certificate OCR callback failed: id=%s (%s)", submission_id, type(exc).__name__)
        return {"status": "failed", "error": "CALLBACK_FAILED"}

    summary = {
        "status": "sent",
        "httpStatus": status_code,
        "success": result["success"],
        "error": result.get("error"),
        "confidence": result.get("confidence"),
        "filledFields": sum(v is not None for v in (result.get("fields") or {}).values()),
    }
    logger.info("Certificate OCR finished: id=%s %s", submission_id, summary)
    return summary


# --------------------------------------------------
# Celery Task
# --------------------------------------------------

@app.task(name=OCR_TASK_NAME, queue=OCR_QUEUE_NAME, max_retries=0)
def predict_certificate_ocr(image_url: str, submission_id: str):
    logger.info("Certificate OCR task started: id=%s", submission_id)
    return execute_certificate_ocr(image_url, submission_id)
