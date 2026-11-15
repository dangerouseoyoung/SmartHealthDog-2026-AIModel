"""Google Cloud Vision DOCUMENT_TEXT_DETECTION 래퍼 (Celery/Backend 의존 없음).

Public contract
    extract_text(image_bytes)     -> {success, rawText, error}
    extract_document(image_bytes) -> {success, rawText, blocks, error}

- 과금 API 호출 직전 ocr.quota_guard 로 사용량을 예약한다 (fail-closed).
- Provider 예외 메시지는 민감정보를 포함할 수 있으므로 로그/반환하지 않는다.
- 인증: GOOGLE_APPLICATION_CREDENTIALS (서비스 계정 JSON 경로).
"""
from __future__ import annotations

import io
import os
from typing import Any

from google.api_core import exceptions as google_errors
from google.cloud import vision
from PIL import Image, UnidentifiedImageError

from ocr.quota_guard import OCRQuotaExceeded, OCRQuotaUnavailable, reserve_document_text_unit

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 프로젝트 보수 상한; OCR_MAX_IMAGE_BYTES 로 조정
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"}
ERROR_CODES = {
    "INVALID_IMAGE", "UNSUPPORTED_FORMAT", "IMAGE_TOO_LARGE",
    "NO_TEXT_DETECTED", "OCR_AUTH_ERROR", "OCR_TIMEOUT",
    "OCR_RATE_LIMIT", "OCR_SERVICE_ERROR", "OCR_MONTHLY_LIMIT", "OCR_QUOTA_GUARD_ERROR",
}

# google.rpc.Code -> 에러 코드 (response.error 로 전달되는 경우)
_RPC_ERROR_CODES = {
    7: "OCR_AUTH_ERROR",     # PERMISSION_DENIED
    16: "OCR_AUTH_ERROR",    # UNAUTHENTICATED
    4: "OCR_TIMEOUT",        # DEADLINE_EXCEEDED
    8: "OCR_RATE_LIMIT",     # RESOURCE_EXHAUSTED
}


class _VisionError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def validate_image(image_bytes: bytes) -> str | None:
    """에러 코드 또는 None. 손상/미지원/과대 파일을 거부한다."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        return "INVALID_IMAGE"
    try:
        max_bytes = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(MAX_IMAGE_BYTES)))
    except ValueError:
        max_bytes = MAX_IMAGE_BYTES
    if max_bytes <= 0:
        max_bytes = MAX_IMAGE_BYTES
    if len(image_bytes) > max_bytes:
        return "IMAGE_TOO_LARGE"
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.format not in ALLOWED_FORMATS:
                return "UNSUPPORTED_FORMAT"
            img.verify()
    except UnidentifiedImageError:
        # PDF 는 이 엔드포인트가 지원하지 않는다.
        if image_bytes.startswith(b"%PDF-"):
            return "UNSUPPORTED_FORMAT"
        return "INVALID_IMAGE"
    except (OSError, ValueError, SyntaxError):
        return "INVALID_IMAGE"
    return None


def _detect(image_bytes: bytes, client: Any, timeout: int):
    """검증 → 사용량 예약 → Vision 호출. 실패 시 _VisionError(code)."""
    error = validate_image(image_bytes)
    if error:
        raise _VisionError(error)
    try:
        ocr_client = client if client is not None else vision.ImageAnnotatorClient()
        # 무료 구간 안전 카운터를 예약하지 못하면 과금 API 를 호출하지 않는다.
        reserve_document_text_unit()
        response = ocr_client.document_text_detection(
            image=vision.Image(content=image_bytes),
            image_context=vision.ImageContext(language_hints=["ko", "en"]),
            timeout=timeout,
        )
    except OCRQuotaExceeded:
        raise _VisionError("OCR_MONTHLY_LIMIT") from None
    except OCRQuotaUnavailable:
        raise _VisionError("OCR_QUOTA_GUARD_ERROR") from None
    except (google_errors.Unauthenticated, google_errors.PermissionDenied):
        raise _VisionError("OCR_AUTH_ERROR") from None
    except (google_errors.DeadlineExceeded, TimeoutError):
        raise _VisionError("OCR_TIMEOUT") from None
    except (google_errors.ResourceExhausted, google_errors.TooManyRequests):
        raise _VisionError("OCR_RATE_LIMIT") from None
    except Exception:
        raise _VisionError("OCR_SERVICE_ERROR") from None

    if response.error.message:
        code = getattr(response.error, "code", 0)
        raise _VisionError(_RPC_ERROR_CODES.get(code, "OCR_SERVICE_ERROR"))

    annotation = response.full_text_annotation
    if not (annotation.text or "").strip():
        raise _VisionError("NO_TEXT_DETECTED")
    return annotation


def extract_text(image_bytes: bytes, *, client: Any = None, timeout: int = 60) -> dict[str, Any]:
    """전체 텍스트만 반환. 오프라인 테스트 시 client 주입."""
    try:
        annotation = _detect(image_bytes, client, timeout)
    except _VisionError as exc:
        return {"success": False, "rawText": None, "error": exc.code}
    return {"success": True, "rawText": annotation.text.strip(), "error": None}


def _paragraph_blocks(annotation) -> list[dict[str, Any]]:
    blocks = []
    for page in annotation.pages:
        for vision_block in page.blocks:
            for paragraph in vision_block.paragraphs:
                words, confidences = [], []
                for word in paragraph.words:
                    words.append("".join(symbol.text for symbol in word.symbols))
                    confidences.append(word.confidence)

                text = " ".join(words).strip()
                if not text:
                    continue

                vertices = paragraph.bounding_box.vertices
                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                blocks.append({
                    "text": text,
                    "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": (x1 + x2) // 2,
                    "center_y": (y1 + y2) // 2,
                    "bbox": [[v.x, v.y] for v in vertices],
                })
    blocks.sort(key=lambda b: (b["center_y"], b["center_x"]))
    return blocks


def extract_document(image_bytes: bytes, *, client: Any = None, timeout: int = 60) -> dict[str, Any]:
    """전체 텍스트 + 좌표/신뢰도가 포함된 문단 블록 반환."""
    try:
        annotation = _detect(image_bytes, client, timeout)
        blocks = _paragraph_blocks(annotation)
    except _VisionError as exc:
        return {"success": False, "rawText": None, "blocks": [], "error": exc.code}
    except Exception:
        return {"success": False, "rawText": None, "blocks": [], "error": "OCR_SERVICE_ERROR"}
    return {"success": True, "rawText": annotation.text.strip(), "blocks": blocks, "error": None}
