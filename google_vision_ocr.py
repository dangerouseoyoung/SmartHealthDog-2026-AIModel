"""Standalone Google Cloud Vision OCR; no Celery/Backend imports or side effects.

Public contract: extract_text(image_bytes) -> {success, rawText, error}.
Never log or return provider exception messages (may contain sensitive details).
"""
from __future__ import annotations

import io
import os
from typing import Any

from google.api_core import exceptions as google_errors
from google.cloud import vision
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # project-side conservative cap; configurable
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"}
ERROR_CODES = {
    "INVALID_IMAGE", "UNSUPPORTED_FORMAT", "IMAGE_TOO_LARGE",
    "NO_TEXT_DETECTED", "OCR_AUTH_ERROR", "OCR_TIMEOUT",
    "OCR_RATE_LIMIT", "OCR_SERVICE_ERROR",
}


def _result(text: str | None = None, error: str | None = None) -> dict[str, Any]:
    return {"success": error is None, "rawText": text if error is None else None, "error": error}


def validate_image(image_bytes: bytes) -> str | None:
    """Return error code or None. Reject invalid, truncated and unsupported files."""
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
        # A PDF is a document, but not an image supported by this endpoint.
        if image_bytes.startswith(b"%PDF-"):
            return "UNSUPPORTED_FORMAT"
        return "INVALID_IMAGE"
    except (OSError, ValueError, SyntaxError):
        return "INVALID_IMAGE"
    return None


def extract_text(image_bytes: bytes, *, client: Any = None, timeout: int = 60) -> dict[str, Any]:
    """OCR a validated image. Inject client for offline unit testing.

    Does not download from S3, enqueue Celery tasks, or call Backend APIs.
    """
    error = validate_image(image_bytes)
    if error:
        return _result(error=error)
    try:
        ocr_client = client if client is not None else vision.ImageAnnotatorClient()
        response = ocr_client.document_text_detection(
            image=vision.Image(content=image_bytes),
            image_context=vision.ImageContext(language_hints=["ko", "en"]),
            timeout=timeout,
        )
        if response.error.message:
            # Google Vision errors can be surfaced as response.error rather than exception.
            code = getattr(response.error, "code", 0)
            if code in (7, 16):
                return _result(error="OCR_AUTH_ERROR")
            if code == 4:
                return _result(error="OCR_TIMEOUT")
            if code == 8:
                return _result(error="OCR_RATE_LIMIT")
            return _result(error="OCR_SERVICE_ERROR")
        text = (response.full_text_annotation.text or "").strip()
        return _result(text=text) if text else _result(error="NO_TEXT_DETECTED")
    except (google_errors.Unauthenticated, google_errors.PermissionDenied):
        return _result(error="OCR_AUTH_ERROR")
    except (google_errors.DeadlineExceeded, TimeoutError):
        return _result(error="OCR_TIMEOUT")
    except (google_errors.ResourceExhausted, google_errors.TooManyRequests):
        return _result(error="OCR_RATE_LIMIT")
    except Exception:
        return _result(error="OCR_SERVICE_ERROR")


def extract_document(image_bytes: bytes, *, client: Any = None, timeout: int = 60) -> dict[str, Any]:
    """Return OCR text and position-aware paragraph blocks."""
    error = validate_image(image_bytes)
    if error:
        return {"success": False, "rawText": None, "blocks": [], "error": error}

    try:
        ocr_client = client if client is not None else vision.ImageAnnotatorClient()
        response = ocr_client.document_text_detection(
            image=vision.Image(content=image_bytes),
            image_context=vision.ImageContext(language_hints=["ko", "en"]),
            timeout=timeout,
        )

        if response.error.message:
            code = getattr(response.error, "code", 0)
            error = {
                7: "OCR_AUTH_ERROR",
                16: "OCR_AUTH_ERROR",
                4: "OCR_TIMEOUT",
                8: "OCR_RATE_LIMIT",
            }.get(code, "OCR_SERVICE_ERROR")
            return {"success": False, "rawText": None, "blocks": [], "error": error}

        annotation = response.full_text_annotation
        raw_text = (annotation.text or "").strip()

        if not raw_text:
            return {"success": False, "rawText": None, "blocks": [], "error": "NO_TEXT_DETECTED"}

        blocks = []

        for page in annotation.pages:
            for vision_block in page.blocks:
                for paragraph in vision_block.paragraphs:
                    words = []
                    confidences = []

                    for word in paragraph.words:
                        word_text = "".join(
                            symbol.text for symbol in word.symbols
                        )
                        words.append(word_text)
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
                        "confidence": (
                            sum(confidences) / len(confidences)
                            if confidences else 0.0
                        ),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2,
                        "bbox": [[v.x, v.y] for v in vertices],
                    })

        blocks.sort(key=lambda b: (b["center_y"], b["center_x"]))

        return {
            "success": True,
            "rawText": raw_text,
            "blocks": blocks,
            "error": None,
        }

    except (google_errors.Unauthenticated, google_errors.PermissionDenied):
        error = "OCR_AUTH_ERROR"
    except (google_errors.DeadlineExceeded, TimeoutError):
        error = "OCR_TIMEOUT"
    except (google_errors.ResourceExhausted, google_errors.TooManyRequests):
        error = "OCR_RATE_LIMIT"
    except Exception:
        error = "OCR_SERVICE_ERROR"

    return {"success": False, "rawText": None, "blocks": [], "error": error}
