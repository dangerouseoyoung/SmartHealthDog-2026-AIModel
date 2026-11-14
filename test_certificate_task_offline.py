
from unittest.mock import patch, Mock

from certificate_ocr_worker import app
from certificate_ocr_task import predict_certificate_ocr


def test_certificate_task():
    # Redis 없이 Celery 작업을 동기 실행
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True

    # 가짜 이미지 다운로드 응답
    image_response = Mock()
    image_response.content = b"fake-image-bytes"
    image_response.raise_for_status.return_value = None

    # 가짜 OCR 및 필드 추출 결과
    mock_result = {
        "success": True,
        "fields": {
            "species": "고양이",
            "breed": "korean short hair",
            "diagnosis_date": "2025-10-31",
            "document_date": "2026-09-28"
        },
        "error": None
    }

    with (
        patch(
            "certificate_ocr_task.get_ocr_auth_token",
            return_value="test-token"
        ),
        patch(
            "certificate_ocr_task.requests.get",
            return_value=image_response
        ) as mock_get,
        patch(
            "certificate_ocr_task.process_certificate",
            return_value=mock_result
        ) as mock_process,
        patch(
            "certificate_ocr_task.send_certificate_result",
            return_value=200
        ) as mock_send
    ):
        task_result = predict_certificate_ocr.apply_async(
            args=[
                "https://example.invalid/certificate.png",
                "123"
            ]
        )

        assert task_result.successful()
        assert task_result.result == mock_result

        mock_get.assert_called_once()
        mock_process.assert_called_once_with(b"fake-image-bytes")
        mock_send.assert_called_once_with(
            submission_id="123",
            token="test-token",
            result=mock_result
        )

    print("Celery OCR 전체 흐름 Mock 테스트 통과!")


if __name__ == "__main__":
    test_certificate_task()
