
from unittest.mock import patch, Mock

import requests

from certificate_ocr_task import predict_certificate_ocr


def run_failure_test(
    test_name,
    image_error=None,
    ocr_result=None,
    expected_reason=None
):
    image_response = Mock()
    image_response.content = b"fake-image-bytes"
    image_response.raise_for_status.return_value = None

    with (
        patch(
            "certificate_ocr_task.get_ocr_auth_token",
            return_value="test-token"
        ),
        patch(
            "certificate_ocr_task.requests.get"
        ) as mock_get,
        patch(
            "certificate_ocr_task.process_certificate"
        ) as mock_process,
        patch(
            "certificate_ocr_task.send_ocr_failure_status",
            return_value=200
        ) as mock_failure,
        patch(
            "certificate_ocr_task.send_certificate_result"
        ) as mock_success
    ):
        if image_error is not None:
            mock_get.side_effect = image_error
        else:
            mock_get.return_value = image_response

        if ocr_result is not None:
            mock_process.return_value = ocr_result

        result = predict_certificate_ocr.run(
            "https://example.invalid/certificate.png",
            "123"
        )

        assert result["status"] == "FAILED"
        assert result["error"] == expected_reason or (
            ocr_result is not None
            and result["error"] == ocr_result["error"]
        )

        mock_failure.assert_called_once_with(
            "123",
            "test-token",
            expected_reason
        )

        mock_success.assert_not_called()

        print(f"PASS: {test_name}")


def main():
    # 1. 이미지 다운로드 타임아웃
    run_failure_test(
        test_name="이미지 다운로드 TIMEOUT",
        image_error=requests.exceptions.Timeout(),
        expected_reason="TIMEOUT"
    )

    # 2. 손상된 이미지
    run_failure_test(
        test_name="잘못된 이미지 INVALID_INPUT",
        ocr_result={
            "success": False,
            "fields": None,
            "error": "INVALID_IMAGE"
        },
        expected_reason="INVALID_INPUT"
    )

    # 3. Google Vision 서비스 오류
    run_failure_test(
        test_name="OCR 서비스 INFERENCE_ERROR",
        ocr_result={
            "success": False,
            "fields": None,
            "error": "OCR_SERVICE_ERROR"
        },
        expected_reason="INFERENCE_ERROR"
    )

    print("\nOCR 실패 상태 전달 Mock 테스트 모두 통과!")


if __name__ == "__main__":
    main()
