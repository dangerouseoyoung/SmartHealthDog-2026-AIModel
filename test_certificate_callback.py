
import os
from unittest.mock import patch, Mock

from certificate_ocr_task import send_certificate_result


def test_certificate_callback():
    submission_id = "123"

    result = {
        "success": True,
        "fields": {
            "species": "고양이",
            "breed": "korean short hair",
            "diagnosis_date": "2025-10-31",
            "document_date": "2026-09-28"
        },
        "error": None
    }

    response = Mock()
    response.status_code = 200

    with patch.dict(os.environ, {
        "AI_MODEL_SERVICE_CERTIFICATE_RESULT_ENDPOINT":
            "https://example.invalid/submissions/{id}/certificate/result"
    }):
        with patch("certificate_ocr_task.requests.patch") as mock_patch:
            mock_patch.return_value = response

            status_code = send_certificate_result(
                submission_id,
                "test-token",
                result
            )

    assert status_code == 200

    mock_patch.assert_called_once()

    args, kwargs = mock_patch.call_args

    assert args[0] == (
        "https://example.invalid/submissions/123/certificate/result"
    )
    assert kwargs["json"] == result
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    response.raise_for_status.assert_called_once()

    print("Backend PATCH Mock 테스트 통과!")


if __name__ == "__main__":
    test_certificate_callback()
