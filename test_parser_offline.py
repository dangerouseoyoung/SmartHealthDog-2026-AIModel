
import json

from ocr.health_check_parser import HealthCheckParser


# Google Vision OCR이 반환하는 좌표 블록을 가상으로 생성
def make_block(text, x1, y1, x2, y2):
    return {
        "text": text,
        "confidence": 0.99,
        "bbox": [],
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": (x1 + x2) // 2,
        "center_y": (y1 + y2) // 2
    }


# 실제 진단서 일부를 기반으로 만든 가상 OCR 결과
mock_ocr_result = {
    "success": True,
    "rawText": None,
    "error": None,
    "blocks": [
        make_block("종류", 100, 100, 180, 120),
        make_block("고양이", 260, 100, 340, 120),

        make_block("품종", 100, 150, 180, 170),
        make_block("korean short hair", 260, 150, 460, 170),

        make_block("성별", 100, 200, 180, 220),
        make_block("여(중성화)", 260, 200, 380, 220),

        make_block("연령", 100, 250, 180, 270),
        make_block("9년1개월", 260, 250, 380, 270),

        make_block("병명", 100, 300, 180, 320),
        make_block("고양이 만성 구내염(FCGS)", 260, 300, 550, 320),

        make_block("진단 연월일", 100, 350, 210, 370),
        make_block("2025년 10월 31일", 260, 350, 460, 370),

        make_block("주요 증상", 100, 400, 210, 420),
        make_block("치은염, 구내염, 목구멍염 등", 260, 400, 550, 420),

        make_block("치료명칭", 100, 450, 210, 470),
        make_block("항생제, 스테로이드 등 내과치료", 260, 450, 600, 470),

        make_block("예후 소견", 100, 500, 210, 520),
        make_block("재발 가능성 높으며 추가 치료 필요", 260, 500, 620, 520),

         # 진단서 하단의 실제 발급일
        make_block("2026년 9월 28일", 260, 650, 460, 670),
    ]
}


def main():
    parser = HealthCheckParser()

    fields = parser.parse(mock_ocr_result)

    print("\n===== 필드 추출 결과 =====")
    print(json.dumps(fields, ensure_ascii=False, indent=2))

    # 주요 필드 검증
    expected = {
        "species": "고양이",
        "breed": "korean short hair",
        "gender": "여(중성화)",
        "age": "9년1개월",
        "diagnosis_date": "2025-10-31",
        "document_date": "2026-09-28",
        "prognosis": "재발 가능성 높으며 추가 치료 필요",
    }

    print("\n===== 테스트 검증 =====")

    failed = False

    for field, expected_value in expected.items():
        actual = fields.get(field)

        if actual == expected_value:
            print(f"PASS {field}: {actual}")
        else:
            print(
                f"FAIL {field}: "
                f"expected={expected_value}, actual={actual}"
            )
            failed = True

    if failed:
        raise AssertionError("일부 필드 추출에 실패했습니다.")

    print("\n기본 필드 추출 테스트 통과!")


if __name__ == "__main__":
    main()

    # 작성일이 없는 문서 테스트
    print("\n===== 작성일 누락 테스트 =====")

    parser = HealthCheckParser()

    # 원본 Mock 데이터는 변경하지 않고 복사
    no_document_date = {
        **mock_ocr_result,
        "blocks": [
            block
            for block in mock_ocr_result["blocks"]
            if "2026년 9월 28일" not in block["text"]
        ]
    }

    fields = parser.parse(no_document_date)

    print("진단일:", fields["diagnosis_date"])
    print("작성일:", fields["document_date"])

    assert fields["diagnosis_date"] == "2025-10-31"
    assert fields["document_date"] is None, (
        "작성일이 없는데 진단일을 작성일로 잘못 추출했습니다."
    )

    print("작성일 누락 테스트 통과!")

