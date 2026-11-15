"""Reparse previously saved Google Vision blocks OFFLINE. No credentials/network/API calls.

Usage (프로젝트 루트에서):
    python -m ocr.reparse_saved_ocr
"""
from __future__ import annotations

import json
from pathlib import Path

from ocr.health_check_parser import HealthCheckParser

SOURCE = Path.home() / ".smarthealthdog-secrets/ocr-debug.json"
DEST = Path.home() / ".smarthealthdog-secrets/ocr-reparsed.json"


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"저장된 Google Vision 파일이 없습니다: {SOURCE}")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    ocr = data.get("ocr") or {}
    blocks = ocr.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise SystemExit("파일에 OCR 블록이 없습니다. API를 호출하지 않았습니다.")

    original = data.get("fields") or {}
    revised = HealthCheckParser().parse(ocr)
    revised_data = {
        "source": str(SOURCE),
        "field_count": len(revised),
        "non_null_count": sum(v is not None for v in revised.values()),
        "fields": revised,
    }
    DEST.write_text(json.dumps(revised_data, ensure_ascii=False, indent=2), encoding="utf-8")
    DEST.chmod(0o600)

    print("이전 OCR 블록 수:", len(blocks))
    print("기존 채워진 필드:", sum(v is not None for v in original.values()))
    print("수정 후 채워진 필드:", revised_data["non_null_count"])
    print("Google Vision 재호출: 없음")
    print("변경된 항목:")
    for field, value in revised.items():
        if original.get(field) != value:
            print(" -", field, ":", repr(original.get(field)), "→", repr(value))
    print("결과 저장:", DEST)
    print("주의: 진단서에 포함된 건강정보·개인정보를 공유하기 전에 가리세요.")


if __name__ == "__main__":
    main()
