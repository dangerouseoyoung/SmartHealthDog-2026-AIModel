"""Manual local smoke test (실제 Vision 호출 1회 과금/카운트): python test/test_google_vision.py path/to/image.png"""
import argparse
import json
import sys
import time
from pathlib import Path

# 루트의 test.py 와 이름이 겹쳐 `python -m test.…` 은 불가 → 루트를 직접 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.google_vision_ocr import extract_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Google Vision document OCR local test")
    parser.add_argument("image", nargs="?", default="health_check_test.png")
    parser.add_argument("--output-dir", default="google_vision_results")
    args = parser.parse_args()
    path = Path(args.image)
    if not path.is_file():
        print(f"Image not found: {path}")
        return 2
    try:
        data = path.read_bytes()
    except OSError:
        print("Unable to read image file")
        return 2
    start = time.perf_counter()
    result = extract_text(data)
    result = {"filename": path.name, **result,
              "processingSeconds": round(time.perf_counter() - start, 2)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{path.stem}_ocr.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{path.stem}_ocr.txt").write_text(
        result["rawText"] or "", encoding="utf-8")
    print(f"success={result['success']} error={result['error']} "
          f"seconds={result['processingSeconds']}")
    print(f"Saved results to {output_dir}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
