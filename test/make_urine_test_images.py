"""
소변 스트립 분석(urine_analysis.check_urine) 테스트용 가상 이미지 생성 + 오프라인 검증.

- 흰색 스트립 위에 10개의 패드를 ANALYTE_ORDER 순서(위→아래)로 그리고,
  각 패드는 REFERENCE_COLOR_TABLE의 기준색으로 칠한다.
- 이미지마다 정답(expected) JSON을 함께 저장한다.
- --verify 옵션: requests.get을 로컬 파일로 대체해 check_urine을 실행하고 정답과 비교.

사용법:
    python test/make_urine_test_images.py            # 이미지 생성
    python test/make_urine_test_images.py --verify   # 생성 + 검증
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import urine_analysis  # noqa: E402
from urine_analysis import ANALYTE_ORDER, REFERENCE_COLOR_TABLE  # noqa: E402

OUT_DIR = ROOT / "urine" / "samples"

# 레이아웃 (px)
IMG_W, IMG_H = 600, 1200
STRIP_W = 80
PAD = 56
GAP = 24
TOP_MARGIN = 40      # 스트립 위쪽 끝 ~ 첫 패드
HANDLE_LEN = 220     # 마지막 패드 아래 손잡이 부분
STRIP_COLOR = (245, 245, 242)   # RGB
BG_COLOR = (96, 110, 128)       # RGB, 회청색 테이블


def scenario_values(name: str, rng: random.Random) -> dict:
    """시나리오별 analyte -> 결과값."""
    if name == "normal":
        return {a: list(REFERENCE_COLOR_TABLE[a].keys())[0] for a in ANALYTE_ORDER}
    if name == "abnormal":
        v = {a: list(REFERENCE_COLOR_TABLE[a].keys())[0] for a in ANALYTE_ORDER}
        v.update({
            "Glucose": "60",
            "Protein": "1.0",
            "Blood": "80 Moderate",
            "Ketone": "Moderate 4.0",
            "Leukocytes": "Moderate 125",
            "Nitrite": "Positive",
            "pH": "8.0",
            "Specific Gravity": "1.030",
        })
        return v
    # random
    return {a: rng.choice(list(REFERENCE_COLOR_TABLE[a].keys())) for a in ANALYTE_ORDER}


VARIANTS = ["clean", "noisy", "tilted", "horizontal", "cluttered"]


def render(values: dict, rng: np.random.Generator, variant: str) -> np.ndarray:
    """RGB로 그린 뒤 BGR로 반환.

    variant:
      clean      - 정위치, 무보정
      noisy      - 조명 그라데이션 + 노이즈 + 블러
      tilted     - noisy + 12도 회전
      horizontal - noisy + 90도 회전(가로 촬영, 왼쪽이 첫 패드)
      cluttered  - noisy + 나뭇결 배경, 주변 물체, -6도 회전
    """
    realistic = variant != "clean"
    img = np.zeros((IMG_H, IMG_W, 3), np.float32)
    img[:] = BG_COLOR
    if variant == "cluttered":
        xs = np.arange(IMG_W, dtype=np.float32)
        wood = 18 * np.sin(xs / 9.0) + 10 * np.sin(xs / 23.0)
        img[:] = np.array([150, 112, 78], np.float32)  # 나무색 책상
        img += wood[None, :, None]
        cv2.rectangle(img, (20, 60), (150, 200), (40, 40, 45), -1)       # 검은 물체
        cv2.circle(img, (500, 1000), 70, (230, 230, 225), -1)            # 흰 원형 물체(뚜껑)
        cv2.line(img, (0, 900), (IMG_W, 860), (60, 60, 60), 6)           # 케이블

    strip_len = TOP_MARGIN + len(ANALYTE_ORDER) * (PAD + GAP) + HANDLE_LEN
    sx = (IMG_W - STRIP_W) // 2
    sy = (IMG_H - strip_len) // 2
    img[sy:sy + strip_len, sx:sx + STRIP_W] = STRIP_COLOR

    px = sx + (STRIP_W - PAD) // 2
    for i, analyte in enumerate(ANALYTE_ORDER):
        py = sy + TOP_MARGIN + i * (PAD + GAP)
        img[py:py + PAD, px:px + PAD] = REFERENCE_COLOR_TABLE[analyte][values[analyte]]

    if realistic:
        # 약한 조명 그라데이션(±4%) + 가우시안 노이즈 + 살짝 블러
        yy = np.linspace(0.96, 1.04, IMG_H, dtype=np.float32)[:, None, None]
        img = img * yy
        img += rng.normal(0, 1.5, img.shape).astype(np.float32)
        img = cv2.GaussianBlur(img, (3, 3), 0)

    img = np.clip(img, 0, 255).astype(np.uint8)
    angle = {"tilted": 12, "cluttered": -6}.get(variant)
    if angle is not None:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif variant == "horizontal":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 첫 패드가 오른쪽으로 감
        img = cv2.flip(img, 1)                           # 첫 패드를 왼쪽에 둠
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def generate(seed: int = 42) -> list:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    made = []
    for scenario in ["normal", "abnormal", "random1", "random2"]:
        values = scenario_values("random" if scenario.startswith("random") else scenario, py_rng)
        for variant in VARIANTS:
            stem = f"urine_{scenario}" + ("" if variant == "clean" else f"_{variant}")
            img_path = OUT_DIR / f"{stem}.png"
            cv2.imwrite(str(img_path), render(values, np_rng, variant))
            expected = [
                {"analyte": a, "value": values[a], "colorRGB": list(REFERENCE_COLOR_TABLE[a][values[a]])}
                for a in ANALYTE_ORDER
            ]
            (OUT_DIR / f"{stem}.expected.json").write_text(
                json.dumps({"results": expected}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            made.append(img_path)
    return made


def verify(paths: list) -> bool:
    ok_all = True
    for p in paths:
        data = p.read_bytes()
        fake = mock.Mock(content=data, raise_for_status=lambda: None)
        with mock.patch.object(urine_analysis.requests, "get", return_value=fake):
            out = urine_analysis.check_urine(f"file://{p.name}")

        expected = json.loads(p.with_suffix("").with_suffix(".expected.json").read_text(encoding="utf-8"))
        if "error" in out:
            print(f"[FAIL] {p.name}: {out['error']}")
            ok_all = False
            continue

        got = {r["analyte"]: r["value"] for r in out["results"]}
        exp = {r["analyte"]: r["value"] for r in expected["results"]}
        diff = {a: (exp[a], got.get(a)) for a in exp if exp[a] != got.get(a)}
        if diff:
            ok_all = False
            print(f"[FAIL] {p.name}: pads={out['pad_count']} mismatches(expected, got)={diff}")
        else:
            print(f"[PASS] {p.name}: pads={out['pad_count']}")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = generate(args.seed)
    print(f"생성 완료: {len(files)}개 -> {os.path.relpath(OUT_DIR, ROOT)}")
    if args.verify:
        sys.exit(0 if verify(files) else 1)
