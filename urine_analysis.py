import cv2
import numpy as np
from math import sqrt
from typing import Dict, Tuple, Union
import requests

# -----------------------------
# Reference Table
# -----------------------------
REFERENCE_COLOR_TABLE = {
    "Leukocytes": {"Neg.": (168,165,140),"Trace 15": (158,150,141),"Small 70": (159,143,139),"Moderate 125": (152,123,145),"Large 500": (128,95,137)},
    "Nitrite": {"Neg.": (169,166,161),"Positive": (175,152,158)},
    "Urobilinogen": {"Normal": (175,166,152),"16": (173,135,124),"Trace 32": (169,97,93),"64+": (171,86,79),"128+": (169,70,65)},
    "Protein": {"Neg.": (164,159,94),"Trace": (147,157,97),"0.3": (137,154,109),"1.0": (113,151,130),"3.0": (122,155,154),"20.0": (101,149,137)},
    "pH": {"5.0": (172,136,43),"6.0": (144,130,25),"6.5": (120,126,29),"7.0": (101,109,34),"7.5": (79,105,34),"8.0": (56,94,33),"8.5": (1,108,139)},
    "Blood": {"Neg.": (176,132,72),"Non hemolyzed 10": (167,130,66),"Hemolyzed 10": (161,125,28),"25 Small": (136,121,34),"80 Moderate": (82,112,43),"200 Large": (20,79,31)},
    "Specific Gravity": {"1.000": (2,107,106),"1.005": (62,96,65),"1.010": (88,106,49),"1.015": (130,122,49),"1.020": (150,134,49),"1.025": (167,138,38),"1.030": (175,144,47)},
    "Ketone": {"Neg.": (180,137,120),"Trace 0.5": (173,116,103),"Small 1.5": (169,87,97),"Moderate 4.0": (160,35,66),"8.0": (114,16,50),"Large 16": (85,12,48)},
    "Bilirubin": {"Neg.": (179,175,169),"Small 17": (172,153,129),"Moderate 50": (175,155,140),"Large 100": (178,146,134)},
    "Glucose": {"Neg.": (125,155,152),"Trace 5": (109,149,105),"15": (106,122,33),"30": (132,105,20),"60": (125,67,31),"110": (101,42,24)},
}

ANALYTE_ORDER = [
    "Urobilinogen","Bilirubin","Ketone","Blood","Protein",
    "Nitrite","Leukocytes","Glucose","Specific Gravity","pH"
]

# -----------------------------
# Color Matching (LAB)
# -----------------------------
def rgb_to_lab(color):
    arr = np.uint8([[color]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0]
    # uint8 그대로 빼면 음수가 255 근처로 wrap-around 되어 거리 계산이 망가지므로 float로 변환
    return lab.astype(np.float32)

def closest_color_lab(ref_colors, r, g, b):
    sample_lab = rgb_to_lab((r,g,b))
    min_dist = float('inf')
    best_key = None
    best_rgb = None

    for key, rgb in ref_colors.items():
        ref_lab = rgb_to_lab(rgb)
        dist = np.linalg.norm(sample_lab - ref_lab)

        if dist < min_dist:
            min_dist = dist
            best_key = key
            best_rgb = rgb

    return best_key, best_rgb

# -----------------------------
# Strip Detection
# -----------------------------
MIN_STRIP_RATIO = 5.0          # 스트립 긴 변 / 짧은 변 최소 비율
MIN_STRIP_AREA_FRAC = 0.002    # 이미지 면적 대비 최소 스트립 면적
STRIP_INSET_FRAC = 0.08        # 스트립 crop 후 짧은 변 기준으로 깎아낼 가장자리 비율


def _strip_candidates(image):
    """스트립 후보 윤곽선. (1) 흰색·저채도 영역 마스크 (2) 엣지 기반 두 경로를 합친다."""
    h, w = image.shape[:2]
    long_k = max(15, int(min(h, w) * 0.02)) | 1

    # (1) 흰 스트립 본체: 밝고 채도가 낮은 영역 (Otsu로 밝기 임계값을 자동 결정)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    v_thr, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = ((hsv[:, :, 1] < 70) & (v >= max(v_thr, 120))).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # (2) 기존 엣지 방식 (배경이 밝아 흰색 마스크가 안 맞는 경우 대비)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((long_k, 5), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, long_k), np.uint8))

    cnts = []
    for mask in (white, edges):
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts.extend(found)
    return cnts


def detect_strip(image):
    """스트립을 찾아 긴 변이 세로가 되도록 회전 보정한 뒤 crop해서 반환한다.

    - 기울어진 스트립: minAreaRect 각도로 회전 보정
    - 가로 스트립: 왼쪽 끝이 위쪽(첫 패드)이 되도록 세운다
    - crop 가장자리의 배경이 패드 검출을 방해하지 않도록 안쪽으로 조금 깎는다
    """
    img_area = image.shape[0] * image.shape[1]
    best_rect, best_area = None, 0.0

    for c in _strip_candidates(image):
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if min(rw, rh) < 8:
            continue
        if max(rw, rh) / min(rw, rh) < MIN_STRIP_RATIO:
            continue
        rect_area = rw * rh
        # 윤곽선이 사각형을 충분히 채워야 함 (선/케이블 같은 것 배제)
        if rect_area < img_area * MIN_STRIP_AREA_FRAC or cv2.contourArea(c) < 0.6 * rect_area:
            continue
        if rect_area > best_area:
            best_area, best_rect = rect_area, ((cx, cy), (rw, rh), ang)

    if best_rect is None:
        return None

    # 긴 변의 방향 벡터 → 세로축과의 각도
    box = cv2.boxPoints(best_rect)
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    long_vec = max(edges, key=lambda e: np.hypot(*e))
    angle = np.degrees(np.arctan2(long_vec[0], long_vec[1]))  # 세로(+y)축 기준
    angle = (angle + 90) % 180 - 90                           # [-90, 90)
    if angle == -90:                                          # 정확히 가로: 왼쪽→위
        angle = 90.0

    (cx, cy), (rw, rh), _ = best_rect
    long_len, short_len = max(rw, rh), min(rw, rh)

    # 회전 후 잘리지 않도록 캔버스를 키워서 회전
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    rotated = cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    inset = int(round(short_len * STRIP_INSET_FRAC)) + 1
    half_w = int(short_len / 2) - inset
    half_h = int(long_len / 2) - inset
    x0, x1 = int(nw / 2) - half_w, int(nw / 2) + half_w
    y0, y1 = int(nh / 2) - half_h, int(nh / 2) + half_h
    if half_w <= 0 or half_h <= 0:
        return None
    return rotated[max(y0, 0):y1, max(x0, 0):x1]

# -----------------------------
# Pad Detection
# -----------------------------
def detect_pads(strip):
    """스트립 내부의 정사각형 패드를 위→아래 순서로 반환한다."""
    sh, sw = strip.shape[:2]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    block = max(21, (sw // 3) | 1)
    thresh = cv2.adaptiveThreshold(
        blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,block,5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # RETR_LIST: 스트립 테두리 윤곽선 안쪽에 있는 패드도 놓치지 않도록
    cnts,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_side = max(10, sw * 0.3)
    cands = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_side or w > sw * 1.05 or h > sw * 1.3:
            continue
        if not (0.7 < w / float(h) < 1.3):
            continue
        if cv2.contourArea(c) < 0.5 * w * h:
            continue
        cands.append((x,y,w,h))

    # 같은 패드의 바깥/안쪽 윤곽선 중복 제거: 큰 것부터 채택, 중심이 겹치면 버림
    pads = []
    for x,y,w,h in sorted(cands, key=lambda r: r[2]*r[3], reverse=True):
        cx, cy = x + w/2, y + h/2
        if any(px <= cx <= px+pw and py <= cy <= py+ph for px,py,pw,ph in pads):
            continue
        pads.append((x,y,w,h))

    # 크기가 다른 패드와 동떨어진 것(잡음) 제거
    if pads:
        med = float(np.median([w for _,_,w,_ in pads]))
        pads = [p for p in pads if 0.65 * med <= p[2] <= 1.35 * med]

    return sorted(pads, key=lambda x: x[1])

# -----------------------------
# Main Pipeline
# -----------------------------
def check_urine(image_url: str):

    # 1. load image
    try:
        res = requests.get(image_url, timeout=10)
        res.raise_for_status()
        img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": str(e)}

    if img is None:
        return {"error": "이미지 디코딩 실패"}

    # 2. strip
    strip = detect_strip(img)
    if strip is None:
        return {"error": "스트립 검출 실패"}

    # 3. pads
    pads = detect_pads(strip)

    if len(pads) < 10:
        return {"error": f"패드 부족 ({len(pads)})"}

    # 4. 색상 추출
    results = []

    for i, (x,y,w,h) in enumerate(pads[:len(ANALYTE_ORDER)]):

        analyte = ANALYTE_ORDER[i]

        roi = strip[y:y+h, x:x+w]
        b,g,r = np.mean(roi, axis=(0,1))

        key, rgb = closest_color_lab(
            REFERENCE_COLOR_TABLE[analyte],
            int(r), int(g), int(b)
        )

        results.append({
            "analyte": analyte,
            "value": key,
            "colorRGB": list(rgb)
        })

    return {
        "results": results,
        "pad_count": len(pads)
    }
