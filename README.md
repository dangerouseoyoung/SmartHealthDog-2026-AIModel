# 🐶 SmartHealthDog AI Model (AI 진단 서비스)

본 저장소는 **똑똑하개 건강하개 애플리케이션**에서 사용하는
AI 기반 반려동물 건강 분석 워커입니다.

백엔드(Spring Boot)가 Redis 큐에 Celery 작업을 넣으면, 이 저장소의 워커가 작업을 꺼내 분석한 뒤
결과를 백엔드 API에 PATCH로 돌려줍니다.

```
앱 업로드 → Backend Submission(PENDING) → Redis 큐 → Celery Worker → 분석
         → AI_MODEL_SERVICE 로그인(JWT) → PATCH 결과/상태 → Backend(COMPLETED / FAILED)
```

---

## 📬 1. Celery 큐 구성

기능마다 큐와 워커를 분리합니다. 각 워커는 **자기 큐만** 소비해야 합니다.

| 큐 | 워커 설정 (`--config`) | 기능 | 태스크 이름 | 인자 |
|---|---|---|---|---|
| `detection` | `smart_health_dog_detection_config` | 개 안구 질환 | `smart_health_dog_disease_detection.predict_dog_disease` | `(image_url, submission_id)` |
| | | 고양이 안구 질환 | `smart_health_dog_disease_detection.predict_cat_disease` | `(image_url, submission_id)` |
| | | 소변 스트립 분석 | `smart_health_dog_disease_detection.predict_urine_analysis` | `(image_url, submission_id)` |
| `respiration` | `smart_health_dog_respiration_config` | 호흡수 측정 (영상) | `smart_health_dog_disease_detection.predict_respiration_rate` | `(video_url, submission_id)` |
| `ocr` | `smart_health_dog_ocr_config` | 진단서 OCR | `predict_certificate_ocr` | `(image_url, submission_id)` |

> 태스크 이름은 백엔드 `CeleryTaskSender`가 보내는 이름과 정확히 같아야 합니다. 함수 이름이나 모듈 위치를 바꿀 때 주의하세요.

### 1.1 `detection` 큐 — 안구 질환 / 소변 분석

코드: [smart_health_dog_disease_detection.py](smart_health_dog_disease_detection.py), [urine_analysis.py](urine_analysis.py)

**개 안구 질환 (`predict_dog_disease`)**
- fastai 단일 모델 10종 (`dog_models/*.pkl`, EfficientNet-B4)
- Blepharitis, Cataract, Conjunctivitis, Entropion, Eyelid Tumor, Epiphora, Non-ulcerative Keratitis, Nuclear Sclerosis, Pigmentary Keratitis, Ulcerative Keratitis

**고양이 안구 질환 (`predict_cat_disease`)**
- 단일 모델 3종: Conjunctivitis, Corneal_Scapula, Corneal Ulcer
- 5-Fold 앙상블 2종(5개 모델 확률 평균): Blepharitis, Non-ulcerative Keratitis

**안구 결과 콜백** — `AI_MODEL_SERVICE_EYE_UPDATE_ENDPOINT`
```json
{ "results": [ { "disease": "Cataract", "probability": 0.1234, "modelMd5Hash": "..." } ] }
```

**소변 분석 (`predict_urine_analysis`)**
- OpenCV로 스트립 검출 → 패드 10개 이상 검출 → 패드별 평균 색을 Lab 색공간에서 기준 색표와 비교
- 결과 콜백: `AI_MODEL_SERVICE_URINE_UPDATE_ENDPOINT`
  ```json
  { "results": [ { "analyte": "...", "value": "...", "colorRGB": [r, g, b] } ] }
  ```
- 결과 전송 후 상태 `COMPLETED`를 `AI_MODEL_SERVICE_STATUS_UPDATE_ENDPOINT`로 전송

**실패 처리** — `AI_MODEL_SERVICE_STATUS_UPDATE_ENDPOINT`로 `{"status": "FAILED", "failureReason": ...}`

| failureReason | 상황 |
|---|---|
| `TIMEOUT` | 이미지 다운로드 시간 초과 |
| `INVALID_INPUT` | 이미지 다운로드/디코딩 실패 (안구) |
| `INFERENCE_ERROR` | 추론 실패, 스트립/패드 검출 실패 (소변) |

### 1.2 `respiration` 큐 — 호흡수 측정

코드: [respiratory_analysis.py](respiratory_analysis.py) (태스크), [respiratory_pipeline.py](respiratory_pipeline.py) (파이프라인), [respiration/src/](respiration/src/)

- 영상 URL(https만 허용)을 임시 디렉터리에 스트리밍 다운로드 (최대 `RESPIRATION_MAX_VIDEO_SIZE_MB`)
- 단계별 subprocess 실행: YOLO 세그멘테이션(`respiration/weights/yolo11s-seg.pt`) → Optical Flow → 신호 처리 → 호흡수 분석
- 각 단계 제한 시간 `RESPIRATION_STAGE_TIMEOUT`, 단계 로그는 작업 디렉터리에 저장
- 결과 콜백: `AI_MODEL_SERVICE_RESPIRATION_UPDATE_ENDPOINT` → `{"respiratoryRate": 25}`
- 이후 `AI_MODEL_SERVICE_RESPIRATION_STATUS_UPDATE_ENDPOINT`로 `COMPLETED` 전송

| failureReason | 상황 |
|---|---|
| `TIMEOUT` | 다운로드/단계 시간 초과 |
| `INVALID_INPUT` | 잘못된 URL, 파일 크기 초과 등 |
| `MEASUREMENT_FAILED` | 분석은 끝났지만 신뢰할 수 있는 호흡수를 얻지 못함 |
| `INFERENCE_ERROR` | 파이프라인 실행 오류 |

### 1.3 `ocr` 큐 — 진단서 OCR

- 이미지 다운로드 → Google Vision `DOCUMENT_TEXT_DETECTION` → 좌표 기반 파서로 17개 항목 추출
- Vision 호출 전 Redis로 사용량 예약 (최근 32일 최대 800회, Redis 장애 시 호출하지 않음)
- 결과 콜백: `AI_MODEL_SERVICE_CERTIFICATE_RESULT_ENDPOINT` (`PATCH /api/submissions/{id}/certificate/result`)
  ```json
  { "success": true, "rawText": "...", "confidence": 0.91, "fields": { "...17개 camelCase..." }, "error": null }
  ```
- 실패도 같은 엔드포인트로 `{"success": false, "fields": null, "error": "<코드>"}` 전송.
  상태 전환(COMPLETED/FAILED/재시도)은 백엔드가 `confidence`(최소 0.75)와 `error` 코드로 판단
- `rawText`는 민감정보이므로 로그·Celery 결과 백엔드에 남기지 않음

---

## 📌 2. 프로젝트 구조

```
SmartHealthDog-2026-AIModel/
├── celery_app.py                          # 공용 Celery 앱
├── smart_health_dog_config.py             # 공용 설정 (.env → broker/backend, 인증)
├── smart_health_dog_detection_config.py   # detection 큐 워커 설정
├── smart_health_dog_respiration_config.py # respiration 큐 워커 설정
├── smart_health_dog_ocr_config.py         # ocr 큐 워커 설정
│
├── smart_health_dog_disease_detection.py  # 안구/소변 태스크
├── urine_analysis.py                      # 소변 스트립 색 분석
├── dog_models/  cat_models/               # 안구 질환 모델 (.pkl, Git 미포함)
│
├── respiratory_analysis.py                # 호흡수 태스크
├── respiratory_pipeline.py                # 호흡수 파이프라인 실행기
├── respiration/                           # 세그멘테이션·Optical Flow·신호 처리 코드
│   └── weights/                           # YOLO 가중치 (Git 미포함)
│
├── ocr/                                   # 진단서 OCR 태스크/Vision/파서
│
├── test/                                  # 테스트 및 벤치마크 스크립트
├── test.py
└── requirements.txt
```

---

## ⚙️ 3. 환경 설정

### 3.1 Git 클론 & 가상환경

```bash
git clone https://github.com/jyzzang02/SmartHealthDog-2026.git
cd SmartHealthDog-2026-AIModel

python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
source venv/bin/activate       # Linux (치타)
```

### 3.2 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🔑 4. 환경 변수 (.env)

프로젝트 루트에 `.env`를 만들고 아래 값을 채웁니다. 코드에서 dotenv로 자동 로딩되며, **Git에 포함하지 않습니다.**

```env
# 공용 — 백엔드 인증 / Redis
AI_MODEL_SERVICE_EMAIL=
AI_MODEL_SERVICE_PASSWORD=
AI_MODEL_SERVICE_LOGIN_ENDPOINT=

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# detection 큐 (안구/소변)
AI_MODEL_SERVICE_EYE_UPDATE_ENDPOINT=http://BACKEND_HOST/api/submissions/{id}/eye
AI_MODEL_SERVICE_URINE_UPDATE_ENDPOINT=
AI_MODEL_SERVICE_STATUS_UPDATE_ENDPOINT=http://BACKEND_HOST/api/submissions/{id}/status

# respiration 큐
AI_MODEL_SERVICE_RESPIRATION_UPDATE_ENDPOINT=
AI_MODEL_SERVICE_RESPIRATION_STATUS_UPDATE_ENDPOINT=
RESPIRATION_MODEL_PATH=          # 기본: respiration/weights/yolo11s-seg.pt
RESPIRATION_DEVICE=0             # GPU 번호 또는 cpu
RESPIRATION_VIDEO_TIMEOUT=180
RESPIRATION_STAGE_TIMEOUT=900
RESPIRATION_MAX_VIDEO_SIZE_MB=100

# ocr 큐
AI_MODEL_SERVICE_CERTIFICATE_RESULT_ENDPOINT=http://BACKEND_HOST/api/submissions/{id}/certificate/result
GOOGLE_APPLICATION_CREDENTIALS=  # 비우면 ocr/ 안의 서비스 계정 키 사용
OCR_MAX_REQUESTS_32D=800         # Vision 호출 상한 (1~800)
```

> `smart_health_dog_config.py`는 로그인/안구/소변/상태 엔드포인트가 비어 있으면 시작 시 오류를 냅니다.
> 모든 워커가 이 공용 설정을 불러오므로, OCR·호흡 워커만 띄울 때도 이 값들은 채워야 합니다.

---

## 🧠 5. 모델 파일 배치 (필수)

모델·가중치 파일은 Git에 포함되지 않습니다.

```
dog_models/*.pkl                       # detection 큐
cat_models/*.pkl                       # detection 큐 (KFold: *_fold1~5.pkl)
respiration/weights/yolo11s-seg.pt     # respiration 큐
ocr/<서비스 계정 키>.json               # ocr 큐 (Google Vision, 절대 커밋 금지)
```

---

## 🧰 6. Redis 실행

```bash
redis-server
redis-cli ping    # → PONG
```

---

## 🚀 7. Celery 워커 실행

큐마다 워커를 따로 띄웁니다. `-Q`로 **자기 큐만** 지정하세요.

```bash
# 안구 질환 / 소변 분석
celery -A celery_app --config smart_health_dog_detection_config worker -Q detection -c 1 -l info

# 호흡수 측정
celery -A celery_app --config smart_health_dog_respiration_config worker -Q respiration -c 1 -l info

# 진단서 OCR
celery -A celery_app --config smart_health_dog_ocr_config worker -Q ocr -c 1 -l info
```

> 운영 Redis에 연결하면 대기 중인 작업을 바로 처리합니다. 처음에는 개발 Redis에서 확인하고,
> 특히 `ocr` 큐는 Google Vision 호출(과금)이 일어나므로 대기 작업 수를 먼저 확인하세요.

---

## 🧪 8. 테스트

```bash
# OCR 오프라인 테스트 (Google/Redis/HTTP 호출 없음)
python -m pytest -q test/test_ocr_contract.py test/test_ocr_real_layout_offline.py test/test_google_vision_unit.py

# 기존 테스트
python test.py
```

---

## ⚠️ 중요 사항

- 모델 경로 (`./dog_models`, `./cat_models`) 변경 금지
- `.env` 없으면 인증 실패
- Redis 실행 안 하면 Celery 동작 안 함
- 각 워커는 자기 큐만 소비 (`detection` / `respiration` / `ocr`)
- Google 서비스 계정 키와 `.env`는 절대 커밋·공유 금지

---

## 📦 개발 환경

| 구분 | 설명 |
|------|------|
| 로컬 | Windows + CPU |
| 서버 | CHEETAH + GPU |
| 큐 | Redis (`detection`, `respiration`, `ocr`) |
| 처리 | Celery |

---

## ✅ 실행 순서 요약

1. clone
2. venv 생성
3. `pip install -r requirements.txt`
4. `.env` 설정
5. 모델/가중치/키 파일 배치
6. Redis 실행
7. 큐별 Celery 워커 실행
