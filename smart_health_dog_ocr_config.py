from smart_health_dog_config import *

# 진단서 OCR 전용 워커 설정: `ocr` 큐만 소비한다 (기존 `celery`/`respiration` 큐와 분리).
imports = (
    "ocr.certificate_ocr_task",
)

task_default_queue = "ocr"
task_routes = {"predict_certificate_ocr": {"queue": "ocr"}}
worker_prefetch_multiplier = 1
