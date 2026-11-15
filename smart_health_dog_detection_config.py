from smart_health_dog_config import *

imports = (
    "smart_health_dog_disease_detection",
)

# 안구 질환 / 소변 분석 워커 설정: `celery` 큐만 소비한다.
task_default_queue = "celery"
task_routes = {
    "smart_health_dog_disease_detection.predict_dog_disease": {"queue": "celery"},
    "smart_health_dog_disease_detection.predict_cat_disease": {"queue": "celery"},
    "smart_health_dog_disease_detection.predict_urine_analysis": {"queue": "celery"},
}
