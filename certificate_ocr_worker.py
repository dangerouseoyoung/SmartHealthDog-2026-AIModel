
import os
import certificate_ocr_task  # OCR 태스크 등록

from dotenv import load_dotenv
from celery_app import app

load_dotenv()

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_password = os.getenv("REDIS_PASSWORD", "")

if redis_password:
    broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
else:
    broker_url = f"redis://{redis_host}:{redis_port}/0"

app.conf.update(
    broker_url=broker_url,
    result_backend=broker_url,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    imports=("certificate_ocr_task",),
)

