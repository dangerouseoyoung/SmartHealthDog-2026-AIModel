import logging

from celery import Celery

# --------------------------------------------------
# Celery Application
# --------------------------------------------------

app = Celery("smart_health_dog")

# smart_health_dog_config.py의 설정을 로드
app.config_from_object("smart_health_dog_config")

logger = logging.getLogger(__name__)

logger.info("Celery application initialized.")