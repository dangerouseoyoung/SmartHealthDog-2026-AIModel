import logging
from celery import Celery

app = Celery("smart_health_dog")

logger = logging.getLogger(__name__)
logger.info("Celery application initialized.")