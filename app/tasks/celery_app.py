"""
Celery application instance.
Workers are started with: celery -A app.tasks.celery_app worker --loglevel=info
"""
from celery import Celery

from app.config import settings

celery_app = Celery(
    "visiondx",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.inference_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,              # acknowledge only after task completes
    worker_prefetch_multiplier=1,     # fair dispatch – important for GPU tasks
    task_soft_time_limit=60,          # soft limit: raise SoftTimeLimitExceeded
    task_time_limit=120,              # hard limit: SIGKILL
    result_expires=86400,             # results expire after 1 day
    beat_schedule={},
)
