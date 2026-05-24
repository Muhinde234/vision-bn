"""
Celery tasks for async inference.
These are the production alternative to FastAPI BackgroundTasks –
use these when you need retry logic, priority queues, and Flower monitoring.
"""
import asyncio

from celery import Task
from celery.utils.log import get_task_logger

from app.tasks.celery_app import celery_app

logger = get_task_logger(__name__)


class AsyncTask(Task):
    """Base task that runs an async function in a new event loop."""
    def run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(
    bind=True,
    base=AsyncTask,
    name="inference.run",
    max_retries=3,
    default_retry_delay=5,
    queue="inference",
)
def run_inference_task(self, image_id: str, diagnosis_id: str):
    """
    Celery task: run YOLO inference for a given image.
    Retries up to 3 times on transient failures (network, GPU OOM, etc.).
    """
    from sqlalchemy import select
    from app.db.session import AsyncSessionLocal
    from app.models.diagnosis import Diagnosis
    from app.models.image import DiagnosticImage, ImageStatus
    from app.services.image_service import ImageService
    from app.services.inference_service import InferenceService
    from app.services.storage_service import StorageService
    from app.core.exceptions import InferenceError

    async def _run():
        async with AsyncSessionLocal() as db:
            image_res = await db.execute(
                select(DiagnosticImage).where(DiagnosticImage.id == image_id)
            )
            image = image_res.scalar_one_or_none()
            diag_res = await db.execute(
                select(Diagnosis).where(Diagnosis.id == diagnosis_id)
            )
            diagnosis = diag_res.scalar_one_or_none()

            if not image or not diagnosis:
                logger.error("Image or diagnosis not found", image_id=image_id)
                return

            storage = StorageService()
            image_service = ImageService(db, storage)
            inference_service = InferenceService(db, storage)

            try:
                await image_service.mark_processing(image)
                result = await inference_service.run(diagnosis, image)
                await image_service.mark_done(image)
                await db.commit()
                logger.info(
                    "Inference complete",
                    diagnosis_id=diagnosis_id,
                    parasitaemia=result.parasitaemia_percent,
                )
            except InferenceError as exc:
                logger.warning("Inference failed – retrying", error=exc.message)
                image.status = ImageStatus.FAILED
                image.error_message = exc.message
                await db.commit()
                raise self.retry(exc=exc)

    self.run_async(_run())
