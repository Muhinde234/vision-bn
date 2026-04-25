"""
Image upload and inference trigger endpoint.

Flow:
  1. POST /images/upload  → validate + store + queue Celery task
  2. Celery task calls InferenceService
  3. GET  /diagnoses/{id} → includes result once completed
"""
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.exceptions import ImageValidationError, InferenceError
from app.core.logging import logger
from app.core.permissions import require_lab_or_admin
from app.db.session import get_db
from app.models.diagnosis import Diagnosis
from app.models.image import DiagnosticImage, ImageStatus
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.result import DiagnosisResultRead
from app.services.image_service import ImageService
from app.services.inference_service import InferenceService
from app.services.storage_service import StorageService

router = APIRouter()


@router.post("/upload", response_model=APIResponse[DiagnosisResultRead], status_code=status.HTTP_202_ACCEPTED)
async def upload_image(
    diagnosis_id: UUID = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a blood smear image for a diagnosis.
    The image is stored immediately; inference runs asynchronously.
    Returns 202 Accepted — poll GET /diagnoses/{id} for results.
    """
    # Verify diagnosis exists
    result = await db.execute(select(Diagnosis).where(Diagnosis.id == diagnosis_id))
    diagnosis = result.scalar_one_or_none()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    data = await file.read()
    storage = StorageService()
    image_service = ImageService(db, storage)

    try:
        image_record = await image_service.upload(
            diagnosis_id=diagnosis_id,
            filename=file.filename,
            content_type=file.content_type,
            data=data,
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.message)

    logger.info("Image uploaded, queuing inference", image_id=str(image_record.id))

    # Queue inference as background task
    # In production switch this to: inference_tasks.run_inference.delay(...)
    background_tasks.add_task(
        _run_inference_bg,
        image_id=str(image_record.id),
        diagnosis_id=str(diagnosis_id),
    )

    return APIResponse(
        message="Image uploaded. Inference queued.",
        data=None,
    )


async def _run_inference_bg(image_id: str, diagnosis_id: str):
    """Background task: runs the full inference pipeline inside a single session."""
    from app.db.session import AsyncSessionLocal

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
            await db.rollback()
            image.status = ImageStatus.FAILED
            image.error_message = exc.message
            await db.commit()
            logger.error("Inference failed", error=exc.message)
        except Exception as exc:
            await db.rollback()
            logger.exception("Unexpected inference error", error=str(exc))
