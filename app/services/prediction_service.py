"""
Prediction service – orchestrates upload → AI inference → persist.
This is the single entry-point for the /predictions API layer.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ImageValidationError, InferenceError, NotFoundError
from app.core.logging import logger
from app.models.prediction import DiseaseType, Prediction, PredictionStatus
from app.models.user import User
from app.services.ai_service import AIService
from app.services.image_service import ImageService
from app.services.storage_service import StorageService


class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.storage = StorageService()
        self.image_service = ImageService(db, self.storage)
        self.ai_service = AIService()

    # ── Create + infer (synchronous path) ────────────────────────────────────

    async def create_and_predict(
        self,
        user: User,
        filename: str,
        content_type: str,
        file_data: bytes,
        disease_type: DiseaseType = DiseaseType.MALARIA,
        diagnosis_id: Optional[UUID] = None,
    ) -> Prediction:
        """
        Full pipeline:
          1. Validate & store image
          2. Create pending Prediction record
          3. Run AI inference (synchronous for <250 ms target)
          4. Persist result
          5. Return completed Prediction
        """
        # ── Step 1: validate, strip EXIF & store ─────────────────────────────
        from app.services.image_service import validate_and_strip_exif

        clean_data, _w, _h = validate_and_strip_exif(content_type, file_data)
        storage_path = await self.storage.save(clean_data, filename, prefix="predictions")

        # ── Step 2: create pending record ─────────────────────────────────────
        prediction = Prediction(
            user_id=user.id,
            original_filename=filename,
            storage_path=storage_path,
            file_size_bytes=len(clean_data),
            content_type=content_type,
            disease_type=disease_type,
            status=PredictionStatus.PROCESSING,
            diagnosis_id=diagnosis_id,
        )
        self.db.add(prediction)
        await self.db.flush()

        # ── Step 3: run inference ─────────────────────────────────────────────
        try:
            result = await self.ai_service.predict(clean_data, disease_type)
        except InferenceError as exc:
            prediction.status = PredictionStatus.FAILED
            prediction.error_message = exc.message
            await self.db.flush()
            raise

        # ── Step 4: persist result ────────────────────────────────────────────
        prediction.status = PredictionStatus.COMPLETED
        prediction.predicted_class = result.predicted_class
        prediction.confidence_score = result.confidence_score
        prediction.severity_level = result.severity_level
        prediction.recommendation = result.recommendation
        prediction.model_version = result.model_version
        prediction.inference_time_ms = result.inference_time_ms
        prediction.raw_output = result.detail.model_dump()

        await self.db.flush()
        return prediction

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, prediction_id: UUID, user: User) -> Prediction:
        result = await self.db.execute(
            select(Prediction).where(
                Prediction.id == prediction_id,
                Prediction.user_id == user.id,
            )
        )
        prediction = result.scalar_one_or_none()
        if not prediction:
            raise NotFoundError("Prediction", str(prediction_id))
        return prediction

    async def get_history(
        self,
        user: User,
        disease_type: Optional[DiseaseType] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Prediction], int]:
        query = select(Prediction).where(Prediction.user_id == user.id)

        if disease_type:
            query = query.where(Prediction.disease_type == disease_type)

        total_result = await self.db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = total_result.scalar_one()

        query = (
            query.order_by(Prediction.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db.execute(query)
        return result.scalars().all(), total
