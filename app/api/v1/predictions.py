"""
/predictions – the primary API surface for the Next.js frontend.

Endpoints:
  POST   /predictions/predict          → upload image + run AI → return result
  GET    /predictions/history          → paginated list of user's predictions
  GET    /predictions/{prediction_id}  → single prediction with full detail
"""
import math
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.exceptions import ImageValidationError, InferenceError
from app.db.session import get_db
from app.models.prediction import DiseaseType
from app.models.user import User
from app.schemas.common import APIResponse, PaginatedResponse
from app.schemas.prediction import PredictionRead, PredictionSummary
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post(
    "/predict",
    response_model=APIResponse[PredictionRead],
    status_code=status.HTTP_201_CREATED,
    summary="Upload image and run AI diagnosis",
)
async def predict(
    file: UploadFile = File(..., description="Blood smear or medical image (JPEG/PNG/TIFF, max 10 MB)"),
    disease_type: DiseaseType = Form(DiseaseType.MALARIA, description="Disease detection module to use"),
    diagnosis_id: Optional[UUID] = Form(None, description="Link to existing clinical diagnosis (optional)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    **Core prediction endpoint.**

    1. Validates and stores the uploaded image
    2. Runs YOLOv9 inference (mock in dev, real in production)
    3. Returns structured result with:
       - `predicted_class` – detected condition
       - `confidence_score` – model confidence (0–1)
       - `severity_level` – negative / mild / moderate / severe
       - `recommendation` – clinical guidance text
       - `raw_output` – full bounding boxes + class probabilities

    The response is identical in structure regardless of disease module,
    so the frontend result card works for all 6 detection types.
    """
    file_data = await file.read()

    service = PredictionService(db)
    try:
        prediction = await service.create_and_predict(
            user=current_user,
            filename=file.filename or "upload.jpg",
            content_type=file.content_type or "image/jpeg",
            file_data=file_data,
            disease_type=disease_type,
            diagnosis_id=diagnosis_id,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        )
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {exc.message}",
        )

    return APIResponse(
        message="Prediction completed successfully",
        data=PredictionRead.model_validate(prediction),
    )


@router.get(
    "/history",
    response_model=APIResponse[PaginatedResponse[PredictionSummary]],
    summary="Get prediction history for current user",
)
async def get_history(
    disease_type: Optional[DiseaseType] = Query(None, description="Filter by disease module"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns paginated prediction history for the authenticated user.
    Used to power the history/results list in the dashboard.
    """
    service = PredictionService(db)
    items, total = await service.get_history(
        user=current_user,
        disease_type=disease_type,
        page=page,
        page_size=page_size,
    )
    pages = math.ceil(total / page_size) if total else 1

    return APIResponse(
        data=PaginatedResponse(
            items=[PredictionSummary.model_validate(p) for p in items],
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )
    )


@router.get(
    "/{prediction_id}",
    response_model=APIResponse[PredictionRead],
    summary="Get a specific prediction with full detail",
)
async def get_prediction(
    prediction_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the full prediction record including bounding boxes and class probabilities.
    Used when the user clicks a result card to see detailed analysis.
    """
    from app.core.exceptions import NotFoundError

    service = PredictionService(db)
    try:
        prediction = await service.get(prediction_id, current_user)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return APIResponse(data=PredictionRead.model_validate(prediction))
