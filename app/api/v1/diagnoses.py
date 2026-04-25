import math
from datetime import date
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.permissions import require_any_role, require_lab_or_admin
from app.db.session import get_db
from app.models.diagnosis import DiagnosisStatus
from app.models.user import User
from app.schemas.common import APIResponse, PaginatedResponse
from app.schemas.diagnosis import DiagnosisCreate, DiagnosisRead, DiagnosisSummary, DiagnosisUpdate
from app.services.diagnosis_service import DiagnosisService

router = APIRouter()


@router.post("", response_model=APIResponse[DiagnosisRead], status_code=status.HTTP_201_CREATED)
async def create_diagnosis(
    payload: DiagnosisCreate,
    current_user: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """Open a new diagnostic session for a patient."""
    service = DiagnosisService(db)
    diagnosis = await service.create(payload, current_user)
    return APIResponse(
        message="Diagnosis created",
        data=DiagnosisRead.model_validate(diagnosis),
    )


@router.get("", response_model=APIResponse[PaginatedResponse[DiagnosisSummary]])
async def list_diagnoses(
    patient_id: Optional[UUID] = Query(None),
    facility_name: Optional[str] = Query(None),
    status: Optional[DiagnosisStatus] = Query(None),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    service = DiagnosisService(db)
    items, total = await service.list(
        patient_id=patient_id,
        facility_name=facility_name,
        status=status,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )
    pages = math.ceil(total / page_size) if total else 1
    return APIResponse(
        data=PaginatedResponse(
            items=[DiagnosisSummary.model_validate(d) for d in items],
            total=total, page=page, page_size=page_size, pages=pages,
        )
    )


@router.get("/{diagnosis_id}", response_model=APIResponse[DiagnosisRead])
async def get_diagnosis(
    diagnosis_id: UUID,
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    service = DiagnosisService(db)
    diagnosis = await service.get(diagnosis_id)
    return APIResponse(data=DiagnosisRead.model_validate(diagnosis))


@router.patch("/{diagnosis_id}", response_model=APIResponse[DiagnosisRead])
async def update_diagnosis(
    diagnosis_id: UUID,
    payload: DiagnosisUpdate,
    _: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    service = DiagnosisService(db)
    diagnosis = await service.update(diagnosis_id, payload)
    return APIResponse(data=DiagnosisRead.model_validate(diagnosis))
