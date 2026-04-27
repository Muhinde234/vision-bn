import math
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.permissions import require_any_role, require_lab_or_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import APIResponse, PaginatedResponse
from app.schemas.diagnosis import DiagnosisRead, DiagnosisSummary
from app.schemas.patient import PatientCreate, PatientRead, PatientUpdate
from app.services.patient_service import PatientService
from app.services.diagnosis_service import DiagnosisService

router = APIRouter()


@router.post("", response_model=APIResponse[PatientRead], status_code=status.HTTP_201_CREATED)
async def create_patient(
    payload: PatientCreate,
    _: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """Register a new patient."""
    service = PatientService(db)
    patient = await service.create(payload)
    return APIResponse(message="Patient created", data=PatientRead.model_validate(patient))


@router.get("", response_model=APIResponse[PaginatedResponse[PatientRead]])
async def list_patients(
    facility_name: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Search by name or patient code"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    service = PatientService(db)
    patients, total = await service.list(
        facility_name=facility_name, search=search, page=page, page_size=page_size
    )
    pages = math.ceil(total / page_size) if total else 1
    return APIResponse(
        data=PaginatedResponse(
            items=[PatientRead.model_validate(p) for p in patients],
            total=total, page=page, page_size=page_size, pages=pages,
        )
    )


@router.get("/{patient_id}", response_model=APIResponse[PatientRead])
async def get_patient(
    patient_id: UUID,
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    service = PatientService(db)
    patient = await service.get(patient_id)
    return APIResponse(data=PatientRead.model_validate(patient))


@router.patch("/{patient_id}", response_model=APIResponse[PatientRead])
async def update_patient(
    patient_id: UUID,
    payload: PatientUpdate,
    _: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    service = PatientService(db)
    patient = await service.update(patient_id, payload)
    return APIResponse(data=PatientRead.model_validate(patient))


@router.delete("/{patient_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def delete_patient(
    patient_id: UUID,
    _: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    service = PatientService(db)
    await service.delete(patient_id)
    return APIResponse(message="Patient deleted")


@router.get("/{patient_id}/history", response_model=APIResponse[PaginatedResponse[DiagnosisSummary]])
async def get_patient_history(
    patient_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    """All diagnoses for a patient in reverse chronological order."""
    patient_service = PatientService(db)
    await patient_service.get(patient_id)  # raises 404 if not found

    diagnosis_service = DiagnosisService(db)
    diagnoses, total = await diagnosis_service.list(
        patient_id=patient_id, page=page, page_size=page_size
    )
    pages = math.ceil(total / page_size) if total else 1
    return APIResponse(
        data=PaginatedResponse(
            items=[DiagnosisSummary.model_validate(d) for d in diagnoses],
            total=total, page=page, page_size=page_size, pages=pages,
        )
    )


@router.get("/{patient_id}/results/{diagnosis_id}", response_model=APIResponse[DiagnosisRead])
async def get_patient_diagnosis(
    patient_id: UUID,
    diagnosis_id: UUID,
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    """Full result for one specific diagnosis belonging to a patient."""
    diagnosis_service = DiagnosisService(db)
    diagnosis = await diagnosis_service.get(diagnosis_id)
    if diagnosis.patient_id != patient_id:
        from app.core.exceptions import NotFoundError
        raise NotFoundError("Diagnosis", str(diagnosis_id))
    return APIResponse(data=DiagnosisRead.model_validate(diagnosis))
