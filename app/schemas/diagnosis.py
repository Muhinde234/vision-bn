from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.diagnosis import DiagnosisSeverity, DiagnosisStatus
from app.schemas.patient import PatientSummary
from app.schemas.result import DiagnosisResultRead
from app.schemas.user import UserSummary


class DiagnosisCreate(BaseModel):
    patient_id: UUID
    clinical_notes: Optional[str] = None
    mobile_sync_id: Optional[str] = None  # device-generated UUID for deduplication


class DiagnosisUpdate(BaseModel):
    clinical_notes: Optional[str] = None
    status: Optional[DiagnosisStatus] = None


class DiagnosisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    patient: PatientSummary
    created_by: Optional[UserSummary]
    facility_name: Optional[str]
    clinical_notes: Optional[str]
    status: DiagnosisStatus
    severity: Optional[DiagnosisSeverity]
    mobile_sync_id: Optional[str]
    result: Optional[DiagnosisResultRead]
    created_at: datetime
    updated_at: datetime


class DiagnosisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    patient_id: UUID
    status: DiagnosisStatus
    severity: Optional[DiagnosisSeverity]
    parasitaemia_percent: Optional[float] = None
    created_at: datetime


# ── Mobile sync ───────────────────────────────────────────────────────────────
class SyncDiagnosisPayload(BaseModel):
    """Batch payload sent from mobile device on reconnection."""
    diagnoses: List[DiagnosisCreate]


class SyncResponse(BaseModel):
    synced: int
    skipped: int
    errors: List[str] = []
