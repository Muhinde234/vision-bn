from datetime import date, datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from app.models.patient import Sex


def _normalize_sex(v):
    if v is None:
        return v
    return str(v).lower()


class PatientCreate(BaseModel):
    full_name: str
    date_of_birth: Optional[date] = None
    sex: Optional[Sex] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    facility_name: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v):
        return _normalize_sex(v)


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    sex: Optional[Sex] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    facility_name: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v):
        return _normalize_sex(v)


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    patient_code: str
    full_name: str
    date_of_birth: Optional[date]
    sex: Optional[Sex]
    phone: Optional[str]
    address: Optional[str]
    facility_name: Optional[str]
    notes: Optional[str]
    created_at: datetime


class PatientSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    patient_code: str
    full_name: str
    facility_name: Optional[str]
