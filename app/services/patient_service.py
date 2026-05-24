"""
Patient CRUD service.
"""
import random
import string
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.patient import Patient
from app.schemas.patient import PatientCreate, PatientUpdate


def _generate_patient_code() -> str:
    """Generate a human-readable patient code like VDX-2026-A3F9."""
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"VDX-2026-{suffix}"


class PatientService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, payload: PatientCreate) -> Patient:
        # Ensure uniqueness of generated code
        while True:
            code = _generate_patient_code()
            existing = await self.db.execute(
                select(Patient).where(Patient.patient_code == code)
            )
            if not existing.scalar_one_or_none():
                break

        patient = Patient(**payload.model_dump(), patient_code=code)
        self.db.add(patient)
        await self.db.flush()
        return patient

    async def get(self, patient_id: UUID) -> Patient:
        result = await self.db.execute(
            select(Patient).where(Patient.id == patient_id)
        )
        patient = result.scalar_one_or_none()
        if not patient:
            raise NotFoundError("Patient", str(patient_id))
        return patient

    async def list(
        self,
        facility_name: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Patient], int]:
        query = select(Patient)

        if facility_name:
            query = query.where(Patient.facility_name == facility_name)
        if search:
            like = f"%{search}%"
            query = query.where(
                Patient.full_name.ilike(like) | Patient.patient_code.ilike(like)
            )

        total_result = await self.db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = total_result.scalar_one()

        query = (
            query.order_by(Patient.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db.execute(query)
        return result.scalars().all(), total

    async def update(self, patient_id: UUID, payload: PatientUpdate) -> Patient:
        patient = await self.get(patient_id)
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(patient, field, value)
        await self.db.flush()
        return patient

    async def delete(self, patient_id: UUID) -> None:
        patient = await self.get(patient_id)
        await self.db.delete(patient)
        await self.db.flush()
