"""
Diagnosis CRUD + sync.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError
from app.models.diagnosis import Diagnosis, DiagnosisStatus
from app.models.user import User
from app.schemas.diagnosis import DiagnosisCreate, DiagnosisUpdate, SyncDiagnosisPayload, SyncResponse


class DiagnosisService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Full load helper ──────────────────────────────────────────────────────

    def _full_load(self):
        return (
            selectinload(Diagnosis.patient),
            selectinload(Diagnosis.created_by),
            selectinload(Diagnosis.result),
        )

    # ── Create ────────────────────────────────────────────────────────────────

    async def create(self, payload: DiagnosisCreate, created_by: User) -> Diagnosis:
        # Deduplication for mobile sync
        if payload.mobile_sync_id:
            result = await self.db.execute(
                select(Diagnosis).where(
                    Diagnosis.mobile_sync_id == payload.mobile_sync_id
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                return existing

        diagnosis = Diagnosis(
            patient_id=payload.patient_id,
            created_by_id=created_by.id,
            facility_name=created_by.facility_name,
            clinical_notes=payload.clinical_notes,
            mobile_sync_id=payload.mobile_sync_id,
            status=DiagnosisStatus.PENDING,
        )
        self.db.add(diagnosis)
        await self.db.flush()
        return diagnosis

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, diagnosis_id: UUID) -> Diagnosis:
        result = await self.db.execute(
            select(Diagnosis)
            .options(*self._full_load())
            .where(Diagnosis.id == diagnosis_id)
        )
        diagnosis = result.scalar_one_or_none()
        if not diagnosis:
            raise NotFoundError("Diagnosis", str(diagnosis_id))
        return diagnosis

    async def list(
        self,
        patient_id: Optional[UUID] = None,
        facility_name: Optional[str] = None,
        status: Optional[DiagnosisStatus] = None,
        date_from=None,
        date_to=None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Diagnosis], int]:
        query = select(Diagnosis).options(*self._full_load())

        if patient_id:
            query = query.where(Diagnosis.patient_id == patient_id)
        if facility_name:
            query = query.where(Diagnosis.facility_name == facility_name)
        if status:
            query = query.where(Diagnosis.status == status)
        if date_from:
            query = query.where(Diagnosis.created_at >= date_from)
        if date_to:
            query = query.where(Diagnosis.created_at <= date_to)

        total_result = await self.db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = total_result.scalar_one()

        query = (
            query.order_by(Diagnosis.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db.execute(query)
        return result.scalars().all(), total

    # ── Update ────────────────────────────────────────────────────────────────

    async def update(self, diagnosis_id: UUID, payload: DiagnosisUpdate) -> Diagnosis:
        diagnosis = await self.get(diagnosis_id)
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(diagnosis, field, value)
        await self.db.flush()
        return diagnosis

    # ── Mobile sync ───────────────────────────────────────────────────────────

    async def batch_sync(
        self, payload: SyncDiagnosisPayload, user: User
    ) -> SyncResponse:
        synced, skipped, errors = 0, 0, []

        for item in payload.diagnoses:
            try:
                if item.mobile_sync_id:
                    existing = await self.db.execute(
                        select(Diagnosis).where(
                            Diagnosis.mobile_sync_id == item.mobile_sync_id
                        )
                    )
                    if existing.scalar_one_or_none():
                        skipped += 1
                        continue

                await self.create(item, user)
                synced += 1
            except Exception as exc:
                errors.append(f"sync_id={item.mobile_sync_id}: {exc}")

        return SyncResponse(synced=synced, skipped=skipped, errors=errors)
