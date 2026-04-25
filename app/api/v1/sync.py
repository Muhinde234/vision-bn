"""
Mobile sync endpoint – batch upsert diagnoses from offline queue.
Designed for idempotent replay: duplicate mobile_sync_ids are silently skipped.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.permissions import require_lab_or_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.diagnosis import SyncDiagnosisPayload, SyncResponse
from app.services.diagnosis_service import DiagnosisService

router = APIRouter()


@router.post("/diagnoses", response_model=APIResponse[SyncResponse])
async def sync_diagnoses(
    payload: SyncDiagnosisPayload,
    current_user: User = Depends(require_lab_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Batch sync diagnoses from offline mobile queue.
    - Duplicate `mobile_sync_id` records are skipped (idempotent).
    - Partial success is supported; errors are reported per item.
    """
    service = DiagnosisService(db)
    result = await service.batch_sync(payload, current_user)
    return APIResponse(
        message=f"Sync complete: {result.synced} synced, {result.skipped} skipped",
        data=result,
    )
