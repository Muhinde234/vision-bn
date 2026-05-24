from datetime import date
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import require_any_role, require_doctor_or_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.analytics import (
    CasesOverTime,
    DashboardStats,
    StageDistribution,
    UserActivityStats,
)
from app.schemas.common import APIResponse
from app.services.analytics_service import AnalyticsService
from app.services.report_service import ReportService

router = APIRouter()


@router.get("/dashboard", response_model=APIResponse[DashboardStats])
async def get_dashboard(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    facility_name: Optional[str] = Query(None),
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    """Aggregated malaria statistics for dashboards."""
    service = AnalyticsService(db)
    stats = await service.get_dashboard(
        date_from=date_from, date_to=date_to, facility_name=facility_name
    )
    return APIResponse(data=stats)


@router.get("/cases-over-time", response_model=APIResponse[List[CasesOverTime]])
async def cases_over_time(
    granularity: Literal["daily", "weekly", "monthly"] = Query("daily"),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    facility_name: Optional[str] = Query(None),
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    """Case counts grouped by day, week, or month."""
    service = AnalyticsService(db)
    data = await service.get_cases_over_time(
        granularity=granularity, date_from=date_from,
        date_to=date_to, facility_name=facility_name,
    )
    return APIResponse(data=data)


@router.get("/stage-distribution", response_model=APIResponse[StageDistribution])
async def stage_distribution(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    facility_name: Optional[str] = Query(None),
    _: User = Depends(require_any_role),
    db: AsyncSession = Depends(get_db),
):
    """Breakdown of parasite stages across all diagnoses."""
    service = AnalyticsService(db)
    data = await service.get_stage_distribution(
        date_from=date_from, date_to=date_to, facility_name=facility_name
    )
    return APIResponse(data=data)


@router.get("/user-activity", response_model=APIResponse[List[UserActivityStats]])
async def user_activity(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    _: User = Depends(require_doctor_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """Diagnoses submitted per technician."""
    service = AnalyticsService(db)
    data = await service.get_user_activity(date_from=date_from, date_to=date_to)
    return APIResponse(data=data)


@router.get("/export/csv")
async def export_csv(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    facility_name: Optional[str] = Query(None),
    _: User = Depends(require_doctor_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """Download diagnostics as CSV."""
    service = ReportService(db)
    csv_bytes = await service.generate_csv(
        date_from=date_from, date_to=date_to, facility_name=facility_name
    )
    filename = f"visiondx_report_{date_from or 'all'}_{date_to or 'now'}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/pdf")
async def export_pdf(
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    facility_name: Optional[str] = Query(None),
    _: User = Depends(require_doctor_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """Download diagnostics as PDF report."""
    service = ReportService(db)
    pdf_bytes = await service.generate_pdf(
        date_from=date_from, date_to=date_to, facility_name=facility_name
    )
    filename = f"visiondx_report_{date_from or 'all'}_{date_to or 'now'}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
