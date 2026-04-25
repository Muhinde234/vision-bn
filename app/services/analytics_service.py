"""
Analytics aggregation service.
"""
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import case, cast, Date, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.diagnosis import Diagnosis, DiagnosisSeverity, DiagnosisStatus
from app.models.result import DiagnosisResult
from app.schemas.analytics import (
    DailyTrend,
    DashboardStats,
    FacilityStats,
    SeverityBreakdown,
    StageBreakdown,
)

# ── Simple process-local TTL cache ───────────────────────────────────────────
_CACHE_TTL_SECONDS = 300  # 5 minutes
_cache: Dict[str, Tuple[Any, float]] = {}  # key → (value, expires_at)


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and entry[1] > time.monotonic():
        return entry[0]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (value, time.monotonic() + _CACHE_TTL_SECONDS)


class AnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        facility_name: Optional[str] = None,
    ) -> DashboardStats:
        cache_key = f"dashboard:{date_from}:{date_to}:{facility_name}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        # Base filter — include both completed and reviewed diagnoses
        filters = [Diagnosis.status.in_([DiagnosisStatus.COMPLETED, DiagnosisStatus.REVIEWED])]
        if date_from:
            filters.append(Diagnosis.created_at >= datetime.combine(date_from, datetime.min.time()))
        if date_to:
            filters.append(Diagnosis.created_at <= datetime.combine(date_to, datetime.max.time()))
        if facility_name:
            filters.append(Diagnosis.facility_name == facility_name)

        # Total / positive counts
        counts_q = await self.db.execute(
            select(
                func.count().label("total"),
                func.count(
                    case((Diagnosis.severity != DiagnosisSeverity.NEGATIVE, 1))
                ).label("positive"),
                func.avg(DiagnosisResult.parasitaemia_percent).label("avg_parasitaemia"),
            )
            .join(DiagnosisResult, DiagnosisResult.diagnosis_id == Diagnosis.id, isouter=True)
            .where(*filters)
        )
        row = counts_q.one()
        total = row.total or 0
        positive = row.positive or 0
        positivity_rate = round((positive / total * 100) if total else 0, 2)
        avg_parasitaemia = round(float(row.avg_parasitaemia or 0), 4)

        # Severity breakdown
        sev_q = await self.db.execute(
            select(Diagnosis.severity, func.count().label("cnt"))
            .where(*filters)
            .group_by(Diagnosis.severity)
        )
        sev_breakdown = SeverityBreakdown()
        for r in sev_q:
            if r.severity:
                setattr(sev_breakdown, r.severity.value, r.cnt)

        # Stage breakdown
        stage_q = await self.db.execute(
            select(
                func.sum(DiagnosisResult.ring_count).label("ring"),
                func.sum(DiagnosisResult.trophozoite_count).label("trophozoite"),
                func.sum(DiagnosisResult.schizont_count).label("schizont"),
                func.sum(DiagnosisResult.gametocyte_count).label("gametocyte"),
            )
            .join(Diagnosis, Diagnosis.id == DiagnosisResult.diagnosis_id)
            .where(*filters)
        )
        sr = stage_q.one()
        stage_breakdown = StageBreakdown(
            ring=int(sr.ring or 0),
            trophozoite=int(sr.trophozoite or 0),
            schizont=int(sr.schizont or 0),
            gametocyte=int(sr.gametocyte or 0),
        )

        # Daily trend (last 30 days)
        trend = await self._daily_trend(filters, days=30)

        # Top facilities
        top_facilities = await self._top_facilities(filters)

        stats = DashboardStats(
            total_diagnoses=total,
            positive_cases=positive,
            positivity_rate=positivity_rate,
            avg_parasitaemia=avg_parasitaemia,
            severity_breakdown=sev_breakdown,
            stage_breakdown=stage_breakdown,
            recent_trend=trend,
            top_facilities=top_facilities,
        )
        _cache_set(cache_key, stats)
        return stats

    async def _daily_trend(self, base_filters, days: int = 30) -> List[DailyTrend]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        trend_q = await self.db.execute(
            select(
                cast(Diagnosis.created_at, Date).label("day"),
                func.count().label("total"),
                func.count(
                    case((Diagnosis.severity != DiagnosisSeverity.NEGATIVE, 1))
                ).label("positive"),
            )
            .where(*base_filters, Diagnosis.created_at >= cutoff)
            .group_by("day")
            .order_by("day")
        )
        trends = []
        for row in trend_q:
            total_day = row.total or 0
            pos_day = row.positive or 0
            trends.append(DailyTrend(
                date=row.day,
                total_cases=total_day,
                positive_cases=pos_day,
                positivity_rate=round((pos_day / total_day * 100) if total_day else 0, 2),
            ))
        return trends

    async def _top_facilities(self, base_filters, limit: int = 10) -> List[FacilityStats]:
        fac_q = await self.db.execute(
            select(
                Diagnosis.facility_name,
                func.count().label("total"),
                func.count(
                    case((Diagnosis.severity != DiagnosisSeverity.NEGATIVE, 1))
                ).label("positive"),
                func.avg(DiagnosisResult.parasitaemia_percent).label("avg_p"),
            )
            .join(DiagnosisResult, DiagnosisResult.diagnosis_id == Diagnosis.id, isouter=True)
            .where(*base_filters, Diagnosis.facility_name.isnot(None))
            .group_by(Diagnosis.facility_name)
            .order_by(func.count().desc())
            .limit(limit)
        )
        return [
            FacilityStats(
                facility_name=r.facility_name,
                total_cases=r.total,
                positive_cases=r.positive,
                avg_parasitaemia=round(float(r.avg_p or 0), 4),
            )
            for r in fac_q
        ]
