from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel


class SeverityBreakdown(BaseModel):
    negative: int = 0
    low: int = 0
    moderate: int = 0
    high: int = 0
    severe: int = 0


class StageBreakdown(BaseModel):
    ring: int = 0
    trophozoite: int = 0
    schizont: int = 0
    gametocyte: int = 0


class DailyTrend(BaseModel):
    date: date
    total_cases: int
    positive_cases: int
    positivity_rate: float


class FacilityStats(BaseModel):
    facility_name: str
    total_cases: int
    positive_cases: int
    avg_parasitaemia: float


class DashboardStats(BaseModel):
    total_diagnoses: int
    positive_cases: int
    positivity_rate: float
    avg_parasitaemia: float
    severity_breakdown: SeverityBreakdown
    stage_breakdown: StageBreakdown
    recent_trend: List[DailyTrend]
    top_facilities: List[FacilityStats]


class AnalyticsFilter(BaseModel):
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    facility_name: Optional[str] = None


class CasesOverTime(BaseModel):
    period: str
    total_cases: int
    positive_cases: int
    positivity_rate: float


class StageDistribution(BaseModel):
    ring: int = 0
    trophozoite: int = 0
    schizont: int = 0
    gametocyte: int = 0
    total_parasites: int = 0
    total_rbc: int = 0
    overall_parasitaemia: float = 0.0


class UserActivityStats(BaseModel):
    user_id: str
    full_name: str
    role: str
    facility_name: Optional[str]
    total_diagnoses: int
    completed_diagnoses: int
    positive_diagnoses: int
