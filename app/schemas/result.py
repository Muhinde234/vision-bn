from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.result import ParasiteStage


class DetectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    stage: ParasiteStage
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DiagnosisResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    diagnosis_id: UUID
    image_id: UUID
    total_rbc_count: int
    total_parasite_count: int
    parasitaemia_percent: float
    ring_count: int
    trophozoite_count: int
    schizont_count: int
    gametocyte_count: int
    model_version: Optional[str]
    inference_time_ms: Optional[float]
    detections: List[DetectionRead] = []
    created_at: datetime


class InferenceRequest(BaseModel):
    """Sent from API to inference microservice."""
    image_url: str
    confidence_threshold: float = 0.35
    diagnosis_id: str


class InferenceResponse(BaseModel):
    """Received from inference microservice."""
    model_version: str
    inference_time_ms: float
    image_width: int
    image_height: int
    total_rbc_count: int
    detections: List[DetectionRead]
