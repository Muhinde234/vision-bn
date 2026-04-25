"""
Pydantic schemas for the prediction / AI inference layer.
These are what the Next.js frontend sends and receives.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.prediction import DiseaseType, PredictionStatus


# ── Request ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Optional JSON body that can accompany the multipart form upload.
    The image file itself is sent as multipart/form-data.
    """
    disease_type: DiseaseType = DiseaseType.MALARIA
    diagnosis_id: Optional[UUID] = Field(
        None,
        description="Link to an existing clinical diagnosis session (optional)",
    )


# ── AI result sub-schemas ─────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float


class AIResultDetail(BaseModel):
    """Full model output – returned in raw_output for advanced consumers."""
    model_version: str
    inference_time_ms: float
    image_width: int
    image_height: int
    class_probabilities: Dict[str, float] = {}
    bounding_boxes: List[BoundingBox] = []


# ── Response ──────────────────────────────────────────────────────────────────

class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    disease_type: DiseaseType
    status: PredictionStatus

    # Primary result – what the frontend cards display
    predicted_class: Optional[str]
    confidence_score: Optional[float]
    recommendation: Optional[str]
    severity_level: Optional[str]

    # File info
    original_filename: str
    file_size_bytes: int

    # Model metadata
    model_version: Optional[str]
    inference_time_ms: Optional[float]

    # Detail (only included when fetching a single prediction)
    raw_output: Optional[Dict[str, Any]] = None

    created_at: datetime
    error_message: Optional[str] = None


class PredictionSummary(BaseModel):
    """Lightweight record for history list."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    disease_type: DiseaseType
    status: PredictionStatus
    predicted_class: Optional[str]
    confidence_score: Optional[float]
    severity_level: Optional[str]
    original_filename: str
    created_at: datetime
