"""
Prediction model – lightweight record of one AI inference request.
Decoupled from the clinical Diagnosis model so the frontend
can query simple upload→predict→result flows independently.
"""
import enum
import uuid
from uuid import UUID
from typing import Optional

from sqlalchemy import BigInteger, Enum, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class PredictionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DiseaseType(str, enum.Enum):
    MALARIA = "malaria"
    TUBERCULOSIS = "tuberculosis"
    PNEUMONIA = "pneumonia"
    DIABETIC_RETINOPATHY = "diabetic_retinopathy"
    SKIN_LESION = "skin_lesion"
    GENERAL = "general"


class Prediction(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "predictions"

    # Who made this request
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # What was uploaded
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # What disease module was targeted
    disease_type: Mapped[DiseaseType] = mapped_column(
        Enum(DiseaseType, name="disease_type"),
        default=DiseaseType.GENERAL,
        nullable=False,
        index=True,
    )

    # AI result
    status: Mapped[PredictionStatus] = mapped_column(
        Enum(PredictionStatus, name="prediction_status"),
        default=PredictionStatus.PENDING,
        nullable=False,
    )
    predicted_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Detailed model output (bounding boxes, all class probabilities, etc.)
    raw_output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Link back to clinical diagnosis session (optional – for malaria workflow)
    diagnosis_id: Mapped[Optional[UUID]] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("diagnoses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # relationships
    user: Mapped[Optional["User"]] = relationship("User", lazy="select")  # noqa: F821

    __table_args__ = (
        Index("ix_predictions_user_created", "user_id", "created_at"),
        Index("ix_predictions_disease_status", "disease_type", "status"),
    )

    def __repr__(self) -> str:
        return f"<Prediction {self.id} [{self.disease_type}] {self.predicted_class}>"
