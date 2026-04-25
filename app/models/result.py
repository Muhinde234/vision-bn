from uuid import UUID
"""
DiagnosisResult  – aggregate result for one diagnostic session
Detection        – individual YOLO bounding-box detection
"""
import enum
from typing import List, Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, JSON, String
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class ParasiteStage(str, enum.Enum):
    RING = "ring"
    TROPHOZOITE = "trophozoite"
    SCHIZONT = "schizont"
    GAMETOCYTE = "gametocyte"


class DiagnosisResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "diagnosis_results"

    diagnosis_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("diagnoses.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    image_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("diagnostic_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Core metrics
    total_rbc_count: Mapped[int] = mapped_column(Integer, default=0)
    total_parasite_count: Mapped[int] = mapped_column(Integer, default=0)
    parasitaemia_percent: Mapped[float] = mapped_column(Float, default=0.0)

    # Per-stage counts
    ring_count: Mapped[int] = mapped_column(Integer, default=0)
    trophozoite_count: Mapped[int] = mapped_column(Integer, default=0)
    schizont_count: Mapped[int] = mapped_column(Integer, default=0)
    gametocyte_count: Mapped[int] = mapped_column(Integer, default=0)

    # Model metadata
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_inference_output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # relationships
    diagnosis: Mapped["Diagnosis"] = relationship(  # noqa: F821
        "Diagnosis", back_populates="result"
    )
    image: Mapped["DiagnosticImage"] = relationship(  # noqa: F821
        "DiagnosticImage", back_populates="result"
    )
    detections: Mapped[List["Detection"]] = relationship(
        "Detection", back_populates="result", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_results_parasitaemia", "parasitaemia_percent"),
    )


class Detection(UUIDMixin, Base):
    """One bounding-box detection returned by the YOLO model."""
    __tablename__ = "detections"

    result_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("diagnosis_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[ParasiteStage] = mapped_column(
        String(50), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Bounding box (relative 0–1 coordinates)
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)

    result: Mapped["DiagnosisResult"] = relationship(
        "DiagnosisResult", back_populates="detections"
    )
