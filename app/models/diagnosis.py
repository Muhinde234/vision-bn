from uuid import UUID
import enum
from typing import List, Optional

from sqlalchemy import Enum, ForeignKey, Index, String, Text
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class DiagnosisSeverity(str, enum.Enum):
    NEGATIVE = "negative"
    LOW = "low"            # parasitaemia < 1 %
    MODERATE = "moderate"  # 1–5 %
    HIGH = "high"          # > 5 %
    SEVERE = "severe"      # > 10 %  (WHO threshold)


class DiagnosisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    FAILED = "failed"


class Diagnosis(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "diagnoses"

    patient_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    facility_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    clinical_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[DiagnosisStatus] = mapped_column(
        Enum(DiagnosisStatus, name="diagnosis_status"),
        default=DiagnosisStatus.PENDING,
        nullable=False,
    )
    severity: Mapped[Optional[DiagnosisSeverity]] = mapped_column(
        Enum(DiagnosisSeverity, name="diagnosis_severity"),
        nullable=True,
    )

    # Mobile-sync fields
    mobile_sync_id: Mapped[Optional[str]] = mapped_column(
        String(100), unique=True, nullable=True, index=True
    )  # UUID generated on device for deduplication

    # relationships
    patient: Mapped["Patient"] = relationship(  # noqa: F821
        "Patient", back_populates="diagnoses"
    )
    created_by: Mapped[Optional["User"]] = relationship(  # noqa: F821
        "User", back_populates="diagnoses"
    )
    images: Mapped[List["DiagnosticImage"]] = relationship(  # noqa: F821
        "DiagnosticImage", back_populates="diagnosis", cascade="all, delete-orphan"
    )
    result: Mapped[Optional["DiagnosisResult"]] = relationship(  # noqa: F821
        "DiagnosisResult", back_populates="diagnosis", uselist=False,
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_diagnoses_patient_created", "patient_id", "created_at"),
        Index("ix_diagnoses_facility_date", "facility_name", "created_at"),
        Index("ix_diagnoses_severity_status", "severity", "status"),
    )
