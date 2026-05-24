from uuid import UUID
import enum
from typing import Optional

from sqlalchemy import BigInteger, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class ImageStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class DiagnosticImage(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "diagnostic_images"

    diagnosis_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False),
        ForeignKey("diagnoses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)  # S3 key or local path
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Image metadata
    width_px: Mapped[Optional[int]] = mapped_column(nullable=True)
    height_px: Mapped[Optional[int]] = mapped_column(nullable=True)
    magnification: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # e.g. "100x"

    status: Mapped[ImageStatus] = mapped_column(
        Enum(ImageStatus, name="image_status"),
        default=ImageStatus.PENDING,
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # relationships
    diagnosis: Mapped["Diagnosis"] = relationship(  # noqa: F821
        "Diagnosis", back_populates="images"
    )
    result: Mapped[Optional["DiagnosisResult"]] = relationship(  # noqa: F821
        "DiagnosisResult", back_populates="image", uselist=False
    )

    __table_args__ = (
        Index("ix_images_diagnosis_status", "diagnosis_id", "status"),
    )
