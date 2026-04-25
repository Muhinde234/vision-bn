import enum
from datetime import date
from typing import List, Optional

from sqlalchemy import Date, Enum, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class Sex(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Patient(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "patients"

    # Personal info
    patient_code: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    sex: Mapped[Optional[Sex]] = mapped_column(
        Enum(Sex, name="sex_enum"), nullable=True
    )
    phone: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    facility_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # relationships
    diagnoses: Mapped[List["Diagnosis"]] = relationship(  # noqa: F821
        "Diagnosis", back_populates="patient", lazy="select"
    )

    __table_args__ = (
        Index("ix_patients_facility", "facility_name"),
        Index("ix_patients_name", "full_name"),
    )

    def __repr__(self) -> str:
        return f"<Patient {self.patient_code} – {self.full_name}>"
