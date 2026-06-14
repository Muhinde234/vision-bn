"""
Database Seeding Script

Populates the database with realistic fake data for users, patients,
diagnoses, and predictions. Useful for development and testing.

Usage:
  1. (First time) pip install Faker
  2. Run from the project root: python scripts/seed_db.py
"""
import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from faker import Faker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Setup path to be able to import app modules from the parent directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.logging import logger
from app.db.base import Base
from app.db.session import engine, get_db
from app.models.diagnosis import Diagnosis, DiagnosisStatus
from app.models.image import DiagnosticImage, ImageStatus
from app.models.patient import Gender, Patient
from app.models.prediction import DiseaseType, Prediction, PredictionStatus
from app.models.user import User, UserRole
from app.schemas.user import UserCreate
from app.services.auth_service import AuthService

# --- Configuration ---
NUM_DOCTORS = 3
NUM_LAB_TECHNICIANS = 5
NUM_PATIENTS = 50
MAX_DIAGNOSES_PER_PATIENT = 5
# ---------------------

fake = Faker("en_US")


async def seed_data():
    """Main function to seed the database."""
    logger.info("--- Starting Database Seeding ---")

    # Create tables if they don't exist
    async with engine.begin() as conn:
        # To wipe the database before seeding, uncomment the next line:
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    db: AsyncSession = await anext(get_db())
    auth_service = AuthService(db)

    try:
        # --- 1. Create Users ---
        logger.info("Seeding users...")
        users = []

        # Admin user (idempotent check)
        admin_email = "admin@visiondx.app"
        res = await db.execute(select(User).where(User.email == admin_email))
        admin_user = res.scalar_one_or_none()
        if not admin_user:
            admin_payload = UserCreate(email=admin_email, password="password", full_name="Admin User", role=UserRole.ADMIN)
            admin_user = await auth_service.register(admin_payload)
            logger.info(f"Created admin: {admin_user.email}")
        else:
            logger.info("Admin user already exists.")
        users.append(admin_user)

        # Doctors and Lab Technicians
        lab_techs = []
        user_payloads = []
        for _ in range(NUM_DOCTORS):
            name = fake.name()
            user_payloads.append(UserCreate(email=f"{name.lower().replace(' ', '.')}{random.randint(1,99)}@hospital.org", password="password", full_name=name, role=UserRole.DOCTOR))
        for _ in range(NUM_LAB_TECHNICIANS):
            name = fake.name()
            user_payloads.append(UserCreate(email=f"{name.lower().replace(' ', '.')}{random.randint(1,99)}@lab.org", password="password", full_name=name, role=UserRole.LAB_TECHNICIAN))

        for payload in user_payloads:
            user = await auth_service.register(payload)
            users.append(user)
            if user.role == UserRole.LAB_TECHNICIAN:
                lab_techs.append(user)

        logger.info(f"Seeded {len(users)} total users.")

        # --- 2. Create Patients ---
        logger.info("Seeding patients...")
        patients = []
        for i in range(NUM_PATIENTS):
            patient = Patient(
                id=uuid4(),
                patient_code=f"P{str(i+1).zfill(5)}",
                full_name=fake.name(),
                date_of_birth=fake.date_of_birth(minimum_age=1, maximum_age=90),
                gender=random.choice(list(Gender)),
                address=fake.address(),
                phone_number=fake.phone_number(),
                facility_name=random.choice(["St. Mary's Hospital", "General Clinic", "Westside Medical Center"]),
            )
            patients.append(patient)
        db.add_all(patients)
        await db.flush()
        logger.info(f"Seeded {len(patients)} patients.")

        # --- 3. Create Diagnoses, Images, and Predictions ---
        logger.info("Seeding diagnoses and predictions...")
        diagnoses_count = 0
        predictions_count = 0
        for patient in patients:
            num_diagnoses = random.randint(1, MAX_DIAGNOSES_PER_PATIENT)
            for _ in range(num_diagnoses):
                created_at = datetime.utcnow() - timedelta(days=random.randint(0, 365))
                lab_tech = random.choice(lab_techs)

                diagnosis = Diagnosis(id=uuid4(), patient_id=patient.id, created_by_id=lab_tech.id, status=random.choice(list(DiagnosisStatus)), notes=fake.sentence() if random.random() > 0.3 else None, created_at=created_at, updated_at=created_at)
                db.add(diagnosis)
                diagnoses_count += 1

                image = DiagnosticImage(id=uuid4(), diagnosis_id=diagnosis.id, storage_path=f"fake/path/{uuid4()}.jpg", original_filename=f"{fake.word()}.jpg", content_type="image/jpeg", file_size_bytes=random.randint(500_000, 5_000_000), width_px=1024, height_px=768, status=ImageStatus.DONE, created_at=created_at)
                db.add(image)

                is_infected = random.random() > 0.4  # 60% chance of being infected
                if is_infected:
                    predicted_class = random.choice(["Ring Stage", "Trophozoite", "Schizont", "Gametocyte"])
                    severity = "mild" if predicted_class == "Ring Stage" else "moderate" if predicted_class in ["Trophozoite", "Gametocyte"] else "severe"
                    confidence = random.uniform(0.65, 0.98)
                else:
                    predicted_class = "Negative"
                    severity = "negative"
                    confidence = random.uniform(0.95, 0.99)

                prediction = Prediction(id=uuid4(), user_id=lab_tech.id, diagnosis_id=diagnosis.id, image_id=image.id, disease_type=DiseaseType.MALARIA, status=PredictionStatus.DONE, predicted_class=predicted_class, confidence_score=confidence, severity_level=severity, recommendation="This is a recommendation based on the mock prediction.", model_version="yolov9n-mock-v2.3.1", raw_output={"mock": True}, original_filename=image.original_filename, file_size_bytes=image.file_size_bytes, created_at=created_at)
                db.add(prediction)
                predictions_count += 1

        await db.commit()
        logger.info(f"Seeded {diagnoses_count} diagnoses.")
        logger.info(f"Seeded {predictions_count} predictions.")

    except Exception as e:
        logger.error(f"An error occurred during seeding: {e}", exc_info=True)
        await db.rollback()
    finally:
        await db.close()

    logger.info("--- Database Seeding Complete ---")


if __name__ == "__main__":
    # This script is safe to run multiple times.
    # It will only add new fake users/patients if they don't already exist.
    asyncio.run(seed_data())

async def _run_inference_bg(image_id: str, diagnosis_id: str):
    """Background task: runs the full inference pipeline inside a single session."""
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        image_res = await db.execute(
            select(DiagnosticImage).where(DiagnosticImage.id == image_id)
        )
        image = image_res.scalar_one_or_none()
        diag_res = await db.execute(
            select(Diagnosis).where(Diagnosis.id == diagnosis_id)
        )
        diagnosis = diag_res.scalar_one_or_none()

        if not image or not diagnosis:
            return

        storage = StorageService()
        image_service = ImageService(db, storage)
        inference_service = InferenceService(db, storage)

        try:
            await image_service.mark_processing(image)
            result = await inference_service.run(diagnosis, image)
            await image_service.mark_done(image)
            await db.commit()
            logger.info("Inference completed successfully", image_id=str(image.id), diagnosis_id=str(diagnosis.id), prediction_id=str(result.id))
        except Exception as exc:
            logger.error("Inference failed", error=str(exc), image_id=str(image.id), diagnosis_id=str(diagnosis.id))
            await image_service.mark_error(image)
            await db.commit()