"""
Database seeder — run once to populate VisionDx with realistic test data.
Usage:
    python seed.py
"""
import json
import os
import uuid
from datetime import date, datetime, timezone, timedelta
import random

import bcrypt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ── Load env from .env file if present ───────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

SYNC_DATABASE_URL = os.environ.get("SYNC_DATABASE_URL", "sqlite:///./visiondx_dev.db")

engine = create_engine(SYNC_DATABASE_URL, echo=False)


def hash_pw(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def uid() -> str:
    return str(uuid.uuid4())


def ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def seed():
    # Import models so Base knows all tables
    import app.models  # noqa
    from app.db.base import Base

    print("Creating tables...")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        # ── Check if already seeded ───────────────────────────────────────────
        existing = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
        if existing and existing > 0:
            print(f"Database already has {existing} users — skipping seed.")
            print("To reseed, run:  python seed.py --force")
            if "--force" not in __import__("sys").argv:
                return
            print("Force flag detected — clearing tables...")
            db.execute(text("DELETE FROM detections"))
            db.execute(text("DELETE FROM diagnosis_results"))
            db.execute(text("DELETE FROM diagnostic_images"))
            db.execute(text("DELETE FROM predictions"))
            db.execute(text("DELETE FROM diagnoses"))
            db.execute(text("DELETE FROM patients"))
            db.execute(text("DELETE FROM refresh_tokens"))
            db.execute(text("DELETE FROM users"))
            db.commit()

        now = datetime.now(timezone.utc)

        # ── USERS ─────────────────────────────────────────────────────────────
        print("Seeding users...")

        admin_id = uid()
        doctor1_id = uid()
        doctor2_id = uid()
        lab1_id = uid()
        lab2_id = uid()
        lab3_id = uid()

        users = [
            {
                "id": admin_id,
                "email": "admin@visiondx.com",
                "full_name": "System Administrator",
                "hashed_password": hash_pw("Admin@1234"),
                "role": "admin",
                "facility_name": "VisionDx HQ",
                "is_active": True,
                "is_verified": True,
                "created_at": ago(60),
                "updated_at": ago(60),
            },
            {
                "id": doctor1_id,
                "email": "dr.uwimana@visiondx.com",
                "full_name": "Dr. Jean Uwimana",
                "hashed_password": hash_pw("Doctor@1234"),
                "role": "doctor",
                "facility_name": "Kigali University Hospital",
                "is_active": True,
                "is_verified": True,
                "created_at": ago(50),
                "updated_at": ago(50),
            },
            {
                "id": doctor2_id,
                "email": "dr.mukamana@visiondx.com",
                "full_name": "Dr. Alice Mukamana",
                "hashed_password": hash_pw("Doctor@1234"),
                "role": "doctor",
                "facility_name": "Butare General Hospital",
                "is_active": True,
                "is_verified": True,
                "created_at": ago(45),
                "updated_at": ago(45),
            },
            {
                "id": lab1_id,
                "email": "lab.niyonzima@visiondx.com",
                "full_name": "Eric Niyonzima",
                "hashed_password": hash_pw("Lab@12345"),
                "role": "lab_technician",
                "facility_name": "Kigali University Hospital",
                "is_active": True,
                "is_verified": True,
                "created_at": ago(40),
                "updated_at": ago(40),
            },
            {
                "id": lab2_id,
                "email": "lab.ingabire@visiondx.com",
                "full_name": "Grace Ingabire",
                "hashed_password": hash_pw("Lab@12345"),
                "role": "lab_technician",
                "facility_name": "Butare General Hospital",
                "is_active": True,
                "is_verified": True,
                "created_at": ago(35),
                "updated_at": ago(35),
            },
            {
                "id": lab3_id,
                "email": "lab.habimana@visiondx.com",
                "full_name": "Patrick Habimana",
                "hashed_password": hash_pw("Lab@12345"),
                "role": "lab_technician",
                "facility_name": "Ruhengeri District Hospital",
                "is_active": True,
                "is_verified": True,
                "created_at": ago(30),
                "updated_at": ago(30),
            },
        ]

        db.execute(text("""
            INSERT INTO users (id, email, full_name, hashed_password, role,
                               facility_name, is_active, is_verified, created_at, updated_at)
            VALUES (:id, :email, :full_name, :hashed_password, :role,
                    :facility_name, :is_active, :is_verified, :created_at, :updated_at)
        """), users)
        db.commit()
        print(f"  ✓ {len(users)} users created")

        # ── PATIENTS ──────────────────────────────────────────────────────────
        print("Seeding patients...")

        patient_ids = [uid() for _ in range(12)]
        facilities = [
            "Kigali University Hospital",
            "Butare General Hospital",
            "Ruhengeri District Hospital",
        ]

        patients_data = [
            ("Mugisha Emmanuel", "1990-03-15", "male",   "+250788100001", "Kigali, Nyarugenge",     facilities[0]),
            ("Uwase Diane",      "1985-07-22", "female", "+250788100002", "Kigali, Gasabo",          facilities[0]),
            ("Nkurunziza Pascal","2000-11-08", "male",   "+250788100003", "Butare, Huye",            facilities[1]),
            ("Kampire Solange",  "1978-05-30", "female", "+250788100004", "Butare, Huye",            facilities[1]),
            ("Hakizimana David", "1995-01-17", "male",   "+250788100005", "Ruhengeri, Musanze",      facilities[2]),
            ("Mukamurenzi Rose", "1988-09-04", "female", "+250788100006", "Ruhengeri, Musanze",      facilities[2]),
            ("Ntirenganya Yves", "2003-06-25", "male",   "+250788100007", "Kigali, Kicukiro",        facilities[0]),
            ("Ingabire Clarisse","1992-12-11", "female", "+250788100008", "Butare, Nyanza",          facilities[1]),
            ("Bizimana Jean",    "1970-04-03", "male",   "+250788100009", "Ruhengeri, Burera",       facilities[2]),
            ("Mukandori Yvonne", "1999-08-19", "female", "+250788100010", "Kigali, Nyarugenge",      facilities[0]),
            ("Ndayishimiye Eric","2005-02-28", "male",   "+250788100011", "Gitarama, Muhanga",       facilities[1]),
            ("Uwimana Christine","1983-10-07", "female", "+250788100012", "Byumba, Gicumbi",         facilities[2]),
        ]

        patients_rows = []
        for i, (name, dob, sex, phone, address, facility) in enumerate(patients_data):
            patients_rows.append({
                "id": patient_ids[i],
                "patient_code": f"VDX-{2026}-{str(i+1).zfill(4)}",
                "full_name": name,
                "date_of_birth": dob,
                "sex": sex,
                "phone": phone,
                "address": address,
                "facility_name": facility,
                "notes": None,
                "created_at": ago(random.randint(5, 55)),
                "updated_at": ago(random.randint(1, 4)),
            })

        db.execute(text("""
            INSERT INTO patients (id, patient_code, full_name, date_of_birth, sex,
                                  phone, address, facility_name, notes, created_at, updated_at)
            VALUES (:id, :patient_code, :full_name, :date_of_birth, :sex,
                    :phone, :address, :facility_name, :notes, :created_at, :updated_at)
        """), patients_rows)
        db.commit()
        print(f"  ✓ {len(patients_rows)} patients created")

        # ── DIAGNOSES ─────────────────────────────────────────────────────────
        print("Seeding diagnoses...")

        diagnosis_ids = [uid() for _ in range(12)]
        severity_map = ["negative", "low", "moderate", "high", "severe"]
        status_map = ["completed", "completed", "completed", "reviewed", "completed"]

        diagnoses_rows = []
        for i in range(12):
            sev_idx = i % 5
            diagnoses_rows.append({
                "id": diagnosis_ids[i],
                "patient_id": patient_ids[i],
                "created_by_id": random.choice([lab1_id, lab2_id, lab3_id]),
                "facility_name": patients_rows[i]["facility_name"],
                "clinical_notes": random.choice([
                    "Patient presents with fever (38.9°C), chills, headache for 3 days.",
                    "Routine malaria screening. Patient returned from high-risk zone.",
                    "Fever and fatigue for 2 days. Suspected malaria.",
                    "Follow-up test after initial positive. Patient on treatment.",
                    "Pregnant patient — routine antenatal malaria check.",
                    "Child with high fever and vomiting. Urgent screening.",
                ]),
                "status": status_map[sev_idx],
                "severity": severity_map[sev_idx],
                "mobile_sync_id": None,
                "created_at": ago(random.randint(1, 30)),
                "updated_at": ago(random.randint(0, 1)),
            })

        db.execute(text("""
            INSERT INTO diagnoses (id, patient_id, created_by_id, facility_name,
                                   clinical_notes, status, severity, mobile_sync_id,
                                   created_at, updated_at)
            VALUES (:id, :patient_id, :created_by_id, :facility_name,
                    :clinical_notes, :status, :severity, :mobile_sync_id,
                    :created_at, :updated_at)
        """), diagnoses_rows)
        db.commit()
        print(f"  ✓ {len(diagnoses_rows)} diagnoses created")

        # ── DIAGNOSTIC IMAGES ─────────────────────────────────────────────────
        print("Seeding diagnostic images...")

        image_ids = [uid() for _ in range(12)]
        image_rows = []
        for i in range(12):
            image_rows.append({
                "id": image_ids[i],
                "diagnosis_id": diagnosis_ids[i],
                "original_filename": f"blood_smear_{i+1:03d}.jpg",
                "storage_path": f"uploads/blood_smear_{i+1:03d}.jpg",
                "content_type": "image/jpeg",
                "file_size_bytes": random.randint(200_000, 900_000),
                "width_px": 1280,
                "height_px": 960,
                "magnification": "100x",
                "status": "done",
                "error_message": None,
                "created_at": diagnoses_rows[i]["created_at"],
                "updated_at": diagnoses_rows[i]["updated_at"],
            })

        db.execute(text("""
            INSERT INTO diagnostic_images (id, diagnosis_id, original_filename,
                storage_path, content_type, file_size_bytes, width_px, height_px,
                magnification, status, error_message, created_at, updated_at)
            VALUES (:id, :diagnosis_id, :original_filename, :storage_path,
                    :content_type, :file_size_bytes, :width_px, :height_px,
                    :magnification, :status, :error_message, :created_at, :updated_at)
        """), image_rows)
        db.commit()
        print(f"  ✓ {len(image_rows)} diagnostic images created")

        # ── DIAGNOSIS RESULTS ─────────────────────────────────────────────────
        print("Seeding diagnosis results...")

        result_data = [
            # (total_rbc, parasites, parasitaemia, ring, troph, schizont, gameto)
            (450, 0,   0.0,  0,  0,  0,  0),   # negative
            (480, 8,   1.7,  5,  2,  1,  0),   # low
            (420, 28,  6.7, 15,  8,  3,  2),   # moderate
            (390, 52, 13.3, 28, 16,  6,  2),   # high
            (350, 89, 25.4, 45, 28, 12,  4),   # severe
            (460, 0,   0.0,  0,  0,  0,  0),   # negative
            (440, 11,  2.5,  7,  3,  1,  0),   # low
            (410, 35,  8.5, 18, 12,  4,  1),   # moderate
            (380, 60, 15.8, 32, 19,  7,  2),   # high
            (330, 95, 28.8, 48, 30, 14,  3),   # severe
            (470, 0,   0.0,  0,  0,  0,  0),   # negative
            (455, 14,  3.1,  9,  4,  1,  0),   # low
        ]

        result_ids = [uid() for _ in range(12)]
        result_rows = []
        for i, (rbc, par, pct, ring, troph, schiz, gameto) in enumerate(result_data):
            result_rows.append({
                "id": result_ids[i],
                "diagnosis_id": diagnosis_ids[i],
                "image_id": image_ids[i],
                "total_rbc_count": rbc,
                "total_parasite_count": par,
                "parasitaemia_percent": pct,
                "ring_count": ring,
                "trophozoite_count": troph,
                "schizont_count": schiz,
                "gametocyte_count": gameto,
                "model_version": "YOLOv9-malaria-v1.2",
                "inference_time_ms": round(random.uniform(120, 280), 1),
                "raw_inference_output": json.dumps({"detections": par, "confidence": round(random.uniform(0.72, 0.97), 2)}),
                "created_at": diagnoses_rows[i]["created_at"],
                "updated_at": diagnoses_rows[i]["updated_at"],
            })

        db.execute(text("""
            INSERT INTO diagnosis_results (id, diagnosis_id, image_id, total_rbc_count,
                total_parasite_count, parasitaemia_percent, ring_count, trophozoite_count,
                schizont_count, gametocyte_count, model_version, inference_time_ms,
                raw_inference_output, created_at, updated_at)
            VALUES (:id, :diagnosis_id, :image_id, :total_rbc_count,
                    :total_parasite_count, :parasitaemia_percent, :ring_count,
                    :trophozoite_count, :schizont_count, :gametocyte_count,
                    :model_version, :inference_time_ms, :raw_inference_output,
                    :created_at, :updated_at)
        """), result_rows)
        db.commit()
        print(f"  ✓ {len(result_rows)} diagnosis results created")

        # ── PREDICTIONS ───────────────────────────────────────────────────────
        print("Seeding predictions...")

        pred_configs = [
            ("Ring Stage",      0.91, "mild",     "Prescribe artemisinin-based combination therapy (ACT). Monitor closely."),
            ("Trophozoite",     0.87, "moderate",  "Administer ACT immediately. Patient requires admission for monitoring."),
            ("Schizont",        0.83, "severe",    "URGENT: Severe malaria detected. IV artesunate required. ICU consideration."),
            ("Gametocyte",      0.78, "mild",      "Gametocytes present. Patient may be infectious. Add primaquine to regimen."),
            ("Negative",        0.95, "negative",  "No malaria parasites detected. Consider other fever causes if symptoms persist."),
            ("Ring Stage",      0.89, "mild",      "Early infection. Start ACT promptly to prevent progression."),
            ("Trophozoite",     0.84, "moderate",  "Active trophozoite infection. Begin ACT and monitor parasitaemia daily."),
            ("Schizont",        0.76, "high",      "High parasite burden. Immediate treatment required."),
            ("Negative",        0.97, "negative",  "Test result negative. Repeat in 24h if fever persists."),
            ("Ring Stage",      0.92, "mild",      "Ring stage malaria. Initiate first-line treatment as per protocol."),
            ("Trophozoite",     0.81, "moderate",  "Moderate infection. Full ACT course required."),
            ("Gametocyte",      0.74, "mild",      "Gametocyte stage only. Patient completing treatment — add gametocytocide."),
        ]

        all_user_ids = [admin_id, doctor1_id, doctor2_id, lab1_id, lab2_id, lab3_id]
        prediction_rows = []
        for i, (cls, conf, sev, rec) in enumerate(pred_configs):
            prediction_rows.append({
                "id": uid(),
                "user_id": random.choice(all_user_ids),
                "original_filename": f"smear_prediction_{i+1:03d}.jpg",
                "storage_path": f"uploads/smear_prediction_{i+1:03d}.jpg",
                "file_size_bytes": random.randint(150_000, 800_000),
                "content_type": "image/jpeg",
                "disease_type": "malaria",
                "status": "completed",
                "predicted_class": cls,
                "confidence_score": conf,
                "recommendation": rec,
                "severity_level": sev,
                "raw_output": json.dumps({
                    "predicted_class": cls,
                    "confidence": conf,
                    "all_classes": {
                        "Ring Stage": round(random.uniform(0.0, 0.4), 2),
                        "Trophozoite": round(random.uniform(0.0, 0.4), 2),
                        "Schizont": round(random.uniform(0.0, 0.3), 2),
                        "Gametocyte": round(random.uniform(0.0, 0.2), 2),
                        "Negative": round(random.uniform(0.0, 0.3), 2),
                    },
                    "bounding_boxes": [
                        {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "conf": conf, "class": cls}
                    ],
                }),
                "model_version": "YOLOv9-malaria-v1.2",
                "inference_time_ms": round(random.uniform(100, 260), 1),
                "error_message": None,
                "diagnosis_id": diagnosis_ids[i] if i < len(diagnosis_ids) else None,
                "created_at": ago(random.randint(0, 25)),
                "updated_at": ago(0),
            })

        db.execute(text("""
            INSERT INTO predictions (id, user_id, original_filename, storage_path,
                file_size_bytes, content_type, disease_type, status, predicted_class,
                confidence_score, recommendation, severity_level, raw_output,
                model_version, inference_time_ms, error_message, diagnosis_id,
                created_at, updated_at)
            VALUES (:id, :user_id, :original_filename, :storage_path,
                    :file_size_bytes, :content_type, :disease_type, :status,
                    :predicted_class, :confidence_score, :recommendation,
                    :severity_level, :raw_output, :model_version, :inference_time_ms,
                    :error_message, :diagnosis_id, :created_at, :updated_at)
        """), prediction_rows)
        db.commit()
        print(f"  ✓ {len(prediction_rows)} predictions created")

        print("\n✅ Database seeded successfully!\n")
        print("=" * 50)
        print("LOGIN CREDENTIALS")
        print("=" * 50)
        print("Admin:          admin@visiondx.com       / Admin@1234")
        print("Doctor 1:       dr.uwimana@visiondx.com  / Doctor@1234")
        print("Doctor 2:       dr.mukamana@visiondx.com / Doctor@1234")
        print("Lab Tech 1:     lab.niyonzima@visiondx.com / Lab@12345")
        print("Lab Tech 2:     lab.ingabire@visiondx.com  / Lab@12345")
        print("Lab Tech 3:     lab.habimana@visiondx.com  / Lab@12345")
        print("=" * 50)
        print(f"Patients: {len(patients_rows)} | Diagnoses: {len(diagnoses_rows)} | Predictions: {len(prediction_rows)}")


if __name__ == "__main__":
    seed()
