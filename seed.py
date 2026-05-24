"""
Database seeder — populates VisionDx with 90 days of realistic analytics data.

Usage:
    python seed.py            # skip if data already exists
    python seed.py --force    # wipe and reseed
"""
import json
import os
import random
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

import bcrypt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from dotenv import load_dotenv
load_dotenv()

SYNC_DATABASE_URL = os.environ.get("SYNC_DATABASE_URL", "sqlite:///./visiondx_dev.db")
engine = create_engine(SYNC_DATABASE_URL, echo=False)

RNG = random.Random(42)   # fixed seed → reproducible data


def hash_pw(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def uid() -> str:
    return str(uuid.uuid4())


def ago(days: float, hour: int = 9) -> datetime:
    """Return a UTC datetime `days` ago at the given hour."""
    base = datetime.now(timezone.utc) - timedelta(days=days)
    return base.replace(hour=hour, minute=RNG.randint(0, 59), second=0, microsecond=0)


# ── Severity distribution weights (realistic Rwanda malaria burden) ───────────
# 38% negative, 28% low, 20% moderate, 9% high, 5% severe
_SEVERITY_WEIGHTS = ["negative"] * 38 + ["low"] * 28 + ["moderate"] * 20 + ["high"] * 9 + ["severe"] * 5

_PARASITAEMIA = {
    "negative": (0,    0,    0,   0,  0,  0,  0),
    "low":      (0.3,  1.8,  0.6, 0.2, 0, 0,  0),     # (pct_min, pct_max, ring_frac, troph_frac, schiz_frac, gameto_frac, ...)
    "moderate": (1.0,  5.0,  0.5, 0.3, 0.15, 0.05, 0),
    "high":     (5.0, 15.0,  0.45, 0.32, 0.18, 0.05, 0),
    "severe":   (15.0, 35.0, 0.42, 0.30, 0.20, 0.08, 0),
}

_STAGE_CLASS = {
    "negative": "Negative",
    "low":      "Ring Stage",
    "moderate": "Trophozoite",
    "high":     "Schizont",
    "severe":   "Schizont",
}

_SEVERITY_CONF = {
    "negative": (0.90, 0.98),
    "low":      (0.82, 0.94),
    "moderate": (0.79, 0.92),
    "high":     (0.74, 0.90),
    "severe":   (0.71, 0.88),
}

_RECOMMENDATIONS = {
    "Negative":    "No malaria parasites detected. Consider other fever causes if symptoms persist.",
    "Ring Stage":  "Early ring-stage infection. Initiate ACT immediately. Monitor parasitaemia every 24 hours.",
    "Trophozoite": "Active trophozoite infection. Begin ACT without delay. Assess for severity markers.",
    "Schizont":    "High-density schizont infection. Urgent parenteral antimalarial treatment indicated. Hospitalise.",
    "Gametocyte":  "Gametocytes detected — patient is infectious. Complete ACT course. Consider primaquine.",
}

_CLINICAL_NOTES = [
    "Patient presents with fever (38.9 °C), chills and headache for 3 days.",
    "Routine malaria screening. Patient returned from high-risk zone last week.",
    "Fever and fatigue for 2 days. Suspected uncomplicated malaria.",
    "Follow-up test after initial positive. Patient currently on artemether-lumefantrine.",
    "Pregnant patient — routine antenatal malaria screening (ANC visit 2).",
    "Child with high fever and vomiting. Urgent blood smear requested.",
    "Community health worker referral. Positive RDT — microscopy confirmation needed.",
    "Post-treatment day 3 check. Patient improving clinically.",
    "Traveller returning from forest region. Fever onset 48 h ago.",
    "School-age child. Pallor noted. Screening for malaria and anaemia.",
]


def _parasite_counts(severity: str, total_rbc: int) -> dict:
    if severity == "negative":
        return dict(pct=0.0, ring=0, troph=0, schiz=0, gameto=0, total_par=0)
    pct_min, pct_max = _PARASITAEMIA[severity][0], _PARASITAEMIA[severity][1]
    ring_f, troph_f, schiz_f, gameto_f = (
        _PARASITAEMIA[severity][2],
        _PARASITAEMIA[severity][3],
        _PARASITAEMIA[severity][4],
        _PARASITAEMIA[severity][5],
    )
    pct = round(RNG.uniform(pct_min, pct_max), 4)
    total_par = max(1, round(total_rbc * pct / 100))
    ring   = round(total_par * ring_f)
    troph  = round(total_par * troph_f)
    schiz  = round(total_par * schiz_f)
    gameto = total_par - ring - troph - schiz
    gameto = max(0, gameto)
    return dict(pct=pct, ring=ring, troph=troph, schiz=schiz, gameto=gameto, total_par=total_par)


def seed():
    import app.models  # noqa — register models
    from app.db.base import Base

    print("Creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        existing = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
        if existing and existing > 0:
            print(f"Database already has {existing} users.")
            if "--force" not in sys.argv:
                print("To reseed, run:  python seed.py --force")
                return
            print("Force flag — clearing all tables...")
            for tbl in ["detections", "diagnosis_results", "diagnostic_images",
                        "predictions", "diagnoses", "patients", "refresh_tokens", "users"]:
                db.execute(text(f"DELETE FROM {tbl}"))
            db.commit()
            print("Tables cleared.")

        # ── USERS ─────────────────────────────────────────────────────────────
        print("Seeding users...")
        admin_id   = uid()
        doctor1_id = uid()
        doctor2_id = uid()
        lab1_id    = uid()
        lab2_id    = uid()
        lab3_id    = uid()

        facilities = [
            "Kigali University Hospital",
            "Butare General Hospital",
            "Ruhengeri District Hospital",
        ]

        users = [
            dict(id=admin_id,   email="admin@visiondx.com",
                 full_name="System Administrator",       hashed_password=hash_pw("Admin@1234"),
                 role="admin",          facility_name="VisionDx HQ",
                 is_active=True, is_verified=True, created_at=ago(90), updated_at=ago(90)),
            dict(id=doctor1_id, email="dr.uwimana@visiondx.com",
                 full_name="Dr. Jean Uwimana",           hashed_password=hash_pw("Doctor@1234"),
                 role="doctor",         facility_name=facilities[0],
                 is_active=True, is_verified=True, created_at=ago(80), updated_at=ago(80)),
            dict(id=doctor2_id, email="dr.mukamana@visiondx.com",
                 full_name="Dr. Alice Mukamana",         hashed_password=hash_pw("Doctor@1234"),
                 role="doctor",         facility_name=facilities[1],
                 is_active=True, is_verified=True, created_at=ago(75), updated_at=ago(75)),
            dict(id=lab1_id,    email="lab.niyonzima@visiondx.com",
                 full_name="Eric Niyonzima",             hashed_password=hash_pw("Lab@12345"),
                 role="lab_technician", facility_name=facilities[0],
                 is_active=True, is_verified=True, created_at=ago(70), updated_at=ago(70)),
            dict(id=lab2_id,    email="lab.ingabire@visiondx.com",
                 full_name="Grace Ingabire",             hashed_password=hash_pw("Lab@12345"),
                 role="lab_technician", facility_name=facilities[1],
                 is_active=True, is_verified=True, created_at=ago(65), updated_at=ago(65)),
            dict(id=lab3_id,    email="lab.habimana@visiondx.com",
                 full_name="Patrick Habimana",           hashed_password=hash_pw("Lab@12345"),
                 role="lab_technician", facility_name=facilities[2],
                 is_active=True, is_verified=True, created_at=ago(60), updated_at=ago(60)),
        ]

        db.execute(text("""
            INSERT INTO users (id, email, full_name, hashed_password, role,
                               facility_name, is_active, is_verified, created_at, updated_at)
            VALUES (:id, :email, :full_name, :hashed_password, :role,
                    :facility_name, :is_active, :is_verified, :created_at, :updated_at)
        """), users)
        db.commit()
        print(f"  [OK] {len(users)} users")

        # ── PATIENTS ──────────────────────────────────────────────────────────
        print("Seeding patients...")
        patient_pool = [
            ("Mugisha Emmanuel",   "1990-03-15", "male",   "+250788100001", "Kigali, Nyarugenge",  facilities[0]),
            ("Uwase Diane",        "1985-07-22", "female", "+250788100002", "Kigali, Gasabo",      facilities[0]),
            ("Nkurunziza Pascal",  "2000-11-08", "male",   "+250788100003", "Butare, Huye",        facilities[1]),
            ("Kampire Solange",    "1978-05-30", "female", "+250788100004", "Butare, Huye",        facilities[1]),
            ("Hakizimana David",   "1995-01-17", "male",   "+250788100005", "Ruhengeri, Musanze",  facilities[2]),
            ("Mukamurenzi Rose",   "1988-09-04", "female", "+250788100006", "Ruhengeri, Musanze",  facilities[2]),
            ("Ntirenganya Yves",   "2003-06-25", "male",   "+250788100007", "Kigali, Kicukiro",    facilities[0]),
            ("Ingabire Clarisse",  "1992-12-11", "female", "+250788100008", "Butare, Nyanza",      facilities[1]),
            ("Bizimana Jean",      "1970-04-03", "male",   "+250788100009", "Ruhengeri, Burera",   facilities[2]),
            ("Mukandori Yvonne",   "1999-08-19", "female", "+250788100010", "Kigali, Nyarugenge",  facilities[0]),
            ("Ndayishimiye Eric",  "2005-02-28", "male",   "+250788100011", "Gitarama, Muhanga",   facilities[1]),
            ("Uwimana Christine",  "1983-10-07", "female", "+250788100012", "Byumba, Gicumbi",     facilities[2]),
            ("Kamanzi Olivier",    "1996-05-14", "male",   "+250788100013", "Kigali, Gasabo",      facilities[0]),
            ("Nyiransabimana Lea", "2001-09-30", "female", "+250788100014", "Butare, Huye",        facilities[1]),
            ("Habimana Samuel",    "1975-11-22", "male",   "+250788100015", "Ruhengeri, Burera",   facilities[2]),
            ("Mukeshimana Aline",  "1993-04-18", "female", "+250788100016", "Kigali, Kicukiro",    facilities[0]),
            ("Nzabonimpa Felix",   "1987-08-07", "male",   "+250788100017", "Butare, Nyanza",      facilities[1]),
            ("Uwineza Marie",      "2002-01-25", "female", "+250788100018", "Ruhengeri, Musanze",  facilities[2]),
            ("Tuyishime Robert",   "1980-06-12", "male",   "+250788100019", "Kigali, Nyarugenge",  facilities[0]),
            ("Gasana Jacqueline",  "1997-03-03", "female", "+250788100020", "Byumba, Gicumbi",     facilities[2]),
        ]
        patient_ids = [uid() for _ in patient_pool]
        patient_rows = []
        for i, (name, dob, sex, phone, address, fac) in enumerate(patient_pool):
            patient_rows.append(dict(
                id=patient_ids[i], patient_code=f"VDX-2026-{str(i+1).zfill(4)}",
                full_name=name, date_of_birth=dob, sex=sex, phone=phone,
                address=address, facility_name=fac, notes=None,
                created_at=ago(RNG.randint(10, 85)), updated_at=ago(RNG.randint(1, 9)),
            ))
        db.execute(text("""
            INSERT INTO patients (id, patient_code, full_name, date_of_birth, sex,
                                  phone, address, facility_name, notes, created_at, updated_at)
            VALUES (:id, :patient_code, :full_name, :date_of_birth, :sex,
                    :phone, :address, :facility_name, :notes, :created_at, :updated_at)
        """), patient_rows)
        db.commit()
        print(f"  [OK] {len(patient_rows)} patients")

        # ── DIAGNOSES + RESULTS (90-day daily data) ───────────────────────────
        print("Seeding 90 days of diagnoses and results...")

        lab_user_ids = [lab1_id, lab2_id, lab3_id]
        facility_lab = {
            lab1_id: facilities[0],
            lab2_id: facilities[1],
            lab3_id: facilities[2],
        }

        diagnosis_rows   = []
        image_rows       = []
        result_rows      = []
        prediction_rows  = []

        # Generate 3–6 diagnoses per day over 90 days
        for day in range(90, 0, -1):
            n_today = RNG.randint(3, 6)
            # Slightly higher case load in rainy season simulation (days 60-90 ago)
            if day > 60:
                n_today = RNG.randint(4, 7)

            for _ in range(n_today):
                hour    = RNG.randint(7, 17)
                ts      = ago(day, hour=hour)
                lab_uid = RNG.choice(lab_user_ids)
                fac     = facility_lab[lab_uid]
                pat_id  = RNG.choice([p for p, (_, _, _, _, _, f) in zip(patient_ids, patient_pool) if f == fac] or patient_ids)
                severity = RNG.choice(_SEVERITY_WEIGHTS)
                d_status = "reviewed" if RNG.random() < 0.15 else "completed"

                diag_id  = uid()
                image_id = uid()
                result_id = uid()

                counts = _parasite_counts(severity, RNG.randint(350, 500))
                total_rbc = RNG.randint(350, 500)

                # ── diagnosis ────────────────────────────────────────────────
                diagnosis_rows.append(dict(
                    id=diag_id, patient_id=pat_id, created_by_id=lab_uid,
                    facility_name=fac,
                    clinical_notes=RNG.choice(_CLINICAL_NOTES),
                    status=d_status, severity=severity,
                    mobile_sync_id=None,
                    created_at=ts, updated_at=ts,
                ))

                # ── diagnostic image ─────────────────────────────────────────
                image_rows.append(dict(
                    id=image_id, diagnosis_id=diag_id,
                    original_filename=f"smear_{diag_id[:8]}.jpg",
                    storage_path=f"uploads/smear_{diag_id[:8]}.jpg",
                    content_type="image/jpeg",
                    file_size_bytes=RNG.randint(180_000, 920_000),
                    width_px=1280, height_px=960,
                    magnification="100x", status="done",
                    error_message=None,
                    created_at=ts, updated_at=ts,
                ))

                # ── diagnosis result ─────────────────────────────────────────
                result_rows.append(dict(
                    id=result_id, diagnosis_id=diag_id, image_id=image_id,
                    total_rbc_count=total_rbc,
                    total_parasite_count=counts["total_par"],
                    parasitaemia_percent=counts["pct"],
                    ring_count=counts["ring"],
                    trophozoite_count=counts["troph"],
                    schizont_count=counts["schiz"],
                    gametocyte_count=counts["gameto"],
                    model_version="YOLOv9-malaria-v1.2",
                    inference_time_ms=round(RNG.uniform(115, 275), 1),
                    raw_inference_output=json.dumps({
                        "detections": counts["total_par"],
                        "confidence": round(RNG.uniform(0.72, 0.97), 2),
                    }),
                    created_at=ts, updated_at=ts,
                ))

                # ── standalone prediction (mirrors diagnosis) ─────────────────
                cls  = _STAGE_CLASS[severity]
                conf = round(RNG.uniform(*_SEVERITY_CONF[severity]), 2)
                prediction_rows.append(dict(
                    id=uid(),
                    user_id=lab_uid,
                    original_filename=f"smear_{diag_id[:8]}.jpg",
                    storage_path=f"uploads/smear_{diag_id[:8]}.jpg",
                    file_size_bytes=RNG.randint(150_000, 900_000),
                    content_type="image/jpeg",
                    disease_type="malaria",
                    status="completed",
                    predicted_class=cls,
                    confidence_score=conf,
                    recommendation=_RECOMMENDATIONS.get(cls, "Consult a specialist."),
                    severity_level=severity,
                    raw_output=json.dumps({
                        "predicted_class": cls, "confidence": conf,
                        "bounding_boxes": [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
                                            "conf": conf, "class": cls}],
                    }),
                    model_version="YOLOv9-malaria-v1.2",
                    inference_time_ms=round(RNG.uniform(100, 265), 1),
                    error_message=None,
                    diagnosis_id=diag_id,
                    created_at=ts, updated_at=ts,
                ))

        # ── batch insert in chunks of 100 ─────────────────────────────────────
        def chunked(lst, n=100):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for chunk in chunked(diagnosis_rows):
            db.execute(text("""
                INSERT INTO diagnoses (id, patient_id, created_by_id, facility_name,
                                       clinical_notes, status, severity, mobile_sync_id,
                                       created_at, updated_at)
                VALUES (:id, :patient_id, :created_by_id, :facility_name,
                        :clinical_notes, :status, :severity, :mobile_sync_id,
                        :created_at, :updated_at)
            """), chunk)
        db.commit()
        print(f"  [OK] {len(diagnosis_rows)} diagnoses")

        for chunk in chunked(image_rows):
            db.execute(text("""
                INSERT INTO diagnostic_images (id, diagnosis_id, original_filename,
                    storage_path, content_type, file_size_bytes, width_px, height_px,
                    magnification, status, error_message, created_at, updated_at)
                VALUES (:id, :diagnosis_id, :original_filename, :storage_path,
                        :content_type, :file_size_bytes, :width_px, :height_px,
                        :magnification, :status, :error_message, :created_at, :updated_at)
            """), chunk)
        db.commit()
        print(f"  [OK] {len(image_rows)} diagnostic images")

        for chunk in chunked(result_rows):
            db.execute(text("""
                INSERT INTO diagnosis_results (id, diagnosis_id, image_id,
                    total_rbc_count, total_parasite_count, parasitaemia_percent,
                    ring_count, trophozoite_count, schizont_count, gametocyte_count,
                    model_version, inference_time_ms, raw_inference_output,
                    created_at, updated_at)
                VALUES (:id, :diagnosis_id, :image_id,
                        :total_rbc_count, :total_parasite_count, :parasitaemia_percent,
                        :ring_count, :trophozoite_count, :schizont_count, :gametocyte_count,
                        :model_version, :inference_time_ms, :raw_inference_output,
                        :created_at, :updated_at)
            """), chunk)
        db.commit()
        print(f"  [OK] {len(result_rows)} diagnosis results")

        for chunk in chunked(prediction_rows):
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
            """), chunk)
        db.commit()
        print(f"  [OK] {len(prediction_rows)} predictions")

        # ── Summary ───────────────────────────────────────────────────────────
        pos = sum(1 for d in diagnosis_rows if d["severity"] != "negative")
        neg = len(diagnosis_rows) - pos
        positivity_rate = round(pos / len(diagnosis_rows) * 100, 1)

        print(f"""
[OK] Seeded successfully!

{'='*52}
  LOGIN CREDENTIALS
{'='*52}
  Admin      admin@visiondx.com          Admin@1234
  Doctor 1   dr.uwimana@visiondx.com     Doctor@1234
  Doctor 2   dr.mukamana@visiondx.com    Doctor@1234
  Lab Tech 1 lab.niyonzima@visiondx.com  Lab@12345
  Lab Tech 2 lab.ingabire@visiondx.com   Lab@12345
  Lab Tech 3 lab.habimana@visiondx.com   Lab@12345
{'='*52}
  ANALYTICS SUMMARY (90-day window)
{'='*52}
  Patients    : {len(patient_rows)}
  Diagnoses   : {len(diagnosis_rows)}
  Positive    : {pos}  ({positivity_rate}% positivity rate)
  Negative    : {neg}
  Facilities  : {len(facilities)}
  Date range  : {(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')} to today
{'='*52}
""")


if __name__ == "__main__":
    seed()
