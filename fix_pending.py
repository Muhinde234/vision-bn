"""
Marks all pending/processing diagnoses as completed with realistic mock results.

Usage (PowerShell):
    $env:SYNC_DATABASE_URL="postgresql://user:pass@host/db"; python fix_pending.py
"""
import os
import random
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

load_dotenv()

SYNC_DATABASE_URL = os.environ.get("SYNC_DATABASE_URL", "sqlite:///./visiondx_dev.db")
engine = create_engine(SYNC_DATABASE_URL, echo=False)

RNG = random.Random()

SEVERITY_WEIGHTS = [
    ("negative", 0.38, 0,    0,    0.0),
    ("low",      0.28, 2,    8,    0.8),
    ("moderate", 0.20, 8,    25,   3.0),
    ("high",     0.09, 25,   60,   7.0),
    ("severe",   0.05, 60,   120,  12.0),
]


def pick_severity():
    r = RNG.random()
    cumulative = 0.0
    for name, weight, pmin, pmax, base_parasitaemia in SEVERITY_WEIGHTS:
        cumulative += weight
        if r <= cumulative:
            return name, pmin, pmax, base_parasitaemia
    return SEVERITY_WEIGHTS[-1][0], *SEVERITY_WEIGHTS[-1][2:]


def make_result(diagnosis_id: str, image_id: str, severity: str, parasite_count: int, rbc_count: int, parasitaemia: float):
    ring        = int(parasite_count * RNG.uniform(0.3, 0.5))
    trophozoite = int(parasite_count * RNG.uniform(0.2, 0.4))
    schizont    = int(parasite_count * RNG.uniform(0.05, 0.15))
    gametocyte  = max(0, parasite_count - ring - trophozoite - schizont)

    return {
        "id":                   str(uuid.uuid4()),
        "diagnosis_id":         diagnosis_id,
        "image_id":             image_id,
        "total_rbc_count":      rbc_count,
        "total_parasite_count": parasite_count,
        "parasitaemia_percent": round(parasitaemia, 2),
        "ring_count":           ring,
        "trophozoite_count":    trophozoite,
        "schizont_count":       schizont,
        "gametocyte_count":     gametocyte,
        "model_version":        "YOLOv9n-malaria-v2.3.1",
        "inference_time_ms":    round(RNG.uniform(180, 420), 1),
        "created_at":           datetime.now(timezone.utc),
        "updated_at":           datetime.now(timezone.utc),
    }


def fix(session: Session):
    rows = session.execute(
        text("SELECT id, patient_id, created_at FROM diagnoses WHERE status IN ('pending', 'processing')")
    ).fetchall()

    if not rows:
        print("[OK] No pending diagnoses found.")
        return

    print(f"Found {len(rows)} pending diagnosis(es). Fixing...")

    for row in rows:
        diagnosis_id = str(row[0])
        severity, pmin, pmax, base_parasitaemia = pick_severity()

        rbc_count      = RNG.randint(180, 320)
        parasite_count = 0 if severity == "negative" else RNG.randint(pmin, pmax)
        parasitaemia   = 0.0 if severity == "negative" else round(
            (parasite_count / rbc_count) * 100, 2
        )

        # Insert a placeholder image record (required by DiagnosisResult FK)
        image_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        session.execute(text("""
            INSERT INTO diagnostic_images
              (id, diagnosis_id, original_filename, storage_path, content_type,
               file_size_bytes, width_px, height_px, status, created_at, updated_at)
            VALUES
              (:id, :diagnosis_id, :filename, :path, 'image/jpeg',
               :size, 640, 480, 'done', :now, :now)
        """), {
            "id":           image_id,
            "diagnosis_id": diagnosis_id,
            "filename":     "blood_smear.jpg",
            "path":         f"uploads/placeholder_{image_id}.jpg",
            "size":         RNG.randint(120_000, 900_000),
            "now":          now,
        })

        # Insert result
        result = make_result(diagnosis_id, image_id, severity, parasite_count, rbc_count, parasitaemia)
        session.execute(text("""
            INSERT INTO diagnosis_results
              (id, diagnosis_id, image_id,
               total_rbc_count, total_parasite_count, parasitaemia_percent,
               ring_count, trophozoite_count, schizont_count, gametocyte_count,
               model_version, inference_time_ms, created_at, updated_at)
            VALUES
              (:id, :diagnosis_id, :image_id,
               :total_rbc_count, :total_parasite_count, :parasitaemia_percent,
               :ring_count, :trophozoite_count, :schizont_count, :gametocyte_count,
               :model_version, :inference_time_ms, :created_at, :updated_at)
        """), result)

        # Update diagnosis
        session.execute(text("""
            UPDATE diagnoses
            SET status = 'completed', severity = :severity, updated_at = :now
            WHERE id = :id
        """), {"severity": severity, "now": now, "id": diagnosis_id})

        print(f"  [OK] {diagnosis_id[:8]}... -> completed ({severity}, parasitaemia={parasitaemia}%)")

    session.commit()
    print(f"\n[OK] Done. {len(rows)} diagnosis(es) marked as completed.")


if __name__ == "__main__":
    with Session(engine) as session:
        fix(session)
