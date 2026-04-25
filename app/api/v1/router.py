from fastapi import APIRouter

from app.api.v1 import auth, users, patients, diagnoses, images, analytics, sync, predictions

api_router = APIRouter()

# ── Frontend-facing endpoints ─────────────────────────────────────────────────
api_router.include_router(auth.router,         prefix="/auth",        tags=["Authentication"])
api_router.include_router(predictions.router,  prefix="/predictions", tags=["AI Predictions"])

# ── Clinical / admin endpoints ────────────────────────────────────────────────
api_router.include_router(users.router,      prefix="/users",      tags=["Users"])
api_router.include_router(patients.router,   prefix="/patients",   tags=["Patients"])
api_router.include_router(diagnoses.router,  prefix="/diagnoses",  tags=["Diagnoses"])
api_router.include_router(images.router,     prefix="/images",     tags=["Images"])
api_router.include_router(analytics.router,  prefix="/analytics",  tags=["Analytics"])
api_router.include_router(sync.router,       prefix="/sync",       tags=["Mobile Sync"])
