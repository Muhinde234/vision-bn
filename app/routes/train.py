"""
POST /train        — start YOLO training in background
GET  /train/status — poll training progress
POST /prepare-dataset — convert JSON → YOLO layout (optional pre-step)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import train as train_svc
from app.services.data_loader import prepare_dataset

router = APIRouter(prefix="/train", tags=["Malaria · Training"])


# ── Request schema ────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int   = Field(50,    ge=1,   le=500,  description="Training epochs")
    batch:  int   = Field(16,    ge=1,   le=128,  description="Batch size")
    imgsz:  int   = Field(640,   ge=320, le=1280, description="Input image size (pixels)")
    device: str   = Field("cpu",                  description="'cpu', '0', '0,1', 'mps' …")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Start model training",
    description=(
        "Launches YOLOv8 training in a background thread. "
        "Poll **GET /train/status** to check progress. "
        "Returns 409 if training is already running."
    ),
)
async def start_training(payload: TrainRequest):
    try:
        train_svc.start_training(
            epochs = payload.epochs,
            batch  = payload.batch,
            imgsz  = payload.imgsz,
            device = payload.device,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "message": "Training started",
        "config":  payload.model_dump(),
        "tip":     "Poll GET /train/status to check progress",
    }


@router.get(
    "/status",
    summary="Get training status",
)
async def training_status():
    return train_svc.get_status()


@router.post(
    "/prepare-dataset",
    summary="Convert malaria JSON → YOLO dataset layout",
    description=(
        "Reads malaria/training.json and malaria/test.json, copies images, "
        "writes YOLO label .txt files and data.yaml. "
        "This step runs automatically when you POST /train, "
        "but you can call it manually first to verify your data."
    ),
)
async def prepare(force: bool = False):
    try:
        result = prepare_dataset(force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
