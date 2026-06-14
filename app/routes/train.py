"""
POST /train        — start YOLO training in background
GET  /train/status — poll training progress
POST /prepare-dataset — convert JSON → YOLO layout (optional pre-step)
"""
import io
import zipfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.services import train as train_svc
from app.services.data_loader import prepare_dataset
from app.services.train import MODELS_DIR

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


@router.get(
    "/plots/{filename}",
    summary="Get training plot images",
    description="Retrieve YOLO output plots like results.png, confusion_matrix.png, PR_curve.png.",
)
async def get_training_plot(filename: str):
    allowed_plots = {
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png"
    }
    
    if filename not in allowed_plots:
        raise HTTPException(status_code=400, detail="Invalid plot filename.")

    plot_path = MODELS_DIR / "runs" / "malaria" / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot not found: {filename}. Ensure training has run.")
        
    return FileResponse(plot_path)


@router.get(
    "/plots/download/all",
    summary="Download all training plots as a ZIP",
    description="Zips all YOLO generated PNG plots and returns them in a single file.",
)
async def download_all_plots():
    run_dir = MODELS_DIR / "runs" / "malaria"
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Training run directory not found.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        has_files = False
        for file_path in run_dir.glob("*.png"):
            zip_file.write(file_path, arcname=file_path.name)
            has_files = True
            
    if not has_files:
        raise HTTPException(status_code=404, detail="No plot images found.")
        
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=malaria_training_plots.zip"}
    )
