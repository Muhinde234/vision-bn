"""
Malaria YOLO model training service.

Trains YOLOv8n on the prepared malaria dataset.
Training runs in a daemon thread so the API stays responsive.
Best weights are saved to  models/best.pt
"""
import shutil
import threading
from pathlib import Path
from typing import Any, Dict

from app.services.data_loader import YOLO_DIR, prepare_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR   = _BACKEND_DIR / "models"
MODEL_PATH   = MODELS_DIR / "best.pt"

# Pretrained backbone — downloaded automatically by ultralytics on first run
_PRETRAINED  = "yolov8n.pt"

# ── Shared training state (read from any request handler) ─────────────────────
_lock   = threading.Lock()
_status: Dict[str, Any] = {
    "running":   False,
    "done":      False,
    "error":     None,
    "metrics":   {},
    "model_path": None,
}


def get_status() -> Dict[str, Any]:
    with _lock:
        return dict(_status)


# ── Core training routine (runs in background thread) ─────────────────────────

def _train(epochs: int, batch: int, imgsz: int, device: str) -> None:
    with _lock:
        _status.update({"running": True, "done": False, "error": None, "metrics": {}})

    try:
        from ultralytics import YOLO

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare YOLO dataset layout (idempotent)
        prepare_dataset()

        yaml_path = YOLO_DIR / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {yaml_path}. "
                                    "Call /prepare-dataset first.")

        model = YOLO(_PRETRAINED)

        results = model.train(
            data        = str(yaml_path),
            epochs      = epochs,
            batch       = batch,
            imgsz       = imgsz,
            device      = device,
            project     = str(MODELS_DIR / "runs"),
            name        = "malaria",
            exist_ok    = True,
            patience    = 20,          # early stopping
            save_period = 10,          # checkpoint every 10 epochs
            # ── augmentation ──────────────────────────────────────────────────
            hsv_h       = 0.015,
            hsv_s       = 0.7,
            hsv_v       = 0.4,
            degrees     = 10.0,
            translate   = 0.1,
            scale       = 0.5,
            flipud      = 0.5,
            fliplr      = 0.5,
            mosaic      = 1.0,
            mixup       = 0.1,
            copy_paste  = 0.1,
        )

        # ── copy best weights ──────────────────────────────────────────────────
        best_src = Path(results.save_dir) / "weights" / "best.pt"
        if best_src.exists():
            shutil.copy2(best_src, MODEL_PATH)

        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = {k: round(float(v), 4) for k, v in results.results_dict.items()
                       if isinstance(v, (int, float))}

        with _lock:
            _status.update({
                "running":    False,
                "done":       True,
                "metrics":    metrics,
                "model_path": str(MODEL_PATH),
            })

    except Exception as exc:
        with _lock:
            _status.update({"running": False, "done": False, "error": str(exc)})
        raise


# ── Public API ────────────────────────────────────────────────────────────────

def start_training(
    epochs: int = 50,
    batch:  int = 16,
    imgsz:  int = 640,
    device: str = "cpu",
) -> None:
    """
    Launch training in a background thread.
    Raises RuntimeError if training is already running.
    """
    with _lock:
        if _status["running"]:
            raise RuntimeError("Training is already in progress.")

    thread = threading.Thread(
        target  = _train,
        args    = (epochs, batch, imgsz, device),
        daemon  = True,
        name    = "malaria-trainer",
    )
    thread.start()
