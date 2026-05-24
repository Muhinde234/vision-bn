"""
Malaria inference service.

Loads models/best.pt (lazy, once) and runs YOLOv8 detection on image bytes.
Also exposes ONNX and TFLite export helpers.
"""
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image as PILImage

from app.utils.yolo_converter import CLASS_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR   = _BACKEND_DIR / "models"
MODEL_PATH   = MODELS_DIR / "best.pt"

# ── Lazy model singleton ──────────────────────────────────────────────────────
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "POST /train to train the model first."
        )
    from ultralytics import YOLO
    _model = YOLO(str(MODEL_PATH))
    return _model


def _reload_model() -> None:
    """Force reload — call this after a new training run completes."""
    global _model
    _model = None
    _load_model()


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes, confidence: float = 0.35) -> List[dict]:
    """
    Run malaria detection on raw image bytes.

    Args:
        image_bytes: Raw bytes of a JPEG / PNG / TIFF image.
        confidence:  Minimum confidence threshold (0–1).

    Returns:
        List of detection dicts::

            {
                "class":      "trophozoite",
                "confidence": 0.87,
                "bbox":       [x, y, w, h]   # pixel coords, top-left origin
            }
    """
    model = _load_model()

    img       = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    results = model.predict(
        source  = img_array,
        conf    = confidence,
        verbose = False,
    )

    detections: List[dict] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = (
                CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            )
            detections.append({
                "class":      class_name,
                "confidence": round(conf_score, 4),
                "bbox":       [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
            })

    return detections


# ── Export helpers ────────────────────────────────────────────────────────────

def export_onnx(dynamic: bool = True) -> str:
    """
    Export best.pt → ONNX.

    Args:
        dynamic: Enable dynamic input shapes (recommended for deployment).

    Returns:
        Absolute path to the exported .onnx file.
    """
    model  = _load_model()
    output = model.export(format="onnx", dynamic=dynamic, simplify=True)
    return str(output)


def export_tflite(int8: bool = False) -> str:
    """
    Export best.pt → TensorFlow Lite.

    Args:
        int8: Enable INT8 quantisation (smaller model, slight accuracy drop).

    Returns:
        Absolute path to the exported .tflite file.
    """
    model  = _load_model()
    output = model.export(format="tflite", int8=int8)
    return str(output)
