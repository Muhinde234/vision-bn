"""
POST /predict              — detect malaria parasites in an uploaded image
POST /predict/export/onnx  — export trained model to ONNX
POST /predict/export/tflite — export trained model to TFLite (mobile)
"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services import inference as inference_svc

router = APIRouter(prefix="/predict", tags=["Malaria · Prediction"])

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Detect malaria parasites",
    description=(
        "Upload a blood smear image (JPEG/PNG/TIFF). "
        "Returns bounding boxes, class names and confidence scores "
        "for every detected object."
    ),
    response_description="List of detections with class, confidence, and bbox.",
)
async def predict(
    file:       UploadFile = File(...,  description="Blood smear image"),
    confidence: float      = Form(0.35, ge=0.01, le=1.0,
                                  description="Minimum detection confidence (0–1)"),
):
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(_ALLOWED_TYPES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        detections = inference_svc.predict(data, confidence=confidence)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # Summarise parasite classes (exclude healthy RBCs and leukocytes for clarity)
    _PARASITE_CLASSES = {"trophozoite", "ring", "schizont", "gametocyte"}
    parasite_hits = [d for d in detections if d["class"] in _PARASITE_CLASSES]

    return {
        "detections": detections,
        "total_objects": len(detections),
        "parasite_detections": len(parasite_hits),
        "infected": len(parasite_hits) > 0,
    }


@router.post(
    "/export/onnx",
    summary="Export model to ONNX",
    description="Exports models/best.pt to ONNX format for cross-platform deployment.",
)
async def export_onnx(dynamic: bool = True):
    try:
        path = inference_svc.export_onnx(dynamic=dynamic)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"message": "ONNX export complete", "path": path}


@router.post(
    "/export/tflite",
    summary="Export model to TensorFlow Lite (mobile)",
    description=(
        "Exports models/best.pt to TFLite for Android / iOS / edge deployment. "
        "Set int8=true for quantised model (smaller, slightly less accurate)."
    ),
)
async def export_tflite(int8: bool = False):
    try:
        path = inference_svc.export_tflite(int8=int8)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"message": "TFLite export complete", "path": path}
