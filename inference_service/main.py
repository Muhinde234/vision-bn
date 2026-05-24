"""
VisionDx Inference Microservice
Wraps YOLOv9 malaria detection model as a FastAPI endpoint.

POST /infer
  Body: { image_url, confidence_threshold, diagnosis_id }
  Returns: { model_version, inference_time_ms, image_width, image_height,
             total_rbc_count, detections }
"""
import io
import time
from typing import List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="VisionDx Inference Service", version="1.0.0")

# ── Model loading ─────────────────────────────────────────────────────────────
# Place your trained YOLOv9 weights at models/malaria_yolov9.pt
MODEL_PATH = "models/malaria_yolov9.pt"
MODEL_VERSION = "yolov9-malaria-v1.0"

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


# ── Classes (must match your training label map) ──────────────────────────────
CLASS_MAP = {0: "ring", 1: "trophozoite", 2: "schizont", 3: "gametocyte"}
RBC_CLASS_ID = 4  # if you trained RBC detection as class 4


# ── Schemas ───────────────────────────────────────────────────────────────────
class InferRequest(BaseModel):
    image_url: str
    confidence_threshold: float = 0.35
    diagnosis_id: str


class DetectionOut(BaseModel):
    stage: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class InferResponse(BaseModel):
    model_version: str
    inference_time_ms: float
    image_width: int
    image_height: int
    total_rbc_count: int
    detections: List[DetectionOut]


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    # Download image
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(request.image_url)
            resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {exc}")

    img_array = np.array(image)
    w, h = image.size

    model = get_model()
    t0 = time.monotonic()
    results = model.predict(
        img_array,
        conf=request.confidence_threshold,
        verbose=False,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000

    detections: List[DetectionOut] = []
    rbc_count = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalized coords

            if cls_id == RBC_CLASS_ID:
                rbc_count += 1
                continue

            stage = CLASS_MAP.get(cls_id)
            if stage:
                detections.append(DetectionOut(
                    stage=stage,
                    confidence=round(conf, 4),
                    x_min=round(x1, 6),
                    y_min=round(y1, 6),
                    x_max=round(x2, 6),
                    y_max=round(y2, 6),
                ))

    return InferResponse(
        model_version=MODEL_VERSION,
        inference_time_ms=round(elapsed_ms, 2),
        image_width=w,
        image_height=h,
        total_rbc_count=rbc_count,
        detections=detections,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_VERSION}
