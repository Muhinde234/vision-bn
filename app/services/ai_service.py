"""
AI Service – central inference abstraction.

Architecture:
  AIService.predict(image_bytes, disease_type) → AIResult

In development / CI:   uses MockYOLOv9Engine (random but structured output)
In production:         swaps to RealYOLOv9Engine (calls inference microservice)

The calling code (prediction_service.py) never changes;
only the engine injected at startup differs.

To plug in your real model:
  1. Set INFERENCE_BACKEND=microservice in .env
  2. Deploy inference_service/ (see docker-compose.yml)
  3. Done. No changes needed here.
"""
from __future__ import annotations

import io
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image as PILImage

from app.config import settings
from app.core.logging import logger
from app.models.prediction import DiseaseType
from app.schemas.prediction import AIResultDetail, BoundingBox

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AIResult:
    predicted_class: str
    confidence_score: float           # 0.0 – 1.0
    severity_level: str               # negative | mild | moderate | severe
    recommendation: str
    detail: AIResultDetail
    model_version: str
    inference_time_ms: float


# ── Knowledge base: disease-specific classes & recommendations ────────────────

_DISEASE_KNOWLEDGE: Dict[DiseaseType, Dict] = {
    DiseaseType.MALARIA: {
        "classes": ["Negative", "Ring Stage", "Trophozoite", "Schizont", "Gametocyte"],
        "recommendations": {
            "Negative": (
                "No malaria parasites detected. "
                "If symptoms persist, repeat the test after 12–24 hours. "
                "Maintain preventive measures (mosquito nets, repellents)."
            ),
            "Ring Stage": (
                "Early malaria infection detected (ring-stage parasites). "
                "Initiate artemisinin-based combination therapy (ACT) immediately. "
                "Monitor parasitaemia every 24 hours. "
                "Report to national surveillance system."
            ),
            "Trophozoite": (
                "Active malaria infection – trophozoite stage detected. "
                "Start ACT without delay. Assess for severity markers "
                "(haemoglobin < 7 g/dL, altered consciousness). "
                "Consider hospitalisation if moderate-severe."
            ),
            "Schizont": (
                "High-density malaria infection (schizont stage). "
                "Urgent parenteral antimalarial treatment indicated. "
                "Immediate clinical assessment for severe malaria criteria. "
                "Hospitalise and monitor closely."
            ),
            "Gametocyte": (
                "Malaria gametocytes detected – patient is infectious. "
                "Complete a full ACT course to clear gametocytes. "
                "Advise on transmission prevention. "
                "Primaquine may be considered per local guidelines."
            ),
        },
    },
    DiseaseType.TUBERCULOSIS: {
        "classes": ["Normal", "TB Suspected", "Active TB", "Pleural Effusion"],
        "recommendations": {
            "Normal": "No signs of tuberculosis on this image. Monitor if symptoms persist.",
            "TB Suspected": (
                "Findings suggestive of tuberculosis. "
                "Refer for sputum smear microscopy and GeneXpert MTB/RIF. "
                "Isolate patient until diagnosis confirmed."
            ),
            "Active TB": (
                "Radiological findings consistent with active TB. "
                "Start standard 6-month DOTS regimen (HRZE/HR). "
                "Notify public health authorities. "
                "Screen household contacts."
            ),
            "Pleural Effusion": (
                "Pleural effusion detected – TB pleuritis possible. "
                "Pleural fluid analysis (ADA, lymphocyte count) required. "
                "Consider thoracocentesis for diagnosis and symptom relief."
            ),
        },
    },
    DiseaseType.PNEUMONIA: {
        "classes": ["Normal", "Bacterial Pneumonia", "Viral Pneumonia", "COVID-19 Pattern"],
        "recommendations": {
            "Normal": "Clear lung fields. No evidence of pneumonia.",
            "Bacterial Pneumonia": (
                "Consolidation pattern consistent with bacterial pneumonia. "
                "Start empirical antibiotics (amoxicillin/clavulanate or azithromycin). "
                "Reassess in 48–72 hours."
            ),
            "Viral Pneumonia": (
                "Bilateral ground-glass opacities suggesting viral pneumonia. "
                "Supportive care; consider antiviral therapy if influenza confirmed. "
                "Monitor oxygen saturation closely."
            ),
            "COVID-19 Pattern": (
                "Imaging pattern consistent with COVID-19 pneumonia. "
                "Isolate patient; perform PCR confirmation. "
                "Assess oxygen requirements; escalate care if SpO₂ < 94%."
            ),
        },
    },
    DiseaseType.DIABETIC_RETINOPATHY: {
        "classes": ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"],
        "recommendations": {
            "No DR": "No diabetic retinopathy detected. Annual screening recommended.",
            "Mild NPDR": "Mild non-proliferative DR. Optimise glycaemic control (HbA1c < 7%). Re-screen in 12 months.",
            "Moderate NPDR": "Moderate NPDR. Urgent ophthalmology referral. Optimise BP and glucose.",
            "Severe NPDR": "Severe NPDR – high risk of progression. Ophthalmology review within 1 week. Pan-retinal photocoagulation may be required.",
            "PDR": "Proliferative DR detected. Urgent ophthalmology. Anti-VEGF therapy or laser treatment needed immediately.",
        },
    },
    DiseaseType.SKIN_LESION: {
        "classes": ["Benign", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Actinic Keratosis"],
        "recommendations": {
            "Benign": "Lesion appears benign. Monitor for changes in size, shape, or colour (ABCDE criteria).",
            "Melanoma": "Features concerning for melanoma. Urgent dermatology referral. Excisional biopsy required.",
            "Basal Cell Carcinoma": "Probable BCC. Refer to dermatology/surgery for excision.",
            "Squamous Cell Carcinoma": "Features consistent with SCC. Excision with clear margins needed. Stage workup if invasive.",
            "Actinic Keratosis": "Pre-cancerous actinic keratosis. Treat with cryotherapy, topical 5-FU, or photodynamic therapy.",
        },
    },
    DiseaseType.GENERAL: {
        "classes": ["Normal", "Abnormality Detected"],
        "recommendations": {
            "Normal": "No significant abnormalities detected. Routine follow-up as clinically indicated.",
            "Abnormality Detected": "Potential abnormality detected. Clinical correlation and specialist review recommended.",
        },
    },
}

_SEVERITY_MAP = {
    "Negative": "negative", "Normal": "negative", "No DR": "negative",
    "No Pneumonia": "negative", "Benign": "negative",
    "Ring Stage": "mild", "Mild NPDR": "mild", "TB Suspected": "mild",
    "Mild": "mild", "Actinic Keratosis": "mild",
    "Trophozoite": "moderate", "Moderate NPDR": "moderate",
    "Bacterial Pneumonia": "moderate", "Viral Pneumonia": "moderate",
    "Schizont": "severe", "Gametocyte": "moderate",
    "Severe NPDR": "severe", "PDR": "severe",
    "Active TB": "severe", "COVID-19 Pattern": "severe",
    "Melanoma": "severe", "Basal Cell Carcinoma": "moderate",
    "Squamous Cell Carcinoma": "severe", "Proliferative DR": "severe",
    "Abnormality Detected": "moderate",
    "Pleural Effusion": "moderate",
}


# ── Abstract engine ───────────────────────────────────────────────────────────

class InferenceEngine(ABC):
    @abstractmethod
    async def infer(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        ...


# ── Mock engine (development / CI) ───────────────────────────────────────────

class MockYOLOv9Engine(InferenceEngine):
    """
    Deterministic-ish mock that returns realistic structured output.
    Weights are seeded from the image content so the same image
    always returns the same result within a session.
    """

    MODEL_VERSION = "yolov9-mock-v1.0"

    async def infer(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        t0 = time.monotonic()

        knowledge = _DISEASE_KNOWLEDGE[disease_type]
        classes: List[str] = knowledge["classes"]

        # Seed from image content for repeatability
        seed = sum(image_bytes[:64]) if image_bytes else 42
        rng = random.Random(seed)

        # Pick predicted class (weighted toward first class for a realistic neg-rate)
        weights = [0.40] + [0.60 / (len(classes) - 1)] * (len(classes) - 1)
        predicted_class = rng.choices(classes, weights=weights)[0]
        confidence = round(rng.uniform(0.72, 0.97), 4)

        # Class probabilities (softmax-like distribution)
        raw_probs = {c: round(rng.random(), 4) for c in classes}
        raw_probs[predicted_class] = confidence
        total = sum(raw_probs.values())
        class_probs = {c: round(v / total, 4) for c, v in raw_probs.items()}

        # Mock bounding boxes for positive findings
        boxes: List[BoundingBox] = []
        if predicted_class not in ("Negative", "Normal", "No DR", "Benign", "No Pneumonia"):
            for _ in range(rng.randint(1, 4)):
                x1 = rng.uniform(0.05, 0.6)
                y1 = rng.uniform(0.05, 0.6)
                boxes.append(BoundingBox(
                    x_min=round(x1, 4),
                    y_min=round(y1, 4),
                    x_max=round(x1 + rng.uniform(0.1, 0.35), 4),
                    y_max=round(y1 + rng.uniform(0.1, 0.35), 4),
                    label=predicted_class,
                    confidence=round(rng.uniform(0.65, confidence), 4),
                ))

        elapsed_ms = (time.monotonic() - t0) * 1000 + rng.uniform(40, 120)  # simulate latency

        detail = AIResultDetail(
            model_version=self.MODEL_VERSION,
            inference_time_ms=round(elapsed_ms, 2),
            image_width=image_width,
            image_height=image_height,
            class_probabilities=class_probs,
            bounding_boxes=boxes,
        )

        recommendation = knowledge["recommendations"].get(
            predicted_class,
            "Please consult a specialist for further evaluation.",
        )
        severity = _SEVERITY_MAP.get(predicted_class, "unknown")

        return AIResult(
            predicted_class=predicted_class,
            confidence_score=confidence,
            severity_level=severity,
            recommendation=recommendation,
            detail=detail,
            model_version=self.MODEL_VERSION,
            inference_time_ms=detail.inference_time_ms,
        )


# ── Real engine (production – calls inference microservice) ───────────────────

class MicroserviceYOLOv9Engine(InferenceEngine):
    """
    Calls the standalone YOLOv9 inference microservice.
    See inference_service/main.py for the server implementation.
    """

    async def infer(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        import httpx
        from app.core.exceptions import InferenceError

        # Upload image as base64 or multipart – here we use multipart
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=settings.INFERENCE_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{settings.INFERENCE_SERVICE_URL}/infer",
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                    data={"disease_type": disease_type.value},
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise InferenceError("Inference microservice timed out")
        except httpx.HTTPStatusError as exc:
            raise InferenceError(f"Inference error {exc.response.status_code}")
        except httpx.RequestError as exc:
            raise InferenceError(f"Inference service unreachable: {exc}")

        elapsed_ms = (time.monotonic() - t0) * 1000
        data = response.json()

        knowledge = _DISEASE_KNOWLEDGE[disease_type]
        predicted_class = data.get("predicted_class", "Unknown")
        confidence = data.get("confidence_score", 0.0)
        recommendation = knowledge["recommendations"].get(
            predicted_class, "Consult a specialist."
        )
        severity = _SEVERITY_MAP.get(predicted_class, "unknown")

        detail = AIResultDetail(
            model_version=data.get("model_version", "unknown"),
            inference_time_ms=round(elapsed_ms, 2),
            image_width=image_width,
            image_height=image_height,
            class_probabilities=data.get("class_probabilities", {}),
            bounding_boxes=[BoundingBox(**b) for b in data.get("bounding_boxes", [])],
        )

        return AIResult(
            predicted_class=predicted_class,
            confidence_score=confidence,
            severity_level=severity,
            recommendation=recommendation,
            detail=detail,
            model_version=data.get("model_version", "unknown"),
            inference_time_ms=round(elapsed_ms, 2),
        )


# ── ONNX inference engine (production — no PyTorch required) ─────────────────

class LocalONNXEngine(InferenceEngine):
    """
    Runs models/best.onnx via onnxruntime.
    No PyTorch / ultralytics needed at runtime — only ~50 MB RAM footprint.
    This is what Render uses. Train locally, export ONNX, commit the file.

    Class IDs match prepare_yolo_dataset.py:
        0 red blood cell  1 trophozoite  2 ring
        3 schizont        4 gametocyte   5 leukocyte
    """

    ONNX_PATH     = Path(__file__).resolve().parents[2] / "models" / "best.onnx"
    MODEL_VERSION = "yolov9n-malaria-onnx-v1.0"
    IMG_SIZE      = 640
    CONF_THRESH   = 0.25
    IOU_THRESH    = 0.45

    # Indices 1-4 are parasites; 0 (RBC) and 5 (leukocyte) are benign
    _TRAIN_CLASSES = [
        "red blood cell", "trophozoite", "ring", "schizont", "gametocyte", "leukocyte"
    ]
    _PARASITE_IDS = {1, 2, 3, 4}   # trophozoite, ring, schizont, gametocyte

    _DISPLAY = {
        "ring":           "Ring Stage",
        "trophozoite":    "Trophozoite",
        "schizont":       "Schizont",
        "gametocyte":     "Gametocyte",
        "red blood cell": "Negative",
        "leukocyte":      "Negative",
    }

    def __init__(self) -> None:
        import onnxruntime as ort
        logger.info("Loading ONNX model", path=str(self.ONNX_PATH))
        self._session = ort.InferenceSession(
            str(self.ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    # ── Async wrapper ─────────────────────────────────────────────────────────

    async def infer(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_sync, image_bytes, disease_type, image_width, image_height
        )

    # ── Sync inference pipeline ───────────────────────────────────────────────

    def _run_sync(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        import numpy as np

        t0 = time.monotonic()

        # 1. Preprocess
        tensor, scale_x, scale_y = self._preprocess(image_bytes)

        # 2. Inference
        raw_out = self._session.run(None, {self._input_name: tensor})[0]
        # raw_out shape: (1, nc+4, num_anchors) — e.g. (1, 10, 8400) for nc=6

        # 3. Decode & NMS
        detections = self._postprocess(raw_out, scale_x, scale_y)

        # 4. Determine diagnosis
        knowledge = _DISEASE_KNOWLEDGE[disease_type]
        parasite_hits: List[tuple] = []

        for x1, y1, x2, y2, conf, cls_id in detections:
            raw_name = self._TRAIN_CLASSES[cls_id] if cls_id < len(self._TRAIN_CLASSES) else "unknown"
            if cls_id not in self._PARASITE_IDS:
                continue
            display = self._DISPLAY[raw_name]
            bbox = BoundingBox(
                x_min      = round(x1 / image_width,  4),
                y_min      = round(y1 / image_height, 4),
                x_max      = round(x2 / image_width,  4),
                y_max      = round(y2 / image_height, 4),
                label      = raw_name,
                confidence = round(float(conf), 4),
            )
            parasite_hits.append((display, float(conf), bbox))

        if parasite_hits:
            best             = max(parasite_hits, key=lambda x: x[1])
            predicted_class  = best[0]
            confidence_score = round(best[1], 4)
            boxes_out        = [h[2] for h in parasite_hits]
        else:
            predicted_class  = "Negative"
            confidence_score = 0.97
            boxes_out        = []

        elapsed_ms = (time.monotonic() - t0) * 1000

        class_probs: Dict[str, float] = {c: 0.0 for c in knowledge["classes"]}
        for display, conf, _ in parasite_hits:
            if display in class_probs:
                class_probs[display] = max(class_probs[display], round(conf, 4))
        if predicted_class == "Negative":
            class_probs["Negative"] = confidence_score

        detail = AIResultDetail(
            model_version       = self.MODEL_VERSION,
            inference_time_ms   = round(elapsed_ms, 2),
            image_width         = image_width,
            image_height        = image_height,
            class_probabilities = class_probs,
            bounding_boxes      = boxes_out,
        )
        recommendation = knowledge["recommendations"].get(
            predicted_class, "Please consult a specialist for further evaluation."
        )
        return AIResult(
            predicted_class   = predicted_class,
            confidence_score  = confidence_score,
            severity_level    = _SEVERITY_MAP.get(predicted_class, "unknown"),
            recommendation    = recommendation,
            detail            = detail,
            model_version     = self.MODEL_VERSION,
            inference_time_ms = detail.inference_time_ms,
        )

    # ── Pre / post processing ─────────────────────────────────────────────────

    def _preprocess(self, image_bytes: bytes):
        import numpy as np

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
        img_resized    = img.resize((self.IMG_SIZE, self.IMG_SIZE), PILImage.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0   # [0,1]
        arr = arr.transpose(2, 0, 1)                            # HWC → CHW
        arr = np.expand_dims(arr, 0)                            # → NCHW
        scale_x = orig_w / self.IMG_SIZE
        scale_y = orig_h / self.IMG_SIZE
        return arr, scale_x, scale_y

    def _postprocess(self, raw: "np.ndarray", scale_x: float, scale_y: float) -> list:
        """
        Decode YOLO output and apply NMS.
        raw shape: (1, nc+4, num_anchors)  e.g. (1, 10, 8400)
        Returns list of (x1, y1, x2, y2, confidence, class_id) in original pixel coords.
        """
        import cv2
        import numpy as np

        pred = raw[0].T   # (num_anchors, nc+4)  — e.g. (8400, 10)
        nc   = pred.shape[1] - 4

        boxes_cxywh = pred[:, :4]          # cx, cy, w, h in 640-space
        cls_scores  = pred[:, 4:]          # (num_anchors, nc)

        class_ids    = np.argmax(cls_scores, axis=1)
        confidences  = cls_scores[np.arange(len(cls_scores)), class_ids]

        # Confidence filter
        mask = confidences >= self.CONF_THRESH
        if not mask.any():
            return []

        boxes_cxywh = boxes_cxywh[mask]
        class_ids   = class_ids[mask]
        confidences = confidences[mask]

        # cx,cy,w,h (640-space) → x,y,w,h (orig pixel space, for cv2.NMSBoxes)
        boxes_xywh = []
        for cx, cy, w, h in boxes_cxywh:
            x = float((cx - w / 2) * scale_x)
            y = float((cy - h / 2) * scale_y)
            boxes_xywh.append([x, y, float(w * scale_x), float(h * scale_y)])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences.tolist(),
            self.CONF_THRESH,
            self.IOU_THRESH,
        )
        if len(indices) == 0:
            return []

        results = []
        for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
            x, y, w, h = boxes_xywh[i]
            results.append((x, y, x + w, y + h, confidences[i], int(class_ids[i])))
        return results


# ── Local trained YOLOv9 engine ───────────────────────────────────────────────

class LocalYOLOv9Engine(InferenceEngine):
    """
    Runs the locally trained YOLOv9n model from models/best.pt.
    Inference is offloaded to a thread-pool so the async loop stays free.
    """

    MODEL_VERSION = "yolov9n-malaria-v1.0"

    # Raw YOLO class names that indicate active infection
    _PARASITE_CLASSES = {"ring", "trophozoite", "schizont", "gametocyte"}

    # Map raw class name → display name used in AIResult / recommendations
    _DISPLAY = {
        "ring":           "Ring Stage",
        "trophozoite":    "Trophozoite",
        "schizont":       "Schizont",
        "gametocyte":     "Gametocyte",
        "red blood cell": "Negative",
        "leukocyte":      "Negative",
    }

    def __init__(self) -> None:
        from ultralytics import YOLO
        logger.info("Loading local YOLOv9 model", path=str(_MODEL_PATH))
        self._model = YOLO(str(_MODEL_PATH))

    async def infer(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_sync,
            image_bytes, disease_type, image_width, image_height,
        )

    def _run_sync(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType,
        image_width: int,
        image_height: int,
    ) -> AIResult:
        import numpy as np

        t0 = time.monotonic()

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(img)

        yolo_results = self._model.predict(source=img_arr, conf=0.25, verbose=False)

        knowledge = _DISEASE_KNOWLEDGE[disease_type]
        parasite_hits: List[tuple] = []   # (display_name, confidence, BoundingBox)

        for r in yolo_results:
            if r.boxes is None:
                continue
            names = r.names  # {0: 'red blood cell', 1: 'trophozoite', ...}
            for box in r.boxes:
                cls_id    = int(box.cls[0])
                conf      = float(box.conf[0])
                raw_name  = names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if raw_name not in self._PARASITE_CLASSES:
                    continue

                bbox = BoundingBox(
                    x_min      = round(x1 / image_width,  4),
                    y_min      = round(y1 / image_height, 4),
                    x_max      = round(x2 / image_width,  4),
                    y_max      = round(y2 / image_height, 4),
                    label      = raw_name,
                    confidence = round(conf, 4),
                )
                parasite_hits.append((self._DISPLAY[raw_name], conf, bbox))

        # ── Determine diagnosis ───────────────────────────────────────────────
        if parasite_hits:
            best             = max(parasite_hits, key=lambda x: x[1])
            predicted_class  = best[0]
            confidence_score = round(best[1], 4)
            boxes_out        = [h[2] for h in parasite_hits]
        else:
            predicted_class  = "Negative"
            confidence_score = 0.97
            boxes_out        = []

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Class probabilities for the frontend result card
        class_probs: Dict[str, float] = {c: 0.0 for c in knowledge["classes"]}
        for display, conf, _ in parasite_hits:
            if display in class_probs:
                class_probs[display] = max(class_probs[display], round(conf, 4))
        if predicted_class == "Negative":
            class_probs["Negative"] = confidence_score

        detail = AIResultDetail(
            model_version     = self.MODEL_VERSION,
            inference_time_ms = round(elapsed_ms, 2),
            image_width       = image_width,
            image_height      = image_height,
            class_probabilities = class_probs,
            bounding_boxes    = boxes_out,
        )

        recommendation = knowledge["recommendations"].get(
            predicted_class, "Please consult a specialist for further evaluation."
        )
        severity = _SEVERITY_MAP.get(predicted_class, "unknown")

        return AIResult(
            predicted_class  = predicted_class,
            confidence_score = confidence_score,
            severity_level   = severity,
            recommendation   = recommendation,
            detail           = detail,
            model_version    = self.MODEL_VERSION,
            inference_time_ms = detail.inference_time_ms,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

class AIService:
    """
    Facade used by routes and services.
    Picks the correct engine based on INFERENCE_BACKEND setting.

    Usage:
        ai = AIService()
        result = await ai.predict(image_bytes, DiseaseType.MALARIA)
    """

    def __init__(self):
        _onnx_path = _MODEL_PATH.with_suffix(".onnx")

        if settings.INFERENCE_BACKEND == "microservice" and not settings.DEBUG:
            self._engine: InferenceEngine = MicroserviceYOLOv9Engine()
            logger.info("AIService using real YOLO microservice")
        elif _onnx_path.exists():
            # ONNX preferred for deployment — no PyTorch dependency, ~50 MB RAM
            self._engine = LocalONNXEngine()
            logger.info("AIService using local ONNX model (onnxruntime)", path=str(_onnx_path))
        elif _MODEL_PATH.exists():
            # PyTorch fallback — for local dev after training, before ONNX export
            self._engine = LocalYOLOv9Engine()
            logger.info("AIService using local PyTorch YOLOv9 model", path=str(_MODEL_PATH))
        else:
            self._engine = MockYOLOv9Engine()
            logger.info("AIService using mock YOLO engine (no trained model found)")

    async def predict(
        self,
        image_bytes: bytes,
        disease_type: DiseaseType = DiseaseType.MALARIA,
    ) -> AIResult:
        # Decode image dimensions
        import io as _io
        img = PILImage.open(_io.BytesIO(image_bytes))
        width, height = img.size

        logger.info(
            "Running inference",
            disease_type=disease_type,
            image_size=f"{width}x{height}",
            engine=type(self._engine).__name__,
        )

        result = await self._engine.infer(image_bytes, disease_type, width, height)

        logger.info(
            "Inference complete",
            predicted_class=result.predicted_class,
            confidence=result.confidence_score,
            time_ms=result.inference_time_ms,
        )
        return result
