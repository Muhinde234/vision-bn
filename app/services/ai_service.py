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
from typing import Dict, List, Optional

from PIL import Image as PILImage

from app.config import settings
from app.core.logging import logger
from app.models.prediction import DiseaseType
from app.schemas.prediction import AIResultDetail, BoundingBox


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
        if settings.INFERENCE_BACKEND == "microservice" and not settings.DEBUG:
            self._engine: InferenceEngine = MicroserviceYOLOv9Engine()
            logger.info("AIService using real YOLO microservice")
        else:
            self._engine = MockYOLOv9Engine()
            logger.info("AIService using mock YOLO engine (development)")

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
