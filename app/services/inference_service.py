"""
AI Inference service – calls the YOLO v9 microservice over HTTP.

The microservice contract (POST /infer):
  Request  : { image_url, confidence_threshold, diagnosis_id }
  Response : { model_version, inference_time_ms, image_width, image_height,
               total_rbc_count, detections: [{stage, confidence, x_min, ...}] }

For local integration (INFERENCE_BACKEND=local) we call the model directly.
"""
import asyncio
import time
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import InferenceError
from app.core.logging import logger
from app.models.diagnosis import Diagnosis, DiagnosisSeverity, DiagnosisStatus
from app.models.image import DiagnosticImage
from app.models.result import Detection, DiagnosisResult
from app.schemas.result import InferenceResponse
from app.services.storage_service import StorageService

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2  # seconds — doubles each attempt: 2s, 4s


def _compute_severity(parasitaemia: float) -> DiagnosisSeverity:
    if parasitaemia == 0:
        return DiagnosisSeverity.NEGATIVE
    if parasitaemia < 1:
        return DiagnosisSeverity.LOW
    if parasitaemia < 5:
        return DiagnosisSeverity.MODERATE
    if parasitaemia < 10:
        return DiagnosisSeverity.HIGH
    return DiagnosisSeverity.SEVERE


class InferenceService:
    def __init__(self, db: AsyncSession, storage: StorageService):
        self.db = db
        self.storage = storage

    async def run(
        self,
        diagnosis: Diagnosis,
        image: DiagnosticImage,
    ) -> DiagnosisResult:
        image_url = await self.storage.get_url(image.storage_path)

        raw: InferenceResponse = await self._call_inference(
            image_url=image_url,
            diagnosis_id=str(diagnosis.id),
        )

        return await self._persist_result(diagnosis, image, raw)

    # ── Microservice call (with retry + exponential back-off) ────────────────

    async def _call_inference(
        self,
        image_url: str,
        diagnosis_id: str,
    ) -> InferenceResponse:
        payload = {
            "image_url": image_url,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "diagnosis_id": diagnosis_id,
        }

        logger.info("Calling inference service", url=settings.INFERENCE_SERVICE_URL)
        t0 = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=settings.INFERENCE_TIMEOUT_SECONDS
                ) as client:
                    response = await client.post(
                        f"{settings.INFERENCE_SERVICE_URL}/infer",
                        json=payload,
                    )
                    response.raise_for_status()

                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.info("Inference completed", elapsed_ms=round(elapsed_ms, 1), attempt=attempt)
                return InferenceResponse(**response.json())

            except httpx.HTTPStatusError as exc:
                # 4xx errors are not retriable (client fault)
                raise InferenceError(
                    f"Inference service error {exc.response.status_code}: {exc.response.text}"
                ) from exc

            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Inference service unavailable, retrying",
                        attempt=attempt,
                        retry_in_seconds=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        raise InferenceError(
            f"Inference service unreachable after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    async def _persist_result(
        self,
        diagnosis: Diagnosis,
        image: DiagnosticImage,
        raw: InferenceResponse,
    ) -> DiagnosisResult:
        # Count detections per stage
        counts = {"ring": 0, "trophozoite": 0, "schizont": 0, "gametocyte": 0}
        for d in raw.detections:
            counts[d.stage.lower()] = counts.get(d.stage.lower(), 0) + 1

        total_parasites = len(raw.detections)
        parasitaemia = (
            (total_parasites / raw.total_rbc_count * 100)
            if raw.total_rbc_count > 0
            else 0.0
        )

        result = DiagnosisResult(
            diagnosis_id=diagnosis.id,
            image_id=image.id,
            total_rbc_count=raw.total_rbc_count,
            total_parasite_count=total_parasites,
            parasitaemia_percent=round(parasitaemia, 4),
            ring_count=counts["ring"],
            trophozoite_count=counts["trophozoite"],
            schizont_count=counts["schizont"],
            gametocyte_count=counts["gametocyte"],
            model_version=raw.model_version,
            inference_time_ms=raw.inference_time_ms,
            raw_inference_output=raw.model_dump(),
        )
        self.db.add(result)
        await self.db.flush()

        # Persist individual detections
        for det in raw.detections:
            self.db.add(Detection(
                result_id=result.id,
                stage=det.stage,
                confidence=det.confidence,
                x_min=det.x_min,
                y_min=det.y_min,
                x_max=det.x_max,
                y_max=det.y_max,
            ))

        # Update diagnosis severity and status
        diagnosis.severity = _compute_severity(parasitaemia)
        diagnosis.status = DiagnosisStatus.COMPLETED
        await self.db.flush()

        return result
