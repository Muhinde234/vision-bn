"""
Validates, processes, and stores diagnostic images.
"""
import io
from uuid import UUID

from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import ImageValidationError
from app.models.diagnosis import Diagnosis, DiagnosisStatus
from app.models.image import DiagnosticImage, ImageStatus
from app.services.storage_service import StorageService

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff"}
MIN_DIMENSION = 256   # pixels

# Map content-type → Pillow save format
_FORMAT_MAP = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/tiff": "TIFF",
}


def validate_blood_smear(image_bytes: bytes) -> None:
    """
    Raises ImageValidationError if the image does not look like a Giemsa-stained
    blood smear microscopy slide. Uses pixel-level colour-distribution heuristics:
      - RBCs appear salmon/pink under Giemsa stain
      - Background is near-white (bright microscopy illumination)
      - Parasites/nuclei appear purple/violet
    Images dominated by green (nature), blue (sky/water), or very dark tones are rejected.
    """
    import numpy as np

    try:
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ImageValidationError("Cannot decode the uploaded image.")

    arr = np.array(img, dtype=np.float32)
    total = float(arr.shape[0] * arr.shape[1])
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    mean_brightness = float(arr.mean()) / 255.0

    # Very dark → not a lit microscopy slide
    if mean_brightness < 0.25:
        raise ImageValidationError(
            "This image does not appear to be a blood smear. "
            "Please upload a properly illuminated microscopy image of a stained blood sample."
        )

    # Colour-ratio checks
    green_ratio  = float(((g > r + 20) & (g > b + 20)).sum()) / total
    blue_ratio   = float(((b > r + 30) & (b > g + 15)).sum()) / total
    orange_ratio = float(((r > 170) & (g > 110) & (b < 100) & ((r - b) > 70)).sum()) / total

    _MSG = (
        "This image does not appear to be a blood smear microscopy slide. "
        "Please upload a Giemsa-stained blood sample image taken under a microscope."
    )

    if green_ratio > 0.30:
        raise ImageValidationError(_MSG)
    if blue_ratio > 0.35:
        raise ImageValidationError(_MSG)

    non_smear_score = green_ratio + blue_ratio + orange_ratio
    if non_smear_score > 0.45:
        raise ImageValidationError(_MSG)

    # Positive blood-smear signature:
    #   pink  – RBCs (high R, R leads G and B, non-orange B channel present)
    #   light – bright microscopy background
    #   purple – stained nuclei / parasites
    pink_ratio   = float(((r > 155) & (r > g + 15) & (r > b + 5) & (b > 85)).sum()) / total
    light_ratio  = float(((r > 195) & (g > 180) & (b > 175)).sum()) / total
    purple_ratio = float(((b > 115) & (r > 85) & (b > g + 12) & (r > g + 5)).sum()) / total

    blood_score = pink_ratio + light_ratio * 0.55 + purple_ratio

    # If all non-smear signals are near zero AND brightness is in range for a lit
    # microscopy slide, the image is clearly not a natural/outdoor photo — accept it
    # regardless of the positive-signature score (handles differently-stained slides).
    clearly_medical = non_smear_score < 0.05 and 0.25 < mean_brightness < 0.95
    if not clearly_medical and blood_score < 0.08:
        raise ImageValidationError(_MSG)


def validate_and_strip_exif(content_type: str, data: bytes) -> tuple[bytes, int, int]:
    """
    Validate image bytes, strip EXIF/metadata, and return (clean_bytes, width, height).
    Raises ImageValidationError on any problem.
    """
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"Unsupported file type '{content_type}'. "
            f"Allowed: JPEG, PNG, TIFF"
        )
    if len(data) > settings.max_image_bytes:
        raise ImageValidationError(
            f"Image exceeds maximum size of {settings.MAX_IMAGE_SIZE_MB} MB"
        )

    try:
        img = PILImage.open(io.BytesIO(data))
        img.verify()  # structural integrity check
    except Exception:
        raise ImageValidationError("File is not a valid image")

    # Re-open (verify() closes the file pointer)
    img = PILImage.open(io.BytesIO(data))
    width, height = img.size

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small: minimum {MIN_DIMENSION}x{MIN_DIMENSION}px"
        )

    # Strip EXIF/metadata by re-encoding into a fresh buffer (no exif= kwarg)
    pil_format = _FORMAT_MAP.get(content_type, "PNG")
    clean_buf = io.BytesIO()
    # Convert to RGB for JPEG (no alpha channel), preserve mode otherwise
    save_img = img.convert("RGB") if pil_format == "JPEG" and img.mode not in ("RGB", "L") else img
    save_img.save(clean_buf, format=pil_format)
    clean_buf.seek(0)
    return clean_buf.read(), width, height


class ImageService:
    def __init__(self, db: AsyncSession, storage: StorageService):
        self.db = db
        self.storage = storage

    async def upload(
        self,
        diagnosis_id: UUID,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> DiagnosticImage:
        # ── Validate + strip EXIF ─────────────────────────────────────────────
        clean_data, width, height = validate_and_strip_exif(content_type, data)

        # ── Store ─────────────────────────────────────────────────────────────
        storage_path = await self.storage.save(clean_data, filename)

        image_record = DiagnosticImage(
            diagnosis_id=diagnosis_id,
            original_filename=filename,
            storage_path=storage_path,
            content_type=content_type,
            file_size_bytes=len(clean_data),
            width_px=width,
            height_px=height,
            status=ImageStatus.PENDING,
        )
        self.db.add(image_record)
        await self.db.flush()
        return image_record

    async def mark_processing(self, image: DiagnosticImage) -> None:
        image.status = ImageStatus.PROCESSING
        await self.db.flush()

    async def mark_done(self, image: DiagnosticImage) -> None:
        image.status = ImageStatus.DONE
        await self.db.flush()

    async def mark_failed(self, image: DiagnosticImage, error: str) -> None:
        image.status = ImageStatus.FAILED
        image.error_message = error
        await self.db.flush()
