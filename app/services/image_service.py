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
