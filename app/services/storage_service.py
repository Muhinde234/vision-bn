"""
Abstracted file storage – local disk, S3/MinIO, or Cloudinary.
Returns a storage_path (local path, S3 key, or Cloudinary public_id) persisted to DB.
"""
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

import aiofiles

from app.config import settings
from app.core.exceptions import StorageError
from app.core.logging import logger


@lru_cache(maxsize=1)
def _get_cloudinary():
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True,
    )
    return cloudinary.uploader


@lru_cache(maxsize=1)
def _get_s3_client():
    """Cached boto3 S3 client — created once per process, not per request."""
    import boto3

    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION,
        endpoint_url=settings.AWS_S3_ENDPOINT_URL or None,
    )


class StorageService:
    """Strategy pattern: chooses local or S3 backend at runtime."""

    async def save(self, data: bytes, original_filename: str, prefix: str = "images") -> str:
        ext = Path(original_filename).suffix.lower()
        key = f"{prefix}/{uuid.uuid4().hex}{ext}"

        if settings.STORAGE_BACKEND == "cloudinary":
            return await self._save_cloudinary(data, prefix)
        if settings.STORAGE_BACKEND == "local":
            return await self._save_local(data, key)
        return await self._save_s3(data, key)

    async def get_url(self, storage_path: str) -> str:
        if settings.STORAGE_BACKEND == "cloudinary":
            return storage_path  # Cloudinary returns a full HTTPS URL as the path
        if settings.STORAGE_BACKEND == "local":
            return f"/static/{storage_path}"
        return await self._presign_s3(storage_path)

    async def delete(self, storage_path: str) -> None:
        if settings.STORAGE_BACKEND == "cloudinary":
            return  # skip deletion for now
        if settings.STORAGE_BACKEND == "local":
            await self._delete_local(storage_path)
        else:
            await self._delete_s3(storage_path)

    # ── Local ─────────────────────────────────────────────────────────────────

    async def _save_local(self, data: bytes, key: str) -> str:
        full_path = Path(settings.UPLOAD_DIR) / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(data)
        logger.info("Saved file locally", path=str(full_path))
        return key

    async def _delete_local(self, key: str) -> None:
        full_path = Path(settings.UPLOAD_DIR) / key
        if full_path.exists():
            full_path.unlink()

    # ── Cloudinary ───────────────────────────────────────────────────────────

    async def _save_cloudinary(self, data: bytes, folder: str) -> str:
        import asyncio
        from io import BytesIO

        uploader = _get_cloudinary()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: uploader.upload(
                    BytesIO(data),
                    folder=f"visiondx/{folder}",
                    resource_type="image",
                ),
            )
            return result["secure_url"]
        except Exception as exc:
            raise StorageError(f"Cloudinary upload failed: {exc}") from exc

    # ── S3 ────────────────────────────────────────────────────────────────────

    async def _save_s3(self, data: bytes, key: str) -> str:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            _get_s3_client().put_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key=key,
                Body=data,
                ServerSideEncryption="AES256",
            )
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 upload failed: {exc}") from exc
        return key

    async def _presign_s3(self, key: str, expires: int = 3600) -> str:
        return _get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.AWS_S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )

    async def _delete_s3(self, key: str) -> None:
        _get_s3_client().delete_object(Bucket=settings.AWS_S3_BUCKET, Key=key)
