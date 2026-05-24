"""
Authentication service: register, login, token refresh, password change.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import AuthenticationError, DuplicateError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.user import UserCreate


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Registration ──────────────────────────────────────────────────────────

    async def register(self, payload: UserCreate) -> User:
        existing = await self.db.execute(
            select(User).where(User.email == payload.email)
        )
        if existing.scalar_one_or_none():
            raise DuplicateError("User", "email")

        user = User(
            email=payload.email,
            full_name=payload.full_name,
            hashed_password=hash_password(payload.password),
            role=payload.role,
            facility_name=payload.facility_name,
        )
        self.db.add(user)
        await self.db.flush()

        # Send email verification (non-blocking — failure won't abort registration)
        try:
            token = await self._create_token(user.id, "email-verification", hours=24)
            verify_link = f"{settings.FRONTEND_URL}/verify-email?token={token}"
            logger.info("Email verification token", email=user.email, link=verify_link)
            from app.services.email_service import send_verification_email
            await send_verification_email(user.email, user.full_name, verify_link)
        except Exception as exc:
            logger.warning("Could not send verification email", error=str(exc))

        return user

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(self, payload: LoginRequest, device_info: Optional[str] = None) -> TokenResponse:
        result = await self.db.execute(
            select(User).where(User.email == payload.email)
        )
        user = result.scalar_one_or_none()

        if not user or not verify_password(payload.password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")

        return await self._issue_tokens(user, device_info)

    # ── Token refresh ─────────────────────────────────────────────────────────

    async def refresh(self, raw_refresh_token: str) -> TokenResponse:
        token_hash = hash_token(raw_refresh_token)
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked.is_(False),
            )
        )
        stored = result.scalar_one_or_none()

        if not stored or stored.expires_at < datetime.now(tz=timezone.utc):
            raise AuthenticationError("Refresh token is invalid or expired")

        # Rotate: revoke old token
        stored.revoked = True
        await self.db.flush()

        user_result = await self.db.execute(select(User).where(User.id == stored.user_id))
        user = user_result.scalar_one_or_none()

        if not user or not user.is_active:
            raise AuthenticationError("User not found or deactivated")

        return await self._issue_tokens(user)

    # ── Password change ───────────────────────────────────────────────────────

    async def change_password(self, user: User, current: str, new: str) -> None:
        if not verify_password(current, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")
        user.hashed_password = hash_password(new)
        await self.db.flush()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _issue_tokens(
        self, user: User, device_info: Optional[str] = None
    ) -> TokenResponse:
        access_token = create_access_token(
            subject=str(user.id), role=user.role.value
        )
        raw_refresh = create_refresh_token()
        expires_at = datetime.now(tz=timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
        refresh_record = RefreshToken(
            user_id=user.id,
            token_hash=hash_token(raw_refresh),
            expires_at=expires_at,
            device_info=device_info,
        )
        self.db.add(refresh_record)
        await self.db.flush()

        return TokenResponse(
            access_token=access_token,
            refresh_token=raw_refresh,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    # ── Password reset ────────────────────────────────────────────────────────

    async def create_reset_token(self, email: str) -> None:
        result = await self.db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            return  # silent no-op — never reveal whether email exists

        token = await self._create_token(user.id, "password-reset", hours=1)
        reset_link = f"{settings.FRONTEND_URL}/reset-password?token={token}"
        logger.info("Password reset token", email=email, link=reset_link)
        try:
            from app.services.email_service import send_password_reset_email
            await send_password_reset_email(user.email, reset_link)
        except Exception as exc:
            logger.warning("Could not send password reset email", error=str(exc))

    async def verify_email(self, raw_token: str) -> User:
        """Consume an email-verification token and mark the user as verified."""
        token_hash = hash_token(raw_token)
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked.is_(False),
                RefreshToken.device_info == "email-verification",
            )
        )
        stored = result.scalar_one_or_none()
        if not stored or stored.expires_at < datetime.now(tz=timezone.utc):
            raise AuthenticationError("Verification token is invalid or expired")

        user_result = await self.db.execute(select(User).where(User.id == stored.user_id))
        user = user_result.scalar_one()
        user.is_verified = True
        stored.revoked = True
        await self.db.flush()
        return user

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _create_token(self, user_id, device_info: str, hours: int) -> str:
        raw_token = create_refresh_token()
        expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=hours)
        record = RefreshToken(
            user_id=user_id,
            token_hash=hash_token(raw_token),
            expires_at=expires_at,
            device_info=device_info,
        )
        self.db.add(record)
        await self.db.flush()
        return raw_token

    async def reset_password(self, raw_token: str, new_password: str) -> None:
        from app.schemas.user import UserCreate
        token_hash = hash_token(raw_token)
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked.is_(False),
                RefreshToken.device_info == "password-reset",
            )
        )
        stored = result.scalar_one_or_none()
        if not stored or stored.expires_at < datetime.now(tz=timezone.utc):
            raise AuthenticationError("Reset token is invalid or expired")

        # Validate new password meets requirements
        if len(new_password) < 8:
            raise AuthenticationError("Password must be at least 8 characters")
        if not any(c.isupper() for c in new_password):
            raise AuthenticationError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in new_password):
            raise AuthenticationError("Password must contain at least one digit")

        user_result = await self.db.execute(select(User).where(User.id == stored.user_id))
        user = user_result.scalar_one()
        user.hashed_password = hash_password(new_password)
        stored.revoked = True
        await self.db.flush()

    async def revoke_all_tokens(self, user_id: UUID) -> None:
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked.is_(False),
            )
        )
        for token in result.scalars().all():
            token.revoked = True
        await self.db.flush()
