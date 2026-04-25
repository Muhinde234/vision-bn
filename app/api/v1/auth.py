from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.core.rate_limiter import limiter
from app.models.user import User
from app.schemas.auth import LoginRequest, PasswordChangeRequest, RefreshRequest, TokenResponse
from app.schemas.common import APIResponse
from app.schemas.user import UserCreate, UserRead
from app.services.auth_service import AuthService

router = APIRouter()


@router.post("/register", response_model=APIResponse[UserRead], status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account."""
    service = AuthService(db)
    user = await service.register(payload)
    return APIResponse(message="User registered successfully", data=UserRead.model_validate(user))


@router.post("/login", response_model=APIResponse[TokenResponse])
@limiter.limit("5/minute")
async def login(
    payload: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and receive JWT access + refresh tokens."""
    service = AuthService(db)
    device_info = request.headers.get("user-agent", "unknown")
    tokens = await service.login(payload, device_info=device_info)
    return APIResponse(data=tokens)


@router.post("/refresh", response_model=APIResponse[TokenResponse])
async def refresh_token(
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rotate refresh token and receive new access token."""
    service = AuthService(db)
    tokens = await service.refresh(payload.refresh_token)
    return APIResponse(data=tokens)


@router.post("/logout", response_model=APIResponse)
async def logout(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke all refresh tokens for the current user (all devices)."""
    service = AuthService(db)
    await service.revoke_all_tokens(current_user.id)
    return APIResponse(message="Logged out from all devices")


@router.post("/change-password", response_model=APIResponse)
async def change_password(
    payload: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change password for authenticated user."""
    service = AuthService(db)
    await service.change_password(current_user, payload.current_password, payload.new_password)
    return APIResponse(message="Password changed successfully")


@router.get("/me", response_model=APIResponse[UserRead])
async def me(current_user: User = Depends(get_current_user)):
    """Return authenticated user's profile."""
    return APIResponse(data=UserRead.model_validate(current_user))
