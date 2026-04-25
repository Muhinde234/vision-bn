"""
VisionDx API – FastAPI application entry point.
"""
import os
import uuid
from contextlib import asynccontextmanager

import sentry_sdk
import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.router import api_router
from app.config import settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DuplicateError,
    ImageValidationError,
    InferenceError,
    NotFoundError,
    VisionDxError,
)
from app.core.logging import logger, setup_logging
from app.core.rate_limiter import limiter

# ── Request body size limit (50 MB hard cap to prevent DoS) ──────────────────
_MAX_REQUEST_BODY_BYTES = 50 * 1024 * 1024


# ── Sentry (production error tracking) ───────────────────────────────────────
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        traces_sample_rate=0.2,
    )


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info("VisionDx API starting", version=settings.APP_VERSION, env=settings.APP_ENV)
    yield
    logger.info("VisionDx API shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered malaria diagnostic API. "
        "Detects parasite life stages from whole blood smear images using YOLOv9."
    ),
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Request size limit middleware ─────────────────────────────────────────────
@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"success": False, "code": "REQUEST_TOO_LARGE", "message": "Request body exceeds 50 MB limit"},
        )
    return await call_next(request)


# ── Request ID / correlation middleware ───────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Domain exception → HTTP mapping ──────────────────────────────────────────
_STATUS_MAP = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    DuplicateError: status.HTTP_409_CONFLICT,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    ImageValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    InferenceError: status.HTTP_502_BAD_GATEWAY,
}


@app.exception_handler(VisionDxError)
async def domain_exception_handler(request: Request, exc: VisionDxError):
    http_status = _STATUS_MAP.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    return JSONResponse(
        status_code=http_status,
        content={"success": False, "code": exc.code, "message": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )


# ── Prometheus metrics ────────────────────────────────────────────────────────
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ── Static files (local storage) ──────────────────────────────────────────────
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.UPLOAD_DIR), name="static")


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")

# ── Malaria detection routes (no auth required for standalone use) ─────────────
from app.routes.train   import router as train_router
from app.routes.predict import router as predict_router

app.include_router(train_router)
app.include_router(predict_router)


# ── Convenience alias: /api/v1/user/profile → /api/v1/auth/me ────────────────
# Next.js auth hooks commonly call /user/profile; we expose a thin alias
# so the frontend doesn't need to know the internal auth route name.
from fastapi import Depends
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.user import UserRead


@app.get(
    "/api/v1/user/profile",
    response_model=APIResponse[UserRead],
    tags=["Authentication"],
    summary="Get current user profile (alias for /auth/me)",
)
async def user_profile(current_user: User = Depends(get_current_user)):
    return APIResponse(data=UserRead.model_validate(current_user))


# ── Health endpoints ──────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], include_in_schema=False)
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/health/db", tags=["Health"], include_in_schema=False)
async def health_db():
    from sqlalchemy import text
    from app.db.session import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": str(exc)},
        )
