"""
Shared rate-limiter instance.
Defined here (not in main.py) to avoid circular imports when
individual routers need to apply per-endpoint limits.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
)
