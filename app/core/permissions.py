"""
Role-based access control (RBAC) helpers.
Use as FastAPI dependencies: `Depends(require_roles([UserRole.ADMIN]))`.
"""
from typing import List

from fastapi import Depends, HTTPException, status

from app.core.deps import get_current_user
from app.models.user import User, UserRole


def require_roles(allowed: List[UserRole]):
    """Factory that returns a dependency checking the caller's role."""
    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {[r.value for r in allowed]}",
            )
        return current_user
    return _check


# Convenience aliases
require_admin = require_roles([UserRole.ADMIN])
require_lab_or_admin = require_roles([UserRole.LAB_TECHNICIAN, UserRole.ADMIN])
require_doctor_or_admin = require_roles([UserRole.DOCTOR, UserRole.ADMIN])
require_any_role = require_roles([UserRole.ADMIN, UserRole.LAB_TECHNICIAN, UserRole.DOCTOR])
