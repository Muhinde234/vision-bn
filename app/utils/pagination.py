"""
Pagination helpers used across services.
"""
import math
from typing import Generic, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def compute_pages(total: int, page_size: int) -> int:
    return max(1, math.ceil(total / page_size)) if page_size > 0 else 1


def paginate_query(query, page: int, page_size: int):
    """Apply LIMIT / OFFSET to a SQLAlchemy select statement."""
    return query.offset((page - 1) * page_size).limit(page_size)
