"""Backward-compatible sync safety facade.

Provides the historical ``app.coordination.sync_safety`` import path while the
canonical exports live in ``app.coordination._exports_sync``.
"""

import app.coordination._exports_sync as _exports_sync
from app.coordination.sync_bloom_filter import (
    DEFAULT_FALSE_POSITIVE_RATE,
    DEFAULT_HASH_COUNT,
    DEFAULT_SIZE,
)
from app.coordination.sync_durability import reset_instances
from app.coordination.sync_integrity import DEFAULT_CHUNK_SIZE, LARGE_CHUNK_SIZE


def _bind_sync_exports() -> list[str]:
    exported_names = list(_exports_sync.__all__)
    for name in exported_names:
        globals()[name] = getattr(_exports_sync, name)
    return exported_names


_SYNC_EXPORTS = _bind_sync_exports()

__all__ = [
    *_SYNC_EXPORTS,
    "DEFAULT_CHUNK_SIZE",
    "LARGE_CHUNK_SIZE",
    "DEFAULT_SIZE",
    "DEFAULT_HASH_COUNT",
    "DEFAULT_FALSE_POSITIVE_RATE",
    "reset_instances",
]
