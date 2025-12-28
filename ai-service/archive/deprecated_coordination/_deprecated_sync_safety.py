"""DEPRECATED: Unified Sync Safety Module - archived December 27, 2025.

This module was a pure re-export wrapper with no external callers.
Import directly from the specialized modules:

- ``app.coordination.sync_durability`` - SyncWAL, DeadLetterQueue
- ``app.coordination.sync_integrity`` - checksum validation
- ``app.coordination.sync_stall_handler`` - stall detection
- ``app.coordination.sync_bloom_filter`` - Bloom filters

For the original implementation, see:
archive/deprecated_coordination/_deprecated_sync_safety.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.sync_safety is deprecated. "
    "Import directly from sync_durability, sync_integrity, sync_stall_handler, "
    "or sync_bloom_filter instead. This module was archived December 27, 2025.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
from app.coordination.sync_durability import (
    SyncWAL,
    SyncWALEntry,
    DeadLetterQueue,
    DeadLetterEntry,
    SyncStatus,
)
# Backward-compat alias
DLQEntry = DeadLetterEntry
from app.coordination.sync_integrity import (
    check_sqlite_integrity,
    compute_file_checksum,
)
from app.coordination.sync_stall_handler import SyncStallHandler
from app.coordination.sync_bloom_filter import SyncBloomFilter, BloomFilterStats

__all__ = [
    "SyncWAL",
    "SyncWALEntry",
    "DeadLetterQueue",
    "DLQEntry",
    "SyncStatus",
    "check_sqlite_integrity",
    "compute_file_checksum",
    "SyncStallHandler",
    "SyncBloomFilter",
    "BloomFilterStats",
]
