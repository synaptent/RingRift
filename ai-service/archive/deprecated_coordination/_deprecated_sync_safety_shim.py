"""DEPRECATED: Backward compatibility shim for sync_safety.

This module was archived December 28, 2025. Import directly from
the specialized modules:

- ``app.coordination.sync_durability`` - SyncWAL, DeadLetterQueue
- ``app.coordination.sync_integrity`` - checksum validation
- ``app.coordination.sync_stall_handler`` - stall detection
- ``app.coordination.sync_bloom_filter`` - Bloom filters

For the original implementation, see:
archive/deprecated_coordination/_deprecated_sync_safety.py
"""

import warnings

warnings.warn(
    "app.coordination.sync_safety is deprecated. "
    "Import directly from sync_durability, sync_integrity, sync_stall_handler, "
    "or sync_bloom_filter instead. This module was archived December 28, 2025.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from archive for backward compatibility
from archive.deprecated_coordination._deprecated_sync_safety import (
    SyncWAL,
    SyncWALEntry,
    DeadLetterQueue,
    DLQEntry,
    SyncStatus,
    check_sqlite_integrity,
    compute_file_checksum,
    SyncStallHandler,
    SyncBloomFilter,
    BloomFilterStats,
)

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
