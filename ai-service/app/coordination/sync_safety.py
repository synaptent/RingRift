"""Unified Sync Safety Module - Single Import for All Reliability Features.

This module provides a unified namespace for all sync safety and reliability features,
consolidating functionality from four specialized modules:

1. **sync_durability.py** - Write-Ahead Log (WAL) and Dead Letter Queue (DLQ)
   - SyncWAL: Crash-safe sync operations with recovery
   - DeadLetterQueue: Track and retry permanently failed syncs
   - Connection pooling for high-performance database access
   - Automatic cleanup and statistics tracking

2. **sync_integrity.py** - Checksum validation and data integrity
   - File checksum computation (SHA256, streaming for large files)
   - SQLite database integrity verification (PRAGMA integrity_check)
   - Full sync integrity reports comparing source and target
   - Structured error reporting with detailed validation

3. **sync_stall_handler.py** - Stall detection and automatic failover
   - Detects stalled sync operations based on timeout
   - Automatic penalty tracking for unreliable sources
   - Alternative source selection for retry
   - Recovery statistics and bounded retry limits

4. **sync_bloom_filter.py** - Bloom filter for efficient P2P sync
   - Set membership testing without storing full data
   - Optimal sizing based on expected items and false positive rate
   - Compression for network transfer
   - Filter merging and intersection ratio calculation

IMPORTANT: This module does NOT duplicate code - it only re-exports from the
specialized modules. The source modules remain unchanged for backward compatibility.

Usage Examples
--------------

Basic WAL + DLQ for crash recovery::

    from app.coordination.sync_safety import SyncWAL, DeadLetterQueue

    # Write-ahead log for crash-safe operations
    wal = SyncWAL(db_path=Path("data/sync_wal.db"))
    entry_id = wal.append(
        game_id="abc123",
        source="node-5",
        target="coordinator",
        data={"moves": [...]}
    )
    wal.mark_complete(entry_id)

    # Dead letter queue for failed syncs
    dlq = DeadLetterQueue(db_path=Path("data/sync_dlq.db"))
    dlq.add(game_id="xyz789", error="Connection timeout", retry_count=3)

Integrity checking::

    from app.coordination.sync_safety import (
        compute_file_checksum,
        verify_sync_integrity,
        check_sqlite_integrity,
    )

    # Compute file checksum
    checksum = compute_file_checksum(Path("data.db"))

    # Verify sync integrity
    report = verify_sync_integrity(
        source=Path("remote/data.db"),
        target=Path("local/data.db")
    )
    if not report.is_valid:
        print(f"Sync failed: {report.errors}")

    # Check SQLite integrity
    is_valid, errors = check_sqlite_integrity(Path("data.db"))

Stall detection and failover::

    from app.coordination.sync_safety import SyncStallHandler

    handler = SyncStallHandler(stall_penalty_seconds=300, max_retries=3)

    # Check for stalls
    if handler.check_stall(sync_id, started_at, timeout=600):
        handler.record_stall(host="node-5", sync_id=sync_id)

        # Get alternative source
        alt = handler.get_alternative_source(
            exclude=["node-5"],
            all_sources=["node-1", "node-2", "node-3"]
        )
        if alt:
            handler.record_recovery(sync_id, alt)

Bloom filter for P2P sync::

    from app.coordination.sync_safety import SyncBloomFilter, create_game_id_filter

    # Create filter for known game IDs
    bf = create_game_id_filter(expected_games=10000)
    for game_id in known_game_ids:
        bf.add(game_id)

    # Serialize for P2P exchange
    data = bf.to_bytes()
    send_to_peer(data)

    # Deserialize peer's filter
    peer_bf = SyncBloomFilter.from_bytes(peer_data)

    # Find games peer doesn't have
    games_to_sync = [g for g in my_games if g not in peer_bf]

Complete sync operation example::

    from app.coordination.sync_safety import (
        SyncWAL,
        DeadLetterQueue,
        SyncStallHandler,
        verify_sync_integrity,
    )

    # Initialize components
    wal = SyncWAL(db_path=Path("data/sync_wal.db"))
    dlq = DeadLetterQueue(db_path=Path("data/sync_dlq.db"))
    stall_handler = SyncStallHandler()

    # Log sync attempt
    entry_id = wal.append(game_id, source, target, data)

    try:
        # Attempt sync with stall detection
        started_at = time.time()
        perform_sync(source, target)

        # Verify integrity
        report = verify_sync_integrity(source_path, target_path)
        if report.is_valid:
            wal.mark_complete(entry_id)
        else:
            raise ValueError(f"Integrity check failed: {report.errors}")

    except Exception as e:
        # Handle stalls and failures
        if stall_handler.check_stall(entry_id, started_at, timeout=600):
            stall_handler.record_stall(host=source, sync_id=entry_id)
            alt = stall_handler.get_alternative_source(exclude=[source])
            if alt:
                # Retry with alternative source
                pass

        # Move to DLQ after max retries
        wal.mark_failed(entry_id, str(e))
        dlq.add(game_id, source, target, error=str(e), retry_count=3)

Module Organization
-------------------

This unified module provides four logical namespaces:

**Durability (WAL + DLQ)**::
    - SyncWAL, DeadLetterQueue
    - SyncWALEntry, DeadLetterEntry
    - WALStats, DLQStats
    - SyncStatus
    - get_sync_wal(), get_dlq()

**Integrity**::
    - compute_file_checksum(), compute_db_checksum()
    - verify_checksum(), verify_sync_integrity()
    - check_sqlite_integrity()
    - IntegrityReport, IntegrityCheckResult
    - DEFAULT_CHUNK_SIZE, LARGE_CHUNK_SIZE

**Stall Handling**::
    - SyncStallHandler
    - get_stall_handler(), reset_stall_handler()

**Bloom Filters**::
    - SyncBloomFilter, BloomFilterStats
    - create_game_id_filter(), create_model_hash_filter(), create_event_dedup_filter()
    - BloomFilter (alias)

See Also
--------
- app/coordination/sync_durability.py - WAL and DLQ implementation
- app/coordination/sync_integrity.py - Integrity checking implementation
- app/coordination/sync_stall_handler.py - Stall detection implementation
- app/coordination/sync_bloom_filter.py - Bloom filter implementation
- app/distributed/unified_data_sync.py - Higher-level sync orchestration
"""

from __future__ import annotations

# =============================================================================
# Re-export from sync_durability.py (WAL + DLQ)
# =============================================================================

from app.coordination.sync_durability import (
    # Core classes
    DeadLetterQueue,
    SyncWAL,
    # Data classes
    DeadLetterEntry,
    DLQStats,
    SyncStatus,
    SyncWALEntry,
    WALStats,
    # Factory functions
    get_dlq,
    get_sync_wal,
    reset_instances,
)

# =============================================================================
# Re-export from sync_integrity.py (Checksum validation)
# =============================================================================

from app.coordination.sync_integrity import (
    # Constants
    DEFAULT_CHUNK_SIZE,
    LARGE_CHUNK_SIZE,
    # Data classes
    IntegrityCheckResult,
    IntegrityReport,
    # Core functions
    check_sqlite_integrity,
    compute_db_checksum,
    compute_file_checksum,
    verify_checksum,
    verify_sync_integrity,
)

# =============================================================================
# Re-export from sync_stall_handler.py (Stall detection/failover)
# =============================================================================

from app.coordination.sync_stall_handler import (
    # Core class
    SyncStallHandler,
    # Factory functions
    get_stall_handler,
    reset_stall_handler,
)

# =============================================================================
# Re-export from sync_bloom_filter.py (P2P Bloom filter)
# =============================================================================

from app.coordination.sync_bloom_filter import (
    # Constants
    DEFAULT_FALSE_POSITIVE_RATE,
    DEFAULT_HASH_COUNT,
    DEFAULT_SIZE,
    # Core classes
    BloomFilter,  # Alias for backward compatibility
    BloomFilterStats,
    SyncBloomFilter,
    # Factory functions
    create_event_dedup_filter,
    create_game_id_filter,
    create_model_hash_filter,
)

# =============================================================================
# Module exports (comprehensive list for IDE autocomplete)
# =============================================================================

__all__ = [
    # === Durability (sync_durability.py) ===
    "SyncWAL",
    "DeadLetterQueue",
    "SyncWALEntry",
    "DeadLetterEntry",
    "WALStats",
    "DLQStats",
    "SyncStatus",
    "get_sync_wal",
    "get_dlq",
    "reset_instances",
    # === Integrity (sync_integrity.py) ===
    "DEFAULT_CHUNK_SIZE",
    "LARGE_CHUNK_SIZE",
    "IntegrityCheckResult",
    "IntegrityReport",
    "check_sqlite_integrity",
    "compute_db_checksum",
    "compute_file_checksum",
    "verify_checksum",
    "verify_sync_integrity",
    # === Stall Handling (sync_stall_handler.py) ===
    "SyncStallHandler",
    "get_stall_handler",
    "reset_stall_handler",
    # === Bloom Filter (sync_bloom_filter.py) ===
    "SyncBloomFilter",
    "BloomFilterStats",
    "BloomFilter",  # Alias
    "DEFAULT_SIZE",
    "DEFAULT_HASH_COUNT",
    "DEFAULT_FALSE_POSITIVE_RATE",
    "create_game_id_filter",
    "create_model_hash_filter",
    "create_event_dedup_filter",
]


# =============================================================================
# Module-level documentation
# =============================================================================

def get_module_summary() -> dict[str, list[str]]:
    """Get a summary of available functionality by category.

    Returns:
        Dict mapping category name to list of available classes/functions

    Example:
        >>> from app.coordination.sync_safety import get_module_summary
        >>> summary = get_module_summary()
        >>> print(summary['durability'])
        ['SyncWAL', 'DeadLetterQueue', ...]
    """
    return {
        "durability": [
            "SyncWAL",
            "DeadLetterQueue",
            "SyncWALEntry",
            "DeadLetterEntry",
            "WALStats",
            "DLQStats",
            "SyncStatus",
            "get_sync_wal",
            "get_dlq",
        ],
        "integrity": [
            "compute_file_checksum",
            "compute_db_checksum",
            "verify_checksum",
            "verify_sync_integrity",
            "check_sqlite_integrity",
            "IntegrityReport",
            "IntegrityCheckResult",
        ],
        "stall_handling": [
            "SyncStallHandler",
            "get_stall_handler",
            "reset_stall_handler",
        ],
        "bloom_filter": [
            "SyncBloomFilter",
            "BloomFilterStats",
            "create_game_id_filter",
            "create_model_hash_filter",
            "create_event_dedup_filter",
        ],
    }
