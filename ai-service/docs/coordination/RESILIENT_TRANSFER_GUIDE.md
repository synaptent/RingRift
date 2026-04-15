# Resilient Transfer Guide

This guide documents the resilient file transfer infrastructure for cluster-wide data synchronization.

## Overview

The RingRift cluster uses a multi-transport, verification-mandatory transfer system to prevent data corruption during sync operations.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSFER REQUEST                                  │
│  File size > 500MB? → REQUIRE BitTorrent (piece-level verification) │
│  File size > 50MB? → Try BitTorrent first                           │
│  File size < 50MB? → Use aria2/rsync with mandatory verification    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────────┐
│                PRE-TRANSFER CHECKSUM EXCHANGE                        │
│  Source: compute SHA256 + file size + (NPZ: array shapes)           │
│  Send metadata to target BEFORE transfer begins                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────────┐
│                     TRANSFER EXECUTION                               │
│  BitTorrent: Piece-level SHA1 verification (automatic)              │
│  aria2 HTTP: Multi-source with checksum flag                        │
│  rsync: --checksum flag + post-transfer verification                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────────┐
│              MANDATORY POST-TRANSFER VERIFICATION                    │
│  1. File size match                                                  │
│  2. SHA256 checksum match                                            │
│  3. NPZ: Array shape validation (features, policy, value)           │
│  4. SQLite: PRAGMA integrity_check                                   │
│  FAIL → Quarantine file → Retry from different source               │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. ResilientTransfer (`app/distributed/resilient_transfer.py`)

Unified entry point for all file transfers with automatic transport selection and mandatory verification.

```python
from app.distributed.resilient_transfer import (
    ResilientTransfer,
    TransferRequest,
    TransferResult,
    transfer_file,  # Convenience function
)

# Simple usage
result = await transfer_file(
    source_node="runpod-h100",
    source_path="/workspace/ringrift/ai-service/data/training/hex8_2p.npz",
    target_path=Path("data/training/hex8_2p.npz"),
    file_type="npz",
)

if result.success:
    print(f"Transferred {result.bytes_transferred} bytes via {result.transport_used}")
else:
    print(f"Transfer failed: {result.error}")

# Full control
transfer = ResilientTransfer()
request = TransferRequest(
    source_node="runpod-h100",
    source_path="/path/to/large_model.pth",
    target_path=Path("models/large_model.pth"),
    expected_checksum="sha256:abc123...",  # Optional, will be fetched if not provided
    expected_size=955_000_000,
    file_type="pth",
    priority="high",
)
result = await transfer.transfer(request)
```

**Transport Selection Logic:**

| File Size | Primary Transport      | Fallback      | Verification           |
| --------- | ---------------------- | ------------- | ---------------------- |
| > 500MB   | BitTorrent (required)  | aria2 → rsync | Piece-level + SHA256   |
| 50-500MB  | BitTorrent (try first) | aria2 → rsync | SHA256 + type-specific |
| < 50MB    | aria2                  | rsync         | SHA256 + type-specific |

### 2. NPZ Validation (`app/coordination/npz_validation.py`)

Deep validation for NPZ training data files that catches corruption checksums miss.

```python
from app.coordination.npz_validation import (
    validate_npz_structure,
    validate_npz_for_training,
    quick_npz_check,
    NPZValidationResult,
)

# Deep validation
result = validate_npz_structure(Path("data/training/hex8_2p.npz"))
if not result.valid:
    print(f"Validation failed: {result.errors}")
else:
    print(f"Valid NPZ with {result.sample_count} samples")
    print(f"Array shapes: {result.array_shapes}")

# Training-specific validation (includes board/player checks)
result = validate_npz_for_training(
    Path("data/training/hex8_2p.npz"),
    board_type="hex8",
    num_players=2,
)

# Quick check for hot paths
is_ok = quick_npz_check(Path("data/training/hex8_2p.npz"))
```

**Validation Checks:**

- File opens as valid NPZ (np.load)
- Required arrays exist: `features`, `values`, optionally `policy_*`
- Array shapes consistent (all have same sample count)
- Array dimensions reasonable (max 1B limit catches corruption)
- Data types correct for each array

### 3. Sync Integrity (`app/coordination/sync_integrity.py`)

Comprehensive integrity verification for all file types.

```python
from app.coordination.sync_integrity import (
    compute_file_checksum,
    verify_checksum,
    check_sqlite_integrity,
    verify_sync_integrity,
    verified_database_copy,
    IntegrityReport,
)

# Compute checksum
checksum = compute_file_checksum(Path("models/model.pth"))

# Verify after transfer
is_match = verify_checksum(Path("models/model.pth"), expected_checksum)

# SQLite-specific integrity
valid, errors = check_sqlite_integrity(Path("data/games/selfplay.db"))

# Full sync verification report
report = verify_sync_integrity(
    source_path=Path("/remote/data.db"),
    target_path=Path("data/local.db"),
)
if not report.is_valid:
    print(f"Sync failed: {report.summary()}")

# Safe database transfer (VACUUM + WAL checkpoint + atomic copy)
success = await verified_database_copy(
    source=Path("/remote/games.db"),
    target=Path("data/games.db"),
)
```

### 4. Transfer Verification (`app/coordination/transfer_verification.py`)

Quarantine mechanism for corrupted files.

```python
from app.coordination.transfer_verification import (
    quarantine_file,
    get_quarantine_path,
    list_quarantined_files,
)

# Quarantine a corrupted file (moves to quarantine directory)
quarantine_file(Path("data/training/corrupted.npz"), reason="checksum_mismatch")

# List quarantined files
for qf in list_quarantined_files():
    print(f"{qf.original_path} - {qf.reason} - {qf.quarantine_time}")
```

## Bandwidth Management

### Provider-Specific Limits

```python
from app.coordination.sync_bandwidth import PROVIDER_BANDWIDTH_HINTS

# Default bandwidth limits (MB/s)
PROVIDER_BANDWIDTH_HINTS = {
    "runpod": 100,
    "nebius": 50,   # Conservative to avoid rate limiting
    "lambda": 100,
    "vast": 50,
    "hetzner": 80,
    "vultr": 50,
}
```

### Bandwidth-Coordinated Rsync

```python
from app.coordination.sync_bandwidth import BandwidthCoordinatedRsync

rsync = BandwidthCoordinatedRsync()
success = await rsync.sync_file(
    host="runpod-h100",
    source="/remote/file.npz",
    target=Path("local/file.npz"),
    bandwidth_limit_mbps=50,  # Optional, uses provider hint if not specified
)
```

## BitTorrent Transport

### When BitTorrent is Used

- **Required**: Files > 500MB (piece-level verification prevents corruption)
- **Preferred**: Files 50-500MB (faster for large files with multiple seeders)
- **Fallback**: aria2/rsync if torrent unavailable

### Torrent Generation

```python
from app.distributed.torrent_manager import TorrentManager

manager = TorrentManager()

# Create torrent for large file
torrent_path = await manager.create_torrent(
    file_path=Path("data/training/large_dataset.npz"),
    announce_urls=["http://coordinator:8770/announce"],
)

# Register in cluster manifest
from app.distributed.cluster_manifest import get_cluster_manifest
manifest = get_cluster_manifest()
manifest.register_torrent(torrent_path, seeders=["runpod-h100", "nebius-h100"])
```

### DHT Bootstrap

Torrents use DHT with bootstrap nodes for decentralized peer discovery:

```python
DHT_BOOTSTRAP_NODES = [
    ("router.bittorrent.com", 6881),
    ("dht.transmissionbt.com", 6881),
]
```

## Event Integration

### Transfer Events

The transfer system emits events for pipeline integration:

```python
from app.distributed.data_events import DataEventType

# Events emitted during transfer
DataEventType.DATA_SYNC_COMPLETED  # Successful transfer
DataEventType.DATA_SYNC_FAILED     # Transfer failed
DataEventType.SYNC_STALLED         # Transfer stalled/timeout
```

### Subscribing to Transfer Events

```python
from app.coordination.event_router import get_router

router = get_router()

@router.subscribe(DataEventType.DATA_SYNC_COMPLETED)
async def on_sync_complete(event):
    print(f"Transfer complete: {event.payload['file_path']}")
    print(f"Checksum verified: {event.payload['checksum_verified']}")
```

## Common Workflows

### 1. Distribute Model After Promotion

```python
# December 2025: Use unified_distribution_daemon (consolidates model + NPZ)
from app.coordination.unified_distribution_daemon import (
    DistributionConfig,
    create_unified_distribution_daemon,
    verify_model_distribution,
)

# Preferred path: use the unified factory, not the deprecated split factories
config = DistributionConfig(
    verify_checksums=True,
    validate_npz_structure=True,
)
daemon = create_unified_distribution_daemon(config)
await daemon.start()

# Distribution is event-driven after MODEL_PROMOTED
success, node_count = await verify_model_distribution("models/canonical_hex8_2p.pth")
print(f"Model visible on {node_count} nodes (ok={success})")
```

### 2. Distribute NPZ Training Data

```python
from app.coordination.unified_distribution_daemon import (
    DistributionConfig,
    get_distribution_daemon,
)

# The same unified daemon automatically subscribes to NPZ_EXPORT_COMPLETE
daemon = get_distribution_daemon(
    DistributionConfig(validate_npz_structure=True)
)
await daemon.start()
```

### 3. Manual Verified Transfer

```python
from app.distributed.sync_utils import rsync_file_verified

success, error = await rsync_file_verified(
    host="runpod-h100",
    remote_path="/workspace/ringrift/ai-service/data/games/selfplay.db",
    local_path=Path("data/games/selfplay.db"),
    expected_checksum="sha256:abc123...",  # Optional
    timeout=300,
)

if not success:
    print(f"Transfer failed: {error}")
```

## Troubleshooting

### Checksum Mismatch

```bash
# Check quarantine directory
ls -la data/quarantine/

# View quarantine manifest
cat data/quarantine/manifest.json
```

### Transfer Timeout

```python
# Increase timeout for large files
result = await transfer_file(
    source_node="runpod-h100",
    source_path="/path/to/huge_file.npz",
    target_path=Path("local/huge_file.npz"),
    timeout=600,  # 10 minutes
)
```

### BitTorrent Unavailable

If BitTorrent fails, transfers automatically fall back to aria2/rsync:

```
[WARN] BitTorrent unavailable for file.npz, falling back to aria2
[INFO] Transferred via aria2 with checksum verification
```

## Configuration

### Environment Variables

| Variable                             | Default         | Description                              |
| ------------------------------------ | --------------- | ---------------------------------------- |
| `RINGRIFT_TRANSFER_TIMEOUT`          | 300             | Default transfer timeout (seconds)       |
| `RINGRIFT_TRANSFER_RETRIES`          | 3               | Number of retry attempts                 |
| `RINGRIFT_BITTORRENT_REQUIRED_SIZE`  | 500MB           | Size above which BitTorrent is required  |
| `RINGRIFT_BITTORRENT_PREFERRED_SIZE` | 50MB            | Size above which BitTorrent is preferred |
| `RINGRIFT_QUARANTINE_DIR`            | data/quarantine | Directory for quarantined files          |

## Key Files Reference

| File                                              | Purpose                             |
| ------------------------------------------------- | ----------------------------------- |
| `app/distributed/resilient_transfer.py`           | Unified transfer abstraction        |
| `app/coordination/npz_validation.py`              | NPZ structure validation            |
| `app/coordination/sync_integrity.py`              | Checksum & SQLite verification      |
| `app/coordination/transfer_verification.py`       | Quarantine mechanism                |
| `app/coordination/sync_bandwidth.py`              | Bandwidth coordination              |
| `app/distributed/torrent_manager.py`              | BitTorrent integration              |
| `app/coordination/unified_distribution_daemon.py` | Model + NPZ distribution (Dec 2025) |

## December 2025 Enhancements

- **Mandatory checksum verification**: All transfers now verify SHA256 post-transfer
- **NPZ structure validation**: Catches corruption that checksums miss (array shapes, sample counts)
- **rsync --checksum flag**: Added to all shell scripts and Python rsync calls
- **BitTorrent preference**: Large files (>50MB) prefer BitTorrent for piece-level verification
- **Quarantine mechanism**: Corrupted files automatically quarantined for investigation
