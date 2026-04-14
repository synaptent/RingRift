# Resilient Transfer Guide

This guide documents the file transfer infrastructure for RingRift's distributed training cluster, with a focus on preventing data corruption during transfers.

## Background

In December 2025, a 955MB NPZ training file was corrupted during transfer despite rsync "succeeding". The root causes were:

1. **rsync with `--partial`** stitched together corrupted segments after 100+ restarts
2. **No post-transfer checksum verification** (only exit code checking)
3. **No NPZ-specific validation** (array shapes not verified)
4. **Corrupted files remained in place** (no quarantine mechanism)

This guide documents the solutions implemented to prevent such issues.

## Architecture Overview

```
                     TRANSFER REQUEST
                           │
         ┌─────────────────┴─────────────────┐
         │       SIZE-BASED ROUTING          │
         │  > 50MB → BitTorrent (primary)    │
         │  < 50MB → aria2/rsync             │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │    PRE-TRANSFER CHECKSUM          │
         │  Source computes: SHA256 + size   │
         │  NPZ: + array shapes              │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │       TRANSFER EXECUTION          │
         │  BitTorrent: piece-level SHA1     │
         │  aria2: multi-source + checksum   │
         │  rsync: --checksum flag           │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │   MANDATORY POST-VERIFICATION     │
         │  1. File size match               │
         │  2. SHA256 checksum match         │
         │  3. NPZ: array shape validation   │
         │  4. SQLite: PRAGMA integrity      │
         │  FAIL → Quarantine + Retry        │
         └───────────────────────────────────┘
```

## Key Components

### 1. Verified Rsync Functions (`app/distributed/sync_utils.py`)

The `rsync_file_verified` and `rsync_push_file_verified` functions add mandatory post-transfer verification:

```python
from app.distributed.sync_utils import rsync_file_verified, rsync_push_file_verified
from app.distributed.hosts import HostConfig

# Pull a file with verification
host = HostConfig(name="node-1", ssh_host="10.0.0.1", ssh_user="ubuntu", ...)
result = rsync_file_verified(
    host=host,
    remote_path="/path/to/remote/file.npz",
    local_path=Path("/local/path/file.npz"),
    expected_checksum="sha256:abc123...",  # Optional
    timeout=120,
)

if result.success and result.verified:
    print("Transfer complete with verified integrity")
elif result.success:
    print(f"Transfer complete but unverified: {result.error}")
else:
    print(f"Transfer failed: {result.error}")

# Push a file with verification
result = rsync_push_file_verified(
    host=host,
    local_path=Path("/local/model.pth"),
    remote_path="/remote/path/model.pth",
    timeout=120,
)
```

**Key features:**

- Adds `--checksum` flag to rsync commands
- Computes SHA256 before and after transfer
- Quarantines corrupted files automatically
- Returns `TransferVerificationResult` with detailed status

### 2. NPZ Structure Validation (`app/coordination/npz_validation.py`)

Beyond checksum verification, NPZ files need structure validation to catch corruption that produces valid-looking but wrong data:

```python
from app.coordination.npz_validation import validate_npz_structure, NPZValidationResult

result = validate_npz_structure(Path("training_data.npz"))

if result.valid:
    print(f"Valid NPZ: {result.sample_count} samples")
    print(f"Arrays: {result.array_shapes}")
else:
    for error in result.errors:
        print(f"ERROR: {error}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
```

**Validation checks:**

- File opens successfully as NPZ
- Required arrays present (`features`, `values`, `policy_*`)
- Sample counts consistent across arrays
- Data types are correct
- No truncated/corrupted array data

### 3. BitTorrent for Large Files (`app/distributed/aria2_transport.py`)

BitTorrent provides piece-level SHA1 verification, making it ideal for large files:

```python
from app.distributed.aria2_transport import Aria2Transport, Aria2Config

transport = Aria2Transport(Aria2Config(
    prefer_torrent_for_large_files=True,
    large_file_threshold_bytes=50_000_000,  # 50MB
    enable_bittorrent=True,
    bt_enable_dht=True,
    bt_enable_lpd=True,
))

# Will automatically use BitTorrent for files > 50MB
success, bytes_transferred, error = await transport.download_with_torrent_fallback(
    file_path="training/hex8_2p.npz",
    output_dir=Path("data/training"),
    expected_size=200_000_000,  # 200MB - will trigger BitTorrent
    expected_checksum="sha256:abc123...",
)
```

**BitTorrent benefits:**

- Piece-level SHA1 verification (catches corruption rsync misses)
- Multi-peer downloads for redundancy
- Resume capability across connection drops
- DHT for trackerless peer discovery in the cluster

### 4. Unified Resilient Transfer (`app/distributed/resilient_transfer.py`)

A unified abstraction that handles all the above:

```python
from app.distributed.resilient_transfer import ResilientTransfer, TransferRequest

transfer = ResilientTransfer()

result = await transfer.transfer(TransferRequest(
    source_node="training-node-1",
    source_path="/path/to/file.npz",
    target_path=Path("/local/file.npz"),
    expected_checksum="sha256:abc123...",
    expected_size=200_000_000,
    file_type="npz",  # Enables NPZ-specific validation
    priority="high",
))

if result.success and result.verification_passed:
    print(f"Transfer complete via {result.transport_used}")
else:
    print(f"Transfer failed after {result.retries} retries: {result.error}")
```

### 5. Unified Distribution with BitTorrent (`app/coordination/unified_distribution_daemon.py`)

The unified distribution daemon now uses BitTorrent for large models:

```python
from app.coordination.unified_distribution_daemon import (
    DistributionConfig,
    get_distribution_daemon,
)

config = DistributionConfig(
    use_bittorrent_for_large_files=True,
    bittorrent_threshold_bytes=50_000_000,  # 50MB
    verify_checksums=True,
)

daemon = get_distribution_daemon(config)
await daemon.start()
```

**Transport priority:**

1. BitTorrent for files > 50MB (piece-level verification)
2. HTTP streaming (fast for smaller files)
3. rsync fallback (reliable but slower)

The same daemon handles NPZ distribution after `NPZ_EXPORT_COMPLETE` events, so
model and training-data delivery share the same transport and verification path.

### 6. Per-Provider Bandwidth Limits (`app/coordination/sync_bandwidth.py`)

Provider-specific bandwidth limits prevent rate limiting and connection resets:

```python
PROVIDER_BANDWIDTH_HINTS = {
    "lambda": 100000,   # 100 MB/s
    "runpod": 100000,   # 100 MB/s
    "nebius": 50000,    # 50 MB/s (has rate limits)
    "vast": 50000,      # 50 MB/s (varies by instance)
    "vultr": 80000,     # 80 MB/s
    "hetzner": 80000,   # 80 MB/s
    "default": 20000,   # 20 MB/s (conservative)
}
```

The AutoSyncDaemon broadcast strategy (`auto_sync_daemon.py` + `sync_push_mixin.py`) automatically applies per-provider limits:

```python
from app.coordination.auto_sync_daemon import get_cluster_data_sync_daemon

# Broadcast strategy daemon exposes get_bandwidth_for_node()
# Note: get_cluster_data_sync_daemon() is deprecated; it returns AutoSyncDaemon
# with strategy="broadcast".
daemon = get_cluster_data_sync_daemon()
bandwidth = daemon.get_bandwidth_for_node("nebius-h100-1", provider="nebius")  # 50000 for Nebius
```

## Quarantine Mechanism

Corrupted files are automatically quarantined:

```python
from app.distributed.sync_utils import _quarantine_file

# Moves file to quarantine directory with timestamp
quarantine_path = _quarantine_file(
    file_path=Path("/data/corrupted.npz"),
    reason="checksum_mismatch",
)
# Result: /data/.quarantine/corrupted.npz.20251226_143000.checksum_mismatch
```

**Quarantine directory:** `.quarantine/` in the same parent directory

**Retention:** Files older than 7 days are automatically cleaned up

## Best Practices

### 1. Always Use Verified Functions for Critical Data

```python
# DON'T: Raw rsync without verification
subprocess.run(["rsync", "-avz", source, target])

# DO: Verified rsync with checksum validation
result = rsync_push_file_verified(host, source, target)
if not result.verified:
    raise TransferError(result.error)
```

### 2. Validate NPZ Files After Export

```python
# After exporting training data
np.savez_compressed(output_path, **arrays)

# Always validate
result = validate_npz_structure(Path(output_path))
if not result.valid:
    raise ValueError(f"Export produced corrupted NPZ: {result.errors}")
```

### 3. Let Large Files Use BitTorrent

Don't force HTTP for large files - BitTorrent's piece verification catches corruption:

```python
# DON'T: Force HTTP for large files
await transport.download_file(url, output)

# DO: Let the transport choose based on file size
await transport.download_with_torrent_fallback(
    file_path, output_dir, expected_size=file_size
)
```

### 4. Handle Verification Failures Gracefully

```python
result = await transfer.transfer(request)

if not result.success:
    logger.error(f"Transfer failed: {result.error}")
    # The file was already quarantined, trigger retry from different source

elif not result.verification_passed:
    logger.warning(f"Transfer unverified: {result.error}")
    # Consider retry or manual verification
```

## Monitoring

### Key Metrics

The following metrics are tracked for monitoring:

- `checksum_failures` - Number of checksum mismatches detected
- `npz_validation_failures` - Number of NPZ structure validation failures
- `quarantine_count` - Number of files moved to quarantine
- `bittorrent_usage_ratio` - Percentage of large file transfers using BitTorrent
- `verification_latency_ms` - Time spent on post-transfer verification

### Health Checks

```python
from app.coordination.unified_distribution_daemon import UnifiedDistributionDaemon

daemon = UnifiedDistributionDaemon()
health = daemon.health_check()

if not health.healthy:
    print(f"Daemon unhealthy: {health.message}")
    if health.details.get("checksum_failures", 0) > 5:
        print("High checksum failure rate - check network stability")
```

## Troubleshooting

### "Checksum mismatch" after transfer

**Cause:** File was corrupted during transfer (likely due to network issues or rsync --partial stitching)

**Solution:**

1. Check quarantine directory for the corrupted file
2. Retry transfer from a different source node
3. If issue persists, check network stability to that node

### "NPZ validation failed: Inconsistent sample counts"

**Cause:** NPZ file has arrays with different numbers of samples (corruption or export bug)

**Solution:**

1. Check the export script for bugs
2. Re-export from the original database
3. If source database is corrupted, restore from backup

### "BitTorrent transfer timeout"

**Cause:** No active seeders or network issues

**Solution:**

1. Check if source node has the .torrent file and is seeding
2. Verify BitTorrent ports (51413, 6881) are open
3. Check DHT connectivity in cluster
4. Fall back to HTTP if BitTorrent unavailable

### "Transfer speed very slow"

**Cause:** Provider bandwidth limits or network congestion

**Solution:**

1. Check PROVIDER_BANDWIDTH_HINTS for the target node
2. Verify network path between nodes
3. Consider using BitTorrent for multi-source downloads

## Related Files

| File                                              | Purpose                            |
| ------------------------------------------------- | ---------------------------------- |
| `app/distributed/sync_utils.py`                   | Verified rsync functions           |
| `app/coordination/npz_validation.py`              | NPZ structure validation           |
| `app/distributed/aria2_transport.py`              | BitTorrent + HTTP transport        |
| `app/distributed/resilient_transfer.py`           | Unified transfer abstraction       |
| `app/coordination/unified_distribution_daemon.py` | Model distribution with BT         |
| `app/coordination/sync_bandwidth.py`              | Provider bandwidth hints           |
| `app/coordination/auto_sync_daemon.py`            | Cluster data sync (broadcast)      |
| `app/coordination/sync_integrity.py`              | Integrity verification utilities   |
| `scripts/cluster_file_sync.py`                    | CLI file sync with verification    |
| `scripts/auto_deploy_models.py`                   | Model deployment with verification |
| `scripts/auto_promote.py`                         | Model promotion with verification  |
| `scripts/export_replay_dataset.py`                | NPZ export with validation         |

## Changelog

### December 2025

- Added post-transfer checksum verification to sync_utils.py
- Created npz_validation.py for NPZ structure validation
- Created resilient_transfer.py unified abstraction
- Added BitTorrent preference for large files (>50MB)
- Updated model_distribution_daemon with BitTorrent support
- Added per-provider bandwidth limits to sync_push_mixin.py (AutoSyncDaemon broadcast strategy)
- Updated auto_deploy_models.py and auto_promote.py to use verified rsync
- Added NPZ validation after export in export_replay_dataset.py
- Added checksum verification to cluster_file_sync.py
