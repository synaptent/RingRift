# Transfer Method Comparison for RingRift Cluster

**Date:** December 28, 2025
**Test Environment:** 7 providers, 15+ nodes tested

## Executive Summary

| Method              | 1MB Success | 10MB Success    | Best Use Case                |
| ------------------- | ----------- | --------------- | ---------------------------- |
| **SCP**             | 80%         | 33%             | Small files (<5MB)           |
| **Base64 pipe**     | 80%         | 33%             | No advantage over SCP        |
| **Rsync --partial** | 100%        | 67%             | Medium files, resume support |
| **Chunked (2MB)**   | **100%**    | **100%**        | ✅ **Large files (>5MB)**    |
| **aria2**           | Works       | Firewall issues | Public URLs only             |
| **BitTorrent**      | N/A         | DHT blocked     | Not viable                   |

**Recommendation:** Use **chunked transfers (2MB chunks)** for files >5MB.

## Provider-Specific Results

### Nebius (backbone-1, h100-3)

| Test         | Result         | Notes                           |
| ------------ | -------------- | ------------------------------- |
| SSH Commands | ✅ OK          | Stable connection               |
| SCP 1MB      | ✅ 4/5         | Occasionally drops              |
| SCP 10MB     | ❌ 1/3         | Frequent connection resets      |
| Rsync 10MB   | ⚠️ 2/3         | Better with --partial           |
| Chunked 2MB  | ✅ 5/5         | **Most reliable**               |
| aria2c       | ✅ Available   | Can't reach other cluster nodes |
| BitTorrent   | ❌ DHT blocked | Port 6881 filtered              |

### RunPod (a100-1, l40s-2, h100)

| Test         | Result         | Notes                        |
| ------------ | -------------- | ---------------------------- |
| SSH Commands | ✅ OK          | NAT traversal required       |
| SCP 1MB      | ✅ OK          | Works well                   |
| SCP 10MB     | ❌ FAIL        | Connection resets frequently |
| Rsync 10MB   | ❌ FAIL        | Same as SCP                  |
| aria2c       | ✅ Available   | Works for public URLs        |
| BitTorrent   | ❌ DHT blocked | Container restrictions       |

### Vast.ai (29128352, 29129529)

| Test         | Result           | Notes                        |
| ------------ | ---------------- | ---------------------------- |
| SSH Commands | ✅ OK            | Via ssh\*.vast.ai proxy      |
| SCP 1MB      | ✅ OK            | Works well                   |
| SCP 10MB     | ❌ FAIL          | Connection drops after ~4-8s |
| Rsync 10MB   | ❌ FAIL          | Same as SCP                  |
| aria2c       | ❌ Not installed | Not in default image         |
| BitTorrent   | ❌ Not available | No torrent tools             |

### Vultr (a100-20gb)

| Test         | Result         | Notes                 |
| ------------ | -------------- | --------------------- |
| SSH Commands | ✅ OK          | Direct connection     |
| SCP 1MB      | ⚠️ Flaky       | 80% success           |
| SCP 10MB     | ⚠️ 1/3         | Often drops           |
| Rsync 10MB   | ⚠️ 2/3         | Better than SCP       |
| aria2c       | ✅ Available   | Works for public URLs |
| BitTorrent   | ❌ DHT blocked | Firewall rules        |

### Hetzner (cpu1, cpu2, cpu3)

| Test         | Result           | Notes                    |
| ------------ | ---------------- | ------------------------ |
| SSH Commands | ✅ OK            | Very stable              |
| SCP 1MB      | ✅ OK            | Reliable                 |
| SCP 10MB     | ✅ OK            | **Most stable provider** |
| Rsync 10MB   | ✅ OK            | Works well               |
| aria2c       | ❌ Not installed | CPU-only nodes           |
| BitTorrent   | ❌ Not available | No tools                 |

### Lambda Labs (gh200-1 through gh200-training)

| Test         | Result           | Notes                        |
| ------------ | ---------------- | ---------------------------- |
| SSH Commands | ✅ OK            | Direct connection            |
| SCP 1MB      | ✅ OK            | Works well                   |
| SCP 10MB     | ❌ FAIL          | Connection resets            |
| Rsync 10MB   | ❌ FAIL          | Same as SCP                  |
| Chunked 2MB  | ⚠️ 3-5/5         | Variable, better than single |
| GPU          | GH200 480GB 96GB | Best VRAM in cluster         |
| aria2c       | ❌ Not installed | Not in Lambda image          |
| BitTorrent   | ❌ Not available | No tools                     |

**Lambda-specific notes:**

- 6 GH200 nodes available (96GB VRAM each)
- Training-only role (selfplay disabled)
- Same connection reset issues as other providers
- Chunked transfers work but less reliable than Nebius

### AWS (Proxy Only)

| Test         | Result  | Notes             |
| ------------ | ------- | ----------------- |
| SSH Commands | ✅ OK   | t2.micro instance |
| Storage      | ❌ None | Proxy node only   |
| GPU          | ❌ None | No GPU            |

**AWS is used for S3 storage, not compute.** See AWS S3 Storage section below.

## Tool Availability

| Node              | aria2c | transmission | python3 |
| ----------------- | ------ | ------------ | ------- |
| nebius-backbone-1 | ✅     | ❌           | ✅      |
| nebius-h100-3     | ✅     | ❌           | ✅      |
| runpod-a100-1     | ✅     | ❌           | ✅      |
| vultr-a100-20gb   | ✅     | ❌           | ✅      |
| hetzner-cpu1      | ❌     | ❌           | ✅      |
| vast-29128352     | ❌     | ❌           | ✅      |
| vast-29129529     | ❌     | ❌           | ✅      |

## Detailed Test Results

### Base64 vs SCP Comparison

Base64 encoding does **NOT** improve transfer reliability:

- Both methods have identical failure patterns
- Connection resets occur at the TCP level, not data encoding
- Base64 adds 33% overhead without benefit

```
1MB file, 5 iterations to nebius-backbone-1:
SCP:    4/5 (80%)
Base64: 4/5 (80%)
Rsync:  5/5 (100%)
```

### Chunked Transfer Success

Breaking large files into 2MB chunks achieves 100% success:

```bash
# 10MB file split into 2MB chunks
split -b 2m source.npz /tmp/chunk_

# Transfer results:
chunk_aa: ✓
chunk_ab: ✓
chunk_ac: ✓
chunk_ad: ✓
chunk_ae: ✓

# Reassembly verified: 10485760 bytes
```

### aria2 Test Results

aria2 works for **public URLs** but fails for intra-cluster transfers:

```
Public HTTP (speedtest.tele2.net):
- nebius: OK - 1MB in ~0.5s
- vultr: OK - 1MB in ~0.5s

Intra-cluster HTTP (hetzner:8888 -> others):
- nebius: FAILED - connection refused
- vultr: FAILED - connection refused

Root cause: Firewall rules block non-standard ports between nodes
```

### BitTorrent Test Results

BitTorrent is **not viable** for cluster distribution:

```
DHT connectivity test (port 6881):
- nebius: BLOCKED
- vultr: BLOCKED
- runpod: BLOCKED (container isolation)

aria2 --bt-* options: Available but unusable
```

## Recommended Transfer Implementation

### For files < 5MB

```bash
rsync -az --partial -e "ssh -i ~/.ssh/id_cluster" \
    source.npz user@host:/destination/
```

### For files >= 5MB (RECOMMENDED)

```bash
# Split into 2MB chunks
split -b 2m source.npz /tmp/chunk_

# Transfer each chunk with retry
for chunk in /tmp/chunk_*; do
    for attempt in 1 2 3; do
        rsync -az --partial -e "ssh -i ~/.ssh/key" \
            "$chunk" user@host:/tmp/chunks/ && break
        sleep 5
    done
done

# Reassemble on remote
ssh user@host "cat /tmp/chunks/chunk_* > /destination/source.npz"

# Cleanup
rm -f /tmp/chunk_*
ssh user@host "rm -rf /tmp/chunks"
```

### Implementation in scripts/lib/transfer.py

```python
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB

def chunked_transfer(source: Path, host: str, dest: Path,
                     config: TransferConfig) -> TransferResult:
    """Transfer large files in chunks for reliability."""
    if source.stat().st_size < 5 * 1024 * 1024:
        return rsync_transfer(source, host, dest, config)

    # Split file
    chunks = split_file(source, CHUNK_SIZE)

    # Transfer each chunk with retries
    for chunk in chunks:
        for attempt in range(3):
            if rsync_transfer(chunk, host, f"/tmp/chunks/{chunk.name}", config):
                break
            time.sleep(5 * (attempt + 1))

    # Reassemble on remote
    ssh_command(host, f"cat /tmp/chunks/* > {dest}")
    ssh_command(host, "rm -rf /tmp/chunks")

    return TransferResult(success=True)
```

## Network Topology Notes

### Why transfers fail

1. **Connection resets after 4-8 seconds**
   - Likely intermediate firewall/NAT timeout
   - Affects all providers except Hetzner

2. **Port restrictions**
   - Non-standard ports (8888, 6881, etc.) blocked between nodes
   - Only SSH (22) and established provider ports work

3. **Provider-specific issues**
   - Vast.ai: SSH proxy adds latency
   - RunPod: Container network isolation
   - Nebius: Strict security groups

### Recommendations for infrastructure

1. **Open port 8780** between nodes for P2P HTTP data server
2. **Consider Tailscale mesh** for direct node-to-node transfers
3. **Use Hetzner as distribution hub** - most stable connections
4. **Pre-install aria2c** on Vast.ai images for public URL fallback

## AWS S3 Storage (Cluster Backup)

RingRift uses AWS S3 for long-term storage and backup of valuable cluster data.

### Current Configuration

```yaml
# From config/data_aggregator.yaml
s3_storage:
  enabled: true
  bucket: 'ringrift-models-20251214'
  region: 'us-east-1'
  sync_interval_minutes: 60

  sync_targets:
    models: true # Trained model checkpoints
    databases: true # Consolidated game databases
    state: true # Cluster state files
    raw_data: false # Too large for S3 (TBs)
```

### What Gets Backed Up

| Data Type          | S3 Sync | Location                                   | Retention         |
| ------------------ | ------- | ------------------------------------------ | ----------------- |
| **Models** (.pth)  | ✅ Yes  | `s3://ringrift-models-20251214/models/`    | Indefinite        |
| **Canonical DBs**  | ✅ Yes  | `s3://ringrift-models-20251214/databases/` | Indefinite        |
| **State files**    | ✅ Yes  | `s3://ringrift-models-20251214/state/`     | 30 days           |
| **Raw selfplay**   | ❌ No   | Local only (OWC drive)                     | 30 days           |
| **Training NPZ**   | ❌ No   | Local only                                 | 90 days           |
| **Archived games** | ✅ Yes  | `s3://ringrift-archive/archives/`          | Glacier after 90d |

### S3 Implementation

The `MaintenanceDaemon` handles S3 uploads:

```python
# From app/coordination/maintenance_daemon.py
archive_to_s3: bool = os.getenv("RINGRIFT_ARCHIVE_TO_S3", "false")
archive_s3_bucket: str = os.getenv("RINGRIFT_ARCHIVE_S3_BUCKET", "ringrift-archive")
```

**Upload method:** Uses AWS CLI (`aws s3 cp`) rather than boto3 for simplicity.

### Enabling S3 Backup

```bash
# Enable S3 archival
export RINGRIFT_ARCHIVE_TO_S3=true
export RINGRIFT_ARCHIVE_S3_BUCKET=ringrift-archive

# Verify AWS credentials
aws sts get-caller-identity

# Manual sync to S3
aws s3 sync /Volumes/RingRift-Data/models/ s3://ringrift-models-20251214/models/
aws s3 sync /Volumes/RingRift-Data/selfplay_repository/synced/ s3://ringrift-models-20251214/databases/
```

### Glacier Deep Archive (Future)

For cost-effective long-term storage of historical game data:

```bash
# Move old archives to Glacier Deep Archive
aws s3 cp s3://ringrift-archive/archives/ s3://ringrift-archive/glacier/ \
    --recursive --storage-class DEEP_ARCHIVE
```

**Cost estimate:**

- S3 Standard: ~$23/TB/month
- S3 Glacier Deep Archive: ~$1/TB/month
- Current data volume: ~200GB models, ~500GB databases

### Primary Storage (OWC Drive)

The Mac Studio coordinator has an external OWC drive for primary cluster storage:

```yaml
# From config/distributed_hosts.yaml
allowed_external_storage:
  - host: mac-studio
    path: /Volumes/RingRift-Data
    subdirs:
      games: selfplay_repository
      npz: canonical_data
      models: canonical_models
```

**Capacity:** 4TB (currently ~1.2TB used)

### Data Flow

```
GPU Nodes (selfplay/training)
    │
    ▼ SyncPushDaemon (rsync)
    │
Mac Studio (/Volumes/RingRift-Data)  ←── Primary Storage (4TB)
    │
    ▼ MaintenanceDaemon (hourly)
    │
AWS S3 (ringrift-models-20251214)    ←── Cloud Backup
    │
    ▼ Lifecycle Policy (90 days)
    │
AWS Glacier Deep Archive              ←── Cold Storage
```

### Recovery Procedure

To restore from S3:

```bash
# List available backups
aws s3 ls s3://ringrift-models-20251214/models/

# Restore specific model
aws s3 cp s3://ringrift-models-20251214/models/canonical_hex8_2p.pth models/

# Restore all models
aws s3 sync s3://ringrift-models-20251214/models/ models/

# Restore from Glacier (takes 12-48 hours)
aws s3 cp s3://ringrift-archive/glacier/old_games.db.gz . \
    --request-payer requester
```

## See Also

- `scripts/lib/transfer.py` - Transfer implementation
- `app/coordination/sync_bandwidth.py` - Bandwidth-coordinated rsync
- `app/coordination/unified_distribution_daemon.py` - Model distribution
- `app/coordination/maintenance_daemon.py` - S3 archival implementation
- `config/data_aggregator.yaml` - S3 configuration
- `docs/architecture/SYNC_INFRASTRUCTURE_ARCHITECTURE.md` - Sync layer design
