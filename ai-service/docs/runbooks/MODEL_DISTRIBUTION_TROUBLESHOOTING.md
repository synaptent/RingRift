# Model Distribution Troubleshooting Runbook

This runbook covers diagnosis and resolution of model distribution failures in the RingRift AI training cluster.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: High

---

## Overview

Model distribution issues prevent trained models from being used for:

- GPU selfplay on cluster nodes
- Model evaluation and gauntlet
- Production inference

Common issues include:

- Symlink configuration errors
- Transport failures (SSH/rsync/HTTP)
- Checksum mismatches
- Model format incompatibilities

---

## Detection Methods

### Method 1: Distribution Daemon Health

```python
from app.coordination.unified_distribution_daemon import get_distribution_daemon

daemon = get_distribution_daemon()
health = daemon.health_check()

print(f"Healthy: {health.is_healthy}")
print(f"Pending: {health.details.get('pending_distributions', 0)}")
print(f"Failed: {health.details.get('failed_last_hour', 0)}")
```

### Method 2: Check Model Availability

```bash
# Check if model exists on coordinator
ls -la models/canonical_hex8_2p.pth

# Check symlinks
ls -la models/ringrift_best_hex8_2p.pth

# Verify symlink target
readlink models/ringrift_best_hex8_2p.pth
```

### Method 3: Cluster-Wide Check

```bash
# Check model on remote nodes
for node in nebius-h100-1 runpod-a100-1 vast-12345; do
  echo "=== $node ==="
  ssh -i ~/.ssh/id_cluster ubuntu@$(get_node_ip $node) \
    "ls -la ~/ringrift/ai-service/models/ringrift_best_hex8_2p.pth" 2>/dev/null || echo "MISSING"
done
```

---

## Common Issues

### Issue 1: Missing Symlinks

**Symptom**: Model file exists but `ringrift_best_*` symlink is missing.

**Impact**: Selfplay uses wrong model or falls back to heuristic.

**Diagnosis**:

```bash
# List all canonical models without symlinks
for model in models/canonical_*.pth; do
  name=$(basename "$model" .pth | sed 's/canonical_//')
  symlink="models/ringrift_best_${name}.pth"
  if [ ! -L "$symlink" ]; then
    echo "MISSING SYMLINK: $symlink -> $model"
  fi
done
```

**Resolution**:

```bash
# Create missing symlinks
for model in models/canonical_*.pth; do
  name=$(basename "$model" .pth | sed 's/canonical_//')
  symlink="models/ringrift_best_${name}.pth"
  if [ ! -L "$symlink" ]; then
    ln -sf "$(basename $model)" "$symlink"
    echo "Created: $symlink"
  fi
done
```

---

### Issue 2: Stale Symlinks

**Symptom**: Symlink points to non-existent file.

**Diagnosis**:

```bash
# Find broken symlinks
find models -type l ! -exec test -e {} \; -print
```

**Resolution**:

```bash
# Remove broken symlinks
find models -type l ! -exec test -e {} \; -delete

# Recreate from canonical models
for model in models/canonical_*.pth; do
  name=$(basename "$model" .pth | sed 's/canonical_//')
  ln -sf "$(basename $model)" "models/ringrift_best_${name}.pth"
done
```

---

### Issue 3: Transport Failures

**Symptom**: Model distribution daemon logs show transfer errors.

**Diagnosis**:

```bash
# Check distribution logs
grep -i "transfer\|failed\|error" logs/distribution.log | tail -50

# Test connectivity to failing nodes
for node in $(grep "Failed to distribute" logs/distribution.log | \
              grep -oP 'to \K[a-z0-9-]+'); do
  echo "Testing $node..."
  ssh -o ConnectTimeout=5 -i ~/.ssh/id_cluster ubuntu@$(get_node_ip $node) "echo OK"
done
```

**Resolution**:

```bash
# Force redistribution using fallback transport
python -c "
from app.coordination.unified_distribution_daemon import UnifiedDistributionDaemon
from app.coordination.unified_distribution_daemon import DataType

daemon = UnifiedDistributionDaemon.get_instance()

# Retry with fallback transport
await daemon.distribute_file(
    file_path='models/canonical_hex8_2p.pth',
    data_type=DataType.MODEL,
    force_transport='rsync',  # Force rsync instead of HTTP
    target_nodes=['nebius-h100-1'],
)
"
```

---

### Issue 4: Checksum Mismatch

**Symptom**: Model distributed but checksum verification fails.

**Diagnosis**:

```bash
# Get local checksum
sha256sum models/canonical_hex8_2p.pth

# Compare with remote
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} \
  "sha256sum ~/ringrift/ai-service/models/canonical_hex8_2p.pth"
```

**Resolution**:

```bash
# Re-sync with verification
rsync -avz --checksum \
  -e "ssh -i ~/.ssh/id_cluster" \
  models/canonical_hex8_2p.pth \
  ubuntu@${NODE_IP}:~/ringrift/ai-service/models/

# Verify after sync
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} \
  "sha256sum ~/ringrift/ai-service/models/canonical_hex8_2p.pth"
```

---

### Issue 5: Model Format Incompatibility

**Symptom**: Model loads on some nodes but fails on others.

**Diagnosis**:

```bash
# Check PyTorch version mismatch
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} \
  "python3 -c 'import torch; print(torch.__version__)'"

# Check if model uses unavailable features
python3 -c "
import torch
ckpt = torch.load('models/canonical_hex8_2p.pth', map_location='cpu')
print(f'Keys: {list(ckpt.keys())}')
print(f'PyTorch version saved: {ckpt.get(\"_versioning_metadata\", {}).get(\"torch_version\", \"unknown\")}')
"
```

**Resolution**:

```bash
# Re-export model with compatibility settings
python -c "
import torch
ckpt = torch.load('models/canonical_hex8_2p.pth', map_location='cpu')
# Save with backward compatibility
torch.save(ckpt, 'models/canonical_hex8_2p.pth', _use_new_zipfile_serialization=False)
"
```

---

## Distribution Flow

### Normal Distribution Flow

```
TRAINING_COMPLETED event
    ↓
EvaluationDaemon (runs gauntlet)
    ↓
PROMOTION_CANDIDATE event
    ↓
AutoPromotionDaemon (checks thresholds)
    ↓
MODEL_PROMOTED event
    ↓
UnifiedDistributionDaemon
    ↓ (parallel distribution to nodes)
MODEL_DISTRIBUTION_COMPLETE event
```

### Manual Distribution

```bash
# Distribute model manually
python -c "
from app.coordination.unified_distribution_daemon import get_distribution_daemon

daemon = get_distribution_daemon()
await daemon.start()

# Distribute to all GPU nodes
result = await daemon.distribute_model(
    model_path='models/canonical_hex8_2p.pth',
    target_nodes=None,  # All GPU nodes
)
print(f'Distributed to {result.success_count} nodes')
"
```

---

## Verification

### Verify Distribution

```bash
# Check model on all active nodes
python -c "
from app.config.cluster_config import get_gpu_nodes

for node in get_gpu_nodes():
    print(f'Checking {node.name}...')
    # SSH check for model
"
```

### Verify Model Loads

```bash
# Test model loading on remote node
ssh -i ~/.ssh/id_cluster ubuntu@${NODE_IP} "
cd ~/ringrift/ai-service
python3 -c '
import torch
model = torch.load(\"models/ringrift_best_hex8_2p.pth\", map_location=\"cpu\")
print(f\"Model loaded: {type(model)}\")
print(f\"Keys: {list(model.keys())[:5]}\")
'
"
```

---

## Prevention

### 1. Pre-Distribution Validation

```python
# In unified_distribution_daemon.py
async def distribute_file(self, file_path, ...):
    # Validate before distributing
    if not self._validate_model(file_path):
        raise DistributionError(f"Model validation failed: {file_path}")
```

### 2. Symlink Management

```bash
# Add to post-promotion hook
models/create_symlinks.sh:
#!/bin/bash
for model in models/canonical_*.pth; do
  name=$(basename "$model" .pth | sed 's/canonical_//')
  ln -sf "$(basename $model)" "models/ringrift_best_${name}.pth"
done
```

### 3. Distribution Monitoring

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

# Check distribution health periodically
health = await dm.get_daemon_health(DaemonType.MODEL_DISTRIBUTION)
if not health.is_healthy:
    # Alert and investigate
    pass
```

---

## Related Documentation

- [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md) - Data sync
- [MODEL_PROMOTION_WORKFLOW.md](MODEL_PROMOTION_WORKFLOW.md) - Promotion process
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General debugging
