# Configuration Source of Truth

This document designates the authoritative configuration file for each concept to eliminate ambiguity and prevent configuration drift.

## Quick Reference

| Concept                  | Authoritative File          | Deprecated Files                     |
| ------------------------ | --------------------------- | ------------------------------------ |
| **Cluster Hosts**        | `distributed_hosts.yaml`    | `cluster.yaml`, `cluster_nodes.yaml` |
| **P2P Configuration**    | `distributed_hosts.yaml`    | `p2p_hosts.yaml`                     |
| **SSH Access**           | `distributed_hosts.yaml`    | `remote_hosts.yaml`                  |
| **Selfplay Workers**     | `selfplay_workers.yaml`     | -                                    |
| **Training Hyperparams** | `training_hyperparams.yaml` | -                                    |
| **Promotion Settings**   | `promotion_daemon.yaml`     | -                                    |
| **Node Policies**        | `node_policies.yaml`        | -                                    |
| **Notification Hooks**   | `notification_hooks.yaml`   | -                                    |
| **Unified Loop**         | `unified_loop.yaml`         | -                                    |

## Authoritative Files

### `distributed_hosts.yaml` - CANONICAL HOST CONFIGURATION

**Status**: Primary source of truth for all host definitions (December 2025)

**Contains**:

- All cluster node definitions (SSH host, port, user, key)
- GPU specifications and memory
- Node roles (coordinator, gpu_selfplay, training, backbone)
- P2P voter configuration
- Sync routing rules
- Auto-sync settings
- ELO sync configuration

**Used by**:

- `ClusterMonitor`
- `AutoSyncDaemon`
- `SyncRouter`
- `P2P backend`
- All SSH-based operations

```yaml
# Example structure
hosts:
  runpod-h100:
    ssh_host: 102.210.171.65
    ssh_port: 30178
    ssh_user: root
    ssh_key: ~/.ssh/id_ed25519
    gpu: H100 PCIe
    gpu_vram_gb: 80
    role: gpu_selfplay_primary
    p2p_voter: true
    p2p_enabled: true
```

---

### `training_hyperparams.yaml` - TRAINING PARAMETERS

**Status**: Authoritative for neural network training settings

**Contains**:

- Learning rates by board type
- Batch sizes
- Epoch counts
- Early stopping thresholds
- Model architecture settings

---

### `promotion_daemon.yaml` - PROMOTION SETTINGS

**Status**: Authoritative for model promotion decisions

**Contains**:

- Win rate thresholds vs baselines
- ELO improvement requirements
- Evaluation game counts
- Promotion cooldowns

---

### `node_policies.yaml` - NODE WORK ASSIGNMENT

**Status**: Authoritative for node work assignment policies

**Contains**:

- Default work allow/deny lists
- Per-node overrides
- Priority weights for work types

---

### `notification_hooks.yaml` - NOTIFICATION ROUTING

**Status**: Authoritative for alert/notification hooks

**Contains**:

- Webhook destinations
- Notification channels per event type
- Retry and rate-limit settings

---

### `unified_loop.yaml` - TRAINING LOOP CONFIG

**Status**: Authoritative for unified training loop settings

**Contains**:

- Pipeline stage ordering
- Timeout settings
- Retry policies
- Stage-specific configurations

---

## Deprecated Files (Do Not Modify)

### `cluster.yaml` - DEPRECATED

**Status**: Superseded by `distributed_hosts.yaml` (December 2025)

**Migration**: All host definitions moved to `distributed_hosts.yaml`

**Reason**: Consolidated into single canonical file for easier maintenance

---

### `cluster_nodes.yaml` - DEPRECATED

**Status**: Superseded by `distributed_hosts.yaml` (December 2025)

**Migration**: Node list now in `distributed_hosts.yaml`

---

### `p2p_hosts.yaml` - DEPRECATED

**Status**: Superseded by `distributed_hosts.yaml` (December 2025)

**Migration**: P2P configuration now under `p2p_voters` and per-host `p2p_enabled` in `distributed_hosts.yaml`

---

### `remote_hosts.yaml` - DEPRECATED

**Status**: Superseded by `distributed_hosts.yaml` (December 2025)

**Migration**: SSH configuration now per-host in `distributed_hosts.yaml`

---

## Configuration Loading Priority

When multiple files might contain the same setting, this is the load priority:

1. **Environment variables** (highest priority)
2. **`distributed_hosts.yaml`** (canonical cluster config)
3. **Specific config files** (`training_hyperparams.yaml`, etc.)
4. **Default values in code** (lowest priority)

## Adding New Configuration

When adding new configuration:

1. **Host-related**: Add to `distributed_hosts.yaml` under the appropriate host
2. **Training-related**: Add to `training_hyperparams.yaml`
3. **Pipeline-related**: Add to `unified_loop.yaml`
4. **New concept**: Create a new dedicated file, document here

## Validation

Run configuration validation:

```bash
cd ai-service
python -c "from app.config.config_validator import validate_all; validate_all()"
```

This checks:

- All referenced hosts exist in `distributed_hosts.yaml`
- No conflicting definitions across files
- Required fields are present

## See Also

- `config/distributed_hosts.template.yaml` - Template for new installations
- `docs/architecture/SYNC_ARCHITECTURE.md` - How sync uses config
- `app/config/loader.py` - Configuration loading implementation
