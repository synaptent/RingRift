# Configuration Source of Truth

This document designates the authoritative configuration file for each concept to eliminate ambiguity and prevent configuration drift.

## Quick Reference

| Concept                         | Authoritative File          | Compatibility / Deprecated Files                           |
| ------------------------------- | --------------------------- | ---------------------------------------------------------- |
| **Cluster Hosts**               | `distributed_hosts.yaml`    | `cluster.yaml`, `cluster_nodes.yaml` (legacy fallbacks)    |
| **P2P Configuration**           | `distributed_hosts.yaml`    | `p2p_hosts.yaml`                                           |
| **SSH Access**                  | `distributed_hosts.yaml`    | `remote_hosts.yaml`                                        |
| **Selfplay Workers**            | `selfplay_workers.yaml`     | -                                                          |
| **Training Hyperparams (NNUE)** | `training_hyperparams.yaml` | `hyperparameters.json` (legacy/runtime CNN overrides)      |
| **Promotion Settings**          | `promotion_daemon.yaml`     | -                                                          |
| **Node Policies**               | `node_policies.yaml`        | -                                                          |
| **Notification Hooks**          | `notification_hooks.yaml`   | -                                                          |
| **Unified Loop**                | `unified_loop.yaml`         | -                                                          |

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
p2p_voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb

hosts:
  runpod-h100:
    ssh_host: 102.210.171.65
    ssh_port: 30178
    ssh_user: root
    ssh_key: ~/.ssh/id_ed25519
    gpu: H100 PCIe
    gpu_vram_gb: 80
    role: gpu_selfplay_primary
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

**Scope note**:

- This file is the authoritative source for NNUE-oriented training settings.
- `config/hyperparameters.json` is still actively consumed by legacy/runtime
  CNN tooling (`app.config.hyperparameters`, `scripts/lib/config.py`,
  `run_nn_training_baseline.py`) and cannot be archived yet.

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

## Compatibility Inputs (Do Not Expand)

These files are no longer the canonical source of truth, but they are still
read by active code paths and therefore cannot be archived yet. Keep them as
compatibility shims while migrations continue.

### `cluster.yaml` - LEGACY P2P / ALERTS FALLBACK

**Status**: Superseded by `distributed_hosts.yaml` for host inventory, but still
loaded by P2P support code for static node metadata and alert thresholds.

**Active consumers**:

- `scripts/p2p/cluster_config.py`
- `scripts/lib/unified_cluster_config.py`
- `scripts/p2p/network_utils.py`
- `scripts/p2p/utils/webhook_notifier.py`

**Policy**: Do not add new inventory here. Prefer `distributed_hosts.yaml` and
only retain the minimum data needed by legacy fallback paths.

---

### `cluster_nodes.yaml` - LEGACY INVENTORY FALLBACK

**Status**: Superseded by `distributed_hosts.yaml`, but still read by legacy SSH
and deployment helpers.

**Active consumers**:

- `scripts/lib/cluster_config.py`
- `scripts/lib/unified_cluster_config.py`
- `scripts/master_cluster_update.sh`
- `scripts/resource_aware_router.py`

**Policy**: Do not expand this file for new deployments. Migrate callers to
`distributed_hosts.yaml` when touching those scripts.

---

### `hyperparameters.json` - LEGACY / RUNTIME CNN OVERRIDES

**Status**: Still active. Not canonical for new loop design, but not removable.

**Active consumers**:

- `app/config/hyperparameters.py`
- `app/config/config_validator.py`
- `scripts/lib/config.py`
- `scripts/run_nn_training_baseline.py`
- `scripts/tune_hyperparameters.py`

---

## Deprecated Files (Do Not Modify)

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
- `SYNC_ARCHITECTURE.md` - How sync uses config
- `app/config/loader.py` - Configuration loading implementation
