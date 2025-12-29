# RingRift Environment Variables Reference

This document provides a curated reference for commonly used `RINGRIFT_*` environment variables in the RingRift AI training infrastructure. For a full listing, see `ENV_REFERENCE_COMPREHENSIVE.md`.

**Created**: December 28, 2025
**Canonical Source**: `app/config/env.py`

---

## Usage

```python
from app.config.env import env

# Access typed, cached values
node_id = env.node_id
log_level = env.log_level
is_coordinator = env.is_coordinator
```

---

## Node Identity

| Variable                  | Type | Default    | Description                                  |
| ------------------------- | ---- | ---------- | -------------------------------------------- |
| `RINGRIFT_NODE_ID`        | str  | `hostname` | Unique identifier for this node              |
| `RINGRIFT_ORCHESTRATOR`   | str  | `unknown`  | Orchestrator ID for cluster coordination     |
| `RINGRIFT_IS_COORDINATOR` | bool | auto       | Whether this node is the cluster coordinator |

---

## Paths

| Variable                         | Type | Default           | Description                  |
| -------------------------------- | ---- | ----------------- | ---------------------------- |
| `RINGRIFT_AI_SERVICE_PATH`       | Path | auto              | Path to ai-service directory |
| `RINGRIFT_DATA_DIR`              | Path | `data`            | Data directory path          |
| `RINGRIFT_CONFIG_PATH`           | Path | None              | Config file path override    |
| `RINGRIFT_ELO_DB`                | Path | None              | Elo database path override   |
| `RINGRIFT_CHECKPOINT_DIR`        | Path | `checkpoints`     | Checkpoint save directory    |
| `RINGRIFT_NFS_COORDINATION_PATH` | Path | `/lambda/nfs/...` | NFS coordination path        |

---

## Logging

| Variable               | Type | Default   | Description                                  |
| ---------------------- | ---- | --------- | -------------------------------------------- |
| `RINGRIFT_LOG_LEVEL`   | str  | `INFO`    | Log level (DEBUG, INFO, WARNING, ERROR)      |
| `RINGRIFT_LOG_FORMAT`  | str  | `default` | Log format style (default, compact, verbose) |
| `RINGRIFT_LOG_JSON`    | bool | `false`   | Enable JSON logging                          |
| `RINGRIFT_LOG_FILE`    | str  | None      | Log file path                                |
| `RINGRIFT_TRACE_DEBUG` | bool | `false`   | Enable trace debugging                       |

---

## P2P / Cluster

| Variable                                      | Type  | Default | Description                              |
| --------------------------------------------- | ----- | ------- | ---------------------------------------- |
| `RINGRIFT_COORDINATOR_URL`                    | str   | `""`    | Central coordinator URL (agent mode)     |
| `RINGRIFT_CLUSTER_AUTH_TOKEN`                 | str   | None    | Cluster authentication token             |
| `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE`            | str   | None    | Path to token file                       |
| `RINGRIFT_P2P_AGENT_MODE`                     | bool  | `false` | Run in agent mode (defer to coordinator) |
| `RINGRIFT_P2P_AUTO_UPDATE`                    | bool  | `false` | Enable automatic P2P updates             |
| `RINGRIFT_P2P_NODE_CIRCUIT_FAILURE_THRESHOLD` | int   | `5`     | Failures before opening per-node circuit |
| `RINGRIFT_P2P_NODE_CIRCUIT_RECOVERY_TIMEOUT`  | float | `60.0`  | Seconds before half-open recovery        |
| `RINGRIFT_HEALTH_PORT`                        | int   | `8790`  | Health check endpoint port               |
| `RINGRIFT_BUILD_VERSION`                      | str   | `dev`   | Build version label                      |

---

## SSH

| Variable               | Type | Default  | Description                   |
| ---------------------- | ---- | -------- | ----------------------------- |
| `RINGRIFT_SSH_USER`    | str  | `ubuntu` | Default SSH user              |
| `RINGRIFT_SSH_KEY`     | str  | None     | Default SSH key path          |
| `RINGRIFT_SSH_TIMEOUT` | int  | `60`     | SSH command timeout (seconds) |

---

## Resource Management

| Variable                        | Type  | Default | Description                              |
| ------------------------------- | ----- | ------- | ---------------------------------------- |
| `RINGRIFT_TARGET_UTIL_MIN`      | float | `60`    | Minimum target GPU utilization %         |
| `RINGRIFT_TARGET_UTIL_MAX`      | float | `80`    | Maximum target GPU utilization %         |
| `RINGRIFT_SCALE_UP_THRESHOLD`   | float | `55`    | GPU % below which to scale up            |
| `RINGRIFT_SCALE_DOWN_THRESHOLD` | float | `85`    | GPU % above which to scale down          |
| `RINGRIFT_IDLE_CHECK_INTERVAL`  | int   | `60`    | Idle check interval (seconds)            |
| `RINGRIFT_IDLE_THRESHOLD`       | float | `10.0`  | GPU % below which is considered idle     |
| `RINGRIFT_IDLE_DURATION`        | int   | `120`   | Duration to wait before action (seconds) |

### PID Controller

| Variable          | Type  | Default | Description       |
| ----------------- | ----- | ------- | ----------------- |
| `RINGRIFT_PID_KP` | float | `0.3`   | Proportional gain |
| `RINGRIFT_PID_KI` | float | `0.05`  | Integral gain     |
| `RINGRIFT_PID_KD` | float | `0.1`   | Derivative gain   |

---

## Process Management

| Variable                                      | Type | Default | Description                                        |
| --------------------------------------------- | ---- | ------- | -------------------------------------------------- |
| `RINGRIFT_JOB_GRACE_PERIOD`                   | int  | `60`    | Seconds before SIGKILL after SIGTERM               |
| `RINGRIFT_GPU_IDLE_THRESHOLD`                 | int  | `600`   | Seconds of GPU idle before killing stuck processes |
| `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD` | int  | `128`   | Max selfplay processes per node                    |

---

## Feature Flags

| Variable                         | Type | Default | Description                                |
| -------------------------------- | ---- | ------- | ------------------------------------------ |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | bool | `false` | Skip shadow contract validation            |
| `RINGRIFT_PARITY_VALIDATION`     | str  | `warn`  | Parity validation mode (warn, strict, off) |
| `RINGRIFT_AUTONOMOUS_MODE`       | bool | `false` | Enable autonomous operation                |
| `RINGRIFT_ALLOW_PENDING_GATE`    | bool | `false` | Allow training with pending_gate databases |
| `RINGRIFT_ALLOW_NONCANONICAL_DB` | bool | `false` | Allow non-canonical database names         |
| `RINGRIFT_ALLOW_STALE_DATA`      | bool | `false` | Allow training with stale data             |
| `RINGRIFT_IDLE_RESOURCE_ENABLED` | bool | `true`  | Enable idle resource detection             |
| `RINGRIFT_LAMBDA_IDLE_ENABLED`   | bool | `true`  | Enable Lambda idle shutdown                |

---

## Pipeline Capabilities

| Variable                    | Type | Default | Description                             |
| --------------------------- | ---- | ------- | --------------------------------------- |
| `RINGRIFT_SELFPLAY_ENABLED` | bool | auto    | Enable selfplay on this node            |
| `RINGRIFT_TRAINING_ENABLED` | bool | auto    | Enable training on this node            |
| `RINGRIFT_GAUNTLET_ENABLED` | bool | auto    | Enable gauntlet evaluation on this node |
| `RINGRIFT_EXPORT_ENABLED`   | bool | auto    | Enable data export on this node         |

---

## Training

| Variable                          | Type  | Default | Description                        |
| --------------------------------- | ----- | ------- | ---------------------------------- |
| `RINGRIFT_TRAINING_THRESHOLD`     | int   | `500`   | Games required before training     |
| `RINGRIFT_MIN_GAMES_FOR_TRAINING` | int   | `100`   | Minimum games for training         |
| `RINGRIFT_LEARNING_RATE`          | float | `0.001` | Default learning rate              |
| `RINGRIFT_BATCH_SIZE`             | int   | `512`   | Default batch size                 |
| `RINGRIFT_EPOCHS`                 | int   | `20`    | Default number of epochs           |
| `RINGRIFT_TRAINING_RETRY_SLEEP`   | float | `2.0`   | Pause between training retries (s) |

---

## Data Export

| Variable                        | Type  | Default | Description                              |
| ------------------------------- | ----- | ------- | ---------------------------------------- |
| `RINGRIFT_DB_LOCK_MAX_RETRIES`  | int   | `5`     | Max retries for locked databases         |
| `RINGRIFT_DB_LOCK_INITIAL_WAIT` | float | `0.5`   | Initial wait between lock retries (s)    |
| `RINGRIFT_DB_LOCK_MAX_WAIT`     | float | `30.0`  | Maximum cumulative wait for DB locks (s) |

---

## Lambda/Cloud Provider

| Variable                         | Type  | Default | Description                   |
| -------------------------------- | ----- | ------- | ----------------------------- |
| `RINGRIFT_LAMBDA_IDLE_INTERVAL`  | int   | `300`   | Idle check interval (seconds) |
| `RINGRIFT_LAMBDA_IDLE_THRESHOLD` | float | `5.0`   | GPU % below which is idle     |
| `RINGRIFT_LAMBDA_IDLE_DURATION`  | int   | `1800`  | Idle duration before shutdown |
| `RINGRIFT_LAMBDA_MIN_NODES`      | int   | `1`     | Minimum nodes to retain       |

### Per-Provider Idle Settings

Use `{PROVIDER}_IDLE_*` prefix (e.g., `VAST_IDLE_ENABLED`, `RUNPOD_IDLE_THRESHOLD`):

| Suffix                 | Type  | Description                     |
| ---------------------- | ----- | ------------------------------- |
| `_IDLE_ENABLED`        | bool  | Enable idle daemon              |
| `_IDLE_THRESHOLD`      | int   | Idle threshold (seconds)        |
| `_IDLE_UTIL_THRESHOLD` | float | GPU % considered idle           |
| `_MIN_NODES`           | int   | Minimum nodes to keep           |
| `_DRAIN_PERIOD`        | int   | Drain period before termination |
| `_IDLE_DRY_RUN`        | bool  | Log only, don't execute         |

---

## Debugging

| Variable                             | Type | Default | Description                   |
| ------------------------------------ | ---- | ------- | ----------------------------- |
| `RINGRIFT_TS_REPLAY_DUMP_DIR`        | str  | None    | Directory for TS replay dumps |
| `RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K` | str  | None    | Dump state at move K          |
| `RINGRIFT_EVENT_TRACE`               | bool | `false` | Enable event tracing          |

---

## Training Data Encoding

| Variable                       | Type | Default | Description                            |
| ------------------------------ | ---- | ------- | -------------------------------------- |
| `RINGRIFT_ENCODING_CHUNK_SIZE` | int  | `64`    | Chunk size for ParallelEncoder batches |

---

## Sync & Bandwidth

| Variable                            | Type  | Default | Description                          |
| ----------------------------------- | ----- | ------- | ------------------------------------ |
| `RINGRIFT_DATA_SYNC_INTERVAL`       | float | `120`   | Games sync interval (seconds)        |
| `RINGRIFT_MODEL_SYNC_INTERVAL`      | float | `600`   | Model sync interval (seconds)        |
| `RINGRIFT_ELO_SYNC_INTERVAL`        | float | `60`    | Elo sync interval (seconds)          |
| `RINGRIFT_REGISTRY_SYNC_INTERVAL`   | float | `120`   | Registry sync interval (seconds)     |
| `RINGRIFT_FAST_SYNC_INTERVAL`       | float | `30`    | Fast sync interval (seconds)         |
| `RINGRIFT_SYNC_FULL_INTERVAL`       | int   | `3600`  | Full sync interval (seconds)         |
| `RINGRIFT_SYNC_TIMEOUT`             | float | `300`   | Sync timeout (seconds)               |
| `RINGRIFT_MIN_SYNC_INTERVAL`        | float | `2.0`   | Minimum auto-sync interval (seconds) |
| `RINGRIFT_AUTO_SYNC_MAX_CONCURRENT` | int   | `6`     | Max concurrent auto-sync transfers   |

---

## Daemon Management

| Variable                          | Type | Default | Description                     |
| --------------------------------- | ---- | ------- | ------------------------------- |
| `RINGRIFT_DAEMON_HEALTH_INTERVAL` | int  | `30`    | Health check interval (seconds) |
| `RINGRIFT_DAEMON_MAX_RESTARTS`    | int  | `5`     | Max restart attempts            |
| `RINGRIFT_DAEMON_RESTART_DELAY`   | int  | `1`     | Initial restart delay (seconds) |

---

## Backpressure

| Variable                                 | Type  | Default | Description                       |
| ---------------------------------------- | ----- | ------- | --------------------------------- |
| `RINGRIFT_BACKPRESSURE_DISK_THRESHOLD`   | float | `80`    | Disk % to trigger backpressure    |
| `RINGRIFT_BACKPRESSURE_MEMORY_THRESHOLD` | float | `85`    | Memory % to trigger backpressure  |
| `RINGRIFT_BACKPRESSURE_GPU_THRESHOLD`    | float | `95`    | GPU % to trigger backpressure     |
| `RINGRIFT_BACKPRESSURE_COOLDOWN`         | int   | `60`    | Cooldown before release (seconds) |

---

## Quick Reference by Use Case

### Training on a Cluster Node

```bash
export RINGRIFT_TRAINING_ENABLED=true
export RINGRIFT_BATCH_SIZE=1024
export RINGRIFT_EPOCHS=50
export RINGRIFT_ALLOW_PENDING_GATE=true  # If no npx available
```

### Running Selfplay

```bash
export RINGRIFT_SELFPLAY_ENABLED=true
export RINGRIFT_IDLE_RESOURCE_ENABLED=true
export RINGRIFT_GPU_IDLE_THRESHOLD=600
```

### Coordinator Node

```bash
export RINGRIFT_IS_COORDINATOR=true
export RINGRIFT_TRAINING_ENABLED=false
export RINGRIFT_SELFPLAY_ENABLED=false
```

### Debug Mode

```bash
export RINGRIFT_LOG_LEVEL=DEBUG
export RINGRIFT_TRACE_DEBUG=true
export RINGRIFT_EVENT_TRACE=true
```

### Autonomous Mode

```bash
export RINGRIFT_AUTONOMOUS_MODE=true
export RINGRIFT_ALLOW_PENDING_GATE=true
export RINGRIFT_ALLOW_STALE_DATA=true
```

---

## Related Documentation

- [app/config/env.py](../app/config/env.py) - Canonical source for all env vars
- [DAEMON_FAILURE_RECOVERY.md](runbooks/DAEMON_FAILURE_RECOVERY.md) - Daemon troubleshooting
- [CLAUDE.md](../CLAUDE.md) - AI assistant context
