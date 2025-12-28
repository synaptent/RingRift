# Comprehensive Environment Variable Reference

All 280+ `RINGRIFT_*` environment variables used in the RingRift AI service.

**Created**: December 28, 2025
**Canonical Source**: `app/config/env.py`
**Extraction Method**: Automated grep of all Python files

---

## Usage

```python
from app.config.env import env

# Typed access with caching (preferred)
node_id = env.node_id
log_level = env.log_level
is_coordinator = env.is_coordinator

# For variables not in env.py
value = env.get("CUSTOM_VAR", "default")
```

---

## Legend

| Column    | Description                 |
| --------- | --------------------------- |
| Variable  | Environment variable name   |
| Type      | str, int, float, bool, Path |
| Default   | Default value if not set    |
| In env.py | ✓ = typed accessor exists   |

---

## Node Identity

| Variable                  | Type | Default    | In env.py | Description                                        |
| ------------------------- | ---- | ---------- | --------- | -------------------------------------------------- |
| `RINGRIFT_NODE_ID`        | str  | hostname   | ✓         | Unique node identifier                             |
| `RINGRIFT_ORCHESTRATOR`   | str  | `unknown`  | ✓         | Orchestrator ID                                    |
| `RINGRIFT_IS_COORDINATOR` | bool | auto       | ✓         | Coordinator-only mode                              |
| `RINGRIFT_NODE_ROLE`      | str  | `selfplay` | ✓         | Node role (coordinator, training, selfplay, voter) |
| `RINGRIFT_BUILD_VERSION`  | str  | `dev`      | ✓         | Build version label                                |
| `RINGRIFT_GIT_COMMIT`     | str  | None       |           | Git commit SHA                                     |

---

## Paths

| Variable                         | Type | Default           | In env.py | Description                 |
| -------------------------------- | ---- | ----------------- | --------- | --------------------------- |
| `RINGRIFT_AI_SERVICE_PATH`       | Path | auto              | ✓         | ai-service directory        |
| `RINGRIFT_AI_SERVICE_DIR`        | Path | None              |           | Alternative path variable   |
| `RINGRIFT_DATA_DIR`              | Path | `data`            | ✓         | Data directory              |
| `RINGRIFT_CONFIG_PATH`           | Path | None              | ✓         | Config file override        |
| `RINGRIFT_ELO_DB`                | Path | None              | ✓         | Elo database path           |
| `RINGRIFT_CHECKPOINT_DIR`        | Path | `checkpoints`     | ✓         | Checkpoint save directory   |
| `RINGRIFT_NFS_COORDINATION_PATH` | Path | `/lambda/nfs/...` | ✓         | NFS coordination path       |
| `RINGRIFT_NFS_PATH`              | Path | None              |           | NFS mount path              |
| `RINGRIFT_SELFPLAY_DB_PATH`      | Path | None              |           | Selfplay database path      |
| `RINGRIFT_WORK_QUEUE_DB`         | Path | None              | ✓         | Work queue database         |
| `RINGRIFT_MODEL_DIR`             | Path | None              |           | Model directory             |
| `RINGRIFT_MODELS_DIR`            | Path | None              |           | Alternative models path     |
| `RINGRIFT_NPZ_DIR`               | Path | None              |           | NPZ training data directory |
| `RINGRIFT_GAMES_DIR`             | Path | None              |           | Games directory             |
| `RINGRIFT_CANONICAL_DIR`         | Path | None              |           | Canonical data directory    |
| `RINGRIFT_ROOT`                  | Path | None              | ✓         | Monorepo root               |
| `RINGRIFT_DIR`                   | Path | None              |           | Alternative root path       |
| `RINGRIFT_BASE_DIR`              | Path | None              |           | Base directory              |
| `RINGRIFT_NPX_PATH`              | Path | None              | ✓         | Path to npx binary          |

---

## Logging

| Variable               | Type | Default   | In env.py | Description                             |
| ---------------------- | ---- | --------- | --------- | --------------------------------------- |
| `RINGRIFT_LOG_LEVEL`   | str  | `INFO`    | ✓         | Log level (DEBUG, INFO, WARNING, ERROR) |
| `RINGRIFT_LOG_FORMAT`  | str  | `default` | ✓         | Log format (default, compact, verbose)  |
| `RINGRIFT_LOG_JSON`    | bool | `false`   | ✓         | Enable JSON logging                     |
| `RINGRIFT_LOG_FILE`    | str  | None      | ✓         | Log file path                           |
| `RINGRIFT_TRACE_DEBUG` | bool | `false`   | ✓         | Enable trace debugging                  |
| `RINGRIFT_ALERT_LEVEL` | str  | None      |           | Alert threshold level                   |

---

## P2P / Cluster

| Variable                            | Type | Default | In env.py | Description                       |
| ----------------------------------- | ---- | ------- | --------- | --------------------------------- |
| `RINGRIFT_COORDINATOR_URL`          | str  | `""`    | ✓         | Central coordinator URL           |
| `RINGRIFT_COORDINATOR_IP`           | str  | None    |           | Coordinator IP address            |
| `RINGRIFT_CLUSTER_AUTH_TOKEN`       | str  | None    | ✓         | Cluster auth token                |
| `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE`  | str  | None    | ✓         | Token file path                   |
| `RINGRIFT_CLUSTER_NAME`             | str  | None    |           | Cluster name                      |
| `RINGRIFT_CLUSTER_API`              | str  | None    | ✓         | Cluster API endpoint              |
| `RINGRIFT_CLUSTER_HOSTS`            | str  | None    |           | Comma-separated host list         |
| `RINGRIFT_P2P_AGENT_MODE`           | bool | `false` | ✓         | Agent mode (defer to coordinator) |
| `RINGRIFT_P2P_AUTO_UPDATE`          | bool | `false` | ✓         | Enable auto-updates               |
| `RINGRIFT_P2P_PORT`                 | int  | `8770`  | ✓         | P2P port                          |
| `RINGRIFT_P2P_SEEDS`                | str  | `""`    | ✓         | P2P seed nodes                    |
| `RINGRIFT_P2P_URL`                  | str  | None    | ✓         | P2P orchestrator URL              |
| `RINGRIFT_P2P_STARTUP_GRACE_PERIOD` | int  | `120`   | ✓         | Startup grace period (seconds)    |
| `RINGRIFT_HEALTH_PORT`              | int  | `8790`  | ✓         | Health check port                 |
| `RINGRIFT_HEARTBEAT_INTERVAL`       | int  | `15`    | ✓         | Heartbeat interval (seconds)      |
| `RINGRIFT_HEARTBEAT_TIMEOUT`        | int  | `60`    | ✓         | Heartbeat timeout (seconds)       |

---

## Consensus & Membership

| Variable                               | Type | Default | In env.py | Description                              |
| -------------------------------------- | ---- | ------- | --------- | ---------------------------------------- |
| `RINGRIFT_CONSENSUS_MODE`              | str  | `bully` | ✓         | Consensus protocol (bully, raft, hybrid) |
| `RINGRIFT_MEMBERSHIP_MODE`             | str  | `http`  | ✓         | Membership protocol (http, swim, hybrid) |
| `RINGRIFT_RAFT_ENABLED`                | bool | `false` | ✓         | Enable Raft consensus                    |
| `RINGRIFT_RAFT_BIND_PORT`              | int  | None    |           | Raft bind port                           |
| `RINGRIFT_RAFT_COMPACTION_MIN_ENTRIES` | int  | None    |           | Min entries before compaction            |
| `RINGRIFT_RAFT_AUTO_UNLOCK_TIME`       | int  | None    |           | Auto-unlock time (seconds)               |
| `RINGRIFT_SWIM_ENABLED`                | bool | `false` | ✓         | Enable SWIM membership                   |
| `RINGRIFT_SWIM_BIND_PORT`              | int  | None    |           | SWIM bind port                           |
| `RINGRIFT_SWIM_PING_INTERVAL`          | int  | None    |           | SWIM ping interval                       |
| `RINGRIFT_SWIM_SUSPICION_TIMEOUT`      | int  | None    |           | SWIM suspicion timeout                   |
| `RINGRIFT_SWIM_FAILURE_TIMEOUT`        | int  | None    |           | SWIM failure timeout                     |
| `RINGRIFT_SWIM_INDIRECT_PING_COUNT`    | int  | None    |           | SWIM indirect pings                      |

---

## SSH

| Variable                       | Type | Default  | In env.py | Description             |
| ------------------------------ | ---- | -------- | --------- | ----------------------- |
| `RINGRIFT_SSH_USER`            | str  | `ubuntu` | ✓         | Default SSH user        |
| `RINGRIFT_SSH_KEY`             | str  | None     | ✓         | SSH key path            |
| `RINGRIFT_SSH_TIMEOUT`         | int  | `60`     | ✓         | SSH timeout (seconds)   |
| `RINGRIFT_SSH_DEFAULT_TIMEOUT` | int  | None     |           | Default command timeout |
| `RINGRIFT_SSH_CONNECT_TIMEOUT` | int  | None     |           | Connection timeout      |
| `RINGRIFT_SSH_MAX_RETRIES`     | int  | None     |           | Max retry attempts      |
| `RINGRIFT_SSH_RETRY_DELAY`     | int  | None     |           | Delay between retries   |
| `RINGRIFT_SOCKS_PROXY`         | str  | None     |           | SOCKS proxy for SSH     |
| `RINGRIFT_USE_AUTOSSH`         | bool | None     |           | Use autossh for tunnels |

---

## Resource Management

| Variable                             | Type  | Default | In env.py | Description                   |
| ------------------------------------ | ----- | ------- | --------- | ----------------------------- |
| `RINGRIFT_TARGET_UTIL_MIN`           | float | `60`    | ✓         | Min target GPU utilization %  |
| `RINGRIFT_TARGET_UTIL_MAX`           | float | `80`    | ✓         | Max target GPU utilization %  |
| `RINGRIFT_SCALE_UP_THRESHOLD`        | float | `55`    | ✓         | Scale up threshold %          |
| `RINGRIFT_SCALE_DOWN_THRESHOLD`      | float | `85`    | ✓         | Scale down threshold %        |
| `RINGRIFT_SCALE_DOWN_IDLE_THRESHOLD` | float | None    |           | Idle threshold for scale down |
| `RINGRIFT_PID_KP`                    | float | `0.3`   | ✓         | PID proportional gain         |
| `RINGRIFT_PID_KI`                    | float | `0.05`  | ✓         | PID integral gain             |
| `RINGRIFT_PID_KD`                    | float | `0.1`   | ✓         | PID derivative gain           |
| `RINGRIFT_IDLE_CHECK_INTERVAL`       | int   | `60`    | ✓         | Idle check interval (seconds) |
| `RINGRIFT_IDLE_THRESHOLD`            | float | `10.0`  | ✓         | GPU idle threshold %          |
| `RINGRIFT_IDLE_DURATION`             | int   | `120`   | ✓         | Idle duration (seconds)       |
| `RINGRIFT_IDLE_RESOURCE_ENABLED`     | bool  | `true`  | ✓         | Enable idle daemon            |
| `RINGRIFT_MAX_MEMORY_GB`             | float | None    |           | Max memory limit              |
| `RINGRIFT_MAX_DISK_PERCENT`          | float | None    |           | Max disk usage %              |
| `RINGRIFT_MAX_NODES`                 | int   | None    |           | Max cluster nodes             |
| `RINGRIFT_MIN_NODES`                 | int   | None    |           | Min cluster nodes             |

---

## Process Management

| Variable                                      | Type | Default | In env.py | Description                   |
| --------------------------------------------- | ---- | ------- | --------- | ----------------------------- |
| `RINGRIFT_JOB_GRACE_PERIOD`                   | int  | `60`    | ✓         | Grace period before SIGKILL   |
| `RINGRIFT_GPU_IDLE_THRESHOLD`                 | int  | `600`   | ✓         | GPU idle threshold (seconds)  |
| `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD` | int  | `128`   | ✓         | Max selfplay processes        |
| `RINGRIFT_PARALLEL_WORKERS`                   | int  | `4`     | ✓         | Parallel workers              |
| `RINGRIFT_PARALLEL_THRESHOLD`                 | int  | None    |           | Parallel processing threshold |
| `RINGRIFT_PARALLEL_MIN_MOVES`                 | int  | None    |           | Min moves for parallelization |
| `RINGRIFT_DISABLE_LOCAL_TASKS`                | bool | `false` | ✓         | Disable local task execution  |
| `RINGRIFT_JOB_ORIGIN`                         | str  | None    | ✓         | Job origin identifier         |
| `RINGRIFT_JOB_REAPER_FALLBACK_ENABLED`        | bool | None    |           | Enable job reaper fallback    |

---

## Feature Flags

| Variable                         | Type | Default | In env.py | Description                        |
| -------------------------------- | ---- | ------- | --------- | ---------------------------------- |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | bool | `false` | ✓         | Skip shadow validation             |
| `RINGRIFT_PARITY_VALIDATION`     | str  | `warn`  | ✓         | Parity mode (warn, strict, off)    |
| `RINGRIFT_AUTONOMOUS_MODE`       | bool | `false` | ✓         | Enable autonomous operation        |
| `RINGRIFT_ALLOW_PENDING_GATE`    | bool | `false` | ✓         | Allow pending_gate databases       |
| `RINGRIFT_ALLOW_NONCANONICAL_DB` | bool | `false` |           | Allow non-canonical databases      |
| `RINGRIFT_AUTO_ROLLBACK`         | bool | None    |           | Enable auto-rollback on regression |
| `RINGRIFT_EXTRACTED_LOOPS`       | bool | `true`  | ✓         | Use extracted loop managers        |
| `RINGRIFT_UNIFIED_LOOP_LEGACY`   | bool | None    |           | Use legacy loop mode               |
| `RINGRIFT_PREEMPTIVE_RECOVERY`   | bool | None    |           | Enable preemptive recovery         |

---

## Pipeline Capabilities

| Variable                    | Type | Default | In env.py | Description     |
| --------------------------- | ---- | ------- | --------- | --------------- |
| `RINGRIFT_SELFPLAY_ENABLED` | bool | auto    | ✓         | Enable selfplay |
| `RINGRIFT_TRAINING_ENABLED` | bool | auto    | ✓         | Enable training |
| `RINGRIFT_GAUNTLET_ENABLED` | bool | auto    | ✓         | Enable gauntlet |
| `RINGRIFT_EXPORT_ENABLED`   | bool | auto    | ✓         | Enable export   |

---

## Training

| Variable                              | Type  | Default | In env.py | Description                  |
| ------------------------------------- | ----- | ------- | --------- | ---------------------------- |
| `RINGRIFT_TRAINING_THRESHOLD`         | int   | `500`   | ✓         | Games before training        |
| `RINGRIFT_MIN_GAMES_FOR_TRAINING`     | int   | `100`   | ✓         | Minimum games required       |
| `RINGRIFT_MIN_TRAINING_GAMES`         | int   | None    |           | Alternative min games        |
| `RINGRIFT_MIN_GAMES_TRAINING`         | int   | None    |           | Alternative min games        |
| `RINGRIFT_LEARNING_RATE`              | float | `0.001` | ✓         | Learning rate                |
| `RINGRIFT_BATCH_SIZE`                 | int   | `512`   | ✓         | Batch size                   |
| `RINGRIFT_TRAINING_BATCH_SIZE`        | int   | None    |           | Training-specific batch size |
| `RINGRIFT_EPOCHS`                     | int   | `20`    | ✓         | Training epochs              |
| `RINGRIFT_DATALOADER_WORKERS`         | int   | None    |           | DataLoader workers           |
| `RINGRIFT_ENABLE_POLICY_TRAINING`     | bool  | None    |           | Enable policy training       |
| `RINGRIFT_ENABLE_INCREMENTAL_EXPORT`  | bool  | None    |           | Enable incremental export    |
| `RINGRIFT_PROFILE_TRAINING`           | bool  | None    |           | Profile training             |
| `RINGRIFT_MIN_HOURS_BETWEEN_TRAINING` | int   | None    |           | Min hours between runs       |

---

## Neural Network / GPU

| Variable                              | Type  | Default | In env.py | Description                     |
| ------------------------------------- | ----- | ------- | --------- | ------------------------------- |
| `RINGRIFT_DISABLE_TORCH_COMPILE`      | bool  | `false` | ✓         | Disable torch.compile()         |
| `RINGRIFT_DISABLE_NEURAL_NET`         | bool  | `false` | ✓         | Disable neural network          |
| `RINGRIFT_REQUIRE_NEURAL_NET`         | bool  | `false` | ✓         | Require neural network          |
| `RINGRIFT_FORCE_CPU`                  | bool  | `false` | ✓         | Force CPU-only mode             |
| `RINGRIFT_GPU_GUMBEL_DISABLE`         | bool  | `false` | ✓         | Disable GPU Gumbel MCTS         |
| `RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE` | bool  | None    |           | Shadow validate Gumbel          |
| `RINGRIFT_GPU_MCTS_DISABLE`           | bool  | None    |           | Disable GPU MCTS                |
| `RINGRIFT_GPU_MCTS_SHADOW_VALIDATE`   | bool  | None    |           | Shadow validate MCTS            |
| `RINGRIFT_GPU_MAXN_DISABLE`           | bool  | None    |           | Disable GPU MaxN                |
| `RINGRIFT_GPU_MAXN_SHADOW_VALIDATE`   | bool  | None    |           | Shadow validate MaxN            |
| `RINGRIFT_GPU_MINIMAX_DISABLE`        | bool  | None    |           | Disable GPU Minimax             |
| `RINGRIFT_GPU_MINIMAX_BATCH_SIZE`     | int   | None    |           | Minimax batch size              |
| `RINGRIFT_GPU_CAPTURE_LEGACY`         | bool  | None    |           | Use legacy GPU capture          |
| `RINGRIFT_GPU_MOVEMENT_LEGACY`        | bool  | None    |           | Use legacy movement             |
| `RINGRIFT_GPU_TREE_SHADOW_RATE`       | float | None    |           | Shadow validation rate          |
| `RINGRIFT_NN_MEMORY_TIER`             | str   | None    |           | Memory tier (low, medium, high) |
| `RINGRIFT_NN_WARN_TIMEOUT`            | int   | None    |           | NN warning timeout              |
| `RINGRIFT_VECTOR_VALUE_HEAD`          | bool  | None    |           | Use vector value head           |

---

## Game Engine

| Variable                            | Type | Default  | In env.py | Description              |
| ----------------------------------- | ---- | -------- | --------- | ------------------------ |
| `RINGRIFT_USE_FAST_TERRITORY`       | bool | `true`   | ✓         | Fast territory scoring   |
| `RINGRIFT_USE_MAKE_UNMAKE`          | bool | `true`   | ✓         | Make/unmake optimization |
| `RINGRIFT_FSM_VALIDATION_MODE`      | str  | `strict` | ✓         | FSM validation mode      |
| `RINGRIFT_STRICT_NO_MOVE_INVARIANT` | bool | `false`  | ✓         | Strict no-move invariant |
| `RINGRIFT_RECORD_SELFPLAY_GAMES`    | bool | `true`   | ✓         | Record selfplay games    |
| `RINGRIFT_BOARD_TYPE`               | str  | None     |           | Board type override      |
| `RINGRIFT_RULES_VERSION`            | str  | None     |           | Rules version            |
| `RINGRIFT_RULES_MUTATOR_FIRST`      | bool | None     |           | Mutator first mode       |

---

## Timeouts

| Variable                         | Type | Default | In env.py | Description             |
| -------------------------------- | ---- | ------- | --------- | ----------------------- |
| `RINGRIFT_HTTP_TIMEOUT`          | int  | `30`    | ✓         | HTTP timeout (seconds)  |
| `RINGRIFT_RSYNC_TIMEOUT`         | int  | `300`   | ✓         | Rsync timeout (seconds) |
| `RINGRIFT_ELO_SYNC_INTERVAL`     | int  | `300`   | ✓         | Elo sync interval       |
| `RINGRIFT_EVENT_HANDLER_TIMEOUT` | int  | `30`    | ✓         | Event handler timeout   |
| `RINGRIFT_LOCK_TIMEOUT`          | int  | `60`    | ✓         | Lock timeout            |
| `RINGRIFT_AI_TIMEOUT`            | int  | None    |           | AI service timeout      |
| `RINGRIFT_TCP_PROBE_TIMEOUT`     | int  | None    |           | TCP probe timeout       |

---

## Circuit Breaker

| Variable                                 | Type | Default | In env.py | Description                   |
| ---------------------------------------- | ---- | ------- | --------- | ----------------------------- |
| `RINGRIFT_CB_FAILURE_THRESHOLD`          | int  | `5`     | ✓         | Failures before open          |
| `RINGRIFT_CB_RECOVERY_TIMEOUT`           | int  | `60`    | ✓         | Recovery timeout              |
| `RINGRIFT_CB_HALF_OPEN_MAX_CALLS`        | int  | `3`     | ✓         | Half-open test calls          |
| `RINGRIFT_TRANSPORT_FAILURE_THRESHOLD`   | int  | None    |           | Transport failure threshold   |
| `RINGRIFT_TRANSPORT_DISABLE_DURATION`    | int  | None    |           | Transport disable duration    |
| `RINGRIFT_TERMINATION_FAILURE_THRESHOLD` | int  | None    |           | Termination failure threshold |

---

## Backpressure

| Variable                                 | Type  | Default | In env.py | Description               |
| ---------------------------------------- | ----- | ------- | --------- | ------------------------- |
| `RINGRIFT_BACKPRESSURE_DISK_THRESHOLD`   | float | `90`    | ✓         | Disk threshold %          |
| `RINGRIFT_BACKPRESSURE_MEMORY_THRESHOLD` | float | `85`    | ✓         | Memory threshold %        |
| `RINGRIFT_BACKPRESSURE_GPU_THRESHOLD`    | float | `90`    | ✓         | GPU threshold %           |
| `RINGRIFT_CPU_WARNING`                   | float | None    |           | CPU warning threshold     |
| `RINGRIFT_CPU_CRITICAL`                  | float | None    |           | CPU critical threshold    |
| `RINGRIFT_MEMORY_WARNING`                | float | None    |           | Memory warning threshold  |
| `RINGRIFT_MEMORY_CRITICAL`               | float | None    |           | Memory critical threshold |
| `RINGRIFT_DISK_WARNING`                  | float | None    |           | Disk warning threshold    |
| `RINGRIFT_DISK_CRITICAL`                 | float | None    |           | Disk critical threshold   |

---

## Lambda/Cloud Provider

| Variable                         | Type  | Default | In env.py | Description                   |
| -------------------------------- | ----- | ------- | --------- | ----------------------------- |
| `RINGRIFT_LAMBDA_IDLE_ENABLED`   | bool  | `true`  | ✓         | Enable Lambda idle daemon     |
| `RINGRIFT_LAMBDA_IDLE_INTERVAL`  | int   | `300`   | ✓         | Idle check interval           |
| `RINGRIFT_LAMBDA_IDLE_THRESHOLD` | float | `5.0`   | ✓         | GPU idle threshold            |
| `RINGRIFT_LAMBDA_IDLE_DURATION`  | int   | `1800`  | ✓         | Idle duration before shutdown |
| `RINGRIFT_LAMBDA_MIN_NODES`      | int   | `1`     | ✓         | Min nodes to retain           |
| `RINGRIFT_LAMBDA_IPS`            | str   | None    |           | Lambda node IPs               |
| `RINGRIFT_VAST_SSH_USER`         | str   | None    |           | Vast.ai SSH user              |
| `RINGRIFT_STORAGE_PROVIDER`      | str   | None    |           | Storage provider              |

---

## Alerting

| Variable                       | Type | Default | In env.py | Description         |
| ------------------------------ | ---- | ------- | --------- | ------------------- |
| `RINGRIFT_DISCORD_WEBHOOK`     | str  | None    | ✓         | Discord webhook     |
| `RINGRIFT_DISCORD_WEBHOOK_URL` | str  | None    | ✓         | Discord webhook URL |
| `RINGRIFT_SLACK_WEBHOOK`       | str  | None    | ✓         | Slack webhook       |
| `RINGRIFT_SLACK_WEBHOOK_URL`   | str  | None    | ✓         | Slack webhook URL   |
| `RINGRIFT_WEBHOOK_URL`         | str  | None    |           | Generic webhook URL |

---

## Debugging

| Variable                             | Type | Default | In env.py | Description                 |
| ------------------------------------ | ---- | ------- | --------- | --------------------------- |
| `RINGRIFT_TS_REPLAY_DUMP_DIR`        | str  | None    | ✓         | TS replay dump directory    |
| `RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K` | str  | None    | ✓         | Dump state at move K        |
| `RINGRIFT_PARITY_DUMP_DIR`           | str  | None    |           | Parity dump directory       |
| `RINGRIFT_PARITY_PROGRESS_EVERY`     | int  | None    |           | Progress report interval    |
| `RINGRIFT_SOAK_FAILURE_DIR`          | str  | None    |           | Soak test failure directory |

---

## Staging

| Variable                                         | Type | Default | In env.py | Description            |
| ------------------------------------------------ | ---- | ------- | --------- | ---------------------- |
| `RINGRIFT_STAGING_ROOT`                          | Path | None    | ✓         | Staging root directory |
| `RINGRIFT_STAGING_SSH_HOST`                      | str  | None    | ✓         | Staging SSH host       |
| `RINGRIFT_STAGING_SSH_PORT`                      | int  | None    |           | Staging SSH port       |
| `RINGRIFT_STAGING_SSH_USER`                      | str  | None    |           | Staging SSH user       |
| `RINGRIFT_STAGING_SSH_KEY`                       | str  | None    |           | Staging SSH key        |
| `RINGRIFT_STAGING_COMPOSE_FILE`                  | str  | None    |           | Staging compose file   |
| `RINGRIFT_STAGING_RESTART_SERVICES`              | str  | None    |           | Services to restart    |
| `RINGRIFT_STAGING_LADDER_HEALTH_URL`             | str  | None    |           | Ladder health URL      |
| `RINGRIFT_STAGING_LADDER_HEALTH_TIMEOUT_SECONDS` | int  | None    |           | Health timeout         |

---

## Search/AI Algorithms

| Variable                           | Type  | Default | In env.py | Description              |
| ---------------------------------- | ----- | ------- | --------- | ------------------------ |
| `RINGRIFT_MCTS_ASYNC_NN_EVAL`      | bool  | None    |           | Async NN evaluation      |
| `RINGRIFT_MCTS_LEAF_BATCH_SIZE`    | int   | None    |           | MCTS leaf batch size     |
| `RINGRIFT_DESCENT_ASYNC_NN_EVAL`   | bool  | None    |           | Descent async eval       |
| `RINGRIFT_DESCENT_LEAF_BATCH_SIZE` | int   | None    |           | Descent leaf batch       |
| `RINGRIFT_DESCENT_GPU_HEURISTIC`   | bool  | None    |           | GPU heuristic mode       |
| `RINGRIFT_DESCENT_UCB`             | bool  | None    |           | Use UCB                  |
| `RINGRIFT_DESCENT_UCB_C`           | float | None    |           | UCB exploration constant |
| `RINGRIFT_USE_EBMO`                | bool  | None    |           | Use EBMO                 |
| `RINGRIFT_USE_IG_GMO`              | bool  | None    |           | Use IG-GMO               |
| `RINGRIFT_USE_OPENING_BOOK`        | bool  | None    |           | Use opening book         |
| `RINGRIFT_USE_MOVE_CACHE`          | bool  | None    |           | Use move cache           |
| `RINGRIFT_MOVE_CACHE_SIZE`         | int   | None    |           | Move cache size          |
| `RINGRIFT_USE_BATCH_EVAL`          | bool  | None    |           | Use batch evaluation     |
| `RINGRIFT_USE_PARALLEL_EVAL`       | bool  | None    |           | Use parallel evaluation  |
| `RINGRIFT_BATCH_EVAL_THRESHOLD`    | int   | None    |           | Batch eval threshold     |

---

## Work Queue

| Variable                         | Type | Default | In env.py | Description               |
| -------------------------------- | ---- | ------- | --------- | ------------------------- |
| `RINGRIFT_WORK_QUEUE_SOFT_LIMIT` | int  | None    |           | Soft limit for queue size |
| `RINGRIFT_WORK_QUEUE_HARD_LIMIT` | int  | None    |           | Hard limit for queue size |
| `RINGRIFT_WORK_QUEUE_RECOVERY`   | bool | None    |           | Enable queue recovery     |

---

## Sync & Bandwidth

| Variable                        | Type | Default | In env.py | Description          |
| ------------------------------- | ---- | ------- | --------- | -------------------- |
| `RINGRIFT_SYNC_STRATEGY`        | str  | None    |           | Sync strategy        |
| `RINGRIFT_SYNC_TARGET`          | str  | None    |           | Sync target node     |
| `RINGRIFT_AUTO_SYNC`            | bool | None    |           | Enable auto sync     |
| `RINGRIFT_ENABLE_VAST_ELO_SYNC` | bool | None    |           | Enable Vast Elo sync |

---

## Evaluation

| Variable                               | Type | Default | In env.py | Description                 |
| -------------------------------------- | ---- | ------- | --------- | --------------------------- |
| `RINGRIFT_EVAL_WORKERS`                | int  | None    |           | Evaluation workers          |
| `RINGRIFT_EVAL_MOVE_SAMPLE_LIMIT`      | int  | None    |           | Move sample limit           |
| `RINGRIFT_EVAL_HEURISTIC_EVAL_MODE`    | str  | None    |           | Heuristic eval mode         |
| `RINGRIFT_SKIP_POST_TRAINING_GAUNTLET` | bool | None    |           | Skip post-training gauntlet |
| `RINGRIFT_SKIP_REGRESSION_TESTS`       | bool | None    |           | Skip regression tests       |

---

## Hyperparameter Tuning

| Variable                           | Type | Default | In env.py | Description           |
| ---------------------------------- | ---- | ------- | --------- | --------------------- |
| `RINGRIFT_ENABLE_AUTO_HP_TUNING`   | bool | None    |           | Enable auto HP tuning |
| `RINGRIFT_MIN_GAMES_FOR_HP_TUNING` | int  | None    |           | Min games for tuning  |
| `RINGRIFT_CMAES_WORKERS`           | int  | None    |           | CMA-ES workers        |

---

## Policy Training

| Variable                           | Type  | Default | In env.py | Description             |
| ---------------------------------- | ----- | ------- | --------- | ----------------------- |
| `RINGRIFT_POLICY_AUTO_KL_LOSS`     | bool  | None    |           | Auto KL loss            |
| `RINGRIFT_POLICY_KL_MIN_COVERAGE`  | float | None    |           | Min KL coverage         |
| `RINGRIFT_POLICY_KL_MIN_SAMPLES`   | int   | None    |           | Min KL samples          |
| `RINGRIFT_LEGACY_POLICY_TRANSFORM` | bool  | None    |           | Legacy policy transform |

---

## Rollback

| Variable                         | Type  | Default | In env.py | Description               |
| -------------------------------- | ----- | ------- | --------- | ------------------------- |
| `RINGRIFT_ROLLBACK_ELO_DROP`     | float | None    |           | Elo drop threshold        |
| `RINGRIFT_ROLLBACK_MIN_GAMES`    | int   | None    |           | Min games before rollback |
| `RINGRIFT_REGRESSION_HARD_BLOCK` | bool  | None    |           | Block on regression       |

---

## Autoscaling

| Variable                                    | Type  | Default | In env.py | Description               |
| ------------------------------------------- | ----- | ------- | --------- | ------------------------- |
| `RINGRIFT_AUTOSCALE_DRY_RUN`                | bool  | None    |           | Dry run mode              |
| `RINGRIFT_AUTOSCALE_MIN_WORKERS`            | int   | None    |           | Min workers               |
| `RINGRIFT_AUTOSCALE_MAX_WORKERS`            | int   | None    |           | Max workers               |
| `RINGRIFT_AUTOSCALE_SCALE_UP_GPH`           | float | None    |           | Games per hour scale up   |
| `RINGRIFT_AUTOSCALE_SCALE_DOWN_GPH`         | float | None    |           | Games per hour scale down |
| `RINGRIFT_AUTOSCALE_TARGET_FRESHNESS_HOURS` | float | None    |           | Target data freshness     |

---

## Coordinator Disk Management

| Variable                           | Type | Default | In env.py | Description      |
| ---------------------------------- | ---- | ------- | --------- | ---------------- |
| `RINGRIFT_COORDINATOR_REMOTE_HOST` | str  | None    |           | Remote sync host |
| `RINGRIFT_COORDINATOR_REMOTE_PATH` | str  | None    |           | Remote sync path |

---

## AI Instance Cache

| Variable                             | Type | Default | In env.py | Description              |
| ------------------------------------ | ---- | ------- | --------- | ------------------------ |
| `RINGRIFT_AI_INSTANCE_CACHE`         | bool | None    |           | Enable AI instance cache |
| `RINGRIFT_AI_INSTANCE_CACHE_MAX`     | int  | None    |           | Max cache size           |
| `RINGRIFT_AI_INSTANCE_CACHE_TTL_SEC` | int  | None    |           | Cache TTL (seconds)      |
| `RINGRIFT_AI_HEALTHCHECK_GAMES`      | int  | None    |           | Healthcheck game count   |

---

## Skip Flags (Testing/Development)

| Variable                                | Type | Default | In env.py | Description                  |
| --------------------------------------- | ---- | ------- | --------- | ---------------------------- |
| `RINGRIFT_SKIP_OPTIONAL_IMPORTS`        | bool | `false` |           | Skip optional imports        |
| `RINGRIFT_SKIP_SCRIPT_INIT_IMPORTS`     | bool | `false` |           | Skip script init imports     |
| `RINGRIFT_SKIP_TORCH_IMPORT`            | bool | `false` |           | Skip torch import            |
| `RINGRIFT_SKIP_RESOURCE_GUARD`          | bool | `false` |           | Skip resource guard          |
| `RINGRIFT_SKIP_MASTER_LOOP_CHECK`       | bool | `false` |           | Skip master loop check       |
| `RINGRIFT_SKIP_SELFPLAY_CONFIG`         | bool | `false` |           | Skip selfplay config         |
| `RINGRIFT_SKIP_SYNC_LOCK_IMPORT`        | bool | `false` |           | Skip sync lock import        |
| `RINGRIFT_SKIP_TRAINING_COORD`          | bool | `false` |           | Skip training coordinator    |
| `RINGRIFT_REQUIRE_CRITICAL_IMPORTS`     | bool | `false` | ✓         | Require critical imports     |
| `RINGRIFT_DISABLE_AUTO_DISCOVERY`       | bool | `false` |           | Disable auto discovery       |
| `RINGRIFT_DISABLE_GPU_DATAGEN`          | bool | `false` |           | Disable GPU data generation  |
| `RINGRIFT_FAIL_ON_SUBSCRIPTION_FAILURE` | bool | `false` |           | Fail on subscription failure |

---

## Quick Reference by Use Case

### Training on Cluster Node

```bash
export RINGRIFT_TRAINING_ENABLED=true
export RINGRIFT_BATCH_SIZE=1024
export RINGRIFT_EPOCHS=50
export RINGRIFT_ALLOW_PENDING_GATE=true
```

### Running Selfplay

```bash
export RINGRIFT_SELFPLAY_ENABLED=true
export RINGRIFT_IDLE_RESOURCE_ENABLED=true
export RINGRIFT_RECORD_SELFPLAY_GAMES=true
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
export RINGRIFT_PARITY_VALIDATION=strict
```

### Autonomous Mode

```bash
export RINGRIFT_AUTONOMOUS_MODE=true
export RINGRIFT_ALLOW_PENDING_GATE=true
export RINGRIFT_AUTO_ROLLBACK=true
```

---

## Related Documentation

- [app/config/env.py](../app/config/env.py) - Canonical source for typed access
- [ENV_REFERENCE.md](ENV_REFERENCE.md) - Summary reference
- [CLAUDE.md](../CLAUDE.md) - AI assistant context
