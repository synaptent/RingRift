# CONFIG_SOURCES.md - RingRift AI Training Infrastructure Configuration Reference

This document provides a comprehensive reference for all configuration sources in the RingRift AI training infrastructure.

## Table of Contents

1. [Configuration Hierarchy and Precedence](#configuration-hierarchy-and-precedence)
2. [YAML Configuration Files](#yaml-configuration-files)
3. [Python Configuration Modules](#python-configuration-modules)
4. [Environment Variables](#environment-variables)
5. [Module Constants](#module-constants)
6. [Command Line Arguments](#command-line-arguments)
7. [Overriding Configuration for Testing](#overriding-configuration-for-testing)
8. [Migration Notes](#migration-notes)

---

## Configuration Hierarchy and Precedence

Configuration values are resolved in the following order (highest priority first):

1. **Command-line arguments** - Explicit CLI flags always take precedence
2. **Environment variables** - `RINGRIFT_*` prefixed variables
3. **YAML configuration files** - `config/*.yaml` files
4. **Python module defaults** - Hardcoded defaults in `app/config/*.py`

### Key Principle

For any given setting:

- CLI args override everything
- Environment variables override YAML and Python defaults
- YAML files override Python module defaults
- Python modules provide safe fallback defaults

---

## YAML Configuration Files

All YAML configuration files are located in `ai-service/config/`.

### distributed_hosts.yaml

**Purpose**: Cluster node configuration for distributed training infrastructure.

**Location**: `config/distributed_hosts.yaml`

**Key Sections**:

| Section        | Description                                               |
| -------------- | --------------------------------------------------------- |
| `sync_routing` | Data sync configuration (disk thresholds, replication)    |
| `auto_sync`    | Automated sync settings (intervals, bandwidth limits)     |
| `elo_sync`     | Elo database synchronization settings                     |
| `p2p_voters`   | P2P voter nodes for leader election (quorum requires 3/5) |
| `hosts`        | Per-host SSH configuration and capabilities               |

**Example**:

```yaml
sync_routing:
  max_disk_usage_percent: 70
  target_disk_usage_percent: 60
  replication_target: 2

auto_sync:
  enabled: true
  interval_seconds: 60
  gossip_interval_seconds: 15
  bandwidth_limit_mbps: 100

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
    ringrift_path: /workspace/ringrift/ai-service
    gpu: H100
    gpu_vram_gb: 80
    role: nn_training_primary
    status: ready
```

### unified_loop.yaml

**Purpose**: Master configuration for the automated training loop.

**Location**: `config/unified_loop.yaml`

**Key Sections**:

| Section            | Description                                     |
| ------------------ | ----------------------------------------------- |
| `data_ingestion`   | Remote data collection settings                 |
| `training`         | Training trigger thresholds and hyperparameters |
| `evaluation`       | Shadow/full tournament settings                 |
| `promotion`        | Model promotion criteria                        |
| `curriculum`       | Adaptive curriculum learning                    |
| `selfplay`         | Selfplay engine configuration                   |
| `cluster`          | Cluster orchestration settings                  |
| `resource_targets` | GPU/CPU utilization targets                     |
| `p2p`              | P2P distributed cluster settings                |
| `auto_scaling`     | Vast.ai auto-scaling configuration              |
| `storage`          | Provider-specific storage paths                 |

**Critical Settings**:

```yaml
training:
  trigger_threshold_games: 500
  min_interval_seconds: 1200
  learning_rate: 0.000934
  batch_size: 256

promotion:
  elo_threshold: 50
  min_games: 20
  significance_level: 0.05

selfplay:
  ai_type_weights:
    gumbel_mcts: 0.70
    policy_only: 0.15
```

### training_hyperparams.yaml

**Purpose**: Board-specific training hyperparameters.

**Location**: `config/training_hyperparams.yaml`

**Key Settings**:

```yaml
default:
  learning_rate: 0.001
  batch_size: 256
  epochs: 50
  weight_decay: 0.0001

board_overrides:
  hexagonal:
    batch_size: 128
    learning_rate: 0.0005
```

### hyperparameters.json

**Purpose**: Per-configuration optimized hyperparameters from Bayesian search.

**Location**: `config/hyperparameters.json`

**Structure**:

```json
{
  "defaults": {
    "learning_rate": 0.0003,
    "batch_size": 256,
    "hidden_dim": 256,
    "epochs": 50
  },
  "configs": {
    "square8_2p": {
      "optimized": true,
      "confidence": "high",
      "hyperparameters": { ... }
    }
  }
}
```

### Other YAML Files

| File                    | Purpose                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `cluster.yaml`          | Cluster-wide settings                                            |
| `node_policies.yaml`    | Per-node policies (sync priority, capabilities)                  |
| `p2p_hosts.yaml`        | Deprecated (P2P peers now derived from `distributed_hosts.yaml`) |
| `promotion_daemon.yaml` | Model promotion daemon settings                                  |
| `selfplay_workers.yaml` | Selfplay worker configuration                                    |
| `alerts.yaml`           | Alerting thresholds and destinations                             |
| `data_aggregator.yaml`  | Data aggregation settings                                        |

---

## Python Configuration Modules

All Python configuration modules are in `ai-service/app/config/`.

### env.py - Environment Variable Configuration

**Purpose**: Centralized typed accessors for all `RINGRIFT_*` environment variables.

**Usage**:

```python
from app.config.env import env

node_id = env.node_id
log_level = env.log_level
is_coordinator = env.is_coordinator
```

**Key Properties**:

| Property           | Type  | Default       | Description                        |
| ------------------ | ----- | ------------- | ---------------------------------- |
| `node_id`          | str   | hostname      | Node identifier                    |
| `log_level`        | str   | "INFO"        | Logging level                      |
| `is_coordinator`   | bool  | (from config) | Whether this is a coordinator node |
| `selfplay_enabled` | bool  | True          | Whether selfplay is enabled        |
| `training_enabled` | bool  | True          | Whether training is enabled        |
| `target_util_min`  | float | 60            | Minimum GPU utilization target     |
| `target_util_max`  | float | 80            | Maximum GPU utilization target     |
| `pid_kp`           | float | 0.3           | PID proportional gain              |
| `pid_ki`           | float | 0.05          | PID integral gain                  |
| `pid_kd`           | float | 0.1           | PID derivative gain                |

### thresholds.py - Quality and Training Thresholds

**Purpose**: Single source of truth for all threshold values.

**Usage**:

```python
from app.config.thresholds import (
    TRAINING_TRIGGER_GAMES,
    ELO_DROP_ROLLBACK,
    MIN_WIN_RATE_VS_RANDOM,
)
```

**Training Thresholds**:

| Constant                        | Value | Description                          |
| ------------------------------- | ----- | ------------------------------------ |
| `TRAINING_TRIGGER_GAMES`        | 500   | Games needed to trigger training     |
| `TRAINING_MIN_INTERVAL_SECONDS` | 1200  | Minimum 20 min between training runs |
| `TRAINING_STALENESS_HOURS`      | 6.0   | Hours before config is stale         |
| `TRAINING_BOOTSTRAP_GAMES`      | 50    | Games for new config bootstrap       |
| `TRAINING_MAX_CONCURRENT`       | 3     | Maximum concurrent training jobs     |

**Promotion Thresholds**:

| Constant                     | Value | Description                        |
| ---------------------------- | ----- | ---------------------------------- |
| `ELO_IMPROVEMENT_PROMOTE`    | 20    | Elo improvement for promotion      |
| `MIN_GAMES_PROMOTE`          | 100   | Minimum games for promotion        |
| `MIN_WIN_RATE_PROMOTE`       | 0.45  | Minimum win rate for promotion     |
| `PROMOTION_COOLDOWN_SECONDS` | 900   | 15 min cooldown between promotions |

**Baseline Gating Thresholds**:

| Constant                    | 2-Player | 3-Player | 4-Player |
| --------------------------- | -------- | -------- | -------- |
| `MIN_WIN_RATE_VS_RANDOM`    | 0.70     | 0.40     | 0.50     |
| `MIN_WIN_RATE_VS_HEURISTIC` | 0.50     | 0.25     | 0.20     |

**Elo Targets**:

| Constant                   | Value  | Description                       |
| -------------------------- | ------ | --------------------------------- |
| `ELO_TARGET_ALL_CONFIGS`   | 2000.0 | Target Elo for all configurations |
| `ELO_PRODUCTION_GATE`      | 2000.0 | Minimum Elo for production        |
| `PRODUCTION_ELO_THRESHOLD` | 1800   | Minimum for production promotion  |

**Gumbel MCTS Budgets**:

| Constant                   | Value | Description            |
| -------------------------- | ----- | ---------------------- |
| `GUMBEL_BUDGET_THROUGHPUT` | 64    | Fast selfplay          |
| `GUMBEL_BUDGET_STANDARD`   | 150   | Default training games |
| `GUMBEL_BUDGET_QUALITY`    | 800   | Evaluation/gauntlet    |
| `GUMBEL_BUDGET_ULTIMATE`   | 1600  | Final benchmarks       |

### constants.py - Network and Process Constants

**Purpose**: Centralized constants for ports, timeouts, and limits.

**Usage**:

```python
from app.config.constants import SSH_CONNECT_TIMEOUT, DEFAULT_BATCH_SIZE
```

**Key Constants**:

| Constant                          | Value | Description                      |
| --------------------------------- | ----- | -------------------------------- |
| `SSH_CONNECT_TIMEOUT`             | 10    | SSH connection timeout (seconds) |
| `SSH_COMMAND_TIMEOUT`             | 30    | SSH command timeout (seconds)    |
| `HTTP_REQUEST_TIMEOUT`            | 30    | HTTP request timeout (seconds)   |
| `HEARTBEAT_INTERVAL_SECONDS`      | 5.0   | P2P heartbeat interval           |
| `PEER_TIMEOUT_SECONDS`            | 30.0  | Mark peer dead after this        |
| `MAX_SELFPLAY_PROCESSES_PER_NODE` | 50    | Prevent runaway accumulation     |
| `DEFAULT_BATCH_SIZE`              | 512   | Default training batch size      |
| `DEFAULT_EPOCHS`                  | 50    | Default training epochs          |

**GPU Memory Thresholds**:

| Config         | Memory Required (GB) |
| -------------- | -------------------- |
| `hex8_2p`      | 4.0                  |
| `hex8_4p`      | 8.0                  |
| `square19_2p`  | 16.0                 |
| `square19_4p`  | 32.0                 |
| `hexagonal_4p` | 48.0                 |

### ports.py - Network Port Configuration

**Purpose**: Single source of truth for all network ports.

**Usage**:

```python
from app.config.ports import P2P_DEFAULT_PORT, HEALTH_CHECK_PORT
```

| Port Constant             | Value | Description        |
| ------------------------- | ----- | ------------------ |
| `P2P_DEFAULT_PORT`        | 8770  | P2P orchestrator   |
| `GOSSIP_PORT`             | 8771  | Gossip protocol    |
| `HEALTH_CHECK_PORT`       | 8765  | Node health checks |
| `METRICS_PORT`            | 9090  | Prometheus metrics |
| `DATA_SERVER_PORT`        | 8766  | Data transfers     |
| `UNIFIED_SYNC_API_PORT`   | 8772  | Unified sync API   |
| `AI_SERVICE_DEFAULT_PORT` | 8000  | FastAPI AI service |

### coordination_defaults.py - Coordination Defaults

**Purpose**: Default values for distributed locking, transport, sync, and scheduling.

**Usage**:

```python
from app.config.coordination_defaults import (
    LockDefaults,
    TransportDefaults,
    SyncDefaults,
)
```

**LockDefaults**:

| Attribute               | Env Override                     | Default | Description               |
| ----------------------- | -------------------------------- | ------- | ------------------------- |
| `LOCK_TIMEOUT`          | `RINGRIFT_LOCK_TIMEOUT`          | 3600    | Max lock hold time        |
| `ACQUIRE_TIMEOUT`       | `RINGRIFT_LOCK_ACQUIRE_TIMEOUT`  | 60      | Lock acquisition timeout  |
| `TRAINING_LOCK_TIMEOUT` | `RINGRIFT_TRAINING_LOCK_TIMEOUT` | 7200    | Training job lock timeout |

**TransportDefaults**:

| Attribute           | Env Override                 | Default | Description            |
| ------------------- | ---------------------------- | ------- | ---------------------- |
| `CONNECT_TIMEOUT`   | `RINGRIFT_CONNECT_TIMEOUT`   | 45      | Connection timeout     |
| `OPERATION_TIMEOUT` | `RINGRIFT_OPERATION_TIMEOUT` | 180     | Large transfer timeout |
| `HTTP_TIMEOUT`      | `RINGRIFT_HTTP_TIMEOUT`      | 30      | HTTP request timeout   |
| `SSH_TIMEOUT`       | `RINGRIFT_SSH_TIMEOUT`       | 60      | SSH operation timeout  |

**SyncDefaults**:

| Attribute                 | Env Override                  | Default | Description        |
| ------------------------- | ----------------------------- | ------- | ------------------ |
| `MAX_CONCURRENT_PER_HOST` | `RINGRIFT_MAX_SYNCS_PER_HOST` | 2       | Syncs per host     |
| `MAX_CONCURRENT_CLUSTER`  | `RINGRIFT_MAX_SYNCS_CLUSTER`  | 10      | Cluster-wide syncs |
| `DATA_SYNC_INTERVAL`      | `RINGRIFT_DATA_SYNC_INTERVAL` | 120     | Data sync interval |

**PIDDefaults**:

| Attribute | Env Override      | Default | Description       |
| --------- | ----------------- | ------- | ----------------- |
| `KP`      | `RINGRIFT_PID_KP` | 0.5     | Proportional gain |
| `KI`      | `RINGRIFT_PID_KI` | 0.1     | Integral gain     |
| `KD`      | `RINGRIFT_PID_KD` | 0.1     | Derivative gain   |

### Other Python Config Modules

| Module                | Purpose                                 |
| --------------------- | --------------------------------------- |
| `loader.py`           | YAML/JSON config file loading utilities |
| `schema.py`           | Config schema validation                |
| `hyperparameters.py`  | Hyperparameter loading from JSON        |
| `training_config.py`  | TrainConfig dataclass                   |
| `perf_budgets.py`     | Performance budget tracking             |
| `tier_eval_config.py` | Tier evaluation configuration           |

---

## Environment Variables

Most internal environment variables use the `RINGRIFT_` prefix and are accessed via
`app.config.env.env`. A small set of runtime and provider integrations use
non-prefixed variables (for example `AI_SERVICE_PORT`, `CORS_ORIGINS`, or
`AWS_REGION`), which are read directly where they are needed.

### Node Identity

| Variable                  | Type   | Default       | Description             |
| ------------------------- | ------ | ------------- | ----------------------- |
| `RINGRIFT_NODE_ID`        | string | hostname      | Node identifier         |
| `RINGRIFT_ORCHESTRATOR`   | string | "unknown"     | Orchestrator identifier |
| `RINGRIFT_IS_COORDINATOR` | bool   | (from config) | Coordinator-only mode   |
| `RINGRIFT_BUILD_VERSION`  | string | "dev"         | Build version label     |

### Paths

| Variable                   | Type | Default | Description                  |
| -------------------------- | ---- | ------- | ---------------------------- |
| `RINGRIFT_AI_SERVICE_PATH` | path | (auto)  | Path to ai-service directory |
| `RINGRIFT_DATA_DIR`        | path | "data"  | Data directory               |
| `RINGRIFT_CONFIG_PATH`     | path | None    | Config file override         |
| `RINGRIFT_ELO_DB`          | path | None    | Elo database path override   |

### Logging

| Variable               | Type   | Default   | Description                          |
| ---------------------- | ------ | --------- | ------------------------------------ |
| `RINGRIFT_LOG_LEVEL`   | string | "INFO"    | Log level (DEBUG/INFO/WARNING/ERROR) |
| `RINGRIFT_LOG_FORMAT`  | string | "default" | Log format style                     |
| `RINGRIFT_LOG_JSON`    | bool   | false     | Enable JSON logging                  |
| `RINGRIFT_LOG_FILE`    | path   | None      | Log file path                        |
| `RINGRIFT_TRACE_DEBUG` | bool   | false     | Enable trace debugging               |

### P2P / Cluster

| Variable                      | Type   | Default | Description                        |
| ----------------------------- | ------ | ------- | ---------------------------------- |
| `RINGRIFT_COORDINATOR_URL`    | string | ""      | Central coordinator URL            |
| `RINGRIFT_CLUSTER_AUTH_TOKEN` | string | None    | Cluster auth token                 |
| `RINGRIFT_P2P_AGENT_MODE`     | bool   | false   | Agent mode (defers to coordinator) |
| `RINGRIFT_HEALTH_PORT`        | int    | 8790    | Health check endpoint port         |
| `RINGRIFT_P2P_AUTO_UPDATE`    | bool   | false   | Auto-update enabled                |

### SSH

| Variable               | Type   | Default  | Description          |
| ---------------------- | ------ | -------- | -------------------- |
| `RINGRIFT_SSH_USER`    | string | "ubuntu" | Default SSH user     |
| `RINGRIFT_SSH_KEY`     | path   | None     | Default SSH key path |
| `RINGRIFT_SSH_TIMEOUT` | int    | 60       | SSH command timeout  |

### Resource Management

| Variable                        | Type  | Default | Description                    |
| ------------------------------- | ----- | ------- | ------------------------------ |
| `RINGRIFT_TARGET_UTIL_MIN`      | float | 60      | Minimum GPU utilization target |
| `RINGRIFT_TARGET_UTIL_MAX`      | float | 80      | Maximum GPU utilization target |
| `RINGRIFT_SCALE_UP_THRESHOLD`   | float | 55      | Scale up threshold             |
| `RINGRIFT_SCALE_DOWN_THRESHOLD` | float | 85      | Scale down threshold           |
| `RINGRIFT_IDLE_CHECK_INTERVAL`  | int   | 60      | Idle check interval (seconds)  |
| `RINGRIFT_IDLE_THRESHOLD`       | float | 10.0    | GPU idle threshold (%)         |
| `RINGRIFT_IDLE_DURATION`        | int   | 120     | Idle duration threshold        |

### PID Controller

| Variable          | Type  | Default | Description       |
| ----------------- | ----- | ------- | ----------------- |
| `RINGRIFT_PID_KP` | float | 0.5     | Proportional gain |
| `RINGRIFT_PID_KI` | float | 0.1     | Integral gain     |
| `RINGRIFT_PID_KD` | float | 0.1     | Derivative gain   |

### Process Management

| Variable                                      | Type | Default | Description                          |
| --------------------------------------------- | ---- | ------- | ------------------------------------ |
| `RINGRIFT_JOB_GRACE_PERIOD`                   | int  | 60      | Seconds before SIGKILL after SIGTERM |
| `RINGRIFT_GPU_IDLE_THRESHOLD`                 | int  | 600     | GPU idle before killing processes    |
| `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD` | int  | 128     | Max selfplay processes per node      |

### Feature Flags

| Variable                         | Type   | Default | Description                                             |
| -------------------------------- | ------ | ------- | ------------------------------------------------------- |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | bool   | true    | Skip shadow contract validation                         |
| `RINGRIFT_PARITY_VALIDATION`     | string | "off"   | Parity validation mode                                  |
| `RINGRIFT_PARITY_BACKEND`        | string | "auto"  | Parity backend (auto, ts, python_only, ts_hashes, skip) |
| `RINGRIFT_SKIP_PARITY`           | bool   | false   | Skip parity validation entirely                         |
| `RINGRIFT_IDLE_RESOURCE_ENABLED` | bool   | true    | Idle resource daemon enabled                            |
| `RINGRIFT_SELFPLAY_ENABLED`      | bool   | true    | Selfplay enabled on this node                           |
| `RINGRIFT_TRAINING_ENABLED`      | bool   | true    | Training enabled on this node                           |
| `RINGRIFT_GAUNTLET_ENABLED`      | bool   | true    | Gauntlet enabled on this node                           |
| `RINGRIFT_EXPORT_ENABLED`        | bool   | true    | Export enabled on this node                             |

### Training

| Variable                      | Type | Default | Description                        |
| ----------------------------- | ---- | ------- | ---------------------------------- |
| `RINGRIFT_TRAINING_THRESHOLD` | int  | 500     | Training trigger threshold (games) |

### Non-prefixed / Integration Variables

Some runtime and provider settings are intentionally **not** prefixed with `RINGRIFT_`
because they map to framework or cloud-provider conventions. Common examples:

| Variable                | Default                  | Description                                         |
| ----------------------- | ------------------------ | --------------------------------------------------- |
| `AI_SERVICE_PORT`       | 8001                     | Port for the FastAPI AI service (`app/main.py`)     |
| `CORS_ORIGINS`          | `*`                      | CORS allow-list for the AI service (`app/main.py`)  |
| `ADMIN_API_KEY`         | auto-generated           | Admin key for protected AI-service endpoints        |
| `GAME_REPLAY_DB_PATH`   | `data/games/selfplay.db` | Replay API DB path (`app/routes/replay.py`)         |
| `AI_SERVICE_DATA_DIR`   | (unset)                  | Override base data directory (`app/utils/paths.py`) |
| `AI_SERVICE_MODELS_DIR` | (unset)                  | Override models directory (`app/utils/paths.py`)    |
| `AI_SERVICE_LOGS_DIR`   | (unset)                  | Override logs directory (`app/utils/paths.py`)      |
| `STORAGE_BACKEND`       | `local`                  | Storage backend (`app/storage/backends.py`)         |
| `STORAGE_BUCKET`        | (unset)                  | S3 bucket (when `STORAGE_BACKEND=s3`)               |
| `STORAGE_PREFIX`        | `ringrift-ai`            | S3 prefix for artifacts                             |
| `STORAGE_BASE_PATH`     | `.`                      | Base path for local storage                         |

Provider credentials and observability flags (for example `VAST_API_KEY`,
`RUNPOD_API_KEY`, `LAMBDA_API_KEY`, `HCLOUD_TOKEN`, `AWS_REGION`, `OTEL_*`)
are consumed directly by their respective integration modules. Search
`os.getenv` usage in `ai-service/app` for the full list.

### Timeouts (via coordination_defaults.py)

| Variable                        | Type | Default | Description              |
| ------------------------------- | ---- | ------- | ------------------------ |
| `RINGRIFT_LOCK_TIMEOUT`         | int  | 3600    | Lock hold timeout        |
| `RINGRIFT_LOCK_ACQUIRE_TIMEOUT` | int  | 60      | Lock acquisition timeout |
| `RINGRIFT_CONNECT_TIMEOUT`      | int  | 45      | Connection timeout       |
| `RINGRIFT_HTTP_TIMEOUT`         | int  | 30      | HTTP timeout             |
| `RINGRIFT_SYNC_LOCK_TIMEOUT`    | int  | 120     | Sync lock timeout        |
| `RINGRIFT_HEARTBEAT_INTERVAL`   | int  | 30      | Heartbeat interval       |
| `RINGRIFT_HEARTBEAT_TIMEOUT`    | int  | 90      | Heartbeat timeout        |

---

## Module Constants

### Gumbel MCTS Budget Tiers

**Location**: `app/config/thresholds.py` (canonical), `app/ai/gumbel_common.py` (imports from thresholds)

| Constant                   | Value | Use Case                   |
| -------------------------- | ----- | -------------------------- |
| `GUMBEL_BUDGET_THROUGHPUT` | 64    | Fast selfplay, high volume |
| `GUMBEL_BUDGET_STANDARD`   | 150   | Default training games     |
| `GUMBEL_BUDGET_QUALITY`    | 800   | Evaluation, gauntlet       |
| `GUMBEL_BUDGET_ULTIMATE`   | 1600  | Final benchmarks           |

### Quality Thresholds

**Location**: `app/config/thresholds.py`

| Constant                   | Value | Description                      |
| -------------------------- | ----- | -------------------------------- |
| `MIN_QUALITY_FOR_TRAINING` | 0.3   | Absolute floor for training data |
| `LOW_QUALITY_THRESHOLD`    | 0.4   | Triggers warnings                |
| `MEDIUM_QUALITY_THRESHOLD` | 0.6   | Baseline acceptable              |
| `HIGH_QUALITY_THRESHOLD`   | 0.7   | Priority data                    |
| `OVERFIT_THRESHOLD`        | 0.15  | Train-val gap detection          |

### Elo Rating Constants

**Location**: `app/config/thresholds.py`

| Constant                 | Value  | Description                 |
| ------------------------ | ------ | --------------------------- |
| `INITIAL_ELO_RATING`     | 1500.0 | Starting Elo for new models |
| `MIN_ELO_RATING`         | 100.0  | Elo floor                   |
| `MAX_ELO_RATING`         | 3000.0 | Elo ceiling                 |
| `BASELINE_ELO_RANDOM`    | 400    | Random AI Elo estimate      |
| `BASELINE_ELO_HEURISTIC` | 1200   | Heuristic AI Elo estimate   |

### GPU Batch Scaling

**Location**: `app/config/thresholds.py`

| GPU Type      | Batch Multiplier | Selfplay Games/Batch |
| ------------- | ---------------- | -------------------- |
| GH200         | 64               | 2000                 |
| H100          | 32               | 1500                 |
| A100          | 16               | 800                  |
| A10           | 8                | 400                  |
| RTX 4090/3090 | 4                | 300-400              |

### SQLite Configuration

**Location**: `app/config/thresholds.py`

| Constant                 | Value    | Description        |
| ------------------------ | -------- | ------------------ |
| `SQLITE_TIMEOUT`         | 30       | Connection timeout |
| `SQLITE_BUSY_TIMEOUT_MS` | 10000    | Lock wait time     |
| `SQLITE_JOURNAL_MODE`    | "WAL"    | Journal mode       |
| `SQLITE_SYNCHRONOUS`     | "NORMAL" | Sync mode          |

---

## Command Line Arguments

### Training CLI (`python -m app.training.train`)

**Basic Options**:

| Argument          | Type  | Default | Description                           |
| ----------------- | ----- | ------- | ------------------------------------- |
| `--config`        | path  | None    | TrainingPipelineConfig YAML/JSON file |
| `--data-path`     | path  | None    | Training data (.npz file)             |
| `--save-path`     | path  | None    | Model save path                       |
| `--epochs`        | int   | None    | Training epochs                       |
| `--batch-size`    | int   | None    | Batch size                            |
| `--learning-rate` | float | None    | Learning rate                         |
| `--seed`          | int   | None    | Random seed                           |

**Model Configuration**:

| Argument           | Type | Choices                            | Description                |
| ------------------ | ---- | ---------------------------------- | -------------------------- |
| `--board-type`     | str  | square8, square19, hex8, hexagonal | Board type                 |
| `--model-version`  | str  | v2, v2_lite, v3, v4, v5, v5-gnn    | Model architecture         |
| `--model-type`     | str  | cnn, gnn, hybrid                   | Model type                 |
| `--num-players`    | int  | 2, 3, 4                            | Number of players          |
| `--num-res-blocks` | int  | None                               | Residual blocks override   |
| `--num-filters`    | int  | None                               | Filters per layer override |

**Training Enhancements**:

| Argument                           | Type | Default | Description                |
| ---------------------------------- | ---- | ------- | -------------------------- |
| `--enable-curriculum`              | flag | false   | Progressive difficulty     |
| `--enable-augmentation`            | flag | false   | Data augmentation          |
| `--enable-elo-weighting`           | flag | true    | Elo-based sample weighting |
| `--enable-auxiliary-tasks`         | flag | true    | Auxiliary prediction tasks |
| `--enable-quality-weighting`       | flag | false   | Quality-weighted training  |
| `--enable-hard-example-mining`     | flag | false   | Hard example curriculum    |
| `--enable-outcome-weighted-policy` | flag | false   | Outcome-weighted loss      |

**Checkpointing**:

| Argument                | Type | Default     | Description                |
| ----------------------- | ---- | ----------- | -------------------------- |
| `--checkpoint-dir`      | path | checkpoints | Checkpoint directory       |
| `--checkpoint-interval` | int  | 5           | Epochs between checkpoints |
| `--resume`              | path | None        | Resume from checkpoint     |
| `--init-weights`        | path | None        | Transfer learning weights  |
| `--freeze-policy`       | flag | false       | Freeze policy head         |

**Learning Rate Scheduling**:

| Argument          | Type  | Choices                            | Description   |
| ----------------- | ----- | ---------------------------------- | ------------- |
| `--lr-scheduler`  | str   | cosine, step, plateau, warmrestart | LR scheduler  |
| `--lr-min`        | float | None                               | Minimum LR    |
| `--warmup-epochs` | int   | None                               | Warmup epochs |

**Data Freshness (December 2025)**:

| Argument                 | Type  | Default | Description                   |
| ------------------------ | ----- | ------- | ----------------------------- |
| `--skip-freshness-check` | flag  | false   | Skip data freshness check     |
| `--max-data-age-hours`   | float | 1.0     | Maximum data age (hours)      |
| `--allow-stale-data`     | flag  | false   | Allow stale data with warning |

### Selfplay CLI (`python scripts/selfplay.py`)

| Argument                 | Type | Choices                                           | Description          |
| ------------------------ | ---- | ------------------------------------------------- | -------------------- |
| `--board`                | str  | square8, square19, hex8, hexagonal                | Board type           |
| `--num-players`          | int  | 2, 3, 4                                           | Number of players    |
| `--num-games`            | int  | 100                                               | Games to generate    |
| `--engine`               | str  | heuristic, gumbel, mcts, nnue-guided, policy-only | Engine type          |
| `--output-dir`           | path | data/games                                        | Output directory     |
| `--emit-pipeline-events` | flag | false                                             | Emit pipeline events |

### Export CLI (`python scripts/export_replay_dataset.py`)

| Argument             | Type  | Description                                                                 |
| -------------------- | ----- | --------------------------------------------------------------------------- |
| `--db`               | path  | Database file path                                                          |
| `--use-discovery`    | flag  | Use GameDiscovery to find databases                                         |
| `--board-type`       | str   | Filter by board type                                                        |
| `--num-players`      | int   | Filter by player count                                                      |
| `--output`           | path  | Output NPZ file                                                             |
| `--min-quality`      | float | Minimum quality score filter                                                |
| `--quality-weighted` | flag  | Emit `sample_weights` + `timestamps` arrays for quality/freshness weighting |

---

## Overriding Configuration for Testing

### Environment Variable Overrides

Set environment variables before running tests:

```bash
# Override timeouts for faster tests
export RINGRIFT_SSH_TIMEOUT=5
export RINGRIFT_HTTP_TIMEOUT=5
export RINGRIFT_LOCK_TIMEOUT=60

# Force coordinator mode
export RINGRIFT_IS_COORDINATOR=true
export RINGRIFT_SELFPLAY_ENABLED=false

# Debug logging
export RINGRIFT_LOG_LEVEL=DEBUG
```

### Python Runtime Overrides

```python
import os

# Set before importing config modules
os.environ["RINGRIFT_LOG_LEVEL"] = "DEBUG"
os.environ["RINGRIFT_TRAINING_THRESHOLD"] = "10"

from app.config.env import env
# env.log_level will be "DEBUG"
```

### YAML Config Override

Create a test-specific config file:

```yaml
# config/test_config.yaml
training:
  trigger_threshold_games: 10
  min_interval_seconds: 60
```

Then use with CLI:

```bash
python -m app.training.train --config config/test_config.yaml
```

### Threshold Updates at Runtime

```python
from app.config.thresholds import update_threshold

# For testing only
update_threshold("disk", "warning", 95)
```

---

## Migration Notes

### Deprecated Configuration

**December 2025 Deprecations**:

1. **`orchestrated_training.py`** - Use `unified_orchestrator.py` instead
2. **`integrated_enhancements.py`** - Use `unified_orchestrator.py` instead
3. **`training_enhancements.DataQualityScorer`** - Use `unified_quality.UnifiedQualityScorer`
4. **Lambda Labs configuration** - Account terminated Dec 2025

### Configuration Changes

**December 2025 Changes**:

1. **PID Controller Gains** - Increased for faster response:
   - KP: 0.3 -> 0.5
   - KI: 0.05 -> 0.1

2. **Sync Intervals** - Reduced for faster training feedback:
   - Data sync: 300s -> 60s
   - Gossip: 60s -> 15s

3. **Concurrent Syncs** - Increased for 43-node cluster:
   - Per-host: 1 -> 2
   - Cluster-wide: 5 -> 10

4. **P2P Voters** - Updated to exclude NAT-blocked and containerized nodes

### Port Migration

When migrating from hardcoded ports, import from `app.config.ports`:

```python
# Before
url = "http://localhost:8770/status"

# After
from app.config.ports import P2P_DEFAULT_PORT
url = f"http://localhost:{P2P_DEFAULT_PORT}/status"
```

### Timeout Migration

Use centralized timeout functions:

```python
# Before
timeout = 30

# After
from app.config.coordination_defaults import get_timeout
timeout = get_timeout("http")  # Returns 30
```

---

## See Also

- `docs/CONFIG_REFERENCE.md` - Quick reference for common settings
- `CLAUDE.md` - Project context and common commands
- `ai-service/CLAUDE.md` - AI service specific documentation
