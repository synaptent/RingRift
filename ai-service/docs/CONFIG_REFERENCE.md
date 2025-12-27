# Configuration Reference

This document provides a comprehensive reference for all configuration options in the RingRift AI service.

## Overview

Configuration is managed through `app/config/unified_config.py` which provides a single source of truth for all settings. Configuration is loaded from `config/unified_loop.yaml` and can be overridden via environment variables.

```python
from app.config.unified_config import get_config

config = get_config()
```

## Environment Variable Overrides

| Variable                              | Description                                 | Default                     |
| ------------------------------------- | ------------------------------------------- | --------------------------- |
| `RINGRIFT_CONFIG_PATH`                | Override config file path                   | `config/unified_loop.yaml`  |
| `RINGRIFT_TRAINING_THRESHOLD`         | Training trigger threshold                  | 500                         |
| `RINGRIFT_MIN_GAMES_FOR_TRAINING`     | Alias for training trigger threshold        | 500                         |
| `RINGRIFT_ELO_DB`                     | Elo database path                           | `data/elo/elo.db`           |
| `AI_SERVICE_PORT`                     | Port for the FastAPI AI service             | `8001`                      |
| `ADMIN_API_KEY`                       | Admin API key (`X-Admin-Key`) for /admin/\* | Generated per boot if unset |
| `CORS_ORIGINS`                        | Comma-separated CORS allowlist              | `*`                         |
| `GAME_REPLAY_DB_PATH`                 | Replay DB path for `/api/replay/*`          | `data/games/selfplay.db`    |
| `RINGRIFT_START_DAEMONS`              | Start daemon manager on boot (`1` = on)     | `0`                         |
| `RINGRIFT_ENV`                        | Environment label for error sanitization    | `development`               |
| `RINGRIFT_AI_TIMEOUT`                 | AI operation timeout (seconds)              | `30.0`                      |
| `RINGRIFT_AI_INSTANCE_CACHE`          | Enable AI instance cache (`1` = on)         | `1`                         |
| `RINGRIFT_AI_INSTANCE_CACHE_TTL_SEC`  | AI instance cache TTL (seconds)             | `1800`                      |
| `RINGRIFT_AI_INSTANCE_CACHE_MAX`      | Max cached AI instances                     | `512`                       |
| `RINGRIFT_TRAINED_HEURISTIC_PROFILES` | Path to trained heuristic profiles JSON     | None                        |

---

## Direct Runtime Environment Flags (RINGRIFT\_\*)

These flags are read directly by runtime modules and bypass `unified_config`. Defaults shown are the code defaults when unset (some are hardware- or platform-dependent).

### Core runtime and logging

| Variable                         | Description                                                 | Default   |
| -------------------------------- | ----------------------------------------------------------- | --------- |
| `RINGRIFT_DEBUG`                 | Enable debug mode (verbose logging)                         | `false`   |
| `RINGRIFT_DEBUG_ENGINE`          | Enable legacy engine debug logging                          | `false`   |
| `RINGRIFT_LOG_LEVEL`             | Log level (DEBUG/INFO/WARNING/ERROR)                        | `INFO`    |
| `RINGRIFT_LOG_FORMAT`            | Log format style (`default`, `compact`, `detailed`, `json`) | `default` |
| `RINGRIFT_LOG_JSON`              | Force JSON logging                                          | `false`   |
| `RINGRIFT_LOG_FILE`              | Log file path                                               | `unset`   |
| `RINGRIFT_SKIP_OPTIONAL_IMPORTS` | Skip optional dependency imports                            | `false`   |
| `RINGRIFT_SKIP_TORCH_IMPORT`     | Skip importing torch                                        | `false`   |
| `RINGRIFT_DISABLE_TORCH_COMPILE` | Disable torch.compile usage                                 | `false`   |
| `RINGRIFT_FORCE_CPU`             | Force CPU device selection (legacy NN/encoding)             | `false`   |
| `RINGRIFT_DISABLE_MPS`           | Disable MPS backend (macOS)                                 | `false`   |
| `RINGRIFT_NN_ARCHITECTURE`       | Legacy NN architecture selection                            | `auto`    |
| `RINGRIFT_NN_MEMORY_TIER`        | Legacy NN memory tier                                       | `high`    |
| `RINGRIFT_NN_RESOLVE_MAX_PROBE`  | Max attempts to resolve NN model path                       | `25`      |
| `RINGRIFT_MAX_MEMORY_GB`         | Memory budget for training/inference buffers                | `16.0`    |
| `RINGRIFT_MAX_LOAD_FACTOR`       | Max load factor relative to CPU count                       | `2.0`     |
| `RINGRIFT_MAX_LOAD_ABSOLUTE`     | Absolute max load average                                   | `100.0`   |
| `RINGRIFT_LOAD_BACKOFF_SECONDS`  | Wait interval when overloaded                               | `30.0`    |

### Notifications

| Variable                       | Description         | Default |
| ------------------------------ | ------------------- | ------- |
| `RINGRIFT_SLACK_WEBHOOK_URL`   | Slack webhook URL   | `unset` |
| `RINGRIFT_DISCORD_WEBHOOK_URL` | Discord webhook URL | `unset` |
| `RINGRIFT_WEBHOOK_URL`         | Generic webhook URL | `unset` |

### Provider Credentials & Cloud

| Variable                | Description                                   | Default |
| ----------------------- | --------------------------------------------- | ------- |
| `VAST_API_KEY`          | Vast.ai API key (node provisioning)           | `unset` |
| `RUNPOD_API_KEY`        | Runpod API key (node provisioning)            | `unset` |
| `LAMBDA_API_KEY`        | Lambda Labs API key (node provisioning)       | `unset` |
| `HCLOUD_TOKEN`          | Hetzner Cloud token                           | `unset` |
| `AWS_REGION`            | AWS region for S3 backups                     | `unset` |
| `AWS_DEFAULT_REGION`    | AWS region fallback (if `AWS_REGION` not set) | `unset` |
| `SLACK_WEBHOOK_URL`     | Legacy Slack webhook (alert_router/promotion) | `unset` |
| `DISCORD_WEBHOOK_URL`   | Legacy Discord webhook (alert_router)         | `unset` |
| `PAGERDUTY_ROUTING_KEY` | PagerDuty routing key (alert_router)          | `unset` |

### Observability & Tracing

| Variable                      | Description                                    | Default       |
| ----------------------------- | ---------------------------------------------- | ------------- |
| `OTEL_TRACING_ENABLED`        | Enable/disable tracing                         | `true`        |
| `OTEL_EXPORTER`               | Exporter (`jaeger`, `otlp`, `console`, `none`) | `none`        |
| `OTEL_SERVICE_NAME`           | Service name for traces                        | `ringrift-ai` |
| `OTEL_JAEGER_AGENT_HOST`      | Jaeger agent host                              | `localhost`   |
| `OTEL_JAEGER_AGENT_PORT`      | Jaeger agent port                              | `6831`        |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL                              | `unset`       |

### Rules, validation, and parity

| Variable                               | Description                                                    | Default           |
| -------------------------------------- | -------------------------------------------------------------- | ----------------- |
| `RINGRIFT_RULES_VERSION`               | Rules engine version tag                                       | `v1`              |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS`       | Skip mutator shadow contract validation                        | `true`            |
| `RINGRIFT_RULES_MUTATOR_FIRST`         | Enable mutator-first orchestration (requires server flag)      | `false`           |
| `RINGRIFT_SERVER_ENABLE_MUTATOR_FIRST` | Allow mutator-first mode at all                                | `false`           |
| `RINGRIFT_FORCE_BOOKKEEPING_MOVES`     | Synthesize required no\_\* bookkeeping moves for current actor | `false`           |
| `RINGRIFT_FSM_VALIDATION_MODE`         | FSM validation mode (`active`, `off`)                          | `active`          |
| `RINGRIFT_STRICT_NO_MOVE_INVARIANT`    | Strict legacy move invariant checks                            | `false`           |
| `RINGRIFT_SKIP_PHASE_INVARIANT`        | Skip legacy phase invariant checks                             | `false`           |
| `RINGRIFT_RECOVERY_STACK_STRIKE_V1`    | Enable recovery stack-strike fallback                          | `true`            |
| `RINGRIFT_PARITY_VALIDATION`           | Parity validation mode (`off`, `warn`, `strict`)               | `off`             |
| `RINGRIFT_PARITY_DUMP_DIR`             | Parity failure dump directory                                  | `parity_failures` |
| `RINGRIFT_NPX_PATH`                    | Override npx path for TS parity                                | `unset`           |
| `RINGRIFT_TS_REPLAY_DUMP_DIR`          | TS replay dump directory                                       | `unset`           |
| `RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K`   | TS replay dump move index                                      | `unset`           |

### Recording and replay metadata

| Variable                         | Description                   | Default                  |
| -------------------------------- | ----------------------------- | ------------------------ |
| `RINGRIFT_RECORD_SELFPLAY_GAMES` | Enable replay recording       | `true`                   |
| `RINGRIFT_SELFPLAY_DB_PATH`      | Selfplay DB path              | `data/games/selfplay.db` |
| `RINGRIFT_SNAPSHOT_INTERVAL`     | Snapshot interval (moves)     | `20`                     |
| `RINGRIFT_RULES_ENGINE_VERSION`  | Tag stored in replay metadata | `unset`                  |
| `RINGRIFT_TS_ENGINE_VERSION`     | Tag stored in replay metadata | `unset`                  |
| `RINGRIFT_AI_SERVICE_VERSION`    | Tag stored in replay metadata | `unset`                  |

### AI evaluation and search toggles

#### Heuristic and move generation

| Variable                        | Description                                         | Default |
| ------------------------------- | --------------------------------------------------- | ------- |
| `RINGRIFT_USE_FAST_TERRITORY`   | Use fast territory evaluator                        | `true`  |
| `RINGRIFT_USE_MOVE_CACHE`       | Enable move cache                                   | `true`  |
| `RINGRIFT_MOVE_CACHE_SIZE`      | Move cache size (entries)                           | `1000`  |
| `RINGRIFT_USE_MAKE_UNMAKE`      | Enable make/unmake optimization                     | `false` |
| `RINGRIFT_USE_BATCH_EVAL`       | Enable batch eval in heuristic AI                   | `false` |
| `RINGRIFT_BATCH_EVAL_THRESHOLD` | Batch eval threshold (moves)                        | `100`   |
| `RINGRIFT_EARLY_TERM_THRESHOLD` | Early termination threshold (0 = disabled)          | `0`     |
| `RINGRIFT_EARLY_TERM_MIN_MOVES` | Min moves before early termination                  | `20`    |
| `RINGRIFT_USE_PARALLEL_EVAL`    | Enable parallel eval                                | `false` |
| `RINGRIFT_PARALLEL_WORKERS`     | Worker count (0 = auto)                             | `0`     |
| `RINGRIFT_PARALLEL_MIN_MOVES`   | Min moves to use parallel eval                      | `50`    |
| `RINGRIFT_EVAL_WORKERS`         | Parallel eval workers (auto = cpu_count - 1, min 1) | `auto`  |
| `RINGRIFT_PARALLEL_THRESHOLD`   | Parallel eval threshold                             | `50`    |

#### Neural net batching and evaluation

| Variable                            | Description                             | Default |
| ----------------------------------- | --------------------------------------- | ------- |
| `RINGRIFT_NN_EVAL_QUEUE`            | Enable NN micro-batching (auto on CUDA) | `auto`  |
| `RINGRIFT_NN_EVAL_MAX_BATCH`        | Max NN micro-batch size                 | `256`   |
| `RINGRIFT_NN_EVAL_BATCH_TIMEOUT_MS` | NN micro-batch timeout (ms)             | `2`     |
| `RINGRIFT_NNUE_ZERO_SUM_EVAL`       | Use zero-sum eval for NNUE              | `true`  |
| `RINGRIFT_MINIMAX_ZERO_SUM_EVAL`    | Use zero-sum eval for minimax           | `true`  |

#### Search engines and GPU paths

| Variable                              | Description                                                 | Default |
| ------------------------------------- | ----------------------------------------------------------- | ------- |
| `RINGRIFT_DISABLE_NEURAL_NET`         | Disable neural nets (heuristic-only)                        | `false` |
| `RINGRIFT_REQUIRE_NEURAL_NET`         | Require neural net; error if missing                        | `false` |
| `RINGRIFT_VECTOR_VALUE_HEAD`          | Use vector value head in search                             | `false` |
| `RINGRIFT_MCTS_ASYNC_NN_EVAL`         | Async NN eval in MCTS                                       | `false` |
| `RINGRIFT_MCTS_LEAF_BATCH_SIZE`       | MCTS NN leaf batch size (auto: 32 CUDA / 16 MPS / 8 CPU)    | `auto`  |
| `RINGRIFT_GPU_MCTS_DISABLE`           | Disable GPU MCTS path                                       | `false` |
| `RINGRIFT_GPU_MCTS_SHADOW_VALIDATE`   | Enable GPU MCTS shadow validation                           | `false` |
| `RINGRIFT_DESCENT_ASYNC_NN_EVAL`      | Async NN eval in Descent                                    | `false` |
| `RINGRIFT_DESCENT_GPU_HEURISTIC`      | Use GPU heuristic in Descent                                | `false` |
| `RINGRIFT_DESCENT_HEURISTIC_FALLBACK` | Allow heuristic fallback in Descent                         | `true`  |
| `RINGRIFT_DESCENT_LEAF_BATCH_SIZE`    | Descent NN leaf batch size (auto: 32 CUDA / 16 MPS / 8 CPU) | `auto`  |
| `RINGRIFT_DESCENT_UCB`                | Enable uncertainty selection in Descent                     | `false` |
| `RINGRIFT_DESCENT_UCB_C`              | UCB coefficient for Descent                                 | `0.25`  |
| `RINGRIFT_GPU_GUMBEL_DISABLE`         | Disable GPU Gumbel MCTS path                                | `false` |
| `RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE` | Enable GPU Gumbel shadow validation                         | `false` |
| `RINGRIFT_GPU_TREE_SHADOW_RATE`       | GPU tree shadow rate                                        | `0.0`   |
| `RINGRIFT_GPU_MINIMAX_DISABLE`        | Disable GPU minimax                                         | `false` |
| `RINGRIFT_GPU_MINIMAX_BATCH_SIZE`     | GPU minimax batch size                                      | `64`    |
| `RINGRIFT_GPU_MAXN_DISABLE`           | Disable GPU MaxN                                            | `false` |
| `RINGRIFT_GPU_MAXN_SHADOW_VALIDATE`   | Enable GPU MaxN shadow validation                           | `false` |
| `RINGRIFT_HYBRID_DISABLE_GPU`         | Disable hybrid GPU policy                                   | `false` |
| `RINGRIFT_GPU_MOVEMENT_LEGACY`        | Use legacy GPU movement path                                | `0`     |
| `RINGRIFT_GPU_CAPTURE_LEGACY`         | Use legacy GPU capture path                                 | `0`     |

#### Model selection and policy toggles

| Variable                   | Description                 | Default |
| -------------------------- | --------------------------- | ------- |
| `RINGRIFT_USE_EBMO`        | Enable EBMO AI              | `false` |
| `RINGRIFT_USE_IG_GMO`      | Enable IG-GMO AI            | `false` |
| `RINGRIFT_USE_HYBRID_D7`   | Enable hybrid D7 policy     | `false` |
| `RINGRIFT_USE_GMO_POLICY`  | Enable GMO policy provider  | `false` |
| `RINGRIFT_EBMO_MODEL_PATH` | EBMO model path override    | `unset` |
| `RINGRIFT_CAGE_MODEL_PATH` | Cage AI model path override | `unset` |

### Training loop and dataset controls

| Variable                               | Description                                         | Default                          |
| -------------------------------------- | --------------------------------------------------- | -------------------------------- |
| `RINGRIFT_MIN_TRAINING_GAMES`          | Minimum games before training can start             | `1000`                           |
| `RINGRIFT_TRAINING_BATCH_SIZE`         | Training batch size                                 | `256`                            |
| `RINGRIFT_MIN_HOURS_BETWEEN_TRAINING`  | Minimum hours between training runs                 | `1.0`                            |
| `RINGRIFT_MIN_GAMES_TRAINING`          | Minimum games for training trigger                  | `300`                            |
| `RINGRIFT_ACCEL_MIN_GAMES`             | Accelerated min games threshold                     | `150`                            |
| `RINGRIFT_HOT_MIN_GAMES`               | Hot-path min games threshold                        | `75`                             |
| `RINGRIFT_AUTO_BATCH_SCALE`            | Auto-scale batch size on GPU                        | `true`                           |
| `RINGRIFT_DISABLE_GPU_DATAGEN`         | Disable GPU data generation                         | `false`                          |
| `RINGRIFT_LEGACY_POLICY_TRANSFORM`     | Use legacy policy index transform for training data | `false`                          |
| `RINGRIFT_AUTO_STREAMING_THRESHOLD_GB` | Auto-enable streaming if dataset exceeds (GB)       | `20`                             |
| `RINGRIFT_DATALOADER_WORKERS`          | Dataloader workers (0 on macOS; else min(4, cpu/2)) | `auto`                           |
| `RINGRIFT_DISABLE_AUTO_DISCOVERY`      | Disable automatic data discovery                    | `false`                          |
| `RINGRIFT_REPETITION_THRESHOLD`        | Repetition draw threshold (must remain 0)           | `0`                              |
| `RINGRIFT_TRAINING_ENABLE_SWAP_RULE`   | Enable swap rule during training                    | `false`                          |
| `RINGRIFT_USE_OPENING_BOOK`            | Enable opening-book selfplay                        | `0`                              |
| `RINGRIFT_SEED`                        | Training RNG seed                                   | `42`                             |
| `RINGRIFT_GIT_COMMIT`                  | Git commit tag for training metadata                | `unset`                          |
| `RINGRIFT_MODEL_DIR`                   | Model storage directory                             | `ai-service/models`              |
| `RINGRIFT_NOTIFICATION_CONFIG`         | Notification hooks config path                      | `config/notification_hooks.yaml` |
| `RINGRIFT_DISTRIBUTED_BACKEND`         | Distributed backend override                        | `auto`                           |
| `RINGRIFT_BOARD_TYPE`                  | Training board type override                        | `square8`                        |
| `RINGRIFT_LEARNING_RATE`               | Training learning rate override                     | `board-specific default`         |
| `RINGRIFT_BATCH_SIZE`                  | Training batch size override                        | `board-specific default`         |
| `RINGRIFT_EPOCHS`                      | Training epochs override                            | `board-specific default`         |
| `RINGRIFT_DATA_DIR`                    | Training data directory override                    | `board-specific default`         |
| `RINGRIFT_CHECKPOINT_DIR`              | Checkpoint directory override                       | `board-specific default`         |

### Training config prefix overrides

#### RINGRIFT*CMAES* (CMA-ES tuning)

| Variable                               | Default         | Description                 |
| -------------------------------------- | --------------- | --------------------------- |
| `RINGRIFT_CMAES_GENERATIONS`           | `20`            | CMA-ES generations          |
| `RINGRIFT_CMAES_POPULATION_SIZE`       | `16`            | Population size             |
| `RINGRIFT_CMAES_SIGMA`                 | `0.5`           | Initial sigma               |
| `RINGRIFT_CMAES_GAMES_PER_EVAL`        | `24`            | Games per evaluation        |
| `RINGRIFT_CMAES_MAX_MOVES`             | `10000`         | Max moves per game          |
| `RINGRIFT_CMAES_EVAL_RANDOMNESS`       | `0.02`          | Eval randomness             |
| `RINGRIFT_CMAES_BOARD_TYPE`            | `square8`       | Board type                  |
| `RINGRIFT_CMAES_EVAL_BOARDS`           | `square8`       | Comma list of boards        |
| `RINGRIFT_CMAES_NUM_PLAYERS`           | `2`             | Player count                |
| `RINGRIFT_CMAES_STATE_POOL_ID`         | `v1`            | State pool id               |
| `RINGRIFT_CMAES_EVAL_MODE`             | `multi-start`   | Eval mode                   |
| `RINGRIFT_CMAES_OPPONENT_MODE`         | `baseline-only` | Opponent mode               |
| `RINGRIFT_CMAES_DISTRIBUTED`           | `false`         | Enable distributed CMA-ES   |
| `RINGRIFT_CMAES_WORKERS`               | `empty`         | Comma list of workers       |
| `RINGRIFT_CMAES_EVAL_TIMEOUT`          | `300.0`         | Eval timeout (seconds)      |
| `RINGRIFT_CMAES_OUTPUT_DIR`            | `logs/cmaes`    | Output directory            |
| `RINGRIFT_CMAES_SEED`                  | `42`            | RNG seed                    |
| `RINGRIFT_CMAES_PROGRESS_INTERVAL_SEC` | `30`            | Progress interval (seconds) |

#### RINGRIFT*NN* (Neural net training)

| Variable                           | Default                     | Description                           |
| ---------------------------------- | --------------------------- | ------------------------------------- |
| `RINGRIFT_NN_HIDDEN_LAYERS`        | `512,512,512,512,512`       | Hidden layer sizes                    |
| `RINGRIFT_NN_INPUT_CHANNELS`       | `17`                        | Input channels                        |
| `RINGRIFT_NN_POLICY_HEAD_CHANNELS` | `32`                        | Policy head channels                  |
| `RINGRIFT_NN_VALUE_HEAD_CHANNELS`  | `32`                        | Value head channels                   |
| `RINGRIFT_NN_BATCH_SIZE`           | `64`                        | Batch size                            |
| `RINGRIFT_NN_LEARNING_RATE`        | `0.0009`                    | Learning rate                         |
| `RINGRIFT_NN_WEIGHT_DECAY`         | `0.00028`                   | Weight decay                          |
| `RINGRIFT_NN_EPOCHS`               | `100`                       | Epochs                                |
| `RINGRIFT_NN_WARMUP_EPOCHS`        | `5`                         | Warmup epochs                         |
| `RINGRIFT_NN_POLICY_WEIGHT`        | `1.66`                      | Policy loss weight                    |
| `RINGRIFT_NN_VALUE_WEIGHT`         | `1.82`                      | Value loss weight                     |
| `RINGRIFT_NN_BOARD_TYPE`           | `square8`                   | Board type                            |
| `RINGRIFT_NN_HISTORY_LENGTH`       | `3`                         | History length                        |
| `RINGRIFT_NN_AUGMENT_ROTATIONS`    | `true`                      | Augment rotations                     |
| `RINGRIFT_NN_AUGMENT_REFLECTIONS`  | `true`                      | Augment reflections                   |
| `RINGRIFT_NN_DEVICE`               | `auto`                      | Device (`auto`, `cpu`, `cuda`, `mps`) |
| `RINGRIFT_NN_NUM_WORKERS`          | `4`                         | Data loader workers                   |
| `RINGRIFT_NN_PIN_MEMORY`           | `true`                      | Pin memory                            |
| `RINGRIFT_NN_MODEL_ID`             | `ringrift_v5_sq8_2p_2xh100` | Model id / checkpoint lineage         |
| `RINGRIFT_NN_CHECKPOINT_DIR`       | `models`                    | Checkpoint directory                  |
| `RINGRIFT_NN_SAVE_EVERY_N_EPOCHS`  | `10`                        | Save frequency                        |
| `RINGRIFT_NN_LOG_DIR`              | `logs/tensorboard`          | Log directory                         |
| `RINGRIFT_NN_LOG_EVERY_N_STEPS`    | `100`                       | Log frequency                         |
| `RINGRIFT_NN_DATA_DIR`             | `data/training`             | Data directory                        |
| `RINGRIFT_NN_SEED`                 | `42`                        | RNG seed                              |

#### RINGRIFT*SELFPLAY* (Legacy selfplay config)

| Variable                                  | Default         | Description           |
| ----------------------------------------- | --------------- | --------------------- |
| `RINGRIFT_SELFPLAY_NUM_GAMES`             | `1000`          | Number of games       |
| `RINGRIFT_SELFPLAY_MAX_MOVES_PER_GAME`    | `10000`         | Max moves per game    |
| `RINGRIFT_SELFPLAY_PARALLEL_GAMES`        | `4`             | Parallel games        |
| `RINGRIFT_SELFPLAY_AI_TYPE`               | `heuristic`     | AI type               |
| `RINGRIFT_SELFPLAY_TEMPERATURE`           | `1.0`           | Temperature           |
| `RINGRIFT_SELFPLAY_TEMPERATURE_DROP_MOVE` | `30`            | Temperature drop move |
| `RINGRIFT_SELFPLAY_EXPLORATION_FRACTION`  | `0.25`          | Exploration fraction  |
| `RINGRIFT_SELFPLAY_BOARD_TYPE`            | `square8`       | Board type            |
| `RINGRIFT_SELFPLAY_NUM_PLAYERS`           | `2`             | Player count          |
| `RINGRIFT_SELFPLAY_OUTPUT_URI`            | `data/selfplay` | Output URI            |
| `RINGRIFT_SELFPLAY_COMPRESS`              | `true`          | Compress output       |
| `RINGRIFT_SELFPLAY_BATCH_SIZE`            | `1000`          | Batch size            |
| `RINGRIFT_SELFPLAY_SEED`                  | `42`            | RNG seed              |

#### RINGRIFT*GPU* (GPU batch scaling)

| Variable                              | Default | Description            |
| ------------------------------------- | ------- | ---------------------- |
| `RINGRIFT_GPU_RESERVED_MEMORY_GB`     | `8.0`   | Reserved GPU memory    |
| `RINGRIFT_GPU_MAX_BATCH_SIZE`         | `16384` | Max scaled batch size  |
| `RINGRIFT_GPU_GH200_BATCH_MULTIPLIER` | `64`    | GH200 batch multiplier |
| `RINGRIFT_GPU_H100_BATCH_MULTIPLIER`  | `32`    | H100 batch multiplier  |
| `RINGRIFT_GPU_A100_BATCH_MULTIPLIER`  | `16`    | A100 batch multiplier  |
| `RINGRIFT_GPU_RTX_BATCH_MULTIPLIER`   | `8`     | RTX batch multiplier   |

### Cluster, P2P, and coordination runtime flags

#### Core cluster and routing

| Variable                           | Description                            | Default                 |
| ---------------------------------- | -------------------------------------- | ----------------------- |
| `RINGRIFT_COORDINATOR_URL`         | Coordinator service URL                | `unset`                 |
| `RINGRIFT_COORDINATOR_IP`          | Coordinator IP address                 | `unset`                 |
| `RINGRIFT_P2P_URL`                 | P2P service URL                        | `http://localhost:8770` |
| `RINGRIFT_P2P_SEEDS`               | Comma list of P2P seed nodes           | `unset`                 |
| `RINGRIFT_P2P_LEADER_URL`          | P2P leader URL override                | `unset`                 |
| `RINGRIFT_P2P_AGENT_MODE`          | Enable P2P agent mode                  | `false`                 |
| `RINGRIFT_CLUSTER_AUTH_TOKEN`      | Cluster auth token                     | `unset`                 |
| `RINGRIFT_CLUSTER_AUTH_TOKEN_FILE` | Cluster auth token file                | `unset`                 |
| `RINGRIFT_BUILD_VERSION`           | Build/version label                    | `dev`                   |
| `RINGRIFT_DISABLE_LOCAL_TASKS`     | Disable local tasks (coordinator-only) | `false`                 |
| `RINGRIFT_HEALTH_PORT`             | Health server port (daemon manager)    | `8790`                  |
| `RINGRIFT_DISCOVERY_INTERVAL`      | Discovery interval (seconds)           | `60`                    |
| `RINGRIFT_IDLE_CHECK_INTERVAL`     | Idle check interval (seconds)          | `60`                    |
| `RINGRIFT_IDLE_THRESHOLD`          | Idle GPU utilization threshold (%)     | `10.0`                  |
| `RINGRIFT_IDLE_DURATION`           | Seconds before a resource is idle      | `120`                   |
| `RINGRIFT_IDLE_RESOURCE_ENABLED`   | Enable idle resource daemon            | `true`                  |
| `RINGRIFT_AUTO_ASSIGN_ENABLED`     | Auto-assign work to idle nodes         | `true`                  |
| `RINGRIFT_MAX_DISK_PERCENT`        | Max disk usage percent                 | `70.0`                  |

#### Node role and workload gating

| Variable                    | Description                                                        | Default |
| --------------------------- | ------------------------------------------------------------------ | ------- |
| `RINGRIFT_IS_COORDINATOR`   | Force coordinator-only mode (no selfplay/training/gauntlet/export) | `auto`  |
| `RINGRIFT_SELFPLAY_ENABLED` | Explicit override for selfplay on this node                        | `auto`  |
| `RINGRIFT_TRAINING_ENABLED` | Explicit override for training on this node                        | `auto`  |
| `RINGRIFT_GAUNTLET_ENABLED` | Explicit override for gauntlet/evaluation on this node             | `auto`  |
| `RINGRIFT_EXPORT_ENABLED`   | Explicit override for export jobs on this node                     | `auto`  |

#### Process and idle safeguards

| Variable                                      | Description                                    | Default |
| --------------------------------------------- | ---------------------------------------------- | ------- |
| `RINGRIFT_JOB_GRACE_PERIOD`                   | Seconds to wait before SIGKILL after SIGTERM   | `60`    |
| `RINGRIFT_GPU_IDLE_THRESHOLD`                 | Seconds of GPU idle before killing processes   | `600`   |
| `RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD` | Max selfplay processes per node before cleanup | `128`   |

#### P2P resource thresholds and behavior

| Variable                                   | Description                           | Default |
| ------------------------------------------ | ------------------------------------- | ------- |
| `RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD`     | Disk critical threshold (%)           | `70`    |
| `RINGRIFT_P2P_DISK_WARNING_THRESHOLD`      | Disk warning threshold (%)            | `65`    |
| `RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD`      | Disk cleanup threshold (%)            | `65`    |
| `RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD`   | Memory critical threshold (%)         | `95`    |
| `RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD`    | Memory warning threshold (%)          | `85`    |
| `RINGRIFT_P2P_MIN_MEMORY_GB`               | Minimum memory for tasks (GB)         | `64`    |
| `RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS`       | Max load for new jobs (%)             | `85`    |
| `RINGRIFT_P2P_TARGET_GPU_UTIL_MIN`         | Target GPU util min (%)               | `60`    |
| `RINGRIFT_P2P_TARGET_GPU_UTIL_MAX`         | Target GPU util max (%)               | `90`    |
| `RINGRIFT_P2P_GH200_MIN_SELFPLAY`          | GH200 min selfplay jobs               | `20`    |
| `RINGRIFT_P2P_GH200_MAX_SELFPLAY`          | GH200 max selfplay jobs               | `100`   |
| `RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS`   | Peer retire timeout (seconds)         | `3600`  |
| `RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL` | Retry retired node interval (seconds) | `3600`  |
| `RINGRIFT_P2P_LOAD_AVG_MAX_MULT`           | Load average max multiplier           | `2.0`   |
| `RINGRIFT_P2P_SPAWN_RATE_LIMIT`            | Spawn rate limit per minute           | `5`     |
| `RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL`   | Git update check interval (seconds)   | `300`   |
| `RINGRIFT_P2P_AUTO_UPDATE`                 | Enable P2P auto-update                | `false` |

#### Distributed execution and storage

| Variable                         | Description                                                    | Default                             |
| -------------------------------- | -------------------------------------------------------------- | ----------------------------------- |
| `RINGRIFT_COORDINATOR_DIR`       | Coordinator state dir                                          | `/tmp/ringrift_coordinator`         |
| `RINGRIFT_WORK_QUEUE_DB`         | Work queue DB path                                             | `data/work_queue.db`                |
| `RINGRIFT_NFS_COORDINATION_PATH` | NFS coordination path (legacy Lambda path; override as needed) | `/lambda/nfs/RingRift/coordination` |
| `RINGRIFT_NFS_PATH`              | NFS data root                                                  | `/mnt/nfs/ringrift`                 |
| `RINGRIFT_AI_SERVICE_DIR`        | AI service working dir                                         | `.`                                 |
| `RINGRIFT_AI_SERVICE_PATH`       | AI service root override                                       | `repo auto-detect`                  |
| `RINGRIFT_P2P_PORT`              | P2P HTTP port                                                  | `8770`                              |
| `RINGRIFT_SSH_USER`              | Default SSH user                                               | `ubuntu`                            |
| `RINGRIFT_SSH_KEY`               | Default SSH key path                                           | `unset`                             |
| `RINGRIFT_VAST_SSH_USER`         | Vast.ai SSH user                                               | `root`                              |
| `RINGRIFT_STORAGE_PROVIDER`      | Storage provider override (`lambda` legacy, `vast`, `local`)   | `auto`                              |
| `RINGRIFT_TARGET_UTIL_MIN`       | Utilization target min (%)                                     | `60`                                |
| `RINGRIFT_TARGET_UTIL_MAX`       | Utilization target max (%)                                     | `80`                                |
| `RINGRIFT_SCALE_UP_THRESHOLD`    | Scale-up threshold (%)                                         | `55`                                |
| `RINGRIFT_SCALE_DOWN_THRESHOLD`  | Scale-down threshold (%)                                       | `85`                                |
| `RINGRIFT_NODE_ID`               | Node id override (default hostname)                            | `auto`                              |
| `RINGRIFT_ORCHESTRATOR`          | Orchestrator id                                                | `unknown`                           |

#### Provider idle controls (Lambda)

> **Note:** Lambda account is suspended; these remain for eventual restoration.

| Variable                         | Description                              | Default |
| -------------------------------- | ---------------------------------------- | ------- |
| `RINGRIFT_LAMBDA_IDLE_ENABLED`   | Enable Lambda idle shutdown daemon       | `true`  |
| `RINGRIFT_LAMBDA_IDLE_INTERVAL`  | Lambda idle check interval (seconds)     | `300`   |
| `RINGRIFT_LAMBDA_IDLE_THRESHOLD` | Lambda GPU idle threshold (%)            | `5.0`   |
| `RINGRIFT_LAMBDA_IDLE_DURATION`  | Lambda idle duration threshold (seconds) | `1800`  |
| `RINGRIFT_LAMBDA_MIN_NODES`      | Minimum Lambda nodes to keep running     | `1`     |

#### Coordination defaults (advanced)

##### Locking and transport

| Variable                            | Description                           | Default |
| ----------------------------------- | ------------------------------------- | ------- |
| `RINGRIFT_LOCK_TIMEOUT`             | Lock timeout (seconds)                | `3600`  |
| `RINGRIFT_LOCK_ACQUIRE_TIMEOUT`     | Lock acquire timeout (seconds)        | `60`    |
| `RINGRIFT_LOCK_RETRY_INTERVAL`      | Lock retry interval (seconds)         | `1.0`   |
| `RINGRIFT_TRAINING_LOCK_TIMEOUT`    | Training lock timeout (seconds)       | `7200`  |
| `RINGRIFT_CONNECT_TIMEOUT`          | Transport connect timeout (seconds)   | `30`    |
| `RINGRIFT_OPERATION_TIMEOUT`        | Transport operation timeout (seconds) | `180`   |
| `RINGRIFT_HTTP_TIMEOUT`             | HTTP timeout (seconds)                | `30`    |
| `RINGRIFT_CIRCUIT_BREAKER_RECOVERY` | Circuit breaker recovery (seconds)    | `300`   |
| `RINGRIFT_SSH_TIMEOUT`              | SSH timeout (seconds)                 | `60`    |
| `RINGRIFT_MAX_RETRIES`              | Transport max retries                 | `3`     |

##### Sync and heartbeat

| Variable                          | Description                       | Default |
| --------------------------------- | --------------------------------- | ------- |
| `RINGRIFT_SYNC_LOCK_TIMEOUT`      | Sync lock timeout (seconds)       | `120`   |
| `RINGRIFT_MAX_SYNCS_PER_HOST`     | Max concurrent syncs per host     | `1`     |
| `RINGRIFT_MAX_SYNCS_CLUSTER`      | Max concurrent syncs cluster-wide | `5`     |
| `RINGRIFT_DATA_SYNC_INTERVAL`     | Data sync interval (seconds)      | `300.0` |
| `RINGRIFT_MODEL_SYNC_INTERVAL`    | Model sync interval (seconds)     | `600.0` |
| `RINGRIFT_ELO_SYNC_INTERVAL`      | Elo sync interval (seconds)       | `60.0`  |
| `RINGRIFT_REGISTRY_SYNC_INTERVAL` | Registry sync interval (seconds)  | `120.0` |
| `RINGRIFT_HEARTBEAT_INTERVAL`     | Heartbeat interval (seconds)      | `30`    |
| `RINGRIFT_HEARTBEAT_TIMEOUT`      | Heartbeat timeout (seconds)       | `90`    |
| `RINGRIFT_STALE_CLEANUP_INTERVAL` | Stale cleanup interval (seconds)  | `60`    |

##### Training scheduler and limits

| Variable                            | Description                             | Default |
| ----------------------------------- | --------------------------------------- | ------- |
| `RINGRIFT_MAX_TRAINING_SAME_CONFIG` | Max concurrent training per config      | `1`     |
| `RINGRIFT_MAX_TRAINING_TOTAL`       | Max concurrent training total           | `3`     |
| `RINGRIFT_TRAINING_TIMEOUT_HOURS`   | Training timeout (hours)                | `24.0`  |
| `RINGRIFT_TRAINING_MIN_INTERVAL`    | Min interval between training (seconds) | `1200`  |
| `RINGRIFT_MIN_MEMORY_GB`            | Min memory for scheduler (GB)           | `64`    |
| `RINGRIFT_MAX_QUEUE_SIZE`           | Scheduler max queue size                | `1000`  |
| `RINGRIFT_MAX_SELFPLAY_CLUSTER`     | Max selfplay tasks cluster-wide         | `500`   |
| `RINGRIFT_HEALTH_CACHE_TTL`         | Scheduler health cache TTL (seconds)    | `30`    |

##### Ephemeral data guard and circuit breaker

| Variable                               | Description                                | Default |
| -------------------------------------- | ------------------------------------------ | ------- |
| `RINGRIFT_CHECKPOINT_INTERVAL`         | Ephemeral checkpoint interval (seconds)    | `300`   |
| `RINGRIFT_EPHEMERAL_HEARTBEAT_TIMEOUT` | Ephemeral heartbeat timeout (seconds)      | `120`   |
| `RINGRIFT_CB_FAILURE_THRESHOLD`        | Circuit breaker failure threshold          | `5`     |
| `RINGRIFT_CB_RECOVERY_TIMEOUT`         | Circuit breaker recovery timeout (seconds) | `60.0`  |
| `RINGRIFT_CB_MAX_BACKOFF`              | Circuit breaker max backoff (seconds)      | `600.0` |
| `RINGRIFT_CB_HALF_OPEN_MAX_CALLS`      | Circuit breaker half-open max calls        | `1`     |

##### Health, utilization, and bandwidth

| Variable                                | Description                                 | Default |
| --------------------------------------- | ------------------------------------------- | ------- |
| `RINGRIFT_HEALTH_SSH_TIMEOUT`           | Health SSH timeout (seconds)                | `5`     |
| `RINGRIFT_HEALTHY_CACHE_TTL`            | Healthy cache TTL (seconds)                 | `60`    |
| `RINGRIFT_UNHEALTHY_CACHE_TTL`          | Unhealthy cache TTL (seconds)               | `30`    |
| `RINGRIFT_MAX_HEALTH_CHECKS`            | Max concurrent health checks                | `10`    |
| `RINGRIFT_MIN_HEALTHY_HOSTS`            | Min healthy hosts                           | `2`     |
| `RINGRIFT_CLUSTER_HEALTH_CACHE_TTL`     | Cluster health cache TTL (seconds)          | `120`   |
| `RINGRIFT_GPU_TARGET_MIN`               | GPU utilization target min (%)              | `60`    |
| `RINGRIFT_GPU_TARGET_MAX`               | GPU utilization target max (%)              | `80`    |
| `RINGRIFT_CPU_TARGET_MIN`               | CPU utilization target min (%)              | `60`    |
| `RINGRIFT_CPU_TARGET_MAX`               | CPU utilization target max (%)              | `80`    |
| `RINGRIFT_UTILIZATION_UPDATE_INTERVAL`  | Utilization update interval (seconds)       | `10`    |
| `RINGRIFT_OPTIMIZATION_INTERVAL`        | Utilization optimization interval (seconds) | `30`    |
| `RINGRIFT_MAX_CONCURRENT_TRANSFERS`     | Max concurrent transfers per host           | `3`     |
| `RINGRIFT_BANDWIDTH_MEASUREMENT_WINDOW` | Bandwidth measurement window (seconds)      | `300`   |
| `RINGRIFT_DEFAULT_UPLOAD_MBPS`          | Default upload limit (MB/s)                 | `100`   |
| `RINGRIFT_DEFAULT_DOWNLOAD_MBPS`        | Default download limit (MB/s)               | `1000`  |

##### Resource limits and PID tuning

| Variable                                 | Description                        | Default |
| ---------------------------------------- | ---------------------------------- | ------- |
| `RINGRIFT_CONSUMER_MAX_SELFPLAY`         | Max selfplay (consumer tier)       | `16`    |
| `RINGRIFT_PROSUMER_MAX_SELFPLAY`         | Max selfplay (prosumer tier)       | `32`    |
| `RINGRIFT_DATACENTER_MAX_SELFPLAY`       | Max selfplay (datacenter tier)     | `64`    |
| `RINGRIFT_HIGH_CPU_MAX_SELFPLAY`         | Max selfplay (high CPU tier)       | `128`   |
| `RINGRIFT_PID_KP`                        | PID proportional gain              | `0.3`   |
| `RINGRIFT_PID_KI`                        | PID integral gain                  | `0.05`  |
| `RINGRIFT_PID_KD`                        | PID derivative gain                | `0.1`   |
| `RINGRIFT_BACKPRESSURE_GPU_THRESHOLD`    | Backpressure GPU threshold (%)     | `90.0`  |
| `RINGRIFT_BACKPRESSURE_MEMORY_THRESHOLD` | Backpressure memory threshold (%)  | `85.0`  |
| `RINGRIFT_BACKPRESSURE_DISK_THRESHOLD`   | Backpressure disk threshold (%)    | `90.0`  |
| `RINGRIFT_RESOURCE_UPDATE_INTERVAL`      | Resource update interval (seconds) | `10`    |
| `RINGRIFT_BACKPRESSURE_COOLDOWN`         | Backpressure cooldown (seconds)    | `30`    |

##### Optimization, metrics, and cache

| Variable                                  | Description                      | Default |
| ----------------------------------------- | -------------------------------- | ------- |
| `RINGRIFT_OPTIMIZATION_PLATEAU_WINDOW`    | Optimization plateau window      | `10`    |
| `RINGRIFT_OPTIMIZATION_PLATEAU_THRESHOLD` | Optimization plateau threshold   | `0.001` |
| `RINGRIFT_OPTIMIZATION_MIN_EPOCHS`        | Min epochs between optimizations | `20`    |
| `RINGRIFT_OPTIMIZATION_COOLDOWN`          | Optimization cooldown (seconds)  | `300.0` |
| `RINGRIFT_OPTIMIZATION_MAX_HISTORY`       | Optimization history max         | `100`   |
| `RINGRIFT_METRICS_WINDOW_SIZE`            | Metrics window size              | `100`   |
| `RINGRIFT_METRICS_PLATEAU_THRESHOLD`      | Metrics plateau threshold        | `0.001` |
| `RINGRIFT_METRICS_PLATEAU_WINDOW`         | Metrics plateau window           | `10`    |
| `RINGRIFT_METRICS_REGRESSION_THRESHOLD`   | Metrics regression threshold     | `0.05`  |
| `RINGRIFT_METRICS_ANOMALY_THRESHOLD`      | Metrics anomaly threshold        | `3.0`   |
| `RINGRIFT_CACHE_DEFAULT_TTL`              | Cache default TTL (seconds)      | `3600`  |
| `RINGRIFT_CACHE_MAX_ENTRIES_PER_NODE`     | Cache max entries per node       | `100`   |
| `RINGRIFT_CACHE_CLEANUP_INTERVAL`         | Cache cleanup interval (seconds) | `300`   |
| `RINGRIFT_CACHE_STALE_THRESHOLD`          | Cache stale threshold (seconds)  | `7200`  |

##### Task lifecycle defaults

| Variable                          | Description                        | Default |
| --------------------------------- | ---------------------------------- | ------- |
| `RINGRIFT_TASK_HEARTBEAT_TIMEOUT` | Task heartbeat timeout (seconds)   | `60.0`  |
| `RINGRIFT_TASK_ORPHAN_GRACE`      | Task orphan grace period (seconds) | `30.0`  |
| `RINGRIFT_TASK_MAX_HISTORY`       | Task history max entries           | `1000`  |
| `RINGRIFT_TASK_CLEANUP_INTERVAL`  | Task cleanup interval (seconds)    | `60`    |

##### Queue, scaling, and duration defaults

| Variable                               | Description                             | Default  |
| -------------------------------------- | --------------------------------------- | -------- |
| `RINGRIFT_QUEUE_TRAINING_SOFT`         | Training queue soft limit               | `100000` |
| `RINGRIFT_QUEUE_TRAINING_HARD`         | Training queue hard limit               | `500000` |
| `RINGRIFT_QUEUE_TRAINING_TARGET`       | Training queue target                   | `50000`  |
| `RINGRIFT_QUEUE_GAMES_SOFT`            | Games queue soft limit                  | `1000`   |
| `RINGRIFT_QUEUE_GAMES_HARD`            | Games queue hard limit                  | `5000`   |
| `RINGRIFT_QUEUE_GAMES_TARGET`          | Games queue target                      | `500`    |
| `RINGRIFT_QUEUE_EVAL_SOFT`             | Eval queue soft limit                   | `50`     |
| `RINGRIFT_QUEUE_EVAL_HARD`             | Eval queue hard limit                   | `200`    |
| `RINGRIFT_QUEUE_EVAL_TARGET`           | Eval queue target                       | `20`     |
| `RINGRIFT_QUEUE_SYNC_SOFT`             | Sync queue soft limit                   | `100`    |
| `RINGRIFT_QUEUE_SYNC_HARD`             | Sync queue hard limit                   | `500`    |
| `RINGRIFT_QUEUE_SYNC_TARGET`           | Sync queue target                       | `50`     |
| `RINGRIFT_SCALE_UP_QUEUE_DEPTH`        | Scale-up queue depth                    | `100`    |
| `RINGRIFT_SCALE_DOWN_QUEUE_DEPTH`      | Scale-down queue depth                  | `10`     |
| `RINGRIFT_SCALE_DOWN_IDLE_MINUTES`     | Scale-down idle minutes                 | `30`     |
| `RINGRIFT_SCALE_UP_COOLDOWN_MINUTES`   | Scale-up cooldown minutes               | `5`      |
| `RINGRIFT_SCALE_DOWN_COOLDOWN_MINUTES` | Scale-down cooldown minutes             | `10`     |
| `RINGRIFT_MAX_INSTANCES`               | Max instances                           | `10`     |
| `RINGRIFT_MIN_INSTANCES`               | Min instances                           | `1`      |
| `RINGRIFT_GPU_SCALE_UP_THRESHOLD`      | GPU scale-up threshold (%)              | `85`     |
| `RINGRIFT_GPU_SCALE_DOWN_THRESHOLD`    | GPU scale-down threshold (%)            | `30`     |
| `RINGRIFT_MAX_HOURLY_COST`             | Max hourly cost                         | `10.0`   |
| `RINGRIFT_DURATION_SELFPLAY`           | Default selfplay duration (seconds)     | `3600`   |
| `RINGRIFT_DURATION_GPU_SELFPLAY`       | Default GPU selfplay duration (seconds) | `7200`   |
| `RINGRIFT_DURATION_TRAINING`           | Default training duration (seconds)     | `14400`  |
| `RINGRIFT_DURATION_CMAES`              | Default CMA-ES duration (seconds)       | `28800`  |
| `RINGRIFT_DURATION_TOURNAMENT`         | Default tournament duration (seconds)   | `1800`   |
| `RINGRIFT_DURATION_EVALUATION`         | Default evaluation duration (seconds)   | `3600`   |
| `RINGRIFT_DURATION_SYNC`               | Default sync duration (seconds)         | `600`    |
| `RINGRIFT_DURATION_EXPORT`             | Default export duration (seconds)       | `300`    |
| `RINGRIFT_DURATION_PIPELINE`           | Default pipeline duration (seconds)     | `21600`  |
| `RINGRIFT_DURATION_IMPROVEMENT`        | Default improvement duration (seconds)  | `43200`  |
| `RINGRIFT_PEAK_HOURS_START`            | Peak hours start (UTC hour)             | `14`     |
| `RINGRIFT_PEAK_HOURS_END`              | Peak hours end (UTC hour)               | `22`     |

##### Sync coordinator, timeouts, and retry defaults

| Variable                            | Description                        | Default |
| ----------------------------------- | ---------------------------------- | ------- |
| `RINGRIFT_SYNC_CRITICAL_STALE`      | Critical stale threshold (seconds) | `3600`  |
| `RINGRIFT_SYNC_FRESHNESS_INTERVAL`  | Sync freshness interval (seconds)  | `60`    |
| `RINGRIFT_SYNC_FULL_INTERVAL`       | Full sync interval (seconds)       | `3600`  |
| `RINGRIFT_HEALTH_CHECK_TIMEOUT`     | Health check timeout (seconds)     | `5`     |
| `RINGRIFT_URL_FETCH_QUICK_TIMEOUT`  | URL fetch quick timeout (seconds)  | `5`     |
| `RINGRIFT_URL_FETCH_TIMEOUT`        | URL fetch timeout (seconds)        | `10`    |
| `RINGRIFT_RSYNC_TIMEOUT`            | Rsync timeout (seconds)            | `30`    |
| `RINGRIFT_ASYNC_SUBPROCESS_TIMEOUT` | Async subprocess timeout (seconds) | `180`   |
| `RINGRIFT_THREAD_JOIN_TIMEOUT`      | Thread join timeout (seconds)      | `5`     |
| `RINGRIFT_FUTURE_RESULT_TIMEOUT`    | Future result timeout (seconds)    | `300`   |
| `RINGRIFT_CHECKPOINT_TIMEOUT`       | Checkpoint timeout (seconds)       | `120`   |
| `RINGRIFT_TRAINING_JOB_TIMEOUT`     | Training job timeout (seconds)     | `14400` |
| `RINGRIFT_RESOURCE_WAIT_TIMEOUT`    | Resource wait timeout (seconds)    | `300`   |
| `RINGRIFT_BATCH_OPERATION_TIMEOUT`  | Batch operation timeout (seconds)  | `1800`  |
| `RINGRIFT_PER_FILE_TIMEOUT`         | Per-file timeout (seconds)         | `120`   |
| `RINGRIFT_DEFAULT_MAX_RETRIES`      | Default retry count                | `3`     |
| `RINGRIFT_RETRY_BASE_DELAY`         | Retry base delay (seconds)         | `1.0`   |
| `RINGRIFT_RETRY_MAX_DELAY`          | Retry max delay (seconds)          | `60.0`  |
| `RINGRIFT_RETRY_BACKOFF_MULTIPLIER` | Retry backoff multiplier           | `2.0`   |
| `RINGRIFT_RETRY_JITTER_FACTOR`      | Retry jitter factor                | `0.1`   |

## DataIngestionConfig

Configuration for data ingestion from remote hosts.

| Option                             | Type | Default            | Description                                 |
| ---------------------------------- | ---- | ------------------ | ------------------------------------------- |
| `poll_interval_seconds`            | int  | 60                 | Interval between data sync polls            |
| `ephemeral_poll_interval_seconds`  | int  | 15                 | Aggressive sync interval for RAM disk hosts |
| `sync_method`                      | str  | "incremental"      | Sync method ("incremental" or "full")       |
| `deduplication`                    | bool | True               | Enable game deduplication                   |
| `min_games_per_sync`               | int  | 5                  | Minimum games before triggering sync        |
| `remote_db_pattern`                | str  | "data/games/\*.db" | Pattern for remote database files           |
| `sync_disabled`                    | bool | False              | Disable sync on this machine                |
| `use_external_sync`                | bool | False              | Use external unified_data_sync.py           |
| `checksum_validation`              | bool | True               | Validate checksums during sync              |
| `retry_max_attempts`               | int  | 3                  | Maximum retry attempts                      |
| `retry_base_delay_seconds`         | int  | 5                  | Base delay between retries                  |
| `dead_letter_enabled`              | bool | True               | Enable dead letter queue for failed syncs   |
| `wal_enabled`                      | bool | True               | Enable write-ahead logging                  |
| `wal_db_path`                      | str  | "data/sync_wal.db" | WAL database path                           |
| `elo_replication_enabled`          | bool | True               | Enable Elo database replication             |
| `elo_replication_interval_seconds` | int  | 60                 | Elo replication interval                    |

---

## TrainingConfig

Configuration for automatic training triggers. **This is the single source of truth for training thresholds.**

| Option                    | Type  | Default                               | Description                                     |
| ------------------------- | ----- | ------------------------------------- | ----------------------------------------------- |
| `trigger_threshold_games` | int   | 500                                   | Games needed to trigger training                |
| `min_interval_seconds`    | int   | 1200                                  | Minimum interval between training runs (20 min) |
| `max_concurrent_jobs`     | int   | 1                                     | Maximum concurrent training jobs                |
| `prefer_gpu_hosts`        | bool  | True                                  | Prefer GPU hosts for training                   |
| `nn_training_script`      | str   | "scripts/run_nn_training_baseline.py" | Neural network training script                  |
| `export_script`           | str   | "scripts/export_replay_dataset.py"    | Dataset export script                           |
| `hex_encoder_version`     | str   | "v3"                                  | Hex encoder version                             |
| `warm_start`              | bool  | True                                  | Enable warm start from previous weights         |
| `validation_split`        | float | 0.1                                   | Validation data split ratio                     |
| `nnue_min_games`          | int   | 10000                                 | Minimum games for NNUE training                 |
| `nnue_policy_min_games`   | int   | 5000                                  | Minimum games for NNUE policy training          |
| `cmaes_min_games`         | int   | 20000                                 | Minimum games for CMA-ES training               |

---

## EvaluationConfig

Configuration for model evaluation (shadow and full tournaments).

| Option                             | Type      | Default                                         | Description                            |
| ---------------------------------- | --------- | ----------------------------------------------- | -------------------------------------- |
| `shadow_interval_seconds`          | int       | 900                                             | Shadow evaluation interval (15 min)    |
| `shadow_games_per_config`          | int       | 15                                              | Games per configuration in shadow eval |
| `full_tournament_interval_seconds` | int       | 3600                                            | Full tournament interval (1 hour)      |
| `full_tournament_games`            | int       | 50                                              | Games in full tournament               |
| `baseline_models`                  | List[str] | ["random", "heuristic", "mcts_100", "mcts_500"] | Baseline models for comparison         |
| `min_games_for_elo`                | int       | 30                                              | Minimum games for Elo calculation      |
| `elo_k_factor`                     | int       | 32                                              | Elo K-factor                           |

---

## PromotionConfig

Configuration for automatic model promotion.

| Option                   | Type  | Default | Description                           |
| ------------------------ | ----- | ------- | ------------------------------------- |
| `auto_promote`           | bool  | True    | Enable automatic promotion            |
| `elo_threshold`          | int   | 25      | Minimum Elo improvement for promotion |
| `min_games`              | int   | 50      | Minimum games before promotion        |
| `significance_level`     | float | 0.05    | Statistical significance level        |
| `sync_to_cluster`        | bool  | True    | Sync promoted models to cluster       |
| `cooldown_seconds`       | int   | 1800    | Cooldown between promotions (30 min)  |
| `max_promotions_per_day` | int   | 10      | Maximum promotions per day            |
| `regression_test`        | bool  | True    | Run regression tests before promotion |

---

## CurriculumConfig

Configuration for adaptive curriculum (Elo-weighted training).

| Option                       | Type  | Default | Description                            |
| ---------------------------- | ----- | ------- | -------------------------------------- |
| `adaptive`                   | bool  | True    | Enable adaptive curriculum             |
| `rebalance_interval_seconds` | int   | 3600    | Curriculum rebalance interval (1 hour) |
| `max_weight_multiplier`      | float | 1.5     | Maximum weight multiplier              |
| `min_weight_multiplier`      | float | 0.7     | Minimum weight multiplier              |
| `ema_alpha`                  | float | 0.3     | EMA smoothing factor                   |
| `min_games_for_weight`       | int   | 100     | Minimum games for weight calculation   |
| `rebalance_on_elo_change`    | bool  | True    | Rebalance on significant Elo changes   |
| `elo_change_threshold`       | int   | 50      | Elo change threshold for rebalance     |

---

## SafeguardsConfig

Process safeguards to prevent uncoordinated process sprawl.

| Option                          | Type | Default             | Description                      |
| ------------------------------- | ---- | ------------------- | -------------------------------- |
| `max_python_processes_per_host` | int  | 20                  | Max Python processes per host    |
| `max_selfplay_processes`        | int  | 2                   | Max selfplay processes           |
| `max_tournament_processes`      | int  | 1                   | Max tournament processes         |
| `max_training_processes`        | int  | 1                   | Max training processes           |
| `single_orchestrator`           | bool | True                | Enforce single orchestrator      |
| `orchestrator_host`             | str  | "nebius-backbone-1" | Designated orchestrator host     |
| `kill_orphans_on_start`         | bool | True                | Kill orphan processes on startup |
| `process_watchdog`              | bool | True                | Enable process watchdog          |
| `watchdog_interval_seconds`     | int  | 60                  | Watchdog check interval          |
| `max_process_age_hours`         | int  | 4                   | Maximum process age before kill  |
| `max_subprocess_depth`          | int  | 2                   | Maximum subprocess nesting depth |
| `subprocess_timeout_seconds`    | int  | 3600                | Subprocess timeout (1 hour)      |

---

## ClusterConfig

Cluster orchestration settings.

| Option                           | Type | Default | Description                           |
| -------------------------------- | ---- | ------- | ------------------------------------- |
| `target_selfplay_games_per_hour` | int  | 1000    | Target selfplay rate                  |
| `health_check_interval_seconds`  | int  | 60      | Health check interval                 |
| `sync_interval_seconds`          | int  | 300     | Data sync interval (5 min)            |
| `sync_interval`                  | int  | 6       | Sync every N iterations               |
| `model_sync_interval`            | int  | 12      | Model sync every N iterations         |
| `model_sync_enabled`             | bool | True    | Enable model sync                     |
| `elo_calibration_interval`       | int  | 72      | Elo calibration interval (iterations) |
| `elo_calibration_games`          | int  | 50      | Games per Elo calibration             |
| `elo_curriculum_enabled`         | bool | True    | Enable Elo-driven curriculum          |
| `elo_match_window`               | int  | 200     | Elo match window                      |
| `elo_underserved_threshold`      | int  | 100     | Underserved config threshold          |
| `auto_scale_interval`            | int  | 12      | Auto-scale check interval             |
| `underutilized_cpu_threshold`    | int  | 30      | CPU underutilization threshold        |
| `underutilized_python_jobs`      | int  | 10      | Min jobs for scaling                  |
| `scale_up_games_per_host`        | int  | 50      | Games per host for scale-up           |
| `adaptive_games_min`             | int  | 30      | Minimum adaptive games                |
| `adaptive_games_max`             | int  | 150     | Maximum adaptive games                |

---

## SSHConfig

SSH execution settings shared across orchestrators.

| Option                              | Type  | Default | Description                        |
| ----------------------------------- | ----- | ------- | ---------------------------------- |
| `max_retries`                       | int   | 3       | Maximum SSH retries                |
| `base_delay_seconds`                | float | 2.0     | Base retry delay                   |
| `max_delay_seconds`                 | float | 30.0    | Maximum retry delay                |
| `connect_timeout_seconds`           | int   | 10      | SSH connection timeout             |
| `command_timeout_seconds`           | int   | 3600    | Command execution timeout (1 hour) |
| `transport_command_timeout_seconds` | int   | 30      | P2P transport command timeout      |
| `retry_delay_seconds`               | float | 1.0     | Retry delay between attempts       |
| `address_cache_ttl_seconds`         | int   | 300     | Address cache TTL (5 min)          |

---

## SlurmConfig

Slurm execution settings for HPC clusters.

| Option                    | Type | Default           | Description                   |
| ------------------------- | ---- | ----------------- | ----------------------------- |
| `enabled`                 | bool | False             | Enable Slurm backend          |
| `partition_training`      | str  | "gpu-train"       | Training partition            |
| `partition_selfplay`      | str  | "gpu-selfplay"    | Selfplay partition            |
| `partition_tournament`    | str  | "cpu-eval"        | Tournament partition          |
| `account`                 | str  | None              | Slurm account                 |
| `qos`                     | str  | None              | Quality of service            |
| `default_time_training`   | str  | "08:00:00"        | Default training time limit   |
| `default_time_selfplay`   | str  | "02:00:00"        | Default selfplay time limit   |
| `default_time_tournament` | str  | "02:00:00"        | Default tournament time limit |
| `gpus_training`           | int  | 1                 | GPUs for training             |
| `cpus_training`           | int  | 16                | CPUs for training             |
| `mem_training`            | str  | "64G"             | Memory for training           |
| `gpus_selfplay`           | int  | 0                 | GPUs for selfplay             |
| `cpus_selfplay`           | int  | 8                 | CPUs for selfplay             |
| `mem_selfplay`            | str  | "16G"             | Memory for selfplay           |
| `job_dir`                 | str  | "data/slurm/jobs" | Slurm job directory           |
| `log_dir`                 | str  | "data/slurm/logs" | Slurm log directory           |
| `poll_interval_seconds`   | int  | 20                | Job status poll interval      |

---

## SafetyConfig

Safety thresholds to prevent bad models from being promoted.

| Option                     | Type  | Default | Description                |
| -------------------------- | ----- | ------- | -------------------------- |
| `overfit_threshold`        | float | 0.15    | Max train/val loss gap     |
| `min_memory_gb`            | int   | 64      | Minimum RAM required       |
| `max_consecutive_failures` | int   | 3       | Failures before stopping   |
| `parity_failure_rate_max`  | float | 0.10    | Max parity failure rate    |
| `data_quality_score_min`   | float | 0.70    | Minimum data quality score |

---

## PlateauDetectionConfig

Plateau detection and automatic hyperparameter search.

| Option                           | Type  | Default | Description                          |
| -------------------------------- | ----- | ------- | ------------------------------------ |
| `elo_plateau_threshold`          | float | 15.0    | Elo gain below this triggers plateau |
| `elo_plateau_lookback`           | int   | 5       | Evaluations to look back             |
| `win_rate_degradation_threshold` | float | 0.40    | Win rate degradation threshold       |
| `plateau_count_for_cmaes`        | int   | 2       | Plateaus before CMA-ES trigger       |
| `plateau_count_for_nas`          | int   | 4       | Plateaus before NAS trigger          |

---

## AlertingConfig

Alerting thresholds for monitoring.

| Option                   | Type | Default | Description                |
| ------------------------ | ---- | ------- | -------------------------- |
| `sync_failure_threshold` | int  | 5       | Sync failures before alert |
| `training_timeout_hours` | int  | 4       | Training timeout           |
| `elo_drop_threshold`     | int  | 50      | Elo drop alert threshold   |
| `games_per_hour_min`     | int  | 100     | Minimum games per hour     |

---

## ReplayBufferConfig

Prioritized experience replay buffer settings.

| Option                     | Type  | Default | Description                       |
| -------------------------- | ----- | ------- | --------------------------------- |
| `priority_alpha`           | float | 0.6     | Priority exponent                 |
| `importance_beta`          | float | 0.4     | Importance sampling exponent      |
| `capacity`                 | int   | 100000  | Maximum buffer capacity           |
| `rebuild_interval_seconds` | int   | 7200    | Buffer rebuild interval (2 hours) |

---

## DistributedConfig

Distributed system component settings.

| Option                       | Type | Default | Description                      |
| ---------------------------- | ---- | ------- | -------------------------------- |
| `degraded_failure_threshold` | int  | 2       | Failures before node is degraded |
| `offline_failure_threshold`  | int  | 5       | Failures before node is offline  |
| `recovery_success_threshold` | int  | 2       | Successes needed for recovery    |

---

## YAML Configuration File

The default configuration file is at `config/unified_loop.yaml`. Example structure:

```yaml
data_ingestion:
  poll_interval_seconds: 60
  sync_method: incremental

training:
  trigger_threshold_games: 500
  min_interval_seconds: 1200
  prefer_gpu_hosts: true

evaluation:
  shadow_games_per_config: 15
  full_tournament_games: 50

promotion:
  auto_promote: true
  elo_threshold: 25

cluster:
  target_selfplay_games_per_hour: 1000
  health_check_interval_seconds: 60
```

---

## Convenience Functions

The config module provides several convenience functions:

```python
from app.config.unified_config import (
    get_config,                  # Get singleton config
    get_training_threshold,      # Training game threshold
    get_elo_db_path,            # Elo database path
    get_min_elo_improvement,     # Promotion Elo threshold
    get_default_board_configs,   # All 9 board configurations
)

# Access nested config
config = get_config()
threshold = config.training.trigger_threshold_games

# Or use convenience functions
threshold = get_training_threshold()
```

---

## Master Loop Profiles

The unified automation entrypoint (`scripts/master_loop.py`) supports `--profile`
to control which daemons are started:

- `minimal`: sync + health (event router, health monitors, auto sync)
- `standard` (default): full automation (selfplay, training, evaluation, promotion)
- `full`: all non-deprecated daemons

Example:

```bash
python scripts/master_loop.py --profile minimal
```
