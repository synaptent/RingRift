# CLAUDE.md - AI Assistant Context for ai-service

This file provides context for AI assistants working on the Python AI service.
It complements AGENTS.md with operational knowledge.

## Project Overview

RingRift is a multiplayer territory control game with:

- **Frontend**: React/TypeScript game client
- **Backend**: Node.js game server
- **AI Service**: Python ML training pipeline + neural network AI opponents
- **Shared**: TypeScript game engine (source of truth for rules)

The Python `ai-service` mirrors the TS engine for training data generation and must maintain **parity** with it.

## Board Types & Configurations

| Board Type  | Grid              | Cells | Player Counts |
| ----------- | ----------------- | ----- | ------------- |
| `square8`   | 8x8               | 64    | 2, 3, 4       |
| `square19`  | 19x19             | 361   | 2, 3, 4       |
| `hex8`      | 9x9 (radius 4)    | 61    | 2, 3, 4       |
| `hexagonal` | 25x25 (radius 12) | 469   | 2, 3, 4       |

## Common Commands

### Training

```bash
# Export training data from database
python scripts/export_replay_dataset.py \
  --db data/games/my_games.db \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Use GameDiscovery to find all databases automatically
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Start training locally
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz \
  --model-version v2 \
  --batch-size 512 --epochs 20
```

### Automated Training Pipeline

One-command training loop that automatically chains: selfplay -> sync -> export -> train -> evaluate -> promote

```bash
# Basic usage - runs full pipeline
python scripts/run_training_loop.py \
  --board-type hex8 --num-players 2 \
  --selfplay-games 1000

# Full options with auto-promotion
python scripts/run_training_loop.py \
  --board-type hex8 --num-players 2 \
  --selfplay-games 1000 \
  --engine gumbel-mcts \
  --training-epochs 50 \
  --auto-promote

# Trigger pipeline on existing data (skip selfplay)
python scripts/run_training_loop.py \
  --board-type hex8 --num-players 2 \
  --skip-selfplay
```

### Master Loop Controller (Recommended for Automation)

The unified automation entry point that orchestrates all background daemons and training loop:

```bash
# Full automation mode (recommended for long-term cluster utilization)
python scripts/master_loop.py

# Watch mode (show status, don't run loop)
python scripts/master_loop.py --watch

# Specific configs only
python scripts/master_loop.py --configs hex8_2p,square8_2p

# Dry run (preview actions without executing)
python scripts/master_loop.py --dry-run

# Skip daemons (for testing)
python scripts/master_loop.py --skip-daemons
```

**What it orchestrates:**

- `SelfplayScheduler` - Priority-based selfplay allocation using curriculum weights, Elo velocities
- `DaemonManager` - Lifecycle for all background daemons (66 types)
- `ClusterMonitor` - Real-time cluster health
- `FeedbackLoopController` - Training feedback signals
- `DataPipelineOrchestrator` - Pipeline stage tracking
- `UnifiedQueuePopulator` - Work queue maintenance until Elo targets met

### Transfer Learning (2p to 4p)

```bash
# Step 1: Resize value head from 2 outputs to 4 outputs
python scripts/transfer_2p_to_4p.py \
  --source models/my_2p_model.pth \
  --output models/my_4p_init.pth \
  --board-type square8

# Step 2: Train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/my_4p_init.pth \
  --data-path data/training/sq8_4p.npz

# Direct transfer (partial loading, value head randomly initialized)
python -m app.training.train \
  --board-type hex8 --num-players 4 \
  --init-weights models/my_hex8_2p.pth \
  --data-path data/training/hex8_4p.npz
```

### Data Discovery & Quality

```bash
# Find all game databases
python -c "
from app.utils.game_discovery import GameDiscovery
d = GameDiscovery()
for db in d.find_all_databases():
    print(f'{db.board_type}_{db.num_players}p: {db.game_count} games - {db.path}')
"

# Check database quality
python -m app.training.data_quality --db data/games/selfplay.db

# Validate NPZ training data
python -m app.training.data_quality --npz data/training/hex8_2p.npz --detailed

# Validate all databases
python scripts/validate_databases.py data/games --check-structure
```

### Parity Testing

```bash
# Check TS/Python parity for a database
python scripts/check_ts_python_replay_parity.py --db data/games/my_games.db

# Run canonical selfplay parity gate
python scripts/run_canonical_selfplay_parity_gate.py --board-type hex8
```

### Model Evaluation

```bash
# Gauntlet evaluation (model vs baselines)
python scripts/quick_gauntlet.py \
  --model models/my_model.pth \
  --board-type hex8 --num-players 2
```

## Key Utilities

### GameDiscovery (`app/utils/game_discovery.py`)

Unified utility for finding game databases across all storage patterns:

- `find_all_databases()` - Find all .db files with game data
- `find_databases_for_config(board_type, num_players)` - Filter by config
- `RemoteGameDiscovery` - SSH-based cluster-wide discovery

### DataQuality (`app/training/data_quality.py`)

Quality checking for training data:

- `DatabaseQualityChecker` - Validate database schema/content
- `TrainingDataValidator` - Validate NPZ files

### Move Data Integrity (Dec 2025)

**Schema v14** enforces that all games have move data before completion. Games without moves are useless for training.

**Key Components:**

| Component                   | Location                                     | Purpose                                                              |
| --------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| `MIN_MOVES_REQUIRED`        | `app/db/game_replay.py`                      | Constant = 1, enforced in `store_game()` and `GameWriter.finalize()` |
| `InvalidGameError`          | `app/errors.py`                              | Raised when game has insufficient move data                          |
| `enforce_moves_on_complete` | SQLite trigger                               | Prevents `game_status='completed'` if `total_moves=0`                |
| `orphaned_games` table      | Schema v14                                   | Quarantine table for games detected without moves                    |
| `IntegrityCheckDaemon`      | `app/coordination/integrity_check_daemon.py` | Hourly scan for orphan games                                         |

**Configuration (environment variables):**

```bash
# IntegrityCheckDaemon settings
RINGRIFT_INTEGRITY_ENABLED=true              # Enable daemon (default: true)
RINGRIFT_INTEGRITY_CHECK_INTERVAL=3600       # Scan interval in seconds (default: 1 hour)
RINGRIFT_INTEGRITY_QUARANTINE_DAYS=7         # Days before cleanup (default: 7)
RINGRIFT_INTEGRITY_MAX_ORPHANS=1000          # Max orphans per scan (default: 1000)
RINGRIFT_INTEGRITY_DATA_DIR=data/games       # Data directory to scan
```

**Cleanup Script:**

```bash
# Preview orphan games (dry run)
python scripts/cleanup_games_without_moves.py --dry-run

# Clean all canonical databases
python scripts/cleanup_games_without_moves.py --execute

# Clean specific database
python scripts/cleanup_games_without_moves.py --db data/games/canonical_hex8.db --execute
```

**Consolidation Validation:**

The `consolidate_jsonl_databases.py` script now validates source databases:

```bash
# Strict mode - fail on any integrity issues
python scripts/consolidate_jsonl_databases.py --strict

# Normal mode - warn and skip invalid games
python scripts/consolidate_jsonl_databases.py
```

**Programmatic Usage:**

```python
from app.coordination.integrity_check_daemon import get_integrity_check_daemon

# Get singleton daemon
daemon = get_integrity_check_daemon()
await daemon.start()

# Check health
health = daemon.health_check()
print(f"Orphans found: {health.details['total_orphans_found']}")
```

### Safe Checkpoint Loading (`app/utils/torch_utils.py`)

Secure model checkpoint loading to mitigate pickle deserialization attacks:

```python
from app.utils.torch_utils import safe_load_checkpoint

# Safe by default - tries weights_only=True first
checkpoint = safe_load_checkpoint("models/my_model.pth")

# For untrusted external models - enforce safe mode
checkpoint = safe_load_checkpoint(external_path, allow_unsafe=False)
```

See `SECURITY.md` for full details.

### Environment Configuration (`app/config/env.py`)

Centralized typed environment variable configuration:

```python
from app.config.env import env

# Get values with proper types and defaults
node_id = env.node_id
log_level = env.log_level
is_coordinator = env.is_coordinator

# Check feature flags
if env.skip_shadow_contracts:
    # Skip validation

# Resource management
util_min = env.target_util_min
util_max = env.target_util_max

# PID controller
kp, ki, kd = env.pid_kp, env.pid_ki, env.pid_kd
```

All RINGRIFT\_\* environment variables accessible via typed cached properties.

### Cluster Configuration (`app/config/cluster_config.py`)

Consolidated helpers for distributed_hosts.yaml access:

```python
from app.config.cluster_config import (
    load_cluster_config,
    get_sync_routing,
    get_auto_sync_config,
    get_host_bandwidth_limit,
    get_host_provider,
    filter_hosts_by_status,
    # December 2025 additions
    get_cluster_nodes,
    get_ready_nodes,
    get_gpu_nodes,
    get_node_bandwidth_kbs,
    get_gpu_types,
    ClusterNode,
)

# Get sync routing config
sync_cfg = get_sync_routing()
max_disk = sync_cfg.max_disk_usage_percent  # 70.0
priority_hosts = sync_cfg.priority_hosts    # ["runpod-a100-1", ...]

# Get bandwidth limit for a host (uses glob patterns)
limit = get_host_bandwidth_limit("vast-12345")  # Returns 50 MB/s

# Get provider from host name
provider = get_host_provider("nebius-h100")  # Returns "nebius"

# Filter hosts by status
ready = filter_hosts_by_status(["ready"])

# December 2025: Node-based helpers
nodes = get_cluster_nodes()  # Dict[str, ClusterNode]
ready_nodes = get_ready_nodes()  # List[ClusterNode] with status='ready'
gpu_types = get_gpu_types()  # Dict[str, int] mapping GPU name to VRAM

# Provider-based bandwidth (in KB/s)
bw = get_node_bandwidth_kbs("vast-12345")  # Uses provider defaults
```

Dataclasses: `SyncRoutingConfig`, `AutoSyncConfig`, `EloSyncConfig`, `ClusterConfig`, `ClusterNode`

**ClusterNode fields** (December 2025):

- `name`, `tailscale_ip`, `ssh_host`, `ssh_port`, `ssh_user`, `ssh_key`
- `status`, `role`, `gpu`, `gpu_vram_gb`, `bandwidth_mbps`
- `is_coordinator`, `data_server_port`, `data_server_url`
- Properties: `best_ip`, `provider`, `is_active`, `is_gpu_node`

Modules consolidated (Dec 2025): `sync_bandwidth.py`, `utilization_optimizer.py`, `database_sync_manager.py`

### Unified SSH (`app/core/ssh.py`)

Canonical SSH utility for all cluster operations:

```python
from app.core.ssh import (
    SSHClient,
    SSHConfig,
    SSHResult,
    get_ssh_client,
    run_ssh_command_async,
)

# Get cached client for a cluster node
client = get_ssh_client("runpod-h100")
result = await client.run_async("nvidia-smi")

# Convenience function
result = await run_ssh_command_async("runpod-h100", "echo hello")

# Sync usage
result = client.run("nvidia-smi", timeout=30)
```

Features: Connection pooling via ControlMaster, multi-transport fallback (Tailscale → Direct), automatic retry.

### Unified NodeInfo (`app/core/node.py`)

Canonical node information dataclass for cluster management:

```python
from app.core.node import NodeInfo, NodeRole, NodeState

# From P2P status
node = NodeInfo.from_p2p_status(p2p_response)

# From SSH discovery
node = NodeInfo.from_ssh_discovery(host, ssh_result)

# Check node properties
if node.is_healthy and node.is_gpu_node:
    score = node.gpu_power_score
    endpoint = node.endpoint
```

Unified structure for: GPUInfo, ResourceMetrics, ConnectionInfo, HealthStatus, ProviderInfo, JobStatus.

### GumbelCommon (`app/ai/gumbel_common.py`)

Unified data structures for all Gumbel MCTS variants:

- `GumbelAction` - Action with policy logit, Gumbel noise, value tracking
- `GumbelNode` - Node for MCTS tree with visit counts and values
- `LeafEvalRequest` - Batched leaf evaluation request
- Budget constants: `GUMBEL_BUDGET_THROUGHPUT` (64), `GUMBEL_BUDGET_STANDARD` (150), `GUMBEL_BUDGET_QUALITY` (800), `GUMBEL_BUDGET_ULTIMATE` (1600)
- `get_budget_for_difficulty(difficulty)` - Map difficulty to budget tier

### GumbelSearchEngine (`app/ai/gumbel_search_engine.py`)

Unified entry point for all Gumbel MCTS search variants:

```python
from app.ai.gumbel_search_engine import GumbelSearchEngine, SearchMode

# For single game play
engine = GumbelSearchEngine(neural_net=my_nn, mode=SearchMode.SINGLE_GAME)
move = engine.search(game_state)

# For selfplay (high throughput)
engine = GumbelSearchEngine(neural_net=my_nn, mode=SearchMode.MULTI_GAME_PARALLEL, num_games=64)
results = engine.search_batch(initial_states)
```

- Modes: `SINGLE_GAME`, `SINGLE_GAME_FAST`, `MULTI_GAME_BATCH`, `MULTI_GAME_PARALLEL`, `AUTO`
- Consolidates: `gumbel_mcts_ai.py`, `tensor_gumbel_tree.py`, `batched_gumbel_mcts.py`, `multi_game_gumbel.py`

### Selfplay (`scripts/selfplay.py`)

Unified CLI entry point for all selfplay:

```bash
# Quick heuristic selfplay (fast bootstrap)
python scripts/selfplay.py --board square8 --num-players 2 --engine heuristic

# GPU Gumbel MCTS (high quality training data)
python scripts/selfplay.py --board hex8 --num-players 2 --engine gumbel --num-games 500

# Full options
python scripts/selfplay.py \
  --board square8 --num-players 4 \
  --num-games 1000 --engine nnue-guided \
  --output-dir data/games/selfplay_sq8_4p
```

Engine modes: `heuristic`, `gumbel`, `mcts`, `nnue-guided`, `policy-only`, `nn-descent`, `mixed`

### SelfplayRunner (`app/training/selfplay_runner.py`)

Base class for programmatic selfplay:

```python
from app.training.selfplay_runner import run_selfplay

# Quick usage
stats = run_selfplay(board_type="hex8", num_players=2, num_games=100, engine="heuristic")
```

- `SelfplayRunner` - Base class with config, model loading, event emission
- `HeuristicSelfplayRunner` - Fast heuristic-only selfplay
- `GumbelMCTSSelfplayRunner` - Quality Gumbel MCTS selfplay

### Coordination Infrastructure (`app/coordination/`)

Unified training pipeline orchestration:

**Core Orchestration:**

- **`selfplay_scheduler.py`**: Priority-based selfplay allocation (curriculum, Elo velocity, feedback signals)
- **`unified_queue_populator.py`**: Maintains work queue until Elo targets met (60% selfplay, 30% training, 10% tournament)
- **`idle_resource_daemon.py`**: Spawns selfplay on idle GPUs using SelfplayScheduler priorities
- **`utilization_optimizer.py`**: Matches GPU capabilities to board sizes, optimizes cluster utilization
- **`feedback_loop_controller.py`**: Manages training feedback signals and quality thresholds
- **`sync_facade.py`**: Unified programmatic entry point for sync (routes to AutoSyncDaemon, SyncRouter, etc.)

**Event System:**

- **`event_router.py`**: Unified event bus with content-based deduplication (SHA256)
- **`pipeline_actions.py`**: Stage invokers with circuit breaker protection
- **`data_pipeline_orchestrator.py`**: Tracks and triggers pipeline stages

**Daemon Management:**

The `DaemonManager` coordinates 60+ background services. See `docs/DAEMON_REGISTRY.md` for full reference.

| Category  | Key Daemon Types                                             | Purpose                      |
| --------- | ------------------------------------------------------------ | ---------------------------- |
| Core      | `EVENT_ROUTER`, `DAEMON_WATCHDOG`                            | Event bus, health monitoring |
| Sync      | `AUTO_SYNC`, `MODEL_DISTRIBUTION`, `ELO_SYNC`                | Data/model synchronization   |
| Training  | `DATA_PIPELINE`, `SELFPLAY_COORDINATOR`, `TRAINING_ACTIVITY` | Pipeline orchestration       |
| Eval      | `EVALUATION`, `AUTO_PROMOTION`                               | Model evaluation/promotion   |
| Health    | `NODE_HEALTH_MONITOR`, `QUALITY_MONITOR`                     | Cluster health               |
| Resources | `IDLE_RESOURCE`, `NODE_RECOVERY`                             | GPU utilization, recovery    |

**Key files:**

- **`daemon_manager.py`**: Lifecycle management, health checks, auto-restart (~2,000 LOC)
- **`daemon_runners.py`**: Async runner functions for all daemon types (~1,100 LOC, Dec 2025 extraction)
- **`daemon_registry.py`**: Declarative daemon specifications (~150 LOC, Dec 2025)
- **`daemon_types.py`**: `DaemonType` enum with all 66 daemon types
- **`sync_bandwidth.py`**: Bandwidth-coordinated rsync with host-level limits
- **`auto_sync_daemon.py`**: Automated P2P data sync with push-from-generator + gossip replication
- **`training_activity_daemon.py`**: Detects training activity, triggers priority sync (Dec 2025)

**Architecture (December 2025):**

The daemon system uses a three-layer architecture:

1. **`daemon_registry.py`** - Declarative configuration (NEW Dec 2025)
   - `DAEMON_REGISTRY`: Dict[DaemonType, DaemonSpec] with all 66 daemon configurations
   - `DaemonSpec` dataclass: runner_name, depends_on, category, auto_restart, health_check_interval
   - `get_daemons_by_category()`, `get_categories()`, `validate_registry()`
   - Replaces ~330 lines of imperative code with ~30 lines of declarations

2. **`daemon_manager.py`** - Lifecycle coordinator
   - `DaemonManager` class handles start/stop, health monitoring, auto-restart
   - `_register_default_factories()` maps DaemonType to runner functions
   - Only `_create_health_server()` remains inline (needs `self.liveness_probe()`)

3. **`daemon_runners.py`** - Runner implementations
   - 62 async runner functions (`create_auto_sync()`, `create_data_pipeline()`, etc.)
   - Each handles: import, instantiation, start, wait-for-completion
   - `get_runner(DaemonType)` - Retrieve runner by type
   - `get_all_runners()` - Get full registry

```python
# Example: Query daemon registry
from app.coordination.daemon_registry import (
    DAEMON_REGISTRY, DaemonSpec, get_daemons_by_category, validate_registry
)
from app.coordination.daemon_types import DaemonType

# Get spec for a specific daemon
spec = DAEMON_REGISTRY[DaemonType.AUTO_SYNC]
print(f"Runner: {spec.runner_name}, Category: {spec.category}")
print(f"Depends on: {[d.name for d in spec.depends_on]}")

# Get all sync-related daemons
sync_daemons = get_daemons_by_category("sync")

# Validate registry at startup
errors = validate_registry()
if errors:
    raise RuntimeError(f"Registry errors: {errors}")
```

```python
# Example: Get and invoke a runner
from app.coordination.daemon_runners import get_runner
from app.coordination.daemon_types import DaemonType

runner = get_runner(DaemonType.AUTO_SYNC)
await runner()  # Starts AutoSyncDaemon
```

**Launching daemons:**

```bash
# Full automation (recommended)
python scripts/master_loop.py

# Specific daemons
python scripts/launch_daemons.py --all
python scripts/launch_daemons.py --status
```

**Data Distribution (Dec 2025):**

- **`scripts/dynamic_data_distribution.py`**: HTTP-based distribution daemon for training data
  - Distributes NPZ/DB files from OWC (mac-studio:8780) to training nodes
  - Rsync fallback for large files (>100MB), aria2/curl fallback for downloads
  - Capacity-aware (skips nodes with <50GB free)
- **`scripts/dynamic_space_manager.py`**: Proactive disk space management
  - Cleanup at 60% usage (before 70% threshold)
  - Deletes old logs, empty databases, old checkpoints
- **`scripts/consolidate_jsonl_databases.py`**: Merge scattered jsonl_aggregated.db files
  - Consolidates 44 source databases (147GB) into per-config canonical databases
  - Deduplicates by game_id
- **`scripts/orchestrated_data_sync.py`**: Config-aware data sync
  - Detects pending training, syncs relevant data only

**Batch Rsync (Dec 2025):**

- **`app/coordination/sync_bandwidth.py`**: Now includes `BatchRsync` class
  - `sync_files()` - Transfer multiple files in single rsync operation
  - `sync_directory()` - Directory sync with include/exclude patterns
  - Bandwidth-coordinated with `BandwidthManager`

**Sync CLI (supported):**

- Use `scripts/unified_data_sync.py` for operational sync CLI usage.
- Avoid importing `app/distributed/unified_data_sync.py` directly (deprecated); use `SyncFacade` or `AutoSyncDaemon` in code.

```bash
# Recommended: Use master_loop.py for full automation
python scripts/master_loop.py

# Or launch specific daemons
python scripts/launch_daemons.py --all

# Check daemon status
python scripts/launch_daemons.py --status
```

### Temperature Scheduling (`app/training/temperature_scheduling.py`)

Exploration/exploitation control during selfplay:

```python
from app.training.temperature_scheduling import create_scheduler

# Presets: default, alphazero, aggressive_exploration, conservative, adaptive, curriculum, cosine
scheduler = create_scheduler("adaptive")
temp = scheduler.get_temperature(move_number=15, game_state=state)
```

- 7 schedule types including adaptive (based on position complexity) and curriculum (based on training progress)
- `AlphaZeroTemperature` for t=1 -> t=0 at move N
- `DirichletNoiseTemperature` for root exploration noise

### Online Learning (`app/training/online_learning.py`)

Continuous learning during gameplay:

```python
from app.training.online_learning import create_online_learner, get_online_learning_config

config = get_online_learning_config("tournament")  # Profiles: default, conservative, aggressive, tournament
learner = create_online_learner(model, learner_type="ebmo", config=config)

# During game
learner.record_transition(state, move, player, next_state)
# After game
learner.update_from_game(winner)
```

## Architecture Notes

### Neural Network (v2)

- 96 channels, 6 residual blocks with SE attention
- Separate policy and value heads
- Policy: position-aware encoding with board geometry
- Value: per-player win probability (softmax for multiplayer)

### Training Pipeline

1. Self-play generates games -> SQLite databases
2. `export_replay_dataset.py` converts to NPZ (features, policy, value)
3. `app.training.train` trains with early stopping
4. Gauntlet evaluation against baselines

### Data Flow

```
Self-play (Python/TS) -> GameReplayDB (.db)
                              |
              export_replay_dataset.py
                              |
                    Training NPZ files
                              |
                   app.training.train
                              |
                    Model checkpoints
```

## Deprecated Components & Migration

The following components are deprecated and will be removed in future releases:

### Daemons (Removal: Q2 2026)

| Deprecated                    | Replacement                                         | Notes                     |
| ----------------------------- | --------------------------------------------------- | ------------------------- |
| `DaemonType.SYNC_COORDINATOR` | `DaemonType.AUTO_SYNC`                              | Use AutoSyncDaemon        |
| `DaemonType.HEALTH_CHECK`     | `DaemonType.NODE_HEALTH_MONITOR`                    | Unified health monitoring |
| `vast_idle_daemon.py`         | `unified_idle_shutdown_daemon.py`                   | Provider-agnostic         |
| `lambda_idle_daemon.py`       | `unified_idle_shutdown_daemon.py`                   | Provider-agnostic         |
| `auto_evaluation_daemon.py`   | `evaluation_daemon.py` + `auto_promotion_daemon.py` | Monolithic → composable   |
| `queue_populator_daemon.py`   | `unified_queue_populator.py`                        | Archived Dec 2025         |
| `node_health_monitor.py`      | `health_check_orchestrator.py`                      | Safe to archive           |

### Training Modules (Deprecated Dec 2025)

| Deprecated                                     | Replacement               | Notes                          |
| ---------------------------------------------- | ------------------------- | ------------------------------ |
| `orchestrated_training.py`                     | `unified_orchestrator.py` | Unified training orchestration |
| `integrated_enhancements.py`                   | `unified_orchestrator.py` | Consolidated enhancements      |
| `DataQualityScorer` (in training_enhancements) | `UnifiedQualityScorer`    | See `unified_quality.py`       |

### Training Entry Points

Three main options - use the recommended one:

| Script                 | Purpose                                 | Recommended?       |
| ---------------------- | --------------------------------------- | ------------------ |
| `master_loop.py`       | Full cluster automation with 66 daemons | ✅ Yes             |
| `run_training_loop.py` | Simple 1-config pipeline                | For single configs |
| `unified_ai_loop.py`   | Legacy wrapper                          | ❌ Deprecated      |

### December 2025 Consolidation Summary

Major consolidation effort completed December 2025:

| Original Modules                                                                         | Consolidated To                          | LOC Saved       | Status   |
| ---------------------------------------------------------------------------------------- | ---------------------------------------- | --------------- | -------- |
| `model_distribution_daemon.py` + `npz_distribution_daemon.py`                            | `unified_distribution_daemon.py`         | ~1,100          | Complete |
| `lambda_idle_daemon.py` + `vast_idle_daemon.py`                                          | `unified_idle_shutdown_daemon.py`        | ~318            | Complete |
| `replication_monitor.py` + `replication_repair_daemon.py`                                | `unified_replication_daemon.py`          | ~600            | Complete |
| `system_health_monitor.py` (scoring)                                                     | `unified_health_manager.py`              | ~200            | Complete |
| 3× GumbelAction/GumbelNode copies                                                        | `gumbel_common.py`                       | ~150            | Complete |
| `distributed/cluster_monitor.py`                                                         | `coordination/cluster_status_monitor.py` | ~40 (shim)      | Complete |
| `EloSyncManager` + `RegistrySyncManager`                                                 | `DatabaseSyncManager` base class         | ~567            | Complete |
| 28 `_init_*()` functions in `coordination_bootstrap.py`                                  | `COORDINATOR_REGISTRY` + generic handler | ~17             | Complete |
| 5× NodeStatus definitions                                                                | `node_status.py`                         | ~200            | Complete |
| DaemonManager factory methods (65 of 66)                                                 | `daemon_runners.py`                      | ~1,580          | Complete |
| `tracing.py` + `distributed_lock.py` + `optional_imports.py` + `yaml_utils.py`           | `core_utils.py`                          | ~0 (re-exports) | Complete |
| `coordinator_base.py` + `coordinator_dependencies.py`                                    | `core_base.py`                           | ~0 (re-exports) | Complete |
| `event_router.py` + `event_mappings.py` + `event_emitters.py` + `event_normalization.py` | `core_events.py`                         | ~0 (re-exports) | Complete |

**New Canonical Modules (December 2025 Wave 2):**

| Module               | Purpose                                                       |
| -------------------- | ------------------------------------------------------------- |
| `node_status.py`     | Unified NodeHealthState enum + NodeMonitoringStatus dataclass |
| `daemon_runners.py`  | 66 daemon runner functions extracted from DaemonManager       |
| `daemon_registry.py` | Declarative DaemonSpec registry for all 66 daemon types       |

**`node_status.py`** consolidates 5 duplicate NodeStatus definitions:

- `NodeHealthState` enum: HEALTHY, DEGRADED, UNHEALTHY, EVICTED, UNKNOWN, OFFLINE
- `NodeMonitoringStatus` dataclass: 20+ fields for node identity, health, GPU, workload
- Backward-compatible aliases: `NodeStatus`, `ClusterNodeStatus`
- Migration: `from app.coordination.node_status import NodeMonitoringStatus as NodeStatus`

**`daemon_runners.py`** extracts runner functions from DaemonManager:

- 62 async runner functions: `create_auto_sync()`, `create_data_pipeline()`, etc.
- `get_runner(DaemonType)` - Retrieve runner function by daemon type
- `get_all_runners()` - Get full registry of all runners
- Only `_create_health_server()` remains in daemon_manager.py (needs `self` access)
- Benefits: Reduced daemon_manager.py from ~3,600 to ~2,000 LOC, runners testable in isolation

**`daemon_registry.py`** provides declarative daemon configuration (Dec 27, 2025):

- `DAEMON_REGISTRY`: Dict[DaemonType, DaemonSpec] with all 66 daemon configurations
- `DaemonSpec` dataclass with frozen=True for immutability:
  - `runner_name`: Function name in daemon_runners.py (e.g., "create_auto_sync")
  - `depends_on`: Tuple of DaemonTypes that must start first
  - `category`: Grouping for management ("sync", "event", "health", "pipeline", "resource", "misc")
  - `auto_restart`: Whether to restart on failure (default: True)
  - `max_restarts`: Max restart attempts (default: 5)
  - `health_check_interval`: Seconds between health checks (default: 30.0)
- Helper functions: `get_daemons_by_category()`, `get_categories()`, `validate_registry()`
- Unit tests: 42 tests in `tests/unit/coordination/test_daemon_registry.py`
- Benefits: Data-driven configuration, dependency graph validation, testable in isolation

**New Canonical Modules (Phase 5 - 157→15 Consolidation):**

| Module           | Exports | Purpose                                                     |
| ---------------- | ------- | ----------------------------------------------------------- |
| `core_utils.py`  | 57      | Tracing, locking, optional imports, YAML utilities          |
| `core_base.py`   | 23      | Coordinator base classes, protocols, registry, dependencies |
| `core_events.py` | 127     | Event router, mappings, emitters, normalization             |

**`core_utils.py`** consolidates utility modules:

```python
from app.coordination.core_utils import (
    # Tracing
    TraceContext, new_trace, span, traced, get_trace_id, set_trace_id,
    # Locking
    DistributedLock, training_lock, acquire_training_lock,
    # Optional imports
    TORCH_AVAILABLE, CUDA_AVAILABLE, get_module, require_module,
    # YAML
    load_yaml, safe_load_yaml, dump_yaml,
)
```

**`core_base.py`** consolidates coordinator infrastructure:

```python
from app.coordination.core_base import (
    # Base classes
    CoordinatorBase, CoordinatorStats, HealthCheckResult,
    # Protocols and enums
    CoordinatorProtocol, CoordinatorStatus,
    # Mixins
    SQLitePersistenceMixin, SingletonMixin, CallbackMixin, EventDrivenMonitorMixin,
    # Registry
    CoordinatorRegistry, get_coordinator_registry, get_all_coordinators, shutdown_all_coordinators,
    # Dependencies
    CoordinatorDependencyGraph, validate_dependencies, get_initialization_order,
)
```

**`core_events.py`** consolidates event system:

```python
from app.coordination.core_events import (
    # Router core
    UnifiedEventRouter, get_router, publish, subscribe, unsubscribe,
    # Event types
    DataEventType, DataEvent, StageEvent, EventBus,
    # Bus access
    get_event_bus, get_stage_event_bus, get_cross_process_queue,
    # Mappings
    STAGE_TO_DATA_EVENT_MAP, DATA_TO_CROSS_PROCESS_MAP,
    # Typed emitters (70+)
    emit_training_complete, emit_selfplay_complete, emit_sync_complete,
    # Normalization
    normalize_event_type, CANONICAL_EVENT_NAMES, is_canonical,
)
```

Old imports still work for backward compatibility but will emit DeprecationWarning in Q1 2026.

**Infrastructure Improvements (December 2025 Wave 2):**

| Improvement                            | Description                                                 |
| -------------------------------------- | ----------------------------------------------------------- |
| DaemonAdapter.health_check()           | Added to base class; all 9 adapter subclasses now comply    |
| Hardcoded port 8770 → P2P_DEFAULT_PORT | node_status.py, unified_node_health_daemon.py fixed         |
| health_check() added to 6 classes      | ClusterMonitor, BandwidthManager, BackpressureMonitor, etc. |
| BaseDaemon deprecation                 | protocols.py BaseDaemon → use base_daemon.py version        |
| CoordinatorStatus/Protocol             | coordinator_base.py now imports from protocols.py           |

**Coordination Bootstrap Refactoring (December 2025):**

Replaced 28 individual `_init_*()` functions with registry-based pattern:

- `COORDINATOR_REGISTRY`: Dict[str, CoordinatorSpec] with 27 coordinator entries
- `_init_coordinator_from_spec()`: Generic initialization handler
- `InitPattern` enum: WIRE, GET, IMPORT, SKIP, DELEGATE patterns
- Benefits: Adding new coordinators now requires 6-8 lines (vs 20-25 for full function)

Special handlers retained for `_init_pipeline_orchestrator` (extra args) and `_init_curriculum_integration_with_verification` (verification logic).

**Event System Improvements (December 2025):**

- `SelfplayScheduler` now wired into coordination bootstrap for feedback events
- P2P orchestrator logs event subscription status for debugging
- Critical exception handlers narrowed from `except Exception:` to specific types (12 handlers fixed)

**Remaining Consolidation Opportunities (Q1 2026):**

| Opportunity                        | Current LOC | Savings | Priority |
| ---------------------------------- | ----------- | ------- | -------- |
| Archive unified_cluster_monitor.py | 951         | ~951    | MEDIUM   |
| Archive node_health_monitor.py     | 386         | ~350    | LOW      |

**Completed Consolidation (December 2025):**

| Completed                             | Original LOC | New LOC         | Savings                  |
| ------------------------------------- | ------------ | --------------- | ------------------------ |
| daemon_manager.py → daemon_runners.py | ~3,600       | ~2,000 + ~1,100 | ~1,580 (deprecated code) |
| HealthCheckMixin extraction           | ~600 (dupl.) | 214             | ~400 (76+ files can use) |

**Bug Fixes (December 2025):**

- Fixed `adaptive_resource_manager.py:414` - was importing non-existent `get_cluster_monitor`
- Wired `emit_task_abandoned` to job cancellation paths in p2p_orchestrator.py

**Enum Collision Fixes (December 27, 2025):**

Fixed CRITICAL naming collisions where multiple enums had the same name but different semantics:

| Original Name    | New Name (module)                               | Purpose                          |
| ---------------- | ----------------------------------------------- | -------------------------------- |
| `NodeRole`       | `LeadershipRole` (leadership_coordinator)       | Raft roles: LEADER, FOLLOWER     |
| `NodeRole`       | `ClusterNodeRole` (multi_provider_orchestrator) | Job roles: TRAINING, SELFPLAY    |
| `RecoveryAction` | `JobRecoveryAction` (unified_health_manager)    | Job-level: RESTART_JOB, KILL_JOB |
| `RecoveryAction` | `SystemRecoveryAction` (recovery_orchestrator)  | System: RESTART_P2P, REBOOT      |
| `RecoveryAction` | `NodeRecoveryAction` (node_recovery_daemon)     | Node: RESTART, FAILOVER          |

Backward-compat aliases retained (deprecated). Use `app/coordination/enums.py` for centralized access:

```python
from app.coordination.enums import (
    LeadershipRole, ClusterNodeRole,
    JobRecoveryAction, SystemRecoveryAction, NodeRecoveryAction,
)
```

**P2P Manager Delegation (December 27, 2025):**

All 7 P2P managers fully delegated (100% coverage):

| Manager               | Status      | LOC Removed | Notes                                   |
| --------------------- | ----------- | ----------- | --------------------------------------- |
| `StateManager`        | ✅ Complete | ~200        | SQLite persistence, epochs              |
| `TrainingCoordinator` | ✅ Complete | ~450        | Job dispatch, model promotion           |
| `JobManager`          | ✅ Complete | ~400        | Selfplay, training, tournaments         |
| `SyncPlanner`         | ✅ Complete | ~60         | Manifest collection, sync planning      |
| `SelfplayScheduler`   | ✅ Complete | ~430        | Priority scheduling, curriculum weights |
| `NodeSelector`        | ✅ Complete | ~50         | Node ranking, job placement             |
| `LoopManager`         | ✅ Complete | ~400        | Background loops (5 migrated)           |

Total: ~1,990 LOC removed from p2p_orchestrator.py (27,889 → 25,899 lines)

See `scripts/P2P_ORCHESTRATOR_REMOVED_CODE.md` for detailed registry of all 22 removed methods (~1,255 LOC).

**Circular Dependency Fixes (December 27, 2025):**

- `selfplay_scheduler.py:84` - backpressure import converted to lazy loading
- `resource_optimizer.py:70` - resource_targets import uses lazy accessor pattern
- Breaks 8-cycle chain, reduces startup module load by ~4,000 LOC

**Phase 8 Improvements (December 27, 2025):**

Daemon health check coverage and code quality improvements:

| Task                              | Description                                             | Status              |
| --------------------------------- | ------------------------------------------------------- | ------------------- |
| Quality→Training wiring           | `ImprovementOptimizer` already integrated               | ✅ Already complete |
| SELFPLAY_COORDINATOR health_check | Added to `selfplay_scheduler.py`                        | ✅ Complete         |
| FEEDBACK_LOOP health_check        | Added to `feedback_loop_controller.py`                  | ✅ Complete         |
| TRAINING_TRIGGER health_check     | Added to `training_trigger_daemon.py`                   | ✅ Complete         |
| facade.py exception handling      | Specific exception types (ValueError, OSError, etc.)    | ✅ Complete         |
| Startup validation                | `_validate_critical_subsystems()` in daemon_manager.py  | ✅ Complete         |
| CLUSTER_WATCHDOG health_check     | Added to `cluster_watchdog_daemon.py`                   | ✅ Complete         |
| NODE_RECOVERY health_check        | Added to `node_recovery_daemon.py`                      | ✅ Complete         |
| EVALUATION get_status()           | Added to `evaluation_daemon.py`                         | ✅ Complete         |
| Hardcoded port 8770 fix           | `p2p_integration.py` now uses `get_p2p_port()`          | ✅ Complete         |
| DB connection leak fix            | `auto_sync_daemon.py` uses context managers             | ✅ Complete         |
| SyncRouter NODE_RECOVERED         | Added event subscription for node recovery              | ✅ Complete         |
| FeedbackLoopController fix        | `_subscribed` flag now reset in `finally` block         | ✅ Complete         |
| P2P_BACKEND health_check          | Added `P2PIntegration` class to `p2p_integration.py`    | ✅ Complete         |
| Config template archival          | Archived 13 example/template files to deprecated_config | ✅ Complete         |
| CoordinatorBase health_check      | Added to `coordinator_base.py` (all subclasses inherit) | ✅ Complete         |
| DaemonManager health_check        | Added to `daemon_manager.py` (lines 1477-1522)          | ✅ Complete         |
| HealthCheckOrchestrator health    | Added to `health_check_orchestrator.py` (581-626)       | ✅ Complete         |
| DataPipeline NEW_GAMES handler    | Subscribes to NEW_GAMES_AVAILABLE event                 | ✅ Complete         |
| DataPipeline REGRESSION handler   | Subscribes to REGRESSION_DETECTED event                 | ✅ Complete         |
| DataPipeline PROMOTION_FAILED     | Subscribes to PROMOTION_FAILED event                    | ✅ Complete         |

Daemon health check coverage increased from 22% to ~85%+ for critical coordinators.

**Integration Verification (December 27, 2025):**

Verified and tested coordination module integration:

| Component                  | Status            | Verification                                                                                  |
| -------------------------- | ----------------- | --------------------------------------------------------------------------------------------- |
| DataPipelineOrchestrator   | ✅ Fully wired    | Lines 690-850+ subscribe to all pipeline events incl. NEW_GAMES, REGRESSION, PROMOTION_FAILED |
| SelfplayScheduler          | ✅ Already wired  | Lines 1163-1164 subscribe to NODE_RECOVERED                                                   |
| P2PAutoDeployer            | ✅ Consolidated   | Now uses cluster_config helpers instead of inline YAML                                        |
| GauntletFeedbackController | ✅ 26 tests added | Full test coverage for feedback loop controller                                               |

**Test Coverage Additions:**

- `test_gauntlet_feedback_controller.py`: 26 tests covering initialization, lifecycle, analysis logic, regression/plateau detection, event handling, metrics, singleton pattern

**Sync Module Status (December 2025):**

| Module                                      | Status     | Notes                                                   |
| ------------------------------------------- | ---------- | ------------------------------------------------------- |
| `app/distributed/sync_coordinator.py`       | **ACTIVE** | Main sync layer for P2P orchestrator                    |
| `app/coordination/database_sync_manager.py` | **ACTIVE** | Base class for Elo/Registry sync (Dec 27, 2025)         |
| `app/coordination/auto_sync_daemon.py`      | **ACTIVE** | Automated P2P data sync with strategies                 |
| `app/coordination/sync_router.py`           | **ACTIVE** | Intelligent routing based on node capabilities          |
| `app/coordination/sync_facade.py`           | **ACTIVE** | Unified programmatic entry point for sync               |
| `app/coordination/sync_coordinator.py`      | DEPRECATED | Archive Q2 2026                                         |
| `app/coordination/cluster_data_sync.py`     | DEPRECATED | Has callers, use `AutoSyncDaemon(strategy="broadcast")` |
| `app/coordination/ephemeral_sync.py`        | DEPRECATED | Has callers, use `AutoSyncDaemon(strategy="ephemeral")` |
| `app/coordination/system_health_monitor.py` | DEPRECATED | Has callers, use `unified_health_manager.py`            |

**New Feedback Loop Modules (December 2025):**

| Module                            | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `evaluation_daemon.py`            | Auto-evaluate models after training          |
| `gauntlet_feedback_controller.py` | Adjust training params based on eval results |
| `feedback_loop_controller.py`     | Central feedback loop orchestration          |
| `model_performance_watchdog.py`   | Track rolling win rates, detect regression   |
| `quality_monitor_daemon.py`       | Monitor selfplay data quality                |

See ADR-007 for P2P orchestrator decomposition details.

### Event System Reference (December 2025)

The coordination infrastructure uses an event-driven architecture with 30+ event types. Key events for pipeline coordination:

| Event                     | Emitter                  | Subscribers                                   | Purpose                               |
| ------------------------- | ------------------------ | --------------------------------------------- | ------------------------------------- |
| `TRAINING_STARTED`        | TrainingCoordinator      | SyncRouter, IdleShutdown, DataPipeline        | Pause idle detection, prioritize sync |
| `TRAINING_COMPLETED`      | TrainingCoordinator      | FeedbackLoop, DataPipeline, ModelDistribution | Trigger evaluation pipeline           |
| `EVALUATION_COMPLETED`    | GameGauntlet             | FeedbackLoop, CurriculumIntegration           | Adjust curriculum weights             |
| `EVALUATION_PROGRESS`     | GameGauntlet             | MetricsAnalysisOrchestrator                   | Real-time tracking of eval progress   |
| `CURRICULUM_REBALANCED`   | CurriculumIntegration    | SelfplayScheduler, SelfplayOrchestrator       | Update selfplay allocation weights    |
| `ELO_VELOCITY_CHANGED`    | QueuePopulator           | SelfplayScheduler, UnifiedFeedback            | Adjust selfplay rate dynamically      |
| `ELO_SIGNIFICANT_CHANGE`  | EloSyncManager           | CurriculumIntegration, DataPipeline           | Trigger curriculum rebalancing        |
| `MODEL_PROMOTED`          | PromotionController      | ModelDistribution, FeedbackLoop               | Distribute new model to cluster       |
| `MODEL_UPDATED`           | ModelRegistry            | UnifiedDistributionDaemon                     | Sync model metadata to nodes          |
| `PROMOTION_FAILED`        | AutoPromotionDaemon      | ModelLifecycleCoordinator, DataPipeline       | Track failed promotions, notify       |
| `ADAPTIVE_PARAMS_CHANGED` | ImprovementOptimizer     | EvaluationFeedbackHandler, TrainingTrigger    | Apply real-time LR adjustments        |
| `DATA_SYNC_COMPLETED`     | P2POrchestrator          | DataPipelineOrchestrator                      | Trigger NPZ export after sync         |
| `NEW_GAMES_AVAILABLE`     | DataPipelineOrchestrator | SelfplayScheduler, ExportScheduler            | Trigger training pipeline             |
| `REGRESSION_DETECTED`     | ModelPerformanceWatchdog | ModelLifecycleCoordinator, DataPipeline       | Rollback bad models                   |

**Subscribing to Events:**

```python
from app.coordination.event_router import DataEventType, get_event_bus

bus = get_event_bus()
bus.subscribe(DataEventType.TRAINING_COMPLETED, my_handler)

async def my_handler(event: dict) -> None:
    config_key = event.get("config_key")
    model_path = event.get("model_path")
    # Process event...
```

### P2P Background Loops (December 2025)

The P2P orchestrator uses `LoopManager` to coordinate 5 background loops. Located in `scripts/p2p/loops/`:

| Loop                       | File                  | Interval | Purpose                                    |
| -------------------------- | --------------------- | -------- | ------------------------------------------ |
| `JobReaperLoop`            | `job_loops.py`        | 5 min    | Clean stale jobs (1hr), stuck jobs (2hr)   |
| `IdleDetectionLoop`        | `job_loops.py`        | 30 sec   | Detect idle GPUs, trigger selfplay         |
| `WorkerPullLoop`           | `job_loops.py`        | 30 sec   | Workers poll leader for work (pull model)  |
| `WorkQueueMaintenanceLoop` | `job_loops.py`        | 5 min    | Check timeouts, cleanup old items (24hr)   |
| `SelfHealingLoop`          | `resilience_loops.py` | 5 min    | Recover stuck jobs, clean stale processes  |
| `PredictiveMonitoringLoop` | `resilience_loops.py` | 5 min    | Track trends, emit alerts before threshold |

**Configuration Example:**

```python
from scripts.p2p.loops.job_loops import JobReaperLoop, JobReaperConfig

config = JobReaperConfig(
    stale_job_threshold_seconds=1800.0,  # 30 minutes
    stuck_job_threshold_seconds=3600.0,  # 1 hour
    max_jobs_to_reap_per_cycle=5,
)
loop = JobReaperLoop(
    get_active_jobs=orchestrator.get_jobs,
    cancel_job=orchestrator.cancel_job,
    config=config,
)
```

**Test Coverage (December 27, 2025):**

- `tests/unit/p2p/test_job_loops.py`: 37 tests for JobReaperLoop, IdleDetectionLoop
- `tests/unit/p2p/test_resilience_loops.py`: 25 tests for SelfHealingLoop, PredictiveMonitoringLoop

### Daemon Health Monitoring (December 2025)

85%+ of critical daemons now support `health_check()`. Query health programmatically:

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

# Get single daemon health
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
# Returns: {"status": "healthy", "running": True, "last_sync": ..., "errors_count": 0}

# Get all daemon health
all_health = dm.get_all_daemon_health()
# Returns dict of DaemonType -> health status

# Liveness probe (for Kubernetes/Docker)
if dm.liveness_probe():
    print("All critical daemons healthy")
```

**Daemons with health_check():**

| Daemon                 | Key Health Metrics                                  |
| ---------------------- | --------------------------------------------------- |
| `AUTO_SYNC`            | Last sync time, files synced, error rate            |
| `DATA_PIPELINE`        | Stage status, last NPZ export, pending games        |
| `FEEDBACK_LOOP`        | Subscribed events, last feedback, training active   |
| `SELFPLAY_COORDINATOR` | Active configs, jobs dispatched, curriculum weights |
| `EVALUATION`           | Last eval, queue depth, pass rate                   |
| `NODE_RECOVERY`        | Recoveries attempted, success rate                  |
| `CLUSTER_WATCHDOG`     | Nodes monitored, alerts generated                   |
| `P2P_BACKEND`          | Connection status, leader ID, peer count            |

### Migration Guides

Full migration documentation is in the archive:

- `archive/deprecated_coordination/README.md` - Coordination module migrations
- `archive/deprecated_scripts/README.md` - Script migrations

## Known Issues & Gotchas

1. **Canonical databases only**: Training scripts enforce `canonical_*.db` naming by default. Use `--allow-noncanonical` to bypass.

2. **Board size conventions**: Hex boards use "radius" convention. hex8 = radius 4 = 9x9 grid = 61 cells.

3. **GPU memory**: v2 models with batch_size=512 need ~8GB VRAM.

4. **PYTHONPATH**: Set `PYTHONPATH=.` when running scripts from the ai-service directory.

## File Locations

```
ai-service/
├── app/
│   ├── ai/              # AI implementations (neural net, MCTS, heuristics)
│   ├── config/          # Centralized configuration
│   │   └── env.py                    # Typed environment variable config
│   ├── coordination/    # Training pipeline orchestration
│   │   ├── event_router.py           # Unified event system
│   │   ├── pipeline_actions.py       # Stage action invokers
│   │   ├── daemon_manager.py         # Daemon lifecycle management
│   │   ├── daemon_runners.py         # Async runner functions for all daemon types
│   │   └── sync_bandwidth.py         # Bandwidth-coordinated transfers
│   ├── core/            # Core shared utilities
│   │   ├── ssh.py                    # Unified SSH client
│   │   └── node.py                   # Unified NodeInfo dataclass
│   ├── db/              # Database utilities (GameReplayDB)
│   ├── distributed/     # Cluster tools (cluster_monitor, data_catalog)
│   ├── monitoring/      # Unified cluster monitoring
│   ├── rules/           # Python rules engine (mirrors TS)
│   ├── training/        # Training pipeline
│   │   ├── train.py                  # Main training entry point
│   │   ├── training_enhancements.py  # Enhanced training utilities (facade)
│   │   ├── enhancements/             # Modular enhancement components
│   │   │   ├── data_quality_scoring.py    # Quality-weighted sampling
│   │   │   ├── hard_example_mining.py     # Curriculum learning
│   │   │   ├── per_sample_loss.py         # Per-sample loss tracking
│   │   │   ├── learning_rate_scheduling.py # Adaptive LR schedulers
│   │   │   ├── gradient_management.py     # Gradient accumulation/clipping
│   │   │   ├── checkpoint_averaging.py    # Checkpoint EMA
│   │   │   └── evaluation_feedback.py     # Training feedback handlers
│   │   ├── temperature_scheduling.py # Selfplay temperature schedules
│   │   └── online_learning.py        # EBMO online learning
│   └── utils/           # Utilities (game_discovery)
├── archive/             # Deprecated code with migration docs
├── config/              # Configuration files (templates provided)
├── data/
│   ├── games/           # Game databases
│   ├── training/        # NPZ training files
│   └── models/          # Trained model checkpoints
├── models/              # Production models by config
├── scripts/             # CLI tools and utilities
└── tests/               # Test suite
```

## Cluster Infrastructure

RingRift uses a P2P mesh network for distributed training across ~32 configured nodes (Dec 2025).

**Note:** Lambda Labs account restored Dec 28, 2025 - GH200 instances being setup.

### Active Cluster (Dec 28, 2025)

| Provider     | Nodes | GPUs                                        | Status  |
| ------------ | ----- | ------------------------------------------- | ------- |
| Lambda GH200 | TBD   | GH200 96GB (being setup)                    | Pending |
| Vast.ai      | 14    | RTX 5090/5080, 4090, 3090, A40, 3060/4060Ti | Active  |
| RunPod       | 8     | H100, A100 (5x), L40S, RTX 3090 Ti          | Active  |
| Nebius       | 3     | H100 80GB (2x), L40S backbone               | Active  |
| Vultr        | 2     | A100 20GB vGPU                              | Active  |
| Hetzner      | 3     | CPU only (P2P voters)                       | Active  |
| Local        | 2     | Mac Studio M3 (coordinator)                 | Active  |

### P2P Cluster Management

```bash
# Check cluster status (when P2P daemon is running)
python -m app.distributed.cluster_monitor

# Watch mode (live updates)
python -m app.distributed.cluster_monitor --watch --interval 10

# Update all nodes to latest code (parallel)
python scripts/update_all_nodes.py --restart-p2p

# Dry run to preview updates
python scripts/update_all_nodes.py --dry-run
```

### Cluster Update Script (NEW - Dec 2025)

The `update_all_nodes.py` script updates the entire cluster in parallel:

```bash
# Update all nodes to latest git commit
python scripts/update_all_nodes.py

# Update with P2P orchestrator restart
python scripts/update_all_nodes.py --restart-p2p

# Update to specific commit
python scripts/update_all_nodes.py --commit abc1234

# Preview changes without applying
python scripts/update_all_nodes.py --dry-run

# Limit parallel updates
python scripts/update_all_nodes.py --max-parallel 5
```

**Features:**

- Parallel updates across all configured nodes
- Automatic path detection by provider (RunPod, Vast, Nebius, etc.)
- Optional P2P orchestrator restart
- Connection retry with timeout
- Detailed summary report

**Update paths by provider:**

- RunPod: `/workspace/ringrift/ai-service`
- Vast.ai: `~/ringrift/ai-service` or `/workspace/ringrift/ai-service`
- Nebius: `~/ringrift/ai-service`
- Vultr/Hetzner: `/root/ringrift/ai-service`
- Mac Studio: `~/Development/RingRift/ai-service`

### P2P Crash Fixes (Dec 2025)

Recent stability improvements to P2P orchestrator:

**1. Dependency Validation (commit 1270b64)**

- Pre-flight checks for aiohttp, psutil, yaml modules
- Clear exit (code 2) with missing dependency message
- Extended validation in node_resilience.py

**2. Gzip Handling (commit 1270b64)**

- Magic byte detection (0x1f 0x8b) before decompression
- Handles clients that set Content-Encoding: gzip but send raw JSON
- Reduces 500 errors from gossip endpoint

**3. Startup Grace Period (commit dade90f)**

- 120-second grace period during P2P startup
- Prevents killing processes during slow state file loading
- Configurable via `RINGRIFT_P2P_STARTUP_GRACE_PERIOD`

**4. SystemExit Handling (commit 6649601)**

- Catches SystemExit in \_safe_task_wrapper
- Prevents "Task exception was never retrieved" log pollution

**5. /dev/shm Fallback (commit 6649601)**

- Graceful fallback to disk storage when /dev/shm unavailable
- Fixes macOS compatibility and permission issues

**6. Port Binding Errors (commit 6649601)**

- Clear error messages for port 8770 conflicts
- Suggested remediation: `lsof -i :8770` and `pkill -f p2p_orchestrator`

### P2P Manager Modules (December 2025)

The P2P orchestrator has been decomposed into modular manager classes at `scripts/p2p/managers/`:

| Manager               | Purpose                                      | Key Methods                                    |
| --------------------- | -------------------------------------------- | ---------------------------------------------- |
| `JobManager`          | Job spawning and lifecycle management        | `run_gpu_selfplay_job()`, `spawn_training()`   |
| `TrainingCoordinator` | Training dispatch, gauntlet, model promotion | `dispatch_training_job()`, `check_readiness()` |
| `SelfplayScheduler`   | Priority-based selfplay config selection     | `pick_weighted_config()`, `get_target_jobs()`  |
| `NodeSelector`        | Node ranking and selection for job dispatch  | `get_best_gpu_node()`, `get_training_nodes()`  |
| `SyncPlanner`         | Manifest collection and sync planning        | `collect_manifest()`, `create_sync_plan()`     |
| `StateManager`        | SQLite persistence, cluster epoch tracking   | `load_state()`, `save_state()`                 |

All managers use dependency injection for testability. See `scripts/p2p/managers/README.md` for architecture details.

**Event System Integration**:

Managers emit events for pipeline coordination:

| Event                 | Emitter             | Subscribers                     |
| --------------------- | ------------------- | ------------------------------- |
| `DATA_SYNC_COMPLETED` | SyncPlanner         | DataPipelineOrchestrator        |
| `TRAINING_STARTED`    | TrainingCoordinator | SyncRouter, IdleShutdown        |
| `TRAINING_COMPLETED`  | TrainingCoordinator | FeedbackLoop, ModelDistribution |
| `HOST_OFFLINE`        | P2POrchestrator     | UnifiedHealthManager            |
| `LEADER_ELECTED`      | P2POrchestrator     | LeadershipCoordinator           |

**Health Check System**:

All managers and loops implement `health_check()` returning `HealthCheckResult` for DaemonManager integration. Health is aggregated at the `/status` endpoint.

**Background Loops** (`scripts/p2p/loops/`):

| Loop                 | Interval | Purpose                              |
| -------------------- | -------- | ------------------------------------ |
| `JobReaperLoop`      | 5 min    | Clean stale/stuck jobs               |
| `IdleDetectionLoop`  | 30 sec   | GPU idle detection, trigger selfplay |
| `WorkerPullLoop`     | 30 sec   | Workers poll leader for work         |
| `SelfHealingLoop`    | 5 min    | Recover stuck jobs                   |
| `EloSyncLoop`        | 5 min    | Elo rating synchronization           |
| `QueuePopulatorLoop` | 1 min    | Work queue maintenance               |

Loops use `BaseLoop` with exponential backoff and `LoopManager` for lifecycle coordination.

### P2P SWIM/Raft Transition (December 2025)

The P2P orchestrator supports optional SWIM/Raft protocols for improved reliability.

**Current Production State**: HTTP polling (membership) + Bully election (consensus)

- These are battle-tested and work reliably
- SWIM/Raft are optional upgrades, not required for production

**SWIM Protocol (Membership)** - OPTIONAL

- Gossip-based membership with 5s failure detection (vs 60-90s HTTP polling)
- Leaderless architecture - no election required for membership changes
- O(1) bandwidth per node regardless of cluster size
- Status: Code ready, requires `swim-p2p>=1.2.0` (not installed in production)

**Raft Protocol (Consensus)** - OPTIONAL

- Replicated work queue with sub-second leader failover
- Distributed locks for exclusive job claims
- Automatic state machine replication across voters
- Status: Code ready, requires `pysyncobj>=0.3.14` (not installed in production)

**Feature Flags**

```bash
# Enable SWIM membership (default: false until deps installed)
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=swim  # Options: http, swim, hybrid

# Enable Raft consensus (default: false until deps installed)
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=raft  # Options: bully, raft, hybrid
```

**Current State**: Feature flags are validated at startup. If flags are enabled but dependencies missing, warnings are logged and system falls back to HTTP polling (membership) and Bully election (consensus).

**New Files**:

- `scripts/p2p/membership_mixin.py` - SWIM membership integration
- `scripts/p2p/consensus_mixin.py` - Raft consensus integration
- `scripts/p2p/handlers/swim.py` - SWIM status endpoints
- `scripts/p2p/handlers/raft.py` - Raft status endpoints
- `app/p2p/hybrid_coordinator.py` - Routes operations based on feature flags
- `app/p2p/swim_adapter.py` - SWIM protocol adapter
- `app/p2p/raft_state.py` - Raft state machines

**Status Endpoint**

The `/status` endpoint now includes SWIM/Raft protocol status:

```json
{
  "swim_raft": {
    "membership_mode": "http",
    "consensus_mode": "bully",
    "swim": { "enabled": false, "available": false },
    "raft": { "enabled": false, "available": false },
    "hybrid_status": { "swim_fallback_active": true, "raft_fallback_active": true }
  }
}
```

### Cluster Configuration

The cluster is configured via `config/distributed_hosts.yaml`:

- **P2P Voters**: 5 stable nodes for leader election (quorum = 3)
  - nebius-backbone-1, nebius-h100-3, hetzner-cpu1, hetzner-cpu2, vultr-a100-20gb
  - Only non-containerized, non-NAT-blocked nodes
- **Auto-sync**: 60s interval (reduced from 300s for fresher data)
- **Gossip interval**: 15s (reduced from 60s for faster propagation)
- **Bandwidth limits**: 100 MB/s (RunPod/Nebius), 50 MB/s (Vast/Vultr)

## Auto-Promotion Workflow

After training, run gauntlet evaluation to promote models:

```bash
PYTHONPATH=. python3 scripts/auto_promote.py --gauntlet \
  --model models/my_model.pth \
  --board-type hex8 --num-players 4 \
  --games 50

# With cluster sync (if configured)
PYTHONPATH=. python3 scripts/auto_promote.py --gauntlet \
  --model models/my_model.pth \
  --board-type hex8 --num-players 4 \
  --games 50 --sync-to-cluster
```

**Promotion thresholds:**

- vs RANDOM: 85% win rate required
- vs HEURISTIC: 60% win rate required

## Recent Changes (Dec 27, 2025)

### Data Distribution Infrastructure

New scripts for automated data distribution from OWC to training nodes:

- **`dynamic_data_distribution.py`**: Daemon running on mac-studio (PID active)
  - HTTP distribution with rsync fallback for large files (>100MB)
  - aria2/curl fallback chain for failed downloads
  - 5-minute distribution cycle
- **`dynamic_space_manager.py`**: Daemon running on mac-studio
  - Proactive cleanup at 60% disk usage
  - Cleans old logs, empty DBs, old checkpoints
- **`consolidate_jsonl_databases.py`**: Database consolidation script
  - Merges 44 jsonl_aggregated.db files (147GB) into per-config databases
  - hex8_4p: 3,210+ games consolidated
- **`scheduled_npz_export.py`**: Updated to search jsonl_aggregated databases
  - Now finds games across all cluster node snapshots

### Batch Rsync

Added `BatchRsync` class to `sync_bandwidth.py`:

```python
from app.coordination.sync_bandwidth import get_batch_rsync

batch = get_batch_rsync()
result = await batch.sync_files(
    source_dir="/data/games/",
    dest="ubuntu@gpu-node:/data/games/",
    host="gpu-node",
    files=["game1.db", "game2.db", "game3.db"],
)
```

### GPU Vectorization Status

Current state of `app/ai/gpu_parallel_games.py`:

- Only 1 `.item()` call remains (statistics tracking, not in hot path)
- 6-10x speedup on CUDA vs CPU
- MPS (Apple Silicon) remains slow due to kernel launch overhead
- 99 `.cpu()` / 96 `.numpy()` calls for data extraction (not in hot loops)

### Active Daemons on mac-studio

- `dynamic_data_distribution.py` - 5min cycle
- `dynamic_space_manager.py` - 30min cycle
- `scheduled_npz_export.py` - 2hr cycle

### Orphan Games Detection & Recovery (Dec 27, 2025)

New event-driven system for detecting and recovering "orphan" games (games generated on nodes that later become unreachable):

**Event Flow:**

```
OrphanDetectionDaemon
    ↓ (emits ORPHAN_GAMES_DETECTED)
DataPipelineOrchestrator._on_orphan_games_detected()
    ↓ (triggers priority sync)
SyncFacade.trigger_priority_sync()
    ↓ (syncs data, emits DATA_SYNC_COMPLETED)
DataPipelineOrchestrator._on_orphan_games_registered()
    ↓ (emits NEW_GAMES_AVAILABLE)
Downstream consumers (export, training)
```

**Key Files:**

- `app/coordination/data_pipeline_orchestrator.py:800-808` - Event subscriptions
- `app/coordination/data_pipeline_orchestrator.py:1053-1136` - Handler implementations
- `app/coordination/sync_facade.py:474-522` - `trigger_priority_sync()` method

**Usage:**

```python
from app.coordination.sync_facade import get_sync_facade

# Trigger urgent sync for orphan game recovery
facade = get_sync_facade()
response = await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
    config_key="hex8_2p",
    data_type="games",
)
```

### Async Cluster Status (Dec 27, 2025)

The `ClusterMonitor` now has async versions of status methods to avoid blocking the event loop:

**New Methods:**

- `_async_run_ssh_command()` - Uses `asyncio.create_subprocess_exec`
- `_async_check_host_connectivity()` - Async connectivity check
- `get_node_status_async()` - Single node status (async)
- `get_cluster_status_async()` - Full cluster status (async)

**Usage:**

```python
from app.coordination.cluster_status_monitor import ClusterMonitor

monitor = ClusterMonitor()

# Async context (preferred for daemons)
status = await monitor.get_cluster_status_async()

# Sync context (for scripts)
status = monitor.get_cluster_status()
```

The `run_forever()` loop now uses `get_cluster_status_async()` to avoid blocking.

### Centralized Paths (Dec 27, 2025)

Extended `app/utils/paths.py` with additional path constants:

```python
from app.utils.paths import (
    # Core paths
    AI_SERVICE_ROOT,
    DATA_DIR,
    GAMES_DIR,
    TRAINING_DIR,       # NPZ training data (NEW)
    COORDINATION_DIR,   # Coordination state DBs (NEW)
    MODELS_DIR,
    LOGS_DIR,

    # Helper functions
    get_games_db_path,
    get_selfplay_db_path,
    get_training_npz_path,  # NEW - returns TRAINING_DIR / f"{config_key}.npz"
)

# Environment variable overrides
# AI_SERVICE_DATA_DIR - Override DATA_DIR and all subdirs
# AI_SERVICE_MODELS_DIR - Override MODELS_DIR
# AI_SERVICE_LOGS_DIR - Override LOGS_DIR
```

### Coordination Defaults (Dec 27, 2025)

Centralized timeout and configuration values in `app/config/coordination_defaults.py`:

```python
from app.config.coordination_defaults import (
    TransportDefaults,   # HTTP_TIMEOUT, SSH_TIMEOUT, CONNECT_TIMEOUT
    LockDefaults,        # LOCK_TIMEOUT, ACQUIRE_TIMEOUT
    SyncDefaults,        # LOCK_TIMEOUT
    get_timeout,         # Convenience function
)

# Usage
http_timeout = get_timeout("http")   # TransportDefaults.HTTP_TIMEOUT
ssh_timeout = get_timeout("ssh")     # TransportDefaults.SSH_TIMEOUT
```

### Smoke Tests for Coordination (Dec 27, 2025)

Added `tests/unit/coordination/test_coordination_smoke.py` with 11 tests:

- `TestEventSubscriptionWiring` - Verifies DataPipelineOrchestrator subscriptions
- `TestClusterStatusMonitorAsync` - Verifies async method availability
- `TestPathConfiguration` - Verifies paths.py exports
- `TestCoordinationDefaults` - Verifies timeout configuration
- `TestDataEventTypes` - Verifies ORPHAN_GAMES events exist

Run with: `pytest tests/unit/coordination/test_coordination_smoke.py -v`

### Integration Fixes (Dec 27, 2025)

**SyncCoordinator.health_check()** - Added to `app/distributed/sync_coordinator.py`:

- Returns `HealthCheckResult` for DaemonManager integration
- Reports: sync health status, consecutive failures, data server health
- Includes transport availability (aria2, P2P, SSH, gossip)

**Orphaned Event Wiring** - Added to `DataPipelineOrchestrator`:

| Event                   | Handler                       | Purpose                                   |
| ----------------------- | ----------------------------- | ----------------------------------------- |
| `REPAIR_COMPLETED`      | `_on_repair_completed()`      | Retrigger sync after data repair          |
| `REPAIR_FAILED`         | `_on_repair_failed()`         | Track repair failures for circuit breaker |
| `QUALITY_SCORE_UPDATED` | `_on_quality_score_updated()` | Aggregate quality metrics                 |
| `CURRICULUM_REBALANCED` | `_on_curriculum_rebalanced()` | Update pipeline priorities                |
| `CURRICULUM_ADVANCED`   | `_on_curriculum_advanced()`   | Track curriculum progression              |

**Singleton Consolidation** - Unified implementations:

- `app/core/singleton_mixin.py` → Re-export with deprecation warning
- `app/coordination/singleton_mixin.py` → Canonical location (503 LOC)

```python
from app.coordination.singleton_mixin import (
    SingletonMixin,      # Mixin for existing classes
    SingletonMeta,       # Metaclass for new classes
    ThreadSafeSingletonMixin,  # Enhanced thread safety
    singleton,           # Decorator (simplest)
)
```

**LogContext Rename** - Clarified naming collision:

- `app/core/logging_config.LogContext` → Renamed to `TemporaryLogLevel`
- `app/utils/logging_utils.LogContext` → Unchanged (structured context)

### Wave 5-6 Infrastructure Improvements (Dec 27, 2025)

**HandlerBase Class** (`app/coordination/handler_base.py`):

Canonical base class for event-driven handlers with 550 LOC, 45 tests:

```python
from app.coordination.handler_base import HandlerBase, HealthCheckResult

class MyDaemon(HandlerBase):
    def __init__(self):
        super().__init__(name="my_daemon", cycle_interval=60.0)

    async def _run_cycle(self) -> None:
        pass  # Main work loop

    def _get_event_subscriptions(self) -> dict:
        return {"my_event": self._on_my_event}

# Supports both _get_event_subscriptions() and legacy _get_subscriptions()
```

**HealthCheckHelper Class** (`app/coordination/health_check_helper.py`):

Reusable health check logic for all coordinators:

```python
from app.coordination.health_check_helper import HealthCheckHelper

# Reusable health check methods
is_ok, msg = HealthCheckHelper.check_error_rate(errors=5, cycles=100, threshold=0.5)
is_ok, msg = HealthCheckHelper.check_uptime_grace(start_time, grace_period=30)
is_ok, msg = HealthCheckHelper.check_queue_depth(queue.qsize(), max_depth=1000)
```

**Missing Daemon Profile Wiring** (Dec 27, 2025):

Added 3 daemons to DaemonManager profiles:

| Daemon                     | Profile       | Purpose                             |
| -------------------------- | ------------- | ----------------------------------- |
| `DATA_CONSOLIDATION`       | coordinator   | Consolidate scattered data files    |
| `DISK_SPACE_MANAGER`       | training_node | Manage disk space on training nodes |
| `COORDINATOR_DISK_MANAGER` | coordinator   | Manage coordinator disk space       |

**Event Wiring Verification Guide** (`docs/runbooks/EVENT_WIRING_VERIFICATION.md`):

New runbook for verifying event emitter/subscriber wiring:

- Quick verification script to find orphan events
- Manual verification procedures for new events
- Critical event wiring matrix (9 key events)
- Automated CI check YAML template
- Troubleshooting guide for event routing issues

**SyncRouter Backpressure Integration** (`app/coordination/sync_router.py`):

Added handlers for backpressure events to pause/resume sync operations:

```python
# Subscribed to BACKPRESSURE_ACTIVATED and BACKPRESSURE_RELEASED
router.is_under_backpressure()  # Query current state
```

**Alert Types Test Coverage** (`tests/unit/coordination/test_alert_types.py`):

30 unit tests covering:

- `AlertSeverity` enum values and ordering
- `AlertCategory` and `AlertState` enums
- `Alert` dataclass properties (`is_critical`, `is_error_or_above`, `to_dict()`)
- `create_alert()` factory function
- Backward-compatible aliases (`ErrorSeverity`, `StallSeverity`, etc.)
- Helper functions (`severity_to_log_level`, `severity_to_color`)

**EVENT_SYSTEM_REFERENCE.md Updates**:

Added 4 missing events to documentation:

| Event                   | Category        | Purpose                           |
| ----------------------- | --------------- | --------------------------------- |
| `ORPHAN_GAMES_DETECTED` | Data Collection | Detect games on unreachable nodes |
| `REPAIR_COMPLETED`      | Data Integrity  | Data repair success               |
| `REPAIR_FAILED`         | Data Integrity  | Data repair failure               |
| `TASK_ABANDONED`        | Job Management  | Track cancelled jobs              |

**DAEMON_FAILURE_RECOVERY.md Runbook** (`docs/runbooks/DAEMON_FAILURE_RECOVERY.md`):

New runbook covering:

- Health endpoint usage (`/health`, `/ready`, `/metrics` on port 8790)
- Programmatic health checks via `DaemonManager`
- CLI status via `launch_daemons.py --status`
- Common failure patterns: Startup failure, Crash loop, Health check failure, Event subscription loss
- Exponential backoff table (1s → 16s max)
- Full system recovery procedures
