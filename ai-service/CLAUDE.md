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
- `DaemonManager` - Lifecycle for all background daemons (30+ types)
- `ClusterMonitor` - Real-time cluster health
- `FeedbackLoopController` - Training feedback signals
- `DataPipelineOrchestrator` - Pipeline stage tracking
- `QueuePopulator` - Work queue maintenance until Elo targets met

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
- **`queue_populator.py`**: Maintains work queue until Elo targets met (60% selfplay, 30% training, 10% tournament)
- **`idle_resource_daemon.py`**: Spawns selfplay on idle GPUs using SelfplayScheduler priorities
- **`utilization_optimizer.py`**: Matches GPU capabilities to board sizes, optimizes cluster utilization
- **`feedback_loop_controller.py`**: Manages training feedback signals and quality thresholds

**Event System:**

- **`event_router.py`**: Unified event bus with content-based deduplication (SHA256)
- **`pipeline_actions.py`**: Stage invokers with circuit breaker protection
- **`data_pipeline_orchestrator.py`**: Tracks and triggers pipeline stages

**Daemon Management:**

- **`daemon_manager.py`**: Lifecycle management for 30+ daemon types
- **`daemon_adapters.py`**: Wrappers for existing daemons (sync, promotion, distillation)
- **`sync_bandwidth.py`**: Bandwidth-coordinated rsync with host-level limits
- **`auto_sync_daemon.py`**: Automated P2P data sync with push-from-generator + gossip replication

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

| Deprecated                    | Replacement                       | Notes                     |
| ----------------------------- | --------------------------------- | ------------------------- |
| `DaemonType.SYNC_COORDINATOR` | `DaemonType.AUTO_SYNC`            | Use AutoSyncDaemon        |
| `DaemonType.HEALTH_CHECK`     | `DaemonType.NODE_HEALTH_MONITOR`  | Unified health monitoring |
| `vast_idle_daemon.py`         | `unified_idle_shutdown_daemon.py` | Provider-agnostic         |
| `lambda_idle_daemon.py`       | `unified_idle_shutdown_daemon.py` | Provider-agnostic         |

### Training Modules (Deprecated Dec 2025)

| Deprecated                                     | Replacement               | Notes                          |
| ---------------------------------------------- | ------------------------- | ------------------------------ |
| `orchestrated_training.py`                     | `unified_orchestrator.py` | Unified training orchestration |
| `integrated_enhancements.py`                   | `unified_orchestrator.py` | Consolidated enhancements      |
| `DataQualityScorer` (in training_enhancements) | `UnifiedQualityScorer`    | See `unified_quality.py`       |

### Training Entry Points

Three main options - use the recommended one:

| Script                 | Purpose                                  | Recommended?       |
| ---------------------- | ---------------------------------------- | ------------------ |
| `master_loop.py`       | Full cluster automation with 30+ daemons | ✅ Yes             |
| `run_training_loop.py` | Simple 1-config pipeline                 | For single configs |
| `unified_ai_loop.py`   | Legacy wrapper                           | ❌ Deprecated      |

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

RingRift uses a P2P mesh network for distributed training across ~52 configured nodes (Dec 2025).

### Active Cluster (Dec 2025)

| Provider | Nodes | GPUs                               | Status |
| -------- | ----- | ---------------------------------- | ------ |
| Vast.ai  | ~30   | RTX 5090, 4090, 3090, A40, 4060 Ti | Active |
| RunPod   | 6     | H100, A100, L40S, RTX 3090 Ti      | Active |
| Nebius   | 4     | H100 (80GB), L40S                  | Active |
| Vultr    | 3     | A100 (20GB vGPU)                   | Active |
| Hetzner  | 4     | CPU only (voters)                  | Active |
| Local    | 2     | Mac Studio M3 (coordinator)        | Active |

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

| Manager               | Lines | Purpose                                        |
| --------------------- | ----- | ---------------------------------------------- |
| `StateManager`        | 629   | SQLite persistence, cluster epoch tracking     |
| `NodeSelector`        | 330   | Node ranking and selection for job dispatch    |
| `SyncPlanner`         | 704   | Manifest collection and sync planning          |
| `JobManager`          | 663   | Job spawning and lifecycle management          |
| `SelfplayScheduler`   | 737   | Priority-based selfplay config selection       |
| `TrainingCoordinator` | 734   | Training dispatch, completion, model promotion |

All managers use dependency injection for testability. See `scripts/p2p/managers/README.md` for architecture details.

### P2P SWIM/Raft Transition (December 2025)

The P2P orchestrator is transitioning to battle-tested protocols for improved reliability:

**SWIM Protocol (Membership)**

- Gossip-based membership with 5s failure detection (vs 60-90s HTTP polling)
- Leaderless architecture - no election required for membership changes
- O(1) bandwidth per node regardless of cluster size
- Status: Integration ready, pending `swim-p2p>=1.2.0` installation

**Raft Protocol (Consensus)**

- Replicated work queue with sub-second leader failover
- Distributed locks for exclusive job claims
- Automatic state machine replication across voters
- Status: Integration ready, pending `pysyncobj>=0.3.14` installation

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
