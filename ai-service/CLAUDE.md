# CLAUDE.md - AI Assistant Memory for RingRift

This file provides context for AI assistants working on this codebase.
It complements AGENTS.md with operational knowledge and current state.

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
| `square8`   | 8×8               | 64    | 2, 3, 4       |
| `square19`  | 19×19             | 361   | 2, 3, 4       |
| `hex8`      | 9×9 (radius 4)    | 61    | 2, 3, 4       |
| `hexagonal` | 25×25 (radius 12) | 469   | 2, 3, 4       |

## Cluster Infrastructure

### Primary Training Nodes (SSH via Tailscale)

```bash
# H100 nodes (primary access points)
ssh -i ~/.ssh/id_cluster ubuntu@100.78.101.123  # lambda-h100 (80GB)
ssh -i ~/.ssh/id_cluster ubuntu@100.97.104.89   # lambda-2xh100 (160GB)

# GH200 nodes (96GB each) - use P2P status to find active ones
# Note: GH200 nodes a, e, f are retired. Active: b-new, d, g, h, i, o, p, q, r, s, t

# A10 node (23GB)
ssh -i ~/.ssh/id_cluster ubuntu@100.91.25.13    # lambda-a10
```

### Cluster Monitoring

```bash
# Quick P2P cluster status (preferred)
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} nodes")
'

# Or use the Python monitor
python -m app.distributed.cluster_monitor

# Watch mode (live updates)
python -m app.distributed.cluster_monitor --watch --interval 10
```

## Common Commands

### Training

```bash
# Export training data from database
python scripts/export_replay_dataset.py \
  --db data/games/canonical_hex8_2p.db \
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

# Start training on cluster
ssh -i ~/.ssh/id_cluster ubuntu@100.123.183.70 \
  "cd ~/ringrift/ai-service && nohup python -m app.training.train \
   --board-type hex8 --num-players 2 \
   --data-path data/training/hex8_2p.npz \
   --model-version v2 --batch-size 512 --epochs 20 \
   > logs/train.log 2>&1 &"
```

### Automated Training Pipeline (NEW Dec 2025)

One-command training loop that automatically chains: selfplay → sync → export → train → evaluate → promote

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

**Alternative: Manual Pipeline with Flags**

```bash
# Selfplay with pipeline event emission
python scripts/selfplay.py \
  --board hex8 --num-players 2 \
  --engine gumbel \
  --emit-pipeline-events

# Training with auto-trigger
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --enable-pipeline-auto-trigger
```

**Environment Variable Alternative**

```bash
export COORDINATOR_AUTO_TRIGGER_PIPELINE=true
python -m app.training.train --board-type hex8 --num-players 2 ...
```

### Transfer Learning (2p → 4p)

```bash
# Step 1: Resize value head from 2 outputs to 4 outputs
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_sq8_2p.pth \
  --output models/transfer_sq8_4p_init.pth \
  --board-type square8

# Step 2: Train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/transfer_sq8_4p_init.pth \
  --data-path data/training/sq8_4p.npz \
  --save-path models/sq8_4p_transfer.pth

# Direct transfer (partial loading, value head randomly initialized)
python -m app.training.train \
  --board-type hex8 --num-players 4 \
  --init-weights models/canonical_hex8_2p.pth \
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
python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db

# Run canonical selfplay parity gate
python scripts/run_canonical_selfplay_parity_gate.py --board-type hex8
```

### Model Evaluation

```bash
# Gauntlet evaluation (model vs baselines)
python -m app.gauntlet.runner \
  --board-type hex8 --num-players 2 \
  --model-path models/hex8_2p/best_model.pt
```

## Key Utilities

### GameDiscovery (`app/utils/game_discovery.py`)

Unified utility for finding game databases across all storage patterns:

- `find_all_databases()` - Find all .db files with game data
- `find_databases_for_config(board_type, num_players)` - Filter by config
- `RemoteGameDiscovery` - SSH-based cluster-wide discovery

### ClusterMonitor (`app/distributed/cluster_monitor.py`)

Real-time cluster monitoring:

- Game counts per node
- Training process detection
- Disk usage monitoring
- CLI with watch mode

### DataQuality (`app/training/data_quality.py`)

Quality checking for training data:

- `DatabaseQualityChecker` - Validate database schema/content
- `TrainingDataValidator` - Validate NPZ files

### GumbelCommon (`app/ai/gumbel_common.py`)

Unified data structures for all Gumbel MCTS variants:

- `GumbelAction` - Action with policy logit, Gumbel noise, value tracking
- `GumbelNode` - Node for MCTS tree with visit counts and values
- `LeafEvalRequest` - Batched leaf evaluation request
- Budget constants: `GUMBEL_BUDGET_THROUGHPUT` (64), `GUMBEL_BUDGET_STANDARD` (150), `GUMBEL_BUDGET_QUALITY` (800), `GUMBEL_BUDGET_ULTIMATE` (1600)
- `get_budget_for_difficulty(difficulty)` - Map difficulty to budget tier

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

- **`event_router.py`**: Unified event bus bridging in-memory, stage, and cross-process events
- **`pipeline_actions.py`**: Stage invokers with circuit breaker protection
- **`data_pipeline_orchestrator.py`**: Tracks and triggers pipeline stages
- **`daemon_manager.py`**: Lifecycle management for all daemons
- **`daemon_adapters.py`**: Wrappers for existing daemons (sync, promotion, distillation)
- **`sync_bandwidth.py`**: Bandwidth-coordinated rsync with host-level limits

```bash
# Launch all daemons under unified management
python scripts/launch_daemons.py --all

# Check daemon status
python scripts/launch_daemons.py --status

# Launch specific daemons
python scripts/launch_daemons.py --sync-only
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
- `AlphaZeroTemperature` for τ=1 → τ=0 at move N
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

- TD-Energy updates: E(s,a) predicts min E(s', a') over next state
- Outcome-weighted contrastive loss
- Rolling buffer for stability

## Current Model State (as of Dec 2025)

| Config  | Status     | Best Accuracy | Location                     |
| ------- | ---------- | ------------- | ---------------------------- |
| hex8_2p | Production | 76.2% policy  | models/canonical_hex8_2p.pth |
| sq8_2p  | Production | -             | models/canonical_sq8_2p.pth  |
| sq8_3p  | Production | -             | models/canonical_sq8_3p.pth  |
| sq19_2p | Production | -             | models/canonical_sq19_2p.pth |

### GPU Selfplay Status

The GPU parallel games engine is production-ready with 100% parity:

- Location: `app/ai/gpu_parallel_games.py`
- Current speedup: ~6.5x on CUDA
- Optimization status: Partial (~80 `.item()` calls remain)
- Full vectorization would yield 10-20x speedup

## Architecture Notes

### Neural Network (v2)

- 96 channels, 6 residual blocks with SE attention
- Separate policy and value heads
- Policy: position-aware encoding with board geometry
- Value: per-player win probability (softmax for multiplayer)

### Training Pipeline

1. Self-play generates games → SQLite databases
2. `export_replay_dataset.py` converts to NPZ (features, policy, value)
3. `app.training.train` trains with early stopping
4. Gauntlet evaluation against baselines

### Data Flow

```
Self-play (Python/TS) → GameReplayDB (.db)
                              ↓
              export_replay_dataset.py
                              ↓
                    Training NPZ files
                              ↓
                   app.training.train
                              ↓
                    Model checkpoints
```

## Known Issues & Gotchas

1. **Canonical databases only**: Training scripts enforce `canonical_*.db` naming by default. Use `--allow-noncanonical` to bypass.

2. **Board size conventions**: Hex boards use "radius" convention. hex8 = radius 4 = 9×9 grid = 61 cells.

3. **Remote module paths**: Cluster nodes have different Python paths. Some modules like `app.ai.heuristic_ai` may not exist remotely.

4. **SSH timeouts**: Lambda nodes can have intermittent connectivity. Use `--timeout 30` for cluster operations.

5. **GPU memory**: v2 models with batch_size=512 need ~8GB VRAM. GH200 nodes have 96GB, plenty of headroom.

## File Locations

```
ai-service/
├── app/
│   ├── ai/              # AI implementations (neural net, MCTS, heuristics)
│   ├── coordination/    # Training pipeline orchestration (NEW Dec 2025)
│   │   ├── event_router.py           # Unified event system
│   │   ├── pipeline_actions.py       # Stage action invokers
│   │   ├── daemon_manager.py         # Daemon lifecycle management
│   │   └── sync_bandwidth.py         # Bandwidth-coordinated transfers
│   ├── db/              # Database utilities (GameReplayDB)
│   ├── distributed/     # Cluster tools (cluster_monitor, data_catalog)
│   ├── monitoring/      # Unified cluster monitoring
│   ├── rules/           # Python rules engine (mirrors TS)
│   ├── training/        # Training pipeline
│   │   ├── temperature_scheduling.py # Selfplay temperature schedules
│   │   └── online_learning.py        # EBMO online learning
│   └── utils/           # Utilities (game_discovery)
├── archive/             # Deprecated code with migration docs
│   ├── deprecated_scripts/
│   └── deprecated_coordination/
├── config/
│   └── distributed_hosts.yaml  # Cluster node configuration
├── data/
│   ├── games/           # Game databases
│   ├── training/        # NPZ training files
│   └── models/          # Trained model checkpoints
├── models/              # Production models by config
├── scripts/             # CLI tools and utilities
└── tests/               # Test suite
```

## Recent Session Context (Dec 2025)

Recent work covered:

- **P2P Cluster**: 21 active nodes with leader election, ~400+ selfplay jobs
- **GPU Parity**: 100% verified (10K seeds tested) - production ready
- **Models**: 4 canonical models in production (hex8_2p, sq8_2p, sq8_3p, sq19_2p)
- **Infrastructure**: Updated voter configuration, fixed node_resilience issues
- **Tests**: 11,274 passing (98.5% pass rate)
- **Auto-Promotion Pipeline**: Added gauntlet-based model promotion (scripts/auto_promote.py)
- **4-Player Gauntlet Fix**: Fixed multiplayer game handling in game_gauntlet.py

### Code Consolidation (Dec 24, 2025)

Major consolidation of duplicated code:

- **`gumbel_common.py`**: Unified 3 copies of GumbelAction/GumbelNode into single source
- **`selfplay_runner.py`**: Unified SelfplayRunner base class for all selfplay variants
- **Budget constants**: Consolidated scattered Gumbel budget defaults into named tiers
- **Export scripts**: Archived `export_replay_dataset_parallel.py` and `export_filtered_training.py` (now flags in main script)

**Coordination Infrastructure** (Dec 24, 2025):

- **`event_router.py`**: Unified event system (supersedes `unified_event_coordinator.py`)
- **`pipeline_actions.py`**: Training pipeline stage invokers (export, train, evaluate, promote)
- **`daemon_adapters.py`**: Daemon wrappers for unified DaemonManager lifecycle
- **`sync_bandwidth.py`**: Bandwidth-coordinated rsync for cluster transfers
- **`launch_daemons.py`**: Master daemon launcher script
- **`auto_promote.py`**: Refactored to use PromotionController

**Already Existing Training Utilities**:

- **`temperature_scheduling.py`**: 7 schedule types (linear, cosine, adaptive, curriculum, etc.)
- **`online_learning.py`**: EBMO online learning with TD-Energy updates

**Quality & Event System** (Dec 24, 2025):

- **`unified_quality.py`**: Single source of truth for quality scoring (`UnifiedQualityScorer`)
- **`event_router.py`**: Fixed fire-and-forget task handling with error callbacks
- **`game_gauntlet.py`**: Now emits EVALUATION_COMPLETED events for curriculum feedback
- **`training_coordinator.py`**: Subscribes to cluster health events for training decisions

**Deprecated Modules** (with runtime warnings):

- `orchestrated_training.py` → use `unified_orchestrator.py`
- `integrated_enhancements.py` → use `unified_orchestrator.py`
- `training_enhancements.DataQualityScorer` → use `unified_quality.UnifiedQualityScorer`

See `archive/deprecated_scripts/README.md` and `archive/deprecated_coordination/README.md` for archived module documentation.

### Auto-Promotion Workflow

After training, run gauntlet evaluation to promote models:

```bash
# On cluster node with model
PYTHONPATH=. python3 scripts/auto_promote.py --gauntlet \
  --model models/my_model.pth \
  --board-type hex8 --num-players 4 \
  --games 50 --sync-to-cluster
```

**Promotion thresholds:**

- vs RANDOM: 85% win rate required
- vs HEURISTIC: 60% win rate required

### Active Training Jobs (Dec 24, 2025)

| Config       | Node           | Status                    |
| ------------ | -------------- | ------------------------- |
| square8_2p   | lambda-gh200-a | Training v2/v3, Exporting |
| hex8_2p      | lambda-gh200-a | Training v2               |
| hex8_4p      | lambda-gh200-a | Training v2               |
| hex8_2p      | lambda-2xh100  | Gumbel selfplay (2 GPUs)  |
| square19_3p  | lambda-gh200-o | Gumbel selfplay           |
| hexagonal_2p | lambda-gh200-o | Gumbel selfplay           |
| square19_4p  | lambda-gh200-o | Gumbel selfplay           |

### Known Cluster Issues

- `node_resilience.py` can kill P2P if `/status` times out - disabled on some nodes
- Tailscale connectivity intermittent - prefer public IPs when available
- GH200 nodes a, e, f retired - removed from voter list
- Export scripts require `PYTHONPATH=.` when running on cluster
