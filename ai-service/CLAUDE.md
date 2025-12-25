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
python -m app.gauntlet.runner \
  --board-type hex8 --num-players 2 \
  --model-path models/my_model.pth
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

- **`event_router.py`**: Unified event bus bridging in-memory, stage, and cross-process events
- **`pipeline_actions.py`**: Stage invokers with circuit breaker protection
- **`data_pipeline_orchestrator.py`**: Tracks and triggers pipeline stages
- **`daemon_manager.py`**: Lifecycle management for all daemons
- **`daemon_adapters.py`**: Wrappers for existing daemons (sync, promotion, distillation)
- **`sync_bandwidth.py`**: Bandwidth-coordinated rsync with host-level limits
- **`auto_sync_daemon.py`**: Automated P2P data sync with push-from-generator + gossip replication

```bash
# Launch all daemons under unified management
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
│   ├── coordination/    # Training pipeline orchestration
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
├── config/              # Configuration files (templates provided)
├── data/
│   ├── games/           # Game databases
│   ├── training/        # NPZ training files
│   └── models/          # Trained model checkpoints
├── models/              # Production models by config
├── scripts/             # CLI tools and utilities
└── tests/               # Test suite
```

## Cluster Infrastructure (Optional)

For distributed training across multiple GPU nodes, configure `config/distributed_hosts.yaml`. See `config/distributed_hosts.template.yaml` for the format.

The P2P mesh network supports:

- Automatic leader election
- Data synchronization across nodes
- Job distribution and monitoring

```bash
# Check cluster status (when P2P daemon is running)
python -m app.distributed.cluster_monitor

# Watch mode (live updates)
python -m app.distributed.cluster_monitor --watch --interval 10
```

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
