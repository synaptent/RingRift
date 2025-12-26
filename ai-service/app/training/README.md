# Training Module

The `app/training/` module contains RingRift's complete neural network training pipeline, from selfplay data generation to model deployment. This README provides an overview of the pipeline architecture, key entry points, and common workflows.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Training Pipeline Flow](#training-pipeline-flow)
- [Main Entry Points](#main-entry-points)
- [Selfplay Data Generation](#selfplay-data-generation)
- [Data Export and Quality](#data-export-and-quality)
- [Model Management](#model-management)
- [Configuration](#configuration)
- [Common Workflows](#common-workflows)
- [Advanced Features](#advanced-features)

## Architecture Overview

The training module consists of 146 Python files organized into functional areas:

```
app/training/
├── Core Training
│   ├── train.py                    # Main training entry point (CLI)
│   ├── train_loop.py               # Iterative selfplay → train → evaluate loop
│   ├── model_factory.py            # Model instantiation and initialization
│   ├── config.py                   # Training configuration and presets
│   └── train_setup.py              # Training preparation utilities
│
├── Selfplay Generation
│   ├── selfplay_runner.py          # Base class for all selfplay variants
│   ├── selfplay_config.py          # Unified selfplay configuration
│   ├── gpu_mcts_selfplay.py        # GPU-accelerated Gumbel MCTS
│   ├── background_selfplay.py      # Continuous selfplay daemon
│   └── temperature_scheduling.py   # Exploration temperature control
│
├── Data Pipeline
│   ├── export_core.py              # Core export logic (DB → NPZ)
│   ├── export_cache.py             # Incremental export with caching
│   ├── data_quality.py             # Quality validation tools
│   ├── data_validation.py          # Schema and integrity checks
│   ├── data_loader.py              # NPZ/DB data loading (76KB)
│   └── data_augmentation.py        # Symmetry-based augmentation
│
├── Model Management
│   ├── unified_model_store.py      # Centralized model registry
│   ├── checkpoint_unified.py       # Unified checkpoint manager
│   ├── model_registry.py           # Model lifecycle tracking
│   └── model_versioning.py         # Architecture version control
│
├── Orchestration
│   ├── unified_orchestrator.py     # Low-level training execution
│   ├── orchestrated_training.py    # High-level service coordination
│   ├── optimization_orchestrator.py # Hyperparameter optimization
│   └── per_orchestrator.py         # Prioritized experience replay
│
├── Evaluation
│   ├── tournament.py               # Round-robin tournaments
│   ├── game_gauntlet.py            # Model vs baseline evaluation
│   ├── elo_service.py              # Elo rating system (56KB)
│   └── background_eval.py          # Async evaluation service
│
└── Advanced Features
    ├── curriculum.py               # Curriculum learning
    ├── distillation.py             # Knowledge distillation
    ├── pbt.py                      # Population-based training
    ├── distributed.py              # Multi-GPU/multi-node training
    ├── online_learning.py          # EBMO online learning
    ├── gradient_surgery.py         # PCGrad multi-task learning
    ├── auxiliary_tasks.py          # Auxiliary training objectives
    └── adversarial_positions.py    # Hard position mining
```

## Quick Start

### 1. Train a Model from Scratch

```bash
cd ai-service

# Export training data from games
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Train the model
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz \
  --model-version v3 \
  --batch-size 512 --epochs 20
```

### 2. Run Selfplay to Generate Training Data

```bash
# Quick heuristic selfplay (fast bootstrap)
python scripts/selfplay.py \
  --board square8 --num-players 2 \
  --engine heuristic --num-games 1000

# High-quality Gumbel MCTS selfplay
python scripts/selfplay.py \
  --board hex8 --num-players 2 \
  --engine gumbel --num-games 500 \
  --batch-size 64 --use-gpu
```

### 3. Automated Training Loop

```bash
# One-command pipeline: selfplay → export → train → evaluate
python scripts/run_training_loop.py \
  --board-type hex8 --num-players 2 \
  --selfplay-games 1000 \
  --training-epochs 50 \
  --auto-promote
```

## Training Pipeline Flow

The complete training pipeline consists of five stages:

```
┌──────────┐      ┌─────────┐      ┌────────┐      ┌───────┐      ┌──────────┐
│ Selfplay │  →   │  Sync/  │  →   │ Export │  →   │ Train │  →   │ Evaluate │
│          │      │ Consol. │      │ to NPZ │      │ Model │      │ & Promote│
└──────────┘      └─────────┘      └────────┘      └───────┘      └──────────┘
     ↑                                                                    │
     └────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Selfplay Generation

Generate game data using various AI engines:

**Engine Modes:**

| Engine        | Speed     | Quality  | Use Case                 |
| ------------- | --------- | -------- | ------------------------ |
| `heuristic`   | Very Fast | Low      | Bootstrap initial data   |
| `random`      | Very Fast | Very Low | Baseline comparisons     |
| `policy-only` | Fast      | Medium   | Quick iterations         |
| `mcts`        | Medium    | Medium   | Balanced approach        |
| `gumbel`      | Slow      | High     | Production training data |
| `nnue-guided` | Medium    | High     | Hybrid NN + search       |
| `mixed`       | Variable  | High     | Engine diversity         |

**Output:** SQLite databases (`data/games/*.db`) with game replays

### Stage 2: Data Export

Convert game replays into neural network training format:

```python
# Export process:
# GameReplayDB (SQLite) → Feature Encoding → NPZ Arrays

# NPZ structure:
{
    "features": (N, C, H, W),      # Board state tensors
    "globals": (N, G),              # Global features (turn, player)
    "values": (N,),                 # Outcome labels (+1 win, -1 loss)
    "policy_indices": (N,),         # Sparse policy targets (object array)
    "policy_values": (N,),          # Policy probabilities (object array)
}
```

**Key Tools:**

- `scripts/export_replay_dataset.py` - Main export CLI
- `app/training/export_core.py` - Core export logic
- `app/training/export_cache.py` - Incremental caching

### Stage 3: Training

Train neural networks using PyTorch with advanced features:

**Model Architectures:**

- **v2**: 96 channels, 6 residual blocks (legacy)
- **v3**: 192 channels, 12 residual blocks, SE attention (current)

**Training Features:**

- Multi-task learning (joint policy + value)
- Gradient surgery (PCGrad to prevent task interference)
- Adaptive learning rates (cosine annealing, warmup)
- Early stopping (loss-based or Elo-based)
- GPU optimization (auto-scaling batch sizes, mixed precision)

**Key Files:**

- `train.py` - Main training script (CLI)
- `train_loop.py` - Iterative training loop
- `model_factory.py` - Model creation

### Stage 4: Evaluation

Assess model strength using gauntlet tournaments:

```bash
# Gauntlet: Play candidate model vs baselines
python scripts/quick_gauntlet.py \
  --model models/hex8_2p_candidate.pt \
  --board-type hex8 --num-players 2 \
  --games 100
```

**Baselines:**

- **RANDOM**: Random move selection
- **HEURISTIC**: Rule-based heuristic AI
- **PRODUCTION**: Use Elo tournaments for full model-vs-model comparisons

**Promotion Thresholds:**

- vs RANDOM: 85%+ win rate required
- vs HEURISTIC: 60%+ win rate required
- vs PREVIOUS: 55%+ win rate required

### Stage 5: Model Promotion

Successful models advance through lifecycle stages:

```
Development → Staging → Production
```

Use `scripts/auto_promote.py` for automated promotion based on gauntlet results.

## Main Entry Points

### Training Scripts

#### `train.py` - Main Training CLI

Primary entry point for neural network training:

```bash
python -m app.training.train \
  --board-type square8 \
  --num-players 2 \
  --data-path data/training/sq8_2p.npz \
  --model-version v3 \
  --batch-size 512 \
  --epochs 50 \
  --learning-rate 0.001 \
  --save-path models/sq8_2p_new.pth
```

**Key Arguments:**

- `--board-type`: Board geometry (square8, square19, hex8, hexagonal)
- `--num-players`: Player count (2, 3, 4)
- `--data-path`: Path to NPZ training file
- `--model-version`: Architecture (v2, v3)
- `--init-weights`: Load pretrained weights (optional)
- `--early-stopping`: Enable early stopping
- `--lr-scheduler`: LR schedule (none, step, cosine)

**Features:**

- Automatic GPU batch size scaling
- Learning rate scheduling with warmup
- Early stopping (loss or Elo-based)
- Gradient accumulation for large batches
- Multi-task gradient surgery (PCGrad)
- Entropy regularization for policy diversity

#### `train_loop.py` - Iterative Training Loop

Runs multiple cycles of selfplay → train → evaluate:

```python
from app.training.train_loop import run_training_loop
from app.training.config import TrainConfig
from app.models import BoardType

config = TrainConfig(
    board_type=BoardType.HEX8,
    iterations=10,           # 10 training cycles
    episodes_per_iter=500,   # 500 games per cycle
    epochs_per_iter=20,      # 20 epochs per cycle
    use_gpu_parallel_datagen=True,
)

run_training_loop(config)
```

**Use Cases:**

- Bootstrap training from scratch
- Continuous improvement loops
- AlphaZero-style self-improvement

### Selfplay Scripts

#### `scripts/selfplay.py` - Unified Selfplay CLI

Single entry point for all selfplay variants:

```bash
python scripts/selfplay.py \
  --board hex8 \
  --num-players 2 \
  --engine gumbel \
  --num-games 1000 \
  --batch-size 64 \
  --use-gpu \
  --output-dir data/games/selfplay_hex8_2p
```

**Engine Modes:**

- `heuristic` - Fast rule-based (bootstrap)
- `gumbel` - Gumbel MCTS (production quality)
- `mcts` - Standard MCTS
- `nnue-guided` - Neural network guided search
- `policy-only` - Direct policy sampling
- `mixed` - Multiple engines for diversity
- `random` - Random baseline

**Output Formats:**

- `--format db` - SQLite database (default)
- `--format jsonl` - JSON Lines file
- `--format npz` - Direct NPZ export

#### Programmatic Selfplay

```python
from app.training.selfplay_runner import (
    HeuristicSelfplayRunner,
    GumbelMCTSSelfplayRunner,
)
from app.training.selfplay_config import SelfplayConfig

# Quick heuristic selfplay
config = SelfplayConfig(
    board_type="square8",
    num_players=2,
    engine_mode="heuristic",
    num_games=100,
)
runner = HeuristicSelfplayRunner(config)
runner.run()

# GPU Gumbel MCTS
config = SelfplayConfig(
    board_type="hex8",
    num_players=2,
    engine_mode="gumbel",
    num_games=500,
    use_gpu=True,
    batch_size=64,
)
runner = GumbelMCTSSelfplayRunner(config)
runner.run()
```

### Data Export Scripts

#### `scripts/export_replay_dataset.py`

Convert game databases to NPZ training format:

```bash
# Basic export
python scripts/export_replay_dataset.py \
  --db data/games/selfplay_hex8_2p.db \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Auto-discover all databases
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Quality filtering
python scripts/export_replay_dataset.py \
  --db data/games/selfplay_hex8_2p.db \
  --board-type hex8 --num-players 2 \
  --require-completed \
  --min-moves 20 \
  --max-moves 500 \
  --output data/training/hex8_2p_filtered.npz

# Incremental export with caching
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 --num-players 2 \
  --use-cache \
  --output data/training/hex8_2p.npz
```

**Features:**

- Parallel processing (multi-core)
- Incremental caching (skip unchanged DBs)
- Quality filtering (completed games, move limits)
- Rank-aware value targets (multiplayer)
- Auto-discovery via `GameDiscovery`

## Selfplay Data Generation

### Gumbel MCTS Selfplay

The recommended engine for production training data:

```python
from app.ai.gumbel_common import (
    GUMBEL_BUDGET_THROUGHPUT,   # 64 simulations
    GUMBEL_BUDGET_STANDARD,     # 150 simulations
    GUMBEL_BUDGET_QUALITY,      # 800 simulations
    GUMBEL_BUDGET_ULTIMATE,     # 1600 simulations
)

config = SelfplayConfig(
    engine_mode="gumbel",
    mcts_simulations=GUMBEL_BUDGET_QUALITY,  # High quality
)
```

**Features:**

- Sequential halving for move selection
- Gumbel noise for exploration
- GPU-accelerated parallel games
- Value and policy targets from tree statistics

### Temperature Scheduling

Control exploration vs exploitation during selfplay:

```python
from app.training.temperature_scheduling import create_scheduler

# Available schedules:
# - alphazero: τ=1 → τ=0 at move N
# - aggressive_exploration: High τ throughout
# - conservative: Low τ from start
# - adaptive: Based on position complexity
# - curriculum: Based on training progress
# - cosine: Smooth annealing

scheduler = create_scheduler("adaptive")
temp = scheduler.get_temperature(move_number=15, game_state=state)
```

### Background Selfplay

Run continuous selfplay as a daemon:

```bash
python -m app.training.background_selfplay \
  --board-type hex8 --num-players 2 \
  --engine gumbel \
  --games-per-batch 100 \
  --daemon
```

## Data Export and Quality

### Feature Encoding

Neural network inputs are encoded as multi-channel tensors:

**Board State Channels** (per player):

- Stack heights (0-4 pieces)
- Control (which player owns the stack)
- Influence (threatened cells)
- Recent moves (move history)

**Global Features:**

- Current turn number
- Player to move
- Forced elimination status (v2+)
- Chain formation for hex boards (v2+)

**Versions:**

- `v1`: Base features (legacy)
- `v2`: Added chain/forced-elimination signals (current)

### Value Target Encoding

Outcome labels are rank-aware for multiplayer:

```python
# 2-player: Binary outcome
winner: +1.0
loser:  -1.0

# 3-player: Ordinal ranks
1st: +1.0
2nd:  0.0
3rd: -1.0

# 4-player: Linear interpolation
1st: +1.00
2nd: +0.33
3rd: -0.33
4th: -1.00
```

### Quality Validation

Check data quality before training:

```bash
# Validate database
python -m app.training.data_quality --db data/games/selfplay.db

# Validate NPZ with details
python -m app.training.data_quality \
  --npz data/training/hex8_2p.npz \
  --detailed

# Scan all discovered databases
python -m app.training.data_quality --all
```

**Programmatic Usage:**

```python
from app.training.data_quality import (
    DatabaseQualityChecker,
    TrainingDataValidator,
)

# Check database
checker = DatabaseQualityChecker()
score = checker.get_quality_score("data/games/selfplay.db")

# Validate NPZ
validator = TrainingDataValidator()
if validator.validate_npz_file("data/training/batch.npz"):
    stats = validator.check_feature_distribution("data/training/batch.npz")
```

## Model Management

### Unified Model Store

Central registry for model lifecycle management:

```python
from app.training.unified_model_store import get_model_store

store = get_model_store()

# Register new model
model_id, version = store.register(
    name="square8_2p_v42",
    model_path="models/trained.pt",
    elo=1650,
)

# Get production model
model = store.get_production("square8_2p")

# Promote through stages
store.promote(model_id, "staging")
store.promote(model_id, "production")
```

### Checkpoint Management

Unified checkpoint system with comprehensive metadata:

```python
from app.training.checkpoint_unified import (
    UnifiedCheckpointManager,
    CheckpointType,
)

manager = UnifiedCheckpointManager(
    checkpoint_dir="data/checkpoints",
    max_checkpoints=5,
)

# Save checkpoint
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    global_step=5000,
    metrics={"loss": 0.45, "elo": 1650},
    checkpoint_type=CheckpointType.BEST,
)

# Resume from latest
state = manager.load_latest_checkpoint(model, optimizer)
```

**Features:**

- Multiple checkpoint types (regular, epoch, best, emergency)
- File integrity verification (SHA256 hashing)
- Adaptive checkpoint frequency
- Automatic cleanup of old checkpoints
- Full training state recovery

### Model Versioning

**v2 Models** (Legacy):

- 96 channels, 6 residual blocks
- Used for square19 and early models

**v3 Models** (Current):

- 192 channels, 12 residual blocks
- SE (Squeeze-and-Excitation) attention
- Position-aware policy encoding
- Spatial policy heads for hex boards

```python
from app.training.config import get_model_version_for_board

version = get_model_version_for_board(BoardType.HEX8)  # Returns "v3"
```

## Configuration

### TrainConfig

Core training hyperparameters:

```python
from app.training.config import TrainConfig, get_training_config_for_board

# Board-specific presets
config = get_training_config_for_board(BoardType.HEX8)

# Manual configuration
config = TrainConfig(
    board_type=BoardType.SQUARE8,
    batch_size=512,              # Auto-scaled for GPU
    learning_rate=0.002,
    epochs_per_iter=20,

    # Learning rate schedule
    lr_scheduler="cosine",
    warmup_epochs=1,
    lr_min=1e-6,

    # Early stopping
    early_stopping_patience=5,
    elo_early_stopping_patience=10,
    elo_min_improvement=5.0,

    # Regularization
    entropy_weight=0.01,
    weight_decay=1e-4,
    policy_label_smoothing=0.0,

    # Multi-task learning
    enable_gradient_surgery=True,  # PCGrad

    # GPU optimization
    gradient_accumulation_steps=1,
)
```

### SelfplayConfig

Unified selfplay settings:

```python
from app.training.selfplay_config import SelfplayConfig

config = SelfplayConfig(
    board_type="hex8",
    num_players=2,
    engine_mode="gumbel",
    mcts_simulations=800,
    num_games=1000,
    batch_size=64,
    use_gpu=True,
)
```

### GPU Scaling

Automatic batch size scaling:

```python
from app.training.config import GpuScalingConfig

config = GpuScalingConfig(
    gh200_batch_multiplier=64,    # 64 * 64 = 4096 batch
    h100_batch_multiplier=32,     # 64 * 32 = 2048 batch
    a100_batch_multiplier=16,     # 64 * 16 = 1024 batch
    reserved_memory_gb=8.0,
)
```

**Environment Variables:**

```bash
export RINGRIFT_DISABLE_GPU_DATAGEN=1  # Disable GPU selfplay
export RINGRIFT_AUTO_BATCH_SCALE=0     # Disable batch auto-scaling
```

## Common Workflows

### Workflow 1: Bootstrap from Scratch

```bash
# 1. Generate initial heuristic data
python scripts/selfplay.py \
  --board square8 --num-players 2 \
  --engine heuristic --num-games 5000

# 2. Export to NPZ
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type square8 --num-players 2 \
  --output data/training/sq8_2p_bootstrap.npz

# 3. Train initial model
python -m app.training.train \
  --board-type square8 --num-players 2 \
  --data-path data/training/sq8_2p_bootstrap.npz \
  --epochs 50 --save-path models/sq8_2p_v1.pt

# 4. Generate higher-quality selfplay
python scripts/selfplay.py \
  --board square8 --num-players 2 \
  --engine gumbel --num-games 2000

# 5. Re-train with improved data
python -m app.training.train \
  --board-type square8 --num-players 2 \
  --data-path data/training/sq8_2p_v2.npz \
  --init-weights models/sq8_2p_v1.pt \
  --epochs 50 --save-path models/sq8_2p_v2.pt
```

### Workflow 2: Transfer Learning (2p → 4p)

```bash
# 1. Resize value head
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_sq8_2p.pth \
  --output models/sq8_4p_init.pth \
  --board-type square8

# 2. Generate 4-player data
python scripts/selfplay.py \
  --board square8 --num-players 4 \
  --engine heuristic --num-games 3000

# 3. Fine-tune
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/sq8_4p_init.pth \
  --learning-rate 0.0005 \
  --epochs 30
```

### Workflow 3: Continuous Improvement

```bash
# Automated loop
python scripts/run_training_loop.py \
  --board-type hex8 --num-players 2 \
  --selfplay-games 1000 \
  --training-epochs 50 \
  --auto-promote
```

### Workflow 4: Model Promotion

```bash
# 1. Train candidate
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --save-path models/hex8_2p_candidate.pt

# 2. Evaluate
python scripts/quick_gauntlet.py \
  --model models/hex8_2p_candidate.pt \
  --board-type hex8 --num-players 2

# 3. Auto-promote
python scripts/auto_promote.py --gauntlet \
  --model models/hex8_2p_candidate.pt \
  --board-type hex8 --num-players 2
```

## Advanced Features

### Curriculum Learning

```python
from app.training.curriculum import CurriculumManager

manager = CurriculumManager(
    initial_difficulty=0.3,
    target_difficulty=1.0,
)
```

### Knowledge Distillation

```bash
python -m app.training.distillation \
  --teacher-model models/hex8_2p_large.pt \
  --student-arch small \
  --temperature 2.0
```

### Population-Based Training

```bash
python -m app.training.pbt \
  --board-type square8 --num-players 2 \
  --population-size 8 \
  --generations 10
```

### Online Learning (EBMO)

```python
from app.training.online_learning import create_online_learner

learner = create_online_learner(model, learner_type="ebmo")

# During game
learner.record_transition(state, move, player, next_state)
learner.update_from_game(winner)
```

## Related Documentation

- [Root CLAUDE.md](../../../CLAUDE.md) - Project overview
- [AI Service CLAUDE.md](../../CLAUDE.md) - AI service context
- [Cluster Monitor](../distributed/README_cluster_monitor.md) - Cluster management

---

**Last Updated**: December 2025
**Module Files**: 146 Python files
**Key Maintainer**: RingRift AI Training Team
