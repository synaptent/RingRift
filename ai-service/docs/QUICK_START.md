# Quick Start Guide

This guide will get you up and running with the RingRift AI service in minutes.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU training)
- 64GB+ RAM recommended for training
- SSH access to cluster nodes (for distributed training)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ringrift.git
cd ringrift/ai-service

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure

```
ai-service/
├── app/                    # Main application code
│   ├── ai/                 # AI implementations (MCTS, NNUE, Neural)
│   ├── config/             # Unified configuration
│   ├── coordination/       # Event routing, orchestration
│   ├── execution/          # Command execution framework
│   ├── storage/            # Storage backends (local, S3, GCS)
│   └── training/           # Training pipeline
├── scripts/                # Operational scripts
├── tests/                  # Test suite
├── data/                   # Data directory (games, models, Elo)
├── config/                 # YAML configuration files
└── docs/                   # Documentation
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_execution.py -v

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

## Configuration

Configuration is centralized in `app/config/unified_config.py`. See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for all options.

```python
from app.config.unified_config import get_config

config = get_config()
print(f"Training threshold: {config.training.trigger_threshold_games}")
```

Override via environment variables:

```bash
export RINGRIFT_TRAINING_THRESHOLD=1000
export RINGRIFT_ELO_DB=data/elo/custom.db
```

## Core Concepts

### 1. AI Players

The system supports multiple AI types:

```python
from app.ai import AIFactory, AIType, create_ai_from_difficulty
from app.models import AIConfig

# Difficulty-based AIs (recommended for gameplay)
random_ai = create_ai_from_difficulty(difficulty=1, player_number=1)
heuristic_ai = create_ai_from_difficulty(difficulty=2, player_number=2)

# Explicit AI type with custom config
config = AIConfig(difficulty=7, think_time=2000, randomness=0.1)
mcts_ai = AIFactory.create(AIType.MCTS, player_number=1, config=config)
```

### 2. Game Execution

Run games using the execution framework:

```python
from app.execution import run_quick_game, run_selfplay_batch

# Quick single game
result = run_quick_game(
    p1_type="mcts",
    p2_type="heuristic",
    board_type="square8",
)

# Parallel self-play batch
results = run_selfplay_batch(
    num_games=100,
    ai_type="mcts",
    difficulty=6,
    board_type="square8",
    num_players=2,
    max_workers=4,
)
```

### 3. Training Pipeline

Training is typically triggered automatically when enough games accumulate.

Manual training (CLI):

```bash
python -m app.training.train \
    --data-path data/training/square8_2p.npz \
    --board-type square8 \
    --epochs 50 \
    --batch-size 256 \
    --save-path models/square8_2p.pth
```

For programmatic training control, see `app/training/ORCHESTRATOR_GUIDE.md`.

### 4. Event System

The system uses an event-driven architecture:

```python
from app.coordination import StageEvent, get_event_router, publish_event_sync

router = get_event_router()

# Subscribe to events
def on_training_complete(event):
    print(f"Training finished: {event.payload.get('model_id')}")

router.subscribe(StageEvent.TRAINING_COMPLETE, on_training_complete)

# Emit events
publish_event_sync(StageEvent.TRAINING_COMPLETE, {"model_id": "v123"}, source="quick_start")
```

### 5. Model Promotion

Models are automatically promoted when they show improvement:

```python
from app.training.promotion_controller import PromotionType, get_promotion_controller

controller = get_promotion_controller()
decision = controller.evaluate_promotion(
    model_id="candidate_v5",
    board_type="square8",
    num_players=2,
    promotion_type=PromotionType.PRODUCTION,
    baseline_model_id="baseline_v4",
)

if decision.should_promote:
    controller.execute_promotion(decision)
```

## Common Tasks

### Run Self-Play Games

```bash
# Local self-play
python scripts/selfplay.py \
  --board square8 \
  --num-players 2 \
  --num-games 100 \
  --engine-mode heuristic

# GPU-accelerated self-play
python scripts/run_gpu_selfplay.py \
  --board square8 \
  --num-players 2 \
  --num-games 1000 \
  --batch-size 64
```

### Train a Model

```bash
# Export dataset from games
python scripts/export_replay_dataset.py --output data/training/dataset.npz

# Train neural network
python -m app.training.train \
    --data-path data/training/dataset.npz \
    --board-type square8 \
    --epochs 50 \
    --save-path models/new_model.pth
```

### Run Tournament

```bash
python scripts/run_tournament.py \
    --players heuristic mcts_100 neural \
    --games-per-pair 50 \
    --board square8
```

### Check Cluster Health

```bash
python scripts/cluster_health_check.py --verbose
```

## Distributed Training

### SSH Cluster Setup

1. Configure hosts in `config/distributed_hosts.yaml`
2. Set up SSH keys for passwordless access
3. Run cluster orchestrator:

```bash
python scripts/p2p_orchestrator.py --node-id mac-studio
```

### Slurm HPC Setup

1. Enable Slurm in config:

```yaml
slurm:
  enabled: true
  partition_training: gpu-train
```

2. Validate and smoke-test the Slurm setup:

```bash
python scripts/slurm_preflight_check.py --config config/unified_loop.yaml
python scripts/slurm_smoke_test.py --work-type training
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("app").setLevel(logging.DEBUG)
```

Or via environment:

```bash
export RINGRIFT_LOG_LEVEL=DEBUG
```

### Profile Performance

```python
from app.utils.profiling import profile_block

with profile_block("my_operation"):
    # Code to profile
    pass
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for coding patterns
- Read [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for all config options
- Explore [infrastructure/](infrastructure) for deployment guides
- Check [runbooks/](runbooks) for operational procedures
