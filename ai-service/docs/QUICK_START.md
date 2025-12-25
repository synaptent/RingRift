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
from app.ai import get_ai_player

# Create different AI players
random_ai = get_ai_player("random")
heuristic_ai = get_ai_player("heuristic")
mcts_ai = get_ai_player("mcts", simulations=1000)
neural_ai = get_ai_player("neural", model_path="models/best.pt")
```

### 2. Game Execution

Run games using the execution framework:

```python
from app.execution import GameExecutor, run_quick_game

# Quick single game
result = await run_quick_game(
    player1_type="mcts",
    player2_type="heuristic",
    board_type="square8",
    num_players=2
)

# Parallel game execution
executor = GameExecutor(board_type="square8", num_players=2)
results = await executor.run_batch(num_games=100)
```

### 3. Training Pipeline

Training is triggered automatically when enough games accumulate:

```python
from app.training import TrainingOrchestrator

orchestrator = TrainingOrchestrator()
await orchestrator.check_and_train()  # Trains if threshold met
```

Manual training:

```bash
python scripts/run_nn_training_baseline.py \
    --config square8_2p \
    --epochs 50 \
    --batch-size 256
```

### 4. Event System

The system uses an event-driven architecture:

```python
from app.coordination import get_event_router

router = get_event_router()

# Subscribe to events
@router.subscribe("training.completed")
async def on_training_complete(event):
    print(f"Training finished: {event.model_id}")

# Emit events
await router.emit("training.completed", {"model_id": "v123"})
```

### 5. Model Promotion

Models are automatically promoted when they show improvement:

```python
from app.training.promotion_controller import get_promotion_controller

controller = get_promotion_controller()
decision = await controller.evaluate_for_promotion(
    model_id="candidate_v5",
    games_played=100,
    elo_delta=35.0
)

if decision.should_promote:
    await controller.promote(decision)
```

## Common Tasks

### Run Self-Play Games

```bash
# Local self-play
python scripts/run_selfplay.py --games 100 --config square8_2p

# GPU-accelerated self-play
python scripts/run_gpu_selfplay.py --games 1000 --batch-size 64
```

### Train a Model

```bash
# Export dataset from games
python scripts/export_replay_dataset.py --output data/training/dataset.pt

# Train neural network
python scripts/run_nn_training_baseline.py \
    --dataset data/training/dataset.pt \
    --output models/new_model.pt
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
python scripts/cluster_orchestrator.py --mode production
```

### Slurm HPC Setup

1. Enable Slurm in config:

```yaml
slurm:
  enabled: true
  partition_training: gpu-train
```

2. Submit jobs:

```bash
python scripts/submit_slurm_job.py --type training
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
