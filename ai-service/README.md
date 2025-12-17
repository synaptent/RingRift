# RingRift AI Service

[![CI](https://github.com/RingRift/RingRift/actions/workflows/ci.yml/badge.svg)](https://github.com/RingRift/RingRift/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python-based AI microservice that powers the intelligent opponents in RingRift. From casual players learning the ropes to experts seeking a challenge, this service provides a 10-level difficulty ladder with AI ranging from random moves to neural network-guided search.

## What's Inside

- **10 Difficulty Levels** — Random (D1) → Heuristic (D2) → Minimax (D3-4) → MCTS (D5-8) → AlphaZero-style Descent (D9-10)
- **Neural Network Integration** — ResNet-style CNNs with policy/value heads for position evaluation
- **Distributed Training** — Self-play generation, Elo tracking, and model promotion across GPU clusters
- **P2P Orchestration** — Automatic cluster coordination with leader election and health monitoring
- **Apple Silicon Support** — MPS-compatible models for M1/M2/M3 training

## Quick Start

### Local Development

```bash
cd ai-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

The service will be available at:

- API: http://localhost:8001
- Interactive docs: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### With Docker

```bash
docker build -t ringrift-ai-service .
docker run -p 8001:8001 ringrift-ai-service
```

### Full Stack (with the main app)

From the project root:

```bash
docker compose up
```

This starts everything: the Node.js backend, AI service, PostgreSQL, Redis, and monitoring.

## API Overview

### Get an AI Move

```bash
curl -X POST http://localhost:8001/ai/move \
  -H "Content-Type: application/json" \
  -d '{
    "game_state": { /* GameState object */ },
    "player_number": 1,
    "difficulty": 5
  }'
```

### Evaluate a Position

```bash
curl -X POST http://localhost:8001/ai/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "game_state": { /* GameState object */ },
    "player_number": 1
  }'
```

### Health Check

```bash
curl http://localhost:8001/health
```

## The Difficulty Ladder

| Level | AI Type   | Description                             | Think Time |
| ----- | --------- | --------------------------------------- | ---------- |
| 1     | Random    | Random valid moves                      | 150ms      |
| 2     | Heuristic | 45+ weighted factors (CMA-ES optimized) | 200ms      |
| 3     | Minimax   | Alpha-beta search, heuristic eval       | 1.8s       |
| 4     | Minimax   | Alpha-beta + NNUE neural eval           | 2.8s       |
| 5     | MCTS      | Monte Carlo tree search                 | 4.0s       |
| 6-8   | MCTS      | MCTS + neural value/policy guidance     | 5.5-9.6s   |
| 9-10  | Descent   | AlphaZero-style UBFM search             | 12.6-16s   |

## Board Support

| Board       | Cells | Description                                |
| ----------- | ----- | ------------------------------------------ |
| `square8`   | 64    | 8×8 grid — Quick games, great for learning |
| `hex8`      | 61    | Radius-4 hex — Fast hex iteration          |
| `square19`  | 361   | 19×19 grid — Go-like strategic depth       |
| `hexagonal` | 469   | Radius-12 hex — Maximum complexity         |

## Project Structure

```
ai-service/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── game_engine.py       # Python rules engine (TS parity)
│   ├── ai/                  # AI implementations
│   │   ├── random_ai.py     # D1: Random moves
│   │   ├── heuristic_ai.py  # D2: Weighted evaluation
│   │   ├── minimax_ai.py    # D3-4: Alpha-beta search
│   │   ├── mcts_ai.py       # D5-8: Monte Carlo search
│   │   ├── descent_ai.py    # D9-10: UBFM/Descent
│   │   ├── nnue.py          # NNUE value network
│   │   ├── nnue_policy.py   # NNUE with policy head
│   │   └── neural_net.py    # CNN architectures
│   ├── training/            # Training pipeline
│   │   ├── training_triggers.py   # 3-signal training decisions
│   │   ├── rollback_manager.py    # Model regression handling
│   │   ├── model_registry.py      # Version lifecycle management
│   │   └── promotion_controller.py # A/B testing & promotion
│   ├── coordination/        # Distributed orchestration
│   ├── integration/         # Unified loop extensions
│   └── rules/               # Game rules (TS parity)
├── scripts/                 # CLI tools (380+)
│   ├── unified_ai_loop.py   # Main training daemon
│   ├── p2p_orchestrator.py  # Cluster coordination
│   ├── train_nnue_policy.py # NNUE policy training
│   └── run_gauntlet.py      # Tournament evaluation
├── config/                  # Configuration templates
├── models/                  # Neural network checkpoints
└── tests/                   # Test suite (1,824 tests)
```

## Training Your Own Models

### Single Machine

```bash
# Generate self-play games
python scripts/run_self_play_soak.py \
  --board-type square8 \
  --num-games 10000 \
  --output data/selfplay/

# Train a model
python -m app.training.train \
  --data data/selfplay/*.npz \
  --epochs 100 \
  --output models/my_model.pth
```

### Distributed Training

For multi-GPU training across a cluster:

1. **Configure your hosts:**

   ```bash
   cp config/distributed_hosts.yaml.example config/distributed_hosts.yaml
   # Edit with your actual server IPs
   ```

2. **Start the training loop:**

   ```bash
   python scripts/unified_ai_loop.py --start
   ```

3. **Monitor progress:**
   ```bash
   python scripts/unified_ai_loop.py --status
   ```

See [docs/UNIFIED_AI_LOOP.md](docs/UNIFIED_AI_LOOP.md) for the full pipeline documentation.

## Cluster Setup

The service includes a P2P orchestration system for distributed training across multiple GPU nodes.

### Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (H100, GH200, A100, or RTX 4090 recommended)
- SSH key-based authentication between nodes
- [Tailscale](https://tailscale.com/) (optional but recommended for easy mesh networking)

### Quick Cluster Start

```bash
# 1. Configure hosts
cp config/distributed_hosts.yaml.example config/distributed_hosts.yaml
vim config/distributed_hosts.yaml  # Add your server IPs

# 2. Deploy code to all nodes
python scripts/update_cluster_code.py --auto-stash

# 3. Start the P2P orchestrator
python scripts/p2p_orchestrator.py --node-id my-node

# 4. Check cluster status
curl http://localhost:8770/health
```

### Configuration Files

| File                            | Purpose                               |
| ------------------------------- | ------------------------------------- |
| `config/distributed_hosts.yaml` | Cluster host definitions (gitignored) |
| `config/remote_hosts.yaml`      | SSH host settings (gitignored)        |
| `config/sync_hosts.env`         | Data sync configuration (gitignored)  |

Use the `.example` templates to create your own configuration.

## Heuristic Training

The service includes tools for optimizing the heuristic evaluation weights using CMA-ES:

```bash
# Run CMA-ES optimization
python scripts/run_cmaes_optimization.py \
  --board square8 \
  --generations 20 \
  --population-size 32 \
  --games-per-eval 100

# Compare baseline vs trained weights
python scripts/run_heuristic_experiment.py \
  --mode baseline-vs-trained \
  --trained-profiles-a logs/cmaes/best_weights.json \
  --boards Square8 \
  --games-per-match 200
```

## Neural Network Architecture

The CNN models use a ResNet-style architecture:

- **Input**: Board state encoded as multi-channel tensor
- **Backbone**: 6-12 residual blocks with batch normalization
- **Policy Head**: Move probability distribution (~7K-92K outputs depending on board)
- **Value Head**: Win/Draw/Loss prediction or scalar evaluation

Variants:

- `RingRiftCNN_v4` — Standard CUDA architecture
- `RingRiftCNN_MPS` — Apple Silicon compatible
- `HexNeuralNet_v3` — Optimized for hexagonal geometry

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Quick health check
curl http://localhost:8001/health
```

## Environment Variables

| Variable         | Description                                | Default               |
| ---------------- | ------------------------------------------ | --------------------- |
| `PYTHON_ENV`     | Environment mode                           | development           |
| `LOG_LEVEL`      | Logging verbosity                          | INFO                  |
| `AI_SERVICE_URL` | URL for the main app to reach this service | http://localhost:8001 |

## Resource Limits

The service enforces 80% max utilization to prevent system overload:

| Resource | Warning | Critical |
| -------- | ------- | -------- |
| CPU      | 70%     | 80%      |
| GPU      | 70%     | 80%      |
| Memory   | 70%     | 80%      |
| Disk     | 65%     | 70%      |

## Documentation

| Document                                               | Description                  |
| ------------------------------------------------------ | ---------------------------- |
| [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md) | System architecture guide    |
| [Unified AI Loop](docs/UNIFIED_AI_LOOP.md)             | Main training pipeline       |
| [NNUE Policy Training](docs/NNUE_POLICY_TRAINING.md)   | Policy training with KL loss |
| [Example Training Run](docs/EXAMPLE_TRAINING_RUN.md)   | Step-by-step tutorial        |
| [Cluster Setup Guide](docs/CLUSTER_SETUP_GUIDE.md)     | Multi-GPU deployment         |
| [AI Training Plan](docs/AI_TRAINING_PLAN.md)           | Training methodology         |
| [Game Record Spec](docs/GAME_RECORD_SPEC.md)           | Data format specification    |
| [P2P Orchestration](docs/VAST_P2P_ORCHESTRATION.md)    | Cluster coordination         |

## Troubleshooting

### Service won't start

- Check Python version: `python --version` (requires 3.11+)
- Verify dependencies: `pip install -r requirements.txt`
- Check if port 8001 is in use: `lsof -i :8001`

### Connection refused from main app

- Ensure the service is running
- Check `AI_SERVICE_URL` environment variable
- Verify Docker network if using containers

### Slow AI responses

- Higher difficulties take longer (D10 = 16 seconds)
- Check CPU/GPU resources
- Consider running on GPU for neural network evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Part of the RingRift project. See the main project LICENSE for details.
