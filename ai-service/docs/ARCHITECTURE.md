# Architecture

This document describes the high-level architecture of the RingRift AI Service.

## Overview

The AI service is a distributed system for training and serving game-playing AI at multiple difficulty levels. It supports:

- **Multi-algorithm AI** spanning 11 difficulty levels (Random → Neural Descent)
- **Distributed training** across GPU clusters
- **P2P coordination** for fault-tolerant data sync and job scheduling
- **Continuous evaluation** with Elo-based model promotion

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                        │
│              /ai/move  /ai/evaluate  /health  /metrics          │
├─────────────────────────────────────────────────────────────────┤
│                       AI Layer (app/ai/)                        │
│    Random │ Heuristic │ Minimax │ MCTS │ Descent │ Neural       │
├─────────────────────────────────────────────────────────────────┤
│                    Game Engine (app/rules/)                     │
│        Board Manager │ Phase Machine │ Capture Logic            │
├─────────────────────────────────────────────────────────────────┤
│                  Training Pipeline (app/training/)              │
│    Data Generation │ Loader │ Trainer │ Checkpoint │ Promotion  │
├─────────────────────────────────────────────────────────────────┤
│                 Coordination (app/coordination/)                │
│     Task Coordinator │ Orchestrator │ Resource Manager          │
├─────────────────────────────────────────────────────────────────┤
│               Distributed (app/distributed/)                    │
│       Data Sync │ P2P Client │ Host Discovery │ Health          │
└─────────────────────────────────────────────────────────────────┘
```

## Major Components

### 1. API Layer (`app/main.py`)

FastAPI service providing:

| Endpoint            | Purpose                           |
| ------------------- | --------------------------------- |
| `POST /ai/move`     | Get AI move for game state        |
| `POST /ai/evaluate` | Evaluate position for all players |
| `GET /health`       | Service health check              |
| `GET /metrics`      | Prometheus metrics                |

AI instances are cached with LRU eviction (512 max) to maintain persistent search tree state during games.

### 2. AI Implementations (`app/ai/`)

Multi-algorithm strategy supporting 11 difficulty levels:

| Level | Algorithm | Description                                      |
| ----- | --------- | ------------------------------------------------ |
| 1     | Random    | Pure random valid moves                          |
| 2     | Heuristic | 45+ CMA-ES optimized weighted evaluation factors |
| 3     | Minimax   | Alpha-beta search with heuristic evaluation      |
| 4     | Minimax   | Alpha-beta search with NNUE neural evaluation    |
| 5     | MCTS      | Monte Carlo Tree Search (heuristic rollouts)     |
| 6-8   | MCTS      | MCTS with neural value/policy guidance           |
| 9-10  | Descent   | AlphaZero-style UBFM search with neural guidance |
| 11    | Ultimate  | Extended Descent with 60s think time             |

Key modules:

- `factory.py` - AIFactory for difficulty → algorithm mapping
- `heuristic_ai.py` - Territory, capture, stability evaluation
- `mcts_ai.py` - UCB/RAVE tree search
- `descent_ai.py` - Upper Confidence Bound From Max
- `neural_net.py` - ResNet-style CNN architectures
- `nnue.py` - NNUE incremental evaluation

### 3. Game Engine (`app/rules/`, `app/game_engine.py`)

**Single Source of Truth Policy**: TypeScript shared engine (`src/shared/engine/`) is the canonical implementation. Python mirrors this behavior.

Supported boards:

- `square8` (8×8 = 64 cells) - Fast iteration
- `square19` (19×19 = 361 cells) - Go-like depth
- `hex8` (radius-4 = 61 cells) - Fast hex
- `hexagonal` (radius-12 = 469 cells) - Maximum complexity

### 4. Training Pipeline (`app/training/`)

```
Self-play → Data Ingestion → Training → Evaluation → Promotion
```

**Data Generation** (`generate_data.py`):

- Parallel self-play with configurable workers
- Zobrist hashing for deduplication
- Priority experience replay

**Training** (`train.py`, `distributed_unified.py`):

- PyTorch with DDP support
- Gradient compression, async SGD
- Mixed precision training
- Adaptive checkpointing

**Model Promotion** (`promotion_controller.py`):

- Staging gate: 50+ games, >52% win rate
- Production gate: 200+ games, min Elo improvement
- Automatic rollback on regression

### 5. Evaluation System (`app/tournament/`)

**Tournament Types**:

- Gauntlet: O(n) efficiency vs fixed baselines
- Round-robin: Full comparison matrix
- Distributed: Parallel evaluation across cluster

**Elo System** (`elo.py`, `unified_elo_db.py`):

- Glicko-2 variant with confidence intervals
- Cross-board Elo normalization
- Cluster-wide replication

### 6. Coordination Layer (`app/coordination/`)

**Task Coordinator** (`task_coordinator.py`):

- SQLite-backed task registry
- Rate limiting and backpressure
- Cross-process mutex

**Orchestrator Registry** (`orchestrator_registry.py`):

- Leader election (Bully algorithm)
- Heartbeat-based liveness
- Split-brain prevention

**Resource Management**:

- PID-controlled cluster scaling
- Per-host utilization targets
- Priority-based job scheduling

### 7. Distributed Infrastructure (`app/distributed/`)

**Data Sync** (multi-tier strategy):

1. SSH/rsync for local cluster
2. Aria2 for parallel downloads
3. HTTP/P2P fallback
4. WAL-based recovery

**Content Deduplication**:

- Game fingerprinting
- 10-30% storage reduction

**Host Discovery** (`hosts.py`):

- Remote memory detection via SSH
- Board-specific memory requirements
- GPU/CPU host classification

## Data Flows

### Self-Play → Training Loop

```
P2P Orchestrator
      │ schedules
      ▼
Selfplay Executor → Game Records
      │ async collection
      ▼
Game Collector → SQLite Database
      │ periodic sync
      ▼
Unified Data Sync → Distributed Hosts
      │ batches (500 games)
      ▼
Training Trigger → Training Loop
      │
      ▼
Model Checkpoint → Registry
```

### Evaluation → Promotion

```
Tournament Executor
      │ gauntlet vs baselines
      ▼
Elo Calculator
      │ update ratings
      ▼
Elo DB Replication
      │ sync across cluster
      ▼
Promotion Controller
      │ staging/production gates
      ▼
Model Promotion or Rollback
```

## Configuration

**Canonical Source**: `app/config/unified_config.py`

All thresholds and parameters are centralized:

```python
from app.config.unified_config import get_config, get_training_threshold

config = get_config()
threshold = get_training_threshold()  # 500 games default
```

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for usage patterns.

## External Integrations

| System     | Purpose                  |
| ---------- | ------------------------ |
| Vast.ai    | Cloud GPU scheduling     |
| Prometheus | Metrics collection       |
| PostgreSQL | Game database (optional) |
| Redis      | Queue backend (optional) |

## Architectural Patterns

1. **Single Responsibility** - Each module handles one concern
2. **Lazy Loading** - Torch imports deferred on inference-only nodes
3. **Fault Tolerance** - WAL recovery, circuit breakers, health checks
4. **Backpressure** - Queue monitors prevent overload
5. **Event-Driven** - Cross-system event bus for model promotion
6. **Configuration as Code** - Unified config eliminates scattered constants
7. **Distributed Consensus** - Leader election + heartbeats
8. **Content Addressing** - Manifest-driven sync with deduplication

## Directory Structure

```
ai-service/
├── app/                    # Core application code
│   ├── ai/                 # AI implementations
│   ├── config/             # Configuration (unified_config.py)
│   ├── coordination/       # Task coordination, orchestration
│   ├── core/               # Logging, utilities
│   ├── distributed/        # P2P, data sync, health
│   ├── rules/              # Game rules engine
│   ├── tournament/         # Evaluation, Elo system
│   └── training/           # Training pipeline
├── scripts/                # CLI tools and daemons
│   ├── unified_ai_loop.py  # Main training orchestrator
│   ├── p2p_orchestrator.py # P2P cluster coordinator
│   └── ...
├── config/                 # Deployment configuration
│   ├── distributed_hosts.yaml
│   └── unified_loop.yaml
├── models/                 # Neural network checkpoints
├── data/                   # Training data and databases
└── docs/                   # Documentation
```

## Scaling

The architecture scales from single-machine training to 20+ node distributed clusters:

- **Single machine**: Local SQLite, direct file access
- **Small cluster**: SSH-based sync, shared storage
- **Large cluster**: P2P coordination, distributed training, content-addressed sync

Leader election and heartbeats ensure cluster stability. WAL-based recovery handles node failures gracefully.
