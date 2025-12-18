# Architecture Overview

This document provides a high-level overview of the RingRift AI Service architecture, including key components, data flows, and integration points.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RingRift AI Service                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   FastAPI    │    │   AI Engine  │    │   Training   │                   │
│  │   Endpoints  │───▶│   (D1-D11)   │◀───│   Pipeline   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    Rules     │    │    Model     │    │   Selfplay   │                   │
│  │   Engine     │    │   Registry   │    │   System     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             ▼                                                │
│                    ┌──────────────┐                                          │
│                    │ P2P Cluster  │                                          │
│                    │ Orchestrator │                                          │
│                    └──────────────┘                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer (`app/main.py`)

FastAPI endpoints serving AI move requests and game evaluation:

| Endpoint            | Purpose                    |
| ------------------- | -------------------------- |
| `POST /ai/move`     | Get AI move for game state |
| `POST /ai/evaluate` | Evaluate position          |
| `GET /health`       | Service health check       |
| `GET /metrics`      | Prometheus metrics         |

### 2. AI Engine (`app/ai/`)

11-level difficulty ladder with pluggable AI implementations:

| Level | AI Type   | Implementation                                      |
| ----- | --------- | --------------------------------------------------- |
| D1    | Random    | `random_ai.py`                                      |
| D2    | Heuristic | `heuristic_ai.py` (45+ CMA-ES optimized weights)    |
| D3    | Minimax   | `minimax_ai.py` (alpha-beta, heuristic eval)        |
| D4    | Minimax   | `minimax_ai.py` (alpha-beta + NNUE neural eval)     |
| D5    | MCTS      | `mcts_ai.py` (heuristic rollouts)                   |
| D6-8  | MCTS      | `mcts_ai.py` (neural policy/value guidance)         |
| D9-10 | Descent   | `descent_ai.py` (AlphaZero-style UBFM search)       |
| D11   | Ultimate  | `descent_ai.py` (60s think time, nearly unbeatable) |

**Key AI Modules:**

- `nnue.py` - NNUE value network
- `nnue_policy.py` - NNUE with policy head
- `mcts_gamestate_adapter.py` - Game state → MCTS interface bridge
- `ensemble_inference.py` - Multi-model evaluation
- `gpu_parallel_games.py` - GPU batch processing

### 3. Training Pipeline (`app/training/`)

End-to-end training infrastructure:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Selfplay  │───▶│   Training  │───▶│  Evaluation │───▶│  Promotion  │
│   System    │    │   Triggers  │    │  Tournament │    │  Controller │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  games.db         model.pt           elo_ratings       production/
```

**Key Training Modules:**

| Module                    | Purpose                                |
| ------------------------- | -------------------------------------- |
| `training_triggers.py`    | 3-signal training decision system      |
| `rollback_manager.py`     | Model regression detection & rollback  |
| `model_registry.py`       | Version tracking, lifecycle management |
| `promotion_controller.py` | A/B testing, auto-promotion            |
| `feedback_accelerator.py` | Training intensity optimization        |

### 4. Model Registry (`app/training/model_registry.py`)

Centralized model version management with lifecycle stages:

```
DEVELOPMENT ──▶ STAGING ──▶ PRODUCTION ──▶ ARCHIVED
      │              │              │
      └── REJECTED   └── ROLLBACK ──┘
```

**Features:**

- SQLite-backed persistence
- SHA256 checksum validation
- Distributed sync across cluster
- Automatic promotion with Elo gates

### 5. Rules Engine (`app/rules/`)

Python implementation maintaining TypeScript parity:

| Module                  | Purpose                             |
| ----------------------- | ----------------------------------- |
| `game_state.py`         | Immutable game state representation |
| `game_engine.py`        | Move application, phase handling    |
| `board_manager.py`      | Board topology, adjacency           |
| `territory.py`          | Territory calculation               |
| `forced_elimination.py` | Elimination rules                   |

### 6. P2P Orchestrator (`scripts/p2p_orchestrator.py`)

Distributed cluster coordination with leader election:

**Endpoint Categories:**

- `/heartbeat`, `/election` - Leadership coordination
- `/training/*` - Distributed training management
- `/sync/*` - Data/model synchronization
- `/tournament/*` - Elo tournament execution
- `/admin/*` - Cluster administration

## Data Flow

### Training Loop

```
1. Selfplay Generation
   ├── GPU Selfplay (parallel games)
   └── MCTS Selfplay (with policy tracking)
            │
            ▼
2. Data Aggregation
   ├── JSONL → SQLite conversion
   └── MCTS policy reanalysis (optional)
            │
            ▼
3. Training Trigger Evaluation
   ├── Data freshness (games since last train)
   ├── Model staleness (time since train)
   └── Performance regression (Elo/win rate)
            │
            ▼
4. Model Training
   ├── NNUE value training
   └── NNUE policy training (KL loss)
            │
            ▼
5. Evaluation
   ├── Elo tournament vs baselines
   └── A/B testing vs production
            │
            ▼
6. Promotion/Rollback
   ├── Auto-promote if +20 Elo
   └── Rollback if -50 Elo
```

### API Request Flow

```
Client Request
      │
      ▼
┌─────────────┐
│  FastAPI    │──▶ Validate Request
└─────────────┘
      │
      ▼
┌─────────────┐
│ Difficulty  │──▶ Select AI (D1-D11)
│   Ladder    │
└─────────────┘
      │
      ▼
┌─────────────┐
│ AI Instance │──▶ Load model (if neural)
└─────────────┘    Apply game state
      │            Run search
      ▼
┌─────────────┐
│   Return    │──▶ Move + metrics
└─────────────┘
```

## Key Integration Points

### Training Triggers (3-Signal System)

```python
# app/training/training_triggers.py
class TrainingTriggers:
    signals:
        - data_freshness: games_since_train / threshold (default: 500)
        - model_staleness: hours_since_train / threshold (default: 6h)
        - performance_regression: win_rate_drop * weight (1.5x)

    priority = freshness*1.0 + staleness*0.8 + regression*1.5
```

### Model Lifecycle

```python
# app/training/model_registry.py
class ModelRegistry:
    lifecycle:
        DEVELOPMENT  # Initial training
        STAGING      # Evaluation phase
        PRODUCTION   # Active deployment
        ARCHIVED     # Historical backup

    promotion_criteria:
        - elo_improvement >= 25
        - games_played >= 50
        - win_rate >= 52%
```

### Rollback Detection

```python
# app/training/rollback_manager.py
class RollbackManager:
    triggers:
        - elo_drop >= 50 points
        - win_rate_drop >= 10%
        - error_rate >= 5%

    min_evaluation_games: 50
```

## Configuration

### Key Config Files

| File                            | Purpose                    |
| ------------------------------- | -------------------------- |
| `config/unified_loop.yaml`      | Training pipeline settings |
| `config/distributed_hosts.yaml` | Cluster node definitions   |
| `config/sync_hosts.env`         | Data sync configuration    |
| `config/ladder_config.py`       | AI difficulty settings     |

### Environment Variables

| Variable         | Purpose           | Default               |
| ---------------- | ----------------- | --------------------- |
| `PYTHON_ENV`     | Environment mode  | development           |
| `LOG_LEVEL`      | Logging verbosity | INFO                  |
| `AI_SERVICE_URL` | Service endpoint  | http://localhost:8001 |

## Monitoring

### Prometheus Metrics

- `ringrift_ai_requests_total` - API request count
- `ringrift_ai_latency_seconds` - Response time histogram
- `ringrift_model_rollbacks_total` - Rollback events
- `ringrift_training_games_total` - Training data volume

### Health Endpoints

- `/health` - Basic liveness check
- `/health/deep` - Full component status
- `/metrics` - Prometheus scrape endpoint

## Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      P2P Cluster Network                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ GH200-a  │    │ GH200-e  │    │ GH200-f  │    │ GH200-g  │  │
│  │ Leader   │◀──▶│ Selfplay │◀──▶│ Training │◀──▶│ Training │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                                         │
│       │              Tailscale Mesh Network                     │
│       ▼                                                         │
│  ┌──────────┐                                                   │
│  │ Vast.ai  │  Dynamic GPU instances                            │
│  │ Nodes    │  (auto-provisioned)                               │
│  └──────────┘                                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Node Roles

| Role           | Responsibilities                               |
| -------------- | ---------------------------------------------- |
| **Leader**     | Coordination, job dispatch, sync orchestration |
| **Selfplay**   | Game generation, data production               |
| **Training**   | Model training, evaluation                     |
| **Evaluation** | Elo tournaments, A/B testing                   |

## See Also

- [Training Pipeline](../training/TRAINING_PIPELINE.md) - Detailed training docs
- [NNUE Policy Training](../algorithms/NNUE_POLICY_TRAINING.md) - Policy training guide
- [Cluster Setup Guide](../infrastructure/CLUSTER_SETUP_GUIDE.md) - Cluster deployment
- [P2P Orchestration](../infrastructure/VAST_P2P_ORCHESTRATION.md) - P2P system docs
