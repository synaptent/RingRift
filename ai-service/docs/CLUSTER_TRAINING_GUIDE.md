# RingRift Cluster Training Guide

**Last Updated:** 2025-12-21
**Purpose:** Document cluster-based training and data generation workflows.

---

## Overview

All heavy computation (training, data generation, tournaments) should run on the cluster, not local machines. The P2P orchestrator manages job scheduling and distribution.

---

## Cluster Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     P2P Orchestrator                         │
│                   (scripts/p2p_orchestrator.py)             │
│                                                              │
│  - Leader election & cluster coordination                   │
│  - Self-play job scheduling                                 │
│  - Training job management                                  │
│  - Model deployment & synchronization                       │
│  - Health monitoring & auto-recovery                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │  Node 1 │    │  Node 2 │    │  Node N │
      │  (GPU)  │    │  (GPU)  │    │  (GPU)  │
      └─────────┘    └─────────┘    └─────────┘
```

---

## Quick Start

### 1. Check Cluster Status

```bash
# Via P2P orchestrator status endpoint
curl localhost:8770/status

# Or use the monitor module
python -m scripts.monitor status
python -m scripts.monitor health
```

### 2. Generate Canonical Training Data

For each board type and player count combination:

```bash
# Square 8x8 boards
python scripts/generate_canonical_selfplay.py --board square8 --num-players 2 --num-games 200
python scripts/generate_canonical_selfplay.py --board square8 --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board square8 --num-players 4 --num-games 200

# Square 19x19 boards
python scripts/generate_canonical_selfplay.py --board square19 --num-players 2 --num-games 200
python scripts/generate_canonical_selfplay.py --board square19 --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board square19 --num-players 4 --num-games 200

# Hex 8 boards
python scripts/generate_canonical_selfplay.py --board hex8 --num-players 2 --num-games 200
python scripts/generate_canonical_selfplay.py --board hex8 --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board hex8 --num-players 4 --num-games 200

# Hexagonal boards
python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 2 --num-games 200
python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 3 --num-games 200
python scripts/generate_canonical_selfplay.py --board hexagonal --num-players 4 --num-games 200
```

### 3. Export to NPZ Format

```bash
# Export game replays to training format
python scripts/export_replay_dataset.py \
    --db databases/canonical_square8_2p.db \
    --output data/training/canonical_square8_2p.npz \
    --feature-version 2
```

### 4. Run Neural Network Training

```bash
# Basic training run
python -m app.training.train \
    --data data/training/canonical_square8_2p.npz \
    --board square8 \
    --epochs 100 \
    --batch-size 256 \
    --save-path models/square8_2p.pt

# With Gumbel MCTS self-play data generation
python scripts/generate_gumbel_selfplay.py \
    --board square8 \
    --num-players 2 \
    --num-games 100 \
    --simulation-budget 200 \
    --output data/selfplay/gumbel_square8_2p.jsonl
```

---

## P2P Orchestrator Commands

The P2P orchestrator manages all cluster operations:

```bash
# Start orchestrator daemon
PYTHONPATH=. venv/bin/python scripts/p2p_orchestrator.py --node-id <name> --port 8770 --peers <url-list>

# Stop orchestrator
pkill -f p2p_orchestrator.py

# Submit self-play job
curl -X POST localhost:8770/jobs/selfplay \
    -d '{"board_type": "square8", "num_players": 2, "num_games": 200}'

# Submit training job
curl -X POST localhost:8770/jobs/train \
    -d '{"data_path": "data/training/combined.npz", "epochs": 100}'

# Check job status
curl localhost:8770/jobs/{job_id}

# List all jobs
curl localhost:8770/jobs
```

---

## Data Pipeline

### Data Flow

```
Self-Play Games → Game Replay DB → NPZ Export → Training → Model Checkpoint
      ↓
  Validation (parity tests)
      ↓
  Canonical DB (TRAINING_DATA_REGISTRY.md)
```

### Canonical Data Requirements

Per TRAINING_DATA_REGISTRY.md, canonical training data must:

1. Pass TS↔Python parity validation
2. Use canonical phase history (7-phase model)
3. Include explicit bookkeeping moves
4. Be schema version 15+ (see `ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md`)

### Coverage Matrix Target

| Board     | 2P   | 3P   | 4P   |
| --------- | ---- | ---- | ---- |
| square8   | 200+ | 200+ | 200+ |
| square19  | 200+ | 200+ | 200+ |
| hex8      | 200+ | 200+ | 200+ |
| hexagonal | 200+ | 200+ | 200+ |

---

## Training Configuration

### Default Hyperparameters

```yaml
# config/hyperparameters.json
{
  'learning_rate': 0.001,
  'batch_size': 256,
  'epochs': 100,
  'weight_decay': 1e-4,
  'lr_scheduler': 'cosine',
  'warmup_epochs': 5,
  'history_length': 3,
  'policy_weight': 1.0,
  'value_weight': 1.0,
}
```

### Exploration Parameters

New AlphaZero-style temperature scheduling (added 2025-12-21):

```python
# In generate_data.py
exploration_moves=30,      # Sample from visit distribution for first 30 moves
exploration_temperature=1.0  # Temperature for sampling (1.0 = proportional to visits)
```

---

## Model Deployment

### Auto-Deployment

The P2P orchestrator handles model deployment automatically:

1. New model trained → Uploaded to central storage
2. P2P orchestrator syncs to all nodes
3. Tournament evaluation against baseline
4. If improvement >= threshold → Promote to production

### Manual Deployment

```bash
# Deploy specific model
python scripts/auto_deploy_models.py \
    --model models/new_model.pt \
    --target production

# Sync models across cluster
python scripts/sync_models.py --distribute
```

---

## Monitoring

### Status Endpoints

```bash
# Cluster health
curl localhost:8770/health

# Active jobs
curl localhost:8770/jobs

# Training metrics
curl localhost:8770/metrics

# ELO ratings
curl localhost:8770/elo
```

### Dashboard

```bash
# Launch dashboard server
python scripts/dashboard_server.py

# Or use CLI status
python -m scripts.monitor status --verbose
```

---

## Troubleshooting

### Common Issues

1. **OOM during training**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model tier

2. **Slow self-play**
   - Check GPU utilization
   - Verify CUDA is enabled
   - Consider hybrid selfplay mode

3. **Parity validation failures**
   - Check game schema version
   - Verify explicit bookkeeping moves
   - See TRAINING_DATA_REGISTRY.md for gate requirements

### Log Locations

```
logs/
├── p2p_orchestrator.log      # Main orchestrator log
├── selfplay/                 # Self-play job logs
├── training/                 # Training job logs
└── tournaments/              # Tournament logs
```

---

## Reference Documentation

- **TRAINING_DATA_REGISTRY.md**: Canonical database requirements
- **AI_IMPROVEMENT_PLAN.md**: Training roadmap and targets
- **RULES_ENGINE_SURFACE_AUDIT.md**: Rules compliance
- **DEPRECATION_TIMELINE.md**: Legacy code removal schedule
- **scripts/DEPRECATED.md**: Script deprecation manifest
