# Pipeline Orchestrator

> **Status:** Production-ready as of 2025-12-13
> **Location:** `ai-service/scripts/pipeline_orchestrator.py`

The pipeline orchestrator provides unified coordination for the complete AI training pipeline across distributed compute resources including local Mac clusters, AWS, Lambda Labs, and Vast.ai instances.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [New Features (2025-12-13)](#new-features-2025-12-13)
4. [Pipeline Phases](#pipeline-phases)
5. [CLI Reference](#cli-reference)
6. [Configuration](#configuration)
7. [State Management](#state-management)
8. [Monitoring](#monitoring)
9. [Backends](#backends)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline orchestrator manages the complete AI improvement loop:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Iteration Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Selfplay │───>│   Sync   │───>│  CMA-ES  │───>│ Training │              │
│  │  Phase   │    │  Phase   │    │  Phase   │    │  Phase   │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │                                               │                      │
│       │              ┌──────────┐    ┌──────────┐    │                      │
│       │              │   Eval   │<───│ Profile  │<───┘                      │
│       │              │  Phase   │    │   Sync   │                           │
│       │              └──────────┘    └──────────┘                           │
│       │                    │                                                 │
│       │              ┌──────────┐    ┌──────────┐                           │
│       └──────────────│   Tier   │───>│ Resource │                           │
│                      │  Gating  │    │ Monitor  │                           │
│                      └──────────┘    └──────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities

- **Distributed Execution:** Coordinates selfplay, training, and evaluation across all compute resources
- **Smart Polling:** Replaces fixed waits with intelligent completion detection
- **Automatic Retry:** SSH commands retry with exponential backoff on transient failures
- **Checkpointing:** Resumes interrupted iterations from last completed phase
- **Elo Tracking:** Maintains model ratings with full history
- **Tier Gating:** Automatic D2→D4→D6→D8 promotion based on performance
- **Model Registry:** Tracks all trained models with lineage and metrics
- **Game Deduplication:** Prevents duplicate games in training data
- **Resource Monitoring:** Tracks CPU/MEM/DISK/GPU across all workers

---

## Architecture

### Worker Configuration

Workers are defined in `config/distributed_hosts.yaml`:

| Worker Type              | Role            | Capabilities           |
| ------------------------ | --------------- | ---------------------- |
| Mac Cluster (Tailscale)  | selfplay, cmaes | All board types        |
| AWS Staging (r5.4xlarge) | selfplay, cmaes | All boards (128GB RAM) |
| Lambda Labs (A10, H100)  | nn_training     | GPU training           |
| Vast.ai (RTX 3090, 5090) | nn_training     | GPU training           |

### Selfplay Job Distribution

Jobs are distributed round-robin across healthy workers with 22 diverse configurations covering:

- **Engine modes:** mixed, heuristic-only, minimax-only, mcts-only, descent-only, nn-only
- **Board types:** square8, square19, hexagonal
- **Player counts:** 2, 3, 4

---

## New Features (2025-12-13)

### SSH Retry with Exponential Backoff

All SSH commands automatically retry on transient failures:

```python
# Configuration constants
SSH_MAX_RETRIES = 3
SSH_BASE_DELAY = 2.0   # seconds
SSH_MAX_DELAY = 30.0   # seconds
SSH_BACKOFF_FACTOR = 2.0
```

Retry delays: 2s → 4s → 8s (with jitter to avoid thundering herd)

### Smart Polling

Replaces fixed 45-minute waits with intelligent completion detection:

| Phase    | Polling Method                           | Default Timeout |
| -------- | ---------------------------------------- | --------------- |
| Selfplay | Poll game counts until threshold reached | 30 min          |
| CMA-ES   | Poll for process completion              | 120 min         |
| Training | Poll for process completion              | 60 min          |

```python
SELFPLAY_MIN_GAMES_THRESHOLD = 50  # Minimum games before proceeding
POLL_INTERVAL_SECONDS = 60         # Check every minute
MAX_PHASE_WAIT_MINUTES = 120       # Maximum wait for any phase
```

### Elo Rating System

Standard Elo formula with K=32:

```python
# Update ratings after tournament match
orchestrator.update_elo_rating("MCTS_D6", "Heuristic_D5", win_rate)

# Get leaderboard (top 10 by rating)
leaderboard = orchestrator.get_elo_leaderboard(top_n=10)
# Returns: [("MCTS_D8", 1687), ("MCTS_D6", 1623), ...]
```

Elo history is persisted in state file for trend analysis.

### Model Registry

Tracks all trained models with lineage:

```python
orchestrator.register_model(
    model_id="square8_2p_iter5",
    model_path="models/square8_2p_iter5.pth",
    config="square8_2p",
    parent_id="square8_2p_iter4",  # Links to previous best
    metrics={"training_iteration": 5, "loss": 0.234}
)

# Find best model for a config (by Elo rating)
best_model_id = orchestrator.get_best_model("square8_2p")

# Deprecate old model
orchestrator.deprecate_model("square8_2p_iter3", reason="Superseded by iter5")
```

### Tier Gating

Automatic promotion through difficulty tiers based on performance:

| Promotion | Win Rate Threshold | Minimum Matches |
| --------- | ------------------ | --------------- |
| D2 → D4   | 55%                | 10+             |
| D4 → D6   | 55%                | 10+             |
| D6 → D8   | 55%                | 10+             |

```python
# Check promotion eligibility
promoted, new_tier = await orchestrator.check_tier_promotion(
    config="square8_2p",
    current_tier="D4",
    win_rate_threshold=0.55
)
```

### Game Deduplication

Prevents duplicate games from polluting training data:

```python
# Games are hashed by: moves + board_type + num_players + outcome
game_hash = orchestrator.hash_game(game_data)  # Returns 16-char SHA256 prefix

# Check and register
if not orchestrator.is_duplicate_game(game_data):
    # Process game for training
    pass

# Bulk deduplication of a database
duplicates_removed = await orchestrator.deduplicate_training_data("data/games/merged.db")
```

### Resource Monitoring

Tracks resource utilization across all workers:

```
=== Resource Usage ===
  mac-studio: CPU=45%, MEM=67%, DISK=34%, GPU=0%
  lambda-h100: CPU=12%, MEM=23%, DISK=15%, GPU=89%
  vast-5090-quad: CPU=78%, MEM=45%, DISK=22%, GPU=95%
```

### Enhanced Error Logging

Full stdout/stderr captured on command failures:

```
logs/pipeline/errors_20251213.log
```

Each failure includes: timestamp, worker, command, return code, stdout (5000 chars), stderr (5000 chars).

### Checkpointing and Resume

Iteration state is saved after each phase completion:

```bash
# Start a long run
python scripts/pipeline_orchestrator.py --iterations 10

# If interrupted, resume from last completed phase
python scripts/pipeline_orchestrator.py --iterations 10 --resume
```

---

## Pipeline Phases

### Phase 1: Selfplay

Generates training games across all board×player configurations using diverse engine modes:

| Config               | Games | Engine Mode    |
| -------------------- | ----- | -------------- |
| square8_2p_mixed     | 40    | mixed          |
| square8_2p_heuristic | 20    | heuristic-only |
| square8_2p_minimax   | 15    | minimax-only   |
| square8_2p_mcts      | 15    | mcts-only      |
| square8_2p_descent   | 20    | descent-only   |
| square8_2p_nn        | 10    | nn-only        |
| ...                  | ...   | ...            |

Total: 22 job configurations distributed round-robin across workers.

### Phase 2: Sync

Pulls data from all workers and deduplicates:

1. Pull selfplay databases via `sync_selfplay_data.sh`
2. Pull tournament JSONL files
3. Merge into local training database
4. Remove duplicate games

### Phase 3: CMA-ES

Runs heuristic optimization for all 9 board×player configurations:

| Config      | Generations | Population |
| ----------- | ----------- | ---------- |
| square8_2p  | 15          | 14         |
| square8_3p  | 12          | 14         |
| square8_4p  | 10          | 14         |
| square19_2p | 12          | 14         |
| square19_3p | 10          | 14         |
| square19_4p | 8           | 14         |
| hex_2p      | 12          | 14         |
| hex_3p      | 10          | 14         |
| hex_4p      | 8           | 14         |

### Phase 4: Training

Trains neural networks for each configuration:

| Config       | Min Games Required | Epochs |
| ------------ | ------------------ | ------ |
| square8_2p   | 100                | 50     |
| square8_3p   | 50                 | 40     |
| square8_4p   | 40                 | 40     |
| square19_2p  | 50                 | 30     |
| hexagonal_2p | 50                 | 30     |

Trained models are registered in the model registry with lineage tracking.

### Phase 5: Profile Sync

Merges trained heuristic profiles from all workers:

1. Pull `trained_heuristic_profiles.json` from each worker
2. Merge by highest fitness score
3. Push unified profiles back to all workers

### Phase 6: Evaluation

Runs tournaments and updates Elo ratings:

| Matchup                      | Board    | Games |
| ---------------------------- | -------- | ----- |
| Heuristic D5 vs MCTS D6      | Square8  | 10    |
| Heuristic D5 vs Minimax D5   | Square8  | 10    |
| MCTS D6 vs Minimax D5        | Square8  | 10    |
| Heuristic D3 vs Heuristic D5 | Square8  | 8     |
| MCTS D5 vs MCTS D7           | Square8  | 8     |
| Heuristic D5 vs MCTS D5      | Square19 | 6     |
| Heuristic D5 vs MCTS D5      | Hex      | 6     |
| MCTS D7 vs MCTS D8           | Square8  | 6     |

Tournament games are saved to `logs/tournaments/` for training data.

### Phase 7: Tier Gating

Checks each configuration for tier promotion eligibility.

### Phase 8: Resource Monitoring

Logs resource utilization and prints Elo leaderboard.

---

## CLI Reference

### Basic Usage

```bash
cd ai-service

# Run a single iteration
python scripts/pipeline_orchestrator.py --iterations 1

# Run continuous improvement loop (10 iterations)
python scripts/pipeline_orchestrator.py --iterations 10

# Resume from interrupted iteration
python scripts/pipeline_orchestrator.py --iterations 5 --resume

# Dry run (show what would execute)
python scripts/pipeline_orchestrator.py --dry-run
```

### Run Specific Phases

```bash
# Selfplay only
python scripts/pipeline_orchestrator.py --phase selfplay

# Sync only
python scripts/pipeline_orchestrator.py --phase sync

# CMA-ES only
python scripts/pipeline_orchestrator.py --phase cmaes

# Training only
python scripts/pipeline_orchestrator.py --phase training

# Profile sync only
python scripts/pipeline_orchestrator.py --phase profile-sync

# Evaluation only
python scripts/pipeline_orchestrator.py --phase evaluation

# Tier gating checks only
python scripts/pipeline_orchestrator.py --phase tier-gating

# Resource monitoring only
python scripts/pipeline_orchestrator.py --phase resources
```

### CLI Options

| Option                | Description                     | Default                    |
| --------------------- | ------------------------------- | -------------------------- |
| `--iterations N`      | Number of pipeline iterations   | 1                          |
| `--start-iteration N` | Starting iteration number       | 0                          |
| `--phase PHASE`       | Run only specific phase         | (full iteration)           |
| `--resume`            | Resume from last saved state    | false                      |
| `--dry-run`           | Show commands without executing | false                      |
| `--config PATH`       | Pipeline configuration JSON     | none                       |
| `--state-path PATH`   | State file location             | `logs/pipeline/state.json` |

---

## Configuration

### Worker Configuration (`config/distributed_hosts.yaml`)

```yaml
hosts:
  mac-studio:
    ssh_host: '100.107.168.125' # Tailscale IP
    ssh_user: 'armand'
    ssh_key: '~/.ssh/id_cluster'
    ringrift_path: '~/Development/RingRift'
    memory_gb: 96
    gpu: 'M3 Max/Ultra (MPS)'
    role: 'nn_training_mps'
    status: 'ready'

  aws-staging:
    ssh_host: '54.198.219.106'
    ssh_user: 'ubuntu'
    ssh_key: '~/.ssh/ringrift-staging-key.pem'
    ringrift_path: '/home/ubuntu/ringrift'
    memory_gb: 128
    role: 'selfplay_cmaes'
    status: 'ready'

  lambda-h100:
    ssh_host: '209.20.157.81'
    ssh_user: 'ubuntu'
    ringrift_path: '~/ringrift'
    memory_gb: 256
    gpu: 'NVIDIA H100 (80GB)'
    role: 'nn_training_primary'
    status: 'ready'

  vast-5090-quad:
    ssh_host: '211.72.13.202'
    ssh_port: 45875
    ssh_user: 'root'
    ringrift_path: '~/ringrift'
    memory_gb: 1024
    gpu: '4x RTX 5090 (128GB)'
    role: 'nn_training_primary'
    status: 'ready'
```

---

## State Management

### State File (`logs/pipeline/state.json`)

```json
{
  "iteration": 5,
  "phase": "complete",
  "phase_completed": {
    "selfplay": true,
    "sync": true,
    "cmaes": true,
    "training": true,
    "evaluation": true
  },
  "games_generated": {
    "iter_5": 234
  },
  "models_trained": ["square8_2p_iter5", "square8_3p_iter5"],
  "elo_ratings": {
    "MCTS_D8": 1687,
    "MCTS_D6": 1623,
    "Heuristic_D5": 1534
  },
  "elo_history": [
    {
      "timestamp": "2025-12-13T10:30:00",
      "player_a": "MCTS_D6",
      "player_b": "Heuristic_D5",
      "score_a": 0.65,
      "rating_a_before": 1600,
      "rating_b_before": 1550,
      "rating_a_after": 1623,
      "rating_b_after": 1534
    }
  ],
  "model_registry": {
    "square8_2p_iter5": {
      "model_id": "square8_2p_iter5",
      "path": "models/square8_2p_iter5.pth",
      "config": "square8_2p",
      "parent_id": "square8_2p_iter4",
      "created_at": "2025-12-13T10:30:00",
      "iteration": 5,
      "status": "active"
    }
  },
  "tier_promotions": {
    "square8_2p": "D6",
    "square8_3p": "D4"
  },
  "seen_game_hashes": ["a1b2c3d4e5f6", "..."],
  "errors": []
}
```

---

## Monitoring

### Log Files

| File                                | Contents                          |
| ----------------------------------- | --------------------------------- |
| `logs/pipeline/state.json`          | Pipeline state with checkpointing |
| `logs/pipeline/errors_YYYYMMDD.log` | Detailed error logs               |
| `logs/selfplay/iter{N}_*.jsonl`     | Selfplay game logs                |
| `logs/tournaments/iter{N}/`         | Tournament game records           |
| `logs/cmaes/iter{N}/`               | CMA-ES optimization logs          |

### Iteration Summary

```
============================================================
=== Iteration 5 Complete ===
============================================================
Selfplay:    {'dispatched': 22, 'total': 22}
CMA-ES:      {'square8_2p': True, 'square8_3p': True, ...}
Training:    {'square8_2p': True, 'square8_3p': True}
Eval:        {'Heuristic5_vs_MCTS6_Square8': 0.4, ...}
Promotions:  {'square8_2p': 'D6'}
============================================================

=== Elo Leaderboard ===
  1. MCTS_D8: 1687
  2. MCTS_D6: 1623
  3. Minimax_D5: 1567
  4. Heuristic_D5: 1534

=== Resource Usage ===
  mac-studio: CPU=45%, MEM=67%, DISK=34%, GPU=0%
  lambda-h100: CPU=12%, MEM=23%, DISK=15%, GPU=89%
```

---

## Backends

The orchestrator supports two backend modes for job dispatch:

### SSH Backend (Default)

Traditional SSH-based job execution:

```bash
python scripts/pipeline_orchestrator.py \
  --iterations 5
```

### P2P Backend

Uses the P2P orchestrator REST API for job dispatch:

```bash
python scripts/pipeline_orchestrator.py \
  --backend p2p \
  --p2p-leader http://lambda-gpu:8770 \
  --p2p-auth-token "your-cluster-token" \
  --iterations 5
```

See [P2P_ORCHESTRATOR_AUTH.md](P2P_ORCHESTRATOR_AUTH.md) for authentication setup.

---

## Troubleshooting

### SSH Connection Failures

1. Check worker health manually:

   ```bash
   ssh user@host "echo healthy"
   ```

2. Verify SSH key:

   ```bash
   ssh -i ~/.ssh/key user@host "echo ok"
   ```

3. Check error logs:
   ```bash
   cat logs/pipeline/errors_$(date +%Y%m%d).log
   ```

### Selfplay Not Generating Games

1. Check worker status:

   ```bash
   python scripts/pipeline_orchestrator.py --phase resources
   ```

2. Verify selfplay script on worker:
   ```bash
   ssh worker "cd ~/ringrift/ai-service && python scripts/run_self_play_soak.py --help"
   ```

### CMA-ES Jobs Not Completing

1. Check for running processes:

   ```bash
   ssh worker "pgrep -f run_iterative_cmaes"
   ```

2. Check CMA-ES logs:
   ```bash
   ssh worker "ls -la ~/ringrift/ai-service/logs/cmaes/"
   ```

### Training Failures

1. Check GPU availability:

   ```bash
   ssh gpu-worker "nvidia-smi"
   ```

2. Check training logs:
   ```bash
   ssh training-worker "tail -100 /tmp/ringrift_bg_*.log"
   ```

### State Corruption

Reset state file and start fresh:

```bash
rm logs/pipeline/state.json
python scripts/pipeline_orchestrator.py --iterations 1
```

---

## Related Documentation

- [DISTRIBUTED_SELFPLAY.md](DISTRIBUTED_SELFPLAY.md) - Cluster setup
- [P2P_ORCHESTRATOR_AUTH.md](P2P_ORCHESTRATOR_AUTH.md) - P2P authentication
- [GPU_PIPELINE_ROADMAP.md](GPU_PIPELINE_ROADMAP.md) - GPU acceleration plans
- [AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md](../../docs/ai/AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md) - Tier gating details
- [TRAINING_DATA_REGISTRY.md](../TRAINING_DATA_REGISTRY.md) - Canonical data sources
