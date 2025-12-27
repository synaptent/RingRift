# Training Loop Ecosystem

This document describes the training loop implementations and their relationships.

## Overview

RingRift has multiple training loop implementations that automate the cycle:
**selfplay → sync → export → train → evaluate → promote**

All loops share the same coordination infrastructure (`event_router`, `pipeline_orchestrator`, `daemon_manager`) but differ in scope and features.

## Loop Comparison

| Loop                            | Lines | Status         | Use Case                       |
| ------------------------------- | ----- | -------------- | ------------------------------ |
| `master_loop.py`                | 1,398 | **Production** | Full system with all features  |
| `continuous_loop.py`            | ~450  | Active         | Lightweight daemon alternative |
| `run_training_loop.py`          | 426   | Active         | Single iteration, manual runs  |
| `multi_config_training_loop.py` | 1,880 | Archived       | Deprecated                     |
| `continuous_training_loop.py`   | 416   | Archived       | Deprecated                     |

## Recommended Usage

### Option A: master_loop.py (Recommended for Production)

The comprehensive all-in-one system with:

- Selfplay scheduling + idle-resource utilization
- Data sync + model distribution via daemon manager
- Evaluation + promotion orchestration (gauntlet/tournament)
- Feedback loops and curriculum weighting
- Health monitoring + queue maintenance

Note: optimization hooks (CMA-ES/NAS) are event-driven and remain off by default.

```bash
# Start the master loop
python scripts/master_loop.py --config config/unified_loop.yaml

# Check status
python scripts/master_loop.py --status

# Watch live status (does not start the loop)
python scripts/master_loop.py --watch
```

### Option B: continuous_loop.py (Lightweight Alternative)

A minimal daemon that:

- Runs selfplay to generate data
- Triggers pipeline auto-events (sync → export → train → evaluate → promote)
- Supports multiple board configs
- Has circuit breaker for failure recovery

```bash
# Via daemon manager
python scripts/launch_daemons.py --continuous

# Direct execution with defaults
python scripts/run_continuous_training.py

# Custom configs
python scripts/run_continuous_training.py --config hex8:2 --config square8:4 --games 1000
```

### Option C: run_training_loop.py (Single Iteration)

For manual, one-off training runs:

```bash
python scripts/run_training_loop.py \
  --board-type hex8 --num-players 2 \
  --selfplay-games 1000
```

## Deferral Behavior

When both loops are available, `continuous_loop.py` **automatically defers** to `master_loop.py`:

```
┌─────────────────────┐     ┌─────────────────────┐
│  master_loop        │     │  continuous_loop    │
│  (running)          │────▶│  (DEFERRED state)   │
└─────────────────────┘     └─────────────────────┘
         │                            │
         │                            │ Checks every 60s
         ▼                            ▼
    Processing...              Waiting for master_loop
                               to stop...
```

- `continuous_loop.py` checks `is_unified_loop_running()` from `app.coordination.helpers`
- If the unified orchestrator (master loop) is active, it enters `DEFERRED` state
- It periodically checks (every 60s) if the master loop has stopped
- Once the master loop stops, it resumes normal operation
- Use `--force` to override deferral (not recommended)

## Architecture Integration

All loops use the same coordination infrastructure:

```
                    ┌─────────────────────────────────┐
                    │     COORDINATION LAYER          │
                    │                                 │
                    │  ┌─────────────────────────┐    │
                    │  │ event_router.py         │    │
                    │  │ (Unified Event System)  │    │
                    │  └─────────────────────────┘    │
                    │             │                   │
                    │  ┌─────────────────────────┐    │
                    │  │ data_pipeline_orchestrator│  │
                    │  │ (Pipeline Stage Tracking)│  │
                    │  └─────────────────────────┘    │
                    │             │                   │
                    │  ┌─────────────────────────┐    │
                    │  │ daemon_manager.py       │    │
                    │  │ (Lifecycle Management)  │    │
                    │  └─────────────────────────┘    │
                    └─────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ master_loop     │    │ continuous_loop │    │ run_training_   │
│ (Full System)   │    │ (Lightweight)   │    │ loop (Single)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Event Flow

When selfplay completes, the pipeline auto-triggers:

```
SELFPLAY_COMPLETE
       │
       ▼
DATA_SYNC_COMPLETED ──▶ NPZ_EXPORT_COMPLETE ──▶ TRAINING_COMPLETED ──▶ EVALUATION_COMPLETED ──▶ MODEL_PROMOTED
       │                │              │            │             │
       ▼                ▼              ▼            ▼             ▼
   sync games      export to      train         gauntlet      promote if
   from cluster    NPZ files      model         evaluation    threshold met
```

This is orchestrated by `wire_pipeline_events(auto_trigger=True)` from `data_pipeline_orchestrator.py`.

## Archived Loops

The following loops are deprecated and archived:

### multi_config_training_loop.py (Archived)

- Location: `scripts/archive/training/`
- Reason: Subprocess-based, doesn't use event infrastructure
- Features: Multi-config, adaptive curriculum, CMA-ES tuning

### continuous_training_loop.py (Archived)

- Location: `scripts/archive/training/`
- Reason: Subprocess-based, old infrastructure
- Superseded by: `continuous_loop.py` (event-based)

## Configuration

### master_loop.py

Uses YAML config at `config/unified_loop.yaml`:

```yaml
data_ingestion:
  poll_interval_seconds: 60

evaluation:
  shadow_interval_seconds: 900
  full_tournament_interval_seconds: 86400

training:
  min_samples_to_trigger: 10000
  auto_trigger: true
```

### continuous_loop.py

Uses CLI arguments or `LoopConfig` dataclass:

```python
LoopConfig(
    configs=[("hex8", 2), ("square8", 2)],
    selfplay_games_per_iteration=1000,
    selfplay_engine="gumbel-mcts",
    max_iterations=0,  # infinite
    iteration_cooldown_seconds=60.0,
    force=False,  # defer to master loop
)
```

## Monitoring

### Check master_loop status

```bash
python scripts/master_loop.py --status
```

### Check continuous_loop status via daemon manager

```bash
python scripts/launch_daemons.py --status
```

### Check if any loop is running

```python
from app.coordination.helpers import is_unified_loop_running
print(f"Unified loop running: {is_unified_loop_running()}")
```

## When to Use Which

| Scenario                      | Recommended Loop                          |
| ----------------------------- | ----------------------------------------- |
| Production cluster deployment | `master_loop.py`                          |
| Development/testing           | `continuous_loop.py`                      |
| Single training run           | `run_training_loop.py`                    |
| Quick iteration check         | `run_training_loop.py --max-iterations 1` |
| Resource-constrained node     | `continuous_loop.py` (lighter)            |
| Full adaptive curriculum      | `master_loop.py`                          |
