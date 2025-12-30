# CLAUDE.md - AI Assistant Context for ai-service

AI assistant context for the Python AI training service. Complements `AGENTS.md` with operational knowledge.

**Last Updated**: December 30, 2025

## Project Overview

RingRift is a multiplayer territory control game. The Python `ai-service` mirrors the TypeScript engine (`src/shared/engine/`) for training data generation and must maintain **parity** with it.

| Board Type  | Grid      | Cells | Players |
| ----------- | --------- | ----- | ------- |
| `square8`   | 8×8       | 64    | 2,3,4   |
| `square19`  | 19×19     | 361   | 2,3,4   |
| `hex8`      | radius 4  | 61    | 2,3,4   |
| `hexagonal` | radius 12 | 469   | 2,3,4   |

## Quick Start

```bash
# Full cluster automation (RECOMMENDED)
python scripts/master_loop.py

# Single config training pipeline
python scripts/run_training_loop.py --board-type hex8 --num-players 2 --selfplay-games 1000

# Manual training
python -m app.training.train --board-type hex8 --num-players 2 --data-path data/training/hex8_2p.npz

# Cluster status
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Leader: {d.get(\"leader_id\")}, Alive: {d.get(\"alive_peers\")}")'

# Update all cluster nodes
python scripts/update_all_nodes.py --restart-p2p
```

## Cluster Infrastructure (Dec 2025)

~36 nodes, ~1.3TB GPU memory across providers:

| Provider     | Nodes | GPUs                        |
| ------------ | ----- | --------------------------- |
| Lambda GH200 | 6     | GH200 96GB (training-only)  |
| Vast.ai      | 14    | RTX 5090/4090/3090, A40     |
| RunPod       | 6     | H100, A100×5, L40S          |
| Nebius       | 3     | H100 80GB×2, L40S           |
| Vultr        | 2     | A100 20GB vGPU              |
| Hetzner      | 3     | CPU only (P2P voters)       |
| Local        | 2     | Mac Studio M3 (coordinator) |

## Key Modules

### Configuration

| Module                                | Purpose                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `app/config/env.py`                   | Typed environment variables (`env.node_id`, `env.log_level`)   |
| `app/config/cluster_config.py`        | Cluster node access (`get_cluster_nodes()`, `get_gpu_nodes()`) |
| `app/config/coordination_defaults.py` | Centralized timeouts, thresholds, priority weights             |
| `app/config/thresholds.py`            | Training/selfplay budget constants                             |

### Coordination Infrastructure (255 modules)

| Module                          | Purpose                                            |
| ------------------------------- | -------------------------------------------------- |
| `daemon_manager.py`             | Lifecycle for 97 daemon types (~2,000 LOC)         |
| `daemon_registry.py`            | Declarative daemon specs (DaemonSpec dataclass)    |
| `daemon_runners.py`             | 89 async runner functions                          |
| `event_router.py`               | Unified event bus (220+ event types, SHA256 dedup) |
| `selfplay_scheduler.py`         | Priority-based selfplay allocation (~3,800 LOC)    |
| `budget_calculator.py`          | Gumbel budget tiers, target games calculation      |
| `progress_watchdog_daemon.py`   | Stall detection for 48h autonomous operation       |
| `p2p_recovery_daemon.py`        | P2P cluster health recovery                        |
| `stale_fallback.py`             | Graceful degradation with older models             |
| `data_pipeline_orchestrator.py` | Pipeline stage tracking                            |
| `auto_sync_daemon.py`           | P2P data synchronization                           |
| `sync_router.py`                | Intelligent sync routing                           |
| `feedback_loop_controller.py`   | Training feedback signals                          |
| `health_facade.py`              | Unified health check API                           |

### AI Components

| Module                           | Purpose                                     |
| -------------------------------- | ------------------------------------------- |
| `app/ai/gpu_parallel_games.py`   | Vectorized GPU selfplay (6-57× speedup)     |
| `app/ai/gumbel_search_engine.py` | Unified MCTS entry point                    |
| `app/ai/gumbel_common.py`        | Shared Gumbel data structures, budget tiers |
| `app/ai/harness/`                | Harness abstraction layer (see below)       |
| `app/ai/nnue.py`                 | NNUE evaluation network (~256 hidden)       |
| `app/ai/nnue_policy.py`          | NNUE with policy head for MCTS              |

### Harness Abstraction Layer (Dec 2025)

The harness abstraction provides a unified interface for evaluating models under different AI algorithms. This enables tracking separate Elo ratings per (model, harness) combination.

**Files:**

| Module                                  | Purpose                                    |
| --------------------------------------- | ------------------------------------------ |
| `app/ai/harness/base_harness.py`        | `AIHarness` abstract base, `HarnessConfig` |
| `app/ai/harness/harness_registry.py`    | Factory, compatibility matrix              |
| `app/ai/harness/evaluation_metadata.py` | `EvaluationMetadata` for Elo tracking      |
| `app/ai/harness/implementations.py`     | Concrete harness implementations           |

**Harness Types:**

| Type          | NN  | NNUE | Policy   | Best For                    |
| ------------- | --- | ---- | -------- | --------------------------- |
| `GUMBEL_MCTS` | ✓   | -    | Required | Training data, high quality |
| `GPU_GUMBEL`  | ✓   | -    | Required | High throughput selfplay    |
| `MINIMAX`     | ✓   | ✓    | -        | 2-player, fast evaluation   |
| `MAXN`        | ✓   | ✓    | -        | 3-4 player multiplayer      |
| `BRS`         | ✓   | ✓    | -        | Fast multiplayer            |
| `POLICY_ONLY` | ✓   | ✓\*  | Required | Baselines, fast play        |
| `DESCENT`     | ✓   | -    | Required | Exploration, research       |
| `HEURISTIC`   | -   | -    | -        | Baselines, bootstrap        |

**Usage:**

```python
from app.ai.harness import create_harness, HarnessType, get_compatible_harnesses

# Create a harness
harness = create_harness(
    HarnessType.MINIMAX,
    model_path="models/nnue_hex8_2p.pt",
    board_type="hex8",
    num_players=2,
    depth=4,
)

# Evaluate position
move, metadata = harness.evaluate(game_state, player_number=1)

# Metadata includes visit distribution for soft targets
print(f"Value: {metadata.value_estimate}, Nodes: {metadata.nodes_visited}")

# Composite ID for Elo: "nnue_hex8_2p:minimax:d4abc123"
participant_id = harness.get_composite_participant_id()
```

### NNUE Evaluation (Dec 2025)

NNUE (Efficiently Updatable Neural Network) provides fast CPU inference for minimax-style search.

**Architecture (V3):**

- 26 spatial planes + 32 global features per position
- ~256 hidden units, ClippedReLU activations
- Output: [-1, 1] scaled to centipawn-like score

**Feature Planes (26):**
| Planes | Content |
|--------|---------|
| 0-3 | Ring presence per player |
| 4-7 | Stack height per player |
| 8-11 | Territory ownership per player |
| 12-15 | Marker presence per player |
| 16-19 | Cap height per player |
| 20-23 | Line threat per direction |
| 24-25 | Capture threat (vulnerability, opportunity) |

**Usage:**

```python
from app.ai.nnue import RingRiftNNUE, clear_nnue_cache

# Load NNUE model
nnue = RingRiftNNUE.from_checkpoint("models/nnue_hex8_2p.pt", board_type="hex8")

# Evaluate position
score = nnue.evaluate(game_state, perspective_player=1)

# Clear cache when done
clear_nnue_cache()
```

### Multi-Harness Gauntlet (Dec 2025)

Evaluates models under all compatible harnesses, producing Elo ratings per combination.

```python
from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

gauntlet = MultiHarnessGauntlet()
results = await gauntlet.evaluate_model(
    model_path="models/canonical_hex8_4p.pth",
    model_type="nn",  # or "nnue", "nnue_mp"
    board_type="hex8",
    num_players=4,
)

for harness, rating in results.ratings.items():
    print(f"  {harness}: Elo {rating.elo:.0f} ({rating.win_rate:.1%})")
```

**Model Types:**

- `nn`: Full neural network (v2-v6), compatible with policy-based harnesses
- `nnue`: NNUE (2-player), uses minimax
- `nnue_mp`: Multi-player NNUE, uses MaxN/BRS

### Architecture Tracker (Dec 2025)

Tracks performance metrics across neural network architectures (v2, v4, v5, etc.) to enable intelligent training compute allocation.

**Key Metrics:**

| Metric                  | Description                                  |
| ----------------------- | -------------------------------------------- |
| `avg_elo`               | Average Elo rating across evaluations        |
| `best_elo`              | Best observed Elo rating                     |
| `elo_per_training_hour` | Efficiency metric (Elo gain / training time) |
| `games_evaluated`       | Total games used in evaluations              |

**Usage:**

```python
from app.training.architecture_tracker import (
    get_architecture_tracker,
    record_evaluation,
    get_best_architecture,
)

# Record evaluation result
record_evaluation(
    architecture="v5",
    board_type="hex8",
    num_players=2,
    elo=1450,
    training_hours=2.5,
    games_evaluated=100,
)

# Get best architecture for allocation decisions
best = get_best_architecture(board_type="hex8", num_players=2)
print(f"Best: {best.architecture} with Elo {best.avg_elo:.0f}")

# Get compute allocation weights
tracker = get_architecture_tracker()
weights = tracker.get_compute_weights(board_type="hex8", num_players=2)
# Returns: {"v4": 0.15, "v5": 0.35, "v5_heavy": 0.50}
```

**Integration with SelfplayScheduler:**

- `ArchitectureFeedbackController` subscribes to `EVALUATION_COMPLETED` events
- Automatically updates architecture weights based on gauntlet results
- Higher-performing architectures receive more selfplay game allocation

### Base Classes

| Class                  | Location                        | Purpose                                       |
| ---------------------- | ------------------------------- | --------------------------------------------- |
| `HandlerBase`          | `handler_base.py`               | Event-driven handlers (550 LOC, 45 tests)     |
| `MonitorBase`          | `monitor_base.py`               | Health monitoring daemons (800 LOC, 41 tests) |
| `P2PMixinBase`         | `scripts/p2p/p2p_mixin_base.py` | P2P mixin utilities (995 LOC)                 |
| `SingletonMixin`       | `singleton_mixin.py`            | Singleton pattern (503 LOC)                   |
| `CircuitBreakerConfig` | `transport_base.py`             | Circuit breaker configuration (canonical)     |

## Daemon System

91 active daemon types, 6 deprecated. Three-layer architecture:

1. **`daemon_registry.py`** - Declarative `DAEMON_REGISTRY: Dict[DaemonType, DaemonSpec]`
2. **`daemon_manager.py`** - Lifecycle coordinator (start/stop, health, auto-restart)
3. **`daemon_runners.py`** - 89 async runner functions

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()
await dm.start(DaemonType.AUTO_SYNC)
health = dm.get_all_daemon_health()
```

**Key Categories:**
| Category | Daemons |
|----------|---------|
| Sync | AUTO_SYNC, MODEL_DISTRIBUTION, ELO_SYNC, GOSSIP_SYNC, OWC_IMPORT |
| Pipeline | DATA_PIPELINE, SELFPLAY_COORDINATOR, TRAINING_NODE_WATCHER |
| Health | NODE_HEALTH_MONITOR, QUALITY_MONITOR, NODE_AVAILABILITY |
| Resources | IDLE_RESOURCE, NODE_RECOVERY |
| Autonomous | PROGRESS_WATCHDOG, P2P_RECOVERY, STALE_FALLBACK, MEMORY_MONITOR |

**Health Monitoring (85%+ coverage):**

```python
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
# {"status": "healthy", "running": True, "last_sync": ..., "errors_count": 0}
```

### Evaluation Daemon Backpressure (Dec 29, 2025)

When evaluation queue exceeds threshold, EvaluationDaemon emits backpressure events:

- `EVALUATION_BACKPRESSURE` at queue_depth >= 70 - signals training to pause
- `EVALUATION_BACKPRESSURE_RELEASED` at queue_depth <= 35 - training can resume
- Hysteresis prevents oscillation between states
- Config: `EvaluationConfig.backpressure_threshold` / `backpressure_release_threshold`

### Training Trigger Confidence-Based Triggering (Dec 29, 2025)

Can trigger training with <5000 samples if statistical confidence is high:

- Minimum 1000 samples as safety floor
- Targets 95% confidence interval width ±2.5%
- Enables faster iteration on high-quality data
- Config: `TrainingTriggerConfig.confidence_early_trigger_enabled` (default: true)

### Evaluation Daemon Retry Strategy (Dec 29, 2025)

Automatic retry for transient failures (GPU OOM, timeouts):

- Up to 3 attempts per model with exponential backoff: 60s, 120s, 240s
- Retryable failures: timeout, GPU OOM, distribution incomplete
- Max retries exceeded: emits `EVALUATION_FAILED` event
- Stats tracked: `retries_queued`, `retries_succeeded`, `retries_exhausted`

## Event System

220+ event types across 3 layers:

1. **In-memory EventBus** - Local daemon communication
2. **Stage events** - Pipeline stage completion
3. **Cross-process queue** - Cluster-wide events

**Critical Event Flows:**

```
Selfplay → NEW_GAMES_AVAILABLE → DataPipeline → TRAINING_THRESHOLD_REACHED
    → Training → TRAINING_COMPLETED → Evaluation → EVALUATION_COMPLETED
    → MODEL_PROMOTED → Distribution → Curriculum rebalance
```

**Key Events:**
| Event | Emitter | Subscribers |
|-------|---------|-------------|
| `TRAINING_COMPLETED` | TrainingCoordinator | FeedbackLoop, DataPipeline, UnifiedQueuePopulator |
| `TRAINING_LOCK_ACQUIRED` | TrainingCoordinator | (Internal tracking) |
| `TRAINING_SLOT_UNAVAILABLE` | TrainingCoordinator | (Internal tracking) |
| `MODEL_PROMOTED` | PromotionController | UnifiedDistributionDaemon, CurriculumIntegration |
| `DATA_SYNC_COMPLETED` | AutoSyncDaemon | DataPipelineOrchestrator, TransferVerification |
| `REGRESSION_DETECTED` | RegressionDetector | TrainingCoordinator, UnifiedFeedback |
| `REGRESSION_CRITICAL` | RegressionDetector | DaemonManager, CurriculumIntegration |
| `EVALUATION_BACKPRESSURE` | EvaluationDaemon | TrainingCoordinator (pauses training) |
| `NEW_GAMES_AVAILABLE` | AutoExportDaemon | DataPipeline, TrainingCoordinator |
| `SELFPLAY_COMPLETE` | SelfplayRunner | DataConsolidation, UnifiedFeedback |

**Complete Event Documentation:**

- `docs/architecture/EVENT_SUBSCRIPTION_MATRIX.md` - Full list of 220+ events with emitters/subscribers
- `docs/architecture/EVENT_FLOW_INTEGRATION.md` - Event flow diagrams and integration patterns

```python
from app.coordination.event_emitters import emit_training_complete
emit_training_complete(config_key="hex8_2p", model_path="models/canonical_hex8_2p.pth")
```

## Common Patterns

### Singleton

```python
from app.coordination.singleton_mixin import SingletonMixin
class MyCoordinator(SingletonMixin): pass
instance = MyCoordinator.get_instance()
```

### Handler Base

```python
from app.coordination.handler_base import HandlerBase

class MyDaemon(HandlerBase):
    def __init__(self):
        super().__init__(name="my_daemon", cycle_interval=60.0)
    async def _run_cycle(self) -> None: pass
    def _get_event_subscriptions(self) -> dict:
        return {"MY_EVENT": self._on_my_event}
```

### Health Checks

```python
from app.coordination.health_facade import get_health_orchestrator, get_system_health_score
score = get_system_health_score()  # 0.0-1.0
orchestrator = get_health_orchestrator()
```

### HTTP Health Endpoints

The DaemonManager exposes HTTP health endpoints on port 8790 (configurable via `RINGRIFT_HEALTH_PORT`):

| Endpoint       | Purpose                                                           |
| -------------- | ----------------------------------------------------------------- |
| `GET /health`  | Liveness probe - returns 200 if daemon manager is running         |
| `GET /ready`   | Readiness probe - returns 200 if all critical daemons are healthy |
| `GET /metrics` | Prometheus-style metrics for monitoring                           |
| `GET /status`  | Detailed daemon status JSON                                       |

```bash
# Check if daemon manager is healthy
curl http://localhost:8790/health

# Get detailed daemon status
curl http://localhost:8790/status | jq

# P2P cluster status (port 8770)
curl http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Leader: {d.get(\"leader_id\")}, Alive: {d.get(\"alive_peers\")}")'
```

## File Structure

```
ai-service/
├── app/
│   ├── ai/              # Neural net, MCTS, heuristics
│   ├── config/          # Environment, cluster, thresholds
│   ├── coordination/    # Daemons, events, pipeline (190+ modules)
│   │   └── node_availability/  # Cloud provider sync (Dec 2025)
│   ├── core/            # SSH, node info, utilities
│   ├── db/              # GameReplayDB
│   ├── distributed/     # Cluster monitor, data catalog
│   ├── rules/           # Python game engine (mirrors TS)
│   ├── training/        # Training pipeline, enhancements
│   └── utils/           # Game discovery, paths
├── config/              # distributed_hosts.yaml
├── data/games/          # SQLite game databases
├── models/              # Model checkpoints (canonical_*.pth)
├── scripts/
│   ├── p2p/             # P2P orchestrator (~28K LOC)
│   │   ├── managers/    # StateManager, JobManager, SelfplayScheduler
│   │   └── loops/       # JobReaper, IdleDetection, EloSync
│   └── master_loop.py   # Main automation entry
└── tests/               # 233 test files
```

## Environment Variables

### Core

| Variable                      | Default  | Purpose                     |
| ----------------------------- | -------- | --------------------------- |
| `RINGRIFT_NODE_ID`            | hostname | Node identifier             |
| `RINGRIFT_LOG_LEVEL`          | INFO     | Logging level               |
| `RINGRIFT_IS_COORDINATOR`     | false    | Coordinator node flag       |
| `RINGRIFT_ALLOW_PENDING_GATE` | false    | Bypass TS parity validation |

### Selfplay Priority Weights

| Variable                       | Default | Purpose                         |
| ------------------------------ | ------- | ------------------------------- |
| `RINGRIFT_STALENESS_WEIGHT`    | 0.30    | Weight for stale configs        |
| `RINGRIFT_ELO_VELOCITY_WEIGHT` | 0.20    | Weight for Elo velocity         |
| `RINGRIFT_CURRICULUM_WEIGHT`   | 0.20    | Weight for curriculum           |
| `RINGRIFT_QUALITY_WEIGHT`      | 0.15    | Weight for data quality         |
| `RINGRIFT_DIVERSITY_WEIGHT`    | 0.10    | Weight for diversity            |
| `RINGRIFT_VOI_WEIGHT`          | 0.05    | Weight for value of information |

### Sync & Timeouts

| Variable                         | Default | Purpose                   |
| -------------------------------- | ------- | ------------------------- |
| `RINGRIFT_DATA_SYNC_INTERVAL`    | 60      | Seconds between syncs     |
| `RINGRIFT_EVENT_HANDLER_TIMEOUT` | 600     | Handler timeout (seconds) |
| `RINGRIFT_HEALTH_CHECK_INTERVAL` | 30      | Health check interval     |

## Deprecated Modules (Removal: Q2 2026)

| Deprecated                     | Replacement                            |
| ------------------------------ | -------------------------------------- |
| `cluster_data_sync.py`         | `AutoSyncDaemon(strategy="broadcast")` |
| `ephemeral_sync.py`            | `AutoSyncDaemon(strategy="ephemeral")` |
| `system_health_monitor.py`     | `unified_health_manager.py`            |
| `node_health_monitor.py`       | `health_check_orchestrator.py`         |
| `queue_populator_daemon.py`    | `unified_queue_populator.py`           |
| `model_distribution_daemon.py` | `unified_distribution_daemon.py`       |

## 48-Hour Autonomous Operation (Dec 2025)

The cluster can run unattended for 48+ hours with these daemons:

| Daemon              | Purpose                                  |
| ------------------- | ---------------------------------------- |
| `PROGRESS_WATCHDOG` | Detects Elo stalls, triggers recovery    |
| `P2P_RECOVERY`      | Restarts unhealthy P2P orchestrator      |
| `STALE_FALLBACK`    | Uses older models when sync fails        |
| `MEMORY_MONITOR`    | Prevents OOM via proactive VRAM tracking |

**Resilience Features:**

- Adaptive circuit breaker cascade prevention (dynamic thresholds 10-20 based on health)
- Multi-transport failover: Tailscale → SSH → Base64 → HTTP
- Stale training fallback after 5 sync failures or 45min timeout
- Automatic parity gate bypass on cluster nodes without Node.js
- Handler timeout protection (600s default, configurable)

**Key Events:**

- `PROGRESS_STALL_DETECTED` - Config Elo stalled >24h
- `PROGRESS_RECOVERED` - Config resumed progress
- `MEMORY_PRESSURE` - GPU VRAM critical, pause spawning

**Budget Tiers** (from `budget_calculator.py`):
| Game Count | Budget | Purpose |
|------------|--------|---------|
| <100 | 64 | Bootstrap tier 1 - max throughput |
| <500 | 150 | Bootstrap tier 2 - fast iteration |
| <1000 | 200 | Bootstrap tier 3 - balanced |
| ≥1000 | Elo-based | STANDARD/QUALITY/ULTIMATE/MASTER |

## Type Consolidation (Dec 2025)

Duplicate type definitions have been consolidated with domain-specific renames and backward-compatible aliases:

| Original Class  | New Name                | Location                                 | Canonical For         |
| --------------- | ----------------------- | ---------------------------------------- | --------------------- |
| `Alert`         | `MonitoringAlert`       | `app/monitoring/base.py`                 | Monitoring alerts     |
| `Alert`         | `RouterAlert`           | `app/monitoring/alert_router.py`         | Alert routing         |
| `Alert`         | (canonical)             | `app/coordination/alert_types.py`        | Coordination layer    |
| `FeedbackState` | `PipelineFeedbackState` | `app/integration/pipeline_feedback.py`   | Global pipeline state |
| `FeedbackState` | (canonical)             | `app/coordination/feedback_state.py`     | Per-config state      |
| `GameResult`    | `GauntletGameResult`    | `app/training/game_gauntlet.py`          | Gauntlet evaluation   |
| `GameResult`    | `DistributedGameResult` | `app/tournament/distributed_gauntlet.py` | Distributed gauntlet  |
| `GameResult`    | `GumbelGameResult`      | `app/ai/multi_game_gumbel.py`            | Multi-game Gumbel     |
| `GameResult`    | (canonical)             | `app/training/selfplay_runner.py`        | Selfplay results      |
| `GameResult`    | (canonical)             | `app/execution/game_executor.py`         | Game execution        |

All renamed classes have backward-compatible `ClassName = NewClassName` aliases that will be deprecated in Q2 2026.

## GPU Vectorization Optimization Status (Dec 2025) - COMPLETE

The GPU selfplay engine (`app/ai/gpu_parallel_games.py`) has been **extensively optimized**. No further `.item()` optimization is needed.

### Current State

| Metric                    | Value                                        |
| ------------------------- | -------------------------------------------- |
| Speedup                   | 6-57× on CUDA vs CPU (batch-dependent)       |
| Remaining `.item()` calls | **6** (down from 80+)                        |
| Hot path `.item()` calls  | **0**                                        |
| Fully vectorized          | Move selection, game state updates, sampling |

### Remaining `.item()` Calls (All Necessary)

| File                      | Line | Purpose                                   | Hot Path? |
| ------------------------- | ---- | ----------------------------------------- | --------- |
| `gpu_parallel_games.py`   | 1475 | Statistics tracking (`move_count.sum()`)  | No        |
| `gpu_move_generation.py`  | 1672 | Chain capture target (live tensor needed) | No\*      |
| `gpu_move_application.py` | 1355 | Max distance for loop bounds              | No        |
| `gpu_move_application.py` | 1719 | History max distance                      | No        |
| `gpu_move_application.py` | 1777 | Max distance for loop bounds              | No        |
| `gpu_move_application.py` | 1933 | Dynamic tensor size for `torch.ones`      | No        |

\*Line 1672 is in chain capture processing, which is infrequent compared to move generation.

### Why No Further Optimization

1. **Statistics (line 1475)**: Only called once per batch for metrics, not per game
2. **Chain capture (line 1672)**: Must use live tensor to correctly find targets after prior captures in chain
3. **Max distance (lines 1355/1719/1777)**: Python loop bounds require scalar values
4. **Dynamic sizes (line 1933)**: `torch.ones()` requires integer size parameter

### Optimization History

- **Dec 13, 2025**: Pre-extract numpy arrays to avoid loop `.item()` calls
- **Dec 14, 2025**: Segment-wise softmax sampling (no per-game loops)
- **Dec 24, 2025**: Fully vectorized move selection, pre-extract metadata batches
- **Dec 2025**: Reduced from 80+ `.item()` calls to 6 remaining necessary calls

**DO NOT** attempt further `.item()` optimization - the remaining calls are architecturally necessary.

## Transfer Learning (2p → 4p)

Train 4-player models using 2-player weights:

```bash
# Step 1: Resize value head from 2 outputs to 4 outputs
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_sq8_2p.pth \
  --output models/transfer_sq8_4p_init.pth \
  --board-type square8

# Step 2: Train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/transfer_sq8_4p_init.pth \
  --data-path data/training/sq8_4p.npz
```

## Known Issues

1. **Parity gates on cluster**: Nodes lack `npx`, so TS validation fails. Set `RINGRIFT_ALLOW_PENDING_GATE=1`.
2. **Board conventions**: Hex boards use radius. hex8 = radius 4 = 61 cells.
3. **GPU memory**: v2 models with batch_size=512 need ~8GB VRAM.
4. **PYTHONPATH**: Set `PYTHONPATH=.` when running scripts from ai-service directory.
5. **Container networking**: Vast.ai/RunPod need `container_tailscale_setup.py` for mesh connectivity.

## P2P Stability Fixes (Dec 2025)

Recent stability improvements to the P2P orchestrator:

- Pre-flight dependency validation (aiohttp, psutil, yaml)
- Gzip magic byte detection in gossip handler
- 120s startup grace period for slow state loading
- SystemExit handling in task wrapper
- /dev/shm fallback for macOS compatibility
- Clear port binding error messages
- Heartbeat interval reduced: 30s → 15s (faster peer discovery)
- Peer timeout reduced: 90s → 60s (faster dead node detection)

## See Also

- `AGENTS.md` - Coding guidelines
- `SECURITY.md` - Security considerations
- `docs/DAEMON_REGISTRY.md` - Full daemon reference
- `docs/EVENT_SYSTEM_REFERENCE.md` - Complete event documentation
- `docs/architecture/EVENT_SUBSCRIPTION_MATRIX.md` - Event emitter/subscriber matrix
- `docs/architecture/EVENT_FLOW_INTEGRATION.md` - Event flow diagrams
- `archive/` - Deprecated modules with migration guides
- `../CLAUDE.md` - Root project context

## Training Loop Improvements (Complete - Dec 30, 2025)

All training loop feedback mechanisms are fully implemented:

| Feature                                 | Location                             | Status      |
| --------------------------------------- | ------------------------------------ | ----------- |
| Quality scores in NPZ export            | `export_replay_dataset.py:1186`      | ✅ Complete |
| Quality-weighted training               | `train.py:3447-3461`                 | ✅ Complete |
| PLATEAU_DETECTED handler                | `feedback_loop_controller.py:1006`   | ✅ Complete |
| Loss anomaly handler                    | `feedback_loop_controller.py:897`    | ✅ Complete |
| DataPipeline → SelfplayScheduler wiring | `data_pipeline_orchestrator.py:1697` | ✅ Complete |
| Elo velocity tracking                   | `selfplay_scheduler.py:3374`         | ✅ Complete |
| Exploration boost emission              | `feedback_loop_controller.py:1048`   | ✅ Complete |

Expected Elo improvement: **+28-45 Elo** across all configs from these feedback loops.
