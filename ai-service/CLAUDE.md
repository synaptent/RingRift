# CLAUDE.md - AI Assistant Context for ai-service

AI assistant context for the Python AI training service. Complements `AGENTS.md` with operational knowledge.

**Last Updated**: April 12, 2026

## Infrastructure Health Snapshot

| Component                 | Status   | Evidence                                                                    |
| ------------------------- | -------- | --------------------------------------------------------------------------- |
| **P2P Control Plane**     | GREEN    | 7/7 GH200 fleet healthy with role-aware deployment and active sync/health   |
| **Training Loop**         | GREEN    | 4 trainer nodes on `minimal_alphazero_loop.py` under systemd                |
| **Policy Selfplay**       | GREEN    | 2 selfplay-worker nodes generating policy-bearing Gumbel JSONL              |
| **Supplemental Pipeline** | PROVING  | Worker shard landing proven on trainers; trainer merge occurs on next cycle |
| **Evaluator**             | ACTIVE   | 1 dedicated evaluator node in the fleet                                     |
| **Code Quality**          | VERIFIED | 33k+ Python tests, cycle-free import audit, role-based autonomy runtime     |

**Key Metrics:**

- 7 GH200 nodes in active role-based deployment
- 4 trainers, 2 selfplay workers, 1 evaluator
- fixed learning rate `5e-5` is the current proven baseline on the supported path
- `hex8_2p` reached `1979.8` Elo on the current minimal-loop line
- 33k+ Python tests in the core suite

## Where To Start Reading

For current-state orientation, use these entrypoints first:

- `../AGENTS.md` for repository invariants and canonical rules/parity expectations
- `../docs/architecture/OVERVIEW.md` for current architecture docs
- `scripts/README.md` for supported Python operational scripts
- `data/sync_policy.yaml` for coordinator remote-to-internal rehydration policy

## Coordinator Sync Safety

Remote-to-internal sync is push-only by default. OWC/S3/P2P pulls that write to
the coordinator's internal disk must pass `app.coordination.sync_policy`, carry
an explicit downstream consumer signal, and satisfy the allowlist in
`data/sync_policy.yaml`. Inactive gauntlet/evaluation DBs are not rehydrated by
default. `master_loop.py` also has a per-process RSS/uptime self-guard; defaults
are `RINGRIFT_MASTER_LOOP_RSS_BUDGET_GB=20` and
`RINGRIFT_MASTER_LOOP_MAX_UPTIME_HOURS=24`.

## Project Overview

RingRift is a multiplayer territory control game. The Python `ai-service` mirrors the TypeScript engine (`src/shared/engine/`) for training data generation and must maintain **parity** with it.

| Board Type  | Grid      | Cells | Players |
| ----------- | --------- | ----- | ------- |
| `square8`   | 8x8       | 64    | 2,3,4   |
| `square19`  | 19x19     | 361   | 2,3,4   |
| `hex8`      | radius 4  | 61    | 2,3,4   |
| `hexagonal` | radius 12 | 469   | 2,3,4   |

## Quick Start

```bash
# Deploy the role-based fleet runtime
cd ai-service
bash scripts/deploy_training_service.sh

# Standalone trainer loop
PYTHONPATH=. python scripts/minimal_alphazero_loop.py \
  --model models/canonical_hex8_2p.pth --board-type hex8 --num-players 2 \
  --iterations 50 --games-per-iter 100 --eval-games 50 --budget 128

# Standalone policy selfplay worker
PYTHONPATH=. python scripts/policy_selfplay_worker.py --help

# Fleet status
PYTHONPATH=. python scripts/autonomy_fleet_check.py

# TS ↔ Python parity
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8_2p.db
```

## Verifying NN Strength Improvement

The master loop demonstrates iterative neural network improvement through Elo rating progression. Use these tools to verify that training produces stronger models over time.

### Progress Report CLI

```bash
# Show Elo progress for all 12 configs
python scripts/elo_progress_report.py

# Filter to specific config
python scripts/elo_progress_report.py --config hex8_2p

# JSON output for scripting
python scripts/elo_progress_report.py --format json

# Custom lookback period (default: 30 days)
python scripts/elo_progress_report.py --days 7
```

**Sample output:**

```
ELO PROGRESS REPORT - Demonstrating Iterative NN Strength Improvement
Generated: 2026-01-16 | Target: 2000.0 Elo

Config           Start  Current    Delta  Iters   Progress Status
---------------------------------------------------------------------------
hex8_2p           1355     1409      +54      9      40.9% IMPROVING
hex8_3p           1590     1610      +20     22      61.0% IMPROVING
...
SUMMARY                     1235       +4     87

Configs improving: 4/12
Avg Elo gain per iteration: +6.0
```

### Progress Endpoint

Query the P2P leader for Elo progress:

```bash
# Get progress for all configs
curl http://localhost:8770/progress | jq '.'

# Filter to specific config
curl "http://localhost:8770/progress?config=hex8_2p" | jq '.'
```

### Key Metrics

| Metric             | Description                    | Target   |
| ------------------ | ------------------------------ | -------- |
| `current_elo`      | Best model's Elo rating        | 2000+    |
| `delta`            | Improvement from starting Elo  | Positive |
| `iterations`       | Training generations completed | Growing  |
| `progress_percent` | Progress toward 2000 Elo       | 100%     |

### Web Dashboard

**Local Development:**

```bash
python scripts/dashboard_server.py --port 8080
```

Then open http://localhost:8080/progress_dashboard.html

**Production Deployment (ringrift.ai):**

1. Copy ai-service to production server
2. Install dependencies:
   ```bash
   pip install flask flask-cors
   ```
3. Start with PM2:
   ```bash
   cd /home/ubuntu/ringrift/ai-service
   pm2 start ecosystem.dashboard.config.js
   ```
4. Reload nginx (config already includes `/ai-dashboard/` proxy):
   ```bash
   sudo nginx -t && sudo nginx -s reload
   ```
5. Access at: https://ringrift.ai/ai-dashboard/progress_dashboard.html

**Available dashboards on ringrift.ai:**

- `/ai-dashboard/progress_dashboard.html` - Elo progress and training iterations
- `/ai-dashboard/elo_dashboard.html` - Detailed Elo ratings
- `/ai-dashboard/model_dashboard.html` - Model performance overview
- `/ai-dashboard/work_queue_dashboard.html` - Work queue status
- `/ai-dashboard/training_dashboard.html` - Training metrics

The dashboard shows:

- Summary stats (total iterations, configs improving, average Elo)
- Config cards with current Elo, delta, and progress bar
- Elo progression chart over time (7/14/30/90 days)

### Tracking Infrastructure

- **`data/elo_progress.db`**: Elo snapshots over time
- **`data/generation_tracking.db`**: Training iteration history
- **`EloProgressTracker`**: Records snapshots on EVALUATION_COMPLETED
- **`GenerationTracker`**: Tracks model lineage and training data

## Cluster Infrastructure

Active runtime fleet:

| Role            | Nodes | Workload                                                    |
| --------------- | ----- | ----------------------------------------------------------- |
| Trainer         | 4     | `minimal_alphazero_loop.py` with `--supplemental-data-dir`  |
| Selfplay worker | 2     | `policy_selfplay_worker.py` generating Gumbel policy shards |
| Evaluator       | 1     | evaluation / gauntlet services                              |

## Key Modules

### Configuration

| Module                                | Purpose                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `app/config/env.py`                   | Typed environment variables (`env.node_id`, `env.log_level`)   |
| `app/config/cluster_config.py`        | Cluster node access (`get_cluster_nodes()`, `get_gpu_nodes()`) |
| `app/config/coordination_defaults.py` | Centralized timeouts, thresholds, priority weights             |
| `app/config/thresholds.py`            | Centralized quality/training/budget thresholds (canonical)     |

### Coordination Infrastructure (276 modules, 235K LOC)

| Module                           | Purpose                                           |
| -------------------------------- | ------------------------------------------------- |
| `daemon_manager.py`              | Lifecycle for 132 daemon types (~2,000 LOC)       |
| `daemon_registry.py`             | Declarative daemon specs (DaemonSpec dataclass)   |
| `daemon_runners.py`              | 124 async runner functions                        |
| `event_router.py`                | Unified event bus (292 event types, SHA256 dedup) |
| `selfplay_scheduler.py`          | Priority-based selfplay allocation (~3,800 LOC)   |
| `budget_calculator.py`           | Gumbel budget tiers, target games calculation     |
| `progress_watchdog_daemon.py`    | Stall detection for 48h autonomous operation      |
| `p2p_recovery_daemon.py`         | P2P cluster health recovery                       |
| `data_pipeline_orchestrator.py`  | Pipeline stage tracking                           |
| `feedback_loop_controller.py`    | Training feedback signals                         |
| `health_facade.py`               | Unified health check API                          |
| `quality_monitor_daemon.py`      | Monitors selfplay data quality, emits events      |
| `training_trigger_daemon.py`     | Automatic training decision logic                 |
| `evaluation_daemon.py`           | Model evaluation with retry and backpressure      |
| `unified_distribution_daemon.py` | Model and NPZ distribution to cluster             |
| `training_coordinator.py`        | Training job management and coordination          |

### AI Components

| Module                                       | Purpose                                           |
| -------------------------------------------- | ------------------------------------------------- |
| `app/ai/gpu_parallel_games.py`               | Vectorized GPU selfplay (6-57x speedup)           |
| `app/ai/gumbel_search_engine.py`             | Unified MCTS entry point                          |
| `app/ai/gumbel_common.py`                    | Shared Gumbel data structures, budget tiers       |
| `app/ai/harness/`                            | Harness abstraction layer                         |
| `app/ai/nnue.py`                             | NNUE evaluation network (~256 hidden)             |
| `app/ai/neural_net/architecture_registry.py` | Encoder/model channel mapping (v2/v3/v4/v5-heavy) |

**Architecture Registry** (`app/ai/neural_net/architecture_registry.py`):

Single source of truth for mapping neural net architectures to their encoders:

| Channels | Architecture | Encoder               | Description                      |
| -------- | ------------ | --------------------- | -------------------------------- |
| 40       | v2           | HexStateEncoder       | 10 base × 4 frames (standard)    |
| 64       | v3/v4        | HexStateEncoderV3     | 16 base × 4 frames (enhanced)    |
| 56       | v5-heavy     | HexStateEncoderV5     | 14 base × 4 frames (heuristics)  |
| 36       | v2-lite      | HexStateEncoder       | 12 base × 3 frames (lightweight) |
| 44       | v3-lite      | HexStateEncoderV3Lite | 12 base × 3 frames + 8 extras    |

```python
from app.ai.neural_net.architecture_registry import (
    get_encoder_for_model,        # Auto-detect encoder from model weights
    get_encoder_class_for_channels,  # Get encoder class by channel count
    validate_encoder_model_match,    # Verify encoder/model compatibility
)

# Auto-detect correct encoder for a loaded model
encoder = get_encoder_for_model(loaded_model)

# Manual validation
is_valid, error = validate_encoder_model_match(encoder, model)
```

**Harness Types** (for model evaluation):

| Type          | NN  | NNUE  | Policy   | Best For                    |
| ------------- | --- | ----- | -------- | --------------------------- |
| `GUMBEL_MCTS` | Yes | -     | Required | Training data, high quality |
| `GPU_GUMBEL`  | Yes | -     | Required | High throughput selfplay    |
| `MINIMAX`     | Yes | Yes   | -        | 2-player, fast evaluation   |
| `MAXN`        | Yes | Yes   | -        | 3-4 player multiplayer      |
| `POLICY_ONLY` | Yes | Yes\* | Required | Baselines, fast play        |
| `HEURISTIC`   | -   | -     | -        | Baselines, bootstrap        |

**Gumbel Budget Tiers** (game count -> simulation budget):

| Game Count | Budget    | Purpose                           |
| ---------- | --------- | --------------------------------- |
| <100       | 64        | Bootstrap tier 1 - max throughput |
| <500       | 150       | Bootstrap tier 2 - fast iteration |
| <1000      | 200       | Bootstrap tier 3 - balanced       |
| >=1000     | Elo-based | STANDARD/QUALITY/ULTIMATE/MASTER  |

### Base Classes

| Class            | Location                        | Purpose                                       |
| ---------------- | ------------------------------- | --------------------------------------------- |
| `HandlerBase`    | `handler_base.py`               | Event-driven handlers (1,300+ LOC, 45+ tests) |
| `MonitorBase`    | `monitor_base.py`               | Health monitoring daemons (800 LOC, 41 tests) |
| `SingletonMixin` | `singleton_mixin.py`            | Singleton pattern (503 LOC)                   |
| `P2PMixinBase`   | `scripts/p2p/p2p_mixin_base.py` | P2P mixin utilities (995 LOC)                 |

## Daemon System

127 daemon types (107 active, 20 deprecated). Three-layer architecture:

1. **`daemon_registry.py`** - Declarative `DAEMON_REGISTRY: Dict[DaemonType, DaemonSpec]`
2. **`daemon_manager.py`** - Lifecycle coordinator (start/stop, health, auto-restart)
3. **`daemon_runners.py`** - 124 async runner functions

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()
await dm.start(DaemonType.AUTO_SYNC)
health = dm.get_all_daemon_health()
```

**Key Categories:**

| Category   | Daemons                                                          |
| ---------- | ---------------------------------------------------------------- |
| Sync       | AUTO_SYNC, MODEL_DISTRIBUTION, ELO_SYNC, GOSSIP_SYNC, OWC_IMPORT |
| Pipeline   | DATA_PIPELINE, SELFPLAY_COORDINATOR, TRAINING_NODE_WATCHER       |
| Health     | NODE_HEALTH_MONITOR, QUALITY_MONITOR, NODE_AVAILABILITY          |
| Resources  | IDLE_RESOURCE, NODE_RECOVERY                                     |
| Autonomous | PROGRESS_WATCHDOG, P2P_RECOVERY, STALE_FALLBACK, MEMORY_MONITOR  |

## Event System

292 event types across 3 layers:

1. **In-memory EventBus** - Local daemon communication
2. **Stage events** - Pipeline stage completion
3. **Cross-process queue** - Cluster-wide events

**Critical Event Flows:**

```
Selfplay -> NEW_GAMES_AVAILABLE -> DataPipeline -> TRAINING_THRESHOLD_REACHED
    -> Training -> TRAINING_COMPLETED -> Evaluation -> EVALUATION_COMPLETED
    -> MODEL_PROMOTED -> Distribution -> Curriculum rebalance
```

**Key Events:**

| Event                     | Emitter             | Subscribers                                       |
| ------------------------- | ------------------- | ------------------------------------------------- |
| `TRAINING_COMPLETED`      | TrainingCoordinator | FeedbackLoop, DataPipeline, UnifiedQueuePopulator |
| `MODEL_PROMOTED`          | PromotionController | UnifiedDistributionDaemon, CurriculumIntegration  |
| `DATA_SYNC_COMPLETED`     | AutoSyncDaemon      | DataPipelineOrchestrator, TransferVerification    |
| `REGRESSION_DETECTED`     | RegressionDetector  | TrainingCoordinator, UnifiedFeedback              |
| `EVALUATION_BACKPRESSURE` | EvaluationDaemon    | TrainingCoordinator (pauses training)             |

```python
from app.coordination.event_router import emit_event
from app.coordination.data_events import DataEventType

emit_event(DataEventType.TRAINING_COMPLETED, {"config_key": "hex8_2p", "model_path": "models/canonical_hex8_2p.pth"})
```

## Common Patterns

### Checking Game Progress (Cluster-Wide)

**ALWAYS use cluster-wide counts** - the coordinator has minimal local data.

```bash
# Correct - cluster-wide view (Local + Cluster + OWC + S3)
python scripts/cluster_data_status.py

# Correct - specific config
python scripts/cluster_data_status.py --config hex8_2p

# Correct - programmatic access
python3 -c "
from app.distributed.data_catalog import get_data_registry
registry = get_data_registry()
status = registry.get_cluster_status()
for config, counts in sorted(status.items()):
    print(f'{config}: total={counts[\"total\"]:,} (local={counts[\"local\"]}, cluster={counts[\"cluster\"]}, owc={counts[\"owc\"]})')
"

# Check if cluster data is available
python3 -c "
from app.distributed.data_catalog import get_data_registry
avail, reason = get_data_registry().is_cluster_data_available()
print(f'Cluster data available: {avail} ({reason})')
"
```

**WRONG** - Local-only queries miss cluster/OWC/S3 data:

```bash
# DON'T DO THIS - only shows coordinator's local games
python3 -c "import sqlite3; conn = sqlite3.connect('data/games/canonical_hex8_2p.db'); ..."
```

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

    async def _run_cycle(self) -> None:
        pass

    def _get_event_subscriptions(self) -> dict:
        return {"MY_EVENT": self._on_my_event}
```

### Health Checks

```python
from app.coordination.health_facade import get_health_orchestrator, get_system_health_score
score = get_system_health_score()  # 0.0-1.0
orchestrator = get_health_orchestrator()
```

**HTTP Health Endpoints:**

| Port | Endpoint   | Purpose                                    |
| ---- | ---------- | ------------------------------------------ |
| 8790 | `/health`  | Liveness probe (daemon manager running)    |
| 8790 | `/ready`   | Readiness probe (critical daemons healthy) |
| 8790 | `/metrics` | Prometheus-style metrics                   |
| 8790 | `/status`  | Detailed daemon status JSON                |
| 8770 | `/status`  | P2P cluster status (leader, alive peers)   |

## Environment Variables

### Core

| Variable                      | Default  | Purpose                     |
| ----------------------------- | -------- | --------------------------- |
| `RINGRIFT_NODE_ID`            | hostname | Node identifier             |
| `RINGRIFT_LOG_LEVEL`          | INFO     | Logging level               |
| `RINGRIFT_IS_COORDINATOR`     | false    | Coordinator node flag       |
| `RINGRIFT_ALLOW_PENDING_GATE` | false    | Bypass TS parity validation |

### Selfplay Priority Weights

| Variable                       | Default | Purpose                  |
| ------------------------------ | ------- | ------------------------ |
| `RINGRIFT_STALENESS_WEIGHT`    | 0.30    | Weight for stale configs |
| `RINGRIFT_ELO_VELOCITY_WEIGHT` | 0.20    | Weight for Elo velocity  |
| `RINGRIFT_CURRICULUM_WEIGHT`   | 0.20    | Weight for curriculum    |
| `RINGRIFT_QUALITY_WEIGHT`      | 0.15    | Weight for data quality  |

## 48-Hour Autonomous Operation

The cluster can run unattended for 48+ hours with these daemons:

| Daemon              | Purpose                                  |
| ------------------- | ---------------------------------------- |
| `PROGRESS_WATCHDOG` | Detects Elo stalls, triggers recovery    |
| `P2P_RECOVERY`      | Restarts unhealthy P2P orchestrator      |
| `STALE_FALLBACK`    | Uses older models when sync fails        |
| `MEMORY_MONITOR`    | Prevents OOM via proactive VRAM tracking |

**4-Layer Resilience Architecture:**

| Layer | Component                     | Purpose                                          |
| ----- | ----------------------------- | ------------------------------------------------ |
| 1     | Sentinel + Watchdog           | OS-level process supervision (launchd/systemd)   |
| 2     | MemoryPressureController      | Proactive memory management (60/70/80/90% tiers) |
| 3     | StandbyCoordinator            | Primary/standby coordinator failover             |
| 4     | ClusterResilienceOrchestrator | Unified health aggregation (30/30/25/15 weights) |

**Memory Pressure Tiers:**

| Tier      | RAM % | Action                                 |
| --------- | ----- | -------------------------------------- |
| CAUTION   | 60%   | Log warning, emit event                |
| WARNING   | 70%   | Pause selfplay, reduce batch sizes     |
| CRITICAL  | 80%   | Kill non-essential daemons, trigger GC |
| EMERGENCY | 90%   | Graceful shutdown, notify standby      |

**Resilience Features:**

- Adaptive circuit breaker cascade prevention
- Multi-transport failover: Tailscale -> SSH -> Base64 -> HTTP
- Stale training fallback after 5 sync failures or 45min timeout
- Automatic parity gate bypass on cluster nodes without Node.js
- Handler timeout protection (600s default)

## Neural Network Architectures

| Version          | Class Name            | Parameters | Description                     |
| ---------------- | --------------------- | ---------- | ------------------------------- |
| `v2`             | HexNeuralNet_v2       | ~2-4M      | Standard architecture (default) |
| `v4`             | RingRiftCNN_v4        | ~3-5M      | Improved residual blocks        |
| `v5-heavy`       | HexNeuralNet_v5_Heavy | ~8-12M     | Wider with heuristic features   |
| `v5-heavy-large` | HexNeuralNet_v5_Heavy | ~25-35M    | Scaled v5-heavy (256 filters)   |

## Transfer Learning (2p -> 4p)

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
5. **Sandbox AI in production**: To enable sandbox AI play in production, set `RINGRIFT_AI_SERVICE_URL=<url>` (e.g., `http://ai-service:8000`). Without this, the client disables AI endpoints to prevent 404 errors.
6. **Lambda nodes run Python 3.10**: Do not use `datetime.UTC` (3.11+), use `datetime.timezone.utc` instead.
7. **56-channel ambiguity**: Square v2 and hex v5-heavy both produce 56 input channels. The architecture registry prefers model metadata over channel-count inference to resolve this.

## Structural Quality Improvements (April 2026)

| Improvement                  | Description                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **BoardEncodingContract**    | Single source of truth for (board_type, model_version) → expected channels. 14 contract tests in CI. |
| **Lean daemon profile**      | `--profile lean` starts 23 essential daemons instead of 47. Cuts RSS from 38-49GB to 200-900MB.      |
| **Encoding rename**          | `hex_in_channels` renamed to `encoding_channels` — applies to ALL board types, not just hex.         |
| **Exception tightening**     | Bare `except Exception` replaced with specific types in training_executor, game_gauntlet.            |
| **Silent failure detection** | evaluation_daemon upgraded from `logger.debug` to `logger.warning` with stack traces.                |
| **20 deprecated daemons**    | Formally deprecated with no-op stubs and helpful migration messages.                                 |

## Training Fixes (Jan 2026)

Critical fixes for NN training quality:

| Fix                | File                          | Change                                              |
| ------------------ | ----------------------------- | --------------------------------------------------- |
| Simulation budget  | `gumbel_search_engine.py:394` | `for_throughput()` → `for_selfplay()` (64→800 sims) |
| Default budget     | `selfplay_config.py:207`      | `simulation_budget: int = 800` (was None)           |
| Opponent diversity | `train_loop.py`               | 50% heuristic, 30% descent, 20% weak                |
| Curriculum stages  | `train_loop.py`               | Progressive difficulty based on games played        |
| Loss monitoring    | `train.py`                    | `LossMonitor` class for stall detection             |

**Root cause**: Selfplay was using 64 simulations (throughput mode) instead of 800 (quality mode), producing garbage training data. Combined with no opponent diversity, the NNs were stuck in a weak-vs-weak cycle.

## File Structure

```
ai-service/
├── app/
│   ├── ai/              # Neural net, MCTS, heuristics
│   ├── config/          # Environment, cluster, thresholds
│   ├── coordination/    # Daemons, events, pipeline (190+ modules)
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

## Deprecated Modules (Removal: Q2 2026)

| Deprecated                     | Replacement                            |
| ------------------------------ | -------------------------------------- |
| `cluster_data_sync.py`         | `AutoSyncDaemon(strategy="broadcast")` |
| `ephemeral_sync.py`            | `AutoSyncDaemon(strategy="ephemeral")` |
| `system_health_monitor.py`     | `unified_health_manager.py`            |
| `node_health_monitor.py`       | `health_check_orchestrator.py`         |
| `queue_populator_daemon.py`    | `unified_queue_populator.py`           |
| `model_distribution_daemon.py` | `unified_distribution_daemon.py`       |

## See Also

- `AGENTS.md` - Coding guidelines
- `SECURITY.md` - Security considerations
- `docs/DAEMON_REGISTRY.md` - Full daemon reference
- `docs/architecture/EVENT_SUBSCRIPTION_MATRIX.md` - Event emitter/subscriber matrix
- `archive/` - Deprecated modules with migration guides
- `../CLAUDE.md` - Root project context

## Archives

Session history and exploration findings have been archived:

- `docs/archive/SPRINT_17_SESSION_HISTORY.md` - Sprint 17 session-by-session logs
- `docs/archive/DEC_2025_EXPLORATION_FINDINGS.md` - Exploration agent findings and improvement priorities
