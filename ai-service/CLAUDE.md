# CLAUDE.md - AI Assistant Context for ai-service

AI assistant context for the Python AI training service. Complements `AGENTS.md` with operational knowledge.

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

### Coordination Infrastructure (190+ modules)

| Module                          | Purpose                                           |
| ------------------------------- | ------------------------------------------------- |
| `daemon_manager.py`             | Lifecycle for 77 daemon types (~2,000 LOC)        |
| `daemon_registry.py`            | Declarative daemon specs (DaemonSpec dataclass)   |
| `daemon_runners.py`             | 62 async runner functions                         |
| `event_router.py`               | Unified event bus (118 event types, SHA256 dedup) |
| `selfplay_scheduler.py`         | Priority-based selfplay allocation (~3,800 LOC)   |
| `budget_calculator.py`          | Gumbel budget tiers, target games calculation     |
| `progress_watchdog_daemon.py`   | Stall detection for 48h autonomous operation      |
| `p2p_recovery_daemon.py`        | P2P cluster health recovery                       |
| `stale_fallback.py`             | Graceful degradation with older models            |
| `data_pipeline_orchestrator.py` | Pipeline stage tracking                           |
| `auto_sync_daemon.py`           | P2P data synchronization                          |
| `sync_router.py`                | Intelligent sync routing                          |
| `feedback_loop_controller.py`   | Training feedback signals                         |
| `health_facade.py`              | Unified health check API                          |

### AI Components

| Module                           | Purpose                                     |
| -------------------------------- | ------------------------------------------- |
| `app/ai/gpu_parallel_games.py`   | Vectorized GPU selfplay (6-57× speedup)     |
| `app/ai/gumbel_search_engine.py` | Unified MCTS entry point                    |
| `app/ai/gumbel_common.py`        | Shared Gumbel data structures, budget tiers |

### Base Classes

| Class                  | Location                        | Purpose                                       |
| ---------------------- | ------------------------------- | --------------------------------------------- |
| `HandlerBase`          | `handler_base.py`               | Event-driven handlers (550 LOC, 45 tests)     |
| `MonitorBase`          | `monitor_base.py`               | Health monitoring daemons (800 LOC, 41 tests) |
| `P2PMixinBase`         | `scripts/p2p/p2p_mixin_base.py` | P2P mixin utilities (995 LOC)                 |
| `SingletonMixin`       | `singleton_mixin.py`            | Singleton pattern (503 LOC)                   |
| `CircuitBreakerConfig` | `transport_base.py`             | Circuit breaker configuration (canonical)     |

## Daemon System

77 active daemon types, 6 deprecated. Three-layer architecture:

1. **`daemon_registry.py`** - Declarative `DAEMON_REGISTRY: Dict[DaemonType, DaemonSpec]`
2. **`daemon_manager.py`** - Lifecycle coordinator (start/stop, health, auto-restart)
3. **`daemon_runners.py`** - 62 async runner functions

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
| Sync | AUTO_SYNC, MODEL_DISTRIBUTION, ELO_SYNC |
| Pipeline | DATA_PIPELINE, SELFPLAY_COORDINATOR |
| Health | NODE_HEALTH_MONITOR, QUALITY_MONITOR, NODE_AVAILABILITY |
| Resources | IDLE_RESOURCE, NODE_RECOVERY |
| Autonomous | PROGRESS_WATCHDOG, P2P_RECOVERY, STALE_FALLBACK, MEMORY_MONITOR |

**Health Monitoring (85%+ coverage):**

```python
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)
# {"status": "healthy", "running": True, "last_sync": ..., "errors_count": 0}
```

## Event System

118 event types across 3 layers:

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
| `TRAINING_COMPLETED` | TrainingCoordinator | FeedbackLoop, DataPipeline |
| `MODEL_PROMOTED` | PromotionController | UnifiedDistributionDaemon |
| `DATA_SYNC_COMPLETED` | AutoSyncDaemon | DataPipelineOrchestrator |
| `REGRESSION_DETECTED` | RegressionDetector | ModelLifecycleCoordinator |

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

## Known Issues

1. **Parity gates on cluster**: Nodes lack `npx`, so TS validation fails. Set `RINGRIFT_ALLOW_PENDING_GATE=1`.
2. **Board conventions**: Hex boards use radius. hex8 = radius 4 = 61 cells.
3. **GPU memory**: v2 models with batch_size=512 need ~8GB VRAM.
4. **PYTHONPATH**: Set `PYTHONPATH=.` when running scripts from ai-service directory.
5. **Container networking**: Vast.ai/RunPod need `container_tailscale_setup.py` for mesh connectivity.

## See Also

- `AGENTS.md` - Coding guidelines
- `SECURITY.md` - Security considerations
- `docs/DAEMON_REGISTRY.md` - Full daemon reference
- `docs/EVENT_SYSTEM_REFERENCE.md` - Complete event documentation
- `archive/` - Deprecated modules with migration guides
- `../CLAUDE.md` - Root project context
