# P2P Manager Modules Architecture

## Overview

The `scripts/p2p/managers/` directory contains domain-specific manager classes extracted from the monolithic `p2p_orchestrator.py` as part of the **Phase 2B refactoring** (December 2025). This decomposition improves modularity, testability, and maintainability of the P2P orchestration layer.

**Current Status (December 28, 2025)**: All 7 managers fully delegated (100% coverage), ~1,990 LOC removed from p2p_orchestrator.py.

---

## P2PEventMixin and Helper Classes

### EventSubscriptionMixin (`p2p_mixin_base.py`)

The `EventSubscriptionMixin` class provides standardized event subscription for all P2P managers. It consolidates ~100 LOC of duplicated event subscription patterns found across 6 manager files.

**Features:**

- Thread-safe double-checked locking for subscription
- Safe event router import with graceful fallback
- Declarative event subscription via `_get_event_subscriptions()`
- Health check integration for subscription status

**Usage:**

```python
from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

class MyManager(EventSubscriptionMixin):
    _subscription_log_prefix = "MyManager"

    def __init__(self):
        self._init_subscription_state()

    def _get_event_subscriptions(self) -> dict:
        """Return mapping of event types to handlers."""
        return {
            "HOST_OFFLINE": self._on_host_offline,
            "NODE_RECOVERED": self._on_node_recovered,
            "TRAINING_COMPLETED": self._on_training_completed,
        }

    async def _on_host_offline(self, event) -> None:
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "")
        self._log_info(f"Host went offline: {node_id}")

# Subscribe during initialization
manager = MyManager()
manager.subscribe_to_events()
```

**Helper Methods:**

| Method                               | Purpose                                     |
| ------------------------------------ | ------------------------------------------- |
| `_init_subscription_state()`         | Initialize `_subscribed` flag and lock      |
| `subscribe_to_events()`              | Thread-safe subscription to declared events |
| `is_subscribed()`                    | Check subscription status                   |
| `get_subscription_status()`          | Get status for health check inclusion       |
| `_extract_event_payload(event)`      | Safely extract payload from event object    |
| `_safe_emit_event(type, payload)`    | Emit event with error handling              |
| `_log_info/debug/warning/error(msg)` | Prefixed logging helpers                    |

### P2PMixinBase (`p2p_mixin_base.py`)

Base class providing shared functionality for all P2P orchestrator mixins:

- Database connection helpers with automatic cleanup
- State initialization patterns
- Peer alive counting
- Event emission with error handling
- Configuration constant loading

### P2PManagerBase (`p2p_mixin_base.py`)

Combined base class inheriting from both `P2PMixinBase` and `EventSubscriptionMixin`:

```python
from scripts.p2p.p2p_mixin_base import P2PManagerBase

class MyManager(P2PManagerBase):
    MIXIN_TYPE = "my_manager"
    _subscription_log_prefix = "MyManager"

    def __init__(self, ...):
        self._init_subscription_state()

    def _get_event_subscriptions(self) -> dict:
        return {"HOST_OFFLINE": self._on_host_offline}

    def health_check(self) -> dict:
        status = "healthy"
        sub_status = self.get_subscription_status()
        if not sub_status["subscribed"]:
            status = "degraded"
        return {
            "status": status,
            "manager": self.MIXIN_TYPE,
            **sub_status,
        }
```

---

## Modules

### 1. StateManager (`state_manager.py`)

**Purpose**: SQLite persistence for P2P orchestrator state.

**Responsibilities**:

- Database initialization and schema management (WAL mode, busy timeout)
- Loading and saving peers, jobs, and leader election state
- Cluster epoch tracking for split-brain resolution
- Thread-safe database operations with explicit locking

**Key Classes**:

- `StateManager` - Main persistence manager
- `PersistedState` - Dataclass containing peers, jobs, and leader state
- `PersistedLeaderState` - Leader election state (lease, role, voter grants)

**Database Tables**:

- `peers` - Node information
- `jobs` - Running job records
- `state` - Key-value state (leader, role)
- `metrics_history` - Observability metrics
- `ab_tests` / `ab_test_games` - A/B testing experiments
- `peer_cache` - Persistent peer storage with reputation
- `config` - Cluster epoch and settings

---

### 2. NodeSelector (`node_selector.py`)

**Purpose**: Node ranking and selection for job dispatch.

**Responsibilities**:

- Rank nodes by GPU processing power for training priority
- Rank nodes by CPU processing power for data processing
- Select best node for specific tasks (training, gauntlet, export)
- Filter nodes by health, availability, and capabilities

**Key Methods**:

- `get_training_primary_nodes(count)` - Top N GPU nodes for training
- `get_training_nodes_ranked()` - All GPU nodes with power rankings
- `get_best_gpu_node_for_training()` - Single best node for NN training
- `get_best_cpu_node_for_export()` - Best node for data export tasks

**GPU Power Scoring**:
Nodes are ranked by GPU power score: H100 > GH200 > A100 > L40S > RTX 4090 > RTX 3090 > consumer GPUs.

---

### 3. SyncPlanner (`sync_planner.py`)

**Purpose**: Data synchronization planning and execution.

**Responsibilities**:

- Local manifest collection (scanning data directory)
- Cluster manifest aggregation (collecting from all peers)
- Sync plan generation (identifying missing files)
- Sync job dispatch and tracking

**Key Classes**:

- `SyncPlanner` - Main sync orchestrator
- `SyncPlannerConfig` - Configuration (cache age, intervals, file limits)
- `SyncStats` - Sync operation statistics

**Configuration Defaults**:

```python
manifest_cache_age_seconds = 300    # 5 minutes
manifest_collection_interval = 60   # 1 minute
max_files_per_sync_job = 50
sync_mtime_tolerance_seconds = 60   # Clock skew tolerance
```

**Phase 2A Extraction** (December 2025): First module extracted as part of god-class decomposition.

---

### 4. JobManager (`job_manager.py`)

**Purpose**: Job spawning and lifecycle management.

**Responsibilities**:

- Spawn selfplay jobs (GPU, hybrid, heuristic)
- Spawn training jobs (local and distributed)
- Spawn export and tournament jobs
- Track running jobs per node
- Monitor job status and cleanup

**Engine Mode Routing**:

```python
# Search modes -> generate_gumbel_selfplay.py
SEARCH_ENGINE_MODES = {"maxn", "brs", "mcts", "gumbel-mcts", "policy-only", "nn-descent", "nn-minimax"}

# Simple modes -> run_gpu_selfplay.py (GPU-optimized)
# random, heuristic, nnue-guided
```

**Key Methods**:

- `run_gpu_selfplay_job()` - Spawn selfplay with automatic script selection
- `get_job_count_for_node()` - Count running jobs on a node
- `cleanup_stale_jobs()` - Remove orphaned job records

---

### 5. SelfplayScheduler (`selfplay_scheduler.py`)

**Purpose**: Priority-based selfplay configuration selection and diversity tracking.

**Responsibilities**:

- Weighted config selection combining multiple priority sources
- Job targeting per node based on hardware and utilization
- Diversity tracking for monitoring
- Integration with backpressure and resource optimization

**Priority Sources**:

1. **Static priority**: Base priority per config (3-8)
2. **Elo-based boost**: +0 to +3 based on model performance
3. **Curriculum weights**: -2 to +3 based on learning curriculum
4. **Board priority overrides**: +0 to +6 from config file

**Key Classes**:

- `SelfplayScheduler` - Main scheduler
- `DiversityMetrics` - Tracking metrics (engine modes, board configs, difficulty)

**Key Methods**:

- `pick_weighted_config(node)` - Select config based on combined priorities
- `get_target_jobs_for_node(node)` - Calculate optimal job count
- `track_diversity(config)` - Record config selection for metrics
- `get_diversity_metrics()` - Get current diversity statistics

See [SELFPLAY_SCHEDULER_USAGE.md](./SELFPLAY_SCHEDULER_USAGE.md) for detailed usage guide.

---

### 6. TrainingCoordinator (`training_coordinator.py`)

**Purpose**: Training job dispatch and completion workflows.

**Responsibilities**:

- Check training readiness based on data thresholds
- Manage training job dispatch and coordination
- Handle training completion workflows (gauntlet evaluation, promotion)
- Prevent duplicate training triggers via hash-based deduplication

**Key Methods**:

- `check_training_readiness()` - Check cluster data for training thresholds
- `dispatch_training_job(config)` - Send training job to best node
- `handle_training_job_completion(job)` - Run gauntlet and promotion workflow
- `_cooldown_ok(job_type, config_key)` - Check training cooldown

**Constants**:

```python
MIN_MEMORY_GB_FOR_TASKS = 8         # Minimum memory for training
LEADERLESS_TRAINING_TIMEOUT = 180   # 3 minutes timeout
```

---

## Dependency Injection Pattern

All managers follow a consistent dependency injection pattern for testability and decoupling:

```python
class Manager:
    def __init__(
        self,
        # Callbacks to access orchestrator state
        get_peers: Callable[[], dict[str, NodeInfo]],
        get_self_info: Callable[[], NodeInfo],
        peers_lock: threading.Lock,

        # Optional feature callbacks
        feature_callback: Callable[..., T] | None = None,

        # Configuration
        config: ManagerConfig | None = None,
        verbose: bool = False,
    ):
        self._get_peers = get_peers
        self._get_self_info = get_self_info
        self._peers_lock = peers_lock
        # ... store other dependencies
```

**Benefits**:

- **Testability**: Managers can be unit tested with mock callbacks
- **Decoupling**: No direct imports of orchestrator internals
- **Flexibility**: Callbacks can return different data sources
- **Composition**: Managers can be composed independently

---

## Integration with p2p_orchestrator.py

Managers are instantiated in `P2POrchestrator.__init__()` and wired with callbacks:

```python
from scripts.p2p.managers import (
    StateManager,
    NodeSelector,
    SyncPlanner,
    JobManager,
    SelfplayScheduler,
    TrainingCoordinator,
)

class P2POrchestrator:
    def __init__(self):
        # Initialize state manager first (needed for loading)
        self.state_manager = StateManager(
            db_path=self.state_db_path,
            verbose=self.verbose,
        )

        # Initialize node selector
        self.node_selector = NodeSelector(
            get_peers=lambda: self.peers,
            get_self_info=lambda: self.self_info,
            peers_lock=self.peers_lock,
        )

        # Initialize sync planner
        self.sync_planner = SyncPlanner(
            node_id=self.node_id,
            data_directory=self.get_data_directory(),
            get_peers=lambda: self.peers,
            get_self_info=lambda: self.self_info,
            peers_lock=self.peers_lock,
            is_leader=lambda: self._is_leader(),
        )

        # Initialize job manager
        self.job_manager = JobManager(
            ringrift_path=self.ringrift_path,
            node_id=self.node_id,
            peers=self.peers,
            peers_lock=self.peers_lock,
            active_jobs=self.active_jobs,
            jobs_lock=self.jobs_lock,
        )

        # Initialize selfplay scheduler
        self.selfplay_scheduler = SelfplayScheduler(
            get_cluster_elo_fn=self._get_cluster_elo_summary,
            load_curriculum_weights_fn=self._load_curriculum_weights,
            verbose=self.verbose,
        )

        # Initialize training coordinator
        self.training_coordinator = TrainingCoordinator(
            ringrift_path=Path(self.ringrift_path),
            get_cluster_data_manifest=lambda: self.cluster_data_manifest,
            get_training_jobs=lambda: self.training_jobs,
            get_training_lock=lambda: self.training_lock,
            get_peers=lambda: self.peers,
            get_peers_lock=lambda: self.peers_lock,
            get_self_info=lambda: self.self_info,
            training_thresholds=self.training_thresholds,
        )
```

---

## Phase 2B Refactoring Status (December 2025)

The managers module is part of the ongoing P2P orchestrator decomposition:

| Phase | Module              | Status   | Description                   |
| ----- | ------------------- | -------- | ----------------------------- |
| 2A    | SyncPlanner         | Complete | Data sync planning extracted  |
| 2B    | StateManager        | Complete | SQLite persistence extracted  |
| 2B    | NodeSelector        | Complete | Node ranking extracted        |
| 2B    | JobManager          | Complete | Job spawning extracted        |
| 2B    | SelfplayScheduler   | Complete | Selfplay scheduling extracted |
| 2B    | TrainingCoordinator | Complete | Training workflows extracted  |

**Original orchestrator size**: ~30,000+ lines
**Per-manager size**: 150-750 lines each

---

## Delegation Status (Updated Dec 28, 2025)

~1,990 LOC removed from `p2p_orchestrator.py` during Dec 27-28 cleanup.

| Manager             | Methods Delegated | Status      | LOC Removed | Notes                           |
| ------------------- | ----------------- | ----------- | ----------- | ------------------------------- |
| StateManager        | 7/7 (100%)        | ✅ Complete | ~200        | SQLite persistence, epochs      |
| NodeSelector        | 6/6 (100%)        | ✅ Complete | ~50         | Node ranking, job placement     |
| TrainingCoordinator | 5/5 (100%)        | ✅ Complete | ~450        | Job dispatch, model promotion   |
| JobManager          | 7/7 (100%)        | ✅ Complete | ~400        | Selfplay, training, tournaments |
| SelfplayScheduler   | 7/7 (100%)        | ✅ Complete | ~430        | Priority scheduling, curriculum |
| SyncPlanner         | 4/4 (100%)        | ✅ Complete | ~60         | Manifest collection, planning   |
| LoopManager         | 5/5 (100%)        | ✅ Complete | ~400        | All background loops migrated   |

**Dec 27-28 Cleanup Highlights**:

- Removed `_dispatch_training_job`, `_handle_training_job_completion`, `_schedule_model_comparison_tournament`, `_run_post_training_gauntlet` (→ TrainingCoordinator)
- Removed `_run_gpu_selfplay_job`, `_run_distributed_tournament`, `_run_distributed_selfplay`, `_export_training_data`, `_run_training`, `_cleanup_old_completed_jobs` (→ JobManager)
- Removed `_target_selfplay_jobs_for_node`, `_get_hybrid_job_targets`, `_should_spawn_cpu_only_jobs`, `_pick_weighted_selfplay_config`, `_get_elo_based_priority_boost`, `_get_diversity_metrics`, `_track_selfplay_diversity` (→ SelfplayScheduler)
- Migrated 5 background loops to LoopManager (`_elo_sync_loop`, `_idle_detection_loop`, `_auto_scaling_loop`, `_job_reaper_loop`, `_queue_populator_loop`)

**All 7 managers 100% delegated** - no remaining partial delegation.

---

## LoopManager and Background Loops

### 7. LoopManager (`../loops/`)

**Purpose**: Centralized management of all P2P background loops.

**Responsibilities**:

- Coordinated start/stop for all loops
- Dependency-ordered startup (topological sort)
- Status aggregation for monitoring
- Graceful shutdown coordination
- Health check aggregation

**Key Classes**:

- `LoopManager` - Central coordinator for all loops
- `BaseLoop` - Abstract base class for loop implementations
- `BackoffConfig` - Exponential backoff configuration
- `LoopStats` - Per-loop execution statistics

**Background Loops (December 2025)**:

| Loop                       | File                          | Interval | Purpose                                    |
| -------------------------- | ----------------------------- | -------- | ------------------------------------------ |
| `JobReaperLoop`            | `job_loops.py`                | 5 min    | Clean stale jobs (1hr), stuck jobs (2hr)   |
| `IdleDetectionLoop`        | `job_loops.py`                | 30 sec   | Detect idle GPUs, trigger selfplay         |
| `WorkerPullLoop`           | `job_loops.py`                | 30 sec   | Workers poll leader for work (pull model)  |
| `WorkQueueMaintenanceLoop` | `job_loops.py`                | 5 min    | Check timeouts, cleanup old items (24hr)   |
| `EloSyncLoop`              | `elo_sync_loop.py`            | 5 min    | Elo rating synchronization                 |
| `QueuePopulatorLoop`       | `queue_populator_loop.py`     | 1 min    | Work queue maintenance                     |
| `SelfHealingLoop`          | `resilience_loops.py`         | 5 min    | Recover stuck jobs, clean stale processes  |
| `PredictiveMonitoringLoop` | `resilience_loops.py`         | 5 min    | Track trends, emit alerts before threshold |
| `ManifestCollectionLoop`   | `manifest_collection_loop.py` | 1 min    | Collect data manifests from peers          |
| `TrainingSyncLoop`         | `training_sync_loop.py`       | 5 min    | Sync training data to training nodes       |

**Usage**:

```python
from scripts.p2p.loops import LoopManager, BaseLoop, JobReaperLoop, IdleDetectionLoop

# Create manager
manager = LoopManager(name="p2p_loops")

# Register loops with dependencies
manager.register(JobReaperLoop(
    get_active_jobs=lambda: orchestrator.active_jobs,
    cancel_job=orchestrator.cancel_job,
))
manager.register(IdleDetectionLoop(
    get_idle_gpus=orchestrator.get_idle_gpus,
    spawn_selfplay=orchestrator.spawn_selfplay,
    depends_on=["job_reaper"],  # Start after JobReaperLoop
))

# Start all loops (respects dependency order)
results = await manager.start_all()

# Monitor health
health = manager.health_check()
print(f"Status: {health['status']}, Loops running: {health['loops_running']}/{health['total_loops']}")

# Graceful shutdown
await manager.stop_all(timeout=30.0)
```

**Creating Custom Loops**:

```python
from scripts.p2p.loops import BaseLoop

class MyCustomLoop(BaseLoop):
    def __init__(self, get_data_fn, process_fn):
        super().__init__(
            name="my_custom_loop",
            interval=60.0,  # Run every 60 seconds
            depends_on=["elo_sync"],  # Optional: start after elo_sync
        )
        self.get_data = get_data_fn
        self.process = process_fn

    async def _run_once(self) -> None:
        """Execute one iteration of the loop."""
        data = self.get_data()
        if data:
            await self.process(data)

    async def _on_start(self) -> None:
        """Called when loop starts."""
        self._log_info("Starting custom loop")

    async def _on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        self._log_error(f"Error: {error}")
```

---

## Health Check Integration with DaemonManager

All managers and loops implement `health_check()` returning results compatible with `DaemonManager` integration.

### Manager Health Check Pattern

```python
def health_check(self) -> HealthCheckResult:
    """Check manager health for DaemonManager integration."""
    try:
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
    except ImportError:
        return {"healthy": True, "status": "running", "message": "OK"}

    # Include subscription status
    sub_status = self.get_subscription_status()
    if not sub_status["subscribed"]:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.DEGRADED,
            message="Event subscriptions not active",
            details=sub_status,
        )

    # Manager-specific health checks
    if self._error_count > 10:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message=f"Too many errors: {self._error_count}",
        )

    return HealthCheckResult(
        healthy=True,
        status=CoordinatorStatus.RUNNING,
        message="Manager operational",
        details={
            **sub_status,
            "jobs_processed": self._stats.jobs_processed,
        },
    )
```

### Health Metrics Reported

| Manager             | Key Health Metrics                                   |
| ------------------- | ---------------------------------------------------- |
| StateManager        | DB connection health, cluster epoch, pending writes  |
| NodeSelector        | Cache freshness, nodes ranked, selection latency     |
| TrainingCoordinator | Training jobs active, cooldown status, last dispatch |
| JobManager          | Active jobs, spawn rate, error count                 |
| SelfplayScheduler   | Active configs, diversity metrics, curriculum state  |
| SyncPlanner         | Sync in progress, manifest freshness, file count     |
| LoopManager         | Loops running, total runs, failing loops             |

### DaemonManager Integration

The P2P orchestrator aggregates health from all managers at the `/status` endpoint:

```python
def health_check(self) -> HealthCheckResult:
    """Aggregate health from all managers."""
    manager_health = self._validate_manager_health()

    if not manager_health["all_healthy"]:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.DEGRADED,
            message=f"Unhealthy managers: {manager_health['unhealthy']}",
            details=manager_health,
        )

    return HealthCheckResult(
        healthy=True,
        status=CoordinatorStatus.RUNNING,
        message="P2P orchestrator operational",
        details={
            "node_id": self.node_id,
            "role": self.role.name,
            "leader_id": self.leader_id,
            "active_peers": len(self.peers),
            "uptime_seconds": time.time() - self._start_time,
        },
    )
```

### Querying Health

```python
# Via P2P /status endpoint
import httpx
response = httpx.get("http://localhost:8770/status")
status = response.json()
print(f"Health: {status['health_check']['status']}")

# Via DaemonManager
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()
health = await dm.get_daemon_health(DaemonType.P2P_BACKEND)
print(f"Status: {health['status']}, Managers: {health['details']['managers']}")
```

---

### Migration Path

1. Import managers: `from scripts.p2p.managers import ManagerName`
2. Initialize in `__init__()` with dependency injection callbacks
3. Replace method calls: `self._method()` -> `self.manager.method()`
4. Remove old methods from orchestrator
5. Add unit tests for isolated manager logic

---

## Module Exports

All managers are exported from `__init__.py`:

```python
from scripts.p2p.managers import (
    DiversityMetrics,
    JobManager,
    NodeSelector,
    SelfplayScheduler,
    StateManager,
    SyncPlanner,
    SyncPlannerConfig,
    SyncStats,
    TrainingCoordinator,
)

# LoopManager and loops from separate package
from scripts.p2p.loops import (
    BackoffConfig,
    BaseLoop,
    LoopManager,
    LoopStats,
    # Individual loops
    EloSyncLoop,
    IdleDetectionLoop,
    JobReaperLoop,
    ManifestCollectionLoop,
    QueuePopulatorLoop,
    SelfHealingLoop,
    TrainingSyncLoop,
    WorkerPullLoop,
    WorkQueueMaintenanceLoop,
)

# Event subscription mixin
from scripts.p2p.p2p_mixin_base import (
    EventSubscriptionMixin,
    P2PManagerBase,
    P2PMixinBase,
)
```

---

## Testing

### Syntax Check

```bash
python3 -m py_compile scripts/p2p/managers/state_manager.py
python3 -m py_compile scripts/p2p/managers/node_selector.py
python3 -m py_compile scripts/p2p/managers/sync_planner.py
python3 -m py_compile scripts/p2p/managers/job_manager.py
python3 -m py_compile scripts/p2p/managers/selfplay_scheduler.py
python3 -m py_compile scripts/p2p/managers/training_coordinator.py
python3 -m py_compile scripts/p2p/loops/base.py
python3 -m py_compile scripts/p2p/p2p_mixin_base.py
```

### Import Check

```bash
python3 -c "from scripts.p2p.managers import StateManager, NodeSelector, SyncPlanner, JobManager, SelfplayScheduler, TrainingCoordinator; print('All imports successful')"
python3 -c "from scripts.p2p.loops import LoopManager, BaseLoop, BackoffConfig; print('Loops imported')"
python3 -c "from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin, P2PManagerBase; print('Mixins imported')"
```

### Unit Testing

Each manager has dedicated unit tests in `tests/unit/p2p/`:

| Test File                  | Tests | Coverage                                   |
| -------------------------- | ----- | ------------------------------------------ |
| `test_state_manager.py`    | 35    | SQLite persistence, cluster epoch, job ops |
| `test_node_selector.py`    | 38    | GPU/CPU ranking, filtering, selection      |
| `test_job_manager.py`      | 30    | Job spawning, lifecycle, script selection  |
| `test_job_loops.py`        | 37    | JobReaperLoop, IdleDetectionLoop           |
| `test_resilience_loops.py` | 25    | SelfHealingLoop, PredictiveMonitoringLoop  |
| `test_loops.py`            | 50+   | LoopManager, BaseLoop, BackoffConfig       |

**Running tests**:

```bash
# All P2P manager tests
PYTHONPATH=. python3 -m pytest tests/unit/p2p/ -v

# Specific manager
PYTHONPATH=. python3 -m pytest tests/unit/p2p/test_node_selector.py -v
```

**Test Pattern Example**:

```python
from scripts.p2p.managers import NodeSelector
from unittest.mock import MagicMock

# Create mock node
mock_node = MagicMock()
mock_node.has_gpu = True
mock_node.gpu_power_score.return_value = 100
mock_node.is_alive.return_value = True

# Create selector with mock data
selector = NodeSelector(
    get_peers=lambda: {"node1": mock_node},
    get_self_info=lambda: None,
)

# Test ranking
nodes = selector.get_training_primary_nodes(count=5)
assert len(nodes) == 1
```

**Async Test Pattern** (for async methods):

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_run_gpu_selfplay_job():
    mgr = JobManager(...)

    with patch('asyncio.create_subprocess_exec') as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_exec.return_value = mock_proc

        await mgr.run_gpu_selfplay_job(...)
        assert mock_exec.called
```

---

## State Machine: Job Lifecycle

Jobs progress through states managed by JobManager and TrainingCoordinator:

```
┌─────────┐   dispatch   ┌─────────┐   started   ┌─────────┐
│ PENDING │──────────────▶ RUNNING │─────────────▶ COMPLETE│
└─────────┘              └─────────┘              └─────────┘
     │                        │                        │
     │ timeout               │ error                  │ gauntlet
     ▼                        ▼                        ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ EXPIRED │              │ FAILED  │              │ PROMOTED│
└─────────┘              └─────────┘              └─────────┘
```

**State Transitions**:

- `PENDING → RUNNING`: Job dispatched to node
- `RUNNING → COMPLETE`: Job finished successfully
- `RUNNING → FAILED`: Job encountered error
- `PENDING → EXPIRED`: Dispatch timeout (5 min)
- `COMPLETE → PROMOTED`: Model passed gauntlet evaluation

---

## State Machine: Sync Workflow

SyncPlanner manages data synchronization across the cluster:

```
┌──────────────┐   collect    ┌──────────────┐
│ IDLE         │──────────────▶ COLLECTING   │
└──────────────┘              └──────────────┘
       ▲                             │
       │                             │ manifests ready
       │                             ▼
┌──────────────┐   complete   ┌──────────────┐
│ SYNCED       │◀─────────────│ SYNCING      │
└──────────────┘              └──────────────┘
       │                             │
       │ cache expire                │ error
       ▼                             ▼
┌──────────────┐              ┌──────────────┐
│ IDLE         │              │ RETRY        │
└──────────────┘              └──────────────┘
```

**Key Timing**:

- Manifest cache: 5 minutes
- Collection interval: 1 minute
- Retry backoff: Exponential (1s → 30s)

---

## Error Handling Strategies

### 1. StateManager: Database Errors

```python
# Retry with exponential backoff
for attempt in range(3):
    try:
        self._execute_query(sql)
        break
    except sqlite3.OperationalError as e:
        if "locked" in str(e):
            time.sleep(0.1 * (2 ** attempt))
        else:
            raise
```

**Recovery**: WAL mode + busy_timeout prevent most locking issues.

### 2. NodeSelector: Missing Nodes

```python
# Graceful degradation when no nodes match criteria
nodes = self.get_training_nodes_ranked()
if not nodes:
    logger.warning("No GPU nodes available for training")
    return None  # Caller handles empty case
```

### 3. SyncPlanner: Network Failures

```python
# Individual peer failures don't fail entire sync
for peer_id, peer in peers.items():
    try:
        manifest = await self._fetch_manifest(peer)
        manifests[peer_id] = manifest
    except Exception as e:
        logger.warning(f"Failed to fetch manifest from {peer_id}: {e}")
        # Continue with other peers
```

### 4. JobManager: Spawn Failures

```python
# Job dispatch failures are logged but don't crash
try:
    result = self._spawn_job(node, config)
    self._track_job(result)
except Exception as e:
    logger.error(f"Job spawn failed: {e}")
    # Job marked as failed, not retried automatically
```

### 5. TrainingCoordinator: Hash-Based Deduplication

```python
# Prevent duplicate training triggers
job_hash = hashlib.sha256(f"{config}:{data_hash}".encode()).hexdigest()[:16]
if job_hash in self._recent_hashes:
    logger.info(f"Skipping duplicate training: {job_hash}")
    return None
self._recent_hashes.add(job_hash)
```

---

## Common Failure Scenarios

| Scenario                    | Manager             | Recovery                             |
| --------------------------- | ------------------- | ------------------------------------ |
| Leader election race        | StateManager        | Epoch comparison, higher epoch wins  |
| Node goes offline           | NodeSelector        | Filtered from rankings automatically |
| Manifest collection timeout | SyncPlanner         | 30s timeout, peer marked unhealthy   |
| Job process dies            | JobManager          | Cleanup on next status check         |
| Training cooldown           | TrainingCoordinator | 5-minute minimum between runs        |
| Database corruption         | StateManager        | WAL checkpoint + VACUUM recovery     |

---

## Performance Considerations

### StateManager

- WAL mode for concurrent reads during writes
- Batch operations with explicit transactions
- 30-second busy timeout for lock contention

### NodeSelector

- Caches node rankings for 10 seconds
- O(n log n) sorting, acceptable for <100 nodes

### SyncPlanner

- Manifest diffing is O(n) per file
- Max 50 files per sync job to limit bandwidth

### JobManager

- Job cleanup runs every 60 seconds
- Stale job detection after 5 minutes of no heartbeat

### SelfplayScheduler

- Config weights cached for 60 seconds
- Diversity tracking limited to last 1000 selections

### TrainingCoordinator

- Recent job hashes limited to 100 entries
- Cooldown state persisted in StateManager

---

## Related Documentation

- [SELFPLAY_SCHEDULER_USAGE.md](./SELFPLAY_SCHEDULER_USAGE.md) - Detailed selfplay scheduler guide
- `../p2p_orchestrator.py` - Main orchestrator (~25,900 LOC after delegation)
- `../p2p_mixin_base.py` - EventSubscriptionMixin and P2PManagerBase classes
- `../loops/` - Background loop implementations and LoopManager
- `../handlers/` - HTTP handler mixins for P2P endpoints
- `../models.py` - Data models (NodeInfo, ClusterJob, etc.)
- `../../CLAUDE.md` - AI service context
- `app/coordination/daemon_manager.py` - DaemonManager for health integration
- `app/coordination/protocols.py` - HealthCheckResult protocol
