# P2P Manager Modules Architecture

## Overview

The `scripts/p2p/managers/` directory contains domain-specific manager classes extracted from the monolithic `p2p_orchestrator.py` as part of the **Phase 2B refactoring** (December 2025). This decomposition improves modularity, testability, and maintainability of the P2P orchestration layer.

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
# Search modes -> run_hybrid_selfplay.py
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

## Delegation Status (Updated Dec 27, 2025)

~1,231 LOC removed from `p2p_orchestrator.py` during Dec 27 cleanup.

| Manager             | Methods Delegated | Status      | LOC Removed | Notes                         |
| ------------------- | ----------------- | ----------- | ----------- | ----------------------------- |
| StateManager        | 7/7 (100%)        | ✅ Complete | ~200        | All delegated                 |
| NodeSelector        | 6/6 (100%)        | ✅ Complete | ~50         | All wrappers removed          |
| TrainingCoordinator | 5/5 (100%)        | ✅ Complete | ~450        | All wrappers removed          |
| JobManager          | 7/7 (100%)        | ✅ Complete | ~400        | All wrappers removed          |
| SelfplayScheduler   | 3/6 (50%)         | ⚠️ Partial  | ~200        | Some targeting methods active |
| SyncPlanner         | 2/4 (50%)         | ⚠️ Partial  | ~60         | `_execute_sync_plan` active   |

**Dec 27 Cleanup Highlights**:

- Removed `_dispatch_training_job`, `_handle_training_job_completion`, `_schedule_model_comparison_tournament`, `_run_post_training_gauntlet` (→ TrainingCoordinator)
- Removed `_run_gpu_selfplay_job`, `_run_distributed_tournament`, `_run_distributed_selfplay`, `_export_training_data`, `_run_training`, `_cleanup_old_completed_jobs` (→ JobManager)
- Removed `_get_hybrid_job_targets`, `_pick_weighted_selfplay_config`, `_get_elo_based_priority_boost` (→ SelfplayScheduler)
- Migrated 5 background loops to LoopManager (`_elo_sync_loop`, `_idle_detection_loop`, `_auto_scaling_loop`, `_job_reaper_loop`, `_queue_populator_loop`)

**Remaining partial delegation** (scheduled for Q2 2026):

- `_execute_sync_plan()` → `SyncPlanner.execute_sync()`
- `_get_selfplay_targeting()` → `SelfplayScheduler.get_targeting()`

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
```

### Import Check

```bash
python3 -c "from scripts.p2p.managers import StateManager, NodeSelector, SyncPlanner, JobManager, SelfplayScheduler, TrainingCoordinator; print('All imports successful')"
```

### Unit Testing

Each manager has dedicated unit tests in `tests/unit/p2p/`:

| Test File               | Tests | Coverage                                   |
| ----------------------- | ----- | ------------------------------------------ |
| `test_state_manager.py` | 35    | SQLite persistence, cluster epoch, job ops |
| `test_node_selector.py` | 38    | GPU/CPU ranking, filtering, selection      |
| `test_job_manager.py`   | 30    | Job spawning, lifecycle, script selection  |
| `test_loops.py`         | 50+   | Loop management, LoopManager integration   |

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
- `../p2p_orchestrator.py` - Main orchestrator (being decomposed)
- `../models.py` - Data models (NodeInfo, ClusterJob, etc.)
- `../../CLAUDE.md` - AI service context
