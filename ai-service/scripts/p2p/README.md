# P2P Orchestrator Module

This package provides the distributed P2P orchestrator for RingRift AI training.
The orchestrator coordinates selfplay, training, and data sync across a cluster of nodes.

## Architecture

```
scripts/p2p/
├── __init__.py           # Package exports, backward compatibility
├── constants.py          # Configuration constants
├── types.py              # Enums: NodeRole, JobType
├── models.py             # Dataclasses: NodeInfo, ClusterJob
├── network.py            # HTTP client, circuit breaker
├── resource.py           # Resource checking utilities
├── cluster_config.py     # Cluster configuration loading
├── client.py             # P2P client for external use
├── utils.py              # General utilities
├── metrics_manager.py    # Metrics recording & history
├── resource_detector.py  # System resource detection
├── network_utils.py      # Peer address & URL utilities
├── p2p_mixin_base.py     # Base class for P2P mixins
├── handlers/             # HTTP handler mixins
│   ├── work_queue.py     # Work queue handlers
│   ├── election.py       # Election handlers
│   └── ...               # Other handler modules
├── managers/             # Domain-specific managers (Dec 2025)
│   ├── job_manager.py         # Job spawning and lifecycle
│   ├── training_coordinator.py # Training dispatch and promotion
│   ├── selfplay_scheduler.py  # Priority-based config selection
│   ├── node_selector.py       # Node ranking for job dispatch
│   ├── sync_planner.py        # Data sync planning
│   └── state_manager.py       # SQLite persistence
├── loops/                # Background loop implementations (Dec 2025)
│   ├── base.py                # BaseLoop with backoff
│   ├── job_loops.py           # Job reaper, idle detection
│   ├── coordination_loops.py  # Auto-scaling, health aggregation
│   ├── resilience_loops.py    # Self-healing, predictive monitoring
│   └── ...                    # Other loop modules
└── README.md             # This file
```

## Manager Architecture (December 2025)

The P2P orchestrator has been decomposed into domain-specific managers for better modularity:

### Core Managers (`scripts/p2p/managers/`)

| Manager               | Purpose                                    | Key Methods                                    |
| --------------------- | ------------------------------------------ | ---------------------------------------------- |
| `JobManager`          | Job spawning and lifecycle management      | `run_gpu_selfplay_job()`, `spawn_training()`   |
| `TrainingCoordinator` | Training dispatch, gauntlet, promotion     | `dispatch_training_job()`, `check_readiness()` |
| `SelfplayScheduler`   | Priority-based selfplay config selection   | `pick_weighted_config()`, `get_target_jobs()`  |
| `NodeSelector`        | Node ranking for job dispatch              | `get_best_gpu_node()`, `get_training_nodes()`  |
| `SyncPlanner`         | Data sync planning and manifest collection | `collect_manifest()`, `create_sync_plan()`     |
| `StateManager`        | SQLite persistence for P2P state           | `load_state()`, `save_state()`                 |

All managers use **dependency injection** for testability:

```python
from scripts.p2p.managers import JobManager, SelfplayScheduler

# Managers receive callbacks instead of direct references
job_manager = JobManager(
    ringrift_path=Path("/path/to/ringrift"),
    node_id="my-node",
    peers=lambda: orchestrator.peers,
    peers_lock=orchestrator.peers_lock,
)

scheduler = SelfplayScheduler(
    get_cluster_elo_fn=lambda: orchestrator._get_cluster_elo_summary(),
    load_curriculum_weights_fn=lambda: orchestrator._load_curriculum_weights(),
)
```

### Background Loops (`scripts/p2p/loops/`)

Background tasks use the `BaseLoop` framework with exponential backoff:

| Loop                       | Interval | Purpose                                  |
| -------------------------- | -------- | ---------------------------------------- |
| `JobReaperLoop`            | 5 min    | Clean stale jobs (1hr), stuck jobs (2hr) |
| `IdleDetectionLoop`        | 30 sec   | Detect idle GPUs, trigger selfplay       |
| `WorkerPullLoop`           | 30 sec   | Workers poll leader for work             |
| `WorkQueueMaintenanceLoop` | 5 min    | Check timeouts, cleanup old items        |
| `SelfHealingLoop`          | 5 min    | Recover stuck jobs, clean stale procs    |
| `PredictiveMonitoringLoop` | 5 min    | Track trends, emit alerts                |
| `EloSyncLoop`              | 5 min    | Synchronize Elo ratings                  |
| `QueuePopulatorLoop`       | 1 min    | Maintain work queue population           |

**LoopManager** coordinates all background loops:

```python
from scripts.p2p.loops import BaseLoop, LoopManager, JobReaperLoop

# Create loop manager
manager = LoopManager()

# Register loops
manager.register(JobReaperLoop(
    get_active_jobs=orchestrator.get_jobs,
    cancel_job=orchestrator.cancel_job,
))

# Start all loops
await manager.start_all()

# Stop gracefully
await manager.stop_all()
```

### Event System Integration

Managers emit events for pipeline coordination via `event_router`:

| Event                 | Emitter             | Subscribers                     |
| --------------------- | ------------------- | ------------------------------- |
| `DATA_SYNC_STARTED`   | SyncPlanner         | DataPipelineOrchestrator        |
| `DATA_SYNC_COMPLETED` | SyncPlanner         | DataPipelineOrchestrator        |
| `TRAINING_STARTED`    | TrainingCoordinator | SyncRouter, IdleShutdown        |
| `TRAINING_COMPLETED`  | TrainingCoordinator | FeedbackLoop, ModelDistribution |
| `HOST_OFFLINE`        | P2POrchestrator     | UnifiedHealthManager            |
| `HOST_ONLINE`         | P2POrchestrator     | UnifiedHealthManager            |
| `LEADER_ELECTED`      | P2POrchestrator     | LeadershipCoordinator           |

Event emission is thread-safe with lazy initialization:

```python
# Managers use _get_event_emitter() for thread-safe access
emitter = _get_event_emitter()
if emitter:
    emitter("TRAINING_COMPLETED", {"config_key": "hex8_2p", "model_path": "/path"})
```

### Health Check System

All managers and loops support `health_check()` for DaemonManager integration:

```python
from app.coordination.protocols import HealthCheckResult

# BaseLoop.health_check() returns:
result = loop.health_check()
# HealthCheckResult(healthy=True, status=CoordinatorStatus.RUNNING, message="...", details={...})

# Health checks track:
# - Consecutive errors (>5 = unhealthy)
# - Success rate (<50% = degraded)
# - Last error message and time
```

The `/status` endpoint aggregates health from all managers:

```json
{
  "node_id": "my-node",
  "role": "leader",
  "managers": {
    "job_manager": {"healthy": true, "jobs_active": 5},
    "sync_planner": {"healthy": true, "last_sync": 1703..},
    "selfplay_scheduler": {"healthy": true, "configs_weighted": 12}
  },
  "loops": {
    "job_reaper": {"running": true, "success_rate": 99.5},
    "idle_detection": {"running": true, "last_run": 1703..}
  }
}
```

## Stability Improvements (December 2025)

Recent fixes for P2P reliability:

1. **Pre-flight dependency validation** - Check aiohttp, psutil, yaml at startup
2. **Gzip magic byte detection** - Handle mixed Content-Encoding in gossip
3. **120s startup grace period** - Allow slow state file loading
4. **SystemExit handling** - Proper task cleanup on shutdown
5. **/dev/shm fallback** - macOS compatibility for shared memory
6. **Clear port binding errors** - Suggest `lsof -i :8770` remediation

## HTTP Handler Mixins

Handlers are organized into mixin classes in `scripts/p2p/handlers/`:

| Module          | Handlers                          | Status   |
| --------------- | --------------------------------- | -------- |
| `work_queue.py` | Work add/claim/complete/fail      | Complete |
| `election.py`   | Leader election, lease management | Complete |
| `relay.py`      | NAT relay for blocked peers       | Complete |
| `gauntlet.py`   | Model evaluation endpoints        | Complete |
| `gossip.py`     | State synchronization             | Complete |
| `admin.py`      | Admin operations                  | Complete |
| `elo_sync.py`   | Elo rating synchronization        | Complete |
| `tournament.py` | Tournament management             | Complete |

P2POrchestrator inherits from these mixins:

- `NetworkUtilsMixin` - Peer address parsing, URL building, Tailscale
- `PeerManagerMixin` - Peer cache and reputation management
- `LeaderElectionMixin` - Voter quorum, consistency checks
- `GossipMetricsMixin` - Gossip protocol metrics

## Usage

### P2P Client (Python)

```python
# External client usage
from scripts.p2p import P2PClient

client = P2PClient("http://localhost:8770")
status = await client.get_status()

# Constants
from scripts.p2p.constants import PEER_TIMEOUT, HEARTBEAT_INTERVAL

# Models
from scripts.p2p.models import NodeInfo, ClusterJob
```

### Cluster Update Script (NEW - Dec 2025)

Update all cluster nodes to latest code:

```bash
# Update all nodes (default: commit 88ca80cb2)
python scripts/update_all_nodes.py

# Update with P2P restart
python scripts/update_all_nodes.py --restart-p2p

# Update to specific commit
python scripts/update_all_nodes.py --commit abc1234

# Dry run (preview only)
python scripts/update_all_nodes.py --dry-run

# Limit parallelism
python scripts/update_all_nodes.py --max-parallel 5
```

**Features:**

- Parallel updates (default: 10 concurrent)
- Auto-detects node paths by provider
- Skips coordinator nodes (local-mac, mac-studio)
- Skips nodes with status != ready
- Optional P2P orchestrator restart
- Git stash before pull
- Commit verification after update
- Summary report with success/failure counts

**Example output:**

```
✅ Successfully updated: 28 nodes
  - runpod-h100: Updated to abc1234, P2P restarted
  - vast-5090-1: Updated to abc1234
  ...

⏭️  Skipped: 3 nodes
  - mac-studio: SKIPPED: Coordinator node
  - vast-retired: SKIPPED: Status is retired

❌ Failed: 1 node
  - nebius-h100-2: Connection failed: timeout

📊 Total: 32 nodes
```

## File Locations

- Main orchestrator: `scripts/p2p_orchestrator.py`
- Package: `scripts/p2p/`
- Tests: `tests/unit/scripts/test_p2p_*.py`
