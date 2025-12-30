# Coordination System API Reference

Comprehensive API documentation for the RingRift coordination layer, providing unified management for background daemons, event routing, synchronization, and health monitoring.

---

## Overview

The coordination system consists of four major components:

1. **Daemon Manager** - Lifecycle management for 89 background services
2. **Event Router** - Unified event bus bridging in-memory, stage, and cross-process events
3. **Sync Facade** - Single entry point for all cluster synchronization operations
4. **Coordination Bootstrap** - Initialization and wiring of all coordinators

### Architecture Diagram

```
+-------------------+     +------------------+     +----------------+
|   DaemonManager   |<--->|   EventRouter    |<--->|   SyncFacade   |
+-------------------+     +------------------+     +----------------+
         |                        |                        |
         v                        v                        v
+-------------------+     +------------------+     +----------------+
| 89 Daemon Types   |     | 3 Event Buses    |     | 5+ Backends    |
| - AUTO_SYNC       |     | - DataEventBus   |     | - AutoSync     |
| - DATA_PIPELINE   |     | - StageEventBus  |     | - ClusterSync  |
| - FEEDBACK_LOOP   |     | - CrossProcess   |     | - Router       |
| - EVALUATION      |     +------------------+     +----------------+
+-------------------+
```

---

## 1. Daemon Manager API

The `DaemonManager` provides centralized lifecycle management for all background services with health monitoring, auto-restart, and graceful shutdown.

### Import

```python
from app.coordination.daemon_manager import (
    DaemonManager,
    get_daemon_manager,
)
from app.coordination.daemon_types import (
    DaemonType,
    DaemonState,
    DaemonInfo,
    DaemonManagerConfig,
)
```

### Getting the Manager

```python
# Get singleton instance (preferred)
manager = get_daemon_manager()

# Or via class method
manager = DaemonManager.get_instance()
```

### Starting Daemons

```python
# Start a single daemon
success = await manager.start(DaemonType.AUTO_SYNC)

# Start with dependency waiting
success = await manager.start(DaemonType.DATA_PIPELINE, wait_for_deps=True)

# Start all registered daemons
await manager.start_all()

# Start all daemons for a specific profile
await manager.start_all(profile="coordinator")  # or "training_node"
```

### Stopping Daemons

```python
# Stop a specific daemon
success = await manager.stop(DaemonType.AUTO_SYNC)

# Graceful shutdown of all daemons
await manager.shutdown()

# Force shutdown with timeout
await manager.shutdown(timeout=10.0, force=True)
```

### Querying Status

```python
# Get status of all daemons
status = manager.get_status()
# Returns: Dict[DaemonType, DaemonInfo]

# Check if a specific daemon is running
is_running = manager.is_running(DaemonType.AUTO_SYNC)

# Get all daemon health status
health_status = manager.get_all_daemon_health()
# Returns: Dict[DaemonType, Dict[str, Any]]

# Liveness probe (for Kubernetes/Docker)
if manager.liveness_probe():
    print("All critical daemons healthy")
```

### Registering Custom Daemons

```python
async def my_daemon_factory():
    """Factory function that runs the daemon."""
    daemon = MyCustomDaemon()
    await daemon.start()
    await daemon.wait_for_shutdown()

manager.register_factory(
    daemon_type=DaemonType.MY_DAEMON,
    factory=my_daemon_factory,
    depends_on=[DaemonType.EVENT_ROUTER],  # Optional dependencies
    health_check_interval=30.0,             # Seconds between health checks
    auto_restart=True,                      # Restart on failure
    max_restarts=5,                         # Max restart attempts
)
```

### Handling Permanently Failed Daemons

```python
# Check if daemon has exceeded restart limit
if manager.is_permanently_failed(DaemonType.SOME_DAEMON):
    # Clear failure status after fixing root cause
    manager.clear_permanently_failed(DaemonType.SOME_DAEMON)
```

### Configuration Options

```python
from app.coordination.daemon_types import DaemonManagerConfig

config = DaemonManagerConfig(
    auto_start=False,                    # Auto-start daemons on init
    enable_coordination_wiring=True,     # Enable event system wiring
    dependency_wait_timeout=30.0,        # Timeout waiting for dependencies
    dependency_poll_interval=0.5,        # Poll interval for dependencies
    health_check_interval=30.0,          # Default health check interval
    shutdown_timeout=30.0,               # Shutdown timeout
    force_kill_timeout=5.0,              # Force kill timeout after shutdown
    auto_restart_failed=True,            # Auto-restart failed daemons
    strict_registry_validation=False,    # Raise on registry errors
    max_restart_attempts=5,              # Max restart attempts per daemon
    recovery_cooldown=10.0,              # Cooldown between restarts
    critical_daemon_health_interval=15.0,# Faster checks for critical daemons
)

manager = DaemonManager(config=config)
```

### Daemon Types

The system supports 89 daemon types organized by category:

| Category       | Key Daemons                                                  | Purpose                      |
| -------------- | ------------------------------------------------------------ | ---------------------------- |
| **Core**       | `EVENT_ROUTER`, `DAEMON_WATCHDOG`                            | Event bus, health monitoring |
| **Sync**       | `AUTO_SYNC`, `MODEL_DISTRIBUTION`, `ELO_SYNC`                | Data/model synchronization   |
| **Training**   | `DATA_PIPELINE`, `SELFPLAY_COORDINATOR`, `TRAINING_TRIGGER`  | Pipeline orchestration       |
| **Evaluation** | `EVALUATION`, `AUTO_PROMOTION`                               | Model evaluation/promotion   |
| **Health**     | `NODE_HEALTH_MONITOR`, `QUALITY_MONITOR`, `CLUSTER_WATCHDOG` | Cluster health               |
| **Resources**  | `IDLE_RESOURCE`, `NODE_RECOVERY`, `UTILIZATION_OPTIMIZER`    | GPU utilization              |

See `docs/DAEMON_REGISTRY.md` for the complete daemon reference.

---

## 2. Event System API

The `UnifiedEventRouter` consolidates three event buses into a single routing layer with deduplication, timeout protection, and dead-letter queue integration.

### Import

```python
from app.coordination.event_router import (
    get_router,
    publish,
    subscribe,
    unsubscribe,
    has_subscribers,
    EventSource,
    RouterEvent,
)
from app.distributed.data_events import DataEventType
```

### Publishing Events

```python
# Async publish (preferred)
router = get_router()
event = await router.publish(
    event_type="TRAINING_COMPLETED",
    payload={"config": "hex8_2p", "success": True, "epochs": 50},
    source="training_daemon",
)

# Using DataEventType enum
event = await router.publish(
    event_type=DataEventType.MODEL_PROMOTED,
    payload={"model_path": "models/canonical_hex8_2p.pth"},
    source="auto_promotion",
)

# Sync publish (for non-async contexts)
event = router.publish_sync(
    event_type="DATA_SYNC_COMPLETED",
    payload={"games_synced": 150, "duration": 12.5},
    source="sync_daemon",
)

# Control routing to specific buses
event = await router.publish(
    event_type="CUSTOM_EVENT",
    payload={"data": "value"},
    route_to_data_bus=True,
    route_to_stage_bus=False,      # Skip stage bus
    route_to_cross_process=True,   # Enable cross-process propagation
)
```

### Module-Level Convenience Function

```python
from app.coordination.event_router import publish

# Direct async publish
await publish(
    DataEventType.TRAINING_COMPLETED,
    payload={"config": "square8_2p"},
    source="trainer"
)
```

### Subscribing to Events

```python
router = get_router()

# Subscribe to specific event type
async def on_training_complete(event: RouterEvent):
    config = event.payload.get("config")
    print(f"Training completed for {config}")

router.subscribe("TRAINING_COMPLETED", on_training_complete)

# Subscribe using enum
router.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)

# Subscribe to ALL events
def on_any_event(event: RouterEvent):
    print(f"Event: {event.event_type}")

router.subscribe(None, on_any_event)  # None = global subscriber

# Unsubscribe
router.unsubscribe("TRAINING_COMPLETED", on_training_complete)
```

### RouterEvent Structure

```python
@dataclass
class RouterEvent:
    event_type: str           # Normalized event type name
    payload: dict[str, Any]   # Event data
    timestamp: float          # Unix timestamp
    source: str               # Component that generated the event
    origin: EventSource       # Which bus it came from (DATA_BUS, STAGE_BUS, etc.)
    event_id: str             # Unique ID for deduplication
    trace_id: str             # Distributed tracing ID
    correlation_id: str       # Request correlation ID
    parent_event_id: str      # Parent event for causality chains
    content_hash: str         # SHA256 hash for content deduplication
```

### Event Sources

```python
class EventSource(str, Enum):
    DATA_BUS = "data_bus"           # From EventBus (data_events.py)
    STAGE_BUS = "stage_bus"         # From StageEventBus (stage_events.py)
    CROSS_PROCESS = "cross_process" # From CrossProcessEventQueue
    ROUTER = "router"               # Originated from this router
```

### Checking Subscribers

```python
from app.coordination.event_router import has_subscribers

# Check if any handlers are registered for an event
if has_subscribers("TRAINING_COMPLETED"):
    print("Handlers registered")
```

### Router Statistics

```python
router = get_router()

# Get routing statistics
stats = router.get_stats()
# Returns: {
#     "events_routed": {"TRAINING_COMPLETED": 15, ...},
#     "events_by_source": {"data_bus": 100, ...},
#     "duplicates_prevented": 42,
#     "content_duplicates_prevented": 8,
# }

# Get recent event history
history = list(router._event_history)  # Last 1000 events
```

### Key Event Types

| Category       | Event Type                                    | Purpose                  |
| -------------- | --------------------------------------------- | ------------------------ |
| **Data Sync**  | `DATA_SYNC_COMPLETED`                         | Sync operation completed |
| **Training**   | `TRAINING_COMPLETED`, `TRAINING_STARTED`      | Training lifecycle       |
| **Evaluation** | `EVALUATION_COMPLETED`, `PROMOTION_CANDIDATE` | Model evaluation         |
| **Promotion**  | `MODEL_PROMOTED`, `PROMOTION_FAILED`          | Model promotion          |
| **Feedback**   | `ELO_VELOCITY_CHANGED`, `PLATEAU_DETECTED`    | Training feedback        |
| **Health**     | `NODE_RECOVERED`, `HOST_OFFLINE`              | Cluster health           |

See `docs/EVENT_SYSTEM_REFERENCE.md` for the complete event catalog (202 event types).

---

## 3. Sync Operations API

The `SyncFacade` provides a unified entry point for all cluster synchronization operations, automatically routing requests to the appropriate backend.

### Import

```python
from app.coordination.sync_facade import (
    SyncFacade,
    SyncRequest,
    SyncResponse,
    SyncBackend,
    sync,              # Convenience function
    get_sync_facade,
)
```

### Basic Sync Operations

```python
# Using convenience function
from app.coordination.sync_facade import sync

# Sync games to all nodes
response = await sync("games")

# Sync models to specific nodes
response = await sync("models", targets=["node-1", "node-2"])

# High-priority sync
response = await sync("games", board_type="hex8", priority="high")
```

### Using SyncFacade Directly

```python
facade = get_sync_facade()

# Sync with full options
response = await facade.sync(
    SyncRequest(
        data_type="games",              # "games", "models", "npz", "all"
        targets=["node-1", "node-2"],   # None = all eligible nodes
        board_type="hex8",
        num_players=2,
        priority="high",                # "low", "normal", "high", "critical"
        timeout_seconds=300.0,
        bandwidth_limit_mbps=50,
        exclude_nodes=["offline-node"],
        prefer_ephemeral=False,         # Use ephemeral sync daemon
        require_confirmation=True,       # Wait for confirmation
    )
)

# Check result
if response.success:
    print(f"Synced {response.nodes_synced} nodes")
    print(f"Transferred {response.bytes_transferred} bytes")
else:
    print(f"Errors: {response.errors}")
```

### Priority Sync for Recovery

```python
# Trigger urgent sync for orphan game recovery
facade = get_sync_facade()
response = await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
    config_key="hex8_2p",
    data_type="games",
)
```

### SyncResponse Structure

```python
@dataclass
class SyncResponse:
    success: bool                  # Overall success status
    backend_used: SyncBackend      # Which backend handled the request
    nodes_synced: int              # Number of nodes synced
    bytes_transferred: int         # Total bytes transferred
    duration_seconds: float        # Time taken
    errors: list[str]              # Error messages if any
    details: dict[str, Any]        # Backend-specific details
```

### Sync Backends

```python
class SyncBackend(Enum):
    AUTO_SYNC = "auto_sync"              # P2P gossip-based sync
    CLUSTER_SYNC = "cluster_data_sync"   # Deprecated alias → AutoSyncDaemon(BROADCAST)
    DISTRIBUTED = "distributed"          # Low-level transport
    EPHEMERAL = "ephemeral_sync"         # Aggressive sync for ephemeral hosts
    ROUTER = "router"                    # Intelligent routing
```

### Backend Selection Logic

The facade automatically selects the appropriate backend:

1. **Ephemeral nodes** -> `EPHEMERAL` backend
2. **High/critical priority** -> `CLUSTER_SYNC` (deprecated alias; broadcast AutoSync)
3. **Specific targets** -> `ROUTER` (intelligent routing)
4. **Default** -> `AUTO_SYNC` (P2P gossip)

### Health Check

```python
facade = get_sync_facade()

# Check sync facade health
result = facade.health_check()
# Returns: HealthCheckResult(
#     healthy=True,
#     status=CoordinatorStatus.RUNNING,
#     message="",
#     details={
#         "operations_count": 42,
#         "errors_count": 2,
#         "backends_loaded": 3,
#         ...
#     }
# )

# Get statistics
stats = facade.get_stats()
# Returns: {
#     "total_syncs": 100,
#     "by_backend": {"auto_sync": 80, "cluster_sync": 20},
#     "total_bytes": 5000000,
#     "total_errors": 2,
# }
```

---

## 4. Coordination Bootstrap API

The `coordination_bootstrap` module initializes coordinators in dependency order and delegates event wiring through `event_subscription_registry` (`INIT_CALL_REGISTRY`, `DELEGATION_REGISTRY`).

### Import

```python
from app.coordination.coordination_bootstrap import (
    bootstrap_coordination,
    shutdown_coordination,
    get_bootstrap_status,
    COORDINATOR_REGISTRY,
)
```

### Initialization

```python
# Initialize all coordination components
bootstrap_coordination()

# Initialize with specific options
bootstrap_coordination(
    enable_metrics=True,
    enable_optimization=True,
    enable_leadership=False,  # Disable for single-node
)
```

### Status Checking

```python
status = get_bootstrap_status()
# Returns: {
#     "initialized": True,
#     "started_at": datetime,
#     "completed_at": datetime,
#     "coordinators": {
#         "task_coordinator": {
#             "initialized": True,
#             "subscribed": True,
#             "error": None,
#         },
#         ...
#     },
#     "errors": [],
# }

print(f"Coordinators initialized: {len(status['coordinators'])}")
```

### Shutdown

```python
# Graceful shutdown
shutdown_coordination()
```

### Initialization Order

Initialization order is defined by `COORDINATOR_REGISTRY` plus the special-case `pipeline_orchestrator` (see `coordination_bootstrap.py` for the canonical list). Current layers:

| Layer | Coordinators (summary)                                                                                                                                                                             | Purpose                         |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 1     | `task_coordinator`, `global_task_coordinator`, `resource_coordinator`, `cache_orchestrator`                                                                                                        | Foundational (no deps)          |
| 2     | `health_manager`, `error_coordinator` (delegated), `recovery_manager` (skipped), `model_coordinator`                                                                                               | Infrastructure support          |
| 3     | `sync_coordinator`, `training_coordinator`                                                                                                                                                         | Sync and training               |
| 4     | `transfer_verifier`, `ephemeral_guard`, `queue_populator`                                                                                                                                          | Data integrity                  |
| 5     | `selfplay_orchestrator`, `selfplay_scheduler`                                                                                                                                                      | Selfplay                        |
| 6     | `pipeline_orchestrator` (special-case), `multi_provider`, `job_scheduler`                                                                                                                          | Pipeline + jobs                 |
| 7     | `auto_export_daemon`, `evaluation_daemon`, `model_distribution_daemon`, `idle_resource_daemon`, `quality_monitor_daemon`, `orphan_detection_daemon`, `training_activity`, `curriculum_integration` | Background automation           |
| 8     | `metrics_orchestrator`, `optimization_coordinator`                                                                                                                                                 | Metrics + optimization          |
| 9     | `leadership_coordinator`                                                                                                                                                                           | Top-level coordination (leader) |

---

## 5. Health Check Patterns

All coordinators and daemons implement a consistent health check interface.

### HealthCheckResult Protocol

```python
from app.coordination.protocols import HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

@dataclass
class HealthCheckResult:
    healthy: bool                    # Overall health status
    status: CoordinatorStatus        # RUNNING, DEGRADED, STOPPED
    message: str                     # Human-readable status message
    details: dict[str, Any]          # Detailed metrics
```

### Implementing Health Checks

```python
from app.coordination.protocols import HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

class MyDaemon:
    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        errors = self._error_count
        cycles = self._cycle_count

        # Determine status
        if errors > cycles * 0.5:
            status = CoordinatorStatus.STOPPED
            message = "High error rate"
        elif errors > cycles * 0.2:
            status = CoordinatorStatus.DEGRADED
            message = "Elevated error rate"
        else:
            status = CoordinatorStatus.RUNNING
            message = "Healthy"

        return HealthCheckResult(
            healthy=(status == CoordinatorStatus.RUNNING),
            status=status,
            message=message,
            details={
                "cycle_count": cycles,
                "error_count": errors,
                "uptime_seconds": self._uptime,
            },
        )
```

### Using HealthCheckHelper

```python
from app.coordination.health_check_helper import HealthCheckHelper

# Reusable health check methods
is_ok, msg = HealthCheckHelper.check_error_rate(
    errors=5, cycles=100, threshold=0.5
)

is_ok, msg = HealthCheckHelper.check_uptime_grace(
    start_time=self._start_time, grace_period=30
)

is_ok, msg = HealthCheckHelper.check_queue_depth(
    depth=queue.qsize(), max_depth=1000
)
```

---

## 6. Common Operation Examples

### Example: Full Daemon Lifecycle

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

async def main():
    manager = get_daemon_manager()

    # Start core daemons in order
    await manager.start(DaemonType.EVENT_ROUTER)
    await manager.start(DaemonType.DATA_PIPELINE)
    await manager.start(DaemonType.AUTO_SYNC)
    await manager.start(DaemonType.FEEDBACK_LOOP)

    # Monitor health
    while True:
        health = manager.get_all_daemon_health()
        for dtype, status in health.items():
            if not status.get("healthy", True):
                print(f"Warning: {dtype.value} unhealthy")
        await asyncio.sleep(30)
```

### Example: Event-Driven Pipeline

```python
from app.coordination.event_router import get_router, RouterEvent

async def setup_pipeline():
    router = get_router()

    # Stage 1: On sync complete, trigger export
    async def on_sync_complete(event: RouterEvent):
        games_synced = event.payload.get("games_synced", 0)
        if games_synced > 100:
            await router.publish(
                "EXPORT_TRIGGERED",
                payload={"threshold_met": True},
                source="pipeline"
            )

    # Stage 2: On export complete, trigger training
    async def on_export_complete(event: RouterEvent):
        samples = event.payload.get("samples_exported", 0)
        if samples > 10000:
            await router.publish(
                "TRAINING_TRIGGERED",
                payload={"samples": samples},
                source="pipeline"
            )

    router.subscribe("DATA_SYNC_COMPLETED", on_sync_complete)
    router.subscribe("NPZ_EXPORT_COMPLETED", on_export_complete)
```

### Example: Priority Sync with Feedback

```python
from app.coordination.sync_facade import get_sync_facade, SyncRequest
from app.coordination.event_router import get_router

async def sync_with_feedback(config_key: str, source_node: str):
    facade = get_sync_facade()
    router = get_router()

    # Trigger sync
    response = await facade.sync(SyncRequest(
        data_type="games",
        targets=[source_node],
        priority="high",
        require_confirmation=True,
    ))

    if response.success:
        # Emit success event for downstream handlers
        await router.publish(
            "PRIORITY_SYNC_COMPLETED",
            payload={
                "config_key": config_key,
                "source_node": source_node,
                "games_synced": response.nodes_synced,
            },
            source="sync_orchestrator"
        )
    else:
        await router.publish(
            "PRIORITY_SYNC_FAILED",
            payload={
                "config_key": config_key,
                "errors": response.errors,
            },
            source="sync_orchestrator"
        )
```

### Example: Health Monitoring Dashboard

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.sync_facade import get_sync_facade
from app.coordination.event_router import get_router

def get_system_health():
    """Aggregate health status from all components."""
    manager = get_daemon_manager()
    facade = get_sync_facade()
    router = get_router()

    # Collect daemon health
    daemon_health = manager.get_all_daemon_health()
    running = sum(1 for h in daemon_health.values() if h.get("state") == "running")
    failed = sum(1 for h in daemon_health.values() if h.get("state") == "failed")

    # Check sync status
    sync_health = facade.health_check()

    # Get event routing stats
    router_stats = router.get_stats()

    return {
        "daemons": {
            "running": running,
            "failed": failed,
            "total": len(daemon_health),
        },
        "sync": {
            "healthy": sync_health.healthy,
            "total_syncs": facade.get_stats()["total_syncs"],
        },
        "events": {
            "routed": sum(router_stats["events_routed"].values()),
            "duplicates_prevented": router_stats["duplicates_prevented"],
        },
    }
```

---

## 7. Environment Variables

Key configuration via environment variables:

| Variable                            | Default | Description                         |
| ----------------------------------- | ------- | ----------------------------------- |
| `RINGRIFT_EVENT_HANDLER_TIMEOUT`    | 600.0   | Event handler timeout (seconds)     |
| `RINGRIFT_EVENT_VALIDATION_STRICT`  | false   | Reject unknown events               |
| `RINGRIFT_P2P_STARTUP_GRACE_PERIOD` | 120     | P2P startup grace period (seconds)  |
| `RINGRIFT_ALLOW_PENDING_GATE`       | false   | Allow pending parity gate databases |
| `RINGRIFT_SWIM_ENABLED`             | false   | Enable SWIM membership protocol     |
| `RINGRIFT_RAFT_ENABLED`             | false   | Enable Raft consensus protocol      |
| `RINGRIFT_JOB_GRACE_PERIOD`         | 60      | Job shutdown grace period (seconds) |

---

## 8. See Also

- `docs/DAEMON_REGISTRY.md` - Complete daemon type reference
- `docs/EVENT_SYSTEM_REFERENCE.md` - Full event catalog (207 types)
- `docs/CLUSTER_INTEGRATION_GUIDE.md` - Cluster architecture integration
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Troubleshooting guide
- `docs/runbooks/EVENT_WIRING_VERIFICATION.md` - Event wiring verification

---

_Last updated: December 2025_
