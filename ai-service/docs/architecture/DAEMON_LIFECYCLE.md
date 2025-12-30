# Daemon Lifecycle Architecture

**Last Updated:** December 28, 2025
**Status:** Production

## Overview

The RingRift AI service uses a daemon-based architecture with 85 daemon types managed by the `DaemonManager`. This document describes the lifecycle, health monitoring, and startup/shutdown behavior of daemons.

## Daemon Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DaemonManager                                │
│  Lifecycle management, health monitoring, auto-restart              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ DaemonSpec    │         │ DaemonRunner  │         │ HealthLoop    │
│ (Registry)    │         │ (Execution)   │         │ (Monitoring)  │
└───────────────┘         └───────────────┘         └───────────────┘
```

## Key Components

### DaemonManager

**File:** `app/coordination/daemon_manager.py`

Central orchestrator for all daemon lifecycle operations.

**Responsibilities:**

- Start/stop individual daemons
- Manage daemon dependencies
- Monitor health and auto-restart failed daemons
- Expose HTTP health endpoints (/health, /ready, /metrics)

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

# Start a daemon
await dm.start(DaemonType.AUTO_SYNC)

# Check health
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)

# Stop daemon
await dm.stop(DaemonType.AUTO_SYNC)
```

### DaemonSpec (Registry)

**File:** `app/coordination/daemon_registry.py`

Declarative configuration for all 85 daemon types.

```python
@dataclass(frozen=True)
class DaemonSpec:
    runner_name: str                 # Function name in daemon_runners.py
    depends_on: tuple                # DaemonTypes that must start first
    health_check_interval: float | None = None
    auto_restart: bool = True        # Restart on failure
    max_restarts: int = 5            # Max restart attempts
    category: str = "misc"           # "sync", "event", "health", etc.
    deprecated: bool = False
    deprecated_message: str = ""
```

**Categories (examples, non-exhaustive):**

- `event`: EVENT_ROUTER, DAEMON_WATCHDOG
- `sync`: AUTO_SYNC, MODEL_DISTRIBUTION, ELO_SYNC
- `pipeline`: DATA_PIPELINE, SELFPLAY_COORDINATOR, TRAINING_NODE_WATCHER
- `evaluation`: EVALUATION, AUTO_PROMOTION
- `health`: NODE_HEALTH_MONITOR, QUALITY_MONITOR
- `resource`: IDLE_RESOURCE, NODE_RECOVERY

### DaemonRunner (Execution)

**File:** `app/coordination/daemon_runners.py`

85 async runner functions that create and start daemon instances.

```python
async def create_auto_sync() -> None:
    """Runner for AUTO_SYNC daemon."""
    from app.coordination.auto_sync_daemon import AutoSyncDaemon

    daemon = AutoSyncDaemon()
    await daemon.start()
    await daemon.wait()  # Block until stopped
```

## Daemon Lifecycle States

```
    INITIALIZING
         │
         ▼
    STARTING ──────────┐
         │             │
         ▼             │ (failure)
     RUNNING ◄─────────┤
         │             │
         │             │
    STOPPING           │
         │             │
         ▼             │
     STOPPED           │
         │             │
         ▼             │
      FAILED ──────────┘
         │          (auto-restart)
         ▼
     RESTARTING
```

### State Transitions

| From         | To         | Trigger                     |
| ------------ | ---------- | --------------------------- |
| INITIALIZING | STARTING   | `start()` called            |
| STARTING     | RUNNING    | `_on_start()` completes     |
| STARTING     | FAILED     | Exception in `_on_start()`  |
| RUNNING      | STOPPING   | `stop()` called             |
| RUNNING      | FAILED     | Exception in `_run_cycle()` |
| STOPPING     | STOPPED    | `_on_stop()` completes      |
| FAILED       | RESTARTING | Auto-restart triggered      |
| RESTARTING   | STARTING   | Restart delay elapsed       |

## Startup Order

Daemons have dependencies that must be satisfied before starting.

### Critical Startup Order (as of Dec 28, 2025)

```python
STARTUP_ORDER = [
    # Phase 1: Core infrastructure
    DaemonType.EVENT_ROUTER,        # Must be first - event bus
    DaemonType.DAEMON_WATCHDOG,     # Health monitoring for daemons

    # Phase 2: Event subscribers (BEFORE emitters)
    DaemonType.FEEDBACK_LOOP,       # Subscribes to TRAINING_COMPLETED
    DaemonType.DATA_PIPELINE,       # Subscribes to DATA_SYNC_COMPLETED

    # Phase 3: Sync daemons (emit events)
    DaemonType.AUTO_SYNC,           # Emits DATA_SYNC_COMPLETED
    DaemonType.MODEL_DISTRIBUTION,  # Emits MODEL_DISTRIBUTED

    # Phase 4: Training coordination
    DaemonType.SELFPLAY_COORDINATOR,
    DaemonType.TRAINING_TRIGGER,

    # Phase 5: Evaluation
    DaemonType.EVALUATION,
    DaemonType.AUTO_PROMOTION,
]
```

**Key Principle:** Event subscribers must start BEFORE event emitters to avoid lost events.

### Dependency Validation

```python
from app.coordination.daemon_registry import DAEMON_REGISTRY, validate_registry

# Validate at startup
errors = validate_registry()
if errors:
    raise RuntimeError(f"Registry validation failed: {errors}")
```

## Health Monitoring

### Health Check Protocol

All daemons implement `health_check()` returning `HealthCheckResult`:

```python
@dataclass
class HealthCheckResult:
    is_healthy: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
```

### Health Check Implementation

```python
class MyDaemon(BaseDaemon):
    def health_check(self) -> HealthCheckResult:
        """Return daemon health status."""
        if self._error_count > 10:
            return HealthCheckResult(
                is_healthy=False,
                message="Too many errors",
                details={"error_count": self._error_count},
            )
        return HealthCheckResult(
            is_healthy=True,
            message="Operating normally",
            details={
                "cycles": self._cycle_count,
                "last_cycle": self._last_cycle_time,
            },
        )
```

### Health Loop

The DaemonManager runs a health loop that:

1. Polls each daemon's `health_check()` at configured interval
2. Logs unhealthy daemons
3. Triggers auto-restart for failed daemons (up to max_restarts)
4. Emits `DAEMON_UNHEALTHY` events

```python
# Health loop is started automatically after daemon start
await dm.start(DaemonType.AUTO_SYNC)
# Health monitoring now active
```

## Auto-Restart Behavior

### Restart Configuration

```python
DaemonSpec(
    runner_name="create_auto_sync",
    auto_restart=True,        # Enable auto-restart
    max_restarts=5,           # Max attempts before giving up
    health_check_interval=30.0,
)
```

### Backoff Strategy

```
Restart 1: 1 second delay
Restart 2: 2 seconds
Restart 3: 4 seconds
Restart 4: 8 seconds
Restart 5: 16 seconds (max)
```

After max_restarts, daemon enters `FAILED` state permanently until manual intervention.

### Manual Restart

```python
dm = get_daemon_manager()

# Reset restart count and force restart
await dm.restart(DaemonType.AUTO_SYNC, reset_count=True)
```

## HTTP Health Endpoints

DaemonManager exposes health endpoints on port 8790:

| Endpoint   | Purpose            | Response                  |
| ---------- | ------------------ | ------------------------- |
| `/health`  | Liveness probe     | `{"status": "ok"}` or 500 |
| `/ready`   | Readiness probe    | `{"ready": true}` or 503  |
| `/metrics` | Prometheus metrics | OpenMetrics format        |

### Kubernetes Integration

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8790
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8790
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Creating New Daemons

### 1. Define the Daemon Class

```python
# app/coordination/my_daemon.py
from app.coordination.base_daemon import BaseDaemon, DaemonConfig

class MyDaemon(BaseDaemon):
    def __init__(self, config: DaemonConfig | None = None):
        super().__init__(config or DaemonConfig())

    def _get_daemon_name(self) -> str:
        return "MyDaemon"

    async def _on_start(self) -> None:
        """Initialize resources."""
        pass

    async def _run_cycle(self) -> None:
        """Main work loop - called repeatedly."""
        pass

    async def _on_stop(self) -> None:
        """Cleanup resources."""
        pass

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        return HealthCheckResult(is_healthy=True, message="OK")
```

### 2. Add Runner Function

```python
# app/coordination/daemon_runners.py
async def create_my_daemon() -> None:
    """Runner for MY_DAEMON."""
    from app.coordination.my_daemon import MyDaemon

    daemon = MyDaemon()
    await daemon.start()
    await daemon.wait()
```

### 3. Add to Registry

```python
# app/coordination/daemon_registry.py
DAEMON_REGISTRY = {
    # ... existing entries ...
    DaemonType.MY_DAEMON: DaemonSpec(
        runner_name="create_my_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="my_category",
        auto_restart=True,
        health_check_interval=60.0,
    ),
}
```

### 4. Add DaemonType

```python
# app/coordination/daemon_types.py
class DaemonType(Enum):
    # ... existing types ...
    MY_DAEMON = "my_daemon"
```

## Event Integration

Daemons can subscribe to and emit events:

### Subscribing

```python
class MyDaemon(BaseDaemon):
    def _get_event_subscriptions(self) -> dict:
        return {
            DataEventType.TRAINING_COMPLETED.value: self._on_training_completed,
            DataEventType.DATA_SYNC_COMPLETED.value: self._on_sync_completed,
        }

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion."""
        config_key = event.get("config_key")
        # Process event...
```

### Emitting

```python
async def _emit_my_event(self, payload: dict) -> None:
    """Emit event for coordination."""
    try:
        from app.distributed.data_events import get_event_bus, DataEventType
        bus = get_event_bus()
        await bus.publish(DataEventType.MY_EVENT.value, payload)
    except Exception as e:
        logger.debug(f"Failed to emit event: {e}")
```

## Troubleshooting

### Daemon Won't Start

1. Check dependencies: `dm.get_missing_dependencies(DaemonType.MY_DAEMON)`
2. Check logs: `tail -f logs/daemons/my_daemon.log`
3. Verify runner exists: `get_runner(DaemonType.MY_DAEMON)`

### Daemon Keeps Restarting

1. Check error logs for root cause
2. Review max_restarts limit
3. Check resource constraints (memory, disk)

### Events Not Delivered

1. Verify EVENT_ROUTER started first
2. Check subscriber is registered before emitter starts
3. Review event type spelling (use enum values)

## See Also

- `docs/DAEMON_REGISTRY.md` - Full daemon type reference
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Recovery procedures
- `docs/EVENT_SYSTEM_REFERENCE.md` - Event wiring details
- `app/coordination/base_daemon.py` - Base class implementation
