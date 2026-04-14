# P2P Manager Integration Guide

**Created:** February 1, 2026
**Purpose:** Consistent patterns for integrating new managers into the P2P orchestrator

---

## Overview

The P2P orchestrator uses a manager-based architecture where specialized managers handle discrete responsibilities. This guide documents the patterns for creating and integrating new managers.

## Manager Architecture

```
P2POrchestrator
    ├── StateManager          # Cluster state tracking
    ├── JobManager            # Job lifecycle management
    ├── TrainingCoordinator   # Training job coordination
    ├── SelfplayScheduler     # Selfplay allocation
    ├── SyncPlanner          # Data synchronization
    ├── NodeSelector         # Node selection logic
    ├── LoopManager          # Background loop lifecycle
    └── StatusMetricsCollector # Parallel metrics gathering
```

## Creating a New Manager

### 1. Directory Structure

Place managers in `scripts/p2p/managers/`:

```
scripts/p2p/managers/
├── __init__.py
├── my_new_manager.py
└── README.md
```

### 2. Basic Manager Template

```python
"""MyNewManager for P2P Orchestrator.

<Date> - <Purpose>

This manager handles <responsibility>.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Avoid circular imports
if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

# Import contracts for health check protocol
from app.coordination.contracts import (
    HealthCheckResult,
    CoordinatorStatus,
)

# Import safe event emission
from app.coordination.safe_event_emitter import safe_emit_event

logger = logging.getLogger(__name__)


@dataclass
class MyManagerConfig:
    """Configuration for MyNewManager."""

    # Configuration parameters with sensible defaults
    cycle_interval: float = 60.0
    timeout: float = 30.0
    max_retries: int = 3


class MyNewManager:
    """Manager for <responsibility>.

    Key features:
    - Feature 1
    - Feature 2
    - Health check integration

    Example:
        manager = MyNewManager(orchestrator, config)
        await manager.start()
        result = await manager.do_work()
    """

    def __init__(
        self,
        orchestrator: P2POrchestrator,
        config: MyManagerConfig | None = None,
    ):
        self._orchestrator = orchestrator
        self.config = config or MyManagerConfig()

        # Track manager state
        self._start_time: float = 0.0
        self._is_running: bool = False
        self._error_count: int = 0
        self._last_error: str = ""

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the manager."""
        self._start_time = time.time()
        self._is_running = True
        logger.info(f"{self.__class__.__name__} started")

    async def stop(self) -> None:
        """Stop the manager gracefully."""
        self._is_running = False
        logger.info(f"{self.__class__.__name__} stopped")

    # =========================================================================
    # Health Check Protocol
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult with current health status
        """
        uptime = time.time() - self._start_time if self._start_time > 0 else 0

        # Compute health based on error rate
        if self._error_count > 10:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"High error count: {self._error_count}",
                details={
                    "uptime_seconds": uptime,
                    "error_count": self._error_count,
                    "last_error": self._last_error,
                },
            )

        if self._error_count > 5:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Elevated errors: {self._error_count}",
                details={
                    "uptime_seconds": uptime,
                    "error_count": self._error_count,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
            details={
                "uptime_seconds": uptime,
                "is_running": self._is_running,
            },
        )

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Safely emit an event.

        Uses safe_emit_event to ensure event failures don't crash the manager.

        Args:
            event_type: Event type to emit
            payload: Event payload

        Returns:
            True if event was emitted successfully
        """
        return safe_emit_event(
            event_type,
            payload,
            source=self.__class__.__name__,
        )

    # =========================================================================
    # Core Functionality
    # =========================================================================

    async def do_work(self) -> dict[str, Any]:
        """Main work method.

        Returns:
            Result dictionary
        """
        try:
            # Do work here
            result = {"status": "success"}

            # Emit completion event
            self._emit_event("MY_WORK_COMPLETED", result)

            return result

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"Work failed: {e}")
            raise


# =============================================================================
# Factory Function
# =============================================================================

def create_my_manager(
    orchestrator: P2POrchestrator,
    config: MyManagerConfig | None = None,
) -> MyNewManager:
    """Create a configured MyNewManager instance.

    Args:
        orchestrator: Parent P2P orchestrator
        config: Optional configuration

    Returns:
        Configured manager instance
    """
    return MyNewManager(orchestrator, config)
```

## Health Check Protocol

All managers must implement `health_check()` returning `HealthCheckResult`:

### HealthCheckResult Fields

| Field       | Type                | Description                        |
| ----------- | ------------------- | ---------------------------------- |
| `healthy`   | `bool`              | Whether the manager is healthy     |
| `status`    | `CoordinatorStatus` | Current status enum                |
| `message`   | `str`               | Human-readable status message      |
| `timestamp` | `float`             | Time of health check (auto-filled) |
| `details`   | `dict`              | Additional metrics and state       |

### Status Levels

| Status     | When to Use                                               |
| ---------- | --------------------------------------------------------- |
| `RUNNING`  | Normal operation                                          |
| `DEGRADED` | Working but with issues (elevated errors, slow responses) |
| `ERROR`    | Not functioning correctly                                 |
| `PAUSED`   | Intentionally paused                                      |
| `STOPPED`  | Cleanly stopped                                           |

### Factory Methods

```python
from app.coordination.contracts import HealthCheckResult

# Healthy result
result = HealthCheckResult.healthy("All systems nominal", jobs_active=5)

# Unhealthy result
result = HealthCheckResult.unhealthy("Database connection failed", retry_count=3)

# Degraded result
result = HealthCheckResult.degraded("High latency detected", avg_latency_ms=500)

# From metrics (auto-computes status)
result = HealthCheckResult.from_metrics(
    uptime_seconds=3600,
    events_processed=1000,
    errors_count=5,
    last_activity_ago=30.0,
    max_inactivity_seconds=300.0,
    max_error_rate=0.1,
)
```

## Event Emission Patterns

### Using SafeEventEmitterMixin (Class-Based)

```python
from app.coordination.safe_event_emitter import SafeEventEmitterMixin

class MyManager(SafeEventEmitterMixin):
    _event_source = "MyManager"  # Required class variable

    def process(self):
        # Sync emission
        self._safe_emit_event("MY_EVENT", {"key": "value"})

        # With logging
        self._safe_emit_event(
            "MY_EVENT",
            {"key": "value"},
            log_before="Starting process",
            log_after="Process event emitted",
        )

    async def process_async(self):
        # Async emission
        await self._safe_emit_event_async("MY_EVENT", {"key": "value"})
```

### Using Module-Level Function

```python
from app.coordination.safe_event_emitter import safe_emit_event

def my_function():
    success = safe_emit_event(
        "MY_EVENT",
        {"key": "value"},
        source="my_module",
    )
    if not success:
        logger.warning("Event emission failed, using fallback")
```

### Common Events

| Event         | When to Emit                         |
| ------------- | ------------------------------------ |
| `*_STARTED`   | When operation begins                |
| `*_COMPLETED` | When operation finishes successfully |
| `*_FAILED`    | When operation fails                 |
| `*_PROGRESS`  | For long-running operations          |

## Circuit Breaker Integration

For managers that call external services, integrate circuit breakers:

```python
from app.distributed import (
    CircuitOpenError,
    CircuitState,
    get_host_breaker,
)

class MyManager:
    def __init__(self, orchestrator, config):
        self._orchestrator = orchestrator
        self.config = config
        self._circuit = get_host_breaker()
        self._circuit_target = "my_manager_external"

    async def call_external(self) -> dict:
        """Call external service with circuit breaker protection."""
        if self._circuit.get_state(self._circuit_target) == CircuitState.OPEN:
            raise RuntimeError("Circuit breaker open")

        try:
            async with self._circuit.protected(self._circuit_target):
                return await self._do_external_call()
        except CircuitOpenError as exc:
            raise RuntimeError("Circuit breaker open") from exc

    def health_check(self) -> HealthCheckResult:
        # Include circuit breaker state in health
        status = self._circuit.get_status(self._circuit_target)
        details = {
            "circuit_state": status.state.value,
            "circuit_failure_count": status.failure_count,
        }

        if status.state == CircuitState.OPEN:
            return HealthCheckResult.degraded(
                "Circuit breaker open",
                **details,
            )

        return HealthCheckResult.healthy("OK", **details)
```

## Integration with P2P Orchestrator

### 1. Add Manager to Orchestrator

In `scripts/p2p_orchestrator.py`:

```python
from scripts.p2p.managers.my_new_manager import (
    MyNewManager,
    MyManagerConfig,
    create_my_manager,
)

class P2POrchestrator:
    def __init__(self, ...):
        # ... existing code ...

        # Initialize manager (lazy or eager)
        self._my_manager: MyNewManager | None = None

    @property
    def my_manager(self) -> MyNewManager:
        """Lazy-load manager on first access."""
        if self._my_manager is None:
            self._my_manager = create_my_manager(self)
        return self._my_manager
```

### 2. Wire to Startup/Shutdown

```python
async def start(self):
    # ... existing startup ...
    await self.my_manager.start()

async def shutdown(self):
    # ... existing shutdown ...
    if self._my_manager:
        await self._my_manager.stop()
```

### 3. Expose via HTTP Endpoint (Optional)

```python
@router.get("/my-manager/status")
async def get_my_manager_status():
    """Get manager status."""
    health = self.my_manager.health_check()
    return health.to_dict()
```

## Example: StatusMetricsCollector

The `StatusMetricsCollector` demonstrates these patterns:

**Location:** `scripts/p2p/managers/status_metrics_collector.py`

**Key Features:**

- Parallel metric gathering with `asyncio.gather()`
- Timeout protection per metric call
- Graceful error handling (captures failures without crashing)
- Configuration via dataclass

**Usage:**

```python
from scripts.p2p.managers.status_metrics_collector import (
    StatusMetricsCollector,
    MetricTask,
    CollectorConfig,
)

# Create collector
collector = StatusMetricsCollector(CollectorConfig(metric_timeout=5.0))

# Add metric tasks
collector.add_task(MetricTask("gossip", self._get_gossip_metrics))
collector.add_task(MetricTask("elo", self._get_elo_summary))
collector.add_task(MetricTask("jobs", self._get_job_stats, is_async=True))

# Collect all metrics in parallel
results = await collector.collect_all_metrics()

# Access results
print(results.metrics)  # {"gossip": {...}, "elo": {...}, "jobs": {...}}
print(results.success_count)  # Number of successful collections
print(results.timed_out_count)  # Number of timeouts
```

## Testing Managers

### Unit Test Template

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from scripts.p2p.managers.my_new_manager import (
    MyNewManager,
    MyManagerConfig,
    create_my_manager,
)
from app.coordination.contracts import CoordinatorStatus


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    return MagicMock()


@pytest.fixture
def manager(mock_orchestrator):
    """Create manager with mock orchestrator."""
    return create_my_manager(mock_orchestrator)


class TestMyNewManager:
    """Tests for MyNewManager."""

    async def test_health_check_healthy(self, manager):
        """Test healthy state."""
        await manager.start()
        result = manager.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    async def test_health_check_degraded_on_errors(self, manager):
        """Test degraded state with errors."""
        await manager.start()
        manager._error_count = 7
        result = manager.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED

    async def test_health_check_unhealthy_on_many_errors(self, manager):
        """Test unhealthy state with many errors."""
        await manager.start()
        manager._error_count = 15
        result = manager.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR

    async def test_lifecycle(self, manager):
        """Test start/stop lifecycle."""
        await manager.start()
        assert manager._is_running is True

        await manager.stop()
        assert manager._is_running is False
```

## Checklist for New Managers

- [ ] Place in `scripts/p2p/managers/`
- [ ] Create configuration dataclass with sensible defaults
- [ ] Implement `health_check()` returning `HealthCheckResult`
- [ ] Use `safe_emit_event()` for all event emissions
- [ ] Add circuit breaker for external service calls
- [ ] Implement `start()` and `stop()` lifecycle methods
- [ ] Create factory function for instantiation
- [ ] Add TYPE_CHECKING guard for orchestrator import
- [ ] Write unit tests with mock orchestrator
- [ ] Add to orchestrator's `__init__` (lazy or eager)
- [ ] Wire to startup/shutdown
- [ ] Update `scripts/p2p/managers/README.md`

## See Also

- `app/coordination/contracts.py` - HealthCheckResult and protocol definitions
- `app/coordination/safe_event_emitter.py` - Event emission utilities
- `app/distributed/circuit_breaker.py` - Shared circuit breaker implementation
- `scripts/p2p/managers/README.md` - Manager directory overview
