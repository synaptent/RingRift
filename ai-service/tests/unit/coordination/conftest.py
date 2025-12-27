"""Shared pytest fixtures for coordination tests.

This module provides common fixtures for testing coordination modules,
including mock event buses, temporary databases, and factory functions.
"""

import asyncio
import contextlib
import sqlite3
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# TEMPORARY DATABASE FIXTURES
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Provide a temporary SQLite database path.

    Yields a Path to a temporary database file that is cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_coordination.db"
        yield db_path


@pytest.fixture
def temp_db_connection(temp_db_path):
    """Provide a temporary SQLite connection.

    Yields an open connection that is closed after the test.
    """
    conn = sqlite3.connect(str(temp_db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def coordination_db(temp_db_path):
    """Create a coordination database with standard schema.

    Returns the path to a database with common coordination tables created.
    """
    conn = sqlite3.connect(str(temp_db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS node_state (
            node_id TEXT PRIMARY KEY,
            cpu_percent REAL DEFAULT 0.0,
            gpu_percent REAL DEFAULT 0.0,
            memory_percent REAL DEFAULT 0.0,
            active_jobs INTEGER DEFAULT 0,
            updated_at REAL DEFAULT 0.0,
            data TEXT
        );

        CREATE TABLE IF NOT EXISTS sync_state (
            host TEXT PRIMARY KEY,
            last_sync_time REAL DEFAULT 0.0,
            games_synced INTEGER DEFAULT 0,
            bytes_transferred INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            data TEXT
        );

        CREATE TABLE IF NOT EXISTS task_state (
            task_id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at REAL DEFAULT 0.0,
            started_at REAL,
            completed_at REAL,
            data TEXT
        );

        CREATE TABLE IF NOT EXISTS training_state (
            model_id TEXT PRIMARY KEY,
            epoch INTEGER DEFAULT 0,
            step INTEGER DEFAULT 0,
            loss REAL,
            status TEXT DEFAULT 'idle',
            updated_at REAL DEFAULT 0.0,
            data TEXT
        );
    """)
    conn.commit()
    conn.close()
    return temp_db_path


# =============================================================================
# MOCK EVENT BUS FIXTURES
# =============================================================================


class MockEventBus:
    """Mock event bus for testing event-driven coordination."""

    def __init__(self):
        self.subscribers: dict[Any, list[Callable]] = {}
        self.emitted_events: list[tuple] = []

    def subscribe(self, event_type: Any, handler: Callable) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Any, handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self.subscribers:
            with contextlib.suppress(ValueError):
                self.subscribers[event_type].remove(handler)

    def emit(self, event_type: Any, payload: Any = None) -> None:
        """Emit an event to all subscribers."""
        self.emitted_events.append((event_type, payload))
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                with contextlib.suppress(Exception):
                    handler(payload)

    async def emit_async(self, event_type: Any, payload: Any = None) -> None:
        """Async emit for async handlers."""
        self.emitted_events.append((event_type, payload))
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(payload)
                    else:
                        handler(payload)
                except Exception:
                    pass

    def clear(self) -> None:
        """Clear all subscribers and emitted events."""
        self.subscribers.clear()
        self.emitted_events.clear()

    def get_emitted(self, event_type: Any = None) -> list[tuple]:
        """Get emitted events, optionally filtered by type."""
        if event_type is None:
            return self.emitted_events.copy()
        return [(t, p) for t, p in self.emitted_events if t == event_type]


@pytest.fixture
def mock_event_bus():
    """Provide a mock event bus for testing."""
    return MockEventBus()


@pytest.fixture
def mock_data_events(mock_event_bus):
    """Mock the data_events module with a test event bus.

    Patches app.distributed.data_events to use the mock event bus.
    """
    # Create mock event types
    mock_event_types = MagicMock()
    mock_event_types.DATA_SYNC_COMPLETED = "DATA_SYNC_COMPLETED"
    mock_event_types.DATA_SYNC_FAILED = "DATA_SYNC_FAILED"
    mock_event_types.NEW_GAMES_AVAILABLE = "NEW_GAMES_AVAILABLE"
    mock_event_types.TRAINING_STARTED = "TRAINING_STARTED"
    mock_event_types.TRAINING_COMPLETED = "TRAINING_COMPLETED"
    mock_event_types.TRAINING_FAILED = "TRAINING_FAILED"
    mock_event_types.TASK_STARTED = "TASK_STARTED"
    mock_event_types.TASK_COMPLETED = "TASK_COMPLETED"
    mock_event_types.TASK_FAILED = "TASK_FAILED"
    mock_event_types.RESOURCE_UPDATE = "RESOURCE_UPDATE"
    mock_event_types.CLUSTER_STATE_CHANGED = "CLUSTER_STATE_CHANGED"

    with patch.dict("sys.modules", {
        "app.distributed.data_events": MagicMock(
            DataEventType=mock_event_types,
            get_event_bus=lambda: mock_event_bus,
        )
    }):
        yield mock_event_bus, mock_event_types


# =============================================================================
# NODE RESOURCES FIXTURES
# =============================================================================


@dataclass
class MockNodeResources:
    """Mock NodeResources for testing."""
    node_id: str
    cpu_percent: float = 50.0
    gpu_percent: float = 50.0
    memory_percent: float = 40.0
    disk_percent: float = 30.0
    gpu_memory_percent: float = 45.0
    cpu_count: int = 8
    gpu_count: int = 1
    memory_gb: float = 32.0
    has_gpu: bool = True
    gpu_name: str = "NVIDIA RTX 4090"
    active_jobs: int = 2
    selfplay_jobs: int = 1
    training_jobs: int = 1
    updated_at: float = field(default_factory=time.time)
    orchestrator: str = "test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cpu_percent": self.cpu_percent,
            "gpu_percent": self.gpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "gpu_memory_percent": self.gpu_memory_percent,
            "cpu_count": self.cpu_count,
            "gpu_count": self.gpu_count,
            "memory_gb": self.memory_gb,
            "has_gpu": self.has_gpu,
            "gpu_name": self.gpu_name,
            "active_jobs": self.active_jobs,
            "selfplay_jobs": self.selfplay_jobs,
            "training_jobs": self.training_jobs,
            "updated_at": self.updated_at,
            "orchestrator": self.orchestrator,
        }


@pytest.fixture
def node_resources_factory():
    """Factory for creating MockNodeResources instances."""

    def _create(
        node_id: str = "test-node-1",
        cpu_percent: float = 50.0,
        gpu_percent: float = 50.0,
        memory_percent: float = 40.0,
        has_gpu: bool = True,
        gpu_name: str = "NVIDIA RTX 4090",
        active_jobs: int = 2,
        **kwargs
    ) -> MockNodeResources:
        return MockNodeResources(
            node_id=node_id,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            memory_percent=memory_percent,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            active_jobs=active_jobs,
            **kwargs
        )

    return _create


@pytest.fixture
def sample_nodes(node_resources_factory):
    """Create a sample set of nodes for testing."""
    return [
        node_resources_factory("gpu-server-1", cpu_percent=65.0, gpu_percent=70.0),
        node_resources_factory("gpu-server-2", cpu_percent=55.0, gpu_percent=60.0),
        node_resources_factory("cpu-server-1", cpu_percent=75.0, has_gpu=False, gpu_percent=0.0),
        node_resources_factory("cpu-server-2", cpu_percent=45.0, has_gpu=False, gpu_percent=0.0),
    ]


# =============================================================================
# CLUSTER STATE FIXTURES
# =============================================================================


@dataclass
class MockClusterState:
    """Mock ClusterState for testing."""
    nodes: list[MockNodeResources] = field(default_factory=list)
    total_cpu_util: float = 0.0
    total_gpu_util: float = 0.0
    total_memory_util: float = 0.0
    total_gpu_memory_util: float = 0.0
    gpu_node_count: int = 0
    cpu_node_count: int = 0
    total_jobs: int = 0
    updated_at: float = field(default_factory=time.time)

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from nodes."""
        if not self.nodes:
            return

        cpu_utils = [n.cpu_percent for n in self.nodes if n.cpu_percent > 0]
        gpu_utils = [n.gpu_percent for n in self.nodes if n.has_gpu and n.gpu_percent > 0]
        mem_utils = [n.memory_percent for n in self.nodes if n.memory_percent > 0]

        self.total_cpu_util = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
        self.total_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        self.total_memory_util = sum(mem_utils) / len(mem_utils) if mem_utils else 0.0

        self.gpu_node_count = sum(1 for n in self.nodes if n.has_gpu)
        self.cpu_node_count = len(self.nodes) - self.gpu_node_count
        self.total_jobs = sum(n.active_jobs for n in self.nodes)


@pytest.fixture
def cluster_state_factory(node_resources_factory):
    """Factory for creating MockClusterState instances."""

    def _create(
        node_count: int = 4,
        gpu_nodes: int = 2,
        avg_cpu_util: float = 60.0,
        avg_gpu_util: float = 65.0,
        **kwargs
    ) -> MockClusterState:
        nodes = []
        for i in range(node_count):
            has_gpu = i < gpu_nodes
            nodes.append(node_resources_factory(
                node_id=f"node-{i}",
                cpu_percent=avg_cpu_util + (i * 5 - 10),  # Vary slightly
                gpu_percent=avg_gpu_util if has_gpu else 0.0,
                has_gpu=has_gpu,
            ))

        state = MockClusterState(nodes=nodes, **kwargs)
        state.compute_aggregates()
        return state

    return _create


# =============================================================================
# COORDINATOR FIXTURES
# =============================================================================


@pytest.fixture
def mock_coordinator_base():
    """Create a mock CoordinatorBase for testing."""
    coord = MagicMock()
    coord.name = "test_coordinator"
    coord.is_running = False
    coord.status = "initializing"
    coord._operations_count = 0
    coord._errors_count = 0
    coord._dependencies = {}

    async def mock_start():
        coord.is_running = True
        coord.status = "running"

    async def mock_stop():
        coord.is_running = False
        coord.status = "stopped"

    coord.start = AsyncMock(side_effect=mock_start)
    coord.stop = AsyncMock(side_effect=mock_stop)
    coord.initialize = AsyncMock()
    coord.pause = AsyncMock()
    coord.resume = AsyncMock()

    return coord


# =============================================================================
# QUEUE MONITOR FIXTURES
# =============================================================================


@pytest.fixture
def mock_queue_monitor():
    """Mock queue monitor for backpressure testing."""
    mock = MagicMock()
    mock.check_backpressure = MagicMock(return_value=False)
    mock.should_throttle_production = MagicMock(return_value=False)
    mock.should_stop_production = MagicMock(return_value=False)
    mock.get_throttle_factor = MagicMock(return_value=1.0)
    mock.report_queue_depth = MagicMock()
    return mock


@pytest.fixture
def mock_queue_monitor_patch(mock_queue_monitor):
    """Patch queue monitor imports."""
    with patch.dict("sys.modules", {
        "app.coordination.queue_monitor": mock_queue_monitor
    }):
        yield mock_queue_monitor


# =============================================================================
# SYNC COORDINATOR FIXTURES
# =============================================================================


@dataclass
class MockHostSyncState:
    """Mock host sync state for testing."""
    host: str
    last_sync_time: float = 0.0
    games_available: int = 0
    games_synced: int = 0
    bytes_transferred: int = 0
    sync_failures: int = 0
    is_stale: bool = False
    priority: int = 0


@pytest.fixture
def host_sync_factory():
    """Factory for creating MockHostSyncState instances."""

    def _create(
        host: str = "test-host",
        games_available: int = 100,
        games_synced: int = 50,
        is_stale: bool = False,
        **kwargs
    ) -> MockHostSyncState:
        return MockHostSyncState(
            host=host,
            games_available=games_available,
            games_synced=games_synced,
            is_stale=is_stale,
            **kwargs
        )

    return _create


# =============================================================================
# TRAINING COORDINATOR FIXTURES
# =============================================================================


@dataclass
class MockTrainingState:
    """Mock training state for testing."""
    model_id: str
    epoch: int = 0
    step: int = 0
    loss: float = 1.0
    status: str = "idle"
    started_at: float = 0.0
    updated_at: float = field(default_factory=time.time)


@pytest.fixture
def training_state_factory():
    """Factory for creating MockTrainingState instances."""

    def _create(
        model_id: str = "model-v1",
        epoch: int = 0,
        step: int = 0,
        loss: float = 1.0,
        status: str = "idle",
        **kwargs
    ) -> MockTrainingState:
        return MockTrainingState(
            model_id=model_id,
            epoch=epoch,
            step=step,
            loss=loss,
            status=status,
            **kwargs
        )

    return _create


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def freeze_time():
    """Fixture to freeze time for deterministic tests.

    Usage:
        def test_something(freeze_time):
            with freeze_time(1000.0):
                assert time.time() == 1000.0
    """
    from contextlib import contextmanager

    @contextmanager
    def _freeze(timestamp: float):
        with patch("time.time", return_value=timestamp):
            yield

    return _freeze


@pytest.fixture
def async_timeout():
    """Fixture providing async timeout helper."""

    async def _timeout(coro, timeout_seconds: float = 5.0):
        """Run a coroutine with a timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            pytest.fail(f"Async operation timed out after {timeout_seconds}s")

    return _timeout


# =============================================================================
# MOCK EVENT FIXTURES (Dec 26, 2025)
# =============================================================================


@pytest.fixture
def mock_event():
    """Provide a mock event class for event handler tests.

    Usage:
        def test_handler(mock_event):
            event = mock_event(payload={"key": "value"})
            await coordinator._on_some_event(event)
    """
    @dataclass
    class MockEvent:
        payload: dict[str, Any]
    return MockEvent


# =============================================================================
# COORDINATOR SINGLETON RESET FIXTURES (Dec 26, 2025)
# =============================================================================


@pytest.fixture
def reset_coordinator_singleton():
    """Factory to reset coordinator singletons during tests.

    Usage:
        def test_singleton(reset_coordinator_singleton):
            reset_coordinator_singleton("app.coordination.leadership_coordinator", "_leadership_coordinator")
    """
    def _reset(module_path: str, singleton_attr: str) -> None:
        """Reset a coordinator singleton."""
        import importlib
        module = importlib.import_module(module_path)
        setattr(module, singleton_attr, None)

    return _reset


@pytest.fixture(autouse=False)
def reset_all_coordination_singletons():
    """Reset all coordination singletons before and after test.

    Usage: Add this fixture to tests that need singleton isolation.
    """
    singletons = [
        ("app.coordination.leadership_coordinator", "_leadership_coordinator"),
        ("app.coordination.model_lifecycle_coordinator", "_model_coordinator"),
        ("app.coordination.task_lifecycle_coordinator", "_task_lifecycle_coordinator"),
        ("app.coordination.recovery_orchestrator", "_recovery_orchestrator"),
        ("app.coordination.health_check_orchestrator", "_health_orchestrator"),
        ("app.coordination.resource_monitoring_coordinator", "_resource_coordinator"),
        ("app.coordination.unified_resource_coordinator", "_coordinator"),
        ("app.coordination.optimization_coordinator", "_optimization_coordinator"),
    ]

    import importlib

    # Reset before test
    for module_path, attr in singletons:
        try:
            module = importlib.import_module(module_path)
            setattr(module, attr, None)
        except (ImportError, AttributeError):
            pass  # Module may not be loaded yet

    yield

    # Reset after test
    for module_path, attr in singletons:
        try:
            module = importlib.import_module(module_path)
            setattr(module, attr, None)
        except (ImportError, AttributeError):
            pass


# =============================================================================
# PROVIDER PATCH FIXTURES (Dec 26, 2025)
# =============================================================================


@pytest.fixture
def patch_provider_managers():
    """Context manager factory to patch all provider managers.

    Usage:
        def test_something(patch_provider_managers):
            with patch_provider_managers("app.coordination.health_check_orchestrator"):
                orchestrator = HealthCheckOrchestrator()
    """
    @contextlib.contextmanager
    def _patch_providers(module_path: str):
        """Patch provider managers for a specific coordination module."""
        with patch(f"{module_path}.LambdaManager"), \
             patch(f"{module_path}.VastManager"), \
             patch(f"{module_path}.HetznerManager"), \
             patch(f"{module_path}.AWSManager"), \
             patch(f"{module_path}.TailscaleManager"):
            yield

    return _patch_providers


@pytest.fixture
def health_check_orchestrator_mocked(patch_provider_managers):
    """Create a HealthCheckOrchestrator with mocked dependencies."""
    with patch_provider_managers("app.coordination.health_check_orchestrator"):
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator
        return HealthCheckOrchestrator(check_interval=60.0)


@pytest.fixture
def recovery_orchestrator_mocked(patch_provider_managers):
    """Create a RecoveryOrchestrator with mocked dependencies."""
    with patch("app.coordination.recovery_orchestrator.get_health_orchestrator") as mock_health:
        with patch_provider_managers("app.coordination.recovery_orchestrator"):
            from app.coordination.recovery_orchestrator import RecoveryOrchestrator

            mock_health_instance = MagicMock()
            mock_health.return_value = mock_health_instance

            return RecoveryOrchestrator(
                health_orchestrator=mock_health_instance,
                slack_webhook_url=None,
            )


# =============================================================================
# COORDINATOR FACTORY FIXTURES (Dec 26, 2025)
# =============================================================================


@pytest.fixture
def leadership_coordinator_fresh():
    """Create a fresh LeadershipCoordinator."""
    from app.coordination.leadership_coordinator import LeadershipCoordinator
    return LeadershipCoordinator(local_node_id="test-node-1")


@pytest.fixture
def model_lifecycle_coordinator_fresh():
    """Create a fresh ModelLifecycleCoordinator."""
    from app.coordination.model_lifecycle_coordinator import ModelLifecycleCoordinator
    return ModelLifecycleCoordinator()


@pytest.fixture
def task_lifecycle_coordinator_fresh():
    """Create a fresh TaskLifecycleCoordinator."""
    from app.coordination.task_lifecycle_coordinator import TaskLifecycleCoordinator
    return TaskLifecycleCoordinator(
        heartbeat_threshold_seconds=60.0,
        orphan_check_interval_seconds=1.0,
    )


@pytest.fixture
def resource_monitoring_coordinator_fresh():
    """Create a fresh ResourceMonitoringCoordinator."""
    from app.coordination.resource_monitoring_coordinator import ResourceMonitoringCoordinator
    return ResourceMonitoringCoordinator()


@pytest.fixture
def unified_resource_coordinator_fresh():
    """Create a fresh UnifiedResourceCoordinator."""
    from app.coordination.unified_resource_coordinator import UnifiedResourceCoordinator
    return UnifiedResourceCoordinator()


@pytest.fixture
def optimization_coordinator_fresh():
    """Create a fresh OptimizationCoordinator."""
    from app.coordination.optimization_coordinator import OptimizationCoordinator
    return OptimizationCoordinator(
        plateau_window=5,
        plateau_threshold=0.001,
        cooldown_seconds=60.0,
    )


# =============================================================================
# DAEMON MANAGER FIXTURES (Phase 1 - Dec 2025)
# =============================================================================


@pytest.fixture
def mock_health_check_result():
    """Factory for mock health check results with different formats.

    Supports creating health check results in various formats that DaemonManager
    should handle: bool, dict, object with .healthy attribute, async callable.

    Usage:
        def test_health_check(mock_health_check_result):
            result = mock_health_check_result("bool", healthy=True)
            assert result() is True

            result = mock_health_check_result("dict", healthy=True, latency_ms=50)
            assert result()["healthy"] is True
    """
    def _create(
        result_type: str = "bool",
        healthy: bool = True,
        **extra_fields
    ):
        """Create a health check result factory.

        Args:
            result_type: One of "bool", "dict", "object", "async"
            healthy: Whether the result indicates healthy state
            **extra_fields: Additional fields for dict/object types

        Returns:
            A callable that returns the health check result
        """
        if result_type == "bool":
            def _result():
                return healthy
            return _result

        elif result_type == "dict":
            def _result():
                return {"healthy": healthy, **extra_fields}
            return _result

        elif result_type == "object":
            class HealthResult:
                def __init__(self):
                    self.healthy = healthy
                    for k, v in extra_fields.items():
                        setattr(self, k, v)
            def _result():
                return HealthResult()
            return _result

        elif result_type == "async":
            async def _result():
                return {"healthy": healthy, **extra_fields}
            return _result

        else:
            raise ValueError(f"Unknown result_type: {result_type}")

    return _create


@pytest.fixture
def daemon_with_slow_health_check():
    """Factory for daemons with configurable slow health checks.

    Creates a daemon factory and corresponding health check function with
    configurable latency for testing health check timeout handling.

    Usage:
        def test_slow_health(daemon_with_slow_health_check):
            factory, health_fn = daemon_with_slow_health_check(
                health_delay=2.0,
                factory_name="slow_daemon"
            )
    """
    def _create(
        health_delay: float = 1.0,
        factory_name: str = "slow_health_daemon",
        healthy: bool = True,
    ):
        """Create daemon factory with slow health check.

        Args:
            health_delay: Seconds to sleep in health check
            factory_name: Name for debugging
            healthy: Health status to return after delay

        Returns:
            Tuple of (daemon_factory, health_check_fn)
        """
        import asyncio

        async def daemon_factory():
            """Daemon that runs forever."""
            while True:
                await asyncio.sleep(1)

        async def health_check():
            """Health check with configurable delay."""
            await asyncio.sleep(health_delay)
            return healthy

        return daemon_factory, health_check

    return _create


@pytest.fixture
def diamond_dependency_setup():
    """Factory for diamond dependency pattern daemon setup.

    Creates the classic diamond dependency graph:
        EVENT_ROUTER
          /        \\
    DATA_PIPELINE  AUTO_SYNC
          \\        /
        FEEDBACK_LOOP

    Usage:
        def test_diamond(diamond_dependency_setup, manager):
            factories, counters = diamond_dependency_setup()
            for dtype, factory in factories.items():
                manager.register_factory(dtype, factory, ...)
    """
    def _create():
        """Create diamond dependency factories with start counters.

        Returns:
            Tuple of (factories_dict, start_counters_dict)
        """
        from app.coordination.daemon_types import DaemonType
        import asyncio

        start_counters = {
            DaemonType.EVENT_ROUTER: 0,
            DaemonType.DATA_PIPELINE: 0,
            DaemonType.AUTO_SYNC: 0,
            DaemonType.FEEDBACK_LOOP: 0,
        }

        def make_factory(dtype: DaemonType):
            async def factory():
                start_counters[dtype] += 1
                while True:
                    await asyncio.sleep(1)
            return factory

        factories = {
            dtype: make_factory(dtype) for dtype in start_counters.keys()
        }

        dependencies = {
            DaemonType.EVENT_ROUTER: [],
            DaemonType.DATA_PIPELINE: [DaemonType.EVENT_ROUTER],
            DaemonType.AUTO_SYNC: [DaemonType.EVENT_ROUTER],
            DaemonType.FEEDBACK_LOOP: [DaemonType.DATA_PIPELINE, DaemonType.AUTO_SYNC],
        }

        return factories, start_counters, dependencies

    return _create


@pytest.fixture
def manager_with_import_error_daemon():
    """Factory for DaemonManager with a daemon in IMPORT_FAILED state.

    Creates a DaemonManager with a pre-configured daemon that has an import
    error, useful for testing import error handling behavior.

    Usage:
        def test_import_error(manager_with_import_error_daemon):
            manager, dtype = manager_with_import_error_daemon(
                error_message="ModuleNotFoundError: No module named 'missing'"
            )
    """
    def _create(
        error_message: str = "ModuleNotFoundError: No module named 'test_module'",
        daemon_type=None,
    ):
        """Create manager with import-failed daemon.

        Args:
            error_message: The import error message
            daemon_type: DaemonType to use (default: MODEL_SYNC)

        Returns:
            Tuple of (manager, daemon_type)
        """
        from app.coordination.daemon_manager import (
            DaemonInfo,
            DaemonManager,
            DaemonManagerConfig,
            DaemonState,
            DaemonType,
        )

        if daemon_type is None:
            daemon_type = DaemonType.MODEL_SYNC

        # Reset and create fresh manager
        DaemonManager.reset_instance()
        config = DaemonManagerConfig(
            health_check_interval=0.1,
            shutdown_timeout=1.0,
            auto_restart_failed=True,
        )
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()

        # Create daemon info with import error
        info = DaemonInfo(daemon_type=daemon_type)
        info.state = DaemonState.FAILED
        info.import_error = error_message
        info.last_failure_time = 0  # Long ago
        info.auto_restart = True
        info.max_restarts = 5
        info.restart_count = 0

        manager._daemons[daemon_type] = info

        return manager, daemon_type

    return _create
