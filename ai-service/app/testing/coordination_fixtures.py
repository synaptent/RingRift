"""Coordination test fixtures and utilities.

This module provides reusable fixtures for testing coordination modules,
including mock event buses, temporary databases, and factory functions.

Usage:
    from app.testing.coordination_fixtures import (
        MockEventBus,
        MockNodeResources,
        create_temp_db,
        create_coordination_db,
    )
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = [
    "MockClusterState",
    "MockEventBus",
    "MockHostSyncState",
    "MockNodeResources",
    "MockTrainingState",
    "create_coordination_db",
    "create_temp_db",
]


# =============================================================================
# TEMPORARY DATABASE UTILITIES
# =============================================================================


def create_temp_db() -> Path:
    """Create a temporary SQLite database path.

    Returns:
        Path to a temporary database file.

    Note:
        The caller is responsible for cleaning up the temporary directory.
        Use within a contextmanager or ensure proper cleanup.
    """
    tmpdir = tempfile.mkdtemp()
    return Path(tmpdir) / "test_coordination.db"


def create_coordination_db(db_path: Path | None = None) -> Path:
    """Create a coordination database with standard schema.

    Args:
        db_path: Path for the database. If None, creates a temp database.

    Returns:
        Path to the created database.
    """
    if db_path is None:
        db_path = create_temp_db()

    conn = sqlite3.connect(str(db_path))
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
    return db_path


# =============================================================================
# MOCK EVENT BUS
# =============================================================================


class MockEventBus:
    """Mock event bus for testing event-driven coordination.

    Usage:
        bus = MockEventBus()
        bus.subscribe("my_event", my_handler)
        bus.emit("my_event", {"key": "value"})
        assert len(bus.get_emitted("my_event")) == 1
    """

    def __init__(self) -> None:
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


# =============================================================================
# MOCK NODE RESOURCES
# =============================================================================


@dataclass
class MockNodeResources:
    """Mock NodeResources for testing cluster state."""

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
        """Convert to dictionary representation."""
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


def create_node_resources(
    node_id: str = "test-node-1",
    cpu_percent: float = 50.0,
    gpu_percent: float = 50.0,
    memory_percent: float = 40.0,
    has_gpu: bool = True,
    gpu_name: str = "NVIDIA RTX 4090",
    active_jobs: int = 2,
    **kwargs,
) -> MockNodeResources:
    """Create a MockNodeResources instance.

    Args:
        node_id: Node identifier
        cpu_percent: CPU utilization percentage
        gpu_percent: GPU utilization percentage
        memory_percent: Memory utilization percentage
        has_gpu: Whether node has GPU
        gpu_name: GPU model name
        active_jobs: Number of active jobs

    Returns:
        MockNodeResources instance
    """
    return MockNodeResources(
        node_id=node_id,
        cpu_percent=cpu_percent,
        gpu_percent=gpu_percent,
        memory_percent=memory_percent,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        active_jobs=active_jobs,
        **kwargs,
    )


# =============================================================================
# MOCK CLUSTER STATE
# =============================================================================


@dataclass
class MockClusterState:
    """Mock ClusterState for testing cluster management."""

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


def create_cluster_state(
    node_count: int = 4,
    gpu_nodes: int = 2,
    avg_cpu_util: float = 60.0,
    avg_gpu_util: float = 65.0,
    **kwargs,
) -> MockClusterState:
    """Create a MockClusterState instance.

    Args:
        node_count: Total number of nodes
        gpu_nodes: Number of nodes with GPUs
        avg_cpu_util: Average CPU utilization
        avg_gpu_util: Average GPU utilization

    Returns:
        MockClusterState instance with computed aggregates
    """
    nodes = []
    for i in range(node_count):
        has_gpu = i < gpu_nodes
        nodes.append(create_node_resources(
            node_id=f"node-{i}",
            cpu_percent=avg_cpu_util + (i * 5 - 10),
            gpu_percent=avg_gpu_util if has_gpu else 0.0,
            has_gpu=has_gpu,
        ))

    state = MockClusterState(nodes=nodes, **kwargs)
    state.compute_aggregates()
    return state


# =============================================================================
# MOCK HOST SYNC STATE
# =============================================================================


@dataclass
class MockHostSyncState:
    """Mock host sync state for testing data synchronization."""

    host: str
    last_sync_time: float = 0.0
    games_available: int = 0
    games_synced: int = 0
    bytes_transferred: int = 0
    sync_failures: int = 0
    is_stale: bool = False
    priority: int = 0


def create_host_sync_state(
    host: str = "test-host",
    games_available: int = 100,
    games_synced: int = 50,
    is_stale: bool = False,
    **kwargs,
) -> MockHostSyncState:
    """Create a MockHostSyncState instance.

    Args:
        host: Host identifier
        games_available: Number of games available
        games_synced: Number of games synced
        is_stale: Whether sync state is stale

    Returns:
        MockHostSyncState instance
    """
    return MockHostSyncState(
        host=host,
        games_available=games_available,
        games_synced=games_synced,
        is_stale=is_stale,
        **kwargs,
    )


# =============================================================================
# MOCK TRAINING STATE
# =============================================================================


@dataclass
class MockTrainingState:
    """Mock training state for testing training coordination."""

    model_id: str
    epoch: int = 0
    step: int = 0
    loss: float = 1.0
    status: str = "idle"
    started_at: float = 0.0
    updated_at: float = field(default_factory=time.time)


def create_training_state(
    model_id: str = "model-v1",
    epoch: int = 0,
    step: int = 0,
    loss: float = 1.0,
    status: str = "idle",
    **kwargs,
) -> MockTrainingState:
    """Create a MockTrainingState instance.

    Args:
        model_id: Model identifier
        epoch: Current epoch
        step: Current step
        loss: Current loss
        status: Training status

    Returns:
        MockTrainingState instance
    """
    return MockTrainingState(
        model_id=model_id,
        epoch=epoch,
        step=step,
        loss=loss,
        status=status,
        **kwargs,
    )
