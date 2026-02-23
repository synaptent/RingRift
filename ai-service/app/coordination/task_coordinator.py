"""
Global Task Coordinator for RingRift AI.

Prevents uncoordinated task spawning across multiple orchestrators by providing:
1. Global task registry with hard limits
2. Mutual exclusion between orchestrators via file locks
3. Rate limiting and backpressure mechanisms
4. Emergency shutdown capability
5. Resource-aware admission control

This module MUST be used by all orchestrators to prevent runaway task spawning.

Architecture Relationship (December 2025):
-----------------------------------------
This module is part of a layered coordination architecture:

1. **TaskCoordinator** (this module)
   - Canonical for TASK ADMISSION CONTROL
   - Decides how many tasks can run based on limits/resources
   - Used by all task spawning code

2. **OrchestratorRegistry** (:mod:`app.coordination.orchestrator_registry`)
   - Canonical for ROLE-BASED COORDINATION
   - Ensures only one orchestrator per role (cluster_orchestrator, etc.)
   - Uses heartbeat-based liveness detection

3. **TrainingCoordinator** (:mod:`app.coordination.training_coordinator`)
   - Specialized facade for TRAINING COORDINATION
   - Adds NFS-based locking for GH200 cluster
   - Delegates to DistributedLock for low-level locking

These modules work together but serve different purposes:
- TaskCoordinator answers: "Can I spawn another task?"
- OrchestratorRegistry answers: "Am I the designated orchestrator?"
- TrainingCoordinator answers: "Can I start training this config?"

Usage:
    coordinator = TaskCoordinator.get_instance()

    # Before spawning any task
    if coordinator.can_spawn_task(TaskType.SELFPLAY, node_id="node-1"):
        coordinator.register_task(task_id, TaskType.SELFPLAY, node_id="node-1")
        # ... spawn task ...
    else:
        logger.warning("Task spawn denied - limits exceeded")

    # When task completes
    coordinator.unregister_task(task_id)
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.coordination.singleton_mixin import SingletonMixin

# December 30, 2025: Extracted ReservationManager as part of god class refactoring
from app.coordination.task_coordinator_reservations import (
    ReservationManager,
    get_reservation_manager,
)

logger = logging.getLogger(__name__)

# January 2026: Migrated to safe_emit_event_async for consistent event handling
from app.coordination.event_emission_helpers import safe_emit_event_async
HAS_CENTRALIZED_EMITTERS = True

# Import queue monitor for backpressure checks
try:
    from app.coordination.queue_monitor import (
        BackpressureLevel,
        QueueType,
        get_queue_monitor,
        get_throttle_factor,
    )
    HAS_QUEUE_MONITOR = True
except ImportError:
    HAS_QUEUE_MONITOR = False
    QueueType = None
    BackpressureLevel = None

# Import host health policy for pre-spawn health checks (December 2025)
try:
    from app.coordination.host_health_policy import (
        check_host_health,
        is_host_healthy,
        mark_host_unhealthy,
        pre_spawn_check,
    )
    HAS_HOST_HEALTH = True
except ImportError:
    HAS_HOST_HEALTH = False
    is_host_healthy = None
    pre_spawn_check = None
    check_host_health = None
    mark_host_unhealthy = None


# ============================================
# Configuration
# ============================================

# TaskType consolidated into app.coordination.types (December 2025)
# Import from canonical source for consistency across codebase
from app.coordination.types import TaskType  # noqa: E402


# December 2025: Import ResourceType from canonical source
from app.coordination.types import ResourceType

# ResourceType is now imported from app.coordination.types
# Canonical values: CPU, GPU, MEMORY, DISK, NETWORK, HYBRID, IO


# Map task types to their primary resource usage
TASK_RESOURCE_MAP: dict = {
    TaskType.SELFPLAY: ResourceType.CPU,
    TaskType.GPU_SELFPLAY: ResourceType.GPU,
    TaskType.HYBRID_SELFPLAY: ResourceType.HYBRID,
    TaskType.TRAINING: ResourceType.GPU,
    TaskType.CMAES: ResourceType.GPU,
    TaskType.TOURNAMENT: ResourceType.CPU,
    TaskType.EVALUATION: ResourceType.CPU,
    TaskType.SYNC: ResourceType.IO,
    TaskType.EXPORT: ResourceType.IO,
    TaskType.PIPELINE: ResourceType.HYBRID,
    TaskType.IMPROVEMENT_LOOP: ResourceType.HYBRID,
    TaskType.BACKGROUND_LOOP: ResourceType.CPU,
}


def get_task_resource_type(task_type: TaskType) -> ResourceType:
    """Get the primary resource type for a task.

    This allows orchestrators to make resource-aware scheduling decisions,
    ensuring CPU-bound and GPU-bound tasks don't block each other.

    Args:
        task_type: The task type to classify

    Returns:
        The primary resource type (CPU, GPU, HYBRID, or IO)
    """
    return TASK_RESOURCE_MAP.get(task_type, ResourceType.CPU)


def is_gpu_task(task_type: TaskType) -> bool:
    """Check if a task is GPU-bound.

    GPU tasks should only be gated by GPU utilization, not CPU.
    """
    resource = get_task_resource_type(task_type)
    return resource in (ResourceType.GPU, ResourceType.HYBRID)


def is_cpu_task(task_type: TaskType) -> bool:
    """Check if a task is CPU-bound.

    CPU tasks should only be gated by CPU utilization, not GPU.
    """
    resource = get_task_resource_type(task_type)
    return resource in (ResourceType.CPU, ResourceType.HYBRID)


# Map task types to relevant queue types for backpressure checks (December 2025)
# Some tasks produce data for queues, others consume from queues
# We check producer queues before spawning producer tasks
TASK_TO_QUEUE_MAP: dict = {}
if HAS_QUEUE_MONITOR:
    TASK_TO_QUEUE_MAP = {
        TaskType.SELFPLAY: QueueType.TRAINING_DATA,      # Produces training data
        TaskType.GPU_SELFPLAY: QueueType.TRAINING_DATA,  # Produces training data
        TaskType.HYBRID_SELFPLAY: QueueType.TRAINING_DATA,
        TaskType.TRAINING: QueueType.EVALUATION_QUEUE,   # Produces models for eval
        TaskType.EVALUATION: QueueType.PROMOTION_QUEUE,  # Produces promotion candidates
        TaskType.SYNC: QueueType.SYNC_QUEUE,             # Sync queue itself
        TaskType.EXPORT: QueueType.EXPORT_QUEUE,         # Export queue itself
    }


def get_queue_for_task(task_type: TaskType) -> Optional["QueueType"]:
    """Get the queue that a task produces to.

    Returns None if the task doesn't have a relevant queue for backpressure.
    """
    return TASK_TO_QUEUE_MAP.get(task_type)


@dataclass
class TaskLimits:
    """Global limits for task spawning."""
    # Per-node limits
    max_selfplay_per_node: int = 32
    max_training_per_node: int = 1
    max_sync_per_node: int = 2
    max_export_per_node: int = 2

    # Cluster-wide limits
    max_total_selfplay: int = 500
    max_total_training: int = 3
    max_total_cmaes: int = 1
    max_total_tournaments: int = 2
    max_total_pipelines: int = 1
    max_total_improvement_loops: int = 1

    # Rate limits (per minute)
    max_task_spawns_per_minute: int = 60
    max_selfplay_spawns_per_minute: int = 30

    # Resource thresholds (halt spawning above these) - raised from 70% to 85% (was starving pipeline)
    halt_on_disk_percent: float = 85.0
    halt_on_memory_percent: float = 95.0
    halt_on_cpu_percent: float = 95.0

    # Backpressure settings
    soft_limit_factor: float = 0.8  # Warn at 80% of limit
    spawn_cooldown_seconds: float = 1.0  # Min time between spawns

    @classmethod
    def conservative(cls) -> 'TaskLimits':
        """Conservative limits for resource-constrained environments."""
        return cls(
            max_selfplay_per_node=8,
            max_total_selfplay=100,
            max_task_spawns_per_minute=20,
            max_selfplay_spawns_per_minute=10,
        )

    @classmethod
    def aggressive(cls) -> 'TaskLimits':
        """Aggressive limits for high-capacity clusters."""
        return cls(
            max_selfplay_per_node=64,
            max_total_selfplay=1000,
            max_task_spawns_per_minute=120,
            max_selfplay_spawns_per_minute=60,
        )


@dataclass
class TaskInfo:
    """Information about a registered task."""
    task_id: str
    task_type: TaskType
    node_id: str
    started_at: float
    pid: int = 0
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 3600.0  # Default 1 hour max runtime

    def is_timed_out(self) -> bool:
        """Check if task has exceeded its timeout."""
        if self.status in ("completed", "failed", "cancelled", "orphaned"):
            return False
        return time.time() - self.started_at > self.timeout_seconds

    def runtime_seconds(self) -> float:
        """Get task runtime in seconds."""
        return time.time() - self.started_at


# December 2025: Import CoordinatorRunState from canonical source
from app.coordination.types import CoordinatorRunState

# CoordinatorRunState is now imported from app.coordination.types
# Canonical values: RUNNING, PAUSED, DRAINING, EMERGENCY, STOPPED
# Backward-compat alias:
CoordinatorState = CoordinatorRunState


# ============================================
# Orchestrator Lock
# ============================================

class OrchestratorLock:
    """
    File-based lock to ensure only one orchestrator runs at a time.

    This prevents multiple orchestrators from spawning tasks simultaneously.
    """

    def __init__(self, lock_name: str = "orchestrator"):
        lock_dir = Path("/tmp/ringrift_locks")
        lock_dir.mkdir(exist_ok=True)
        self.lock_file = lock_dir / f"{lock_name}.lock"
        self._fd: int | None = None
        self._holder_pid: int | None = None

    def acquire(self, blocking: bool = False, timeout: float = 10.0) -> bool:
        """
        Acquire the orchestrator lock.

        Returns True if lock acquired, False otherwise.
        """
        try:
            self._fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR)

            if blocking:
                start = time.time()
                while time.time() - start < timeout:
                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.1)
                else:
                    return False
            else:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(self._fd)
                    self._fd = None
                    return False

            # Write our PID to the lock file
            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, f"{os.getpid()}\n".encode())
            self._holder_pid = os.getpid()

            return True

        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            if self._fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        """Release the orchestrator lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            self._holder_pid = None

    def get_holder(self) -> int | None:
        """Get PID of current lock holder, if any."""
        try:
            if self.lock_file.exists():
                content = self.lock_file.read_text().strip()
                if content:
                    return int(content)
        except (OSError, ValueError):
            pass
        return None

    def is_held(self) -> bool:
        """Check if lock is currently held."""
        holder = self.get_holder()
        if holder is None:
            return False

        # Check if holder process is alive
        try:
            os.kill(holder, 0)
            return True
        except OSError:
            return False

    def __enter__(self):
        if not self.acquire(blocking=True):
            raise RuntimeError("Failed to acquire orchestrator lock")
        return self

    def __exit__(self, *args):
        self.release()


# ============================================
# Rate Limiter
# ============================================

class RateLimiter:
    """Token bucket rate limiter for task spawning."""

    def __init__(self, rate: float, burst: int = 10):
        """
        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_update = time.time()
        self._lock = threading.RLock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def tokens_available(self) -> float:
        """Get current token count."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            return min(self.burst, self._tokens + elapsed * self.rate)


# ============================================
# Task Registry Database
# ============================================

class TaskRegistry:
    """
    SQLite-based task registry for persistence across restarts.

    Stores active tasks and allows recovery after crashes.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite database connection for task tracking.

        Lazily initializes connection on first access for the current thread.
        Each thread gets its own connection to avoid SQLite threading issues.

        Returns:
            sqlite3.Connection with Row factory enabled for dict-like access.
        """
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    pid INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_node ON tasks(node_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

                CREATE TABLE IF NOT EXISTS spawn_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    spawned_at REAL NOT NULL,
                    allowed INTEGER NOT NULL,
                    reason TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_spawn_log_time ON spawn_log(spawned_at);

                CREATE TABLE IF NOT EXISTS coordinator_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def register_task(self, task: TaskInfo) -> None:
        """Register a new task."""
        self.conn.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, task_type, node_id, started_at, pid, status, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.task_type.value,
            task.node_id,
            task.started_at,
            task.pid,
            task.status,
            json.dumps(task.metadata),
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def unregister_task(self, task_id: str) -> None:
        """Remove a task from registry."""
        self.conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        self.conn.commit()

    def update_task_status(self, task_id: str, status: str) -> None:
        """Update task status."""
        self.conn.execute(
            "UPDATE tasks SET status = ? WHERE task_id = ?",
            (status, task_id)
        )
        self.conn.commit()

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get a task by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_task(row)

    def get_tasks_by_node(self, node_id: str) -> list[TaskInfo]:
        """Get all tasks for a node."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE node_id = ? AND status = 'running'",
            (node_id,)
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_tasks_by_type(self, task_type: TaskType) -> list[TaskInfo]:
        """Get all tasks of a type."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE task_type = ? AND status = 'running'",
            (task_type.value,)
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_all_running_tasks(self) -> list[TaskInfo]:
        """Get all running tasks."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE status = 'running'"
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def count_by_type(self, task_type: TaskType) -> int:
        """Count running tasks of a type."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_type = ? AND status = 'running'",
            (task_type.value,)
        )
        return cursor.fetchone()[0]

    def count_by_node(self, node_id: str, task_type: TaskType | None = None) -> int:
        """Count running tasks on a node."""
        if task_type:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE node_id = ? AND task_type = ? AND status = 'running'",
                (node_id, task_type.value)
            )
        else:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE node_id = ? AND status = 'running'",
                (node_id,)
            )
        return cursor.fetchone()[0]

    def log_spawn_attempt(
        self,
        task_type: TaskType,
        node_id: str,
        allowed: bool,
        reason: str = ""
    ) -> None:
        """Log a spawn attempt for auditing."""
        self.conn.execute("""
            INSERT INTO spawn_log (task_type, node_id, spawned_at, allowed, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (task_type.value, node_id, time.time(), int(allowed), reason))
        self.conn.commit()

    def get_spawn_count(self, minutes: int = 1) -> int:
        """Get spawn count in recent minutes."""
        cutoff = time.time() - (minutes * 60)
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM spawn_log WHERE spawned_at > ? AND allowed = 1",
            (cutoff,)
        )
        return cursor.fetchone()[0]

    def cleanup_stale_tasks(self, max_age_hours: float = 24.0) -> int:
        """Remove tasks older than max_age."""
        cutoff = time.time() - (max_age_hours * 3600)
        cursor = self.conn.execute(
            "DELETE FROM tasks WHERE started_at < ?", (cutoff,)
        )
        count = cursor.rowcount
        self.conn.commit()
        return count

    def set_state(self, key: str, value: str) -> None:
        """Set coordinator state."""
        self.conn.execute("""
            INSERT OR REPLACE INTO coordinator_state (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))
        self.conn.commit()

    def get_state(self, key: str) -> str | None:
        """Get coordinator state."""
        cursor = self.conn.execute(
            "SELECT value FROM coordinator_state WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _row_to_task(self, row) -> TaskInfo:
        """Convert database row to TaskInfo."""
        metadata = json.loads(row['metadata_json'] or '{}')
        return TaskInfo(
            task_id=row['task_id'],
            task_type=TaskType(row['task_type']),
            node_id=row['node_id'],
            started_at=row['started_at'],
            pid=row['pid'],
            status=row['status'],
            metadata=metadata,
            timeout_seconds=metadata.get('timeout_seconds', 3600.0),
        )

    def update_heartbeat(self, task_id: str) -> None:
        """Update task heartbeat timestamp."""
        self.conn.execute("""
            UPDATE tasks SET metadata_json = json_set(
                COALESCE(metadata_json, '{}'),
                '$.last_heartbeat',
                ?
            ) WHERE task_id = ?
        """, (time.time(), task_id))
        self.conn.commit()

    def get_orphaned_tasks(self, timeout_seconds: float = 300.0) -> list[TaskInfo]:
        """Get tasks that haven't sent heartbeat within timeout.

        Args:
            timeout_seconds: Seconds without heartbeat to consider task orphaned

        Returns:
            List of orphaned TaskInfo objects
        """
        cutoff = time.time() - timeout_seconds
        cursor = self.conn.execute("""
            SELECT * FROM tasks
            WHERE status = 'running'
            AND (
                json_extract(metadata_json, '$.last_heartbeat') IS NULL
                OR json_extract(metadata_json, '$.last_heartbeat') < ?
            )
            AND started_at < ?
        """, (cutoff, cutoff))
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_timed_out_tasks(self) -> list[TaskInfo]:
        """Get tasks that have exceeded their timeout.

        Returns:
            List of timed-out TaskInfo objects
        """
        cursor = self.conn.execute("""
            SELECT * FROM tasks
            WHERE status = 'running'
        """)
        tasks = [self._row_to_task(row) for row in cursor.fetchall()]
        return [t for t in tasks if t.is_timed_out()]


# ============================================
# Task Heartbeat Monitor (December 2025)
# ============================================

class TaskHeartbeatMonitor:
    """Monitors tasks for heartbeat timeouts and task duration timeouts.

    Features:
    - Detects orphaned tasks (no heartbeat within timeout)
    - Detects timed-out tasks (exceeded their maximum runtime)
    - Emits events for monitoring/alerting

    Usage:
        monitor = TaskHeartbeatMonitor(registry, timeout=300)
        monitor.start()  # Background thread
        ...
        monitor.stop()
    """

    def __init__(
        self,
        registry: TaskRegistry,
        timeout_seconds: float = 300.0,
        check_interval_seconds: float = 60.0,
    ):
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval_seconds
        self._running = False
        self._thread: threading.Thread | None = None

        # January 2026: Migrated to safe_emit_event_async - always available
        self._emit_orphaned_enabled = True

    def start(self) -> None:
        """Start monitoring in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"[HeartbeatMonitor] Started (timeout={self.timeout_seconds}s)")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[HeartbeatMonitor] Stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self.check_for_orphans()
                self.check_for_timeouts()
            except Exception as e:
                logger.warning(f"[HeartbeatMonitor] Check failed: {e}")
            time.sleep(self.check_interval)

    def check_for_orphans(self) -> list[TaskInfo]:
        """Check for orphaned tasks and emit events."""
        orphans = self.registry.get_orphaned_tasks(self.timeout_seconds)

        for task in orphans:
            logger.warning(
                f"[HeartbeatMonitor] Task orphaned: {task.task_id} "
                f"(type={task.task_type.value}, node={task.node_id})"
            )

            # Mark as orphaned
            self.registry.update_task_status(task.task_id, "orphaned")

            # Emit event
            if self._emit_orphaned_enabled:
                try:
                    import asyncio
                    last_hb = task.metadata.get('last_heartbeat', task.started_at)
                    asyncio.get_running_loop().create_task(safe_emit_event_async(
                        "TASK_ORPHANED",
                        {
                            "task_id": task.task_id,
                            "task_type": task.task_type.value,
                            "node_id": task.node_id,
                            "last_heartbeat_seconds_ago": time.time() - last_hb,
                        },
                        context="heartbeat_monitor",
                    ))
                except RuntimeError:
                    pass  # No event loop

        return orphans

    def check_for_timeouts(self) -> list[TaskInfo]:
        """Check for tasks that have exceeded their timeout and cancel them.

        Tasks are marked as 'timed_out' and a TASK_TIMEOUT event is emitted.
        """
        timed_out = self.registry.get_timed_out_tasks()

        for task in timed_out:
            runtime = task.runtime_seconds()
            logger.warning(
                f"[HeartbeatMonitor] Task timed out: {task.task_id} "
                f"(type={task.task_type.value}, runtime={runtime:.0f}s, "
                f"timeout={task.timeout_seconds}s)"
            )

            # Mark as timed out
            self.registry.update_task_status(task.task_id, "timed_out")

            # Emit event (using orphaned event for now, could add dedicated event)
            if self._emit_orphaned_enabled:
                try:
                    import asyncio
                    asyncio.get_running_loop().create_task(safe_emit_event_async(
                        "TASK_ORPHANED",
                        {
                            "task_id": task.task_id,
                            "task_type": task.task_type.value,
                            "node_id": task.node_id,
                            "last_heartbeat_seconds_ago": 0,  # N/A for timeout
                        },
                        context="timeout_monitor",
                    ))
                except RuntimeError:
                    pass  # No event loop

        return timed_out

    def health_check(self) -> "HealthCheckResult":
        """Return health status of the heartbeat monitor.

        Returns:
            HealthCheckResult with monitor health status
        """
        from app.coordination.protocols import HealthCheckResult

        is_healthy = self._running and self._thread is not None and self._thread.is_alive()

        details = {
            "running": self._running,
            "thread_alive": self._thread is not None and self._thread.is_alive(),
            "timeout_seconds": self.timeout_seconds,
            "check_interval": self.check_interval,
        }

        if is_healthy:
            return HealthCheckResult(
                status="healthy",
                message="TaskHeartbeatMonitor running",
                details=details,
            )
        elif not self._running:
            return HealthCheckResult(
                status="stopped",
                message="TaskHeartbeatMonitor stopped",
                details=details,
            )
        else:
            return HealthCheckResult(
                status="degraded",
                message="TaskHeartbeatMonitor thread not alive",
                details=details,
            )


# ============================================
# Task Coordinator
# ============================================

class TaskCoordinator(SingletonMixin):
    """
    Global task coordinator singleton.

    Provides centralized control over task spawning across all orchestrators.

    December 27, 2025: Migrated to SingletonMixin (Wave 4 Phase 1).
    """

    @classmethod
    def reset_instance(cls) -> None:
        """Override to call _shutdown() before clearing instance."""
        with cls._get_lock():
            if cls.has_instance():
                instance = cls.get_instance()
                instance._shutdown()
            super().reset_instance()

    def __init__(self):
        # Configuration
        self.limits = TaskLimits()

        # State
        self.state = CoordinatorState.RUNNING
        self._state_lock = threading.RLock()

        # Data directory
        data_dir = Path(os.environ.get(
            "RINGRIFT_COORDINATOR_DIR",
            "/tmp/ringrift_coordinator"
        ))
        data_dir.mkdir(parents=True, exist_ok=True)

        # Registry
        self.registry = TaskRegistry(data_dir / "tasks.db")

        # Rate limiters
        self._spawn_limiter = RateLimiter(
            rate=self.limits.max_task_spawns_per_minute / 60,
            burst=10
        )
        self._selfplay_limiter = RateLimiter(
            rate=self.limits.max_selfplay_spawns_per_minute / 60,
            burst=5
        )

        # Callbacks
        self._on_limit_reached: list[Callable] = []
        self._on_emergency: list[Callable] = []

        # Last spawn time per node (for cooldown)
        self._last_spawn: dict[str, float] = {}

        # Resource cache (refreshed periodically)
        self._resource_cache: dict[str, dict[str, float]] = {}
        self._resource_cache_time: float = 0

        # December 30, 2025: Delegate reservations to extracted ReservationManager
        # This was previously inline state, now managed by ReservationManager singleton
        self._reservation_manager = get_reservation_manager()

        # Heartbeat monitor for orphan detection (December 2025)
        self._heartbeat_monitor = TaskHeartbeatMonitor(
            registry=self.registry,
            timeout_seconds=300.0,  # 5 minutes
            check_interval_seconds=60.0,  # Check every minute
        )
        self._heartbeat_monitor.start()

        # Resource recovery loop for auto-resumption (December 2025)
        self._resource_recovery_task: threading.Thread | None = None
        self._resource_recovery_interval = 30.0  # Check every 30s
        self._resource_recovery_threshold = 0.90  # Resume when below 90% of critical
        self._paused_due_to_resources = False
        self._stop_recovery_loop = threading.Event()

        logger.info("Task coordinator initialized (with heartbeat monitor)")

    def _shutdown(self) -> None:
        """Cleanup on shutdown."""
        # Stop resource recovery loop
        if hasattr(self, '_stop_recovery_loop'):
            self._stop_recovery_loop.set()
        if hasattr(self, '_resource_recovery_task') and self._resource_recovery_task:
            self._resource_recovery_task.join(timeout=5.0)

        if hasattr(self, '_heartbeat_monitor'):
            self._heartbeat_monitor.stop()

    # ==========================================
    # State Management
    # ==========================================

    def set_state(self, state: CoordinatorState) -> None:
        """Set coordinator state."""
        with self._state_lock:
            old_state = self.state
            self.state = state
            self.registry.set_state("coordinator_state", state.value)
            logger.info(f"Coordinator state: {old_state.value} -> {state.value}")

            if state == CoordinatorState.EMERGENCY:
                for callback in self._on_emergency:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Emergency callback error: {e}")

    def get_state(self) -> CoordinatorState:
        """Get current coordinator state."""
        with self._state_lock:
            return self.state

    def pause(self) -> None:
        """Pause task spawning."""
        self.set_state(CoordinatorState.PAUSED)

    def resume(self) -> None:
        """Resume task spawning."""
        self.set_state(CoordinatorState.RUNNING)
        self._paused_due_to_resources = False

    def start_resource_recovery_loop(self) -> None:
        """Start the resource recovery monitoring loop.

        This loop periodically checks resource usage and auto-resumes
        if the coordinator was paused due to resource pressure and
        resources have recovered below the threshold.
        """
        if self._resource_recovery_task and self._resource_recovery_task.is_alive():
            return  # Already running

        self._stop_recovery_loop.clear()
        self._resource_recovery_task = threading.Thread(
            target=self._resource_recovery_loop,
            daemon=True,
            name="resource-recovery-loop",
        )
        self._resource_recovery_task.start()
        logger.info("Resource recovery loop started")

    def stop_resource_recovery_loop(self) -> None:
        """Stop the resource recovery monitoring loop."""
        self._stop_recovery_loop.set()
        if self._resource_recovery_task:
            self._resource_recovery_task.join(timeout=5.0)
            self._resource_recovery_task = None

    def _resource_recovery_loop(self) -> None:
        """Background loop that checks resources and auto-resumes if recovered."""
        while not self._stop_recovery_loop.is_set():
            try:
                self._check_resource_recovery()
            except Exception as e:
                logger.error(f"Resource recovery check error: {e}")

            # Wait for next check
            self._stop_recovery_loop.wait(timeout=self._resource_recovery_interval)

    def _check_resource_recovery(self) -> None:
        """Check if resources have recovered enough to auto-resume."""
        # Only check if we're paused due to resources
        if not self._paused_due_to_resources:
            return

        if self.state != CoordinatorState.PAUSED:
            return

        try:
            import psutil
        except ImportError:
            return

        # Check current resource usage
        disk_percent = psutil.disk_usage("/").percent
        memory_percent = psutil.virtual_memory().percent

        # Calculate recovery thresholds (90% of critical thresholds)
        disk_recovery = self.limits.halt_on_disk_percent * self._resource_recovery_threshold
        memory_recovery = self.limits.halt_on_memory_percent * self._resource_recovery_threshold

        # Check if both resources are below recovery threshold
        if disk_percent < disk_recovery and memory_percent < memory_recovery:
            logger.info(
                f"Resources recovered: disk={disk_percent:.0f}% < {disk_recovery:.0f}%, "
                f"mem={memory_percent:.0f}% < {memory_recovery:.0f}%. Auto-resuming."
            )
            self.resume()

    def pause_for_resources(self) -> None:
        """Pause due to resource pressure (enables auto-recovery)."""
        self._paused_due_to_resources = True
        self.set_state(CoordinatorState.PAUSED)

        # Start recovery loop if not already running
        if not self._resource_recovery_task or not self._resource_recovery_task.is_alive():
            self.start_resource_recovery_loop()

    def emergency_stop(self) -> None:
        """Emergency stop - halt all spawning and signal shutdown."""
        self.set_state(CoordinatorState.EMERGENCY)

    # ==========================================
    # Reservation Methods (Delegated to ReservationManager)
    # Extracted Dec 30, 2025 as part of god class refactoring
    # ==========================================

    def reserve_for_gauntlet(self, node_ids: list[str]) -> list[str]:
        """Reserve workers for gauntlet evaluation."""
        return self._reservation_manager.reserve_for_gauntlet(node_ids)

    def release_from_gauntlet(self, node_ids: list[str]) -> None:
        """Release workers from gauntlet reservation."""
        self._reservation_manager.release_from_gauntlet(node_ids)

    def release_all_gauntlet(self) -> int:
        """Release all workers from gauntlet reservation."""
        return self._reservation_manager.release_all_gauntlet()

    def is_reserved_for_gauntlet(self, node_id: str) -> bool:
        """Check if a worker is reserved for gauntlet."""
        return self._reservation_manager.is_reserved_for_gauntlet(node_id)

    def get_gauntlet_reserved(self) -> set[str]:
        """Get set of all workers reserved for gauntlet."""
        return self._reservation_manager.get_gauntlet_reserved()

    def get_available_for_gauntlet(self, all_nodes: list[str], count: int = 2) -> list[str]:
        """Get available nodes that can be reserved for gauntlet."""
        return self._reservation_manager.get_available_for_gauntlet(all_nodes, count)

    def reserve_for_training(
        self,
        node_ids: list[str],
        duration_seconds: float = 7200.0,
        config_key: str = "",
    ) -> list[str]:
        """Reserve GPU nodes for training jobs."""
        return self._reservation_manager.reserve_for_training(
            node_ids, duration_seconds, config_key
        )

    def release_from_training(self, node_ids: list[str]) -> None:
        """Release nodes from training reservation."""
        self._reservation_manager.release_from_training(node_ids)

    def release_all_training(self) -> int:
        """Release all nodes from training reservation."""
        return self._reservation_manager.release_all_training()

    def is_reserved_for_training(self, node_id: str) -> bool:
        """Check if a node is reserved for training."""
        return self._reservation_manager.is_reserved_for_training(node_id)

    def get_training_reserved(self) -> set[str]:
        """Get set of all nodes reserved for training."""
        return self._reservation_manager.get_training_reserved()

    def get_available_for_training(
        self,
        all_nodes: list[str],
        gpu_nodes_only: bool = True,
        exclude_gauntlet: bool = True,
    ) -> list[str]:
        """Get available nodes that can be reserved for training."""
        return self._reservation_manager.get_available_for_training(
            all_nodes, gpu_nodes_only, exclude_gauntlet
        )

    def is_any_node_reserved(self, node_id: str) -> bool:
        """Check if a node is reserved for any purpose."""
        return self._reservation_manager.is_any_reserved(node_id)

    # ==========================================
    # Admission Control
    # ==========================================

    def can_spawn_task(
        self,
        task_type: TaskType,
        node_id: str,
        check_resources: bool = True,
        check_backpressure: bool = True,
        check_health: bool = True,
    ) -> tuple:
        """
        Check if a task can be spawned.

        Args:
            task_type: Type of task to spawn
            node_id: Node where task will run
            check_resources: Whether to check CPU/memory/disk resources
            check_backpressure: Whether to check queue backpressure (December 2025)
            check_health: Whether to check node health via SSH (December 2025)

        Returns: (allowed: bool, reason: str)
        """
        # Check coordinator state
        state = self.get_state()
        if state != CoordinatorState.RUNNING:
            return (False, f"Coordinator {state.value}")

        # Check rate limit
        if (task_type in (TaskType.SELFPLAY, TaskType.GPU_SELFPLAY, TaskType.HYBRID_SELFPLAY)
                and not self._selfplay_limiter.acquire()):
            return (False, "Selfplay rate limit exceeded")

        if not self._spawn_limiter.acquire():
            return (False, "Global rate limit exceeded")

        # Check cooldown
        last = self._last_spawn.get(node_id, 0)
        if time.time() - last < self.limits.spawn_cooldown_seconds:
            return (False, "Spawn cooldown active")

        # Check node reservations for selfplay tasks (December 2025)
        # Training and gauntlet get priority over selfplay
        if task_type in (TaskType.SELFPLAY, TaskType.GPU_SELFPLAY, TaskType.HYBRID_SELFPLAY):
            if self.is_reserved_for_training(node_id):
                return (False, f"Node {node_id} reserved for training")
            if self.is_reserved_for_gauntlet(node_id):
                return (False, f"Node {node_id} reserved for gauntlet")

        # Check node health (December 2025)
        # Uses cached SSH connectivity checks to avoid spawning on unreachable hosts
        if check_health:
            denied, reason = self._check_node_health(node_id)
            if denied:
                return (False, reason)

        # Check per-node limits
        denied, reason = self._check_node_limits(task_type, node_id)
        if denied:
            return (False, reason)

        # Check cluster-wide limits
        denied, reason = self._check_cluster_limits(task_type)
        if denied:
            return (False, reason)

        # Check resources
        if check_resources:
            denied, reason = self._check_resources(node_id)
            if denied:
                return (False, reason)

        # Check queue backpressure (December 2025)
        if check_backpressure:
            denied, reason = self._check_backpressure(task_type)
            if denied:
                return (False, reason)

        return (True, "OK")

    def _check_node_limits(self, task_type: TaskType, node_id: str) -> tuple:
        """Check per-node limits."""
        if task_type in (TaskType.SELFPLAY, TaskType.GPU_SELFPLAY, TaskType.HYBRID_SELFPLAY):
            count = self.registry.count_by_node(node_id, TaskType.SELFPLAY)
            count += self.registry.count_by_node(node_id, TaskType.GPU_SELFPLAY)
            count += self.registry.count_by_node(node_id, TaskType.HYBRID_SELFPLAY)
            if count >= self.limits.max_selfplay_per_node:
                return (True, f"Node selfplay limit ({self.limits.max_selfplay_per_node})")

        elif task_type == TaskType.TRAINING:
            count = self.registry.count_by_node(node_id, TaskType.TRAINING)
            if count >= self.limits.max_training_per_node:
                return (True, f"Node training limit ({self.limits.max_training_per_node})")

        elif task_type == TaskType.SYNC:
            count = self.registry.count_by_node(node_id, TaskType.SYNC)
            if count >= self.limits.max_sync_per_node:
                return (True, f"Node sync limit ({self.limits.max_sync_per_node})")

        elif task_type == TaskType.EXPORT:
            count = self.registry.count_by_node(node_id, TaskType.EXPORT)
            if count >= self.limits.max_export_per_node:
                return (True, f"Node export limit ({self.limits.max_export_per_node})")

        return (False, "")

    def _check_cluster_limits(self, task_type: TaskType) -> tuple:
        """Check cluster-wide limits."""
        if task_type in (TaskType.SELFPLAY, TaskType.GPU_SELFPLAY, TaskType.HYBRID_SELFPLAY):
            count = self.registry.count_by_type(TaskType.SELFPLAY)
            count += self.registry.count_by_type(TaskType.GPU_SELFPLAY)
            count += self.registry.count_by_type(TaskType.HYBRID_SELFPLAY)
            if count >= self.limits.max_total_selfplay:
                self._fire_limit_reached("selfplay", count, self.limits.max_total_selfplay)
                return (True, f"Cluster selfplay limit ({self.limits.max_total_selfplay})")

        elif task_type == TaskType.TRAINING:
            count = self.registry.count_by_type(TaskType.TRAINING)
            if count >= self.limits.max_total_training:
                return (True, f"Cluster training limit ({self.limits.max_total_training})")

        elif task_type == TaskType.CMAES:
            count = self.registry.count_by_type(TaskType.CMAES)
            if count >= self.limits.max_total_cmaes:
                return (True, f"Cluster CMA-ES limit ({self.limits.max_total_cmaes})")

        elif task_type == TaskType.TOURNAMENT:
            count = self.registry.count_by_type(TaskType.TOURNAMENT)
            if count >= self.limits.max_total_tournaments:
                return (True, f"Cluster tournament limit ({self.limits.max_total_tournaments})")

        elif task_type == TaskType.PIPELINE:
            count = self.registry.count_by_type(TaskType.PIPELINE)
            if count >= self.limits.max_total_pipelines:
                return (True, f"Cluster pipeline limit ({self.limits.max_total_pipelines})")

        elif task_type == TaskType.IMPROVEMENT_LOOP:
            count = self.registry.count_by_type(TaskType.IMPROVEMENT_LOOP)
            if count >= self.limits.max_total_improvement_loops:
                return (True, f"Cluster improvement loop limit ({self.limits.max_total_improvement_loops})")

        return (False, "")

    def _check_node_health(self, node_id: str) -> tuple:
        """Check if node is healthy via SSH connectivity check (December 2025).

        Uses cached health checks to avoid overloading hosts with SSH probes.
        Results are cached for 60s (healthy) or 30s (unhealthy).

        Args:
            node_id: Node identifier to check

        Returns: (denied: bool, reason: str)
        """
        if not HAS_HOST_HEALTH:
            return (False, "")  # No health checking available, allow

        # Skip health check for localhost
        if node_id in ("localhost", "local", socket.gethostname()):
            return (False, "")

        try:
            # Use pre_spawn_check which includes load checking
            can_spawn, reason = pre_spawn_check(
                host=node_id,
                check_load=True,
                max_load_per_cpu=0.8,
            )

            if not can_spawn:
                logger.debug(f"[TaskCoordinator] Node {node_id} health check failed: {reason}")
                return (True, f"Node unhealthy: {reason}")

            return (False, "")

        except Exception as e:
            # Fail open on health check errors to avoid blocking all spawns
            logger.warning(f"[TaskCoordinator] Health check error for {node_id}: {e}")
            return (False, "")

    def _check_resources(self, node_id: str) -> tuple:
        """Check if resources allow spawning."""
        resources = self._resource_cache.get(node_id, {})

        if resources.get("disk_percent", 0) >= self.limits.halt_on_disk_percent:
            return (True, f"Disk usage critical ({resources['disk_percent']:.0f}%)")

        if resources.get("memory_percent", 0) >= self.limits.halt_on_memory_percent:
            return (True, f"Memory usage critical ({resources['memory_percent']:.0f}%)")

        if resources.get("cpu_percent", 0) >= self.limits.halt_on_cpu_percent:
            return (True, f"CPU usage critical ({resources['cpu_percent']:.0f}%)")

        return (False, "")

    def _check_backpressure(self, task_type: TaskType) -> tuple:
        """Check if queue backpressure should block spawning (December 2025).

        Prevents spawning tasks when their downstream queues are overloaded.

        Returns: (denied: bool, reason: str)
        """
        if not HAS_QUEUE_MONITOR:
            return (False, "")

        queue_type = get_queue_for_task(task_type)
        if queue_type is None:
            return (False, "")  # No queue to check

        try:
            monitor = get_queue_monitor()
            level = monitor.check_backpressure(queue_type)

            if level == BackpressureLevel.STOP:
                return (True, f"Queue {queue_type.value} at STOP backpressure")
            elif level == BackpressureLevel.HARD and hash(time.time()) % 10 != 0:
                # For hard backpressure, deny 90% of spawns
                return (True, f"Queue {queue_type.value} at HARD backpressure")
            elif level == BackpressureLevel.SOFT and hash(time.time()) % 2 != 0:
                # For soft backpressure, deny 50% of spawns
                return (True, f"Queue {queue_type.value} at SOFT backpressure")

            return (False, "")
        except Exception as e:
            logger.debug(f"Backpressure check failed: {e}")
            return (False, "")  # Fail open if check fails

    def get_queue_backpressure(self, task_type: TaskType) -> str | None:
        """Get current backpressure level for a task's queue.

        Args:
            task_type: Type of task to check

        Returns:
            Backpressure level string ("none", "soft", "hard", "stop"),
            or None if no queue is associated
        """
        if not HAS_QUEUE_MONITOR:
            return None

        queue_type = get_queue_for_task(task_type)
        if queue_type is None:
            return None

        try:
            monitor = get_queue_monitor()
            level = monitor.check_backpressure(queue_type)
            return level.value
        except (ImportError, AttributeError, RuntimeError):
            return None

    def get_throttle_factor_for_task(self, task_type: TaskType) -> float:
        """Get throttle factor for a task based on queue backpressure.

        Returns a value between 0.0 (stop) and 1.0 (full speed).
        """
        if not HAS_QUEUE_MONITOR:
            return 1.0

        queue_type = get_queue_for_task(task_type)
        if queue_type is None:
            return 1.0

        try:
            return get_throttle_factor(queue_type)
        except (ImportError, AttributeError, RuntimeError):
            return 1.0

    def _fire_limit_reached(self, limit_name: str, current: int, max_val: int) -> None:
        """Fire callbacks when limit is reached."""
        for callback in self._on_limit_reached:
            try:
                callback(limit_name, current, max_val)
            except Exception as e:
                logger.error(f"Limit reached callback error: {e}")

    # ==========================================
    # Task Registration
    # ==========================================

    def register_task(
        self,
        task_id: str,
        task_type: TaskType,
        node_id: str,
        pid: int = 0,
        metadata: dict | None = None
    ) -> None:
        """Register a newly spawned task."""
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            node_id=node_id,
            started_at=time.time(),
            pid=pid,
            status="running",
            metadata=metadata or {}
        )

        self.registry.register_task(task)
        self._last_spawn[node_id] = time.time()
        self.registry.log_spawn_attempt(task_type, node_id, True, "registered")

        logger.debug(f"Registered task {task_id} ({task_type.value}) on {node_id}")

        # Emit TASK_SPAWNED event (December 2025)
        self._emit_task_spawned(task)

    def unregister_task(self, task_id: str) -> None:
        """Unregister a completed/stopped task."""
        self.registry.unregister_task(task_id)
        logger.debug(f"Unregistered task {task_id}")

    def heartbeat_task(self, task_id: str) -> None:
        """Update heartbeat timestamp for a task.

        December 30, 2025: Added to fix orphan detection bug. The event handler
        _on_task_heartbeat() calls this method to update the last_heartbeat
        timestamp, preventing healthy tasks from being marked as orphaned.

        Args:
            task_id: The task ID to update heartbeat for
        """
        self.registry.update_heartbeat(task_id)
        logger.debug(f"Updated heartbeat for task {task_id}")

    def complete_task(
        self,
        task_id: str,
        success: bool = True,
        result_data: dict[str, Any] | None = None,
    ) -> None:
        """Complete a task and emit appropriate StageEvent.

        This is the preferred method for task completion as it:
        1. Updates task status
        2. Emits completion events for downstream consumers
        3. Cleans up task registration

        Args:
            task_id: The task ID to complete
            success: Whether the task succeeded
            result_data: Optional result data (games_generated, model_path, etc.)
        """
        # Get task info before unregistering
        task = self.registry.get_task(task_id)
        if not task:
            logger.warning(f"Completing unknown task: {task_id}")
            return

        # Update status
        status = "completed" if success else "failed"
        self.registry.update_task_status(task_id, status)

        # Emit task completion event (December 2025)
        self._emit_task_event(task, success, result_data or {})

        # Emit TASK_COMPLETED or TASK_FAILED via data_events (December 2025)
        self._emit_task_completed_or_failed(task, success, result_data or {})

        # Unregister
        self.registry.unregister_task(task_id)
        logger.debug(f"Completed task {task_id} ({task.task_type.value}): {status}")

    def fail_task(
        self,
        task_id: str,
        error: str | None = None,
    ) -> None:
        """Fail a task and emit appropriate StageEvent.

        Args:
            task_id: The task ID to fail
            error: Optional error message
        """
        self.complete_task(
            task_id,
            success=False,
            result_data={"error": error} if error else None,
        )

    def _emit_task_event(
        self,
        task: TaskInfo,
        success: bool,
        result_data: dict[str, Any],
    ) -> None:
        """Emit StageEvent for task completion via centralized emitters.

        Uses event_emitters.py which handles mapping TaskType to StageEvent:
        - selfplay → SELFPLAY_COMPLETE
        - training → TRAINING_COMPLETE/TRAINING_FAILED
        - evaluation → EVALUATION_COMPLETE
        - sync → SYNC_COMPLETE
        """
        import asyncio

        duration_seconds = time.time() - task.started_at

        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(
                safe_emit_event_async(
                    "TASK_COMPLETE",
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "success": success,
                        "node_id": task.node_id,
                        "duration_seconds": duration_seconds,
                        "result_data": result_data,
                    },
                    context="task_coordinator",
                )
            )
        except RuntimeError:
            # No event loop - run synchronously
            asyncio.run(
                safe_emit_event_async(
                    "TASK_COMPLETE",
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type.value,
                        "success": success,
                        "node_id": task.node_id,
                        "duration_seconds": duration_seconds,
                        "result_data": result_data,
                    },
                    context="task_coordinator",
                )
            )

        logger.debug(f"Emitted TASK_COMPLETE for {task.task_id}")

    def _emit_task_spawned(self, task: TaskInfo) -> None:
        """Emit TASK_SPAWNED event via data_events (December 2025).

        This provides visibility into task lifecycle for monitoring
        and coordination systems like TaskLifecycleCoordinator.
        """
        try:
            import asyncio

            from app.coordination.event_router import emit_task_spawned

            try:
                asyncio.get_running_loop()
                asyncio.create_task(emit_task_spawned(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    node_id=task.node_id,
                    source="task_coordinator",
                ))
            except RuntimeError:
                asyncio.run(emit_task_spawned(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    node_id=task.node_id,
                    source="task_coordinator",
                ))

            logger.debug(f"Emitted TASK_SPAWNED for {task.task_id}")

        except ImportError:
            logger.debug("emit_task_spawned not available")
        except Exception as e:
            logger.debug(f"Failed to emit TASK_SPAWNED: {e}")

    def _emit_task_completed_or_failed(
        self,
        task: TaskInfo,
        success: bool,
        result_data: dict[str, Any],
    ) -> None:
        """Emit TASK_COMPLETED or TASK_FAILED event (December 2025)."""
        try:
            import asyncio

            from app.coordination.event_router import emit_task_completed, emit_task_failed

            duration = time.time() - task.started_at

            if success:
                emit_fn = emit_task_completed
                event_name = "TASK_COMPLETED"
            else:
                emit_fn = emit_task_failed
                event_name = "TASK_FAILED"

            try:
                asyncio.get_running_loop()
                asyncio.create_task(emit_fn(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    node_id=task.node_id,
                    duration_seconds=duration,
                    source="task_coordinator",
                ))
            except RuntimeError:
                asyncio.run(emit_fn(
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    node_id=task.node_id,
                    duration_seconds=duration,
                    source="task_coordinator",
                ))

            logger.debug(f"Emitted {event_name} for {task.task_id}")

        except ImportError:
            logger.debug("Task lifecycle events not available")
        except Exception as e:
            logger.debug(f"Failed to emit task lifecycle event: {e}")

    def update_task_status(self, task_id: str, status: str) -> None:
        """Update task status."""
        self.registry.update_task_status(task_id, status)

    # ==========================================
    # Resource Updates
    # ==========================================

    def update_node_resources(
        self,
        node_id: str,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float
    ) -> None:
        """Update cached resource information for a node."""
        self._resource_cache[node_id] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "updated_at": time.time()
        }
        self._resource_cache_time = time.time()

        # Auto-pause on critical resource usage (with auto-recovery)
        if (disk_percent >= self.limits.halt_on_disk_percent or
            memory_percent >= self.limits.halt_on_memory_percent) and self.state == CoordinatorState.RUNNING:
            logger.warning(
                f"Critical resources on {node_id}: "
                f"disk={disk_percent:.0f}%, mem={memory_percent:.0f}%. "
                f"Auto-pausing with recovery monitoring."
            )
            self.pause_for_resources()

    # ==========================================
    # Callbacks
    # ==========================================

    def on_limit_reached(self, callback: Callable[[str, int, int], None]) -> None:
        """Register callback invoked when task spawn count reaches per-type limit.

        Used for monitoring, adaptive throttling, and alerting when the cluster
        is at capacity for a specific task type.

        Args:
            callback: Function with signature (task_type: str, current: int, max_val: int)
                - task_type: The type of task that hit its limit (e.g., "selfplay", "training")
                - current: Current number of active tasks of this type
                - max_val: Maximum allowed tasks of this type

        Example:
            def on_limit(task_type, current, max_val):
                logger.warning(f"{task_type} at capacity: {current}/{max_val}")
            coordinator.on_limit_reached(on_limit)
        """
        self._on_limit_reached.append(callback)

    def on_emergency(self, callback: Callable[[str], None]) -> None:
        """Register callback invoked during emergency stop conditions.

        Called when the coordinator detects critical issues requiring immediate
        intervention, such as runaway process counts or resource exhaustion.

        Args:
            callback: Function with signature (reason: str) describing the emergency
        """
        self._on_emergency.append(callback)

    # ==========================================
    # Statistics
    # ==========================================

    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        tasks = self.registry.get_all_running_tasks()

        by_type: dict[str, int] = {}
        by_node: dict[str, int] = {}

        for task in tasks:
            by_type[task.task_type.value] = by_type.get(task.task_type.value, 0) + 1
            by_node[task.node_id] = by_node.get(task.node_id, 0) + 1

        stats = {
            "state": self.state.value,
            "total_tasks": len(tasks),
            "by_type": by_type,
            "by_node": by_node,
            "spawns_last_minute": self.registry.get_spawn_count(1),
            "limits": asdict(self.limits),
            "rate_limiter_tokens": self._spawn_limiter.tokens_available(),
        }

        # Add backpressure info (December 2025)
        if HAS_QUEUE_MONITOR:
            try:
                from app.coordination.queue_monitor import get_queue_stats
                stats["queue_backpressure"] = get_queue_stats()
            except (ImportError, AttributeError):
                pass

        return stats

    def get_tasks(
        self,
        task_type: TaskType | None = None,
        node_id: str | None = None
    ) -> list[TaskInfo]:
        """Get tasks with optional filtering."""
        if task_type:
            return self.registry.get_tasks_by_type(task_type)
        elif node_id:
            return self.registry.get_tasks_by_node(node_id)
        else:
            return self.registry.get_all_running_tasks()

    # ==========================================
    # Cleanup
    # ==========================================

    def cleanup_stale_tasks(self) -> int:
        """Remove stale tasks from registry."""
        count = self.registry.cleanup_stale_tasks()
        if count > 0:
            logger.info(f"Cleaned up {count} stale tasks")
        return count

    def verify_tasks(self) -> dict[str, Any]:
        """Verify registered tasks are still running."""
        tasks = self.registry.get_all_running_tasks()
        verified = 0
        removed = 0

        for task in tasks:
            if task.pid > 0:
                try:
                    os.kill(task.pid, 0)  # Check if alive
                    verified += 1
                except OSError:
                    # Process dead
                    self.registry.unregister_task(task.task_id)
                    removed += 1
            else:
                # No PID, assume still valid
                verified += 1

        return {"verified": verified, "removed": removed}

    def health_check(self) -> "HealthCheckResult":
        """Check task coordinator health status.

        December 2025: Added for daemon health monitoring coverage (Phase 13).

        Returns:
            HealthCheckResult indicating coordinator health
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        stats = self.get_stats()

        # Map CoordinatorState to HealthCheckResult status
        state_to_status = {
            CoordinatorState.RUNNING: CoordinatorStatus.RUNNING,
            CoordinatorState.PAUSED: CoordinatorStatus.PAUSED,
            CoordinatorState.DRAINING: CoordinatorStatus.STOPPING,
            CoordinatorState.EMERGENCY: CoordinatorStatus.ERROR,
            CoordinatorState.STOPPED: CoordinatorStatus.STOPPED,
        }

        # Check state
        if self.state == CoordinatorState.EMERGENCY:
            return HealthCheckResult.unhealthy(
                "Emergency stop active",
                state=self.state.value,
                total_tasks=stats["total_tasks"],
            )

        if self.state == CoordinatorState.STOPPED:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Coordinator stopped",
                details={"state": self.state.value},
            )

        # Check if paused due to resources
        if self._paused_due_to_resources:
            return HealthCheckResult(
                healthy=True,  # Paused but operational
                status=CoordinatorStatus.PAUSED,
                message="Paused for resource recovery",
                details={
                    "state": self.state.value,
                    "paused_due_to_resources": True,
                    "total_tasks": stats["total_tasks"],
                },
            )

        # Check for critical resource exhaustion
        for node_id, resources in self._resource_cache.items():
            if resources.get("disk_percent", 0) >= self.limits.halt_on_disk_percent:
                return HealthCheckResult.degraded(
                    f"Critical disk usage on {node_id}: {resources['disk_percent']:.0f}%",
                    critical_node=node_id,
                    disk_percent=resources["disk_percent"],
                )
            if resources.get("memory_percent", 0) >= self.limits.halt_on_memory_percent:
                return HealthCheckResult.degraded(
                    f"Critical memory usage on {node_id}: {resources['memory_percent']:.0f}%",
                    critical_node=node_id,
                    memory_percent=resources["memory_percent"],
                )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=state_to_status.get(self.state, CoordinatorStatus.RUNNING),
            message=f"Managing {stats['total_tasks']} tasks across {len(stats['by_node'])} nodes",
            details={
                "state": self.state.value,
                "total_tasks": stats["total_tasks"],
                "tasks_by_type": stats["by_type"],
                "tasks_by_node": stats["by_node"],
                "spawns_last_minute": stats["spawns_last_minute"],
                "rate_limiter_tokens": stats["rate_limiter_tokens"],
                "gauntlet_reserved": len(self.get_gauntlet_reserved()),
                "training_reserved": len(self.get_training_reserved()),
            },
        )


# ============================================
# Context Manager for Coordinated Tasks
# ============================================

class CoordinatedTask:
    """
    Context manager for coordinated task execution.

    Usage:
        async with CoordinatedTask(TaskType.TRAINING, "node-1") as task:
            if task.allowed:
                # Run the task
                await run_training()
            else:
                logger.warning(f"Task denied: {task.reason}")
    """

    def __init__(
        self,
        task_type: TaskType,
        node_id: str,
        task_id: str | None = None,
        pid: int = 0
    ):
        self.task_type = task_type
        self.node_id = node_id
        self.task_id = task_id or f"{task_type.value}_{node_id}_{int(time.time())}"
        self.pid = pid
        self.allowed = False
        self.reason = ""
        self._coordinator = TaskCoordinator.get_instance()

    async def __aenter__(self):
        self.allowed, self.reason = self._coordinator.can_spawn_task(
            self.task_type,
            self.node_id
        )

        if self.allowed:
            self._coordinator.register_task(
                self.task_id,
                self.task_type,
                self.node_id,
                self.pid
            )
        else:
            self._coordinator.registry.log_spawn_attempt(
                self.task_type,
                self.node_id,
                False,
                self.reason
            )

        return self

    async def __aexit__(self, *args):
        if self.allowed:
            self._coordinator.unregister_task(self.task_id)


# ============================================
# Utility Functions
# ============================================

def get_coordinator() -> TaskCoordinator:
    """Get the global task coordinator instance."""
    return TaskCoordinator.get_instance()


def can_spawn(task_type: TaskType, node_id: str) -> tuple:
    """Quick check if spawning is allowed.

    Returns:
        tuple: (allowed: bool, reason: str) - whether spawning is allowed and why not if blocked
    """
    return get_coordinator().can_spawn_task(task_type, node_id)


def emergency_stop_all() -> None:
    """Emergency stop all task spawning."""
    get_coordinator().emergency_stop()


# ============================================
# Main
# ============================================

def main() -> None:
    """Test the task coordinator."""
    import uuid

    # Get coordinator
    coordinator = TaskCoordinator.get_instance()

    # Register limit callback
    def on_limit(name, current, max_val):
        print(f"LIMIT REACHED: {name} = {current}/{max_val}")

    coordinator.on_limit_reached(on_limit)

    # Test spawning
    for i in range(10):
        task_id = f"test_{uuid.uuid4().hex[:8]}"
        allowed, reason = coordinator.can_spawn_task(TaskType.SELFPLAY, "node-1")
        print(f"Spawn {i}: allowed={allowed}, reason={reason}")

        if allowed:
            coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1")

    # Print stats
    stats = coordinator.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    # Cleanup
    verified = coordinator.verify_tasks()
    print(f"\nVerified: {verified}")


if __name__ == "__main__":
    main()


def wire_task_coordinator_events() -> TaskCoordinator:
    """Wire task coordinator to the event bus for automatic updates.

    Subscribes to:
    - TASK_SPAWNED: Track new tasks
    - TASK_HEARTBEAT: Update task heartbeats
    - TASK_COMPLETED: Mark tasks complete
    - TASK_FAILED: Handle task failures
    - TASK_CANCELLED: Handle task cancellation

    Returns:
        The configured TaskCoordinator instance
    """
    coordinator = get_coordinator()

    try:
        # Use unified event router (consolidated from data_events)
        from app.coordination.event_router import get_router
        from app.coordination.event_router import DataEventType  # Types still needed

        router = get_router()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_task_spawned(event: Any) -> None:
            """Handle task spawn event."""
            payload = _event_payload(event)
            task_id = payload.get("task_id")
            task_type_str = payload.get("task_type")
            node_id = payload.get("node_id") or payload.get("host")
            pid = payload.get("pid", 0)
            if task_id and task_type_str and node_id:
                try:
                    task_type = TaskType(task_type_str)
                    coordinator.register_task(task_id, task_type, node_id, pid)
                except ValueError:
                    logger.warning(f"Unknown task type: {task_type_str}")

        def _on_task_heartbeat(event: Any) -> None:
            """Handle task heartbeat."""
            payload = _event_payload(event)
            task_id = payload.get("task_id")
            if task_id:
                coordinator.heartbeat_task(task_id)

        def _on_task_completed(event: Any) -> None:
            """Handle task completion."""
            payload = _event_payload(event)
            task_id = payload.get("task_id")
            if task_id:
                coordinator.unregister_task(task_id)

        def _on_task_failed(event: Any) -> None:
            """Handle task failure."""
            payload = _event_payload(event)
            task_id = payload.get("task_id")
            error = payload.get("error", "Unknown error")
            if task_id:
                coordinator.fail_task(task_id, error)

        def _on_task_cancelled(event: Any) -> None:
            """Handle task cancellation."""
            payload = _event_payload(event)
            task_id = payload.get("task_id")
            if task_id:
                coordinator.unregister_task(task_id)

        router.subscribe(DataEventType.TASK_SPAWNED.value, _on_task_spawned)
        router.subscribe(DataEventType.TASK_HEARTBEAT.value, _on_task_heartbeat)
        router.subscribe(DataEventType.TASK_COMPLETED.value, _on_task_completed)
        router.subscribe(DataEventType.TASK_FAILED.value, _on_task_failed)
        router.subscribe(DataEventType.TASK_CANCELLED.value, _on_task_cancelled)

        logger.info("[TaskCoordinator] Wired to event router (TASK_SPAWNED, TASK_HEARTBEAT, TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED)")

    except ImportError:
        logger.warning("[TaskCoordinator] data_events not available, running without event bus")

    return coordinator


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "CoordinatedTask",
    "CoordinatorState",
    "OrchestratorLock",
    "RateLimiter",
    "ResourceType",
    "TaskCoordinator",
    "TaskHeartbeatMonitor",
    "TaskInfo",
    # Classes
    "TaskLimits",
    "TaskRegistry",
    # Enums
    "TaskType",
    "can_spawn",
    "emergency_stop_all",
    # Convenience functions
    "get_coordinator",
    "get_queue_for_task",
    # Functions
    "get_task_resource_type",
    "is_cpu_task",
    "is_gpu_task",
    # Event wiring
    "wire_task_coordinator_events",
]
