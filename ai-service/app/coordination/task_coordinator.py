"""
Global Task Coordinator for RingRift AI.

Prevents uncoordinated task spawning across multiple orchestrators by providing:
1. Global task registry with hard limits
2. Mutual exclusion between orchestrators via file locks
3. Rate limiting and backpressure mechanisms
4. Emergency shutdown capability
5. Resource-aware admission control

This module MUST be used by all orchestrators to prevent runaway task spawning.

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

import asyncio
import fcntl
import json
import logging
import os
import signal
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable

logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

class TaskType(Enum):
    """Types of tasks that can be spawned."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"
    HYBRID_SELFPLAY = "hybrid_selfplay"
    TRAINING = "training"
    CMAES = "cmaes"
    TOURNAMENT = "tournament"
    EVALUATION = "evaluation"
    SYNC = "sync"
    EXPORT = "export"
    PIPELINE = "pipeline"
    IMPROVEMENT_LOOP = "improvement_loop"
    BACKGROUND_LOOP = "background_loop"


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

    # Resource thresholds (halt spawning above these)
    halt_on_disk_percent: float = 90.0
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
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoordinatorState(Enum):
    """State of the task coordinator."""
    RUNNING = "running"
    PAUSED = "paused"       # Temporarily paused - no new tasks
    DRAINING = "draining"   # Stopping - let existing tasks finish
    EMERGENCY = "emergency" # Emergency stop - kill all tasks
    STOPPED = "stopped"


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
        self._fd: Optional[int] = None
        self._holder_pid: Optional[int] = None

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
                try:
                    os.close(self._fd)
                except:
                    pass
                self._fd = None
            return False

    def release(self) -> None:
        """Release the orchestrator lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except:
                pass
            self._fd = None
            self._holder_pid = None

    def get_holder(self) -> Optional[int]:
        """Get PID of current lock holder, if any."""
        try:
            if self.lock_file.exists():
                content = self.lock_file.read_text().strip()
                if content:
                    return int(content)
        except:
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
        self._lock = threading.Lock()

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

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get a task by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_task(row)

    def get_tasks_by_node(self, node_id: str) -> List[TaskInfo]:
        """Get all tasks for a node."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE node_id = ? AND status = 'running'",
            (node_id,)
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_tasks_by_type(self, task_type: TaskType) -> List[TaskInfo]:
        """Get all tasks of a type."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE task_type = ? AND status = 'running'",
            (task_type.value,)
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    def get_all_running_tasks(self) -> List[TaskInfo]:
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

    def count_by_node(self, node_id: str, task_type: Optional[TaskType] = None) -> int:
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

    def get_state(self, key: str) -> Optional[str]:
        """Get coordinator state."""
        cursor = self.conn.execute(
            "SELECT value FROM coordinator_state WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _row_to_task(self, row) -> TaskInfo:
        """Convert database row to TaskInfo."""
        return TaskInfo(
            task_id=row['task_id'],
            task_type=TaskType(row['task_type']),
            node_id=row['node_id'],
            started_at=row['started_at'],
            pid=row['pid'],
            status=row['status'],
            metadata=json.loads(row['metadata_json'] or '{}')
        )


# ============================================
# Task Coordinator
# ============================================

class TaskCoordinator:
    """
    Global task coordinator singleton.

    Provides centralized control over task spawning across all orchestrators.
    """

    _instance: Optional['TaskCoordinator'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'TaskCoordinator':
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance._shutdown()
            cls._instance = None

    def __init__(self):
        # Configuration
        self.limits = TaskLimits()

        # State
        self.state = CoordinatorState.RUNNING
        self._state_lock = threading.Lock()

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
        self._on_limit_reached: List[Callable] = []
        self._on_emergency: List[Callable] = []

        # Last spawn time per node (for cooldown)
        self._last_spawn: Dict[str, float] = {}

        # Resource cache (refreshed periodically)
        self._resource_cache: Dict[str, Dict[str, float]] = {}
        self._resource_cache_time: float = 0

        logger.info("Task coordinator initialized")

    def _shutdown(self) -> None:
        """Cleanup on shutdown."""
        pass

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

    def emergency_stop(self) -> None:
        """Emergency stop - halt all spawning and signal shutdown."""
        self.set_state(CoordinatorState.EMERGENCY)

    # ==========================================
    # Admission Control
    # ==========================================

    def can_spawn_task(
        self,
        task_type: TaskType,
        node_id: str,
        check_resources: bool = True
    ) -> tuple:
        """
        Check if a task can be spawned.

        Returns: (allowed: bool, reason: str)
        """
        # Check coordinator state
        state = self.get_state()
        if state != CoordinatorState.RUNNING:
            return (False, f"Coordinator {state.value}")

        # Check rate limit
        if task_type in (TaskType.SELFPLAY, TaskType.GPU_SELFPLAY, TaskType.HYBRID_SELFPLAY):
            if not self._selfplay_limiter.acquire():
                return (False, "Selfplay rate limit exceeded")

        if not self._spawn_limiter.acquire():
            return (False, "Global rate limit exceeded")

        # Check cooldown
        last = self._last_spawn.get(node_id, 0)
        if time.time() - last < self.limits.spawn_cooldown_seconds:
            return (False, "Spawn cooldown active")

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
        metadata: Optional[Dict] = None
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

    def unregister_task(self, task_id: str) -> None:
        """Unregister a completed/stopped task."""
        self.registry.unregister_task(task_id)
        logger.debug(f"Unregistered task {task_id}")

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

        # Auto-pause on critical resource usage
        if (disk_percent >= self.limits.halt_on_disk_percent or
            memory_percent >= self.limits.halt_on_memory_percent):
            if self.state == CoordinatorState.RUNNING:
                logger.warning(
                    f"Critical resources on {node_id}: "
                    f"disk={disk_percent:.0f}%, mem={memory_percent:.0f}%"
                )

    # ==========================================
    # Callbacks
    # ==========================================

    def on_limit_reached(self, callback: Callable) -> None:
        """Register callback for when limits are reached."""
        self._on_limit_reached.append(callback)

    def on_emergency(self, callback: Callable) -> None:
        """Register callback for emergency stop."""
        self._on_emergency.append(callback)

    # ==========================================
    # Statistics
    # ==========================================

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        tasks = self.registry.get_all_running_tasks()

        by_type: Dict[str, int] = {}
        by_node: Dict[str, int] = {}

        for task in tasks:
            by_type[task.task_type.value] = by_type.get(task.task_type.value, 0) + 1
            by_node[task.node_id] = by_node.get(task.node_id, 0) + 1

        return {
            "state": self.state.value,
            "total_tasks": len(tasks),
            "by_type": by_type,
            "by_node": by_node,
            "spawns_last_minute": self.registry.get_spawn_count(1),
            "limits": asdict(self.limits),
            "rate_limiter_tokens": self._spawn_limiter.tokens_available()
        }

    def get_tasks(
        self,
        task_type: Optional[TaskType] = None,
        node_id: Optional[str] = None
    ) -> List[TaskInfo]:
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

    def verify_tasks(self) -> Dict[str, Any]:
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
        task_id: Optional[str] = None,
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


def can_spawn(task_type: TaskType, node_id: str) -> bool:
    """Quick check if spawning is allowed."""
    allowed, _ = get_coordinator().can_spawn_task(task_type, node_id)
    return allowed


def emergency_stop_all() -> None:
    """Emergency stop all task spawning."""
    get_coordinator().emergency_stop()


# ============================================
# Main
# ============================================

def main():
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
