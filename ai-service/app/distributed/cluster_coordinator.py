#!/usr/bin/env python3
"""Cluster Coordination Module - Prevents runaway processes and task collisions.

DEPRECATION NOTICE:
    This module is being superseded by app.coordination modules:

    - For orchestrator mutual exclusion:
        Use app.coordination.orchestrator_registry
        (OrchestratorRole, acquire_orchestrator_role, release_orchestrator_role)

    - For task spawning limits:
        Use app.coordination.task_coordinator
        (TaskCoordinator, TaskType, can_spawn_task)

    Migration example:
        # OLD:
        from app.distributed.cluster_coordinator import ClusterCoordinator, TaskRole
        coordinator = ClusterCoordinator()
        if coordinator.is_role_held(TaskRole.ORCHESTRATOR):
            ...

        # NEW:
        from app.coordination import (
            OrchestratorRole,
            acquire_orchestrator_role,
            release_orchestrator_role,
        )
        if acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR):
            # Role acquired
            ...
            release_orchestrator_role()

This module provides centralized coordination for all improvement orchestrators:
1. Global task locking - Only one orchestrator can run per role per host
2. Process limits - Enforces maximum concurrent processes per host
3. Task registry - Tracks active tasks across the cluster
4. Collision prevention - Rejects conflicting task requests

Usage (deprecated):
    from app.distributed.cluster_coordinator import ClusterCoordinator, TaskRole

    coordinator = ClusterCoordinator()

    # Acquire a role (blocks if already held)
    with coordinator.acquire_role(TaskRole.SELFPLAY) as lock:
        # Run selfplay tasks
        pass

    # Check if we can spawn more processes
    if coordinator.can_spawn_process("selfplay"):
        # Spawn process
        coordinator.register_process(pid, "selfplay")
"""
from __future__ import annotations

import warnings

warnings.warn(
    "app.distributed.cluster_coordinator is deprecated. "
    "Use app.coordination.orchestrator_registry and app.coordination.task_coordinator instead.",
    DeprecationWarning,
    stacklevel=2,
)

import fcntl
import json
import os
import signal
import socket
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

# Use centralized path constants
from app.utils.paths import DATA_DIR

COORDINATION_DIR = DATA_DIR / "coordination"
LOCK_DIR = COORDINATION_DIR / "locks"
REGISTRY_DB = COORDINATION_DIR / "task_registry.db"


class TaskRole(Enum):
    """Mutually exclusive task roles - only one instance per role per host."""
    ORCHESTRATOR = "orchestrator"  # Main improvement controller
    SELFPLAY = "selfplay"  # Selfplay generation
    TRAINING = "training"  # Model training
    TOURNAMENT = "tournament"  # ELO tournaments
    HP_TUNING = "hp_tuning"  # Hyperparameter tuning
    DATA_AGGREGATION = "data_aggregation"  # Data collection


@dataclass
class ProcessLimits:
    """Process limits per host to prevent resource exhaustion.

    Note: Memory and CPU limits are inherited from app.utils.resource_guard
    to ensure consistent 80% utilization limits across the codebase.
    """
    max_python_processes: int = 50  # Maximum Python processes
    max_selfplay_workers: int = 16  # Maximum selfplay worker processes
    max_training_workers: int = 4  # Maximum concurrent training jobs
    max_tournament_workers: int = 8  # Maximum tournament workers
    # Use consistent 80% limits from resource_guard (previously was 90% for CPU)
    max_memory_percent: float = 80.0  # Max memory usage before throttling
    max_cpu_percent: float = 80.0  # Max CPU usage before throttling (fixed from 90%)


@dataclass
class TaskInfo:
    """Information about a registered task."""
    task_id: str
    role: str
    host: str
    pid: int
    started_at: str
    description: str = ""
    parent_task: str | None = None


class ClusterCoordinator:
    """Coordinates tasks across the cluster to prevent collisions and runaway processes."""

    def __init__(self, limits: ProcessLimits | None = None):
        self.limits = limits or ProcessLimits()
        self.hostname = socket.gethostname()
        self._ensure_dirs()
        self._init_registry_db()
        self._active_locks: dict[str, int] = {}  # role -> lock fd

    def _ensure_dirs(self):
        """Create coordination directories if needed."""
        COORDINATION_DIR.mkdir(parents=True, exist_ok=True)
        LOCK_DIR.mkdir(parents=True, exist_ok=True)

    def _init_registry_db(self):
        """Initialize the task registry database."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                host TEXT NOT NULL,
                pid INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                description TEXT DEFAULT '',
                parent_task TEXT,
                last_heartbeat TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_registry (
                pid INTEGER,
                host TEXT,
                category TEXT NOT NULL,
                started_at TEXT NOT NULL,
                parent_pid INTEGER,
                PRIMARY KEY (pid, host)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_role ON tasks(role)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_host ON tasks(host)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procs_host ON process_registry(host)")
        conn.commit()
        conn.close()

    def _lock_file_path(self, role: TaskRole) -> Path:
        """Get the lock file path for a role."""
        return LOCK_DIR / f"{role.value}.{self.hostname}.lock"

    def is_role_held(self, role: TaskRole) -> bool:
        """Check if a role is currently held on this host."""
        lock_path = self._lock_file_path(role)
        if not lock_path.exists():
            return False
        try:
            fd = os.open(str(lock_path), os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False  # Lock is free
            except OSError:
                return True  # Lock is held
            finally:
                os.close(fd)
        except FileNotFoundError:
            return False

    def get_role_holder_pid(self, role: TaskRole) -> int | None:
        """Get the PID of the process holding a role lock."""
        lock_path = self._lock_file_path(role)
        if lock_path.exists():
            try:
                content = lock_path.read_text().strip()
                if content:
                    return int(content)
            except (ValueError, FileNotFoundError):
                pass
        return None

    @contextmanager
    def acquire_role(self, role: TaskRole, blocking: bool = True, description: str = ""):
        """Acquire exclusive access to a task role.

        Args:
            role: The task role to acquire
            blocking: If True, wait for the lock. If False, raise if unavailable.
            description: Description of what this task is doing

        Raises:
            RuntimeError: If role is already held and blocking=False
        """
        lock_path = self._lock_file_path(role)
        lock_path.touch(exist_ok=True)

        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            # Try to acquire lock
            if blocking:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    existing_pid = self.get_role_holder_pid(role)
                    raise RuntimeError(
                        f"Role {role.value} is already held by PID {existing_pid} on {self.hostname}"
                    )

            # Write our PID to the lock file
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, str(os.getpid()).encode())

            # Register task
            task_id = f"{role.value}_{self.hostname}_{os.getpid()}"
            self._register_task(TaskInfo(
                task_id=task_id,
                role=role.value,
                host=self.hostname,
                pid=os.getpid(),
                started_at=datetime.now(timezone.utc).isoformat(),
                description=description
            ))

            self._active_locks[role.value] = fd

            try:
                yield task_id
            finally:
                # Cleanup
                self._unregister_task(task_id)
                if role.value in self._active_locks:
                    del self._active_locks[role.value]
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

        except (OSError, IOError):
            os.close(fd)
            raise

    def _register_task(self, task: TaskInfo):
        """Register a task in the database."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, role, host, pid, started_at, description, parent_task, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.role, task.host, task.pid,
            task.started_at, task.description, task.parent_task,
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        conn.close()

    def _unregister_task(self, task_id: str):
        """Unregister a task from the database."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        conn.commit()
        conn.close()

    def heartbeat(self, task_id: str):
        """Update heartbeat for a task."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tasks SET last_heartbeat = ? WHERE task_id = ?",
            (datetime.now(timezone.utc).isoformat(), task_id)
        )
        conn.commit()
        conn.close()

    def get_active_tasks(self, role: TaskRole | None = None, host: str | None = None) -> list[TaskInfo]:
        """Get all active tasks, optionally filtered by role or host."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()

        query = "SELECT task_id, role, host, pid, started_at, description, parent_task FROM tasks WHERE 1=1"
        params = []

        if role:
            query += " AND role = ?"
            params.append(role.value)
        if host:
            query += " AND host = ?"
            params.append(host)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            TaskInfo(
                task_id=r[0], role=r[1], host=r[2], pid=r[3],
                started_at=r[4], description=r[5], parent_task=r[6]
            )
            for r in rows
        ]

    def count_processes(self, category: str | None = None) -> int:
        """Count Python processes on this host, optionally by category."""
        if category:
            conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM process_registry WHERE host = ? AND category = ?",
                (self.hostname, category)
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        else:
            # Count actual Python processes
            count = 0
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return count

    def can_spawn_process(self, category: str) -> bool:
        """Check if we can spawn another process in a category."""
        current = self.count_processes(category)

        limits_map = {
            "selfplay": self.limits.max_selfplay_workers,
            "training": self.limits.max_training_workers,
            "tournament": self.limits.max_tournament_workers,
        }

        max_allowed = limits_map.get(category, self.limits.max_python_processes)

        if current >= max_allowed:
            return False

        # Also check overall system resources
        if psutil.virtual_memory().percent > self.limits.max_memory_percent:
            return False
        return not psutil.cpu_percent(interval=0.1) > self.limits.max_cpu_percent

    def register_process(self, pid: int, category: str, parent_pid: int | None = None):
        """Register a spawned process."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO process_registry (pid, host, category, started_at, parent_pid)
            VALUES (?, ?, ?, ?, ?)
        """, (pid, self.hostname, category, datetime.now(timezone.utc).isoformat(), parent_pid))
        conn.commit()
        conn.close()

    def unregister_process(self, pid: int):
        """Unregister a process."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM process_registry WHERE pid = ? AND host = ?",
            (pid, self.hostname)
        )
        conn.commit()
        conn.close()

    def cleanup_stale_locks(self) -> list[str]:
        """Clean up lock files held by dead processes.

        Returns list of cleaned up lock files.
        """
        cleaned = []
        for lock_file in LOCK_DIR.glob(f"*.{self.hostname}.lock"):
            try:
                content = lock_file.read_text().strip()
                if content:
                    pid = int(content)
                    if not psutil.pid_exists(pid):
                        # Process is dead, remove the lock
                        lock_file.unlink()
                        cleaned.append(str(lock_file))
                        print(f"[ClusterCoordinator] Cleaned stale lock: {lock_file.name} (dead PID {pid})")
            except (ValueError, FileNotFoundError, PermissionError):
                # Invalid PID or file issues - try to clean anyway
                try:
                    lock_file.unlink()
                    cleaned.append(str(lock_file))
                except (OSError, PermissionError):
                    pass
        return cleaned

    def cleanup_stale_entries(self, max_age_hours: float = 24.0):
        """Clean up stale task and process entries."""
        # First clean up stale lock files
        self.cleanup_stale_locks()

        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()

        # Clean up processes that no longer exist
        cursor.execute("SELECT pid FROM process_registry WHERE host = ?", (self.hostname,))
        for (pid,) in cursor.fetchall():
            if not psutil.pid_exists(pid):
                cursor.execute(
                    "DELETE FROM process_registry WHERE pid = ? AND host = ?",
                    (pid, self.hostname)
                )

        # Clean up tasks with no heartbeat in max_age_hours
        cursor.execute(f"""
            DELETE FROM tasks WHERE last_heartbeat < datetime('now', '-{int(max_age_hours)} hours')
        """)

        conn.commit()
        conn.close()

    def kill_processes_by_category(self, category: str, signal_num: int = signal.SIGTERM):
        """Kill all processes in a category on this host."""
        conn = sqlite3.connect(str(REGISTRY_DB), timeout=30)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pid FROM process_registry WHERE host = ? AND category = ?",
            (self.hostname, category)
        )
        pids = [r[0] for r in cursor.fetchall()]
        conn.close()

        killed = []
        for pid in pids:
            try:
                os.kill(pid, signal_num)
                killed.append(pid)
            except (ProcessLookupError, PermissionError):
                pass

        # Unregister killed processes
        for pid in killed:
            self.unregister_process(pid)

        return killed

    def get_status_summary(self) -> dict[str, Any]:
        """Get a summary of coordination status."""
        tasks = self.get_active_tasks()

        # Group by role
        by_role = {}
        for task in tasks:
            if task.role not in by_role:
                by_role[task.role] = []
            by_role[task.role].append(asdict(task))

        # Get process counts
        python_count = self.count_processes()

        # Get memory/CPU
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)

        return {
            "host": self.hostname,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_tasks": len(tasks),
            "tasks_by_role": by_role,
            "python_processes": python_count,
            "memory_percent": mem.percent,
            "cpu_percent": cpu,
            "limits": asdict(self.limits),
            "can_spawn": {
                "selfplay": self.can_spawn_process("selfplay"),
                "training": self.can_spawn_process("training"),
                "tournament": self.can_spawn_process("tournament"),
            }
        }


def check_and_abort_if_role_held(role: TaskRole) -> None:
    """Utility to check if a role is held and abort if so."""
    coordinator = ClusterCoordinator()
    if coordinator.is_role_held(role):
        holder_pid = coordinator.get_role_holder_pid(role)
        print(f"ERROR: Role {role.value} is already held by PID {holder_pid}")
        print("Kill that process first or wait for it to complete.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster Coordinator CLI")
    parser.add_argument("--status", action="store_true", help="Show coordination status")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale entries")
    parser.add_argument("--kill-category", type=str, help="Kill all processes in a category")
    parser.add_argument("--check-role", type=str, help="Check if a role is held")

    args = parser.parse_args()

    coordinator = ClusterCoordinator()

    if args.status:
        status = coordinator.get_status_summary()
        print(json.dumps(status, indent=2))
    elif args.cleanup:
        coordinator.cleanup_stale_entries()
        print("Cleaned up stale entries")
    elif args.kill_category:
        killed = coordinator.kill_processes_by_category(args.kill_category)
        print(f"Killed {len(killed)} processes: {killed}")
    elif args.check_role:
        role = TaskRole(args.check_role)
        if coordinator.is_role_held(role):
            pid = coordinator.get_role_holder_pid(role)
            print(f"Role {role.value} is held by PID {pid}")
        else:
            print(f"Role {role.value} is available")
    else:
        parser.print_help()
