#!/usr/bin/env python3
"""
DEPRECATED: This module is deprecated. Use app.coordination.task_coordinator instead.

The task_coordinator provides superior functionality using SQLite for coordination
instead of JSON files, with proper rate limiting and cluster-wide task management.

Migration:
    # OLD (deprecated)
    from app.coordination.cluster_lock import (
        acquire_orchestrator_lock,
        can_spawn_task,
        register_task,
    )

    # NEW (canonical)
    from app.coordination.task_coordinator import (
        TaskCoordinator,
        TaskType,
        atomic_write_json,  # Utility functions are preserved here
        safe_read_json,
    )

    coordinator = TaskCoordinator.get_instance()
    if coordinator.can_spawn_task(TaskType.SELFPLAY, node_id="node-1"):
        coordinator.register_task(task_id, TaskType.SELFPLAY, node_id="node-1")

This file is kept for backwards compatibility but will be removed in a future version.

---
Original docstring:

Unified cluster-wide coordination and resource management.
"""
import warnings

warnings.warn(
    "cluster_lock.py is deprecated. Use app.coordination.task_coordinator instead. "
    "The task_coordinator provides SQLite-based coordination with proper rate limiting.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export utilities from task_coordinator for backwards compatibility
from app.coordination.task_coordinator import atomic_write_json, safe_read_json

import fcntl
import json
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================
# Atomic File Operations
# ============================================

def atomic_write_json(filepath: Path, data: Any, indent: int = 2) -> None:
    """
    Write JSON data to file atomically.

    Uses write-to-temp-then-rename pattern to prevent corruption
    if process crashes during write.

    Args:
        filepath: Target file path
        data: Data to serialize as JSON
        indent: JSON indentation (default 2)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.name}.",
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # Ensure data hits disk

        # Atomic rename (POSIX guarantees atomicity)
        os.rename(temp_path, filepath)
    except:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise


def safe_read_json(filepath: Path, default: Any = None) -> Any:
    """
    Safely read JSON file with fallback on corruption.

    Args:
        filepath: File to read
        default: Value to return if file doesn't exist or is corrupt

    Returns:
        Parsed JSON data or default value
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return default

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        # Attempt to read backup if it exists
        backup = filepath.with_suffix(filepath.suffix + '.bak')
        if backup.exists():
            try:
                with open(backup, 'r') as f:
                    print(f"Recovered from backup: {backup}")
                    return json.load(f)
            except:
                pass
        return default


# Coordination files
COORDINATION_DIR = Path("/tmp/ringrift_coordination")
ORCHESTRATOR_LOCK = COORDINATION_DIR / "orchestrator.lock"
PROCESS_REGISTRY = COORDINATION_DIR / "process_registry.json"
CLUSTER_STATE = COORDINATION_DIR / "cluster_state.json"

# Resource limits
MAX_LOAD_THRESHOLD = 50  # Don't spawn if load > this
MAX_PROCESSES_PER_HOST = {
    "lambda_h100": 50,
    "lambda_2xh100": 80,
    "lambda_a10": 40,
    "gh200": 60,  # Default for GH200 hosts
    "aws": 20,
    "default": 30,
}

# Process types and their resource weights
TASK_WEIGHTS = {
    "selfplay": 1.0,
    "training": 5.0,
    "tournament": 2.0,
    "hp_tuning": 3.0,
    "export": 1.5,
}


@dataclass
class TaskInfo:
    """Information about a running task."""
    host: str
    task_type: str
    pid: int
    started_at: str
    command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        return cls(**data)


@dataclass
class HostStatus:
    """Status of a host."""
    name: str
    load_1m: float = 0.0
    load_5m: float = 0.0
    load_15m: float = 0.0
    cpu_count: int = 1
    active_tasks: int = 0
    last_check: str = ""
    healthy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HostStatus":
        return cls(**data)


def ensure_coordination_dir():
    """Ensure coordination directory exists."""
    COORDINATION_DIR.mkdir(parents=True, exist_ok=True)


def acquire_orchestrator_lock(orchestrator_name: str, timeout: int = 5) -> bool:
    """Acquire exclusive orchestrator lock.

    Only one orchestrator should run at a time across the entire cluster.
    Returns True if lock acquired, False if another orchestrator is running.
    """
    ensure_coordination_dir()

    try:
        lock_fd = os.open(str(ORCHESTRATOR_LOCK), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            os.close(lock_fd)
            # Check who holds the lock
            try:
                with open(ORCHESTRATOR_LOCK, 'r') as f:
                    lock_info = json.load(f)
                    print(f"Lock held by: {lock_info.get('orchestrator', 'unknown')} "
                          f"(pid: {lock_info.get('pid', 'unknown')}) "
                          f"since {lock_info.get('acquired_at', 'unknown')}")
            except:
                pass
            return False

        # Write lock info
        lock_info = {
            "orchestrator": orchestrator_name,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": datetime.now().isoformat(),
        }
        os.write(lock_fd, json.dumps(lock_info).encode())
        os.fsync(lock_fd)

        # Store fd for later release
        acquire_orchestrator_lock._lock_fd = lock_fd
        return True

    except Exception as e:
        print(f"Error acquiring orchestrator lock: {e}")
        return False


def release_orchestrator_lock():
    """Release the orchestrator lock."""
    try:
        if hasattr(acquire_orchestrator_lock, '_lock_fd'):
            fd = acquire_orchestrator_lock._lock_fd
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            del acquire_orchestrator_lock._lock_fd
    except Exception as e:
        print(f"Error releasing lock: {e}")


def get_host_load(host: str, ssh_user: str = "ubuntu") -> Optional[HostStatus]:
    """Get current load for a host via SSH."""
    try:
        if host in ["localhost", "local", socket.gethostname()]:
            # Local host
            load = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return HostStatus(
                name=host,
                load_1m=load[0],
                load_5m=load[1],
                load_15m=load[2],
                cpu_count=cpu_count,
                last_check=datetime.now().isoformat(),
                healthy=True,
            )

        # Remote host
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             f"{ssh_user}@{host}",
             "cat /proc/loadavg && nproc"],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            return HostStatus(name=host, healthy=False, last_check=datetime.now().isoformat())

        lines = result.stdout.strip().split('\n')
        load_parts = lines[0].split()
        cpu_count = int(lines[1]) if len(lines) > 1 else 1

        return HostStatus(
            name=host,
            load_1m=float(load_parts[0]),
            load_5m=float(load_parts[1]),
            load_15m=float(load_parts[2]),
            cpu_count=cpu_count,
            last_check=datetime.now().isoformat(),
            healthy=True,
        )
    except Exception as e:
        return HostStatus(name=host, healthy=False, last_check=datetime.now().isoformat())


def can_spawn_task(host: str, task_type: str = "selfplay", ssh_user: str = "ubuntu") -> bool:
    """Check if it's safe to spawn a new task on the host.

    Returns True if:
    1. Host load is below threshold
    2. Active tasks are below limit
    3. Host is healthy
    """
    status = get_host_load(host, ssh_user)
    if not status or not status.healthy:
        print(f"[GATE] Host {host} is unhealthy, blocking spawn")
        return False

    # Check load (normalized by CPU count)
    normalized_load = status.load_1m / max(status.cpu_count, 1) * 100
    if normalized_load > MAX_LOAD_THRESHOLD:
        print(f"[GATE] Host {host} load too high: {status.load_1m:.1f} "
              f"({normalized_load:.1f}% normalized), blocking spawn")
        return False

    # Check active task count
    registry = load_process_registry()
    host_tasks = [t for t in registry if t.host == host]

    # Get max for this host type
    max_tasks = MAX_PROCESSES_PER_HOST.get("default", 30)
    for prefix, limit in MAX_PROCESSES_PER_HOST.items():
        if prefix in host.lower():
            max_tasks = limit
            break

    if len(host_tasks) >= max_tasks:
        print(f"[GATE] Host {host} at task limit: {len(host_tasks)}/{max_tasks}")
        return False

    print(f"[GATE] Host {host} OK: load={status.load_1m:.1f}, tasks={len(host_tasks)}/{max_tasks}")
    return True


def load_process_registry() -> List[TaskInfo]:
    """Load the process registry from disk safely."""
    ensure_coordination_dir()
    try:
        data = safe_read_json(PROCESS_REGISTRY, default=[])
        return [TaskInfo.from_dict(t) for t in data]
    except Exception as e:
        print(f"Error loading registry: {e}")
    return []


def save_process_registry(tasks: List[TaskInfo]):
    """Save the process registry to disk atomically.

    Uses atomic write (temp file + rename) to prevent corruption
    if process crashes during write.
    """
    ensure_coordination_dir()
    atomic_write_json(PROCESS_REGISTRY, [t.to_dict() for t in tasks])


def register_task(host: str, task_type: str, pid: int, command: str = "") -> bool:
    """Register a new task in the process registry."""
    try:
        registry = load_process_registry()
        registry.append(TaskInfo(
            host=host,
            task_type=task_type,
            pid=pid,
            started_at=datetime.now().isoformat(),
            command=command,
        ))
        save_process_registry(registry)
        return True
    except Exception as e:
        print(f"Error registering task: {e}")
        return False


def unregister_task(host: str, pid: int) -> bool:
    """Remove a task from the registry."""
    try:
        registry = load_process_registry()
        registry = [t for t in registry if not (t.host == host and t.pid == pid)]
        save_process_registry(registry)
        return True
    except Exception as e:
        print(f"Error unregistering task: {e}")
        return False


def cleanup_stale_tasks(ssh_user: str = "ubuntu"):
    """Remove tasks from registry that are no longer running."""
    registry = load_process_registry()
    active_tasks = []

    for task in registry:
        # Check if process is still running
        try:
            if task.host in ["localhost", "local", socket.gethostname()]:
                # Local check
                os.kill(task.pid, 0)  # Doesn't kill, just checks
                active_tasks.append(task)
            else:
                # Remote check
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", f"{ssh_user}@{task.host}",
                     f"kill -0 {task.pid} 2>/dev/null && echo running"],
                    capture_output=True, text=True, timeout=10
                )
                if "running" in result.stdout:
                    active_tasks.append(task)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            print(f"[CLEANUP] Removing stale task: {task.task_type} on {task.host} (pid {task.pid})")

    save_process_registry(active_tasks)
    return len(registry) - len(active_tasks)


def get_cluster_status() -> Dict[str, Any]:
    """Get full cluster status including all hosts and tasks."""
    registry = load_process_registry()

    # Group tasks by host
    tasks_by_host = {}
    for task in registry:
        if task.host not in tasks_by_host:
            tasks_by_host[task.host] = []
        tasks_by_host[task.host].append(task.to_dict())

    # Check lock status
    lock_info = None
    try:
        if ORCHESTRATOR_LOCK.exists():
            with open(ORCHESTRATOR_LOCK, 'r') as f:
                lock_info = json.load(f)
    except:
        pass

    return {
        "timestamp": datetime.now().isoformat(),
        "orchestrator_lock": lock_info,
        "total_tasks": len(registry),
        "tasks_by_host": tasks_by_host,
        "tasks_by_type": {},  # TODO: aggregate
    }


def emergency_cluster_halt():
    """Emergency function to stop all orchestrators and signal cleanup."""
    ensure_coordination_dir()
    halt_file = COORDINATION_DIR / "EMERGENCY_HALT"
    halt_file.write_text(json.dumps({
        "halted_at": datetime.now().isoformat(),
        "halted_by": socket.gethostname(),
        "reason": "emergency_cluster_halt called",
    }))
    print("[EMERGENCY] Cluster halt signal written. All orchestrators should stop.")


def check_emergency_halt() -> bool:
    """Check if emergency halt has been signaled."""
    halt_file = COORDINATION_DIR / "EMERGENCY_HALT"
    return halt_file.exists()


def clear_emergency_halt():
    """Clear the emergency halt signal."""
    halt_file = COORDINATION_DIR / "EMERGENCY_HALT"
    if halt_file.exists():
        halt_file.unlink()
        print("[EMERGENCY] Halt signal cleared.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster coordination utilities")
    parser.add_argument("--status", action="store_true", help="Show cluster status")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup stale tasks")
    parser.add_argument("--halt", action="store_true", help="Signal emergency halt")
    parser.add_argument("--clear-halt", action="store_true", help="Clear emergency halt")
    args = parser.parse_args()

    if args.status:
        status = get_cluster_status()
        print(json.dumps(status, indent=2))
    elif args.cleanup:
        removed = cleanup_stale_tasks()
        print(f"Removed {removed} stale tasks")
    elif args.halt:
        emergency_cluster_halt()
    elif args.clear_halt:
        clear_emergency_halt()
    else:
        parser.print_help()
