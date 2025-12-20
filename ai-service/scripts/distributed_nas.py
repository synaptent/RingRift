#!/usr/bin/env python3
"""Distributed Neural Architecture Search (NAS) for RingRift AI.

Distributes architecture evaluations across multiple machines for faster
search. Supports both synchronous and asynchronous evaluation modes.

Architecture:
- Coordinator: Manages NAS state, generates candidates, aggregates results
- Workers: Evaluate architectures on remote machines (CPU/GPU)
- Communication: SSH-based remote execution + shared SQLite database

Features:
- Parallel evaluation across cluster
- Fault tolerance with automatic retry
- Load balancing based on host capabilities
- Real-time progress tracking
- Resume support from checkpoint

Usage:
    # Start distributed NAS with evolutionary search
    python scripts/distributed_nas.py \
        --strategy evolutionary \
        --population 40 \
        --generations 100 \
        --workers 10

    # Resume previous run
    python scripts/distributed_nas.py --resume nas_dist_12345

    # Use Bayesian search with async evaluation
    python scripts/distributed_nas.py \
        --strategy bayesian \
        --trials 200 \
        --async-eval \
        --max-concurrent 20
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neural_architecture_search import (
    SEARCH_SPACE,
    Architecture,
    bayesian_acquisition,
    crossover_architectures,
    mutate_architecture,
    sample_architecture,
    tournament_selection,
    update_pareto_front,
)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        check_gpu_memory,
        require_resources,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    resource_can_proceed = lambda **kwargs: True  # type: ignore
    check_disk_space = lambda *args, **kwargs: True  # type: ignore
    check_memory = lambda *args, **kwargs: True  # type: ignore
    check_gpu_memory = lambda *args, **kwargs: True  # type: ignore
    require_resources = lambda *args, **kwargs: True  # type: ignore
    RESOURCE_LIMITS = None  # type: ignore

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("distributed_nas")


# =============================================================================
# Worker Configuration
# =============================================================================

@dataclass
class WorkerConfig:
    """Configuration for a distributed worker."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: str | None = None
    ringrift_path: str = "~/ringrift/ai-service"
    venv_activate: str = "source venv/bin/activate"
    has_gpu: bool = False
    gpu_name: str = ""
    memory_gb: int = 16
    cpus: int = 4
    max_concurrent_evals: int = 1  # How many evals this worker can run
    enabled: bool = True
    shared_filesystem: bool = False  # True if on shared filesystem (e.g., GH200 cluster)
    shared_work_dir: str | None = None  # Path to shared work directory


@dataclass
class EvalTask:
    """A task to evaluate an architecture on a worker."""
    task_id: str
    arch: Architecture
    worker_name: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    retries: int = 0


@dataclass
class DistributedNASState:
    """State for distributed NAS run."""
    run_id: str
    strategy: str
    board_type: str
    num_players: int
    search_space: dict[str, Any]
    population: list[Architecture]
    generation: int = 0
    total_evaluations: int = 0
    best_performance: float = 0.0
    best_architecture: dict[str, Any] | None = None
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    # Distributed-specific fields
    pending_tasks: list[EvalTask] = field(default_factory=list)
    completed_tasks: list[EvalTask] = field(default_factory=list)
    failed_tasks: list[EvalTask] = field(default_factory=list)
    worker_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    created_at: str = ""
    updated_at: str = ""


# =============================================================================
# Database for Distributed State
# =============================================================================

class NASDatabase:
    """SQLite database for distributed NAS coordination."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Architectures table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS architectures (
                arch_id TEXT PRIMARY KEY,
                params TEXT NOT NULL,
                performance REAL DEFAULT 0.0,
                flops INTEGER DEFAULT 0,
                param_count INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0,
                evaluated INTEGER DEFAULT 0,
                generation INTEGER DEFAULT 0,
                parent_ids TEXT,
                created_at TEXT,
                evaluated_at TEXT,
                worker_name TEXT
            )
        """)

        # Evaluation tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_tasks (
                task_id TEXT PRIMARY KEY,
                arch_id TEXT NOT NULL,
                worker_name TEXT,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT,
                retries INTEGER DEFAULT 0,
                FOREIGN KEY (arch_id) REFERENCES architectures(arch_id)
            )
        """)

        # NAS state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nas_state (
                run_id TEXT PRIMARY KEY,
                strategy TEXT,
                board_type TEXT,
                num_players INTEGER,
                generation INTEGER DEFAULT 0,
                total_evaluations INTEGER DEFAULT 0,
                best_performance REAL DEFAULT 0.0,
                best_architecture TEXT,
                pareto_front TEXT,
                history TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Worker stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS worker_stats (
                worker_name TEXT PRIMARY KEY,
                total_evals INTEGER DEFAULT 0,
                successful_evals INTEGER DEFAULT 0,
                failed_evals INTEGER DEFAULT 0,
                avg_eval_time REAL DEFAULT 0.0,
                last_seen TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_architecture(self, arch: Architecture, generation: int = 0):
        """Save architecture to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO architectures
            (arch_id, params, performance, flops, param_count, latency_ms,
             evaluated, generation, parent_ids, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            arch.arch_id,
            json.dumps(arch.params),
            arch.performance,
            arch.flops,
            arch.param_count,
            arch.latency_ms,
            1 if arch.evaluated else 0,
            generation,
            json.dumps(arch.parent_ids),
            arch.created_at,
        ))

        conn.commit()
        conn.close()

    def load_architecture(self, arch_id: str) -> Architecture | None:
        """Load architecture from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT arch_id, params, performance, flops, param_count,
                   latency_ms, evaluated, parent_ids, created_at
            FROM architectures
            WHERE arch_id = ?
        """, (arch_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Architecture(
            arch_id=row[0],
            params=json.loads(row[1]),
            performance=row[2],
            flops=row[3],
            param_count=row[4],
            latency_ms=row[5],
            evaluated=bool(row[6]),
            parent_ids=json.loads(row[7]) if row[7] else [],
            created_at=row[8],
        )

    def get_unevaluated_architectures(self, limit: int = 10) -> list[Architecture]:
        """Get architectures that haven't been evaluated yet."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT arch_id, params, performance, flops, param_count,
                   latency_ms, evaluated, parent_ids, created_at
            FROM architectures
            WHERE evaluated = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))

        architectures = []
        for row in cursor.fetchall():
            architectures.append(Architecture(
                arch_id=row[0],
                params=json.loads(row[1]),
                performance=row[2],
                flops=row[3],
                param_count=row[4],
                latency_ms=row[5],
                evaluated=bool(row[6]),
                parent_ids=json.loads(row[7]) if row[7] else [],
                created_at=row[8],
            ))

        conn.close()
        return architectures

    def update_evaluation_result(
        self,
        arch_id: str,
        performance: float,
        flops: int,
        param_count: int,
        latency_ms: float,
        worker_name: str,
    ):
        """Update architecture with evaluation results."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE architectures
            SET performance = ?,
                flops = ?,
                param_count = ?,
                latency_ms = ?,
                evaluated = 1,
                evaluated_at = ?,
                worker_name = ?
            WHERE arch_id = ?
        """, (
            performance,
            flops,
            param_count,
            latency_ms,
            datetime.utcnow().isoformat() + "Z",
            worker_name,
            arch_id,
        ))

        conn.commit()
        conn.close()

    def create_eval_task(self, task: EvalTask):
        """Create evaluation task in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO eval_tasks
            (task_id, arch_id, worker_name, status, started_at, completed_at,
             result, error, retries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.arch.arch_id,
            task.worker_name,
            task.status,
            datetime.fromtimestamp(task.started_at).isoformat() if task.started_at else None,
            datetime.fromtimestamp(task.completed_at).isoformat() if task.completed_at else None,
            json.dumps(task.result) if task.result else None,
            task.error,
            task.retries,
        ))

        conn.commit()
        conn.close()

    def update_eval_task(self, task: EvalTask):
        """Update evaluation task status."""
        self.create_eval_task(task)  # Uses INSERT OR REPLACE

    def get_pending_tasks(self, worker_name: str | None = None) -> list[EvalTask]:
        """Get pending evaluation tasks."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if worker_name:
            cursor.execute("""
                SELECT t.task_id, t.arch_id, t.worker_name, t.status,
                       t.started_at, t.completed_at, t.result, t.error, t.retries
                FROM eval_tasks t
                WHERE t.status = 'pending' AND t.worker_name = ?
            """, (worker_name,))
        else:
            cursor.execute("""
                SELECT t.task_id, t.arch_id, t.worker_name, t.status,
                       t.started_at, t.completed_at, t.result, t.error, t.retries
                FROM eval_tasks t
                WHERE t.status = 'pending'
            """)

        tasks = []
        for row in cursor.fetchall():
            arch = self.load_architecture(row[1])
            if arch:
                tasks.append(EvalTask(
                    task_id=row[0],
                    arch=arch,
                    worker_name=row[2] or "",
                    status=row[3],
                    started_at=datetime.fromisoformat(row[4]).timestamp() if row[4] else None,
                    completed_at=datetime.fromisoformat(row[5]).timestamp() if row[5] else None,
                    result=json.loads(row[6]) if row[6] else None,
                    error=row[7],
                    retries=row[8],
                ))

        conn.close()
        return tasks

    def update_worker_stats(
        self,
        worker_name: str,
        eval_time: float,
        success: bool,
    ):
        """Update worker statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get current stats
        cursor.execute("""
            SELECT total_evals, successful_evals, failed_evals, avg_eval_time
            FROM worker_stats
            WHERE worker_name = ?
        """, (worker_name,))

        row = cursor.fetchone()
        if row:
            total = row[0] + 1
            successful = row[1] + (1 if success else 0)
            failed = row[2] + (0 if success else 1)
            # Running average
            avg_time = (row[3] * row[0] + eval_time) / total
        else:
            total = 1
            successful = 1 if success else 0
            failed = 0 if success else 1
            avg_time = eval_time

        cursor.execute("""
            INSERT OR REPLACE INTO worker_stats
            (worker_name, total_evals, successful_evals, failed_evals,
             avg_eval_time, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            worker_name,
            total,
            successful,
            failed,
            avg_time,
            datetime.utcnow().isoformat() + "Z",
        ))

        conn.commit()
        conn.close()

    def save_nas_state(self, state: DistributedNASState):
        """Save NAS state to database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO nas_state
            (run_id, strategy, board_type, num_players, generation,
             total_evaluations, best_performance, best_architecture,
             pareto_front, history, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.run_id,
            state.strategy,
            state.board_type,
            state.num_players,
            state.generation,
            state.total_evaluations,
            state.best_performance,
            json.dumps(state.best_architecture) if state.best_architecture else None,
            json.dumps(state.pareto_front),
            json.dumps(state.history[-100:]),  # Keep last 100
            state.created_at,
            datetime.utcnow().isoformat() + "Z",
        ))

        conn.commit()
        conn.close()

    def load_nas_state(self, run_id: str) -> DistributedNASState | None:
        """Load NAS state from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT run_id, strategy, board_type, num_players, generation,
                   total_evaluations, best_performance, best_architecture,
                   pareto_front, history, created_at, updated_at
            FROM nas_state
            WHERE run_id = ?
        """, (run_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        # Load all architectures for this run
        cursor.execute("""
            SELECT arch_id, params, performance, flops, param_count,
                   latency_ms, evaluated, parent_ids, created_at
            FROM architectures
            ORDER BY created_at ASC
        """)

        population = []
        for arch_row in cursor.fetchall():
            population.append(Architecture(
                arch_id=arch_row[0],
                params=json.loads(arch_row[1]),
                performance=arch_row[2],
                flops=arch_row[3],
                param_count=arch_row[4],
                latency_ms=arch_row[5],
                evaluated=bool(arch_row[6]),
                parent_ids=json.loads(arch_row[7]) if arch_row[7] else [],
                created_at=arch_row[8],
            ))

        conn.close()

        return DistributedNASState(
            run_id=row[0],
            strategy=row[1],
            board_type=row[2],
            num_players=row[3],
            search_space=SEARCH_SPACE,
            population=population,
            generation=row[4],
            total_evaluations=row[5],
            best_performance=row[6],
            best_architecture=json.loads(row[7]) if row[7] else None,
            pareto_front=json.loads(row[8]) if row[8] else [],
            history=json.loads(row[9]) if row[9] else [],
            created_at=row[10],
            updated_at=row[11],
        )


# =============================================================================
# Shared Filesystem Task Coordination (for GH200 cluster)
# =============================================================================

class SharedFSTaskQueue:
    """File-based task queue for shared filesystem coordination.

    Uses file locking for coordination between coordinator and workers
    on a shared filesystem (NFS, Lustre, etc.).

    Directory structure:
        work_dir/
            pending/      - Tasks waiting to be picked up
            running/      - Tasks currently being processed
            completed/    - Completed task results
            failed/       - Failed task info
    """

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.pending_dir = self.work_dir / "pending"
        self.running_dir = self.work_dir / "running"
        self.completed_dir = self.work_dir / "completed"
        self.failed_dir = self.work_dir / "failed"

        # Create directories
        for d in [self.pending_dir, self.running_dir, self.completed_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def submit_task(self, task: EvalTask):
        """Submit a task to the pending queue."""
        task_file = self.pending_dir / f"{task.task_id}.json"
        with open(task_file, "w") as f:
            json.dump({
                "task_id": task.task_id,
                "arch_id": task.arch.arch_id,
                "arch_params": task.arch.params,
                "worker_name": task.worker_name,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat() + "Z",
            }, f, indent=2)

    def claim_task(self, worker_name: str) -> dict[str, Any] | None:
        """Claim a pending task for a worker.

        Uses atomic rename for file locking on shared filesystem.
        Returns task data or None if no tasks available.
        """
        for task_file in sorted(self.pending_dir.glob("*.json")):
            try:
                # Try atomic rename to running dir
                running_file = self.running_dir / task_file.name
                task_file.rename(running_file)

                # Read and update task
                with open(running_file) as f:
                    task_data = json.load(f)

                task_data["worker_name"] = worker_name
                task_data["status"] = "running"
                task_data["started_at"] = datetime.utcnow().isoformat() + "Z"

                with open(running_file, "w") as f:
                    json.dump(task_data, f, indent=2)

                return task_data

            except (FileNotFoundError, OSError):
                # Another worker claimed this task
                continue

        return None

    def complete_task(self, task_id: str, result: dict[str, Any], success: bool = True):
        """Mark a task as completed with results."""
        running_file = self.running_dir / f"{task_id}.json"
        target_dir = self.completed_dir if success else self.failed_dir
        target_file = target_dir / f"{task_id}.json"

        try:
            with open(running_file) as f:
                task_data = json.load(f)
        except FileNotFoundError:
            # Task file missing, create fresh
            task_data = {"task_id": task_id}

        task_data["status"] = "completed" if success else "failed"
        task_data["completed_at"] = datetime.utcnow().isoformat() + "Z"
        task_data["result"] = result

        with open(target_file, "w") as f:
            json.dump(task_data, f, indent=2)

        # Remove from running
        try:
            running_file.unlink()
        except FileNotFoundError:
            pass

    def get_completed_tasks(self) -> list[dict[str, Any]]:
        """Get all completed tasks."""
        results = []
        for task_file in self.completed_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    results.append(json.load(f))
            except Exception:
                continue
        return results

    def get_failed_tasks(self) -> list[dict[str, Any]]:
        """Get all failed tasks."""
        results = []
        for task_file in self.failed_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    results.append(json.load(f))
            except Exception:
                continue
        return results

    def cleanup_completed(self, older_than_hours: int = 24):
        """Remove old completed task files."""
        cutoff = datetime.utcnow().timestamp() - (older_than_hours * 3600)

        for task_file in self.completed_dir.glob("*.json"):
            if task_file.stat().st_mtime < cutoff:
                task_file.unlink()


def run_shared_fs_worker(work_dir: Path, worker_name: str, max_tasks: int = -1):
    """Run a worker process on a shared filesystem.

    This function is meant to be run as a separate process on each
    worker machine in the shared filesystem cluster.

    Args:
        work_dir: Path to shared work directory
        worker_name: Unique identifier for this worker
        max_tasks: Maximum tasks to process (-1 for unlimited)
    """
    queue = SharedFSTaskQueue(work_dir)
    tasks_completed = 0

    logger.info(f"Worker {worker_name} starting, watching {work_dir}")

    while max_tasks == -1 or tasks_completed < max_tasks:
        # Try to claim a task
        task_data = queue.claim_task(worker_name)

        if task_data is None:
            # No tasks available, wait and retry
            time.sleep(2)
            continue

        task_id = task_data["task_id"]
        arch_id = task_data["arch_id"]
        arch_params = task_data["arch_params"]

        logger.info(f"Worker {worker_name} claimed task {task_id} (arch: {arch_id})")

        try:
            # Create architecture and evaluate
            arch = Architecture(
                arch_id=arch_id,
                params=arch_params,
                created_at=task_data.get("created_at", ""),
            )

            # Evaluate locally
            start_time = time.time()
            evaluate_architecture_local(arch)
            eval_time = time.time() - start_time

            result = {
                "arch_id": arch.arch_id,
                "performance": arch.performance,
                "flops": arch.flops,
                "param_count": arch.param_count,
                "latency_ms": arch.latency_ms,
                "eval_time": eval_time,
                "worker_name": worker_name,
            }

            queue.complete_task(task_id, result, success=True)
            tasks_completed += 1

            logger.info(
                f"Worker {worker_name} completed {arch_id}: "
                f"perf={arch.performance:.4f} ({eval_time:.1f}s)"
            )

        except Exception as e:
            logger.error(f"Worker {worker_name} failed on {arch_id}: {e}")
            queue.complete_task(task_id, {"error": str(e)}, success=False)

    logger.info(f"Worker {worker_name} finished after {tasks_completed} tasks")


# =============================================================================
# Worker Management
# =============================================================================

def load_workers_config() -> list[WorkerConfig]:
    """Load worker configurations from YAML files."""
    workers = []
    config_dir = AI_SERVICE_ROOT / "config"

    # Load distributed_hosts.yaml
    distributed_file = config_dir / "distributed_hosts.yaml"
    if distributed_file.exists():
        with open(distributed_file) as f:
            data = yaml.safe_load(f) or {}

        for name, cfg in data.get("hosts", {}).items():
            if cfg.get("status") == "ssh_key_issue":
                continue

            ringrift_path = cfg.get("ringrift_path", "~/ringrift/ai-service")
            if not ringrift_path.endswith("ai-service"):
                ringrift_path = ringrift_path + "/ai-service"

            # Check if this is a shared filesystem host (e.g., GH200 cluster)
            is_shared_fs = cfg.get("shared_filesystem", False)
            # Auto-detect GH200 hosts as shared filesystem
            if "gh200" in name.lower() or "GH200" in cfg.get("gpu", ""):
                is_shared_fs = True

            workers.append(WorkerConfig(
                name=name,
                ssh_host=cfg.get("ssh_host", ""),
                ssh_user=cfg.get("ssh_user", "ubuntu"),
                ssh_port=cfg.get("ssh_port", 22),
                ssh_key=cfg.get("ssh_key"),
                ringrift_path=ringrift_path,
                venv_activate=cfg.get("venv_activate") or "source venv/bin/activate",
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu="gpu" in cfg,
                gpu_name=cfg.get("gpu", ""),
                # Allow more concurrent evals on powerful machines (more for GH200s)
                max_concurrent_evals=cfg.get("max_concurrent_evals", max(2 if is_shared_fs else 1, cfg.get("cpus", 4) // 8)),
                shared_filesystem=is_shared_fs,
                shared_work_dir=cfg.get("shared_work_dir"),
            ))

    return workers


def ssh_cmd(
    worker: WorkerConfig,
    cmd: str,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Execute SSH command on worker."""
    ssh_args = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
    ]

    if worker.ssh_port != 22:
        ssh_args.extend(["-p", str(worker.ssh_port)])

    if worker.ssh_key:
        key_path = os.path.expanduser(worker.ssh_key)
        if os.path.exists(key_path):
            ssh_args.extend(["-i", key_path])

    ssh_args.append(f"{worker.ssh_user}@{worker.ssh_host}")
    ssh_args.append(cmd)

    try:
        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "SSH timeout"
    except Exception as e:
        return 1, "", str(e)


def check_worker_available(worker: WorkerConfig) -> bool:
    """Check if worker is available and responsive."""
    code, out, err = ssh_cmd(worker, "echo ok", timeout=15)
    return code == 0 and "ok" in out


# =============================================================================
# Architecture Evaluation
# =============================================================================

def evaluate_architecture_local(arch: Architecture) -> Architecture:
    """Evaluate architecture locally (for testing or single-machine runs).

    In production, this builds and trains the model for a few epochs,
    then evaluates on a validation set.
    """
    from neural_architecture_search import evaluate_architecture
    evaluate_architecture(arch)
    return arch


async def evaluate_architecture_remote(
    worker: WorkerConfig,
    arch: Architecture,
    board_type: str = "square8",
    num_players: int = 2,
    quick_eval: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate architecture on a remote worker.

    Args:
        worker: Worker configuration
        arch: Architecture to evaluate
        board_type: Board type for evaluation
        num_players: Number of players
        quick_eval: Use quick evaluation (fewer training steps)

    Returns:
        Tuple of (success, result_dict)
    """
    # Serialize architecture to JSON
    arch_json = json.dumps({
        "arch_id": arch.arch_id,
        "params": arch.params,
    })

    # Create remote evaluation script
    eval_script = f"""
import json
import sys
sys.path.insert(0, '.')

from scripts.neural_architecture_search import (
    Architecture,
    evaluate_architecture,
    estimate_architecture_cost,
)

# Load architecture
arch_data = json.loads('''{arch_json}''')
arch = Architecture(
    arch_id=arch_data['arch_id'],
    params=arch_data['params'],
    created_at='',
)

# Evaluate
evaluate_architecture(arch, quick_eval={'true' if quick_eval else 'false'})

# Output results
print(json.dumps({{
    'arch_id': arch.arch_id,
    'performance': arch.performance,
    'flops': arch.flops,
    'param_count': arch.param_count,
    'latency_ms': arch.latency_ms,
}}))
"""

    # Write script to temp file and copy to worker
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(eval_script)
        local_script = f.name

    remote_script = f"/tmp/nas_eval_{arch.arch_id}.py"

    try:
        # Copy script to worker
        scp_args = [
            "scp",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
        ]
        if worker.ssh_port != 22:
            scp_args.extend(["-P", str(worker.ssh_port)])
        if worker.ssh_key:
            key_path = os.path.expanduser(worker.ssh_key)
            if os.path.exists(key_path):
                scp_args.extend(["-i", key_path])

        scp_args.extend([
            local_script,
            f"{worker.ssh_user}@{worker.ssh_host}:{remote_script}"
        ])

        result = subprocess.run(scp_args, capture_output=True, timeout=30)
        if result.returncode != 0:
            return False, {"error": f"Failed to copy script: {result.stderr.decode()}"}

        # Run evaluation on worker
        cmd = f"""
cd {worker.ringrift_path} && \
({worker.venv_activate}) 2>/dev/null || true && \
python3 {remote_script}
"""

        code, out, err = ssh_cmd(worker, cmd, timeout=600)  # 10 minute timeout

        if code != 0:
            return False, {"error": f"Evaluation failed: {err}"}

        # Parse results
        try:
            # Find JSON in output
            for line in out.split("\n"):
                if line.strip().startswith("{"):
                    result_data = json.loads(line.strip())
                    return True, result_data

            return False, {"error": f"No JSON output found: {out}"}

        except json.JSONDecodeError as e:
            return False, {"error": f"Failed to parse results: {e}\nOutput: {out}"}

    finally:
        # Cleanup local temp file
        try:
            os.unlink(local_script)
        except Exception:
            pass

        # Cleanup remote script
        ssh_cmd(worker, f"rm -f {remote_script}", timeout=10)


# =============================================================================
# Distributed NAS Coordinator
# =============================================================================

class DistributedNASCoordinator:
    """Coordinates distributed NAS across multiple workers."""

    def __init__(
        self,
        run_id: str,
        strategy: str = "evolutionary",
        board_type: str = "square8",
        num_players: int = 2,
        population_size: int = 20,
        max_concurrent: int = 10,
        quick_eval: bool = True,
        db_path: Path | None = None,
        shared_work_dir: Path | None = None,
    ):
        self.run_id = run_id
        self.strategy = strategy
        self.board_type = board_type
        self.num_players = num_players
        self.population_size = population_size
        self.max_concurrent = max_concurrent
        self.quick_eval = quick_eval

        # Setup database
        if db_path is None:
            db_path = AI_SERVICE_ROOT / "logs" / "nas" / run_id / "nas_distributed.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = NASDatabase(db_path)

        # Load workers
        self.workers = load_workers_config()
        self.available_workers: list[WorkerConfig] = []
        self.ssh_workers: list[WorkerConfig] = []  # Workers using SSH
        self.shared_fs_workers: list[WorkerConfig] = []  # Workers on shared filesystem

        # Shared filesystem task queue (for GH200 cluster)
        if shared_work_dir:
            self.shared_work_dir = Path(shared_work_dir)
        else:
            # Default to project-relative path
            self.shared_work_dir = AI_SERVICE_ROOT / "data" / "nas_tasks" / run_id
        self.shared_fs_queue: SharedFSTaskQueue | None = None

        # State
        self.state: DistributedNASState | None = None
        self._running = False
        self._active_tasks: dict[str, EvalTask] = {}  # task_id -> task
        self._shared_fs_task_ids: set[str] = set()  # Track tasks submitted to shared FS

    async def initialize(self, resume: bool = False):
        """Initialize coordinator state."""
        if resume:
            self.state = self.db.load_nas_state(self.run_id)
            if self.state:
                logger.info(f"Resumed NAS run {self.run_id} from generation {self.state.generation}")
            else:
                logger.warning(f"No state found for {self.run_id}, starting fresh")
                resume = False

        if not resume:
            # Create new state
            self.state = DistributedNASState(
                run_id=self.run_id,
                strategy=self.strategy,
                board_type=self.board_type,
                num_players=self.num_players,
                search_space=SEARCH_SPACE,
                population=[],
                created_at=datetime.utcnow().isoformat() + "Z",
            )

            # Initialize population
            if self.strategy == "evolutionary":
                for i in range(self.population_size):
                    arch = sample_architecture(f"gen000_arch{i:03d}")
                    self.state.population.append(arch)
                    self.db.save_architecture(arch, generation=0)

            self.db.save_nas_state(self.state)

        # Check worker availability and categorize by type
        logger.info(f"Checking {len(self.workers)} workers...")
        has_shared_fs_workers = False

        for worker in self.workers:
            if worker.shared_filesystem:
                # Shared filesystem workers - check if accessible
                # (assume available if shared filesystem is configured)
                self.available_workers.append(worker)
                self.shared_fs_workers.append(worker)
                has_shared_fs_workers = True
                logger.info(f"  {worker.name}: shared filesystem worker (GPU: {worker.has_gpu})")
            elif check_worker_available(worker):
                self.available_workers.append(worker)
                self.ssh_workers.append(worker)
                logger.info(f"  {worker.name}: SSH worker available (GPU: {worker.has_gpu})")
            else:
                logger.warning(f"  {worker.name}: unavailable")

        # Initialize shared filesystem task queue if we have shared FS workers
        if has_shared_fs_workers:
            self.shared_fs_queue = SharedFSTaskQueue(self.shared_work_dir)
            logger.info(f"Initialized shared filesystem queue at: {self.shared_work_dir}")
            logger.info(f"  Run workers on GH200s with: python scripts/distributed_nas.py --worker --work-dir {self.shared_work_dir}")

        logger.info(f"{len(self.available_workers)} workers available "
                    f"({len(self.shared_fs_workers)} shared FS, {len(self.ssh_workers)} SSH)")

    def _select_worker(self) -> WorkerConfig | None:
        """Select best available worker for next evaluation."""
        if not self.available_workers:
            return None

        # Count active tasks per worker
        worker_loads = {w.name: 0 for w in self.available_workers}
        for task in self._active_tasks.values():
            if task.worker_name in worker_loads:
                worker_loads[task.worker_name] += 1

        # Find worker with lowest load that hasn't exceeded max concurrent
        for worker in sorted(self.available_workers, key=lambda w: worker_loads[w.name]):
            if worker_loads[worker.name] < worker.max_concurrent_evals:
                return worker

        return None

    async def _run_evaluation(self, task: EvalTask):
        """Run a single evaluation task."""
        worker = next(
            (w for w in self.available_workers if w.name == task.worker_name),
            None
        )
        if not worker:
            task.status = "failed"
            task.error = "Worker not found"
            self.db.update_eval_task(task)
            return

        task.status = "running"
        task.started_at = time.time()
        self.db.update_eval_task(task)

        try:
            success, result = await evaluate_architecture_remote(
                worker,
                task.arch,
                self.board_type,
                self.num_players,
                self.quick_eval,
            )

            if success:
                task.status = "completed"
                task.result = result
                task.completed_at = time.time()

                # Update architecture in state
                task.arch.performance = result.get("performance", 0.0)
                task.arch.flops = result.get("flops", 0)
                task.arch.param_count = result.get("param_count", 0)
                task.arch.latency_ms = result.get("latency_ms", 0.0)
                task.arch.evaluated = True

                # Update database
                self.db.update_evaluation_result(
                    task.arch.arch_id,
                    task.arch.performance,
                    task.arch.flops,
                    task.arch.param_count,
                    task.arch.latency_ms,
                    worker.name,
                )

                eval_time = task.completed_at - task.started_at
                self.db.update_worker_stats(worker.name, eval_time, success=True)

                logger.info(
                    f"Evaluated {task.arch.arch_id} on {worker.name}: "
                    f"perf={task.arch.performance:.4f} "
                    f"({eval_time:.1f}s)"
                )
            else:
                task.status = "failed"
                task.error = result.get("error", "Unknown error")
                task.retries += 1

                self.db.update_worker_stats(worker.name, 0, success=False)
                logger.warning(f"Evaluation failed for {task.arch.arch_id}: {task.error}")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.retries += 1
            logger.error(f"Exception evaluating {task.arch.arch_id}: {e}")

        self.db.update_eval_task(task)

    async def _submit_shared_fs_tasks(self, architectures: list[Architecture]):
        """Submit tasks to shared filesystem queue for GH200 workers."""
        if not self.shared_fs_queue or not architectures:
            return

        for arch in architectures:
            task = EvalTask(
                task_id=f"eval_{arch.arch_id}_{int(time.time() * 1000)}",
                arch=arch,
                worker_name="shared_fs",  # Will be assigned by worker
            )
            self.shared_fs_queue.submit_task(task)
            self._shared_fs_task_ids.add(task.task_id)
            self.db.create_eval_task(task)

        logger.info(f"Submitted {len(architectures)} tasks to shared filesystem queue")

    async def _collect_shared_fs_results(self) -> int:
        """Collect completed tasks from shared filesystem queue.

        Returns number of results collected.
        """
        if not self.shared_fs_queue:
            return 0

        collected = 0
        completed_tasks = self.shared_fs_queue.get_completed_tasks()

        for task_data in completed_tasks:
            task_id = task_data.get("task_id")
            if task_id not in self._shared_fs_task_ids:
                continue  # Not our task

            result = task_data.get("result", {})
            arch_id = result.get("arch_id") or task_data.get("arch_id")

            if not arch_id:
                continue

            # Update architecture in population
            for arch in self.state.population:
                if arch.arch_id == arch_id:
                    arch.performance = result.get("performance", 0.0)
                    arch.flops = result.get("flops", 0)
                    arch.param_count = result.get("param_count", 0)
                    arch.latency_ms = result.get("latency_ms", 0.0)
                    arch.evaluated = True

                    # Update database
                    self.db.update_evaluation_result(
                        arch_id,
                        arch.performance,
                        arch.flops,
                        arch.param_count,
                        arch.latency_ms,
                        result.get("worker_name", "shared_fs"),
                    )

                    collected += 1
                    self._shared_fs_task_ids.discard(task_id)

                    logger.info(
                        f"Collected result for {arch_id}: "
                        f"perf={arch.performance:.4f}"
                    )
                    break

        # Also check failed tasks
        failed_tasks = self.shared_fs_queue.get_failed_tasks()
        for task_data in failed_tasks:
            task_id = task_data.get("task_id")
            if task_id in self._shared_fs_task_ids:
                logger.warning(f"Task {task_id} failed: {task_data.get('result', {}).get('error', 'unknown')}")
                self._shared_fs_task_ids.discard(task_id)

        return collected

    async def _evolutionary_step(self):
        """Run one generation of distributed evolutionary NAS."""
        # Ensure all architectures in current generation are evaluated
        unevaluated = [a for a in self.state.population if not a.evaluated]

        if unevaluated:
            # Separate tasks for shared FS vs SSH workers
            shared_fs_archs = []
            ssh_tasks = []

            for arch in unevaluated:
                # Prefer shared filesystem for batch submission (more efficient)
                if self.shared_fs_queue and self.shared_fs_workers:
                    shared_fs_archs.append(arch)
                else:
                    # Use SSH-based evaluation
                    worker = self._select_worker()
                    if not worker:
                        break

                    task = EvalTask(
                        task_id=f"eval_{arch.arch_id}_{int(time.time())}",
                        arch=arch,
                        worker_name=worker.name,
                    )
                    self.db.create_eval_task(task)
                    self._active_tasks[task.task_id] = task
                    ssh_tasks.append(self._run_evaluation(task))

            # Submit shared FS tasks (non-blocking, workers will pick them up)
            if shared_fs_archs:
                await self._submit_shared_fs_tasks(shared_fs_archs)

            # Run SSH evaluations concurrently
            if ssh_tasks:
                await asyncio.gather(*ssh_tasks)

            # Wait for shared FS results (poll until all complete or timeout)
            if shared_fs_archs:
                max_wait = 600  # 10 minute timeout
                start_wait = time.time()
                while self._shared_fs_task_ids and (time.time() - start_wait) < max_wait:
                    collected = await self._collect_shared_fs_results()
                    if collected == 0:
                        await asyncio.sleep(5)  # Poll every 5 seconds

            # Cleanup completed SSH tasks
            completed_ids = [
                tid for tid, t in self._active_tasks.items()
                if t.status in ("completed", "failed")
            ]
            for tid in completed_ids:
                task = self._active_tasks.pop(tid)
                if task.status == "completed":
                    self.state.completed_tasks.append(task)
                else:
                    self.state.failed_tasks.append(task)

        # Update population from database
        for arch in self.state.population:
            db_arch = self.db.load_architecture(arch.arch_id)
            if db_arch and db_arch.evaluated:
                arch.performance = db_arch.performance
                arch.flops = db_arch.flops
                arch.param_count = db_arch.param_count
                arch.latency_ms = db_arch.latency_ms
                arch.evaluated = True

        # Check if all evaluated
        evaluated_count = sum(1 for a in self.state.population if a.evaluated)
        if evaluated_count < len(self.state.population):
            logger.info(f"Waiting for evaluations: {evaluated_count}/{len(self.state.population)}")
            return

        # Sort by performance
        self.state.population.sort(key=lambda a: a.performance, reverse=True)

        # Update best
        if self.state.population[0].performance > self.state.best_performance:
            self.state.best_performance = self.state.population[0].performance
            self.state.best_architecture = copy.deepcopy(self.state.population[0].params)
            logger.info(f"New best: perf={self.state.best_performance:.4f}")

        # Generate next generation
        gen_id = self.state.generation + 1
        elite_count = max(1, self.population_size // 10)
        new_population = self.state.population[:elite_count]

        arch_counter = 0
        while len(new_population) < self.population_size:
            arch_id = f"gen{gen_id:03d}_arch{arch_counter:03d}"

            if random.random() < 0.5 and len(self.state.population) >= 2:
                # Crossover
                p1 = tournament_selection(self.state.population)
                p2 = tournament_selection(self.state.population)
                child = crossover_architectures(p1, p2, arch_id)
                if random.random() < 0.3:
                    child = mutate_architecture(child, arch_id, 0.5)
            else:
                # Mutation
                parent = tournament_selection(self.state.population)
                child = mutate_architecture(parent, arch_id, 0.3)

            new_population.append(child)
            self.db.save_architecture(child, generation=gen_id)
            arch_counter += 1

        self.state.population = new_population
        self.state.generation = gen_id
        self.state.total_evaluations += evaluated_count
        self.state.pareto_front = update_pareto_front(self.state.population)

        # Record history
        self.state.history.append({
            "generation": self.state.generation,
            "best_performance": self.state.best_performance,
            "avg_performance": np.mean([a.performance for a in self.state.population if a.evaluated]),
            "pareto_size": len(self.state.pareto_front),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        self.db.save_nas_state(self.state)
        logger.info(f"Completed generation {gen_id}, best={self.state.best_performance:.4f}")

    async def _bayesian_step(self):
        """Run one step of distributed Bayesian NAS."""
        # Get all evaluated architectures
        evaluated = [a for a in self.state.population if a.evaluated]

        # Generate candidates using acquisition function
        if len(evaluated) < 10:
            arch = sample_architecture(f"bayes_{self.state.generation:03d}")
        else:
            arch = bayesian_acquisition(evaluated, SEARCH_SPACE)
            arch.arch_id = f"bayes_{self.state.generation:03d}"

        self.state.population.append(arch)
        self.db.save_architecture(arch, generation=self.state.generation)

        # Evaluate
        worker = self._select_worker()
        if not worker:
            logger.warning("No workers available")
            return

        task = EvalTask(
            task_id=f"eval_{arch.arch_id}_{int(time.time())}",
            arch=arch,
            worker_name=worker.name,
        )
        self.db.create_eval_task(task)
        self._active_tasks[task.task_id] = task

        await self._run_evaluation(task)

        # Update state
        if task.status == "completed":
            self.state.total_evaluations += 1

            if arch.performance > self.state.best_performance:
                self.state.best_performance = arch.performance
                self.state.best_architecture = copy.deepcopy(arch.params)
                logger.info(f"New best: perf={self.state.best_performance:.4f}")

        self.state.generation += 1
        self.state.pareto_front = update_pareto_front(self.state.population)

        self.state.history.append({
            "generation": self.state.generation,
            "best_performance": self.state.best_performance,
            "total_evaluations": self.state.total_evaluations,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        self.db.save_nas_state(self.state)

        # Cleanup
        del self._active_tasks[task.task_id]

    async def run(self, max_iterations: int):
        """Run distributed NAS for specified iterations."""
        self._running = True

        for i in range(max_iterations):
            if not self._running:
                break

            logger.info(f"=== Iteration {i + 1}/{max_iterations} ===")

            if self.strategy == "evolutionary":
                await self._evolutionary_step()
            elif self.strategy == "bayesian":
                await self._bayesian_step()
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # Print progress
            if (i + 1) % 5 == 0:
                self._print_status()

        self._running = False
        self._print_final_results()

    def stop(self):
        """Stop the coordinator."""
        self._running = False

    def _print_status(self):
        """Print current NAS status."""
        print("\n" + "=" * 70)
        print(f"DISTRIBUTED NAS STATUS - Generation {self.state.generation}")
        print("=" * 70)
        print(f"Strategy: {self.strategy}")
        print(f"Total Evaluations: {self.state.total_evaluations}")
        print(f"Best Performance: {self.state.best_performance:.4f}")
        print(f"Active Workers: {len(self.available_workers)}")
        print(f"Active Tasks: {len(self._active_tasks)}")

        if self.state.pareto_front:
            print(f"\nPareto Front ({len(self.state.pareto_front)} architectures):")
            for i, arch in enumerate(self.state.pareto_front[:3]):
                print(f"  {i+1}. perf={arch['performance']:.4f}, "
                      f"latency={arch['latency_ms']:.2f}ms")

        print("=" * 70 + "\n")

    def _print_final_results(self):
        """Print final NAS results."""
        print("\n" + "=" * 70)
        print("DISTRIBUTED NAS COMPLETE")
        print("=" * 70)
        print(f"Strategy: {self.strategy}")
        print(f"Total Evaluations: {self.state.total_evaluations}")
        print(f"Generations: {self.state.generation}")
        print(f"Best Performance: {self.state.best_performance:.4f}")

        if self.state.best_architecture:
            print("\nBest Architecture:")
            for key, value in self.state.best_architecture.items():
                print(f"  {key}: {value}")

        # Worker statistics
        print("\nWorker Statistics:")
        for worker in self.available_workers:
            print(f"  {worker.name}: checked")

        # Save final state
        output_dir = AI_SERVICE_ROOT / "logs" / "nas" / self.run_id

        # Save best architecture
        if self.state.best_architecture:
            best_file = output_dir / "best_architecture.json"
            with open(best_file, "w") as f:
                json.dump({
                    "performance": self.state.best_performance,
                    "params": self.state.best_architecture,
                }, f, indent=2)
            print(f"\nResults saved to: {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

async def async_main(args):
    """Async main entry point."""
    # Generate run ID
    if args.resume:
        run_id = args.resume
    else:
        run_id = f"nas_dist_{args.strategy}_{int(time.time())}"

    # Determine shared work directory
    shared_work_dir = None
    if args.shared_work_dir:
        shared_work_dir = Path(args.shared_work_dir)

    # Create coordinator
    coordinator = DistributedNASCoordinator(
        run_id=run_id,
        strategy=args.strategy,
        board_type=args.board,
        num_players=args.players,
        population_size=args.population,
        max_concurrent=args.max_concurrent,
        quick_eval=args.quick_eval,
        shared_work_dir=shared_work_dir,
    )

    # Initialize
    await coordinator.initialize(resume=bool(args.resume))

    # Run NAS
    if args.strategy == "evolutionary":
        max_iterations = args.generations
    else:
        max_iterations = args.trials

    await coordinator.run(max_iterations)


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Neural Architecture Search for RingRift AI"
    )

    # Mode selection
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run as worker process (for shared filesystem clusters like GH200)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="Shared work directory for task coordination (required for --worker)",
    )
    parser.add_argument(
        "--worker-name",
        type=str,
        help="Worker name/ID (defaults to hostname)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=-1,
        help="Maximum tasks for worker to process (-1 for unlimited)",
    )

    # Coordinator options
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["evolutionary", "bayesian"],
        default="evolutionary",
        help="Search strategy",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=40,
        help="Population size (for evolutionary)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (for evolutionary)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials (for bayesian)",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        help="Board type",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent evaluations",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        default=True,
        help="Use quick evaluation (default: true)",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Use full evaluation (overrides --quick-eval)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from existing NAS run ID",
    )
    parser.add_argument(
        "--shared-work-dir",
        type=str,
        help="Shared work directory for GH200 cluster coordination",
    )

    args = parser.parse_args()

    # Entry point resource validation (enforced 2025-12-16)
    # NAS requires significant resources for model evaluation
    if HAS_RESOURCE_GUARD:
        if not resource_can_proceed(check_disk=True, check_mem=True, check_gpu=True):
            logger.error("Insufficient resources to start distributed NAS")
            logger.error("CPU/Memory/GPU usage exceeds 80% limit or disk exceeds 70%")
            logger.error("Free up resources before starting NAS")
            return
        logger.info("Resource check passed: disk, memory, GPU within limits")

    # Worker mode
    if args.worker:
        if not args.work_dir:
            parser.error("--work-dir is required when running as --worker")

        import socket
        worker_name = args.worker_name or f"worker_{socket.gethostname()}_{os.getpid()}"

        logger.info(f"Starting NAS worker: {worker_name}")
        logger.info(f"Work directory: {args.work_dir}")

        run_shared_fs_worker(
            work_dir=Path(args.work_dir),
            worker_name=worker_name,
            max_tasks=args.max_tasks,
        )
        return

    # Coordinator mode
    if args.full_eval:
        args.quick_eval = False

    logger.info(f"Starting Distributed NAS Coordinator with strategy={args.strategy}")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
