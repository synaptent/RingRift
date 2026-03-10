"""Coordinator Resource Governor - Cross-process resource budget enforcement.

THIS IS THE FIX for the recurring mac-studio kernel panics (Mar 6, 2026).

Problem:
    The coordinator runs two independent process trees (P2P orchestrator +
    master_loop) that each spawn heavy work (exports, evaluations, S3 uploads,
    tournament matches) without knowing what the other is doing. Individual
    per-operation caps (MAX_CONCURRENT_EVALUATIONS, MAX_EXPORT_SAMPLES, etc.)
    are insufficient because they don't account for COMBINED load.

    Example crash sequence: evaluation daemon spawns 478 workers while export
    accumulates 17GB in RAM while 4 S3 uploads saturate disk I/O.

Solution:
    A shared SQLite-backed resource governor that enforces a global budget.
    Before any heavy operation starts, it must acquire a "slot" from the
    governor. The governor tracks total resource usage across all processes
    and rejects new work when the machine is at capacity.

Architecture:
    - Single SQLite database (data/coordinator_governor.db) shared by all processes
    - Lock table with operation type, PID, start time, estimated resource usage
    - Stale lock cleanup via PID liveness check (same pattern as ExportCoordinator)
    - Resource budget based on actual system state (psutil), not just slot counts

Usage:
    from app.utils.coordinator_governor import get_governor, OperationType

    gov = get_governor()

    # Before starting heavy work:
    slot = gov.try_acquire(OperationType.EVALUATION, estimated_ram_gb=4.0)
    if slot is None:
        logger.info("Governor denied evaluation: system at capacity")
        return

    try:
        run_evaluation(...)
    finally:
        gov.release(slot)

    # Or as context manager:
    async with gov.acquire_or_skip(OperationType.EXPORT, estimated_ram_gb=6.0) as slot:
        if slot is None:
            return  # Skipped due to resource pressure
        run_export(...)

Design Principles:
    1. FAIL OPEN: If the governor DB is broken, allow the operation (don't block the pipeline)
    2. STALE CLEANUP: Dead PIDs automatically release their slots on next check
    3. BUDGET-BASED: Limits based on actual available RAM/CPU, not arbitrary slot counts
    4. SINGLE SOURCE OF TRUTH: All resource limits live here, not scattered across env vars
    5. CROSS-PROCESS: Works across master_loop + P2P + any subprocess

March 6, 2026: Created after 3 kernel panics from uncoordinated resource consumption.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of heavy operations that need resource governance."""

    EVALUATION = "evaluation"  # Gauntlet evaluation (~4GB RAM, 50+ threads)
    EXPORT = "export"  # NPZ export (~4-17GB RAM, I/O heavy)
    S3_UPLOAD = "s3_upload"  # S3 cp/sync (~500MB RAM, disk I/O heavy)
    TOURNAMENT = "tournament"  # Tournament match (~2GB RAM, CPU heavy)
    TRAINING = "training"  # Training subprocess (~8GB+ RAM, GPU)
    SELFPLAY = "selfplay"  # Selfplay subprocess (~2GB RAM, GPU)


@dataclass
class ResourceBudget:
    """Resource budget for the coordinator.

    These are the TOTAL resources available for heavy operations,
    after reserving headroom for the OS, P2P, master_loop, and daemons.
    """

    # Total system RAM available for heavy ops (GB)
    # Mac Studio M3 Ultra has 96GB; reserve 20GB for system + daemons
    max_heavy_ram_gb: float = 40.0

    # Max concurrent heavy operations by type
    max_evaluations: int = 1
    max_exports: int = 1
    max_s3_uploads: int = 2
    max_tournaments: int = 0  # Coordinator should NEVER run tournaments
    max_training: int = 0  # Coordinator doesn't train
    max_selfplay: int = 0  # Coordinator doesn't selfplay

    # Total concurrent heavy operations across all types
    max_total_heavy_ops: int = 3

    # Minimum free RAM before ANY heavy op is allowed (GB)
    min_free_ram_gb: float = 16.0

    # Minimum free disk before I/O-heavy ops (percent)
    min_free_disk_percent: float = 10.0

    @classmethod
    def for_coordinator(cls) -> ResourceBudget:
        """Budget for coordinator nodes (mac-studio)."""
        return cls(
            max_heavy_ram_gb=40.0,
            max_evaluations=1,
            max_exports=1,
            max_s3_uploads=2,
            max_tournaments=0,
            max_training=0,
            max_selfplay=0,
            max_total_heavy_ops=3,
            min_free_ram_gb=16.0,
        )

    @classmethod
    def for_worker(cls) -> ResourceBudget:
        """Budget for GPU worker nodes."""
        return cls(
            max_heavy_ram_gb=60.0,
            max_evaluations=2,
            max_exports=2,
            max_s3_uploads=4,
            max_tournaments=2,
            max_training=1,
            max_selfplay=8,
            max_total_heavy_ops=12,
            min_free_ram_gb=8.0,
        )


# Estimated RAM usage per operation type (GB)
_ESTIMATED_RAM: dict[OperationType, float] = {
    OperationType.EVALUATION: 4.0,
    OperationType.EXPORT: 6.0,
    OperationType.S3_UPLOAD: 0.5,
    OperationType.TOURNAMENT: 2.0,
    OperationType.TRAINING: 8.0,
    OperationType.SELFPLAY: 2.0,
}

# Maximum slot age (seconds) before TTL cleanup releases it.
# These are generous upper bounds — normal operations finish well within these.
# The TTL exists to catch leaked slots from unexecuted finally blocks.
_DEFAULT_MAX_SLOT_AGE = 7200  # 2 hours
_MAX_SLOT_AGE: dict[str, float] = {
    OperationType.EVALUATION.value: 14400,  # 4 hours (large board gauntlets)
    OperationType.EXPORT.value: 7200,  # 2 hours
    OperationType.S3_UPLOAD.value: 3600,  # 1 hour
    OperationType.TOURNAMENT.value: 7200,  # 2 hours
    OperationType.TRAINING.value: 18000,  # 5 hours
    OperationType.SELFPLAY.value: 14400,  # 4 hours
}


@dataclass
class Slot:
    """A resource slot acquired from the governor."""

    slot_id: int
    operation: OperationType
    pid: int
    acquired_at: float
    estimated_ram_gb: float


class CoordinatorGovernor:
    """Cross-process resource governor using SQLite for coordination.

    This is the SINGLE POINT OF CONTROL for all heavy operations on a node.
    Both P2P orchestrator and master_loop must acquire slots before starting
    heavy work.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        budget: ResourceBudget | None = None,
    ):
        if db_path is None:
            base = Path(__file__).resolve().parent.parent.parent
            db_path = base / "data" / "coordinator_governor.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if budget is None:
            is_coordinator = os.environ.get(
                "RINGRIFT_IS_COORDINATOR", ""
            ).lower() in ("true", "1", "yes")
            budget = (
                ResourceBudget.for_coordinator()
                if is_coordinator
                else ResourceBudget.for_worker()
            )
        self.budget = budget
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the slots table if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    acquired_at REAL NOT NULL,
                    estimated_ram_gb REAL NOT NULL DEFAULT 0.0,
                    description TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_slots_operation
                ON active_slots(operation)
            """)
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.warning(f"[Governor] Failed to create schema: {e}")

    def _cleanup_stale(self, conn: sqlite3.Connection) -> int:
        """Remove slots held by dead processes OR exceeding TTL.

        Two cleanup mechanisms:
        1. Dead PID: process no longer exists -> slot leaked at process death
        2. TTL expired: slot held longer than max age -> finally block never ran
           (e.g. async cancellation, unhandled error in long-lived process)

        The TTL fix is critical for master_loop (PID stays alive for days)
        where daemon cycle methods can leak slots without killing the process.
        """
        now = time.time()
        rows = conn.execute(
            "SELECT id, pid, operation, acquired_at, description "
            "FROM active_slots"
        ).fetchall()
        stale_ids = []
        for slot_id, pid, operation, acquired_at, description in rows:
            age_seconds = now - acquired_at
            max_age = _MAX_SLOT_AGE.get(operation, _DEFAULT_MAX_SLOT_AGE)

            if not _is_pid_alive(pid):
                stale_ids.append(slot_id)
                logger.info(
                    f"[Governor] Cleaning dead-PID slot {slot_id}: "
                    f"{operation} pid={pid} desc={description}"
                )
            elif age_seconds > max_age:
                stale_ids.append(slot_id)
                logger.warning(
                    f"[Governor] Cleaning TTL-expired slot {slot_id}: "
                    f"{operation} held for {age_seconds:.0f}s "
                    f"(max {max_age:.0f}s) pid={pid} desc={description}"
                )

        if stale_ids:
            placeholders = ",".join("?" * len(stale_ids))
            conn.execute(
                f"DELETE FROM active_slots WHERE id IN ({placeholders})",
                stale_ids,
            )
            conn.commit()
            logger.info(
                f"[Governor] Cleaned {len(stale_ids)} stale slots"
            )
        return len(stale_ids)

    def _get_type_limit(self, op: OperationType) -> int:
        """Get the max concurrent limit for an operation type."""
        limits = {
            OperationType.EVALUATION: self.budget.max_evaluations,
            OperationType.EXPORT: self.budget.max_exports,
            OperationType.S3_UPLOAD: self.budget.max_s3_uploads,
            OperationType.TOURNAMENT: self.budget.max_tournaments,
            OperationType.TRAINING: self.budget.max_training,
            OperationType.SELFPLAY: self.budget.max_selfplay,
        }
        return limits.get(op, 1)

    def try_acquire(
        self,
        operation: OperationType,
        estimated_ram_gb: float | None = None,
        description: str = "",
    ) -> Slot | None:
        """Try to acquire a resource slot. Returns None if denied.

        Checks:
        1. Per-type concurrent limit not exceeded
        2. Total concurrent heavy ops not exceeded
        3. Estimated total RAM within budget
        4. Actual free RAM above minimum threshold
        5. Descendant process count below hard cap
        """
        if estimated_ram_gb is None:
            estimated_ram_gb = _ESTIMATED_RAM.get(operation, 2.0)

        type_limit = self._get_type_limit(operation)
        if type_limit <= 0:
            logger.info(
                f"[Governor] DENIED {operation.value}: "
                f"type limit is 0 (not allowed on this node)"
            )
            return None

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_table_exists(conn)
            self._cleanup_stale(conn)

            # Check 1: Per-type limit
            type_count = conn.execute(
                "SELECT COUNT(*) FROM active_slots WHERE operation = ?",
                (operation.value,),
            ).fetchone()[0]

            if type_count >= type_limit:
                conn.close()
                logger.info(
                    f"[Governor] DENIED {operation.value}: "
                    f"{type_count}/{type_limit} slots in use"
                )
                return None

            # Check 2: Total heavy ops limit
            total_count = conn.execute(
                "SELECT COUNT(*) FROM active_slots"
            ).fetchone()[0]

            if total_count >= self.budget.max_total_heavy_ops:
                conn.close()
                logger.info(
                    f"[Governor] DENIED {operation.value}: "
                    f"{total_count}/{self.budget.max_total_heavy_ops} "
                    f"total heavy ops"
                )
                return None

            # Check 3: RAM budget
            total_estimated_ram = conn.execute(
                "SELECT COALESCE(SUM(estimated_ram_gb), 0) FROM active_slots"
            ).fetchone()[0]

            if (
                total_estimated_ram + estimated_ram_gb
                > self.budget.max_heavy_ram_gb
            ):
                conn.close()
                logger.info(
                    f"[Governor] DENIED {operation.value}: "
                    f"estimated RAM {total_estimated_ram + estimated_ram_gb:.1f}GB "
                    f"> budget {self.budget.max_heavy_ram_gb:.0f}GB"
                )
                return None

            # Check 4: Actual free RAM
            free_ram_gb = _get_free_ram_gb()
            if free_ram_gb is not None and free_ram_gb < self.budget.min_free_ram_gb:
                conn.close()
                logger.info(
                    f"[Governor] DENIED {operation.value}: "
                    f"free RAM {free_ram_gb:.1f}GB "
                    f"< minimum {self.budget.min_free_ram_gb:.0f}GB"
                )
                return None

            # Check 5: Descendant process count
            descendant_count = _count_descendant_processes()
            max_descendants = (
                _COORDINATOR_MAX_DESCENDANTS
                if self.budget.max_training == 0  # coordinator
                else _WORKER_MAX_DESCENDANTS
            )
            if descendant_count is not None and descendant_count >= max_descendants:
                conn.close()
                logger.info(
                    f"[Governor] DENIED {operation.value}: "
                    f"descendant processes {descendant_count} >= limit {max_descendants}"
                )
                return None

            # All checks passed — acquire slot
            pid = os.getpid()
            now = time.time()
            cursor = conn.execute(
                "INSERT INTO active_slots "
                "(operation, pid, acquired_at, estimated_ram_gb, description) "
                "VALUES (?, ?, ?, ?, ?)",
                (operation.value, pid, now, estimated_ram_gb, description),
            )
            slot_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(
                f"[Governor] ACQUIRED {operation.value} slot {slot_id} "
                f"(pid={pid}, est_ram={estimated_ram_gb:.1f}GB, "
                f"active={total_count + 1}/{self.budget.max_total_heavy_ops})"
            )

            return Slot(
                slot_id=slot_id,
                operation=operation,
                pid=pid,
                acquired_at=now,
                estimated_ram_gb=estimated_ram_gb,
            )

        except (sqlite3.Error, OSError) as e:
            logger.warning(
                f"[Governor] DB error acquiring {operation.value}: {e}. "
                "Failing OPEN (allowing operation)."
            )
            # Fail open: return a synthetic slot so work isn't blocked
            return Slot(
                slot_id=-1,
                operation=operation,
                pid=os.getpid(),
                acquired_at=time.time(),
                estimated_ram_gb=estimated_ram_gb,
            )

    def release(self, slot: Slot) -> None:
        """Release a resource slot."""
        if slot.slot_id == -1:
            return  # Synthetic slot from fail-open path

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute(
                "DELETE FROM active_slots WHERE id = ?",
                (slot.slot_id,),
            )
            conn.commit()
            conn.close()
            elapsed = time.time() - slot.acquired_at
            logger.info(
                f"[Governor] RELEASED {slot.operation.value} slot {slot.slot_id} "
                f"after {elapsed:.0f}s"
            )
        except (sqlite3.Error, OSError) as e:
            logger.warning(f"[Governor] Failed to release slot {slot.slot_id}: {e}")

    @contextmanager
    def acquire_or_skip(
        self,
        operation: OperationType,
        estimated_ram_gb: float | None = None,
        description: str = "",
    ):
        """Context manager that yields a Slot or None if denied.

        Usage:
            with gov.acquire_or_skip(OperationType.EXPORT) as slot:
                if slot is None:
                    return  # Denied
                do_export()
        """
        slot = self.try_acquire(operation, estimated_ram_gb, description)
        try:
            yield slot
        finally:
            if slot is not None:
                self.release(slot)

    def get_status(self) -> dict[str, Any]:
        """Get current governor status for monitoring."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            self._cleanup_stale(conn)

            slots = conn.execute(
                "SELECT operation, pid, acquired_at, estimated_ram_gb, description "
                "FROM active_slots ORDER BY acquired_at"
            ).fetchall()
            conn.close()

            by_type: dict[str, int] = {}
            total_ram = 0.0
            active = []
            now = time.time()
            for op, pid, acquired, ram, desc in slots:
                by_type[op] = by_type.get(op, 0) + 1
                total_ram += ram
                max_age = _MAX_SLOT_AGE.get(op, _DEFAULT_MAX_SLOT_AGE)
                age = now - acquired
                active.append({
                    "operation": op,
                    "pid": pid,
                    "running_seconds": int(age),
                    "ttl_remaining_seconds": int(max_age - age),
                    "estimated_ram_gb": ram,
                    "description": desc,
                })

            return {
                "active_slots": len(slots),
                "max_slots": self.budget.max_total_heavy_ops,
                "estimated_ram_gb": round(total_ram, 1),
                "max_ram_gb": self.budget.max_heavy_ram_gb,
                "free_ram_gb": round(_get_free_ram_gb() or 0, 1),
                "descendant_processes": _count_descendant_processes(),
                "by_type": by_type,
                "active": active,
            }
        except (sqlite3.Error, OSError) as e:
            return {"error": str(e)}

    def _ensure_table_exists(self, conn: sqlite3.Connection) -> None:
        """Ensure the table exists (for robustness)."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                pid INTEGER NOT NULL,
                acquired_at REAL NOT NULL,
                estimated_ram_gb REAL NOT NULL DEFAULT 0.0,
                description TEXT DEFAULT ''
            )
        """)


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _get_free_ram_gb() -> float | None:
    """Get free RAM in GB, or None if psutil unavailable."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except ImportError:
        return None


def _count_descendant_processes() -> int | None:
    """Count all descendant processes of the current process.

    Returns None if psutil is unavailable.
    """
    try:
        import psutil

        proc = psutil.Process()
        return len(proc.children(recursive=True))
    except (ImportError, psutil.Error):
        return None


# Hard limits on descendant process count
_COORDINATOR_MAX_DESCENDANTS = 120  # Coordinator manages 15+ nodes via SSH
_WORKER_MAX_DESCENDANTS = 200


class ProcessWatchdog:
    """Daemon thread that kills excess descendant processes to prevent kernel panics.

    Checks descendant process count every `check_interval` seconds. If count
    exceeds `max_processes`, kills the newest excess processes via SIGTERM.
    Does nothing if psutil is unavailable.

    Usage:
        watchdog = ProcessWatchdog(max_processes=60)
        watchdog.start()  # Runs as daemon thread, dies with parent
    """

    def __init__(
        self,
        max_processes: int = _COORDINATOR_MAX_DESCENDANTS,
        check_interval: float = 10.0,
    ):
        self.max_processes = max_processes
        self.check_interval = check_interval
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the watchdog daemon thread."""
        import threading

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ProcessWatchdog",
        )
        self._thread.start()
        logger.info(
            f"[ProcessWatchdog] Started (max_processes={self.max_processes}, "
            f"interval={self.check_interval}s)"
        )

    def _run(self) -> None:
        """Main watchdog loop."""
        try:
            import psutil
        except ImportError:
            logger.warning("[ProcessWatchdog] psutil unavailable, watchdog disabled")
            return

        while True:
            try:
                time.sleep(self.check_interval)
                self._check_and_kill(psutil)
            except Exception as e:
                logger.warning(f"[ProcessWatchdog] Error in check cycle: {e}")

    # Never kill these processes — they're system-critical infrastructure
    PROTECTED_NAMES = frozenset({
        "tailscale", "tailscaled",  # VPN connectivity
        "ssh", "sshd",  # Remote operations
        "rsync",  # File sync
        "aws", "s3",  # S3 operations
        "pgrep", "pkill", "ps",  # Process utilities
        "git",  # Version control
        "launchd", "launchctl",  # macOS service management
        "mdworker", "mds",  # Spotlight (shouldn't be child but be safe)
    })

    def _check_and_kill(self, psutil_mod: Any) -> None:
        """Check descendant count and kill excess processes.

        Only kills Python worker processes. System processes (ssh, tailscale,
        rsync, etc.) are protected and never killed.
        """
        try:
            proc = psutil_mod.Process()
            children = proc.children(recursive=True)
            count = len(children)

            if count <= self.max_processes:
                return

            excess = count - self.max_processes
            # Sort by creation time descending (newest first) to kill newest
            # Only consider killable (non-protected) processes
            killable = []
            protected_count = 0
            for child in children:
                try:
                    child_name = child.name()
                    if child_name in self.PROTECTED_NAMES:
                        protected_count += 1
                        continue
                    killable.append((child.create_time(), child, child_name))
                except (psutil_mod.NoSuchProcess, psutil_mod.AccessDenied):
                    continue

            killable.sort(reverse=True)  # newest first

            killed = 0
            for _ctime, child, child_name in killable[:excess]:
                try:
                    child_pid = child.pid
                    child.terminate()  # SIGTERM
                    killed += 1
                    logger.critical(
                        f"[ProcessWatchdog] KILLED excess process: "
                        f"pid={child_pid} name={child_name} "
                        f"(descendants={count}/{self.max_processes})"
                    )
                except (psutil_mod.NoSuchProcess, psutil_mod.AccessDenied):
                    continue

            if killed:
                logger.critical(
                    f"[ProcessWatchdog] Killed {killed}/{excess} excess processes "
                    f"(total descendants: {count} > limit {self.max_processes}, "
                    f"protected={protected_count})"
                )
            elif excess > 0 and not killable:
                logger.warning(
                    f"[ProcessWatchdog] {count} descendants > limit {self.max_processes}, "
                    f"but all {protected_count} excess are protected processes — "
                    f"not killing"
                )

        except psutil_mod.NoSuchProcess:
            pass  # Parent process gone


# Singleton
_governor: CoordinatorGovernor | None = None


def get_governor() -> CoordinatorGovernor:
    """Get the singleton governor instance."""
    global _governor
    if _governor is None:
        _governor = CoordinatorGovernor()
    return _governor
