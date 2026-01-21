"""Memory Monitor Daemon (December 2025).

Monitors GPU VRAM and process memory (RSS) to detect memory pressure
before OOM conditions occur. Part of 48-hour autonomous operation.

Key thresholds (Jan 21, 2026 - Phase 4 lowered for earlier intervention):
- GPU VRAM: 65% caution → 70% warning → 80% critical → Emit MEMORY_PRESSURE
- System RAM: 70% caution → 75% warning → 85% critical → Emit RESOURCE_CONSTRAINT
- Process RSS: 24GB warning → 28GB critical → SIGTERM → wait 60s → SIGKILL

Rollback: RINGRIFT_MEMORY_LEGACY_THRESHOLDS=true (uses 75/85% GPU, 80/90% RAM, 32GB RSS)

Usage:
    from app.coordination.memory_monitor_daemon import MemoryMonitorDaemon

    daemon = MemoryMonitorDaemon.get_instance()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


# Environment variable prefix
_ENV_PREFIX = "RINGRIFT_MEMORY_"


def _env_float(name: str, default: float) -> float:
    """Get float from environment."""
    return float(os.environ.get(f"{_ENV_PREFIX}{name}", str(default)))


def _env_int(name: str, default: int) -> int:
    """Get int from environment."""
    return int(os.environ.get(f"{_ENV_PREFIX}{name}", str(default)))


def _env_bool(name: str, default: bool) -> bool:
    """Get bool from environment."""
    val = os.environ.get(f"{_ENV_PREFIX}{name}", str(default).lower())
    return val.lower() in ("true", "1", "yes")


@dataclass
class MemoryThresholds:
    """Memory threshold configuration.

    All thresholds are percentages (0.0 to 1.0).

    Jan 21, 2026: Phase 4 - Lowered thresholds for earlier intervention.
    - Added CAUTION tier (65% GPU, 70% RAM) for early warning
    - Lowered WARNING tier (70% GPU, 75% RAM) to reduce batch sizes
    - Lowered CRITICAL tier (80% GPU, 85% RAM) to pause spawning earlier
    - Lowered process RSS thresholds (24GB warning, 28GB critical)
    Rollback: RINGRIFT_MEMORY_LEGACY_THRESHOLDS=true (uses 75/85% GPU, 80/90% RAM)
    """

    # GPU VRAM thresholds
    # Jan 21, 2026: Added caution tier, lowered warning/critical for earlier OOM prevention
    gpu_caution: float = 0.65  # 65% - log + emit MEMORY_CAUTION
    gpu_warning: float = 0.70  # 70% - reduce batch sizes (was 75%)
    gpu_critical: float = 0.80  # 80% - pause spawning (was 85%)

    # System RAM thresholds
    # Jan 21, 2026: Added caution tier, lowered warning/critical
    ram_caution: float = 0.70  # 70% - log + emit MEMORY_CAUTION
    ram_warning: float = 0.75  # 75% - reduce batch sizes (was 80%)
    ram_critical: float = 0.85  # 85% - emit RESOURCE_CONSTRAINT (was 90%)

    # Process RSS thresholds (bytes)
    # Jan 21, 2026: Added warning tier, lowered critical
    process_rss_warning_bytes: int = 24 * 1024 * 1024 * 1024  # 24GB warning
    process_rss_critical_bytes: int = 28 * 1024 * 1024 * 1024  # 28GB critical (was 32GB)

    # Grace period before SIGKILL (seconds)
    sigkill_grace_period: float = 60.0


@dataclass
class MemoryMonitorConfig:
    """Configuration for MemoryMonitorDaemon."""

    enabled: bool = True
    check_interval_seconds: float = 30.0

    # Thresholds
    thresholds: MemoryThresholds = field(default_factory=MemoryThresholds)

    # Whether to actually kill processes (vs just logging)
    kill_enabled: bool = True

    # Cool down between MEMORY_PRESSURE events (seconds)
    event_cooldown_seconds: float = 60.0

    # Enable/disable specific monitors
    monitor_gpu: bool = True
    monitor_ram: bool = True
    monitor_processes: bool = True

    @classmethod
    def from_env(cls) -> MemoryMonitorConfig:
        """Create config from environment variables.

        Jan 21, 2026: Phase 4 - Added RINGRIFT_MEMORY_LEGACY_THRESHOLDS=true
        for rollback to old thresholds (75/85% GPU, 80/90% RAM, 32GB RSS).
        """
        # Check for legacy mode rollback
        legacy_mode = os.environ.get(
            "RINGRIFT_MEMORY_LEGACY_THRESHOLDS", ""
        ).lower() in ("true", "1", "yes")

        if legacy_mode:
            # Legacy thresholds (pre-Phase 4)
            thresholds = MemoryThresholds(
                gpu_caution=0.70,  # No caution in legacy, set to warning level
                gpu_warning=_env_float("GPU_WARNING_THRESHOLD", 0.75),
                gpu_critical=_env_float("GPU_CRITICAL_THRESHOLD", 0.85),
                ram_caution=0.75,  # No caution in legacy, set to warning level
                ram_warning=_env_float("RAM_WARNING_THRESHOLD", 0.80),
                ram_critical=_env_float("RAM_CRITICAL_THRESHOLD", 0.90),
                process_rss_warning_bytes=_env_int(
                    "PROCESS_RSS_WARNING_GB", 28
                ) * 1024 * 1024 * 1024,
                process_rss_critical_bytes=_env_int(
                    "PROCESS_RSS_CRITICAL_GB", 32
                ) * 1024 * 1024 * 1024,
                sigkill_grace_period=_env_float("SIGKILL_GRACE_PERIOD", 60.0),
            )
        else:
            # New Phase 4 thresholds (Jan 21, 2026)
            thresholds = MemoryThresholds(
                gpu_caution=_env_float("GPU_CAUTION_THRESHOLD", 0.65),
                gpu_warning=_env_float("GPU_WARNING_THRESHOLD", 0.70),
                gpu_critical=_env_float("GPU_CRITICAL_THRESHOLD", 0.80),
                ram_caution=_env_float("RAM_CAUTION_THRESHOLD", 0.70),
                ram_warning=_env_float("RAM_WARNING_THRESHOLD", 0.75),
                ram_critical=_env_float("RAM_CRITICAL_THRESHOLD", 0.85),
                process_rss_warning_bytes=_env_int(
                    "PROCESS_RSS_WARNING_GB", 24
                ) * 1024 * 1024 * 1024,
                process_rss_critical_bytes=_env_int(
                    "PROCESS_RSS_CRITICAL_GB", 28
                ) * 1024 * 1024 * 1024,
                sigkill_grace_period=_env_float("SIGKILL_GRACE_PERIOD", 60.0),
            )
        return cls(
            enabled=_env_bool("ENABLED", True),
            check_interval_seconds=_env_float("CHECK_INTERVAL", 30.0),
            thresholds=thresholds,
            kill_enabled=_env_bool("KILL_ENABLED", True),
            event_cooldown_seconds=_env_float("EVENT_COOLDOWN", 60.0),
            monitor_gpu=_env_bool("MONITOR_GPU", True),
            monitor_ram=_env_bool("MONITOR_RAM", True),
            monitor_processes=_env_bool("MONITOR_PROCESSES", True),
        )


@dataclass
class MemoryStatus:
    """Current memory status snapshot.

    Jan 21, 2026: Phase 4 - Added caution flags and process warning flag.
    """

    # GPU memory
    gpu_used_bytes: int = 0
    gpu_total_bytes: int = 0
    gpu_utilization: float = 0.0
    gpu_available: bool = False

    # System RAM
    ram_used_bytes: int = 0
    ram_total_bytes: int = 0
    ram_utilization: float = 0.0

    # Largest selfplay process
    largest_process_pid: int = 0
    largest_process_rss_bytes: int = 0

    # Status flags - Jan 21, 2026: Added caution tier
    gpu_caution: bool = False  # 65% - early warning
    gpu_warning: bool = False  # 70% - reduce batch sizes
    gpu_critical: bool = False  # 80% - pause spawning
    ram_caution: bool = False  # 70% - early warning
    ram_warning: bool = False  # 75% - reduce batch sizes
    ram_critical: bool = False  # 85% - emit RESOURCE_CONSTRAINT
    process_warning: bool = False  # 24GB - log warning
    process_critical: bool = False  # 28GB - kill process

    @property
    def any_critical(self) -> bool:
        """Check if any resource is in critical state."""
        return self.gpu_critical or self.ram_critical or self.process_critical

    @property
    def any_caution(self) -> bool:
        """Check if any resource is in caution state (but not warning/critical)."""
        return (self.gpu_caution or self.ram_caution) and not self.any_warning

    @property
    def any_warning(self) -> bool:
        """Check if any resource is in warning state (but not critical)."""
        return (self.gpu_warning or self.ram_warning or self.process_warning) and not self.any_critical


class MemoryMonitorDaemon(HandlerBase):
    """Monitors GPU VRAM and process memory to prevent OOM.

    December 2025: Part of 48-hour autonomous operation optimization.
    Replaces 2hr process timeout for OOM recovery with proactive detection.

    Inherits from HandlerBase providing:
    - Singleton pattern via get_instance()
    - Standardized health check format
    - Lifecycle management (start/stop)
    """

    def __init__(self, config: MemoryMonitorConfig | None = None):
        self._memory_config = config or MemoryMonitorConfig.from_env()
        super().__init__(
            name="memory_monitor",
            config=self._memory_config,
            cycle_interval=self._memory_config.check_interval_seconds,
        )

        # State tracking
        self._last_memory_pressure_event: float = 0.0
        self._last_resource_constraint_event: float = 0.0
        self._pending_kills: dict[int, float] = {}  # pid -> SIGTERM sent time
        self._last_status: MemoryStatus | None = None

        # Stats
        self._gpu_warnings_emitted = 0
        self._gpu_criticals_emitted = 0
        self._ram_warnings_emitted = 0
        self._ram_criticals_emitted = 0
        self._processes_killed = 0

    @property
    def config(self) -> MemoryMonitorConfig:
        """Get the monitor configuration."""
        return self._memory_config

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """No event subscriptions - this daemon is cycle-based."""
        return {}

    async def _on_start(self) -> None:
        """Initialize monitoring."""
        if not self._memory_config.enabled:
            logger.info("[MemoryMonitor] Disabled by configuration")
            return

        logger.info(
            f"[MemoryMonitor] Started with thresholds: "
            f"GPU={self._memory_config.thresholds.gpu_critical:.0%}, "
            f"RAM={self._memory_config.thresholds.ram_critical:.0%}, "
            f"RSS={self._memory_config.thresholds.process_rss_critical_bytes / (1024**3):.0f}GB"
        )

    async def _on_stop(self) -> None:
        """Cleanup on stop."""
        logger.info("[MemoryMonitor] Stopped")

    async def _run_cycle(self) -> None:
        """Main monitoring cycle."""
        if not self._memory_config.enabled:
            return

        try:
            status = await self._collect_memory_status()
            self._last_status = status

            # Check GPU memory
            if self._memory_config.monitor_gpu and status.gpu_available:
                await self._check_gpu_memory(status)

            # Check system RAM
            if self._memory_config.monitor_ram:
                await self._check_ram(status)

            # Check large processes
            if self._memory_config.monitor_processes:
                await self._check_large_processes(status)

            # Process pending kills
            await self._process_pending_kills()

        except Exception as e:  # noqa: BLE001
            logger.error(f"[MemoryMonitor] Cycle error: {e}")

    async def _collect_memory_status(self) -> MemoryStatus:
        """Collect current memory status."""
        status = MemoryStatus()

        # Collect GPU memory
        try:
            gpu_info = await self._get_gpu_memory()
            if gpu_info:
                status.gpu_used_bytes, status.gpu_total_bytes = gpu_info
                if status.gpu_total_bytes > 0:
                    status.gpu_utilization = status.gpu_used_bytes / status.gpu_total_bytes
                    status.gpu_available = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[MemoryMonitor] GPU memory query failed: {e}")

        # Collect system RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            status.ram_used_bytes = mem.used
            status.ram_total_bytes = mem.total
            status.ram_utilization = mem.percent / 100.0
        except ImportError:
            logger.debug("[MemoryMonitor] psutil not available for RAM monitoring")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[MemoryMonitor] RAM query failed: {e}")

        # Find largest selfplay process
        try:
            pid, rss = await self._find_largest_selfplay_process()
            status.largest_process_pid = pid
            status.largest_process_rss_bytes = rss
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[MemoryMonitor] Process scan failed: {e}")

        # Apply thresholds - Jan 21, 2026: Phase 4 added caution tier
        thresholds = self._memory_config.thresholds
        status.gpu_caution = status.gpu_utilization >= thresholds.gpu_caution
        status.gpu_warning = status.gpu_utilization >= thresholds.gpu_warning
        status.gpu_critical = status.gpu_utilization >= thresholds.gpu_critical
        status.ram_caution = status.ram_utilization >= thresholds.ram_caution
        status.ram_warning = status.ram_utilization >= thresholds.ram_warning
        status.ram_critical = status.ram_utilization >= thresholds.ram_critical
        status.process_warning = (
            status.largest_process_rss_bytes >= thresholds.process_rss_warning_bytes
        )
        status.process_critical = (
            status.largest_process_rss_bytes >= thresholds.process_rss_critical_bytes
        )

        return status

    async def _get_gpu_memory(self) -> tuple[int, int] | None:
        """Get GPU memory usage (used_bytes, total_bytes).

        December 2025: Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.debug("[MemoryMonitor] nvidia-smi timed out")
                return None

            if proc.returncode == 0:
                lines = stdout.decode().strip().split("\n")
                if lines:
                    # Sum across all GPUs
                    total_used = 0
                    total_total = 0
                    for line in lines:
                        parts = line.split(",")
                        if len(parts) == 2:
                            used_mb = int(parts[0].strip())
                            total_mb = int(parts[1].strip())
                            total_used += used_mb * 1024 * 1024
                            total_total += total_mb * 1024 * 1024
                    return total_used, total_total
        except FileNotFoundError:
            # nvidia-smi not available (no NVIDIA GPU)
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[MemoryMonitor] nvidia-smi failed: {e}")
        return None

    async def _find_largest_selfplay_process(self) -> tuple[int, int]:
        """Find the selfplay process with largest RSS.

        Returns:
            (pid, rss_bytes) of largest selfplay process, or (0, 0) if none found
        """
        try:
            import psutil
            largest_pid = 0
            largest_rss = 0

            for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                    if not cmdline:
                        continue

                    # Check if this is a selfplay process
                    cmd_str = " ".join(cmdline).lower()
                    if "selfplay" in cmd_str or "gpu_parallel_games" in cmd_str:
                        mem_info = proc.info.get("memory_info")
                        if mem_info and mem_info.rss > largest_rss:
                            largest_rss = mem_info.rss
                            largest_pid = proc.info.get("pid", 0)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return largest_pid, largest_rss
        except ImportError:
            return 0, 0

    async def _check_gpu_memory(self, status: MemoryStatus) -> None:
        """Check GPU memory and emit events.

        Jan 21, 2026: Phase 4 - Added caution tier for earlier intervention.
        """
        if status.gpu_critical:
            self._gpu_criticals_emitted += 1
            await self._emit_memory_pressure(status, "gpu")
            logger.warning(
                f"[MemoryMonitor] GPU VRAM CRITICAL: {status.gpu_utilization:.1%} "
                f"({status.gpu_used_bytes / (1024**3):.1f}GB / "
                f"{status.gpu_total_bytes / (1024**3):.1f}GB)"
            )
        elif status.gpu_warning:
            self._gpu_warnings_emitted += 1
            logger.warning(
                f"[MemoryMonitor] GPU VRAM warning: {status.gpu_utilization:.1%}"
            )
        elif status.gpu_caution:
            # Jan 21, 2026: Phase 4 - Emit caution event for early awareness
            logger.info(
                f"[MemoryMonitor] GPU VRAM caution: {status.gpu_utilization:.1%}"
            )
            await self._emit_memory_caution(status, "gpu")

    async def _check_ram(self, status: MemoryStatus) -> None:
        """Check system RAM and emit events.

        Jan 21, 2026: Phase 4 - Added caution tier for earlier intervention.
        """
        if status.ram_critical:
            self._ram_criticals_emitted += 1
            await self._emit_resource_constraint(status, "ram")
            logger.warning(
                f"[MemoryMonitor] System RAM CRITICAL: {status.ram_utilization:.1%} "
                f"({status.ram_used_bytes / (1024**3):.1f}GB / "
                f"{status.ram_total_bytes / (1024**3):.1f}GB)"
            )
        elif status.ram_warning:
            self._ram_warnings_emitted += 1
            logger.warning(
                f"[MemoryMonitor] System RAM warning: {status.ram_utilization:.1%}"
            )
        elif status.ram_caution:
            # Jan 21, 2026: Phase 4 - Emit caution event for early awareness
            logger.info(
                f"[MemoryMonitor] System RAM caution: {status.ram_utilization:.1%}"
            )
            await self._emit_memory_caution(status, "ram")

    async def _emit_memory_caution(self, status: MemoryStatus, resource_type: str) -> None:
        """Emit MEMORY_CAUTION event for early warning.

        Jan 21, 2026: Phase 4 - New event type for caution tier.
        """
        self._safe_emit_event(
            "MEMORY_CAUTION",
            {
                "resource_type": resource_type,
                "utilization": status.gpu_utilization if resource_type == "gpu" else status.ram_utilization,
                "node_id": getattr(self, "node_id", None) or os.environ.get("RINGRIFT_NODE_ID", "unknown"),
            },
        )

    async def _check_large_processes(self, status: MemoryStatus) -> None:
        """Check for oversized selfplay processes and kill if needed.

        Jan 21, 2026: Phase 4 - Added warning tier logging before critical action.
        """
        # Log warning tier first (24GB threshold)
        if status.process_warning and not status.process_critical:
            pid = status.largest_process_pid
            rss_gb = status.largest_process_rss_bytes / (1024**3)
            warning_threshold_gb = self._memory_config.thresholds.process_rss_warning_bytes / (1024**3)
            logger.warning(
                f"[MemoryMonitor] Process RSS warning: PID {pid} using "
                f"{rss_gb:.1f}GB (threshold: {warning_threshold_gb:.0f}GB)"
            )

        if not status.process_critical:
            return

        pid = status.largest_process_pid
        rss_gb = status.largest_process_rss_bytes / (1024**3)
        threshold_gb = self._memory_config.thresholds.process_rss_critical_bytes / (1024**3)

        logger.warning(
            f"[MemoryMonitor] Process RSS CRITICAL: PID {pid} using "
            f"{rss_gb:.1f}GB (threshold: {threshold_gb:.0f}GB)"
        )

        if self._memory_config.kill_enabled and pid > 0:
            await self._schedule_kill(pid)

    async def _schedule_kill(self, pid: int) -> None:
        """Schedule a process for termination."""
        if pid in self._pending_kills:
            # Already scheduled
            return

        logger.warning(f"[MemoryMonitor] Scheduling SIGTERM for PID {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
            self._pending_kills[pid] = time.time()
        except ProcessLookupError:
            logger.info(f"[MemoryMonitor] PID {pid} already terminated")
        except PermissionError:
            logger.error(f"[MemoryMonitor] Permission denied to kill PID {pid}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[MemoryMonitor] Failed to SIGTERM PID {pid}: {e}")

    async def _process_pending_kills(self) -> None:
        """Process pending kills - escalate to SIGKILL if needed."""
        now = time.time()
        grace_period = self._memory_config.thresholds.sigkill_grace_period
        completed = []

        for pid, sigterm_time in self._pending_kills.items():
            elapsed = now - sigterm_time

            # Check if process still exists
            try:
                import psutil
                if not psutil.pid_exists(pid):
                    completed.append(pid)
                    self._processes_killed += 1
                    logger.info(f"[MemoryMonitor] PID {pid} terminated (SIGTERM)")
                    continue
            except ImportError:
                # Try os.kill with signal 0 to check
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    completed.append(pid)
                    self._processes_killed += 1
                    logger.info(f"[MemoryMonitor] PID {pid} terminated (SIGTERM)")
                    continue
                except PermissionError:
                    pass  # Process exists but we can't check

            # Escalate to SIGKILL if grace period elapsed
            if elapsed >= grace_period:
                logger.warning(
                    f"[MemoryMonitor] Escalating to SIGKILL for PID {pid} "
                    f"(SIGTERM sent {elapsed:.0f}s ago)"
                )
                try:
                    os.kill(pid, signal.SIGKILL)
                    completed.append(pid)
                    self._processes_killed += 1
                except ProcessLookupError:
                    completed.append(pid)
                    self._processes_killed += 1
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[MemoryMonitor] Failed to SIGKILL PID {pid}: {e}")
                    completed.append(pid)  # Remove from pending anyway

        for pid in completed:
            self._pending_kills.pop(pid, None)

    async def _emit_memory_pressure(self, status: MemoryStatus, source: str) -> None:
        """Emit MEMORY_PRESSURE event with cooldown."""
        now = time.time()
        if now - self._last_memory_pressure_event < self._memory_config.event_cooldown_seconds:
            return

        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            if safe_emit_event(
                DataEventType.MEMORY_PRESSURE,
                {
                    "source": source,
                    "gpu_utilization": status.gpu_utilization,
                    "ram_utilization": status.ram_utilization,
                    "gpu_used_gb": status.gpu_used_bytes / (1024**3),
                    "gpu_total_gb": status.gpu_total_bytes / (1024**3),
                    "timestamp": now,
                },
                context="MemoryMonitor",
            ):
                self._last_memory_pressure_event = now
                logger.info(f"[MemoryMonitor] Emitted MEMORY_PRESSURE ({source})")
        except ImportError:
            logger.debug("[MemoryMonitor] event_emission_helpers not available")

    async def _emit_resource_constraint(self, status: MemoryStatus, source: str) -> None:
        """Emit RESOURCE_CONSTRAINT event with cooldown."""
        now = time.time()
        if now - self._last_resource_constraint_event < self._memory_config.event_cooldown_seconds:
            return

        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            if safe_emit_event(
                DataEventType.RESOURCE_CONSTRAINT,
                {
                    "source": source,
                    "resource_type": "memory",
                    "ram_utilization": status.ram_utilization,
                    "ram_used_gb": status.ram_used_bytes / (1024**3),
                    "ram_total_gb": status.ram_total_bytes / (1024**3),
                    "timestamp": now,
                },
                context="MemoryMonitor",
            ):
                self._last_resource_constraint_event = now
                logger.info(f"[MemoryMonitor] Emitted RESOURCE_CONSTRAINT ({source})")
        except ImportError:
            logger.debug("[MemoryMonitor] event_emission_helpers not available")

    def health_check(self) -> HealthCheckResult:
        """Check monitor health.

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.contracts import CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="MemoryMonitor not running",
            )

        if not self._memory_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="MemoryMonitor disabled by configuration",
                details={"enabled": False},
            )

        # Check if we're in a critical state
        details = self.get_stats()
        if self._last_status and self._last_status.any_critical:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message="MemoryMonitor: memory pressure detected",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="MemoryMonitor running normally",
            details=details,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        stats: dict[str, Any] = {
            "running": self._running,
            "enabled": self._memory_config.enabled,
            "gpu_warnings_emitted": self._gpu_warnings_emitted,
            "gpu_criticals_emitted": self._gpu_criticals_emitted,
            "ram_warnings_emitted": self._ram_warnings_emitted,
            "ram_criticals_emitted": self._ram_criticals_emitted,
            "processes_killed": self._processes_killed,
            "pending_kills": len(self._pending_kills),
        }

        if self._last_status:
            stats["last_status"] = {
                "gpu_utilization": self._last_status.gpu_utilization,
                "gpu_available": self._last_status.gpu_available,
                "ram_utilization": self._last_status.ram_utilization,
                "gpu_critical": self._last_status.gpu_critical,
                "ram_critical": self._last_status.ram_critical,
                "process_critical": self._last_status.process_critical,
            }

        return stats


# Singleton accessors
def get_memory_monitor() -> MemoryMonitorDaemon:
    """Get or create the singleton MemoryMonitorDaemon instance."""
    return MemoryMonitorDaemon.get_instance()


def reset_memory_monitor() -> None:
    """Reset the singleton instance (for testing)."""
    MemoryMonitorDaemon.reset_instance()


async def start_memory_monitor() -> MemoryMonitorDaemon:
    """Start the memory monitor (convenience function)."""
    monitor = get_memory_monitor()
    await monitor.start()
    return monitor


__all__ = [
    "MemoryMonitorConfig",
    "MemoryMonitorDaemon",
    "MemoryStatus",
    "MemoryThresholds",
    "get_memory_monitor",
    "reset_memory_monitor",
    "start_memory_monitor",
]
