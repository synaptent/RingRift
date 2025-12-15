"""
Safeguards for RingRift AI Task Spawning.

Provides additional protection layers to prevent runaway task spawning:
1. Circuit breakers - Auto-disable spawning on repeated failures
2. Backpressure - Slow down spawning when system is stressed
3. Global kill switch - Emergency stop all spawning
4. Resource monitors - Proactive detection of resource exhaustion
5. Spawn auditing - Track and limit spawn rates

This module should be integrated with all orchestrators.

Usage:
    from app.coordination.safeguards import Safeguards

    safeguards = Safeguards.get_instance()

    # Before spawning
    if safeguards.allow_spawn("selfplay", "node-1"):
        safeguards.record_spawn("selfplay", "node-1")
        # ... spawn task ...
    else:
        print(f"Spawn blocked: {safeguards.get_block_reason()}")

    # After failure
    safeguards.record_failure("selfplay", "node-1", "OOM")
"""

import json
import logging
import os
import psutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

@dataclass
class SafeguardConfig:
    """Configuration for safeguards."""
    # Circuit breaker settings
    failure_threshold: int = 5          # Failures before circuit opens
    recovery_timeout: float = 300.0     # Seconds before retry after open
    half_open_max_calls: int = 3        # Test calls in half-open state

    # Backpressure settings
    spawn_window_seconds: float = 60.0  # Window for rate calculation
    max_spawns_per_window: int = 30     # Max spawns in window
    backpressure_threshold: float = 0.8 # Slow down at 80% of limit

    # Resource thresholds
    disk_critical_percent: float = 90.0
    disk_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    memory_warning_percent: float = 85.0
    cpu_critical_percent: float = 98.0
    load_critical_multiplier: float = 2.0  # load > cpus * this = critical

    # Emergency settings
    emergency_file: str = "/tmp/ringrift_coordination/EMERGENCY_HALT"
    auto_emergency_on_oom: bool = True

    # Per-node limits
    max_selfplay_per_node: int = 32
    max_total_selfplay: int = 200      # Reduced from 500 for safety

    # Spawn delays (backpressure)
    normal_delay: float = 0.0
    warning_delay: float = 1.0
    critical_delay: float = 5.0


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking all calls
    HALF_OPEN = "half_open"  # Testing if recovered


# ============================================
# Circuit Breaker
# ============================================

@dataclass
class CircuitBreaker:
    """
    Circuit breaker for a specific spawn target.

    Prevents repeated failures by temporarily blocking spawns
    after too many failures.
    """
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_transition_time: float = 0.0
    config: SafeguardConfig = field(default_factory=SafeguardConfig)

    def record_success(self) -> None:
        """Record a successful spawn."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_max_calls:
                self._transition(CircuitState.CLOSED)
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            # Decay failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self, reason: str = "") -> None:
        """Record a failed spawn."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self.failure_count} failures"
                )
                self._transition(CircuitState.OPEN)

    def allow_call(self) -> bool:
        """Check if a call is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_transition_time >= self.config.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.success_count < self.config.half_open_max_calls

        return False

    def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.last_transition_time = time.time()
        self.success_count = 0

        logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")


# ============================================
# Spawn Rate Tracker
# ============================================

class SpawnRateTracker:
    """
    Tracks spawn rates for backpressure.

    Uses a sliding window to calculate spawn rate and
    apply backpressure when rate is too high.
    """

    def __init__(self, config: SafeguardConfig):
        self.config = config
        self._spawns: deque = deque()
        self._lock = threading.Lock()

    def record_spawn(self) -> None:
        """Record a spawn event."""
        with self._lock:
            now = time.time()
            self._spawns.append(now)
            # Cleanup old entries
            cutoff = now - self.config.spawn_window_seconds
            while self._spawns and self._spawns[0] < cutoff:
                self._spawns.popleft()

    def get_rate(self) -> float:
        """Get current spawn rate (spawns per window)."""
        with self._lock:
            now = time.time()
            cutoff = now - self.config.spawn_window_seconds
            # Count spawns in window
            count = sum(1 for t in self._spawns if t >= cutoff)
            return count

    def get_delay(self) -> float:
        """Get recommended delay based on current rate."""
        rate = self.get_rate()
        ratio = rate / self.config.max_spawns_per_window

        if ratio >= 1.0:
            return self.config.critical_delay
        elif ratio >= self.config.backpressure_threshold:
            # Linear interpolation between warning and critical
            factor = (ratio - self.config.backpressure_threshold) / (1.0 - self.config.backpressure_threshold)
            return self.config.warning_delay + factor * (self.config.critical_delay - self.config.warning_delay)
        else:
            return self.config.normal_delay

    def at_limit(self) -> bool:
        """Check if at or above spawn limit."""
        return self.get_rate() >= self.config.max_spawns_per_window


# ============================================
# Resource Monitor
# ============================================

class ResourceMonitor:
    """
    Monitors local system resources.

    Provides proactive detection of resource exhaustion.
    """

    def __init__(self, config: SafeguardConfig):
        self.config = config
        self._last_check: Dict[str, Any] = {}
        self._last_check_time: float = 0
        self._check_interval: float = 5.0  # Cache for 5 seconds

    def get_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        now = time.time()
        if now - self._last_check_time < self._check_interval and self._last_check:
            return self._last_check

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load = os.getloadavg()
            cpu_count = os.cpu_count() or 1

            self._last_check = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "load_1m": load[0],
                "load_5m": load[1],
                "load_15m": load[2],
                "cpu_count": cpu_count,
                "load_per_cpu": load[0] / cpu_count,
                "timestamp": now,
            }
            self._last_check_time = now

        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            self._last_check = {"error": str(e), "timestamp": now}
            self._last_check_time = now

        return self._last_check

    def is_critical(self) -> Tuple[bool, str]:
        """Check if resources are at critical levels."""
        resources = self.get_resources()

        if "error" in resources:
            return False, ""

        if resources["disk_percent"] >= self.config.disk_critical_percent:
            return True, f"Disk critical: {resources['disk_percent']:.1f}%"

        if resources["memory_percent"] >= self.config.memory_critical_percent:
            return True, f"Memory critical: {resources['memory_percent']:.1f}%"

        if resources["cpu_percent"] >= self.config.cpu_critical_percent:
            return True, f"CPU critical: {resources['cpu_percent']:.1f}%"

        if resources["load_per_cpu"] >= self.config.load_critical_multiplier:
            return True, f"Load critical: {resources['load_1m']:.1f} ({resources['load_per_cpu']:.1f}x)"

        return False, ""

    def is_warning(self) -> Tuple[bool, str]:
        """Check if resources are at warning levels."""
        resources = self.get_resources()

        if "error" in resources:
            return False, ""

        if resources["disk_percent"] >= self.config.disk_warning_percent:
            return True, f"Disk warning: {resources['disk_percent']:.1f}%"

        if resources["memory_percent"] >= self.config.memory_warning_percent:
            return True, f"Memory warning: {resources['memory_percent']:.1f}%"

        return False, ""


# ============================================
# Safeguards Manager
# ============================================

class Safeguards:
    """
    Main safeguards manager singleton.

    Coordinates all safeguard mechanisms.
    """

    _instance: Optional['Safeguards'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'Safeguards':
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, config: Optional[SafeguardConfig] = None):
        self.config = config or SafeguardConfig()

        # Circuit breakers by target (node_id or task_type)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._cb_lock = threading.Lock()

        # Spawn rate tracking
        self._global_tracker = SpawnRateTracker(self.config)
        self._node_trackers: Dict[str, SpawnRateTracker] = {}

        # Resource monitor
        self._resource_monitor = ResourceMonitor(self.config)

        # Task counts
        self._task_counts: Dict[str, Dict[str, int]] = {}  # node_id -> {task_type: count}
        self._counts_lock = threading.Lock()

        # Block reason (for debugging)
        self._last_block_reason: str = ""

        # Emergency state
        self._emergency_active: bool = False

        logger.info("Safeguards initialized")

    # ==========================================
    # Main API
    # ==========================================

    def allow_spawn(
        self,
        task_type: str,
        node_id: str,
        check_resources: bool = True
    ) -> bool:
        """
        Check if spawning is allowed.

        Returns True if spawn is allowed, False otherwise.
        Use get_block_reason() to get the reason for blocking.
        """
        self._last_block_reason = ""

        # Check emergency halt
        if self.is_emergency_active():
            self._last_block_reason = "Emergency halt active"
            return False

        # Check circuit breakers
        cb = self._get_circuit_breaker(node_id)
        if not cb.allow_call():
            self._last_block_reason = f"Circuit breaker open for {node_id}"
            return False

        cb_type = self._get_circuit_breaker(f"type:{task_type}")
        if not cb_type.allow_call():
            self._last_block_reason = f"Circuit breaker open for {task_type}"
            return False

        # Check rate limits
        if self._global_tracker.at_limit():
            self._last_block_reason = "Global spawn rate limit"
            return False

        node_tracker = self._get_node_tracker(node_id)
        if node_tracker.at_limit():
            self._last_block_reason = f"Node {node_id} spawn rate limit"
            return False

        # Check task count limits
        if not self._check_task_limits(task_type, node_id):
            return False  # Block reason set by _check_task_limits

        # Check resources
        if check_resources:
            critical, reason = self._resource_monitor.is_critical()
            if critical:
                self._last_block_reason = reason
                return False

        return True

    def record_spawn(self, task_type: str, node_id: str, pid: int = 0) -> None:
        """Record a successful spawn."""
        # Update trackers
        self._global_tracker.record_spawn()
        self._get_node_tracker(node_id).record_spawn()

        # Update circuit breakers
        self._get_circuit_breaker(node_id).record_success()
        self._get_circuit_breaker(f"type:{task_type}").record_success()

        # Update task counts
        with self._counts_lock:
            if node_id not in self._task_counts:
                self._task_counts[node_id] = {}
            self._task_counts[node_id][task_type] = self._task_counts[node_id].get(task_type, 0) + 1

        logger.debug(f"Recorded spawn: {task_type} on {node_id}")

    def record_completion(self, task_type: str, node_id: str) -> None:
        """Record task completion."""
        with self._counts_lock:
            if node_id in self._task_counts and task_type in self._task_counts[node_id]:
                self._task_counts[node_id][task_type] = max(
                    0, self._task_counts[node_id][task_type] - 1
                )

    def record_failure(self, task_type: str, node_id: str, reason: str = "") -> None:
        """Record a spawn failure."""
        self._get_circuit_breaker(node_id).record_failure(reason)
        self._get_circuit_breaker(f"type:{task_type}").record_failure(reason)

        logger.warning(f"Spawn failure: {task_type} on {node_id}: {reason}")

        # Auto emergency on OOM
        if self.config.auto_emergency_on_oom and "oom" in reason.lower():
            logger.error("OOM detected - activating emergency halt")
            self.activate_emergency()

    def get_delay(self) -> float:
        """Get recommended delay before next spawn."""
        delay = self._global_tracker.get_delay()

        # Add resource-based delay
        warning, _ = self._resource_monitor.is_warning()
        if warning:
            delay = max(delay, self.config.warning_delay)

        return delay

    def get_block_reason(self) -> str:
        """Get the reason for the last block."""
        return self._last_block_reason

    # ==========================================
    # Emergency Controls
    # ==========================================

    def activate_emergency(self) -> None:
        """Activate emergency halt."""
        self._emergency_active = True

        # Write emergency file
        emergency_file = Path(self.config.emergency_file)
        emergency_file.parent.mkdir(parents=True, exist_ok=True)
        emergency_file.write_text(json.dumps({
            "activated_at": datetime.now().isoformat(),
            "reason": "safeguards_triggered",
        }))

        logger.error("EMERGENCY HALT ACTIVATED")

    def deactivate_emergency(self) -> None:
        """Deactivate emergency halt."""
        self._emergency_active = False

        emergency_file = Path(self.config.emergency_file)
        if emergency_file.exists():
            emergency_file.unlink()

        logger.info("Emergency halt deactivated")

    def is_emergency_active(self) -> bool:
        """Check if emergency halt is active."""
        if self._emergency_active:
            return True

        # Also check file
        return Path(self.config.emergency_file).exists()

    # ==========================================
    # Statistics
    # ==========================================

    def get_stats(self) -> Dict[str, Any]:
        """Get safeguard statistics."""
        resources = self._resource_monitor.get_resources()

        with self._counts_lock:
            task_counts = dict(self._task_counts)

        circuit_states = {}
        with self._cb_lock:
            for name, cb in self._circuit_breakers.items():
                circuit_states[name] = {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                }

        return {
            "emergency_active": self.is_emergency_active(),
            "global_spawn_rate": self._global_tracker.get_rate(),
            "recommended_delay": self.get_delay(),
            "resources": resources,
            "task_counts": task_counts,
            "circuit_breakers": circuit_states,
            "last_block_reason": self._last_block_reason,
        }

    def get_task_count(self, node_id: str, task_type: Optional[str] = None) -> int:
        """Get task count for a node."""
        with self._counts_lock:
            if node_id not in self._task_counts:
                return 0
            if task_type:
                return self._task_counts[node_id].get(task_type, 0)
            return sum(self._task_counts[node_id].values())

    def sync_task_counts(self, counts: Dict[str, Dict[str, int]]) -> None:
        """Sync task counts from external source (e.g., P2P orchestrator)."""
        with self._counts_lock:
            self._task_counts = counts

    # ==========================================
    # Internal Methods
    # ==========================================

    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self._cb_lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, config=self.config)
            return self._circuit_breakers[name]

    def _get_node_tracker(self, node_id: str) -> SpawnRateTracker:
        """Get or create a spawn rate tracker for a node."""
        if node_id not in self._node_trackers:
            self._node_trackers[node_id] = SpawnRateTracker(self.config)
        return self._node_trackers[node_id]

    def _check_task_limits(self, task_type: str, node_id: str) -> bool:
        """Check if task counts are within limits."""
        with self._counts_lock:
            # Per-node selfplay limit
            if task_type in ("selfplay", "gpu_selfplay", "hybrid_selfplay"):
                node_selfplay = 0
                if node_id in self._task_counts:
                    for t in ("selfplay", "gpu_selfplay", "hybrid_selfplay"):
                        node_selfplay += self._task_counts[node_id].get(t, 0)

                if node_selfplay >= self.config.max_selfplay_per_node:
                    self._last_block_reason = f"Node {node_id} at selfplay limit ({node_selfplay})"
                    return False

                # Total selfplay limit
                total_selfplay = 0
                for node_counts in self._task_counts.values():
                    for t in ("selfplay", "gpu_selfplay", "hybrid_selfplay"):
                        total_selfplay += node_counts.get(t, 0)

                if total_selfplay >= self.config.max_total_selfplay:
                    self._last_block_reason = f"Total selfplay limit reached ({total_selfplay})"
                    return False

        return True


# ============================================
# Integration Helpers
# ============================================

def patch_p2p_start_job():
    """
    Patch function to integrate safeguards with P2P orchestrator.

    Call this before starting the P2P orchestrator to add safeguards.
    """
    safeguards = Safeguards.get_instance()

    def wrapped_start_job(original_func):
        async def wrapper(self, *args, **kwargs):
            # Extract job info
            job_type = kwargs.get("job_type", args[0] if args else "selfplay")
            node_id = getattr(self, "node_id", "unknown")

            # Check safeguards
            if not safeguards.allow_spawn(str(job_type), node_id):
                logger.warning(
                    f"Job blocked by safeguards: {job_type} on {node_id}: "
                    f"{safeguards.get_block_reason()}"
                )
                return None

            # Apply backpressure delay
            delay = safeguards.get_delay()
            if delay > 0:
                await asyncio.sleep(delay)

            # Call original
            result = await original_func(self, *args, **kwargs)

            # Record result
            if result:
                safeguards.record_spawn(str(job_type), node_id)
            else:
                safeguards.record_failure(str(job_type), node_id, "start_failed")

            return result

        return wrapper

    return wrapped_start_job


def check_before_spawn(task_type: str, node_id: str) -> Tuple[bool, str]:
    """
    Simple check function for existing code.

    Returns: (allowed, reason_if_blocked)
    """
    safeguards = Safeguards.get_instance()
    if safeguards.allow_spawn(task_type, node_id):
        return True, ""
    return False, safeguards.get_block_reason()


# ============================================
# CLI
# ============================================

def main():
    """CLI for safeguards management."""
    import argparse

    parser = argparse.ArgumentParser(description="Safeguards management")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--emergency", action="store_true", help="Activate emergency")
    parser.add_argument("--clear-emergency", action="store_true", help="Clear emergency")
    parser.add_argument("--test-spawn", type=str, help="Test spawn for node")
    args = parser.parse_args()

    safeguards = Safeguards.get_instance()

    if args.status:
        stats = safeguards.get_stats()
        print(json.dumps(stats, indent=2, default=str))

    elif args.emergency:
        safeguards.activate_emergency()
        print("Emergency halt activated")

    elif args.clear_emergency:
        safeguards.deactivate_emergency()
        print("Emergency halt cleared")

    elif args.test_spawn:
        node = args.test_spawn
        allowed = safeguards.allow_spawn("selfplay", node)
        print(f"Spawn allowed: {allowed}")
        if not allowed:
            print(f"Reason: {safeguards.get_block_reason()}")

    else:
        parser.print_help()


if __name__ == "__main__":
    import asyncio
    main()
