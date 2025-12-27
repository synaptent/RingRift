"""Resource Optimizer for RingRift AI Cluster.

This module provides cooperative resource scheduling across orchestrators,
targeting 60-80% CPU and GPU utilization for optimal training throughput.

Key Features:
1. Shared resource state via SQLite coordination database
2. PID controller for adaptive workload adjustment
3. Cross-orchestrator communication and scheduling
4. Prometheus metrics for utilization tracking
5. Active optimization toward target utilization range

Usage:
    from app.coordination.resource_optimizer import (
        ResourceOptimizer,
        get_resource_optimizer,
        should_scale_up,
        should_scale_down,
        get_optimal_concurrency,
    )

    optimizer = get_resource_optimizer()

    # Check if we should adjust workloads
    if optimizer.should_scale_up("gpu"):
        # Increase GPU selfplay jobs
        pass

    # Get optimal job count for a node
    optimal_jobs = optimizer.get_optimal_concurrency(
        node_id="gpu-server-1",
        resource_type="gpu",
        current_util=55.0,
    )
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.config.env import env

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Lazy initialization of targets to avoid circular import with resource_targets
# December 2025: Moved from module-level import to lazy accessor pattern
_cached_targets: dict[str, float] | None = None


def _get_targets() -> dict[str, float]:
    """Lazy load resource targets to avoid circular import."""
    global _cached_targets
    if _cached_targets is not None:
        return _cached_targets

    try:
        from app.coordination.resource_targets import get_resource_targets
        _targets = get_resource_targets()
        _cached_targets = {
            "util_min": _targets.cpu_min,
            "util_max": _targets.cpu_max,
            "util_optimal": _targets.cpu_target,
            "scale_up": _targets.cpu_min - 5,
            "scale_down": _targets.cpu_max + 5,
        }
    except (ImportError, AttributeError) as e:
        # Fallback to centralized env config (Dec 2025: Added AttributeError handling)
        logger.debug(f"Using env fallback for resource targets: {e}")
        _cached_targets = {
            "util_min": env.target_util_min,
            "util_max": env.target_util_max,
            "util_optimal": (env.target_util_min + env.target_util_max) / 2,
            "scale_up": env.scale_up_threshold,
            "scale_down": env.scale_down_threshold,
        }
    return _cached_targets


# Module-level constants use env defaults; actual values come from _get_targets()
TARGET_UTIL_MIN = env.target_util_min  # 60%
TARGET_UTIL_MAX = env.target_util_max  # 80%
TARGET_UTIL_OPTIMAL = (TARGET_UTIL_MIN + TARGET_UTIL_MAX) / 2  # 70%
SCALE_UP_THRESHOLD = env.scale_up_threshold  # 55%
SCALE_DOWN_THRESHOLD = env.scale_down_threshold  # 85%

# PID controller parameters - use centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import PIDDefaults, UtilizationDefaults
    PID_KP = PIDDefaults.KP
    PID_KI = PIDDefaults.KI
    PID_KD = PIDDefaults.KD
    UTILIZATION_UPDATE_INTERVAL = UtilizationDefaults.UPDATE_INTERVAL
    OPTIMIZATION_INTERVAL = UtilizationDefaults.OPTIMIZATION_INTERVAL
except ImportError:
    # Fallback to centralized env config
    PID_KP = env.pid_kp
    PID_KI = env.pid_ki
    PID_KD = env.pid_kd
    UTILIZATION_UPDATE_INTERVAL = 10  # seconds
    OPTIMIZATION_INTERVAL = 30  # seconds

# Database path
from app.utils.paths import DATA_DIR

COORDINATION_DB_PATH = DATA_DIR / "coordination" / "resource_state.db"


# December 2025: Import ResourceType from canonical source
from app.coordination.types import ResourceType

# ResourceType is now imported from app.coordination.types
# Canonical values: CPU, GPU, MEMORY, DISK, NETWORK, HYBRID, IO


class ScaleAction(str, Enum):
    """Scaling actions."""
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"


@dataclass
class NodeResources:
    """Resource state for a single node."""
    node_id: str
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_count: int = 0
    gpu_count: int = 0  # Number of GPUs on node
    memory_gb: float = 0.0
    has_gpu: bool = False
    gpu_name: str = ""
    active_jobs: int = 0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    updated_at: float = 0.0
    orchestrator: str = ""  # Which orchestrator reported this

    def get_max_gpu_jobs(self) -> int:
        """Get max GPU jobs based on GPU count and type.

        Conservative limits to prevent GPU starvation:
        - 2-4 jobs per high-end GPU (H100, A100)
        - 1-2 jobs per consumer GPU
        """
        if not self.has_gpu or self.gpu_count == 0:
            return 0
        # High-end datacenter GPUs can handle more parallelism
        if any(g in self.gpu_name.upper() for g in ["H100", "H200", "A100", "L40"]):
            jobs_per_gpu = 4
        elif any(g in self.gpu_name.upper() for g in ["A10", "4090", "5090", "3090"]):
            jobs_per_gpu = 3
        else:
            jobs_per_gpu = 2
        return self.gpu_count * jobs_per_gpu

    def get_max_cpu_jobs(self) -> int:
        """Get max CPU jobs based on CPU count.

        Reasonable limits to prevent CPU starvation:
        - Up to 1 job per 2 cores for hybrid selfplay
        - Cap at reasonable total for memory constraints
        """
        if self.cpu_count == 0:
            return 8  # Default fallback
        # ~1 job per 2 cores, capped by memory (assume ~1GB per job)
        cpu_based = max(1, self.cpu_count // 2)
        memory_based = max(1, int(self.memory_gb / 2)) if self.memory_gb > 0 else 32
        return min(cpu_based, memory_based, 48)  # Hard cap at 48

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeResources:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClusterState:
    """Aggregated cluster resource state."""
    nodes: list[NodeResources]
    total_cpu_util: float = 0.0
    total_gpu_util: float = 0.0
    total_memory_util: float = 0.0
    total_gpu_memory_util: float = 0.0  # GPU VRAM utilization
    gpu_node_count: int = 0
    cpu_node_count: int = 0
    total_jobs: int = 0
    updated_at: float = 0.0

    # GPU memory thresholds
    GPU_MEMORY_WARNING: float = 80.0  # Start throttling new GPU work
    GPU_MEMORY_CRITICAL: float = 90.0  # Stop spawning GPU jobs

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from node data."""
        if not self.nodes:
            return

        cpu_utils = [n.cpu_percent for n in self.nodes if n.cpu_percent > 0]
        gpu_utils = [n.gpu_percent for n in self.nodes if n.has_gpu and n.gpu_percent > 0]
        mem_utils = [n.memory_percent for n in self.nodes if n.memory_percent > 0]
        gpu_mem_utils = [n.gpu_memory_percent for n in self.nodes if n.has_gpu and n.gpu_memory_percent > 0]

        self.total_cpu_util = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
        self.total_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        self.total_memory_util = sum(mem_utils) / len(mem_utils) if mem_utils else 0.0
        self.total_gpu_memory_util = sum(gpu_mem_utils) / len(gpu_mem_utils) if gpu_mem_utils else 0.0

        self.gpu_node_count = len([n for n in self.nodes if n.has_gpu])
        self.cpu_node_count = len(self.nodes)
        self.total_jobs = sum(n.active_jobs for n in self.nodes)
        self.updated_at = time.time()

    def is_gpu_memory_constrained(self) -> bool:
        """Check if GPU memory is constrained across the cluster.

        Returns:
            True if GPU memory is above warning threshold on any GPU node
        """
        return any(node.has_gpu and node.gpu_memory_percent > self.GPU_MEMORY_WARNING for node in self.nodes)

    def is_gpu_memory_critical(self) -> bool:
        """Check if GPU memory is critically high.

        Returns:
            True if GPU memory is above critical threshold on any GPU node
        """
        return any(node.has_gpu and node.gpu_memory_percent > self.GPU_MEMORY_CRITICAL for node in self.nodes)

    def get_gpu_memory_status(self) -> str:
        """Get GPU memory status string.

        Returns:
            Status: 'ok', 'warning', or 'critical'
        """
        if self.is_gpu_memory_critical():
            return "critical"
        elif self.is_gpu_memory_constrained():
            return "warning"
        return "ok"


@dataclass
class OptimizationResult:
    """Result of an optimization decision."""
    action: ScaleAction
    resource_type: ResourceType
    current_util: float
    target_util: float
    adjustment: int  # Suggested job count change
    nodes_affected: list[str]
    reason: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "resource_type": self.resource_type.value,
            "current_util": self.current_util,
            "target_util": self.target_util,
            "adjustment": self.adjustment,
            "nodes_affected": self.nodes_affected,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class PIDController:
    """PID controller for smooth utilization targeting.

    Uses proportional, integral, and derivative control to smoothly
    adjust workloads toward the target utilization.

    Supports:
    - Config-driven parameter tuning
    - Gain scheduling (adjust gains based on error magnitude)
    - Output smoothing (reduce sudden changes)
    - Minimum update interval (prevent excessive updates)
    """

    def __init__(
        self,
        kp: float = PID_KP,
        ki: float = PID_KI,
        kd: float = PID_KD,
        setpoint: float = TARGET_UTIL_OPTIMAL,
        integral_clamp: float = 100.0,
        min_update_interval: float = 30.0,
        output_smoothing: float = 0.3,
        gain_scheduling: bool = True,
        large_error_threshold: float = 15.0,
        large_error_gain_multiplier: float = 1.5,
        small_error_threshold: float = 5.0,
        small_error_gain_multiplier: float = 0.7,
    ):
        # Base gains
        self.kp_base = kp
        self.ki_base = ki
        self.kd_base = kd
        self.setpoint = setpoint

        # Current effective gains (may be adjusted by gain scheduling)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Anti-windup
        self.integral_clamp = integral_clamp

        # Update throttling
        self.min_update_interval = min_update_interval

        # Output smoothing
        self.output_smoothing = output_smoothing
        self._prev_output = 0.0

        # Gain scheduling
        self.gain_scheduling = gain_scheduling
        self.large_error_threshold = large_error_threshold
        self.large_error_gain_multiplier = large_error_gain_multiplier
        self.small_error_threshold = small_error_threshold
        self.small_error_gain_multiplier = small_error_gain_multiplier

        # Internal state
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_update = 0.0

    @classmethod
    def from_config(cls, config: dict, setpoint: float = TARGET_UTIL_OPTIMAL) -> PIDController:
        """Create a PIDController from config dictionary.

        Args:
            config: PID config dict (from unified_loop.yaml resource_targets.pid)
            setpoint: Target utilization percentage

        Returns:
            Configured PIDController instance
        """
        return cls(
            kp=config.get("kp", PID_KP),
            ki=config.get("ki", PID_KI),
            kd=config.get("kd", PID_KD),
            setpoint=setpoint,
            integral_clamp=config.get("integral_clamp", 100.0),
            min_update_interval=config.get("min_update_interval", 30.0),
            output_smoothing=config.get("output_smoothing", 0.3),
            gain_scheduling=config.get("gain_scheduling", True),
            large_error_threshold=config.get("large_error_threshold", 15.0),
            large_error_gain_multiplier=config.get("large_error_gain_multiplier", 1.5),
            small_error_threshold=config.get("small_error_threshold", 5.0),
            small_error_gain_multiplier=config.get("small_error_gain_multiplier", 0.7),
        )

    def _apply_gain_scheduling(self, error_magnitude: float) -> None:
        """Adjust gains based on error magnitude.

        Large errors get higher gains for faster response.
        Small errors get lower gains for stability.
        """
        if not self.gain_scheduling:
            return

        if error_magnitude > self.large_error_threshold:
            # Large error: increase gains for faster response
            multiplier = self.large_error_gain_multiplier
        elif error_magnitude < self.small_error_threshold:
            # Small error: reduce gains for stability
            multiplier = self.small_error_gain_multiplier
        else:
            # Normal range: use base gains
            multiplier = 1.0

        self.kp = self.kp_base * multiplier
        self.ki = self.ki_base * multiplier
        self.kd = self.kd_base * multiplier

    def update(self, current_value: float, dt: float | None = None) -> float:
        """Calculate PID output for current utilization.

        Args:
            current_value: Current utilization percentage
            dt: Time delta since last update (auto-computed if None)

        Returns:
            Control output (positive = need more work, negative = reduce)
        """
        now = time.time()

        # Throttle updates
        if self._last_update > 0:
            elapsed = now - self._last_update
            if elapsed < self.min_update_interval:
                return self._prev_output

        if dt is None:
            dt = max(0.1, now - self._last_update) if self._last_update > 0 else 1.0
        self._last_update = now

        # Error: how far from target
        error = self.setpoint - current_value
        error_magnitude = abs(error)

        # Apply gain scheduling based on error magnitude
        self._apply_gain_scheduling(error_magnitude)

        # Proportional term
        p_term = self.kp * error

        # Integral term (anti-windup: clamp to prevent runaway)
        self._integral += error * dt
        self._integral = max(-self.integral_clamp, min(self.integral_clamp, self._integral))
        i_term = self.ki * self._integral

        # Derivative term
        d_term = self.kd * (error - self._prev_error) / dt if dt > 0 else 0
        self._prev_error = error

        # Raw output
        raw_output = p_term + i_term + d_term

        # Apply output smoothing (exponential moving average)
        if self.output_smoothing > 0:
            smoothed_output = (
                self.output_smoothing * self._prev_output +
                (1 - self.output_smoothing) * raw_output
            )
        else:
            smoothed_output = raw_output

        self._prev_output = smoothed_output
        return smoothed_output

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_update = 0.0
        self._prev_output = 0.0
        # Reset gains to base values
        self.kp = self.kp_base
        self.ki = self.ki_base
        self.kd = self.kd_base

    def get_state(self) -> dict:
        """Get current controller state for monitoring.

        Returns:
            Dictionary with controller state
        """
        return {
            "kp_effective": self.kp,
            "ki_effective": self.ki,
            "kd_effective": self.kd,
            "integral": self._integral,
            "prev_error": self._prev_error,
            "prev_output": self._prev_output,
            "setpoint": self.setpoint,
        }


class UtilizationPredictor:
    """Predictive scaling based on utilization trends.

    Uses exponential smoothing and linear regression to predict
    future utilization and proactively adjust job rates.

    Features:
    - Historical utilization buffer (configurable window)
    - Exponential moving average for smoothing
    - Linear regression for trend prediction
    - Confidence-weighted predictions
    """

    def __init__(
        self,
        history_window_seconds: float = 600.0,  # 10 minutes of history
        prediction_horizon_seconds: float = 120.0,  # Predict 2 minutes ahead
        ema_alpha: float = 0.2,  # Smoothing factor for EMA
        min_samples_for_prediction: int = 10,  # Min data points for prediction
    ):
        self.history_window_seconds = history_window_seconds
        self.prediction_horizon_seconds = prediction_horizon_seconds
        self.ema_alpha = ema_alpha
        self.min_samples_for_prediction = min_samples_for_prediction

        # Historical data: list of (timestamp, cpu_util, gpu_util, gpu_mem_util)
        self._history: list[tuple[float, float, float, float]] = []
        self._lock = threading.RLock()

        # EMA values
        self._ema_cpu: float | None = None
        self._ema_gpu: float | None = None
        self._ema_gpu_mem: float | None = None

    def record_sample(
        self,
        cpu_util: float,
        gpu_util: float,
        gpu_mem_util: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        """Record a utilization sample.

        Args:
            cpu_util: CPU utilization percentage (0-100)
            gpu_util: GPU utilization percentage (0-100)
            gpu_mem_util: GPU memory utilization percentage (0-100)
            timestamp: Sample timestamp (defaults to now)
        """
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            # Add to history
            self._history.append((ts, cpu_util, gpu_util, gpu_mem_util))

            # Update EMA
            if self._ema_cpu is None:
                self._ema_cpu = cpu_util
                self._ema_gpu = gpu_util
                self._ema_gpu_mem = gpu_mem_util
            else:
                self._ema_cpu = self.ema_alpha * cpu_util + (1 - self.ema_alpha) * self._ema_cpu
                self._ema_gpu = self.ema_alpha * gpu_util + (1 - self.ema_alpha) * self._ema_gpu
                self._ema_gpu_mem = self.ema_alpha * gpu_mem_util + (1 - self.ema_alpha) * self._ema_gpu_mem

            # Prune old samples
            cutoff = ts - self.history_window_seconds
            self._history = [(t, c, g, m) for t, c, g, m in self._history if t > cutoff]

    def _calculate_trend(self, data: list[tuple[float, float]]) -> tuple[float, float]:
        """Calculate linear trend using least squares regression.

        Args:
            data: List of (timestamp, value) pairs

        Returns:
            Tuple of (slope, intercept) for the trend line
        """
        if len(data) < 2:
            return 0.0, data[0][1] if data else 0.0

        n = len(data)
        sum_t = sum(t for t, _ in data)
        sum_v = sum(v for _, v in data)
        sum_tv = sum(t * v for t, v in data)
        sum_t2 = sum(t * t for t, _ in data)

        # Least squares
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-9:
            return 0.0, sum_v / n

        slope = (n * sum_tv - sum_t * sum_v) / denom
        intercept = (sum_v - slope * sum_t) / n

        return slope, intercept

    def predict(self) -> dict[str, Any] | None:
        """Predict future utilization based on historical trends.

        Returns:
            Prediction dictionary with expected utilization and confidence,
            or None if insufficient data
        """
        with self._lock:
            if len(self._history) < self.min_samples_for_prediction:
                return None

            now = time.time()
            future_ts = now + self.prediction_horizon_seconds

            # Extract time series for each metric
            cpu_data = [(t, c) for t, c, _, _ in self._history]
            gpu_data = [(t, g) for t, _, g, _ in self._history]
            gpu_mem_data = [(t, m) for t, _, _, m in self._history]

            # Calculate trends
            cpu_slope, cpu_intercept = self._calculate_trend(cpu_data)
            gpu_slope, gpu_intercept = self._calculate_trend(gpu_data)
            gpu_mem_slope, gpu_mem_intercept = self._calculate_trend(gpu_mem_data)

            # Predict future values
            predicted_cpu = cpu_intercept + cpu_slope * future_ts
            predicted_gpu = gpu_intercept + gpu_slope * future_ts
            predicted_gpu_mem = gpu_mem_intercept + gpu_mem_slope * future_ts

            # Clamp predictions to valid range
            predicted_cpu = max(0.0, min(100.0, predicted_cpu))
            predicted_gpu = max(0.0, min(100.0, predicted_gpu))
            predicted_gpu_mem = max(0.0, min(100.0, predicted_gpu_mem))

            # Calculate confidence based on data consistency
            # More samples and lower variance = higher confidence
            sample_count = len(self._history)
            max_samples = int(self.history_window_seconds / UTILIZATION_UPDATE_INTERVAL)
            sample_confidence = min(1.0, sample_count / max_samples)

            # Trend stability (lower slope variance = higher confidence)
            cpu_variance = sum((v - (cpu_intercept + cpu_slope * t)) ** 2 for t, v in cpu_data) / sample_count
            trend_stability = max(0.1, 1.0 - min(1.0, cpu_variance / 100.0))

            confidence = sample_confidence * trend_stability

            return {
                "timestamp": now,
                "prediction_horizon_seconds": self.prediction_horizon_seconds,
                "predicted_cpu": predicted_cpu,
                "predicted_gpu": predicted_gpu,
                "predicted_gpu_mem": predicted_gpu_mem,
                "cpu_trend": "rising" if cpu_slope > 0.5 else "falling" if cpu_slope < -0.5 else "stable",
                "gpu_trend": "rising" if gpu_slope > 0.5 else "falling" if gpu_slope < -0.5 else "stable",
                "cpu_slope_per_min": cpu_slope * 60,  # % change per minute
                "gpu_slope_per_min": gpu_slope * 60,
                "confidence": confidence,
                "samples_used": sample_count,
                "ema_cpu": self._ema_cpu,
                "ema_gpu": self._ema_gpu,
                "ema_gpu_mem": self._ema_gpu_mem,
            }

    def get_proactive_adjustment(self) -> dict[str, Any] | None:
        """Get proactive scaling recommendation based on predictions.

        Returns:
            Recommendation dictionary or None if no action needed
        """
        prediction = self.predict()
        if prediction is None:
            return None

        # Only act on high-confidence predictions
        if prediction["confidence"] < 0.5:
            return None

        action = None
        reason = None

        # Check if we're heading toward underutilization
        if prediction["predicted_cpu"] < TARGET_UTIL_MIN - 10 or \
           prediction["predicted_gpu"] < TARGET_UTIL_MIN - 10:
            # Utilization trending down toward underutilization
            action = "scale_up"
            reason = f"Predicted underutilization: CPU={prediction['predicted_cpu']:.1f}%, GPU={prediction['predicted_gpu']:.1f}%"

        # Check if we're heading toward overutilization
        elif prediction["predicted_cpu"] > TARGET_UTIL_MAX + 10 or \
             prediction["predicted_gpu"] > TARGET_UTIL_MAX + 10:
            action = "scale_down"
            reason = f"Predicted overutilization: CPU={prediction['predicted_cpu']:.1f}%, GPU={prediction['predicted_gpu']:.1f}%"

        # Check GPU memory trend
        elif prediction["predicted_gpu_mem"] > ClusterState.GPU_MEMORY_WARNING:
            action = "scale_down"
            reason = f"Predicted GPU memory pressure: {prediction['predicted_gpu_mem']:.1f}%"

        if action is None:
            return None

        # Calculate suggested rate multiplier
        if action == "scale_up":
            # How far below target are we heading?
            gap = min(
                TARGET_UTIL_OPTIMAL - prediction["predicted_cpu"],
                TARGET_UTIL_OPTIMAL - prediction["predicted_gpu"]
            )
            multiplier = 1.0 + min(0.3, gap / 50.0)  # Max 30% increase
        else:
            # How far above target are we heading?
            gap = max(
                prediction["predicted_cpu"] - TARGET_UTIL_OPTIMAL,
                prediction["predicted_gpu"] - TARGET_UTIL_OPTIMAL,
                prediction["predicted_gpu_mem"] - ClusterState.GPU_MEMORY_WARNING
            )
            multiplier = max(0.7, 1.0 - gap / 50.0)  # Max 30% decrease

        return {
            "action": action,
            "reason": reason,
            "rate_multiplier": multiplier,
            "confidence": prediction["confidence"],
            "prediction": prediction,
        }

    def clear(self) -> None:
        """Clear history and reset state."""
        with self._lock:
            self._history.clear()
            self._ema_cpu = None
            self._ema_gpu = None
            self._ema_gpu_mem = None


class ResourceOptimizer:
    """Cooperative resource optimizer for RingRift cluster.

    This class provides:
    1. Shared resource state across orchestrators (via SQLite)
    2. PID-controlled workload adjustment
    3. Scaling recommendations to target 60-80% utilization
    4. Metrics for Prometheus integration
    """

    _instance: ResourceOptimizer | None = None
    _lock = threading.RLock()

    def __new__(cls) -> ResourceOptimizer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._db_path = COORDINATION_DB_PATH
        self._db_lock = threading.RLock()

        # Load PID config from unified_loop.yaml if available
        pid_config = self._load_pid_config()

        # PID controllers for each resource type
        if pid_config:
            self._pid_cpu = PIDController.from_config(pid_config, setpoint=TARGET_UTIL_OPTIMAL)
            self._pid_gpu = PIDController.from_config(pid_config, setpoint=TARGET_UTIL_OPTIMAL)
        else:
            self._pid_cpu = PIDController(setpoint=TARGET_UTIL_OPTIMAL)
            self._pid_gpu = PIDController(setpoint=TARGET_UTIL_OPTIMAL)

        # Local node ID (for identifying this orchestrator's reports)
        self._node_id = env.node_id

        # Orchestrator identifier
        self._orchestrator_id = env.orchestrator_id

        # Cached state
        self._cached_cluster_state: ClusterState | None = None
        self._cache_updated_at = 0.0

        # Predictive scaling
        self._predictor = UtilizationPredictor()
        self._predictive_scaling_enabled = True

        # Initialize database
        self._init_db()

        logger.info(
            f"ResourceOptimizer initialized: node={self._node_id}, "
            f"target={TARGET_UTIL_MIN}-{TARGET_UTIL_MAX}%"
        )

    def _load_pid_config(self) -> dict | None:
        """Load PID controller config from unified_loop.yaml.

        Returns:
            PID config dict or None if not found
        """
        try:
            import yaml
        except ImportError as e:
            logger.warning(f"Failed to import yaml module: {e}")
            return None

        try:
            # Try to find unified_loop.yaml
            config_paths = [
                Path(__file__).parent.parent.parent / "config" / "unified_loop.yaml",
                Path("config/unified_loop.yaml"),
                Path("/etc/ringrift/unified_loop.yaml"),
            ]

            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    pid_config = config.get("resource_targets", {}).get("pid", {})
                    if pid_config:
                        logger.info(f"Loaded PID config from {config_path}: kp={pid_config.get('kp')}, "
                                   f"ki={pid_config.get('ki')}, kd={pid_config.get('kd')}")
                        return pid_config

            logger.debug("No PID config found, using defaults")
            return None

        except (IOError, OSError, yaml.YAMLError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load PID config: {e}")
            return None

    def _init_db(self) -> None:
        """Initialize the coordination database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS node_resources (
                    node_id TEXT PRIMARY KEY,
                    cpu_percent REAL DEFAULT 0,
                    gpu_percent REAL DEFAULT 0,
                    memory_percent REAL DEFAULT 0,
                    disk_percent REAL DEFAULT 0,
                    gpu_memory_percent REAL DEFAULT 0,
                    cpu_count INTEGER DEFAULT 0,
                    gpu_count INTEGER DEFAULT 0,
                    memory_gb REAL DEFAULT 0,
                    has_gpu INTEGER DEFAULT 0,
                    gpu_name TEXT DEFAULT '',
                    active_jobs INTEGER DEFAULT 0,
                    selfplay_jobs INTEGER DEFAULT 0,
                    training_jobs INTEGER DEFAULT 0,
                    orchestrator TEXT DEFAULT '',
                    updated_at REAL DEFAULT 0
                );
            """)

            # Migration: Add gpu_count column if it doesn't exist (for older databases)
            try:
                conn.execute("SELECT gpu_count FROM node_resources LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE node_resources ADD COLUMN gpu_count INTEGER DEFAULT 0")

            conn.executescript("""

                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    current_util REAL NOT NULL,
                    target_util REAL NOT NULL,
                    adjustment INTEGER NOT NULL,
                    nodes_affected TEXT NOT NULL,
                    reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS utilization_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    node_id TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    gpu_percent REAL,
                    memory_percent REAL
                );

                CREATE INDEX IF NOT EXISTS idx_util_timestamp
                    ON utilization_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_util_node
                    ON utilization_metrics(node_id, timestamp);

                -- Rate negotiation between orchestrators
                CREATE TABLE IF NOT EXISTS selfplay_rate (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_rate INTEGER NOT NULL DEFAULT 1000,
                    min_rate INTEGER NOT NULL DEFAULT 100,
                    max_rate INTEGER NOT NULL DEFAULT 5000,
                    updated_at REAL NOT NULL,
                    updated_by TEXT
                );

                CREATE TABLE IF NOT EXISTS rate_negotiations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    requestor TEXT NOT NULL,
                    requested_rate INTEGER NOT NULL,
                    reason TEXT,
                    approved_rate INTEGER NOT NULL,
                    current_utilization REAL,
                    response TEXT
                );

                -- Config weighting for selfplay distribution
                CREATE TABLE IF NOT EXISTS config_weights (
                    config_key TEXT PRIMARY KEY,
                    weight REAL NOT NULL DEFAULT 1.0,
                    game_count INTEGER DEFAULT 0,
                    games_per_hour REAL DEFAULT 0.0,
                    reason TEXT,
                    updated_at REAL NOT NULL
                );

                -- Insert default rate if not exists
                INSERT OR IGNORE INTO selfplay_rate (id, current_rate, min_rate, max_rate, updated_at, updated_by)
                VALUES (1, 1000, 100, 5000, 0, 'init');
            """)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # Resource State Reporting
    # =========================================================================

    def report_node_resources(self, resources: NodeResources) -> None:
        """Report resource state for a node.

        Called by orchestrators to share their node's resource state.

        Args:
            resources: Current resource state for the node
        """
        resources.updated_at = time.time()
        resources.orchestrator = self._orchestrator_id

        with self._db_lock, self._get_connection() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO node_resources (
                        node_id, cpu_percent, gpu_percent, memory_percent,
                        disk_percent, gpu_memory_percent, cpu_count, gpu_count, memory_gb,
                        has_gpu, gpu_name, active_jobs, selfplay_jobs,
                        training_jobs, orchestrator, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                resources.node_id, resources.cpu_percent, resources.gpu_percent,
                resources.memory_percent, resources.disk_percent,
                resources.gpu_memory_percent, resources.cpu_count, resources.gpu_count,
                resources.memory_gb, int(resources.has_gpu), resources.gpu_name,
                resources.active_jobs, resources.selfplay_jobs,
                resources.training_jobs, resources.orchestrator,
                resources.updated_at,
            ))

            # Also record in metrics history
            conn.execute("""
                    INSERT INTO utilization_metrics
                        (timestamp, node_id, cpu_percent, gpu_percent, memory_percent)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                resources.updated_at, resources.node_id, resources.cpu_percent,
                resources.gpu_percent if resources.has_gpu else None,
                resources.memory_percent,
            ))
            conn.commit()

        # Invalidate cache
        self._cache_updated_at = 0.0

    def record_utilization(
        self,
        node_id: str,
        cpu_percent: float,
        gpu_percent: float | None = None,
        memory_percent: float | None = None,
    ) -> None:
        """Record a utilization sample for metrics tracking.

        Args:
            node_id: Node identifier
            cpu_percent: CPU utilization percentage
            gpu_percent: GPU utilization percentage (if applicable)
            memory_percent: Memory utilization percentage
        """
        now = time.time()
        with self._db_lock, self._get_connection() as conn:
            conn.execute("""
                    INSERT INTO utilization_metrics
                        (timestamp, node_id, cpu_percent, gpu_percent, memory_percent)
                    VALUES (?, ?, ?, ?, ?)
                """, (now, node_id, cpu_percent, gpu_percent, memory_percent))
            conn.commit()

    # =========================================================================
    # Cluster State Queries
    # =========================================================================

    def get_cluster_state(self, max_age_seconds: float = 60) -> ClusterState:
        """Get current cluster resource state.

        Args:
            max_age_seconds: Maximum age for cached state

        Returns:
            ClusterState with all node resources
        """
        now = time.time()

        # Return cached if fresh
        if (self._cached_cluster_state is not None and
            now - self._cache_updated_at < max_age_seconds):
            return self._cached_cluster_state

        # Query all nodes (exclude stale nodes older than 5 minutes)
        cutoff = now - 300
        nodes = []

        with self._db_lock, self._get_connection() as conn:
            rows = conn.execute("""
                    SELECT * FROM node_resources WHERE updated_at > ?
                """, (cutoff,)).fetchall()

            for row in rows:
                node = NodeResources(
                    node_id=row["node_id"],
                    cpu_percent=row["cpu_percent"],
                    gpu_percent=row["gpu_percent"],
                    memory_percent=row["memory_percent"],
                    disk_percent=row["disk_percent"],
                    gpu_memory_percent=row["gpu_memory_percent"],
                    cpu_count=row["cpu_count"],
                    gpu_count=row["gpu_count"],
                    memory_gb=row["memory_gb"],
                    has_gpu=bool(row["has_gpu"]),
                    gpu_name=row["gpu_name"],
                    active_jobs=row["active_jobs"],
                    selfplay_jobs=row["selfplay_jobs"],
                    training_jobs=row["training_jobs"],
                    orchestrator=row["orchestrator"],
                    updated_at=row["updated_at"],
                )
                nodes.append(node)

        state = ClusterState(nodes=nodes)
        state.compute_aggregates()

        # Record sample for predictive scaling
        if self._predictive_scaling_enabled and state.cpu_node_count > 0:
            self._predictor.record_sample(
                cpu_util=state.total_cpu_util,
                gpu_util=state.total_gpu_util,
                gpu_mem_util=state.total_gpu_memory_util,
                timestamp=now,
            )

        self._cached_cluster_state = state
        self._cache_updated_at = now

        return state

    def get_node_resources(self, node_id: str) -> NodeResources | None:
        """Get resources for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            NodeResources or None if not found
        """
        with self._db_lock, self._get_connection() as conn:
            row = conn.execute("""
                    SELECT * FROM node_resources WHERE node_id = ?
                """, (node_id,)).fetchone()

            if row:
                return NodeResources(
                    node_id=row["node_id"],
                    cpu_percent=row["cpu_percent"],
                    gpu_percent=row["gpu_percent"],
                    memory_percent=row["memory_percent"],
                    disk_percent=row["disk_percent"],
                    gpu_memory_percent=row["gpu_memory_percent"],
                    cpu_count=row["cpu_count"],
                    gpu_count=row["gpu_count"],
                    memory_gb=row["memory_gb"],
                    has_gpu=bool(row["has_gpu"]),
                    gpu_name=row["gpu_name"],
                    active_jobs=row["active_jobs"],
                    selfplay_jobs=row["selfplay_jobs"],
                    training_jobs=row["training_jobs"],
                    orchestrator=row["orchestrator"],
                    updated_at=row["updated_at"],
                )
        return None

    # =========================================================================
    # Scaling Decisions
    # =========================================================================

    def should_scale_up(self, resource_type: str = "cpu") -> bool:
        """Check if we should add more workload.

        Returns True if utilization is below target minimum.

        Args:
            resource_type: "cpu" or "gpu"

        Returns:
            True if scaling up is recommended
        """
        state = self.get_cluster_state()

        if resource_type == "gpu":
            return state.total_gpu_util < SCALE_UP_THRESHOLD and state.gpu_node_count > 0
        else:
            return state.total_cpu_util < SCALE_UP_THRESHOLD

    def should_scale_down(self, resource_type: str = "cpu") -> bool:
        """Check if we should reduce workload.

        Returns True if utilization is above target maximum.

        Args:
            resource_type: "cpu" or "gpu"

        Returns:
            True if scaling down is recommended
        """
        state = self.get_cluster_state()

        if resource_type == "gpu":
            return state.total_gpu_util > SCALE_DOWN_THRESHOLD and state.gpu_node_count > 0
        else:
            return state.total_cpu_util > SCALE_DOWN_THRESHOLD

    def get_scale_action(self, resource_type: str = "cpu") -> ScaleAction:
        """Get recommended scaling action.

        Args:
            resource_type: "cpu" or "gpu"

        Returns:
            ScaleAction recommendation
        """
        state = self.get_cluster_state()

        if resource_type == "gpu":
            util = state.total_gpu_util
        else:
            util = state.total_cpu_util

        if util < SCALE_UP_THRESHOLD:
            return ScaleAction.SCALE_UP
        elif util > SCALE_DOWN_THRESHOLD:
            return ScaleAction.SCALE_DOWN
        elif abs(util - TARGET_UTIL_OPTIMAL) > 10:
            return ScaleAction.REBALANCE
        else:
            return ScaleAction.NONE

    # =========================================================================
    # Optimal Concurrency Calculation
    # =========================================================================

    def get_optimal_concurrency(
        self,
        node_id: str,
        resource_type: str = "cpu",
        current_jobs: int = 0,
        current_util: float | None = None,
    ) -> int:
        """Calculate optimal job count for a node.

        Uses PID control to smoothly adjust toward target utilization,
        with hardware-aware limits based on GPU/CPU count.

        Args:
            node_id: Node identifier
            resource_type: "cpu" or "gpu"
            current_jobs: Current number of jobs on node
            current_util: Current utilization (auto-fetched if None)

        Returns:
            Optimal number of concurrent jobs
        """
        # Get node resources for hardware-aware limits
        node = self.get_node_resources(node_id)
        if node is None:
            # No node data - use conservative defaults
            return min(current_jobs, 4 if resource_type == "gpu" else 8)

        # Get current utilization if not provided
        if current_util is None:
            current_util = node.gpu_percent if resource_type == "gpu" else node.cpu_percent

        # Use PID controller to get adjustment
        pid = self._pid_gpu if resource_type == "gpu" else self._pid_cpu
        adjustment = pid.update(current_util)

        # Convert PID output to job count change
        # PID output is roughly "utilization points to add"
        # Scale this to job changes (assume ~5% util per job as baseline)
        util_per_job = 5.0 if resource_type == "gpu" else 3.0
        job_adjustment = int(adjustment / util_per_job)

        # Calculate new job count with bounds
        new_jobs = max(1, current_jobs + job_adjustment)

        # Use hardware-aware limits instead of hardcoded values
        if resource_type == "gpu":
            max_jobs = node.get_max_gpu_jobs() if node.has_gpu else 0
            if max_jobs == 0:
                max_jobs = 4  # Fallback for unknown GPU
        else:
            max_jobs = node.get_max_cpu_jobs()

        new_jobs = min(new_jobs, max_jobs)

        return new_jobs

    def get_optimization_recommendation(self) -> OptimizationResult:
        """Get comprehensive optimization recommendation.

        Returns:
            OptimizationResult with recommended action and affected nodes
        """
        state = self.get_cluster_state()

        # Check GPU utilization first (higher value optimization)
        if state.gpu_node_count > 0:
            gpu_util = state.total_gpu_util
            if gpu_util < SCALE_UP_THRESHOLD:
                # Find underutilized GPU nodes
                underutilized = [
                    n.node_id for n in state.nodes
                    if n.has_gpu and n.gpu_percent < SCALE_UP_THRESHOLD
                ]
                return OptimizationResult(
                    action=ScaleAction.SCALE_UP,
                    resource_type=ResourceType.GPU,
                    current_util=gpu_util,
                    target_util=TARGET_UTIL_OPTIMAL,
                    adjustment=len(underutilized) * 2,  # Add 2 jobs per underutilized GPU node
                    nodes_affected=underutilized,
                    reason=f"GPU utilization {gpu_util:.1f}% below target {TARGET_UTIL_MIN}%",
                    confidence=min(1.0, (TARGET_UTIL_MIN - gpu_util) / 20),
                )
            elif gpu_util > SCALE_DOWN_THRESHOLD:
                overloaded = [
                    n.node_id for n in state.nodes
                    if n.has_gpu and n.gpu_percent > SCALE_DOWN_THRESHOLD
                ]
                return OptimizationResult(
                    action=ScaleAction.SCALE_DOWN,
                    resource_type=ResourceType.GPU,
                    current_util=gpu_util,
                    target_util=TARGET_UTIL_OPTIMAL,
                    adjustment=-len(overloaded),
                    nodes_affected=overloaded,
                    reason=f"GPU utilization {gpu_util:.1f}% above target {TARGET_UTIL_MAX}%",
                    confidence=min(1.0, (gpu_util - TARGET_UTIL_MAX) / 15),
                )

        # Check CPU utilization
        cpu_util = state.total_cpu_util
        if cpu_util < SCALE_UP_THRESHOLD:
            underutilized = [
                n.node_id for n in state.nodes
                if n.cpu_percent < SCALE_UP_THRESHOLD
            ]
            return OptimizationResult(
                action=ScaleAction.SCALE_UP,
                resource_type=ResourceType.CPU,
                current_util=cpu_util,
                target_util=TARGET_UTIL_OPTIMAL,
                adjustment=len(underutilized) * 2,
                nodes_affected=underutilized,
                reason=f"CPU utilization {cpu_util:.1f}% below target {TARGET_UTIL_MIN}%",
                confidence=min(1.0, (TARGET_UTIL_MIN - cpu_util) / 20),
            )
        elif cpu_util > SCALE_DOWN_THRESHOLD:
            overloaded = [
                n.node_id for n in state.nodes
                if n.cpu_percent > SCALE_DOWN_THRESHOLD
            ]
            return OptimizationResult(
                action=ScaleAction.SCALE_DOWN,
                resource_type=ResourceType.CPU,
                current_util=cpu_util,
                target_util=TARGET_UTIL_OPTIMAL,
                adjustment=-len(overloaded),
                nodes_affected=overloaded,
                reason=f"CPU utilization {cpu_util:.1f}% above target {TARGET_UTIL_MAX}%",
                confidence=min(1.0, (cpu_util - TARGET_UTIL_MAX) / 15),
            )

        return OptimizationResult(
            action=ScaleAction.NONE,
            resource_type=ResourceType.CPU,
            current_util=cpu_util,
            target_util=TARGET_UTIL_OPTIMAL,
            adjustment=0,
            nodes_affected=[],
            reason=f"Utilization {cpu_util:.1f}% within target range {TARGET_UTIL_MIN}-{TARGET_UTIL_MAX}%",
            confidence=1.0,
        )

    # =========================================================================
    # Utilization History and Metrics
    # =========================================================================

    def get_utilization_history(
        self,
        node_id: str | None = None,
        hours: float = 1.0,
        resolution_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Get utilization history for metrics/graphing.

        Args:
            node_id: Specific node (None for cluster average)
            hours: Hours of history to fetch
            resolution_seconds: Bucket size for aggregation

        Returns:
            List of utilization samples
        """
        cutoff = time.time() - (hours * 3600)

        with self._db_lock, self._get_connection() as conn:
            if node_id:
                rows = conn.execute("""
                        SELECT
                            CAST(timestamp / ? AS INTEGER) * ? as bucket,
                            AVG(cpu_percent) as cpu,
                            AVG(gpu_percent) as gpu,
                            AVG(memory_percent) as memory
                        FROM utilization_metrics
                        WHERE node_id = ? AND timestamp > ?
                        GROUP BY bucket
                        ORDER BY bucket
                    """, (resolution_seconds, resolution_seconds, node_id, cutoff)).fetchall()
            else:
                rows = conn.execute("""
                        SELECT
                            CAST(timestamp / ? AS INTEGER) * ? as bucket,
                            AVG(cpu_percent) as cpu,
                            AVG(gpu_percent) as gpu,
                            AVG(memory_percent) as memory
                        FROM utilization_metrics
                        WHERE timestamp > ?
                        GROUP BY bucket
                        ORDER BY bucket
                    """, (resolution_seconds, resolution_seconds, cutoff)).fetchall()

        return [
            {
                "timestamp": row["bucket"],
                "cpu_percent": row["cpu"],
                "gpu_percent": row["gpu"],
                "memory_percent": row["memory"],
            }
            for row in rows
        ]

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics suitable for Prometheus exposition.

        Returns:
            Dict with metric names and values
        """
        state = self.get_cluster_state()
        rec = self.get_optimization_recommendation()

        # GPU memory status codes: 0=ok, 1=warning, 2=critical
        gpu_mem_status_codes = {"ok": 0, "warning": 1, "critical": 2}

        return {
            # Cluster-wide utilization
            "ringrift_cluster_cpu_utilization": state.total_cpu_util / 100,
            "ringrift_cluster_gpu_utilization": state.total_gpu_util / 100,
            "ringrift_cluster_memory_utilization": state.total_memory_util / 100,
            "ringrift_cluster_gpu_memory_utilization": state.total_gpu_memory_util / 100,

            # Target range
            "ringrift_target_util_min": TARGET_UTIL_MIN / 100,
            "ringrift_target_util_max": TARGET_UTIL_MAX / 100,
            "ringrift_target_util_optimal": TARGET_UTIL_OPTIMAL / 100,

            # Node counts
            "ringrift_cluster_gpu_nodes": state.gpu_node_count,
            "ringrift_cluster_cpu_nodes": state.cpu_node_count,
            "ringrift_cluster_total_jobs": state.total_jobs,

            # Optimization state
            "ringrift_optimization_action": rec.action.value,
            "ringrift_optimization_adjustment": rec.adjustment,
            "ringrift_optimization_confidence": rec.confidence,

            # Within target range
            "ringrift_cpu_in_target_range": int(
                TARGET_UTIL_MIN <= state.total_cpu_util <= TARGET_UTIL_MAX
            ),
            "ringrift_gpu_in_target_range": int(
                TARGET_UTIL_MIN <= state.total_gpu_util <= TARGET_UTIL_MAX
            ) if state.gpu_node_count > 0 else 1,

            # GPU memory status (0=ok, 1=warning, 2=critical)
            "ringrift_gpu_memory_status": gpu_mem_status_codes.get(
                state.get_gpu_memory_status(), 0
            ),
            "ringrift_gpu_memory_constrained": int(state.is_gpu_memory_constrained()),
        }

    # =========================================================================
    # Rate Negotiation Between Orchestrators
    # =========================================================================

    def negotiate_selfplay_rate(
        self,
        requested_rate: int,
        reason: str,
        requestor: str,
    ) -> int:
        """Negotiate a new selfplay rate based on current utilization.

        This allows unified_ai_loop and p2p_orchestrator to coordinate
        selfplay rates to maintain 60-80% utilization.

        Args:
            requested_rate: Desired games per hour
            reason: Why the rate change is requested
            requestor: "unified_loop" or "p2p"

        Returns:
            Approved rate (may differ from requested based on utilization)
        """
        now = time.time()

        # Get current cluster utilization
        cluster_state = self.get_cluster_state()
        current_util = (
            cluster_state.total_gpu_util
            if cluster_state.gpu_node_count > 0
            else cluster_state.total_cpu_util
        )

        # Calculate approved rate based on utilization and GPU memory
        approved_rate = self._calculate_approved_rate(
            requested_rate, current_util, reason, cluster_state
        )

        # Build response
        if approved_rate == requested_rate:
            response = "approved"
        elif approved_rate > requested_rate:
            response = f"increased_from_{requested_rate}_underutilized_{current_util:.0f}pct"
        else:
            response = f"reduced_from_{requested_rate}_overutilized_{current_util:.0f}pct"

        # Persist to DB
        with self._db_lock, self._get_connection() as conn:
            conn.execute("""
                    INSERT INTO rate_negotiations
                    (timestamp, requestor, requested_rate, reason, approved_rate, current_utilization, response)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (now, requestor, requested_rate, reason, approved_rate, current_util, response))

            conn.execute("""
                    UPDATE selfplay_rate SET current_rate = ?, updated_at = ?, updated_by = ?
                    WHERE id = 1
                """, (approved_rate, now, requestor))

            conn.commit()

        logger.info(
            f"[RateNegotiation] {requestor} requested {requested_rate}, "
            f"approved {approved_rate} ({response})"
        )

        return approved_rate

    def _calculate_approved_rate(
        self,
        requested_rate: int,
        current_utilization: float,
        reason: str,
        cluster_state: ClusterState | None = None,
    ) -> int:
        """Calculate approved rate based on current utilization and GPU memory.

        Args:
            requested_rate: Requested selfplay rate
            current_utilization: Current CPU/GPU utilization percentage
            reason: Reason for the request
            cluster_state: Optional cluster state for GPU memory checks

        Returns:
            Approved rate (may be reduced for high GPU memory usage)
        """

        MIN_RATE = 100
        MAX_RATE = 5000

        # Emergency reasons always get approved
        if "emergency" in reason.lower() or "critical" in reason.lower():
            return max(MIN_RATE, min(requested_rate, MAX_RATE))

        # Check GPU memory constraints first (if cluster_state available)
        gpu_memory_multiplier = 1.0
        if cluster_state is not None:
            if cluster_state.is_gpu_memory_critical():
                # Critical GPU memory: aggressive throttling
                gpu_memory_multiplier = 0.3
                logger.warning(
                    f"GPU memory critical ({cluster_state.total_gpu_memory_util:.1f}%), "
                    f"throttling rate to {gpu_memory_multiplier*100:.0f}%"
                )
            elif cluster_state.is_gpu_memory_constrained():
                # Warning: moderate throttling
                gpu_memory_multiplier = 0.7
                logger.info(
                    f"GPU memory constrained ({cluster_state.total_gpu_memory_util:.1f}%), "
                    f"reducing rate to {gpu_memory_multiplier*100:.0f}%"
                )

        # Calculate adjustment based on utilization gap from target
        if current_utilization < TARGET_UTIL_MIN:
            # Underutilized - increase rate
            # Gap of 20% (60-40) should give ~1.4x multiplier
            multiplier = 1.0 + (TARGET_UTIL_MIN - current_utilization) / 50.0
            adjusted_rate = int(requested_rate * min(1.5, multiplier))

        elif current_utilization > TARGET_UTIL_MAX:
            # Overutilized - decrease rate
            # Each 10% over max reduces by 20%
            over = current_utilization - TARGET_UTIL_MAX
            multiplier = max(0.5, 1.0 - (over / 50.0))
            adjusted_rate = int(requested_rate * multiplier)

        elif current_utilization > 90.0:  # Critical
            # Aggressive reduction
            adjusted_rate = int(requested_rate * 0.3)

        else:
            # Within target range
            adjusted_rate = requested_rate

        # Apply GPU memory constraint multiplier
        if gpu_memory_multiplier < 1.0:
            adjusted_rate = int(adjusted_rate * gpu_memory_multiplier)

        # Apply bounds
        return max(MIN_RATE, min(adjusted_rate, MAX_RATE))

    def get_current_selfplay_rate(self) -> int:
        """Get the current negotiated selfplay rate."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT current_rate FROM selfplay_rate WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    return row[0]
        except sqlite3.Error:
            pass
        return 1000  # Default

    def get_rate_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent rate negotiation history."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM rate_negotiations
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
                return [dict(row) for row in cursor]
        except sqlite3.Error:
            return []

    # =========================================================================
    # Data-Aware Config Weighting
    # =========================================================================

    def update_config_weights(
        self,
        game_counts: dict[str, int],
        throughput: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Update config weights based on game data distribution.

        Underserved configs get higher weights, overserved get lower weights.

        Args:
            game_counts: Current game count per config (e.g., {"square8_2p": 1000})
            throughput: Games per hour per config (optional)

        Returns:
            Dict of config -> weight (0.5-2.0)
        """
        now = time.time()
        throughput = throughput or {}

        if not game_counts:
            return {}

        # Calculate total and target per config
        total_games = sum(game_counts.values())
        num_configs = len(game_counts)
        avg_per_config = total_games / num_configs if num_configs else 0

        weights = {}

        for config, count in game_counts.items():
            # Base weight from data distribution
            if avg_per_config > 0:
                ratio = count / avg_per_config
                # Underserved configs (< 70% of avg) get higher weight
                if ratio < 0.7:
                    weight = min(2.0, 1.5 / max(0.1, ratio))
                    reason = f"underserved_{count}_games_{ratio:.0%}_of_avg"
                # Overserved configs (> 130% of avg) get lower weight
                elif ratio > 1.3:
                    weight = max(0.5, 0.8 / ratio)
                    reason = f"well_represented_{count}_games_{ratio:.0%}_of_avg"
                else:
                    weight = 1.0
                    reason = "balanced"
            else:
                weight = 1.0
                reason = "no_data"

            # Boost configs with low throughput (harder to generate)
            gph = throughput.get(config, 0)
            if gph > 0 and gph < 50:  # Less than 50 games/hour is slow
                weight = min(2.0, weight * 1.2)
                reason += f"_slow_throughput_{gph:.0f}gph"

            weights[config] = weight

            # Persist to DB
            with self._db_lock, self._get_connection() as conn:
                conn.execute("""
                        INSERT OR REPLACE INTO config_weights
                        (config_key, weight, game_count, games_per_hour, reason, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (config, weight, count, gph, reason, now))
                conn.commit()

        logger.info(f"[ConfigWeights] Updated weights for {len(weights)} configs")
        return weights

    def get_config_weights(self) -> dict[str, float]:
        """Get current config weights for selfplay distribution."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT config_key, weight FROM config_weights")
                return {row["config_key"]: row["weight"] for row in cursor}
        except sqlite3.Error:
            return {}

    def get_config_weight_details(self) -> list[dict[str, Any]]:
        """Get detailed config weight information."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM config_weights ORDER BY weight DESC
                """)
                return [dict(row) for row in cursor]
        except sqlite3.Error:
            return []

    # =========================================================================
    # Utilization Feedback Loop
    # =========================================================================

    def apply_feedback_adjustment(self, requestor: str = "feedback_loop") -> int:
        """Automatically adjust selfplay rate based on current utilization.

        This is the core feedback loop for maintaining 60-80% utilization.
        Call periodically (e.g., every 5 minutes) to adjust rates.

        Returns:
            New selfplay rate
        """
        cluster_state = self.get_cluster_state()
        current_rate = self.get_current_selfplay_rate()

        # Use GPU utilization for GPU-heavy cluster, otherwise CPU
        if cluster_state.gpu_node_count > 0:
            current_util = cluster_state.total_gpu_util
        else:
            current_util = cluster_state.total_cpu_util

        # Calculate recommended rate adjustment
        if current_util < TARGET_UTIL_MIN:
            # Underutilized - scale up
            gap = TARGET_UTIL_OPTIMAL - current_util
            multiplier = 1.0 + (gap / 50.0)
            new_rate = int(current_rate * min(1.3, multiplier))  # Max 30% increase
            reason = f"feedback_underutilized_{current_util:.0f}pct"
        elif current_util > TARGET_UTIL_MAX:
            # Overutilized - scale down
            over = current_util - TARGET_UTIL_OPTIMAL
            multiplier = max(0.7, 1.0 - (over / 40.0))  # Max 30% decrease
            new_rate = int(current_rate * multiplier)
            reason = f"feedback_overutilized_{current_util:.0f}pct"
        else:
            # Within target range - no change
            return current_rate

        # Negotiate the new rate
        return self.negotiate_selfplay_rate(new_rate, reason, requestor)

    def get_utilization_status(self) -> dict[str, Any]:
        """Get current utilization status for monitoring.

        Returns a summary of cluster utilization relative to targets.
        """
        cluster_state = self.get_cluster_state()

        # Calculate status
        cpu_status = "optimal"
        if cluster_state.total_cpu_util < TARGET_UTIL_MIN:
            cpu_status = "underutilized"
        elif cluster_state.total_cpu_util > TARGET_UTIL_MAX:
            cpu_status = "overutilized"

        gpu_status = "optimal"
        if cluster_state.gpu_node_count > 0:
            if cluster_state.total_gpu_util < TARGET_UTIL_MIN:
                gpu_status = "underutilized"
            elif cluster_state.total_gpu_util > TARGET_UTIL_MAX:
                gpu_status = "overutilized"
        else:
            gpu_status = "no_gpu"

        # Determine overall status
        overall_status = "optimal"
        if cpu_status == "underutilized" or gpu_status == "underutilized":
            overall_status = "below"
        elif cpu_status == "overutilized" or gpu_status == "overutilized":
            overall_status = "above"

        # GPU memory status
        gpu_memory_status = cluster_state.get_gpu_memory_status()

        # Include GPU memory in overall status determination
        if gpu_memory_status == "critical":
            overall_status = "gpu_memory_critical"
        elif gpu_memory_status == "warning" and overall_status == "optimal":
            overall_status = "gpu_memory_warning"

        return {
            "timestamp": time.time(),
            "active_nodes": cluster_state.cpu_node_count,
            "total_jobs": cluster_state.total_jobs,
            "cpu_util": cluster_state.total_cpu_util,
            "gpu_util": cluster_state.total_gpu_util,
            "gpu_memory_util": cluster_state.total_gpu_memory_util,
            "status": overall_status,
            "current_rate": self.get_current_selfplay_rate(),
            "cpu": {
                "avg_percent": cluster_state.total_cpu_util,
                "status": cpu_status,
                "target_min": TARGET_UTIL_MIN,
                "target_max": TARGET_UTIL_MAX,
            },
            "gpu": {
                "avg_percent": cluster_state.total_gpu_util,
                "status": gpu_status,
                "gpu_nodes": cluster_state.gpu_node_count,
                "target_min": TARGET_UTIL_MIN,
                "target_max": TARGET_UTIL_MAX,
            },
            "gpu_memory": {
                "avg_percent": cluster_state.total_gpu_memory_util,
                "status": gpu_memory_status,
                "warning_threshold": ClusterState.GPU_MEMORY_WARNING,
                "critical_threshold": ClusterState.GPU_MEMORY_CRITICAL,
            },
            "selfplay_rate": self.get_current_selfplay_rate(),
            "recommendation": self._get_recommendation(cluster_state),
        }

    def _get_recommendation(self, cluster_state: ClusterState) -> str:
        """Generate a recommendation based on current state."""
        if cluster_state.cpu_node_count == 0:
            return "No active nodes - check cluster connectivity"

        util = (
            cluster_state.total_gpu_util
            if cluster_state.gpu_node_count > 0
            else cluster_state.total_cpu_util
        )

        if util < TARGET_UTIL_MIN:
            gap = TARGET_UTIL_MIN - util
            return f"Scale UP: {gap:.0f}% below target, increase selfplay jobs"
        elif util > TARGET_UTIL_MAX:
            over = util - TARGET_UTIL_MAX
            return f"Scale DOWN: {over:.0f}% above target, reduce selfplay jobs"
        else:
            return f"OPTIMAL: Utilization {util:.0f}% within 60-80% target range"

    # =========================================================================
    # Predictive Scaling
    # =========================================================================

    def get_prediction(self) -> dict[str, Any] | None:
        """Get utilization prediction.

        Returns:
            Prediction dictionary or None if insufficient data
        """
        return self._predictor.predict()

    def get_proactive_adjustment(self) -> dict[str, Any] | None:
        """Get proactive scaling recommendation based on predictions.

        Returns:
            Recommendation dictionary or None if no action needed
        """
        if not self._predictive_scaling_enabled:
            return None
        return self._predictor.get_proactive_adjustment()

    def apply_proactive_adjustment(self, requestor: str = "predictive") -> int | None:
        """Apply proactive rate adjustment based on utilization predictions.

        This should be called periodically to enable proactive scaling.

        Args:
            requestor: Identifier for the requestor

        Returns:
            New approved rate, or None if no adjustment needed
        """
        if not self._predictive_scaling_enabled:
            return None

        adjustment = self.get_proactive_adjustment()
        if adjustment is None:
            return None

        # Get current rate
        current_rate = self.get_current_selfplay_rate()

        # Apply multiplier
        new_rate = int(current_rate * adjustment["rate_multiplier"])

        # Use negotiation to validate and persist
        reason = f"proactive_{adjustment['action']}: {adjustment['reason']}"
        approved_rate = self.negotiate_selfplay_rate(new_rate, reason, requestor)

        logger.info(
            f"Proactive adjustment: {adjustment['action']} from {current_rate} to {approved_rate} "
            f"(confidence={adjustment['confidence']:.2f})"
        )

        return approved_rate

    def set_predictive_scaling_enabled(self, enabled: bool) -> None:
        """Enable or disable predictive scaling.

        Args:
            enabled: Whether to enable predictive scaling
        """
        self._predictive_scaling_enabled = enabled
        if not enabled:
            self._predictor.clear()
        logger.info(f"Predictive scaling {'enabled' if enabled else 'disabled'}")

    def get_predictor_state(self) -> dict[str, Any]:
        """Get predictor state for monitoring.

        Returns:
            Dictionary with predictor state
        """
        prediction = self._predictor.predict()
        return {
            "enabled": self._predictive_scaling_enabled,
            "sample_count": len(self._predictor._history),
            "history_window_seconds": self._predictor.history_window_seconds,
            "prediction": prediction,
            "proactive_adjustment": self.get_proactive_adjustment(),
        }

    # =========================================================================
    # History Recording
    # =========================================================================

    def record_optimization_action(self, result: OptimizationResult) -> None:
        """Record an optimization action for history tracking.

        Args:
            result: Optimization result that was applied
        """
        with self._db_lock, self._get_connection() as conn:
            conn.execute("""
                    INSERT INTO optimization_history
                        (timestamp, action, resource_type, current_util,
                         target_util, adjustment, nodes_affected, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                time.time(), result.action.value, result.resource_type.value,
                result.current_util, result.target_util, result.adjustment,
                json.dumps(result.nodes_affected), result.reason,
            ))
            conn.commit()

    def cleanup_old_data(self, days: int = 7) -> int:
        """Clean up old utilization data.

        Args:
            days: Delete data older than this many days

        Returns:
            Number of rows deleted
        """
        cutoff = time.time() - (days * 86400)
        deleted = 0

        with self._db_lock, self._get_connection() as conn:
            cursor = conn.execute("""
                    DELETE FROM utilization_metrics WHERE timestamp < ?
                """, (cutoff,))
            deleted += cursor.rowcount

            cursor = conn.execute("""
                    DELETE FROM optimization_history WHERE timestamp < ?
                """, (cutoff,))
            deleted += cursor.rowcount

            conn.commit()

        return deleted


# =============================================================================
# Module-level convenience functions
# =============================================================================

_optimizer: ResourceOptimizer | None = None


def get_resource_optimizer() -> ResourceOptimizer:
    """Get the singleton resource optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ResourceOptimizer()
    return _optimizer


def should_scale_up(resource_type: str = "cpu") -> bool:
    """Check if scaling up is recommended."""
    return get_resource_optimizer().should_scale_up(resource_type)


def should_scale_down(resource_type: str = "cpu") -> bool:
    """Check if scaling down is recommended."""
    return get_resource_optimizer().should_scale_down(resource_type)


def get_optimal_concurrency(
    node_id: str,
    resource_type: str = "cpu",
    current_jobs: int = 0,
    current_util: float | None = None,
) -> int:
    """Get optimal job concurrency for a node."""
    return get_resource_optimizer().get_optimal_concurrency(
        node_id, resource_type, current_jobs, current_util
    )


def record_utilization(
    node_id: str,
    cpu_percent: float,
    gpu_percent: float | None = None,
    memory_percent: float | None = None,
) -> None:
    """Record a utilization sample."""
    get_resource_optimizer().record_utilization(
        node_id, cpu_percent, gpu_percent, memory_percent
    )


def get_cluster_utilization() -> tuple[float, float, float]:
    """Get current cluster utilization (cpu%, gpu%, memory%).

    Returns:
        Tuple of (cpu_percent, gpu_percent, memory_percent)
    """
    state = get_resource_optimizer().get_cluster_state()
    return (state.total_cpu_util, state.total_gpu_util, state.total_memory_util)


# =============================================================================
# Rate Negotiation Functions
# =============================================================================

def negotiate_selfplay_rate(
    requested_rate: int,
    reason: str,
    requestor: str,
) -> int:
    """Negotiate a selfplay rate between orchestrators.

    Args:
        requested_rate: Desired games per hour
        reason: Why the rate change is requested
        requestor: "unified_loop" or "p2p"

    Returns:
        Approved rate (may differ based on utilization)
    """
    return get_resource_optimizer().negotiate_selfplay_rate(
        requested_rate, reason, requestor
    )


def get_current_selfplay_rate() -> int:
    """Get the current negotiated selfplay rate."""
    return get_resource_optimizer().get_current_selfplay_rate()


def apply_feedback_adjustment(requestor: str = "feedback_loop") -> int:
    """Apply automatic rate adjustment based on utilization.

    Call this periodically to maintain 60-80% utilization.
    """
    return get_resource_optimizer().apply_feedback_adjustment(requestor)


def get_utilization_status() -> dict[str, Any]:
    """Get current utilization status for monitoring."""
    return get_resource_optimizer().get_utilization_status()


# =============================================================================
# Hardware-Aware Selfplay Limits (Single Source of Truth)
# =============================================================================

def get_max_selfplay_for_node(
    node_id: str,
    gpu_count: int = 0,
    gpu_name: str = "",
    cpu_count: int = 0,
    memory_gb: float = 0,
    has_gpu: bool = False,
    gpu_vram_gb: float = 0,
) -> int:
    """Get hardware-aware max selfplay jobs for a node.

    This is the SINGLE SOURCE OF TRUTH for max_selfplay calculations.
    Used by: resource_targets.py, p2p_orchestrator.py, unified_ai_loop.py

    The calculation considers:
    1. GPU type and count (datacenter GPUs can handle more parallelism)
    2. GPU VRAM (more VRAM = more jobs, ~1.5-2GB per job)
    3. CPU count (for CPU-bound parts of hybrid selfplay)
    4. System memory constraints (each job needs ~2GB RAM)
    5. High-CPU bonus: Machines with many CPUs can run hybrid jobs more efficiently

    Args:
        node_id: Node identifier (used for logging/debugging)
        gpu_count: Number of GPUs on node
        gpu_name: GPU model name (e.g., "H100", "RTX 4090")
        cpu_count: Number of CPU cores
        memory_gb: Total system memory in GB
        has_gpu: Whether node has GPU capability
        gpu_vram_gb: Total GPU VRAM in GB (optional, estimated from gpu_name if 0)

    Returns:
        Maximum recommended selfplay jobs for this node
    """
    # Limits calibrated from real workloads:
    # - GH200 with 64 cores runs 40-50 jobs at 60-80% GPU util
    # - Consumer GPUs need VRAM-aware limits
    # Use centralized defaults (December 2025)
    try:
        from app.config.coordination_defaults import ResourceLimitsDefaults
        CONSUMER_MAX = ResourceLimitsDefaults.CONSUMER_MAX
        PROSUMER_MAX = ResourceLimitsDefaults.PROSUMER_MAX
        DATACENTER_MAX = ResourceLimitsDefaults.DATACENTER_MAX
        HIGH_CPU_MAX = ResourceLimitsDefaults.HIGH_CPU_MAX
    except ImportError:
        CONSUMER_MAX = 16  # Consumer GPUs
        PROSUMER_MAX = 32  # High-end consumer
        DATACENTER_MAX = 64  # Datacenter GPUs (H100, GH200, A100)
        HIGH_CPU_MAX = 128  # For machines with 100+ CPU cores

    # High-CPU bonus: machines with 100+ cores can efficiently run more jobs
    # because they can handle more CPU-side work per GPU operation
    high_cpu_multiplier = 1.0
    if cpu_count >= 256:
        high_cpu_multiplier = 2.0  # 256+ cores = 2x multiplier
    elif cpu_count >= 128:
        high_cpu_multiplier = 1.5  # 128+ cores = 1.5x multiplier
    elif cpu_count >= 64:
        high_cpu_multiplier = 1.25  # 64+ cores = 1.25x multiplier

    if has_gpu and gpu_count > 0:
        gpu_upper = gpu_name.upper() if gpu_name else ""

        # Estimate VRAM if not provided
        estimated_vram = gpu_vram_gb
        if estimated_vram <= 0:
            # Estimate VRAM from GPU name
            if any(g in gpu_upper for g in ["GH200"]):
                estimated_vram = 480 * gpu_count  # 480GB unified memory
            elif any(g in gpu_upper for g in ["H100", "H200"]):
                estimated_vram = 80 * gpu_count
            elif any(g in gpu_upper for g in ["A100"]):
                estimated_vram = 80 * gpu_count  # Could be 40 or 80GB
            elif any(g in gpu_upper for g in ["L40"]):
                estimated_vram = 48 * gpu_count
            elif any(g in gpu_upper for g in ["5090"]):
                estimated_vram = 32 * gpu_count
            elif any(g in gpu_upper for g in ["4090", "3090", "A10"]):
                estimated_vram = 24 * gpu_count
            elif any(g in gpu_upper for g in ["4060TI", "4060 TI"]):
                estimated_vram = 16 * gpu_count  # 4060Ti has 16GB
            elif any(g in gpu_upper for g in ["4080"]):
                estimated_vram = 16 * gpu_count
            elif any(g in gpu_upper for g in ["4070"]):
                estimated_vram = 12 * gpu_count
            elif any(g in gpu_upper for g in ["3080"]):
                estimated_vram = 10 * gpu_count
            elif any(g in gpu_upper for g in ["3070"]):
                estimated_vram = 8 * gpu_count
            elif any(g in gpu_upper for g in ["2060", "3060"]):
                estimated_vram = 8 * gpu_count  # 2060S has 8GB
            else:
                estimated_vram = 8 * gpu_count  # Conservative default

        # VRAM-based limit: ~2GB VRAM per job for inference
        vram_based = int(estimated_vram / 2) if estimated_vram > 0 else 8

        # Datacenter GPUs: scale with CPU cores (they have massive VRAM)
        if any(g in gpu_upper for g in ["GH200"]):
            # GH200 has 480GB unified memory - CPU is the bottleneck
            cpu_based = int(cpu_count * 0.8) if cpu_count > 0 else 48
            max_selfplay = min(cpu_based, vram_based, DATACENTER_MAX)

        elif any(g in gpu_upper for g in ["H100", "H200"]):
            # H100 has 80GB VRAM - can handle ~20-30 jobs per GPU
            gpu_based = gpu_count * 24
            cpu_based = int(cpu_count * 0.5) if cpu_count > 0 else 32
            max_selfplay = min(gpu_based, cpu_based, vram_based, DATACENTER_MAX)

        elif any(g in gpu_upper for g in ["A100", "L40"]):
            # A100 (40-80GB), L40 (48GB) - moderate parallelism
            gpu_based = gpu_count * 16
            cpu_based = int(cpu_count * 0.4) if cpu_count > 0 else 24
            max_selfplay = min(gpu_based, cpu_based, vram_based, DATACENTER_MAX)

        elif any(g in gpu_upper for g in ["5090"]):
            # RTX 5090 (32GB VRAM) - very high-end consumer
            gpu_based = gpu_count * 12
            cpu_based = int(cpu_count * 0.3 * high_cpu_multiplier) if cpu_count > 0 else 48
            max_selfplay = min(gpu_based, cpu_based, vram_based, HIGH_CPU_MAX)

        elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
            # High-end consumer/prosumer (24GB VRAM)
            gpu_based = gpu_count * 10  # Raised from 8
            cpu_based = int(cpu_count * 0.35 * high_cpu_multiplier) if cpu_count > 0 else 16
            max_selfplay = min(gpu_based, cpu_based, vram_based, PROSUMER_MAX)

        elif any(g in gpu_upper for g in ["4080", "4060TI", "4060 TI"]):
            # 16GB VRAM cards - can handle more jobs
            gpu_based = gpu_count * 8  # Raised from 6
            cpu_based = int(cpu_count * 0.3 * high_cpu_multiplier) if cpu_count > 0 else 12
            max_selfplay = min(gpu_based, cpu_based, vram_based, PROSUMER_MAX)

        elif any(g in gpu_upper for g in ["4070", "3080", "4060"]):
            # 10-12GB VRAM cards
            gpu_based = gpu_count * 6
            cpu_based = int(cpu_count * 0.25 * high_cpu_multiplier) if cpu_count > 0 else 10
            max_selfplay = min(gpu_based, cpu_based, vram_based, CONSUMER_MAX)

        elif any(g in gpu_upper for g in ["3070", "3060", "2080", "2070", "2060"]):
            # 8GB VRAM - more constrained by VRAM
            # But high-CPU machines can still benefit from more parallelism
            gpu_based = gpu_count * 4
            cpu_based = int(cpu_count * 0.2 * high_cpu_multiplier) if cpu_count > 0 else 8
            # Allow higher limit for high-CPU machines (up to VRAM limit)
            tier_max = 12 if cpu_count >= 64 else 10
            max_selfplay = min(gpu_based, cpu_based, vram_based, tier_max)

        else:
            # Unknown GPU - conservative but respect high-CPU
            gpu_based = gpu_count * 4
            cpu_based = int(cpu_count * 0.2 * high_cpu_multiplier) if cpu_count > 0 else 8
            max_selfplay = min(gpu_based, cpu_based, vram_based, CONSUMER_MAX)

        # System memory constraint (~2GB RAM per job)
        if memory_gb > 0:
            mem_based = int(memory_gb / 2)
            max_selfplay = min(max_selfplay, mem_based)

    else:
        # CPU-only nodes: scale with CPU count
        # Higher multiplier for very high-CPU machines
        if cpu_count > 0:
            base_multiplier = 0.3 if cpu_count < 64 else 0.4 if cpu_count < 128 else 0.5
            cpu_based = int(cpu_count * base_multiplier)
            mem_based = int(memory_gb / 2) if memory_gb > 0 else 64
            # Higher cap for high-CPU machines
            cap = 32 if cpu_count < 64 else 64 if cpu_count < 256 else 128
            max_selfplay = min(cpu_based, mem_based, cap)
        else:
            max_selfplay = 8  # Conservative default

    # Low memory systems need extra constraints
    if memory_gb > 0 and memory_gb < 16:
        max_selfplay = min(max_selfplay, max(2, int(memory_gb / 2)))

    return max(1, max_selfplay)


def get_max_cpu_only_selfplay(
    node_id: str,
    cpu_count: int = 0,
    memory_gb: float = 0,
    gpu_jobs_running: int = 0,
) -> int:
    """Get max ADDITIONAL CPU-only selfplay jobs for a node.

    This is for hybrid mode: run GPU-accelerated jobs up to VRAM limit,
    then run additional CPU-only jobs to utilize remaining CPU capacity.

    For nodes with high CPU counts but limited GPU VRAM (like Vast hosts),
    this enables much better resource utilization.

    Example:
        vast-2060s (88 CPUs, 8GB VRAM):
        - GPU jobs: 4 (limited by VRAM)
        - CPU-only jobs: ~20 additional
        - Total: 24 jobs utilizing both GPU and excess CPU

    Args:
        node_id: Node identifier
        cpu_count: Number of CPU cores
        memory_gb: Total system memory in GB
        gpu_jobs_running: Number of GPU selfplay jobs already running

    Returns:
        Maximum additional CPU-only selfplay jobs
    """
    if cpu_count <= 0:
        return 0

    # CPU-only jobs are lighter (~0.5 cores per job vs 1 core for GPU jobs)
    # Reserve cores for GPU jobs (assume ~1.5 cores per GPU job)
    reserved_for_gpu = int(gpu_jobs_running * 1.5)
    available_cores = max(0, cpu_count - reserved_for_gpu)

    # Scale CPU-only jobs
    # - Small machines (<32 cores): 0.3 jobs per core
    # - Medium machines (32-128 cores): 0.4 jobs per core
    # - Large machines (128-256 cores): 0.5 jobs per core
    # - Huge machines (256+ cores): 0.6 jobs per core
    if available_cores < 32:
        multiplier = 0.3
        cap = 8
    elif available_cores < 128:
        multiplier = 0.4
        cap = 32
    elif available_cores < 256:
        multiplier = 0.5
        cap = 64
    else:
        multiplier = 0.6
        cap = 128

    cpu_based = int(available_cores * multiplier)

    # Memory constraint: CPU-only jobs use less memory (~1.5GB vs 2GB)
    if memory_gb > 0:
        # Reserve memory for GPU jobs (~2GB each)
        available_mem = max(0, memory_gb - (gpu_jobs_running * 2))
        mem_based = int(available_mem / 1.5)
    else:
        mem_based = cap

    return min(cpu_based, mem_based, cap)


def get_hybrid_selfplay_limits(
    node_id: str,
    gpu_count: int = 0,
    gpu_name: str = "",
    cpu_count: int = 0,
    memory_gb: float = 0,
    has_gpu: bool = False,
    gpu_vram_gb: float = 0,
) -> dict[str, int]:
    """Get both GPU and CPU-only selfplay limits for hybrid mode.

    Returns separate limits for GPU-accelerated and CPU-only jobs,
    enabling maximum resource utilization on high-CPU machines.

    Args:
        node_id: Node identifier
        gpu_count: Number of GPUs
        gpu_name: GPU model name
        cpu_count: Number of CPU cores
        memory_gb: Total system memory in GB
        has_gpu: Whether node has GPU capability
        gpu_vram_gb: Total GPU VRAM in GB

    Returns:
        Dict with 'gpu_jobs', 'cpu_only_jobs', 'total_jobs'
    """
    # Get GPU-accelerated job limit
    gpu_jobs = get_max_selfplay_for_node(
        node_id=node_id,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        has_gpu=has_gpu,
        gpu_vram_gb=gpu_vram_gb,
    )

    # Get additional CPU-only job limit
    cpu_only_jobs = get_max_cpu_only_selfplay(
        node_id=node_id,
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        gpu_jobs_running=gpu_jobs,
    )

    return {
        "gpu_jobs": gpu_jobs,
        "cpu_only_jobs": cpu_only_jobs,
        "total_jobs": gpu_jobs + cpu_only_jobs,
    }


def get_node_hardware_info(node_id: str) -> dict[str, Any] | None:
    """Get hardware info for a node from the coordination DB.

    Returns:
        Dict with gpu_count, gpu_name, cpu_count, memory_gb, has_gpu
        or None if node not found
    """
    node = get_resource_optimizer().get_node_resources(node_id)
    if node is None:
        return None
    return {
        "gpu_count": node.gpu_count,
        "gpu_name": node.gpu_name,
        "cpu_count": node.cpu_count,
        "memory_gb": node.memory_gb,
        "has_gpu": node.has_gpu,
    }


def get_max_selfplay_for_node_by_id(node_id: str) -> int:
    """Get hardware-aware max selfplay by looking up node info.

    Convenience function that queries the coordination DB for hardware info.
    Falls back to conservative defaults if node not found.
    """
    hw = get_node_hardware_info(node_id)
    if hw is None:
        # Node not in DB - use conservative default based on hostname patterns
        node_lower = node_id.lower()
        if any(g in node_lower for g in ["h100", "h200", "gh200"]):
            return 12  # HIGH_END
        elif any(g in node_lower for g in ["a100", "4090", "a10"]):
            return 8  # MID_TIER
        else:
            return 6  # Conservative default

    return get_max_selfplay_for_node(
        node_id=node_id,
        gpu_count=hw["gpu_count"],
        gpu_name=hw["gpu_name"],
        cpu_count=hw["cpu_count"],
        memory_gb=hw["memory_gb"],
        has_gpu=hw["has_gpu"],
    )


# =============================================================================
# Config Weighting Functions
# =============================================================================

def update_config_weights(
    game_counts: dict[str, int],
    throughput: dict[str, float] | None = None,
) -> dict[str, float]:
    """Update config weights based on game data distribution."""
    return get_resource_optimizer().update_config_weights(game_counts, throughput)


def get_config_weights() -> dict[str, float]:
    """Get current config weights for selfplay distribution."""
    return get_resource_optimizer().get_config_weights()


def get_config_weight_details() -> list[dict[str, Any]]:
    """Get detailed config weight information."""
    return get_resource_optimizer().get_config_weight_details()


# =============================================================================
# Predictive Scaling Functions
# =============================================================================

def get_prediction() -> dict[str, Any] | None:
    """Get utilization prediction based on historical data."""
    return get_resource_optimizer().get_prediction()


def get_proactive_adjustment() -> dict[str, Any] | None:
    """Get proactive scaling recommendation based on predictions."""
    return get_resource_optimizer().get_proactive_adjustment()


def apply_proactive_adjustment(requestor: str = "predictive") -> int | None:
    """Apply proactive rate adjustment based on utilization predictions.

    Call this periodically (e.g., every 2-5 minutes) to enable proactive scaling.
    """
    return get_resource_optimizer().apply_proactive_adjustment(requestor)


def get_predictor_state() -> dict[str, Any]:
    """Get predictor state for monitoring."""
    return get_resource_optimizer().get_predictor_state()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "ClusterState",
    # Data classes
    "NodeResources",
    "OptimizationResult",
    # Classes
    "PIDController",
    "ResourceOptimizer",
    # Enums
    "ResourceType",
    "ScaleAction",
    "UtilizationPredictor",
    "apply_feedback_adjustment",
    "apply_proactive_adjustment",
    "get_cluster_utilization",
    "get_current_selfplay_rate",
    "get_max_cpu_only_selfplay",
    "get_max_selfplay_for_node",
    "get_optimal_concurrency",
    "get_predictor_state",
    "get_proactive_adjustment",
    # Functions
    "get_resource_optimizer",
    "get_utilization_status",
    "negotiate_selfplay_rate",
    "record_utilization",
    "should_scale_down",
    "should_scale_up",
]
