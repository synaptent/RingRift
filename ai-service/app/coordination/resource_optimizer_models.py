"""PID and utilization prediction helpers for resource optimization."""

from __future__ import annotations

import threading
import time
from typing import Any

from app.coordination.resource_optimizer_shared import (
    ClusterState,
    PID_KD,
    PID_KI,
    PID_KP,
    TARGET_UTIL_MAX,
    TARGET_UTIL_MIN,
    TARGET_UTIL_OPTIMAL,
    UTILIZATION_UPDATE_INTERVAL,
)


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
