"""
Job Stall Detection for Cluster Coordination.

Ported from deprecated SyncStalledHandler (December 2025).

This module detects stalled jobs and manages node penalties:
- Detects jobs with no progress for threshold seconds
- Applies time-based penalties with exponential backoff
- Tracks stall counts and automatic recovery

Usage:
    from app.coordination.stall_detection import (
        JobStallDetector,
        get_stall_detector,
    )

    # Get detector instance
    detector = get_stall_detector()

    # Check if job is stalled
    if detector.is_job_stalled(job_id, last_progress_time):
        detector.report_stall(job_id, node_id)

    # Check if node is penalized before spawning
    if detector.is_node_penalized(node_id):
        logger.info(f"Skipping {node_id}: under stall penalty")
        return False
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StallSeverity(str, Enum):
    """Severity level of a detected stall."""

    MINOR = "minor"      # < 2x threshold: short penalty
    MODERATE = "moderate"  # 2-5x threshold: medium penalty
    SEVERE = "severe"    # > 5x threshold: long penalty


@dataclass
class StallRecord:
    """Record of a stall event."""

    job_id: str
    node_id: str
    stall_time: float
    duration_seconds: float
    severity: StallSeverity
    recovery_action: str = ""


@dataclass
class NodePenalty:
    """Penalty applied to a node after stalls."""

    node_id: str
    penalty_until: float
    stall_count: int
    last_stall_time: float
    backoff_level: int = 0  # Exponential backoff multiplier


@dataclass
class StallDetectorConfig:
    """Configuration for stall detection."""

    # Stall thresholds
    stall_threshold_seconds: float = 300.0  # 5 minutes without progress = stalled
    severe_stall_multiplier: float = 5.0    # 5x threshold = severe

    # Penalty settings
    base_penalty_seconds: float = 300.0     # 5 minutes base penalty
    max_penalty_seconds: float = 3600.0     # 1 hour max penalty
    penalty_decay_factor: float = 0.5       # How fast penalty level decays

    # Recovery settings
    max_stalls_before_permanent: int = 5    # After this many stalls, node marked unhealthy
    recovery_check_interval: float = 60.0   # Check for recovery every minute

    # Backoff settings
    backoff_base: float = 2.0               # Exponential backoff base
    max_backoff_level: int = 4              # Max backoff level (16x penalty)


class JobStallDetector:
    """Detects stalled jobs and manages node penalties.

    This class monitors job progress and:
    1. Detects when jobs have no progress for threshold seconds
    2. Applies time-based penalties to nodes with stalled jobs
    3. Uses exponential backoff for repeated offenders
    4. Automatically clears penalties after recovery period
    """

    def __init__(self, config: StallDetectorConfig | None = None):
        """Initialize the stall detector.

        Args:
            config: Configuration for thresholds and penalties
        """
        self.config = config or StallDetectorConfig()

        # Node penalties: node_id -> NodePenalty
        self._penalties: dict[str, NodePenalty] = {}

        # Stall history: list of recent stalls
        self._stall_history: list[StallRecord] = []

        # Active jobs: job_id -> (node_id, last_progress_time)
        self._active_jobs: dict[str, tuple[str, float]] = {}

        # Statistics
        self._total_stalls = 0
        self._total_recoveries = 0
        self._last_cleanup = 0.0

    def register_job(self, job_id: str, node_id: str) -> None:
        """Register a new job to track.

        Args:
            job_id: Unique job identifier
            node_id: Node running the job
        """
        self._active_jobs[job_id] = (node_id, time.time())
        logger.debug(f"[StallDetector] Registered job {job_id} on {node_id}")

    def update_progress(self, job_id: str) -> None:
        """Update progress timestamp for a job.

        Args:
            job_id: Job that made progress
        """
        if job_id in self._active_jobs:
            node_id = self._active_jobs[job_id][0]
            self._active_jobs[job_id] = (node_id, time.time())

    def complete_job(self, job_id: str, success: bool = True) -> None:
        """Mark a job as completed.

        Args:
            job_id: Completed job identifier
            success: Whether job completed successfully
        """
        if job_id in self._active_jobs:
            node_id = self._active_jobs[job_id][0]
            del self._active_jobs[job_id]

            if success:
                # Successful completion reduces backoff level
                self._reduce_penalty_level(node_id)

    def is_job_stalled(
        self,
        job_id: str,
        last_progress_time: float | None = None,
    ) -> bool:
        """Check if a job is stalled.

        Args:
            job_id: Job to check
            last_progress_time: Optional override for progress time

        Returns:
            True if job is stalled
        """
        now = time.time()

        if last_progress_time is None:
            if job_id not in self._active_jobs:
                return False
            last_progress_time = self._active_jobs[job_id][1]

        elapsed = now - last_progress_time
        return elapsed > self.config.stall_threshold_seconds

    def get_stall_duration(self, job_id: str) -> float:
        """Get how long a job has been stalled.

        Args:
            job_id: Job to check

        Returns:
            Duration in seconds (0 if not tracked or not stalled)
        """
        if job_id not in self._active_jobs:
            return 0.0

        now = time.time()
        last_progress = self._active_jobs[job_id][1]
        elapsed = now - last_progress

        if elapsed > self.config.stall_threshold_seconds:
            return elapsed
        return 0.0

    def report_stall(
        self,
        job_id: str,
        node_id: str,
        duration_seconds: float | None = None,
    ) -> StallRecord:
        """Report a stalled job and apply penalty.

        Args:
            job_id: Stalled job identifier
            node_id: Node with the stalled job
            duration_seconds: How long the job was stalled

        Returns:
            StallRecord describing the stall
        """
        now = time.time()
        self._total_stalls += 1

        # Calculate duration and severity
        if duration_seconds is None:
            if job_id in self._active_jobs:
                last_progress = self._active_jobs[job_id][1]
                duration_seconds = now - last_progress
            else:
                duration_seconds = self.config.stall_threshold_seconds

        severity = self._calculate_severity(duration_seconds)

        # Create stall record
        record = StallRecord(
            job_id=job_id,
            node_id=node_id,
            stall_time=now,
            duration_seconds=duration_seconds,
            severity=severity,
        )

        # Apply penalty to node
        self._apply_penalty(node_id, severity)
        record.recovery_action = f"Penalty applied: {severity.value}"

        # Store in history
        self._stall_history.append(record)
        if len(self._stall_history) > 1000:
            self._stall_history = self._stall_history[-500:]

        logger.warning(
            f"[StallDetector] Stall #{self._total_stalls}: job={job_id} "
            f"node={node_id} duration={duration_seconds:.0f}s severity={severity.value}"
        )

        return record

    def _calculate_severity(self, duration_seconds: float) -> StallSeverity:
        """Calculate severity based on stall duration."""
        ratio = duration_seconds / self.config.stall_threshold_seconds

        if ratio < 2.0:
            return StallSeverity.MINOR
        elif ratio < self.config.severe_stall_multiplier:
            return StallSeverity.MODERATE
        else:
            return StallSeverity.SEVERE

    def _apply_penalty(self, node_id: str, severity: StallSeverity) -> None:
        """Apply penalty to a node based on stall severity."""
        now = time.time()

        # Get or create penalty record
        if node_id in self._penalties:
            penalty = self._penalties[node_id]
            penalty.stall_count += 1
            penalty.last_stall_time = now
            # Increase backoff level
            penalty.backoff_level = min(
                penalty.backoff_level + 1,
                self.config.max_backoff_level,
            )
        else:
            penalty = NodePenalty(
                node_id=node_id,
                penalty_until=0,
                stall_count=1,
                last_stall_time=now,
                backoff_level=0,
            )
            self._penalties[node_id] = penalty

        # Calculate penalty duration with backoff
        base_multiplier = {
            StallSeverity.MINOR: 1.0,
            StallSeverity.MODERATE: 2.0,
            StallSeverity.SEVERE: 4.0,
        }[severity]

        backoff_multiplier = self.config.backoff_base ** penalty.backoff_level
        penalty_duration = min(
            self.config.base_penalty_seconds * base_multiplier * backoff_multiplier,
            self.config.max_penalty_seconds,
        )

        penalty.penalty_until = now + penalty_duration

        logger.info(
            f"[StallDetector] Applied {penalty_duration:.0f}s penalty to {node_id} "
            f"(stall_count={penalty.stall_count}, backoff_level={penalty.backoff_level})"
        )

    def _reduce_penalty_level(self, node_id: str) -> None:
        """Reduce penalty level after successful job completion."""
        if node_id not in self._penalties:
            return

        penalty = self._penalties[node_id]
        if penalty.backoff_level > 0:
            penalty.backoff_level -= 1
            self._total_recoveries += 1
            logger.debug(
                f"[StallDetector] Reduced backoff level for {node_id} to {penalty.backoff_level}"
            )

    def is_node_penalized(self, node_id: str) -> bool:
        """Check if a node is currently under penalty.

        Args:
            node_id: Node to check

        Returns:
            True if node is penalized
        """
        self._cleanup_expired_penalties()

        if node_id not in self._penalties:
            return False

        penalty = self._penalties[node_id]
        return time.time() < penalty.penalty_until

    def get_penalty_remaining(self, node_id: str) -> float:
        """Get remaining penalty time for a node.

        Args:
            node_id: Node to check

        Returns:
            Seconds remaining (0 if not penalized)
        """
        if node_id not in self._penalties:
            return 0.0

        penalty = self._penalties[node_id]
        remaining = penalty.penalty_until - time.time()
        return max(0.0, remaining)

    def is_node_unhealthy(self, node_id: str) -> bool:
        """Check if a node has exceeded max stalls and is marked unhealthy.

        Args:
            node_id: Node to check

        Returns:
            True if node has had too many stalls
        """
        if node_id not in self._penalties:
            return False

        penalty = self._penalties[node_id]
        return penalty.stall_count >= self.config.max_stalls_before_permanent

    def clear_penalty(self, node_id: str) -> None:
        """Manually clear penalty for a node.

        Args:
            node_id: Node to clear
        """
        if node_id in self._penalties:
            del self._penalties[node_id]
            logger.info(f"[StallDetector] Cleared penalty for {node_id}")

    def _cleanup_expired_penalties(self) -> None:
        """Clean up expired penalties periodically."""
        now = time.time()
        if now - self._last_cleanup < self.config.recovery_check_interval:
            return

        self._last_cleanup = now
        expired = [
            node_id for node_id, penalty in self._penalties.items()
            if penalty.penalty_until < now and penalty.backoff_level == 0
        ]

        for node_id in expired:
            del self._penalties[node_id]

        if expired:
            logger.debug(f"[StallDetector] Cleaned up {len(expired)} expired penalties")

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        self._cleanup_expired_penalties()

        active_penalties = sum(
            1 for p in self._penalties.values()
            if time.time() < p.penalty_until
        )

        return {
            "total_stalls": self._total_stalls,
            "total_recoveries": self._total_recoveries,
            "active_jobs": len(self._active_jobs),
            "penalized_nodes": active_penalties,
            "unhealthy_nodes": sum(
                1 for p in self._penalties.values()
                if p.stall_count >= self.config.max_stalls_before_permanent
            ),
            "recent_stalls": len([
                s for s in self._stall_history
                if time.time() - s.stall_time < 3600
            ]),
        }

    def get_stall_history(self, limit: int = 100) -> list[StallRecord]:
        """Get recent stall history.

        Args:
            limit: Maximum records to return

        Returns:
            List of recent stall records
        """
        return self._stall_history[-limit:]


# Module-level singleton
_stall_detector: JobStallDetector | None = None


def get_stall_detector(config: StallDetectorConfig | None = None) -> JobStallDetector:
    """Get the singleton stall detector.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        JobStallDetector instance
    """
    global _stall_detector
    if _stall_detector is None:
        _stall_detector = JobStallDetector(config=config)
    return _stall_detector


def reset_stall_detector() -> None:
    """Reset the singleton for testing."""
    global _stall_detector
    _stall_detector = None


__all__ = [
    "JobStallDetector",
    "NodePenalty",
    "StallDetectorConfig",
    "StallRecord",
    "StallSeverity",
    "get_stall_detector",
    "reset_stall_detector",
]
