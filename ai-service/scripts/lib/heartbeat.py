"""Heartbeat utilities for process supervision.

This module provides heartbeat file management for the hierarchical process
supervision system. The sentinel process monitors these heartbeat files and
restarts processes when heartbeats go stale.

Architecture:
    launchd/systemd
        ↓ KeepAlive
    Sentinel (C binary)
        ↓ monitors /tmp/ringrift_watchdog.heartbeat
    master_loop_watchdog.py (writes heartbeat)
        ↓ supervises
    master_loop.py
        ↓ supervises
    DaemonManager

Usage:
    from scripts.lib.heartbeat import HeartbeatWriter, HeartbeatReader

    # Writer (in watchdog)
    heartbeat = HeartbeatWriter()
    while running:
        heartbeat.pulse()
        time.sleep(30)

    # Reader (in sentinel or monitoring)
    reader = HeartbeatReader()
    if reader.is_stale():
        restart_watchdog()

January 2026: Created for cluster resilience architecture.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default paths
DEFAULT_WATCHDOG_HEARTBEAT = Path("/tmp/ringrift_watchdog.heartbeat")
DEFAULT_MASTER_LOOP_HEARTBEAT = Path("/tmp/ringrift_master_loop.heartbeat")
DEFAULT_P2P_HEARTBEAT = Path("/tmp/ringrift_p2p.heartbeat")

# Thresholds
DEFAULT_STALE_THRESHOLD = 120.0  # 2 minutes
PULSE_INTERVAL = 30.0  # Expected pulse frequency


@dataclass
class HeartbeatInfo:
    """Information stored in heartbeat file."""

    timestamp: float = field(default_factory=time.time)
    pid: int = field(default_factory=os.getpid)
    node_id: str = ""
    iteration: int = 0
    status: str = "running"
    memory_percent: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> HeartbeatInfo:
        """Deserialize from JSON string."""
        try:
            d = json.loads(data)
            return cls(**d)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse heartbeat JSON: {e}")
            return cls()


class HeartbeatWriter:
    """Writes heartbeat files for process supervision.

    The sentinel process monitors these files and restarts processes
    when heartbeats become stale (file mtime too old).

    Example:
        heartbeat = HeartbeatWriter()
        while running:
            heartbeat.pulse()  # Updates file mtime and content
            # ... main loop work ...
            time.sleep(30)

        # With context (optional metadata)
        heartbeat.pulse(
            iteration=loop_count,
            status="syncing",
            extra={"queue_depth": 42}
        )
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_WATCHDOG_HEARTBEAT,
        node_id: str = "",
    ):
        """Initialize heartbeat writer.

        Args:
            path: Path to heartbeat file
            node_id: Node identifier for multi-node deployments
        """
        self.path = Path(path)
        self.node_id = node_id or os.environ.get("RINGRIFT_NODE_ID", "local")
        self._iteration = 0
        self._last_pulse: float = 0.0

    def pulse(
        self,
        iteration: int | None = None,
        status: str = "running",
        memory_percent: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write heartbeat to file.

        This updates the file's mtime (which the sentinel monitors) and
        writes current process information as JSON content.

        Args:
            iteration: Current loop iteration number
            status: Current status (running, syncing, training, etc.)
            memory_percent: Current memory usage percentage
            extra: Additional metadata to include
        """
        if iteration is not None:
            self._iteration = iteration
        else:
            self._iteration += 1

        # Get memory usage if not provided
        if memory_percent is None:
            try:
                import psutil

                memory_percent = psutil.virtual_memory().percent
            except ImportError:
                memory_percent = 0.0

        info = HeartbeatInfo(
            timestamp=time.time(),
            pid=os.getpid(),
            node_id=self.node_id,
            iteration=self._iteration,
            status=status,
            memory_percent=memory_percent,
            extra=extra or {},
        )

        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Write content (also updates mtime)
            self.path.write_text(info.to_json())
            self._last_pulse = time.time()

            logger.debug(
                f"[Heartbeat] Pulse written to {self.path} "
                f"(iter={self._iteration}, mem={memory_percent:.1f}%)"
            )

        except OSError as e:
            logger.error(f"[Heartbeat] Failed to write heartbeat: {e}")

    def touch(self) -> None:
        """Quick heartbeat - just update mtime without full content."""
        try:
            self.path.touch()
            self._last_pulse = time.time()
        except OSError as e:
            logger.error(f"[Heartbeat] Failed to touch heartbeat: {e}")

    @property
    def last_pulse_age(self) -> float:
        """Seconds since last pulse was written."""
        if self._last_pulse == 0.0:
            return float("inf")
        return time.time() - self._last_pulse


class HeartbeatReader:
    """Reads and monitors heartbeat files.

    Used by the sentinel process or monitoring tools to detect stale heartbeats.

    Example:
        reader = HeartbeatReader()
        if reader.is_stale():
            logger.error("Watchdog heartbeat stale, restarting...")
            restart_watchdog()

        # Get detailed info
        info = reader.read()
        print(f"Last beat from PID {info.pid} at {info.timestamp}")
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_WATCHDOG_HEARTBEAT,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
    ):
        """Initialize heartbeat reader.

        Args:
            path: Path to heartbeat file to monitor
            stale_threshold: Seconds after which heartbeat is considered stale
        """
        self.path = Path(path)
        self.stale_threshold = stale_threshold

    def exists(self) -> bool:
        """Check if heartbeat file exists."""
        return self.path.exists()

    def get_age(self) -> float:
        """Get age of heartbeat file in seconds.

        Returns:
            Seconds since last modification, or inf if file doesn't exist
        """
        if not self.path.exists():
            return float("inf")

        try:
            mtime = self.path.stat().st_mtime
            return time.time() - mtime
        except OSError:
            return float("inf")

    def is_stale(self, threshold: float | None = None) -> bool:
        """Check if heartbeat is stale.

        Args:
            threshold: Override default stale threshold (seconds)

        Returns:
            True if heartbeat is stale or missing
        """
        threshold = threshold or self.stale_threshold
        return self.get_age() > threshold

    def is_fresh(self, threshold: float | None = None) -> bool:
        """Check if heartbeat is fresh (opposite of stale)."""
        return not self.is_stale(threshold)

    def read(self) -> HeartbeatInfo:
        """Read heartbeat file content.

        Returns:
            HeartbeatInfo from file content, or default if unreadable
        """
        if not self.path.exists():
            return HeartbeatInfo()

        try:
            content = self.path.read_text()
            return HeartbeatInfo.from_json(content)
        except OSError as e:
            logger.warning(f"[Heartbeat] Failed to read heartbeat: {e}")
            return HeartbeatInfo()

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive heartbeat status.

        Returns:
            Dictionary with all heartbeat status information
        """
        age = self.get_age()
        info = self.read() if self.exists() else HeartbeatInfo()

        return {
            "path": str(self.path),
            "exists": self.exists(),
            "age_seconds": age if age != float("inf") else None,
            "is_stale": self.is_stale(),
            "stale_threshold": self.stale_threshold,
            "pid": info.pid,
            "node_id": info.node_id,
            "iteration": info.iteration,
            "status": info.status,
            "memory_percent": info.memory_percent,
            "last_timestamp": info.timestamp,
            "extra": info.extra,
        }


# Convenience functions for common heartbeat paths


def get_watchdog_heartbeat(node_id: str = "") -> HeartbeatWriter:
    """Get heartbeat writer for the watchdog process."""
    return HeartbeatWriter(DEFAULT_WATCHDOG_HEARTBEAT, node_id=node_id)


def get_master_loop_heartbeat(node_id: str = "") -> HeartbeatWriter:
    """Get heartbeat writer for the master loop process."""
    return HeartbeatWriter(DEFAULT_MASTER_LOOP_HEARTBEAT, node_id=node_id)


def get_p2p_heartbeat(node_id: str = "") -> HeartbeatWriter:
    """Get heartbeat writer for the P2P orchestrator."""
    return HeartbeatWriter(DEFAULT_P2P_HEARTBEAT, node_id=node_id)


def check_watchdog_heartbeat(stale_threshold: float = DEFAULT_STALE_THRESHOLD) -> bool:
    """Quick check if watchdog heartbeat is fresh.

    Returns:
        True if heartbeat is fresh, False if stale or missing
    """
    reader = HeartbeatReader(DEFAULT_WATCHDOG_HEARTBEAT, stale_threshold)
    return reader.is_fresh()


def check_all_heartbeats(
    stale_threshold: float = DEFAULT_STALE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Check status of all standard heartbeat files.

    Returns:
        Dictionary mapping heartbeat name to status dict
    """
    heartbeats = {
        "watchdog": DEFAULT_WATCHDOG_HEARTBEAT,
        "master_loop": DEFAULT_MASTER_LOOP_HEARTBEAT,
        "p2p": DEFAULT_P2P_HEARTBEAT,
    }

    return {
        name: HeartbeatReader(path, stale_threshold).get_status()
        for name, path in heartbeats.items()
    }
