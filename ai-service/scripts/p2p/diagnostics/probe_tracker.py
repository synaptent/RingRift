"""Track probe success rates and false positive detection."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger("p2p.diagnostics.probe")


@dataclass
class ProbeResult:
    """Record of a probe attempt."""

    node_id: str
    success: bool
    latency_ms: float | None
    timestamp: float
    transport: str = "unknown"
    recovered_after: bool = field(default=False)  # Did node come back after failed probe?


class ProbeEffectivenessTracker:
    """Track probe effectiveness to detect false positives.

    A "false positive" is when a probe fails but the node recovers shortly after,
    indicating the probe result was misleading (network blip, timeout too short, etc.)

    Features:
    - Track success/failure rates per node and overall
    - Detect false positives (failed probe followed by quick recovery)
    - Identify nodes with poor probe reliability
    - Track latency distributions
    """

    def __init__(self, window: float = 1800.0, max_probes: int = 10000) -> None:
        """Initialize the tracker.

        Args:
            window: Time window in seconds for analysis (default 30 min)
            max_probes: Maximum probes to keep in memory
        """
        self._probes: list[ProbeResult] = []
        self._window = window
        self._max_probes = max_probes
        self._recovery_times: dict[str, float] = {}  # node -> last recovery timestamp

    def record_probe(
        self,
        node_id: str,
        success: bool,
        latency_ms: float | None = None,
        transport: str = "unknown",
    ) -> None:
        """Record a probe result.

        Args:
            node_id: The probed peer's node ID
            success: Whether the probe succeeded
            latency_ms: Round-trip latency in milliseconds (if successful)
            transport: Transport used for probe
        """
        probe = ProbeResult(
            node_id=node_id,
            success=success,
            latency_ms=latency_ms,
            timestamp=time.time(),
            transport=transport,
        )
        self._probes.append(probe)
        self._prune_old()

        # Check for false positive detection
        # If this is a successful probe, check if we had recent failures
        if success:
            self._check_false_positives(node_id)
            self._recovery_times[node_id] = time.time()

        # Log significant events
        if success:
            if latency_ms and latency_ms > 1000:
                logger.warning(
                    f"PROBE_SLOW: {node_id[:20]} latency={latency_ms:.0f}ms "
                    f"transport={transport}"
                )
        else:
            logger.info(f"PROBE_FAIL: {node_id[:20]} transport={transport}")

    def _check_false_positives(self, node_id: str) -> None:
        """Mark recent failed probes as false positives if node recovered quickly."""
        now = time.time()
        false_positive_window = 120.0  # 2 minutes

        for probe in reversed(self._probes):
            if probe.node_id != node_id:
                continue
            if probe.success:
                break  # Stop at last successful probe
            if now - probe.timestamp < false_positive_window:
                if not probe.recovered_after:
                    probe.recovered_after = True
                    logger.info(
                        f"PROBE_FALSE_POSITIVE: {node_id[:20]} recovered "
                        f"{now - probe.timestamp:.0f}s after failed probe"
                    )

    def _prune_old(self) -> None:
        """Remove old probes outside the window."""
        cutoff = time.time() - self._window
        self._probes = [p for p in self._probes if p.timestamp > cutoff]

        # Also enforce max size
        if len(self._probes) > self._max_probes:
            self._probes = self._probes[-self._max_probes :]

    def get_success_rate(self, node_id: str | None = None) -> float:
        """Get probe success rate.

        Args:
            node_id: Filter to specific node (None for all)
        """
        self._prune_old()
        probes = self._probes
        if node_id:
            probes = [p for p in probes if p.node_id == node_id]

        if not probes:
            return 1.0  # No data = assume healthy

        successes = sum(1 for p in probes if p.success)
        return successes / len(probes)

    def get_false_positive_rate(self, node_id: str | None = None) -> float:
        """Calculate false positive rate (failed probes where node recovered).

        A high false positive rate indicates:
        - Timeouts too aggressive
        - Network instability (not node failure)
        - Transport reliability issues
        """
        self._prune_old()

        failed = [p for p in self._probes if not p.success]
        if node_id:
            failed = [p for p in failed if p.node_id == node_id]

        if not failed:
            return 0.0

        false_positives = sum(1 for p in failed if p.recovered_after)
        return false_positives / len(failed)

    def get_average_latency(self, node_id: str | None = None) -> float | None:
        """Get average latency in ms for successful probes.

        Args:
            node_id: Filter to specific node (None for all)
        """
        self._prune_old()
        probes = [p for p in self._probes if p.success and p.latency_ms is not None]
        if node_id:
            probes = [p for p in probes if p.node_id == node_id]

        if not probes:
            return None

        return sum(p.latency_ms for p in probes) / len(probes)  # type: ignore

    def get_unreliable_nodes(self, success_threshold: float = 0.5) -> list[str]:
        """Get nodes with probe success rate below threshold."""
        self._prune_old()

        # Get per-node stats
        by_node: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0})
        for p in self._probes:
            if p.success:
                by_node[p.node_id]["success"] += 1
            else:
                by_node[p.node_id]["fail"] += 1

        # Find unreliable nodes with sufficient data
        unreliable = []
        for nid, stats in by_node.items():
            total = stats["success"] + stats["fail"]
            if total >= 3:  # Need at least 3 probes
                rate = stats["success"] / total
                if rate < success_threshold:
                    unreliable.append(nid)

        return unreliable

    def get_diagnostics(self) -> dict[str, Any]:
        """Return probe diagnostics.

        Returns dict with:
        - total_probes_30min: Count in window
        - success_rate: Overall success rate
        - false_positive_rate: Rate of false positives
        - average_latency_ms: Average successful probe latency
        - worst_success_rates: Top 5 worst nodes by success rate
        - unreliable_nodes: Nodes with <50% success rate
        - by_transport: Success rates per transport
        """
        self._prune_old()

        total = len(self._probes)
        successes = sum(1 for p in self._probes if p.success)

        # Per-node stats
        by_node: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0})
        for p in self._probes:
            if p.success:
                by_node[p.node_id]["success"] += 1
            else:
                by_node[p.node_id]["fail"] += 1

        # Per-transport stats
        by_transport: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "fail": 0}
        )
        for p in self._probes:
            if p.success:
                by_transport[p.transport]["success"] += 1
            else:
                by_transport[p.transport]["fail"] += 1

        # Worst success rates (nodes with at least 3 probes)
        node_rates = [
            (nid[:20], round(stats["success"] / (stats["success"] + stats["fail"]), 2))
            for nid, stats in by_node.items()
            if stats["success"] + stats["fail"] >= 3
        ]
        worst = sorted(node_rates, key=lambda x: x[1])[:5]

        # Transport success rates
        transport_rates = {
            transport: round(
                stats["success"] / (stats["success"] + stats["fail"]), 2
            )
            for transport, stats in by_transport.items()
            if stats["success"] + stats["fail"] > 0
        }

        return {
            "total_probes_30min": total,
            "success_rate": round(successes / total, 3) if total else 1.0,
            "false_positive_rate": round(self.get_false_positive_rate(), 3),
            "average_latency_ms": round(self.get_average_latency() or 0, 1),
            "worst_success_rates": worst,
            "unreliable_nodes": [n[:20] for n in self.get_unreliable_nodes()],
            "by_transport": transport_rates,
        }

    def clear(self) -> None:
        """Clear all tracking data."""
        self._probes.clear()
        self._recovery_times.clear()
