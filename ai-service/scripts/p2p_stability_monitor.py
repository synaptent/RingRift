#!/usr/bin/env python3
"""P2P Cluster Stability Monitor.

Jan 23, 2026: Created as Phase 6 of P2P stability plan.

Monitors cluster health at configurable intervals and logs observations.
Designed for 60-minute monitoring cycles (6 checks @ 10 min) to detect
intermittent stability issues.

Usage:
    # 6 checks @ 10 min = 60 min monitoring
    python scripts/p2p_stability_monitor.py --cycles 6 --interval 10

    # 24 checks @ 10 min = 4 hour monitoring
    python scripts/p2p_stability_monitor.py --cycles 24 --interval 10

    # Continuous monitoring with 5-minute intervals
    python scripts/p2p_stability_monitor.py --continuous --interval 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "p2p_stability_monitor.log"


class StabilityMonitor:
    """Monitors P2P cluster stability at regular intervals."""

    def __init__(
        self,
        p2p_host: str = "localhost",
        p2p_port: int = 8770,
        log_file: Path | None = None,
    ):
        self.p2p_url = f"http://{p2p_host}:{p2p_port}"
        self.log_file = log_file or LOG_FILE
        self.observations: list[dict[str, Any]] = []

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _log_to_file(self, message: str) -> None:
        """Append message to log file."""
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def _get_cluster_status(self) -> dict[str, Any] | None:
        """Fetch cluster status from P2P orchestrator."""
        try:
            response = requests.get(f"{self.p2p_url}/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"P2P status returned {response.status_code}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Failed to get P2P status: {e}")
            return None

    def _extract_provider(self, node_id: str) -> str:
        """Extract provider from node ID."""
        if not node_id:
            return "Unknown"
        node_lower = node_id.lower()
        providers = ["lambda", "vast", "runpod", "nebius", "vultr", "hetzner", "local", "mac"]
        for provider in providers:
            if node_lower.startswith(provider):
                return provider.capitalize()
        if "mac" in node_lower or "local" in node_lower:
            return "Local"
        return "Other"

    def _get_provider_breakdown(self, peers: dict[str, Any]) -> dict[str, int]:
        """Count peers by provider."""
        breakdown: dict[str, int] = {}
        for node_id, info in peers.items():
            # Only count alive peers
            is_alive = info.get("is_alive", False) if isinstance(info, dict) else getattr(info, "is_alive", lambda: False)()
            if callable(is_alive):
                is_alive = is_alive()
            if not is_alive:
                continue
            provider = self._extract_provider(node_id)
            breakdown[provider] = breakdown.get(provider, 0) + 1
        return breakdown

    def _check_split_brain(self, status: dict[str, Any]) -> dict[str, Any] | None:
        """Check for split-brain conditions."""
        consensus = status.get("leader_consensus", {})
        leader_agreement = consensus.get("leader_agreement", 0)
        total_voters = consensus.get("total_voters", 0)

        if total_voters >= 3:
            agreement_ratio = leader_agreement / total_voters
            if agreement_ratio < 0.5:
                return {
                    "detected": True,
                    "reason": f"Low leader agreement: {leader_agreement}/{total_voters} ({agreement_ratio:.0%})",
                    "consensus_leader": consensus.get("consensus_leader"),
                }
        return None

    def take_observation(self) -> dict[str, Any]:
        """Take a single observation of cluster state."""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        observation = {
            "timestamp": timestamp,
            "unix_time": time.time(),
            "status": "UNKNOWN",
            "alive_peers": 0,
            "leader_id": None,
            "quorum_ok": False,
            "breakdown": {},
            "split_brain": None,
            "error": None,
        }

        status = self._get_cluster_status()
        if status is None:
            observation["status"] = "ERROR"
            observation["error"] = "Could not reach P2P orchestrator"
            return observation

        # Extract key metrics
        observation["leader_id"] = status.get("leader_id") or status.get("effective_leader_id")
        observation["alive_peers"] = status.get("alive_peers", 0)

        # Get voter health
        voter_health = status.get("voter_health", {})
        observation["quorum_ok"] = voter_health.get("quorum_ok", False)
        observation["voters_alive"] = voter_health.get("voters_alive", 0)
        observation["voters_total"] = voter_health.get("voters_total", 0)

        # Count alive peers by provider
        peers = status.get("peers", {}) or status.get("all_peers", {})
        observation["breakdown"] = self._get_provider_breakdown(peers)

        # Check for split-brain
        observation["split_brain"] = self._check_split_brain(status)

        # Determine status
        alive = observation["alive_peers"]
        if observation["split_brain"]:
            observation["status"] = "UNSTABLE"
        elif alive >= 20:
            observation["status"] = "EXCELLENT"
        elif alive >= 15:
            observation["status"] = "GOOD"
        elif alive >= 10:
            observation["status"] = "MODERATE"
        elif alive >= 5:
            observation["status"] = "DEGRADED"
        elif alive >= 1:
            observation["status"] = "MINIMUM"
        else:
            observation["status"] = "CRITICAL"

        self.observations.append(observation)
        return observation

    def format_observation(self, obs: dict[str, Any]) -> str:
        """Format observation for display/logging."""
        lines = [f"--- Check at {obs['timestamp']} ---"]
        if obs["error"]:
            lines.append(f"  ERROR: {obs['error']}")
            return "\n".join(lines)

        lines.append(f"  Leader: {obs['leader_id'] or 'None'}")
        lines.append(f"  ALIVE: {obs['alive_peers']} nodes")

        if obs.get("voters_total"):
            lines.append(f"  Voters: {obs.get('voters_alive', 0)}/{obs['voters_total']}")

        lines.append(f"  Quorum: {obs['quorum_ok']}")

        if obs["breakdown"]:
            breakdown_str = ", ".join(
                f"{k}:{v}" for k, v in sorted(obs["breakdown"].items(), key=lambda x: -x[1])
            )
            lines.append(f"  Breakdown: {breakdown_str}")

        if obs["split_brain"]:
            lines.append(f"  SPLIT-BRAIN DETECTED: {obs['split_brain']['reason']}")

        lines.append(f"  Status: {obs['status']}")

        return "\n".join(lines)

    def run_monitoring(
        self,
        cycles: int = 6,
        interval_minutes: float = 10,
        continuous: bool = False,
    ) -> int:
        """Run monitoring for specified number of cycles.

        Args:
            cycles: Number of observations to take (ignored if continuous=True)
            interval_minutes: Minutes between observations
            continuous: If True, run indefinitely

        Returns:
            0 if all observations healthy, 1 otherwise
        """
        interval_seconds = interval_minutes * 60

        # Log header
        header = f"\n=== P2P Stability Monitor ===\n"
        header += f"Started: {datetime.datetime.now().strftime('%c')}\n"
        if continuous:
            header += f"Mode: Continuous monitoring\n"
        else:
            header += f"Target: {cycles} checks at {interval_minutes}-minute intervals\n"
        header += f"Total duration: {'indefinite' if continuous else f'{cycles * interval_minutes} minutes'}\n"

        logger.info(header.strip())
        self._log_to_file(header)

        check_num = 0
        unhealthy_count = 0

        try:
            while continuous or check_num < cycles:
                check_num += 1

                obs = self.take_observation()
                formatted = self.format_observation(obs)

                logger.info(formatted)
                self._log_to_file(formatted)

                # Track unhealthy observations
                if obs["status"] in ("ERROR", "CRITICAL", "DEGRADED", "UNSTABLE"):
                    unhealthy_count += 1

                # Sleep until next check (unless last iteration)
                if continuous or check_num < cycles:
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nMonitoring interrupted by user")
            self._log_to_file("\n=== Monitoring interrupted by user ===\n")

        # Log summary
        summary = self._generate_summary()
        logger.info(summary)
        self._log_to_file(summary)

        # Return exit code
        return 1 if unhealthy_count > 0 else 0

    def _generate_summary(self) -> str:
        """Generate summary of all observations."""
        if not self.observations:
            return "\n=== No observations recorded ==="

        lines = ["\n=== Monitoring Summary ==="]
        lines.append(f"Total observations: {len(self.observations)}")

        # Count by status
        status_counts: dict[str, int] = {}
        for obs in self.observations:
            status = obs["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        lines.append("Status distribution:")
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(self.observations)
            lines.append(f"  {status}: {count} ({pct:.0f}%)")

        # Stability assessment
        healthy_count = sum(
            1 for obs in self.observations
            if obs["status"] in ("EXCELLENT", "GOOD", "MODERATE")
        )
        stability_pct = 100 * healthy_count / len(self.observations)
        lines.append(f"\nStability score: {stability_pct:.0f}%")

        if stability_pct >= 90:
            lines.append("Assessment: STABLE - Cluster is meeting stability targets")
        elif stability_pct >= 70:
            lines.append("Assessment: MOSTLY STABLE - Minor intermittent issues")
        elif stability_pct >= 50:
            lines.append("Assessment: UNSTABLE - Significant issues detected")
        else:
            lines.append("Assessment: CRITICAL - Major stability problems")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor P2P cluster stability at regular intervals"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=6,
        help="Number of observation cycles (default: 6)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10,
        help="Minutes between observations (default: 10)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously until interrupted",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="P2P orchestrator host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8770,
        help="P2P orchestrator port (default: 8770)",
    )

    args = parser.parse_args()

    monitor = StabilityMonitor(
        p2p_host=args.host,
        p2p_port=args.port,
    )

    exit_code = monitor.run_monitoring(
        cycles=args.cycles,
        interval_minutes=args.interval,
        continuous=args.continuous,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
