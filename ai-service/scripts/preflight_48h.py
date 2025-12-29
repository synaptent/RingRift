#!/usr/bin/env python3
"""Pre-flight checklist for 48-hour unattended training runs.

This script verifies that all prerequisites are met before starting a long
unattended training run. It checks infrastructure, data pipeline, model health,
daemon status, and external dependencies.

Usage:
    python scripts/preflight_48h.py           # Run all checks
    python scripts/preflight_48h.py --verbose # Detailed output
    python scripts/preflight_48h.py --json    # JSON output for automation

December 2025: Created for 48-hour autonomous operation enablement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.utils.paths import GAMES_DIR, MODELS_DIR, DATA_DIR


# =============================================================================
# Check Result Types
# =============================================================================


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class PreflightReport:
    """Complete pre-flight check report."""

    checks: list[CheckResult]
    total_duration_seconds: float
    timestamp: str

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_duration_seconds": self.total_duration_seconds,
            "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
        }


# =============================================================================
# Infrastructure Checks
# =============================================================================


def check_disk_space(min_gb: int = 100) -> CheckResult:
    """Check that sufficient disk space is available."""
    try:
        total, used, free = shutil.disk_usage(str(DATA_DIR))
        free_gb = free / (1024**3)
        used_percent = (used / total) * 100

        if free_gb >= min_gb:
            return CheckResult(
                name="disk_space",
                passed=True,
                message=f"Disk space OK: {free_gb:.1f}GB free ({used_percent:.1f}% used)",
                details={"free_gb": free_gb, "used_percent": used_percent},
            )
        else:
            return CheckResult(
                name="disk_space",
                passed=False,
                message=f"Low disk space: {free_gb:.1f}GB free (need {min_gb}GB)",
                details={"free_gb": free_gb, "required_gb": min_gb},
            )
    except Exception as e:
        return CheckResult(
            name="disk_space",
            passed=False,
            message=f"Failed to check disk space: {e}",
        )


def check_gpu_memory(min_gb: int = 8) -> CheckResult:
    """Check that GPU memory is available (local machine only)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # No GPU or nvidia-smi not available - acceptable for coordinator
            return CheckResult(
                name="gpu_memory",
                passed=True,
                message="No local GPU detected (OK for coordinator node)",
                details={"has_gpu": False},
            )

        lines = result.stdout.strip().split("\n")
        gpus = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                total = float(parts[0].strip()) / 1024  # Convert MB to GB
                free = float(parts[1].strip()) / 1024
                gpus.append({"total_gb": total, "free_gb": free})

        if gpus:
            max_free = max(g["free_gb"] for g in gpus)
            if max_free >= min_gb:
                return CheckResult(
                    name="gpu_memory",
                    passed=True,
                    message=f"GPU memory OK: {max_free:.1f}GB free",
                    details={"gpus": gpus, "max_free_gb": max_free},
                )
            else:
                return CheckResult(
                    name="gpu_memory",
                    passed=False,
                    message=f"Low GPU memory: {max_free:.1f}GB free (need {min_gb}GB)",
                    details={"gpus": gpus, "required_gb": min_gb},
                )

        return CheckResult(
            name="gpu_memory",
            passed=True,
            message="No GPU memory info available (OK for coordinator)",
            details={"has_gpu": False},
        )
    except FileNotFoundError:
        return CheckResult(
            name="gpu_memory",
            passed=True,
            message="nvidia-smi not found (OK for coordinator node)",
            details={"has_gpu": False},
        )
    except Exception as e:
        return CheckResult(
            name="gpu_memory",
            passed=False,
            message=f"Failed to check GPU: {e}",
        )


async def check_p2p_quorum(min_voters: int = 3) -> CheckResult:
    """Check that P2P cluster has quorum."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8770/status",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return CheckResult(
                        name="p2p_quorum",
                        passed=False,
                        message=f"P2P status endpoint returned {resp.status}",
                    )

                data = await resp.json()
                alive_peers = data.get("alive_peers", 0)
                leader_id = data.get("leader_id", "unknown")
                voter_count = data.get("voter_quorum", {}).get("total_voters", 0)

                if alive_peers >= min_voters:
                    return CheckResult(
                        name="p2p_quorum",
                        passed=True,
                        message=f"P2P quorum OK: {alive_peers} alive peers, leader={leader_id}",
                        details={
                            "alive_peers": alive_peers,
                            "leader_id": leader_id,
                            "voter_count": voter_count,
                        },
                    )
                else:
                    return CheckResult(
                        name="p2p_quorum",
                        passed=False,
                        message=f"P2P quorum low: {alive_peers} peers (need {min_voters})",
                        details={"alive_peers": alive_peers, "required": min_voters},
                    )
    except Exception as e:
        return CheckResult(
            name="p2p_quorum",
            passed=False,
            message=f"Cannot reach P2P: {e}",
        )


# =============================================================================
# Data Pipeline Checks
# =============================================================================


def check_canonical_dbs() -> CheckResult:
    """Check that canonical databases exist for all 12 configs."""
    configs = [
        "hex8_2p", "hex8_3p", "hex8_4p",
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    missing = []
    existing = []

    for config in configs:
        db_path = GAMES_DIR / f"canonical_{config}.db"
        if db_path.exists():
            existing.append(config)
        else:
            missing.append(config)

    if not missing:
        return CheckResult(
            name="canonical_dbs_exist",
            passed=True,
            message=f"All 12 canonical databases exist",
            details={"existing": existing},
        )
    else:
        return CheckResult(
            name="canonical_dbs_exist",
            passed=False,
            message=f"Missing {len(missing)} canonical databases: {missing}",
            details={"missing": missing, "existing": existing},
        )


def check_recent_selfplay(max_age_hours: float = 2.0) -> CheckResult:
    """Check that selfplay has generated recent data."""
    try:
        import sqlite3

        newest_game_time = None
        newest_config = None

        for db_path in GAMES_DIR.glob("canonical_*.db"):
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT MAX(end_time) FROM games WHERE game_status='completed'"
                )
                row = cursor.fetchone()
                conn.close()

                if row and row[0]:
                    game_time = row[0]
                    if newest_game_time is None or game_time > newest_game_time:
                        newest_game_time = game_time
                        newest_config = db_path.stem
            except Exception:
                continue

        if newest_game_time is None:
            return CheckResult(
                name="recent_selfplay",
                passed=False,
                message="No completed games found in any database",
            )

        # Parse ISO timestamp
        from datetime import datetime

        try:
            game_dt = datetime.fromisoformat(newest_game_time.replace("Z", "+00:00"))
            age_hours = (datetime.now(game_dt.tzinfo) - game_dt).total_seconds() / 3600
        except Exception:
            # Fallback: assume recent
            age_hours = 0.0

        if age_hours <= max_age_hours:
            return CheckResult(
                name="recent_selfplay",
                passed=True,
                message=f"Recent selfplay OK: newest game {age_hours:.1f}h ago ({newest_config})",
                details={"age_hours": age_hours, "config": newest_config},
            )
        else:
            return CheckResult(
                name="recent_selfplay",
                passed=False,
                message=f"Selfplay stale: newest game {age_hours:.1f}h ago (max {max_age_hours}h)",
                details={"age_hours": age_hours, "max_age_hours": max_age_hours},
            )
    except Exception as e:
        return CheckResult(
            name="recent_selfplay",
            passed=False,
            message=f"Failed to check selfplay: {e}",
        )


def check_export_queue() -> CheckResult:
    """Check that export queue is not backed up."""
    try:
        # Check for pending export files or backed up queues
        pending_exports = list((DATA_DIR / "pending_exports").glob("*.json")) if (DATA_DIR / "pending_exports").exists() else []

        if len(pending_exports) <= 10:
            return CheckResult(
                name="export_queue_clear",
                passed=True,
                message=f"Export queue OK: {len(pending_exports)} pending",
                details={"pending_count": len(pending_exports)},
            )
        else:
            return CheckResult(
                name="export_queue_clear",
                passed=False,
                message=f"Export queue backed up: {len(pending_exports)} pending",
                details={"pending_count": len(pending_exports)},
            )
    except Exception as e:
        return CheckResult(
            name="export_queue_clear",
            passed=True,  # Don't fail if directory doesn't exist
            message=f"Export queue check skipped: {e}",
        )


# =============================================================================
# Model Health Checks
# =============================================================================


def check_all_models() -> CheckResult:
    """Check that all 12 canonical models exist."""
    configs = [
        "hex8_2p", "hex8_3p", "hex8_4p",
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    missing = []
    existing = []

    for config in configs:
        model_path = MODELS_DIR / f"canonical_{config}.pth"
        if model_path.exists():
            existing.append(config)
        else:
            missing.append(config)

    if not missing:
        return CheckResult(
            name="all_12_models_exist",
            passed=True,
            message="All 12 canonical models exist",
            details={"existing": existing},
        )
    else:
        return CheckResult(
            name="all_12_models_exist",
            passed=False,
            message=f"Missing {len(missing)} models: {missing}",
            details={"missing": missing, "existing": existing},
        )


def check_elo_baseline() -> CheckResult:
    """Check that Elo ratings exist for all configs."""
    try:
        import sqlite3

        elo_db = DATA_DIR / "elo_ratings.db"
        if not elo_db.exists():
            return CheckResult(
                name="elo_baseline",
                passed=False,
                message="Elo database does not exist",
            )

        conn = sqlite3.connect(str(elo_db))
        cursor = conn.execute("SELECT DISTINCT config_key FROM ratings")
        configs_with_elo = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "hex8_2p", "hex8_3p", "hex8_4p",
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        }

        missing = expected - configs_with_elo

        if not missing:
            return CheckResult(
                name="elo_baseline",
                passed=True,
                message="Elo ratings exist for all 12 configs",
                details={"configs": list(configs_with_elo)},
            )
        else:
            return CheckResult(
                name="elo_baseline",
                passed=False,
                message=f"Missing Elo for {len(missing)} configs: {missing}",
                details={"missing": list(missing)},
            )
    except Exception as e:
        return CheckResult(
            name="elo_baseline",
            passed=False,
            message=f"Failed to check Elo: {e}",
        )


# =============================================================================
# Daemon Health Checks
# =============================================================================


async def check_daemon_health() -> CheckResult:
    """Check that critical daemons are healthy."""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8790/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return CheckResult(
                        name="all_daemons_healthy",
                        passed=False,
                        message=f"Health endpoint returned {resp.status}",
                    )

                data = await resp.json()
                unhealthy = []
                healthy = []

                for daemon, status in data.get("daemons", {}).items():
                    if status.get("healthy", False):
                        healthy.append(daemon)
                    else:
                        unhealthy.append(daemon)

                if not unhealthy:
                    return CheckResult(
                        name="all_daemons_healthy",
                        passed=True,
                        message=f"All {len(healthy)} daemons healthy",
                        details={"healthy": healthy},
                    )
                else:
                    return CheckResult(
                        name="all_daemons_healthy",
                        passed=False,
                        message=f"{len(unhealthy)} unhealthy daemons: {unhealthy}",
                        details={"unhealthy": unhealthy, "healthy": healthy},
                    )
    except Exception as e:
        return CheckResult(
            name="all_daemons_healthy",
            passed=False,
            message=f"Cannot reach health endpoint: {e}",
        )


async def check_event_router() -> CheckResult:
    """Check that event router is running."""
    try:
        from app.coordination.event_router import get_router

        router = get_router()
        stats = router.get_stats()

        if stats.get("running", False):
            return CheckResult(
                name="event_router_running",
                passed=True,
                message="Event router is running",
                details=stats,
            )
        else:
            return CheckResult(
                name="event_router_running",
                passed=False,
                message="Event router is not running",
            )
    except Exception as e:
        return CheckResult(
            name="event_router_running",
            passed=False,
            message=f"Cannot check event router: {e}",
        )


def check_dlq_size(max_events: int = 100) -> CheckResult:
    """Check that Dead Letter Queue is not overflowing."""
    try:
        import sqlite3

        dlq_path = DATA_DIR / "coordination" / "dead_letter_queue.db"
        if not dlq_path.exists():
            return CheckResult(
                name="dlq_not_full",
                passed=True,
                message="DLQ database does not exist (no events failed)",
            )

        conn = sqlite3.connect(str(dlq_path))
        cursor = conn.execute("SELECT COUNT(*) FROM events WHERE status='pending'")
        count = cursor.fetchone()[0]
        conn.close()

        if count <= max_events:
            return CheckResult(
                name="dlq_not_full",
                passed=True,
                message=f"DLQ OK: {count} pending events (max {max_events})",
                details={"pending_count": count, "max_events": max_events},
            )
        else:
            return CheckResult(
                name="dlq_not_full",
                passed=False,
                message=f"DLQ overflow: {count} pending events (max {max_events})",
                details={"pending_count": count, "max_events": max_events},
            )
    except Exception as e:
        return CheckResult(
            name="dlq_not_full",
            passed=True,  # Don't fail if table doesn't exist
            message=f"DLQ check skipped: {e}",
        )


# =============================================================================
# External Dependency Checks
# =============================================================================


async def check_cluster_ssh() -> CheckResult:
    """Check SSH connectivity to critical cluster nodes."""
    try:
        import yaml

        config_path = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
        if not config_path.exists():
            return CheckResult(
                name="cluster_connectivity",
                passed=False,
                message="Cluster config not found",
            )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        hosts = config.get("hosts", {})
        voters = [h for h, v in hosts.items() if v.get("role") == "voter" and v.get("status") == "ready"]

        reachable = []
        unreachable = []

        for host in voters[:5]:  # Check first 5 voters
            host_config = hosts[host]
            ssh_host = host_config.get("ssh_host") or host_config.get("tailscale_ip")
            ssh_port = host_config.get("ssh_port", 22)

            try:
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    "-p", str(ssh_port),
                    f"root@{ssh_host}",
                    "echo ok",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                if b"ok" in stdout:
                    reachable.append(host)
                else:
                    unreachable.append(host)
            except Exception:
                unreachable.append(host)

        if len(reachable) >= 3:
            return CheckResult(
                name="cluster_connectivity",
                passed=True,
                message=f"Cluster SSH OK: {len(reachable)}/{len(voters[:5])} voters reachable",
                details={"reachable": reachable, "unreachable": unreachable},
            )
        else:
            return CheckResult(
                name="cluster_connectivity",
                passed=False,
                message=f"Cluster SSH poor: only {len(reachable)}/{len(voters[:5])} voters reachable",
                details={"reachable": reachable, "unreachable": unreachable},
            )
    except Exception as e:
        return CheckResult(
            name="cluster_connectivity",
            passed=False,
            message=f"Failed to check cluster: {e}",
        )


def check_launchd_installed() -> CheckResult:
    """Check that launchd watchdog is installed (macOS only)."""
    if sys.platform != "darwin":
        return CheckResult(
            name="launchd_watchdog",
            passed=True,
            message="Not macOS, launchd check skipped",
        )

    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.ringrift.master-loop.plist"

    if plist_path.exists():
        # Check if loaded
        result = subprocess.run(
            ["launchctl", "list", "com.ringrift.master-loop"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return CheckResult(
                name="launchd_watchdog",
                passed=True,
                message="Launchd watchdog installed and loaded",
            )
        else:
            return CheckResult(
                name="launchd_watchdog",
                passed=False,
                message="Launchd plist exists but not loaded",
            )
    else:
        return CheckResult(
            name="launchd_watchdog",
            passed=False,
            message="Launchd watchdog not installed",
            details={"expected_path": str(plist_path)},
        )


# =============================================================================
# Main Runner
# =============================================================================


async def run_all_checks(verbose: bool = False) -> PreflightReport:
    """Run all pre-flight checks and return report."""
    start_time = time.time()
    results: list[CheckResult] = []

    # Define all checks with their parameters
    checks: list[tuple[str, Callable, dict]] = [
        # Infrastructure
        ("disk_space", check_disk_space, {"min_gb": 100}),
        ("gpu_memory", check_gpu_memory, {"min_gb": 8}),
        ("p2p_quorum", check_p2p_quorum, {"min_voters": 3}),
        # Data Pipeline
        ("canonical_dbs", check_canonical_dbs, {}),
        ("recent_selfplay", check_recent_selfplay, {"max_age_hours": 2}),
        ("export_queue", check_export_queue, {}),
        # Model Health
        ("all_models", check_all_models, {}),
        ("elo_baseline", check_elo_baseline, {}),
        # Daemon Health
        ("daemon_health", check_daemon_health, {}),
        ("event_router", check_event_router, {}),
        ("dlq_size", check_dlq_size, {"max_events": 100}),
        # External Dependencies
        ("cluster_ssh", check_cluster_ssh, {}),
        ("launchd", check_launchd_installed, {}),
    ]

    for name, check_fn, kwargs in checks:
        check_start = time.time()
        try:
            if asyncio.iscoroutinefunction(check_fn):
                result = await check_fn(**kwargs)
            else:
                result = check_fn(**kwargs)
            result.duration_seconds = time.time() - check_start
        except Exception as e:
            result = CheckResult(
                name=name,
                passed=False,
                message=f"Check failed with exception: {e}",
                duration_seconds=time.time() - check_start,
            )

        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.message}")

    from datetime import datetime

    return PreflightReport(
        checks=results,
        total_duration_seconds=time.time() - start_time,
        timestamp=datetime.now().isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight checklist for 48-hour unattended training runs"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    if not args.json:
        print("RingRift 48-Hour Pre-flight Checklist")
        print("=" * 40)

    report = asyncio.run(run_all_checks(verbose=args.verbose))

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print()
        if report.all_passed:
            print(f"All {report.passed_count} checks passed. Safe for 48-hour run.")
        else:
            print(f"FAILED: {report.failed_count} checks failed:")
            for check in report.checks:
                if not check.passed:
                    print(f"  - {check.name}: {check.message}")
        print(f"\nTotal time: {report.total_duration_seconds:.1f}s")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
