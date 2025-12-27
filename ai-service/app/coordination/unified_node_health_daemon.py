"""Unified Node Health Daemon.

Main daemon for maintaining cluster health across all providers.
Combines health checking, recovery, and utilization optimization.

Features:
- Continuous health monitoring across Lambda, Vast, Hetzner, AWS
- Automated recovery with escalation ladder
- Utilization optimization to keep all nodes productive
- Config file sync for distributed_hosts.yaml
- Alerting via logs and Slack

Usage:
    # Start daemon
    python -m app.coordination.unified_node_health_daemon

    # Or import and run
    from app.coordination.unified_node_health_daemon import (
        UnifiedNodeHealthDaemon,
        run_daemon,
    )

    daemon = UnifiedNodeHealthDaemon()
    await daemon.run()
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.providers import (
    AWSManager,
    HetznerManager,
    LambdaManager,
    TailscaleManager,
    VastManager,
)
from app.coordination.health_check_orchestrator import (
    ClusterHealthSummary,
    HealthCheckOrchestrator,
    NodeHealthState,
)
from app.coordination.recovery_orchestrator import RecoveryOrchestrator
from app.coordination.utilization_optimizer import UtilizationOptimizer
from app.coordination.p2p_auto_deployer import P2PAutoDeployer, P2PDeploymentConfig

logger = logging.getLogger(__name__)

# Health event emission imports (December 2025 - Phase 21)
try:
    from app.distributed.data_events import (
        emit_p2p_cluster_healthy,
        emit_p2p_cluster_unhealthy,
        emit_node_unhealthy,
    )
    HAS_HEALTH_EVENTS = True
except ImportError:
    HAS_HEALTH_EVENTS = False
    emit_p2p_cluster_healthy = None
    emit_p2p_cluster_unhealthy = None
    emit_node_unhealthy = None


@dataclass
class DaemonConfig:
    """Configuration for the health daemon."""

    # Check intervals (seconds)
    health_check_interval: float = 60.0
    recovery_check_interval: float = 120.0
    optimization_interval: float = 300.0
    config_sync_interval: float = 600.0
    p2p_deploy_interval: float = 300.0  # Check P2P coverage every 5 minutes

    # Thresholds
    min_healthy_percent: float = 80.0  # Alert if < 80% healthy
    max_offline_count: int = 5  # Alert if > 5 nodes offline
    min_p2p_coverage_percent: float = 90.0  # Alert if < 90% have P2P

    # Paths
    hosts_config_path: str = "config/distributed_hosts.yaml"
    alerts_config_path: str = "config/alerts.yaml"

    # Features
    enable_recovery: bool = True
    enable_optimization: bool = True
    enable_config_sync: bool = True
    enable_alerting: bool = True
    enable_p2p_auto_deploy: bool = True  # Auto-deploy P2P to missing nodes

    # P2P port
    p2p_port: int = 8770


class UnifiedNodeHealthDaemon:
    """Main daemon for cluster health maintenance.

    Orchestrates:
    1. Health checks across all providers
    2. Automated recovery of failing nodes
    3. Utilization optimization
    4. Config file synchronization
    5. Alerting
    """

    def __init__(self, config: DaemonConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or DaemonConfig()

        # Provider managers
        self.lambda_mgr = LambdaManager()
        self.vast_mgr = VastManager()
        self.hetzner_mgr = HetznerManager()
        self.aws_mgr = AWSManager()
        self.tailscale_mgr = TailscaleManager()

        # Orchestrators
        self.health_orchestrator = HealthCheckOrchestrator(
            check_interval=self.config.health_check_interval,
            p2p_port=self.config.p2p_port,
        )
        self.recovery_orchestrator = RecoveryOrchestrator(
            health_orchestrator=self.health_orchestrator,
            alerts_config_path=self.config.alerts_config_path,
        )
        self.utilization_optimizer = UtilizationOptimizer(
            health_orchestrator=self.health_orchestrator,
        )

        # P2P auto-deployer
        p2p_config = P2PDeploymentConfig(
            check_interval_seconds=self.config.p2p_deploy_interval,
            p2p_port=self.config.p2p_port,
            min_coverage_percent=self.config.min_p2p_coverage_percent,
            hosts_config_path=self.config.hosts_config_path,
        )
        self.p2p_auto_deployer = P2PAutoDeployer(
            config=p2p_config,
            health_orchestrator=self.health_orchestrator,
        )

        # State
        self._running = False
        self._last_health_check = 0.0
        self._last_recovery_check = 0.0
        self._last_optimization = 0.0
        self._last_config_sync = 0.0
        self._last_p2p_deploy = 0.0

        # Tasks
        self._tasks: list[asyncio.Task] = []

        # Stats
        self._start_time: float = 0.0
        self._health_checks_run = 0
        self._recoveries_attempted = 0
        self._optimizations_run = 0
        self._p2p_deploys_run = 0

    async def run(self) -> None:
        """Run the daemon main loop."""
        self._running = True
        self._start_time = time.time()

        logger.info("=" * 60)
        logger.info("UNIFIED NODE HEALTH DAEMON")
        logger.info("=" * 60)
        logger.info(f"Health check interval: {self.config.health_check_interval}s")
        logger.info(f"Recovery check interval: {self.config.recovery_check_interval}s")
        logger.info(f"Optimization interval: {self.config.optimization_interval}s")
        logger.info(f"Config sync interval: {self.config.config_sync_interval}s")
        logger.info(f"P2P deploy interval: {self.config.p2p_deploy_interval}s")
        logger.info(f"Recovery enabled: {self.config.enable_recovery}")
        logger.info(f"Optimization enabled: {self.config.enable_optimization}")
        logger.info(f"P2P auto-deploy enabled: {self.config.enable_p2p_auto_deploy}")
        logger.info("=" * 60)

        # Initial health check
        await self._run_health_check()

        # Main loop
        while self._running:
            try:
                await self._daemon_cycle()
            except Exception as e:
                logger.error(f"[Daemon] Cycle error: {e}")

            await asyncio.sleep(10)  # Base loop interval

        logger.info("[Daemon] Shutting down...")
        await self._cleanup()

    async def _daemon_cycle(self) -> None:
        """Run one daemon cycle."""
        now = time.time()

        # Health checks
        if now - self._last_health_check >= self.config.health_check_interval:
            await self._run_health_check()

        # Recovery
        if (
            self.config.enable_recovery
            and now - self._last_recovery_check >= self.config.recovery_check_interval
        ):
            await self._run_recovery_check()

        # Optimization
        if (
            self.config.enable_optimization
            and now - self._last_optimization >= self.config.optimization_interval
        ):
            await self._run_optimization()

        # Config sync
        if (
            self.config.enable_config_sync
            and now - self._last_config_sync >= self.config.config_sync_interval
        ):
            await self._run_config_sync()

        # P2P auto-deployment
        if (
            self.config.enable_p2p_auto_deploy
            and now - self._last_p2p_deploy >= self.config.p2p_deploy_interval
        ):
            await self._run_p2p_deploy()

    async def _run_health_check(self) -> None:
        """Run health check cycle."""
        logger.info("[Daemon] Running health check cycle...")
        self._last_health_check = time.time()
        self._health_checks_run += 1

        try:
            await self.health_orchestrator.run_full_health_check()
            summary = await self.health_orchestrator.get_cluster_health()

            # Check if we need to alert
            if self.config.enable_alerting:
                await self._check_alerts(summary)

        except Exception as e:
            logger.error(f"[Daemon] Health check failed: {e}")

    async def _run_recovery_check(self) -> None:
        """Run recovery on unhealthy nodes."""
        logger.info("[Daemon] Running recovery check...")
        self._last_recovery_check = time.time()

        try:
            results = await self.recovery_orchestrator.recover_all_unhealthy()

            successful = sum(1 for r in results if r.success)
            if results:
                logger.info(
                    f"[Daemon] Recovery: {successful}/{len(results)} successful"
                )
                self._recoveries_attempted += len(results)

        except Exception as e:
            logger.error(f"[Daemon] Recovery check failed: {e}")

    async def _run_optimization(self) -> None:
        """Run utilization optimization."""
        logger.info("[Daemon] Running utilization optimization...")
        self._last_optimization = time.time()
        self._optimizations_run += 1

        try:
            results = await self.utilization_optimizer.optimize_cluster()

            successful = sum(1 for r in results if r.success)
            if results:
                logger.info(
                    f"[Daemon] Optimization: {successful}/{len(results)} jobs spawned"
                )

        except Exception as e:
            logger.error(f"[Daemon] Optimization failed: {e}")

    async def _run_config_sync(self) -> None:
        """Sync distributed_hosts.yaml with current state."""
        logger.info("[Daemon] Syncing config file...")
        self._last_config_sync = time.time()

        try:
            await self._sync_hosts_config()
        except Exception as e:
            logger.error(f"[Daemon] Config sync failed: {e}")

    async def _run_p2p_deploy(self) -> None:
        """Check P2P coverage and deploy to nodes missing it."""
        logger.info("[Daemon] Running P2P auto-deployment check...")
        self._last_p2p_deploy = time.time()
        self._p2p_deploys_run += 1

        try:
            report = await self.p2p_auto_deployer.check_and_deploy()

            # Alert if coverage below threshold
            if (
                self.config.enable_alerting
                and report.coverage_percent < self.config.min_p2p_coverage_percent
            ):
                await self._send_alert(
                    f"⚠️ P2P coverage low: {report.coverage_percent:.0f}% "
                    f"({report.nodes_with_p2p}/{report.total_nodes} nodes with P2P). "
                    f"Missing: {report.nodes_needing_deployment[:5]}"
                )

        except Exception as e:
            logger.error(f"[Daemon] P2P deployment check failed: {e}")

    async def _sync_hosts_config(self) -> None:
        """Update distributed_hosts.yaml with current node health."""
        config_path = Path(self.config.hosts_config_path)

        if not config_path.exists():
            logger.warning(f"[Daemon] Config not found: {config_path}")
            return

        try:
            # Load existing config
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            hosts = config.get("hosts", {})
            updated = False

            # Update each host with current health state
            for host_id, host_info in hosts.items():
                health = self.health_orchestrator.get_node_health(host_id)
                if health:
                    # Update health tracking fields
                    host_info["last_health_check"] = datetime.now().isoformat()
                    host_info["health_state"] = health.state.value
                    host_info["consecutive_failures"] = health.consecutive_failures

                    # Update Tailscale IP if discovered
                    if health.instance and health.instance.tailscale_ip:
                        if host_info.get("tailscale_ip") != health.instance.tailscale_ip:
                            host_info["tailscale_ip"] = health.instance.tailscale_ip
                            updated = True

            # Write if updated
            if updated:
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info("[Daemon] Updated hosts config")

        except Exception as e:
            logger.error(f"[Daemon] Config sync error: {e}")

    async def _check_alerts(self, summary: ClusterHealthSummary) -> None:
        """Check if alerts should be sent."""
        active_nodes = summary.total_nodes - summary.retired
        if active_nodes == 0:
            return

        healthy_percent = (summary.healthy + summary.degraded) / active_nodes * 100
        healthy_nodes = summary.healthy + summary.degraded

        # Emit cluster health events (December 2025 - Phase 21)
        if HAS_HEALTH_EVENTS:
            await self._emit_cluster_health_events(
                healthy_percent=healthy_percent,
                healthy_nodes=healthy_nodes,
                active_nodes=active_nodes,
                summary=summary,
            )

        # Alert if below threshold
        if healthy_percent < self.config.min_healthy_percent:
            await self._send_alert(
                f"⚠️ Cluster health low: {healthy_percent:.0f}% "
                f"({summary.healthy + summary.degraded}/{active_nodes} nodes available)"
            )

        # Alert if too many offline
        if summary.offline > self.config.max_offline_count:
            await self._send_alert(
                f"🔴 High offline count: {summary.offline} nodes offline"
            )

    async def _emit_cluster_health_events(
        self,
        healthy_percent: float,
        healthy_nodes: int,
        active_nodes: int,
        summary: ClusterHealthSummary,
    ) -> None:
        """Emit cluster health events for event-driven coordination."""
        try:
            if healthy_percent >= self.config.min_healthy_percent:
                await emit_p2p_cluster_healthy(
                    healthy_nodes=healthy_nodes,
                    node_count=active_nodes,
                    source="unified_node_health_daemon",
                )
            else:
                alerts = []
                if summary.offline > 0:
                    alerts.append(f"{summary.offline} nodes offline")
                if summary.unhealthy > 0:
                    alerts.append(f"{summary.unhealthy} nodes unhealthy")
                await emit_p2p_cluster_unhealthy(
                    healthy_nodes=healthy_nodes,
                    node_count=active_nodes,
                    alerts=alerts,
                    source="unified_node_health_daemon",
                )
        except Exception as e:
            logger.debug(f"[Daemon] Failed to emit cluster health events: {e}")

    async def _send_alert(self, message: str) -> None:
        """Send alert via configured channels."""
        logger.warning(f"[ALERT] {message}")

        # Slack alert via recovery orchestrator
        if self.recovery_orchestrator.slack_webhook_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.recovery_orchestrator.slack_webhook_url,
                        json={"text": message},
                        timeout=aiohttp.ClientTimeout(total=10),
                    )
            except Exception as e:
                logger.error(f"[Daemon] Slack alert failed: {e}")

    async def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        await self.health_orchestrator.stop()
        await self.lambda_mgr.close()

    def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0

        # Get latest P2P coverage
        p2p_coverage = None
        latest_report = self.p2p_auto_deployer.get_latest_coverage()
        if latest_report:
            p2p_coverage = {
                "coverage_percent": latest_report.coverage_percent,
                "nodes_with_p2p": latest_report.nodes_with_p2p,
                "total_nodes": latest_report.total_nodes,
                "nodes_needing_deployment": latest_report.nodes_needing_deployment,
            }

        return {
            "uptime_seconds": uptime,
            "health_checks_run": self._health_checks_run,
            "recoveries_attempted": self._recoveries_attempted,
            "optimizations_run": self._optimizations_run,
            "p2p_deploys_run": self._p2p_deploys_run,
            "p2p_coverage": p2p_coverage,
            "running": self._running,
        }


async def run_daemon(config: DaemonConfig | None = None) -> None:
    """Run the unified node health daemon.

    Args:
        config: Daemon configuration
    """
    daemon = UnifiedNodeHealthDaemon(config)

    # Handle signals
    # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("[Daemon] Received shutdown signal")
        daemon.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await daemon.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Node Health Daemon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--health-interval",
        type=float,
        default=60.0,
        help="Health check interval (seconds)",
    )
    parser.add_argument(
        "--recovery-interval",
        type=float,
        default=120.0,
        help="Recovery check interval (seconds)",
    )
    parser.add_argument(
        "--optimization-interval",
        type=float,
        default=300.0,
        help="Optimization interval (seconds)",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable automatic recovery",
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable utilization optimization",
    )
    parser.add_argument(
        "--hosts-config",
        type=str,
        default="config/distributed_hosts.yaml",
        help="Path to hosts config file",
    )
    parser.add_argument(
        "--p2p-port",
        type=int,
        default=8770,
        help="P2P daemon port",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config
    config = DaemonConfig(
        health_check_interval=args.health_interval,
        recovery_check_interval=args.recovery_interval,
        optimization_interval=args.optimization_interval,
        enable_recovery=not args.no_recovery,
        enable_optimization=not args.no_optimization,
        hosts_config_path=args.hosts_config,
        p2p_port=args.p2p_port,
    )

    # Run
    try:
        asyncio.run(run_daemon(config))
    except KeyboardInterrupt:
        logger.info("[Daemon] Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
