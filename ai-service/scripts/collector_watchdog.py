#!/usr/bin/env python3
"""Streaming Collector Watchdog - Ensures continuous data collection.

DEPRECATED: This script is deprecated in favor of unified_data_sync.py
Please use: python scripts/unified_data_sync.py --watchdog
All functionality has been preserved in the unified service.

This script monitors the streaming data collector and restarts it if it crashes
or becomes unresponsive. Provides high availability for the data sync pipeline.

Features:
1. Monitor collector health via HTTP API
2. Auto-restart on crash or unresponsiveness
3. Leader election for multi-node deployments
4. Graceful handoff between instances

Usage:
    # Run as watchdog daemon (DEPRECATED - use unified_data_sync.py --watchdog instead)
    python scripts/collector_watchdog.py

    # With custom settings
    python scripts/collector_watchdog.py --check-interval 30 --max-restarts 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import warnings

# Emit deprecation warning at runtime
warnings.warn(
    "collector_watchdog.py is deprecated. "
    "Please use: python scripts/unified_data_sync.py --watchdog\n"
    "All functionality has been preserved in the unified service.",
    DeprecationWarning,
    stacklevel=2,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Watchdog] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Import coordination helpers for leader election
sys.path.insert(0, str(AI_SERVICE_ROOT))
from app.coordination.helpers import (
    has_coordination,
    get_registry_safe,
    get_orchestrator_roles,
    warn_if_orchestrator_running,
)

HAS_LEADER_ELECTION = has_coordination()
OrchestratorRole = get_orchestrator_roles()
get_registry = get_registry_safe


@dataclass
class WatchdogConfig:
    """Configuration for the watchdog."""
    collector_url: str = "http://localhost:8772"
    check_interval: int = 30  # Seconds between health checks
    health_timeout: int = 10  # Timeout for health check
    max_restarts: int = 10  # Max restarts before giving up
    restart_cooldown: int = 60  # Seconds between restart attempts
    unresponsive_threshold: int = 3  # Failed checks before restart
    collector_script: str = "scripts/streaming_data_collector.py"
    collector_args: str = ""


@dataclass
class WatchdogState:
    """State tracking for the watchdog."""
    collector_pid: Optional[int] = None
    collector_process: Optional[subprocess.Popen] = None
    last_healthy: float = 0.0
    consecutive_failures: int = 0
    total_restarts: int = 0
    last_restart: float = 0.0
    is_leader: bool = False


class CollectorWatchdog:
    """Monitors and restarts the streaming data collector."""

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self.state = WatchdogState()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._node_id = socket.gethostname()

    async def _check_health(self) -> bool:
        """Check collector health via HTTP API.

        Returns True if healthy.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.collector_url}/health",
                    timeout=aiohttp.ClientTimeout(total=self.config.health_timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status") == "healthy" and data.get("running", False)

        except ImportError:
            # aiohttp not available, try urllib
            import urllib.request
            import urllib.error

            try:
                req = urllib.request.Request(f"{self.config.collector_url}/health")
                with urllib.request.urlopen(req, timeout=self.config.health_timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return data.get("status") == "healthy" and data.get("running", False)
            except Exception:
                return False

        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

        return False

    async def _check_process_alive(self) -> bool:
        """Check if collector process is still running."""
        if self.state.collector_process is None:
            return False

        # Check if process is still alive
        poll_result = self.state.collector_process.poll()
        return poll_result is None

    def _start_collector(self) -> bool:
        """Start the collector process.

        Returns True on success.
        """
        if self.state.collector_process is not None:
            # Check if still running
            if self.state.collector_process.poll() is None:
                logger.warning("Collector already running, killing first")
                self._stop_collector()

        collector_path = AI_SERVICE_ROOT / self.config.collector_script
        if not collector_path.exists():
            logger.error(f"Collector script not found: {collector_path}")
            return False

        try:
            cmd = [sys.executable, str(collector_path)]
            if self.config.collector_args:
                cmd.extend(self.config.collector_args.split())

            logger.info(f"Starting collector: {' '.join(cmd)}")

            self.state.collector_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(AI_SERVICE_ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self.state.collector_pid = self.state.collector_process.pid
            self.state.last_restart = time.time()
            self.state.total_restarts += 1

            logger.info(f"Collector started with PID {self.state.collector_pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start collector: {e}")
            return False

    def _stop_collector(self) -> None:
        """Stop the collector process gracefully."""
        if self.state.collector_process is None:
            return

        try:
            # Send SIGTERM for graceful shutdown
            self.state.collector_process.terminate()

            # Wait up to 30 seconds
            try:
                self.state.collector_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Collector did not stop gracefully, sending SIGKILL")
                self.state.collector_process.kill()
                self.state.collector_process.wait(timeout=10)

        except Exception as e:
            logger.error(f"Error stopping collector: {e}")

        finally:
            self.state.collector_process = None
            self.state.collector_pid = None

    async def _try_become_leader(self) -> bool:
        """Try to become the leader for collector management.

        Returns True if this node is the leader.
        """
        if not HAS_LEADER_ELECTION:
            # No leader election available, assume we're the leader
            return True

        try:
            registry = get_registry()
            # Use WATCHDOG role (or create a new one)
            # For now, use DATA_SYNC role since watchdog manages data collection
            is_leader = registry.try_acquire(OrchestratorRole.DATA_SYNC, self._node_id)

            if is_leader != self.state.is_leader:
                if is_leader:
                    logger.info(f"Became leader for collector management")
                else:
                    logger.info(f"Lost leadership, stopping collector")
                    self._stop_collector()

            self.state.is_leader = is_leader
            return is_leader

        except Exception as e:
            logger.error(f"Leader election error: {e}")
            return self.state.is_leader  # Keep previous state on error

    async def _heartbeat_leader(self) -> None:
        """Send heartbeat to maintain leadership."""
        if not HAS_LEADER_ELECTION or not self.state.is_leader:
            return

        try:
            registry = get_registry()
            registry.heartbeat(OrchestratorRole.DATA_SYNC)
        except Exception as e:
            logger.error(f"Leader heartbeat error: {e}")

    async def run(self) -> None:
        """Main watchdog loop."""
        self._running = True
        logger.info(f"Starting watchdog (check interval: {self.config.check_interval}s)")

        while self._running:
            try:
                # Try to become leader (in multi-node setups)
                is_leader = await self._try_become_leader()

                if not is_leader:
                    # Not the leader, wait and try again
                    await asyncio.sleep(self.config.check_interval)
                    continue

                # Heartbeat to maintain leadership
                await self._heartbeat_leader()

                # Check collector health
                process_alive = await self._check_process_alive()
                http_healthy = await self._check_health()

                if process_alive and http_healthy:
                    # All good
                    self.state.consecutive_failures = 0
                    self.state.last_healthy = time.time()
                    logger.debug("Collector healthy")
                else:
                    self.state.consecutive_failures += 1
                    reason = []
                    if not process_alive:
                        reason.append("process dead")
                    if not http_healthy:
                        reason.append("HTTP unhealthy")
                    logger.warning(f"Collector check failed ({', '.join(reason)}): {self.state.consecutive_failures}/{self.config.unresponsive_threshold}")

                    # Check if we should restart
                    if self.state.consecutive_failures >= self.config.unresponsive_threshold:
                        # Check restart limits
                        if self.state.total_restarts >= self.config.max_restarts:
                            logger.error(f"Max restarts ({self.config.max_restarts}) exceeded, giving up")
                            break

                        # Check cooldown
                        since_restart = time.time() - self.state.last_restart
                        if since_restart < self.config.restart_cooldown:
                            logger.info(f"Restart cooldown active ({since_restart:.0f}s < {self.config.restart_cooldown}s)")
                        else:
                            # Restart collector
                            logger.info("Restarting collector...")
                            self._stop_collector()
                            await asyncio.sleep(2)  # Brief pause

                            if self._start_collector():
                                self.state.consecutive_failures = 0
                                # Give it time to start
                                await asyncio.sleep(10)
                            else:
                                logger.error("Failed to restart collector")

                # Wait for next check
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.check_interval,
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(self.config.check_interval)

        # Cleanup
        logger.info("Watchdog stopping")
        self._stop_collector()

        # Release leadership
        if HAS_LEADER_ELECTION and self.state.is_leader:
            try:
                registry = get_registry()
                registry.release(OrchestratorRole.DATA_SYNC)
            except Exception:
                pass

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()

    def get_status(self) -> Dict[str, Any]:
        """Get watchdog status."""
        return {
            "running": self._running,
            "is_leader": self.state.is_leader,
            "collector_pid": self.state.collector_pid,
            "last_healthy": self.state.last_healthy,
            "consecutive_failures": self.state.consecutive_failures,
            "total_restarts": self.state.total_restarts,
            "last_restart": self.state.last_restart,
            "node_id": self._node_id,
        }


async def run_with_http_api(watchdog: CollectorWatchdog, http_port: int) -> None:
    """Run watchdog with HTTP API for status queries."""
    try:
        from aiohttp import web
    except ImportError:
        logger.warning("aiohttp not available, running without HTTP API")
        await watchdog.run()
        return

    async def handle_health(request):
        return web.json_response({"status": "healthy"})

    async def handle_status(request):
        return web.json_response(watchdog.get_status())

    async def handle_restart(request):
        """POST /restart - Force restart collector."""
        watchdog._stop_collector()
        await asyncio.sleep(2)
        success = watchdog._start_collector()
        return web.json_response({"success": success, "pid": watchdog.state.collector_pid})

    app = web.Application()
    app.router.add_get('/health', handle_health)
    app.router.add_get('/status', handle_status)
    app.router.add_post('/restart', handle_restart)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', http_port)
    await site.start()
    logger.info(f"Watchdog HTTP API listening on port {http_port}")

    try:
        await watchdog.run()
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Streaming Collector Watchdog")
    parser.add_argument("--collector-url", type=str, default="http://localhost:8772", help="Collector HTTP URL")
    parser.add_argument("--check-interval", type=int, default=30, help="Seconds between health checks")
    parser.add_argument("--max-restarts", type=int, default=10, help="Max restarts before giving up")
    parser.add_argument("--restart-cooldown", type=int, default=60, help="Seconds between restart attempts")
    parser.add_argument("--unresponsive-threshold", type=int, default=3, help="Failed checks before restart")
    parser.add_argument("--collector-args", type=str, default="", help="Additional args for collector")
    parser.add_argument("--http-port", type=int, default=8773, help="Watchdog HTTP API port")
    parser.add_argument("--no-http", action="store_true", help="Disable HTTP API")

    args = parser.parse_args()

    config = WatchdogConfig(
        collector_url=args.collector_url,
        check_interval=args.check_interval,
        max_restarts=args.max_restarts,
        restart_cooldown=args.restart_cooldown,
        unresponsive_threshold=args.unresponsive_threshold,
        collector_args=args.collector_args,
    )

    watchdog = CollectorWatchdog(config)

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Shutdown requested")
        watchdog.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.no_http:
        asyncio.run(watchdog.run())
    else:
        asyncio.run(run_with_http_api(watchdog, args.http_port))


if __name__ == "__main__":
    main()
