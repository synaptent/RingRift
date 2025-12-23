#!/usr/bin/env python3
"""Cluster Supervisor - Auto-restart daemon for RingRift orchestration services.

This supervisor ensures critical cluster services stay running by:
1. Monitoring P2P orchestrator, unified_ai_loop, and other daemons
2. Auto-restarting crashed processes with exponential backoff
3. Logging all restarts with detailed diagnostics
4. Integrating with systemd for proper service management
5. Providing HTTP endpoint for status monitoring

Usage:
    # Start supervisor (foreground)
    python scripts/cluster_supervisor.py

    # Start with custom config
    python scripts/cluster_supervisor.py --config config/supervisor.yaml

    # Check status
    python scripts/cluster_supervisor.py --status

    # With systemd (recommended)
    systemctl start ringrift-supervisor
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import systemd notification
try:
    import sdnotify
    HAS_SYSTEMD = True
except ImportError:
    HAS_SYSTEMD = False
    sdnotify = None

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

DEFAULT_HTTP_PORT = 8775
CHECK_INTERVAL = 10  # seconds
MAX_RESTART_ATTEMPTS = 10
INITIAL_BACKOFF = 5  # seconds
MAX_BACKOFF = 300  # 5 minutes


class ProcessState(str, Enum):
    """State of a managed process."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    BACKOFF = "backoff"


@dataclass
class ManagedProcess:
    """Configuration and state for a managed process."""
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    enabled: bool = True
    critical: bool = True  # If critical, supervisor exits if it can't be restarted

    # Runtime state
    process: Optional[subprocess.Popen] = None
    state: ProcessState = ProcessState.STOPPED
    pid: Optional[int] = None
    started_at: Optional[float] = None
    restart_count: int = 0
    last_restart: Optional[float] = None
    last_exit_code: Optional[int] = None
    backoff_until: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "script": self.script,
            "state": self.state.value,
            "pid": self.pid,
            "started_at": self.started_at,
            "uptime_seconds": time.time() - self.started_at if self.started_at else None,
            "restart_count": self.restart_count,
            "last_restart": self.last_restart,
            "last_exit_code": self.last_exit_code,
            "enabled": self.enabled,
            "critical": self.critical,
        }


class ClusterSupervisor:
    """Main supervisor daemon that manages cluster processes."""

    def __init__(
        self,
        http_port: int = DEFAULT_HTTP_PORT,
        node_id: Optional[str] = None,
        ringrift_path: Optional[str] = None,
    ):
        self.http_port = http_port
        self.node_id = node_id or socket.gethostname()
        self.ringrift_path = Path(ringrift_path) if ringrift_path else PROJECT_ROOT.parent

        self.managed_processes: dict[str, ManagedProcess] = {}
        self.running = False
        self.start_time = time.time()

        # Systemd notifier
        self.sd_notifier = sdnotify.SystemdNotifier() if HAS_SYSTEMD else None

        # HTTP server for status
        self.http_server: Optional[HTTPServer] = None
        self.http_thread: Optional[Thread] = None

        # Setup default managed processes
        self._setup_default_processes()

    def _setup_default_processes(self):
        """Configure default processes to manage."""
        ai_service_path = self.ringrift_path / "ai-service"

        # P2P Orchestrator
        self.managed_processes["p2p_orchestrator"] = ManagedProcess(
            name="p2p_orchestrator",
            script="scripts/p2p_orchestrator.py",
            args=["--node-id", self.node_id, "--port", "8770", "--supervised"],
            env={"PYTHONPATH": str(ai_service_path)},
            working_dir=str(ai_service_path),
            critical=True,
        )

        # Unified AI Loop (only on controller nodes)
        self.managed_processes["unified_ai_loop"] = ManagedProcess(
            name="unified_ai_loop",
            script="scripts/unified_ai_loop.py",
            args=["--foreground", "--verbose"],
            env={"PYTHONPATH": str(ai_service_path)},
            working_dir=str(ai_service_path),
            enabled=self._is_controller_node(),
            critical=False,
        )

        # Node Resilience (fallback selfplay)
        self.managed_processes["node_resilience"] = ManagedProcess(
            name="node_resilience",
            script="scripts/node_resilience.py",
            args=[],
            env={"PYTHONPATH": str(ai_service_path)},
            working_dir=str(ai_service_path),
            critical=False,
        )

    def _is_controller_node(self) -> bool:
        """Check if this is a controller node that should run unified_ai_loop."""
        # Controller nodes are typically high-memory nodes or specifically designated
        controller_nodes = ["lambda-2xh100", "lambda-gh200-e", "mac-studio"]
        return self.node_id in controller_nodes or "controller" in self.node_id.lower()

    def _get_python_executable(self) -> str:
        """Get the Python executable path."""
        ai_service_path = self.ringrift_path / "ai-service"
        venv_python = ai_service_path / "venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        return sys.executable

    def _calculate_backoff(self, restart_count: int) -> float:
        """Calculate exponential backoff time."""
        backoff = INITIAL_BACKOFF * (2 ** min(restart_count, 6))
        return min(backoff, MAX_BACKOFF)

    async def start_process(self, proc: ManagedProcess) -> bool:
        """Start a managed process."""
        if not proc.enabled:
            logger.info(f"Process {proc.name} is disabled, skipping")
            return False

        if proc.state == ProcessState.RUNNING and proc.process and proc.process.poll() is None:
            logger.debug(f"Process {proc.name} already running (PID {proc.pid})")
            return True

        # Check backoff
        if proc.backoff_until and time.time() < proc.backoff_until:
            remaining = proc.backoff_until - time.time()
            logger.debug(f"Process {proc.name} in backoff, {remaining:.1f}s remaining")
            return False

        proc.state = ProcessState.STARTING
        logger.info(f"Starting process: {proc.name}")

        try:
            python_exec = self._get_python_executable()
            working_dir = proc.working_dir or str(self.ringrift_path / "ai-service")

            # Build command
            cmd = [python_exec, proc.script] + proc.args

            # Build environment
            env = os.environ.copy()
            env.update(proc.env)

            # Start process
            log_file = Path(working_dir) / "logs" / f"{proc.name}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as log_handle:
                log_handle.write(f"\n{'='*60}\n")
                log_handle.write(f"Supervisor starting {proc.name} at {datetime.now().isoformat()}\n")
                log_handle.write(f"Command: {' '.join(cmd)}\n")
                log_handle.write(f"{'='*60}\n\n")

                proc.process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=working_dir,
                    preexec_fn=os.setsid,  # Create new process group for clean shutdown
                )

            proc.pid = proc.process.pid
            proc.started_at = time.time()
            proc.state = ProcessState.RUNNING
            proc.last_restart = time.time()
            proc.backoff_until = None

            logger.info(f"Started {proc.name} with PID {proc.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start {proc.name}: {e}")
            proc.state = ProcessState.FAILED
            proc.restart_count += 1
            proc.backoff_until = time.time() + self._calculate_backoff(proc.restart_count)
            return False

    async def stop_process(self, proc: ManagedProcess, timeout: float = 10.0) -> bool:
        """Stop a managed process gracefully."""
        if not proc.process or proc.process.poll() is not None:
            proc.state = ProcessState.STOPPED
            proc.pid = None
            return True

        proc.state = ProcessState.STOPPING
        logger.info(f"Stopping process: {proc.name} (PID {proc.pid})")

        try:
            # Send SIGTERM first
            os.killpg(os.getpgid(proc.process.pid), signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                proc.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                logger.warning(f"Process {proc.name} didn't stop gracefully, sending SIGKILL")
                os.killpg(os.getpgid(proc.process.pid), signal.SIGKILL)
                proc.process.wait(timeout=5)

            proc.last_exit_code = proc.process.returncode
            proc.state = ProcessState.STOPPED
            proc.pid = None
            logger.info(f"Stopped {proc.name} (exit code: {proc.last_exit_code})")
            return True

        except Exception as e:
            logger.error(f"Error stopping {proc.name}: {e}")
            return False

    async def check_and_restart(self, proc: ManagedProcess) -> bool:
        """Check if process is running and restart if needed."""
        if not proc.enabled:
            return True

        if proc.process and proc.process.poll() is None:
            # Process is running
            if proc.state != ProcessState.RUNNING:
                proc.state = ProcessState.RUNNING
            return True

        # Process has exited
        if proc.process:
            proc.last_exit_code = proc.process.returncode
            logger.warning(
                f"Process {proc.name} exited with code {proc.last_exit_code} "
                f"(restart #{proc.restart_count})"
            )

        # Check restart limits
        if proc.restart_count >= MAX_RESTART_ATTEMPTS:
            if proc.critical:
                logger.critical(
                    f"Critical process {proc.name} exceeded restart limit ({MAX_RESTART_ATTEMPTS}), "
                    "supervisor will exit"
                )
                self.running = False
                return False
            else:
                logger.error(
                    f"Process {proc.name} exceeded restart limit, disabling"
                )
                proc.enabled = False
                proc.state = ProcessState.FAILED
                return False

        # Attempt restart with backoff
        proc.restart_count += 1
        if await self.start_process(proc):
            return True

        return False

    async def monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            for name, proc in self.managed_processes.items():
                if proc.enabled:
                    await self.check_and_restart(proc)

            # Notify systemd watchdog
            if self.sd_notifier:
                self.sd_notifier.notify("WATCHDOG=1")

            await asyncio.sleep(CHECK_INTERVAL)

    def _start_http_server(self):
        """Start HTTP status server."""
        supervisor = self

        class StatusHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

            def do_GET(self):
                if self.path == "/status" or self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    status = supervisor.get_status()
                    self.wfile.write(json.dumps(status, indent=2).encode())
                elif self.path == "/health":
                    # Health check for load balancers
                    all_critical_running = all(
                        p.state == ProcessState.RUNNING
                        for p in supervisor.managed_processes.values()
                        if p.critical and p.enabled
                    )
                    if all_critical_running:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"OK")
                    else:
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(b"UNHEALTHY")
                else:
                    self.send_response(404)
                    self.end_headers()

        try:
            self.http_server = HTTPServer(("0.0.0.0", self.http_port), StatusHandler)
            self.http_thread = Thread(target=self.http_server.serve_forever, daemon=True)
            self.http_thread.start()
            logger.info(f"HTTP status server running on port {self.http_port}")
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get supervisor status."""
        return {
            "supervisor": {
                "node_id": self.node_id,
                "uptime_seconds": time.time() - self.start_time,
                "running": self.running,
                "http_port": self.http_port,
            },
            "processes": {
                name: proc.to_dict()
                for name, proc in self.managed_processes.items()
            },
        }

    async def run(self):
        """Main entry point."""
        self.running = True
        self.start_time = time.time()

        logger.info(f"Cluster Supervisor starting on {self.node_id}")

        # Start HTTP server
        self._start_http_server()

        # Notify systemd we're ready
        if self.sd_notifier:
            self.sd_notifier.notify("READY=1")

        # Start all enabled processes
        for name, proc in self.managed_processes.items():
            if proc.enabled:
                await self.start_process(proc)

        # Enter monitoring loop
        try:
            await self.monitor_loop()
        except asyncio.CancelledError:
            logger.info("Supervisor received shutdown signal")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Supervisor shutting down...")
        self.running = False

        # Stop all managed processes
        for name, proc in self.managed_processes.items():
            if proc.state == ProcessState.RUNNING:
                await self.stop_process(proc)

        # Stop HTTP server
        if self.http_server:
            self.http_server.shutdown()

        # Notify systemd
        if self.sd_notifier:
            self.sd_notifier.notify("STOPPING=1")

        logger.info("Supervisor shutdown complete")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Cluster Supervisor Daemon")
    parser.add_argument("--node-id", help="Node identifier")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="HTTP status port")
    parser.add_argument("--ringrift-path", help="Path to RingRift repository")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.status:
        # Query running supervisor
        import urllib.request
        try:
            url = f"http://localhost:{args.port}/status"
            with urllib.request.urlopen(url, timeout=5) as resp:
                status = json.loads(resp.read().decode())
                print(json.dumps(status, indent=2))
        except Exception as e:
            print(f"Failed to get status: {e}")
            sys.exit(1)
        return

    supervisor = ClusterSupervisor(
        http_port=args.port,
        node_id=args.node_id,
        ringrift_path=args.ringrift_path,
    )

    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        supervisor.running = False

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        loop.run_until_complete(supervisor.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
