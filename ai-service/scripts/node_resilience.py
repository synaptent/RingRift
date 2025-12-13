#!/usr/bin/env python3
"""Node resilience daemon - keeps nodes utilized even when disconnected from P2P.

This script runs as a background daemon on each node and ensures:
1. P2P orchestrator is running and connected
2. If P2P is unavailable, runs local selfplay as fallback
3. Periodically attempts to reconnect to P2P network
4. Auto-registers with coordinator when IP changes

Usage:
    # Run as daemon
    python scripts/node_resilience.py --node-id vast-5090-quad --coordinator http://192.222.53.22:8770

    # Run once (for cron)
    python scripts/node_resilience.py --node-id vast-5090-quad --coordinator http://192.222.53.22:8770 --once
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_log_file = (os.environ.get("RINGRIFT_NODE_RESILIENCE_LOG_FILE") or "").strip()
if _log_file:
    try:
        logger.addHandler(logging.FileHandler(_log_file))
    except Exception as e:
        logger.warning(f"Failed to add file logger { _log_file }: {e}")


@dataclass
class NodeConfig:
    """Configuration for this node."""
    node_id: str
    coordinator_url: str
    ai_service_dir: str
    num_gpus: int
    selfplay_script: str = "scripts/run_gpu_selfplay.py"
    p2p_port: int = 8770
    peers: str = ""  # comma-separated list for p2p_orchestrator.py --peers
    check_interval: int = 60  # seconds
    reconnect_interval: int = 300  # seconds
    max_local_selfplay_procs: int = 4
    disk_threshold: int = 80  # percent - trigger cleanup above this
    min_free_gb: float = 2.0  # trigger cleanup if free space is low
    fallback_board: str = "square8"
    fallback_num_players: int = 2
    fallback_num_games_gpu: int = 3000
    fallback_num_games_cpu: int = 300
    fallback_batch_size_gpu: int = 16


class NodeResilience:
    """Keeps a node utilized and connected to the cluster."""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.local_selfplay_pids: List[int] = self._discover_fallback_pids()
        self.last_p2p_check = 0
        self.last_registration = 0
        self.p2p_connected = False
        self.running = True

    def get_public_ip(self) -> Optional[str]:
        """Get this machine's public IP address."""
        services = [
            "https://api.ipify.org",
            "https://icanhazip.com",
            "https://ifconfig.me/ip",
        ]
        for url in services:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    ip = response.read().decode().strip()
                    if ip:
                        return ip
            except Exception:
                continue
        return None

    def check_p2p_health(self) -> bool:
        """Check if P2P orchestrator is running and connected."""
        try:
            url = f"http://localhost:{self.config.p2p_port}/health"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                if "healthy" in data:
                    return bool(data.get("healthy"))
                # Back-compat for older health payloads
                return data.get("status") == "ok"
        except Exception:
            return False

    def check_coordinator_reachable(self) -> bool:
        """Check if the coordinator is reachable."""
        try:
            url = f"{self.config.coordinator_url.rstrip('/')}/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                if "healthy" in data:
                    return bool(data.get("healthy"))
                return data.get("status") == "ok"
        except Exception:
            return False

    def register_with_coordinator(self) -> bool:
        """Register this node with the coordinator."""
        ip = self.get_public_ip()
        if not ip:
            logger.warning("Failed to get public IP for registration")
            return False

        try:
            url = f"{self.config.coordinator_url.rstrip('/')}/register"
            payload = {
                "node_id": self.config.node_id,
                "host": ip,
                "port": self._get_ssh_port(),
            }
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            token = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN") or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode())
                if result.get("success"):
                    logger.info(f"Registered {self.config.node_id} at {ip}")
                    return True
        except Exception as e:
            logger.warning(f"Registration failed: {e}")
        return False

    def _get_ssh_port(self) -> int:
        """Get SSH port for dynamic registry registration.

        Prefer explicit env (SSH_PORT), otherwise attempt to read the local
        distributed host config so Vast nodes register the externally-forwarded
        SSH port (not the internal port 22).
        """
        env_port = (os.environ.get("SSH_PORT") or "").strip()
        if env_port:
            try:
                return int(env_port)
            except ValueError:
                pass

        try:
            cfg_path = Path(self.config.ai_service_dir) / "config" / "distributed_hosts.yaml"
            if not cfg_path.exists():
                return 22
            try:
                import yaml  # type: ignore
            except Exception:
                return 22
            data = yaml.safe_load(cfg_path.read_text()) or {}
            hosts = data.get("hosts", {}) or {}
            node_cfg = hosts.get(self.config.node_id, {}) or {}
            port = node_cfg.get("ssh_port", 22) or 22
            return int(port)
        except Exception:
            return 22

    def _discover_fallback_pids(self) -> List[int]:
        """Recover fallback selfplay PIDs from process table (for daemon restarts)."""
        marker = f"fallback/{self.config.node_id}"
        try:
            out = subprocess.run(
                ["pgrep", "-f", marker],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0:
                return []
            pids = []
            for token in out.stdout.strip().split():
                try:
                    pids.append(int(token))
                except ValueError:
                    continue
            return sorted(set(pids))
        except Exception:
            return []

    def _ringrift_root(self) -> str:
        """Infer RingRift repo root from ai-service dir."""
        return str(Path(self.config.ai_service_dir).resolve().parent)

    def start_p2p_orchestrator(self) -> bool:
        """Start the P2P orchestrator if not running."""
        if self.check_p2p_health():
            return True

        logger.info("Starting P2P orchestrator...")
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.config.ai_service_dir

            # Ensure we have peers so cloud nodes can find the coordinator.
            peers = self.config.peers.strip() or self.config.coordinator_url.strip()

            log_dir = Path(self.config.ai_service_dir) / "logs" / "node_resilience"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "p2p_orchestrator.log"

            log_handle = open(log_path, "a")
            try:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(self.config.ai_service_dir, "scripts/p2p_orchestrator.py"),
                        "--node-id", self.config.node_id,
                        "--port", str(self.config.p2p_port),
                        "--peers", peers,
                        "--ringrift-path", self._ringrift_root(),
                    ],
                    cwd=self.config.ai_service_dir,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            finally:
                log_handle.close()
            time.sleep(3)
            if proc.poll() is None and self.check_p2p_health():
                logger.info(f"P2P orchestrator started (PID {proc.pid})")
                return True
        except Exception as e:
            logger.error(f"Failed to start P2P orchestrator: {e}")
        return False

    def _start_gpu_fallback_selfplay(self, num_to_start: int) -> None:
        """Start GPU fallback selfplay workers."""
        if self.config.num_gpus <= 0 or num_to_start <= 0:
            return

        logger.info(f"Starting {num_to_start} GPU fallback selfplay workers")
        for i in range(num_to_start):
            gpu_id = (len(self.local_selfplay_pids) + i) % max(self.config.num_gpus, 1)
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = self.config.ai_service_dir
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                env["RINGRIFT_JOB_ORIGIN"] = "resilience_fallback"

                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    self.config.ai_service_dir,
                    "data/selfplay/fallback",
                    self.config.node_id,
                    f"gpu{gpu_id}",
                    ts,
                )

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(self.config.ai_service_dir, "scripts/run_gpu_selfplay.py"),
                        "--board", self.config.fallback_board,
                        "--num-players", str(self.config.fallback_num_players),
                        "--num-games", str(self.config.fallback_num_games_gpu),
                        "--batch-size", str(self.config.fallback_batch_size_gpu),
                        "--output-dir", output_dir,
                        "--seed", str(int(time.time()) + gpu_id),
                    ],
                    cwd=self.config.ai_service_dir,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.local_selfplay_pids.append(proc.pid)
                logger.info(f"Started GPU fallback selfplay on GPU {gpu_id} (PID {proc.pid})")
            except Exception as e:
                logger.error(f"Failed to start GPU fallback selfplay on GPU {gpu_id}: {e}")

    def _start_cpu_fallback_selfplay(self, num_to_start: int) -> None:
        """Start CPU fallback selfplay workers (for non-GPU nodes)."""
        if num_to_start <= 0:
            return

        logger.info(f"Starting {num_to_start} CPU fallback selfplay workers")
        for i in range(num_to_start):
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = self.config.ai_service_dir
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                env["RINGRIFT_JOB_ORIGIN"] = "resilience_fallback"

                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                out_dir = Path(self.config.ai_service_dir) / "data" / "selfplay" / "fallback" / self.config.node_id / "cpu" / ts
                out_dir.mkdir(parents=True, exist_ok=True)

                log_jsonl = str(out_dir / "games.jsonl")
                summary_json = str(out_dir / "summary.json")

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(self.config.ai_service_dir, "scripts/run_self_play_soak.py"),
                        "--num-games", str(self.config.fallback_num_games_cpu),
                        "--board-type", self.config.fallback_board,
                        "--num-players", str(self.config.fallback_num_players),
                        "--engine-mode", "mixed",
                        "--difficulty-band", "light",
                        "--log-jsonl", log_jsonl,
                        "--summary-json", summary_json,
                        "--lean-db",
                        "--record-db", str(out_dir / "games.db"),
                        "--verbose", "0",
                    ],
                    cwd=self.config.ai_service_dir,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.local_selfplay_pids.append(proc.pid)
                logger.info(f"Started CPU fallback selfplay (PID {proc.pid})")
            except Exception as e:
                logger.error(f"Failed to start CPU fallback selfplay: {e}")

    def start_local_fallback_work(self) -> None:
        """Start local fallback work when P2P is unavailable."""
        # Clean up dead processes
        self.local_selfplay_pids = [
            pid for pid in self.local_selfplay_pids
            if self._process_running(pid)
        ]

        # Avoid spawning new work when disk is under pressure.
        if not self.check_and_cleanup_disk():
            logger.warning("Skipping fallback work due to disk pressure")
            return

        max_procs = max(1, self.config.max_local_selfplay_procs)
        num_to_start = max_procs - len(self.local_selfplay_pids)

        if num_to_start <= 0:
            return

        if self.config.num_gpus > 0:
            # Prefer one worker per GPU, but respect the overall cap.
            self._start_gpu_fallback_selfplay(num_to_start=min(num_to_start, self.config.num_gpus))
        else:
            # CPU-only nodes: keep it conservative by default.
            self._start_cpu_fallback_selfplay(num_to_start=min(num_to_start, 1))

    def stop_local_selfplay(self) -> None:
        """Stop all local selfplay processes (when P2P reconnects)."""
        for pid in self.local_selfplay_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Stopped local selfplay (PID {pid})")
            except ProcessLookupError:
                pass
        self.local_selfplay_pids = []

    def check_and_cleanup_disk(self) -> bool:
        """Check disk usage and run cleanup if needed."""
        try:
            stat = os.statvfs("/")
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used_percent = ((total - free) / total) * 100 if total > 0 else 0
            free_gb = free / (1024 ** 3) if free > 0 else 0.0

            if used_percent > self.config.disk_threshold or free_gb < self.config.min_free_gb:
                logger.warning(f"Disk usage {used_percent:.1f}% exceeds threshold {self.config.disk_threshold}%")
                if free_gb < self.config.min_free_gb:
                    logger.warning(f"Low disk headroom: {free_gb:.2f}GB free (< {self.config.min_free_gb}GB)")

                # Try to run disk_monitor.py if available
                disk_monitor = os.path.join(self.config.ai_service_dir, "scripts/disk_monitor.py")
                if os.path.exists(disk_monitor):
                    logger.info("Running disk cleanup...")
                    cmd = [
                        sys.executable,
                        disk_monitor,
                        "--threshold",
                        str(self.config.disk_threshold),
                        "--ringrift-path",
                        self._ringrift_root(),
                    ]
                    # When very low on space, force cleanup even if the percent calculation is skewed.
                    if free_gb < self.config.min_free_gb:
                        cmd.append("--force")
                    cmd.append("--aggressive")
                    result = subprocess.run(
                        cmd,
                        cwd=self.config.ai_service_dir,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        logger.info("Disk cleanup completed successfully")
                        return True
                    else:
                        logger.warning(f"Disk cleanup failed: {result.stderr}")
                else:
                    logger.warning("disk_monitor.py not found, skipping cleanup")

                return False
            return True
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False

    def _process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def detect_num_gpus(self) -> int:
        """Detect number of available GPUs."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split("\n"))
        except Exception:
            pass
        return 0

    def check_gpu_health(self) -> bool:
        """Check GPU health and kill stuck processes.

        Detects stuck jobs: processes running but GPU utilization at 0% for extended period.
        Returns True if healthy, False if stuck processes were killed.
        """
        if self.config.num_gpus <= 0:
            return True

        try:
            # Get GPU utilization
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return True  # Can't check, assume healthy

            utilizations = []
            for line in result.stdout.strip().split('\n'):
                try:
                    util = int(line.strip().replace('%', '').replace(' ', ''))
                    utilizations.append(util)
                except ValueError:
                    continue

            if not utilizations:
                return True

            # Count selfplay processes
            selfplay_procs = 0
            try:
                ps_result = subprocess.run(
                    ["pgrep", "-f", "run_gpu_selfplay|run_self_play"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if ps_result.returncode == 0:
                    selfplay_procs = len(ps_result.stdout.strip().split('\n'))
            except Exception:
                pass

            # Stuck detection: processes running but all GPUs at 0%
            all_idle = all(u < 2 for u in utilizations)  # Allow 1-2% idle noise

            if selfplay_procs > 0 and all_idle:
                # Track how long GPUs have been idle
                if not hasattr(self, '_gpu_idle_since'):
                    self._gpu_idle_since = time.time()
                    logger.warning(f"GPU idle detected: {selfplay_procs} procs, util={utilizations}")
                    return True  # Give it time

                idle_duration = time.time() - self._gpu_idle_since
                if idle_duration > 300:  # 5 minutes of idle
                    logger.error(f"Stuck processes detected: {selfplay_procs} procs, GPU idle for {idle_duration:.0f}s")
                    logger.info("Killing stuck selfplay processes...")

                    # Kill stuck processes
                    subprocess.run(["pkill", "-9", "-f", "run_gpu_selfplay"], timeout=10)
                    subprocess.run(["pkill", "-9", "-f", "run_self_play_soak"], timeout=10)

                    # Reset tracking
                    self._gpu_idle_since = None
                    self.local_selfplay_pids = []

                    return False
            else:
                # GPU is working, reset idle tracking
                self._gpu_idle_since = None

            return True

        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
            return True

    def run_once(self) -> None:
        """Run a single check cycle."""
        now = time.time()

        # Check disk and cleanup if needed (critical for Vast instances)
        self.check_and_cleanup_disk()

        # Check GPU health and kill stuck processes
        self.check_gpu_health()

        # Check P2P health
        p2p_healthy = self.check_p2p_health()
        coordinator_reachable = self.check_coordinator_reachable()

        if p2p_healthy and coordinator_reachable:
            if not self.p2p_connected:
                logger.info("P2P connection restored - stopping local fallback")
                self.stop_local_selfplay()
            self.p2p_connected = True
        else:
            if self.p2p_connected:
                logger.warning("P2P connection lost - starting local fallback")
            self.p2p_connected = False

            # Try to start P2P orchestrator
            if not p2p_healthy:
                self.start_p2p_orchestrator()

            # Start local fallback work
            self.start_local_fallback_work()

        # Periodic registration
        if now - self.last_registration > self.config.reconnect_interval:
            if coordinator_reachable:
                self.register_with_coordinator()
            self.last_registration = now

    def run_daemon(self) -> None:
        """Run as a continuous daemon."""
        logger.info(f"Node resilience daemon started for {self.config.node_id}")
        logger.info(f"Coordinator: {self.config.coordinator_url}")
        logger.info(f"GPUs detected: {self.config.num_gpus}")

        def handle_signal(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False
            self.stop_local_selfplay()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            time.sleep(self.config.check_interval)


def main():
    parser = argparse.ArgumentParser(description="Node resilience daemon")
    parser.add_argument("--node-id", required=True, help="Node identifier")
    parser.add_argument("--coordinator", required=True, help="Coordinator URL")
    parser.add_argument("--ai-service-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="AI service directory")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (auto-detect if 0)")
    parser.add_argument("--p2p-port", type=int, default=8770, help="P2P orchestrator port")
    parser.add_argument("--peers", default="", help="Comma-separated peer list for P2P orchestrator (defaults to coordinator URL)")
    parser.add_argument("--check-interval", type=int, default=60, help="Health check interval (seconds)")
    parser.add_argument("--max-local-procs", type=int, default=4, help="Max fallback workers to run when disconnected")
    parser.add_argument("--disk-threshold", type=int, default=80, help="Disk usage percent threshold for cleanup")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="Minimum free GB headroom before forcing cleanup")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for cron)")

    args = parser.parse_args()

    config = NodeConfig(
        node_id=args.node_id,
        coordinator_url=args.coordinator,
        ai_service_dir=args.ai_service_dir,
        num_gpus=args.num_gpus,
        p2p_port=args.p2p_port,
        peers=args.peers,
        check_interval=args.check_interval,
        max_local_selfplay_procs=args.max_local_procs,
        disk_threshold=args.disk_threshold,
        min_free_gb=args.min_free_gb,
    )

    resilience = NodeResilience(config)

    # Auto-detect GPUs if not specified
    if config.num_gpus == 0:
        config.num_gpus = resilience.detect_num_gpus()
        logger.info(f"Auto-detected {config.num_gpus} GPUs")

    if args.once:
        resilience.run_once()
    else:
        resilience.run_daemon()


if __name__ == "__main__":
    main()
