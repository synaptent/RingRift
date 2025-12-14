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
import fcntl
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

# If a node reports hundreds/thousands of selfplay processes, it almost always
# indicates job tracking was lost and stale processes are accumulating.
RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = int(
    os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD", "128") or 128
)


def _acquire_singleton_lock(node_id: str):
    """Acquire a non-blocking singleton lock so we don't run duplicate daemons.

    This prevents double-start situations from tmux/systemd + cron overlap, which
    can otherwise spawn multiple P2P orchestrators and fallback jobs.
    """
    lock_path = os.environ.get("RINGRIFT_NODE_RESILIENCE_LOCK_FILE") or f"/tmp/ringrift_node_resilience_{node_id}.lock"
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
    except Exception:
        pass
    return fh


def _load_cluster_auth_token() -> str:
    token = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN") or "").strip()
    if token:
        return token
    token_file = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN_FILE") or "").strip()
    if token_file:
        try:
            return Path(token_file).read_text().strip()
        except Exception:
            return ""
    return ""


@dataclass
class NodeConfig:
    """Configuration for this node."""
    node_id: str
    coordinator_url: str
    ai_service_dir: str
    num_gpus: int
    # Fallback selfplay script for GPU nodes. Prefer hybrid for rule fidelity; can
    # be overridden per-node via --selfplay-script or env.
    selfplay_script: str = "scripts/run_hybrid_selfplay.py"
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
        self.gpu_idle_since: Optional[float] = None
        self.last_p2p_check = 0
        self.last_registration = 0
        self.p2p_connected = False
        self.running = True
        self._last_good_coordinator: str = ""

    def _coordinator_urls(self) -> List[str]:
        raw = (self.config.coordinator_url or "").strip()
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        return urls

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

    def get_tailscale_ip(self) -> Optional[str]:
        """Get this machine's Tailscale IPv4 (100.x) if available."""
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            ip = (result.stdout or "").strip().splitlines()[0].strip()
            return ip or None
        except FileNotFoundError:
            return None
        except Exception:
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

    def check_p2p_managing_jobs(self) -> bool:
        """Check if P2P is actively managing jobs (to avoid cron conflicts).

        LEARNED LESSONS - Don't start fallback selfplay if P2P is managing jobs,
        even if P2P health check is slightly delayed. This prevents job conflicts.
        """
        try:
            url = f"http://localhost:{self.config.p2p_port}/status"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                # Only treat P2P as "managing jobs" when it is actually running
                # work on this node. A common failure mode is a partitioned node
                # electing itself leader (peers unreachable) while dispatching no
                # jobs; in that case we still want fallback selfplay to keep the
                # machine utilized.
                selfplay_jobs = int(data.get("selfplay_jobs", 0) or 0)
                training_jobs = int(data.get("training_jobs", 0) or 0)
                return (selfplay_jobs + training_jobs) > 0
        except Exception:
            return False

    def check_coordinator_reachable(self) -> bool:
        """Check if the coordinator is reachable."""
        for base in self._coordinator_urls():
            try:
                url = f"{base.rstrip('/')}/health"
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    if "healthy" in data:
                        ok = bool(data.get("healthy"))
                    else:
                        ok = data.get("status") == "ok"
                    if ok:
                        self._last_good_coordinator = base
                        return True
            except Exception:
                continue
        return False

    def register_with_coordinator(self) -> bool:
        """Register this node with the coordinator."""
        public_ip = self.get_public_ip()
        tailscale_ip = self.get_tailscale_ip()
        ip = public_ip or tailscale_ip
        if not ip:
            logger.warning("Failed to get IP for registration")
            return False

        payload = {
            "node_id": self.config.node_id,
            "host": ip,
            "port": self._get_ssh_port(),
        }
        if tailscale_ip:
            payload["tailscale_ip"] = tailscale_ip
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = _load_cluster_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        for base in self._coordinator_urls():
            try:
                url = f"{base.rstrip('/')}/register"
                request = urllib.request.Request(
                    url,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    result = json.loads(response.read().decode())
                    if result.get("success"):
                        self._last_good_coordinator = base
                        logger.info(f"Registered {self.config.node_id} at {ip} via {base}")
                        return True
            except Exception as e:
                logger.warning(f"Registration failed via {base}: {e}")
                continue
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

    def _python_can_import(self, python_executable: str, module: str) -> bool:
        try:
            result = subprocess.run(
                [python_executable, "-c", f"import {module}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _python_for_orchestrator(self) -> str:
        """Select a python executable that can import aiohttp (required by p2p).

        Prefer the current interpreter, then ai-service venv, then system python3.
        """
        candidates: List[str] = []
        if sys.executable:
            candidates.append(sys.executable)
        venv_py = Path(self.config.ai_service_dir) / "venv" / "bin" / "python"
        if venv_py.exists() and os.access(venv_py, os.X_OK):
            candidates.append(str(venv_py))
        candidates.append("python3")

        for cand in candidates:
            if cand and self._python_can_import(cand, "aiohttp"):
                return cand
        return sys.executable or "python3"

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
                        self._python_for_orchestrator(),
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

                script = (self.config.selfplay_script or "scripts/run_hybrid_selfplay.py").strip()
                script_path = script if os.path.isabs(script) else os.path.join(self.config.ai_service_dir, script)

                cmd: List[str]
                if script_path.endswith("run_gpu_selfplay.py"):
                    cmd = [
                        sys.executable,
                        script_path,
                        "--board", self.config.fallback_board,
                        "--num-players", str(self.config.fallback_num_players),
                        "--num-games", str(self.config.fallback_num_games_gpu),
                        "--batch-size", str(self.config.fallback_batch_size_gpu),
                        "--max-moves", "10000",  # Avoid draws due to move limit
                        "--output-dir", output_dir,
                        "--seed", str(int(time.time()) + gpu_id),
                    ]
                else:
                    # Default: hybrid selfplay (CPU rules + GPU eval).
                    cmd = [
                        sys.executable,
                        script_path,
                        "--board-type", self.config.fallback_board,
                        "--num-players", str(self.config.fallback_num_players),
                        "--num-games", str(self.config.fallback_num_games_gpu),
                        "--max-moves", "10000",  # Avoid draws due to move limit
                        "--output-dir", output_dir,
                        "--engine-mode", "mixed",
                        "--seed", str(int(time.time()) + gpu_id),
                    ]

                proc = subprocess.Popen(
                    cmd,
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
                        "--max-moves", "10000",  # Avoid draws due to move limit
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
        # LEARNED LESSONS - Don't start fallback if P2P is already managing jobs
        # This prevents cron conflicts with P2P job management
        if self.check_p2p_managing_jobs():
            logger.info("P2P is managing jobs, skipping fallback work")
            return

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
            for line in result.stdout.strip().split("\n"):
                try:
                    util = int(line.strip().replace("%", "").replace(" ", ""))
                    utilizations.append(util)
                except ValueError:
                    continue

            if not utilizations:
                return True

            # Only consider fallback processes we started/tracked.
            self.local_selfplay_pids = [
                pid for pid in self.local_selfplay_pids if self._process_running(pid)
            ]
            selfplay_procs = len(self.local_selfplay_pids)
            if selfplay_procs == 0:
                self.gpu_idle_since = None
                return True

            # Stuck detection: processes running but all GPUs at 0%
            all_idle = all(u < 2 for u in utilizations)  # Allow 1-2% idle noise

            if selfplay_procs > 0 and all_idle:
                # Track how long GPUs have been idle
                if self.gpu_idle_since is None:
                    self.gpu_idle_since = time.time()
                    logger.warning(f"GPU idle detected: {selfplay_procs} procs, util={utilizations}")
                    return True  # Give it time

                idle_duration = time.time() - self.gpu_idle_since
                if idle_duration > 300:  # 5 minutes of idle
                    logger.error(f"Stuck processes detected: {selfplay_procs} procs, GPU idle for {idle_duration:.0f}s")
                    logger.info("Killing stuck selfplay processes...")

                    # Kill tracked fallback processes.
                    for pid in self.local_selfplay_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except Exception:
                            continue

                    # Reset tracking
                    self.gpu_idle_since = None
                    self.local_selfplay_pids = []

                    return False
            else:
                # GPU is working, reset idle tracking
                self.gpu_idle_since = None

            return True

        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
            return True

    def _count_selfplay_processes(self) -> int:
        """Count selfplay-related processes (best-effort) to detect runaway states."""
        patterns = [
            "run_self_play_soak.py",
            "run_hybrid_selfplay.py",
            "run_gpu_selfplay.py",
            "run_random_selfplay.py",
        ]
        pids: set[int] = set()
        for pattern in patterns:
            try:
                out = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode != 0:
                    continue
                for tok in (out.stdout or "").strip().split():
                    try:
                        pids.add(int(tok))
                    except Exception:
                        continue
            except Exception:
                continue
        # Never count ourselves.
        pids.discard(int(os.getpid()))
        return len(pids)

    def _kill_selfplay_processes(self) -> int:
        """Kill selfplay-related processes (only used for runaway recovery)."""
        patterns = [
            "run_self_play_soak.py",
            "run_hybrid_selfplay.py",
            "run_gpu_selfplay.py",
            "run_random_selfplay.py",
        ]
        before = self._count_selfplay_processes()
        for pattern in patterns:
            try:
                # First try SIGTERM.
                subprocess.run(
                    ["pkill", "-TERM", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:
                pass
        time.sleep(2)
        for pattern in patterns:
            try:
                # Then SIGKILL any stragglers.
                subprocess.run(
                    ["pkill", "-KILL", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:
                pass

        after = self._count_selfplay_processes()
        return max(0, before - after)

    def check_runaway_selfplay(self) -> bool:
        """Detect runaway selfplay counts and trigger a restart sweep.

        Returns True when no runaway condition is detected.
        """
        # Prefer the orchestrator-reported job counts when available.
        if self.check_p2p_health():
            try:
                url = f"http://localhost:{self.config.p2p_port}/status"
                with urllib.request.urlopen(url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                count = int(data.get("selfplay_jobs", 0) or 0)
                if count < RUNAWAY_SELFPLAY_PROCESS_THRESHOLD:
                    return True

                logger.error(
                    f"Runaway selfplay detected via P2P status: {count} >= {RUNAWAY_SELFPLAY_PROCESS_THRESHOLD}"
                )

                headers = {"Content-Type": "application/json"}
                token = _load_cluster_auth_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                req = urllib.request.Request(
                    f"http://localhost:{self.config.p2p_port}/restart_stuck_jobs",
                    data=b"{}",
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    _ = resp.read()
                logger.info("Requested /restart_stuck_jobs for runaway recovery")
                return False
            except Exception as e:
                logger.warning(f"Runaway selfplay check (via P2P) failed: {e}")

        # Fallback: count processes directly.
        try:
            count = self._count_selfplay_processes()
            if count < RUNAWAY_SELFPLAY_PROCESS_THRESHOLD:
                return True
            logger.error(
                f"Runaway selfplay detected via process table: {count} >= {RUNAWAY_SELFPLAY_PROCESS_THRESHOLD}"
            )
            killed_est = self._kill_selfplay_processes()
            logger.warning(f"Runaway recovery: issued pkill sweep (killed_est={killed_est})")
            return False
        except Exception as e:
            logger.warning(f"Runaway selfplay check (via pgrep) failed: {e}")
            return True

    def run_once(self) -> None:
        """Run a single check cycle."""
        now = time.time()

        # Check disk and cleanup if needed (critical for Vast instances)
        self.check_and_cleanup_disk()

        # Check GPU health and kill stuck processes
        self.check_gpu_health()

        # Detect runaway selfplay states (lost tracking / manual runaway processes)
        self.check_runaway_selfplay()

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
    parser.add_argument("--coordinator", required=True, help="Comma-separated seed coordinator URLs")
    parser.add_argument("--ai-service-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="AI service directory")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (auto-detect if 0)")
    parser.add_argument("--p2p-port", type=int, default=8770, help="P2P orchestrator port")
    parser.add_argument("--peers", default="", help="Comma-separated peer list for P2P orchestrator (defaults to coordinator URL)")
    parser.add_argument("--check-interval", type=int, default=60, help="Health check interval (seconds)")
    parser.add_argument("--max-local-procs", type=int, default=4, help="Max fallback workers to run when disconnected")
    parser.add_argument("--disk-threshold", type=int, default=80, help="Disk usage percent threshold for cleanup")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="Minimum free GB headroom before forcing cleanup")
    parser.add_argument(
        "--selfplay-script",
        default=os.environ.get("RINGRIFT_FALLBACK_SELFPLAY_SCRIPT", "scripts/run_hybrid_selfplay.py"),
        help="Fallback selfplay script for GPU nodes (relative to ai-service dir unless absolute)",
    )
    parser.add_argument("--once", action="store_true", help="Run once and exit (for cron)")

    args = parser.parse_args()

    lock_handle = _acquire_singleton_lock(args.node_id)
    if lock_handle is None:
        logger.info(f"node_resilience already running for {args.node_id}; exiting")
        return

    config = NodeConfig(
        node_id=args.node_id,
        coordinator_url=args.coordinator,
        ai_service_dir=args.ai_service_dir,
        num_gpus=args.num_gpus,
        selfplay_script=args.selfplay_script,
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
