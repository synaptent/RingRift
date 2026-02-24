#!/usr/bin/env python3
"""Node resilience daemon - keeps nodes utilized even when disconnected from P2P.

This script runs as a background daemon on each node and ensures:
1. P2P orchestrator is running and connected
2. If P2P is unavailable, runs local selfplay as fallback
3. Periodically attempts to reconnect to P2P network
4. Auto-registers with coordinator when IP changes

Usage:
    # Run as daemon
    python scripts/node_resilience.py --node-id <node-id> --coordinator http://<coordinator-ip>:8770

    # Run once (for cron)
    python scripts/node_resilience.py --node-id <node-id> --coordinator http://<coordinator-ip>:8770 --once
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import port configuration
try:
    from app.config.ports import P2P_DEFAULT_PORT
except ImportError:
    P2P_DEFAULT_PORT = 8770

# Diverse selfplay profiles for high-quality training data
# Based on benchmark results: MaxN >> Descent in 3P/4P, Gumbel >> all in 2P
# Dec 28, 2025: CRITICAL FIX - Changed "hex" to "hex8" in all profiles.
# "hex" was normalized to "hexagonal" (large 469-cell board) instead of
# "hex8" (small 61-cell board), starving hex8_* configs of selfplay games.
# Dec 31, 2025: Added random, nn-minimax, descent-only, gpu-gumbel harnesses
# for broader harness diversity. Total = 100%.
DIVERSE_ENGINE_PROFILES = [
    # Neural-guided Gumbel MCTS (24%) - highest quality training data
    {"engine_mode": "gumbel-mcts", "board_type": "hex8", "num_players": 2, "weight": 0.18},
    {"engine_mode": "gumbel-mcts", "board_type": "square8", "num_players": 3, "weight": 0.06},
    # GPU Gumbel (3%) - high throughput neural search
    {"engine_mode": "gpu-gumbel", "board_type": "hex8", "num_players": 2, "weight": 0.02},
    {"engine_mode": "gpu-gumbel", "board_type": "square8", "num_players": 2, "weight": 0.01},
    # Policy-only (11%) - fast neural moves without search
    {"engine_mode": "policy-only", "board_type": "hex8", "num_players": 2, "weight": 0.08},
    {"engine_mode": "policy-only", "board_type": "square8", "num_players": 4, "weight": 0.03},
    # NNUE-guided (12%) - efficient value + policy guidance
    {"engine_mode": "nnue-guided", "board_type": "square8", "num_players": 2, "weight": 0.08},
    {"engine_mode": "nnue-guided", "board_type": "hex8", "num_players": 3, "weight": 0.04},
    # Standard MCTS (6%)
    {"engine_mode": "mcts", "board_type": "hex8", "num_players": 2, "weight": 0.06},
    # NN-Minimax (7%) - strong 2-player search with neural evaluation
    {"engine_mode": "nn-minimax", "board_type": "square8", "num_players": 2, "weight": 0.04},
    {"engine_mode": "nn-minimax", "board_type": "hex8", "num_players": 2, "weight": 0.03},
    # MaxN/BRS multiplayer (15%) - best for 3P/4P games
    {"engine_mode": "maxn", "board_type": "hex8", "num_players": 3, "weight": 0.05},
    {"engine_mode": "maxn", "board_type": "square8", "num_players": 4, "weight": 0.04},
    {"engine_mode": "brs", "board_type": "hex8", "num_players": 3, "weight": 0.03},
    {"engine_mode": "brs", "board_type": "square8", "num_players": 4, "weight": 0.03},
    # Descent-only (4%) - exploration via sequential halving
    {"engine_mode": "descent-only", "board_type": "hex8", "num_players": 2, "weight": 0.02},
    {"engine_mode": "descent-only", "board_type": "square8", "num_players": 2, "weight": 0.02},
    # Heuristic throughput (12%) - fast baseline games
    {"engine_mode": "heuristic-only", "board_type": "hex8", "num_players": 2, "weight": 0.05},
    {"engine_mode": "heuristic-only", "board_type": "square8", "num_players": 2, "weight": 0.04},
    {"engine_mode": "heuristic-only", "board_type": "hex8", "num_players": 4, "weight": 0.03},
    # Random (5%) - baseline calibration, exploration
    {"engine_mode": "random", "board_type": "hex8", "num_players": 2, "weight": 0.03},
    {"engine_mode": "random", "board_type": "square8", "num_players": 4, "weight": 0.02},
    # Mixed exploration (1%)
    {"engine_mode": "mixed", "board_type": "square19", "num_players": 2, "weight": 0.01},
]

# Add project root to path for scripts.lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import setup_script_logging

_log_file = (os.environ.get("RINGRIFT_NODE_RESILIENCE_LOG_FILE") or "").strip() or None
logger = setup_script_logging("node_resilience", log_file=_log_file)

# If a node reports hundreds/thousands of selfplay processes, it almost always
# indicates job tracking was lost and stale processes are accumulating.
# Dec 26 2025: Lowered from 500 to 100 for earlier intervention.
# Process limit enforcement in IdleResourceDaemon handles soft limits at 50.
RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = int(
    os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD", "100") or 100
)

STATE_DIR = Path(__file__).parent.parent / "logs" / "node_resilience"
STATE_DIR.mkdir(parents=True, exist_ok=True)


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
    except (OSError, IOError):
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
        except (FileNotFoundError, OSError, PermissionError):
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
    p2p_port: int = P2P_DEFAULT_PORT
    peers: str = ""  # comma-separated list for p2p_orchestrator.py --peers
    check_interval: int = 60  # seconds
    reconnect_interval: int = 300  # seconds
    max_local_selfplay_procs: int = 4
    disk_threshold: int = 70  # percent - DISK_SYNC_TARGET_PERCENT from app.config.thresholds
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
        self.state_path = STATE_DIR / f"state_{self.config.node_id}.json"
        state = self._load_state()
        self.fallback_job_prefix = f"resilience_fallback_{self.config.node_id}_"
        self.fallback_job_ids: list[str] = [
            str(j).strip()
            for j in (state.get("fallback_job_ids") or [])
            if str(j).strip()
        ]
        self.local_selfplay_pids: list[int] = self._discover_fallback_pids(state=state)
        self.gpu_idle_since: float | None = None
        self.last_p2p_check = 0
        self.last_registration = 0
        self.p2p_connected = False
        self.p2p_unhealthy_since: float | None = None  # Grace period tracking
        self.p2p_last_start_attempt: float | None = None  # Startup grace period tracking
        self.running = True
        self._last_good_coordinator: str = ""

    def _load_state(self) -> dict[str, Any]:
        try:
            if not self.state_path.exists():
                return {}
            raw = self.state_path.read_text().strip()
            if not raw:
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, OSError, PermissionError, json.JSONDecodeError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            payload = {
                "node_id": self.config.node_id,
                "updated_at": datetime.utcnow().isoformat(),
                "fallback_job_ids": list(self.fallback_job_ids),
                "fallback_pids": list(self.local_selfplay_pids),
            }
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp.replace(self.state_path)
        except (OSError, PermissionError, ValueError, TypeError):
            pass

    def _coordinator_urls(self) -> list[str]:
        raw = (self.config.coordinator_url or "").strip()
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        return urls

    def get_public_ip(self) -> str | None:
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
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                continue
        return None

    def get_tailscale_ip(self) -> str | None:
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
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
            return None

    def check_p2p_health(self) -> bool:
        """Check if P2P orchestrator is running and connected.

        Uses retry logic to avoid false negatives from transient network issues.
        The /status endpoint can be slow when the orchestrator is busy, so we use
        generous timeouts to avoid killing healthy but busy orchestrators.
        """
        for attempt in range(3):
            try:
                url = f"http://localhost:{self.config.p2p_port}/health"
                # Progressive timeout: 15s, 25s, 35s (increased from 5s, 8s, 11s)
                # The P2P orchestrator can be slow under load
                timeout = 15 + attempt * 10
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    data = json.loads(response.read().decode())
                    if "healthy" in data:
                        return bool(data.get("healthy"))
                    # Back-compat for older health payloads
                    return data.get("status") == "ok"
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
                if attempt < 2:
                    time.sleep(2)  # Longer pause before retry
        return False

    def check_autossh_tunnel(self) -> bool:
        """Check if autossh P2P tunnel is running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "autossh"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def start_autossh_tunnel(self, relay_host: str | None = None) -> bool:
        """Start autossh reverse tunnel when P2P connectivity fails.

        This is useful for Vast instances behind carrier NAT where Tailscale
        may be unreliable. The tunnel forwards the local P2P port to the relay.

        Args:
            relay_host: Relay IP address. If not provided, uses RINGRIFT_RELAY_HOST
                        environment variable or loads from distributed_hosts.yaml.
        """
        if relay_host is None:
            relay_host = os.environ.get("RINGRIFT_RELAY_HOST")
            if not relay_host:
                # Try to load from config
                config_path = Path(self.config.ai_service_dir) / "config" / "distributed_hosts.yaml"
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path) as f:
                            config = yaml.safe_load(f) or {}
                        # Use coordinator as relay
                        coord = config.get("elo_sync", {}).get("coordinator", "mac-studio")
                        hosts = config.get("hosts", {})
                        if coord in hosts:
                            relay_host = hosts[coord].get("tailscale_ip")
                    except (ImportError, ModuleNotFoundError, OSError, AttributeError, KeyError):
                        pass
            if not relay_host:
                logger.warning("No relay host configured for autossh tunnel")
                return False
        if self.check_autossh_tunnel():
            logger.debug("Autossh tunnel already running")
            return True

        # Only use autossh on Vast instances or when explicitly enabled
        node_id = self.config.node_id
        if not node_id.startswith("vast") and not os.environ.get("RINGRIFT_USE_AUTOSSH"):
            return False

        tunnel_script = Path(self.config.ai_service_dir) / "scripts" / "autossh_p2p_tunnel.sh"
        if not tunnel_script.exists():
            logger.warning(f"Autossh tunnel script not found: {tunnel_script}")
            return False

        try:
            logger.info(f"Starting autossh reverse tunnel to {relay_host}")
            result = subprocess.run(
                [
                    str(tunnel_script),
                    "reverse",
                    "--relay", f"ubuntu@{relay_host}",
                    "--node-id", node_id,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.config.ai_service_dir,
            )
            if result.returncode == 0:
                logger.info("Autossh tunnel started successfully")
                return True
            else:
                logger.warning(f"Autossh tunnel failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error starting autossh tunnel: {e}")
            return False

    def _systemd_usable(self) -> bool:
        """Return True when systemd appears to be the active init system.

        NodeResilience runs on a mix of environments (full VMs vs containers).
        Only use `systemctl` when systemd is actually running; otherwise fall
        back to direct process supervision.
        """
        try:
            if not Path("/etc/systemd/system").exists():
                return False
            result = subprocess.run(
                ["systemctl", "is-system-running"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            state = (result.stdout or "").strip().lower()
            # "degraded" is still usable for our purposes.
            return state in {"running", "degraded"}
        except FileNotFoundError:
            return False
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
            return False

    def _systemd_unit_exists(self, unit: str) -> bool:
        try:
            result = subprocess.run(
                ["systemctl", "show", "-p", "LoadState", "--value", unit],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                return False
            load_state = (result.stdout or "").strip().lower()
            return bool(load_state) and load_state != "not-found"
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _local_orchestrator_url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        return f"http://localhost:{self.config.p2p_port}{path}"

    def _local_orchestrator_headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        token = _load_cluster_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _local_orchestrator_get_json(self, path: str, *, timeout: int = 5) -> dict[str, Any] | None:
        try:
            url = self._local_orchestrator_url(path)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = json.loads(response.read().decode())
            return data if isinstance(data, dict) else None
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _local_orchestrator_post_json(
        self, path: str, payload: dict[str, Any], *, timeout: int = 15
    ) -> dict[str, Any] | None:
        try:
            url = self._local_orchestrator_url(path)
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers=self._local_orchestrator_headers(json_body=True),
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode())
            return decoded if isinstance(decoded, dict) else None
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _list_local_jobs(self, *, status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        try:
            qs: dict[str, str] = {"local": "1", "limit": str(int(limit))}
            if status:
                qs["status"] = str(status)
            query = urllib.parse.urlencode(qs)
            data = self._local_orchestrator_get_json(f"/api/jobs?{query}", timeout=10) or {}
            jobs = data.get("jobs") or []
            return jobs if isinstance(jobs, list) else []
        except (ValueError, TypeError, AttributeError):
            return []

    def check_p2p_managing_jobs(self) -> bool:
        """Check if P2P is actively managing jobs (to avoid cron conflicts).

        LEARNED LESSONS - Don't start fallback selfplay if P2P is managing jobs,
        even if P2P health check is slightly delayed. This prevents job conflicts.
        """
        if not self.check_p2p_health():
            return False
        jobs = self._list_local_jobs(status="running", limit=500)
        # Any running job indicates the local orchestrator is already doing work.
        return len(jobs) > 0

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
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
                continue
        return False

    def check_cluster_connected(self) -> bool:
        """Check whether the local orchestrator appears connected to any peers/leader.

        This is intentionally not tied to a single hard-coded coordinator URL; it
        uses local P2P state so leadership can move without falsely triggering
        fallback work.
        """
        if not self.check_p2p_health():
            return False
        status = self._local_orchestrator_get_json("/status", timeout=15) or {}
        try:
            alive_peers = int(status.get("alive_peers", 0) or 0)
        except (ValueError, TypeError):
            alive_peers = 0
        effective_leader_id = str(status.get("effective_leader_id") or "").strip()
        if alive_peers > 0:
            return True
        return bool(effective_leader_id and effective_leader_id != self.config.node_id)

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
            except (ValueError, TypeError):
                pass

        try:
            cfg_path = Path(self.config.ai_service_dir) / "config" / "distributed_hosts.yaml"
            if not cfg_path.exists():
                return 22
            try:
                import yaml  # type: ignore
            except (ImportError, ModuleNotFoundError):
                return 22
            data = yaml.safe_load(cfg_path.read_text()) or {}
            hosts = data.get("hosts", {}) or {}
            node_cfg = hosts.get(self.config.node_id, {}) or {}
            port = node_cfg.get("ssh_port", 22) or 22
            return int(port)
        except (FileNotFoundError, OSError, PermissionError, ValueError, TypeError, AttributeError, KeyError):
            return 22

    def _discover_fallback_pids(self, *, state: dict[str, Any] | None = None) -> list[int]:
        """Recover fallback selfplay PIDs from local state (for daemon restarts)."""
        state = state or {}
        raw_pids = state.get("fallback_pids") or []
        pids: list[int] = []
        for token in raw_pids:
            try:
                pids.append(int(token))
            except (ValueError, TypeError):
                continue
        return sorted(set(pids))

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
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _python_for_orchestrator(self) -> str:
        """Select a python executable that can import P2P dependencies.

        Prefer the current interpreter, then ai-service venv, then system python3.
        Required modules: aiohttp, psutil, yaml (pyyaml).
        """
        candidates: list[str] = []
        if sys.executable:
            candidates.append(sys.executable)
        venv_py = Path(self.config.ai_service_dir) / "venv" / "bin" / "python"
        if venv_py.exists() and os.access(venv_py, os.X_OK):
            candidates.append(str(venv_py))
        candidates.append("python3")

        # P2P requires aiohttp, psutil, and yaml - check all three
        required_modules = ["aiohttp", "psutil", "yaml"]
        for cand in candidates:
            if cand and all(self._python_can_import(cand, mod) for mod in required_modules):
                return cand

        # Log which modules are missing for debugging
        for cand in candidates:
            if cand:
                missing = [mod for mod in required_modules if not self._python_can_import(cand, mod)]
                if missing:
                    logger.warning(f"Python {cand} missing modules: {missing}")

        return sys.executable or "python3"

    def _check_port_available(self, port: int) -> bool:
        """Check if a port is available for binding."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return True
        except OSError:
            return False

    def _kill_port_holder(self, port: int) -> bool:
        """Kill any process holding a port. Returns True if port becomes available.

        IMPORTANT: Before killing, we do a final health check with a generous timeout
        to avoid killing a healthy but slow orchestrator.
        """
        # Final safety check: try one more health check with a very long timeout
        # before killing. This prevents killing a healthy but busy orchestrator.
        try:
            url = f"http://localhost:{port}/health"
            with urllib.request.urlopen(url, timeout=45) as response:
                data = json.loads(response.read().decode())
                if data.get("healthy") or data.get("status") == "ok":
                    logger.info(f"Port {port} holder responded to health check - not killing")
                    return False  # Don't kill, it's actually healthy
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
            pass  # Continue to kill if health check failed

        try:
            # Find PIDs using the port via fuser
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split()
                for pid in pids:
                    try:
                        pid_int = int(pid.strip())
                        logger.warning(f"Killing process {pid_int} holding port {port}")
                        os.kill(pid_int, signal.SIGKILL)
                    except (ValueError, ProcessLookupError):
                        pass
                time.sleep(1)
                return self._check_port_available(port)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError) as e:
            logger.debug(f"fuser check failed: {e}")

        # Fallback: kill any p2p_orchestrator processes
        try:
            subprocess.run(
                ["pkill", "-9", "-f", "p2p_orchestrator.py"],
                capture_output=True,
                timeout=5,
            )
            time.sleep(2)
            return self._check_port_available(port)
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return False

    def _is_in_startup_grace_period(self) -> bool:
        """Check if we're within the startup grace period since last P2P start attempt.

        During this period (default 120s), avoid killing port holders to give P2P
        time to load state and become responsive. This prevents restart loops when
        P2P takes time to load large state files.
        """
        if self.p2p_last_start_attempt is None:
            return False
        grace_period = int(os.environ.get("RINGRIFT_P2P_STARTUP_GRACE_PERIOD", "120"))
        elapsed = time.time() - self.p2p_last_start_attempt
        return elapsed < grace_period

    def start_p2p_orchestrator(self) -> bool:
        """Start the P2P orchestrator if not running.

        Includes retry logic with port availability checking to handle
        race conditions where the port is briefly unavailable after crashes.
        """
        if self.check_p2p_health():
            return True

        logger.info("Starting P2P orchestrator...")

        # Check port availability and clear if needed
        if not self._check_port_available(self.config.p2p_port):
            # Don't kill port holder during startup grace period
            if self._is_in_startup_grace_period():
                grace_remaining = int(os.environ.get("RINGRIFT_P2P_STARTUP_GRACE_PERIOD", "120")) - (time.time() - (self.p2p_last_start_attempt or 0))
                logger.info(
                    f"Port {self.config.p2p_port} in use but within startup grace period "
                    f"({grace_remaining:.0f}s remaining) - waiting for P2P to initialize"
                )
                return False  # Don't kill, just wait

            logger.warning(f"Port {self.config.p2p_port} is in use, attempting to clear...")
            if not self._kill_port_holder(self.config.p2p_port):
                logger.error(f"Could not free port {self.config.p2p_port}")
                return False
            logger.info(f"Port {self.config.p2p_port} is now available")

        # Mark the start attempt timestamp before launching
        self.p2p_last_start_attempt = time.time()

        try:
            # Prefer systemd on full VM hosts to avoid split-brain between
            # systemd units and a directly-spawned process (which can lead to
            # bind failures and "ghost" orchestrators after restarts).
            if self._systemd_usable() and self._systemd_unit_exists("ringrift-p2p.service"):
                logger.info("Starting P2P orchestrator via systemd (ringrift-p2p.service)")
                subprocess.run(
                    ["systemctl", "start", "ringrift-p2p.service"],
                    capture_output=True,
                    text=True,
                    timeout=60,  # Increased from 10s - P2P needs time to initialize
                )
                deadline = time.time() + 60  # Increased from 20s
                while time.time() < deadline:
                    if self.check_p2p_health():
                        return True
                    time.sleep(1)
                logger.warning("systemd start issued but /health did not become ready in time")
                return False

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

            # Wait longer for P2P to initialize (loading state, connecting to peers)
            deadline = time.time() + 15
            while time.time() < deadline:
                if proc.poll() is not None:
                    logger.warning(f"P2P orchestrator exited with code {proc.returncode}")
                    break
                if self.check_p2p_health():
                    logger.info(f"P2P orchestrator started (PID {proc.pid})")
                    return True
                time.sleep(1)

            # If process is still running but not healthy, give it more time
            if proc.poll() is None:
                logger.info("P2P process running but not healthy yet, waiting...")
                for _ in range(10):
                    time.sleep(1)
                    if self.check_p2p_health():
                        logger.info(f"P2P orchestrator started (PID {proc.pid})")
                        return True
                logger.warning("P2P did not become healthy within timeout")
        except Exception as e:
            logger.error(f"Failed to start P2P orchestrator: {e}")
        return False

    def _start_gpu_fallback_selfplay(self, num_to_start: int) -> None:
        """Start GPU fallback selfplay workers with diverse profiles.

        Uses weighted random selection from DIVERSE_ENGINE_PROFILES to ensure
        high-quality, diverse training data including MaxN/BRS for multiplayer.
        """
        if self.config.num_gpus <= 0 or num_to_start <= 0:
            return

        logger.info(f"Starting {num_to_start} GPU fallback selfplay workers with diverse profiles")

        # Select diverse profiles for each worker
        weights = [p["weight"] for p in DIVERSE_ENGINE_PROFILES]
        selected_profiles = random.choices(DIVERSE_ENGINE_PROFILES, weights=weights, k=num_to_start)

        for i, profile in enumerate(selected_profiles):
            gpu_id = (len(self.local_selfplay_pids) + i) % max(self.config.num_gpus, 1)
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = self.config.ai_service_dir
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                env["RINGRIFT_JOB_ORIGIN"] = "resilience_fallback"

                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                # Include engine mode in output dir for easier tracking
                engine_mode = profile["engine_mode"]
                board_type = profile["board_type"]
                num_players = profile["num_players"]
                output_dir = os.path.join(
                    self.config.ai_service_dir,
                    "data/selfplay/fallback",
                    self.config.node_id,
                    f"gpu{gpu_id}",
                    f"{engine_mode}_{board_type}_{num_players}p",
                    ts,
                )

                script = (self.config.selfplay_script or "scripts/run_hybrid_selfplay.py").strip()
                script_path = script if os.path.isabs(script) else os.path.join(self.config.ai_service_dir, script)

                cmd: list[str]
                if script_path.endswith("run_gpu_selfplay.py"):
                    cmd = [
                        sys.executable,
                        script_path,
                        "--board", board_type,
                        "--num-players", str(num_players),
                        "--num-games", str(self.config.fallback_num_games_gpu),
                        "--batch-size", str(self.config.fallback_batch_size_gpu),
                        "--max-moves", "10000",  # Avoid draws due to move limit
                        "--output-dir", output_dir,
                        "--seed", str(int(time.time()) + gpu_id),
                    ]
                else:
                    # Default: hybrid selfplay (CPU rules + GPU eval) with diverse engine mode
                    cmd = [
                        sys.executable,
                        script_path,
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--num-games", str(self.config.fallback_num_games_gpu),
                        "--max-moves", "10000",  # Avoid draws due to move limit
                        "--output-dir", output_dir,
                        "--engine-mode", engine_mode,
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
                logger.info(f"Started GPU fallback selfplay on GPU {gpu_id}: {engine_mode} {board_type} {num_players}P (PID {proc.pid})")
            except Exception as e:
                logger.error(f"Failed to start GPU fallback selfplay on GPU {gpu_id}: {e}")

    def _start_cpu_fallback_selfplay(self, num_to_start: int) -> None:
        """Start CPU fallback selfplay workers (for non-GPU nodes)."""
        if num_to_start <= 0:
            return

        logger.info(f"Starting {num_to_start} CPU fallback selfplay workers")
        for _i in range(num_to_start):
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
                        "--engine-mode", "descent-only",  # Prefer descent for CPU (efficient)
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

    def _target_fallback_job_ids(self) -> list[str]:
        max_procs = max(1, int(self.config.max_local_selfplay_procs or 1))
        prefix = self.fallback_job_prefix
        if self.config.num_gpus > 0:
            target = min(int(self.config.num_gpus or 0), max_procs)
            return [f"{prefix}gpu{gpu_id}" for gpu_id in range(target)]
        return [f"{prefix}cpu{i}" for i in range(max_procs)]

    def _start_fallback_jobs_via_orchestrator(self) -> bool:
        """Start fallback work using the local P2P orchestrator so jobs are tracked."""
        if not self.check_p2p_health():
            return False

        desired_job_ids = self._target_fallback_job_ids()

        running_jobs = self._list_local_jobs(status="running", limit=500)
        running_fallback = {
            str(j.get("job_id") or "")
            for j in running_jobs
            if str(j.get("job_id") or "").startswith(self.fallback_job_prefix)
        }

        started_any = False
        for job_id in desired_job_ids:
            if job_id in running_fallback:
                continue

            if self.config.num_gpus > 0:
                job_type = "hybrid_selfplay"
                cuda_visible_devices = job_id.split("gpu")[-1] if "gpu" in job_id else None
            else:
                job_type = "selfplay"
                cuda_visible_devices = None

            payload: dict[str, Any] = {
                "job_id": job_id,
                "job_type": job_type,
                "board_type": self.config.fallback_board,
                "num_players": int(self.config.fallback_num_players or 2),
                "engine_mode": "nn-only",  # Use neural network for quality training data
            }
            if cuda_visible_devices is not None:
                payload["cuda_visible_devices"] = str(cuda_visible_devices)

            resp = self._local_orchestrator_post_json("/start_job", payload, timeout=30) or {}
            if resp.get("success"):
                started_any = True
                if job_id not in self.fallback_job_ids:
                    self.fallback_job_ids.append(job_id)
                    self._save_state()
            else:
                # If orchestrator start fails, don't spam retries in a tight loop.
                err = resp.get("error") if isinstance(resp, dict) else None
                logger.warning(f"Failed to start fallback via orchestrator (job_id={job_id}): {err or 'unknown_error'}")

        return started_any or bool(running_fallback)

    def _start_fallback_processes_direct(self) -> None:
        """Direct fallback selfplay (legacy). Prefer orchestrator-managed jobs when possible."""
        # LEARNED LESSONS - Don't start direct fallback if P2P is already managing jobs.
        if self.check_p2p_managing_jobs():
            logger.info("P2P is managing jobs, skipping direct fallback work")
            return

        # Clean up dead processes
        self.local_selfplay_pids = [pid for pid in self.local_selfplay_pids if self._process_running(pid)]

        max_procs = max(1, self.config.max_local_selfplay_procs)
        num_to_start = max_procs - len(self.local_selfplay_pids)
        if num_to_start <= 0:
            return

        if self.config.num_gpus > 0:
            self._start_gpu_fallback_selfplay(num_to_start=min(num_to_start, self.config.num_gpus))
        else:
            self._start_cpu_fallback_selfplay(num_to_start=min(num_to_start, 1))

        self._save_state()

    def start_local_fallback_work(self) -> None:
        """Start local fallback work when P2P is unavailable."""
        # Avoid spawning new work when disk is under pressure.
        if not self.check_and_cleanup_disk():
            logger.warning("Skipping fallback work due to disk pressure")
            return

        # Prefer orchestrator-managed jobs (tracked + visible to the leader).
        if self._start_fallback_jobs_via_orchestrator():
            return

        # Last resort: start direct processes if orchestrator is unavailable.
        self._start_fallback_processes_direct()

    def stop_direct_fallback_processes(self) -> None:
        """Stop only the direct (untracked) fallback selfplay processes."""
        for pid in self.local_selfplay_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Stopped direct fallback selfplay (PID {pid})")
            except ProcessLookupError:
                pass
        self.local_selfplay_pids = []
        self._save_state()

    def stop_orchestrator_fallback_jobs(self) -> None:
        """Stop orchestrator-tracked fallback jobs (identified by job_id prefix)."""
        if not self.check_p2p_health():
            return
        running_jobs = self._list_local_jobs(status="running", limit=500)
        fallback_job_ids = [
            str(j.get("job_id") or "")
            for j in running_jobs
            if str(j.get("job_id") or "").startswith(self.fallback_job_prefix)
        ]
        for job_id in fallback_job_ids:
            resp = self._local_orchestrator_post_json("/stop_job", {"job_id": job_id}, timeout=15) or {}
            if resp.get("success"):
                logger.info(f"Stopped orchestrator fallback job {job_id}")
        self.fallback_job_ids = [j for j in self.fallback_job_ids if j not in set(fallback_job_ids)]
        self._save_state()

    def stop_local_fallback(self) -> None:
        """Stop all fallback work (orchestrator jobs + direct processes)."""
        self.stop_orchestrator_fallback_jobs()
        self.stop_direct_fallback_processes()

    def check_and_cleanup_disk(self) -> bool:
        """Check disk usage and run cleanup if needed."""
        try:
            # Measure disk pressure on the volume that actually contains the
            # RingRift checkout/data. On macOS (APFS split volumes) and some
            # container overlays, checking "/" can under-report the data volume.
            stat = os.statvfs(self._ringrift_root())
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
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
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
                # GPU idle threshold: 10 minutes default (increased Dec 2025)
                # MCTS simulations can legitimately run for several minutes
                gpu_idle_threshold = int(os.environ.get("RINGRIFT_GPU_IDLE_THRESHOLD", "600"))
                if idle_duration > gpu_idle_threshold:
                    logger.error(f"Stuck processes detected: {selfplay_procs} procs, GPU idle for {idle_duration:.0f}s")
                    logger.info("Sending SIGTERM to stuck selfplay processes (graceful shutdown)...")

                    # Send SIGTERM first for graceful shutdown (Dec 2025 fix)
                    for pid in self.local_selfplay_pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except (ProcessLookupError, OSError):
                            continue

                    # Wait briefly for graceful shutdown before SIGKILL
                    time.sleep(10)
                    for pid in self.local_selfplay_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except (ProcessLookupError, OSError):
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
                    except (ValueError, TypeError):
                        continue
            except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
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
            except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
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
            except (subprocess.SubprocessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        after = self._count_selfplay_processes()
        return max(0, before - after)

    def check_runaway_selfplay(self) -> bool:
        """Detect runaway selfplay counts and trigger a restart sweep.

        Returns True when no runaway condition is detected.
        """
        # Prefer the orchestrator-tracked job list when available.
        if self.check_p2p_health():
            try:
                jobs = self._list_local_jobs(status="running", limit=1000)
                count = sum(
                    1
                    for j in jobs
                    if str(j.get("job_type") or "") in {"selfplay", "gpu_selfplay", "hybrid_selfplay"}
                )
                if count < RUNAWAY_SELFPLAY_PROCESS_THRESHOLD:
                    return True

                logger.error(
                    f"Runaway selfplay detected via local P2P jobs: {count} >= {RUNAWAY_SELFPLAY_PROCESS_THRESHOLD}"
                )

                req = urllib.request.Request(
                    self._local_orchestrator_url("/restart_stuck_jobs"),
                    data=b"{}",
                    headers=self._local_orchestrator_headers(json_body=True),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    _ = resp.read()
                logger.info("Requested /restart_stuck_jobs for runaway recovery")
                return False
            except Exception as e:
                logger.warning(f"Runaway selfplay check (via local P2P jobs) failed: {e}")

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
        cluster_connected = self.check_cluster_connected()

        if p2p_healthy and cluster_connected:
            # P2P is healthy - clear grace period tracking
            self.p2p_unhealthy_since = None
            self.p2p_last_start_attempt = None  # Clear startup grace period
            if not self.p2p_connected:
                logger.info("P2P connection restored - stopping direct fallback processes")
                self.stop_direct_fallback_processes()
            self.p2p_connected = True
        else:
            # P2P appears unhealthy - track duration and apply grace period
            if self.p2p_unhealthy_since is None:
                self.p2p_unhealthy_since = now

            unhealthy_duration = now - self.p2p_unhealthy_since
            # Grace period reduced from 600s (10min) to 180s (3min) for faster failover (Dec 2025)
            grace_period = int(os.environ.get("RINGRIFT_P2P_GRACE_PERIOD", "180"))

            if unhealthy_duration < grace_period:
                # Still in grace period - just log and wait
                if self.p2p_connected:
                    logger.warning(
                        f"P2P appears unhealthy ({unhealthy_duration:.0f}s), "
                        f"waiting {grace_period - unhealthy_duration:.0f}s before intervention"
                    )
                return  # Don't take action yet

            # Grace period expired - take action
            if self.p2p_connected:
                logger.warning(
                    f"P2P unhealthy for {unhealthy_duration:.0f}s (>{grace_period}s) - "
                    "starting local fallback"
                )
            self.p2p_connected = False

            # Try to start P2P orchestrator
            if not p2p_healthy:
                self.start_p2p_orchestrator()

            # For Vast instances, try autossh tunnel as backup connectivity
            if not cluster_connected and self.config.node_id.startswith("vast"):
                self.start_autossh_tunnel()

            # Start local fallback work
            self.start_local_fallback_work()

        # Periodic registration
        if now - self.last_registration > self.config.reconnect_interval:
            if self.check_coordinator_reachable():
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
            self.stop_local_fallback()
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
    parser.add_argument("--disk-threshold", type=int, default=70, help="Disk usage percent threshold for cleanup (70% enforced 2025-12-15)")
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
