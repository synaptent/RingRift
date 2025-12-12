#!/usr/bin/env python3
"""Distributed P2P Orchestrator - Self-healing compute cluster for RingRift AI training.

This orchestrator runs on each node in the cluster and:
1. Discovers other nodes via broadcast UDP or known peer list
2. Participates in leader election for coordination tasks
3. Monitors local resources and shares status with peers
4. Auto-starts selfplay/training jobs based on cluster needs
5. Self-heals when nodes go offline or IPs change

Architecture:
- Each node runs this script as a daemon
- Nodes communicate via HTTP REST API (port 8770)
- Leader election uses Bully algorithm (highest node_id wins)
- Heartbeats every 30 seconds detect failures
- Nodes maintain local SQLite state for crash recovery

Usage:
    # On each node:
    python scripts/p2p_orchestrator.py --node-id mac-studio
    python scripts/p2p_orchestrator.py --node-id vast-5090-quad --port 8770

    # With known peers (for cloud nodes without broadcast):
    python scripts/p2p_orchestrator.py --node-id vast-3090 --peers 100.107.168.125:8770,100.66.142.46:8770
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

# HTTP server imports
try:
    from aiohttp import web, ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("Warning: aiohttp not installed. Install with: pip install aiohttp")

# ============================================
# Configuration
# ============================================

DEFAULT_PORT = 8770
HEARTBEAT_INTERVAL = 30  # seconds
PEER_TIMEOUT = 90  # seconds without heartbeat = node considered dead
ELECTION_TIMEOUT = 10  # seconds to wait for election responses
JOB_CHECK_INTERVAL = 60  # seconds between job status checks
DISCOVERY_PORT = 8771  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# LEARNED LESSONS from PLAN.md - Disk and resource thresholds
DISK_CRITICAL_THRESHOLD = 90  # Stop all new jobs at 90% disk
DISK_WARNING_THRESHOLD = 80   # Reduce job count at 80% disk
DISK_CLEANUP_THRESHOLD = 85   # Trigger automatic cleanup at 85%
MEMORY_CRITICAL_THRESHOLD = 95  # OOM prevention - stop jobs at 95%
MEMORY_WARNING_THRESHOLD = 85   # Reduce jobs at 85% memory

# LEARNED LESSONS - Connection robustness
HTTP_CONNECT_TIMEOUT = 10     # Fast timeout for connection phase
HTTP_TOTAL_TIMEOUT = 30       # Total request timeout
MAX_CONSECUTIVE_FAILURES = 3  # Mark node dead after 3 failures
RETRY_DEAD_NODE_INTERVAL = 300  # Retry dead nodes every 5 minutes

# Git auto-update settings
GIT_UPDATE_CHECK_INTERVAL = 300  # Check for updates every 5 minutes
GIT_REMOTE_NAME = "origin"       # Git remote to check
GIT_BRANCH_NAME = "main"         # Branch to track
AUTO_UPDATE_ENABLED = True       # Enable automatic updates
GRACEFUL_SHUTDOWN_BEFORE_UPDATE = True  # Stop jobs before updating

# Path to local state database
STATE_DIR = Path(__file__).parent.parent / "logs" / "p2p_orchestrator"
STATE_DIR.mkdir(parents=True, exist_ok=True)


class NodeRole(str, Enum):
    """Role a node plays in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class JobType(str, Enum):
    """Types of jobs nodes can run."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated parallel selfplay
    TRAINING = "training"
    CMAES = "cmaes"


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    role: NodeRole = NodeRole.FOLLOWER
    last_heartbeat: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    has_gpu: bool = False
    gpu_name: str = ""
    memory_gb: int = 0
    capabilities: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    # LEARNED LESSONS - Track connection failures for adaptive retry
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    # LEARNED LESSONS - Track resource issues
    disk_cleanup_needed: bool = False
    oom_events: int = 0
    last_oom_time: float = 0.0

    def is_alive(self) -> bool:
        """Check if node is considered alive based on last heartbeat."""
        return time.time() - self.last_heartbeat < PEER_TIMEOUT

    def is_healthy(self) -> bool:
        """Check if node is healthy for new jobs (not just reachable)."""
        if not self.is_alive():
            return False
        # LEARNED LESSONS - Don't start jobs on resource-constrained nodes
        if self.disk_percent >= DISK_CRITICAL_THRESHOLD:
            return False
        if self.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
            return False
        return True

    def should_retry(self) -> bool:
        """Check if we should retry connecting to a failed node."""
        if self.consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            return True
        # LEARNED LESSONS - Retry dead nodes periodically
        return time.time() - self.last_failure_time > RETRY_DEAD_NODE_INTERVAL

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['role'] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeInfo':
        """Create from dictionary."""
        d = d.copy()
        d['role'] = NodeRole(d.get('role', 'follower'))
        # Handle missing new fields gracefully
        d.setdefault('consecutive_failures', 0)
        d.setdefault('last_failure_time', 0.0)
        d.setdefault('disk_cleanup_needed', False)
        d.setdefault('oom_events', 0)
        d.setdefault('last_oom_time', 0.0)
        return cls(**d)


@dataclass
class ClusterJob:
    """A job running in the cluster."""
    job_id: str
    job_type: JobType
    node_id: str
    board_type: str = "square8"
    num_players: int = 2
    engine_mode: str = "descent-only"
    pid: int = 0
    started_at: float = 0.0
    status: str = "running"

    def to_dict(self) -> dict:
        d = asdict(self)
        d['job_type'] = self.job_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ClusterJob':
        d = d.copy()
        d['job_type'] = JobType(d.get('job_type', 'selfplay'))
        return cls(**d)


class P2POrchestrator:
    """Main P2P orchestrator class that runs on each node."""

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        known_peers: List[str] = None,
        ringrift_path: str = None,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.known_peers = known_peers or []
        self.ringrift_path = ringrift_path or self._detect_ringrift_path()

        # Node state
        self.role = NodeRole.FOLLOWER
        self.leader_id: Optional[str] = None
        self.peers: Dict[str, NodeInfo] = {}
        self.local_jobs: Dict[str, ClusterJob] = {}

        # Locks for thread safety
        self.peers_lock = threading.Lock()
        self.jobs_lock = threading.Lock()

        # State persistence
        self.db_path = STATE_DIR / f"{node_id}_state.db"
        self._init_database()

        # Event flags
        self.running = True
        self.election_in_progress = False

        # Load persisted state
        self._load_state()

        # Self info
        self.self_info = self._create_self_info()

        print(f"[P2P] Initialized node {node_id} on {host}:{port}")
        print(f"[P2P] RingRift path: {self.ringrift_path}")
        print(f"[P2P] Known peers: {self.known_peers}")

    def _detect_ringrift_path(self) -> str:
        """Detect the RingRift installation path."""
        # Try common locations
        candidates = [
            Path.home() / "Development" / "RingRift",
            Path.home() / "ringrift",
            Path("/home/ubuntu/ringrift"),
            Path("/root/ringrift"),
        ]
        for path in candidates:
            if (path / "ai-service").exists():
                return str(path)
        return str(Path(__file__).parent.parent.parent)

    def _init_database(self):
        """Initialize SQLite database for state persistence."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Peers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS peers (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                last_heartbeat REAL,
                info_json TEXT
            )
        """)

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT,
                node_id TEXT,
                board_type TEXT,
                num_players INTEGER,
                engine_mode TEXT,
                pid INTEGER,
                started_at REAL,
                status TEXT
            )
        """)

        # State table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load persisted state from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Load peers
            cursor.execute("SELECT node_id, info_json FROM peers")
            for row in cursor.fetchall():
                try:
                    info = NodeInfo.from_dict(json.loads(row[1]))
                    self.peers[row[0]] = info
                except Exception as e:
                    print(f"[P2P] Failed to load peer {row[0]}: {e}")

            # Load jobs
            cursor.execute("SELECT * FROM jobs WHERE status = 'running'")
            for row in cursor.fetchall():
                job = ClusterJob(
                    job_id=row[0],
                    job_type=JobType(row[1]),
                    node_id=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    engine_mode=row[5],
                    pid=row[6],
                    started_at=row[7],
                    status=row[8],
                )
                self.local_jobs[job.job_id] = job

            # Load leader
            cursor.execute("SELECT value FROM state WHERE key = 'leader_id'")
            row = cursor.fetchone()
            if row:
                self.leader_id = row[0]

            conn.close()
            print(f"[P2P] Loaded state: {len(self.peers)} peers, {len(self.local_jobs)} jobs")
        except Exception as e:
            print(f"[P2P] Failed to load state: {e}")

    def _save_state(self):
        """Save current state to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Save peers
            with self.peers_lock:
                for node_id, info in self.peers.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO peers (node_id, host, port, last_heartbeat, info_json)
                        VALUES (?, ?, ?, ?, ?)
                    """, (node_id, info.host, info.port, info.last_heartbeat, json.dumps(info.to_dict())))

            # Save jobs
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO jobs
                        (job_id, job_type, node_id, board_type, num_players, engine_mode, pid, started_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (job.job_id, job.job_type.value, job.node_id, job.board_type,
                          job.num_players, job.engine_mode, job.pid, job.started_at, job.status))

            # Save leader
            cursor.execute("""
                INSERT OR REPLACE INTO state (key, value) VALUES ('leader_id', ?)
            """, (self.leader_id,))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[P2P] Failed to save state: {e}")

    def _create_self_info(self) -> NodeInfo:
        """Create NodeInfo for this node."""
        # Detect GPU
        has_gpu, gpu_name = self._detect_gpu()

        # Detect memory
        memory_gb = self._detect_memory()

        # Detect capabilities based on hardware
        capabilities = ["selfplay"]
        if has_gpu:
            capabilities.extend(["training", "cmaes"])
        if memory_gb >= 64:
            capabilities.append("large_boards")

        return NodeInfo(
            node_id=self.node_id,
            host=self._get_local_ip(),
            port=self.port,
            role=self.role,
            last_heartbeat=time.time(),
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            capabilities=capabilities,
        )

    def _detect_gpu(self) -> Tuple[bool, str]:
        """Detect if GPU is available and its name."""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip().split('\n')[0]
        except:
            pass

        try:
            # Try MPS (Apple Silicon)
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True, text=True, timeout=10
            )
            if "True" in result.stdout:
                return True, "Apple MPS"
        except:
            pass

        return False, ""

    def _detect_memory(self) -> int:
        """Detect total system memory in GB."""
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                return int(result.stdout.strip()) // (1024**3)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) // (1024**2)
        except:
            pass
        return 16  # Default assumption

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        result = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "gpu_percent": 0.0,
            "gpu_memory_percent": 0.0,
        }

        try:
            # CPU
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["ps", "-A", "-o", "%cpu"],
                    capture_output=True, text=True, timeout=5
                )
                cpus = [float(x) for x in out.stdout.strip().split('\n')[1:] if x.strip()]
                result["cpu_percent"] = min(100.0, sum(cpus) / os.cpu_count())
            else:
                with open("/proc/loadavg") as f:
                    load = float(f.read().split()[0])
                    result["cpu_percent"] = min(100.0, load * 100 / os.cpu_count())

            # Memory
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=5
                )
                # Parse vm_stat output
                lines = out.stdout.strip().split('\n')
                stats = {}
                for line in lines[1:]:
                    if ':' in line:
                        key, val = line.split(':')
                        stats[key.strip()] = int(val.strip().rstrip('.'))
                page_size = 16384  # Usually 16KB on M1
                free = stats.get('Pages free', 0) * page_size
                total = self._detect_memory() * (1024**3)
                result["memory_percent"] = 100.0 * (1 - free / total) if total > 0 else 0.0
            else:
                with open("/proc/meminfo") as f:
                    mem = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            mem[parts[0].rstrip(':')] = int(parts[1])
                    total = mem.get('MemTotal', 1)
                    avail = mem.get('MemAvailable', mem.get('MemFree', 0))
                    result["memory_percent"] = 100.0 * (1 - avail / total)

            # Disk
            import shutil
            usage = shutil.disk_usage(self.ringrift_path)
            result["disk_percent"] = 100.0 * usage.used / usage.total

            # GPU (NVIDIA)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0:
                    parts = out.stdout.strip().split(',')
                    result["gpu_percent"] = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    result["gpu_memory_percent"] = 100.0 * mem_used / mem_total
            except:
                pass

        except Exception as e:
            print(f"[P2P] Resource check error: {e}")

        return result

    def _count_local_jobs(self) -> Tuple[int, int]:
        """Count running selfplay and training jobs on this node."""
        selfplay = 0
        training = 0

        try:
            # Count python processes running selfplay
            out = subprocess.run(
                ["pgrep", "-f", "run_self_play"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                selfplay = len(out.stdout.strip().split('\n'))

            # Count training processes
            out = subprocess.run(
                ["pgrep", "-f", "train_"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                training = len([p for p in out.stdout.strip().split('\n') if p])
        except:
            pass

        return selfplay, training

    # ============================================
    # Git Auto-Update Methods
    # ============================================

    def _get_local_git_commit(self) -> Optional[str]:
        """Get the current local git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get local git commit: {e}")
        return None

    def _get_local_git_branch(self) -> Optional[str]:
        """Get the current local git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get local git branch: {e}")
        return None

    def _get_remote_git_commit(self) -> Optional[str]:
        """Fetch and get the remote branch's latest commit hash."""
        try:
            # First fetch to update remote refs
            fetch_result = subprocess.run(
                ["git", "fetch", GIT_REMOTE_NAME, GIT_BRANCH_NAME],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=60
            )
            if fetch_result.returncode != 0:
                print(f"[P2P] Git fetch failed: {fetch_result.stderr}")
                return None

            # Get remote branch commit
            result = subprocess.run(
                ["git", "rev-parse", f"{GIT_REMOTE_NAME}/{GIT_BRANCH_NAME}"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get remote git commit: {e}")
        return None

    def _check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if there are updates available from GitHub.

        Returns: (has_updates, local_commit, remote_commit)
        """
        local_commit = self._get_local_git_commit()
        remote_commit = self._get_remote_git_commit()

        if not local_commit or not remote_commit:
            return False, local_commit, remote_commit

        has_updates = local_commit != remote_commit
        return has_updates, local_commit, remote_commit

    def _get_commits_behind(self, local_commit: str, remote_commit: str) -> int:
        """Get the number of commits the local branch is behind remote."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", f"{local_commit}..{remote_commit}"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to count commits behind: {e}")
        return 0

    def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # If there's output, there are uncommitted changes
                return bool(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to check local changes: {e}")
        return True  # Assume changes exist on error (safer)

    async def _stop_all_local_jobs(self) -> int:
        """Stop all local jobs gracefully before update.

        Returns: Number of jobs stopped
        """
        stopped = 0
        with self.jobs_lock:
            for job_id, job in list(self.local_jobs.items()):
                try:
                    if job.pid > 0:
                        os.kill(job.pid, signal.SIGTERM)
                        print(f"[P2P] Sent SIGTERM to job {job_id} (PID {job.pid})")
                        stopped += 1
                        job.status = "stopping"
                except ProcessLookupError:
                    # Process already gone
                    job.status = "stopped"
                except Exception as e:
                    print(f"[P2P] Failed to stop job {job_id}: {e}")

        # Wait for processes to terminate
        if stopped > 0:
            await asyncio.sleep(5)

            # Force kill any remaining
            with self.jobs_lock:
                for job_id, job in list(self.local_jobs.items()):
                    if job.status == "stopping" and job.pid > 0:
                        try:
                            os.kill(job.pid, signal.SIGKILL)
                            print(f"[P2P] Force killed job {job_id}")
                        except:
                            pass
                        job.status = "stopped"

        return stopped

    async def _perform_git_update(self) -> Tuple[bool, str]:
        """Perform git pull to update the codebase.

        Returns: (success, message)
        """
        # Check for local changes
        if self._check_local_changes():
            return False, "Local changes detected. Cannot auto-update. Please commit or stash changes."

        # Stop jobs if configured
        if GRACEFUL_SHUTDOWN_BEFORE_UPDATE:
            stopped = await self._stop_all_local_jobs()
            if stopped > 0:
                print(f"[P2P] Stopped {stopped} jobs before update")

        try:
            # Perform git pull
            result = subprocess.run(
                ["git", "pull", GIT_REMOTE_NAME, GIT_BRANCH_NAME],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                return False, f"Git pull failed: {result.stderr}"

            print(f"[P2P] Git pull successful: {result.stdout}")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            return False, "Git pull timed out"
        except Exception as e:
            return False, f"Git pull error: {e}"

    async def _restart_orchestrator(self):
        """Restart the orchestrator process after update."""
        print("[P2P] Restarting orchestrator to apply updates...")

        # Save state before restart
        self._save_state()

        # Get current script path and arguments
        script_path = Path(__file__).resolve()
        args = sys.argv[1:]

        # Schedule restart
        await asyncio.sleep(2)

        # Use exec to replace current process
        os.execv(sys.executable, [sys.executable, str(script_path)] + args)

    async def _git_update_loop(self):
        """Background loop to periodically check for and apply updates."""
        if not AUTO_UPDATE_ENABLED:
            print("[P2P] Auto-update disabled")
            return

        print(f"[P2P] Git auto-update loop started (interval: {GIT_UPDATE_CHECK_INTERVAL}s)")

        while self.running:
            try:
                await asyncio.sleep(GIT_UPDATE_CHECK_INTERVAL)

                if not self.running:
                    break

                # Check for updates
                has_updates, local_commit, remote_commit = self._check_for_updates()

                if has_updates and local_commit and remote_commit:
                    commits_behind = self._get_commits_behind(local_commit, remote_commit)
                    print(f"[P2P] Update available: {commits_behind} commits behind")
                    print(f"[P2P] Local:  {local_commit[:8]}")
                    print(f"[P2P] Remote: {remote_commit[:8]}")

                    # Perform update
                    success, message = await self._perform_git_update()

                    if success:
                        print(f"[P2P] Update successful, restarting...")
                        await self._restart_orchestrator()
                    else:
                        print(f"[P2P] Update failed: {message}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Git update loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry on error

    # ============================================
    # HTTP API Handlers
    # ============================================

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle heartbeat from peer node."""
        try:
            data = await request.json()
            peer_info = NodeInfo.from_dict(data)
            peer_info.last_heartbeat = time.time()

            with self.peers_lock:
                self.peers[peer_info.node_id] = peer_info

            # Return our info
            self._update_self_info()
            return web.json_response(self.self_info.to_dict())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return cluster status."""
        self._update_self_info()

        with self.peers_lock:
            peers = {k: v.to_dict() for k, v in self.peers.items()}

        with self.jobs_lock:
            jobs = {k: v.to_dict() for k, v in self.local_jobs.items()}

        return web.json_response({
            "node_id": self.node_id,
            "role": self.role.value,
            "leader_id": self.leader_id,
            "self": self.self_info.to_dict(),
            "peers": peers,
            "local_jobs": jobs,
            "alive_peers": len([p for p in self.peers.values() if p.is_alive()]),
        })

    async def handle_election(self, request: web.Request) -> web.Response:
        """Handle election message from another node."""
        try:
            data = await request.json()
            candidate_id = data.get("candidate_id")

            # If our ID is higher, we respond with "ALIVE" (Bully algorithm)
            if self.node_id > candidate_id:
                # Start our own election
                asyncio.create_task(self._start_election())
                return web.json_response({"response": "ALIVE", "node_id": self.node_id})
            else:
                return web.json_response({"response": "OK"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_coordinator(self, request: web.Request) -> web.Response:
        """Handle coordinator announcement from new leader."""
        try:
            data = await request.json()
            new_leader = data.get("leader_id")

            print(f"[P2P] New leader announced: {new_leader}")
            self.leader_id = new_leader
            if new_leader == self.node_id:
                self.role = NodeRole.LEADER
            else:
                self.role = NodeRole.FOLLOWER

            self._save_state()
            return web.json_response({"accepted": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_start_job(self, request: web.Request) -> web.Response:
        """Handle request to start a job (from leader)."""
        try:
            data = await request.json()
            job_type = JobType(data.get("job_type", "selfplay"))
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            engine_mode = data.get("engine_mode", "descent-only")

            job = await self._start_local_job(job_type, board_type, num_players, engine_mode)

            if job:
                return web.json_response({"success": True, "job": job.to_dict()})
            else:
                return web.json_response({"success": False, "error": "Failed to start job"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_stop_job(self, request: web.Request) -> web.Response:
        """Handle request to stop a job."""
        try:
            data = await request.json()
            job_id = data.get("job_id")

            with self.jobs_lock:
                if job_id in self.local_jobs:
                    job = self.local_jobs[job_id]
                    try:
                        os.kill(job.pid, signal.SIGTERM)
                        job.status = "stopped"
                    except:
                        pass
                    return web.json_response({"success": True})

            return web.json_response({"success": False, "error": "Job not found"}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_cleanup(self, request: web.Request) -> web.Response:
        """Handle cleanup request (from leader or manual).

        LEARNED LESSONS - This endpoint allows remote nodes to trigger disk cleanup
        when the leader detects disk usage approaching critical thresholds.
        """
        try:
            print(f"[P2P] Cleanup request received")

            # Run cleanup in background to avoid blocking the request
            asyncio.create_task(self._cleanup_local_disk())

            # Return current disk usage
            usage = self._get_resource_usage()
            return web.json_response({
                "success": True,
                "disk_percent_before": usage["disk_percent"],
                "message": "Cleanup initiated",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request.

        LEARNED LESSONS - Simple health endpoint for monitoring and load balancers.
        Returns node health status without full cluster state.
        """
        try:
            self._update_self_info()
            is_healthy = self.self_info.is_healthy()

            return web.json_response({
                "healthy": is_healthy,
                "node_id": self.node_id,
                "role": self.role.value,
                "disk_percent": self.self_info.disk_percent,
                "memory_percent": self.self_info.memory_percent,
                "cpu_percent": self.self_info.cpu_percent,
                "selfplay_jobs": self.self_info.selfplay_jobs,
                "training_jobs": self.self_info.training_jobs,
            })
        except Exception as e:
            return web.json_response({"error": str(e), "healthy": False}, status=500)

    async def handle_git_status(self, request: web.Request) -> web.Response:
        """Get git status for this node.

        Returns local/remote commit info and whether updates are available.
        """
        try:
            local_commit = self._get_local_git_commit()
            local_branch = self._get_local_git_branch()
            has_local_changes = self._check_local_changes()

            # Check for remote updates (this does a git fetch)
            has_updates, _, remote_commit = self._check_for_updates()
            commits_behind = 0
            if has_updates and local_commit and remote_commit:
                commits_behind = self._get_commits_behind(local_commit, remote_commit)

            return web.json_response({
                "local_commit": local_commit[:8] if local_commit else None,
                "local_commit_full": local_commit,
                "local_branch": local_branch,
                "remote_commit": remote_commit[:8] if remote_commit else None,
                "remote_commit_full": remote_commit,
                "has_updates": has_updates,
                "commits_behind": commits_behind,
                "has_local_changes": has_local_changes,
                "auto_update_enabled": AUTO_UPDATE_ENABLED,
                "ringrift_path": self.ringrift_path,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_git_update(self, request: web.Request) -> web.Response:
        """Manually trigger a git update on this node.

        This will stop jobs, pull updates, and restart the orchestrator.
        """
        try:
            # Check for updates first
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if not has_updates:
                return web.json_response({
                    "success": True,
                    "message": "Already up to date",
                    "local_commit": local_commit[:8] if local_commit else None,
                })

            # Perform the update
            success, message = await self._perform_git_update()

            if success:
                # Schedule restart
                asyncio.create_task(self._restart_orchestrator())
                return web.json_response({
                    "success": True,
                    "message": "Update successful, restarting...",
                    "old_commit": local_commit[:8] if local_commit else None,
                    "new_commit": remote_commit[:8] if remote_commit else None,
                })
            else:
                return web.json_response({
                    "success": False,
                    "message": message,
                }, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ============================================
    # Core Logic
    # ============================================

    def _update_self_info(self):
        """Update self info with current resource usage."""
        usage = self._get_resource_usage()
        selfplay, training = self._count_local_jobs()

        self.self_info.cpu_percent = usage["cpu_percent"]
        self.self_info.memory_percent = usage["memory_percent"]
        self.self_info.disk_percent = usage["disk_percent"]
        self.self_info.gpu_percent = usage["gpu_percent"]
        self.self_info.gpu_memory_percent = usage["gpu_memory_percent"]
        self.self_info.selfplay_jobs = selfplay
        self.self_info.training_jobs = training
        self.self_info.role = self.role
        self.self_info.last_heartbeat = time.time()

    async def _send_heartbeat_to_peer(self, peer_host: str, peer_port: int) -> Optional[NodeInfo]:
        """Send heartbeat to a peer and return their info."""
        try:
            self._update_self_info()

            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{peer_host}:{peer_port}/heartbeat"
                async with session.post(url, json=self.self_info.to_dict()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return NodeInfo.from_dict(data)
        except Exception as e:
            pass
        return None

    async def _heartbeat_loop(self):
        """Send heartbeats to all known peers."""
        while self.running:
            try:
                # Send to known peers from config
                for peer_addr in self.known_peers:
                    parts = peer_addr.split(':')
                    host = parts[0]
                    port = int(parts[1]) if len(parts) > 1 else DEFAULT_PORT

                    info = await self._send_heartbeat_to_peer(host, port)
                    if info:
                        with self.peers_lock:
                            info.last_heartbeat = time.time()
                            self.peers[info.node_id] = info

                # Send to discovered peers
                with self.peers_lock:
                    peer_list = list(self.peers.values())

                for peer in peer_list:
                    if peer.node_id != self.node_id:
                        info = await self._send_heartbeat_to_peer(peer.host, peer.port)
                        if info:
                            with self.peers_lock:
                                info.last_heartbeat = time.time()
                                self.peers[info.node_id] = info

                # Check for dead peers
                self._check_dead_peers()

                # Save state periodically
                self._save_state()

            except Exception as e:
                print(f"[P2P] Heartbeat error: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    def _check_dead_peers(self):
        """Check for peers that have stopped responding."""
        with self.peers_lock:
            dead_peers = []
            for node_id, info in self.peers.items():
                if not info.is_alive() and node_id != self.node_id:
                    dead_peers.append(node_id)

            for node_id in dead_peers:
                print(f"[P2P] Peer {node_id} is dead (no heartbeat for {PEER_TIMEOUT}s)")
                # Don't remove, just mark as dead for historical tracking

        # If leader is dead, start election
        if self.leader_id and self.leader_id != self.node_id:
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
                if leader and not leader.is_alive():
                    print(f"[P2P] Leader {self.leader_id} is dead, starting election")
                    asyncio.create_task(self._start_election())

    async def _start_election(self):
        """Start leader election using Bully algorithm."""
        if self.election_in_progress:
            return

        self.election_in_progress = True
        self.role = NodeRole.CANDIDATE
        print(f"[P2P] Starting election, my ID: {self.node_id}")

        try:
            # Send election message to all nodes with higher IDs
            higher_nodes = []
            with self.peers_lock:
                higher_nodes = [
                    p for p in self.peers.values()
                    if p.node_id > self.node_id and p.is_alive()
                ]

            got_response = False

            timeout = ClientTimeout(total=ELECTION_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
                for peer in higher_nodes:
                    try:
                        url = f"http://{peer.host}:{peer.port}/election"
                        async with session.post(url, json={"candidate_id": self.node_id}) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("response") == "ALIVE":
                                    got_response = True
                                    print(f"[P2P] Higher node {peer.node_id} responded")
                    except:
                        pass

            # If no higher node responded, we become leader
            if not got_response:
                await self._become_leader()
            else:
                # Wait for coordinator message
                await asyncio.sleep(ELECTION_TIMEOUT * 2)

        finally:
            self.election_in_progress = False

    async def _become_leader(self):
        """Become the cluster leader."""
        print(f"[P2P] I am now the leader: {self.node_id}")
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id

        # Announce to all peers
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=5)
        async with ClientSession(timeout=timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = f"http://{peer.host}:{peer.port}/coordinator"
                        await session.post(url, json={"leader_id": self.node_id})
                    except:
                        pass

        self._save_state()

    async def _job_management_loop(self):
        """Leader-only: Manage jobs across the cluster."""
        while self.running:
            try:
                if self.role == NodeRole.LEADER:
                    await self._manage_cluster_jobs()
            except Exception as e:
                print(f"[P2P] Job management error: {e}")

            await asyncio.sleep(JOB_CHECK_INTERVAL)

    async def _manage_cluster_jobs(self):
        """Manage jobs across the cluster (leader only).

        LEARNED LESSONS incorporated:
        - Check disk space BEFORE starting jobs (Vast.ai 91-93% disk issue)
        - Check memory to prevent OOM (AWS instance crashed at 31GB+)
        - Trigger cleanup when approaching limits
        - Use is_healthy() not just is_alive()
        """
        print("[P2P] Leader: Managing cluster jobs...")

        # Gather cluster state
        with self.peers_lock:
            alive_peers = [p for p in self.peers.values() if p.is_alive()]

        # Add self
        self._update_self_info()
        all_nodes = alive_peers + [self.self_info]

        # Phase 1: Handle resource warnings and cleanup
        for node in all_nodes:
            # LEARNED LESSONS - Proactive disk cleanup before hitting critical
            if node.disk_percent >= DISK_CLEANUP_THRESHOLD:
                print(f"[P2P] {node.node_id}: Disk at {node.disk_percent:.0f}% - triggering cleanup")
                if node.node_id == self.node_id:
                    await self._cleanup_local_disk()
                else:
                    await self._request_remote_cleanup(node)
                continue  # Skip job creation this cycle

            # LEARNED LESSONS - Memory warning - reduce jobs
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                print(f"[P2P] {node.node_id}: Memory at {node.memory_percent:.0f}% - reducing jobs")
                # Don't start new jobs, let existing ones complete

        # Phase 2: Calculate desired job distribution for healthy nodes
        for node in all_nodes:
            # LEARNED LESSONS - Use is_healthy() to check both connectivity AND resources
            if not node.is_healthy():
                reason = []
                if not node.is_alive():
                    reason.append("unreachable")
                if node.disk_percent >= DISK_CRITICAL_THRESHOLD:
                    reason.append(f"disk={node.disk_percent:.0f}%")
                if node.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
                    reason.append(f"mem={node.memory_percent:.0f}%")
                print(f"[P2P] Skipping {node.node_id}: {', '.join(reason)}")
                continue

            # LEARNED LESSONS - Reduce target when approaching limits
            target_selfplay = 2  # Base minimum
            if node.memory_gb >= 64:
                target_selfplay = 4
            if node.has_gpu and "5090" in node.gpu_name.lower():
                target_selfplay = 8  # More for powerful GPUs

            # LEARNED LESSONS - Reduce target if resources are under pressure
            if node.disk_percent >= DISK_WARNING_THRESHOLD:
                target_selfplay = min(target_selfplay, 2)
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                target_selfplay = min(target_selfplay, 1)

            # Check if node needs more jobs
            if node.selfplay_jobs < target_selfplay:
                needed = target_selfplay - node.selfplay_jobs
                print(f"[P2P] {node.node_id} needs {needed} more selfplay jobs")

                # Start jobs (max 2 at a time to avoid overwhelming)
                for _ in range(min(needed, 2)):
                    # Choose GPU selfplay for GPU nodes, CPU selfplay otherwise
                    job_type = JobType.GPU_SELFPLAY if node.has_gpu else JobType.SELFPLAY

                    if node.node_id == self.node_id:
                        await self._start_local_job(job_type)
                    else:
                        await self._request_remote_job(node, job_type)

    async def _cleanup_local_disk(self):
        """Clean up disk space on local node.

        LEARNED LESSONS - Automatically archive old data:
        - Remove deprecated selfplay databases
        - Compress and archive old logs
        - Clear /tmp files older than 24h
        """
        print("[P2P] Running local disk cleanup...")
        try:
            # Find and remove old .db files in deprecated locations
            deprecated_patterns = [
                f"{self.ringrift_path}/ai-service/data/games/deprecated_*",
                f"{self.ringrift_path}/ai-service/data/selfplay_old/*",
                "/tmp/*.db",  # Temporary test databases
            ]

            for pattern in deprecated_patterns:
                import glob
                files = glob.glob(pattern)
                for f in files:
                    try:
                        path = Path(f)
                        if path.exists():
                            # Only delete files older than 1 day
                            if time.time() - path.stat().st_mtime > 86400:
                                path.unlink()
                                print(f"[P2P] Cleaned: {f}")
                    except Exception as e:
                        print(f"[P2P] Failed to clean {f}: {e}")

            # Clear old log files
            log_dirs = [
                f"{self.ringrift_path}/ai-service/logs",
            ]
            for log_dir in log_dirs:
                for logfile in Path(log_dir).rglob("*.log"):
                    if time.time() - logfile.stat().st_mtime > 7 * 86400:  # 7 days
                        logfile.unlink()
                        print(f"[P2P] Cleaned old log: {logfile}")

        except Exception as e:
            print(f"[P2P] Disk cleanup error: {e}")

    async def _request_remote_cleanup(self, node: NodeInfo):
        """Request a remote node to clean up disk space."""
        try:
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{node.host}:{node.port}/cleanup"
                async with session.post(url, json={}) as resp:
                    if resp.status == 200:
                        print(f"[P2P] Cleanup requested on {node.node_id}")
        except Exception as e:
            print(f"[P2P] Failed to request cleanup from {node.node_id}: {e}")

    async def _start_local_job(
        self,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "descent-only",
    ) -> Optional[ClusterJob]:
        """Start a job on the local node."""
        try:
            job_id = str(uuid.uuid4())[:8]

            if job_type == JobType.SELFPLAY:
                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_self_play_soak.py",
                    "--num-games", "1000",
                    "--board-type", board_type,
                    "--num-players", str(num_players),
                    "--engine-mode", engine_mode,
                    "--log-jsonl", f"{self.ringrift_path}/ai-service/data/selfplay/{board_type}_{num_players}p/games.jsonl",
                ]

                # Create output directory
                output_dir = Path(f"{self.ringrift_path}/ai-service/data/selfplay/{board_type}_{num_players}p")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Start process
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = subprocess.Popen(
                    cmd,
                    stdout=open(output_dir / "run.log", "a"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=self.ringrift_path,
                )

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode=engine_mode,
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started {job_type.value} job {job_id} (PID {proc.pid})")
                self._save_state()
                return job

            elif job_type == JobType.GPU_SELFPLAY:
                # GPU-accelerated parallel selfplay using run_gpu_selfplay.py
                # Only start on nodes with GPU (check done in _manage_cluster_jobs)
                batch_size = 256 if "5090" in self.self_info.gpu_name.lower() else 128

                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_gpu_selfplay.py",
                    "--num-games", "1000",
                    "--board-size", "8" if board_type == "square8" else "19",
                    "--num-players", str(num_players),
                    "--batch-size", str(batch_size),
                    "--output-dir", f"{self.ringrift_path}/ai-service/data/selfplay/gpu_{board_type}_{num_players}p",
                ]

                # Create output directory
                output_dir = Path(f"{self.ringrift_path}/ai-service/data/selfplay/gpu_{board_type}_{num_players}p")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Start process with GPU environment
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                # Ensure CUDA is visible
                if "CUDA_VISIBLE_DEVICES" not in env:
                    env["CUDA_VISIBLE_DEVICES"] = "0"

                proc = subprocess.Popen(
                    cmd,
                    stdout=open(output_dir / "gpu_run.log", "a"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=self.ringrift_path,
                )

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode="gpu",
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started GPU selfplay job {job_id} (PID {proc.pid}, batch={batch_size})")
                self._save_state()
                return job

        except Exception as e:
            print(f"[P2P] Failed to start job: {e}")
        return None

    async def _request_remote_job(self, node: NodeInfo, job_type: JobType):
        """Request a remote node to start a job."""
        try:
            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                url = f"http://{node.host}:{node.port}/start_job"
                async with session.post(url, json={"job_type": job_type.value}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            print(f"[P2P] Started remote job on {node.node_id}")
                        else:
                            print(f"[P2P] Failed to start remote job: {data.get('error')}")
        except Exception as e:
            print(f"[P2P] Failed to request remote job from {node.node_id}: {e}")

    async def _discovery_loop(self):
        """Broadcast UDP discovery messages to find peers on local network."""
        while self.running:
            try:
                # Create UDP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(1.0)

                # Broadcast our presence
                message = json.dumps({
                    "type": "p2p_discovery",
                    "node_id": self.node_id,
                    "host": self.self_info.host,
                    "port": self.port,
                }).encode()

                try:
                    sock.sendto(message, ('<broadcast>', DISCOVERY_PORT))
                except:
                    pass

                # Listen for responses
                try:
                    while True:
                        data, addr = sock.recvfrom(1024)
                        msg = json.loads(data.decode())
                        if msg.get("type") == "p2p_discovery" and msg.get("node_id") != self.node_id:
                            # Found a peer!
                            peer_addr = f"{msg.get('host')}:{msg.get('port')}"
                            if peer_addr not in self.known_peers:
                                self.known_peers.append(peer_addr)
                                print(f"[P2P] Discovered peer: {msg.get('node_id')} at {peer_addr}")
                except socket.timeout:
                    pass

                sock.close()

            except Exception as e:
                pass

            await asyncio.sleep(DISCOVERY_INTERVAL)

    async def run(self):
        """Main entry point - start the orchestrator."""
        if not HAS_AIOHTTP:
            print("Error: aiohttp is required. Install with: pip install aiohttp")
            return

        # Set up HTTP server
        app = web.Application()
        app.router.add_post('/heartbeat', self.handle_heartbeat)
        app.router.add_get('/status', self.handle_status)
        app.router.add_post('/election', self.handle_election)
        app.router.add_post('/coordinator', self.handle_coordinator)
        app.router.add_post('/start_job', self.handle_start_job)
        app.router.add_post('/stop_job', self.handle_stop_job)
        app.router.add_post('/cleanup', self.handle_cleanup)
        app.router.add_get('/health', self.handle_health)
        app.router.add_get('/git/status', self.handle_git_status)
        app.router.add_post('/git/update', self.handle_git_update)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"[P2P] HTTP server started on {self.host}:{self.port}")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._job_management_loop()),
            asyncio.create_task(self._discovery_loop()),
        ]

        # Add git update loop if enabled
        if AUTO_UPDATE_ENABLED:
            tasks.append(asyncio.create_task(self._git_update_loop()))

        # If no leader known, start election after short delay
        await asyncio.sleep(5)
        if not self.leader_id:
            await self._start_election()

        # Run forever
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="P2P Orchestrator for RingRift cluster")
    parser.add_argument("--node-id", required=True, help="Unique identifier for this node")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--peers", help="Comma-separated list of known peers (host:port)")
    parser.add_argument("--ringrift-path", help="Path to RingRift installation")

    args = parser.parse_args()

    known_peers = []
    if args.peers:
        known_peers = [p.strip() for p in args.peers.split(',')]

    orchestrator = P2POrchestrator(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        known_peers=known_peers,
        ringrift_path=args.ringrift_path,
    )

    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n[P2P] Shutting down...")
        orchestrator.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
