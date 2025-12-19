#!/usr/bin/env python
"""
Lightweight HTTP worker for local Mac cluster CMA-ES evaluation.

Run this script on each worker Mac to accept evaluation tasks from the
coordinator. The worker exposes a simple HTTP API:

    GET  /health   - Health check endpoint
    POST /evaluate - Evaluate a single CMA-ES candidate

Usage:
------
    # Start worker on default port 8765
    python scripts/cluster_worker.py

    # Start worker on custom port with explicit ID
    python scripts/cluster_worker.py --port 8766 --worker-id macbook-pro-2

    # Start with Bonjour/mDNS service registration for auto-discovery
    python scripts/cluster_worker.py --register-bonjour

Prerequisites:
--------------
    - Python environment with RingRift dependencies
    - For Bonjour support: pip install zeroconf
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import BoardType, GameState
from app.utils.json_utils import json_default
from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)
from app.training.eval_pools import load_state_pool

# Import evaluation functions from CMA-ES script
from scripts.run_cmaes_optimization import (
    evaluate_fitness,
    BOARD_NAME_TO_TYPE,
)
from app.distributed.game_collector import InMemoryGameCollector

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        get_memory_usage as unified_get_memory_usage,
        check_memory,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_get_memory_usage = None
    check_memory = None
    RESOURCE_LIMITS = None

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("cluster_worker", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory Detection & Requirements
# ---------------------------------------------------------------------------

# Memory requirements per board type (in GB)
# Based on empirical testing:
# - 16GB machine: only 8x8 works reliably
# - 64GB machine: can run 19x19/hex with memory pressure
# - 96GB machine: runs everything comfortably
BOARD_MEMORY_REQUIREMENTS: Dict[str, int] = {
    "square8": 8,  # 8GB minimum for 8x8 games
    "square19": 48,  # 48GB minimum for 19x19 games
    "hexagonal": 48,  # 48GB minimum for hex games
    "hex": 48,  # Alias for hexagonal
}


def get_memory_info() -> Dict[str, Any]:
    """Get memory information for this machine.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement across the codebase.

    Returns dict with:
        - total_gb: Total physical RAM in GB
        - available_gb: Available RAM in GB (free + inactive/reclaimable)
        - eligible_boards: List of board types this worker can handle
    """
    total_gb = 8
    available_gb = 4

    # Try unified resource_guard first
    if HAS_RESOURCE_GUARD and unified_get_memory_usage is not None:
        try:
            _, avail_gb, total_gb_res = unified_get_memory_usage()
            total_gb = int(total_gb_res)
            available_gb = int(avail_gb)
        except Exception:
            pass  # Fall through to platform-specific implementations

    # Fallback to platform-specific implementations if resource_guard failed
    if total_gb == 8:  # Still at default, resource_guard didn't work
        try:
            # macOS: use sysctl for total
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                bytes_total = int(result.stdout.strip())
                total_gb = bytes_total // (1024**3)

            # macOS: use vm_stat for available (free + inactive pages)
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                page_size = 4096  # Default
                free_pages = 0
                inactive_pages = 0

                for line in result.stdout.split("\n"):
                    if "page size of" in line:
                        match = re.search(r"page size of (\d+)", line)
                        if match:
                            page_size = int(match.group(1))
                    elif "Pages free:" in line:
                        free_pages = int(line.split(":")[1].strip().rstrip("."))
                    elif "Pages inactive:" in line:
                        inactive_pages = int(line.split(":")[1].strip().rstrip("."))

                available_bytes = (free_pages + inactive_pages) * page_size
                available_gb = available_bytes // (1024**3)
        except Exception:
            pass

    # Fallback: try Linux /proc/meminfo
    if total_gb == 8:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_gb = kb // (1024 * 1024)
                    elif line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        available_gb = kb // (1024 * 1024)
        except Exception:
            pass

    # Determine eligible boards based on total RAM
    eligible_boards = []
    for board_type, required_gb in BOARD_MEMORY_REQUIREMENTS.items():
        if total_gb >= required_gb:
            eligible_boards.append(board_type)

    # Deduplicate (hex/hexagonal)
    eligible_boards = list(set(eligible_boards))

    return {
        "total_gb": total_gb,
        "available_gb": available_gb,
        "eligible_boards": sorted(eligible_boards),
    }


# ---------------------------------------------------------------------------
# Worker State
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    """Statistics tracked by the worker."""

    tasks_completed: int = 0
    tasks_failed: int = 0
    total_games_played: int = 0
    total_evaluation_time_sec: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_task_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_games_played": self.total_games_played,
            "total_evaluation_time_sec": round(self.total_evaluation_time_sec, 2),
            "uptime_sec": round(uptime, 2),
            "last_task_time": (self.last_task_time.isoformat() if self.last_task_time else None),
        }


# Global worker state
WORKER_ID: str = "unknown"
WORKER_STATS = WorkerStats()

# Cache for loaded state pools to avoid re-loading on every task
STATE_POOL_CACHE: Dict[str, List[GameState]] = {}
STATE_POOL_CACHE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# State Pool Handling
# ---------------------------------------------------------------------------


def get_cached_state_pool(
    board_type: BoardType,
    num_players: int,
    pool_id: str,
) -> List[GameState]:
    """Get a state pool from cache or load it."""
    cache_key = f"{board_type.value}_{num_players}p_{pool_id}"

    with STATE_POOL_CACHE_LOCK:
        if cache_key in STATE_POOL_CACHE:
            return STATE_POOL_CACHE[cache_key]

    # Load outside the lock to avoid blocking other requests
    logger.info(f"Loading state pool: {cache_key}")
    pool = load_state_pool(
        board_type=board_type,
        pool_id=pool_id,
        num_players=num_players,
    )
    if pool is None:
        pool = []

    with STATE_POOL_CACHE_LOCK:
        STATE_POOL_CACHE[cache_key] = pool

    logger.info(f"Loaded state pool {cache_key} with {len(pool)} states")
    return pool


# ---------------------------------------------------------------------------
# Task Evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single CMA-ES candidate.

    Expected task format:
    {
        "task_id": "uuid",
        "candidate_id": 0,
        "weights": {"WEIGHT_STACK_CONTROL": 5.0, ...},
        "baseline_weights": {"WEIGHT_STACK_CONTROL": 5.0, ...},
        "board_type": "square8",
        "num_players": 2,
        "games_per_eval": 24,
        "eval_mode": "multi-start",
        "state_pool_id": "v1",
        "max_moves": 200,
        "eval_randomness": 0.02,
        "seed": 42,
        "record_games": false  # Optional: if true, return game data
    }
    """
    global WORKER_STATS

    task_id = task.get("task_id", "unknown")
    candidate_id = task.get("candidate_id", -1)

    logger.info(f"Starting task {task_id} (candidate {candidate_id})")
    start_time = time.time()

    try:
        # Parse configuration from task
        candidate_weights: HeuristicWeights = task["weights"]
        baseline_weights: HeuristicWeights = task.get("baseline_weights", BASE_V1_BALANCED_WEIGHTS)
        board_type_str = task.get("board_type", "square8")
        board_type = BOARD_NAME_TO_TYPE.get(board_type_str, BoardType.SQUARE8)
        num_players = task.get("num_players", 2)
        games_per_eval = task.get("games_per_eval", 24)
        eval_mode = task.get("eval_mode", "multi-start")
        state_pool_id = task.get("state_pool_id", "v1")
        max_moves = task.get("max_moves", 10000)
        eval_randomness = task.get("eval_randomness", 0.0)
        seed = task.get("seed")
        record_games = task.get("record_games", False)

        # Pre-load state pool if using multi-start mode
        if eval_mode == "multi-start":
            get_cached_state_pool(board_type, num_players, state_pool_id)

        # Create game collector if recording is requested
        game_collector = InMemoryGameCollector() if record_games else None

        # Evaluate fitness using the existing function
        fitness = evaluate_fitness(
            candidate_weights=candidate_weights,
            baseline_weights=baseline_weights,
            games_per_eval=games_per_eval,
            board_type=board_type,
            verbose=False,
            max_moves=max_moves,
            eval_mode=eval_mode,
            state_pool_id=state_pool_id,
            eval_randomness=eval_randomness,
            seed=seed,
            game_db=game_collector,  # Pass collector as game_db
        )

        elapsed = time.time() - start_time

        # Update stats
        WORKER_STATS.tasks_completed += 1
        WORKER_STATS.total_games_played += games_per_eval
        WORKER_STATS.total_evaluation_time_sec += elapsed
        WORKER_STATS.last_task_time = datetime.now()

        logger.info(
            f"Task {task_id} complete: fitness={fitness:.3f}, "
            f"games={games_per_eval}, time={elapsed:.1f}s"
            + (f", recorded={len(game_collector.get_games())}" if game_collector else "")
        )

        result = {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "fitness": fitness,
            "games_played": games_per_eval,
            "evaluation_time_sec": round(elapsed, 2),
            "worker_id": WORKER_ID,
            "status": "success",
        }

        # Include game replays if recording was requested
        if game_collector:
            result["game_replays"] = game_collector.get_serialized_games()

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        WORKER_STATS.tasks_failed += 1
        WORKER_STATS.last_task_time = datetime.now()

        logger.exception(f"Task {task_id} failed: {e}")

        return {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "fitness": 0.0,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "evaluation_time_sec": round(elapsed, 2),
            "worker_id": WORKER_ID,
            "status": "error",
        }


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------


class WorkerRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for worker API."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        """Custom logging format."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        """Send a JSON response."""
        body = json.dumps(data, default=json_default).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/stats":
            self._handle_stats()
        elif self.path == "/pools":
            self._handle_pools()
        else:
            self.send_json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/evaluate":
            self._handle_evaluate()
        elif self.path == "/preload-pool":
            self._handle_preload_pool()
        else:
            self.send_json_response(404, {"error": "Not found"})

    def _handle_health(self) -> None:
        """Health check endpoint with memory info for capacity-aware routing."""
        memory_info = get_memory_info()
        self.send_json_response(
            200,
            {
                "status": "healthy",
                "worker_id": WORKER_ID,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "tasks_completed": WORKER_STATS.tasks_completed,
                # Memory info for capacity-aware job routing
                "memory": {
                    "total_gb": memory_info["total_gb"],
                    "available_gb": memory_info["available_gb"],
                    "eligible_boards": memory_info["eligible_boards"],
                },
            },
        )

    def _handle_stats(self) -> None:
        """Detailed statistics endpoint."""
        stats = WORKER_STATS.to_dict()
        stats["worker_id"] = WORKER_ID
        stats["hostname"] = socket.gethostname()
        stats["cached_pools"] = list(STATE_POOL_CACHE.keys())
        self.send_json_response(200, stats)

    def _handle_pools(self) -> None:
        """List cached state pools."""
        with STATE_POOL_CACHE_LOCK:
            pools = {key: len(pool) for key, pool in STATE_POOL_CACHE.items()}
        self.send_json_response(200, {"pools": pools})

    def _handle_evaluate(self) -> None:
        """Handle candidate evaluation request."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_json_response(400, {"error": "Empty request body"})
                return

            body = self.rfile.read(content_length)
            task = json.loads(body.decode("utf-8"))

            # Validate required fields
            if "weights" not in task:
                self.send_json_response(400, {"error": "Missing required field: weights"})
                return

            result = evaluate_candidate_task(task)

            if result["status"] == "success":
                self.send_json_response(200, result)
            else:
                self.send_json_response(500, result)

        except json.JSONDecodeError as e:
            self.send_json_response(400, {"error": f"Invalid JSON: {e}"})
        except Exception as e:
            logger.exception(f"Request handling error: {e}")
            self.send_json_response(500, {"error": str(e), "traceback": traceback.format_exc()})

    def _handle_preload_pool(self) -> None:
        """Pre-load a state pool into cache."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request = json.loads(body.decode("utf-8"))

            board_type_str = request.get("board_type", "square8")
            board_type = BOARD_NAME_TO_TYPE.get(board_type_str, BoardType.SQUARE8)
            num_players = request.get("num_players", 2)
            pool_id = request.get("pool_id", "v1")

            pool = get_cached_state_pool(board_type, num_players, pool_id)

            self.send_json_response(
                200,
                {
                    "status": "success",
                    "pool_key": f"{board_type.value}_{num_players}p_{pool_id}",
                    "pool_size": len(pool),
                },
            )

        except Exception as e:
            logger.exception(f"Pool preload error: {e}")
            self.send_json_response(500, {"error": str(e)})


# ---------------------------------------------------------------------------
# Bonjour/mDNS Service Registration
# ---------------------------------------------------------------------------


class BonjourRegistration:
    """Register worker as a Bonjour/mDNS service for auto-discovery."""

    SERVICE_TYPE = "_ringrift-worker._tcp.local."

    def __init__(self, port: int, worker_id: str):
        self.port = port
        self.worker_id = worker_id
        self.zeroconf = None
        self.service_info = None

    def register(self) -> bool:
        """Register the worker service. Returns True on success."""
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            logger.warning("zeroconf not installed. Run 'pip install zeroconf' for " "Bonjour service registration.")
            return False

        try:
            hostname = socket.gethostname()
            # Get local IP address
            local_ip = self._get_local_ip()

            self.zeroconf = Zeroconf()
            self.service_info = ServiceInfo(
                self.SERVICE_TYPE,
                f"ringrift-worker-{self.worker_id}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties={
                    "worker_id": self.worker_id,
                    "hostname": hostname,
                },
            )

            self.zeroconf.register_service(self.service_info)
            logger.info(f"Registered Bonjour service: {self.worker_id} at {local_ip}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to register Bonjour service: {e}")
            return False

    def unregister(self) -> None:
        """Unregister the service."""
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                logger.info("Unregistered Bonjour service")
            except Exception as e:
                logger.error(f"Error unregistering Bonjour service: {e}")

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local IP address on the network."""
        try:
            # Connect to an external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    global WORKER_ID

    parser = argparse.ArgumentParser(description="RingRift CMA-ES evaluation worker for local Mac cluster")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (default: hostname)",
    )
    parser.add_argument(
        "--register-bonjour",
        action="store_true",
        help="Register as Bonjour/mDNS service for auto-discovery",
    )
    parser.add_argument(
        "--preload-pools",
        type=str,
        default=None,
        help="Comma-separated list of pools to preload (e.g., 'square8_2p_v1,hex_2p_v1')",
    )
    args = parser.parse_args()

    # Set worker ID
    WORKER_ID = args.worker_id or socket.gethostname()

    logger.info(f"Starting RingRift worker: {WORKER_ID}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")

    # Pre-load state pools if requested
    if args.preload_pools:
        pool_specs = args.preload_pools.split(",")
        for spec in pool_specs:
            try:
                # Parse spec like "square8_2p_v1"
                parts = spec.strip().split("_")
                if len(parts) >= 3:
                    board_type_str = parts[0]
                    num_players = int(parts[1].replace("p", ""))
                    pool_id = "_".join(parts[2:])
                    board_type = BOARD_NAME_TO_TYPE.get(board_type_str, BoardType.SQUARE8)
                    logger.info(f"Preloading pool: {spec}")
                    get_cached_state_pool(board_type, num_players, pool_id)
            except Exception as e:
                logger.warning(f"Failed to preload pool {spec}: {e}")

    # Register Bonjour service if requested
    bonjour = None
    if args.register_bonjour:
        bonjour = BonjourRegistration(args.port, WORKER_ID)
        bonjour.register()

    # Start HTTP server
    server_address = ("0.0.0.0", args.port)
    httpd = HTTPServer(server_address, WorkerRequestHandler)

    local_ip = BonjourRegistration._get_local_ip()
    logger.info(f"Worker listening on http://{local_ip}:{args.port}")
    logger.info(f"Health check: curl http://{local_ip}:{args.port}/health")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
    finally:
        if bonjour:
            bonjour.unregister()
        httpd.server_close()


if __name__ == "__main__":
    main()
