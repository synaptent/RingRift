#!/usr/bin/env python
"""
Queue-based CMA-ES worker for cloud deployment.

This worker is designed to run in containerized environments (Docker, ECS, etc.)
and receives tasks from a queue (Redis or SQS). It's the cloud-native counterpart
to cluster_worker.py which uses HTTP.

Features:
- Pulls tasks from Redis or SQS queue
- Downloads state pools from cloud storage on startup
- Graceful shutdown handling for spot instance termination
- Health check endpoint for load balancer
- Automatic task retry on failure

Usage:
    # With Redis (local development / small-scale)
    QUEUE_BACKEND=redis REDIS_URL=redis://localhost:6379 python scripts/cmaes_cloud_worker.py

    # With SQS (production AWS deployment)
    QUEUE_BACKEND=sqs \
    SQS_TASK_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/tasks \
    SQS_RESULT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/results \
    python scripts/cmaes_cloud_worker.py

Environment variables:
    QUEUE_BACKEND: redis, sqs (default: redis)
    REDIS_URL: Redis connection URL
    SQS_TASK_QUEUE_URL: SQS queue URL for tasks
    SQS_RESULT_QUEUE_URL: SQS queue URL for results
    WORKER_ID: Unique worker identifier (default: hostname)
    STORAGE_BACKEND: local, s3, gcs (default: local)
    STORAGE_BUCKET: S3/GCS bucket for state pools
    PRELOAD_POOLS: Comma-separated pool specs to preload (e.g., "square8_2p_v1,hex_3p_v1")
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.distributed.queue import (
    TaskQueue,
    EvalTask,
    EvalResult,
    get_task_queue,
)
from app.models import BoardType, GameState
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
from app.training.eval_pools import load_state_pool
from scripts.run_cmaes_optimization import (
    evaluate_fitness,
    BOARD_NAME_TO_TYPE,
)
from app.distributed.game_collector import InMemoryGameCollector

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("cmaes_cloud_worker")


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
    current_task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_games_played": self.total_games_played,
            "total_evaluation_time_sec": round(self.total_evaluation_time_sec, 2),
            "uptime_sec": round(uptime, 2),
            "last_task_time": (self.last_task_time.isoformat() if self.last_task_time else None),
            "current_task_id": self.current_task_id,
        }


# Global state
WORKER_ID: str = os.getenv("WORKER_ID", socket.gethostname())
WORKER_STATS = WorkerStats()
SHUTDOWN_REQUESTED = threading.Event()

# State pool cache
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

    # Load outside the lock
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


def preload_pools(pool_specs: List[str]) -> None:
    """Pre-load specified state pools into cache."""
    for spec in pool_specs:
        try:
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


# ---------------------------------------------------------------------------
# Task Processing
# ---------------------------------------------------------------------------


def process_task(task: EvalTask) -> EvalResult:
    """Process a single evaluation task."""
    global WORKER_STATS

    WORKER_STATS.current_task_id = task.task_id
    start_time = time.time()

    try:
        # Parse configuration
        board_type = BOARD_NAME_TO_TYPE.get(task.board_type, BoardType.SQUARE8)
        baseline_weights = task.baseline_weights or BASE_V1_BALANCED_WEIGHTS

        # Pre-load state pool if using multi-start mode
        if task.eval_mode == "multi-start":
            get_cached_state_pool(board_type, task.num_players, task.state_pool_id)

        # Create game collector if recording is requested
        game_collector = InMemoryGameCollector() if task.record_games else None

        # Evaluate fitness
        fitness = evaluate_fitness(
            candidate_weights=task.weights,
            baseline_weights=baseline_weights,
            games_per_eval=task.games_per_eval,
            board_type=board_type,
            verbose=False,
            max_moves=task.max_moves,
            eval_mode=task.eval_mode,
            state_pool_id=task.state_pool_id,
            eval_randomness=task.eval_randomness,
            seed=task.seed,
            game_db=game_collector,  # Pass collector as game_db
        )

        elapsed = time.time() - start_time

        # Update stats
        WORKER_STATS.tasks_completed += 1
        WORKER_STATS.total_games_played += task.games_per_eval
        WORKER_STATS.total_evaluation_time_sec += elapsed
        WORKER_STATS.last_task_time = datetime.now()
        WORKER_STATS.current_task_id = None

        logger.info(
            f"Task {task.task_id} complete: fitness={fitness:.3f}, "
            f"games={task.games_per_eval}, time={elapsed:.1f}s"
            + (f", recorded={len(game_collector.get_games())}" if game_collector else "")
        )

        # Get game replays if recording was requested
        game_replays = game_collector.get_serialized_games() if game_collector else None

        return EvalResult(
            task_id=task.task_id,
            candidate_id=task.candidate_id,
            fitness=fitness,
            games_played=task.games_per_eval,
            evaluation_time_sec=round(elapsed, 2),
            worker_id=WORKER_ID,
            status="success",
            game_replays=game_replays,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        WORKER_STATS.tasks_failed += 1
        WORKER_STATS.last_task_time = datetime.now()
        WORKER_STATS.current_task_id = None

        logger.exception(f"Task {task.task_id} failed: {e}")

        return EvalResult(
            task_id=task.task_id,
            candidate_id=task.candidate_id,
            fitness=0.0,
            evaluation_time_sec=round(elapsed, 2),
            worker_id=WORKER_ID,
            status="error",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Worker Loop
# ---------------------------------------------------------------------------


def run_worker_loop(queue: TaskQueue, poll_timeout: float = 30.0) -> None:
    """Main worker loop that processes tasks from the queue."""
    logger.info(f"Worker {WORKER_ID} starting task loop")

    while not SHUTDOWN_REQUESTED.is_set():
        try:
            # Try to get a task
            task = queue.consume_task(timeout=poll_timeout)

            if task is None:
                # No task available, loop and try again
                continue

            # Process the task
            result = process_task(task)

            # Publish the result (this also acks the task)
            queue.publish_result(result)

        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            traceback.print_exc()
            # Brief pause before retrying
            time.sleep(1)

    logger.info(f"Worker {WORKER_ID} shutting down gracefully")


# ---------------------------------------------------------------------------
# Health Check Server
# ---------------------------------------------------------------------------


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""

    def log_message(self, format: str, *args) -> None:
        pass  # Suppress logging

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond_health()
        elif self.path == "/stats":
            self._respond_stats()
        else:
            self.send_error(404)

    def _respond_health(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        import json

        response = {
            "status": "healthy",
            "worker_id": WORKER_ID,
            "tasks_completed": WORKER_STATS.tasks_completed,
        }
        self.wfile.write(json.dumps(response).encode())

    def _respond_stats(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        import json

        stats = WORKER_STATS.to_dict()
        stats["worker_id"] = WORKER_ID
        stats["cached_pools"] = list(STATE_POOL_CACHE.keys())
        self.wfile.write(json.dumps(stats).encode())


def start_health_server(port: int) -> HTTPServer:
    """Start a background HTTP server for health checks."""
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server started on port {port}")
    return server


# ---------------------------------------------------------------------------
# Signal Handlers
# ---------------------------------------------------------------------------


def handle_shutdown(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    SHUTDOWN_REQUESTED.set()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    global WORKER_ID

    parser = argparse.ArgumentParser(description="Queue-based CMA-ES worker for cloud deployment")
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (default: hostname or WORKER_ID env var)",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8080,
        help="Port for health check HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--preload-pools",
        type=str,
        default=None,
        help="Comma-separated pool specs to preload (e.g., 'square8_2p_v1,hex_3p_v1')",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds when polling for tasks (default: 30)",
    )
    args = parser.parse_args()

    # Set worker ID
    if args.worker_id:
        WORKER_ID = args.worker_id
    elif os.getenv("WORKER_ID"):
        WORKER_ID = os.getenv("WORKER_ID")
    else:
        WORKER_ID = socket.gethostname()

    logger.info(f"Starting cloud worker: {WORKER_ID}")

    # Setup signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Start health check server
    health_server = start_health_server(args.health_port)

    # Pre-load state pools if specified
    preload_specs = args.preload_pools or os.getenv("PRELOAD_POOLS", "")
    if preload_specs:
        preload_pools(preload_specs.split(","))

    # Connect to queue
    try:
        queue = get_task_queue()
        logger.info(f"Connected to queue backend: {type(queue).__name__}")
    except Exception as e:
        logger.error(f"Failed to connect to queue: {e}")
        sys.exit(1)

    try:
        # Run the worker loop
        run_worker_loop(queue, poll_timeout=args.poll_timeout)
    finally:
        # Cleanup
        queue.close()
        health_server.shutdown()
        logger.info(f"Worker {WORKER_ID} shutdown complete")


if __name__ == "__main__":
    main()
