#!/usr/bin/env python
"""HTTP CMA-ES worker for distributed evaluation.

Endpoints:
  - GET /health
  - GET /stats
  - POST /preload-pool
  - POST /evaluate
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Add app/ and scripts/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, ConfigDict
    import uvicorn
except ImportError:  # pragma: no cover - runtime dependency guard
    print(
        "FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn pydantic",
        file=sys.stderr,
    )
    sys.exit(1)

from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS
from app.distributed.game_collector import InMemoryGameCollector
from app.models import BoardType, GameState
from app.training.eval_pools import load_state_pool as load_state_pool_raw
from app.utils.resource_guard import get_memory_usage
from scripts import run_cmaes_optimization as cmaes
from scripts.lib.cli import parse_board_type
from scripts.lib.logging_config import setup_script_logging


SERVICE_TYPE = "_ringrift-worker._tcp.local."

# Must stay in sync with app.distributed.client.BOARD_MEMORY_REQUIREMENTS
BOARD_MEMORY_REQUIREMENTS: dict[str, int] = {
    "square8": 8,
    "square19": 48,
    "hexagonal": 48,
    "hex": 48,
    "full_hex": 48,
    "hex24": 48,
}


@dataclass
class WorkerStats:
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_games_played: int = 0
    total_evaluation_time_sec: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_task_time: datetime | None = None
    current_task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_games_played": self.total_games_played,
            "total_evaluation_time_sec": round(self.total_evaluation_time_sec, 2),
            "uptime_sec": round(uptime, 2),
            "last_task_time": self.last_task_time.isoformat() if self.last_task_time else None,
            "current_task_id": self.current_task_id,
        }


WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())
WORKER_STATS = WorkerStats()
STATS_LOCK = threading.Lock()
EVALUATION_LOCK = threading.Lock()

STATE_POOL_CACHE: dict[str, list[GameState]] = {}
STATE_POOL_INDEX: dict[tuple[str, str], list[str]] = {}
STATE_POOL_LOCK = threading.RLock()


def _cache_key(board_type: BoardType, pool_id: str, num_players: int | None) -> str:
    players_tag = f"{num_players}p" if num_players is not None else "any"
    return f"{board_type.value}_{players_tag}_{pool_id}"


def get_cached_state_pool(
    board_type: BoardType,
    pool_id: str,
    num_players: int | None,
) -> list[GameState]:
    key = _cache_key(board_type, pool_id, num_players)
    with STATE_POOL_LOCK:
        if key in STATE_POOL_CACHE:
            return STATE_POOL_CACHE[key]

    pool = load_state_pool_raw(
        board_type=board_type,
        pool_id=pool_id,
        max_states=None,
        num_players=num_players,
    )

    with STATE_POOL_LOCK:
        STATE_POOL_CACHE[key] = pool
        STATE_POOL_INDEX.setdefault((board_type.value, pool_id), []).append(key)

    return pool


def load_state_pool_cached(
    board_type: BoardType,
    pool_id: str = "v1",
    max_states: int | None = None,
    num_players: int | None = None,
) -> list[GameState]:
    if num_players is None:
        with STATE_POOL_LOCK:
            keys = STATE_POOL_INDEX.get((board_type.value, pool_id), [])
            if keys:
                pool = STATE_POOL_CACHE.get(keys[0], [])
                return pool[:max_states] if max_states is not None else pool

    pool = get_cached_state_pool(board_type, pool_id, num_players)
    return pool[:max_states] if max_states is not None else pool


# Patch the CMA-ES module to use cached state pools.
cmaes.load_state_pool = load_state_pool_cached  # type: ignore[assignment]

evaluate_fitness = cmaes.evaluate_fitness
evaluate_fitness_multiplayer = cmaes.evaluate_fitness_multiplayer


def _get_memory_info() -> dict[str, Any]:
    used_percent, available_gb, total_gb = get_memory_usage()
    eligible = [
        board for board, required in BOARD_MEMORY_REQUIREMENTS.items()
        if total_gb >= required
    ]
    return {
        "total_gb": round(total_gb, 2),
        "available_gb": round(available_gb, 2),
        "used_percent": round(used_percent, 1),
        "eligible_boards": sorted(set(eligible)),
    }


def _get_advertise_ip(host: str) -> str:
    if host and host not in ("0.0.0.0", "::") and not host.startswith("127."):
        return host
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _register_bonjour(worker_id: str, host: str, port: int) -> tuple[Any, Any]:
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        raise RuntimeError("zeroconf not installed; run: pip install zeroconf")

    hostname = socket.gethostname()
    address = socket.inet_aton(_get_advertise_ip(host))
    service_name = f"{worker_id}.{SERVICE_TYPE}"
    properties = {
        "worker_id": worker_id,
        "hostname": hostname,
    }

    info = ServiceInfo(
        type_=SERVICE_TYPE,
        name=service_name,
        addresses=[address],
        port=port,
        properties=properties,
        server=f"{hostname}.local.",
    )
    zeroconf = Zeroconf()
    zeroconf.register_service(info)
    return zeroconf, info


class PreloadPoolRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    board_type: str
    num_players: int = 2
    pool_id: str = "v1"


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    candidate_id: int
    weights: dict[str, float]
    baseline_weights: dict[str, float] | None = None
    board_type: str = "square8"
    num_players: int = 2
    games_per_eval: int = 24
    eval_mode: str = "multi-start"
    state_pool_id: str = "v1"
    max_moves: int = 10000
    eval_randomness: float = 0.0
    seed: int | None = None
    record_games: bool = False


def build_app() -> FastAPI:
    app = FastAPI(title="RingRift CMA-ES Worker")

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        return {
            "status": "healthy",
            "worker_id": WORKER_ID,
            "tasks_completed": WORKER_STATS.tasks_completed,
            "memory": _get_memory_info(),
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        data = WORKER_STATS.to_dict()
        data.update(
            {
                "status": "healthy",
                "worker_id": WORKER_ID,
                "memory": _get_memory_info(),
                "cached_pools": sorted(STATE_POOL_CACHE.keys()),
            }
        )
        return data

    @app.post("/preload-pool")
    def preload_pool(request: PreloadPoolRequest) -> dict[str, Any]:
        try:
            board_type = parse_board_type(request.board_type)
            pool = get_cached_state_pool(board_type, request.pool_id, request.num_players)
            return {
                "status": "success",
                "pool_id": request.pool_id,
                "pool_size": len(pool),
                "cache_key": _cache_key(board_type, request.pool_id, request.num_players),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
            }

    @app.post("/evaluate")
    def evaluate(request: EvaluateRequest) -> dict[str, Any]:
        start_time = time.time()
        game_collector = InMemoryGameCollector() if request.record_games else None

        with EVALUATION_LOCK:
            WORKER_STATS.current_task_id = request.task_id

            try:
                board_type = parse_board_type(request.board_type)
                baseline_weights = request.baseline_weights or BASE_V1_BALANCED_WEIGHTS

                if request.eval_mode == "multi-start":
                    get_cached_state_pool(board_type, request.state_pool_id, request.num_players)

                recording_context = {
                    "worker_id": WORKER_ID,
                    "task_id": request.task_id,
                    "candidate_id": request.candidate_id,
                }

                if request.num_players > 2:
                    raw_score = evaluate_fitness_multiplayer(
                        candidate_weights=request.weights,
                        baseline_weights=baseline_weights,
                        num_players=request.num_players,
                        games_per_eval=request.games_per_eval,
                        boards=[board_type],
                        state_pool_id=request.state_pool_id,
                        seed=request.seed,
                        game_db=game_collector,
                        recording_context=recording_context,
                    )
                    fitness = (raw_score + 1.0) / 2.0
                else:
                    fitness = evaluate_fitness(
                        candidate_weights=request.weights,
                        baseline_weights=baseline_weights,
                        games_per_eval=request.games_per_eval,
                        board_type=board_type,
                        verbose=False,
                        max_moves=request.max_moves,
                        eval_mode=request.eval_mode,
                        state_pool_id=request.state_pool_id,
                        eval_randomness=request.eval_randomness,
                        seed=request.seed,
                        game_db=game_collector,
                        recording_context=recording_context,
                    )

                elapsed = time.time() - start_time
                game_replays = (
                    game_collector.get_serialized_games(include_history_entries=False)
                    if game_collector
                    else None
                )

                with STATS_LOCK:
                    WORKER_STATS.tasks_completed += 1
                    WORKER_STATS.total_games_played += request.games_per_eval
                    WORKER_STATS.total_evaluation_time_sec += elapsed
                    WORKER_STATS.last_task_time = datetime.utcnow()
                    WORKER_STATS.current_task_id = None

                return {
                    "task_id": request.task_id,
                    "candidate_id": request.candidate_id,
                    "fitness": fitness,
                    "games_played": request.games_per_eval,
                    "evaluation_time_sec": round(elapsed, 2),
                    "worker_id": WORKER_ID,
                    "status": "success",
                    "game_replays": game_replays,
                }

            except Exception as exc:
                elapsed = time.time() - start_time
                with STATS_LOCK:
                    WORKER_STATS.tasks_failed += 1
                    WORKER_STATS.last_task_time = datetime.utcnow()
                    WORKER_STATS.current_task_id = None

                return {
                    "task_id": request.task_id,
                    "candidate_id": request.candidate_id,
                    "fitness": 0.0,
                    "games_played": 0,
                    "evaluation_time_sec": round(elapsed, 2),
                    "worker_id": WORKER_ID,
                    "status": "error",
                    "error": str(exc),
                }

    return app


def main() -> None:
    global WORKER_ID

    parser = argparse.ArgumentParser(description="RingRift CMA-ES HTTP worker")
    parser.add_argument("--host", default=os.getenv("WORKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WORKER_PORT", "8765")))
    parser.add_argument("--worker-id", default=os.getenv("WORKER_ID", socket.gethostname()))
    parser.add_argument("--register-bonjour", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("WORKER_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    WORKER_ID = args.worker_id
    logger = setup_script_logging("cluster_worker", level=args.log_level)

    logger.info("Starting CMA-ES worker", extra={"worker_id": WORKER_ID, "host": args.host, "port": args.port})

    zeroconf = None
    service_info = None
    if args.register_bonjour:
        try:
            zeroconf, service_info = _register_bonjour(WORKER_ID, args.host, args.port)
            logger.info("Registered Bonjour service", extra={"service": SERVICE_TYPE, "worker_id": WORKER_ID})
        except Exception as exc:
            logger.error(f"Failed to register Bonjour service: {exc}")
            sys.exit(1)

    try:
        uvicorn.run(
            build_app(),
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
        )
    finally:
        if zeroconf and service_info:
            try:
                zeroconf.unregister_service(service_info)
            except Exception as exc:
                logger.warning(f"Failed to unregister Bonjour service: {exc}")
            zeroconf.close()


if __name__ == "__main__":
    main()
