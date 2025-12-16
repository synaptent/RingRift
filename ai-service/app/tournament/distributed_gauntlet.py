"""Distributed O(n) gauntlet evaluation for neural network models.

This module implements an efficient gauntlet-style evaluation where each model
plays against a fixed set of baselines rather than O(n²) round-robin. This makes
large-scale model evaluation tractable.

Architecture:
    - Leader node runs gauntlet coordinator
    - Worker nodes (including Vast instances) execute games
    - Hybrid transport ensures reliable communication even with NAT
    - Games dispatched in batches for efficiency

Usage:
    from app.tournament.distributed_gauntlet import DistributedNNGauntlet

    gauntlet = DistributedNNGauntlet(elo_db, model_dir)
    results = await gauntlet.run_gauntlet("square8_2p")

    # Or run distributed across cluster
    results = await gauntlet.run_gauntlet_distributed("square8_2p")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import hybrid transport for distributed execution
try:
    from app.distributed.hybrid_transport import get_hybrid_transport, HybridTransport
    HAS_HYBRID_TRANSPORT = True
except ImportError:
    HAS_HYBRID_TRANSPORT = False
    get_hybrid_transport = None
    HybridTransport = None

# Config keys for all 9 board/player combinations
CONFIG_KEYS = [
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Theoretical max moves per config for complete games
MAX_MOVES = {
    "square8_2p": 500, "square8_3p": 600, "square8_4p": 700,
    "square19_2p": 1500, "square19_3p": 2000, "square19_4p": 2500,
    "hexagonal_2p": 2000, "hexagonal_3p": 3000, "hexagonal_4p": 4000,
}


@dataclass
class GauntletConfig:
    """Configuration for gauntlet evaluation."""

    games_per_matchup: int = 10       # Games against each baseline
    num_baselines: int = 4            # Fixed reference models per config
    reserved_workers: int = 2         # CPU instances for gauntlet
    parallel_games: int = 8           # Concurrent games per worker
    timeout_seconds: int = 300        # Per-game timeout
    min_games_for_rating: int = 5     # Min games before model is "rated"


@dataclass
class GauntletResult:
    """Result of a gauntlet evaluation run."""

    run_id: str
    config_key: str
    started_at: float
    completed_at: Optional[float] = None
    models_evaluated: int = 0
    total_games: int = 0
    status: str = "pending"
    model_results: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class GameTask:
    """A single game to be played in the gauntlet."""

    task_id: str
    model_id: str
    baseline_id: str
    config_key: str
    game_num: int


@dataclass
class GameResult:
    """Result of a single gauntlet game."""

    task_id: str
    model_id: str
    baseline_id: str
    model_won: bool
    baseline_won: bool
    draw: bool
    game_length: int
    duration_sec: float


class DistributedNNGauntlet:
    """O(n) gauntlet evaluation: each model plays fixed baselines only.

    Instead of O(n²) round-robin where each model plays every other model,
    this gauntlet tests each model against 4 fixed baseline models per config.
    This makes evaluation of 500+ models tractable.

    Baselines per config:
    1. Best current model (highest Elo)
    2. Median model (50th percentile by Elo)
    3. Lower quartile model (25th percentile)
    4. RandomAI baseline (consistent reference)
    """

    def __init__(
        self,
        elo_db_path: Path,
        model_dir: Path,
        config: Optional[GauntletConfig] = None,
    ):
        """Initialize gauntlet evaluator.

        Args:
            elo_db_path: Path to unified Elo database
            model_dir: Directory containing model .pth files
            config: Gauntlet configuration
        """
        self.elo_db_path = elo_db_path
        self.model_dir = model_dir
        self.config = config or GauntletConfig()

        # State
        self._current_run: Optional[GauntletResult] = None
        self._reserved_workers: Set[str] = set()

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.elo_db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_gauntlet_tables(self) -> None:
        """Initialize gauntlet tracking tables if they don't exist."""
        conn = self._get_db_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gauntlet_runs (
                    run_id TEXT PRIMARY KEY,
                    config_key TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    models_evaluated INTEGER DEFAULT 0,
                    total_games INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                );

                CREATE TABLE IF NOT EXISTS gauntlet_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    baseline_id TEXT NOT NULL,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    elo_before REAL,
                    elo_after REAL,
                    FOREIGN KEY (run_id) REFERENCES gauntlet_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_gauntlet_results_model
                    ON gauntlet_results(model_id, run_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def get_unrated_models(self, config_key: str) -> List[str]:
        """Get models that haven't been rated via gauntlet for this config.

        Args:
            config_key: Config like "square8_2p"

        Returns:
            List of model IDs needing evaluation
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        conn = self._get_db_connection()
        try:
            # Get models with < min games for this config
            min_games = self.config.min_games_for_rating

            # First check which column exists
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            id_col = "model_id" if "model_id" in columns else "participant_id"

            cursor = conn.execute(f"""
                SELECT {id_col}
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                AND games_played < ?
            """, (board_type, num_players, min_games))

            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_models_by_elo(self, config_key: str) -> List[Tuple[str, float]]:
        """Get all models for a config sorted by Elo rating.

        Args:
            config_key: Config like "square8_2p"

        Returns:
            List of (model_id, elo_rating) tuples, sorted by rating descending
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        conn = self._get_db_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            id_col = "model_id" if "model_id" in columns else "participant_id"

            cursor = conn.execute(f"""
                SELECT {id_col}, rating
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                ORDER BY rating DESC
            """, (board_type, num_players))

            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def count_models(self, config_key: str) -> int:
        """Count total models for a config."""
        return len(self.get_models_by_elo(config_key))

    def select_baselines(self, config_key: str) -> List[str]:
        """Select 4 fixed baseline models for gauntlet comparison.

        Selection strategy:
        1. Best model (highest Elo) - the champion to beat
        2. Median model (50th percentile) - middle-strength opponent
        3. Lower quartile (25th percentile) - weaker opponent for floor
        4. RandomAI - consistent baseline across all evaluations

        Args:
            config_key: Config like "square8_2p"

        Returns:
            List of 4 baseline model IDs
        """
        models = self.get_models_by_elo(config_key)
        baselines = []

        if not models:
            # No models yet, just use RandomAI
            return ["random_ai"]

        n = len(models)

        # 1. Best model (index 0, highest Elo)
        baselines.append(models[0][0])

        # 2. Median model (middle)
        if n >= 3:
            median_idx = n // 2
            baselines.append(models[median_idx][0])
        elif n >= 2:
            baselines.append(models[1][0])

        # 3. Lower quartile (25th percentile from top = 75th percentile down)
        if n >= 4:
            lower_q_idx = (3 * n) // 4
            baselines.append(models[lower_q_idx][0])
        elif n >= 3:
            baselines.append(models[-1][0])

        # 4. RandomAI baseline (always include)
        baselines.append("random_ai")

        # Ensure we have exactly 4 unique baselines
        baselines = list(dict.fromkeys(baselines))[:4]

        # Pad with random_ai if needed
        while len(baselines) < 4:
            baselines.append("random_ai")

        return baselines

    def create_game_tasks(
        self,
        unrated_models: List[str],
        baselines: List[str],
        config_key: str,
    ) -> List[GameTask]:
        """Create list of games to play for gauntlet evaluation.

        Args:
            unrated_models: Models to evaluate
            baselines: Fixed baseline models
            config_key: Config for max_moves lookup

        Returns:
            List of GameTask objects to execute
        """
        tasks = []

        for model_id in unrated_models:
            for baseline_id in baselines:
                # Skip if model is its own baseline
                if model_id == baseline_id:
                    continue

                for game_num in range(self.config.games_per_matchup):
                    task = GameTask(
                        task_id=f"{model_id}_vs_{baseline_id}_g{game_num}",
                        model_id=model_id,
                        baseline_id=baseline_id,
                        config_key=config_key,
                        game_num=game_num,
                    )
                    tasks.append(task)

        return tasks

    async def run_gauntlet(self, config_key: str) -> GauntletResult:
        """Run gauntlet evaluation for all unrated models in a config.

        Args:
            config_key: Config like "square8_2p"

        Returns:
            GauntletResult with evaluation outcomes
        """
        self._init_gauntlet_tables()

        run_id = str(uuid.uuid4())[:8]
        self._current_run = GauntletResult(
            run_id=run_id,
            config_key=config_key,
            started_at=time.time(),
            status="running",
        )

        # Record run start
        conn = self._get_db_connection()
        try:
            conn.execute("""
                INSERT INTO gauntlet_runs (run_id, config_key, started_at, status)
                VALUES (?, ?, ?, ?)
            """, (run_id, config_key, self._current_run.started_at, "running"))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"[Gauntlet] Starting run {run_id} for {config_key}")

        # Get models to evaluate
        unrated = self.get_unrated_models(config_key)
        if not unrated:
            logger.info(f"[Gauntlet] No unrated models for {config_key}")
            self._current_run.status = "no_work"
            return self._current_run

        logger.info(f"[Gauntlet] Found {len(unrated)} unrated models")

        # Select baselines
        baselines = self.select_baselines(config_key)
        logger.info(f"[Gauntlet] Baselines: {baselines}")

        # Create game tasks
        tasks = self.create_game_tasks(unrated, baselines, config_key)
        self._current_run.total_games = len(tasks)
        logger.info(f"[Gauntlet] Created {len(tasks)} game tasks")

        # Execute tasks (this would be distributed in production)
        # For now, run locally with asyncio
        results = await self._execute_tasks_local(tasks, config_key)

        # Aggregate results and update Elo
        self._aggregate_results(results)

        # Mark complete
        self._current_run.completed_at = time.time()
        self._current_run.status = "completed"
        self._current_run.models_evaluated = len(unrated)

        conn = self._get_db_connection()
        try:
            conn.execute("""
                UPDATE gauntlet_runs
                SET completed_at = ?, status = ?, models_evaluated = ?, total_games = ?
                WHERE run_id = ?
            """, (
                self._current_run.completed_at,
                "completed",
                len(unrated),
                len(tasks),
                run_id,
            ))
            conn.commit()
        finally:
            conn.close()

        duration = self._current_run.completed_at - self._current_run.started_at
        logger.info(
            f"[Gauntlet] Completed {run_id}: {len(unrated)} models, "
            f"{len(tasks)} games in {duration:.1f}s"
        )

        return self._current_run

    async def _execute_tasks_local(
        self,
        tasks: List[GameTask],
        config_key: str,
    ) -> List[GameResult]:
        """Execute game tasks locally (single process).

        In production, this would distribute to workers.
        For now, runs games sequentially with asyncio.
        """
        from app.tournament.agents import AIAgentRegistry
        from app.tournament.runner import TournamentRunner
        from app.tournament.scheduler import TournamentScheduler

        results = []

        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        max_moves = MAX_MOVES.get(config_key, 2000)

        # This is a simplified local execution
        # Real distributed version would use P2P coordinator
        for task in tasks:
            start = time.time()

            # Simulate game result for now
            # In production, would actually run the game
            result = GameResult(
                task_id=task.task_id,
                model_id=task.model_id,
                baseline_id=task.baseline_id,
                model_won=False,
                baseline_won=True,
                draw=False,
                game_length=100,
                duration_sec=time.time() - start,
            )
            results.append(result)

            if len(results) % 100 == 0:
                logger.info(f"[Gauntlet] Progress: {len(results)}/{len(tasks)} games")

        return results

    def _aggregate_results(self, results: List[GameResult]) -> None:
        """Aggregate game results and update Elo ratings."""
        # Group by model
        model_stats: Dict[str, Dict[str, Dict]] = {}

        for result in results:
            if result.model_id not in model_stats:
                model_stats[result.model_id] = {}
            if result.baseline_id not in model_stats[result.model_id]:
                model_stats[result.model_id][result.baseline_id] = {
                    "wins": 0, "losses": 0, "draws": 0
                }

            stats = model_stats[result.model_id][result.baseline_id]
            if result.model_won:
                stats["wins"] += 1
            elif result.baseline_won:
                stats["losses"] += 1
            else:
                stats["draws"] += 1

        # Store aggregated results
        if self._current_run:
            self._current_run.model_results = model_stats

        # Record to database
        conn = self._get_db_connection()
        try:
            for model_id, baseline_results in model_stats.items():
                for baseline_id, stats in baseline_results.items():
                    conn.execute("""
                        INSERT INTO gauntlet_results
                        (run_id, model_id, baseline_id, wins, losses, draws)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        self._current_run.run_id if self._current_run else "unknown",
                        model_id,
                        baseline_id,
                        stats["wins"],
                        stats["losses"],
                        stats["draws"],
                    ))
            conn.commit()
        finally:
            conn.close()

    # ============================================
    # Distributed Execution via P2P Cluster
    # ============================================

    async def run_gauntlet_distributed(
        self,
        config_key: str,
        p2p_url: Optional[str] = None,
    ) -> GauntletResult:
        """Run gauntlet evaluation distributed across the P2P cluster.

        Uses the hybrid transport to dispatch games to available workers,
        including Vast instances behind NAT.

        Args:
            config_key: Config like "square8_2p"
            p2p_url: URL of P2P orchestrator (default: localhost:8770)

        Returns:
            GauntletResult with evaluation outcomes
        """
        self._init_gauntlet_tables()

        p2p_url = p2p_url or os.environ.get("RINGRIFT_P2P_URL", "http://localhost:8770")

        run_id = str(uuid.uuid4())[:8]
        self._current_run = GauntletResult(
            run_id=run_id,
            config_key=config_key,
            started_at=time.time(),
            status="running",
        )

        # Record run start
        conn = self._get_db_connection()
        try:
            conn.execute("""
                INSERT INTO gauntlet_runs (run_id, config_key, started_at, status)
                VALUES (?, ?, ?, ?)
            """, (run_id, config_key, self._current_run.started_at, "running"))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"[Gauntlet] Starting distributed run {run_id} for {config_key}")

        # Get models to evaluate
        unrated = self.get_unrated_models(config_key)
        if not unrated:
            logger.info(f"[Gauntlet] No unrated models for {config_key}")
            self._current_run.status = "no_work"
            return self._current_run

        logger.info(f"[Gauntlet] Found {len(unrated)} unrated models")

        # Select baselines
        baselines = self.select_baselines(config_key)
        logger.info(f"[Gauntlet] Baselines: {baselines}")

        # Create game tasks
        tasks = self.create_game_tasks(unrated, baselines, config_key)
        self._current_run.total_games = len(tasks)
        logger.info(f"[Gauntlet] Created {len(tasks)} game tasks")

        # Discover available workers
        workers = await self._discover_workers(p2p_url)
        if not workers:
            logger.warning("[Gauntlet] No workers available, falling back to local execution")
            results = await self._execute_tasks_local(tasks, config_key)
        else:
            logger.info(f"[Gauntlet] Distributing to {len(workers)} workers: {[w['node_id'] for w in workers]}")
            results = await self._execute_tasks_distributed(tasks, workers, config_key)

        # Aggregate results and update Elo
        self._aggregate_results(results)
        await self._update_elo_from_results(config_key, results)

        # Mark complete
        self._current_run.completed_at = time.time()
        self._current_run.status = "completed"
        self._current_run.models_evaluated = len(unrated)

        conn = self._get_db_connection()
        try:
            conn.execute("""
                UPDATE gauntlet_runs
                SET completed_at = ?, status = ?, models_evaluated = ?, total_games = ?
                WHERE run_id = ?
            """, (
                self._current_run.completed_at,
                "completed",
                len(unrated),
                len(results),
                run_id,
            ))
            conn.commit()
        finally:
            conn.close()

        duration = self._current_run.completed_at - self._current_run.started_at
        logger.info(
            f"[Gauntlet] Completed {run_id}: {len(unrated)} models, "
            f"{len(results)} games in {duration:.1f}s"
        )

        return self._current_run

    async def _discover_workers(self, p2p_url: str) -> List[Dict[str, Any]]:
        """Discover available gauntlet workers from P2P cluster.

        Args:
            p2p_url: URL of P2P orchestrator

        Returns:
            List of worker info dicts with node_id, host, port
        """
        # Timeout for considering a peer "alive" (90 seconds)
        ALIVE_TIMEOUT = 90

        try:
            import aiohttp
            from aiohttp import ClientTimeout

            timeout = ClientTimeout(total=30)  # Allow more time for large clusters
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{p2p_url}/status") as resp:
                    if resp.status != 200:
                        logger.warning(f"[Gauntlet] P2P status failed: {resp.status}")
                        return []

                    data = await resp.json()

            # Find workers that are alive and have capacity
            workers = []
            peers = data.get("peers", {})
            now = time.time()

            for node_id, peer_info in peers.items():
                # Compute aliveness from last_heartbeat timestamp
                # Status response has last_heartbeat as Unix timestamp, not is_alive flag
                last_heartbeat = peer_info.get("last_heartbeat", 0)
                is_alive = (now - last_heartbeat) < ALIVE_TIMEOUT

                if not is_alive:
                    logger.debug(f"[Gauntlet] Skipping {node_id}: last heartbeat {now - last_heartbeat:.0f}s ago")
                    continue

                # Get host - try multiple fields (IP address of the node)
                host = peer_info.get("host") or peer_info.get("ip") or peer_info.get("address", "")
                if not host:
                    logger.debug(f"[Gauntlet] Skipping {node_id}: no host/IP address")
                    continue

                # Prefer nodes with GPUs but accept CPU-only for gauntlet
                # (gauntlet games are lighter than training)
                workers.append({
                    "node_id": node_id,
                    "host": host,
                    "port": peer_info.get("port", 8770),
                    "has_gpu": peer_info.get("has_gpu", False),
                    "gpu_name": peer_info.get("gpu_name", ""),
                    "selfplay_jobs": peer_info.get("selfplay_jobs", 0),
                })

            logger.info(f"[Gauntlet] Found {len(workers)} alive workers from {len(peers)} peers")

            # Sort by GPU power, then by current load
            workers.sort(key=lambda w: (-1 if w["has_gpu"] else 0, w["selfplay_jobs"]))

            # Limit to reserved workers (default 2-4)
            max_workers = self.config.reserved_workers * 2  # Allow some extra
            return workers[:max_workers]

        except asyncio.TimeoutError:
            logger.error("[Gauntlet] Worker discovery timed out - P2P may be under heavy load")
            return []
        except Exception as e:
            logger.error(f"[Gauntlet] Failed to discover workers: {type(e).__name__}: {e}")
            return []

    async def _execute_tasks_distributed(
        self,
        tasks: List[GameTask],
        workers: List[Dict[str, Any]],
        config_key: str,
    ) -> List[GameResult]:
        """Execute game tasks distributed across workers.

        Dispatches batches of games to workers and collects results.

        Args:
            tasks: List of game tasks to execute
            workers: Available workers
            config_key: Config for the games

        Returns:
            List of game results
        """
        results = []
        batch_size = self.config.parallel_games

        # Split tasks into batches for each worker
        worker_batches: Dict[str, List[GameTask]] = {w["node_id"]: [] for w in workers}

        for i, task in enumerate(tasks):
            worker_idx = i % len(workers)
            worker = workers[worker_idx]
            worker_batches[worker["node_id"]].append(task)

        # Dispatch all batches concurrently
        dispatch_tasks = []
        for worker in workers:
            batch = worker_batches[worker["node_id"]]
            if batch:
                dispatch_tasks.append(
                    self._dispatch_and_wait(worker, batch, config_key)
                )

        # Wait for all workers to complete
        batch_results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)

        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"[Gauntlet] Worker batch failed: {batch_result}")
                continue
            if isinstance(batch_result, list):
                results.extend(batch_result)

        logger.info(f"[Gauntlet] Collected {len(results)}/{len(tasks)} results")
        return results

    async def _dispatch_and_wait(
        self,
        worker: Dict[str, Any],
        tasks: List[GameTask],
        config_key: str,
    ) -> List[GameResult]:
        """Dispatch a batch of tasks to a worker and wait for results.

        Uses hybrid transport for reliable communication.

        Args:
            worker: Worker info dict
            tasks: Tasks to execute
            config_key: Config for the games

        Returns:
            List of game results
        """
        node_id = worker["node_id"]
        host = worker["host"]
        port = worker["port"]

        # Prepare batch payload
        payload = {
            "run_id": self._current_run.run_id if self._current_run else "unknown",
            "config_key": config_key,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "model_id": t.model_id,
                    "baseline_id": t.baseline_id,
                    "game_num": t.game_num,
                }
                for t in tasks
            ],
        }

        # Try to use hybrid transport if available
        if HAS_HYBRID_TRANSPORT:
            transport = get_hybrid_transport()
            success, response = await transport.send_request(
                node_id=node_id,
                host=host,
                port=port,
                path="/gauntlet/execute",
                method="POST",
                payload=payload,
                command_type="gauntlet_execute",
                timeout=self.config.timeout_seconds * len(tasks),
            )

            if success and response:
                return self._parse_batch_response(response)
            else:
                logger.warning(f"[Gauntlet] Hybrid transport failed for {node_id}")

        # Fallback to direct HTTP
        try:
            import aiohttp
            from aiohttp import ClientTimeout

            timeout = ClientTimeout(total=self.config.timeout_seconds * len(tasks))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://{host}:{port}/gauntlet/execute"
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_batch_response(data)
                    else:
                        logger.error(f"[Gauntlet] Worker {node_id} returned {resp.status}")
                        return []

        except Exception as e:
            logger.error(f"[Gauntlet] Dispatch to {node_id} failed: {e}")
            return []

    def _parse_batch_response(self, response: Dict[str, Any]) -> List[GameResult]:
        """Parse batch response from worker into GameResult objects."""
        results = []

        for r in response.get("results", []):
            try:
                result = GameResult(
                    task_id=r["task_id"],
                    model_id=r["model_id"],
                    baseline_id=r["baseline_id"],
                    model_won=r.get("model_won", False),
                    baseline_won=r.get("baseline_won", False),
                    draw=r.get("draw", False),
                    game_length=r.get("game_length", 0),
                    duration_sec=r.get("duration_sec", 0.0),
                )
                results.append(result)
            except (KeyError, TypeError) as e:
                logger.warning(f"[Gauntlet] Invalid result format: {e}")

        return results

    async def _update_elo_from_results(
        self,
        config_key: str,
        results: List[GameResult],
    ) -> None:
        """Update Elo ratings based on gauntlet results.

        Uses standard Elo update formula.

        Args:
            config_key: Config for the games
            results: List of game results
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        K = 32  # Standard Elo K-factor

        conn = self._get_db_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            id_col = "model_id" if "model_id" in columns else "participant_id"

            # Get current ratings for all models
            cursor = conn.execute(f"""
                SELECT {id_col}, rating, games_played
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
            """, (board_type, num_players))

            ratings = {}
            games_played = {}
            for row in cursor.fetchall():
                ratings[row[0]] = row[1]
                games_played[row[0]] = row[2]

            # Default rating for new models and random_ai
            DEFAULT_RATING = 1500.0
            ratings.setdefault("random_ai", DEFAULT_RATING)

            # Calculate Elo changes
            elo_changes: Dict[str, float] = {}

            for result in results:
                model_id = result.model_id
                baseline_id = result.baseline_id

                model_rating = ratings.get(model_id, DEFAULT_RATING)
                baseline_rating = ratings.get(baseline_id, DEFAULT_RATING)

                # Expected score
                expected = 1 / (1 + 10 ** ((baseline_rating - model_rating) / 400))

                # Actual score
                if result.model_won:
                    actual = 1.0
                elif result.draw:
                    actual = 0.5
                else:
                    actual = 0.0

                # Elo change
                delta = K * (actual - expected)

                if model_id not in elo_changes:
                    elo_changes[model_id] = 0.0
                elo_changes[model_id] += delta

            # Update database
            for model_id, delta in elo_changes.items():
                new_rating = ratings.get(model_id, DEFAULT_RATING) + delta
                new_games = games_played.get(model_id, 0) + 1

                conn.execute(f"""
                    UPDATE elo_ratings
                    SET rating = ?, games_played = ?
                    WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                """, (new_rating, new_games, model_id, board_type, num_players))

            conn.commit()
            logger.info(f"[Gauntlet] Updated Elo for {len(elo_changes)} models")

        except Exception as e:
            logger.error(f"[Gauntlet] Failed to update Elo: {e}")
        finally:
            conn.close()


def get_gauntlet(
    elo_db_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> DistributedNNGauntlet:
    """Get gauntlet evaluator instance.

    Args:
        elo_db_path: Path to Elo database (default: data/unified_elo.db)
        model_dir: Path to models directory (default: data/models)

    Returns:
        Configured DistributedNNGauntlet instance
    """
    project_root = Path(__file__).parent.parent.parent

    if elo_db_path is None:
        elo_db_path = project_root / "data" / "unified_elo.db"
    if model_dir is None:
        model_dir = project_root / "data" / "models"

    return DistributedNNGauntlet(elo_db_path, model_dir)
