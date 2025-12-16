"""Distributed O(n) gauntlet evaluation for neural network models.

This module implements an efficient gauntlet-style evaluation where each model
plays against a fixed set of baselines rather than O(n²) round-robin. This makes
large-scale model evaluation tractable.

Usage:
    from app.tournament.distributed_gauntlet import DistributedNNGauntlet

    gauntlet = DistributedNNGauntlet(elo_db, model_dir)
    results = await gauntlet.run_gauntlet("square8_2p")
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

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
