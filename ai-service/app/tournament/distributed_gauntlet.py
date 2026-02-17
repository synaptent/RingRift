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
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.cluster_config import get_host_provider

logger = logging.getLogger(__name__)

# Try to import hybrid transport for distributed execution
try:
    from app.distributed.hybrid_transport import HybridTransport, get_hybrid_transport
    HAS_HYBRID_TRANSPORT = True
except ImportError:
    HAS_HYBRID_TRANSPORT = False
    get_hybrid_transport = None
    HybridTransport = None

# Config keys for all 12 board/player combinations
CONFIG_KEYS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Theoretical max moves per config for complete games
MAX_MOVES = {
    "hex8_2p": 400, "hex8_3p": 500, "hex8_4p": 600,
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
    stale_run_timeout: int = 3600     # Max time (sec) a run can be "running" before auto-cleanup
    max_distributed_wait: int = 1800  # Max wait for distributed execution (30 min)


@dataclass
class GauntletResult:
    """Result of a gauntlet evaluation run."""

    run_id: str
    config_key: str
    started_at: float
    completed_at: float | None = None
    models_evaluated: int = 0
    total_games: int = 0
    status: str = "pending"
    model_results: dict[str, dict] = field(default_factory=dict)


@dataclass
class GameTask:
    """A single game to be played in the gauntlet."""

    task_id: str
    model_id: str
    baseline_id: str
    config_key: str
    game_num: int


@dataclass
class DistributedGameResult:
    """Result of a single distributed gauntlet game.

    December 2025: Renamed from GameResult to avoid collision with
    app.training.selfplay_runner.GameResult (canonical for selfplay) and
    app.execution.game_executor.GameResult (canonical for execution).
    """

    task_id: str
    model_id: str
    baseline_id: str
    model_won: bool
    baseline_won: bool
    draw: bool
    game_length: int
    duration_sec: float


# Backward-compat alias (deprecated Dec 2025)
GameResult = DistributedGameResult


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
        config: GauntletConfig | None = None,
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
        self._current_run: GauntletResult | None = None
        self._reserved_workers: set[str] = set()

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

    def _cleanup_stale_runs(self, config_key: str | None = None) -> int:
        """Clean up stale gauntlet runs that have been 'running' too long.

        This prevents gauntlet deadlock when previous runs crashed or timed out.
        Should be called before starting a new gauntlet.

        Args:
            config_key: If provided, only clean up runs for this config

        Returns:
            Number of stale runs cleaned up
        """
        conn = self._get_db_connection()
        try:
            cutoff_time = time.time() - self.config.stale_run_timeout

            if config_key:
                cursor = conn.execute("""
                    SELECT run_id, config_key, started_at
                    FROM gauntlet_runs
                    WHERE status = 'running'
                    AND started_at < ?
                    AND config_key = ?
                """, (cutoff_time, config_key))
            else:
                cursor = conn.execute("""
                    SELECT run_id, config_key, started_at
                    FROM gauntlet_runs
                    WHERE status = 'running'
                    AND started_at < ?
                """, (cutoff_time,))

            stale_runs = cursor.fetchall()

            for run_id, cfg, started_at in stale_runs:
                age_hours = (time.time() - started_at) / 3600
                logger.warning(
                    f"[Gauntlet] Cleaning up stale run {run_id} ({cfg}) - "
                    f"stuck for {age_hours:.1f} hours"
                )
                conn.execute("""
                    UPDATE gauntlet_runs
                    SET status = 'timeout', completed_at = ?
                    WHERE run_id = ?
                """, (time.time(), run_id))

            conn.commit()

            if stale_runs:
                logger.info(f"[Gauntlet] Cleaned up {len(stale_runs)} stale runs")

            return len(stale_runs)
        finally:
            conn.close()

    def get_unrated_models(self, config_key: str) -> list[str]:
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

            # Filter out archived models
            archived_filter = ""
            if "archived_at" in columns:
                archived_filter = "AND (archived_at IS NULL OR archived_at = 0)"

            cursor = conn.execute(f"""
                SELECT {id_col}
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                AND games_played < ?
                {archived_filter}
            """, (board_type, num_players, min_games))

            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_models_by_elo(self, config_key: str) -> list[tuple[str, float]]:
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

    def discover_and_register_models(self, config_key: str) -> int:
        """Scan models directory and register untracked models in Elo database.

        This ensures all .pth files are tracked for gauntlet evaluation.

        Args:
            config_key: Config like "square8_2p"

        Returns:
            Number of newly registered models
        """

        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Get currently registered models
        registered = {m[0] for m in self.get_models_by_elo(config_key)}

        # Scan models directory for matching .pth files
        # Map board types to their filename prefixes
        # e.g., config "square8_2p" matches files like "sq8_2p_*", "square8_2p_*"
        board_prefixes = {
            "square8": ["sq8", "square8"],
            "square19": ["sq19", "square19"],
            "hexagonal": ["hex", "hexagonal"],
        }
        prefixes = board_prefixes.get(board_type, [board_type])
        patterns = [f"{prefix}_{num_players}p" for prefix in prefixes]

        new_models = []
        for pth_file in self.model_dir.glob("*.pth"):
            from app.training.composite_participant import normalize_nn_id
            model_id = normalize_nn_id(pth_file.stem) or pth_file.stem
            # Check if matches any pattern
            matches = any(model_id.startswith(p) or f"_{p}_" in model_id for p in patterns)
            if matches and model_id not in registered:
                new_models.append(model_id)

        if not new_models:
            return 0

        # Register new models with default rating
        DEFAULT_RATING = 1200.0
        DEFAULT_RD = 350.0

        conn = self._get_db_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            id_col = "model_id" if "model_id" in columns else "participant_id"

            for model_id in new_models:
                try:
                    conn.execute(f"""
                        INSERT OR IGNORE INTO elo_ratings
                        ({id_col}, board_type, num_players, rating, games_played,
                         wins, losses, draws, rating_deviation, last_update)
                        VALUES (?, ?, ?, ?, 0, 0, 0, 0, ?, datetime('now'))
                    """, (model_id, board_type, num_players, DEFAULT_RATING, DEFAULT_RD))
                except Exception as e:
                    logger.warning(f"[Gauntlet] Failed to register {model_id}: {e}")

            conn.commit()
            logger.info(f"[Gauntlet] Registered {len(new_models)} new models for {config_key}")
        finally:
            conn.close()

        return len(new_models)

    def select_baselines(self, config_key: str) -> list[str]:
        """Select diverse baseline opponents for gauntlet comparison.

        January 10, 2026: Expanded from 4 to include diverse opponent types.
        This ensures models are tested against different playing styles:
        - NN models (best/median) - test vs learned strategies
        - Heuristic - test vs hand-crafted evaluation
        - Random - test basic competence
        - Canonical model - test vs current production (if different from best)

        Selection strategy:
        1. Canonical model for this config (production baseline)
        2. Best model (highest Elo) - the champion to beat
        3. Heuristic AI - tests against strategic evaluation
        4. Random AI - consistent competence baseline

        Args:
            config_key: Config like "square8_2p"

        Returns:
            List of baseline model IDs (diverse opponent types)
        """
        models = self.get_models_by_elo(config_key)
        baselines = []

        # 1. Canonical model for this config (if it exists and is different from best)
        canonical_id = f"ringrift_best_{config_key}"
        baselines.append(canonical_id)

        if not models:
            # No trained models yet - use diverse AI baselines
            # January 10, 2026: Include heuristic for strategy testing
            return [canonical_id, f"ringrift_best_{config_key}:policy_only:t0p3", "heuristic", "random_ai"]

        n = len(models)

        # 2. Best model (index 0, highest Elo) - skip if same as canonical
        if models[0][0] != canonical_id:
            baselines.append(models[0][0])

        # 3. Median model (middle) - for mid-range testing
        if n >= 3:
            median_idx = n // 2
            if models[median_idx][0] not in baselines:
                baselines.append(models[median_idx][0])
        elif n >= 2:
            if models[1][0] not in baselines:
                baselines.append(models[1][0])

        # 4. Heuristic AI - tests strategic play without neural network
        # January 10, 2026: Always include for diversity
        baselines.append("heuristic")

        # 5. RandomAI baseline (always include for competence floor)
        baselines.append("random_ai")

        # Ensure we have unique baselines
        baselines = list(dict.fromkeys(baselines))

        return baselines

    def _ensure_baselines_registered(
        self, baselines: list[str], config_key: str
    ) -> None:
        """Auto-register heuristic/random baselines as participants.

        Without registered baselines, Elo ratings are only relative to
        neural net self-variants. Baselines provide absolute calibration.
        """
        BASELINE_TYPES = {
            "heuristic": ("baseline", "HEURISTIC", 2, False),
            "random_ai": ("baseline", "RANDOM", 1, False),
        }

        if not hasattr(self, "_elo_service") or self._elo_service is None:
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except (ImportError, Exception):
                pass

        elo_svc = getattr(self, "_elo_service", None)
        if elo_svc is None:
            return

        for baseline_id in baselines:
            meta = BASELINE_TYPES.get(baseline_id)
            if meta is None:
                continue
            ptype, ai_type, difficulty, use_nn = meta
            try:
                elo_svc.elo_db.register_participant(
                    participant_id=baseline_id,
                    participant_type=ptype,
                    ai_type=ai_type,
                    difficulty=difficulty,
                    use_neural_net=use_nn,
                )
            except Exception as e:
                logger.debug(f"[Gauntlet] Baseline registration for {baseline_id}: {e}")

    def create_game_tasks(
        self,
        unrated_models: list[str],
        baselines: list[str],
        config_key: str,
    ) -> list[GameTask]:
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
        # Feb 2026: Skip gauntlet on coordinator nodes to prevent
        # tournament_*.db and gauntlet_*.db from accumulating locally
        try:
            from app.config.env import env
            if not env.gauntlet_enabled:
                logger.warning("Gauntlet disabled on coordinator node")
                return GauntletResult(config_key=config_key, status="skipped")
        except ImportError:
            pass

        self._init_gauntlet_tables()

        # Clean up any stale runs before starting (prevents deadlock)
        self._cleanup_stale_runs(config_key)

        # Discover and register any new models from the models directory
        new_count = self.discover_and_register_models(config_key)
        if new_count > 0:
            logger.info(f"[Gauntlet] Discovered {new_count} new models for {config_key}")

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

        # Auto-register baseline participants (heuristic/random) so they
        # appear in unified_elo.db and provide absolute Elo calibration
        self._ensure_baselines_registered(baselines, config_key)

        # Create game tasks
        tasks = self.create_game_tasks(unrated, baselines, config_key)
        self._current_run.total_games = len(tasks)
        logger.info(f"[Gauntlet] Created {len(tasks)} game tasks")

        # Execute tasks (this would be distributed in production)
        # For now, run locally with asyncio
        results = await self._execute_tasks_local(tasks, config_key)

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
        tasks: list[GameTask],
        config_key: str,
    ) -> list[GameResult]:
        """Execute game tasks locally using thread pool.

        Runs actual games using the game engine with NN agents.
        Uses thread pool to avoid blocking the event loop.
        """
        import concurrent.futures

        results = []

        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        max_moves = MAX_MOVES.get(config_key, 2000)

        # Run games in parallel using thread pool
        max_workers = min(self.config.parallel_games, len(tasks))  # Use config parallelism

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(
                    self._run_game_sync,
                    task, board_type, num_players, max_moves,
                )
                futures[future] = task

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning(f"[Gauntlet] Game {task.task_id} failed: {e}")
                    results.append(GameResult(
                        task_id=task.task_id,
                        model_id=task.model_id,
                        baseline_id=task.baseline_id,
                        model_won=False,
                        baseline_won=False,
                        draw=True,
                        game_length=0,
                        duration_sec=0.0,
                    ))

                if len(results) % 10 == 0:
                    logger.info(f"[Gauntlet] Progress: {len(results)}/{len(tasks)} games")

        return results

    def _run_game_sync(
        self,
        task: GameTask,
        board_type: str,
        num_players: int,
        max_moves: int,
    ) -> GameResult:
        """Run a single game synchronously.

        Called from thread pool. Uses GameExecutor for consistent game execution.
        """
        start_time = time.time()

        try:
            from app.execution.game_executor import GameExecutor

            # Derive per-game, per-player seeds for varied behavior
            # Use game_num to ensure different games have different seeds
            base_seed = (task.game_num * 1_000_003 + 12345) & 0x7FFFFFFF

            # Map model IDs to player configs
            player_configs = []

            # Model agent (player 0)
            # Use mcts_25 for fast gauntlet evaluation (25 simulations = ~0.15s/move)
            player_0_seed = (base_seed * 104729 + 1 * 7919) & 0x7FFFFFFF
            if task.model_id == "random_ai":
                player_configs.append({"ai_type": "random", "difficulty": 1, "rngSeed": player_0_seed})
            elif task.model_id.startswith("gmo"):
                # GMO model - use GMO AI type
                player_configs.append({
                    "ai_type": "gmo",
                    "difficulty": 5,
                    "nn_model_id": task.model_id,
                    "rngSeed": player_0_seed,
                })
            else:
                model_path = self.model_dir / f"{task.model_id}.pth"
                if model_path.exists():
                    # Use MCTS with neural guidance - 25 sims for speed
                    player_configs.append({
                        "ai_type": "mcts_25",
                        "difficulty": 5,
                        "nn_model_id": task.model_id,
                        "rngSeed": player_0_seed,
                    })
                else:
                    # Model file not found, use MCTS fallback without NN
                    player_configs.append({"ai_type": "mcts_25", "difficulty": 4, "rngSeed": player_0_seed})

            # Baseline agent (player 1)
            player_1_seed = (base_seed * 104729 + 2 * 7919) & 0x7FFFFFFF
            if task.baseline_id == "random_ai":
                player_configs.append({"ai_type": "random", "difficulty": 1, "rngSeed": player_1_seed})
            elif task.baseline_id.startswith("gmo"):
                # GMO baseline - use GMO AI type
                player_configs.append({
                    "ai_type": "gmo",
                    "difficulty": 5,
                    "nn_model_id": task.baseline_id,
                    "rngSeed": player_1_seed,
                })
            else:
                baseline_path = self.model_dir / f"{task.baseline_id}.pth"
                if baseline_path.exists():
                    player_configs.append({
                        "ai_type": "mcts_25",
                        "difficulty": 5,
                        "nn_model_id": task.baseline_id,
                        "rngSeed": player_1_seed,
                    })
                else:
                    player_configs.append({"ai_type": "mcts_25", "difficulty": 4, "rngSeed": player_1_seed})

            # Add random players for 3p/4p games
            for extra_player_idx in range(len(player_configs), num_players):
                extra_seed = (base_seed * 104729 + (extra_player_idx + 1) * 7919) & 0x7FFFFFFF
                player_configs.append({"ai_type": "random", "difficulty": 1, "rngSeed": extra_seed})

            # Run game using GameExecutor
            executor = GameExecutor(board_type=board_type, num_players=num_players)
            result = executor.run_game(
                player_configs=player_configs,
                max_moves=max_moves,
            )

            game_length = result.move_count
            duration = time.time() - start_time

            # Convert executor result to gauntlet result
            # winner is 1-indexed in GameExecutor, 0 means player 1 won (model)
            if result.winner is None or result.outcome.value == "draw":
                return GameResult(
                    task_id=task.task_id,
                    model_id=task.model_id,
                    baseline_id=task.baseline_id,
                    model_won=False,
                    baseline_won=False,
                    draw=True,
                    game_length=game_length,
                    duration_sec=duration,
                )
            elif result.winner == 1:  # Player 1 (model) won
                return GameResult(
                    task_id=task.task_id,
                    model_id=task.model_id,
                    baseline_id=task.baseline_id,
                    model_won=True,
                    baseline_won=False,
                    draw=False,
                    game_length=game_length,
                    duration_sec=duration,
                )
            else:  # Player 2 (baseline) won
                return GameResult(
                    task_id=task.task_id,
                    model_id=task.model_id,
                    baseline_id=task.baseline_id,
                    model_won=False,
                    baseline_won=True,
                    draw=False,
                    game_length=game_length,
                    duration_sec=duration,
                )

        except Exception as e:
            logger.warning(f"[Gauntlet] Game error: {e}")
            return GameResult(
                task_id=task.task_id,
                model_id=task.model_id,
                baseline_id=task.baseline_id,
                model_won=False,
                baseline_won=False,
                draw=True,
                game_length=0,
                duration_sec=time.time() - start_time,
            )

    def _aggregate_results(self, results: list[GameResult]) -> None:
        """Aggregate game results and update Elo ratings."""
        # Group by model
        model_stats: dict[str, dict[str, dict]] = {}

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
        p2p_url: str | None = None,
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

        # Clean up any stale runs before starting (prevents deadlock)
        self._cleanup_stale_runs(config_key)

        # Discover and register any new models from the models directory
        new_count = self.discover_and_register_models(config_key)
        if new_count > 0:
            logger.info(f"[Gauntlet] Discovered {new_count} new models for {config_key}")

        from app.config.ports import get_local_p2p_url
        p2p_url = p2p_url or os.environ.get("RINGRIFT_P2P_URL") or get_local_p2p_url()

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

        try:
            # Get models to evaluate
            unrated = self.get_unrated_models(config_key)
            if not unrated:
                logger.info(f"[Gauntlet] No unrated models for {config_key}")
                self._current_run.status = "no_work"
                self._mark_run_complete(run_id, "no_work", 0, 0)
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

            self._mark_run_complete(run_id, "completed", len(unrated), len(results))

            duration = self._current_run.completed_at - self._current_run.started_at
            logger.info(
                f"[Gauntlet] Completed {run_id}: {len(unrated)} models, "
                f"{len(results)} games in {duration:.1f}s"
            )

            return self._current_run

        except Exception as e:
            # Mark run as failed on any exception
            logger.error(f"[Gauntlet] Run {run_id} failed with error: {e}")
            self._current_run.status = "failed"
            self._current_run.completed_at = time.time()
            self._mark_run_complete(run_id, "failed", 0, 0)
            raise

    def _mark_run_complete(
        self, run_id: str, status: str, models_evaluated: int, total_games: int
    ) -> None:
        """Mark a gauntlet run as complete in the database.

        Args:
            run_id: The run ID
            status: Final status (completed, failed, no_work, timeout)
            models_evaluated: Number of models evaluated
            total_games: Total games played
        """
        conn = self._get_db_connection()
        try:
            conn.execute("""
                UPDATE gauntlet_runs
                SET completed_at = ?, status = ?, models_evaluated = ?, total_games = ?
                WHERE run_id = ?
            """, (time.time(), status, models_evaluated, total_games, run_id))
            conn.commit()
        finally:
            conn.close()

    async def _discover_workers(self, p2p_url: str) -> list[dict[str, Any]]:
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

            # Use /api/cluster/status?local=1 which is faster than /status
            # and avoids proxying to leader (local=1 flag)
            timeout = ClientTimeout(total=45)  # Allow more time for large clusters
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{p2p_url}/api/cluster/status?local=1") as resp:
                    if resp.status != 200:
                        logger.warning(f"[Gauntlet] P2P status failed: {resp.status}")
                        return []

                    data = await resp.json()

            # Find workers that are alive and have capacity
            # Note: /api/cluster/status returns peers as array with last_seen, not dict with last_heartbeat
            workers = []
            peers = data.get("peers", [])
            now = time.time()

            for peer_info in peers:
                node_id = peer_info.get("node_id", "")
                if not node_id:
                    continue

                # Compute aliveness from last_seen timestamp
                last_seen = peer_info.get("last_seen", 0)
                is_alive = (now - last_seen) < ALIVE_TIMEOUT

                if not is_alive:
                    logger.debug(f"[Gauntlet] Skipping {node_id}: last seen {now - last_seen:.0f}s ago")
                    continue

                # Get host - try multiple fields (IP address of the node)
                host = peer_info.get("host") or peer_info.get("effective_host") or ""
                if not host:
                    logger.debug(f"[Gauntlet] Skipping {node_id}: no host/IP address")
                    continue

                # Collect worker info for gauntlet evaluation
                workers.append({
                    "node_id": node_id,
                    "host": host,
                    "port": peer_info.get("port", 8770),
                    "has_gpu": peer_info.get("has_gpu", False),
                    "gpu_name": peer_info.get("gpu_name", ""),
                    "selfplay_jobs": peer_info.get("selfplay_jobs", 0),
                    "training_jobs": peer_info.get("training_jobs", 0),
                })

            logger.info(f"[Gauntlet] Found {len(workers)} alive workers from {len(peers)} peers")

            # Sort workers to prefer Vast nodes with strong CPUs for gauntlet
            # Priority: Vast nodes > CPU-only nodes > idle Lambda nodes
            # Gauntlet is CPU-bound, so prefer strong CPUs over GPUs
            def gauntlet_worker_priority(w):
                node_id = w["node_id"]
                provider = get_host_provider(node_id)
                is_vast = provider == "vast"
                is_lambda_gpu = provider == "lambda" and w["has_gpu"]
                total_jobs = w["selfplay_jobs"] + w["training_jobs"]

                # Lower score = higher priority
                # Vast nodes get priority 0, CPU-only 1, busy Lambda GPU 2+
                if is_vast:
                    return (0, total_jobs)  # Vast nodes first, prefer idle
                elif not w["has_gpu"]:
                    return (1, total_jobs)  # CPU-only nodes second
                elif is_lambda_gpu:
                    # Lambda GPU nodes last - they should focus on training
                    return (2, -total_jobs)  # Deprioritize busy GPU nodes
                else:
                    return (1, total_jobs)  # Other nodes

            workers.sort(key=gauntlet_worker_priority)

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
        tasks: list[GameTask],
        workers: list[dict[str, Any]],
        config_key: str,
    ) -> list[GameResult]:
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

        # Split tasks into batches for each worker
        worker_batches: dict[str, list[GameTask]] = {w["node_id"]: [] for w in workers}

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

        # Wait for all workers to complete with overall timeout
        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*dispatch_tasks, return_exceptions=True),
                timeout=self.config.max_distributed_wait,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[Gauntlet] Distributed execution timed out after "
                f"{self.config.max_distributed_wait}s - returning partial results"
            )
            batch_results = []

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
        worker: dict[str, Any],
        tasks: list[GameTask],
        config_key: str,
    ) -> list[GameResult]:
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

    def _parse_batch_response(self, response: dict[str, Any]) -> list[GameResult]:
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
        results: list[GameResult],
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

        # Scale K-factor for multiplayer games
        # In N-player games, each game provides (N-1) pairwise matchups worth of info
        # Dividing by (N-1) ensures consistent rating change magnitude across player counts
        K_BASE = 32  # Standard Elo K-factor for 2-player
        base_k = K_BASE / (num_players - 1) if num_players > 2 else K_BASE

        def get_adaptive_k(games_model: int, games_baseline: int) -> float:
            """Adaptive K-factor: higher for new models, lower for established."""
            min_games = min(games_model, games_baseline)
            if min_games < 30:
                return base_k * 1.5  # Provisional: fast convergence
            elif min_games < 100:
                return base_k * 1.25  # Developing
            elif min_games < 300:
                return base_k * 1.0  # Established
            else:
                return base_k * 0.75  # Mature: stability

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
            elo_changes: dict[str, float] = {}

            for result in results:
                model_id = result.model_id
                baseline_id = result.baseline_id

                model_rating = ratings.get(model_id, DEFAULT_RATING)
                baseline_rating = ratings.get(baseline_id, DEFAULT_RATING)

                # Expected score for pairwise matchup (model vs baseline)
                # This is the probability that model finishes ahead of baseline
                # Note: In multiplayer, random players can affect outcomes, but
                # the pairwise expected score between model and baseline is still
                # based on their rating difference
                expected = 1 / (1 + 10 ** ((baseline_rating - model_rating) / 400))

                # Actual score (did model finish ahead of baseline?)
                if result.model_won:
                    actual = 1.0
                elif result.draw:
                    actual = 0.5
                else:
                    actual = 0.0

                # Elo change (adaptive K based on games played)
                model_games = games_played.get(model_id, 0)
                baseline_games = games_played.get(baseline_id, 0)
                K = get_adaptive_k(model_games, baseline_games)
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
                    SET rating = ?,
                        games_played = ?,
                        peak_rating = MAX(peak_rating, ?),
                        last_update = ?
                    WHERE {id_col} = ? AND board_type = ? AND num_players = ?
                """, (new_rating, new_games, new_rating, time.time(),
                      model_id, board_type, num_players))

            conn.commit()
            logger.info(f"[Gauntlet] Updated Elo for {len(elo_changes)} models")

        except Exception as e:
            logger.error(f"[Gauntlet] Failed to update Elo: {e}")
        finally:
            conn.close()


def get_gauntlet(
    elo_db_path: Path | None = None,
    model_dir: Path | None = None,
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
