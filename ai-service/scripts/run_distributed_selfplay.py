#!/usr/bin/env python
"""Distributed self-play worker for neural network training data generation.

This script is designed to run on cloud VMs (AWS, GCP, Azure) or as part of
a distributed job queue system. It generates training samples from self-play
games and uploads them to cloud storage (S3, GCS) or writes locally.

Usage Examples
==============

Local testing::

    python scripts/run_distributed_selfplay.py \
        --num-games 100 \
        --board-type square8 \
        --output file:///tmp/training_data.jsonl

Cloud deployment with S3::

    WORKER_ID=worker-01 python scripts/run_distributed_selfplay.py \
        --num-games 10000 \
        --board-type square8 \
        --output s3://ringrift-training-data/selfplay/square8 \
        --seed 42

Cloud deployment with GCS::

    WORKER_ID=worker-01 python scripts/run_distributed_selfplay.py \
        --num-games 10000 \
        --board-type hexagonal \
        --output gs://ringrift-training-data/selfplay/hex \
        --seed 42

Job Queue Mode (with Redis)::

    # Worker pulls jobs from Redis queue
    python scripts/run_distributed_selfplay.py \
        --job-queue redis://localhost:6379/0 \
        --queue-name selfplay_jobs \
        --output s3://bucket/prefix

AWS Spot Instance Mode::

    # Handles preemption gracefully with checkpointing
    python scripts/run_distributed_selfplay.py \
        --num-games 5000 \
        --board-type square8 \
        --output s3://bucket/prefix \
        --checkpoint-interval 500 \
        --checkpoint-path /tmp/worker_checkpoint.json

Training Sample Format
======================

Each sample is a JSON object with:
- state: Full GameState dict
- outcome: 1.0 (win), 0.0 (loss), 0.5 (draw/timeout)
- board_type: Board type string
- game_id: Source game ID
- move_number: Position in game
- ply_to_end: Moves until game end
- move: Move that was played (optional, for policy training)
- metadata: Source info, engine config, etc.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

# Ensure app.* imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.game_engine import GameEngine
from app.main import _create_ai_instance, _get_difficulty_profile
from app.models import (
    AIConfig,
    AIType,
    BoardType,
    GameState,
    GameStatus,
)
from app.training.cloud_storage import (
    TrainingSample,
    get_storage,
)
from app.training.env import (
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD,
    TrainingEnvConfig,
    get_theoretical_max_moves,
    make_env,
)
from app.training.selfplay_config import SelfplayConfig, create_argument_parser

# ---------------------------------------------------------------------------
# Model Pool for Diverse Selfplay
# ---------------------------------------------------------------------------

@dataclass
class ModelPoolEntry:
    """Entry in the model pool for diverse selfplay."""
    model_id: str
    path: str
    ai_types: list[str]  # Which AI types can use this model: "descent", "mcts", "minimax"
    elo_estimate: float | None = None  # Estimated Elo if known


def scan_model_pool(
    models_dir: str | None = None,
    include_baselines: bool = True,
) -> list[ModelPoolEntry]:
    """Scan for available model checkpoints to build a diverse pool.

    Returns a list of ModelPoolEntry objects representing available models
    that can be assigned to neural-based AI types in selfplay.

    Args:
        models_dir: Directory to scan for .pth files. Defaults to ai-service/models/
        include_baselines: If True, include non-neural baselines (random, heuristic)

    Returns:
        List of ModelPoolEntry objects
    """
    pool: list[ModelPoolEntry] = []

    # Add non-neural baselines first (always available)
    if include_baselines:
        pool.append(ModelPoolEntry(
            model_id="random",
            path="",
            ai_types=["random"],
            elo_estimate=800.0,
        ))
        pool.append(ModelPoolEntry(
            model_id="heuristic_v1",
            path="",
            ai_types=["heuristic"],
            elo_estimate=1000.0,
        ))
        pool.append(ModelPoolEntry(
            model_id="minimax_heuristic",
            path="",
            ai_types=["minimax"],
            elo_estimate=1200.0,
        ))
        pool.append(ModelPoolEntry(
            model_id="mcts_heuristic",
            path="",
            ai_types=["mcts"],
            elo_estimate=1300.0,
        ))

    # Scan for neural network checkpoints
    if models_dir is None:
        models_dir = os.path.join(ROOT, "models")

    if os.path.isdir(models_dir):
        for filename in os.listdir(models_dir):
            if not filename.endswith(".pth"):
                continue

            filepath = os.path.join(models_dir, filename)
            model_id = filename.replace(".pth", "")

            # Skip MPS-specific variants (they're duplicates)
            if "_mps" in model_id:
                continue

            # Determine which AI types can use this model
            ai_types = ["descent", "mcts"]  # Neural models work with both
            if "nnue" in model_id.lower():
                ai_types.append("minimax")  # NNUE models can also be used with minimax

            # Try to extract Elo estimate from filename or metadata
            elo_estimate = None
            meta_path = filepath.replace(".pth", ".meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                        elo_estimate = meta.get("elo_rating") or meta.get("elo")
                except Exception:
                    pass

            pool.append(ModelPoolEntry(
                model_id=model_id,
                path=filepath,
                ai_types=ai_types,
                elo_estimate=elo_estimate,
            ))

    logger.info(f"Model pool: {len(pool)} entries ({len([p for p in pool if p.path])} neural)")
    return pool


def get_models_for_ai_type(pool: list[ModelPoolEntry], ai_type: str) -> list[ModelPoolEntry]:
    """Filter pool to models compatible with a given AI type."""
    return [m for m in pool if ai_type in m.ai_types]


import contextlib

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_distributed_selfplay")


@dataclass
class WorkerConfig:
    """Configuration for a distributed worker."""

    worker_id: str
    num_games: int
    board_type: BoardType
    num_players: int
    max_moves: int
    seed: int | None
    engine_mode: str
    difficulty_band: str
    output_uri: str
    checkpoint_interval: int
    checkpoint_path: str | None
    sample_every_n_moves: int
    gc_interval: int
    telemetry_path: str | None = None
    telemetry_interval: int = 50


@dataclass
class WorkerStats:
    """Statistics from a worker run."""

    worker_id: str
    games_completed: int
    samples_generated: int
    wins_by_player: dict[int, int]
    avg_game_length: float
    total_time_sec: float
    games_per_second: float
    samples_per_second: float
    storage_stats: dict[str, Any]


class Telemetry:
    """Real-time telemetry for distributed selfplay monitoring.

    Tracks metrics like win rates, game lengths, and throughput,
    and periodically writes them to a JSON file for monitoring dashboards.
    """

    def __init__(
        self,
        worker_id: str,
        output_path: str | None = None,
        log_interval: int = 50,
    ):
        self.worker_id = worker_id
        self.output_path = output_path
        self.log_interval = log_interval
        self.start_time = time.time()

        # Metrics
        self.games_completed = 0
        self.samples_generated = 0
        self.wins_by_player: dict[int, int] = {}
        self.game_lengths: list[int] = []
        self.draws = 0

        # Rolling window for recent stats (last 100 games)
        self._recent_lengths: list[int] = []
        self._recent_winners: list[int] = []
        self._max_recent = 100

    def record_game(
        self,
        winner: int,
        move_count: int,
        samples_count: int,
    ) -> None:
        """Record a completed game."""
        self.games_completed += 1
        self.samples_generated += samples_count
        self.game_lengths.append(move_count)

        if winner == 0:
            self.draws += 1
        else:
            self.wins_by_player[winner] = self.wins_by_player.get(winner, 0) + 1

        # Rolling window
        self._recent_lengths.append(move_count)
        self._recent_winners.append(winner)
        if len(self._recent_lengths) > self._max_recent:
            self._recent_lengths.pop(0)
            self._recent_winners.pop(0)

        # Periodic logging and export
        if self.games_completed % self.log_interval == 0:
            self._log_progress()
            if self.output_path:
                self._write_metrics()

    def _log_progress(self) -> None:
        """Log current progress with detailed metrics."""
        elapsed = time.time() - self.start_time
        games_per_sec = self.games_completed / elapsed if elapsed > 0 else 0
        self.samples_generated / elapsed if elapsed > 0 else 0

        # Win rates
        total_decided = sum(self.wins_by_player.values())
        win_rates = {}
        for p, wins in sorted(self.wins_by_player.items()):
            win_rates[f"P{p}"] = f"{100 * wins / total_decided:.1f}%" if total_decided > 0 else "N/A"

        # Recent game lengths
        recent_avg = sum(self._recent_lengths) / len(self._recent_lengths) if self._recent_lengths else 0
        recent_std = (
            (sum((x - recent_avg) ** 2 for x in self._recent_lengths) / len(self._recent_lengths)) ** 0.5
            if len(self._recent_lengths) > 1 else 0
        )

        logger.info(
            f"[{self.worker_id}] Games: {self.games_completed} | "
            f"Samples: {self.samples_generated} | "
            f"{games_per_sec:.2f} g/s | "
            f"Win rates: {win_rates} | "
            f"Moves: {recent_avg:.1f}±{recent_std:.1f}"
        )

    def _write_metrics(self) -> None:
        """Write metrics to JSON file for monitoring."""
        elapsed = time.time() - self.start_time
        total_decided = sum(self.wins_by_player.values())

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_id": self.worker_id,
            "elapsed_seconds": elapsed,
            "games_completed": self.games_completed,
            "samples_generated": self.samples_generated,
            "games_per_second": self.games_completed / elapsed if elapsed > 0 else 0,
            "samples_per_second": self.samples_generated / elapsed if elapsed > 0 else 0,
            "draws": self.draws,
            "draw_rate": self.draws / self.games_completed if self.games_completed > 0 else 0,
            "wins_by_player": self.wins_by_player,
            "win_rates": {
                p: wins / total_decided if total_decided > 0 else 0
                for p, wins in self.wins_by_player.items()
            },
            "avg_game_length": sum(self.game_lengths) / len(self.game_lengths) if self.game_lengths else 0,
            "recent_avg_length": sum(self._recent_lengths) / len(self._recent_lengths) if self._recent_lengths else 0,
        }

        try:
            with open(self.output_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write telemetry: {e}")

    def get_final_stats(self) -> dict[str, Any]:
        """Get final statistics for summary."""
        elapsed = time.time() - self.start_time
        total_decided = sum(self.wins_by_player.values())

        return {
            "worker_id": self.worker_id,
            "games_completed": self.games_completed,
            "samples_generated": self.samples_generated,
            "elapsed_seconds": elapsed,
            "games_per_second": self.games_completed / elapsed if elapsed > 0 else 0,
            "draws": self.draws,
            "win_rates": {
                p: wins / total_decided if total_decided > 0 else 0
                for p, wins in self.wins_by_player.items()
            },
            "avg_game_length": sum(self.game_lengths) / len(self.game_lengths) if self.game_lengths else 0,
            "length_std": (
                (sum((x - sum(self.game_lengths) / len(self.game_lengths)) ** 2 for x in self.game_lengths) / len(self.game_lengths)) ** 0.5
                if len(self.game_lengths) > 1 else 0
            ),
        }


class GracefulShutdown:
    """Handler for graceful shutdown on SIGTERM/SIGINT."""

    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(
            "Received signal %s, initiating graceful shutdown...",
            signum,
        )
        self.should_stop = True


def parse_board_type(name: str) -> BoardType:
    """Parse board type from string."""
    name = name.lower()
    if name == "square8":
        return BoardType.SQUARE8
    if name == "square19":
        return BoardType.SQUARE19
    if name == "hexagonal" or name == "hex":
        return BoardType.HEXAGONAL
    raise ValueError(f"Unknown board type: {name}")


@dataclass
class PlayerAssignment:
    """Track AI assignment for a player including model info."""
    player_number: int
    ai_type: str
    difficulty: int
    model_id: str | None = None
    model_path: str | None = None
    elo_estimate: float | None = None


def build_ai_pool(
    game_index: int,
    player_numbers: list[int],
    engine_mode: str,
    base_seed: int | None,
    board_type: BoardType,
    difficulty_band: str,
    model_pool: list[ModelPoolEntry] | None = None,
) -> tuple[dict[int, Any], list[PlayerAssignment]]:
    """Build AI instances for all players in a game.

    Args:
        game_index: Index of this game for seeding
        player_numbers: List of player numbers to create AIs for
        engine_mode: "descent-only", "mixed", or "diverse"
        base_seed: Base random seed
        board_type: Board type for heuristic config
        difficulty_band: "light" or "full" difficulty range
        model_pool: Optional pool of models for diverse mode

    Returns:
        Tuple of (ai_by_player dict, player_assignments list)
        The assignments list tracks which model was assigned to each player
        for Elo tracking purposes.
    """
    ai_by_player: dict[int, Any] = {}
    assignments: list[PlayerAssignment] = []

    game_rng = random.Random((base_seed + game_index) if base_seed is not None else None)

    if engine_mode == "descent-only":
        from app.ai.descent_ai import DescentAI

        # In descent-only mode, optionally use different neural models per player
        descent_models = []
        if model_pool:
            descent_models = get_models_for_ai_type(model_pool, "descent")

        for pnum in player_numbers:
            # Pick a random neural model if available
            model_id = None
            model_path = None
            elo_estimate = None
            if descent_models:
                chosen = game_rng.choice(descent_models)
                model_id = chosen.model_id
                model_path = chosen.path if chosen.path else None
                elo_estimate = chosen.elo_estimate

            cfg = AIConfig(
                difficulty=5,
                think_time=0,
                randomness=0.1,
                rngSeed=(base_seed or 0) + pnum + game_index,
                nn_model_id=model_id,
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
            assignments.append(PlayerAssignment(
                player_number=pnum,
                ai_type="descent",
                difficulty=5,
                model_id=model_id,
                model_path=model_path,
                elo_estimate=elo_estimate,
            ))

        return ai_by_player, assignments

    # Mixed mode - use diverse AI types from difficulty profiles
    # In "diverse" mode, also vary the neural models used
    difficulty_choices = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    if difficulty_band == "light":
        difficulty_choices = [1, 2, 4, 5]

    for pnum in player_numbers:
        difficulty = game_rng.choice(difficulty_choices)
        profile = _get_difficulty_profile(difficulty)
        ai_type = profile["ai_type"]

        heuristic_profile_id = None
        heuristic_eval_mode = None
        model_id = None
        model_path = None
        elo_estimate = None

        # Apply weight noise for heuristic AIs to increase training diversity
        weight_noise = 0.1 if ai_type == AIType.HEURISTIC else 0.0

        if ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")
            heuristic_eval_mode = TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.get(board_type, "full")

        # For neural-capable AI types, randomly select a model from the pool
        if model_pool and engine_mode == "diverse":
            ai_type_str = ai_type.value.lower() if hasattr(ai_type, 'value') else str(ai_type).lower()
            compatible_models = get_models_for_ai_type(model_pool, ai_type_str)

            if compatible_models:
                chosen = game_rng.choice(compatible_models)
                model_id = chosen.model_id
                model_path = chosen.path if chosen.path else None
                elo_estimate = chosen.elo_estimate

        cfg = AIConfig(
            difficulty=difficulty,
            randomness=profile["randomness"],
            think_time=0,
            rngSeed=game_rng.randrange(0, 2**31),
            heuristic_profile_id=heuristic_profile_id,
            heuristic_eval_mode=heuristic_eval_mode,
            nn_model_id=model_id,
            weight_noise=weight_noise,
        )
        ai_by_player[pnum] = _create_ai_instance(ai_type, pnum, cfg)

        ai_type_name = ai_type.value if hasattr(ai_type, 'value') else str(ai_type)
        assignments.append(PlayerAssignment(
            player_number=pnum,
            ai_type=ai_type_name,
            difficulty=difficulty,
            model_id=model_id,
            model_path=model_path,
            elo_estimate=elo_estimate,
        ))

    return ai_by_player, assignments


def extract_training_samples(
    game_id: str,
    state_history: list[GameState],
    move_history: list[Any],
    final_state: GameState,
    board_type: BoardType,
    sample_interval: int,
    metadata: dict[str, Any],
) -> list[TrainingSample]:
    """Extract training samples from a completed game.

    For each sampled state, assigns the outcome from the perspective
    of the current player at that state.
    """
    samples = []
    total_moves = len(state_history)
    winner = final_state.winner

    for i, state in enumerate(state_history):
        # Sample every N moves, but always sample first few and last few
        is_critical = i < 5 or i >= total_moves - 5
        if not is_critical and i % sample_interval != 0:
            continue

        # Determine outcome from current player's perspective
        current_player = state.current_player
        if winner is None:
            outcome = 0.5  # Draw/timeout
        elif winner == current_player:
            outcome = 1.0  # Win
        else:
            outcome = 0.0  # Loss

        # Get the move that was played (if available)
        move_json = None
        if i < len(move_history):
            with contextlib.suppress(Exception):
                move_json = move_history[i].model_dump_json()

        sample = TrainingSample(
            state_json=state.model_dump_json(),
            outcome=outcome,
            board_type=board_type.value,
            game_id=game_id,
            move_number=i,
            ply_to_end=total_moves - i,
            move_json=move_json,
            metadata=metadata,
        )
        samples.append(sample)

    return samples


def run_single_game(
    config: WorkerConfig,
    game_index: int,
    env: Any,
    model_pool: list[ModelPoolEntry] | None = None,
) -> tuple[GameState | None, list[GameState], list[Any], dict[str, Any]]:
    """Run a single self-play game.

    Args:
        config: Worker configuration
        game_index: Index of this game
        env: Game environment
        model_pool: Optional pool of models for diverse selfplay

    Returns:
        Tuple of (final_state, state_history, move_history, game_info)
        Returns (None, [], [], {}) if game fails.
    """
    game_seed = None if config.seed is None else config.seed + game_index
    game_id = f"{config.worker_id}_{game_index}_{uuid.uuid4().hex[:8]}"

    try:
        state = env.reset(seed=game_seed)
    except Exception as exc:
        logger.error(f"Game {game_index}: Failed to reset: {exc}")
        return None, [], [], {}

    player_numbers = [p.player_number for p in state.players]
    ai_by_player, assignments = build_ai_pool(
        game_index,
        player_numbers,
        config.engine_mode,
        config.seed,
        config.board_type,
        config.difficulty_band,
        model_pool=model_pool,
    )

    state_history: list[GameState] = [state.model_copy(deep=True)]
    move_history: list[Any] = []
    move_count = 0

    while True:
        if state.game_status != GameStatus.ACTIVE:
            break

        legal_moves = env.legal_moves()
        if not legal_moves:
            break

        current_player = state.current_player
        ai = ai_by_player.get(current_player)
        if ai is None:
            break

        move = ai.select_move(state)
        if not move:
            break

        try:
            state, _reward, done, _info = env.step(move)
            move_history.append(move)
            state_history.append(state.model_copy(deep=True))
        except Exception as exc:
            logger.warning(f"Game {game_index}: Step exception: {exc}")
            break

        move_count += 1
        if move_count >= config.max_moves or done:
            break

    # Build player assignment info for Elo tracking
    player_info = {}
    for assign in assignments:
        player_info[f"player_{assign.player_number}"] = {
            "ai_type": assign.ai_type,
            "difficulty": assign.difficulty,
            "model_id": assign.model_id,
            "elo_estimate": assign.elo_estimate,
        }

    game_info = {
        "game_id": game_id,
        "game_index": game_index,
        "move_count": move_count,
        "winner": state.winner,
        "status": state.game_status.value,
        "player_assignments": player_info,
        "engine_mode": config.engine_mode,
    }

    return state, state_history, move_history, game_info


def save_checkpoint(
    config: WorkerConfig,
    games_completed: int,
    samples_generated: int,
    wins_by_player: dict[int, int],
) -> None:
    """Save checkpoint for resumption after preemption."""
    if not config.checkpoint_path:
        return

    checkpoint = {
        "worker_id": config.worker_id,
        "games_completed": games_completed,
        "samples_generated": samples_generated,
        "wins_by_player": wins_by_player,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(config.checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved: {games_completed} games completed")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(config: WorkerConfig) -> dict[str, Any] | None:
    """Load checkpoint if it exists."""
    if not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        return None

    try:
        with open(config.checkpoint_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def run_worker(config: WorkerConfig) -> WorkerStats:
    """Run the distributed worker loop.

    Generates self-play games, extracts training samples, and uploads
    to cloud storage. Handles graceful shutdown and checkpointing.
    """
    shutdown = GracefulShutdown()
    start_time = time.time()

    # Initialize storage backend
    storage = get_storage(
        config.output_uri,
        buffer_size=1000,
        compress=True,
    )

    # Initialize environment via canonical factory
    env_config = TrainingEnvConfig(
        board_type=config.board_type,
        num_players=config.num_players,
        max_moves=config.max_moves,
        reward_mode="terminal",
    )
    env = make_env(env_config)

    # Scan model pool for diverse selfplay
    model_pool: list[ModelPoolEntry] | None = None
    if config.engine_mode in ("diverse", "descent-only"):
        model_pool = scan_model_pool(include_baselines=(config.engine_mode == "diverse"))
        logger.info(f"Loaded model pool with {len(model_pool)} entries for {config.engine_mode} mode")

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(config)
    start_game = 0
    samples_generated = 0
    wins_by_player: dict[int, int] = {}
    total_game_lengths: list[int] = []

    if checkpoint:
        start_game = checkpoint.get("games_completed", 0)
        samples_generated = checkpoint.get("samples_generated", 0)
        wins_by_player = checkpoint.get("wins_by_player", {})
        logger.info(f"Resuming from checkpoint: {start_game} games completed")

    # Initialize telemetry for real-time monitoring
    telemetry = Telemetry(
        worker_id=config.worker_id,
        output_path=config.telemetry_path,
        log_interval=config.telemetry_interval,
    )
    # Restore telemetry state from checkpoint
    telemetry.games_completed = start_game
    telemetry.samples_generated = samples_generated
    telemetry.wins_by_player = wins_by_player.copy()

    logger.info(
        f"Worker {config.worker_id} starting: "
        f"{config.num_games} games, board={config.board_type.value}, "
        f"engine_mode={config.engine_mode}, output={config.output_uri}"
    )
    if config.telemetry_path:
        logger.info(f"Telemetry enabled: {config.telemetry_path}")

    games_completed = start_game

    for game_idx in range(start_game, config.num_games):
        if shutdown.should_stop:
            logger.info("Shutdown requested, stopping gracefully...")
            break

        # Run single game with model pool for diverse assignment
        final_state, state_history, move_history, game_info = run_single_game(
            config, game_idx, env, model_pool=model_pool
        )

        if final_state is None:
            continue

        games_completed += 1
        total_game_lengths.append(game_info.get("move_count", 0))

        # Track winner stats
        winner = game_info.get("winner")
        if winner is not None:
            wins_by_player[winner] = wins_by_player.get(winner, 0) + 1

        # Extract training samples with player assignment metadata for Elo tracking
        samples = extract_training_samples(
            game_id=game_info["game_id"],
            state_history=state_history,
            move_history=move_history,
            final_state=final_state,
            board_type=config.board_type,
            sample_interval=config.sample_every_n_moves,
            metadata={
                "worker_id": config.worker_id,
                "engine_mode": config.engine_mode,
                "difficulty_band": config.difficulty_band,
                "seed": config.seed,
                "player_assignments": game_info.get("player_assignments", {}),
            },
        )

        # Telemetry (best-effort) for real-time monitoring dashboards
        try:
            winner_value = game_info.get("winner")
            winner_int = int(winner_value) if winner_value is not None else 0
            telemetry.record_game(
                winner=winner_int,
                move_count=int(game_info.get("move_count", 0) or 0),
                samples_count=len(samples),
            )
        except Exception as e:
            logger.debug(f"Telemetry record failed: {e}")

        # Write samples to storage
        for sample in samples:
            storage.write_training_sample(sample)
            samples_generated += 1

        # Checkpointing
        if config.checkpoint_interval > 0 and games_completed % config.checkpoint_interval == 0:
            storage.flush()
            save_checkpoint(
                config,
                games_completed,
                samples_generated,
                wins_by_player,
            )

        # Garbage collection
        if config.gc_interval > 0 and games_completed % config.gc_interval == 0:
            GameEngine.clear_cache()
            gc.collect()

    # Final flush and close
    storage.flush()
    storage.close()

    # Log final telemetry summary
    final_telemetry = telemetry.get_final_stats()
    logger.info(
        f"[{config.worker_id}] Final: {final_telemetry['games_completed']} games, "
        f"{final_telemetry['samples_generated']} samples, "
        f"{final_telemetry['games_per_second']:.2f} g/s, "
        f"avg_length={final_telemetry['avg_game_length']:.1f}±{final_telemetry['length_std']:.1f}"
    )
    if config.telemetry_path:
        telemetry._write_metrics()  # Final metrics write

    # Compute stats
    elapsed = time.time() - start_time
    avg_length = sum(total_game_lengths) / len(total_game_lengths) if total_game_lengths else 0.0

    stats = WorkerStats(
        worker_id=config.worker_id,
        games_completed=games_completed,
        samples_generated=samples_generated,
        wins_by_player=wins_by_player,
        avg_game_length=avg_length,
        total_time_sec=elapsed,
        games_per_second=games_completed / elapsed if elapsed > 0 else 0,
        samples_per_second=samples_generated / elapsed if elapsed > 0 else 0,
        storage_stats=storage.get_stats(),
    )

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command line arguments using unified SelfplayConfig parser."""
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="Distributed self-play worker for training data generation",
        include_gpu=True,
        include_ramdrive=False,
    )

    # Add script-specific arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output URI: file://, s3://, or gs://",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum moves per game (auto-calculated from board/players if not set)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=3,
        help="Sample every N moves (default: 3)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path for checkpoint file (for resumption)",
    )
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=50,
        help="Run garbage collection every N games (default: 50)",
    )
    # Override engine-mode choices for this script's specific modes
    parser.add_argument(
        "--distributed-engine-mode",
        choices=["mixed", "descent-only", "diverse"],
        default="mixed",
        dest="distributed_engine_mode",
        help=(
            "AI engine mode: 'mixed' uses difficulty-based AI types, "
            "'descent-only' uses only neural descent, "
            "'diverse' uses all AI types with random model assignments "
            "(default: mixed)"
        ),
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create SelfplayConfig from parsed args for tracking
    selfplay_config = SelfplayConfig(
        board_type=args.board,
        num_players=args.num_players,
        num_games=args.num_games,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        worker_id=args.worker_id,
        telemetry_path=args.telemetry_path,
        telemetry_interval=args.telemetry_interval,
        nn_batch_enabled=args.nn_batch_enabled,
        nn_batch_timeout_ms=args.nn_batch_timeout_ms,
        nn_max_batch_size=args.nn_max_batch_size,
        difficulty_band=args.difficulty_band,
        source="run_distributed_selfplay.py",
        extra_options={
            "output_uri": args.output,
            "sample_interval": args.sample_interval,
            "checkpoint_path": args.checkpoint_path,
            "gc_interval": args.gc_interval,
            "distributed_engine_mode": args.distributed_engine_mode,
        },
    )

    # Configure neural network batching if requested
    # This must be done before importing AI modules
    if selfplay_config.nn_batch_enabled:
        os.environ["RINGRIFT_NN_EVAL_QUEUE"] = "1"
        os.environ["RINGRIFT_NN_EVAL_BATCH_TIMEOUT_MS"] = str(selfplay_config.nn_batch_timeout_ms)
        os.environ["RINGRIFT_NN_EVAL_MAX_BATCH"] = str(selfplay_config.nn_max_batch_size)
        logger.info(f"Neural batching enabled: timeout={selfplay_config.nn_batch_timeout_ms}ms, max_batch={selfplay_config.nn_max_batch_size}")

    # Generate worker ID if not provided
    worker_id = selfplay_config.worker_id or os.environ.get("WORKER_ID")
    if not worker_id:
        worker_id = f"worker_{uuid.uuid4().hex[:8]}"

    # Auto-calculate max_moves from board type and player count if not specified
    board_type = parse_board_type(args.board)
    max_moves = args.max_moves
    if max_moves is None:
        max_moves = get_theoretical_max_moves(board_type, args.num_players)
        logger.info(f"Auto-calculated max_moves={max_moves} for {args.board} {args.num_players}p")

    config = WorkerConfig(
        worker_id=worker_id,
        num_games=selfplay_config.num_games,
        board_type=board_type,
        num_players=selfplay_config.num_players,
        max_moves=max_moves,
        seed=selfplay_config.seed,
        engine_mode=selfplay_config.extra_options["distributed_engine_mode"],
        difficulty_band=selfplay_config.difficulty_band,
        output_uri=selfplay_config.extra_options["output_uri"],
        checkpoint_interval=selfplay_config.checkpoint_interval,
        checkpoint_path=selfplay_config.extra_options["checkpoint_path"],
        sample_every_n_moves=selfplay_config.extra_options["sample_interval"],
        gc_interval=selfplay_config.extra_options["gc_interval"],
        telemetry_path=selfplay_config.telemetry_path,
        telemetry_interval=selfplay_config.telemetry_interval,
    )

    logger.info(f"Starting distributed self-play worker: {worker_id}")
    logger.info(f"Config: {json.dumps(asdict(config), default=str, indent=2)}")

    stats = run_worker(config)

    # Print summary
    print("\n" + "=" * 60)
    print("WORKER SUMMARY")
    print("=" * 60)
    print(f"Worker ID:        {stats.worker_id}")
    print(f"Games completed:  {stats.games_completed}")
    print(f"Samples generated: {stats.samples_generated}")
    print(f"Total time:       {stats.total_time_sec:.1f}s")
    print(f"Games/sec:        {stats.games_per_second:.2f}")
    print(f"Samples/sec:      {stats.samples_per_second:.2f}")
    print(f"Avg game length:  {stats.avg_game_length:.1f}")
    print(f"Wins by player:   {stats.wins_by_player}")
    print("=" * 60)

    # Write final stats
    stats_path = f"/tmp/worker_{worker_id}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
