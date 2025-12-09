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
import logging
import os
import random
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Ensure app.* imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.main import _create_ai_instance, _get_difficulty_profile  # noqa: E402
from app.models import (  # noqa: E402
    AIConfig,
    AIType,
    BoardType,
    GameState,
    GameStatus,
)
from app.training.env import (  # noqa: E402
    TrainingEnvConfig,
    make_env,
    TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD,
)
from app.training.cloud_storage import (  # noqa: E402
    TrainingSample,
    get_storage,
)
from app.game_engine import GameEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a distributed worker."""

    worker_id: str
    num_games: int
    board_type: BoardType
    num_players: int
    max_moves: int
    seed: Optional[int]
    engine_mode: str
    difficulty_band: str
    output_uri: str
    checkpoint_interval: int
    checkpoint_path: Optional[str]
    sample_every_n_moves: int
    gc_interval: int


@dataclass
class WorkerStats:
    """Statistics from a worker run."""

    worker_id: str
    games_completed: int
    samples_generated: int
    wins_by_player: Dict[int, int]
    avg_game_length: float
    total_time_sec: float
    games_per_second: float
    samples_per_second: float
    storage_stats: Dict[str, Any]


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


def build_ai_pool(
    game_index: int,
    player_numbers: List[int],
    engine_mode: str,
    base_seed: Optional[int],
    board_type: BoardType,
    difficulty_band: str,
) -> Dict[int, Any]:
    """Build AI instances for all players in a game."""
    ai_by_player: Dict[int, Any] = {}

    if engine_mode == "descent-only":
        from app.ai.descent_ai import DescentAI

        for pnum in player_numbers:
            cfg = AIConfig(
                difficulty=5,
                think_time=0,
                randomness=0.1,
                rngSeed=(base_seed or 0) + pnum + game_index,
            )
            ai_by_player[pnum] = DescentAI(pnum, cfg)
        return ai_by_player

    # Mixed mode
    difficulty_choices = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    if difficulty_band == "light":
        difficulty_choices = [1, 2, 4, 5]

    game_rng = random.Random((base_seed + game_index) if base_seed is not None else None)

    for pnum in player_numbers:
        difficulty = game_rng.choice(difficulty_choices)
        profile = _get_difficulty_profile(difficulty)
        ai_type = profile["ai_type"]

        heuristic_profile_id = None
        heuristic_eval_mode = None
        if ai_type == AIType.HEURISTIC:
            heuristic_profile_id = profile.get("profile_id")
            heuristic_eval_mode = TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD.get(board_type, "full")

        cfg = AIConfig(
            difficulty=difficulty,
            randomness=profile["randomness"],
            think_time=0,
            rngSeed=game_rng.randrange(0, 2**31),
            heuristic_profile_id=heuristic_profile_id,
            heuristic_eval_mode=heuristic_eval_mode,
        )
        ai_by_player[pnum] = _create_ai_instance(ai_type, pnum, cfg)

    return ai_by_player


def extract_training_samples(
    game_id: str,
    state_history: List[GameState],
    move_history: List[Any],
    final_state: GameState,
    board_type: BoardType,
    sample_interval: int,
    metadata: Dict[str, Any],
) -> List[TrainingSample]:
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
            try:
                move_json = move_history[i].model_dump_json()
            except Exception:
                pass

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
) -> Tuple[Optional[GameState], List[GameState], List[Any], Dict[str, Any]]:
    """Run a single self-play game.

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
    ai_by_player = build_ai_pool(
        game_index,
        player_numbers,
        config.engine_mode,
        config.seed,
        config.board_type,
        config.difficulty_band,
    )

    state_history: List[GameState] = [state.model_copy(deep=True)]
    move_history: List[Any] = []
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

    game_info = {
        "game_id": game_id,
        "game_index": game_index,
        "move_count": move_count,
        "winner": state.winner,
        "status": state.game_status.value,
    }

    return state, state_history, move_history, game_info


def save_checkpoint(
    config: WorkerConfig,
    games_completed: int,
    samples_generated: int,
    wins_by_player: Dict[int, int],
) -> None:
    """Save checkpoint for resumption after preemption."""
    if not config.checkpoint_path:
        return

    checkpoint = {
        "worker_id": config.worker_id,
        "games_completed": games_completed,
        "samples_generated": samples_generated,
        "wins_by_player": wins_by_player,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        with open(config.checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved: {games_completed} games completed")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(config: WorkerConfig) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists."""
    if not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        return None

    try:
        with open(config.checkpoint_path, "r") as f:
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

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(config)
    start_game = 0
    samples_generated = 0
    wins_by_player: Dict[int, int] = {}
    total_game_lengths: List[int] = []

    if checkpoint:
        start_game = checkpoint.get("games_completed", 0)
        samples_generated = checkpoint.get("samples_generated", 0)
        wins_by_player = checkpoint.get("wins_by_player", {})
        logger.info(f"Resuming from checkpoint: {start_game} games completed")

    logger.info(
        f"Worker {config.worker_id} starting: "
        f"{config.num_games} games, board={config.board_type.value}, "
        f"output={config.output_uri}"
    )

    games_completed = start_game

    for game_idx in range(start_game, config.num_games):
        if shutdown.should_stop:
            logger.info("Shutdown requested, stopping gracefully...")
            break

        # Run single game
        final_state, state_history, move_history, game_info = run_single_game(config, game_idx, env)

        if final_state is None:
            continue

        games_completed += 1
        total_game_lengths.append(game_info.get("move_count", 0))

        # Track winner stats
        winner = game_info.get("winner")
        if winner is not None:
            wins_by_player[winner] = wins_by_player.get(winner, 0) + 1

        # Extract training samples
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
            },
        )

        # Write samples to storage
        for sample in samples:
            storage.write_training_sample(sample)
            samples_generated += 1

        # Progress logging
        if games_completed % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = games_completed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {games_completed}/{config.num_games} games, "
                f"{samples_generated} samples, "
                f"{games_per_sec:.2f} games/sec"
            )

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed self-play worker for training data generation")

    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to play (default: 1000)",
    )
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hexagonal"],
        default="square8",
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed for reproducibility",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["mixed", "descent-only"],
        default="mixed",
        help="AI engine mode (default: mixed)",
    )
    parser.add_argument(
        "--difficulty-band",
        choices=["canonical", "light"],
        default="light",
        help="Difficulty band for mixed mode (default: light)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output URI: file://, s3://, or gs://",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=3,
        help="Sample every N moves (default: 3)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N games (default: 100)",
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
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (auto-generated if not provided)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Generate worker ID if not provided
    worker_id = args.worker_id or os.environ.get("WORKER_ID")
    if not worker_id:
        worker_id = f"worker_{uuid.uuid4().hex[:8]}"

    config = WorkerConfig(
        worker_id=worker_id,
        num_games=args.num_games,
        board_type=parse_board_type(args.board_type),
        num_players=args.num_players,
        max_moves=args.max_moves,
        seed=args.seed,
        engine_mode=args.engine_mode,
        difficulty_band=args.difficulty_band,
        output_uri=args.output,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint_path,
        sample_every_n_moves=args.sample_interval,
        gc_interval=args.gc_interval,
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
