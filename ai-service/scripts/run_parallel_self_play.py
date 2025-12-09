#!/usr/bin/env python3
"""Parallel Self-Play Script for AI Training Data Generation.

This script runs multiple self-play games in parallel using ProcessPoolExecutor
to speed up training data generation. Each worker runs independent games with
different seeds for reproducibility.

Usage:
    python scripts/run_parallel_self_play.py \\
        --num-games 16 --num-workers 4 --output-dir ./data

Example:
    python scripts/run_parallel_self_play.py \\
        --num-games 100 \\
        --num-workers 8 \\
        --output-dir ./training_data \\
        --ai-type heuristic
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add the app directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.descent_ai import DescentAI  # noqa: E402
from app.ai.heuristic_ai import HeuristicAI  # noqa: E402
from app.ai.random_ai import RandomAI  # noqa: E402
from app.models import AIConfig, GameStatus  # noqa: E402
from app.training.env import TrainingEnvConfig, make_env  # noqa: E402
from app.utils.memory_config import MemoryConfig  # noqa: E402
from app.utils.progress_reporter import ProgressReporter  # noqa: E402

try:
    from app.ai.mcts_ai import MCTSAI  # noqa: E402
except ImportError:
    MCTSAI = None  # type: ignore

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result from a single self-play game."""

    features: np.ndarray
    globals: np.ndarray
    values: np.ndarray
    policy_indices: np.ndarray
    policy_values: np.ndarray
    game_length: int
    winner: Optional[int]
    seed: int
    worker_id: int
    termination_reason: str = "unknown"


@dataclass
class WorkerTask:
    """Task assignment for a worker process."""

    worker_id: int
    game_indices: List[int]
    base_seed: int
    ai_type: str
    memory_config_gb: float


def create_ai_instance(
    ai_type: str,
    player_number: int,
    seed: int,
    memory_config: Optional[MemoryConfig] = None,
):
    """Create an AI instance based on type string."""
    if ai_type == "heuristic":
        config = AIConfig(
            difficulty=5,
            randomness=0.0,
            think_time=1000,
            rngSeed=seed,
        )
        return HeuristicAI(player_number, config)
    elif ai_type == "mcts":
        if MCTSAI is None:
            raise ImportError("MCTSAI not available")
        config = AIConfig(
            difficulty=5,
            randomness=0.0,
            think_time=1000,
            rngSeed=seed,
        )
        return MCTSAI(player_number, config, memory_config=memory_config)
    elif ai_type == "minimax":
        # DescentAI uses minimax-style search
        config = AIConfig(
            difficulty=5,
            randomness=0.0,
            think_time=1000,
            rngSeed=seed,
        )
        return DescentAI(player_number, config, memory_config=memory_config)
    elif ai_type == "random":
        config = AIConfig(
            difficulty=1,
            randomness=1.0,
            think_time=100,
            rngSeed=seed,
        )
        return RandomAI(player_number, config)
    else:
        raise ValueError(f"Unknown AI type: {ai_type}")


def play_single_game(
    seed: int,
    ai_type: str,
    memory_config: Optional[MemoryConfig] = None,
) -> Tuple[int, Optional[int], str]:
    """Play a single self-play game and return result.

    Args:
        seed: Random seed for reproducibility
        ai_type: Type of AI to use ('heuristic', 'mcts', 'minimax', 'random')
        memory_config: Memory configuration for AI instances

    Returns:
        Tuple of (game_length, winner, termination_reason)
    """
    # Increase env limit to avoid early cutoff
    env_config = TrainingEnvConfig(max_moves=500)
    env = make_env(env_config)
    state = env.reset(seed=seed)

    # Create AI instances for both players
    ai_p1 = create_ai_instance(ai_type, 1, seed, memory_config)
    ai_p2 = create_ai_instance(ai_type, 2, seed + 1000000, memory_config)

    move_count = 0
    max_moves = 500  # Safety limit to prevent infinite games
    termination_reason = "unknown"

    while True:
        # Check if game has ended
        if state.game_status != GameStatus.ACTIVE:
            termination_reason = f"status:{state.game_status.value}"
            break

        if move_count >= max_moves:
            termination_reason = "max_moves_reached"
            break

        current_player = state.current_player
        ai = ai_p1 if current_player == 1 else ai_p2

        # Get legal moves
        legal_moves = env.legal_moves()
        if not legal_moves:
            # This should not happen for ACTIVE games per rules invariant
            termination_reason = "no_legal_moves_for_current_player"
            logger.warning(
                f"Seed {seed}: No legal moves for player {current_player} "
                f"at move {move_count}, game status is still ACTIVE"
            )
            break

        # Select and apply move
        move = ai.select_move(state)
        if move is None:
            termination_reason = "ai_returned_no_move"
            logger.warning(f"Seed {seed}: AI returned no move for player " f"{current_player} at move {move_count}")
            break

        try:
            state, _reward, done, _info = env.step(move)
            move_count += 1
            if done:
                termination_reason = "env_done_flag"
                break
        except Exception as e:
            termination_reason = f"step_exception:{type(e).__name__}"
            logger.error(f"Seed {seed}: Step exception at move {move_count}: {e}")
            break

    # Determine winner from state
    winner = state.winner

    return move_count, winner, termination_reason


def worker_process(task: WorkerTask) -> List[GameResult]:
    """Worker process that runs multiple games.

    Each worker runs its assigned games sequentially and returns results.
    This function runs in a separate process.
    """
    # Reconfigure logging for this worker process
    log_fmt = f"%(asctime)s [Worker-{task.worker_id}] " "%(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    worker_logger = logging.getLogger(f"worker_{task.worker_id}")

    # Create memory config for this worker
    memory_config = None
    if task.memory_config_gb > 0:
        memory_config = MemoryConfig(max_memory_gb=task.memory_config_gb)

    worker_logger.info(
        f"Starting worker {task.worker_id} with "
        f"{len(task.game_indices)} games, "
        f"memory budget: {task.memory_config_gb:.2f} GB"
    )

    results = []
    for game_idx in task.game_indices:
        # Use a unique seed for each game based on base seed and game index
        game_seed = task.base_seed + game_idx * 1000

        try:
            start_time = time.time()
            game_length, winner, termination_reason = play_single_game(
                seed=game_seed,
                ai_type=task.ai_type,
                memory_config=memory_config,
            )
            elapsed = time.time() - start_time

            # Create a minimal result for counting purposes
            # In production, this would include actual training data
            result = GameResult(
                features=np.array([]),
                globals=np.array([]),
                values=np.array([], dtype=np.float32),
                policy_indices=np.array([], dtype=np.int32),
                policy_values=np.array([], dtype=np.float32),
                game_length=game_length,
                winner=winner,
                seed=game_seed,
                worker_id=task.worker_id,
                termination_reason=termination_reason,
            )
            results.append(result)

            # Log with warning if abnormal termination
            if winner is None and not termination_reason.startswith("status:"):
                worker_logger.warning(
                    f"Game {game_idx} ABNORMAL: length={game_length}, "
                    f"reason={termination_reason}, time={elapsed:.2f}s"
                )
            else:
                worker_logger.info(
                    f"Game {game_idx} complete: length={game_length}, "
                    f"winner={winner}, reason={termination_reason}, "
                    f"time={elapsed:.2f}s"
                )

        except Exception as e:
            worker_logger.error(
                f"Game {game_idx} failed with error: {e}",
                exc_info=True,
            )
            # Continue with remaining games even if one fails
            continue

    worker_logger.info(f"Worker {task.worker_id} finished: " f"{len(results)}/{len(task.game_indices)} games completed")
    return results


def aggregate_results(
    results: List[GameResult],
    output_dir: Path,
) -> dict:
    """Aggregate results from all workers and save to output directory.

    Returns:
        Dictionary with aggregation statistics
    """
    if not results:
        return {"total_games": 0, "total_positions": 0}

    total_positions = 0
    total_game_length = 0
    winners: dict = {1: 0, 2: 0, None: 0}

    for result in results:
        total_positions += len(result.features) if len(result.features) else 0
        total_game_length += result.game_length
        winners[result.winner] = winners.get(result.winner, 0) + 1

    # Save a summary metadata file instead of large .npz when no features
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"self_play_summary_{timestamp}.npz"

    # Save summary data
    np.savez_compressed(
        output_path,
        total_games=np.array([len(results)]),
        total_positions=np.array([total_positions]),
        avg_game_length=np.array([total_game_length / len(results) if results else 0]),
        wins_p1=np.array([winners[1]]),
        wins_p2=np.array([winners[2]]),
        draws=np.array([winners[None]]),
    )

    logger.info(f"Saved aggregated data to {output_path}")

    return {
        "total_games": len(results),
        "total_positions": total_positions,
        "output_path": str(output_path),
        "wins_p1": winners[1],
        "wins_p2": winners[2],
        "draws": winners[None],
    }


def divide_memory_budget(total_memory_gb: float, num_workers: int) -> float:
    """Divide total memory budget across workers.

    Args:
        total_memory_gb: Total memory budget in GB
        num_workers: Number of workers

    Returns:
        Per-worker memory budget in GB
    """
    if num_workers <= 0:
        return 0.0
    return total_memory_gb / num_workers


def distribute_games_to_workers(
    num_games: int,
    num_workers: int,
) -> List[List[int]]:
    """Distribute game indices evenly across workers.

    Returns:
        List of game index lists, one per worker
    """
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    distributions = []
    current_idx = 0

    for worker_id in range(num_workers):
        # Give one extra game to earlier workers if there's a remainder
        worker_games = games_per_worker + (1 if worker_id < remainder else 0)
        game_indices = list(range(current_idx, current_idx + worker_games))
        distributions.append(game_indices)
        current_idx += worker_games

    return distributions


def create_worker_tasks(
    num_games: int,
    num_workers: int,
    per_worker_memory_gb: float,
    base_seed: int,
    ai_type: str,
) -> List[WorkerTask]:
    """Create worker tasks with game distributions.

    Args:
        num_games: Total number of games
        num_workers: Number of workers
        per_worker_memory_gb: Memory budget per worker
        base_seed: Base seed for reproducibility
        ai_type: Type of AI to use

    Returns:
        List of WorkerTask objects
    """
    game_distributions = distribute_games_to_workers(num_games, num_workers)

    tasks = []
    for worker_id, game_indices in enumerate(game_distributions):
        if game_indices:  # Only create task if worker has games
            task = WorkerTask(
                worker_id=worker_id,
                game_indices=game_indices,
                base_seed=base_seed,
                ai_type=ai_type,
                memory_config_gb=per_worker_memory_gb,
            )
            tasks.append(task)

    return tasks


def run_parallel_self_play(
    num_games: int,
    num_workers: int,
    output_dir: Path,
    ai_type: str,
    base_seed: int = 42,
    memory_budget_gb: Optional[float] = None,
) -> dict:
    """Run parallel self-play games and aggregate results.

    Args:
        num_games: Total number of games to play
        num_workers: Number of parallel workers
        output_dir: Directory to save output data
        ai_type: Type of AI to use
        base_seed: Base seed for reproducibility
        memory_budget_gb: Total memory budget (if None, read from env)

    Returns:
        Dictionary with run statistics
    """
    start_time = time.time()

    # Time-based progress reporter so that long parallel runs do not
    # appear stalled when individual games are slow. This emits
    # throttled updates (~10s) with completed games, rate, and ETA.
    context_label = f"ai={ai_type} workers={num_workers}"
    progress_reporter = ProgressReporter(
        total_units=num_games,
        unit_name="games",
        report_interval_sec=10.0,
        context_label=context_label,
    )

    # Get memory configuration from environment and divide by workers
    if memory_budget_gb is not None:
        per_worker_memory_gb = divide_memory_budget(memory_budget_gb, num_workers)
    else:
        try:
            total_memory_config = MemoryConfig.from_env()
            per_worker_memory_gb = divide_memory_budget(total_memory_config.max_memory_gb, num_workers)
        except Exception:
            # Default to 1GB per worker if no config available
            per_worker_memory_gb = 1.0

    logger.info(f"Starting parallel self-play: {num_games} games " f"across {num_workers} workers")
    logger.info(f"AI type: {ai_type}, " f"per-worker memory: {per_worker_memory_gb:.2f} GB")

    # Create worker tasks
    tasks = create_worker_tasks(
        num_games=num_games,
        num_workers=num_workers,
        per_worker_memory_gb=per_worker_memory_gb,
        base_seed=base_seed,
        ai_type=ai_type,
    )

    # Run workers in parallel
    all_results = []
    completed_games = 0
    failed_workers = 0

    # Try to import tqdm for progress tracking
    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=num_games, desc="Self-play games", unit="game")
    except ImportError:
        logger.warning("tqdm not installed, progress bar disabled")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(worker_process, task): task for task in tasks}

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results = future.result()
                all_results.extend(results)
                games_completed = len(results)
                completed_games += games_completed

                if pbar is not None:
                    pbar.update(games_completed)

                # Throttled, time-based progress reporting with
                # completed games and any failed workers so far.
                progress_reporter.update(
                    completed=completed_games,
                    extra_metrics={
                        "failed_workers": failed_workers,
                    },
                )

                logger.info(f"Worker {task.worker_id} completed: " f"{games_completed} games")
            except Exception as e:
                failed_workers += 1
                logger.error(
                    f"Worker {task.worker_id} failed: {e}",
                    exc_info=True,
                )

    if pbar is not None:
        pbar.close()

    # Aggregate and save results
    stats = aggregate_results(all_results, output_dir)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["games_per_second"] = completed_games / elapsed if elapsed > 0 else 0
    stats["failed_workers"] = failed_workers

    # Emit a final summary line from the shared progress reporter.
    progress_reporter.finish(
        extra_metrics={
            "games_per_second": stats["games_per_second"],
            "failed_workers": failed_workers,
        },
    )

    logger.info(f"Parallel self-play complete in {elapsed:.2f}s")
    logger.info(
        f"Stats: {stats['total_games']} games, "
        f"{stats['total_positions']} positions, "
        f"{stats['games_per_second']:.2f} games/sec"
    )

    return stats


def main():
    """Main entry point for parallel self-play script."""
    parser = argparse.ArgumentParser(description="Run parallel self-play games for training data generation")
    parser.add_argument(
        "--num-games",
        type=int,
        required=True,
        help="Total number of games to play",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./self_play_data",
        help="Directory to save output data (default: ./self_play_data)",
    )
    parser.add_argument(
        "--ai-type",
        type=str,
        choices=["heuristic", "mcts", "minimax", "random"],
        default="heuristic",
        help="Type of AI to use for self-play (default: heuristic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.num_games < 1:
        parser.error("--num-games must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")

    output_dir = Path(args.output_dir)

    logger.info("Configuration:")
    logger.info(f"  Games: {args.num_games}")
    logger.info(f"  Workers: {args.num_workers}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  AI Type: {args.ai_type}")
    logger.info(f"  Seed: {args.seed}")

    stats = run_parallel_self_play(
        num_games=args.num_games,
        num_workers=args.num_workers,
        output_dir=output_dir,
        ai_type=args.ai_type,
        base_seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PARALLEL SELF-PLAY SUMMARY")
    print("=" * 60)
    print(f"Total games:      {stats['total_games']}")
    print(f"Total positions:  {stats['total_positions']}")
    print(f"Player 1 wins:    {stats.get('wins_p1', 0)}")
    print(f"Player 2 wins:    {stats.get('wins_p2', 0)}")
    print(f"Draws:            {stats.get('draws', 0)}")
    print(f"Failed workers:   {stats.get('failed_workers', 0)}")
    print(f"Elapsed time:     {stats['elapsed_seconds']:.2f}s")
    print(f"Games/second:     {stats['games_per_second']:.2f}")
    if "output_path" in stats:
        print(f"Output file:      {stats['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
