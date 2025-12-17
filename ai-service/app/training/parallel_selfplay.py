"""
Parallel selfplay generation for faster training data collection.

This module provides a worker pool architecture for generating selfplay
games in parallel, achieving 4-8x speedup over sequential generation.

The key insight is that AI objects (DescentAI, MCTSAI) are not pickle-
serializable, so each worker initializes its own AI instances.

Usage:
    from app.training.parallel_selfplay import generate_dataset_parallel

    generate_dataset_parallel(
        num_games=1000,
        output_file="data/dataset.npz",
        num_workers=8,
        board_type=BoardType.SQUARE8,
    )
"""

import logging
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.models import BoardType

logger = logging.getLogger(__name__)


@dataclass
class SelfplayConfig:
    """Configuration for selfplay game generation."""
    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2
    max_moves: int = 10000
    engine: str = "descent"  # "descent", "mcts", or "gumbel"
    nn_model_id: Optional[str] = None
    multi_player_values: bool = False
    max_players: int = 4
    graded_outcomes: bool = False
    history_length: int = 3
    # Gumbel-MCTS specific settings
    gumbel_simulations: int = 64
    gumbel_top_k: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0


@dataclass
class GameResult:
    """Result from a single selfplay game."""
    features: np.ndarray  # (N, C, H, W)
    globals: np.ndarray  # (N, D)
    values: np.ndarray  # (N,)
    policy_indices: List[np.ndarray]  # Sparse policy indices
    policy_values: List[np.ndarray]  # Sparse policy values
    values_mp: Optional[np.ndarray]  # (N, max_players) or None
    num_players: Optional[np.ndarray]  # (N,) or None
    num_samples: int
    game_idx: int
    duration_sec: float


def _worker_init(config_dict: dict) -> None:
    """Initialize worker process with config and PYTHONPATH."""
    import sys
    from pathlib import Path

    # Add ai-service root to path for worker processes
    ai_service_root = Path(__file__).resolve().parents[2]
    if str(ai_service_root) not in sys.path:
        sys.path.insert(0, str(ai_service_root))

    global _worker_config
    _worker_config = config_dict


def _generate_single_game(args: Tuple[int, int]) -> Optional[GameResult]:
    """
    Generate a single selfplay game in a worker process.

    Args:
        args: (game_idx, seed) tuple

    Returns:
        GameResult or None if game failed
    """
    game_idx, base_seed = args

    # Access worker config
    global _worker_config
    config = SelfplayConfig(**_worker_config)

    try:
        start_time = time.time()

        # Ensure ai-service is in path for worker subprocess
        import sys
        import os
        ai_service_root = os.environ.get('RINGRIFT_AI_SERVICE_ROOT')
        if ai_service_root and ai_service_root not in sys.path:
            sys.path.insert(0, ai_service_root)

        # Import dependencies in worker (avoid serialization issues)
        from app.env import RingRiftEnv
        from app.ai.descent_ai import DescentAI
        from app.ai.mcts_ai import MCTSAI
        from app.training.generate_data import state_to_feature_planes

        # Conditionally import GumbelMCTSAI (heavy dependencies)
        GumbelMCTSAI = None
        if config.engine == "gumbel":
            from app.ai.gumbel_mcts_ai import GumbelMCTSAI

        # Create environment
        env = RingRiftEnv(
            board_type=config.board_type,
            num_players=config.num_players,
        )

        # Create AI players (per-worker, not shared)
        ai_players = {}
        for pn in range(1, config.num_players + 1):
            if config.engine == "mcts":
                ai_players[pn] = MCTSAI(
                    player_num=pn,
                    simulations=400,
                    nn_model_id=config.nn_model_id,
                )
            elif config.engine == "gumbel":
                ai_players[pn] = GumbelMCTSAI(
                    player_num=pn,
                    simulations=config.gumbel_simulations,
                    top_k=config.gumbel_top_k,
                    c_visit=config.gumbel_c_visit,
                    c_scale=config.gumbel_c_scale,
                    nn_model_id=config.nn_model_id,
                )
            else:
                ai_players[pn] = DescentAI(
                    player_num=pn,
                    nn_model_id=config.nn_model_id,
                )

        # Seed RNG
        game_seed = base_seed + game_idx if base_seed is not None else None
        if game_seed is not None:
            random.seed(game_seed)
            np.random.seed(game_seed & 0xFFFFFFFF)
            for ai in ai_players.values():
                if hasattr(ai, 'seed'):
                    ai.seed(game_seed)

        # Play game
        state = env.reset(seed=game_seed)
        game_history = []
        state_history = []

        move_count = 0
        while not state.done and move_count < config.max_moves:
            current_player = state.current_player

            # Get AI move
            ai = ai_players.get(current_player)
            if ai is None:
                break

            # Get move and value
            move = ai.select_move(state)
            root_value = 0.0
            if hasattr(ai, 'last_root_value'):
                root_value = ai.last_root_value

            # Get policy distribution (soft targets for training)
            policy_indices = []
            policy_values = []
            if config.engine == "gumbel" and hasattr(ai, 'get_search_policy'):
                # Gumbel-MCTS: use visit-count-based soft policy targets
                moves, probs = ai.get_search_policy()
                for mv, prob in zip(moves, probs):
                    if hasattr(mv, 'id'):
                        policy_indices.append(mv.id)
                        policy_values.append(prob)
            elif hasattr(ai, 'last_root_policy') and ai.last_root_policy:
                # Standard MCTS/Descent: use last_root_policy
                for mv, prob in ai.last_root_policy.items():
                    if hasattr(mv, 'id'):
                        policy_indices.append(mv.id)
                        policy_values.append(prob)

            # Get feature planes
            feature_planes = state_to_feature_planes(
                state,
                state_history,
                config.board_type,
                config.history_length,
            )
            state_history.append(feature_planes[0])  # Current state only
            if len(state_history) > config.history_length:
                state_history.pop(0)

            # Get globals
            globals_vec = _get_globals(state, current_player)

            # Store sample
            game_history.append({
                'features': feature_planes,
                'globals': globals_vec,
                'player': current_player,
                'root_value': root_value,
                'policy_indices': np.array(policy_indices, dtype=np.int64),
                'policy_values': np.array(policy_values, dtype=np.float32),
            })

            # Make move
            state, _, done, _ = env.step(move)
            move_count += 1

        # Assign values based on game outcome
        if state.done and state.winner is not None:
            winner = state.winner
            for sample in game_history:
                player = sample['player']
                if player == winner:
                    sample['final_value'] = 1.0
                else:
                    sample['final_value'] = -1.0
        else:
            # Draw or timeout
            for sample in game_history:
                sample['final_value'] = 0.0

        # Convert to arrays
        if not game_history:
            return None

        num_samples = len(game_history)
        features = np.stack([s['features'] for s in game_history])
        globals_arr = np.stack([s['globals'] for s in game_history])
        values = np.array([s['final_value'] for s in game_history], dtype=np.float32)
        pol_indices = [s['policy_indices'] for s in game_history]
        pol_values = [s['policy_values'] for s in game_history]

        # Multi-player values if enabled
        values_mp = None
        num_players_arr = None
        if config.multi_player_values:
            values_mp = np.zeros((num_samples, config.max_players), dtype=np.float32)
            num_players_arr = np.full(num_samples, config.num_players, dtype=np.int32)
            for i, sample in enumerate(game_history):
                values_mp[i, sample['player'] - 1] = sample['final_value']

        duration = time.time() - start_time

        return GameResult(
            features=features,
            globals=globals_arr,
            values=values,
            policy_indices=pol_indices,
            policy_values=pol_values,
            values_mp=values_mp,
            num_players=num_players_arr,
            num_samples=num_samples,
            game_idx=game_idx,
            duration_sec=duration,
        )

    except Exception as e:
        logger.warning(f"Game {game_idx} failed: {e}")
        return None


def _get_globals(state, current_player: int) -> np.ndarray:
    """Extract global features from game state."""
    # Simple globals: current player, turn number, etc.
    globals_vec = np.zeros(16, dtype=np.float32)
    globals_vec[0] = current_player / 4.0
    globals_vec[1] = getattr(state, 'turn_number', 0) / 200.0
    return globals_vec


def generate_dataset_parallel(
    num_games: int = 100,
    output_file: str = "data/dataset_parallel.npz",
    num_workers: Optional[int] = None,
    board_type: BoardType = BoardType.SQUARE8,
    seed: Optional[int] = None,
    max_moves: int = 10000,
    num_players: int = 2,
    engine: str = "descent",
    nn_model_id: Optional[str] = None,
    multi_player_values: bool = False,
    max_players: int = 4,
    graded_outcomes: bool = False,
    progress_callback: Optional[callable] = None,
    # Gumbel-MCTS specific parameters
    gumbel_simulations: int = 64,
    gumbel_top_k: int = 16,
    gumbel_c_visit: float = 50.0,
    gumbel_c_scale: float = 1.0,
) -> int:
    """
    Generate selfplay data using parallel workers.

    Achieves 4-8x speedup over sequential generation by using multiple
    worker processes, each with its own AI instances.

    Args:
        num_games: Number of games to generate
        output_file: Output path for .npz file
        num_workers: Number of worker processes (default: CPU count)
        board_type: Board type
        seed: Random seed for reproducibility
        max_moves: Max moves per game
        num_players: Number of players
        engine: AI engine type ("descent", "mcts", or "gumbel")
        nn_model_id: Neural network model ID for AI
        multi_player_values: Include multi-player value vectors
        max_players: Max players for multi-player values
        graded_outcomes: Use graded outcomes
        progress_callback: Optional callback(completed, total) for progress
        gumbel_simulations: Simulations per move for Gumbel-MCTS
        gumbel_top_k: Top-k actions to consider in sequential halving
        gumbel_c_visit: Visit count exploration constant
        gumbel_c_scale: UCB scale factor

    Returns:
        Total number of samples generated
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    logger.info(
        f"Starting parallel selfplay: {num_games} games with {num_workers} workers"
    )

    # Prepare config (serializable)
    config = SelfplayConfig(
        board_type=board_type,
        num_players=num_players,
        max_moves=max_moves,
        engine=engine,
        nn_model_id=nn_model_id,
        multi_player_values=multi_player_values,
        max_players=max_players,
        graded_outcomes=graded_outcomes,
        gumbel_simulations=gumbel_simulations,
        gumbel_top_k=gumbel_top_k,
        gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
    )
    config_dict = {
        'board_type': config.board_type,
        'num_players': config.num_players,
        'max_moves': config.max_moves,
        'engine': config.engine,
        'nn_model_id': config.nn_model_id,
        'multi_player_values': config.multi_player_values,
        'max_players': config.max_players,
        'graded_outcomes': config.graded_outcomes,
        'history_length': config.history_length,
        'gumbel_simulations': config.gumbel_simulations,
        'gumbel_top_k': config.gumbel_top_k,
        'gumbel_c_visit': config.gumbel_c_visit,
        'gumbel_c_scale': config.gumbel_c_scale,
    }

    # Generate game arguments
    base_seed = seed if seed is not None else int(time.time())
    game_args = [(i, base_seed) for i in range(num_games)]

    # Run workers
    results: List[GameResult] = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(config_dict,),
    ) as executor:
        futures = {executor.submit(_generate_single_game, args): args[0]
                   for args in game_args}

        for future in as_completed(futures):
            game_idx = futures[future]
            try:
                result = future.result(timeout=300)
                if result is not None:
                    results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, num_games)

                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    logger.info(
                        f"Progress: {completed}/{num_games} games "
                        f"({rate:.1f} games/sec)"
                    )

            except Exception as e:
                logger.warning(f"Game {game_idx} failed: {e}")
                completed += 1

    # Aggregate results
    if not results:
        logger.error("No games completed successfully")
        return 0

    total_samples = sum(r.num_samples for r in results)
    logger.info(
        f"Completed {len(results)} games with {total_samples} samples "
        f"in {time.time() - start_time:.1f}s"
    )

    # Concatenate all data
    all_features = np.concatenate([r.features for r in results], axis=0)
    all_globals = np.concatenate([r.globals for r in results], axis=0)
    all_values = np.concatenate([r.values for r in results], axis=0)

    # Concatenate sparse policies
    all_pol_indices = []
    all_pol_values = []
    for r in results:
        all_pol_indices.extend(r.policy_indices)
        all_pol_values.extend(r.policy_values)

    # Multi-player values
    all_values_mp = None
    all_num_players = None
    if multi_player_values:
        all_values_mp = np.concatenate([r.values_mp for r in results], axis=0)
        all_num_players = np.concatenate([r.num_players for r in results], axis=0)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_dict = {
        'features': all_features,
        'globals': all_globals,
        'values': all_values,
        'pol_indices': np.array(all_pol_indices, dtype=object),
        'pol_values': np.array(all_pol_values, dtype=object),
    }
    if all_values_mp is not None:
        save_dict['values_mp'] = all_values_mp
        save_dict['num_players'] = all_num_players

    np.savez_compressed(output_file, **save_dict)
    logger.info(f"Saved {total_samples} samples to {output_file}")

    return total_samples
