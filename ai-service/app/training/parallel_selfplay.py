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
from typing import Any

import numpy as np

from app.models import BoardType
from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

# Worker process global configuration (set by _init_worker)
_worker_config: dict[str, Any] = {}


@dataclass
class ParallelSelfplayConfig:
    """Configuration for parallel selfplay game generation workers.

    This is the internal config for the parallel worker pool.
    For unified per-run configuration, see :class:`app.training.selfplay_config.SelfplayConfig`.
    """
    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2
    max_moves: int = 10000
    engine: str = "descent"  # "descent", "mcts", "gumbel", or "ebmo"
    nn_model_id: str | None = None
    multi_player_values: bool = False
    max_players: int = 4
    graded_outcomes: bool = False
    history_length: int = 3
    feature_version: int = 1
    # Gumbel-MCTS specific settings
    gumbel_simulations: int = 64
    gumbel_top_k: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    # Temperature scheduling for exploration/exploitation
    temperature: float = 1.0  # Move selection temperature
    use_temperature_decay: bool = False  # Enable temperature decay per game
    move_temp_threshold: int = 30  # Use higher temp for first N moves
    opening_temperature: float = 1.5  # Temperature for opening moves


# Backward compatibility alias (December 2025)
# This config is for parallel worker pool internals.
# For unified per-run configuration, use app.training.selfplay_config.SelfplayConfig
SelfplayConfig = ParallelSelfplayConfig


@dataclass
class GameResult:
    """Result from a single selfplay game."""
    features: np.ndarray  # (N, C, H, W)
    globals: np.ndarray  # (N, D)
    values: np.ndarray  # (N,)
    policy_indices: list[np.ndarray]  # Sparse policy indices
    policy_values: list[np.ndarray]  # Sparse policy values
    values_mp: np.ndarray | None  # (N, max_players) or None
    num_players: np.ndarray | None  # (N,) or None
    num_samples: int
    game_idx: int
    duration_sec: float
    effective_temps: np.ndarray | None = None  # (N,) Per-sample effective temperature
    # Auxiliary task targets (2025-12)
    game_lengths: np.ndarray | None = None  # (N,) Total game length (same for all samples in game)
    piece_counts: np.ndarray | None = None  # (N,) Piece count at each sample
    outcomes: np.ndarray | None = None  # (N,) Outcome class: 0=loss, 1=draw, 2=win


def _worker_init(config_dict: dict) -> None:
    """Initialize worker process with config and PYTHONPATH."""
    import sys

    # Add ai-service root to path for worker processes (passed from main process)
    ai_service_root = config_dict.get('_ai_service_root')
    if ai_service_root and ai_service_root not in sys.path:
        sys.path.insert(0, ai_service_root)

    _worker_config = config_dict


def _generate_single_game(args: tuple[int, int]) -> GameResult | None:
    """
    Generate a single selfplay game in a worker process.

    Args:
        args: (game_idx, seed) tuple

    Returns:
        GameResult or None if game failed
    """
    game_idx, base_seed = args

    # Access worker config
    # Filter out internal keys before creating SelfplayConfig
    config_fields = {k: v for k, v in _worker_config.items() if not k.startswith('_')}
    config = SelfplayConfig(**config_fields)

    try:
        start_time = time.time()

        # Ensure ai-service is in path for worker subprocess (use path from config)
        import sys
        ai_service_root = _worker_config.get('_ai_service_root')
        if ai_service_root and ai_service_root not in sys.path:
            sys.path.insert(0, ai_service_root)

        # Import dependencies in worker (avoid serialization issues)
        from app.ai.descent_ai import DescentAI
        from app.ai.mcts_ai import MCTSAI
        from app.ai.neural_net import encode_move_for_board
        from app.ai.nnue import FEATURE_PLANES, extract_features_from_gamestate, get_board_size
        from app.models import AIConfig
        from app.training.env import RingRiftEnv

        # Conditionally import GumbelMCTSAI (heavy dependencies)
        GumbelMCTSAI = None
        if config.engine == "gumbel":
            from app.ai.gumbel_mcts_ai import GumbelMCTSAI

        # Conditionally import EBMO_AI
        EBMO_AI = None
        if config.engine == "ebmo":
            from app.ai.ebmo_ai import EBMO_AI

        # Create environment
        env = RingRiftEnv(
            board_type=config.board_type,
            num_players=config.num_players,
        )

        # Create AI players (per-worker, not shared)
        ai_players = {}
        for pn in range(1, config.num_players + 1):
            # Create AIConfig for this player
            ai_config = AIConfig(
                difficulty=5,  # Mid-range difficulty for selfplay
                nn_model_id=config.nn_model_id,
                self_play=True,  # Enable exploration for selfplay
                use_neural_net=config.nn_model_id is not None,
            )

            if config.engine == "mcts":
                ai_players[pn] = MCTSAI(
                    player_number=pn,
                    config=ai_config,
                )
            elif config.engine == "gumbel":
                # Gumbel-MCTS needs board_type for move encoding
                ai_players[pn] = GumbelMCTSAI(
                    player_number=pn,
                    config=ai_config,
                    board_type=config.board_type,
                )
            elif config.engine == "ebmo":
                # EBMO uses gradient descent on action embeddings
                ai_players[pn] = EBMO_AI(
                    player_number=pn,
                    config=ai_config,
                )
            else:
                ai_players[pn] = DescentAI(
                    player_number=pn,
                    config=ai_config,
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
        done = False

        move_count = 0
        while not done and move_count < config.max_moves:
            current_player = state.current_player

            # Get AI move
            ai = ai_players.get(current_player)
            if ai is None:
                break

            # Calculate temperature for this move (higher early in game for exploration)
            if config.use_temperature_decay:
                if move_count < config.move_temp_threshold:
                    # Interpolate from opening to standard temperature
                    progress = move_count / config.move_temp_threshold
                    temp = config.opening_temperature * (1 - progress) + config.temperature * progress
                else:
                    temp = config.temperature
            else:
                temp = config.temperature

            # Apply temperature to AI if supported
            if hasattr(ai, 'temperature'):
                ai.temperature = temp
            if hasattr(ai, 'config') and hasattr(ai.config, 'temperature'):
                ai.config.temperature = temp

            # Get move and value
            move = ai.select_move(state)
            root_value = 0.0
            if hasattr(ai, 'last_root_value'):
                root_value = ai.last_root_value

            # Get policy distribution (soft targets for training)
            policy_indices = []
            policy_values = []
            if config.engine == "gumbel" and hasattr(ai, 'get_visit_distribution'):
                # Gumbel-MCTS: use visit-count-based soft policy targets
                moves, probs = ai.get_visit_distribution()
                for mv, prob in zip(moves, probs, strict=False):
                    idx = encode_move_for_board(mv, state.board)
                    if idx >= 0:  # Valid move encoding
                        policy_indices.append(idx)
                        policy_values.append(prob)
            elif hasattr(ai, 'last_root_policy') and ai.last_root_policy:
                # Standard MCTS/Descent: use last_root_policy
                for mv, prob in ai.last_root_policy.items():
                    idx = encode_move_for_board(mv, state.board)
                    if idx >= 0:  # Valid move encoding
                        policy_indices.append(idx)
                        policy_values.append(prob)

            # Get feature planes using NNUE feature extraction
            # Extract 1D features and reshape to (C, H, W)
            features_1d = extract_features_from_gamestate(state, current_player)
            board_size = get_board_size(config.board_type)
            num_channels = FEATURE_PLANES  # 12 planes
            current_features = features_1d.reshape(num_channels, board_size, board_size)

            # Build stacked features with history
            hist_frames = list(state_history[-config.history_length:])
            while len(hist_frames) < config.history_length:
                hist_frames.insert(0, np.zeros_like(current_features))
            stacked_features = np.concatenate([current_features, *hist_frames], axis=0)

            # Update history
            state_history.append(current_features.copy())
            if len(state_history) > config.history_length:
                state_history.pop(0)

            # Get globals
            globals_vec = _get_globals(state, current_player)

            # Count total pieces on board for auxiliary task
            piece_count = _count_pieces(state)

            # Store sample (including effective temperature used for this move)
            game_history.append({
                'features': stacked_features,
                'globals': globals_vec,
                'player': current_player,
                'root_value': root_value,
                'policy_indices': np.array(policy_indices, dtype=np.int64),
                'policy_values': np.array(policy_values, dtype=np.float32),
                'effective_temp': temp,  # Track temperature used for this sample
                'piece_count': piece_count,  # Auxiliary task target
                'move_number': move_count,  # For game_length target
            })

            # Make move
            state, _, done, _ = env.step(move)
            move_count += 1

        # Assign values based on game outcome
        if done and state.winner is not None:
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
        effective_temps = np.array([s['effective_temp'] for s in game_history], dtype=np.float32)

        # Multi-player values if enabled
        values_mp = None
        num_players_arr = None
        if config.multi_player_values:
            values_mp = np.zeros((num_samples, config.max_players), dtype=np.float32)
            num_players_arr = np.full(num_samples, config.num_players, dtype=np.int32)
            for i, sample in enumerate(game_history):
                values_mp[i, sample['player'] - 1] = sample['final_value']  # type: ignore[call-overload]

        # Auxiliary task targets (2025-12)
        # Game length: total moves in the game (same for all samples)
        game_length = move_count
        game_lengths = np.full(num_samples, game_length, dtype=np.int32)

        # Piece counts: number of pieces at each sample
        piece_counts = np.array([s['piece_count'] for s in game_history], dtype=np.int32)

        # Outcomes: convert values to class labels (0=loss, 1=draw, 2=win)
        outcomes = np.zeros(num_samples, dtype=np.int64)
        for i, sample in enumerate(game_history):
            val = sample['final_value']
            if val > 0.3:
                outcomes[i] = 2  # Win
            elif val < -0.3:
                outcomes[i] = 0  # Loss
            else:
                outcomes[i] = 1  # Draw

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
            effective_temps=effective_temps,
            game_lengths=game_lengths,
            piece_counts=piece_counts,
            outcomes=outcomes,
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


def _count_pieces(state) -> int:
    """Count total pieces (rings) on the board."""
    total = 0
    if hasattr(state, 'board') and hasattr(state.board, 'stacks'):
        for stack in state.board.stacks.values():
            if hasattr(stack, 'rings'):
                total += len(stack.rings)
            elif isinstance(stack, (list, tuple)):
                total += len(stack)
    return total


def generate_dataset_parallel(
    num_games: int = 100,
    output_file: str = "data/dataset_parallel.npz",
    num_workers: int | None = None,
    board_type: BoardType = BoardType.SQUARE8,
    seed: int | None = None,
    max_moves: int = 10000,
    num_players: int = 2,
    history_length: int = 3,
    feature_version: int = 1,
    engine: str = "descent",
    nn_model_id: str | None = None,
    multi_player_values: bool = False,
    max_players: int = 4,
    graded_outcomes: bool = False,
    progress_callback: callable | None = None,
    # Gumbel-MCTS specific parameters
    gumbel_simulations: int = 64,
    gumbel_top_k: int = 16,
    gumbel_c_visit: float = 50.0,
    gumbel_c_scale: float = 1.0,
    # Temperature scheduling parameters
    temperature: float = 1.0,
    use_temperature_decay: bool = False,
    opening_temperature: float = 1.5,
    move_temp_threshold: int = 30,
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
        engine: AI engine type ("descent", "mcts", "gumbel", or "ebmo")
        history_length: Number of history frames to stack in features
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
        history_length=history_length,
        feature_version=feature_version,
        gumbel_simulations=gumbel_simulations,
        gumbel_top_k=gumbel_top_k,
        gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
        temperature=temperature,
        use_temperature_decay=use_temperature_decay,
        move_temp_threshold=move_temp_threshold,
        opening_temperature=opening_temperature,
    )
    # Get ai-service root path to pass to workers
    ai_service_root = str(AI_SERVICE_ROOT)

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
        'temperature': config.temperature,
        'use_temperature_decay': config.use_temperature_decay,
        'move_temp_threshold': config.move_temp_threshold,
        'opening_temperature': config.opening_temperature,
        '_ai_service_root': ai_service_root,  # Path for worker processes
    }

    # Generate game arguments
    base_seed = seed if seed is not None else int(time.time())
    game_args = [(i, base_seed) for i in range(num_games)]

    # Run workers
    results: list[GameResult] = []
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

                # Enhanced progress logging with ETA
                progress_interval = max(1, min(50, num_games // 20))
                if completed % progress_interval == 0 or completed == num_games:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = num_games - completed
                    eta_seconds = remaining / rate if rate > 0 else 0
                    pct = completed / num_games * 100
                    samples_so_far = sum(r.num_samples for r in results)
                    logger.info(
                        "[parallel-selfplay] Game %d/%d (%.1f%%) | %.2f games/s | "
                        "ETA: %.0fs | %d samples so far",
                        completed,
                        num_games,
                        pct,
                        rate,
                        eta_seconds,
                        samples_so_far,
                    )

            except Exception as e:
                logger.warning(f"Game {game_idx} failed: {e}")
                completed += 1

    # Aggregate results
    if not results:
        logger.error("No games completed successfully")
        return 0

    total_samples = sum(r.num_samples for r in results)
    elapsed_total = time.time() - start_time
    logger.info(
        "[parallel-selfplay] Completed %d/%d games with %d samples in %.1fs (%.2f games/s)",
        len(results),
        num_games,
        total_samples,
        elapsed_total,
        len(results) / elapsed_total if elapsed_total > 0 else 0,
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

    # Concatenate per-sample effective temperatures
    all_effective_temps = np.concatenate([r.effective_temps for r in results], axis=0)

    # Concatenate auxiliary task targets (2025-12)
    all_game_lengths = np.concatenate([r.game_lengths for r in results], axis=0)
    all_piece_counts = np.concatenate([r.piece_counts for r in results], axis=0)
    all_outcomes = np.concatenate([r.outcomes for r in results], axis=0)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_dict = {
        'features': all_features,
        'globals': all_globals,
        'values': all_values,
        'pol_indices': np.array(all_pol_indices, dtype=object),
        'pol_values': np.array(all_pol_values, dtype=object),
        'policy_encoding': np.asarray("board_aware"),
        'history_length': np.asarray(int(config.history_length)),
        'feature_version': np.asarray(int(config.feature_version)),
        # Per-sample effective temperature (for temperature-aware training)
        'effective_temps': all_effective_temps,
        # Temperature config metadata (for reproducibility/analysis)
        'temp_config': np.array([
            config.temperature,
            config.opening_temperature,
            float(config.move_temp_threshold),
            float(config.use_temperature_decay),
        ], dtype=np.float32),
        # Auxiliary task targets (2025-12)
        'game_lengths': all_game_lengths,
        'piece_counts': all_piece_counts,
        'outcomes': all_outcomes,
    }
    if all_values_mp is not None:
        save_dict['values_mp'] = all_values_mp
        save_dict['num_players'] = all_num_players

    np.savez_compressed(output_file, **save_dict)
    logger.info(f"Saved {total_samples} samples to {output_file}")

    return total_samples
