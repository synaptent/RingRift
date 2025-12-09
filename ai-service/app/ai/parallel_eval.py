"""
Parallel move evaluation using multiprocessing.

This module provides parallel evaluation of candidate moves across multiple
CPU cores, achieving near-linear speedup for move evaluation on large boards.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .batch_eval import BoardArrays

# Number of worker processes (default to CPU count - 1, min 1)
NUM_WORKERS = max(1, int(os.getenv('RINGRIFT_EVAL_WORKERS', str(max(1, mp.cpu_count() - 1)))))

# Minimum moves to trigger parallel evaluation
PARALLEL_THRESHOLD = int(os.getenv('RINGRIFT_PARALLEL_THRESHOLD', '50'))

# Global process pool (lazy initialized)
_process_pool: Optional[ProcessPoolExecutor] = None


def get_process_pool() -> ProcessPoolExecutor:
    """Get or create the global process pool."""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=NUM_WORKERS)
    return _process_pool


def shutdown_pool():
    """Shutdown the global process pool."""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=False)
        _process_pool = None


def evaluate_moves_parallel(
    state_data: Dict,
    move_data: List[Tuple],
    player_number: int,
    weights: Dict[str, float],
    num_workers: int = None,
) -> np.ndarray:
    """
    Evaluate moves in parallel across multiple processes.

    Args:
        state_data: Serialized lightweight state data
        move_data: List of (from_key, to_key, player, move_type) tuples
        player_number: Player to evaluate for
        weights: Weight dictionary
        num_workers: Number of workers (default: NUM_WORKERS)

    Returns:
        Array of scores for each move
    """
    if num_workers is None:
        num_workers = NUM_WORKERS

    num_moves = len(move_data)

    # For small number of moves, don't bother with parallelism
    if num_moves < PARALLEL_THRESHOLD or num_workers <= 1:
        return _evaluate_chunk(state_data, move_data, player_number, weights)

    # Split moves into chunks for each worker
    chunk_size = max(1, num_moves // num_workers)
    chunks = []
    for i in range(0, num_moves, chunk_size):
        chunks.append(move_data[i:i + chunk_size])

    # Use process pool for parallel evaluation
    pool = get_process_pool()

    # Submit all chunks
    futures = []
    for chunk in chunks:
        future = pool.submit(
            _evaluate_chunk,
            state_data,
            chunk,
            player_number,
            weights,
        )
        futures.append(future)

    # Collect results
    all_scores = []
    for future in futures:
        chunk_scores = future.result()
        all_scores.extend(chunk_scores)

    return np.array(all_scores, dtype=np.float64)


def _evaluate_chunk(
    state_data: Dict,
    move_data: List[Tuple],
    player_number: int,
    weights: Dict[str, float],
) -> List[float]:
    """
    Evaluate a chunk of moves (runs in worker process).

    This function is designed to be pickle-friendly and run in a separate process.
    """
    # Import here to avoid circular imports in worker processes
    from .batch_eval import batch_evaluate_positions

    # Reconstruct BoardArrays from serialized data
    arrays = _reconstruct_board_arrays(state_data)

    # Evaluate all moves in this chunk
    scores = batch_evaluate_positions(
        arrays,
        move_data,
        player_number,
        weights,
    )

    return scores.tolist()


def serialize_state_for_parallel(arrays: 'BoardArrays') -> Dict:
    """
    Serialize BoardArrays for passing to worker processes.

    Returns a dict that can be pickled efficiently.
    """
    return {
        'board_type': arrays.board_type,
        'board_size': arrays.board_size,
        'num_positions': arrays.num_positions,
        'position_to_idx': arrays.position_to_idx,
        'idx_to_position': arrays.idx_to_position,
        'stack_owner': arrays.stack_owner.tobytes(),
        'stack_height': arrays.stack_height.tobytes(),
        'marker_owner': arrays.marker_owner.tobytes(),
        'is_collapsed': arrays.is_collapsed.tobytes(),
        'territory_owner': arrays.territory_owner.tobytes(),
        'player_rings_in_hand': arrays.player_rings_in_hand.tobytes(),
        'player_eliminated': arrays.player_eliminated.tobytes(),
        'player_territory': arrays.player_territory.tobytes(),
        'center_mask': arrays.center_mask.tobytes(),
        'neighbors': arrays.neighbors.tobytes(),
        'num_neighbors': arrays.num_neighbors.tobytes(),
        'victory_rings': arrays.victory_rings,
        'victory_territory': arrays.victory_territory,
    }


def _reconstruct_board_arrays(data: Dict) -> 'BoardArrays':
    """Reconstruct BoardArrays from serialized data."""
    from .batch_eval import BoardArrays

    # Create empty arrays object
    arrays = BoardArrays.__new__(BoardArrays)
    arrays.board_type = data['board_type']
    arrays.board_size = data['board_size']
    arrays.num_positions = data['num_positions']
    arrays.position_to_idx = data['position_to_idx']
    arrays.idx_to_position = data['idx_to_position']
    arrays.victory_rings = data['victory_rings']
    arrays.victory_territory = data['victory_territory']

    # Reconstruct numpy arrays from bytes
    arrays.stack_owner = np.frombuffer(data['stack_owner'], dtype=np.int8).copy()
    arrays.stack_height = np.frombuffer(data['stack_height'], dtype=np.int8).copy()
    arrays.marker_owner = np.frombuffer(data['marker_owner'], dtype=np.int8).copy()
    arrays.is_collapsed = np.frombuffer(data['is_collapsed'], dtype=np.bool_).copy()
    arrays.territory_owner = np.frombuffer(data['territory_owner'], dtype=np.int8).copy()
    arrays.player_rings_in_hand = np.frombuffer(data['player_rings_in_hand'], dtype=np.int16).copy()
    arrays.player_eliminated = np.frombuffer(data['player_eliminated'], dtype=np.int16).copy()
    arrays.player_territory = np.frombuffer(data['player_territory'], dtype=np.int16).copy()
    arrays.center_mask = np.frombuffer(data['center_mask'], dtype=np.bool_).copy()

    # Reconstruct 2D arrays
    max_neighbors = 8 if arrays.board_type != 2 else 6  # BOARD_HEX = 2
    arrays.neighbors = np.frombuffer(
        data['neighbors'], dtype=np.int32
    ).copy().reshape(arrays.num_positions, max_neighbors)
    arrays.num_neighbors = np.frombuffer(data['num_neighbors'], dtype=np.int8).copy()

    return arrays


# Thread-based parallel evaluation (for when multiprocessing overhead is too high)
def evaluate_moves_threaded(
    arrays: 'BoardArrays',
    move_data: List[Tuple],
    player_number: int,
    weights: Dict[str, float],
    num_threads: int = None,
) -> np.ndarray:
    """
    Evaluate moves using thread pool (less overhead than processes).

    Note: Due to Python GIL, this is only effective if the evaluation
    code releases the GIL (e.g., numpy operations).
    """
    from .batch_eval import batch_evaluate_positions

    if num_threads is None:
        num_threads = NUM_WORKERS

    num_moves = len(move_data)

    if num_moves < PARALLEL_THRESHOLD or num_threads <= 1:
        return batch_evaluate_positions(arrays, move_data, player_number, weights)

    # Split into chunks
    chunk_size = max(1, num_moves // num_threads)
    chunks = []
    for i in range(0, num_moves, chunk_size):
        chunks.append(move_data[i:i + chunk_size])

    # Evaluate in thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                batch_evaluate_positions,
                arrays,
                chunk,
                player_number,
                weights,
            )
            for chunk in chunks
        ]

        all_scores = []
        for future in futures:
            chunk_scores = future.result()
            all_scores.extend(chunk_scores.tolist())

    return np.array(all_scores, dtype=np.float64)
