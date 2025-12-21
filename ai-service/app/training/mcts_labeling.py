"""MCTS Quality Labeling Utilities.

This module provides shared utilities for extracting quality labels
from MCTS search results, used by multiple data generation scripts.

Consolidates duplicated logic from:
- scripts/generate_mcts_data_56ch.py
- scripts/generate_search_labeled_data.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.ai.gumbel_mcts_ai import GumbelMCTSAI
    from app.models import Move


def build_move_key(move: Move) -> str:
    """Build a canonical key string for a move.

    This is the standard format used for move-to-index mapping
    in MCTS-based data generators.

    Args:
        move: A Move object with type, from_pos, and to attributes.

    Returns:
        A string key like "place_ring_3,4" or "move_1,2_3,4".
    """
    move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
    key = move_type

    if hasattr(move, "from_pos") and move.from_pos:
        key += f"_{move.from_pos.x},{move.from_pos.y}"
    if hasattr(move, "to") and move.to:
        key += f"_{move.to.x},{move.to.y}"

    return key


def build_move_to_index_map(valid_moves: list[Move]) -> dict[str, int]:
    """Build mapping from move key to index in valid_moves list.

    Args:
        valid_moves: List of valid moves in the current state.

    Returns:
        Dictionary mapping move keys to their indices.
    """
    return {build_move_key(move): i for i, move in enumerate(valid_moves)}


def extract_mcts_quality(
    mcts_ai: GumbelMCTSAI,
    valid_moves: list[Move],
) -> list[tuple[int, float]]:
    """Extract move quality labels from MCTS search results.

    Returns visit-fraction-based quality labels for each move
    that was explored during MCTS search.

    Args:
        mcts_ai: A GumbelMCTSAI instance after performing a search.
        valid_moves: List of valid moves in the current state.

    Returns:
        List of (move_index, visit_fraction) tuples, sorted by visit_fraction descending.
    """
    if mcts_ai._last_search_actions is None:
        return []

    total_visits = sum(a.visit_count for a in mcts_ai._last_search_actions)
    if total_visits == 0:
        return []

    move_to_idx = build_move_to_index_map(valid_moves)

    results = []
    for action in mcts_ai._last_search_actions:
        key = build_move_key(action.move)

        if key in move_to_idx:
            idx = move_to_idx[key]
            visit_fraction = action.visit_count / total_visits
            results.append((idx, visit_fraction))

    results.sort(key=lambda x: -x[1])
    return results


def extract_mcts_quality_with_values(
    mcts_ai: GumbelMCTSAI,
    valid_moves: list[Move],
) -> list[tuple[int, float, float]]:
    """Extract move quality labels with Q-values from MCTS search results.

    Like extract_mcts_quality but also includes Q-values for each move.

    Args:
        mcts_ai: A GumbelMCTSAI instance after performing a search.
        valid_moves: List of valid moves in the current state.

    Returns:
        List of (move_index, visit_fraction, q_value) tuples,
        sorted by visit_fraction descending.
    """
    if mcts_ai._last_search_actions is None:
        return []

    total_visits = sum(a.visit_count for a in mcts_ai._last_search_actions)
    if total_visits == 0:
        return []

    move_to_idx = build_move_to_index_map(valid_moves)

    results = []
    for action in mcts_ai._last_search_actions:
        key = build_move_key(action.move)

        if key in move_to_idx:
            idx = move_to_idx[key]
            visit_fraction = action.visit_count / total_visits
            q_value = action.value_sum / action.visit_count if action.visit_count > 0 else 0.0
            results.append((idx, visit_fraction, q_value))

    results.sort(key=lambda x: -x[1])
    return results


def get_best_move_index(
    mcts_quality: list[tuple[int, float, ...]],
) -> int | None:
    """Get the index of the best move (highest visit fraction).

    Args:
        mcts_quality: Output from extract_mcts_quality or extract_mcts_quality_with_values.

    Returns:
        Index of the best move, or None if no quality labels.
    """
    if not mcts_quality:
        return None
    return mcts_quality[0][0]


def compute_relative_scores(
    mcts_quality: list[tuple[int, float, ...]],
    num_moves: int,
) -> list[float]:
    """Compute relative scores for all moves based on MCTS visit fractions.

    Returns a score for each move index, where the best move gets 1.0
    and others get their visit_fraction / best_visit_fraction.

    Args:
        mcts_quality: Output from extract_mcts_quality.
        num_moves: Total number of valid moves.

    Returns:
        List of relative scores for each move index (0.0 for unexplored moves).
    """
    if not mcts_quality:
        return [0.0] * num_moves

    scores = [0.0] * num_moves
    best_fraction = mcts_quality[0][1]

    if best_fraction > 0:
        for move_idx, visit_fraction, *_ in mcts_quality:
            scores[move_idx] = visit_fraction / best_fraction

    return scores
