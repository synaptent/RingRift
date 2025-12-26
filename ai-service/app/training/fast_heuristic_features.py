"""Heuristic feature extraction for V5/V6 neural network training.

This module provides two extraction modes for heuristic features:

1. **Fast mode (21 features)**: O(1) extraction using `_compute_component_scores`.
   Best for inference and high-throughput selfplay.

2. **Full mode (49 features)**: Uses `extract_linear_features` to decompose
   HeuristicAI into all 49 weight contributions. Best for maximum-strength
   training data.

The fast features capture aggregated strategic signals:
- Material: stack control, rings, eliminations, markers
- Positional: territory, center control, closure, safety
- Tactical: opponent threats, vulnerability, overtake potential
- Mobility: movement options, stack mobility
- Strategic: victory proximity, opponent threats, LPS advantage

The full features expose all 49 heuristic weight contributions including:
- Line patterns: two-in-row, three-in-row, four-in-row
- Swap evaluation: opening strength, corner penalties, etc.
- Recovery: buried ring value, recovery eligibility, etc.
- Move-level: safe move bonuses, blocked stack penalties

Usage:
    from app.training.fast_heuristic_features import (
        extract_heuristic_features,      # Fast 21-feature extraction
        extract_full_heuristic_features, # Full 49-feature extraction
        NUM_HEURISTIC_FEATURES,          # 21
        NUM_HEURISTIC_FEATURES_FULL,     # 49
    )

    # Fast extraction (O(1) per state)
    fast_feats = extract_heuristic_features(game_state, player_number=1)

    # Full extraction (O(50) per state, maximum information)
    full_feats = extract_full_heuristic_features(game_state, player_number=1)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import GameState

# Canonical ordered list of heuristic feature names
# These correspond to the keys returned by HeuristicAI._compute_component_scores()
HEURISTIC_FEATURE_NAMES: tuple[str, ...] = (
    # Tier 0 (core)
    "stack_control",
    "territory",
    "rings_in_hand",
    "center_control",
    "eliminated_rings",
    "marker_count",
    # Tier 1 (local/mobility)
    "opponent_threats",
    "mobility",
    "stack_mobility",
    # Tier 1 (strategic)
    "victory_proximity",
    "opponent_victory_threat",
    "multi_leader_threat",
    # Tier 2 (structural) - only computed in "full" mode
    "line_potential",
    "line_connectivity",
    "vulnerability",
    "overtake_potential",
    "territory_closure",
    "territory_safety",
    "forced_elimination_risk",
    "lps_action_advantage",
    "recovery_potential",
)

NUM_HEURISTIC_FEATURES = len(HEURISTIC_FEATURE_NAMES)
NUM_HEURISTIC_FEATURES_FAST = NUM_HEURISTIC_FEATURES  # Alias for clarity

# Full 49-feature extraction (all heuristic weight keys)
# Import lazily to avoid circular imports
_FULL_FEATURE_NAMES: tuple[str, ...] | None = None
NUM_HEURISTIC_FEATURES_FULL = 49  # Constant, matches HEURISTIC_WEIGHT_KEYS length


def _get_full_feature_names() -> tuple[str, ...]:
    """Get the ordered list of all 49 heuristic weight keys."""
    global _FULL_FEATURE_NAMES
    if _FULL_FEATURE_NAMES is None:
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS
        _FULL_FEATURE_NAMES = tuple(HEURISTIC_WEIGHT_KEYS)
    return _FULL_FEATURE_NAMES


# Default feature order for consistent indexing
_FEATURE_INDEX = {name: i for i, name in enumerate(HEURISTIC_FEATURE_NAMES)}


def extract_heuristic_features(
    game_state: "GameState",
    player_number: int,
    eval_mode: str = "full",
    normalize: bool = True,
) -> np.ndarray:
    """Extract heuristic features from a game state.

    This is a fast O(1) extraction that directly computes heuristic component
    scores without the 49x overhead of the linear feature decomposition.

    Args:
        game_state: The game state to evaluate
        player_number: Perspective player (1-indexed)
        eval_mode: Evaluation mode ("full" or "light")
        normalize: Whether to normalize features to reasonable ranges

    Returns:
        Feature vector of shape (NUM_HEURISTIC_FEATURES,) = (21,)
    """
    from app.ai.heuristic_ai import HeuristicAI
    from app.models import AIConfig

    # Create a lightweight HeuristicAI instance
    ai = HeuristicAI(
        player_number=player_number,
        config=AIConfig(difficulty=4, randomness=0.0),
    )
    ai.eval_mode = eval_mode

    # Get component scores (this is the efficient path)
    scores = ai._compute_component_scores(game_state)

    # Convert to ordered feature vector
    features = np.zeros(NUM_HEURISTIC_FEATURES, dtype=np.float32)
    for name, value in scores.items():
        if name in _FEATURE_INDEX:
            features[_FEATURE_INDEX[name]] = value

    if normalize:
        # Normalize to roughly [-1, 1] range based on typical value ranges
        # These are heuristic scaling factors based on observed ranges
        normalization_scales = {
            "stack_control": 100.0,
            "territory": 100.0,
            "rings_in_hand": 30.0,
            "center_control": 50.0,
            "eliminated_rings": 100.0,
            "marker_count": 50.0,
            "opponent_threats": 50.0,
            "mobility": 100.0,
            "stack_mobility": 50.0,
            "victory_proximity": 200.0,
            "opponent_victory_threat": 100.0,
            "multi_leader_threat": 50.0,
            "line_potential": 100.0,
            "line_connectivity": 50.0,
            "vulnerability": 100.0,
            "overtake_potential": 100.0,
            "territory_closure": 100.0,
            "territory_safety": 50.0,
            "forced_elimination_risk": 50.0,
            "lps_action_advantage": 30.0,
            "recovery_potential": 50.0,
        }
        for name, scale in normalization_scales.items():
            if name in _FEATURE_INDEX:
                idx = _FEATURE_INDEX[name]
                features[idx] = np.clip(features[idx] / scale, -1.0, 1.0)

    return features


def extract_heuristic_features_batch(
    game_states: list["GameState"],
    player_numbers: list[int],
    eval_mode: str = "full",
    normalize: bool = True,
) -> np.ndarray:
    """Batch extract heuristic features from multiple game states.

    Args:
        game_states: List of game states
        player_numbers: List of perspective players (1-indexed)
        eval_mode: Evaluation mode ("full" or "light")
        normalize: Whether to normalize features

    Returns:
        Feature array of shape (N, NUM_HEURISTIC_FEATURES)
    """
    if len(game_states) != len(player_numbers):
        raise ValueError(
            f"game_states and player_numbers must have same length, "
            f"got {len(game_states)} and {len(player_numbers)}"
        )

    if not game_states:
        return np.zeros((0, NUM_HEURISTIC_FEATURES), dtype=np.float32)

    features_list = []
    for state, player in zip(game_states, player_numbers):
        features = extract_heuristic_features(
            state, player, eval_mode=eval_mode, normalize=normalize
        )
        features_list.append(features)

    return np.stack(features_list, axis=0)


# Mapping from 49 HEURISTIC_WEIGHT_KEYS to fast feature indices
# For weights that don't have direct component mappings, we use -1
def get_weight_to_feature_mapping() -> dict[str, int]:
    """Get mapping from weight keys to fast feature indices.

    Some weights (like line pattern weights) don't have direct mappings
    because they're internal to the line_potential computation.

    Returns:
        Dict mapping weight key -> feature index (or -1 if no direct mapping)
    """
    from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS

    # Map weight keys to feature names (strip WEIGHT_ prefix, lowercase)
    mapping = {}
    for weight_key in HEURISTIC_WEIGHT_KEYS:
        # Convert WEIGHT_STACK_CONTROL -> stack_control
        feature_name = weight_key.replace("WEIGHT_", "").lower()

        # Handle special cases
        special_mappings = {
            "opponent_threat": "opponent_threats",
            "stack_height": "stack_control",  # Combined in stack_control
            "cap_height": "stack_control",    # Combined in stack_control
            "adjacency": "opponent_threats",  # Part of threat computation
        }
        feature_name = special_mappings.get(feature_name, feature_name)

        if feature_name in _FEATURE_INDEX:
            mapping[weight_key] = _FEATURE_INDEX[feature_name]
        else:
            mapping[weight_key] = -1  # No direct mapping

    return mapping


# =============================================================================
# Full 49-Feature Extraction (Maximum Strength)
# =============================================================================

# Normalization scales for full features (based on typical value ranges)
_FULL_NORMALIZATION_SCALES: dict[str, float] = {
    # Core material/positional
    "WEIGHT_STACK_CONTROL": 100.0,
    "WEIGHT_STACK_HEIGHT": 50.0,
    "WEIGHT_CAP_HEIGHT": 50.0,
    "WEIGHT_TERRITORY": 100.0,
    "WEIGHT_RINGS_IN_HAND": 30.0,
    "WEIGHT_CENTER_CONTROL": 50.0,
    "WEIGHT_ADJACENCY": 30.0,
    "WEIGHT_OPPONENT_THREAT": 50.0,
    "WEIGHT_MOBILITY": 100.0,
    "WEIGHT_ELIMINATED_RINGS": 100.0,
    "WEIGHT_LINE_POTENTIAL": 100.0,
    "WEIGHT_VICTORY_PROXIMITY": 200.0,
    "WEIGHT_MARKER_COUNT": 50.0,
    "WEIGHT_VULNERABILITY": 100.0,
    "WEIGHT_OVERTAKE_POTENTIAL": 100.0,
    "WEIGHT_TERRITORY_CLOSURE": 100.0,
    "WEIGHT_LINE_CONNECTIVITY": 50.0,
    "WEIGHT_TERRITORY_SAFETY": 50.0,
    "WEIGHT_STACK_MOBILITY": 50.0,
    # Strategic
    "WEIGHT_OPPONENT_VICTORY_THREAT": 100.0,
    "WEIGHT_FORCED_ELIMINATION_RISK": 50.0,
    "WEIGHT_LPS_ACTION_ADVANTAGE": 30.0,
    "WEIGHT_MULTI_LEADER_THREAT": 50.0,
    # Move-level penalties/bonuses
    "WEIGHT_NO_STACKS_PENALTY": 50.0,
    "WEIGHT_SINGLE_STACK_PENALTY": 20.0,
    "WEIGHT_STACK_DIVERSITY_BONUS": 10.0,
    "WEIGHT_SAFE_MOVE_BONUS": 10.0,
    "WEIGHT_NO_SAFE_MOVES_PENALTY": 10.0,
    "WEIGHT_VICTORY_THRESHOLD_BONUS": 1000.0,
    "WEIGHT_RINGS_PROXIMITY_FACTOR": 50.0,
    "WEIGHT_TERRITORY_PROXIMITY_FACTOR": 50.0,
    # Line patterns
    "WEIGHT_TWO_IN_ROW": 20.0,
    "WEIGHT_THREE_IN_ROW": 20.0,
    "WEIGHT_FOUR_IN_ROW": 20.0,
    "WEIGHT_CONNECTED_NEIGHBOR": 20.0,
    "WEIGHT_GAP_POTENTIAL": 10.0,
    "WEIGHT_BLOCKED_STACK_PENALTY": 20.0,
    # Swap evaluation
    "WEIGHT_SWAP_OPENING_CENTER": 20.0,
    "WEIGHT_SWAP_OPENING_ADJACENCY": 10.0,
    "WEIGHT_SWAP_OPENING_HEIGHT": 10.0,
    "WEIGHT_SWAP_CORNER_PENALTY": 10.0,
    "WEIGHT_SWAP_EDGE_BONUS": 10.0,
    "WEIGHT_SWAP_DIAGONAL_BONUS": 10.0,
    "WEIGHT_SWAP_OPENING_STRENGTH": 30.0,
    "WEIGHT_SWAP_EXPLORATION_TEMPERATURE": 1.0,
    # Recovery
    "WEIGHT_RECOVERY_POTENTIAL": 50.0,
    "WEIGHT_RECOVERY_ELIGIBILITY": 20.0,
    "WEIGHT_BURIED_RING_VALUE": 20.0,
    "WEIGHT_RECOVERY_THREAT": 30.0,
}


def extract_full_heuristic_features(
    game_state: "GameState",
    player_number: int,
    normalize: bool = True,
    include_bias: bool = False,
) -> np.ndarray:
    """Extract all 49 heuristic weight features from a game state.

    This uses `extract_linear_features` to decompose HeuristicAI into its
    full 49-weight representation. More expensive than fast extraction
    (O(50) evaluations vs O(1)) but provides maximum information for training.

    Args:
        game_state: The game state to evaluate
        player_number: Perspective player (1-indexed)
        normalize: Whether to normalize features to roughly [-1, 1]
        include_bias: If True, returns 50 features (49 weights + 1 bias)

    Returns:
        Feature vector of shape (49,) or (50,) if include_bias=True
    """
    from app.training.heuristic_features import extract_linear_features

    result = extract_linear_features(game_state, player_number)
    feature_names = _get_full_feature_names()

    # Features are already in order matching HEURISTIC_WEIGHT_KEYS
    features = result.features.copy()

    if normalize:
        for i, name in enumerate(feature_names):
            scale = _FULL_NORMALIZATION_SCALES.get(name, 100.0)
            features[i] = np.clip(features[i] / scale, -1.0, 1.0)

    if include_bias:
        bias = result.bias
        if normalize:
            bias = np.clip(bias / 1000.0, -1.0, 1.0)
        return np.append(features, bias).astype(np.float32)

    return features.astype(np.float32)


def extract_full_heuristic_features_batch(
    game_states: list["GameState"],
    player_numbers: list[int],
    normalize: bool = True,
    include_bias: bool = False,
) -> np.ndarray:
    """Batch extract all 49 heuristic features from multiple game states.

    Args:
        game_states: List of game states
        player_numbers: List of perspective players (1-indexed)
        normalize: Whether to normalize features
        include_bias: If True, returns 50 features per state

    Returns:
        Feature array of shape (N, 49) or (N, 50) if include_bias=True
    """
    if len(game_states) != len(player_numbers):
        raise ValueError(
            f"game_states and player_numbers must have same length, "
            f"got {len(game_states)} and {len(player_numbers)}"
        )

    if not game_states:
        dim = NUM_HEURISTIC_FEATURES_FULL + (1 if include_bias else 0)
        return np.zeros((0, dim), dtype=np.float32)

    features_list = []
    for state, player in zip(game_states, player_numbers):
        features = extract_full_heuristic_features(
            state, player, normalize=normalize, include_bias=include_bias
        )
        features_list.append(features)

    return np.stack(features_list, axis=0)


__all__ = [
    # Fast extraction (21 features)
    "extract_heuristic_features",
    "extract_heuristic_features_batch",
    "HEURISTIC_FEATURE_NAMES",
    "NUM_HEURISTIC_FEATURES",
    "NUM_HEURISTIC_FEATURES_FAST",
    # Full extraction (49 features)
    "extract_full_heuristic_features",
    "extract_full_heuristic_features_batch",
    "NUM_HEURISTIC_FEATURES_FULL",
    # Utilities
    "get_weight_to_feature_mapping",
]
