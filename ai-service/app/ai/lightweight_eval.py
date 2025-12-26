"""
Lightweight evaluation functions for fast make/unmake move evaluation.

These functions operate on LightweightState objects instead of Pydantic
GameState objects, avoiding Pydantic validation overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lightweight_state import LightweightState

# Pre-computed center positions for different board types
# These are the most valuable positions for center control
_CENTER_POSITIONS_SQUARE8 = frozenset([
    "3,3", "3,4", "4,3", "4,4",  # Inner center (2x2)
    "2,2", "2,3", "2,4", "2,5",  # Ring around center
    "3,2", "3,5", "4,2", "4,5",
    "5,2", "5,3", "5,4", "5,5",
])

_CENTER_POSITIONS_SQUARE19 = frozenset([
    f"{x},{y}" for x in range(7, 12) for y in range(7, 12)  # 5x5 center
])

# For hex board (axial coordinates), center is around (0,0)
_CENTER_POSITIONS_HEX = frozenset([
    "0,0",
    "1,0", "0,1", "-1,1", "-1,0", "0,-1", "1,-1",  # Ring 1
    "2,0", "1,1", "0,2", "-1,2", "-2,2", "-2,1",   # Ring 2
    "-2,0", "-1,-1", "0,-2", "1,-2", "2,-2", "2,-1",
])


def evaluate_stack_control_light(
    state: LightweightState,
    player_number: int,
    weight_stack_control: float,
    weight_no_stacks_penalty: float,
    weight_single_stack_penalty: float,
    weight_stack_diversity_bonus: float,
) -> float:
    """Evaluate stack control for lightweight state."""
    my_stacks = []
    opponent_stacks = []

    for stack in state.stacks.values():
        if stack.controlling_player == player_number:
            my_stacks.append(stack)
        else:
            opponent_stacks.append(stack)

    my_count = len(my_stacks)
    opp_count = len(opponent_stacks)

    # For multi-player games, average opponent count
    num_opponents = max(1, len(state.players) - 1) if hasattr(state, 'players') else 1
    opp_count_avg = opp_count / num_opponents

    # Base score: difference in stack counts (symmetric)
    score = (my_count - opp_count_avg) * weight_stack_control

    # Penalties for having too few stacks - SYMMETRIC
    # Compute my penalty
    my_penalty = 0.0
    if my_count == 0:
        my_penalty = weight_no_stacks_penalty
    elif my_count == 1:
        my_penalty = weight_single_stack_penalty

    # Compute opponent penalty (averaged)
    opp_penalty = 0.0
    if opp_count_avg < 0.5:  # Effectively 0 stacks
        opp_penalty = weight_no_stacks_penalty
    elif opp_count_avg < 1.5:  # Effectively 1 stack
        opp_penalty = weight_single_stack_penalty

    # Symmetric penalty: my_penalty - opp_penalty
    score += my_penalty - opp_penalty

    # Bonus for stack diversity - SYMMETRIC
    my_diversity = 0.0
    if my_count >= 2:
        # Simple diversity metric: count unique row/column pairs
        unique_positions = set()
        for stack in my_stacks:
            # Extract row/col from key "x,y"
            parts = stack.position_key.split(",")
            if len(parts) >= 2:
                row = int(parts[1]) // 2  # Group into regions
                col = int(parts[0]) // 2
                unique_positions.add((row, col))
        my_diversity = (len(unique_positions) / my_count) * weight_stack_diversity_bonus

    opp_diversity = 0.0
    if len(opponent_stacks) >= 2:
        unique_positions = set()
        for stack in opponent_stacks:
            parts = stack.position_key.split(",")
            if len(parts) >= 2:
                row = int(parts[1]) // 2
                col = int(parts[0]) // 2
                unique_positions.add((row, col))
        opp_diversity = (len(unique_positions) / len(opponent_stacks)) * weight_stack_diversity_bonus

    # Symmetric diversity advantage
    score += my_diversity - opp_diversity

    return score


def evaluate_territory_light(
    state: LightweightState,
    player_number: int,
    weight_territory: float,
) -> float:
    """Evaluate territory control for lightweight state."""
    my_territory = 0
    opp_territory = 0

    for owner in state.territories.values():
        if owner == player_number:
            my_territory += 1
        elif owner != 0:  # 0 typically means neutral/unclaimed
            opp_territory += 1

    return (my_territory - opp_territory) * weight_territory


def evaluate_rings_in_hand_light(
    state: LightweightState,
    player_number: int,
    weight_rings_in_hand: float,
) -> float:
    """Evaluate rings in hand for lightweight state."""
    player = state.players.get(player_number)
    if not player:
        return 0.0

    # Find opponent(s)
    opp_rings = 0
    opp_count = 0
    for pnum, p in state.players.items():
        if pnum != player_number:
            opp_rings += p.rings_in_hand
            opp_count += 1

    avg_opp_rings = opp_rings / max(1, opp_count)
    return (player.rings_in_hand - avg_opp_rings) * weight_rings_in_hand


def evaluate_center_control_light(
    state: LightweightState,
    player_number: int,
    weight_center_control: float,
) -> float:
    """Evaluate center control for lightweight state."""
    # Select center positions based on board type
    if state.board_type.value == "square8":
        center_positions = _CENTER_POSITIONS_SQUARE8
    elif state.board_type.value == "square19":
        center_positions = _CENTER_POSITIONS_SQUARE19
    else:  # hexagonal
        center_positions = _CENTER_POSITIONS_HEX

    my_center_stacks = 0
    opp_center_stacks = 0

    for key, stack in state.stacks.items():
        if key in center_positions:
            if stack.controlling_player == player_number:
                my_center_stacks += 1
            else:
                opp_center_stacks += 1

    return (my_center_stacks - opp_center_stacks) * weight_center_control


def evaluate_eliminated_rings_light(
    state: LightweightState,
    player_number: int,
    weight_eliminated_rings: float,
) -> float:
    """Evaluate eliminated rings for lightweight state (symmetric)."""
    player = state.players.get(player_number)
    if not player:
        return 0.0

    # Compute opponent average for symmetric evaluation
    opp_eliminated = 0
    opp_count = 0
    for pnum, p in state.players.items():
        if pnum != player_number:
            opp_eliminated += p.eliminated_rings
            opp_count += 1

    opp_eliminated_avg = opp_eliminated / max(1, opp_count)

    # Symmetric: positive means we've eliminated more than opponents
    return (player.eliminated_rings - opp_eliminated_avg) * weight_eliminated_rings


def evaluate_victory_proximity_light(
    state: LightweightState,
    player_number: int,
    weight_victory_proximity: float,
    weight_victory_threshold_bonus: float,
    weight_rings_proximity_factor: float,
    weight_territory_proximity_factor: float,
) -> float:
    """Evaluate how close player is to winning (symmetric vs opponents)."""
    player = state.players.get(player_number)
    if not player:
        return 0.0

    def compute_proximity(p) -> float:
        """Compute victory proximity for a player."""
        rings_needed = max(1, state.victory_rings - p.eliminated_rings)
        territory_needed = max(1, state.victory_territory - p.territory_spaces)

        score = 0.0
        # Near-victory bonus
        if rings_needed <= 3 or territory_needed <= 5:
            score += weight_victory_threshold_bonus

        # Proximity scoring (inverse of distance to victory)
        score += (1.0 / rings_needed) * weight_rings_proximity_factor
        score += (1.0 / territory_needed) * weight_territory_proximity_factor
        return score

    my_proximity = compute_proximity(player)

    # Compute max opponent proximity for symmetric evaluation
    max_opp_proximity = 0.0
    for pnum, p in state.players.items():
        if pnum != player_number:
            opp_prox = compute_proximity(p)
            max_opp_proximity = max(max_opp_proximity, opp_prox)

    # Symmetric: positive means we're closer to winning than opponents
    return (my_proximity - max_opp_proximity) * weight_victory_proximity


def evaluate_marker_count_light(
    state: LightweightState,
    player_number: int,
    weight_marker_count: float,
) -> float:
    """Evaluate marker count for lightweight state."""
    my_markers = 0
    opp_markers = 0

    for marker in state.markers.values():
        if marker.player == player_number:
            my_markers += 1
        else:
            opp_markers += 1

    return (my_markers - opp_markers) * weight_marker_count


def evaluate_mobility_light(
    state: LightweightState,
    player_number: int,
    weight_mobility: float,
    board_size: int,
) -> float:
    """Simplified mobility evaluation for lightweight state (symmetric).

    Full mobility calculation requires valid move generation which is expensive.
    This provides a fast approximation based on stack positions and board occupancy.
    """
    my_stacks = [s for s in state.stacks.values() if s.controlling_player == player_number]
    opp_stacks = [s for s in state.stacks.values() if s.controlling_player != player_number]

    # Approximate mobility: count empty adjacent cells around stacks
    occupied = set(state.stacks.keys()) | set(state.markers.keys())

    def compute_stack_mobility(stacks):
        if not stacks:
            return 0.0
        total = 0
        for stack in stacks:
            parts = stack.position_key.split(",")
            if len(parts) < 2:
                continue
            x, y = int(parts[0]), int(parts[1])

            # Check 8 directions for square boards
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    neighbor_key = f"{nx},{ny}"
                    if neighbor_key not in occupied:
                        total += 1
        return total / max(1, len(stacks))

    my_mobility = compute_stack_mobility(my_stacks)
    opp_mobility = compute_stack_mobility(opp_stacks)

    # Symmetric: positive means we have more mobility than opponents
    return (my_mobility - opp_mobility) * weight_mobility


def evaluate_position_light(
    state: LightweightState,
    player_number: int,
    weights: dict[str, float],
    eval_mode: str = "light",
) -> float:
    """
    Fast position evaluation using lightweight state.

    This is a streamlined version of HeuristicAI._compute_component_scores
    that operates on LightweightState instead of GameState.

    Args:
        state: LightweightState to evaluate
        player_number: Player to evaluate for
        weights: Dict of weight constants from HeuristicAI
        eval_mode: "light" for fast eval, "full" for complete eval

    Returns:
        Position score
    """
    score = 0.0

    # Core features (Tier 0) - always computed
    score += evaluate_stack_control_light(
        state, player_number,
        weights.get('WEIGHT_STACK_CONTROL', 10.0),
        weights.get('WEIGHT_NO_STACKS_PENALTY', -50.0),
        weights.get('WEIGHT_SINGLE_STACK_PENALTY', -10.0),
        weights.get('WEIGHT_STACK_DIVERSITY_BONUS', 5.0),
    )

    score += evaluate_territory_light(
        state, player_number,
        weights.get('WEIGHT_TERRITORY', 15.0),
    )

    score += evaluate_rings_in_hand_light(
        state, player_number,
        weights.get('WEIGHT_RINGS_IN_HAND', 3.0),
    )

    score += evaluate_center_control_light(
        state, player_number,
        weights.get('WEIGHT_CENTER_CONTROL', 8.0),
    )

    score += evaluate_eliminated_rings_light(
        state, player_number,
        weights.get('WEIGHT_ELIMINATED_RINGS', 20.0),
    )

    score += evaluate_victory_proximity_light(
        state, player_number,
        weights.get('WEIGHT_VICTORY_PROXIMITY', 25.0),
        weights.get('WEIGHT_VICTORY_THRESHOLD_BONUS', 30.0),
        weights.get('WEIGHT_RINGS_PROXIMITY_FACTOR', 10.0),
        weights.get('WEIGHT_TERRITORY_PROXIMITY_FACTOR', 5.0),
    )

    score += evaluate_marker_count_light(
        state, player_number,
        weights.get('WEIGHT_MARKER_COUNT', 2.0),
    )

    # Simplified mobility (Tier 1)
    score += evaluate_mobility_light(
        state, player_number,
        weights.get('WEIGHT_MOBILITY', 5.0),
        state.board_size,
    )

    return score


def extract_weights_from_ai(ai) -> dict[str, float]:
    """Extract weight constants from a HeuristicAI instance."""
    weights = {}
    for attr in dir(ai):
        if attr.startswith('WEIGHT_'):
            weights[attr] = getattr(ai, attr)
    return weights
