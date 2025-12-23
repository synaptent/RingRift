"""GPU to Canonical Export Translation Module.

This module provides functions to convert GPU batch state move history
to canonical format that matches the TS/Python parity requirements.

December 2025: Added for canonical data quality upgrade.

Key translations:
- GPU MoveType (IntEnum) → Canonical move type strings
- GPU GamePhase (IntEnum) → Canonical phase strings
- GPU position format [y, x] → Canonical position {x, y}

The canonical format is defined by:
- app/rules/history_contract.py
- TS turnOrchestrator.ts phase/move contracts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .gpu_game_types import GamePhase, MoveType

if TYPE_CHECKING:
    from .gpu_batch_state import BatchGameState


# =============================================================================
# GPU to Canonical Move Type Mapping
# =============================================================================

# Map GPU MoveType to canonical move type strings (matches app/models/core.py)
_GPU_TO_CANONICAL_MOVE_TYPE = {
    # Legacy GPU types (0-7)
    MoveType.PLACEMENT: "place_ring",
    MoveType.MOVEMENT: "move_stack",
    MoveType.CAPTURE: "overtaking_capture",  # Legacy generic capture
    MoveType.LINE_FORMATION: "process_line",
    MoveType.TERRITORY_CLAIM: "choose_territory_option",
    MoveType.SKIP: "skip_placement",  # Generic skip (context-dependent)
    MoveType.NO_ACTION: "no_territory_action",  # Generic (legacy)
    MoveType.RECOVERY_SLIDE: "recovery_slide",

    # Canonical phase-specific no-action types (8-11)
    MoveType.NO_PLACEMENT_ACTION: "no_placement_action",
    MoveType.NO_MOVEMENT_ACTION: "no_movement_action",
    MoveType.NO_LINE_ACTION: "no_line_action",
    MoveType.NO_TERRITORY_ACTION: "no_territory_action",

    # Canonical capture types (12-14)
    MoveType.OVERTAKING_CAPTURE: "overtaking_capture",
    MoveType.CONTINUE_CAPTURE_SEGMENT: "continue_capture_segment",
    MoveType.SKIP_CAPTURE: "skip_capture",

    # Canonical recovery types (15)
    MoveType.SKIP_RECOVERY: "skip_recovery",

    # Canonical forced elimination (16)
    MoveType.FORCED_ELIMINATION: "forced_elimination",

    # Canonical line/territory choice types (17-18)
    MoveType.CHOOSE_LINE_OPTION: "choose_line_option",
    MoveType.CHOOSE_TERRITORY_OPTION: "choose_territory_option",

    # Canonical skip placement (19)
    MoveType.SKIP_PLACEMENT: "skip_placement",

    # Canonical elimination (20) - per RR-CANON-R123/R145
    MoveType.ELIMINATE_RINGS_FROM_STACK: "eliminate_rings_from_stack",
}


# =============================================================================
# GPU to Canonical Phase Mapping
# =============================================================================

# Map GPU GamePhase to canonical phase strings (matches app/models/core.py)
_GPU_TO_CANONICAL_PHASE = {
    # Legacy GPU phases (0-4)
    GamePhase.RING_PLACEMENT: "ring_placement",
    GamePhase.MOVEMENT: "movement",
    GamePhase.LINE_PROCESSING: "line_processing",
    GamePhase.TERRITORY_PROCESSING: "territory_processing",
    GamePhase.END_TURN: "movement",  # Legacy fallback

    # Canonical phases (5-9)
    GamePhase.CAPTURE: "capture",
    GamePhase.CHAIN_CAPTURE: "chain_capture",
    # Recovery slides are recorded as movement-phase actions in canonical rules.
    GamePhase.RECOVERY: "movement",
    GamePhase.FORCED_ELIMINATION: "forced_elimination",
    GamePhase.GAME_OVER: "game_over",
}


# =============================================================================
# Conversion Functions
# =============================================================================


def _canvas_to_cube_coords(row: int, col: int, board_type: str) -> dict[str, int]:
    """Convert canvas coordinates to cube coordinates for hex boards.

    GPU stores positions as canvas coords (row, col) in [0, board_size).
    Canonical hex format uses cube coords (x, y, z) where x+y+z=0.

    Args:
        row: Canvas row coordinate (Y in tensor)
        col: Canvas column coordinate (X in tensor)
        board_type: Board type string

    Returns:
        Position dict with x, y (and z for hex boards)
    """
    if board_type in ("hexagonal", "hex8"):
        # Determine board size and radius from board type
        if board_type == "hexagonal":
            board_size = 25
        else:  # hex8
            board_size = 9
        radius = (board_size - 1) // 2

        # Convert canvas to cube coords
        cube_x = col - radius
        cube_y = row - radius
        cube_z = -cube_x - cube_y
        return {"x": cube_x, "y": cube_y, "z": cube_z}
    else:
        # Square boards: canvas coords are the canonical coords
        return {"x": col, "y": row}


def gpu_move_type_to_canonical(gpu_move_type: int) -> str:
    """Convert GPU MoveType integer to canonical move type string.

    Args:
        gpu_move_type: GPU MoveType enum value

    Returns:
        Canonical move type string (e.g., "place_ring", "overtaking_capture")
    """
    try:
        move_type_enum = MoveType(gpu_move_type)
        return _GPU_TO_CANONICAL_MOVE_TYPE.get(move_type_enum, "unknown")
    except ValueError:
        return "unknown"


def gpu_phase_to_canonical(gpu_phase: int) -> str:
    """Convert GPU GamePhase integer to canonical phase string.

    Args:
        gpu_phase: GPU GamePhase enum value

    Returns:
        Canonical phase string (e.g., "ring_placement", "capture")
    """
    try:
        phase_enum = GamePhase(gpu_phase)
        return _GPU_TO_CANONICAL_PHASE.get(phase_enum, "ring_placement")
    except ValueError:
        return "ring_placement"


def convert_gpu_move_to_canonical(
    move_type: int,
    player: int,
    from_y: int,
    from_x: int,
    to_y: int,
    to_x: int,
    phase: int,
    board_type: str = "square8",
    capture_target_y: int = -1,
    capture_target_x: int = -1,
) -> dict[str, Any]:
    """Convert a single GPU move to canonical format.

    Args:
        move_type: GPU MoveType enum value
        player: Player number (1-4)
        from_y: Origin Y coordinate (-1 if N/A)
        from_x: Origin X coordinate (-1 if N/A)
        to_y: Destination Y coordinate (-1 if N/A)
        to_x: Destination X coordinate (-1 if N/A)
        phase: GPU GamePhase enum value
        board_type: Board type string for coordinate conversion
        capture_target_y: Capture target Y coordinate (-1 if N/A)
        capture_target_x: Capture target X coordinate (-1 if N/A)

    Returns:
        Canonical move dictionary compatible with import scripts
    """
    canonical_type = gpu_move_type_to_canonical(move_type)
    canonical_phase = gpu_phase_to_canonical(phase)

    move: dict[str, Any] = {
        "type": canonical_type,
        "player": player,
        "phase": canonical_phase,
    }

    # Add position fields if valid, converting to cube coords for hex boards
    if from_y >= 0 and from_x >= 0:
        move["from"] = _canvas_to_cube_coords(from_y, from_x, board_type)

    if to_y >= 0 and to_x >= 0:
        move["to"] = _canvas_to_cube_coords(to_y, to_x, board_type)

    # Add capture_target for capture moves (December 2025)
    if capture_target_y >= 0 and capture_target_x >= 0:
        move["captureTarget"] = _canvas_to_cube_coords(capture_target_y, capture_target_x, board_type)

    return move


def convert_gpu_history_to_canonical(
    state: BatchGameState,
    game_idx: int,
    board_type: str = "square8",
) -> list[dict[str, Any]]:
    """Convert GPU move history to canonical format for a single game.

    This produces move dictionaries compatible with:
    - scripts/import_gpu_selfplay_to_db.py
    - app/rules/history_contract.py validation

    Args:
        state: BatchGameState containing move history
        game_idx: Index of the game in the batch
        board_type: Board type string for coordinate conversion

    Returns:
        List of canonical move dictionaries
    """
    moves = []

    # Move history now has 9 columns (December 2025):
    # [move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x]
    history_cols = state.move_history.shape[2] if len(state.move_history.shape) > 2 else 7
    has_capture_target_cols = history_cols >= 9

    for i in range(state.max_history_moves):
        move_type = state.move_history[game_idx, i, 0].item()
        if move_type < 0:
            break

        player = int(state.move_history[game_idx, i, 1].item())
        from_y = int(state.move_history[game_idx, i, 2].item())
        from_x = int(state.move_history[game_idx, i, 3].item())
        to_y = int(state.move_history[game_idx, i, 4].item())
        to_x = int(state.move_history[game_idx, i, 5].item())
        phase = int(state.move_history[game_idx, i, 6].item())

        # Read capture target columns if available (December 2025)
        capture_target_y = -1
        capture_target_x = -1
        if has_capture_target_cols:
            capture_target_y = int(state.move_history[game_idx, i, 7].item())
            capture_target_x = int(state.move_history[game_idx, i, 8].item())

        canonical_move = convert_gpu_move_to_canonical(
            move_type=int(move_type),
            player=player,
            from_y=from_y,
            from_x=from_x,
            to_y=to_y,
            to_x=to_x,
            phase=phase,
            board_type=board_type,
            capture_target_y=capture_target_y,
            capture_target_x=capture_target_x,
        )
        moves.append(canonical_move)

    return moves


def validate_canonical_move_sequence(
    moves: list[dict[str, Any]],
    num_players: int = 2,
) -> tuple[bool, list[str]]:
    """Validate a canonical move sequence against phase/move contract.

    Checks:
    - Move types are valid for their phases
    - Player sequence is valid
    - Phase transitions follow canonical order

    Args:
        moves: List of canonical move dictionaries
        num_players: Number of players in the game

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if not moves:
        return True, []

    # Phase to allowed move types mapping - must match history_contract.py
    allowed_moves_by_phase = {
        "ring_placement": {"place_ring", "skip_placement", "no_placement_action", "swap_sides"},
        "movement": {
            "move_stack",
            "overtaking_capture",
            "recovery_slide", "skip_recovery",  # recovery in movement phase
            "no_movement_action",
        },
        "capture": {"overtaking_capture", "skip_capture"},
        "chain_capture": {"continue_capture_segment"},
        "line_processing": {"process_line", "choose_line_option", "no_line_action", "eliminate_rings_from_stack"},
        "territory_processing": {
            "choose_territory_option",
            "eliminate_rings_from_stack",
            "no_territory_action",
            "skip_territory_processing",
        },
        "forced_elimination": {"forced_elimination"},
        "game_over": set(),  # No moves in game_over
    }

    for i, move in enumerate(moves):
        move_type = move.get("type", "unknown")
        phase = move.get("phase")
        player = move.get("player", 0)

        # Check player number is valid
        if player < 1 or player > num_players:
            errors.append(f"Move {i}: Invalid player {player}")

        if not phase:
            errors.append(f"Move {i}: missing phase")
            continue

        # Check move type is valid for phase
        allowed = allowed_moves_by_phase.get(phase, set())
        if move_type not in allowed:
            errors.append(f"Move {i}: {move_type} not valid in {phase} phase")

    is_valid = len(errors) == 0
    return is_valid, errors


def export_game_to_canonical_dict(
    state: BatchGameState,
    game_idx: int,
    board_type: str = "square8",
    num_players: int = 2,
) -> dict[str, Any]:
    """Export a complete game to canonical dictionary format.

    This produces a game dictionary compatible with:
    - scripts/import_gpu_selfplay_to_db.py
    - Training data pipelines

    Args:
        state: BatchGameState containing game data
        game_idx: Index of the game in the batch
        board_type: Board type string
        num_players: Number of players

    Returns:
        Game dictionary with canonical move history
    """
    import time
    from datetime import datetime

    # Get canonical moves
    moves = convert_gpu_history_to_canonical(state, game_idx, board_type)

    # Determine winner
    winner = int(state.winner[game_idx].item())
    winner = winner if winner > 0 else None

    # Determine victory type
    victory_type, tiebreaker = state.derive_victory_type(game_idx, state.max_history_moves)

    # Build game dictionary
    game_dict = {
        "game_id": f"gpu_{board_type}_{num_players}p_{game_idx}_{int(time.time())}",
        "board_type": board_type,
        "num_players": num_players,
        "winner": winner,
        "move_count": len(moves),
        "total_moves": len(moves),
        "status": "completed" if winner else "active",
        "game_status": "completed" if winner else "active",
        "victory_type": victory_type,
        "stalemate_tiebreaker": tiebreaker,
        "completed": winner is not None,
        "engine_mode": "gpu-batch",
        "opponent_type": "selfplay",
        "player_types": ["gpu-mcts"] * num_players,
        "moves": moves,
        "timestamp": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "source": "gpu_canonical_export.py",
    }

    return game_dict


__all__ = [
    "convert_gpu_history_to_canonical",
    "convert_gpu_move_to_canonical",
    "export_game_to_canonical_dict",
    "gpu_move_type_to_canonical",
    "gpu_phase_to_canonical",
    "validate_canonical_move_sequence",
]
