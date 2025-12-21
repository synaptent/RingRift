#!/usr/bin/env python
"""Convert JSONL selfplay data to NPZ format for neural network training.

This script converts game records from JSONL format (produced by run_self_play_soak.py)
into compressed NumPy arrays (.npz) suitable for direct neural network training.

The script replays each game from its initial_state using the moves list,
extracting features at each position using the appropriate encoder:
- Hexagonal boards: HexStateEncoder (v2, 40 channels) or HexStateEncoderV3 (v3, 64 channels)
- Square boards: NeuralNetAI (56 channels)

**Checkpointing**: The script saves intermediate chunks to disk every N games
(default: 100) to prevent data loss on interruption. Use --checkpoint-dir to
enable this feature. Chunks are merged at the end into the final NPZ file.

Output NPZ format (compatible with train.py):
- features: (N, C, H, W) float32 - Full feature channels (C depends on encoder)
- globals: (N, 20) float32 - Global features
- values: (N,) float32 - Game outcomes (-1 to +1)
- policy_indices: (N,) object - Sparse action indices per sample
- policy_values: (N,) object - Sparse action probabilities per sample
- move_numbers: (N,) int32 - Move index within game
- total_game_moves: (N,) int32 - Total moves in source game
- phases: (N,) object - Game phase at each position
- values_mp: (N, 4) float32 - Multi-player value vectors
- num_players: (N,) int32 - Player count per sample

Usage:
    # Basic conversion (replays games properly)
    PYTHONPATH=. python scripts/jsonl_to_npz.py \\
        --input data/selfplay/combined_cloud.jsonl \\
        --output data/training/from_jsonl.npz \\
        --board-type square8 \\
        --num-players 2

    # With checkpointing (saves progress every 100 games)
    PYTHONPATH=. python scripts/jsonl_to_npz.py \\
        --input-dir data/selfplay/ \\
        --output data/training/sampled.npz \\
        --board-type square19 \\
        --num-players 2 \\
        --checkpoint-dir /tmp/npz_checkpoints \\
        --checkpoint-interval 100

    # Resume from checkpoint after interruption
    PYTHONPATH=. python scripts/jsonl_to_npz.py \\
        --input-dir data/selfplay/ \\
        --output data/training/sampled.npz \\
        --board-type square19 \\
        --num-players 2 \\
        --checkpoint-dir /tmp/npz_checkpoints \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ensure app.* imports resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force CPU to avoid MPS/OMP issues during batch conversion
os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

from scripts.lib.cli import BOARD_TYPE_MAP
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("jsonl_to_npz")

from app.ai.neural_net import INVALID_MOVE_INDEX, NeuralNetAI, encode_move_for_board
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameState, Move, MoveType, Position
from app.rules.serialization import deserialize_game_state
from app.training.encoding import HexStateEncoder, HexStateEncoderV3

# Phase transition moves for completing turns
NO_ACTION_MOVES = {
    "movement": MoveType.NO_MOVEMENT_ACTION,
    "line_processing": MoveType.NO_LINE_ACTION,
    "territory_processing": MoveType.NO_TERRITORY_ACTION,
}

# Valid move types per phase
PHASE_VALID_MOVES = {
    "ring_placement": [MoveType.PLACE_RING, MoveType.NO_PLACEMENT_ACTION],
    "movement": [MoveType.MOVE_STACK, MoveType.RECOVERY_SLIDE, MoveType.NO_MOVEMENT_ACTION],
    "line_processing": [MoveType.PROCESS_LINE, MoveType.NO_LINE_ACTION],
    "territory_processing": [MoveType.PROCESS_TERRITORY_REGION, MoveType.NO_TERRITORY_ACTION],
}


def _get_phase_str(state: GameState) -> str:
    """Get the phase as a string."""
    phase = state.current_phase
    if hasattr(phase, "value"):
        return str(phase.value)
    return str(phase)


def _complete_remaining_phases(
    state: GameState,
    target_player: int,
    target_move_type: MoveType | None = None,
) -> GameState:
    """Apply no-action moves to advance through phases until we can apply the target move.

    This handles GPU selfplay format where only place_ring/move_stack moves are recorded,
    skipping the phase transition moves (no_movement_action, no_line_action, no_territory_action).

    Args:
        state: Current game state
        target_player: Player who will make the next move
        target_move_type: The type of move we're trying to apply (to determine target phase)
    """
    # Map move types to phases where they're valid (use actual GamePhase.value strings)
    MOVE_VALID_PHASES: dict[MoveType, list[str]] = {
        MoveType.PLACE_RING: ["ring_placement"],
        MoveType.NO_PLACEMENT_ACTION: ["ring_placement"],
        MoveType.MOVE_STACK: ["movement"],
        MoveType.MOVE_RING: ["movement"],
        MoveType.BUILD_STACK: ["movement"],
        MoveType.RECOVERY_SLIDE: ["movement"],
        MoveType.NO_MOVEMENT_ACTION: ["movement"],
        MoveType.PROCESS_LINE: ["line_processing"],
        MoveType.CHOOSE_LINE_REWARD: ["line_processing"],
        MoveType.NO_LINE_ACTION: ["line_processing"],
        MoveType.PROCESS_TERRITORY_REGION: ["territory_processing"],
        MoveType.SKIP_TERRITORY_PROCESSING: ["territory_processing"],
        MoveType.NO_TERRITORY_ACTION: ["territory_processing"],
        MoveType.OVERTAKING_CAPTURE: ["capture"],
        MoveType.SKIP_CAPTURE: ["capture"],
        MoveType.CONTINUE_CAPTURE_SEGMENT: ["chain_capture"],
    }

    target_phases = MOVE_VALID_PHASES.get(target_move_type, []) if target_move_type else []

    max_iterations = 20  # Prevent infinite loops
    iterations = 0

    while iterations < max_iterations:
        phase_str = _get_phase_str(state)
        current_player = state.current_player

        # If target move type is specified, check if we're already in the right phase for it
        if target_move_type and target_phases and phase_str in target_phases and current_player == target_player:
            return state

        # Fallback for no target_move_type: stop at ring_placement for target player
        if not target_move_type and phase_str == "ring_placement" and current_player == target_player:
            return state

        # If we're at ring_placement for wrong player, apply no_placement
        if phase_str == "ring_placement" and current_player != target_player:
            no_place_move = Move(
                id="auto_no_placement",
                type=MoveType.NO_PLACEMENT_ACTION,
                player=current_player,
                timestamp="1970-01-01T00:00:00Z",
                thinkTime=0,
                moveNumber=0,
            )
            try:
                state = GameEngine.apply_move(state, no_place_move)
            except Exception:
                # Can't apply - must place a ring
                return state
            iterations += 1
            continue

        # If we're at ring_placement for target player but need a different phase,
        # we can't skip ring placement - return as-is
        if phase_str == "ring_placement" and current_player == target_player and target_move_type and target_phases and "ring_placement" not in target_phases:
            # This shouldn't happen - we need ring_placement but target wants different phase
            # Return as-is and let caller handle
            return state

        # Check if we can apply a no-action move for the current phase
        if phase_str in NO_ACTION_MOVES:
            no_action_type = NO_ACTION_MOVES[phase_str]
            no_action_move = Move(
                id=f"auto_{no_action_type.value}",
                type=no_action_type,
                player=current_player,
                timestamp="1970-01-01T00:00:00Z",
                thinkTime=0,
                moveNumber=0,
            )
            try:
                state = GameEngine.apply_move(state, no_action_move)
            except Exception:
                # Can't apply - might have mandatory moves
                return state
        else:
            # Unknown phase or can't advance, return as-is
            return state

        iterations += 1

    return state


def _value_from_winner(winner: int, perspective: int, num_players: int) -> float:
    """Compute value from winner field directly (for GPU selfplay)."""
    if winner == 0:  # Draw
        return 0.0
    if winner == perspective:
        return 1.0
    elif num_players == 2:
        return -1.0
    else:
        # Multi-player: non-winner gets -1
        return -1.0


def _compute_multi_player_values_from_winner(winner: int, num_players: int) -> list[float]:
    """Compute value vector for all players from winner field."""
    values = []
    for p in range(1, 5):  # Always 4 slots
        if p <= num_players:
            values.append(_value_from_winner(winner, p, num_players))
        else:
            values.append(0.0)
    return values


def _process_gpu_selfplay_record(
    record: dict[str, Any],
    encoder: NeuralNetAI,
    history_length: int,
    sample_every: int,
) -> tuple[
    list[np.ndarray],  # features
    list[np.ndarray],  # globals
    list[float],       # values
    list[np.ndarray],  # values_mp
    list[int],         # num_players
    list[np.ndarray],  # policy_indices
    list[np.ndarray],  # policy_values
    list[int],         # move_numbers
    list[int],         # total_game_moves
    list[str],         # phases
    int,               # positions_extracted
]:
    """Process a GPU selfplay record with proper policy extraction.

    GPU selfplay uses bulk ring placement (all P1 rings first, then P2) which
    doesn't match canonical FSM turn alternation. We replay through the game
    using _complete_remaining_phases to handle phase transitions, and extract
    proper policy targets from the moves that were actually played.

    This produces training samples with both value AND policy targets.
    """
    features_list: list[np.ndarray] = []
    globals_list: list[np.ndarray] = []
    values_list: list[float] = []
    values_mp_list: list[np.ndarray] = []
    num_players_list: list[int] = []
    policy_indices_list: list[np.ndarray] = []
    policy_values_list: list[np.ndarray] = []
    move_numbers_list: list[int] = []
    total_game_moves_list: list[int] = []
    phases_list: list[str] = []

    # Extract data from record
    initial_state_dict = record.get("initial_state")
    moves_list = record.get("moves", [])
    winner = record.get("winner", 0)
    num_players = record.get("num_players", 2)

    if not initial_state_dict or not moves_list:
        return (features_list, globals_list, values_list, values_mp_list,
                num_players_list, policy_indices_list, policy_values_list,
                move_numbers_list, total_game_moves_list, phases_list, 0)

    # Parse initial state
    initial_state = deserialize_game_state(initial_state_dict)

    # Parse moves
    moves = [parse_move(m) for m in moves_list]
    total_moves = len(moves)

    # Compute value targets from winner field directly
    values_vec = np.array(
        _compute_multi_player_values_from_winner(winner, num_players),
        dtype=np.float32,
    )

    # Replay game and extract features with proper policy targets
    current_state = initial_state
    history_frames: list[np.ndarray] = []
    positions_extracted = 0

    for move_idx, move in enumerate(moves):
        # Get move type (handle string or enum)
        move_type = move.type if isinstance(move.type, MoveType) else _move_type_from_str(str(move.type))

        # Sample every N moves
        if sample_every > 1 and (move_idx % sample_every) != 0:
            # Still need to apply move and update history
            try:
                base_features, _ = encoder._extract_features(current_state)
                history_frames.append(base_features)
                if len(history_frames) > history_length + 1:
                    history_frames.pop(0)
                # Complete phase transitions before applying recorded move
                current_state = _complete_remaining_phases(current_state, move.player, move_type)
                current_state = GameEngine.apply_move(current_state, move)
            except Exception:
                break  # Stop on error
            continue

        try:
            # Complete phase transitions to get to the right player/phase
            current_state = _complete_remaining_phases(current_state, move.player, move_type)

            # Encode state with history BEFORE applying the move
            stacked, globals_vec = encode_state_with_history(
                encoder, current_state, history_frames, history_length
            )

            # Encode the actual move as the policy target (board-aware encoding)
            action_idx = encode_move_for_board(move, current_state.board)
            if action_idx == INVALID_MOVE_INDEX:
                # Skip invalid moves but still apply to continue
                base_features, _ = encoder._extract_features(current_state)
                history_frames.append(base_features)
                if len(history_frames) > history_length + 1:
                    history_frames.pop(0)
                current_state = GameEngine.apply_move(current_state, move)
                continue

            # Value from perspective of current player
            value = _value_from_winner(winner, current_state.current_player, num_players)

            # Phase string
            phase_str = (
                str(current_state.current_phase.value)
                if hasattr(current_state.current_phase, "value")
                else str(current_state.current_phase)
            )

            # Store sample with proper policy target
            features_list.append(stacked)
            globals_list.append(globals_vec)
            values_list.append(float(value))
            values_mp_list.append(values_vec.copy())
            num_players_list.append(num_players)
            policy_indices_list.append(np.array([action_idx], dtype=np.int32))
            policy_values_list.append(np.array([1.0], dtype=np.float32))
            move_numbers_list.append(move_idx)
            total_game_moves_list.append(total_moves)
            phases_list.append(phase_str)

            positions_extracted += 1

            # Update history
            base_features, _ = encoder._extract_features(current_state)
            history_frames.append(base_features)
            if len(history_frames) > history_length + 1:
                history_frames.pop(0)

            # Apply move for next iteration
            current_state = GameEngine.apply_move(current_state, move)

        except Exception:
            # Stop on error - state may be desynced
            break

    return (features_list, globals_list, values_list, values_mp_list,
            num_players_list, policy_indices_list, policy_values_list,
            move_numbers_list, total_game_moves_list, phases_list, positions_extracted)


def _move_type_from_str(type_str: str) -> MoveType | None:
    """Convert move type string to MoveType enum."""
    try:
        return MoveType(type_str)
    except ValueError:
        return None


class HexEncoderWrapper:
    """Wrapper for HexStateEncoder/V3 to provide _extract_features interface."""

    def __init__(self, encoder, board_size: int = 25):
        self._encoder = encoder
        self.board_size = board_size

    def _extract_features(self, state: GameState):
        """Extract features using the hex encoder's encode method."""
        return self._encoder.encode(state)


def build_encoder(
    board_type: BoardType,
    encoder_version: str = "v2",
    feature_version: int = 2,
):
    """Build an encoder instance for feature extraction.

    For hexagonal boards, uses HexStateEncoder (v2, 10 channels) or
    HexStateEncoderV3 (v3, 16 channels) for proper compatibility with
    HexNeuralNet_v2 (40 total channels) or HexNeuralNet_v3 (64 total).

    For square boards, uses NeuralNetAI (14 base channels).

    Args:
        board_type: The board type
        encoder_version: "v2" for 10-channel hex encoder (40 total),
                        "v3" for 16-channel hex encoder (64 total)

    Returns:
        An encoder with _extract_features(state) method
    """
    if board_type == BoardType.HEXAGONAL:
        # Use proper hex encoders for compatible training data
        if encoder_version == "v3":
            hex_encoder = HexStateEncoderV3(feature_version=feature_version)
            logger.info("Using HexStateEncoderV3 (16 base channels -> 64 total)")
        else:
            hex_encoder = HexStateEncoder(feature_version=feature_version)
            logger.info("Using HexStateEncoder (10 base channels -> 40 total)")
        return HexEncoderWrapper(hex_encoder, board_size=25)

    # For square boards, use NeuralNetAI (14 base channels)
    config = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        rngSeed=None,
        heuristic_profile_id=None,
        nn_model_id=None,
        heuristic_eval_mode=None,
        use_neural_net=True,
    )
    encoder = NeuralNetAI(player_number=1, config=config)
    encoder.feature_version = int(feature_version)
    encoder.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)
    return encoder


def parse_position(pos_data: dict[str, Any] | list | None) -> Position | None:
    """Parse position dict or list to Position object.

    Handles both formats:
    - Dict: {"x": 6, "y": 6, "z": None}
    - List: [6, 6] or [6, 6, z]
    """
    if pos_data is None:
        return None
    if isinstance(pos_data, list):
        # GPU selfplay format: [y, x] or [y, x, z]
        return Position(
            x=pos_data[1] if len(pos_data) > 1 else 0,
            y=pos_data[0] if len(pos_data) > 0 else 0,
            z=pos_data[2] if len(pos_data) > 2 else None,
        )
    return Position(
        x=pos_data.get("x", 0),
        y=pos_data.get("y", 0),
        z=pos_data.get("z"),
    )


def parse_move(move_dict: dict[str, Any]) -> Move:
    """Parse move dict from JSONL to Move object.

    Handles both canonical format and GPU selfplay format:
    - Canonical: {"type": "place_ring", "from": {...}, "to": {...}}
    - GPU selfplay: {"move_type": "PLACEMENT", "from_pos": [...], "to_pos": [...]}
    """
    # Handle move_type vs type field
    move_type = move_dict.get("type") or move_dict.get("move_type", "unknown")
    # Normalize GPU selfplay move types (UPPERCASE) to canonical (lowercase)
    gpu_type_map = {
        # Ring placement
        "PLACEMENT": "place_ring",
        "SKIP_PLACEMENT": "skip_placement",
        "NO_PLACEMENT_ACTION": "no_placement_action",
        # Movement
        "MOVEMENT": "move_stack",
        "NO_MOVEMENT_ACTION": "no_movement_action",
        "RECOVERY_SLIDE": "recovery_slide",
        # Capture
        "CAPTURE": "overtaking_capture",
        "OVERTAKING_CAPTURE": "overtaking_capture",
        "CONTINUE_CAPTURE_SEGMENT": "continue_capture_segment",
        "SKIP_CAPTURE": "skip_capture",
        # Line processing
        "LINE": "process_line",
        "PROCESS_LINE": "process_line",
        "CHOOSE_LINE_OPTION": "choose_line_option",
        "CHOOSE_LINE_REWARD": "choose_line_reward",
        "NO_LINE_ACTION": "no_line_action",
        # Territory processing
        "TERRITORY": "process_territory_region",
        "PROCESS_TERRITORY_REGION": "process_territory_region",
        "CHOOSE_TERRITORY_OPTION": "choose_territory_option",
        "NO_TERRITORY_ACTION": "no_territory_action",
        # Other
        "FORCED_ELIMINATION": "forced_elimination",
        "ELIMINATE_RINGS_FROM_STACK": "eliminate_rings_from_stack",
    }
    if move_type in gpu_type_map:
        move_type = gpu_type_map[move_type]

    return Move(
        id=move_dict.get("id", "imported"),
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=parse_position(move_dict.get("from_pos") or move_dict.get("from")),
        to=parse_position(move_dict.get("to_pos") or move_dict.get("to")),
        capture_target=parse_position(move_dict.get("capture_target")),
        captured_stacks=move_dict.get("captured_stacks"),
        capture_chain=move_dict.get("capture_chain"),
        overtaken_rings=move_dict.get("overtaken_rings"),
        placed_on_stack=move_dict.get("placed_on_stack", False),
        placement_count=move_dict.get("placement_count"),
        stack_moved=move_dict.get("stack_moved"),
        minimum_distance=move_dict.get("minimum_distance"),
        # Required fields with defaults - use epoch time if timestamp missing
        timestamp=move_dict.get("timestamp") or "1970-01-01T00:00:00Z",
        thinkTime=move_dict.get("think_time", move_dict.get("thinkTime", 0)),
        moveNumber=move_dict.get("move_number", move_dict.get("moveNumber", 0)),
    )


def encode_state_with_history(
    encoder: NeuralNetAI,
    state: GameState,
    history_frames: list[np.ndarray],
    history_length: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode state with history frames, matching export_replay_dataset.py."""
    base_features, globals_vec = encoder._extract_features(state)

    # Stack history frames
    c, h, w = base_features.shape
    total_channels = c * (history_length + 1)
    stacked = np.zeros((total_channels, h, w), dtype=np.float32)

    # Current frame first
    stacked[:c] = base_features

    # Historical frames
    for i, hist_frame in enumerate(reversed(history_frames[-history_length:])):
        offset = c * (i + 1)
        if offset + c <= total_channels:
            stacked[offset:offset + c] = hist_frame

    return stacked, globals_vec


def value_from_final_ranking(
    final_state: GameState,
    perspective: int,
    num_players: int,
) -> float:
    """Compute value from final ranking (rank-aware for multiplayer)."""
    # Get rankings from final state
    rankings = []
    for p in final_state.players:
        score = p.eliminated_rings  # Higher eliminated = better
        rankings.append((p.player_number, score))

    # Sort by score descending
    rankings.sort(key=lambda x: -x[1])

    # Find perspective player's rank (0-indexed)
    rank = 0
    for i, (pnum, _) in enumerate(rankings):
        if pnum == perspective:
            rank = i
            break

    # Convert rank to value
    if num_players == 2:
        return 1.0 if rank == 0 else -1.0
    elif num_players == 3:
        return [1.0, 0.0, -1.0][rank]
    else:  # 4 players
        return [1.0, 0.33, -0.33, -1.0][rank]


def compute_multi_player_values(final_state: GameState, num_players: int) -> list[float]:
    """Compute value vector for all players."""
    values = []
    for p in range(1, 5):  # Always 4 slots
        if p <= num_players:
            values.append(value_from_final_ranking(final_state, p, num_players))
        else:
            values.append(0.0)
    return values


@dataclass
class ConversionStats:
    """Statistics from conversion."""
    files_processed: int = 0
    games_processed: int = 0
    games_skipped_filter: int = 0
    games_skipped_no_data: int = 0
    games_skipped_error: int = 0
    positions_extracted: int = 0


class CheckpointManager:
    """Manages checkpointing for long-running NPZ conversions.

    Saves intermediate chunks to disk periodically to prevent data loss.
    Supports resume from checkpoint after interruption.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_interval: int = 100,
        enabled: bool = True,
        history_length: int = 3,
        feature_version: int = 2,
        policy_encoding: str = "board_aware",
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.enabled = enabled and self.checkpoint_dir is not None
        self.chunk_count = 0
        self.progress_file: Path | None = None
        self.progress: dict[str, Any] = {}
        self.history_length = int(history_length)
        self.feature_version = int(feature_version)
        self.policy_encoding = policy_encoding

        if self.enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.progress_file = self.checkpoint_dir / "progress.json"

    def load_progress(self) -> dict[str, Any]:
        """Load progress from checkpoint if exists."""
        if not self.enabled or not self.progress_file.exists():
            return {"games_completed": 0, "chunks": [], "stats": {}}

        try:
            with open(self.progress_file) as f:
                self.progress = json.load(f)
                logger.info(f"Loaded checkpoint: {self.progress.get('games_completed', 0)} games completed")
                return self.progress
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {"games_completed": 0, "chunks": [], "stats": {}}

    def save_progress(self, games_completed: int, stats: ConversionStats):
        """Save current progress."""
        if not self.enabled:
            return

        self.progress = {
            "games_completed": games_completed,
            "chunks": [f"chunk_{i:04d}.npz" for i in range(self.chunk_count)],
            "stats": {
                "files_processed": stats.files_processed,
                "games_processed": stats.games_processed,
                "games_skipped_filter": stats.games_skipped_filter,
                "games_skipped_no_data": stats.games_skipped_no_data,
                "games_skipped_error": stats.games_skipped_error,
                "positions_extracted": stats.positions_extracted,
            },
            "timestamp": time.time(),
        }

        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def save_chunk(
        self,
        features: list[np.ndarray],
        globals_vec: list[np.ndarray],
        values: list[float],
        values_mp: list[np.ndarray],
        num_players: list[int],
        policy_indices: list[np.ndarray],
        policy_values: list[np.ndarray],
        move_numbers: list[int],
        total_game_moves: list[int],
        phases: list[str],
    ) -> Path | None:
        """Save a chunk of data to disk."""
        if not self.enabled or not features:
            return None

        chunk_path = self.checkpoint_dir / f"chunk_{self.chunk_count:04d}.npz"

        try:
            np.savez_compressed(
                chunk_path,
                features=np.stack(features, axis=0).astype(np.float32),
                globals=np.stack(globals_vec, axis=0).astype(np.float32),
                values=np.array(values, dtype=np.float32),
                values_mp=np.stack(values_mp, axis=0).astype(np.float32),
                num_players=np.array(num_players, dtype=np.int32),
                policy_indices=np.array(policy_indices, dtype=object),
                policy_values=np.array(policy_values, dtype=object),
                move_numbers=np.array(move_numbers, dtype=np.int32),
                total_game_moves=np.array(total_game_moves, dtype=np.int32),
                phases=np.array(phases, dtype=object),
                history_length=np.asarray(int(self.history_length)),
                feature_version=np.asarray(int(self.feature_version)),
                policy_encoding=np.asarray(self.policy_encoding),
            )

            self.chunk_count += 1
            logger.info(f"Saved checkpoint chunk {self.chunk_count}: {len(features)} samples to {chunk_path}")
            return chunk_path

        except Exception as e:
            logger.error(f"Failed to save chunk: {e}")
            return None

    def merge_chunks(self, output_path: Path) -> bool:
        """Merge all chunks into final NPZ file."""
        if not self.enabled:
            return False

        chunk_files = sorted(self.checkpoint_dir.glob("chunk_*.npz"))
        if not chunk_files:
            logger.warning("No chunks to merge")
            return False

        logger.info(f"Merging {len(chunk_files)} chunks into {output_path}...")

        # Load all chunks
        all_features = []
        all_globals = []
        all_values = []
        all_values_mp = []
        all_num_players = []
        all_policy_indices = []
        all_policy_values = []
        all_move_numbers = []
        all_total_game_moves = []
        all_phases = []

        for chunk_file in chunk_files:
            try:
                with np.load(chunk_file, allow_pickle=True) as data:
                    all_features.append(data["features"])
                    all_globals.append(data["globals"])
                    all_values.append(data["values"])
                    all_values_mp.append(data["values_mp"])
                    all_num_players.append(data["num_players"])
                    all_policy_indices.extend(data["policy_indices"])
                    all_policy_values.extend(data["policy_values"])
                    all_move_numbers.append(data["move_numbers"])
                    all_total_game_moves.append(data["total_game_moves"])
                    all_phases.extend(data["phases"])
            except Exception as e:
                logger.error(f"Failed to load chunk {chunk_file}: {e}")
                return False

        # Concatenate arrays
        features_arr = np.concatenate(all_features, axis=0)
        globals_arr = np.concatenate(all_globals, axis=0)
        values_arr = np.concatenate(all_values, axis=0)
        values_mp_arr = np.concatenate(all_values_mp, axis=0)
        num_players_arr = np.concatenate(all_num_players, axis=0)
        move_numbers_arr = np.concatenate(all_move_numbers, axis=0)
        total_game_moves_arr = np.concatenate(all_total_game_moves, axis=0)
        policy_indices_arr = np.array(all_policy_indices, dtype=object)
        policy_values_arr = np.array(all_policy_values, dtype=object)
        phases_arr = np.array(all_phases, dtype=object)

        # Save merged file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            features=features_arr,
            globals=globals_arr,
            values=values_arr,
            policy_indices=policy_indices_arr,
            policy_values=policy_values_arr,
            move_numbers=move_numbers_arr,
            total_game_moves=total_game_moves_arr,
            phases=phases_arr,
            values_mp=values_mp_arr,
            num_players=num_players_arr,
            history_length=np.asarray(int(self.history_length)),
            feature_version=np.asarray(int(self.feature_version)),
            policy_encoding=np.asarray(self.policy_encoding),
        )

        logger.info(f"Merged {len(features_arr)} samples into {output_path}")
        return True

    def cleanup(self):
        """Remove checkpoint directory after successful completion."""
        if self.enabled and self.checkpoint_dir.exists():
            try:
                shutil.rmtree(self.checkpoint_dir)
                logger.info(f"Cleaned up checkpoint directory: {self.checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint dir: {e}")


def process_jsonl_file(
    filepath: Path,
    encoder: NeuralNetAI,
    board_type: BoardType,
    board_filter: str | None,
    players_filter: int | None,
    max_games: int | None,
    sample_every: int,
    history_length: int,
    current_games: int,
    gpu_selfplay_mode: bool = False,
) -> tuple[
    list[np.ndarray],  # features
    list[np.ndarray],  # globals
    list[float],       # values
    list[np.ndarray],  # values_mp
    list[int],         # num_players
    list[np.ndarray],  # policy_indices
    list[np.ndarray],  # policy_values
    list[int],         # move_numbers
    list[int],         # total_game_moves
    list[str],         # phases
    ConversionStats,
]:
    """Process a single JSONL file and extract training data."""
    features_list = []
    globals_list = []
    values_list = []
    values_mp_list = []
    num_players_list = []
    policy_indices_list = []
    policy_values_list = []
    move_numbers_list = []
    total_game_moves_list = []
    phases_list = []

    stats = ConversionStats()
    games_in_file = 0

    with open(filepath) as f:
        for line in f:
            if max_games and (current_games + games_in_file) >= max_games:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.games_skipped_error += 1
                continue

            # Apply filters
            board_type_str = record.get("board_type", "square8")
            num_players = record.get("num_players", 2)

            if board_filter and board_type_str != board_filter:
                stats.games_skipped_filter += 1
                continue
            if players_filter and num_players != players_filter:
                stats.games_skipped_filter += 1
                continue

            # Check required data
            initial_state_dict = record.get("initial_state")
            moves_list = record.get("moves", [])

            if not initial_state_dict or not moves_list:
                stats.games_skipped_no_data += 1
                continue

            try:
                # GPU selfplay mode: use simplified processing without FSM validation
                if gpu_selfplay_mode:
                    (
                        gpu_features, gpu_globals, gpu_values, gpu_values_mp,
                        gpu_num_players, gpu_policy_idx, gpu_policy_val,
                        gpu_move_nums, gpu_total_moves, gpu_phases, extracted
                    ) = _process_gpu_selfplay_record(
                        record, encoder, history_length, sample_every
                    )

                    features_list.extend(gpu_features)
                    globals_list.extend(gpu_globals)
                    values_list.extend(gpu_values)
                    values_mp_list.extend(gpu_values_mp)
                    num_players_list.extend(gpu_num_players)
                    policy_indices_list.extend(gpu_policy_idx)
                    policy_values_list.extend(gpu_policy_val)
                    move_numbers_list.extend(gpu_move_nums)
                    total_game_moves_list.extend(gpu_total_moves)
                    phases_list.extend(gpu_phases)

                    stats.positions_extracted += extracted
                    games_in_file += 1
                    stats.games_processed += 1
                    continue

                # Standard mode: parse initial state and replay through FSM
                initial_state = deserialize_game_state(initial_state_dict)

                # Parse moves
                moves = [parse_move(m) for m in moves_list]
                len(moves)

                # Replay game and extract features
                current_state = initial_state
                history_frames: list[np.ndarray] = []

                # Compute final state for value targets
                # Replay until we hit an error, use that as "final" state
                final_state = initial_state
                moves_succeeded = 0
                for move in moves:
                    try:
                        # Handle GPU selfplay simplified format:
                        # Complete phase transitions before applying the recorded move
                        if gpu_selfplay_mode:
                            move_type = move.type if isinstance(move.type, MoveType) else _move_type_from_str(str(move.type))
                            final_state = _complete_remaining_phases(final_state, move.player, move_type)
                        final_state = GameEngine.apply_move(final_state, move)
                        moves_succeeded += 1
                    except Exception:
                        # Stop at first error - state is now desynced
                        break

                min_moves_threshold = int(os.environ.get("RINGRIFT_MIN_MOVES", "6"))
                if moves_succeeded < min_moves_threshold:
                    # Need at least min_moves_threshold successful moves to have meaningful data
                    raise ValueError(f"Only {moves_succeeded}/{len(moves)} moves succeeded (min={min_moves_threshold})")

                # Precompute multi-player values
                values_vec = np.array(
                    compute_multi_player_values(final_state, num_players),
                    dtype=np.float32,
                )

                # Only process up to moves_succeeded moves
                for move_idx, move in enumerate(moves[:moves_succeeded]):
                    # Get move type for phase completion
                    move_type_for_phase = move.type if isinstance(move.type, MoveType) else _move_type_from_str(str(move.type))

                    # Sample every N moves
                    if sample_every > 1 and (move_idx % sample_every) != 0:
                        # Still need to apply move and update history
                        base_features, _ = encoder._extract_features(current_state)
                        history_frames.append(base_features)
                        if len(history_frames) > history_length + 1:
                            history_frames.pop(0)
                        # Handle GPU selfplay simplified format
                        if gpu_selfplay_mode:
                            current_state = _complete_remaining_phases(current_state, move.player, move_type_for_phase)
                        current_state = GameEngine.apply_move(current_state, move)
                        continue

                    # Encode state with history
                    stacked, globals_vec = encode_state_with_history(
                        encoder, current_state, history_frames, history_length
                    )

                    # Update history
                    base_features, _ = encoder._extract_features(current_state)
                    history_frames.append(base_features)
                    if len(history_frames) > history_length + 1:
                        history_frames.pop(0)

                    # Encode action (sparse, board-aware encoding)
                    action_idx = encode_move_for_board(move, current_state.board)
                    if action_idx == INVALID_MOVE_INDEX:
                        # Skip invalid moves
                        current_state = GameEngine.apply_move(current_state, move)
                        continue

                    # Value from perspective of current player
                    value = value_from_final_ranking(
                        final_state, current_state.current_player, num_players
                    )

                    # Phase string
                    phase_str = (
                        str(current_state.current_phase.value)
                        if hasattr(current_state.current_phase, "value")
                        else str(current_state.current_phase)
                    )

                    # Store sample
                    features_list.append(stacked)
                    globals_list.append(globals_vec)
                    values_list.append(float(value))
                    values_mp_list.append(values_vec.copy())
                    num_players_list.append(num_players)
                    policy_indices_list.append(np.array([action_idx], dtype=np.int32))
                    policy_values_list.append(np.array([1.0], dtype=np.float32))
                    move_numbers_list.append(move_idx)
                    total_game_moves_list.append(moves_succeeded)
                    phases_list.append(phase_str)

                    stats.positions_extracted += 1

                    # Apply move for next iteration
                    # Handle GPU selfplay simplified format
                    if gpu_selfplay_mode:
                        current_state = _complete_remaining_phases(current_state, move.player, move_type_for_phase)
                    current_state = GameEngine.apply_move(current_state, move)

                games_in_file += 1
                stats.games_processed += 1

            except Exception as e:
                stats.games_skipped_error += 1
                if stats.games_skipped_error <= 3:
                    import traceback
                    logger.warning(f"Error processing game: {e}")
                    logger.warning(traceback.format_exc())
                continue

    stats.files_processed = 1
    return (
        features_list, globals_list, values_list, values_mp_list,
        num_players_list, policy_indices_list, policy_values_list,
        move_numbers_list, total_game_moves_list, phases_list, stats
    )


def find_jsonl_files(input_path: Path, recursive: bool = True) -> list[Path]:
    """Find all JSONL files in the given path."""
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(input_path.glob(pattern))


def convert_jsonl_to_npz(
    input_paths: list[Path],
    output_path: Path,
    board_type_str: str,
    players_filter: int | None = None,
    max_games: int | None = None,
    sample_every: int = 1,
    history_length: int = 3,
    feature_version: int = 2,
    gpu_selfplay_mode: bool = False,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 100,
    resume: bool = False,
    encoder_version: str = "v2",
) -> ConversionStats:
    """Convert JSONL files to NPZ training data.

    Args:
        input_paths: List of JSONL files to process
        output_path: Output NPZ file path
        board_type_str: Board type (square8, square19, hexagonal)
        players_filter: Filter games by player count
        max_games: Maximum number of games to process
        sample_every: Sample every Nth position
        history_length: Number of history frames to stack
        feature_version: Feature encoding version for global feature layout
        gpu_selfplay_mode: Use GPU selfplay simplified format
        checkpoint_dir: Directory for checkpoint chunks (enables checkpointing)
        checkpoint_interval: Save checkpoint every N games
        resume: Resume from existing checkpoint
    """
    board_type = BOARD_TYPE_MAP.get(board_type_str, BoardType.SQUARE8)
    encoder = build_encoder(
        board_type,
        encoder_version=encoder_version,
        feature_version=feature_version,
    )

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        enabled=checkpoint_dir is not None,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding="board_aware",
    )

    # Check for resume
    games_to_skip = 0
    if resume and checkpoint_mgr.enabled:
        progress = checkpoint_mgr.load_progress()
        games_to_skip = progress.get("games_completed", 0)
        checkpoint_mgr.chunk_count = len(progress.get("chunks", []))
        if games_to_skip > 0:
            logger.info(f"Resuming from checkpoint: skipping first {games_to_skip} games")

    all_features = []
    all_globals = []
    all_values = []
    all_values_mp = []
    all_num_players = []
    all_policy_indices = []
    all_policy_values = []
    all_move_numbers = []
    all_total_game_moves = []
    all_phases = []

    total_stats = ConversionStats()
    games_since_checkpoint = 0

    logger.info(f"Processing {len(input_paths)} JSONL files...")
    logger.info(f"Board type: {board_type_str}, Players: {players_filter or 'any'}")
    if checkpoint_mgr.enabled:
        logger.info(f"Checkpointing enabled: saving every {checkpoint_interval} games to {checkpoint_dir}")

    for i, filepath in enumerate(input_paths):
        if max_games and total_stats.games_processed >= max_games:
            break

        (features, globals_vec, values, values_mp, num_players,
         policy_indices, policy_values, move_numbers, total_moves,
         phases, stats) = process_jsonl_file(
            filepath=filepath,
            encoder=encoder,
            board_type=board_type,
            board_filter=board_type_str,
            players_filter=players_filter,
            max_games=max_games,
            sample_every=sample_every,
            history_length=history_length,
            current_games=total_stats.games_processed,
            gpu_selfplay_mode=gpu_selfplay_mode,
        )

        all_features.extend(features)
        all_globals.extend(globals_vec)
        all_values.extend(values)
        all_values_mp.extend(values_mp)
        all_num_players.extend(num_players)
        all_policy_indices.extend(policy_indices)
        all_policy_values.extend(policy_values)
        all_move_numbers.extend(move_numbers)
        all_total_game_moves.extend(total_moves)
        all_phases.extend(phases)

        total_stats.files_processed += stats.files_processed
        total_stats.games_processed += stats.games_processed
        total_stats.games_skipped_filter += stats.games_skipped_filter
        total_stats.games_skipped_no_data += stats.games_skipped_no_data
        total_stats.games_skipped_error += stats.games_skipped_error
        total_stats.positions_extracted += stats.positions_extracted

        games_since_checkpoint += stats.games_processed

        # Save checkpoint if interval reached
        if checkpoint_mgr.enabled and games_since_checkpoint >= checkpoint_interval:
            checkpoint_mgr.save_chunk(
                all_features, all_globals, all_values, all_values_mp,
                all_num_players, all_policy_indices, all_policy_values,
                all_move_numbers, all_total_game_moves, all_phases,
            )
            checkpoint_mgr.save_progress(total_stats.games_processed, total_stats)

            # Clear buffers after saving chunk
            all_features.clear()
            all_globals.clear()
            all_values.clear()
            all_values_mp.clear()
            all_num_players.clear()
            all_policy_indices.clear()
            all_policy_values.clear()
            all_move_numbers.clear()
            all_total_game_moves.clear()
            all_phases.clear()
            games_since_checkpoint = 0

        if (i + 1) % 10 == 0 or i == len(input_paths) - 1:
            logger.info(
                f"Processed {i + 1}/{len(input_paths)} files, "
                f"{total_stats.games_processed} games, "
                f"{total_stats.positions_extracted} positions"
            )

    # Save any remaining data as final chunk
    if checkpoint_mgr.enabled and all_features:
        checkpoint_mgr.save_chunk(
            all_features, all_globals, all_values, all_values_mp,
            all_num_players, all_policy_indices, all_policy_values,
            all_move_numbers, all_total_game_moves, all_phases,
        )
        checkpoint_mgr.save_progress(total_stats.games_processed, total_stats)

    # Merge chunks or save directly
    if checkpoint_mgr.enabled and checkpoint_mgr.chunk_count > 0:
        # Merge all chunks into final output
        if checkpoint_mgr.merge_chunks(output_path):
            checkpoint_mgr.cleanup()
        else:
            logger.error("Failed to merge chunks - checkpoint data preserved")
            return total_stats
    elif all_features:
        # No checkpointing - save directly (original behavior)
        logger.info("Converting to numpy arrays...")
        features_arr = np.stack(all_features, axis=0).astype(np.float32)
        globals_arr = np.stack(all_globals, axis=0).astype(np.float32)
        values_arr = np.array(all_values, dtype=np.float32)
        values_mp_arr = np.stack(all_values_mp, axis=0).astype(np.float32)
        num_players_arr = np.array(all_num_players, dtype=np.int32)
        policy_indices_arr = np.array(all_policy_indices, dtype=object)
        policy_values_arr = np.array(all_policy_values, dtype=object)
        move_numbers_arr = np.array(all_move_numbers, dtype=np.int32)
        total_game_moves_arr = np.array(all_total_game_moves, dtype=np.int32)
        phases_arr = np.array(all_phases, dtype=object)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {output_path}...")

        np.savez_compressed(
            output_path,
            features=features_arr,
            globals=globals_arr,
            values=values_arr,
            policy_indices=policy_indices_arr,
            policy_values=policy_values_arr,
            move_numbers=move_numbers_arr,
            total_game_moves=total_game_moves_arr,
            phases=phases_arr,
            values_mp=values_mp_arr,
            num_players=num_players_arr,
            history_length=np.asarray(int(history_length)),
            feature_version=np.asarray(int(feature_version)),
            policy_encoding=np.asarray("board_aware"),
        )
    else:
        logger.warning("No training data extracted!")
        return total_stats

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {output_size_mb:.2f} MB")

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL selfplay data to NPZ format for NN training"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output NPZ file path",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        required=True,
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Board type (required for proper feature encoding)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter games by player count",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to process",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Use every Nth move as a training sample (default: 1 = all)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of historical frames to stack (default: 3)",
    )
    parser.add_argument(
        "--feature-version",
        type=int,
        default=2,
        help=(
            "Feature encoding version for global feature layout (default: 2). "
            "Use 1 to keep legacy hex globals without chain/FE flags."
        ),
    )
    parser.add_argument(
        "--gpu-selfplay",
        action="store_true",
        help="Enable GPU selfplay mode: auto-inject phase transitions for simplified move format",
    )
    parser.add_argument(
        "--encoder-version",
        type=str,
        default="v2",
        choices=["v2", "v3"],
        help="Encoder version for hexagonal boards: v2 (10 base channels -> 40 total, "
             "compatible with HexNeuralNet_v2) or v3 (16 base channels -> 64 total, "
             "compatible with HexNeuralNet_v3). Default: v2",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory for checkpoint chunks (enables incremental saves to prevent data loss)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N games (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint in --checkpoint-dir",
    )

    args = parser.parse_args()

    # Collect input files
    input_paths = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.input_dir:
        input_paths.extend(find_jsonl_files(Path(args.input_dir)))

    if not input_paths:
        parser.error("Must specify --input or --input-dir")

    logger.info(f"Found {len(input_paths)} JSONL files")

    stats = convert_jsonl_to_npz(
        input_paths=input_paths,
        output_path=Path(args.output),
        board_type_str=args.board_type,
        players_filter=args.num_players,
        max_games=args.max_games,
        sample_every=args.sample_every,
        history_length=args.history_length,
        feature_version=args.feature_version,
        gpu_selfplay_mode=args.gpu_selfplay,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        encoder_version=args.encoder_version,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats.files_processed}")
    logger.info(f"Games processed: {stats.games_processed}")
    logger.info(f"Games skipped (filter): {stats.games_skipped_filter}")
    logger.info(f"Games skipped (no data): {stats.games_skipped_no_data}")
    logger.info(f"Games skipped (error): {stats.games_skipped_error}")
    logger.info(f"Positions extracted: {stats.positions_extracted}")


if __name__ == "__main__":
    main()
