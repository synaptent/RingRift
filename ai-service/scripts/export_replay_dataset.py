#!/usr/bin/env python
"""
Export training samples from existing GameReplayDB replays.

This script walks one or more GameReplayDB SQLite files and converts completed
games into a neural-net training dataset in the same NPZ format used by
app.training.generate_data:

    - features: (N, C, H, W) float32
    - globals:  (N, G)       float32
    - values:   (N,)         float32   (from final ranking, per-state perspective)
    - policy_indices: (N,)   object    → np.ndarray[int32] of indices per sample
    - policy_values:  (N,)   object    → np.ndarray[float32] of probs per sample

Each sample corresponds to a (state_before_move, move_taken) pair drawn from
recorded games, with an outcome label derived from the final player ranking.

**Value Target Encoding (rank-aware for multiplayer):**
  - 2-player: winner=+1, loser=-1 (unchanged)
  - 3-player: 1st=+1, 2nd=0, 3rd=-1
  - 4-player: 1st=+1, 2nd=+0.33, 3rd=-0.33, 4th=-1

**Quality Filtering:**
  - Use --require-completed to only include games with normal termination
  - Use --min-moves N to exclude trivially short games
  - Use --max-moves N to exclude abnormally long games

Usage examples (from ai-service/):

    # Basic: export square8 2p samples (parallel by default, uses CPU_COUNT-1 workers)
    python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --output data/training/from_replays.square8_2p.npz

    # Quality-filtered export with explicit worker count
    python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_3p.db \\
        --board-type square8 \\
        --num-players 3 \\
        --workers 16 \\
        --require-completed \\
        --min-moves 20 \\
        --output data/training/from_replays.square8_3p.npz

    # Single-threaded mode (for debugging or when parallel causes issues)
    python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --single-threaded \\
        --output data/training/debug.npz

    # Incremental export with caching (skip if DBs unchanged)
    python scripts/export_replay_dataset.py \\
        --db data/games/consolidated.db \\
        --board-type square8 --num-players 2 \\
        --output data/training/square8_2p.npz \\
        --use-cache

    # Force re-export even with valid cache
    python scripts/export_replay_dataset.py \\
        --db data/games/consolidated.db \\
        --board-type square8 --num-players 2 \\
        --output data/training/square8_2p.npz \\
        --use-cache --force-export
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

from app.ai.neural_net import INVALID_MOVE_INDEX, NeuralNetAI, encode_move_for_board
from app.db import GameReplayDB
from app.models import AIConfig, BoardType, GameState, Move, Position
from app.training.canonical_sources import enforce_canonical_sources


def _normalize_hex_board_size(board: "BoardState") -> "BoardState":
    """Normalize hex board size from legacy Convention A to Convention B.

    Legacy hexagonal games stored board.size=13 (radius+1), but the encoder
    expects board.size=25 (2*radius+1 = bounding box). This function creates
    a normalized board with the correct size for encoding.

    Note: This modifies only the size attribute. Position data in board.stacks
    should already be in cube coordinates [-12, 12].
    """
    from app.models import BoardState, BoardType

    if board.type != BoardType.HEXAGONAL:
        return board

    # Convention A: size = radius + 1 = 13
    # Convention B: size = 2*radius + 1 = 25
    if board.size == 13:
        # Create a new BoardState with corrected size
        return BoardState(
            type=board.type,
            size=25,  # Correct bounding box size
            stacks=board.stacks,
        )

    return board


def _normalize_hex_move_coords(move: Move, board_type: BoardType, board_size: int) -> Move:
    """Normalize move positions from canvas to cube coords for hex boards.

    Legacy GPU selfplay stored hexagonal positions in canvas coords [0, board_size).
    The encoder expects cube coords [-radius, radius]. This function detects and
    converts canvas coords to cube coords for hex boards.

    Detection heuristic: If any coord > radius, it's likely canvas coords.
    For unambiguous cases (coords in [0, radius]), we assume cube coords.
    """
    if board_type not in (BoardType.HEXAGONAL, BoardType.HEX8):
        return move

    radius = (board_size - 1) // 2

    def maybe_convert(pos: Position | None) -> Position | None:
        if pos is None:
            return None

        # Check if coords look like canvas (any coord > radius means canvas)
        x, y = pos.x, pos.y
        if x > radius or y > radius or x < -radius or y < -radius:
            # These are definitely canvas coords [0, board_size) - convert to cube
            cube_x = x - radius
            cube_y = y - radius
            cube_z = -cube_x - cube_y
            return Position(x=cube_x, y=cube_y, z=cube_z)
        else:
            # Ambiguous or already cube coords - assume cube (z might need fixing)
            if pos.z is None:
                z = -x - y
                return Position(x=x, y=y, z=z)
            return pos

    # Create new Move with normalized positions
    return Move(
        id=move.id,
        type=move.type,
        player=move.player,
        from_pos=maybe_convert(move.from_pos),
        to=maybe_convert(move.to),
        capture_target=move.capture_target,
        captured_stacks=move.captured_stacks,
        capture_chain=move.capture_chain,
        overtaken_rings=move.overtaken_rings,
        placed_on_stack=move.placed_on_stack,
        placement_count=move.placement_count,
        stack_moved=move.stack_moved,
        minimum_distance=move.minimum_distance,
        actual_distance=move.actual_distance,
        marker_left=move.marker_left,
        line_index=move.line_index,
        formed_lines=move.formed_lines,
        collapsed_markers=move.collapsed_markers,
        claimed_territory=move.claimed_territory,
        disconnected_regions=move.disconnected_regions,
        recovery_option=move.recovery_option,
        recovery_mode=move.recovery_mode,
        collapse_positions=move.collapse_positions,
        extraction_stacks=move.extraction_stacks,
        eliminated_rings=move.eliminated_rings,
        elimination_context=move.elimination_context,
        timestamp=move.timestamp,
        think_time=move.think_time,
        move_number=move.move_number,
        phase=move.phase,
    )


from app.training.encoding import get_encoder_for_board_type
from app.training.export_cache import get_export_cache
from app.training.export_core import (
    compute_multi_player_values,
    encode_state_with_history,
    value_from_final_ranking,
    value_from_final_winner,
)
from scripts.lib.cli import BOARD_TYPE_MAP


def _enforce_canonical_db_policy(
    db_paths: list[str],
    output_path: str,
    *,
    allow_noncanonical: bool,
) -> None:
    """Refuse to label outputs as canonical when source DBs are non-canonical."""
    if allow_noncanonical:
        return

    if not os.path.basename(output_path).startswith("canonical_"):
        return

    noncanonical = [path for path in db_paths if not os.path.basename(path).startswith("canonical_")]
    if noncanonical:
        joined = ", ".join(noncanonical)
        raise SystemExit(
            "[export-replay-dataset] Refusing to export canonical_* dataset from non-canonical DB(s): "
            f"{joined}\n"
            "Use --allow-noncanonical to override, or rename the output to avoid canonical_ prefix."
        )


def build_encoder(
    board_type: BoardType,
    encoder_version: str = "default",
    feature_version: int = 2,
) -> NeuralNetAI:
    """
    Construct a NeuralNetAI instance for feature and policy encoding.

    This uses a lightweight AIConfig and treats player_number=1 purely as a
    placeholder; we never call select_move(), only the encoding helpers.

    For hexagonal boards, encoder_version can be:
      - "default": Maps to "v3" (HexStateEncoderV3, 16 channels)
      - "v2": Use HexStateEncoder (10 channels) for HexNeuralNet_v2
      - "v3": Use HexStateEncoderV3 (16 channels) for HexNeuralNet_v3

    feature_version controls the global feature layout for encoders.

    IMPORTANT: Hex boards ALWAYS use specialized encoders to ensure consistent
    feature shapes across all games. The "default" option maps to v3.
    """
    # Prefer CPU by default to avoid accidental MPS/OMP issues; callers can
    # override via env (e.g. RINGRIFT_FORCE_CPU=0) if they want GPU.
    os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

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
    # Ensure the encoder's board_size hint is consistent with the dataset.
    encoder.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEX8: 9,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)

    # For hex boards, ALWAYS attach a specialized encoder to ensure consistent
    # feature shapes. Default to v3 (newest, 16 channels).
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        effective_version = encoder_version if encoder_version in ("v2", "v3") else "v3"
        encoder._hex_encoder = get_encoder_for_board_type(
            board_type,
            effective_version,
            feature_version=feature_version,
        )
        encoder._hex_encoder_version = effective_version

    return encoder


# Value computation and encoding functions are now imported from app.training.export_core


def export_replay_dataset_multi(
    db_paths: list[str],
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    feature_version: int = 2,
    sample_every: int = 1,
    max_games: int | None = None,
    require_completed: bool = False,
    min_moves: int | None = None,
    max_moves: int | None = None,
    max_move_index: int | None = None,
    use_rank_aware_values: bool = True,
    parity_fixtures_dir: str | None = None,
    exclude_recovery: bool = False,
    use_board_aware_encoding: bool = False,
    append: bool = False,
    encoder_version: str = "default",
    require_moves: bool = True,
) -> None:
    """
    Export training samples from multiple GameReplayDB files into an NPZ dataset
    with automatic deduplication by game_id.

    This function processes databases in order, tracking game_ids to skip
    duplicates that appear in multiple sources. This enables aggregated training
    from siloed data across multiple nodes without double-counting games.

    Args:
        db_paths: List of paths to GameReplayDB SQLite files
        board_type: Board type to filter games by
        num_players: Number of players to filter games by
        output_path: Path to output .npz dataset
        history_length: Number of past feature frames to stack (default: 3)
        feature_version: Feature encoding version for global feature layout
        sample_every: Use every Nth move as a training sample (default: 1)
        max_games: Optional cap on total number of games to process across all DBs
        require_completed: If True, only include games with normal termination
        min_moves: Minimum move count to include a game
        max_moves: Maximum move count to include a game
        use_rank_aware_values: If True, use rank-based values for multiplayer
        use_board_aware_encoding: If True, use board-specific policy encoding
        append: If True, append to existing output NPZ
        encoder_version: Encoder version for hex boards ('default', 'v2', 'v3')
        require_moves: If True, only include games with move data (default: True)
    """
    encoder = build_encoder(
        board_type,
        encoder_version=encoder_version,
        feature_version=feature_version,
    )

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
    victory_types_list: list[str] = []  # For victory-type-balanced sampling

    # Track seen game_ids for deduplication across databases
    seen_game_ids: set = set()
    games_processed = 0
    games_skipped = 0
    games_deduplicated = 0
    games_skipped_recovery = 0

    # Build query filters
    query_filters: dict[str, Any] = {
        "board_type": board_type,
        "num_players": num_players,
        "require_moves": require_moves,
    }
    if min_moves is not None:
        query_filters["min_moves"] = min_moves
    if max_moves is not None:
        query_filters["max_moves"] = max_moves

    # Optional parity cutoffs
    parity_cutoffs: dict[str, int] = {}
    if parity_fixtures_dir:
        fixtures_path = os.path.abspath(parity_fixtures_dir)
        if os.path.isdir(fixtures_path):
            for name in os.listdir(fixtures_path):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(fixtures_path, name)
                try:
                    with open(path, encoding="utf-8") as f:
                        fixture = json.load(f)
                except Exception:
                    continue
                game_id = fixture.get("game_id")
                diverged_at = fixture.get("diverged_at")
                if not isinstance(game_id, str) or not isinstance(diverged_at, int) or diverged_at <= 0:
                    continue
                safe_max_move = diverged_at - 1
                prev = parity_cutoffs.get(game_id)
                if prev is None or safe_max_move < prev:
                    parity_cutoffs[game_id] = safe_max_move

    # Log filter configuration
    filter_desc = []
    if require_moves:
        filter_desc.append("require move data")
    if require_completed:
        filter_desc.append("completed games only")
    if min_moves is not None:
        filter_desc.append(f"min {min_moves} moves")
    if max_moves is not None:
        filter_desc.append(f"max {max_moves} moves")
    if exclude_recovery:
        filter_desc.append("excluding recovery games")
    if filter_desc:
        print(f"Quality filters: {', '.join(filter_desc)}")
    print(f"Value targets: {'rank-aware' if use_rank_aware_values else 'binary winner/loser'}")
    print(f"Processing {len(db_paths)} database(s) with deduplication")

    # Process each database
    for db_idx, db_path in enumerate(db_paths):
        if not os.path.exists(db_path):
            print(f"  [{db_idx+1}/{len(db_paths)}] Skipping missing: {db_path}")
            continue

        print(f"  [{db_idx+1}/{len(db_paths)}] Processing: {os.path.basename(db_path)}...")

        try:
            db = GameReplayDB(db_path)
        except Exception as e:
            print(f"    Error opening database: {e}")
            continue

        db_games = 0
        db_samples = 0
        db_dedup = 0

        for meta, initial_state, moves in db.iterate_games(**query_filters):
            game_id = meta.get("game_id")

            # Deduplication: skip if we've seen this game_id before
            if game_id in seen_game_ids:
                db_dedup += 1
                games_deduplicated += 1
                continue
            seen_game_ids.add(game_id)

            if require_completed:
                status = str(meta.get("game_status", ""))
                term = str(meta.get("termination_reason", ""))
                if status != "completed":
                    games_skipped += 1
                    continue
                if term and not (term.startswith("status:completed") or term == "env_done_flag"):
                    games_skipped += 1
                    continue

            # Extract victory type for balanced sampling (normalize to standard categories)
            victory_type_raw = str(meta.get("victory_type", meta.get("termination_reason", "unknown")))
            if "territory" in victory_type_raw.lower():
                victory_type = "territory"
            elif "elimination" in victory_type_raw.lower() or "ring" in victory_type_raw.lower():
                victory_type = "elimination"
            elif "lps" in victory_type_raw.lower() or "last_player" in victory_type_raw.lower():
                victory_type = "lps"
            elif "stalemate" in victory_type_raw.lower():
                victory_type = "stalemate"
            elif "timeout" in victory_type_raw.lower():
                victory_type = "timeout"
            else:
                victory_type = "other"

            total_moves = meta.get("total_moves")
            if total_moves is None:
                total_moves = len(moves) if moves else 0
            total_moves = int(total_moves)
            if total_moves <= 0 or not moves:
                continue

            if exclude_recovery:
                has_recovery = any(
                    "recovery" in str(getattr(m, "type", "")).lower()
                    for m in moves
                )
                if has_recovery:
                    games_skipped_recovery += 1
                    continue

            max_safe_move_index: int | None = None
            if parity_cutoffs:
                cutoff = parity_cutoffs.get(game_id)
                if cutoff is not None:
                    max_safe_move_index = cutoff
                    if max_safe_move_index <= 0:
                        games_skipped += 1
                        continue

            # NOTE: We defer final_state computation until after incremental replay
            # to avoid slow db.get_state_at_move() call. The incremental replay will
            # give us the final state naturally.
            final_state_index = total_moves - 1
            if max_safe_move_index is not None:
                final_state_index = min(final_state_index, max_safe_move_index)

            num_players_in_game = len(initial_state.players)

            # Collect samples first, then compute values after we have final state
            game_samples: list[tuple[np.ndarray, np.ndarray, int, int, str]] = []
            history_frames: list[np.ndarray] = []
            samples_before = len(features_list)

            # Use incremental state updates instead of replaying from scratch for each move.
            # This reduces complexity from O(n²) to O(n) per game.
            from app.game_engine import GameEngine

            current_state = initial_state
            replay_succeeded = True
            for move_index, move in enumerate(moves):
                if max_safe_move_index is not None and move_index > max_safe_move_index:
                    break
                if max_move_index is not None and move_index > max_move_index:
                    break

                # state_before is the state BEFORE this move is applied
                state_before = current_state

                # Apply move to get next state (for next iteration)
                try:
                    # Use trace_mode=True for canonical replay behavior
                    current_state = GameEngine.apply_move(current_state, move, trace_mode=True)
                except Exception as e:
                    logger.debug(f"Skipping game {game_id} at move {move_index}: {e}")
                    replay_succeeded = False
                    break

                # Skip if not sampling this move
                if sample_every > 1 and (move_index % sample_every) != 0:
                    continue

                stacked, globals_vec = encode_state_with_history(
                    encoder, state_before, history_frames, history_length=history_length
                )

                # Use the same encoder path as encode_state_with_history for consistent shapes
                hex_encoder = getattr(encoder, "_hex_encoder", None)
                if hex_encoder is not None:
                    base_features, _ = hex_encoder.encode_state(state_before)
                else:
                    base_features, _ = encoder._extract_features(state_before)
                history_frames.append(base_features)
                if len(history_frames) > history_length + 1:
                    history_frames.pop(0)

                # Normalize hex board size and move coords for legacy data
                normalized_board = _normalize_hex_board_size(state_before.board)
                normalized_move = _normalize_hex_move_coords(move, board_type, 25 if board_type == BoardType.HEXAGONAL else initial_state.board.size)

                if use_board_aware_encoding:
                    idx = encode_move_for_board(normalized_move, normalized_board)
                else:
                    idx = encoder.encode_move(normalized_move, normalized_board)
                if idx == INVALID_MOVE_INDEX:
                    continue

                # Store sample with perspective for later value computation
                phase_str = (
                    str(state_before.current_phase.value)
                    if hasattr(state_before.current_phase, "value")
                    else str(state_before.current_phase)
                )
                game_samples.append((
                    stacked, globals_vec, idx, state_before.current_player,
                    move_index, phase_str
                ))

            # Skip this game if replay failed
            if not replay_succeeded or not game_samples:
                games_skipped += 1
                continue

            # Now we have final_state = current_state from incremental replay
            final_state = current_state

            # Skip games without a valid winner - these produce value=0 which corrupts training
            if getattr(final_state, 'winner', None) is None or final_state.winner == 0:
                games_skipped += 1
                continue

            # Compute values using the final replayed state
            if use_rank_aware_values:
                values_vec = np.asarray(
                    compute_multi_player_values(final_state, num_players=num_players_in_game),
                    dtype=np.float32,
                )
            else:
                values_vec = np.zeros(4, dtype=np.float32)
                for p in final_state.players:
                    base = value_from_final_winner(final_state, p.player_number)
                    values_vec[p.player_number - 1] = float(base)

            # Add all samples from this game with computed values
            # NOTE: For scalar value targets, we use the CURRENT PLAYER's perspective.
            # This matches the feature encoding (which uses current player's view)
            # and the inference code (which expects current player's value and
            # flips it if needed, see gumbel_mcts_ai.py lines 790-791).
            # For multi-player training, values_mp provides per-player values.
            for stacked, globals_vec, idx, perspective, move_index, phase_str in game_samples:
                # Use current player's perspective (stored in 'perspective' variable)
                if use_rank_aware_values:
                    value = value_from_final_ranking(
                        final_state, perspective=perspective, num_players=num_players
                    )
                else:
                    value = value_from_final_winner(final_state, perspective=perspective)

                features_list.append(stacked)
                globals_list.append(globals_vec)
                values_list.append(float(value))
                policy_indices_list.append(np.array([idx], dtype=np.int32))
                policy_values_list.append(np.array([1.0], dtype=np.float32))
                values_mp_list.append(values_vec)
                num_players_list.append(num_players_in_game)
                move_numbers_list.append(move_index)
                total_game_moves_list.append(total_moves)
                phases_list.append(phase_str)
                victory_types_list.append(victory_type)

            samples_added = len(features_list) - samples_before
            if samples_added > 0:
                db_games += 1
                db_samples += samples_added

            games_processed += 1
            if max_games is not None and games_processed >= max_games:
                break

        print(f"    -> {db_games} games, {db_samples} samples, {db_dedup} deduplicated")

        if max_games is not None and games_processed >= max_games:
            print(f"  Reached max_games limit ({max_games})")
            break

    if not features_list:
        print(f"No samples generated from {len(db_paths)} database(s) "
              f"(board={board_type}, players={num_players}).")
        return

    # Stack into arrays
    features_arr = np.stack(features_list, axis=0).astype(np.float32)
    globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
    values_arr = np.array(values_list, dtype=np.float32)
    policy_indices_arr = np.array(policy_indices_list, dtype=object)
    policy_values_arr = np.array(policy_values_list, dtype=object)
    values_mp_arr = np.stack(values_mp_list, axis=0).astype(np.float32)
    num_players_arr = np.array(num_players_list, dtype=np.int32)
    move_numbers_arr = np.array(move_numbers_list, dtype=np.int32)
    total_game_moves_arr = np.array(total_game_moves_list, dtype=np.int32)
    phases_arr = np.array(phases_list, dtype=object)
    victory_types_arr = np.array(victory_types_list, dtype=object)  # For balanced sampling

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    write_mp = True
    if os.path.exists(output_path) and not append:
        archived = f"{output_path}.archived_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            os.rename(output_path, archived)
            print(f"[export] archived existing output -> {archived}", file=sys.stderr)
        except OSError as exc:
            print(f"[export] Warning: failed to archive {output_path}: {exc}", file=sys.stderr)

    if os.path.exists(output_path) and append:
        try:
            with np.load(output_path, allow_pickle=True) as data:
                if "features" in data:
                    existing_features = data["features"]
                    existing_globals = data["globals"]
                    existing_values = data["values"]
                    existing_policy_indices = data["policy_indices"]
                    existing_policy_values = data["policy_values"]

                    has_mp = "values_mp" in data and "num_players" in data
                    if has_mp:
                        existing_values_mp = data["values_mp"]
                        existing_num_players = data["num_players"]
                        values_mp_arr = np.concatenate([existing_values_mp, values_mp_arr], axis=0)
                        num_players_arr = np.concatenate([existing_num_players, num_players_arr], axis=0)
                    else:
                        write_mp = False

                    features_arr = np.concatenate([existing_features, features_arr], axis=0)
                    globals_arr = np.concatenate([existing_globals, globals_arr], axis=0)
                    values_arr = np.concatenate([existing_values, values_arr], axis=0)
                    policy_indices_arr = np.concatenate([existing_policy_indices, policy_indices_arr], axis=0)
                    policy_values_arr = np.concatenate([existing_policy_values, policy_values_arr], axis=0)
                    # Handle victory_types if present in existing data
                    if "victory_types" in data:
                        existing_victory_types = data["victory_types"]
                        victory_types_arr = np.concatenate([existing_victory_types, victory_types_arr], axis=0)
                    print(f"Appended to existing dataset at {output_path}; new total samples: {values_arr.shape[0]}")
        except Exception as exc:
            print(f"Warning: failed to append to existing {output_path}: {exc}")

    save_kwargs = {
        "features": features_arr,
        "globals": globals_arr,
        "values": values_arr,
        "policy_indices": policy_indices_arr,
        "policy_values": policy_values_arr,
        "board_type": np.asarray(board_type.value),
        "board_size": np.asarray(int(features_arr.shape[-1])),
        "history_length": np.asarray(int(history_length)),
        "feature_version": np.asarray(int(feature_version)),
        "policy_encoding": np.asarray("board_aware" if use_board_aware_encoding else "legacy_max_n"),
        "move_numbers": move_numbers_arr,
        "total_game_moves": total_game_moves_arr,
        "phases": phases_arr,
        "victory_types": victory_types_arr,  # For victory-type-balanced sampling
    }
    if write_mp:
        save_kwargs.update({"values_mp": values_mp_arr, "num_players": num_players_arr})

    np.savez_compressed(output_path, **save_kwargs)

    print(f"Exported {features_arr.shape[0]} samples from {games_processed} games "
          f"({games_deduplicated} deduplicated) into {output_path}")


def export_replay_dataset(
    db_path: str,
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    feature_version: int = 2,
    sample_every: int = 1,
    max_games: int | None = None,
    require_completed: bool = False,
    min_moves: int | None = None,
    max_moves: int | None = None,
    max_move_index: int | None = None,
    use_rank_aware_values: bool = True,
    parity_fixtures_dir: str | None = None,
    exclude_recovery: bool = False,
    use_board_aware_encoding: bool = False,
    append: bool = False,
    encoder_version: str = "default",
    require_moves: bool = True,
) -> None:
    """
    Export training samples from a single GameReplayDB into an NPZ dataset.

    This is a convenience wrapper around export_replay_dataset_multi for
    single-database exports. For multi-source exports with deduplication,
    use export_replay_dataset_multi directly.
    """
    # Delegate to multi-source function with single DB
    export_replay_dataset_multi(
        db_paths=[db_path],
        board_type=board_type,
        num_players=num_players,
        output_path=output_path,
        history_length=history_length,
        feature_version=feature_version,
        sample_every=sample_every,
        max_games=max_games,
        require_completed=require_completed,
        min_moves=min_moves,
        max_moves=max_moves,
        max_move_index=max_move_index,
        use_rank_aware_values=use_rank_aware_values,
        parity_fixtures_dir=parity_fixtures_dir,
        exclude_recovery=exclude_recovery,
        use_board_aware_encoding=use_board_aware_encoding,
        append=append,
        require_moves=require_moves,
        encoder_version=encoder_version,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NN training samples from existing GameReplayDB replays.",
    )
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        dest="db_paths",
        required=True,
        help=(
            "Path to a GameReplayDB SQLite file. Can be specified multiple times "
            "for multi-source export with automatic deduplication by game_id. "
            "Example: --db db1.db --db db2.db"
        ),
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hex8", "hexagonal"],
        required=True,
        help="Board type to filter games by.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        required=True,
        help="Number of players to filter games by.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npz dataset.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append to an existing output NPZ if present (legacy behavior). "
            "Default is to archive any existing output and rebuild from scratch."
        ),
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of past feature frames to stack (default: 3).",
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
        "--sample-every",
        type=int,
        default=1,
        help="Use every Nth move as a training sample (default: 1 = every move).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to process (default: all).",
    )
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="Only include games that completed normally (not timeout/disconnect).",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=None,
        help="Minimum move count to include a game (filters out trivially short games).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum move count to include a game (filters out abnormally long games).",
    )
    parser.add_argument(
        "--max-move-index",
        type=int,
        default=None,
        help=(
            "Maximum move index to sample within each game (limits replay depth). "
            "Use this to speed up export for games with many moves by only sampling "
            "early-game positions where replay is fast. E.g., --max-move-index 100 "
            "only samples moves 0-100 regardless of game length."
        ),
    )
    parser.add_argument(
        "--no-rank-aware-values",
        action="store_true",
        help="Use binary winner/loser values instead of rank-aware values for multiplayer.",
    )
    parser.add_argument(
        "--parity-fixtures-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing TS↔Python replay parity fixtures "
            "(as produced by scripts/check_ts_python_replay_parity.py "
            "with --emit-fixtures-dir). When provided, export only uses "
            "pre-divergence states per game."
        ),
    )
    parser.add_argument(
        "--exclude-recovery",
        action="store_true",
        help=(
            "Exclude games that contain recovery slide moves. "
            "Use this for training data purity when recovery rules have changed."
        ),
    )
    parser.add_argument(
        "--board-aware-encoding",
        action="store_true",
        help=(
            "Use board-specific policy encoding (compact action space). "
            "square8: 7000 actions, square19: 67000 actions. "
            "Recommended for new training runs. Legacy default uses ~55000 actions "
            "for all square boards."
        ),
    )
    parser.add_argument(
        "--encoder-version",
        type=str,
        choices=["default", "v2", "v3"],
        default="default",
        help=(
            "Encoder version for hex boards. "
            "'default' maps to 'v3' (recommended), "
            "'v2' uses HexStateEncoder (10 channels for HexNeuralNet_v2), "
            "'v3' uses HexStateEncoderV3 (16 channels for HexNeuralNet_v3). "
            "Hex boards ALWAYS use specialized encoders for consistent shapes."
        ),
    )
    parser.add_argument(
        "--no-require-moves",
        action="store_true",
        help=(
            "Disable the require_moves filter. By default, only games with "
            "actual move data in the game_moves table are included. Use this "
            "flag to include games without move data (they will be skipped "
            "anyway, but without this optimization the query is slower)."
        ),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "Enable incremental export caching. Skips export if source DBs "
            "haven't changed since last export. Significantly speeds up "
            "repeated training runs."
        ),
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help=(
            "Force re-export even if cache indicates no changes. "
            "Use with --use-cache to rebuild cache."
        ),
    )
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help=(
            "Disable parallel encoding and use single-threaded mode. "
            "Default is parallel mode which is 10-20x faster on multi-core systems."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for parallel mode (default: CPU count - 1). "
            "Set to 1 for single-threaded mode (same as --single-threaded)."
        ),
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help=(
            "Allow exporting from non-canonical DBs even when the output name "
            "starts with canonical_. Use for legacy/experimental datasets only."
        ),
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help=(
            "Allow DBs marked pending_gate in TRAINING_DATA_REGISTRY.md "
            "(still requires gate summary to be canonical_ok when present)."
        ),
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to TRAINING_DATA_REGISTRY.md (default: repo root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    board_type = BOARD_TYPE_MAP[args.board_type]
    if args.history_length < 0:
        raise ValueError("--history-length must be >= 0")
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

    _enforce_canonical_db_policy(
        args.db_paths,
        args.output,
        allow_noncanonical=bool(args.allow_noncanonical),
    )
    # Use central canonical source validation
    allowed_statuses = ["canonical", "pending_gate"] if args.allow_pending_gate else ["canonical"]
    enforce_canonical_sources(
        [Path(p) for p in args.db_paths],
        registry_path=Path(args.registry) if args.registry else None,
        allowed_statuses=allowed_statuses,
        allow_noncanonical=bool(args.allow_noncanonical),
        error_prefix="export-replay-dataset",
    )

    # Determine parallelism: default is parallel unless --single-threaded or --workers=1
    use_parallel = not args.single_threaded and (args.workers is None or args.workers > 1)

    # Use parallel export by default (10-20x faster on multi-core systems)
    if use_parallel:
        from scripts.export_replay_dataset_parallel import export_parallel
        num_workers = args.workers
        if num_workers is None:
            import os
            num_workers = max(1, (os.cpu_count() or 4) - 1)
        print(f"[PARALLEL] Using {num_workers} worker processes for encoding")
        result = export_parallel(
            db_paths=args.db_paths,
            board_type=board_type,
            num_players=args.num_players,
            output_path=args.output,
            num_workers=num_workers,
            encoder_version=args.encoder_version,
            history_length=args.history_length,
            feature_version=args.feature_version,
            sample_every=args.sample_every,
            max_games=args.max_games,
            require_completed=args.require_completed,
            min_moves=args.min_moves,
            max_moves=args.max_moves,
            use_board_aware_encoding=args.board_aware_encoding,
            require_moves=not args.no_require_moves,
            use_cache=args.use_cache,
            force_export=args.force_export,
        )
        return 0 if result else 1

    # Single-threaded mode (legacy, for debugging or when parallelism causes issues)
    print("[SINGLE-THREADED] Using sequential encoding (use --workers N for parallel mode)")

    # Check cache if enabled
    if args.use_cache:
        cache = get_export_cache()
        if not cache.needs_export(
            db_paths=args.db_paths,
            output_path=args.output,
            board_type=args.board_type,
            num_players=args.num_players,
            history_length=args.history_length,
            feature_version=args.feature_version,
            policy_encoding="board_aware" if args.board_aware_encoding else "legacy_max_n",
            force=args.force_export,
        ):
            cache_info = cache.get_cache_info(
                args.output,
                args.board_type,
                args.num_players,
                history_length=args.history_length,
                feature_version=args.feature_version,
                policy_encoding="board_aware" if args.board_aware_encoding else "legacy_max_n",
            )
            samples = cache_info.get("samples_exported", "?") if cache_info else "?"
            print("[CACHE HIT] Skipping export - source DBs unchanged since last export")
            print(f"  Output: {args.output}")
            print(f"  Cached samples: {samples}")
            return 0
        print("[CACHE MISS] Export needed - source DBs have changed")

    # Use multi-source export with deduplication
    export_replay_dataset_multi(
        db_paths=args.db_paths,
        board_type=board_type,
        num_players=args.num_players,
        output_path=args.output,
        history_length=args.history_length,
        feature_version=args.feature_version,
        sample_every=args.sample_every,
        max_games=args.max_games,
        require_completed=args.require_completed,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        max_move_index=args.max_move_index,
        use_rank_aware_values=not args.no_rank_aware_values,
        parity_fixtures_dir=args.parity_fixtures_dir,
        exclude_recovery=args.exclude_recovery,
        use_board_aware_encoding=args.board_aware_encoding,
        append=bool(args.append),
        encoder_version=args.encoder_version,
        require_moves=not args.no_require_moves,
    )

    # Update cache if enabled
    if args.use_cache:
        # Read sample count from output file
        samples_exported = 0
        games_exported = 0
        try:
            with np.load(args.output, allow_pickle=True) as data:
                if "values" in data:
                    samples_exported = len(data["values"])
        except Exception:
            pass

        cache.record_export(
            db_paths=args.db_paths,
            output_path=args.output,
            board_type=args.board_type,
            num_players=args.num_players,
            history_length=args.history_length,
            feature_version=args.feature_version,
            policy_encoding="board_aware" if args.board_aware_encoding else "legacy_max_n",
            samples_exported=samples_exported,
            games_exported=games_exported,
        )
        print(f"[CACHE] Recorded export: {samples_exported} samples")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
