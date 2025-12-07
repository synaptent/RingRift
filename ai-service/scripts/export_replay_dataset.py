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

    # Basic: export square8 2p samples from a single DB
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \\
      python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --output data/training/from_replays.square8_2p.npz

    # Quality-filtered export with rank-aware values
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \\
      python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_3p.db \\
        --board-type square8 \\
        --num-players 3 \\
        --require-completed \\
        --min-moves 20 \\
        --output data/training/from_replays.square8_3p.npz
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.db import GameReplayDB
from app.models import AIConfig, BoardType, GameState, Move
from app.ai.neural_net import NeuralNetAI, INVALID_MOVE_INDEX


BOARD_TYPE_MAP: Dict[str, BoardType] = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}


def build_encoder(board_type: BoardType) -> NeuralNetAI:
    """
    Construct a NeuralNetAI instance for feature and policy encoding.

    This uses a lightweight AIConfig and treats player_number=1 purely as a
    placeholder; we never call select_move(), only the encoding helpers.
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
    # Ensure the encoder's board_size hint is consistent with the dataset.
    encoder.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)
    return encoder


def encode_state_with_history(
    encoder: NeuralNetAI,
    state: GameState,
    history_frames: List[np.ndarray],
    history_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a GameState + history into (stacked_features, globals_vec).

    This mirrors the stacking logic in NeuralNetAI.evaluate_batch /
    encode_state_for_model: current features followed by up to history_length
    previous feature frames, newest-first, padded with zeros as needed.
    """
    # Use the internal feature extractor; this is stable tooling code.
    features, globals_vec = encoder._extract_features(state)  # type: ignore[attr-defined]

    hist = history_frames[::-1][:history_length]
    while len(hist) < history_length:
        hist.append(np.zeros_like(features))

    stacked = np.concatenate([features] + hist, axis=0)
    return stacked.astype(np.float32), globals_vec.astype(np.float32)


def value_from_final_winner(final_state: GameState, perspective: int) -> float:
    """
    DEPRECATED: Use value_from_final_ranking for multiplayer support.
    Map final winner to a scalar value from the perspective of `perspective`.
    """
    winner = getattr(final_state, "winner", None)
    if winner is None:
        return 0.0
    if winner == perspective:
        return 1.0
    return -1.0


def value_from_final_ranking(
    final_state: GameState,
    perspective: int,
    num_players: int,
) -> float:
    """
    Compute rank-aware value from final game state.

    Maps final ranking to a scalar value in [-1, +1] using linear interpolation:
      - 1st place: +1
      - Last place: -1
      - Intermediate positions: linearly interpolated

    Formula: value = 1 - 2 * (rank - 1) / (num_players - 1)
      - 2-player: 1st=+1, 2nd=-1
      - 3-player: 1st=+1, 2nd=0, 3rd=-1
      - 4-player: 1st=+1, 2nd=+0.333, 3rd=-0.333, 4th=-1

    Ranking is determined by eliminated_rings (more = better), with ties broken
    by territory_spaces.

    Args:
        final_state: The completed game state
        perspective: Player number (1-indexed) to compute value for
        num_players: Total number of players in the game

    Returns:
        Value in [-1, +1] representing expected outcome for this player
    """
    winner = getattr(final_state, "winner", None)

    # Handle incomplete games or draws
    if winner is None or not final_state.players:
        return 0.0

    # For 2-player games, use simple winner/loser logic
    if num_players == 2:
        if winner == perspective:
            return 1.0
        return -1.0

    # For multiplayer, compute ranking based on eliminated_rings (primary)
    # and territory_spaces (tiebreaker)
    player_scores = []
    for player in final_state.players:
        # Score: eliminated rings as primary, territory as tiebreaker
        # Higher score = better ranking
        score = (player.eliminated_rings, player.territory_spaces)
        player_scores.append((player.number, score))

    # Sort by score descending (best first)
    player_scores.sort(key=lambda x: x[1], reverse=True)

    # Find rank of perspective player (1-indexed)
    rank = 1
    for i, (player_num, _) in enumerate(player_scores):
        if player_num == perspective:
            rank = i + 1
            break

    # Linear interpolation: 1st=+1, last=-1, intermediate=interpolated
    # value = 1 - 2 * (rank - 1) / (num_players - 1)
    if num_players <= 1:
        return 0.0

    value = 1.0 - 2.0 * (rank - 1) / (num_players - 1)
    return float(value)


def compute_multi_player_values(
    final_state: GameState,
    num_players: int,
    max_players: int = 4,
) -> List[float]:
    """
    Compute value vector for all player positions.

    This is used with RingRiftCNN_MultiPlayer which outputs values for all
    players simultaneously instead of just the current player's perspective.

    Args:
        final_state: The completed game state
        num_players: Number of active players in the game (2, 3, or 4)
        max_players: Maximum players the model supports (default: 4)

    Returns:
        List of values of length max_players, where:
        - Active players (0 to num_players-1) have values in [-1, +1]
        - Inactive slots are filled with 0.0

    Examples:
        >>> # 2-player game where P1 wins
        >>> values = compute_multi_player_values(state, num_players=2)
        >>> # values = [1.0, -1.0, 0.0, 0.0]

        >>> # 3-player game ranking P2, P1, P3
        >>> values = compute_multi_player_values(state, num_players=3)
        >>> # values = [0.0, 1.0, -1.0, 0.0]  (P2=1st, P1=2nd, P3=3rd)
    """
    # Initialize with zeros for inactive slots
    values = [0.0] * max_players

    winner = getattr(final_state, "winner", None)

    # Handle incomplete games
    if winner is None or not final_state.players:
        return values

    # Compute ranking based on eliminated_rings and territory_spaces
    player_scores = []
    for player in final_state.players:
        score = (player.eliminated_rings, player.territory_spaces)
        player_scores.append((player.number, score))

    # Sort by score descending (best = rank 1)
    player_scores.sort(key=lambda x: x[1], reverse=True)

    # Build rank lookup: player_number -> rank (1-indexed)
    player_ranks: Dict[int, int] = {}
    for rank, (player_num, _) in enumerate(player_scores, start=1):
        player_ranks[player_num] = rank

    # Compute value for each active player position
    for player in final_state.players:
        player_idx = player.number - 1  # 0-indexed for array
        if player_idx >= max_players:
            continue

        rank = player_ranks.get(player.number, num_players)

        if num_players <= 1:
            values[player_idx] = 0.0
        else:
            # Linear interpolation: 1st=+1, last=-1
            values[player_idx] = 1.0 - 2.0 * (rank - 1) / (num_players - 1)

    return values


def export_replay_dataset(
    db_path: str,
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    sample_every: int = 1,
    max_games: Optional[int] = None,
    require_completed: bool = False,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    use_rank_aware_values: bool = True,
    parity_fixtures_dir: Optional[str] = None,
) -> None:
    """
    Export training samples from a single GameReplayDB into an NPZ dataset.

    Args:
        db_path: Path to GameReplayDB SQLite file
        board_type: Board type to filter games by
        num_players: Number of players to filter games by
        output_path: Path to output .npz dataset
        history_length: Number of past feature frames to stack (default: 3)
        sample_every: Use every Nth move as a training sample (default: 1)
        max_games: Optional cap on number of games to process
        require_completed: If True, only include games with normal termination
        min_moves: Minimum move count to include a game
        max_moves: Maximum move count to include a game
        use_rank_aware_values: If True, use rank-based values for multiplayer;
            if False, use binary winner/loser values (default: True)
    """
    db = GameReplayDB(db_path)
    encoder = build_encoder(board_type)

    features_list: List[np.ndarray] = []
    globals_list: List[np.ndarray] = []
    values_list: List[float] = []
    # Optional vector value targets and num_players metadata for v2
    values_mp_list: List[np.ndarray] = []
    num_players_list: List[int] = []
    policy_indices_list: List[np.ndarray] = []
    policy_values_list: List[np.ndarray] = []

    # Metadata for weighted sampling
    move_numbers_list: List[int] = []
    total_game_moves_list: List[int] = []
    phases_list: List[str] = []

    # Build query filters for quality-based selection
    query_filters: Dict[str, Any] = {
        "board_type": board_type,
        "num_players": num_players,
    }
    if require_completed:
        # Only include games that completed normally (not timeout/disconnect)
        query_filters["termination_reason"] = "env_done_flag"
    if min_moves is not None:
        query_filters["min_moves"] = min_moves
    if max_moves is not None:
        query_filters["max_moves"] = max_moves

    # Optional parity cutoffs derived from TS↔Python replay parity fixtures.
    # When provided, we truncate each game's usable move range to strictly
    # pre-divergence states so that exported samples only come from positions
    # where TS and Python agree on the replayed state (up to the first
    # detected mismatch).
    parity_cutoffs: Dict[str, int] = {}
    if parity_fixtures_dir:
        fixtures_path = os.path.abspath(parity_fixtures_dir)
        if os.path.isdir(fixtures_path):
            for name in os.listdir(fixtures_path):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(fixtures_path, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        fixture = json.load(f)
                except Exception:
                    continue

                game_id = fixture.get("game_id")
                diverged_at = fixture.get("diverged_at")
                # diverged_at refers to TS step index k (state AFTER move k-1).
                # Python get_state_at_move(k-1) first mismatches at this step.
                # States used as inputs in this exporter are:
                #   state_before(move_index=m) = get_state_at_move(m-1)
                # So we require (m-1) < (diverged_at - 1) → m <= diverged_at - 1.
                if (
                    not isinstance(game_id, str)
                    or not isinstance(diverged_at, int)
                    or diverged_at <= 0
                ):
                    continue

                safe_max_move = diverged_at - 1
                prev = parity_cutoffs.get(game_id)
                if prev is None or safe_max_move < prev:
                    parity_cutoffs[game_id] = safe_max_move

    games_iter = db.iterate_games(**query_filters)

    # Log filter configuration
    filter_desc = []
    if require_completed:
        filter_desc.append("completed games only")
    if min_moves is not None:
        filter_desc.append(f"min {min_moves} moves")
    if max_moves is not None:
        filter_desc.append(f"max {max_moves} moves")
    if filter_desc:
        print(f"Quality filters: {', '.join(filter_desc)}")
    print(f"Value targets: {'rank-aware' if use_rank_aware_values else 'binary winner/loser'}")

    games_processed = 0
    games_skipped = 0
    for meta, initial_state, moves in games_iter:
        game_id = meta.get("game_id")
        total_moves = int(meta.get("total_moves", len(moves)))
        if total_moves <= 0 or not moves:
            continue

        # If parity cutoffs are available, truncate this game to a safe
        # pre-divergence prefix, dropping any samples whose state_before
        # would come from a divergent replay step.
        max_safe_move_index: Optional[int] = None
        if parity_cutoffs:
            cutoff = parity_cutoffs.get(game_id)
            if cutoff is not None:
                max_safe_move_index = cutoff
                if max_safe_move_index <= 0:
                    # Divergence at or before the first post-initial step;
                    # skip this game entirely as there is no safe prefix.
                    games_skipped += 1
                    continue

        # Compute final state once for value targets.
        final_state_index = total_moves - 1
        if max_safe_move_index is not None:
            # Ensure the value target at least reflects a state that is
            # consistent between TS and Python replays. We conservatively
            # use the last safe prefix state when available.
            final_state_index = min(final_state_index, max_safe_move_index)

        final_state = db.get_state_at_move(game_id, final_state_index)
        if final_state is None:
            continue

        # Determine number of players in this game and precompute
        # multi-player value vector for v2 targets.
        num_players_in_game = len(final_state.players)
        if use_rank_aware_values:
            # Use the same rank-aware semantics as scalar values but
            # computed for all player positions at once.
            values_vec = np.asarray(
                compute_multi_player_values(
                    final_state,
                    num_players=num_players_in_game,
                ),
                dtype=np.float32,
            )
        else:
            # Fallback: use winner/loser style values per player.
            values_vec = np.zeros(4, dtype=np.float32)
            for p in final_state.players:
                base = value_from_final_winner(final_state, p.number)
                values_vec[p.number - 1] = float(base)

        history_frames: List[np.ndarray] = []

        for move_index, move in enumerate(moves):
            # Respect parity cutoffs when present: stop sampling once we
            # reach the first move whose state_before would rely on a
            # divergent replay step.
            if max_safe_move_index is not None and move_index > max_safe_move_index:
                break

            if sample_every > 1 and (move_index % sample_every) != 0:
                continue

            # State BEFORE this move: initial_state for move 0, otherwise
            # state after the previous move.
            if move_index == 0:
                state_before = initial_state
            else:
                state_before = db.get_state_at_move(game_id, move_index - 1)
                if state_before is None:
                    break

            # Encode features + globals with history.
            stacked, globals_vec = encode_state_with_history(
                encoder,
                state_before,
                history_frames,
                history_length=history_length,
            )

            # Update history with the base features for this state.
            base_features, _ = encoder._extract_features(state_before)  # type: ignore[attr-defined]
            history_frames.append(base_features)
            if len(history_frames) > history_length + 1:
                history_frames.pop(0)

            # Encode the action taken at this state.
            idx = encoder.encode_move(move, state_before.board)
            if idx == INVALID_MOVE_INDEX:
                continue

            # Value from the perspective of the player to move at this state
            # (scalar) plus vector value for all players.
            if use_rank_aware_values:
                value = value_from_final_ranking(
                    final_state,
                    perspective=state_before.current_player,
                    num_players=num_players,
                )
            else:
                value = value_from_final_winner(
                    final_state,
                    perspective=state_before.current_player,
                )

            features_list.append(stacked)
            globals_list.append(globals_vec)
            values_list.append(float(value))
            policy_indices_list.append(np.array([idx], dtype=np.int32))
            policy_values_list.append(np.array([1.0], dtype=np.float32))
            # Vector value and num_players metadata reuse the precomputed
            # game-level values_vec / num_players_in_game.
            values_mp_list.append(values_vec)
            num_players_list.append(num_players_in_game)

            # Capture metadata for weighted sampling
            move_numbers_list.append(move_index)
            total_game_moves_list.append(total_moves)
            phase_str = (
                str(state_before.current_phase.value)
                if hasattr(state_before.current_phase, "value")
                else str(state_before.current_phase)
            )
            phases_list.append(phase_str)

        games_processed += 1
        if max_games is not None and games_processed >= max_games:
            break

    if not features_list:
        print(f"No samples generated from {db_path} (board={board_type}, players={num_players}).")
        return

    # Stack into arrays; policies remain sparse per-sample arrays.
    features_arr = np.stack(features_list, axis=0).astype(np.float32)
    globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
    values_arr = np.array(values_list, dtype=np.float32)
    policy_indices_arr = np.array(policy_indices_list, dtype=object)
    policy_values_arr = np.array(policy_values_list, dtype=object)
    # Multi-player value vectors and num_players metadata (v2 targets)
    values_mp_arr = np.stack(values_mp_list, axis=0).astype(np.float32)
    num_players_arr = np.array(num_players_list, dtype=np.int32)

    # Metadata arrays for weighted sampling
    move_numbers_arr = np.array(move_numbers_list, dtype=np.int32)
    total_game_moves_arr = np.array(total_game_moves_list, dtype=np.int32)
    phases_arr = np.array(phases_list, dtype=object)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Append to existing dataset if present, mirroring generate_data.py.
    write_mp = True
    if os.path.exists(output_path):
        try:
            with np.load(output_path, allow_pickle=True) as data:
                if "features" in data:
                    existing_features = data["features"]
                    existing_globals = data["globals"]
                    existing_values = data["values"]
                    existing_policy_indices = data["policy_indices"]
                    existing_policy_values = data["policy_values"]

                    # Optional multi-player targets / metadata from previous
                    # runs. Only keep writing these when the existing file
                    # already has them to avoid shape mismatches with legacy
                    # datasets.
                    has_mp = (
                        "values_mp" in data
                        and "num_players" in data
                    )
                    if has_mp:
                        existing_values_mp = data["values_mp"]
                        existing_num_players = data["num_players"]

                        values_mp_arr = np.concatenate(
                            [existing_values_mp, values_mp_arr],
                            axis=0,
                        )
                        num_players_arr = np.concatenate(
                            [existing_num_players, num_players_arr],
                            axis=0,
                        )
                    else:
                        write_mp = False

                    features_arr = np.concatenate(
                        [existing_features, features_arr],
                        axis=0,
                    )
                    globals_arr = np.concatenate(
                        [existing_globals, globals_arr],
                        axis=0,
                    )
                    values_arr = np.concatenate(
                        [existing_values, values_arr],
                        axis=0,
                    )
                    policy_indices_arr = np.concatenate(
                        [existing_policy_indices, policy_indices_arr],
                        axis=0,
                    )
                    policy_values_arr = np.concatenate(
                        [existing_policy_values, policy_values_arr],
                        axis=0,
                    )
                    print(
                        f"Appended to existing dataset at {output_path}; "
                        f"new total samples: {values_arr.shape[0]}"
                    )
        except Exception as exc:
            print(f"Warning: failed to append to existing {output_path}: {exc}")

    save_kwargs = {
        "features": features_arr,
        "globals": globals_arr,
        "values": values_arr,
        "policy_indices": policy_indices_arr,
        "policy_values": policy_values_arr,
        # Metadata for weighted sampling (optional, used by WeightedRingRiftDataset)
        "move_numbers": move_numbers_arr,
        "total_game_moves": total_game_moves_arr,
        "phases": phases_arr,
    }
    if write_mp:
        save_kwargs.update(
            {
                "values_mp": values_mp_arr,
                "num_players": num_players_arr,
            }
        )

    np.savez_compressed(output_path, **save_kwargs)

    print(
        f"Exported {features_arr.shape[0]} samples "
        f"from {games_processed} games into {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NN training samples from existing GameReplayDB replays.",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to a GameReplayDB SQLite file (e.g. data/games/selfplay_square8_2p.db).",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal"],
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
        "--history-length",
        type=int,
        default=3,
        help="Number of past feature frames to stack (default: 3).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    board_type = BOARD_TYPE_MAP[args.board_type]
    if args.history_length < 0:
        raise ValueError("--history-length must be >= 0")
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

    export_replay_dataset(
        db_path=args.db,
        board_type=board_type,
        num_players=args.num_players,
        output_path=args.output,
        history_length=args.history_length,
        sample_every=args.sample_every,
        max_games=args.max_games,
        require_completed=args.require_completed,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        use_rank_aware_values=not args.no_rank_aware_values,
        parity_fixtures_dir=args.parity_fixtures_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
