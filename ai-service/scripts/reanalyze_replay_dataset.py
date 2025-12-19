#!/usr/bin/env python
"""Reanalyze replay games with search-derived soft policy targets.

Mirrors `scripts/export_replay_dataset.py` but replaces 1-hot played policy
targets with search outputs:
  - `mcts_visits`: MCTS visit-count distribution.
  - `descent_softmax`: Softmax over Descent root child values.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

from app.db import GameReplayDB
from app.models import AIConfig, BoardType, GameState, Move
from app.ai.neural_net import NeuralNetAI, INVALID_MOVE_INDEX, encode_move_for_board
from app.training.encoding import get_encoder_for_board_type

from scripts.export_replay_dataset import (
    BOARD_TYPE_MAP,
    encode_state_with_history,
    value_from_final_winner,
    value_from_final_ranking,
    compute_multi_player_values,
)


def _build_encoder(
    board_type: BoardType,
    nn_model_id: Optional[str],
    encoder_version: str = "v3",
    feature_version: int = 2,
) -> NeuralNetAI:
    os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")
    cfg = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        nn_model_id=nn_model_id,
        use_neural_net=True,
    )
    enc = NeuralNetAI(player_number=1, config=cfg)
    enc.feature_version = int(feature_version)
    enc.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEX8: 9,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)

    # For hex boards, attach specialized encoder for consistent feature shapes
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        effective_version = encoder_version if encoder_version in ("v2", "v3") else "v3"
        enc._hex_encoder = get_encoder_for_board_type(
            board_type,
            effective_version,
            feature_version=feature_version,
        )
        enc._hex_encoder_version = effective_version

    return enc


def _encode_sparse_policy_from_moves(
    encoder: NeuralNetAI,
    state: GameState,
    moves: Sequence[Move],
    probs: Sequence[float],
    *,
    use_board_aware_encoding: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    p_indices: List[int] = []
    p_values: List[float] = []
    for move, prob in zip(moves, probs):
        idx = (
            encode_move_for_board(move, state.board)
            if use_board_aware_encoding
            else encoder.encode_move(move, state.board)
        )
        if idx == INVALID_MOVE_INDEX:
            continue
        p_indices.append(int(idx))
        p_values.append(float(prob))
    if p_values:
        total = float(sum(p_values))
        if total > 0:
            p_values = [v / total for v in p_values]
    return np.array(p_indices, dtype=np.int32), np.array(p_values, dtype=np.float32)


def _reanalyze_mcts_policy(
    ai: Any,
    state: GameState,
    encoder: NeuralNetAI,
    *,
    use_board_aware_encoding: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    ai.select_move(state)
    moves, probs = ai.get_visit_distribution()
    if not moves:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    return _encode_sparse_policy_from_moves(
        encoder,
        state,
        moves,
        probs,
        use_board_aware_encoding=use_board_aware_encoding,
    )


def _reanalyze_descent_policy(
    ai: Any,
    state: GameState,
    encoder: NeuralNetAI,
    *,
    temperature: float,
    use_board_aware_encoding: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    ai.select_move(state)
    state_key = ai._get_state_key(state)
    entry = ai.transposition_table.get(state_key)
    if entry is None:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    _cur, children_values, _status, _rem, _visits = ai._unpack_tt_entry(entry)
    if not children_values:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    child_moves: List[Move] = []
    child_vals: List[float] = []
    for data in children_values.values():
        child_moves.append(data[0])
        child_vals.append(float(data[1]))
    if not child_moves:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    temp = float(temperature) if temperature and temperature > 0 else 1.0
    logits = np.asarray(child_vals, dtype=np.float32) / temp
    logits = logits - float(np.max(logits))
    exps = np.exp(logits)
    probs = exps / float(np.sum(exps))
    return _encode_sparse_policy_from_moves(
        encoder,
        state,
        child_moves,
        probs.tolist(),
        use_board_aware_encoding=use_board_aware_encoding,
    )


def reanalyze_replay_dataset(
    db_path: str,
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    feature_version: int = 2,
    sample_every: int = 1,
    max_games: Optional[int] = None,
    require_completed: bool = False,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    use_rank_aware_values: bool = True,
    parity_fixtures_dir: Optional[str] = None,
    exclude_recovery: bool = False,
    policy_target: str = "mcts_visits",
    policy_search_think_time_ms: int = 50,
    policy_temperature: float = 1.0,
    nn_model_id: Optional[str] = None,
    use_board_aware_encoding: bool = False,
) -> None:
    if policy_target not in {"mcts_visits", "descent_softmax"}:
        raise ValueError("policy_target must be mcts_visits or descent_softmax")

    db = GameReplayDB(db_path)
    encoder = _build_encoder(board_type, nn_model_id, feature_version=feature_version)

    mcts_ais: Dict[int, Any] = {}
    descent_ais: Dict[int, Any] = {}

    def _get_mcts_ai(perspective: int) -> Any:
        if perspective not in mcts_ais:
            from app.ai.mcts_ai import MCTSAI
            cfg = AIConfig(
                difficulty=7,
                think_time=policy_search_think_time_ms,
                randomness=0.0,
                nn_model_id=nn_model_id,
                use_neural_net=True,
                self_play=False,
            )
            mcts_ais[perspective] = MCTSAI(perspective, cfg)
        return mcts_ais[perspective]

    def _get_descent_ai(perspective: int) -> Any:
        if perspective not in descent_ais:
            from app.ai.descent_ai import DescentAI
            cfg = AIConfig(
                difficulty=9,
                think_time=policy_search_think_time_ms,
                randomness=0.0,
                nn_model_id=nn_model_id,
                use_neural_net=True,
            )
            descent_ais[perspective] = DescentAI(perspective, cfg)
        return descent_ais[perspective]

    features_list: List[np.ndarray] = []
    globals_list: List[np.ndarray] = []
    values_list: List[float] = []
    values_mp_list: List[np.ndarray] = []
    num_players_list: List[int] = []
    policy_indices_list: List[np.ndarray] = []
    policy_values_list: List[np.ndarray] = []

    move_numbers_list: List[int] = []
    total_game_moves_list: List[int] = []
    phases_list: List[str] = []

    query_filters: Dict[str, Any] = {
        "board_type": board_type,
        "num_players": num_players,
    }
    if require_completed:
        query_filters["termination_reason"] = "env_done_flag"
    if min_moves is not None:
        query_filters["min_moves"] = min_moves
    if max_moves is not None:
        query_filters["max_moves"] = max_moves

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
                if not isinstance(game_id, str) or not isinstance(diverged_at, int) or diverged_at <= 0:
                    continue
                safe_max_move = diverged_at - 1
                prev = parity_cutoffs.get(game_id)
                if prev is None or safe_max_move < prev:
                    parity_cutoffs[game_id] = safe_max_move

    games_iter = db.iterate_games(**query_filters)
    games_processed = 0

    for meta, initial_state, moves in games_iter:
        game_id = meta.get("game_id")
        total_moves = int(meta.get("total_moves", len(moves)))
        if total_moves <= 0 or not moves:
            continue

        if exclude_recovery:
            if any("recovery" in str(m.get("type", "")).lower() for m in moves):
                continue

        max_safe_move_index: Optional[int] = None
        if parity_cutoffs:
            cutoff = parity_cutoffs.get(game_id)
            if cutoff is not None:
                max_safe_move_index = cutoff
                if max_safe_move_index <= 0:
                    continue

        final_state_index = total_moves - 1
        if max_safe_move_index is not None:
            final_state_index = min(final_state_index, max_safe_move_index)
        final_state = db.get_state_at_move(game_id, final_state_index)
        if final_state is None:
            continue

        num_players_in_game = len(final_state.players)
        if use_rank_aware_values:
            values_vec = np.asarray(
                compute_multi_player_values(final_state, num_players=num_players_in_game),
                dtype=np.float32,
            )
        else:
            values_vec = np.zeros(4, dtype=np.float32)
            for p in final_state.players:
                values_vec[p.number - 1] = float(value_from_final_winner(final_state, p.number))

        history_frames: List[np.ndarray] = []
        for move_index, move in enumerate(moves):
            if max_safe_move_index is not None and move_index > max_safe_move_index:
                break
            if sample_every > 1 and (move_index % sample_every) != 0:
                continue
            state_before = initial_state if move_index == 0 else db.get_state_at_move(game_id, move_index - 1)
            if state_before is None:
                break

            stacked, globals_vec = encode_state_with_history(
                encoder, state_before, history_frames, history_length=history_length
            )
            base_features, _ = encoder._extract_features(state_before)  # type: ignore[attr-defined]
            history_frames.append(base_features)
            if len(history_frames) > history_length + 1:
                history_frames.pop(0)

            played_idx = (
                encode_move_for_board(move, state_before.board)
                if use_board_aware_encoding
                else encoder.encode_move(move, state_before.board)
            )
            if played_idx == INVALID_MOVE_INDEX:
                continue

            if use_rank_aware_values:
                value = value_from_final_ranking(
                    final_state,
                    perspective=state_before.current_player,
                    num_players=num_players,
                )
            else:
                value = value_from_final_winner(
                    final_state, perspective=state_before.current_player
                )

            if policy_target == "mcts_visits":
                ai = _get_mcts_ai(state_before.current_player)
                p_indices, p_values = _reanalyze_mcts_policy(
                    ai,
                    state_before,
                    encoder,
                    use_board_aware_encoding=use_board_aware_encoding,
                )
            else:
                ai = _get_descent_ai(state_before.current_player)
                p_indices, p_values = _reanalyze_descent_policy(
                    ai,
                    state_before,
                    encoder,
                    temperature=policy_temperature,
                    use_board_aware_encoding=use_board_aware_encoding,
                )

            if p_indices.size == 0:
                p_indices = np.array([played_idx], dtype=np.int32)
                p_values = np.array([1.0], dtype=np.float32)

            features_list.append(stacked)
            globals_list.append(globals_vec)
            values_list.append(float(value))
            policy_indices_list.append(p_indices)
            policy_values_list.append(p_values)
            values_mp_list.append(values_vec)
            num_players_list.append(num_players_in_game)

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
        print(f"No samples generated from {db_path}.")
        return

    policy_encoding = "board_aware" if use_board_aware_encoding else "legacy_max_n"
    np.savez_compressed(
        output_path,
        features=np.stack(features_list, axis=0).astype(np.float32),
        globals=np.stack(globals_list, axis=0).astype(np.float32),
        values=np.array(values_list, dtype=np.float32),
        policy_indices=np.array(policy_indices_list, dtype=object),
        policy_values=np.array(policy_values_list, dtype=object),
        values_mp=np.stack(values_mp_list, axis=0).astype(np.float32),
        num_players=np.array(num_players_list, dtype=np.int32),
        move_numbers=np.array(move_numbers_list, dtype=np.int32),
        total_game_moves=np.array(total_game_moves_list, dtype=np.int32),
        phases=np.array(phases_list, dtype=object),
        history_length=np.asarray(int(history_length)),
        feature_version=np.asarray(int(feature_version)),
        policy_encoding=np.asarray(policy_encoding),
    )

    print(
        f"Reanalyzed export: {len(features_list)} samples from {games_processed} games into {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--board-type", choices=["square8", "square19", "hex8", "hexagonal"], required=True)
    p.add_argument("--num-players", type=int, choices=[2, 3, 4], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--history-length", type=int, default=3)
    p.add_argument(
        "--feature-version",
        type=int,
        default=2,
        help=(
            "Feature encoding version for global feature layout (default: 2). "
            "Use 1 to keep legacy hex globals without chain/FE flags."
        ),
    )
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--require-completed", action="store_true")
    p.add_argument("--min-moves", type=int, default=None)
    p.add_argument("--max-moves", type=int, default=None)
    p.add_argument("--no-rank-aware-values", action="store_true")
    p.add_argument("--parity-fixtures-dir", type=str, default=None)
    p.add_argument("--exclude-recovery", action="store_true")
    p.add_argument("--policy-target", choices=["mcts_visits", "descent_softmax"], default="mcts_visits")
    p.add_argument("--policy-search-think-time-ms", type=int, default=50)
    p.add_argument("--policy-temperature", type=float, default=1.0)
    p.add_argument("--nn-model-id", type=str, default=None)
    p.add_argument(
        "--board-aware-encoding",
        action="store_true",
        help=(
            "Use board-specific policy encoding (compact action space). "
            "Recommended for v3 training on square boards."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    board_type = BOARD_TYPE_MAP[args.board_type]
    reanalyze_replay_dataset(
        db_path=args.db,
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
        use_rank_aware_values=not args.no_rank_aware_values,
        parity_fixtures_dir=args.parity_fixtures_dir,
        exclude_recovery=args.exclude_recovery,
        policy_target=args.policy_target,
        policy_search_think_time_ms=args.policy_search_think_time_ms,
        policy_temperature=args.policy_temperature,
        nn_model_id=args.nn_model_id,
        use_board_aware_encoding=bool(args.board_aware_encoding),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
