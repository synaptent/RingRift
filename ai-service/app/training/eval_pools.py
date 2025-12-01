"""Minimal loader for evaluation state pools (JSONL GameState snapshots).

The pools referenced here are intended primarily for heuristic training and
diagnostics in the multi-board, multi-start regime:

- The canonical ``"v1"`` pools for each board (Square8, Square19, Hexagonal)
  are biased toward **mid- and late-game** positions where heuristic features
  matter most. They are generated via long self-play soaks using
  :mod:`scripts.run_self_play_soak` with mid-game sampling.
- Multi-player variants use **distinct pool_ids** and directory roots so that
  2-player optimisation jobs never accidentally mix 3p/4p positions into their
  evaluation schedule.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import os

from app.models import GameState, BoardType  # type: ignore


# Canonical mapping from (BoardType, pool_id) to JSONL pool paths.
# 2-player CMA-ES / GA runs should use the ``"v1"`` pools below together with
# eval_mode="multi-start" so that evaluation is driven from fixed mid/late-game
# snapshots rather than only the empty starting position.
#
# Multi-player state pools are kept separate via explicit pool_ids so that
# callers must opt in to 3p/4p evaluation (for example,
# ``square19_3p_pool_v1`` or ``hex_4p_pool_v1``), and 2-player training code
# that passes the default pool_id="v1" will never see those states.
POOL_PATHS: Dict[Tuple[BoardType, str], str] = {
    # Canonical 2-player evaluation pools (mid/late-game heavy).
    (BoardType.SQUARE8, "v1"): "data/eval_pools/square8/pool_v1.jsonl",
    (BoardType.SQUARE19, "v1"): "data/eval_pools/square19/pool_v1.jsonl",
    (BoardType.HEXAGONAL, "v1"): "data/eval_pools/hex/pool_v1.jsonl",
    # Multi-player evaluation pools (diagnostic / experimental).
    (BoardType.SQUARE19, "square19_3p_pool_v1"): (
        "data/eval_pools/square19_3p/pool_v1.jsonl"
    ),
    (BoardType.HEXAGONAL, "hex_4p_pool_v1"): (
        "data/eval_pools/hex_4p/pool_v1.jsonl"
    ),
}


def load_state_pool(
    board_type: BoardType,
    pool_id: str = "v1",
    max_states: Optional[int] = None,
    num_players: Optional[int] = None,
) -> List[GameState]:
    """Load a deterministically ordered pool of GameState records.

    Parameters
    ----------
    board_type:
        BoardType for which to load a state pool.
    pool_id:
        Logical pool identifier (e.g. "v1" or a more specific label such
        as "square19_3p_pool_v1" for multi-player pools).
    max_states:
        Optional maximum number of states to load. If None, load all
        available states. If <= 0, return an empty list.
    num_players:
        Optional number of players to filter by. When provided, this
        helper enforces that every loaded state has exactly this many
        players (mirroring the strict board_type check below).

    Returns
    -------
    List[GameState]
        Parsed GameState instances in file order.

    Raises
    ------
    ValueError
        If no pool path is configured for (board_type, pool_id) or if the
        file contains states with a mismatched board_type.
    FileNotFoundError
        If the configured pool file does not exist.
    """
    key = (board_type, pool_id)
    if key not in POOL_PATHS:
        raise ValueError(
            f"No evaluation state pool configured for "
            f"board_type={board_type!r}, pool_id={pool_id!r}"
        )

    rel_path = POOL_PATHS[key]
    path = os.path.abspath(rel_path)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"State pool file not found at {path!r} "
            f"for board_type={board_type!r}, pool_id={pool_id!r}"
        )

    states: List[GameState] = []
    if max_states is not None and max_states <= 0:
        return states

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_states is not None and len(states) >= max_states:
                break
            line = line.strip()
            if not line:
                continue

            state = GameState.model_validate_json(
                line  # type: ignore[attr-defined]
            )

            if state.board_type != board_type:
                raise ValueError(
                    "State pool contains state with mismatched board_type: "
                    f"expected={board_type!r}, got={state.board_type!r}"
                )

            if num_players is not None:
                actual_players = len(state.players)
                if actual_players != num_players:
                    raise ValueError(
                        "State pool contains state with mismatched "
                        f"num_players: expected={num_players!r}, "
                        f"got={actual_players!r}"
                    )

            states.append(state)

    return states