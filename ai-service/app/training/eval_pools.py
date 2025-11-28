"""Minimal loader for evaluation state pools (JSONL GameState snapshots)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import os

from app.models import GameState, BoardType  # type: ignore


POOL_PATHS: Dict[Tuple[BoardType, str], str] = {
    (BoardType.SQUARE8, "v1"): "data/eval_pools/square8/pool_v1.jsonl",
    (BoardType.SQUARE19, "v1"): "data/eval_pools/square19/pool_v1.jsonl",
    (BoardType.HEXAGONAL, "v1"): "data/eval_pools/hex/pool_v1.jsonl",
}


def load_state_pool(
    board_type: BoardType,
    pool_id: str = "v1",
    max_states: Optional[int] = None,
) -> List[GameState]:
    """Load a deterministically ordered pool of GameState records.

    Parameters
    ----------
    board_type:
        BoardType for which to load a state pool.
    pool_id:
        Logical pool identifier (e.g. "v1").
    max_states:
        Optional maximum number of states to load. If None, load all
        available states. If <= 0, return an empty list.

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

            states.append(state)

    return states