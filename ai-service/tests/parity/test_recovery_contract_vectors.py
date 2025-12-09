"""Parity contract vectors for recovery_action (RR-CANON-R110–R115).

These tests load the TS-authored recovery contract vectors and assert that
the Python GameEngine:
  - accepts the recovery_slide move,
  - applies it, and
  - produces the expected board deltas (collapsed spaces, markers removed).
"""

from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path

import pytest

from app.game_engine import GameEngine  # type: ignore[import]
from app.models import GameState, Move, MoveType, Position  # type: ignore[import]
from app.rules.recovery import apply_recovery_slide  # type: ignore[import]


VECTORS_PATH = Path(__file__).parent / "vectors" / "recovery_action.vectors.json"


def _convert_ts_state_to_python(state_dict: dict) -> dict:
    """Translate TS contract vector state into Python model shape."""
    result = copy.deepcopy(state_dict)

    stacks = result.get("board", {}).get("stacks", {})
    if isinstance(stacks, dict):
        for stack in stacks.values():
            rings = stack.get("rings")
            if isinstance(rings, list):
                # TS rings are top→bottom; Python expects bottom→top.
                stack["rings"] = list(reversed(rings))

    # Populate required top-level fields with sensible defaults when omitted.
    result.setdefault("id", result.get("gameId", "recovery-contract"))
    board_type = result.get("board", {}).get("type", "square8")
    result.setdefault("boardType", board_type)

    now_iso = datetime.now().isoformat()
    result.setdefault("createdAt", now_iso)
    result.setdefault("lastMoveAt", now_iso)
    result.setdefault(
        "timeControl",
        {"initialTime": 60, "increment": 0, "type": "untimed"},
    )
    result.setdefault("isRated", False)

    players = result.get("players") or []
    for idx, player in enumerate(players, start=1):
        player.setdefault("id", f"p{idx}")
        player.setdefault("username", f"p{idx}")
        player.setdefault("type", "human")
        player.setdefault("isReady", True)
        player.setdefault("timeRemaining", 60)
        player.setdefault("aiDifficulty", None)
    result["players"] = players

    result.setdefault("maxPlayers", len(players) or 2)
    result.setdefault("totalRingsInPlay", 0)
    result.setdefault("totalRingsEliminated", 0)
    result.setdefault("victoryThreshold", 3)
    result.setdefault("territoryVictoryThreshold", 10)

    return result


@pytest.mark.skipif(
    not VECTORS_PATH.exists(),
    reason="Recovery contract vectors not present. Run generate_test_vectors.py first.",
)
def test_recovery_contract_vectors_apply_successfully() -> None:
    """Ensure Python accepts and applies the TS recovery vectors."""
    payload = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    vectors = payload.get("vectors", [])
    assert vectors, "No recovery vectors found"

    for vector in vectors:
        ts_state = vector["input"]["state"]
        ts_move = dict(vector["input"]["initialMove"])

        # Map TS move field names to Python model aliases.
        if "option" in ts_move and "recoveryOption" not in ts_move:
            ts_move["recoveryOption"] = ts_move.pop("option")

        state = GameState(**_convert_ts_state_to_python(ts_state))
        move = Move(**ts_move)

        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        assert any(
            m.type == MoveType.RECOVERY_SLIDE
            and isinstance(m.from_pos, Position)
            and isinstance(move.from_pos, Position)
            and m.from_pos.x == move.from_pos.x
            and m.from_pos.y == move.from_pos.y
            and isinstance(m.to, Position)
            and isinstance(move.to, Position)
            and m.to.x == move.to.x
            and m.to.y == move.to.y
            and m.recovery_option == move.recovery_option
        for m in valid_moves), f"Move not enumerated for vector {vector['id']}"

        state_after = copy.deepcopy(state)
        outcome = apply_recovery_slide(
            state_after,
            move,
            option=getattr(move, "recovery_option", None),
            collapse_positions=getattr(move, "collapse_positions", None),
        )
        assert outcome.success, f"Recovery slide failed for {vector['id']}: {outcome.error}"

        board_changes = vector.get("assertions", {}).get("boardChanges", {})

        collapsed_expected = set(board_changes.get("collapsedSpacesAdded", []))
        if collapsed_expected:
            collapsed_actual = {
                pos.to_key() for pos in (outcome.collapsed_positions or [])
            }
            assert collapsed_expected == collapsed_actual, (
                f"Collapsed spaces mismatch for {vector['id']}: "
                f"expected {collapsed_expected}, got {collapsed_actual}"
            )

        markers_expected = board_changes.get("markersRemaining")
        if markers_expected is not None:
            # Canonical markers after collapse: remove collapsed positions
            collapsed_keys = {
                pos.to_key() for pos in (outcome.collapsed_positions or [])
            }
            marker_keys = {
                key for key in state_after.board.markers.keys() if key not in collapsed_keys
            }
            assert marker_keys == set(markers_expected), (
                f"Markers remaining mismatch for {vector['id']}: "
                f"expected {set(markers_expected)}, got {marker_keys}"
            )
