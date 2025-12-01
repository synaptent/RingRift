"""Parity tests for TS ComparableSnapshot plateau fixtures (seeds 1 and 18).

These tests load the TS-generated ComparableSnapshot JSON fixtures for
square8 / 2 AI plateau states (seed 1 and seed 18), hydrate them into
Python GameState/BoardState models, then reconstruct a Python-side
ComparableSnapshot shape and assert deep equality with the TS snapshot.

This ensures that:

* TS → JSON snapshot export (tests/unit/ExportSeed1Snapshot.test.ts and
  tests/unit/ExportSeed18Snapshot.test.ts) produces a shape that can be
  losslessly mapped into Python models, and
* Python's models (BoardState, Player, GameState) can represent the same
  plateau states faithfully enough to reconstruct the original snapshot.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure app package is importable when running tests directly
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.models.core import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    Position,
    RingStack,
    TimeControl,
)


BASE_DIR = Path(__file__).resolve().parent
PARITY_DIR = BASE_DIR

SEED18_SNAPSHOT = PARITY_DIR / "square8_2p_seed18_plateau.snapshot.json"
SEED1_SNAPSHOT = PARITY_DIR / "square8_2p_seed1_plateau.snapshot.json"


def _build_board_from_snapshot(snapshot: Dict[str, Any]) -> BoardState:
    """Hydrate a Python BoardState from a TS ComparableSnapshot.

    This focuses on the fields that ComparableSnapshot cares about:
    stacks, markers, and collapsedSpaces. Other BoardState fields are
    initialised with empty/default values.
    """

    board_type = BoardType(snapshot["boardType"])

    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    elif board_type == BoardType.HEXAGONAL:
        # Mirror TS BOARD_CONFIGS for hexagonal boards (size = 11).
        # The underlying axial coordinates use a radius of 10; BoardState.size
        # reflects the backing array dimensions rather than radius.
        size = 11
    else:
        raise AssertionError(f"Unsupported boardType for plateau snapshot: {board_type}")

    stacks: Dict[str, RingStack] = {}
    for entry in snapshot.get("stacks", []):
        key = entry["key"]
        parts = [int(p) for p in key.split(",")]
        if len(parts) == 2:
            pos = Position(x=parts[0], y=parts[1])
        elif len(parts) == 3:
            pos = Position(x=parts[0], y=parts[1], z=parts[2])
        else:
            raise AssertionError(f"Unexpected stack key format: {key}")

        stack = RingStack(
            position=pos,
            rings=list(entry["rings"]),
            stack_height=entry["stackHeight"],
            cap_height=entry["capHeight"],
            controlling_player=entry["controllingPlayer"],
        )
        stacks[key] = stack

    markers: Dict[str, Any] = {}
    for entry in snapshot.get("markers", []):
        key = entry["key"]
        parts = [int(p) for p in key.split(",")]
        if len(parts) == 2:
            pos = Position(x=parts[0], y=parts[1])
        elif len(parts) == 3:
            pos = Position(x=parts[0], y=parts[1], z=parts[2])
        else:
            raise AssertionError(f"Unexpected marker key format: {key}")

        # Python MarkerInfo currently includes a `type` field; plateau
        # snapshots only need ownership, so we mark all as regular.
        from app.models.core import MarkerInfo  # local import to avoid cycles

        markers[key] = MarkerInfo(player=entry["player"], position=pos, type="regular")

    collapsed_spaces: Dict[str, int] = {}
    for entry in snapshot.get("collapsedSpaces", []):
        collapsed_spaces[entry["key"]] = entry["player"]

    return BoardState(
        type=board_type,
        size=size,
        stacks=stacks,
        markers=markers,
        collapsed_spaces=collapsed_spaces,
        eliminated_rings={},
        formed_lines=[],
        territories={},
    )


def _build_players_from_snapshot(snapshot: Dict[str, Any]) -> List[Player]:
    """Hydrate minimal Player models from a TS ComparableSnapshot.

    TS ComparableSnapshot exports only playerNumber, type, ringsInHand,
    eliminatedRings, and territorySpaces. The Python Player model has
    additional metadata (id, username, readiness, time, etc.), which we
    stub with deterministic defaults that do not affect parity.
    """

    players: List[Player] = []
    for entry in snapshot.get("players", []):
        num = entry["playerNumber"]
        players.append(
            Player(
                id=f"p{num}",
                username=f"p{num}",
                type=entry["type"],
                player_number=num,
                is_ready=True,
                time_remaining=0,
                ai_difficulty=None,
                rings_in_hand=entry["ringsInHand"],
                eliminated_rings=entry["eliminatedRings"],
                territory_spaces=entry["territorySpaces"],
            )
        )

    # Ensure deterministic order
    players.sort(key=lambda p: p.player_number)
    return players


def _build_game_state_from_snapshot(snapshot: Dict[str, Any]) -> GameState:
    """Construct a Python GameState equivalent to a TS ComparableSnapshot.

    This fills in non-snapshot fields with benign defaults while ensuring
    that all snapshot-relevant fields (boardType, players, stacks, markers,
    collapsedSpaces, currentPlayer, phase/status, and ring counts) match
    the TS source.
    """

    board = _build_board_from_snapshot(snapshot)
    players = _build_players_from_snapshot(snapshot)

    board_type = BoardType(snapshot["boardType"])
    current_phase = GamePhase(snapshot["currentPhase"])
    game_status = GameStatus(snapshot["gameStatus"])

    time_control = TimeControl(initial_time=0, increment=0, type="rapid")

    return GameState(
        id=f"{snapshot['boardType']}_plateau",
        board_type=board_type,
        rng_seed=None,
        board=board,
        players=players,
        current_phase=current_phase,
        current_player=snapshot["currentPlayer"],
        move_history=[],
        time_control=time_control,
        spectators=[],
        game_status=game_status,
        winner=None,
        created_at=datetime.utcfromtimestamp(0),
        last_move_at=datetime.utcfromtimestamp(0),
        is_rated=False,
        max_players=len(players),
        total_rings_in_play=snapshot["totalRingsInPlay"],
        total_rings_eliminated=snapshot["totalRingsEliminated"],
        victory_threshold=0,
        territory_victory_threshold=0,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
    )


def _python_comparable_snapshot(label: str, state: GameState) -> Dict[str, Any]:
    """Build a Python-side ComparableSnapshot dict that mirrors TS shape.

    Field names and ordering are chosen to match tests/utils/stateSnapshots.ts
    in the TS project, so we can compare JSON dumps directly after sorting
    collections and ignoring the `label` field.
    """

    players = [
        {
            "playerNumber": p.player_number,
            "type": p.type,
            "ringsInHand": p.rings_in_hand,
            "eliminatedRings": p.eliminated_rings,
            "territorySpaces": p.territory_spaces,
        }
        for p in state.players
    ]
    players.sort(key=lambda p: p["playerNumber"])

    stacks: List[Dict[str, Any]] = []
    for key, stack in state.board.stacks.items():
        stacks.append(
            {
                "key": key,
                "controllingPlayer": stack.controlling_player,
                "stackHeight": stack.stack_height,
                "capHeight": stack.cap_height,
                "rings": list(stack.rings),
            }
        )
    stacks.sort(key=lambda s: s["key"])

    markers: List[Dict[str, Any]] = []
    for key, marker in state.board.markers.items():
        markers.append({"key": key, "player": marker.player})
    markers.sort(key=lambda m: m["key"])

    collapsed_spaces: List[Dict[str, Any]] = []
    for key, owner in state.board.collapsed_spaces.items():
        collapsed_spaces.append({"key": key, "player": owner})
    collapsed_spaces.sort(key=lambda c: c["key"])

    return {
        "label": label,
        "boardType": state.board_type.value,
        "currentPlayer": state.current_player,
        "currentPhase": state.current_phase.value,
        "gameStatus": state.game_status.value,
        "totalRingsInPlay": state.total_rings_in_play,
        "totalRingsEliminated": state.total_rings_eliminated,
        "players": players,
        "stacks": stacks,
        "markers": markers,
        "collapsedSpaces": collapsed_spaces,
    }


def _normalise_for_comparison(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise snapshots to match TS snapshotsEqual semantics.

    We drop the `label` field and rely on sorted collections to ensure
    deterministic ordering.
    """

    out = dict(snapshot)
    out.pop("label", None)
    return out


@pytest.mark.parametrize("fixture_path", [SEED18_SNAPSHOT, SEED1_SNAPSHOT])
def test_ts_vs_python_plateau_snapshots(fixture_path: Path) -> None:
    """Assert that Python can faithfully reconstruct TS plateau snapshots.

    If the TS snapshot JSON for a seed plateau has not yet been generated,
    this test is skipped, mirroring the pattern used by the rules-parity
    fixture tests.
    """

    if not fixture_path.exists():
        pytest.skip(
            "TS plateau snapshot fixture not found. Run the TS exporters "
            "(ExportSeed18Snapshot/ExportSeed1Snapshot Jest tests) first."
        )

    with fixture_path.open("r", encoding="utf-8") as f:
        ts_snapshot: Dict[str, Any] = json.load(f)

    state = _build_game_state_from_snapshot(ts_snapshot)
    py_snapshot = _python_comparable_snapshot(ts_snapshot.get("label", "py"), state)

    assert _normalise_for_comparison(py_snapshot) == _normalise_for_comparison(ts_snapshot)
