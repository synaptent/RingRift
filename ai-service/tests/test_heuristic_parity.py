"""Cross-language heuristic parity scaffolding.

This test suite loads shared heuristic fixtures defined in the TS
project (tests/fixtures/heuristic/v1/*.json), hydrates them into
Python GameState instances, and verifies that the Python HeuristicAI
respects the same ordering constraints as the TS-side
`evaluateHeuristicState`.

The goal is *ordering* agreement on curated micro-positions rather
than exact numeric score parity. Each fixture encodes a small set of
GameState snapshots plus a list of `(better, worse)` pairs for a given
heuristic profile id. For each pair we assert:

    HeuristicAI(profile).evaluate_position(better) > evaluate_position(worse)

This file intentionally stays lightweight so additional fixtures can be
added without changing the test harness.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from app.ai.heuristic_ai import HeuristicAI
from app.models.core import (
    AIConfig,
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


# Repository root: ai-service/tests/.. -> ai-service/.. (project root)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HEURISTIC_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "heuristic" / "v1"
_SIMPLE_STACKS_FIXTURE = _HEURISTIC_FIXTURE_DIR / "square8_2p_simple_stacks.v1.json"


def _load_fixture(path: Path) -> Dict[str, Any]:
    if not path.exists():
        pytest.skip(f"Heuristic fixture not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_board_from_fixture(snapshot: Dict[str, Any]) -> BoardState:
    board_data = snapshot["board"]

    board_type = BoardType(board_data["type"])
    size = int(board_data["size"])

    stacks: Dict[str, RingStack] = {}
    for key, entry in board_data.get("stacks", {}).items():
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
            stack_height=int(entry["stackHeight"]),
            cap_height=int(entry["capHeight"]),
            controlling_player=int(entry["controllingPlayer"]),
        )
        stacks[key] = stack

    return BoardState(
        type=board_type,
        size=size,
        stacks=stacks,
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
        formed_lines=[],
        territories={},
    )


def _build_players_from_fixture(snapshot: Dict[str, Any]) -> list[Player]:
    players: list[Player] = []
    for entry in snapshot.get("players", []):
        num = int(entry["playerNumber"])
        players.append(
            Player(
                id=f"p{num}",
                username=f"p{num}",
                type=entry["type"],
                player_number=num,
                is_ready=True,
                time_remaining=0,
                ai_difficulty=None,
                rings_in_hand=int(entry.get("ringsInHand", 0)),
                eliminated_rings=int(entry.get("eliminatedRings", 0)),
                territory_spaces=int(entry.get("territorySpaces", 0)),
            )
        )

    players.sort(key=lambda p: p.player_number)
    return players


def _build_game_state_from_fixture_state(state_entry: Dict[str, Any]) -> GameState:
    game_state_snapshot = state_entry["gameState"]

    board = _build_board_from_fixture(game_state_snapshot)
    players = _build_players_from_fixture(game_state_snapshot)

    board_type = BoardType(game_state_snapshot["boardType"])
    game_status = GameStatus(game_state_snapshot["gameStatus"])

    time_control = TimeControl(initial_time=0, increment=0, type="rapid")

    return GameState(
        id=str(state_entry.get("id", "heuristic_fixture")),
        board_type=board_type,
        rng_seed=None,
        board=board,
        players=players,
        current_phase=GamePhase.MOVEMENT,
        current_player=int(game_state_snapshot["currentPlayer"]),
        move_history=[],
        time_control=time_control,
        spectators=[],
        game_status=game_status,
        winner=None,
        created_at=datetime.utcfromtimestamp(0),
        last_move_at=datetime.utcfromtimestamp(0),
        is_rated=False,
        max_players=len(players),
        total_rings_in_play=int(game_state_snapshot.get("totalRingsInPlay", 0)),
        total_rings_eliminated=int(
            game_state_snapshot.get("totalRingsEliminated", 0)
        ),
        victory_threshold=0,
        territory_victory_threshold=0,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
    )


@pytest.mark.parametrize("fixture_path", [_SIMPLE_STACKS_FIXTURE])
def test_python_heuristic_respects_orderings(fixture_path: Path) -> None:
    """For each (better, worse) pair in the fixture, assert Python ordering.

    The fixture currently encodes only ordering for the
    ``heuristic_v1_balanced`` profile, but the harness is generic enough
    to support additional personas and board types.
    """

    data = _load_fixture(fixture_path)

    # Default profile id for this fixture; individual orderings may override
    # this via an explicit "profileId" field to exercise multiple personas
    # against the same set of states.
    base_profile_id = data.get("profileId", "heuristic_v1_balanced")

    # Index states by id for quick lookup.
    states_by_id: Dict[str, Dict[str, Any]] = {
        s["id"]: s for s in data.get("states", [])
    }

    for ordering in data.get("orderings", []):
        better_id = ordering["better"]
        worse_id = ordering["worse"]
        profile_id = ordering.get("profileId", base_profile_id)

        better_entry = states_by_id.get(better_id)
        worse_entry = states_by_id.get(worse_id)

        assert better_entry is not None, f"Unknown state id in fixture: {better_id}"
        assert worse_entry is not None, f"Unknown state id in fixture: {worse_id}"

        better_state = _build_game_state_from_fixture_state(better_entry)
        worse_state = _build_game_state_from_fixture_state(worse_entry)

        player_number = int(better_entry.get("playerNumber", 1))

        # Configure HeuristicAI with the requested profile id. We pin
        # difficulty to a heuristic-tier value (2) but the exact number
        # is not important for evaluation semantics as long as the
        # profile id is applied.
        config = AIConfig(difficulty=2, heuristic_profile_id=profile_id)
        ai = HeuristicAI(player_number=player_number, config=config)

        better_score = ai.evaluate_position(better_state)
        worse_score = ai.evaluate_position(worse_state)

        msg = "Expected {better} > {worse} for profile {pid}".format(
            better=better_id,
            worse=worse_id,
            pid=profile_id,
        )
        assert better_score > worse_score, msg
