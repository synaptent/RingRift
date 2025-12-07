import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

import pytest

# Ensure app package is importable, matching other ai-service tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

PARITY_FIXTURES_DIR = Path(ROOT) / "parity_fixtures"

from app.db import GameReplayDB  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.models import GameState, BoardType, Move  # noqa: E402
from app.rules.mutable_state import MutableGameState, MoveUndo  # noqa: E402

# A small, representative subset of large-board parity fixtures.
# These were originally written by scripts/check_ts_python_replay_parity.py
# and capture specific (db_path, game_id, move_index) slices from
# self-play DBs.
#
# For SearchBoard / MutableGameState parity we treat them as:
#   - canonical Python trajectories via GameEngine.apply_move (trace_mode=True)
#   - reference sequences of Move objects to replay via MutableGameState.
LARGE_BOARD_FIXTURES: List[str] = [
    # Square19 2p midgame capture-heavy position
    (
        "selfplay_square19_2p__"
        "02aa8d91-47aa-4d3e-a506-cdd493bda33a__k118.json"
    ),
    # Hexagonal 2p early-game movement position
    (
        "selfplay_hexagonal_2p__"
        "41c7c746-af99-48cb-8887-435b03b5eac7__k8.json"
    ),
    # Hexagonal 3p midgame territory/capture mix
    (
        "selfplay_hexagonal_3p__"
        "1f7a10cc-e41c-48eb-80a9-8c4bfde8d3d0__k87.json"
    ),
]


def _load_fixture_payload(fixture_name: str) -> dict:
    """Load a parity fixture JSON payload.

    Returns the JSON object loaded from the fixture file. If the
    fixture (or its referenced DB) is missing in the current
    environment, the test will be skipped rather than failing.
    """
    fixture_path = PARITY_FIXTURES_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Parity fixture not found: {fixture_path}")

    with fixture_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    db_path = payload.get("db_path")
    game_id = payload.get("game_id")
    if not isinstance(db_path, str) or not isinstance(game_id, str):
        pytest.skip(f"Fixture {fixture_name} missing db_path/game_id")

    if not Path(db_path).exists():
        pytest.skip(f"Referenced GameReplayDB not found: {db_path}")

    if "canonical_move_index" not in payload:
        # Older fixtures may only have diverged_at; in those cases we fall
        # back to using diverged_at as the comparison index.
        if "diverged_at" in payload:
            payload["canonical_move_index"] = payload["diverged_at"] - 1
        else:
            pytest.skip(
                "Fixture "
                f"{fixture_name} missing canonical_move_index/diverged_at"
            )

    return payload


def _replay_canonical_sequence(
    db: GameReplayDB,
    game_id: str,
    move_index: int,
) -> Tuple[GameState, List[Move]]:
    """Replay 0..move_index via canonical GameEngine.apply_move.

    Returns:
        (final_state, moves) where final_state is the GameState after
        applying moves[0..move_index] with trace_mode=True.
    """
    initial_state = db.get_initial_state(game_id)
    if initial_state is None:
        raise RuntimeError(f"Initial state not found for game_id={game_id}")

    # Fetch the move slice once to drive both canonical and SearchBoard paths.
    moves = db.get_moves(game_id, start=0, end=move_index + 1)

    state = initial_state
    for mv in moves:
        state = GameEngine.apply_move(state, mv, trace_mode=True)

    return state, moves


def _replay_via_mutable(
    initial_state: GameState,
    moves: List[Move],
) -> GameState:
    """Replay the same move sequence via MutableGameState (SearchBoard).

    This uses the make/unmake SearchBoard path as a pure implementation
    detail:

    - Initializes MutableGameState from the immutable initial_state.
    - Applies each Move in the sequence via make_move (no unmake in this
      pass).
    - Converts back to GameState via to_immutable() for comparison.

    Phase / LPS semantics inside MutableGameState intentionally use a
    simplified phase machine for search; this helper focuses on
    *structural* parity (board, markers, collapsed spaces, eliminations,
    and player ring/territory counts) rather than re-validating phase
    transitions.
    """
    mutable = MutableGameState.from_immutable(initial_state)
    for mv in moves:
        mutable.make_move(mv)
    return mutable.to_immutable()


def _assert_structural_parity(expected: GameState, actual: GameState) -> None:
    """Assert structural parity between two GameState instances.

    This focuses on the parts that MutableGameState maintains incrementally:

    - Board geometry: type, size
    - Stacks: keys, rings, heights, controlling_player
    - Markers: keys, player, position, type
    - Collapsed spaces: ownership map
    - Board-level eliminated_rings
    - Per-player rings_in_hand, eliminated_rings, territory_spaces
    - total_rings_eliminated
    - Zobrist hash when present on both states

    Phase / current_player / game_status are *not* asserted here, because
    MutableGameState intentionally uses a search-oriented phase machine
    that is allowed to differ from the canonical engine's turn
    orchestrator while still producing structurally valid positions for
    AI search.
    """
    # Board geometry
    assert expected.board.type == actual.board.type
    assert expected.board.size == actual.board.size

    # Stacks
    exp_stacks = expected.board.stacks
    act_stacks = actual.board.stacks
    assert set(exp_stacks.keys()) == set(act_stacks.keys())

    for key, exp_stack in exp_stacks.items():
        act_stack = act_stacks[key]
        assert list(exp_stack.rings) == list(act_stack.rings), (
            f"rings mismatch at {key}"
        )
        assert exp_stack.stack_height == act_stack.stack_height, (
            f"stack_height mismatch at {key}"
        )
        assert exp_stack.cap_height == act_stack.cap_height, (
            f"cap_height mismatch at {key}"
        )
        assert (
            exp_stack.controlling_player == act_stack.controlling_player
        ), f"controlling_player mismatch at {key}"

    # Markers
    exp_markers = expected.board.markers
    act_markers = actual.board.markers
    assert set(exp_markers.keys()) == set(act_markers.keys())

    for key, exp_marker in exp_markers.items():
        act_marker = act_markers[key]
        assert exp_marker.player == act_marker.player, (
            f"marker player mismatch at {key}"
        )
        assert exp_marker.position == act_marker.position, (
            f"marker position mismatch at {key}"
        )
        assert exp_marker.type == act_marker.type, (
            f"marker type mismatch at {key}"
        )

    # Collapsed spaces
    assert expected.board.collapsed_spaces == actual.board.collapsed_spaces

    # Board-level eliminated rings mapping
    assert expected.board.eliminated_rings == actual.board.eliminated_rings

    # Player ring / territory state
    exp_players = {p.player_number: p for p in expected.players}
    act_players = {p.player_number: p for p in actual.players}
    assert set(exp_players.keys()) == set(act_players.keys())

    for pnum, exp_player in exp_players.items():
        act_player = act_players[pnum]
        assert (
            exp_player.rings_in_hand == act_player.rings_in_hand
        ), f"rings_in_hand mismatch for player {pnum}"
        assert (
            exp_player.eliminated_rings == act_player.eliminated_rings
        ), f"eliminated_rings mismatch for player {pnum}"
        assert (
            exp_player.territory_spaces == act_player.territory_spaces
        ), f"territory_spaces mismatch for player {pnum}"

    # Aggregate totals
    assert expected.total_rings_eliminated == actual.total_rings_eliminated

    # Zobrist hash parity when available on both states
    if expected.zobrist_hash is not None and actual.zobrist_hash is not None:
        assert expected.zobrist_hash == actual.zobrist_hash


@pytest.mark.parametrize("fixture_name", LARGE_BOARD_FIXTURES)
def test_mutable_state_large_board_replay_matches_canonical_board(
    fixture_name: str,
) -> None:
    """SearchBoard replay matches canonical GameEngine replay on large boards.

    For each selected large-board parity fixture, this test:

    1. Loads the corresponding self-play DB and game_id from the JSON
       fixture.
    2. Replays moves 0..canonical_move_index via:
       - Canonical path: GameEngine.apply_move(..., trace_mode=True)
       - SearchBoard path: MutableGameState.make_move + to_immutable()
    3. Compares the resulting GameState instances using structural parity
       assertions focused on board and player material.

    This provides a targeted guarantee that the SearchBoard/make-unmake
    layer faithfully mirrors canonical board/ownership semantics for
    complex large positions (Square19 and Hex) without re-encoding the
    full phase machine.
    """
    payload = _load_fixture_payload(fixture_name)
    db_path = payload["db_path"]
    game_id = payload["game_id"]
    move_index = int(payload["canonical_move_index"])

    db = GameReplayDB(db_path)

    # Ensure we do not hit auto-injected NO_*_ACTION moves in
    # get_state_at_move. In some environments the referenced DB may exist
    # but the specific game_id may have been pruned; in that case, skip
    # rather than failing hard.
    moves_for_game = db.get_moves(game_id)
    total_moves = len(moves_for_game)
    if total_moves == 0 or move_index >= total_moves:
        pytest.skip(
            "No moves found for game_id "
            f"{game_id!r} in {db_path} (total_moves={total_moves}, "
            f"canonical_move_index={move_index})"
        )

    canonical_final, moves = _replay_canonical_sequence(
        db,
        game_id,
        move_index,
    )

    # Sanity: we are in fact dealing with large boards
    assert canonical_final.board.type in {
        BoardType.SQUARE19,
        BoardType.HEXAGONAL,
    }

    mutable_final = _replay_via_mutable(
        db.get_initial_state(game_id),  # type: ignore[arg-type]
        moves,
    )

    _assert_structural_parity(canonical_final, mutable_final)


@pytest.mark.parametrize(
    "fixture_name",
    [
        (
            "selfplay_square19_2p__"
            "02aa8d91-47aa-4d3e-a506-cdd493bda33a__k118.json"
        ),
        (
            "selfplay_hexagonal_2p__"
            "41c7c746-af99-48cb-8887-435b03b5eac7__k8.json"
        ),
    ],
)
def test_mutable_state_large_board_make_unmake_roundtrip(
    fixture_name: str,
) -> None:
    """Large-board make/unmake roundtrip restores the original state.

    Starting from a midgame large-board position, this test:

    1. Wraps the canonical GameState in MutableGameState.
    2. Applies a short sequence of valid moves using make_move.
    3. Un-applies them in reverse using unmake_move.
    4. Asserts that structural state (board, markers, collapsed spaces,
       eliminated rings, per-player material, totals, and hash) is
       restored.

    This is a focused regression guard for make/unmake correctness on
    large boards (Square19 and Hex), complementing the existing
    square8-focused `benchmark_make_unmake` roundtrip test.
    """
    payload = _load_fixture_payload(fixture_name)
    db_path = payload["db_path"]
    game_id = payload["game_id"]
    move_index = int(payload["canonical_move_index"])

    db = GameReplayDB(db_path)
    base_state = db.get_state_at_move(game_id, move_index)
    if base_state is None:
        pytest.skip(
            "Could not reconstruct state at move "
            f"{move_index} for {game_id}"
        )

    # Sanity: ensure large-board geometry
    assert base_state.board.type in {
        BoardType.SQUARE19,
        BoardType.HEXAGONAL,
    }

    from app.rules.default_engine import DefaultRulesEngine

    rules_engine = DefaultRulesEngine()
    mutable = MutableGameState.from_immutable(base_state)
    original_state = base_state

    undos: List[MoveUndo] = []
    max_steps = 5

    for _ in range(max_steps):
        current_state = mutable.to_immutable()
        valid_moves = rules_engine.get_valid_moves(
            current_state,
            current_state.current_player,
        )
        if not valid_moves:
            break

        # Prefer genuinely interactive moves when available; fall back to
        # any.
        interactive_moves = [
            m
            for m in valid_moves
            if m.type
            not in {
                "no_placement_action",
                "no_movement_action",
                "no_line_action",
                "no_territory_action",
            }
        ]
        move = interactive_moves[0] if interactive_moves else valid_moves[0]

        undo = mutable.make_move(move)
        undos.append(undo)

    # Unwind moves in reverse
    for undo in reversed(undos):
        mutable.unmake_move(undo)

    roundtrip_state = mutable.to_immutable()
    _assert_structural_parity(original_state, roundtrip_state)