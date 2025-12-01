import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure app package is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    LineInfo,
    MoveType,
    Player,
    Position,
    RingStack,
    Territory,
    TimeControl,
)
from app.game_engine import GameEngine  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402
from tests.parity.test_ts_seed_plateau_snapshot_parity import (  # noqa: E402
    _build_game_state_from_snapshot,
    _normalise_for_comparison,
    _python_comparable_snapshot,
)


BASE_DIR = Path(__file__).resolve().parent
PARITY_DIR = BASE_DIR

LINE_TERRITORY_SNAPSHOT_BY_BOARD = {
    BoardType.SQUARE8: PARITY_DIR / "line_territory_scenario_square8.snapshot.json",
    BoardType.SQUARE19: PARITY_DIR / "line_territory_scenario_square19.snapshot.json",
    BoardType.HEXAGONAL: PARITY_DIR / "line_territory_scenario_hexagonal.snapshot.json",
}


REQUIRED_LENGTH_BY_BOARD = {
    BoardType.SQUARE8: 3,
    BoardType.SQUARE19: 4,
    BoardType.HEXAGONAL: 4,
}


def create_base_state(board_type: BoardType) -> GameState:
    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    else:
        # Mirror TS BOARD_CONFIGS for hexagonal boards (size = 11, radius 10)
        size = 11

    return GameState(
        id="line-territory-scenario",
        boardType=board_type,
        board=BoardState(type=board_type, size=size),
        players=[
            Player(
                id="p1",
                username="Player1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p2",
                username="Player2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ],
        currentPhase=GamePhase.LINE_PROCESSING,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
    )


@pytest.mark.parametrize("board_type", [
    BoardType.SQUARE8,
    BoardType.SQUARE19,
    BoardType.HEXAGONAL,
])
def test_line_and_territory_scenario_parity(board_type: BoardType) -> None:
    """Python analogue of TS Q7/Q20/Q22 line+territory scenario."""
    state = create_base_state(board_type)
    board = state.board
    required_length = REQUIRED_LENGTH_BY_BOARD[board_type]

    # Synthetic overlong line for Player 1: length = requiredLength + 1.
    line_positions = [
        Position(x=i, y=0) for i in range(required_length + 1)
    ]

    # Region: single-cell disconnected territory at (5,5) with a P2 stack.
    region_pos = Position(x=5, y=5)
    region_key = region_pos.to_key()
    p2_stack = RingStack(
        position=region_pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    board.stacks[region_key] = p2_stack

    # P1 stack outside the region (for self-elimination prerequisite).
    outside_pos = Position(x=7, y=7)
    outside_key = outside_pos.to_key()
    p1_rings = [1, 1]
    p1_stack = RingStack(
        position=outside_pos,
        rings=p1_rings,
        stackHeight=len(p1_rings),
        capHeight=len(p1_rings),
        controllingPlayer=1,
    )
    board.stacks[outside_key] = p1_stack

    # Capture baseline metrics.
    player1 = next(p for p in state.players if p.player_number == 1)
    initial_territory = player1.territory_spaces
    initial_eliminated = player1.eliminated_rings
    initial_total_eliminated = state.total_rings_eliminated
    initial_collapsed_count = len(board.collapsed_spaces)

    # Build synthetic line and region Territory mirroring TS tests.
    synthetic_line = LineInfo(
        positions=line_positions,
        player=1,
        length=len(line_positions),
        direction=Position(x=1, y=0),
    )
    region_territory = Territory(
        spaces=[region_pos],
        controllingPlayer=1,
        isDisconnected=True,
    )

    # Monkeypatch BoardManager helpers to isolate semantics, as TS tests do.
    orig_find_all_lines = BoardManager.find_all_lines
    orig_find_disconnected_regions = BoardManager.find_disconnected_regions
    orig_get_border_markers = BoardManager.get_border_marker_positions

    try:
        BoardManager.find_all_lines = staticmethod(
            lambda b: [synthetic_line]
        )
        BoardManager.find_disconnected_regions = staticmethod(
            lambda b, moving_player: [region_territory]
        )
        BoardManager.get_border_marker_positions = staticmethod(
            lambda spaces, b: []
        )

        # === 1) Line processing (Option 2: min collapse, no elimination) ===
        line_moves = GameEngine._get_line_processing_moves(state, 1)

        # Canonical path: CHOOSE_LINE_REWARD (TS-aligned).
        # Legacy path: CHOOSE_LINE_OPTION with placement_count == 2.
        option_moves = [
            m
            for m in line_moves
            if (
                m.type == MoveType.CHOOSE_LINE_REWARD
                or (
                    m.type == MoveType.CHOOSE_LINE_OPTION
                    and (m.placement_count or 0) == 2
                )
            )
        ]
        assert option_moves, "Expected at least one Option 2/CHOOSE_LINE_REWARD move"
        line_move = option_moves[0]

        GameEngine._apply_line_formation(state, line_move)

        player1_after_lines = next(
            p for p in state.players if p.player_number == 1
        )
        collapsed_for_p1 = [
            key
            for key, owner in board.collapsed_spaces.items()
            if owner == 1
        ]
        delta_collapsed = len(collapsed_for_p1) - initial_collapsed_count

        # Exactly requiredLength markers collapsed, no elimination.
        assert delta_collapsed == required_length
        assert player1_after_lines.eliminated_rings == initial_eliminated
        assert state.total_rings_eliminated == initial_total_eliminated
        # Line collapse updates board.collapsed_spaces but does not yet
        # change per-player territorySpaces in the TS-aligned semantics.
        assert player1_after_lines.territory_spaces == initial_territory

        # === 2) Territory processing for the disconnected region ===
        territory_moves = GameEngine._get_territory_processing_moves(
            state, 1
        )
        assert territory_moves, "Expected at least one territory move"
        territory_move = territory_moves[0]

        GameEngine._apply_territory_claim(state, territory_move)

        player1_after_territory = next(
            p for p in state.players if p.player_number == 1
        )

        # Region stack removed and cell collapsed to Player 1.
        assert board.stacks.get(region_key) is None
        assert board.collapsed_spaces.get(region_key) == 1

        # TerritorySpaces: +1 from the processed region. Line-collapse
        # rewards are already reflected in board.collapsed_spaces and do not
        # directly bump territory_spaces in the TS-aligned semantics.
        assert (
            player1_after_territory.territory_spaces
            == initial_territory + 1
        )

        # Elimination accounting: 1 ring from the region (P2 stack) credited
        # to Player 1. Mandatory self-elimination is now modelled as a separate
        # explicit ELIMINATE_RINGS_FROM_STACK decision rather than being baked
        # into PROCESS_TERRITORY_REGION.
        delta_elim_p1 = (
            player1_after_territory.eliminated_rings - initial_eliminated
        )
        delta_total_elim = (
            state.total_rings_eliminated - initial_total_eliminated
        )
        assert delta_elim_p1 == 1
        assert delta_total_elim == 1
        assert board.eliminated_rings.get("1") == 1

    finally:
        BoardManager.find_all_lines = orig_find_all_lines
        BoardManager.find_disconnected_regions = orig_find_disconnected_regions
        BoardManager.get_border_marker_positions = orig_get_border_markers


@pytest.mark.parametrize(
    "board_type",
    [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL],
)
def test_line_and_territory_ts_snapshot_parity(board_type: BoardType) -> None:
    """
    TS snapshot-based parity for the combined line+territory scenario.

    For each board type, this test:
    - Loads the TS-generated ComparableSnapshot JSON fixture for the
      line+territory scenario.
    - Hydrates an equivalent Python GameState from the snapshot.
    - Reconstructs a Python ComparableSnapshot shape.
    - Asserts deep equality with the TS snapshot (modulo the `label`
      field), mirroring plateau snapshot parity tests.
    """
    fixture_path = LINE_TERRITORY_SNAPSHOT_BY_BOARD[board_type]

    if not fixture_path.exists():
        pytest.skip(
            "TS line+territory snapshot fixture not found. "
            "Run the TS exporter (ExportLineAndTerritorySnapshot Jest test) first."
        )

    with fixture_path.open("r", encoding="utf-8") as f:
        ts_snapshot = json.load(f)

    state = _build_game_state_from_snapshot(ts_snapshot)
    py_snapshot = _python_comparable_snapshot(ts_snapshot.get("label", "py"), state)

    assert _normalise_for_comparison(py_snapshot) == _normalise_for_comparison(ts_snapshot)
