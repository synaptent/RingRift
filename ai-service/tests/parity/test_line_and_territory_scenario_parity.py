
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

MULTI_REGION_SNAPSHOT = PARITY_DIR / "line_territory_multi_region_square8.snapshot.json"


REQUIRED_LENGTH_BY_BOARD = {
    # TODO: This constant uses 3 for square8 but the canonical rules specify
    # required_length=4 for 2-player games. The test_overlength_line_option2_segments_exhaustive
    # test has a parity issue between this constant and what GameEngine uses.
    # See canonical rules Section 4.5: square8 2-player = 4-in-a-row.
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
        # Mirror TS BOARD_CONFIGS for hexagonal boards (size = 13, radius 12)
        size = 13

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
def test_line_and_territory_scenario_parity(board_type: BoardType, monkeypatch: pytest.MonkeyPatch) -> None:
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
    # Use frozen copies to avoid closure issues with lambdas
    frozen_line = synthetic_line
    frozen_region = region_territory

    def mock_find_all_lines(board, num_players=3):
        return [frozen_line]

    def mock_find_disconnected_regions(board, moving_player):
        return [frozen_region]

    def mock_get_border_markers(spaces, board):
        return []

    monkeypatch.setattr(BoardManager, "find_all_lines", staticmethod(mock_find_all_lines))
    monkeypatch.setattr(BoardManager, "find_disconnected_regions", staticmethod(mock_find_disconnected_regions))
    monkeypatch.setattr(BoardManager, "get_border_marker_positions", staticmethod(mock_get_border_markers))

    # === 1) Line processing (Option 2: min collapse, no elimination) ===
    line_moves = GameEngine._get_line_processing_moves(state, 1)

    # Canonical path: CHOOSE_LINE_REWARD (TS-aligned).
    # Option 2 = minimum collapse = collapsed_markers length equals required_length.
    # Legacy path: CHOOSE_LINE_OPTION with placement_count == 2.
    option_moves = [
        m
        for m in line_moves
        if (
            (
                m.type == MoveType.CHOOSE_LINE_REWARD
                and hasattr(m, "collapsed_markers")
                and m.collapsed_markers is not None
                and len(m.collapsed_markers) == required_length
            )
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
    # Line collapse updates board.collapsed_spaces AND credits territory
    # to the acting player per canonical rule RR-CANON-R041.
    assert player1_after_lines.territory_spaces == initial_territory + required_length

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

    # TerritorySpaces: +required_length from line collapse (per RR-CANON-R041)
    # plus +1 from the processed region.
    assert (
        player1_after_territory.territory_spaces
        == initial_territory + required_length + 1
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


@pytest.mark.parametrize("board_type", [
    BoardType.SQUARE8,
    BoardType.SQUARE19,
    BoardType.HEXAGONAL,
])
def test_get_valid_moves_line_processing_surfaces_only_line_decisions(
    board_type: BoardType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision-phase parity: LINE_PROCESSING get_valid_moves surface.

    For an overlength line, GameEngine.get_valid_moves in LINE_PROCESSING
    must surface only PROCESS_LINE and CHOOSE_LINE_REWARD moves whose
    geometry matches the synthetic line used in this scenario, for all
    supported board types.
    """
    # Clear move cache to ensure our mocked find_all_lines is used
    GameEngine.clear_cache()
    state = create_base_state(board_type)
    required_length = REQUIRED_LENGTH_BY_BOARD[board_type]

    # Synthetic overlength line for Player 1: length = requiredLength + 1.
    line_positions = [Position(x=i, y=0) for i in range(required_length + 1)]
    synthetic_line = LineInfo(
        positions=line_positions,
        player=1,
        length=len(line_positions),
        direction=Position(x=1, y=0),
    )

    # Use frozen copy to avoid closure issues
    frozen_line = synthetic_line

    def mock_find_all_lines(board, num_players=2):
        return [frozen_line]

    monkeypatch.setattr(BoardManager, "find_all_lines", staticmethod(mock_find_all_lines))

    state.current_phase = GamePhase.LINE_PROCESSING
    state.current_player = 1

    moves = GameEngine.get_valid_moves(state, 1)
    assert moves, "Expected line-processing moves in LINE_PROCESSING"

    # Only PROCESS_LINE and CHOOSE_LINE_REWARD should be surfaced.
    allowed_types = {MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_REWARD}
    assert all(m.type in allowed_types for m in moves)

    process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
    reward_moves = [m for m in moves if m.type == MoveType.CHOOSE_LINE_REWARD]
    assert process_moves, "Expected at least one PROCESS_LINE move"
    assert reward_moves, "Expected at least one CHOOSE_LINE_REWARD move"

    # Reward moves must reference the same synthetic line and use either the
    # full line or required-length segments as collapsed_markers.
    for m in reward_moves:
        assert m.formed_lines
        line = list(m.formed_lines)[0]
        actual_positions = list(line.positions)

        # All positions must lie on the synthetic overlength line.
        assert all(pos in line_positions for pos in actual_positions)

        # Length must be either the full synthetic line or a contiguous
        # required-length segment of it.
        length = len(actual_positions)
        assert length in (required_length, len(line_positions))

        xs = sorted(pos.x for pos in actual_positions)
        ys = {pos.y for pos in actual_positions}
        # All positions on the same row and contiguous in x.
        assert len(ys) == 1
        assert all(xs[i + 1] - xs[i] == 1 for i in range(len(xs) - 1))

        if length == len(line_positions):
            # Full-line reward: geometry must match the synthetic line.
            assert set(actual_positions) == set(line_positions)

        collapsed = getattr(m, "collapsed_markers", None)
        assert collapsed is not None
        collapsed_positions = list(collapsed)
        # Collapsed markers must form a non-empty contiguous subset of the
        # synthetic overlength line.
        assert collapsed_positions
        assert all(pos in line_positions for pos in collapsed_positions)
        xs_collapsed = sorted(pos.x for pos in collapsed_positions)
        ys_collapsed = {pos.y for pos in collapsed_positions}
        assert len(ys_collapsed) == 1
        assert all(
            xs_collapsed[i + 1] - xs_collapsed[i] == 1
            for i in range(len(xs_collapsed) - 1)
        )


def test_get_valid_moves_territory_processing_pre_elimination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision-phase parity: TERRITORY_PROCESSING pre-elim surface.

    When at least one disconnected region is eligible, TERRITORY_PROCESSING
    get_valid_moves must surface only PROCESS_TERRITORY_REGION moves whose
    geometry points at the region spaces.
    """
    state = create_base_state(BoardType.SQUARE8)
    board = state.board

    # Disconnected region: single cell at (5,5) claimed by Player 1.
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

    region_territory = Territory(
        spaces=[region_pos],
        controllingPlayer=1,
        isDisconnected=True,
    )

    # Use frozen copy for closure safety
    frozen_region = region_territory

    def mock_find_disconnected_regions(board, player_number):
        return [frozen_region]

    monkeypatch.setattr(
        BoardManager, "find_disconnected_regions", staticmethod(mock_find_disconnected_regions)
    )

    state.current_phase = GamePhase.TERRITORY_PROCESSING
    state.current_player = 1

    moves = GameEngine.get_valid_moves(state, 1)
    assert moves, "Expected territory-processing moves"

    # Only PROCESS_TERRITORY_REGION should be surfaced.
    assert all(
        m.type == MoveType.PROCESS_TERRITORY_REGION for m in moves
    )

    # Each move should carry the disconnected region geometry and point
    # to one of its spaces.
    for m in moves:
        assert m.disconnected_regions
        region = list(m.disconnected_regions)[0]
        assert list(region.spaces) == list(region_territory.spaces)
        assert m.to in region.spaces


def test_get_valid_moves_territory_processing_self_elimination_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision-phase parity: TERRITORY_PROCESSING elimination surface.

    When no disconnected regions are eligible but the player controls stacks,
    TERRITORY_PROCESSING get_valid_moves must surface only
    ELIMINATE_RINGS_FROM_STACK decisions, one per candidate stack.
    """
    state = create_base_state(BoardType.SQUARE8)
    board = state.board

    # Two P1 stacks eligible for elimination.
    pos_a = Position(x=1, y=1)
    pos_b = Position(x=2, y=2)
    board.stacks[pos_a.to_key()] = RingStack(
        position=pos_a,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    board.stacks[pos_b.to_key()] = RingStack(
        position=pos_b,
        rings=[1, 1, 1],
        stackHeight=3,
        capHeight=3,
        controllingPlayer=1,
    )

    # No disconnected regions should be reported for this test.
    def mock_find_disconnected_regions(board, player_number):
        return []

    monkeypatch.setattr(
        BoardManager, "find_disconnected_regions", staticmethod(mock_find_disconnected_regions)
    )

    state.current_phase = GamePhase.TERRITORY_PROCESSING
    state.current_player = 1

    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 0

    moves = GameEngine.get_valid_moves(state, 1)
    assert moves, "Expected elimination decisions in TERRITORY_PROCESSING"

    # Only ELIMINATE_RINGS_FROM_STACK should be surfaced.
    assert all(
        m.type == MoveType.ELIMINATE_RINGS_FROM_STACK for m in moves
    )

    # Candidates must match the player stacks with positive capHeight.
    expected_keys = {
        key
        for key, stack in board.stacks.items()
        if stack.controlling_player == 1 and stack.cap_height > 0
    }
    target_keys = {m.to.to_key() for m in moves if m.to is not None}
    assert target_keys == expected_keys


def _multi_phase_vectors_path() -> Path:
    """Return the path to the TS multi_phase_turn v2 vectors bundle."""
    # Repo root is three levels up from this file:
    # ai-service/tests/parity/test_line_and_territory_scenario_parity.py
    repo_root = BASE_DIR.parent.parent.parent
    return repo_root / "tests" / "fixtures" / "contract-vectors" / "v2" / "multi_phase_turn.vectors.json"


@pytest.mark.skipif(
    not _multi_phase_vectors_path().exists(),
    reason="TS multi_phase_turn.vectors.json not found; run TS tests/fixtures generation first.",
)
def test_turn_line_then_territory_sequence_metadata() -> None:
    """
    Ensure the TS mixed line+territory multi-phase sequences are present and tagged.

    This ties the Python line+territory parity suite back to the TS v2
    contract bundle via the `sequence:turn.line_then_territory.*`
    tags on the corresponding multi_phase vectors.
    """
    path = _multi_phase_vectors_path()
    with path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    vectors = bundle.get("vectors", [])
    by_id = {v.get("id"): v for v in vectors}

    # Square8 base sequence
    base = by_id.get("multi_phase.full_sequence_with_territory")
    assert base is not None, "Expected multi_phase.full_sequence_with_territory vector to exist"
    base_tags = set(base.get("tags", []))
    assert (
        "sequence:turn.line_then_territory.square8" in base_tags
    ), "multi_phase.full_sequence_with_territory missing sequence:turn.line_then_territory.square8 tag"

    # Square19 analogue
    sq19 = by_id.get("multi_phase.full_sequence_with_territory_square19")
    assert (
        sq19 is not None
    ), "Expected multi_phase.full_sequence_with_territory_square19 vector to exist"
    sq19_tags = set(sq19.get("tags", []))
    assert (
        "sequence:turn.line_then_territory.square19" in sq19_tags
    ), "multi_phase.full_sequence_with_territory_square19 missing sequence:turn.line_then_territory.square19 tag"

    # Hex analogue
    hex_vec = by_id.get("multi_phase.full_sequence_with_territory_hex")
    assert (
        hex_vec is not None
    ), "Expected multi_phase.full_sequence_with_territory_hex vector to exist"
    hex_tags = set(hex_vec.get("tags", []))
    assert (
        "sequence:turn.line_then_territory.hex" in hex_tags
    ), "multi_phase.full_sequence_with_territory_hex missing sequence:turn.line_then_territory.hex tag"

    # Mixed multi-region metadata (sequence only; vectors are metadata and
    # validated via TS/backend↔sandbox parity tests).
    for v_id, tag in [
        (
            "multi_phase.line_then_multi_region_territory.square8.step1_line",
            "sequence:turn.line_then_territory.multi_region.square8",
        ),
        (
            "multi_phase.line_then_multi_region_territory.square19.step1_line",
            "sequence:turn.line_then_territory.multi_region.square19",
        ),
        (
            "multi_phase.line_then_multi_region_territory.hex.step1_line",
            "sequence:turn.line_then_territory.multi_region.hex",
        ),
    ]:
        vec = by_id.get(v_id)
        assert (
            vec is not None
        ), f"Expected {v_id} vector to exist for mixed multi-region sequence"
        tags = set(vec.get("tags", []))
        assert tag in tags, f"{v_id} missing {tag} tag"


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


def test_line_and_territory_multi_region_ts_snapshot_parity() -> None:
    """
    TS snapshot-based parity for the mixed line+multi-region scenario (square8).

    This uses the TS-exported snapshot from
    ExportLineAndTerritoryMultiRegionSnapshot.test.ts and checks that the
    Python engine reconstructs an equivalent ComparableSnapshot.
    """
    if not MULTI_REGION_SNAPSHOT.exists():
        pytest.skip(
            "TS mixed line+multi-region snapshot fixture not found. "
            "Run the TS exporter (ExportLineAndTerritoryMultiRegionSnapshot Jest test) first."
        )

    with MULTI_REGION_SNAPSHOT.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    ts_snapshot = payload["snapshot"]

    # For the mixed multi-region case we now treat the TS ComparableSnapshot
    # exactly like the single-region line+territory fixtures: hydrate a
    # Python GameState from the snapshot and assert full structural parity
    # (modulo the label field).
    state = _build_game_state_from_snapshot(ts_snapshot)
    py_snapshot = _python_comparable_snapshot(ts_snapshot.get("label", "py"), state)

    assert _normalise_for_comparison(py_snapshot) == _normalise_for_comparison(ts_snapshot)


@pytest.mark.parametrize(
    "board_type",
    [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL],
)
def test_overlength_line_option2_segments_exhaustive(
    board_type: BoardType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure CHOOSE_LINE_REWARD enumerates all TS-legal Option-2 segments.

    For an overlength line of length L > requiredLength, TS semantics expose:
    - 1 Option-1 reward (collapse-all), and
    - (L - requiredLength + 1) distinct Option-2 rewards, one per contiguous
      requiredLength-length segment of the line.

    This test constructs a synthetic overlength line and asserts that
    GameEngine.get_valid_moves surfaces exactly that set of Option-2 segments
    for all supported board types.
    """
    # Clear move cache to ensure our mocked find_all_lines is used
    GameEngine.clear_cache()
    state = create_base_state(board_type)
    required_length = REQUIRED_LENGTH_BY_BOARD[board_type]

    # Use a synthetic overlength line with two extra markers beyond the
    # minimum required length to exercise multiple Option-2 segments.
    line_positions = [Position(x=i, y=0) for i in range(required_length + 2)]
    synthetic_line = LineInfo(
        positions=line_positions,
        player=1,
        length=len(line_positions),
        direction=Position(x=1, y=0),
    )

    # Force the board to report exactly our synthetic overlength line.
    # Make a fresh copy of positions to avoid any mutation issues.
    frozen_line = LineInfo(
        positions=[Position(x=p.x, y=p.y) for p in synthetic_line.positions],
        player=synthetic_line.player,
        length=synthetic_line.length,
        direction=Position(x=synthetic_line.direction.x, y=synthetic_line.direction.y),
    )

    # Create a closure function to avoid lambda issues
    def mock_find_all_lines(board, num_players=3):
        return [frozen_line]

    # Use monkeypatch for proper test isolation and automatic cleanup
    monkeypatch.setattr(BoardManager, "find_all_lines", staticmethod(mock_find_all_lines))

    # Verify the mock works
    test_result = BoardManager.find_all_lines(state.board, 2)
    assert len(test_result) == 1, f"Mock should return 1 line, got {len(test_result)}"
    assert test_result[0].length == frozen_line.length, (
        f"Mock line length mismatch: expected {frozen_line.length}, "
        f"got {test_result[0].length}"
    )

    state.current_phase = GamePhase.LINE_PROCESSING
    state.current_player = 1

    moves = GameEngine.get_valid_moves(state, 1)
    assert moves, "Expected line-processing moves in LINE_PROCESSING"

    reward_moves = [m for m in moves if m.type == MoveType.CHOOSE_LINE_REWARD]
    assert reward_moves, "Expected CHOOSE_LINE_REWARD moves for overlength line"

    line_len = len(line_positions)
    # Option‑1: collapse‑all reward, identified by the canonical "-all" suffix
    # used by GameEngine._get_line_processing_moves.
    option1_moves = [
        m for m in reward_moves
        if m.id.endswith("-all")
    ]
    # Option‑2: minimum‑collapse segments of exactly required_length markers.
    option2_moves = [
        m for m in reward_moves
        if m.collapsed_markers is not None
        and len(m.collapsed_markers) == required_length
    ]

    # Exactly one collapse-all reward and the TS-expected number of
    # minimum-collapse segments.
    expected_option2_count = line_len - required_length + 1

    # Python CHOOSE_LINE_REWARD enumeration must now exactly match the TS
    # decision surface for this overlength line:
    # - 1 collapse-all (Option-1) reward; and
    # - (L - required_length + 1) distinct contiguous Option-2 segments.
    assert len(option1_moves) == 1
    assert len(option2_moves) == expected_option2_count

    def _segment_key(segment: list[Position]) -> tuple[tuple[int, int, int | None], ...]:
        return tuple((p.x, p.y, getattr(p, "z", None)) for p in segment)

    expected_segments = {
        _segment_key(line_positions[start : start + required_length])
        for start in range(expected_option2_count)
    }
    actual_segments = {
        _segment_key(list(m.collapsed_markers or ()))
        for m in option2_moves
    }

    # Full parity: Python's Option-2 choices must equal the TS-legal set of
    # contiguous required-length windows of the overlength line.
    assert actual_segments == expected_segments
