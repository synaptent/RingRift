from __future__ import annotations

"""
Hex territory / forced-elimination fixtures (HX-A, HX-C).

This module provides Python mirrors for the TS-side HX-A/HX-C fixtures
defined in tests/fixtures/hexTerritoryFeFixtures.ts and exercised by
tests/unit/TerritoryAggregate.hex.feParity.test.ts.

Scope:

- HX-A: Compact hex territory mini-region with mixed internal stacks and
  at least one acting-player stack outside the region. Exercises:
  - Self-elimination prerequisite (must have a stack outside region).
  - Internal eliminations + crediting and region collapse geometry.

- HX-C: Hex scenario where:
  - In territory_processing, the acting player has no processable
    regions.
  - Territory phase records a no_territory_action bookkeeping move.
  - Forced elimination is then surfaced via a dedicated
    forced_elimination phase with explicit forced_elimination moves,
    never as territory_processing moves.

These tests are parity-focused and must not change GameEngine semantics.
They are intended to become part of the broader parity/gating story for
hex boards.
"""

import os
import sys
from datetime import datetime

import pytest

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.board_manager import BoardManager  # noqa: E402
from app.game_engine import (  # noqa: E402
    GameEngine,
    PhaseRequirementType,
)
from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    Territory,
    TimeControl,
)


# ────────────────────────────────────────────────────────────────────────────────
# HX-A: hex mini-region fixture (mixed internal stacks)
# ────────────────────────────────────────────────────────────────────────────────


def create_hex_territory_fe_board_hx_a() -> tuple[GameState, Territory]:
    """
    Build a minimal hex GameState + Territory mirroring the TS HX-A fixture.

    Geometry (hex radius-12, size 13 board):

    - Board type: hexagonal (BoardType.HEXAGONAL), size = 13.
    - Stacks:
        (0, 0, 0)  -> P1 internal stack, height 1
        (1, -1, 0) -> P2 internal stack, height 2
        (0, -1, 1) -> P2 internal stack, height 1
        (2, -2, 0) -> P1 stack outside the region, height 1
    - No collapsed spaces.

    Territory region for HX-A / Player 1:

        spaces: (0, 0, 0), (1, -1, 0), (0, -1, 1)
        controllingPlayer: 1
        isDisconnected: true

    The GameState is initialised directly in TERRITORY_PROCESSING for
    player 1 so tests can call GameEngine._can_process_disconnected_region
    and GameEngine.apply_move with PROCESS_TERRITORY_REGION decisions.
    """
    board = BoardState(type=BoardType.HEXAGONAL, size=13)

    # Internal P1 stack at (0,0,0)
    pos_p1_internal = Position(x=0, y=0, z=0)
    board.stacks[pos_p1_internal.to_key()] = RingStack(
        position=pos_p1_internal,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    # Internal P2 stack at (1,-1,0), height 2
    pos_p2_internal_tall = Position(x=1, y=-1, z=0)
    board.stacks[pos_p2_internal_tall.to_key()] = RingStack(
        position=pos_p2_internal_tall,
        rings=[2, 2],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=2,
    )

    # Internal P2 stack at (0,-1,1), height 1
    pos_p2_internal = Position(x=0, y=-1, z=1)
    board.stacks[pos_p2_internal.to_key()] = RingStack(
        position=pos_p2_internal,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )

    # P1 stack outside the region at (2,-2,0), height 1
    pos_p1_outside = Position(x=2, y=-2, z=0)
    board.stacks[pos_p1_outside.to_key()] = RingStack(
        position=pos_p1_outside,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    # No collapsed spaces or territories; eliminated_rings starts empty.
    now = datetime.now()

    players = [
        Player(
            id="p1",
            username="Player1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="Player2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    total_rings_in_play = sum(
        stack.stack_height for stack in board.stacks.values()
    )

    game_state = GameState(  # type: ignore[call-arg]
        id="hex-hx-a-territory-fe",
        boardType=BoardType.HEXAGONAL,
        board=board,
        players=players,
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=total_rings_in_play,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )

    region = Territory(
        spaces=[
            Position(x=0, y=0, z=0),
            Position(x=1, y=-1, z=0),
            Position(x=0, y=-1, z=1),
        ],
        controllingPlayer=1,
        isDisconnected=True,
    )

    return game_state, region


def _region_keys(region: Territory) -> set[str]:
    """Helper: return a set of position keys for the region's spaces."""
    return {pos.to_key() for pos in region.spaces}


def test_hex_hx_a_self_elimination_prerequisite_uses_outside_stacks() -> None:
    """
    HX-A self-elimination prerequisite: must have a stack outside region.

    Mirrors the TS HX-A test:

    - With the full HX-A board, player 1 controls stacks both inside and
      outside the region, so the region is processable for self-elimination.
    - After removing all player-1 stacks outside the region, the region is
      no longer processable for player 1.
    """
    game_state, region = create_hex_territory_fe_board_hx_a()

    assert game_state.board.type == BoardType.HEXAGONAL
    assert game_state.current_phase == GamePhase.TERRITORY_PROCESSING
    assert game_state.current_player == 1

    region_keys = _region_keys(region)

    # Collect all P1 stacks outside the HX-A region.
    p1_stacks = BoardManager.get_player_stacks(
        game_state.board,
        player_number=1,
    )
    outside_keys = [
        stack.position.to_key()
        for stack in p1_stacks
        if stack.position.to_key() not in region_keys
    ]

    # Fixture sanity: there must be at least one P1 stack outside the region.
    assert outside_keys, (
        "Expected at least one P1 stack outside HX-A mini-region"
    )

    # With the original board, the region should satisfy the
    # self-elimination prerequisite.
    can_process = GameEngine._can_process_disconnected_region(
        game_state,
        region,
        player_number=1,
    )
    assert can_process is True

    # Remove all P1 stacks outside the region and re-check.
    board_no_outside = game_state.board.model_copy()
    # Follow GameEngine.apply_move pattern: copy stacks dict explicitly.
    board_no_outside.stacks = game_state.board.stacks.copy()
    for key in outside_keys:
        board_no_outside.stacks.pop(key, None)

    state_no_outside = game_state.model_copy(
        update={"board": board_no_outside},
    )

    can_process_after = GameEngine._can_process_disconnected_region(
        state_no_outside,
        region,
        player_number=1,
    )
    assert (
        can_process_after is False
    ), (
        "HX-A mini-region should not be processable when P1 has no stacks "
        "outside it"
    )


def test_hex_hx_a_internal_eliminations_and_credit() -> None:
    """
    HX-A internal eliminations + crediting.

    Mirrors the TS HX-A crediting test:

    - All stacks inside the HX-A mini-region are eliminated.
    - All region spaces are collapsed to player 1.
    - The total internal ring count across the region is fully credited to
      player 1 (and to total_rings_eliminated).
    - The input GameState remains unchanged (immutability of apply_move).
    """
    base_state, region = create_hex_territory_fe_board_hx_a()

    assert base_state.board.type == BoardType.HEXAGONAL
    assert base_state.current_phase == GamePhase.TERRITORY_PROCESSING
    assert base_state.current_player == 1

    # Representative internal stack at (1,-1,0)
    internal_pos = Position(x=1, y=-1, z=0)
    internal_key = internal_pos.to_key()
    internal_stack = base_state.board.stacks.get(internal_key)
    assert internal_stack is not None, (
        "Expected a stack at (1,-1,0) in the HX-A mini-region"
    )
    internal_height = internal_stack.stack_height
    assert internal_height > 0

    # Total internal height across all region spaces.
    total_internal_height = 0
    for pos in region.spaces:
        key = pos.to_key()
        stack = base_state.board.stacks.get(key)
        if stack is not None:
            total_internal_height += stack.stack_height
    assert total_internal_height >= internal_height
    assert total_internal_height > 0

    # Player-1 elimination stats and global elimination count before
    # applying the move.
    before_p1_elims = 0
    for p in base_state.players:
        if p.player_number == 1:
            before_p1_elims = p.eliminated_rings
            break

    before_total_elims = base_state.total_rings_eliminated

    # Construct a PROCESS_TERRITORY_REGION move that carries the HX-A region
    # explicitly via disconnectedRegions, so that _apply_territory_claim
    # processes exactly this region.
    move = Move(  # type: ignore[call-arg]
        id="process-hex-terr-fe-mini-region-hx-a",
        type=MoveType.PROCESS_TERRITORY_REGION,
        player=base_state.current_player,
        to=internal_pos,
        disconnectedRegions=(region,),
        timestamp=base_state.last_move_at,
        thinkTime=0,
        moveNumber=len(base_state.move_history) + 1,
    )

    # Apply the move via the canonical GameEngine.apply_move surface, which
    # clones the input state and invokes _apply_territory_claim internally.
    next_state = GameEngine.apply_move(base_state, move)

    # Input state must remain unchanged.
    assert internal_key in base_state.board.stacks
    assert base_state.total_rings_eliminated == before_total_elims

    # Representative internal stack is removed on the resulting board.
    assert internal_key not in next_state.board.stacks

    # All region spaces are collapsed to player 1, with no stacks or
    # markers remaining.
    for pos in region.spaces:
        key = pos.to_key()
        owner = next_state.board.collapsed_spaces.get(key)
        assert owner == 1, (
            f"Expected collapsed owner 1 at {key}, got {owner!r}"
        )
        assert key not in next_state.board.stacks
        assert key not in next_state.board.markers

    # All rings from stacks in the HX-A mini-region are credited to
    # player 1 as eliminations, and the global elimination count increases
    # by the same amount.
    after_p1_elims = 0
    for p in next_state.players:
        if p.player_number == 1:
            after_p1_elims = p.eliminated_rings
            break

    after_total_elims = next_state.total_rings_eliminated

    assert after_p1_elims == before_p1_elims + total_internal_height, (
        "Expected all internal rings from HX-A mini-region to be credited "
        "to player 1; "
        f"before={before_p1_elims}, "
        f"total_internal_height={total_internal_height}, "
        f"after={after_p1_elims}"
    )

    assert after_total_elims == before_total_elims + total_internal_height, (
        "Expected total_rings_eliminated to increase by "
        "total_internal_height; "
        f"before={before_total_elims}, "
        f"total_internal_height={total_internal_height}, "
        f"after={after_total_elims}"
    )


# ────────────────────────────────────────────────────────────────────────────────
# HX-C: hex single-cell territory no-op → forced elimination
# ────────────────────────────────────────────────────────────────────────────────


def create_hex_single_cell_forced_elimination_state() -> GameState:
    """
    Create a minimal hex GameState for HX-C ANM/FE semantics.

    Board / state shape:

    - Board type: hexagonal.
    - Board is a single valid hex cell (size=1, radius=0) at (0,0,0).
    - Player 1 controls a single stack on that cell (height 1).
    - Player 1 has no rings in hand.
    - Current phase: territory_processing.
    - Current player: 1.

    This guarantees:

    - No legal movement or capture actions (no neighbours are in-bounds).
    - No disconnected territory regions to process.
    - Forced elimination is required once the territory phase is exhausted.

    The test uses this state to assert that:
    - NO_TERRITORY_ACTION is recorded in TERRITORY_PROCESSING.
    - After that bookkeeping move, the phase machine advances to
      FORCED_ELIMINATION.
    - Forced elimination moves are only surfaced in the dedicated
      forced_elimination phase, never as TERRITORY_PROCESSING moves.
    """
    board = BoardState(type=BoardType.HEXAGONAL, size=1)

    origin = Position(x=0, y=0, z=0)
    board.stacks[origin.to_key()] = RingStack(
        position=origin,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    now = datetime.now()

    players = [
        Player(
            id="p1",
            username="Player1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=0,  # No rings in hand; only the on-board stack.
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="Player2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=0,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    total_rings_in_play = sum(
        stack.stack_height for stack in board.stacks.values()
    )

    return GameState(  # type: ignore[call-arg]
        id="hex-hx-c-single-cell",
        boardType=BoardType.HEXAGONAL,
        board=board,
        players=players,
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=total_rings_in_play,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def test_hex_territory_noop_then_forced_elimination_decision() -> None:
    """
    HX-C: no territory action then forced elimination as a separate phase.

    Python analogue of the TS HX-C test:

    - Start in TERRITORY_PROCESSING on a hex single-cell board with a
      single P1 stack and no rings in hand.
    - There are no processable territory regions, so hosts must emit an
      explicit NO_TERRITORY_ACTION bookkeeping move.
    - After that move, the phase machine advances to FORCED_ELIMINATION.
    - Forced-elimination moves are only surfaced in FORCED_ELIMINATION,
      never as TERRITORY_PROCESSING moves.
    """
    state = create_hex_single_cell_forced_elimination_state()

    assert state.board.type == BoardType.HEXAGONAL
    assert state.current_phase == GamePhase.TERRITORY_PROCESSING
    assert state.current_player == 1

    # Precondition: in TERRITORY_PROCESSING there should be no forced
    # elimination moves; any interactive moves here would be
    # PROCESS_TERRITORY_REGION, and in this HX-C slice there should be
    # none.
    GameEngine.clear_cache()
    territory_moves = GameEngine.get_valid_moves(state, 1)
    assert all(
        m.type != MoveType.FORCED_ELIMINATION for m in territory_moves
    ), "Forced elimination must not be surfaced in TERRITORY_PROCESSING"

    # From the engine's perspective, the lack of territory decisions is
    # expressed via a NO_TERRITORY_ACTION_REQUIRED phase requirement.
    requirement = GameEngine.get_phase_requirement(state, 1)
    assert requirement is not None
    assert (
        requirement.type
        == PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED
    )

    # Synthesize and apply the canonical NO_TERRITORY_ACTION bookkeeping
    # move.
    no_territory_move = GameEngine.synthesize_bookkeeping_move(
        requirement,
        state,
    )
    assert no_territory_move.type == MoveType.NO_TERRITORY_ACTION

    next_state = GameEngine.apply_move(state, no_territory_move)

    # After NO_TERRITORY_ACTION, the phase machine must leave
    # TERRITORY_PROCESSING and enter FORCED_ELIMINATION for this HX-C
    # ANM slice.
    assert next_state.current_phase == GamePhase.FORCED_ELIMINATION
    assert next_state.current_player == 1

    # In the forced_elimination phase, forced_elimination moves are surfaced
    # directly as interactive decisions via get_valid_moves rather than via a
    # additional phase requirement.
    origin_key = Position(x=0, y=0, z=0).to_key()

    # And get_valid_moves in FORCED_ELIMINATION must surface only explicit
    # FORCED_ELIMINATION moves, never territory-processing moves, and they
    # must all target the origin stack.
    GameEngine.clear_cache()
    fe_moves = GameEngine.get_valid_moves(next_state, 1)
    assert fe_moves, (
        "Expected forced_elimination moves in FORCED_ELIMINATION phase"
    )
    assert all(m.type == MoveType.FORCED_ELIMINATION for m in fe_moves)
    target_keys = {
        m.to.to_key()
        for m in fe_moves
        if m.to is not None
    }
    assert target_keys == {origin_key}


if __name__ == "__main__":  # pragma: no cover - manual debug helper
    # Allow running this module directly for quick iteration:
    #   cd ai-service && python -m pytest \
    #       tests/test_hex_territory_fe_fixtures.py -q
    pytest.main([__file__])