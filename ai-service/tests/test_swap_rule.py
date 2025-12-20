import os
import sys
from datetime import datetime

import pytest

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.board_manager import BoardManager
from app.game_engine import GameEngine
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    TimeControl,
)
from app.rules.default_engine import DefaultRulesEngine


def _make_swap_enabled_state() -> GameState:
    """Construct a minimal 2-player SQUARE8 GameState with the pie rule enabled."""
    now = datetime.now()
    board = BoardState(type=BoardType.SQUARE8, size=8)
    players = [
        Player(
            id="p1",
            username="P1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="P2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=20,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    return GameState(
        id="swap-rule-test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=40,
        totalRingsEliminated=0,
        victoryThreshold=21,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
        rngSeed=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
        rulesOptions={"swapRuleEnabled": True},
    )


def _contains_swap_sides(moves) -> bool:
    return any(m.type == MoveType.SWAP_SIDES for m in moves)


def _count_swap_sides(moves) -> int:
    return sum(1 for m in moves if m.type == MoveType.SWAP_SIDES)


def _dummy_move(
    *,
    move_type: MoveType,
    player: int,
    move_number: int,
    ts: datetime,
) -> Move:
    return Move(
        id=f"dummy-{player}-{move_number}",
        type=move_type,
        player=player,
        to=Position(x=0, y=0),
        timestamp=ts,
        thinkTime=0,
        moveNumber=move_number,
    )


def test_swap_rule_offered_exactly_once_for_player_two_after_p1_first_move() -> None:
    """
    GameEngine.get_valid_moves should surface a single SWAP_SIDES move for
    Player 2 immediately after Player 1's first non-swap move, and never
    before or after that window.
    """
    state = _make_swap_enabled_state()

    # Before any moves, swap_sides should not be offered to either player.
    p1_moves_initial = GameEngine.get_valid_moves(state, 1)
    p2_moves_initial = GameEngine.get_valid_moves(state, 2)

    assert not _contains_swap_sides(p1_moves_initial)
    assert not _contains_swap_sides(p2_moves_initial)

    # Apply Player 1's first ring placement via the canonical engine.
    p1_placement = next(
        (m for m in p1_moves_initial if m.type == MoveType.PLACE_RING),
        None,
    )
    assert p1_placement is not None, "Expected at least one PLACE_RING move for P1"

    state_after_p1 = GameEngine.apply_move(state, p1_placement)

    # Synthesise the start of Player 2's interactive turn by updating
    # current_player while preserving board + history. This mirrors the
    # TS backend context in which the pie rule is offered.
    state_for_p2_turn = state_after_p1.model_copy(
        update={"current_player": 2}
    )

    # Immediately after P1's first move, P2 should see exactly one SWAP_SIDES move.
    p2_moves_after_p1 = GameEngine.get_valid_moves(state_for_p2_turn, 2)
    assert _count_swap_sides(p2_moves_after_p1) == 1

    # P1 should never see swap_sides in this window.
    p1_moves_after_p1 = GameEngine.get_valid_moves(state_after_p1, 1)
    assert not _contains_swap_sides(p1_moves_after_p1)

    # After P2 takes any non-swap move, swap_sides should no longer be offered.
    non_swap_for_p2 = next(
        (m for m in p2_moves_after_p1 if m.type != MoveType.SWAP_SIDES),
        None,
    )
    if non_swap_for_p2 is not None:
        state_after_p2_non_swap = GameEngine.apply_move(
            state_for_p2_turn, non_swap_for_p2
        )
        state_after_p2_for_gate = state_after_p2_non_swap.model_copy(
            update={"current_player": 2}
        )
        p2_moves_after_non_swap = GameEngine.get_valid_moves(
            state_after_p2_for_gate, 2
        )
        assert not _contains_swap_sides(p2_moves_after_non_swap)


def test_default_rules_engine_apply_swap_sides_swaps_identities_not_stats() -> None:
    """
    DefaultRulesEngine.apply_move must mirror GameEngine.apply_move for
    SWAP_SIDES:

    - Player identities (id/username/type/aiDifficulty) for seats 1 and 2
      are swapped.
    - Per-seat statistics (ringsInHand, eliminatedRings, territorySpaces,
      timeRemaining) remain attached to the original seats.
    - Board geometry and hash remain unchanged.
    - currentPlayer and currentPhase remain unchanged.
    """
    base_state = _make_swap_enabled_state()
    p1_moves = GameEngine.get_valid_moves(base_state, 1)
    p1_placement = next(
        (m for m in p1_moves if m.type == MoveType.PLACE_RING),
        None,
    )
    assert p1_placement is not None, "Expected at least one PLACE_RING move for P1"

    state_after_p1 = GameEngine.apply_move(base_state, p1_placement)
    state_for_p2_turn = state_after_p1.model_copy(update={"current_player": 2})
    p2_moves = GameEngine.get_valid_moves(state_for_p2_turn, 2)

    swap_move = next(
        (m for m in p2_moves if m.type == MoveType.SWAP_SIDES),
        None,
    )
    assert swap_move is not None, "Expected SWAP_SIDES to be offered to P2"

    # Capture a deep snapshot of the pre-swap state for comparison.
    before = state_for_p2_turn.model_copy(deep=True)
    before_hash = BoardManager.hash_game_state(before)

    engine = DefaultRulesEngine(mutator_first=False)
    after = engine.apply_move(state_for_p2_turn, swap_move)

    # Board hash and meta should be unchanged by swap_sides.
    after_hash = BoardManager.hash_game_state(after)
    assert after_hash == before_hash

    # currentPlayer / currentPhase are preserved.
    assert after.current_player == before.current_player
    assert after.current_phase == before.current_phase

    # Per-seat stats should remain attached to seats 1 and 2.
    before_p1 = next(p for p in before.players if p.player_number == 1)
    before_p2 = next(p for p in before.players if p.player_number == 2)
    after_p1 = next(p for p in after.players if p.player_number == 1)
    after_p2 = next(p for p in after.players if p.player_number == 2)

    assert after_p1.rings_in_hand == before_p1.rings_in_hand
    assert after_p1.eliminated_rings == before_p1.eliminated_rings
    assert after_p1.territory_spaces == before_p1.territory_spaces

    assert after_p2.rings_in_hand == before_p2.rings_in_hand
    assert after_p2.eliminated_rings == before_p2.eliminated_rings
    assert after_p2.territory_spaces == before_p2.territory_spaces

    # Identity/meta fields should be swapped between seats 1 and 2.
    assert after_p1.id == before_p2.id
    assert after_p1.username == before_p2.username
    assert after_p1.type == before_p2.type
    assert after_p1.ai_difficulty == before_p2.ai_difficulty

    assert after_p2.id == before_p1.id
    assert after_p2.username == before_p1.username
    assert after_p2.type == before_p1.type
    assert after_p2.ai_difficulty == before_p1.ai_difficulty


def test_swap_sides_not_offered_again_after_swap_is_applied() -> None:
    """swap_sides must be offered at most once even with move caching.

    Regression guard: swap_sides is a meta-move that does not change the
    canonical structural hash_game_state. If the move cache key ignores
    move_history, swap_sides can remain cached and be offered repeatedly,
    leading to non-terminating self-play loops.
    """
    GameEngine.clear_cache()
    state = _make_swap_enabled_state()

    p1_moves = GameEngine.get_valid_moves(state, 1)
    p1_placement = next(
        (m for m in p1_moves if m.type == MoveType.PLACE_RING),
        None,
    )
    assert p1_placement is not None
    state_after_p1 = GameEngine.apply_move(state, p1_placement)

    state_for_p2_turn = state_after_p1.model_copy(update={"current_player": 2})
    p2_moves = GameEngine.get_valid_moves(state_for_p2_turn, 2)
    swap_move = next((m for m in p2_moves if m.type == MoveType.SWAP_SIDES), None)
    assert swap_move is not None

    state_after_swap = GameEngine.apply_move(state_for_p2_turn, swap_move)

    p2_moves_after_swap = GameEngine.get_valid_moves(state_after_swap, 2)
    assert not _contains_swap_sides(p2_moves_after_swap)


def test_swap_sides_cache_key_respects_rules_options_swap_rule_enabled() -> None:
    """swap_sides must not leak across rulesOptions variants via caching.

    Regression guard: swap_sides eligibility depends on rulesOptions.swapRuleEnabled,
    but that flag does not affect the canonical structural hash_game_state. If the
    move cache key ignores rulesOptions, a cached move surface from a swap-enabled
    caller can incorrectly be reused for a swap-disabled caller.
    """
    GameEngine.clear_cache()
    state = _make_swap_enabled_state()

    p1_moves = GameEngine.get_valid_moves(state, 1)
    p1_placement = next((m for m in p1_moves if m.type == MoveType.PLACE_RING), None)
    assert p1_placement is not None
    state_after_p1 = GameEngine.apply_move(state, p1_placement)

    state_for_p2_turn = state_after_p1.model_copy(update={"current_player": 2})
    p2_moves_enabled = GameEngine.get_valid_moves(state_for_p2_turn, 2)
    assert _count_swap_sides(p2_moves_enabled) == 1

    # Same board/history, but swap disabled.
    state_for_p2_turn_disabled = state_for_p2_turn.model_copy(
        update={"rules_options": {"swapRuleEnabled": False}}
    )
    p2_moves_disabled = GameEngine.get_valid_moves(state_for_p2_turn_disabled, 2)
    assert _count_swap_sides(p2_moves_disabled) == 0


def test_swap_sides_cache_key_accounts_for_p2_non_swap_move_presence() -> None:
    """swap_sides eligibility depends on move history composition.

    Regression guard: in canonical history, a single player may record
    multiple moves per turn (bookkeeping + choices). Using only
    ``move_history`` length in the move cache key can therefore cause a
    swap-enabled move surface to leak into a state where Player 2 has
    already taken a non-swap move, producing illegal SWAP_SIDES actions
    in self-play.
    """
    GameEngine.clear_cache()
    base = _make_swap_enabled_state()
    ts = datetime.now()

    p1_only_history = [
        _dummy_move(
            move_type=MoveType.PLACE_RING,
            player=1,
            move_number=i + 1,
            ts=ts,
        )
        for i in range(7)
    ]

    swap_eligible = base.model_copy(
        update={
            "current_player": 2,
            "move_history": p1_only_history,
        }
    )
    moves_swap_eligible = GameEngine.get_valid_moves(swap_eligible, 2)
    assert _count_swap_sides(moves_swap_eligible) == 1

    not_swap_eligible = base.model_copy(
        update={
            "current_player": 2,
            "move_history": [*p1_only_history[:6], _dummy_move(move_type=MoveType.SKIP_PLACEMENT, player=2, move_number=7, ts=ts)],
        }
    )
    moves_not_swap_eligible = GameEngine.get_valid_moves(not_swap_eligible, 2)
    assert _count_swap_sides(moves_not_swap_eligible) == 0
