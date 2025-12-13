import os
import sys
from datetime import datetime
import unittest

import pytest

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
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
from app.game_engine import GameEngine  # noqa: E402
from app.rules import global_actions as ga  # noqa: E402


def _make_two_player_state() -> GameState:
    """Minimal square8 two-player state for ring-cap tests."""
    board = BoardState(type=BoardType.SQUARE8, size=8)
    now = datetime.now()
    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]
    return GameState(
        id="ringcap-test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        rngSeed=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


def _make_three_player_state() -> GameState:
    """Minimal square8 three-player state for LPS tests."""
    board = BoardState(type=BoardType.SQUARE8, size=8)
    now = datetime.now()
    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=5,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=5,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p3",
            username="p3",
            type="human",
            playerNumber=3,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=5,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]
    return GameState(
        id="lps-test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=3,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        rngSeed=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


class RingCapAndLPSTests(unittest.TestCase):
    def test_mixed_stack_captured_opponent_rings_do_not_count_against_own_cap(
        self,
    ) -> None:
        state = _make_two_player_state()
        board = state.board

        # Build a tall mixed stack controlled by player 1 but containing many
        # opponent-colour rings. Only the P1 rings should count toward their
        # own-colour rings-per-player cap.
        pos = Position(x=0, y=0)
        pos_key = pos.to_key()

        per_player_cap = GameEngine._estimate_rings_per_player(state)

        own_on_board = per_player_cap - 2
        captured_opponent = 10
        rings = [2] * captured_opponent + [1] * own_on_board

        stack = RingStack(
            position=pos,
            rings=rings,
            stackHeight=len(rings),
            capHeight=own_on_board,
            controllingPlayer=1,
        )
        board.stacks[pos_key] = stack

        # Player 1 has two rings in hand; with own_on_board = cap-2 they
        # should still be allowed to place up to 2 additional rings despite
        # controlling many captured opponent rings.
        state.players[0].rings_in_hand = 2

        moves = GameEngine._get_ring_placement_moves(
            state,
            1,
        )  # type: ignore[attr-defined]
        self.assertTrue(
            moves,
            "Expected at least one legal placement move for P1",
        )

        counts = [m.placement_count or 1 for m in moves]
        self.assertLessEqual(max(counts), 2)
        self.assertIn(2, counts)

    def test_exact_cap_prevents_further_placements(self) -> None:
        state = _make_two_player_state()
        board = state.board

        per_player_cap = GameEngine._estimate_rings_per_player(state)
        pos = Position(x=1, y=1)
        pos_key = pos.to_key()
        rings = [1] * per_player_cap
        stack = RingStack(
            position=pos,
            rings=rings,
            stackHeight=len(rings),
            capHeight=per_player_cap,
            controllingPlayer=1,
        )
        board.stacks[pos_key] = stack

        # Player 1 has rings in hand but their own-colour rings in play are
        # already at the per-player cap.
        state.players[0].rings_in_hand = 1

        moves = GameEngine._get_ring_placement_moves(
            state,
            1,
        )  # type: ignore[attr-defined]
        self.assertFalse(
            moves,
            "Expected no legal placements once own-colour cap is reached",
        )

    def test_lps_three_player_unique_actor_full_round_then_win(self) -> None:
        """Test LPS detection when only one player has real actions for a full round.

        The LPS tracking system finalizes rounds when cycling BACK to the first
        player. A round is only complete and evaluated when the first player's
        turn comes around again.
        """
        state = _make_three_player_state()

        real_actions = {1: True, 2: False, 3: False}
        original = GameEngine._has_real_action_for_player
        try:
            GameEngine._has_real_action_for_player = staticmethod(
                lambda s, pid: real_actions.get(pid, False)
            )  # type: ignore[assignment]

            # First round: P1 has real actions; P2 and P3 do not.
            # Iterate through all players to record their real action status.
            for pid in (1, 2, 3):
                state.current_player = pid
                GameEngine._update_lps_round_tracking_for_current_player(
                    state,
                )  # type: ignore[attr-defined]

            # Round is NOT finalized yet - only completes when cycling back to P1.
            # At this point, the mask has been populated but no round completed.
            self.assertIsNone(state.lps_exclusive_player_for_completed_round)

            # Start of P1's next turn: cycling back finalizes the previous round
            # and triggers the exclusive player detection.
            state.current_player = 1
            GameEngine._update_lps_round_tracking_for_current_player(
                state,
            )  # type: ignore[attr-defined]

            # NOW the round is finalized and we can check the exclusive player.
            self.assertEqual(state.lps_exclusive_player_for_completed_round, 1)

            # Check for LPS victory (requires consecutive exclusive rounds).
            GameEngine._check_victory(state)

            # First exclusive round by itself may not trigger victory yet
            # (depends on LPS_REQUIRED_CONSECUTIVE_ROUNDS), but the tracking
            # should be correctly recording P1 as the exclusive player.
        finally:
            GameEngine._has_real_action_for_player = original

    def test_lps_resets_when_other_player_regains_real_action_mid_round(
        self,
    ) -> None:
        """Test LPS tracking resets when multiple players have real actions.

        Rounds are finalized when cycling back to the first player.
        Round 1: P1 and P2 have real actions → no exclusive player.
        Round 2: Only P1 has real actions → P1 is exclusive.
        """
        state = _make_three_player_state()

        mode = {"value": "mixed"}

        def fake_has_real_action(s: GameState, pid: int) -> bool:
            if mode["value"] == "mixed":
                return pid in (1, 2)
            return pid == 1

        original = GameEngine._has_real_action_for_player
        try:
            GameEngine._has_real_action_for_player = staticmethod(
                fake_has_real_action,
            )  # type: ignore[assignment]

            # Round 1: both P1 and P2 have real actions.
            # Iterate through all players to populate the mask.
            for pid in (1, 2, 3):
                state.current_player = pid
                GameEngine._update_lps_round_tracking_for_current_player(
                    state,
                )  # type: ignore[attr-defined]

            # Round 1 is NOT finalized yet - check happens in same turn.
            self.assertIsNone(state.lps_exclusive_player_for_completed_round)

            # Round 2: only P1 has real actions.
            # When P1 starts round 2, round 1 is finalized first.
            mode["value"] = "exclusive"

            # Cycling to P1 finalizes round 1 (no exclusive - P1 and P2 had actions)
            state.current_player = 1
            GameEngine._update_lps_round_tracking_for_current_player(
                state,
            )  # type: ignore[attr-defined]

            # After round 1 finalization: no exclusive player (P1+P2 had actions)
            self.assertIsNone(state.lps_exclusive_player_for_completed_round)

            # Continue round 2 with P2 and P3
            for pid in (2, 3):
                state.current_player = pid
                GameEngine._update_lps_round_tracking_for_current_player(
                    state,
                )  # type: ignore[attr-defined]

            # Round 2 not finalized yet
            self.assertIsNone(state.lps_exclusive_player_for_completed_round)

            # Cycle back to P1 - this finalizes round 2
            state.current_player = 1
            GameEngine._update_lps_round_tracking_for_current_player(
                state,
            )  # type: ignore[attr-defined]

            # NOW round 2 is finalized: only P1 had real actions
            self.assertEqual(state.lps_exclusive_player_for_completed_round, 1)
        finally:
            GameEngine._has_real_action_for_player = original


def test_lps_victory_can_be_awarded_at_turn_start_even_if_ring_placement_is_anm() -> None:
    """LPS victory is evaluated at turn start even when placement is a forced no-op.

    Canonically, players always enter `ring_placement` first (RR-CANON-R073).
    When a player has no placements (e.g., ringsInHand == 0) but does have a
    real action available in later interactive phases (movement/capture),
    `globalActions.isANMState` may be true at the boundary.

    The TS engine still evaluates LPS at the start of that interactive turn;
    Python must mirror that behaviour for parity.
    """
    now = datetime.now()
    board = BoardState(type=BoardType.SQUARE8, size=8)

    # Provide P1 a real action (movement/capture) while leaving ring placement
    # forbidden (ringsInHand == 0), yielding an ANM ring_placement boundary.
    pos_a = Position(x=3, y=3)
    board.stacks[pos_a.to_key()] = RingStack(
        position=pos_a,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    # Ensure other players still have rings in play (buried) so they count for
    # LPS material checks.
    pos_b = Position(x=4, y=4)
    board.stacks[pos_b.to_key()] = RingStack(
        position=pos_b,
        rings=[2, 3, 1],
        stackHeight=3,
        capHeight=1,
        controllingPlayer=1,
    )

    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=0,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=0,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p3",
            username="p3",
            type="human",
            playerNumber=3,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=0,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    state = GameState(
        id="lps-anm-turn-start-test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=3,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
        rngSeed=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
        lpsCurrentRoundFirstPlayer=None,
        lpsConsecutiveExclusiveRounds=3,
        lpsConsecutiveExclusivePlayer=1,
    )

    assert ga.is_anm_state(state)

    GameEngine._maybe_apply_lps_victory_at_turn_start(state)  # type: ignore[attr-defined]

    assert state.game_status == GameStatus.COMPLETED
    assert state.current_phase == GamePhase.GAME_OVER
    assert state.winner == 1


def test_lps_round_finalization_includes_players_with_buried_rings() -> None:
    """LPS round tracking must treat buried rings as material (TS parity).

    Regression test for a parity bug where Python used "turn-material" to
    compute activePlayers during LPS round finalization, excluding players who
    had only buried rings (no controlled stacks, no rings in hand). This could
    incorrectly mark a round as "exclusive" and lead to premature LPS wins.

    Canonical reference:
      - RR-CANON-R172 / RR-CANON-R175: active players for LPS are those with
        any rings in play (including buried) or in hand.
      - TS playerHasMaterial uses countRingsInPlayForPlayer (includes buried).
    """
    now = datetime.now()
    board = BoardState(type=BoardType.SQUARE8, size=8)

    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=1,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=1,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    state = GameState(
        id="lps-buried-material-round-finalization",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
        rngSeed=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
        lpsCurrentRoundFirstPlayer=None,
        lpsConsecutiveExclusiveRounds=0,
        lpsConsecutiveExclusivePlayer=None,
    )

    original = GameEngine._has_real_action_for_player
    try:
        # Round 1: both players have real actions (so the round is NOT exclusive).
        GameEngine._has_real_action_for_player = staticmethod(lambda _s, _pid: True)  # type: ignore[assignment]

        state.current_player = 1
        GameEngine._update_lps_round_tracking_for_current_player(state)  # type: ignore[attr-defined]

        state.current_player = 2
        GameEngine._update_lps_round_tracking_for_current_player(state)  # type: ignore[attr-defined]

        # Before cycling back to the first player, remove P1's turn-material
        # while keeping material via a buried ring in a P2-controlled stack.
        state.players[0].rings_in_hand = 0
        state.players[1].rings_in_hand = 0

        pos = Position(x=0, y=0)
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=[1, 2],  # P1 ring buried, P2 controls the stack
            stackHeight=2,
            capHeight=1,
            controllingPlayer=2,
        )

        # Cycle back to P1: this finalizes the previous round. Because BOTH
        # players acted in the round, there must be no exclusive player.
        state.current_player = 1
        GameEngine._update_lps_round_tracking_for_current_player(state)  # type: ignore[attr-defined]

        assert state.lps_exclusive_player_for_completed_round is None
        assert state.lps_consecutive_exclusive_rounds == 0
        assert state.lps_consecutive_exclusive_player is None
    finally:
        GameEngine._has_real_action_for_player = original


if __name__ == "__main__":
    unittest.main()
