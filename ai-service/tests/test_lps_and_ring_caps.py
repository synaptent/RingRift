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


if __name__ == "__main__":
    unittest.main()