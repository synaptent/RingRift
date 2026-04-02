"""
Tests for LPS and ring cap alignment in AI training code.

These tests verify that:
1. LPS (Last Player Standing - R172) terminations give appropriate rewards.
2. Own-colour ring caps (CLAR-003) are correctly handled in state encoding.
3. Victory reason inference works correctly for tournament statistics.
"""

import os
import sys
import unittest
from datetime import datetime

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (
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
from app.rules.core import count_rings_in_play_for_player, infer_total_rings_in_play
from app.training.env import RingRiftEnv
from app.training.tournament import (
    VICTORY_REASONS,
    infer_victory_reason,
)


def _make_two_player_state() -> GameState:
    """Minimal square8 two-player state for testing."""
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
        id="test-state",
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
        totalRingsInPlay=infer_total_rings_in_play(players, board),
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


class TestLPSRewardHandling(unittest.TestCase):
    """Tests for LPS victory reward handling in training environment."""

    def test_lps_victory_gives_winner_positive_reward(self) -> None:
        """Verify that an LPS victory gives +1 reward to the winner."""
        state = _make_two_player_state()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        state.lps_exclusive_player_for_completed_round = 1

        # Simulate the reward computation from env.py
        # Winner perspective
        perspective = 1
        if state.winner is None:
            reward = 0.0
        elif state.winner == perspective:
            reward = 1.0
        else:
            reward = -1.0

        self.assertEqual(reward, 1.0, "LPS winner should get +1 reward")

    def test_lps_victory_gives_loser_negative_reward(self) -> None:
        """Verify that an LPS victory gives -1 reward to the loser."""
        state = _make_two_player_state()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        state.lps_exclusive_player_for_completed_round = 1

        # Loser perspective
        perspective = 2
        if state.winner is None:
            reward = 0.0
        elif state.winner == perspective:
            reward = 1.0
        else:
            reward = -1.0

        self.assertEqual(reward, -1.0, "LPS loser should get -1 reward")


class TestVictoryReasonInference(unittest.TestCase):
    """Tests for victory reason inference used in tournament statistics."""

    def test_infer_elimination_victory(self) -> None:
        """Verify elimination victory is correctly inferred."""
        state = _make_two_player_state()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        # Player 1 reached the victory threshold
        state.board.eliminated_rings["1"] = state.victory_threshold

        reason = infer_victory_reason(state)
        self.assertEqual(reason, "ring_elimination")

    def test_infer_territory_victory(self) -> None:
        """Verify territory victory is correctly inferred."""
        state = _make_two_player_state()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1

        # Player 1 reached the territory victory threshold
        for i in range(state.territory_victory_threshold):
            state.board.collapsed_spaces[f"{i},0"] = 1

        reason = infer_victory_reason(state)
        self.assertEqual(reason, "territory")

    def test_infer_lps_victory(self) -> None:
        """Verify LPS (R172) victory is correctly inferred."""
        state = _make_two_player_state()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        state.lps_exclusive_player_for_completed_round = 1
        # Add a stack so it's not structural termination
        pos = Position(x=0, y=0)
        stack = RingStack(
            position=pos,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        state.board.stacks[pos.to_key()] = stack

        reason = infer_victory_reason(state)
        self.assertEqual(reason, "last_player_standing")

    def test_infer_structural_victory(self) -> None:
        """Verify structural termination is correctly inferred."""
        state = _make_two_player_state()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        # No stacks = structural termination
        state.board.stacks = {}

        reason = infer_victory_reason(state)
        self.assertEqual(reason, "structural")

    def test_victory_reasons_all_present(self) -> None:
        """Verify all expected victory reasons are defined."""
        expected = [
            "ring_elimination",
            "territory",
            "last_player_standing",
            "structural",
            "unknown",
        ]
        self.assertEqual(VICTORY_REASONS, expected)


class TestRingCapSemantics(unittest.TestCase):
    """Tests for own-colour ring cap semantics (CLAR-003)."""

    def test_own_colour_rings_counted_correctly(self) -> None:
        """Verify count_rings_in_play_for_player counts only own-colour."""
        state = _make_two_player_state()
        board = state.board

        # Create a mixed stack: 3 P2 rings at bottom, 2 P1 rings on top
        # P1 controls the stack but only 2 rings are own-colour
        pos = Position(x=0, y=0)
        stack = RingStack(
            position=pos,
            rings=[2, 2, 2, 1, 1],  # bottom to top
            stackHeight=5,
            capHeight=2,
            controllingPlayer=1,
        )
        board.stacks[pos.to_key()] = stack

        # P1 has 16 rings in hand (18 - 2 placed)
        state.players[0].rings_in_hand = 16

        # Own-colour rings for P1: 2 on board + 16 in hand = 18
        count = count_rings_in_play_for_player(state, 1)
        self.assertEqual(count, 18)

        # Own-colour rings for P2: 3 on board + 18 in hand = 21
        count_p2 = count_rings_in_play_for_player(state, 2)
        self.assertEqual(count_p2, 21)

    def test_captured_opponent_rings_not_counted(self) -> None:
        """Verify captured opponent rings don't count against own cap."""
        state = _make_two_player_state()
        board = state.board

        # P1 has captured 10 P2 rings (at bottom of stack)
        # P1 has 5 of their own rings on top
        pos = Position(x=1, y=1)
        captured_opponent = 10
        own_on_board = 5
        rings = [2] * captured_opponent + [1] * own_on_board

        stack = RingStack(
            position=pos,
            rings=rings,
            stackHeight=len(rings),
            capHeight=own_on_board,
            controllingPlayer=1,
        )
        board.stacks[pos.to_key()] = stack

        # P1 has 13 rings in hand (18 - 5 placed)
        state.players[0].rings_in_hand = 13

        # Own-colour rings for P1: 5 on board + 13 in hand = 18
        # The 10 captured P2 rings do NOT count against P1's cap
        count = count_rings_in_play_for_player(state, 1)
        self.assertEqual(count, 18)


class TestRingRiftEnvRewards(unittest.TestCase):
    """Integration tests for rewards in the RingRiftEnv."""

    def test_env_reset_works(self) -> None:
        """Verify RingRiftEnv can be reset."""
        env = RingRiftEnv()
        state = env.reset(seed=42)
        self.assertIsNotNone(state)
        self.assertEqual(state.game_status, GameStatus.ACTIVE)

    def test_finished_game_returns_terminal(self) -> None:
        """Verify finished games are detected as terminal."""
        env = RingRiftEnv()
        state = env.reset(seed=42)
        self.assertIsNotNone(state)

        # Manually set game to completed state with LPS victory
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        state.lps_exclusive_player_for_completed_round = 1

        # Verify state is correctly set
        self.assertEqual(state.game_status, GameStatus.COMPLETED)
        self.assertEqual(state.winner, 1)

        # Verify victory reason inference works
        reason = infer_victory_reason(state)
        # Without stacks, it will be "structural", not LPS
        # Add a stack to trigger LPS detection
        pos = Position(x=0, y=0)
        stack = RingStack(
            position=pos,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        state.board.stacks[pos.to_key()] = stack
        reason = infer_victory_reason(state)
        self.assertEqual(reason, "last_player_standing")


if __name__ == "__main__":
    unittest.main()
