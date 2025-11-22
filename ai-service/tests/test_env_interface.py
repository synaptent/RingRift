import unittest
import sys
import os
from unittest.mock import patch

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import BoardType, GameStatus
from app.game_engine import GameEngine
from app.training.env import RingRiftEnv


class TestRingRiftEnv(unittest.TestCase):
    def test_reset_and_step_parity(self):
        """
        Verify that env.step() produces the same state as GameEngine.apply_move()
        """
        env = RingRiftEnv(BoardType.SQUARE8)
        state = env.reset()

        # Get a valid move
        moves = env.legal_moves()
        self.assertTrue(len(moves) > 0)
        move = moves[0]

        # Apply via env
        new_state_env, reward, done, info = env.step(move)

        # Apply via engine directly
        new_state_direct = GameEngine.apply_move(state, move)

        # Check parity
        # We can't compare objects directly because apply_move creates deep copies
        # and timestamps might differ slightly if generated inside apply_move
        # (though they shouldn't)
        # But the board state should be identical
        self.assertEqual(
            new_state_env.board.stacks, new_state_direct.board.stacks
        )
        self.assertEqual(
            new_state_env.current_player, new_state_direct.current_player
        )
        self.assertEqual(
            new_state_env.current_phase, new_state_direct.current_phase
        )

        # Check internal env state update
        self.assertEqual(env.state.id, new_state_env.id)

    def test_terminal_reward_semantics(self):
        """
        Verify terminal reward logic (+1/-1/0)
        """
        env = RingRiftEnv(BoardType.SQUARE8, reward_on="terminal")
        env.reset()

        # Force a terminal state
        # We'll manually set the winner on the internal state to simulate end
        # of game. This is a bit hacky but avoids playing a full game
        if env._state:
            env._state.game_status = GameStatus.FINISHED
            env._state.winner = 1

        # Create a dummy move from player 1 (winner)
        # We need a move to pass to step(), even if state is already finished
        # In reality, step() calls apply_move() which might reject moves if
        # game is finished. So we need to be careful.
        # Instead, let's mock a state just BEFORE winning.

        # Actually, let's just test the reward logic by inspecting the code
        # path or by using a scenario where a move causes a win.
        # Constructing a winning state is complex.
        # Let's rely on the fact that step() returns reward based on
        # state.winner

        # Let's simulate the effect of a winning move by mocking
        # GameEngine.apply_move to return a finished state.

        # Use a context-managed patch to avoid leaking global state between
        # tests. This ensures GameEngine.apply_move is always restored
        # automatically, even if assertions fail.
        def mock_apply_win(game_state, move):
            new_state = game_state.model_copy(deep=True)
            new_state.game_status = GameStatus.FINISHED
            new_state.winner = 1
            return new_state

        with patch.object(GameEngine, "apply_move", side_effect=mock_apply_win):
            # Case 1: P1 moves and wins
            # Perspective: P1 (mover)
            # Reward should be +1
            dummy_move_p1 = env.legal_moves()[0]
            # Ensure move is from P1
            if dummy_move_p1.player != 1:
                # Force player 1 move
                dummy_move_p1 = dummy_move_p1.model_copy(update={"player": 1})

            _, reward, done, _ = env.step(dummy_move_p1)
            self.assertTrue(done)
            self.assertEqual(reward, 1.0)

            # Reset for next case
            env.reset()

            # Case 2: P2 moves and P1 wins (e.g. P2 ran out of time or made a
            # bad move? Or P2 self-eliminated?)
            # If P2 moves and P1 wins, P2 lost.
            # Perspective: P2 (mover)
            # Reward should be -1
            dummy_move_p2 = env.legal_moves()[0].model_copy(
                update={"player": 2}
            )

            _, reward, done, _ = env.step(dummy_move_p2)
            self.assertTrue(done)
            self.assertEqual(reward, -1.0)

    def test_legal_moves_consistency(self):
        """
        Verify env.legal_moves() matches GameEngine.get_valid_moves()
        """
        env = RingRiftEnv(BoardType.SQUARE8)
        state = env.reset()
        
        env_moves = env.legal_moves()
        engine_moves = GameEngine.get_valid_moves(state, state.current_player)
        
        self.assertEqual(len(env_moves), len(engine_moves))
        # We assume order is preserved or at least set content is same
        # Since Move objects are Pydantic models, equality works if fields match
        # But IDs might differ if generated dynamically?
        # GameEngine generates IDs dynamically.
        # Let's compare types and coordinates.

        self.assertEqual(env_moves[0].type, engine_moves[0].type)
        self.assertEqual(env_moves[0].to, engine_moves[0].to)


if __name__ == '__main__':
    unittest.main()
