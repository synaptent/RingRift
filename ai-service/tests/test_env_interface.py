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

    def test_reset_multi_player_initial_state(self):
        """Verify that RingRiftEnv.reset() respects num_players for N-player games.

        This exercises the Python initial-state helper used by the training
        environment and checks that key thresholds match the shared
        TypeScript/BOARD_CONFIGS semantics for square8.
        """
        # square8 configuration mirrored from src/shared/types/game.ts
        rings_per_player = 18
        total_spaces = 64

        for num_players in (3, 4):
            env = RingRiftEnv(BoardType.SQUARE8, num_players=num_players)
            state = env.reset()

            # Player list and max_players should reflect num_players.
            self.assertEqual(len(state.players), num_players)
            self.assertEqual(state.max_players, num_players)

            player_numbers = sorted(p.player_number for p in state.players)
            self.assertEqual(player_numbers, list(range(1, num_players + 1)))

            # Victory threshold scales with total rings across all players.
            total_rings = rings_per_player * num_players
            expected_victory_threshold = (total_rings // 2) + 1
            expected_territory_threshold = (total_spaces // 2) + 1

            self.assertEqual(state.victory_threshold, expected_victory_threshold)
            self.assertEqual(
                state.territory_victory_threshold,
                expected_territory_threshold,
            )

    def test_turn_rotation_multi_player(self):
        """Smoke-test that current_player behaves sensibly in N-player games.

        We do not assert an exact sequence of moves (that is rules-engine
        territory), but we do require that the turn logic only ever yields
        player indices in the expected range and that, when there are legal
        moves, the env can make progress without getting stuck.
        """
        for num_players in (3, 4):
            env = RingRiftEnv(BoardType.SQUARE8, num_players=num_players)
            state = env.reset()

            seen = set()
            # Play up to a few full rounds of turns, or until the game ends
            # or no legal moves are available for the current player.
            for _ in range(num_players * 4):
                seen.add(state.current_player)
                moves = env.legal_moves()
                if not moves:
                    break

                move = moves[0]
                state, _reward, done, _info = env.step(move)
                if done:
                    break

            expected_players = set(range(1, num_players + 1))
            # All observed players must be within the configured range.
            self.assertTrue(
                seen.issubset(expected_players),
                f"Observed invalid player numbers {seen} for num_players={num_players}",
            )
            # As a minimal sanity check, we should see at least two distinct
            # players in typical games.
            self.assertGreaterEqual(
                len(seen),
                min(2, num_players),
                f"Expected to observe at least two players taking turns for num_players={num_players}",
            )


if __name__ == '__main__':
    unittest.main()
