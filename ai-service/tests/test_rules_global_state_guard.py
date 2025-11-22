import unittest
from unittest.mock import MagicMock

# Import from the local app package (tests adjust sys.path similarly elsewhere)
from app.game_engine import GameEngine
from app.board_manager import BoardManager


class TestRulesGlobalStateGuard(unittest.TestCase):
    """Guard tests to detect accidental global mocking of core rules classes.

    These tests intentionally assert on *types* of key engine entrypoints so
    that any test which globally replaces them with MagicMocks (or similar)
    without proper patching/teardown will cause an immediate, easy-to-diagnose
    failure.

    This is a regression guard for historical issues where tests globally
    monkeypatched GameEngine methods, which in turn broke the FastAPI
    /rules/evaluate_move endpoint and rules-parity harness by returning
    MagicMocks instead of real Move/GameState models.
    """

    def test_game_engine_methods_not_globally_mocked(self) -> None:
        """GameEngine core methods should not be left as MagicMocks globally.

        Tests that need mocking must use context-managed patching (e.g.
        ``with patch.object(...)``) or decorators so that these methods are
        always restored for other tests, especially API and parity tests.
        """

        self.assertFalse(
            isinstance(GameEngine.get_valid_moves, MagicMock),
            msg="GameEngine.get_valid_moves appears to be globally mocked; use\n"
                "context-managed patching in the specific test instead.",
        )
        self.assertFalse(
            isinstance(GameEngine.apply_move, MagicMock),
            msg="GameEngine.apply_move appears to be globally mocked; use\n"
                "context-managed patching in the specific test instead.",
        )

    def test_board_manager_helpers_not_globally_mocked(self) -> None:
        """BoardManager helpers used by parity/API must not be globally mocked."""

        self.assertFalse(
            isinstance(BoardManager.hash_game_state, MagicMock),
            msg="BoardManager.hash_game_state appears to be globally mocked;\n"
                "this will break TS↔Python rules parity.",
        )
        self.assertFalse(
            isinstance(BoardManager.compute_progress_snapshot, MagicMock),
            msg="BoardManager.compute_progress_snapshot appears to be globally\n"
                "mocked; this will break TS↔Python rules parity and S-invariant\n"
                "checks.",
        )


if __name__ == "__main__":
    unittest.main()
