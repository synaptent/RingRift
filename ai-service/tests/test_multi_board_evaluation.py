import os
import sys
import unittest

# Ensure app.* and scripts.* imports resolve when running tests
# directly under ai-service/.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
)
from app.models import BoardType  # type: ignore  # noqa: E402
from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    evaluate_fitness_over_boards,
)


class MultiBoardEvaluationTest(unittest.TestCase):
    def test_single_board_aggregate_matches_per_board(self) -> None:
        baseline = dict(BASE_V1_BALANCED_WEIGHTS)
        boards = [BoardType.SQUARE8]

        agg, per_board = evaluate_fitness_over_boards(
            candidate_weights=baseline,
            baseline_weights=baseline,
            games_per_eval=2,
            boards=boards,
            opponent_mode="baseline-only",
            max_moves=10,
            verbose=False,
            eval_mode="initial-only",
            state_pool_id=None,
            seed=12345,
        )

        self.assertIn(BoardType.SQUARE8, per_board)
        self.assertAlmostEqual(agg, per_board[BoardType.SQUARE8])
        self.assertGreaterEqual(agg, 0.0)
        self.assertLessEqual(agg, 1.0)


if __name__ == "__main__":
    unittest.main()