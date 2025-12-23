"""
Multi-board evaluation fitness tests for CMA-ES optimization.

NOTE: These tests play full AI games across multiple board types and can
exceed the default pytest timeout (60s). They are skipped by default in CI.
Run locally with: pytest tests/test_multi_board_evaluation.py -v --timeout=300
"""

import os
import sys
import unittest

import pytest

# Ensure app.* and scripts.* imports resolve when running tests
# directly under ai-service/.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.heuristic_weights import (  # type: ignore
    BASE_V1_BALANCED_WEIGHTS,
)
from app.models import BoardType  # type: ignore
from scripts.run_cmaes_optimization import (  # type: ignore
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

    # SKIP-REASON: KEEP-SKIPPED - slow integration test (full AI games), run locally with --timeout=300
    @pytest.mark.skip(
        reason="Slow integration test: plays full AI games across 3 board types. "
        "Run locally with: pytest tests/test_multi_board_evaluation.py -v --timeout=300"
    )
    def test_zero_profile_is_worse_than_baseline_across_boards(self) -> None:
        """
        Sanity-check that the multi-board fitness helper meaningfully
        distinguishes a clearly bad heuristic profile (all weights = 0)
        from the canonical balanced baseline when evaluating across the
        standard board set.

        This mirrors the single-board CMA-ES sanity tests but exercises
        evaluate_fitness_over_boards over multiple BoardType values.
        """
        baseline = dict(BASE_V1_BALANCED_WEIGHTS)
        zero_profile = dict.fromkeys(baseline.keys(), 0.0)

        boards = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]

        baseline_agg, _baseline_per_board = evaluate_fitness_over_boards(
            candidate_weights=baseline,
            baseline_weights=baseline,
            games_per_eval=2,
            boards=boards,
            opponent_mode="baseline-only",
            max_moves=50,
            verbose=False,
            eval_mode="initial-only",
            state_pool_id=None,
            seed=12345,
        )

        zero_agg, _zero_per_board = evaluate_fitness_over_boards(
            candidate_weights=zero_profile,
            baseline_weights=baseline,
            games_per_eval=2,
            boards=boards,
            opponent_mode="baseline-only",
            max_moves=50,
            verbose=False,
            eval_mode="initial-only",
            state_pool_id=None,
            seed=12345,
        )

        # Aggregate fitness must remain in [0,1] and the zero profile
        # should not outperform the baseline across the combined boards.
        self.assertGreaterEqual(baseline_agg, 0.0)
        self.assertLessEqual(baseline_agg, 1.0)
        self.assertGreaterEqual(zero_agg, 0.0)
        self.assertLessEqual(zero_agg, 1.0)
        self.assertLess(zero_agg, baseline_agg)


if __name__ == "__main__":
    unittest.main()
