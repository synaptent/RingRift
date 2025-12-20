import os
import sys

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.heuristic_weights import (  # type: ignore
    BASE_V1_BALANCED_WEIGHTS,
)
from app.models import BoardType  # type: ignore
from app.training.env import RingRiftEnv  # type: ignore
from scripts.diagnose_policy_equivalence import (  # type: ignore
    compare_moves_on_states,
)


def test_compare_moves_on_states_baseline_vs_baseline() -> None:
    """Baseline vs baseline should yield zero move differences."""
    env = RingRiftEnv(BoardType.SQUARE8)
    state = env.reset()
    states = [state]

    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    stats_same = compare_moves_on_states(baseline, baseline, states)

    assert stats_same["different_moves"] == 0
    assert stats_same["difference_rate"] == 0.0


def test_compare_moves_on_states_baseline_vs_zero_profile() -> None:
    """Baseline vs an all-zero profile should evaluate on real states."""
    env = RingRiftEnv(BoardType.SQUARE8)
    state = env.reset()
    states = [state]

    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    zero_profile = dict.fromkeys(baseline.keys(), 0.0)

    stats_diff = compare_moves_on_states(baseline, zero_profile, states)

    # We do not require a specific difference rate, only that the helper
    # runs end-to-end on a real GameState and returns a sane summary.
    assert stats_diff["total_states"] >= 1
    assert 0.0 <= stats_diff["difference_rate"] <= 1.0
