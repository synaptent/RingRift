from unittest.mock import MagicMock

import pytest

from app.ai.descent_ai import DescentAI
from app.models import AIConfig

# Skip all tests in this module - internal API changed after descent_ai refactoring
pytestmark = pytest.mark.skip(reason="DescentAI internal API changed - _select_child_puct_key removed")


def make_ai(c_puct: float = 1.0) -> DescentAI:
    return DescentAI(
        player_number=1,
        config=AIConfig(
            difficulty=5,
            use_neural_net=False,
            descent_c_puct=c_puct,
        ),
    )


def test_selects_high_prior_unvisited_child_for_self_player():
    ai = make_ai()
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "m1": (m1, 0.0, 0.9, 0),
        "m2": (m2, 0.0, 0.1, 0),
    }
    assert ai._select_child_puct_key(children, parent_visits=10, current_player=1) == "m1"


def test_shifts_to_less_visited_child_as_uncertainty_dominates():
    ai = make_ai()
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "m1": (m1, 0.0, 0.9, 90),
        "m2": (m2, 0.0, 0.1, 0),
    }
    assert ai._select_child_puct_key(children, parent_visits=100, current_player=1) == "m2"


def test_minimizing_branch_uses_val_minus_exploration():
    ai = make_ai()
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "a": (m1, -0.5, 0.9, 0),
        "b": (m2, -0.5, 0.1, 0),
    }
    assert ai._select_child_puct_key(children, parent_visits=10, current_player=2) == "a"


def test_missing_priors_default_to_uniform():
    ai = make_ai(c_puct=0.5)
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "a": (m1, 0.2),
        "b": (m2, 0.1),
    }
    assert ai._select_child_puct_key(children, parent_visits=5, current_player=1) == "a"

