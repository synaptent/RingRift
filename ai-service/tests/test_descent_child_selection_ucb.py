from __future__ import annotations

from app.ai.descent_ai import DescentAI
from app.models import AIConfig


def _make_ai() -> DescentAI:
    return DescentAI(
        player_number=1,
        config=AIConfig(
            difficulty=5,
            use_neural_net=False,
        ),
    )


def test_descent_ucb_prefers_less_visited_child_when_values_equal() -> None:
    ai = _make_ai()
    ai.use_uncertainty_selection = True
    ai.uncertainty_ucb_c = 1.0

    children = {
        "a": (object(), 0.25, 0.5, 50),
        "b": (object(), 0.25, 0.5, 0),
    }

    assert ai._select_child_key(children, parent_visits=100, maximizing=True) == "b"


def test_descent_greedy_selection_uses_value_then_prior() -> None:
    ai = _make_ai()
    ai.use_uncertainty_selection = False

    children = {
        "a": (object(), 0.10, 0.9, 0),
        "b": (object(), 0.10, 0.1, 0),
        "c": (object(), 0.20, 0.0, 0),
    }

    assert ai._select_child_key(children, parent_visits=10, maximizing=True) == "c"
