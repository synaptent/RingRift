"""Tests for DescentAI child selection with uncertainty-based exploration.

These tests verify the _select_child_key method which implements UCB-style
exploration bonuses for the Descent algorithm.
"""

from unittest.mock import MagicMock

from app.ai.descent_ai import DescentAI
from app.models import AIConfig


def make_ai(uncertainty_ucb_c: float = 1.0, use_uncertainty: bool = True) -> DescentAI:
    """Create a DescentAI instance configured for uncertainty-based selection."""
    ai = DescentAI(
        player_number=1,
        config=AIConfig(
            difficulty=5,
            use_neural_net=False,
        ),
    )
    # Configure uncertainty selection parameters
    ai.use_uncertainty_selection = use_uncertainty
    ai.uncertainty_ucb_c = uncertainty_ucb_c
    return ai


def test_selects_high_prior_unvisited_child_for_maximizing():
    """When maximizing, unvisited child with higher prior should be preferred."""
    ai = make_ai()
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "m1": (m1, 0.0, 0.9, 0),  # (state, value, prior, visits)
        "m2": (m2, 0.0, 0.1, 0),
    }
    assert ai._select_child_key(children, parent_visits=10, maximizing=True) == "m1"


def test_shifts_to_less_visited_child_as_uncertainty_dominates():
    """With many visits on one child, exploration bonus should favor unvisited child."""
    ai = make_ai()
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "m1": (m1, 0.0, 0.9, 90),  # High prior but heavily visited
        "m2": (m2, 0.0, 0.1, 0),   # Low prior but unvisited
    }
    assert ai._select_child_key(children, parent_visits=100, maximizing=True) == "m2"


def test_minimizing_branch_uses_val_minus_exploration():
    """When minimizing (opponent's turn), higher prior should still be preferred for ties."""
    ai = make_ai()
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "a": (m1, -0.5, 0.9, 0),  # Same value, higher prior
        "b": (m2, -0.5, 0.1, 0),  # Same value, lower prior
    }
    # When minimizing with equal values, should prefer higher prior as tie-breaker
    assert ai._select_child_key(children, parent_visits=10, maximizing=False) == "a"


def test_missing_priors_default_to_uniform():
    """Children without explicit prior/visits should use defaults."""
    ai = make_ai(uncertainty_ucb_c=0.5, use_uncertainty=False)
    m1, m2 = MagicMock(), MagicMock()
    children = {
        "a": (m1, 0.2),  # Only state and value, no prior/visits
        "b": (m2, 0.1),
    }
    # With uncertainty disabled, should pick highest value
    assert ai._select_child_key(children, parent_visits=5, maximizing=True) == "a"

