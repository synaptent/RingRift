"""Tests for Gumbel MCTS common data structures.

Tests cover:
- GumbelAction creation and properties
- GumbelNode tree operations
- LeafEvalRequest construction
- Budget tier constants and helpers
"""

import pytest

from app.ai.gumbel_common import (
    GumbelAction,
    GumbelNode,
    LeafEvalRequest,
    GUMBEL_BUDGET_THROUGHPUT,
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_ULTIMATE,
    GUMBEL_DEFAULT_BUDGET,
    GUMBEL_DEFAULT_K,
    get_budget_for_difficulty,
)


class TestGumbelAction:
    """Test GumbelAction data class."""

    def test_create_basic_action(self):
        """Test basic action creation."""
        action = GumbelAction(
            move=None,  # Mock move
            policy_logit=1.5,
            gumbel_noise=0.5,
            perturbed_value=2.0,
        )
        assert action.policy_logit == 1.5
        assert action.gumbel_noise == 0.5
        assert action.perturbed_value == 2.0
        assert action.visit_count == 0
        assert action.total_value == 0.0

    def test_mean_value_with_no_visits(self):
        """Test mean_value returns 0 when no visits."""
        action = GumbelAction(
            move=None,
            policy_logit=1.0,
            gumbel_noise=0.0,
            perturbed_value=1.0,
        )
        assert action.mean_value == 0.0

    def test_mean_value_with_visits(self):
        """Test mean_value calculation."""
        action = GumbelAction(
            move=None,
            policy_logit=1.0,
            gumbel_noise=0.0,
            perturbed_value=1.0,
            visit_count=10,
            total_value=5.0,
        )
        assert action.mean_value == 0.5

    def test_completed_q_no_visits(self):
        """Test completed_q falls back to prior when no visits."""
        action = GumbelAction(
            move=None,
            policy_logit=2.0,
            gumbel_noise=0.0,
            perturbed_value=2.0,
        )
        # Should return policy_logit / 10
        assert action.completed_q(max_visits=100) == 0.2

    def test_completed_q_with_visits(self):
        """Test completed_q mixes prior and empirical value."""
        action = GumbelAction(
            move=None,
            policy_logit=1.0,
            gumbel_noise=0.0,
            perturbed_value=1.0,
            visit_count=50,
            total_value=25.0,  # mean = 0.5
        )
        q = action.completed_q(max_visits=50, c_visit=50.0)
        # With c_visit=50 and max_visits=50:
        # mix = 50 / (50 + 50) = 0.5
        # q = (1 - 0.5) * 0.5 + 0.5 * (1.0 / 10) = 0.25 + 0.05 = 0.3
        assert abs(q - 0.3) < 0.01

    def test_from_gumbel_score_classmethod(self):
        """Test creating action from gumbel score."""
        action = GumbelAction.from_gumbel_score(move=None, gumbel_score=1.5)
        assert action.perturbed_value == 1.5
        assert action.visit_count == 0


class TestGumbelNode:
    """Test GumbelNode tree structure."""

    def test_create_root_node(self):
        """Test creating root node."""
        node = GumbelNode(move=None, parent=None)
        assert node.move is None
        assert node.parent is None
        assert len(node.children) == 0
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.prior == 0.0

    def test_create_child_node(self):
        """Test creating child node with parent reference."""
        root = GumbelNode(move=None, parent=None)
        child = GumbelNode(move=None, parent=root, prior=0.5)
        assert child.parent is root
        assert child.prior == 0.5

    def test_add_child_to_tree(self):
        """Test adding children to tree structure."""
        root = GumbelNode(move=None, parent=None)
        root.children["move1"] = GumbelNode(move=None, parent=root)
        root.children["move2"] = GumbelNode(move=None, parent=root)
        assert len(root.children) == 2
        assert "move1" in root.children
        assert "move2" in root.children


class TestLeafEvalRequest:
    """Test LeafEvalRequest batch processing structure."""

    def test_create_request(self):
        """Test creating leaf evaluation request."""
        request = LeafEvalRequest(
            game_state=None,  # Mock state
            is_opponent_perspective=True,
            action_idx=1,
            simulation_idx=0,
        )
        assert request.action_idx == 1
        assert request.simulation_idx == 0
        assert request.is_opponent_perspective is True

    def test_opponent_perspective_field(self):
        """Test opponent perspective tracking."""
        request_opponent = LeafEvalRequest(
            game_state=None,
            is_opponent_perspective=True,
            action_idx=0,
            simulation_idx=0,
        )
        assert request_opponent.is_opponent_perspective is True

        request_same = LeafEvalRequest(
            game_state=None,
            is_opponent_perspective=False,
            action_idx=0,
            simulation_idx=0,
        )
        assert request_same.is_opponent_perspective is False


class TestBudgetConstants:
    """Test Gumbel budget tier constants."""

    def test_budget_tiers_ordered(self):
        """Test budget tiers are properly ordered."""
        assert GUMBEL_BUDGET_THROUGHPUT < GUMBEL_BUDGET_STANDARD
        assert GUMBEL_BUDGET_STANDARD <= GUMBEL_BUDGET_QUALITY  # Equal after Dec 2025 change
        assert GUMBEL_BUDGET_QUALITY < GUMBEL_BUDGET_ULTIMATE

    def test_budget_tier_values(self):
        """Test specific budget values.

        Dec 2025: STANDARD raised from 150 to 800 to match AlphaZero.
        Lower budgets produce weak training data that plateaus at ~1400 Elo.
        """
        assert GUMBEL_BUDGET_THROUGHPUT == 64
        assert GUMBEL_BUDGET_STANDARD == 800  # Dec 2025: raised from 150
        assert GUMBEL_BUDGET_QUALITY == 800
        assert GUMBEL_BUDGET_ULTIMATE == 1600

    def test_default_budget_matches_standard(self):
        """Test default budget equals standard tier."""
        assert GUMBEL_DEFAULT_BUDGET == GUMBEL_BUDGET_STANDARD

    def test_default_k_value(self):
        """Test default k for Gumbel-Top-K."""
        assert GUMBEL_DEFAULT_K == 16


class TestGetBudgetForDifficulty:
    """Test difficulty-to-budget mapping helper."""

    def test_low_difficulty_gets_standard(self):
        """Test difficulties 1-6 get standard budget."""
        for diff in range(1, 7):
            assert get_budget_for_difficulty(diff) == GUMBEL_BUDGET_STANDARD

    def test_medium_difficulty_gets_quality(self):
        """Test difficulties 7-9 get quality budget."""
        for diff in range(7, 10):
            assert get_budget_for_difficulty(diff) == GUMBEL_BUDGET_QUALITY

    def test_high_difficulty_gets_ultimate_or_master(self):
        """Test difficulty 10 gets ultimate, 11+ gets master budget."""
        # Difficulty 10: ULTIMATE (1600)
        assert get_budget_for_difficulty(10) == GUMBEL_BUDGET_ULTIMATE
        # Difficulty 11+: MASTER (3200) - December 2025 addition for 2000+ Elo
        from app.ai.gumbel_common import GUMBEL_BUDGET_MASTER
        for diff in range(11, 13):
            assert get_budget_for_difficulty(diff) == GUMBEL_BUDGET_MASTER
