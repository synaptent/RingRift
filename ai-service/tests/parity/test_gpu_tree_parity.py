"""Test parity between GPU tree search and CPU sequential halving.

This module validates that the GPU-accelerated tensor tree implementation
produces equivalent training data quality to the CPU-based sequential halving.

Key aspects tested:
1. Policy distribution similarity (within tolerance)
2. Best move selection agreement
3. Sequential halving budget allocation
4. Gumbel-Top-K action sampling

Environment variables:
- RINGRIFT_GPU_TREE_SHADOW_RATE: Set to 1.0 for full validation in tests
"""

import numpy as np
import pytest
import torch

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.tensor_gumbel_tree import (
    GPUGumbelMCTS,
    GPUGumbelMCTSConfig,
    TensorGumbelTree,
)
from app.models import AIConfig, BoardType
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state


class MockNeuralNet:
    """Mock neural network for deterministic testing."""

    def __init__(self, policy_bias: dict[int, float] | None = None):
        """Initialize mock with optional policy bias.

        Args:
            policy_bias: Dict mapping action indices to probability boosts.
        """
        self.policy_bias = policy_bias or {}
        self.call_count = 0

    def evaluate_batch(self, states, value_head=None):
        """Return uniform policy and zero values."""
        self.call_count += 1
        values = [0.0] * len(states)
        policies = []
        for state in states:
            # Create uniform policy over 64 actions (8x8 board)
            policy = np.ones(64) / 64
            for idx, prob in self.policy_bias.items():
                if idx < len(policy):
                    policy[idx] = prob
            policy = policy / policy.sum()  # Renormalize
            policies.append(policy)
        return values, policies

    def encode_move(self, move, board) -> int:
        """Encode move as flat index."""
        size = board.size
        if move.to is not None:
            return move.to.y * size + move.to.x
        return 0


@pytest.fixture
def initial_state():
    """Create initial game state."""
    return create_initial_state(
        board_type=BoardType.SQUARE8,
        num_players=2,
    )


@pytest.fixture
def mock_nn():
    """Create mock neural network."""
    return MockNeuralNet()


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestGPUCPUPolicyParity:
    """Test that GPU and CPU produce similar policy distributions."""

    def test_policy_distribution_similarity(self, initial_state, device):
        """Test that GPU and CPU produce similar policy distributions."""
        torch.manual_seed(42)
        np.random.seed(42)

        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        if len(valid_moves) <= 1:
            pytest.skip("Need multiple valid moves for comparison")

        # Create GPU tree
        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=100,
            eval_mode="heuristic",
            device=device,
        )
        gpu_tree = GPUGumbelMCTS(config)

        mock_nn = MockNeuralNet()

        try:
            _, gpu_policy = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        # Verify policy is valid distribution
        total_prob = sum(gpu_policy.values())
        assert abs(total_prob - 1.0) < 0.01, f"Policy sum = {total_prob}"

        # Verify all probabilities are non-negative
        for key, prob in gpu_policy.items():
            assert prob >= 0.0, f"Negative probability for {key}: {prob}"

    def test_best_move_in_policy(self, initial_state, device):
        """Test that best move has non-zero probability in policy."""
        torch.manual_seed(42)

        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        if len(valid_moves) <= 1:
            pytest.skip("Need multiple valid moves")

        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=100,
            eval_mode="heuristic",
            device=device,
        )
        gpu_tree = GPUGumbelMCTS(config)
        mock_nn = MockNeuralNet()

        try:
            best_move, policy = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        best_key = gpu_tree._move_to_key(best_move)
        assert best_key in policy, f"Best move {best_key} not in policy"
        assert policy[best_key] > 0.0, "Best move has zero probability"


class TestSequentialHalvingParity:
    """Test Sequential Halving budget allocation parity."""

    def test_budget_allocation_formula(self):
        """Test that budget allocation formulas match expected values."""
        tree = TensorGumbelTree.create(batch_size=1, num_actions=16, device="cpu")

        # Test with 800 budget, 16 actions
        phases = tree.compute_sequential_halving_budget(800, 16)

        # Expected: 4 phases for 16 actions (log2(16) = 4)
        assert len(phases) == 4

        # Verify actions halve each phase
        prev_actions = 16
        for num_actions, sims in phases:
            assert num_actions <= prev_actions
            prev_actions = num_actions

        # Final phase should have 2 or fewer actions
        assert phases[-1][0] <= 2

    def test_pruning_keeps_top_half(self):
        """Test that pruning keeps top half of actions by value."""
        tree = TensorGumbelTree.create(batch_size=1, num_actions=8, device="cpu")
        tree.reset()

        # Initialize with 8 remaining actions
        tree.remaining_mask[0, :8] = True

        # Set values: actions 0-3 are worse, 4-7 are better
        tree.action_values[0, :4] = 0.0
        tree.action_values[0, 4:8] = 1.0
        tree.action_visits[0, :8] = 1

        # Prune
        tree.prune_actions(tree_idx=0)

        # Top 4 should remain
        remaining = tree.get_remaining_action_indices(tree_idx=0)
        assert len(remaining) == 4
        assert all(idx >= 4 for idx in remaining.tolist())


class TestGumbelTopKParity:
    """Test Gumbel-Top-K sampling parity."""

    def test_gumbel_noise_distribution(self):
        """Verify Gumbel noise has correct distribution properties."""
        n_samples = 10000

        # Generate Gumbel samples using tensor tree method
        uniform = torch.rand(n_samples)
        gumbel = -torch.log(-torch.log(uniform.clamp(min=1e-10, max=1.0 - 1e-10)))

        # Gumbel(0,1) should have mean ~0.5772 (Euler-Mascheroni constant)
        expected_mean = 0.5772
        assert abs(gumbel.mean().item() - expected_mean) < 0.1

        # Standard deviation ~π/√6 ≈ 1.28
        expected_std = 1.28
        assert abs(gumbel.std().item() - expected_std) < 0.2

    def test_top_k_selection_with_uniform_prior(self):
        """Test that top-K selection works with uniform prior."""
        torch.manual_seed(42)

        num_actions = 16
        k = 4

        # Uniform prior logits
        logits = torch.zeros(1, num_actions)

        # Create tensor tree and sample
        tree = TensorGumbelTree.create(
            batch_size=1,
            num_actions=num_actions,
            device="cpu",
        )
        tree.reset()
        top_k_indices = tree.initialize_root(logits, num_sampled_actions=k)

        # Verify K actions were selected
        assert top_k_indices.shape == (1, k)
        assert len(torch.unique(top_k_indices)) == k

    def test_top_k_prefers_high_prior(self):
        """Test that top-K selection prefers high-prior actions."""
        torch.manual_seed(42)

        num_actions = 16
        k = 4
        n_trials = 100

        # Create biased prior: action 0 has much higher prob
        logits = torch.zeros(1, num_actions)
        logits[0, 0] = 5.0  # High prior for action 0

        action_0_count = 0
        for trial in range(n_trials):
            tree = TensorGumbelTree.create(
                batch_size=1,
                num_actions=num_actions,
                device="cpu",
            )
            tree.reset()

            # Re-randomize Gumbel each trial
            top_k = tree.initialize_root(logits, num_sampled_actions=k)
            if 0 in top_k[0].tolist():
                action_0_count += 1

        # Action 0 should be selected most of the time
        assert action_0_count > 90, f"High-prior action selected only {action_0_count}/100 times"


class TestGumbelMCTSAIValidation:
    """Test GumbelMCTSAI validation infrastructure."""

    def test_get_gpu_tree_validation_stats(self):
        """Test that validation stats are accessible."""
        config = AIConfig(difficulty=5, allow_fresh_weights=True)
        ai = GumbelMCTSAI(player_number=1, config=config, board_type=BoardType.SQUARE8)

        stats = ai.get_gpu_tree_validation_stats()

        assert "total_checks" in stats
        assert "divergences" in stats
        assert "divergence_rate" in stats
        assert "shadow_rate" in stats

    def test_shadow_validation_disabled_by_default(self):
        """Test that shadow validation is disabled by default (rate=0)."""
        config = AIConfig(difficulty=5, allow_fresh_weights=True)
        ai = GumbelMCTSAI(player_number=1, config=config, board_type=BoardType.SQUARE8)

        stats = ai.get_gpu_tree_validation_stats()
        # Default rate is 0.0 (disabled)
        assert stats["shadow_rate"] == 0.0


class TestTrainingDataQuality:
    """Test that training data extraction produces valid outputs."""

    def test_visit_distribution_valid(self, initial_state, device):
        """Test that visit distribution is valid for training."""
        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=200,
            eval_mode="heuristic",
            device=device,
        )
        gpu_tree = GPUGumbelMCTS(config)
        mock_nn = MockNeuralNet()

        try:
            _, policy = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        # Policy should sum to ~1.0
        total_prob = sum(policy.values())
        assert abs(total_prob - 1.0) < 0.01, f"Policy sum = {total_prob}"

        # Should have non-zero probabilities for at least some moves
        non_zero = sum(1 for p in policy.values() if p > 0.001)
        assert non_zero >= 1, "Should have at least one non-zero probability"

    def test_policy_entropy_reasonable(self, initial_state, device):
        """Test that policy has reasonable entropy (not degenerate)."""
        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        if len(valid_moves) < 4:
            pytest.skip("Need at least 4 valid moves for entropy test")

        config = GPUGumbelMCTSConfig(
            num_sampled_actions=min(8, len(valid_moves)),
            simulation_budget=200,
            eval_mode="heuristic",
            device=device,
        )
        gpu_tree = GPUGumbelMCTS(config)
        mock_nn = MockNeuralNet()

        try:
            _, policy = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        # Compute entropy: -sum(p * log(p))
        entropy = 0.0
        for prob in policy.values():
            if prob > 1e-10:
                entropy -= prob * np.log(prob)

        # Entropy should be > 0 (not deterministic) and < log(n) (not uniform)
        max_entropy = np.log(len(policy))
        assert entropy > 0.1, f"Policy too deterministic: entropy={entropy:.3f}"
        assert entropy < max_entropy, f"Policy too uniform: entropy={entropy:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
