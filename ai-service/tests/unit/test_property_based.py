"""Property-based tests using Hypothesis.

These tests verify mathematical properties and invariants that should
hold for all valid inputs, catching edge cases that unit tests might miss.

Run with: pytest tests/unit/test_property_based.py -v --hypothesis-show-statistics
"""

import numpy as np
import pytest
import torch
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)

from app.ai.neural_losses import (
    MAX_PLAYERS,
    build_rank_targets,
    masked_policy_kl,
    multi_player_value_loss,
)

# =============================================================================
# Custom Strategies
# =============================================================================

@st.composite
def probability_distributions(draw, size, batch_size=1):
    """Generate valid probability distributions that sum to 1."""
    raw = draw(st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size,
    ))
    total = sum(raw)
    normalized = [x / total for x in raw]
    if batch_size == 1:
        return torch.tensor([normalized])
    return torch.tensor([normalized] * batch_size)


@st.composite
def value_predictions(draw, batch_size, num_players):
    """Generate valid multi-player value predictions."""
    values = []
    for _ in range(batch_size):
        sample = draw(st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=MAX_PLAYERS,
            max_size=MAX_PLAYERS,
        ))
        values.append(sample)
    return torch.tensor(values, dtype=torch.float32)


# =============================================================================
# Property Tests for Loss Functions
# =============================================================================

class TestMaskedPolicyKLProperties:
    """Property-based tests for masked_policy_kl."""

    @given(st.integers(min_value=2, max_value=20))
    @settings(max_examples=50)
    def test_zero_loss_for_identical_distributions(self, policy_size):
        """Identical distributions should have (near) zero KL divergence."""
        # Create a valid probability distribution
        raw = torch.rand(1, policy_size) + 0.01  # Avoid zeros
        probs = raw / raw.sum(dim=1, keepdim=True)
        log_probs = torch.log(probs)

        loss = masked_policy_kl(log_probs, probs)

        # Should be very close to zero
        assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"

    @given(st.integers(min_value=2, max_value=20))
    @settings(max_examples=50)
    def test_loss_non_negative(self, policy_size):
        """KL divergence is always non-negative."""
        raw_pred = torch.rand(1, policy_size) + 0.01
        raw_target = torch.rand(1, policy_size) + 0.01

        pred = raw_pred / raw_pred.sum(dim=1, keepdim=True)
        target = raw_target / raw_target.sum(dim=1, keepdim=True)

        loss = masked_policy_kl(torch.log(pred), target)

        assert loss.item() >= 0, f"KL divergence should be non-negative, got {loss.item()}"

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=30)
    def test_empty_targets_return_zero(self, batch_size):
        """All-zero targets should return zero loss."""
        policy_log_probs = torch.log_softmax(torch.randn(batch_size, 10), dim=1)
        policy_targets = torch.zeros(batch_size, 10)

        loss = masked_policy_kl(policy_log_probs, policy_targets)

        assert loss.item() == 0.0


class TestBuildRankTargetsProperties:
    """Property-based tests for build_rank_targets."""

    @given(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50)
    def test_rank_distributions_sum_to_one(self, batch_size, num_players):
        """Each player's rank distribution should sum to 1."""
        values_mp = torch.randn(batch_size, MAX_PLAYERS)

        rank_targets, _active_mask = build_rank_targets(values_mp, num_players=num_players)

        for b in range(batch_size):
            for p in range(num_players):
                rank_sum = rank_targets[b, p].sum().item()
                assert abs(rank_sum - 1.0) < 1e-5, (
                    f"Rank distribution for batch {b}, player {p} sums to {rank_sum}"
                )

    @given(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50)
    def test_active_mask_correct_shape(self, batch_size, num_players):
        """Active mask should have correct shape and values."""
        values_mp = torch.randn(batch_size, MAX_PLAYERS)

        _rank_targets, active_mask = build_rank_targets(values_mp, num_players=num_players)

        assert active_mask.shape == (batch_size, MAX_PLAYERS)
        # First num_players slots should be active
        assert active_mask[:, :num_players].all()
        # Remaining slots should be inactive
        if num_players < MAX_PLAYERS:
            assert not active_mask[:, num_players:].any()

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=30)
    def test_all_ranks_non_negative(self, batch_size):
        """All rank probabilities should be non-negative."""
        values_mp = torch.randn(batch_size, MAX_PLAYERS)

        rank_targets, _ = build_rank_targets(values_mp, num_players=4)

        assert (rank_targets >= 0).all(), "All rank probabilities should be non-negative"

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=30)
    def test_rank_probabilities_at_most_one(self, batch_size):
        """All rank probabilities should be at most 1."""
        values_mp = torch.randn(batch_size, MAX_PLAYERS)

        rank_targets, _ = build_rank_targets(values_mp, num_players=4)

        assert (rank_targets <= 1.0 + 1e-6).all(), "All rank probabilities should be at most 1"


class TestMultiPlayerValueLossProperties:
    """Property-based tests for multi_player_value_loss."""

    @given(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50)
    def test_zero_loss_for_identical_predictions(self, batch_size, num_players):
        """Identical predictions and targets should have zero loss."""
        values = torch.randn(batch_size, MAX_PLAYERS)

        loss = multi_player_value_loss(values, values.clone(), num_players=num_players)

        assert loss.item() == pytest.approx(0.0)

    @given(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50)
    def test_loss_non_negative(self, batch_size, num_players):
        """MSE loss is always non-negative."""
        pred = torch.randn(batch_size, MAX_PLAYERS)
        target = torch.randn(batch_size, MAX_PLAYERS)

        loss = multi_player_value_loss(pred, target, num_players=num_players)

        assert loss.item() >= 0

    @given(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=50)
    def test_inactive_slots_dont_affect_loss(self, batch_size, num_players):
        """Changing inactive slots should not affect loss."""
        pred = torch.randn(batch_size, MAX_PLAYERS)
        target = torch.randn(batch_size, MAX_PLAYERS)

        loss1 = multi_player_value_loss(pred, target, num_players=num_players)

        # Modify inactive slots
        pred_modified = pred.clone()
        target_modified = target.clone()
        if num_players < MAX_PLAYERS:
            pred_modified[:, num_players:] = torch.randn(batch_size, MAX_PLAYERS - num_players)
            target_modified[:, num_players:] = torch.randn(batch_size, MAX_PLAYERS - num_players)

        loss2 = multi_player_value_loss(pred_modified, target_modified, num_players=num_players)

        assert loss1.item() == pytest.approx(loss2.item())


# =============================================================================
# Property Tests for Training Helpers
# =============================================================================

class TestHeuristicWeightProperties:
    """Property-based tests for heuristic weight functions."""

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_flatten_reconstruct_roundtrip(self, seed):
        """Flatten and reconstruct should preserve all weights."""
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS
        from app.training.train import (
            _flatten_heuristic_weights,
            _reconstruct_heuristic_profile,
        )

        # Create a profile with random values
        np.random.seed(seed)
        profile = {key: float(np.random.uniform(-10, 10)) for key in HEURISTIC_WEIGHT_KEYS}

        # Roundtrip
        keys, values = _flatten_heuristic_weights(profile)
        reconstructed = _reconstruct_heuristic_profile(keys, values)

        # Verify all keys preserved
        assert set(reconstructed.keys()) == set(profile.keys())

        # Verify all values preserved
        for key in profile:
            assert abs(reconstructed[key] - profile[key]) < 1e-9, (
                f"Key {key}: expected {profile[key]}, got {reconstructed[key]}"
            )

    @given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                    min_size=10, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_flatten_produces_correct_length(self, weights):
        """Flattening should produce list of same length as input profile."""
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS
        from app.training.train import _flatten_heuristic_weights

        profile = {key: weights[i % len(weights)] for i, key in enumerate(HEURISTIC_WEIGHT_KEYS)}

        keys, values = _flatten_heuristic_weights(profile)

        assert len(keys) == len(HEURISTIC_WEIGHT_KEYS)
        assert len(values) == len(HEURISTIC_WEIGHT_KEYS)


# =============================================================================
# Property Tests for Tensor Operations
# =============================================================================

class TestTensorOperationProperties:
    """Property-based tests for tensor operations used in training."""

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=50)
    def test_softmax_sums_to_one(self, batch_size, dim):
        """Softmax outputs should sum to 1 along the specified dimension."""
        logits = torch.randn(batch_size, dim)
        probs = torch.softmax(logits, dim=1)

        sums = probs.sum(dim=1)
        for i, s in enumerate(sums):
            assert abs(s.item() - 1.0) < 1e-5, f"Batch {i}: softmax sum is {s.item()}"

    @given(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=50)
    def test_log_softmax_consistent_with_softmax(self, batch_size, dim):
        """log_softmax should be consistent with log(softmax)."""
        logits = torch.randn(batch_size, dim)

        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        expected_log_probs = torch.log(probs)

        assert torch.allclose(log_probs, expected_log_probs, atol=1e-6)
