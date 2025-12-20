"""Tests for app/ai/neural_losses.py - Neural network loss functions.

Tests cover:
- multi_player_value_loss MSE computation
- rank_distribution_loss cross-entropy
- ranks_from_game_result ranking computation
"""

import pytest
import torch

from app.ai.neural_losses import (
    MAX_PLAYERS,
    build_rank_targets,
    masked_policy_kl,
    multi_player_value_loss,
    rank_distribution_loss,
    ranks_from_game_result,
)


class TestMultiPlayerValueLoss:
    """Tests for multi_player_value_loss function."""

    def test_perfect_prediction(self):
        """Test zero loss for perfect predictions."""
        batch_size = 4
        pred = torch.tensor([[0.8, 0.1, 0.05, 0.05]] * batch_size)
        target = pred.clone()

        loss = multi_player_value_loss(pred, target, num_players=2)

        assert loss.item() == pytest.approx(0.0)

    def test_masks_inactive_players(self):
        """Test that inactive players are masked out."""
        pred = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        target = torch.tensor([[0.5, 0.5, 1.0, 1.0]])  # Only differ in inactive slots

        loss = multi_player_value_loss(pred, target, num_players=2)

        # Loss should be 0 because only slots 0-1 are active and match
        assert loss.item() == pytest.approx(0.0)

    def test_per_sample_num_players(self):
        """Test with per-sample num_players tensor."""
        pred = torch.tensor([
            [0.8, 0.2, 0.0, 0.0],
            [0.6, 0.3, 0.1, 0.0],
        ])
        target = torch.tensor([
            [0.8, 0.2, 0.5, 0.5],  # Sample 0: 2 players, slots 2-3 masked
            [0.6, 0.3, 0.1, 0.5],  # Sample 1: 3 players, slot 3 masked
        ])
        num_players = torch.tensor([2, 3])

        loss = multi_player_value_loss(pred, target, num_players=num_players)

        # Loss should be 0 because all active slots match
        assert loss.item() == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        pred = torch.randn(4, 4)
        target = torch.randn(4, 3)

        with pytest.raises(ValueError, match="share the same shape"):
            multi_player_value_loss(pred, target, num_players=2)

    def test_wrong_ndim_raises(self):
        """Test that wrong ndim raises ValueError."""
        pred = torch.randn(4)  # 1D instead of 2D
        target = torch.randn(4)

        with pytest.raises(ValueError, match="2D tensors"):
            multi_player_value_loss(pred, target, num_players=2)

    def test_invalid_num_players_raises(self):
        """Test that invalid num_players raises ValueError."""
        pred = torch.randn(2, 4)
        target = torch.randn(2, 4)

        with pytest.raises(ValueError, match="num_players must be in"):
            multi_player_value_loss(pred, target, num_players=0)

        with pytest.raises(ValueError, match="num_players must be in"):
            multi_player_value_loss(pred, target, num_players=5)

    def test_mse_computation(self):
        """Test correct MSE computation."""
        pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        loss = multi_player_value_loss(pred, target, num_players=2)

        # MSE = ((1-0)^2 + (0-0)^2) / 2 = 0.5
        assert loss.item() == pytest.approx(0.5)

    def test_all_four_players(self):
        """Test with all four players active."""
        pred = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        target = torch.tensor([[0.5, 0.0, 0.5, 0.0]])

        loss = multi_player_value_loss(pred, target, num_players=4)

        # MSE = mean of (0.25-0.5)^2 + (0.25-0)^2 + (0.25-0.5)^2 + (0.25-0)^2
        # = mean of 0.0625 + 0.0625 + 0.0625 + 0.0625 = 0.0625
        assert loss.item() == pytest.approx(0.0625)


class TestRankDistributionLoss:
    """Tests for rank_distribution_loss function."""

    def test_perfect_prediction(self):
        """Test low loss for confident correct predictions."""
        pred_dist = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0],   # Player 0 predicted rank 0 with 100%
             [0.0, 1.0, 0.0, 0.0],   # Player 1 predicted rank 1 with 100%
             [0.0, 0.0, 1.0, 0.0],   # Player 2 predicted rank 2 with 100%
             [0.25, 0.25, 0.25, 0.25]],  # Player 3 inactive
        ])
        target_ranks = torch.tensor([[0, 1, 2, 3]])

        loss = rank_distribution_loss(pred_dist, target_ranks, num_players=3)

        # Loss should be near 0 (clamped to avoid log(1))
        assert loss.item() < 1e-5

    def test_wrong_prediction_high_loss(self):
        """Test high loss for wrong predictions."""
        pred_dist = torch.tensor([
            [[0.1, 0.1, 0.1, 0.7],   # Player 0 predicted wrong rank
             [0.1, 0.1, 0.7, 0.1],   # Player 1 predicted wrong rank
             [0.25, 0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25, 0.25]],
        ])
        target_ranks = torch.tensor([[0, 1, 2, 3]])

        loss = rank_distribution_loss(pred_dist, target_ranks, num_players=2)

        # Loss should be high due to wrong predictions
        assert loss.item() > 1.0

    def test_masks_inactive_players(self):
        """Test that inactive players are masked."""
        pred_dist = torch.zeros(1, 4, 4)
        pred_dist[0, 0, 0] = 1.0  # Player 0 correct
        pred_dist[0, 1, 1] = 1.0  # Player 1 correct
        pred_dist[0, 2, 0] = 1.0  # Player 2 wrong, but inactive
        pred_dist[0, 3, 0] = 1.0  # Player 3 wrong, but inactive

        target_ranks = torch.tensor([[0, 1, 2, 3]])

        loss = rank_distribution_loss(pred_dist, target_ranks, num_players=2)

        # Loss should only consider players 0 and 1
        assert loss.item() < 1e-5


class TestRanksFromGameResult:
    """Tests for ranks_from_game_result function."""

    def test_two_player_winner(self):
        """Test 2-player game winner gets rank 0."""
        ranks = ranks_from_game_result(winner=0, num_players=2)

        assert ranks[0].item() == 0
        assert ranks[1].item() == 1
        assert ranks[2].item() == MAX_PLAYERS  # Inactive get MAX_PLAYERS
        assert ranks[3].item() == MAX_PLAYERS  # Inactive get MAX_PLAYERS

    def test_three_player_ranking(self):
        """Test 3-player game ranking."""
        ranks = ranks_from_game_result(
            winner=1,
            num_players=3,
            player_territories=[5, 10, 7],  # Winner has most but won anyway
        )

        assert ranks[1].item() == 0  # Winner
        # Remaining sorted by territory: p2 (7) > p0 (5)
        assert ranks[2].item() == 1
        assert ranks[0].item() == 2

    def test_four_player_ranking_with_tiebreakers(self):
        """Test 4-player ranking with multiple tiebreakers."""
        ranks = ranks_from_game_result(
            winner=0,
            num_players=4,
            player_territories=[0, 5, 5, 3],  # p1 and p2 tied on territory
            player_eliminated_rings=[0, 2, 3, 1],  # p2 has more elim rings
        )

        assert ranks[0].item() == 0  # Winner
        # p2 beats p1 on eliminated_rings tiebreaker
        assert ranks[2].item() == 1
        assert ranks[1].item() == 2
        assert ranks[3].item() == 3

    def test_default_max_players_value(self):
        """Test MAX_PLAYERS constant is 4."""
        assert MAX_PLAYERS == 4

    def test_inactive_players_get_max_players_rank(self):
        """Test inactive players get rank equal to MAX_PLAYERS (4)."""
        ranks = ranks_from_game_result(winner=0, num_players=2)

        assert ranks[2].item() == MAX_PLAYERS
        assert ranks[3].item() == MAX_PLAYERS

    def test_elimination_order_tiebreaker(self):
        """Test elimination order as final tiebreaker."""
        ranks = ranks_from_game_result(
            winner=0,
            num_players=3,
            player_territories=[0, 0, 0],  # All tied
            player_eliminated_rings=[0, 0, 0],  # All tied
            player_markers_on_board=[0, 0, 0],  # All tied
            elimination_order=[2, 1],  # p2 eliminated first, p1 second
        )

        assert ranks[0].item() == 0  # Winner
        # p1 eliminated later, so ranks higher
        assert ranks[1].item() == 1
        assert ranks[2].item() == 2


class TestMaskedPolicyKL:
    """Tests for masked_policy_kl function."""

    def test_basic_kl_loss(self):
        """Test basic KL divergence computation."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        policy_targets = torch.tensor([[0.6, 0.3, 0.1]])

        loss = masked_policy_kl(policy_log_probs, policy_targets)

        assert loss.item() > 0  # KL divergence is non-negative
        assert torch.isfinite(loss)

    def test_identical_distributions_low_loss(self):
        """Test that identical distributions have near-zero loss."""
        probs = torch.tensor([[0.5, 0.3, 0.2]])
        policy_log_probs = torch.log(probs)
        policy_targets = probs.clone()

        loss = masked_policy_kl(policy_log_probs, policy_targets)

        assert loss.item() < 0.01  # Should be very small

    def test_empty_targets_return_zero(self):
        """Test that all-zero targets return zero loss."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
        policy_targets = torch.zeros(1, 3)

        loss = masked_policy_kl(policy_log_probs, policy_targets)

        assert loss.item() == 0.0

    def test_mixed_valid_invalid_targets(self):
        """Test batch with mix of valid and invalid targets."""
        policy_log_probs = torch.log(torch.tensor([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
        ]))
        policy_targets = torch.tensor([
            [0.6, 0.3, 0.1],  # Valid
            [0.0, 0.0, 0.0],  # Invalid - should be masked
        ])

        loss = masked_policy_kl(policy_log_probs, policy_targets)

        # Should only compute loss for the first sample
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_batch_computation(self):
        """Test loss computation over a batch."""
        batch_size = 4
        policy_size = 10
        policy_log_probs = torch.log_softmax(
            torch.randn(batch_size, policy_size), dim=1
        )
        policy_targets = torch.softmax(
            torch.randn(batch_size, policy_size), dim=1
        )

        loss = masked_policy_kl(policy_log_probs, policy_targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0


class TestBuildRankTargets:
    """Tests for build_rank_targets function."""

    def test_basic_two_players(self):
        """Test rank targets for 2-player game."""
        # Player 0 has higher value, should be rank 0
        values_mp = torch.tensor([[0.8, 0.2, 0.0, 0.0]])

        rank_targets, active_mask = build_rank_targets(values_mp, num_players=2)

        # Check shapes
        assert rank_targets.shape == (1, 4, 4)
        assert active_mask.shape == (1, 4)

        # Check active mask
        assert active_mask[0, 0].item() is True
        assert active_mask[0, 1].item() is True
        assert active_mask[0, 2].item() is False
        assert active_mask[0, 3].item() is False

        # Player 0 (value 0.8) should have rank 0 (first place)
        assert rank_targets[0, 0, 0].item() == 1.0

        # Player 1 (value 0.2) should have rank 1 (second place)
        assert rank_targets[0, 1, 1].item() == 1.0

    def test_three_players_with_tie(self):
        """Test rank targets with tied values."""
        # Players 0 and 1 are tied, player 2 is last
        values_mp = torch.tensor([[0.5, 0.5, 0.2, 0.0]])

        rank_targets, _active_mask = build_rank_targets(values_mp, num_players=3)

        # Players 0 and 1 should share ranks 0 and 1
        assert rank_targets[0, 0, 0].item() == pytest.approx(0.5)
        assert rank_targets[0, 0, 1].item() == pytest.approx(0.5)
        assert rank_targets[0, 1, 0].item() == pytest.approx(0.5)
        assert rank_targets[0, 1, 1].item() == pytest.approx(0.5)

        # Player 2 should have rank 2 (third place)
        assert rank_targets[0, 2, 2].item() == 1.0

    def test_per_sample_num_players(self):
        """Test with per-sample num_players tensor."""
        values_mp = torch.tensor([
            [0.8, 0.2, 0.0, 0.0],
            [0.6, 0.4, 0.3, 0.0],
        ])
        num_players = torch.tensor([2, 3])

        _rank_targets, active_mask = build_rank_targets(values_mp, num_players)

        # First sample: 2 players
        assert active_mask[0, :2].all()
        assert not active_mask[0, 2:].any()

        # Second sample: 3 players
        assert active_mask[1, :3].all()
        assert not active_mask[1, 3].item()

    def test_batch_processing(self):
        """Test processing multiple samples in a batch."""
        batch_size = 8
        values_mp = torch.randn(batch_size, 4)

        rank_targets, active_mask = build_rank_targets(values_mp, num_players=4)

        assert rank_targets.shape == (batch_size, 4, 4)
        assert active_mask.shape == (batch_size, 4)

        # All players should be active
        assert active_mask.all()

        # Each player's rank distribution should sum to 1
        for b in range(batch_size):
            for p in range(4):
                rank_sum = rank_targets[b, p].sum().item()
                assert abs(rank_sum - 1.0) < 1e-6


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with neural_net.py imports."""

    def test_import_from_neural_net(self):
        """Test that functions can still be imported from neural_net.py."""
        # Verify they're the same functions
        from app.ai.neural_losses import (
            multi_player_value_loss,
            rank_distribution_loss,
            ranks_from_game_result,
        )
        from app.ai.neural_net import (
            multi_player_value_loss as mlvl,
            rank_distribution_loss as rdl,
            ranks_from_game_result as rfgr,
        )

        assert mlvl is multi_player_value_loss
        assert rdl is rank_distribution_loss
        assert rfgr is ranks_from_game_result
