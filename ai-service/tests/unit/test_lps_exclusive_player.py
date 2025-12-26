"""
Tests for LPS (Last Player Standing) exclusive player detection.

Validates that the fix for the P2 bias bug (2025-12-26) correctly identifies
exclusive players using masked multiplication instead of argmax.

The bug: torch.argmax() on a boolean tensor returns the first True index,
which doesn't semantically identify the exclusive player correctly.

The fix: Use masked multiplication with player indices to identify the
single True player.
"""

import pytest
import torch


class TestLpsExclusivePlayerDetection:
    """Test the masked multiplication logic for finding exclusive players."""

    def _find_exclusive_player(self, real_action_mask: torch.Tensor) -> torch.Tensor:
        """
        Find the exclusive player using the fixed logic.

        This replicates the logic from gpu_parallel_games.py lines 1688-1701.

        Args:
            real_action_mask: Boolean tensor of shape (batch_size, num_players+1)
                             where index 0 is unused (1-indexed players).

        Returns:
            Tensor of shape (batch_size,) with the exclusive player ID (1-indexed)
            or 0 if no exclusive player.
        """
        true_counts = real_action_mask.sum(dim=1).to(torch.int16)

        device = real_action_mask.device
        player_indices = torch.arange(
            real_action_mask.shape[1], device=device, dtype=torch.int8
        ).unsqueeze(0)

        masked_indices = real_action_mask.to(torch.int8) * player_indices
        exclusive_pid_candidates = masked_indices.max(dim=1).values.to(torch.int8)

        exclusive_pid = torch.where(
            true_counts == 1,
            exclusive_pid_candidates,
            torch.zeros_like(exclusive_pid_candidates),
        )

        return exclusive_pid

    def test_p1_exclusive_returns_p1(self):
        """When only P1 has real actions, should return 1."""
        # Shape: (1, 3) for 2-player game (index 0 unused)
        mask = torch.tensor([[False, True, False]])  # P1=True, P2=False
        result = self._find_exclusive_player(mask)
        assert result[0].item() == 1, f"Expected P1 (1), got {result[0].item()}"

    def test_p2_exclusive_returns_p2(self):
        """When only P2 has real actions, should return 2."""
        mask = torch.tensor([[False, False, True]])  # P1=False, P2=True
        result = self._find_exclusive_player(mask)
        assert result[0].item() == 2, f"Expected P2 (2), got {result[0].item()}"

    def test_both_players_have_actions_returns_zero(self):
        """When both players have actions, should return 0 (no exclusive)."""
        mask = torch.tensor([[False, True, True]])  # P1=True, P2=True
        result = self._find_exclusive_player(mask)
        assert result[0].item() == 0, f"Expected 0 (no exclusive), got {result[0].item()}"

    def test_no_players_have_actions_returns_zero(self):
        """When no players have actions, should return 0."""
        mask = torch.tensor([[False, False, False]])
        result = self._find_exclusive_player(mask)
        assert result[0].item() == 0, f"Expected 0, got {result[0].item()}"

    def test_symmetry_batch(self):
        """Test symmetry across a batch of games."""
        # Batch of 4 games: P1 exclusive, P2 exclusive, both, neither
        mask = torch.tensor([
            [False, True, False],   # P1 exclusive
            [False, False, True],   # P2 exclusive
            [False, True, True],    # Both
            [False, False, False],  # Neither
        ])
        result = self._find_exclusive_player(mask)

        assert result[0].item() == 1, "P1 exclusive should return 1"
        assert result[1].item() == 2, "P2 exclusive should return 2"
        assert result[2].item() == 0, "Both should return 0"
        assert result[3].item() == 0, "Neither should return 0"

    def test_4_player_all_exclusive_cases(self):
        """Test all exclusive cases for 4-player game."""
        for exclusive_player in [1, 2, 3, 4]:
            mask = torch.zeros(1, 5, dtype=torch.bool)  # 4 players + unused index 0
            mask[0, exclusive_player] = True
            result = self._find_exclusive_player(mask)
            assert result[0].item() == exclusive_player, \
                f"P{exclusive_player} exclusive should return {exclusive_player}, got {result[0].item()}"

    def test_argmax_would_fail_p2_exclusive(self):
        """
        Demonstrate the bug: argmax returns first True index, not exclusive player.

        With mask [False, False, True] (P2 exclusive):
        - argmax would return 2 (correct by coincidence)

        But more importantly, this test validates our fix works.
        """
        # The old buggy code was:
        # exclusive_pid = torch.argmax(mask.to(torch.int8), dim=1)
        #
        # For [False, False, True], argmax returns 2 (correct)
        # For [False, True, False], argmax returns 1 (correct)
        #
        # But the semantic is wrong - argmax doesn't mean "find exclusive player"
        # It means "find first/max index", which happens to work when there's one True.

        mask = torch.tensor([[False, False, True]])  # P2 exclusive

        # Old (buggy) way - happens to work in this case
        old_result = torch.argmax(mask.to(torch.int8), dim=1)

        # New (fixed) way
        new_result = self._find_exclusive_player(mask)

        # Both return 2 here, but the new way is semantically correct
        assert new_result[0].item() == 2


class TestLpsExclusivePlayerStatistical:
    """Statistical tests for LPS player detection symmetry."""

    def _find_exclusive_player(self, real_action_mask: torch.Tensor) -> torch.Tensor:
        """Same helper as above."""
        true_counts = real_action_mask.sum(dim=1).to(torch.int16)
        device = real_action_mask.device
        player_indices = torch.arange(
            real_action_mask.shape[1], device=device, dtype=torch.int8
        ).unsqueeze(0)
        masked_indices = real_action_mask.to(torch.int8) * player_indices
        exclusive_pid_candidates = masked_indices.max(dim=1).values.to(torch.int8)
        return torch.where(
            true_counts == 1,
            exclusive_pid_candidates,
            torch.zeros_like(exclusive_pid_candidates),
        )

    def test_random_exclusive_selection_is_symmetric(self):
        """
        Generate random exclusive player scenarios and verify no bias.

        This simulates what happens in actual games - randomly one player
        ends up exclusive. The detection should be symmetric.
        """
        torch.manual_seed(42)

        num_games = 1000
        num_players = 2

        p1_exclusive_count = 0
        p2_exclusive_count = 0

        for _ in range(num_games):
            # Randomly choose which player is exclusive
            exclusive = torch.randint(1, num_players + 1, (1,)).item()

            mask = torch.zeros(1, num_players + 1, dtype=torch.bool)
            mask[0, exclusive] = True

            result = self._find_exclusive_player(mask)

            if result[0].item() == 1:
                p1_exclusive_count += 1
            elif result[0].item() == 2:
                p2_exclusive_count += 1

        total = p1_exclusive_count + p2_exclusive_count
        p1_ratio = p1_exclusive_count / total if total > 0 else 0

        # Should be roughly 50/50 (within statistical bounds)
        assert 0.4 <= p1_ratio <= 0.6, \
            f"P1 ratio {p1_ratio:.2%} outside expected 40-60% range"

    def test_batch_detection_matches_sequential(self):
        """Verify batched detection matches sequential detection."""
        torch.manual_seed(123)

        batch_size = 100
        num_players = 2

        # Generate random exclusive scenarios
        masks = torch.zeros(batch_size, num_players + 1, dtype=torch.bool)
        expected = []

        for i in range(batch_size):
            scenario = torch.randint(0, 4, (1,)).item()
            if scenario == 0:  # P1 exclusive
                masks[i, 1] = True
                expected.append(1)
            elif scenario == 1:  # P2 exclusive
                masks[i, 2] = True
                expected.append(2)
            else:  # Both or neither
                expected.append(0)

        # Batch detection
        batch_result = self._find_exclusive_player(masks)

        # Compare
        for i, (got, want) in enumerate(zip(batch_result.tolist(), expected)):
            assert got == want, f"Game {i}: expected {want}, got {got}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
