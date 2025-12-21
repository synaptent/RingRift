"""Integration tests for full GPU game simulation.

Tests complete game simulations from start to finish using the GPU parallel
game runner to verify:
1. Games complete without errors
2. Victory conditions are detected
3. Move counts are reasonable
4. State invariants are maintained throughout
"""

import pytest
import torch

try:
    from app.ai.gpu_game_types import GameStatus
    from app.ai.gpu_parallel_games import ParallelGameRunner

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="GPU modules not available"
)


class TestFullGameSimulation:
    """Integration tests for complete game simulations."""

    @pytest.fixture
    def device(self):
        """Use CPU for testing to ensure consistent behavior."""
        return torch.device("cpu")

    def test_single_game_completes(self, device):
        """A single game should complete within reasonable moves."""
        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
            swap_enabled=True,
        )

        weights = [runner._default_weights()]
        max_steps = 1000

        for _step in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if runner.state.game_status[0].item() == GameStatus.COMPLETED:
                break

        # Game should complete
        assert (
            runner.state.game_status[0].item() == GameStatus.COMPLETED
        ), f"Game did not complete after {max_steps} steps"

        # Should have a winner (1 or 2) or be a draw (0)
        winner = runner.state.winner[0].item()
        assert winner in [0, 1, 2], f"Invalid winner: {winner}"

    def test_batch_games_complete(self, device):
        """A batch of games should all complete."""
        batch_size = 8
        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
            swap_enabled=True,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]
        max_steps = 1000

        for _step in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            # Check if all games completed
            if (runner.state.game_status == GameStatus.COMPLETED).all():
                break

        # Most games should complete (some may hit max steps with random play)
        completed_count = (runner.state.game_status == GameStatus.COMPLETED).sum().item()
        min_required = batch_size * 3 // 4  # At least 75% should complete
        assert completed_count >= min_required, f"Only {completed_count}/{batch_size} games completed (need {min_required})"

    def test_games_have_reasonable_length(self, device):
        """Games should have a reasonable number of moves (not too short/long)."""
        batch_size = 4
        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
            swap_enabled=True,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]
        max_steps = 1000

        for _step in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if (runner.state.game_status == GameStatus.COMPLETED).all():
                break

        # Check move counts
        move_counts = runner.state.move_count.tolist()
        for i, count in enumerate(move_counts):
            # Games should have at least some moves (not instant)
            assert count >= 5, f"Game {i} finished too quickly: {count} moves"
            # Games should not run forever (max is 500 canonical)
            assert count <= 500, f"Game {i} exceeded move limit: {count} moves"

    def test_victory_types_are_valid(self, device):
        """Completed games should have valid victory types."""
        batch_size = 4
        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
            swap_enabled=True,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]
        max_steps = 1000

        for _step in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if (runner.state.game_status == GameStatus.COMPLETED).all():
                break

        # Check victory types
        valid_victory_types = {"elimination", "territory", "lps", "draw", "move_limit"}
        for g in range(batch_size):
            if runner.state.game_status[g].item() == GameStatus.COMPLETED:
                victory_type, _details = runner.state.derive_victory_type(g, max_moves=500)
                assert (
                    victory_type in valid_victory_types
                ), f"Game {g} has invalid victory type: {victory_type}"

    def test_state_invariants_after_completion(self, device):
        """State invariants should hold after game completion."""
        batch_size = 2
        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
            swap_enabled=True,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]
        max_steps = 1000

        for _step in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if (runner.state.game_status == GameStatus.COMPLETED).all():
                break

        state = runner.state

        # Stack heights should be non-negative
        assert (state.stack_height >= 0).all()

        # Cap height should not exceed stack height (only check where stacks exist)
        has_stack = state.stack_height > 0
        if has_stack.any():
            assert (state.cap_height[has_stack] <= state.stack_height[has_stack]).all()

        # Rings in hand should be non-negative
        assert (state.rings_in_hand >= 0).all()

        # Where there are stacks, owner should be valid
        has_stack = state.stack_height > 0
        valid_owners = (state.stack_owner[has_stack] >= 1) & (
            state.stack_owner[has_stack] <= 2
        )
        assert valid_owners.all(), "Invalid stack owners found"


class TestDeterministicGameplay:
    """Tests for deterministic game behavior with seeded RNG."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_same_seed_same_outcome(self, device):
        """Games with the same seed should produce identical outcomes."""
        seed = 42
        results = []

        for _ in range(2):
            torch.manual_seed(seed)
            runner = ParallelGameRunner(
                batch_size=1,
                board_size=8,
                num_players=2,
                device=device,
                shadow_validation=False,
                state_validation=False,
                swap_enabled=True,
            )

            weights = [runner._default_weights()]
            max_steps = 500

            for _step in range(max_steps):
                runner._step_games(weights)
                runner._check_victory_conditions()

                if runner.state.game_status[0].item() == GameStatus.COMPLETED:
                    break

            results.append(
                {
                    "winner": runner.state.winner[0].item(),
                    "move_count": runner.state.move_count[0].item(),
                }
            )

        assert results[0] == results[1], "Same seed produced different outcomes"

    def test_different_seeds_different_outcomes(self, device):
        """Different seeds should (usually) produce different games."""
        outcomes = []

        for seed in [1, 2, 3, 4, 5]:
            torch.manual_seed(seed)
            runner = ParallelGameRunner(
                batch_size=1,
                board_size=8,
                num_players=2,
                device=device,
                shadow_validation=False,
                state_validation=False,
                swap_enabled=True,
            )

            weights = [runner._default_weights()]
            max_steps = 500

            for _step in range(max_steps):
                runner._step_games(weights)
                runner._check_victory_conditions()

                if runner.state.game_status[0].item() == GameStatus.COMPLETED:
                    break

            outcomes.append(runner.state.move_count[0].item())

        # Not all games should have the same move count
        unique_counts = set(outcomes)
        assert (
            len(unique_counts) > 1
        ), "All different seeds produced identical move counts"
