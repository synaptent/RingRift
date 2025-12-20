"""Performance regression tests for CI.

These tests measure key performance metrics and compare against established
baselines. They fail if performance degrades significantly.

Run locally with:
    pytest tests/benchmarks/test_performance_regression.py -v --tb=short

Performance baselines are calibrated for CI runners (ubuntu-latest with 2 cores).
Local machines may be faster.
"""

import time
import pytest
import torch

try:
    from app.ai.gpu_parallel_games import ParallelGameRunner
    from app.ai.gpu_batch_state import BatchGameState
    from app.ai.gpu_move_generation import generate_placement_moves_batch
    from app.ai.gpu_game_types import GameStatus

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="GPU modules not available"
)


# Performance baselines (calibrated for CI runners - 2 core ubuntu-latest)
# These are conservative to avoid flaky failures
BASELINES = {
    # Game simulation: games per second
    "game_simulation_gps": 50,  # Minimum 50 games/sec for batch of 32
    # Move generation: moves per second
    "move_generation_mps": 5000,  # Minimum 5k moves/sec
    # State creation: states per second
    "state_creation_sps": 500,  # Minimum 500 states/sec
    # Full game completion: max seconds per game
    "full_game_max_seconds": 2.0,  # Max 2 seconds per game
}

# Tolerance for variance (allow 20% worse than baseline)
TOLERANCE = 0.20


class TestGameSimulationPerformance:
    """Benchmark game simulation throughput."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_batch_simulation_throughput(self, device):
        """Measure games per second for batch simulation."""
        batch_size = 32

        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]

        # Warm up
        for _ in range(5):
            runner._step_games(weights)

        # Benchmark
        num_steps = 100
        start = time.perf_counter()
        for _ in range(num_steps):
            runner._step_games(weights)
        elapsed = time.perf_counter() - start

        # Calculate games per second (batch_size * steps / time)
        total_game_steps = batch_size * num_steps
        gps = total_game_steps / elapsed

        min_gps = BASELINES["game_simulation_gps"] * (1 - TOLERANCE)
        print(f"\nGame simulation: {gps:.1f} game-steps/sec (min: {min_gps:.1f})")

        assert gps >= min_gps, (
            f"Game simulation too slow: {gps:.1f} GPS < {min_gps:.1f} GPS minimum"
        )

    def test_small_batch_overhead(self, device):
        """Verify small batches don't have excessive overhead."""
        batch_size = 4

        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]

        # Benchmark
        num_steps = 100
        start = time.perf_counter()
        for _ in range(num_steps):
            runner._step_games(weights)
        elapsed = time.perf_counter() - start

        # Should still be reasonably fast even with small batch
        gps = (batch_size * num_steps) / elapsed
        min_gps = 20  # More lenient for small batches

        print(f"\nSmall batch simulation: {gps:.1f} GPS (min: {min_gps:.1f})")

        assert gps >= min_gps, f"Small batch too slow: {gps:.1f} GPS"


class TestMoveGenerationPerformance:
    """Benchmark move generation speed."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_move_generation_throughput(self, device):
        """Measure moves per second for batch move generation."""
        batch_size = 16

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Set up for placement phase (many valid moves)
        state.rings_in_hand[:, 1] = 18
        state.rings_in_hand[:, 2] = 18

        # Warm up
        for _ in range(3):
            _ = generate_placement_moves_batch(state)

        # Benchmark
        num_iterations = 50
        total_moves = 0
        start = time.perf_counter()
        for _ in range(num_iterations):
            moves = generate_placement_moves_batch(state)
            total_moves += moves.total_moves
        elapsed = time.perf_counter() - start

        mps = total_moves / elapsed
        min_mps = BASELINES["move_generation_mps"] * (1 - TOLERANCE)

        print(f"\nMove generation: {mps:.0f} moves/sec (min: {min_mps:.0f})")

        assert mps >= min_mps, (
            f"Move generation too slow: {mps:.0f} MPS < {min_mps:.0f} MPS minimum"
        )


class TestStateCreationPerformance:
    """Benchmark state creation speed."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_batch_state_creation_throughput(self, device):
        """Measure states per second for batch creation."""
        batch_size = 16

        # Warm up
        for _ in range(3):
            _ = BatchGameState.create_batch(
                batch_size=batch_size,
                board_size=8,
                num_players=2,
                device=device,
            )

        # Benchmark
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = BatchGameState.create_batch(
                batch_size=batch_size,
                board_size=8,
                num_players=2,
                device=device,
            )
        elapsed = time.perf_counter() - start

        sps = (batch_size * num_iterations) / elapsed
        min_sps = BASELINES["state_creation_sps"] * (1 - TOLERANCE)

        print(f"\nState creation: {sps:.0f} states/sec (min: {min_sps:.0f})")

        assert sps >= min_sps, (
            f"State creation too slow: {sps:.0f} SPS < {min_sps:.0f} SPS minimum"
        )


class TestFullGamePerformance:
    """Benchmark complete game simulation."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_full_game_completion_time(self, device):
        """Verify games complete within reasonable time."""
        torch.manual_seed(42)  # Deterministic for consistent timing

        runner = ParallelGameRunner(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )

        weights = [runner._default_weights()]
        max_steps = 500

        start = time.perf_counter()
        for _ in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if runner.state.game_status[0].item() == GameStatus.COMPLETED:
                break
        elapsed = time.perf_counter() - start

        max_seconds = BASELINES["full_game_max_seconds"]
        print(f"\nFull game time: {elapsed:.3f}s (max: {max_seconds:.1f}s)")

        assert elapsed < max_seconds, (
            f"Full game too slow: {elapsed:.2f}s > {max_seconds}s maximum"
        )

    def test_batch_game_completion_time(self, device):
        """Verify batch games complete within reasonable time."""
        torch.manual_seed(42)
        batch_size = 8

        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]
        max_steps = 500

        start = time.perf_counter()
        for _ in range(max_steps):
            runner._step_games(weights)
            runner._check_victory_conditions()

            if (runner.state.game_status == GameStatus.COMPLETED).all():
                break
        elapsed = time.perf_counter() - start

        # Batch should be faster per-game than single game
        max_per_game = BASELINES["full_game_max_seconds"] * 1.5  # 50% overhead for batch
        max_total = max_per_game * batch_size

        print(f"\nBatch games time: {elapsed:.3f}s for {batch_size} games")
        print(f"  Per-game avg: {elapsed/batch_size:.3f}s (max: {max_per_game:.1f}s)")

        assert elapsed < max_total, (
            f"Batch games too slow: {elapsed:.2f}s > {max_total}s maximum"
        )


class TestScalingPerformance:
    """Test performance scaling with batch size."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_linear_scaling_with_batch_size(self, device):
        """Verify throughput scales roughly linearly with batch size."""
        results = {}

        for batch_size in [4, 8, 16, 32]:
            runner = ParallelGameRunner(
                batch_size=batch_size,
                board_size=8,
                num_players=2,
                device=device,
                shadow_validation=False,
                state_validation=False,
            )

            weights = [runner._default_weights() for _ in range(batch_size)]

            # Benchmark
            num_steps = 50
            start = time.perf_counter()
            for _ in range(num_steps):
                runner._step_games(weights)
            elapsed = time.perf_counter() - start

            gps = (batch_size * num_steps) / elapsed
            results[batch_size] = gps

        print("\nScaling results:")
        for bs, gps in sorted(results.items()):
            print(f"  batch_size={bs}: {gps:.1f} GPS")

        # Larger batches should be more efficient (higher GPS)
        # Allow 30% variance
        assert results[32] >= results[4] * 0.7, (
            f"Scaling degradation: batch=32 ({results[32]:.1f} GPS) "
            f"should be >= 70% of batch=4 ({results[4]:.1f} GPS)"
        )


class TestMemoryPerformance:
    """Test memory usage doesn't grow unexpectedly."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_no_memory_growth_in_simulation(self, device):
        """Verify memory doesn't grow during extended simulation."""
        import gc

        batch_size = 8

        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
            shadow_validation=False,
            state_validation=False,
        )

        weights = [runner._default_weights() for _ in range(batch_size)]

        # Run initial steps to establish baseline
        for _ in range(50):
            runner._step_games(weights)
            runner._check_victory_conditions()

        gc.collect()
        initial_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

        # Run many more steps
        for _ in range(200):
            runner._step_games(weights)
            runner._check_victory_conditions()

        gc.collect()
        final_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

        growth = final_tensors - initial_tensors
        max_growth = 50  # Allow some variance but not unbounded growth

        print(f"\nTensor count: initial={initial_tensors}, final={final_tensors}, growth={growth}")

        assert growth <= max_growth, (
            f"Possible memory leak: tensor count grew by {growth} > {max_growth}"
        )
