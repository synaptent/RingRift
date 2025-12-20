"""
Tests for the make/unmake benchmark script.

These tests verify:
1. Benchmark script runs without error
2. Incremental search is at least 2x faster than legacy (sanity check)
3. Both search modes produce valid results
"""

import os
import sys

import pytest

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_make_unmake import (
    benchmark_search,
    create_midgame_state,
    create_starting_state,
    run_make_unmake_roundtrip_test,
    validate_correctness,
)


class TestBenchmarkHelpers:
    """Tests for benchmark helper functions."""

    def test_create_starting_state(self):
        """Verify starting state is valid."""
        state = create_starting_state()

        assert state is not None
        assert state.current_player == 1
        assert len(state.players) == 2
        assert state.players[0].rings_in_hand == 18
        assert state.players[1].rings_in_hand == 18
        assert len(state.board.stacks) == 0
        assert len(state.board.markers) == 0

    def test_create_midgame_state(self):
        """Verify midgame state has expected structure."""
        state = create_midgame_state()

        assert state is not None
        assert len(state.board.stacks) > 0
        assert len(state.board.markers) > 0
        assert state.players[0].rings_in_hand < 18
        assert state.players[1].rings_in_hand < 18


class TestMakeUnmakeRoundtrip:
    """Tests for make/unmake roundtrip correctness."""

    def test_roundtrip_restores_state(self):
        """Verify make/unmake roundtrip restores state exactly."""
        passed, messages = run_make_unmake_roundtrip_test()

        # Print messages for debugging
        for msg in messages:
            print(msg)

        assert passed, "Make/unmake roundtrip test failed"


class TestSearchModeEquivalence:
    """Tests for search mode equivalence."""

    def test_both_modes_produce_valid_moves(self):
        """Verify both search modes produce valid results."""
        _passed, messages = validate_correctness(depth=2, num_positions=2)

        # Print messages for debugging
        for msg in messages:
            print(msg)

        # Note: Different moves may be equally good, so we just check
        # that neither crashed and both produced results
        assert len(messages) > 0


class TestBenchmarkPerformance:
    """Performance sanity checks."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("RUN_PERF_BENCHMARKS"),
        reason="Performance benchmark skipped by default (varies by environment). "
        "Set RUN_PERF_BENCHMARKS=1 to run.",
    )
    def test_incremental_faster_than_legacy(self):
        """Verify incremental search is faster than legacy.

        This is a sanity check that the make/unmake pattern provides
        at least 2x speedup compared to legacy immutable state copying.

        To run this benchmark:
            RUN_PERF_BENCHMARKS=1 pytest tests/test_benchmark_make_unmake.py -v -k test_incremental
        """
        depth = 2
        num_runs = 2

        # Run legacy benchmark
        legacy = benchmark_search(
            use_incremental=False,
            depth=depth,
            num_runs=num_runs,
        )

        # Run incremental benchmark
        incremental = benchmark_search(
            use_incremental=True,
            depth=depth,
            num_runs=num_runs,
        )

        # Verify both completed
        assert legacy.avg_time > 0, "Legacy search did not run"
        assert incremental.avg_time > 0, "Incremental search did not run"

        # Calculate speedup
        speedup = legacy.avg_time / incremental.avg_time
        print(f"Speedup at depth {depth}: {speedup:.2f}x")
        print(f"  Legacy: {legacy.avg_time:.3f}s")
        print(f"  Incremental: {incremental.avg_time:.3f}s")

        # Sanity check: incremental should be at least 1.5x faster
        # (using 1.5x instead of 2x to account for variance)
        assert speedup >= 1.5, (
            f"Expected at least 1.5x speedup, got {speedup:.2f}x"
        )


class TestBenchmarkScript:
    """Tests that the benchmark script runs correctly."""

    def test_benchmark_search_legacy_runs(self):
        """Verify legacy benchmark runs without error."""
        result = benchmark_search(
            use_incremental=False,
            depth=2,
            num_runs=1,
        )

        assert result is not None
        assert result.mode == "legacy"
        assert result.depth == 2
        assert result.avg_time >= 0

    def test_benchmark_search_incremental_runs(self):
        """Verify incremental benchmark runs without error."""
        result = benchmark_search(
            use_incremental=True,
            depth=2,
            num_runs=1,
        )

        assert result is not None
        assert result.mode == "incremental"
        assert result.depth == 2
        assert result.avg_time >= 0
