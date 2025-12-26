"""Tests for app.evaluation.benchmark_suite - Benchmark Suite.

This module tests the benchmark suite classes including:
- BenchmarkCategory enum
- BenchmarkResult dataclass
- BenchmarkSuiteResult dataclass
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.evaluation.benchmark_suite import (
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuiteResult,
)


# =============================================================================
# BenchmarkCategory Tests
# =============================================================================


class TestBenchmarkCategory:
    """Tests for BenchmarkCategory enum."""

    def test_performance_category(self):
        """Should have PERFORMANCE category."""
        assert BenchmarkCategory.PERFORMANCE.value == "performance"

    def test_quality_category(self):
        """Should have QUALITY category."""
        assert BenchmarkCategory.QUALITY.value == "quality"

    def test_tactical_category(self):
        """Should have TACTICAL category."""
        assert BenchmarkCategory.TACTICAL.value == "tactical"

    def test_strategic_category(self):
        """Should have STRATEGIC category."""
        assert BenchmarkCategory.STRATEGIC.value == "strategic"

    def test_robustness_category(self):
        """Should have ROBUSTNESS category."""
        assert BenchmarkCategory.ROBUSTNESS.value == "robustness"

    def test_efficiency_category(self):
        """Should have EFFICIENCY category."""
        assert BenchmarkCategory.EFFICIENCY.value == "efficiency"

    def test_category_count(self):
        """Should have exactly 6 categories."""
        assert len(BenchmarkCategory) == 6


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_required_fields(self):
        """Should create result with required fields."""
        result = BenchmarkResult(
            benchmark_name="test_bench",
            category=BenchmarkCategory.PERFORMANCE,
            score=0.95,
            unit="win_rate",
            higher_is_better=True,
        )
        assert result.benchmark_name == "test_bench"
        assert result.category == BenchmarkCategory.PERFORMANCE
        assert result.score == 0.95
        assert result.unit == "win_rate"
        assert result.higher_is_better is True

    def test_default_details(self):
        """Should have empty dict as default details."""
        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.QUALITY,
            score=1.0,
            unit="score",
            higher_is_better=True,
        )
        assert result.details == {}

    def test_custom_details(self):
        """Should accept custom details."""
        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.QUALITY,
            score=1.0,
            unit="score",
            higher_is_better=True,
            details={"games_played": 100, "draws": 5},
        )
        assert result.details["games_played"] == 100
        assert result.details["draws"] == 5

    def test_default_timestamp(self):
        """Should have timestamp set by default."""
        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.TACTICAL,
            score=0.5,
            unit="accuracy",
            higher_is_better=True,
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_default_duration(self):
        """Should have zero duration by default."""
        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.EFFICIENCY,
            score=100.0,
            unit="ms",
            higher_is_better=False,
        )
        assert result.duration_seconds == 0.0

    def test_to_dict(self):
        """Should convert to dict correctly."""
        result = BenchmarkResult(
            benchmark_name="test_bench",
            category=BenchmarkCategory.PERFORMANCE,
            score=0.95,
            unit="win_rate",
            higher_is_better=True,
            duration_seconds=10.5,
        )
        d = result.to_dict()
        assert d["benchmark_name"] == "test_bench"
        assert d["category"] == "performance"
        assert d["score"] == 0.95
        assert d["unit"] == "win_rate"
        assert d["higher_is_better"] is True
        assert d["duration_seconds"] == 10.5
        assert "timestamp" in d

    def test_to_dict_category_is_string(self):
        """Should convert category enum to string."""
        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.STRATEGIC,
            score=0.75,
            unit="score",
            higher_is_better=True,
        )
        d = result.to_dict()
        assert d["category"] == "strategic"
        assert isinstance(d["category"], str)


# =============================================================================
# BenchmarkSuiteResult Tests
# =============================================================================


class TestBenchmarkSuiteResult:
    """Tests for BenchmarkSuiteResult dataclass."""

    def test_required_fields(self):
        """Should create suite result with required fields."""
        suite = BenchmarkSuiteResult(
            suite_name="test_suite",
            model_id="model_v1",
        )
        assert suite.suite_name == "test_suite"
        assert suite.model_id == "model_v1"

    def test_default_results(self):
        """Should have empty results list by default."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
        )
        assert suite.results == []

    def test_default_metadata(self):
        """Should have empty metadata dict by default."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
        )
        assert suite.metadata == {}

    def test_get_score_existing(self):
        """Should get score for existing benchmark."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[
                BenchmarkResult(
                    benchmark_name="bench_a",
                    category=BenchmarkCategory.QUALITY,
                    score=0.95,
                    unit="rate",
                    higher_is_better=True,
                ),
                BenchmarkResult(
                    benchmark_name="bench_b",
                    category=BenchmarkCategory.PERFORMANCE,
                    score=100.0,
                    unit="ms",
                    higher_is_better=False,
                ),
            ],
        )
        assert suite.get_score("bench_a") == 0.95
        assert suite.get_score("bench_b") == 100.0

    def test_get_score_nonexistent(self):
        """Should return None for nonexistent benchmark."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[],
        )
        assert suite.get_score("nonexistent") is None

    def test_get_category_scores(self):
        """Should get all scores for a category."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[
                BenchmarkResult(
                    benchmark_name="quality_1",
                    category=BenchmarkCategory.QUALITY,
                    score=0.90,
                    unit="rate",
                    higher_is_better=True,
                ),
                BenchmarkResult(
                    benchmark_name="quality_2",
                    category=BenchmarkCategory.QUALITY,
                    score=0.85,
                    unit="rate",
                    higher_is_better=True,
                ),
                BenchmarkResult(
                    benchmark_name="perf_1",
                    category=BenchmarkCategory.PERFORMANCE,
                    score=50.0,
                    unit="ms",
                    higher_is_better=False,
                ),
            ],
        )
        quality_scores = suite.get_category_scores(BenchmarkCategory.QUALITY)
        assert len(quality_scores) == 2
        assert quality_scores["quality_1"] == 0.90
        assert quality_scores["quality_2"] == 0.85

        perf_scores = suite.get_category_scores(BenchmarkCategory.PERFORMANCE)
        assert len(perf_scores) == 1
        assert perf_scores["perf_1"] == 50.0

    def test_get_category_scores_empty(self):
        """Should return empty dict for category with no results."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[],
        )
        scores = suite.get_category_scores(BenchmarkCategory.TACTICAL)
        assert scores == {}

    def test_compute_aggregate_score_empty(self):
        """Should return 0 for empty results."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[],
        )
        assert suite.compute_aggregate_score() == 0.0

    def test_compute_aggregate_score_single(self):
        """Should return the score for single result."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[
                BenchmarkResult(
                    benchmark_name="test",
                    category=BenchmarkCategory.QUALITY,
                    score=0.8,
                    unit="rate",
                    higher_is_better=True,
                ),
            ],
        )
        assert suite.compute_aggregate_score() == 0.8

    def test_compute_aggregate_score_multiple(self):
        """Should compute mean of normalized scores."""
        suite = BenchmarkSuiteResult(
            suite_name="test",
            model_id="model",
            results=[
                BenchmarkResult(
                    benchmark_name="a",
                    category=BenchmarkCategory.QUALITY,
                    score=0.9,
                    unit="rate",
                    higher_is_better=True,
                ),
                BenchmarkResult(
                    benchmark_name="b",
                    category=BenchmarkCategory.QUALITY,
                    score=0.7,
                    unit="rate",
                    higher_is_better=True,
                ),
            ],
        )
        # Mean of (0.9, 0.7) = 0.8
        assert suite.compute_aggregate_score() == 0.8

    def test_to_dict(self):
        """Should convert to dict correctly."""
        suite = BenchmarkSuiteResult(
            suite_name="test_suite",
            model_id="model_v1",
            results=[
                BenchmarkResult(
                    benchmark_name="test",
                    category=BenchmarkCategory.QUALITY,
                    score=0.95,
                    unit="rate",
                    higher_is_better=True,
                ),
            ],
            total_duration_seconds=120.5,
            metadata={"board_type": "square8"},
        )
        d = suite.to_dict()
        assert d["suite_name"] == "test_suite"
        assert d["model_id"] == "model_v1"
        assert len(d["results"]) == 1
        assert d["total_duration_seconds"] == 120.5
        assert d["metadata"]["board_type"] == "square8"
        assert "aggregate_score" in d
        assert "timestamp" in d


# =============================================================================
# Integration Tests
# =============================================================================


class TestBenchmarkSuiteIntegration:
    """Integration tests for benchmark suite."""

    def test_full_workflow(self):
        """Test creating and analyzing a complete benchmark suite."""
        # Create multiple benchmark results
        results = [
            BenchmarkResult(
                benchmark_name="win_rate_vs_random",
                category=BenchmarkCategory.QUALITY,
                score=0.95,
                unit="win_rate",
                higher_is_better=True,
                details={"games": 100, "wins": 95, "losses": 5},
            ),
            BenchmarkResult(
                benchmark_name="inference_time",
                category=BenchmarkCategory.PERFORMANCE,
                score=15.2,
                unit="ms",
                higher_is_better=False,
                details={"median": 15.2, "p99": 25.0},
            ),
            BenchmarkResult(
                benchmark_name="tactical_accuracy",
                category=BenchmarkCategory.TACTICAL,
                score=0.88,
                unit="accuracy",
                higher_is_better=True,
                details={"positions_tested": 500},
            ),
        ]

        # Create suite
        suite = BenchmarkSuiteResult(
            suite_name="full_evaluation",
            model_id="hex8_2p_v3",
            results=results,
            total_duration_seconds=3600.0,
            metadata={
                "board_type": "hex8",
                "num_players": 2,
                "gpu": "H100",
            },
        )

        # Verify queries work
        assert suite.get_score("win_rate_vs_random") == 0.95
        assert suite.get_score("inference_time") == 15.2

        # Verify category grouping
        quality_scores = suite.get_category_scores(BenchmarkCategory.QUALITY)
        assert len(quality_scores) == 1

        # Verify dict conversion
        d = suite.to_dict()
        assert d["suite_name"] == "full_evaluation"
        assert len(d["results"]) == 3
        assert d["aggregate_score"] > 0  # Should have computed aggregate
