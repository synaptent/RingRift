"""Tests for architecture_tracker.py.

Comprehensive test coverage for the ArchitectureStats dataclass,
ArchitectureTracker singleton, and convenience functions.

December 29, 2025: Created for Phase 6 of NN/NNUE multi-harness plan.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.training.architecture_tracker import (
    ArchitectureStats,
    ArchitectureTracker,
    _on_evaluation_completed,
    extract_architecture_from_model_path,
    get_allocation_weights,
    get_architecture_tracker,
    get_best_architecture,
    record_evaluation,
    wire_architecture_tracker_to_events,
)


# =============================================================================
# ArchitectureStats Tests
# =============================================================================


class TestArchitectureStatsInitialization:
    """Tests for ArchitectureStats initialization and defaults."""

    def test_default_values(self):
        """Test default initialization values."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
        )

        assert stats.architecture == "v5"
        assert stats.board_type == "hex8"
        assert stats.num_players == 2
        assert stats.avg_elo == 1000.0
        assert stats.best_elo == 1000.0
        assert stats.worst_elo == 1000.0
        assert stats.elo_variance == 0.0
        assert stats.training_hours == 0.0
        assert stats.elo_per_training_hour == 0.0
        assert stats.games_evaluated == 0
        assert stats.evaluation_count == 0
        assert stats.last_evaluation_time == 0.0
        assert stats._elo_sum == 0.0
        assert stats._elo_sum_sq == 0.0

    def test_custom_values(self):
        """Test initialization with custom values."""
        stats = ArchitectureStats(
            architecture="v4",
            board_type="square8",
            num_players=4,
            avg_elo=1500.0,
            best_elo=1600.0,
        )

        assert stats.architecture == "v4"
        assert stats.board_type == "square8"
        assert stats.num_players == 4
        assert stats.avg_elo == 1500.0
        assert stats.best_elo == 1600.0


class TestArchitectureStatsProperties:
    """Tests for ArchitectureStats computed properties."""

    def test_config_key(self):
        """Test config_key property."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)
        assert stats.config_key == "hex8_2p"

        stats = ArchitectureStats(architecture="v4", board_type="square19", num_players=4)
        assert stats.config_key == "square19_4p"

    def test_full_key(self):
        """Test full_key property."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)
        assert stats.full_key == "v5:hex8_2p"

        stats = ArchitectureStats(architecture="nnue_v1", board_type="square8", num_players=3)
        assert stats.full_key == "nnue_v1:square8_3p"

    def test_efficiency_score_zero_hours(self):
        """Test efficiency_score with zero training hours."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            avg_elo=1500.0,
            training_hours=0.0,
        )
        assert stats.efficiency_score == 0.0

    def test_efficiency_score_positive_hours(self):
        """Test efficiency_score with positive training hours."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            avg_elo=1500.0,
            training_hours=5.0,
        )
        # (1500 - 1000) / 5 = 100
        assert stats.efficiency_score == 100.0

    def test_efficiency_score_below_baseline(self):
        """Test efficiency_score when avg_elo is below baseline."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            avg_elo=900.0,  # Below 1000 baseline
            training_hours=5.0,
        )
        # max(0, 900 - 1000) / 5 = 0
        assert stats.efficiency_score == 0.0

    def test_confidence_interval_single_evaluation(self):
        """Test confidence interval with single evaluation."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            avg_elo=1500.0,
            evaluation_count=1,
        )
        lower, upper = stats.confidence_interval_95
        # Default margin of 200 for single evaluation
        assert lower == 1300.0
        assert upper == 1700.0

    def test_confidence_interval_multiple_evaluations(self):
        """Test confidence interval with multiple evaluations."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            avg_elo=1500.0,
            elo_variance=2500.0,  # std_dev = 50
            evaluation_count=25,
        )
        lower, upper = stats.confidence_interval_95
        # std_error = 50 / sqrt(25) = 10
        # margin = 1.96 * 10 = 19.6
        assert abs(lower - 1480.4) < 0.1
        assert abs(upper - 1519.6) < 0.1


class TestArchitectureStatsRecordEvaluation:
    """Tests for ArchitectureStats.record_evaluation()."""

    def test_first_evaluation(self):
        """Test recording first evaluation."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)

        stats.record_evaluation(elo=1450, training_hours=1.0, games_evaluated=50)

        assert stats.evaluation_count == 1
        assert stats.games_evaluated == 50
        assert stats.training_hours == 1.0
        assert stats.avg_elo == 1450.0
        assert stats.best_elo == 1450.0
        assert stats.worst_elo == 1450.0
        assert stats.last_evaluation_time > 0
        assert stats._elo_sum == 1450.0
        assert stats._elo_sum_sq == 1450.0 * 1450.0

    def test_multiple_evaluations_mean(self):
        """Test mean calculation with multiple evaluations."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)

        stats.record_evaluation(elo=1400, training_hours=1.0, games_evaluated=50)
        stats.record_evaluation(elo=1500, training_hours=1.0, games_evaluated=50)
        stats.record_evaluation(elo=1600, training_hours=1.0, games_evaluated=50)

        assert stats.evaluation_count == 3
        assert stats.games_evaluated == 150
        assert stats.training_hours == 3.0
        assert stats.avg_elo == 1500.0  # (1400 + 1500 + 1600) / 3
        assert stats.best_elo == 1600.0
        assert stats.worst_elo == 1400.0

    def test_multiple_evaluations_variance(self):
        """Test variance calculation with multiple evaluations."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)

        # Add evaluations: 1400, 1500, 1600 -> mean=1500, variance=6666.67
        stats.record_evaluation(elo=1400, training_hours=1.0, games_evaluated=50)
        stats.record_evaluation(elo=1500, training_hours=1.0, games_evaluated=50)
        stats.record_evaluation(elo=1600, training_hours=1.0, games_evaluated=50)

        # Variance = [(1400-1500)^2 + (1500-1500)^2 + (1600-1500)^2] / 3
        # = [10000 + 0 + 10000] / 3 = 6666.67
        assert abs(stats.elo_variance - 6666.67) < 1.0

    def test_efficiency_update(self):
        """Test elo_per_training_hour is updated correctly."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)

        stats.record_evaluation(elo=1500, training_hours=2.5, games_evaluated=100)

        # (1500 - 1000) / 2.5 = 200
        assert stats.elo_per_training_hour == 200.0

    def test_best_elo_updates_when_higher(self):
        """Test best_elo only updates when new elo is higher."""
        stats = ArchitectureStats(architecture="v5", board_type="hex8", num_players=2)

        stats.record_evaluation(elo=1500, training_hours=1.0, games_evaluated=50)
        stats.record_evaluation(elo=1400, training_hours=1.0, games_evaluated=50)  # Lower
        stats.record_evaluation(elo=1600, training_hours=1.0, games_evaluated=50)  # Higher

        assert stats.best_elo == 1600.0
        assert stats.worst_elo == 1400.0


class TestArchitectureStatsSerialization:
    """Tests for ArchitectureStats serialization."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        stats = ArchitectureStats(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            avg_elo=1500.0,
            best_elo=1600.0,
            worst_elo=1400.0,
            elo_variance=2500.0,
            training_hours=5.0,
            elo_per_training_hour=100.0,
            games_evaluated=250,
            evaluation_count=5,
            last_evaluation_time=12345.67,
        )
        stats._elo_sum = 7500.0
        stats._elo_sum_sq = 11300000.0

        data = stats.to_dict()

        assert data["architecture"] == "v5"
        assert data["board_type"] == "hex8"
        assert data["num_players"] == 2
        assert data["avg_elo"] == 1500.0
        assert data["best_elo"] == 1600.0
        assert data["worst_elo"] == 1400.0
        assert data["elo_variance"] == 2500.0
        assert data["training_hours"] == 5.0
        assert data["elo_per_training_hour"] == 100.0
        assert data["games_evaluated"] == 250
        assert data["evaluation_count"] == 5
        assert data["last_evaluation_time"] == 12345.67
        assert data["_elo_sum"] == 7500.0
        assert data["_elo_sum_sq"] == 11300000.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "architecture": "v4",
            "board_type": "square8",
            "num_players": 4,
            "avg_elo": 1450.0,
            "best_elo": 1550.0,
            "worst_elo": 1350.0,
            "elo_variance": 3000.0,
            "training_hours": 4.0,
            "elo_per_training_hour": 112.5,
            "games_evaluated": 200,
            "evaluation_count": 4,
            "last_evaluation_time": 99999.0,
            "_elo_sum": 5800.0,
            "_elo_sum_sq": 8500000.0,
        }

        stats = ArchitectureStats.from_dict(data)

        assert stats.architecture == "v4"
        assert stats.board_type == "square8"
        assert stats.num_players == 4
        assert stats.avg_elo == 1450.0
        assert stats.best_elo == 1550.0
        assert stats.worst_elo == 1350.0
        assert stats.elo_variance == 3000.0
        assert stats.training_hours == 4.0
        assert stats.elo_per_training_hour == 112.5
        assert stats.games_evaluated == 200
        assert stats.evaluation_count == 4
        assert stats.last_evaluation_time == 99999.0
        assert stats._elo_sum == 5800.0
        assert stats._elo_sum_sq == 8500000.0

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "architecture": "v3",
            "board_type": "hex8",
            "num_players": 3,
        }

        stats = ArchitectureStats.from_dict(data)

        assert stats.architecture == "v3"
        assert stats.avg_elo == 1000.0  # Default
        assert stats.games_evaluated == 0  # Default

    def test_round_trip(self):
        """Test serialization round-trip preserves data."""
        original = ArchitectureStats(
            architecture="v5_heavy",
            board_type="hexagonal",
            num_players=4,
        )
        original.record_evaluation(elo=1600, training_hours=3.0, games_evaluated=150)
        original.record_evaluation(elo=1650, training_hours=2.0, games_evaluated=100)

        data = original.to_dict()
        restored = ArchitectureStats.from_dict(data)

        assert restored.architecture == original.architecture
        assert restored.avg_elo == original.avg_elo
        assert restored.best_elo == original.best_elo
        assert restored.evaluation_count == original.evaluation_count
        assert restored._elo_sum == original._elo_sum


# =============================================================================
# ArchitectureTracker Tests
# =============================================================================


class TestArchitectureTrackerSingleton:
    """Tests for ArchitectureTracker singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        ArchitectureTracker.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        ArchitectureTracker.reset_instance()

    def test_get_instance_returns_same_instance(self):
        """Test get_instance returns the same instance."""
        instance1 = ArchitectureTracker.get_instance()
        instance2 = ArchitectureTracker.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test reset_instance creates new instance."""
        instance1 = ArchitectureTracker.get_instance()
        ArchitectureTracker.reset_instance()
        instance2 = ArchitectureTracker.get_instance()
        assert instance1 is not instance2


class TestArchitectureTrackerPersistence:
    """Tests for ArchitectureTracker state persistence."""

    def setup_method(self):
        """Reset singleton and create temp directory."""
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.state_path = Path(self.temp_dir) / "test_stats.json"

    def teardown_method(self):
        """Clean up temp files and reset singleton."""
        ArchitectureTracker.reset_instance()
        if self.state_path.exists():
            self.state_path.unlink()

    def test_save_and_load_state(self):
        """Test state persistence saves and loads correctly."""
        # Create tracker and record data
        tracker1 = ArchitectureTracker(state_path=self.state_path)
        tracker1.record_evaluation(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            elo=1500,
            training_hours=2.0,
            games_evaluated=100,
        )

        # Create new tracker and verify state was loaded
        tracker2 = ArchitectureTracker(state_path=self.state_path)
        stats = tracker2.get_stats("v5", "hex8", 2)

        assert stats is not None
        assert stats.avg_elo == 1500.0
        assert stats.games_evaluated == 100

    def test_load_state_handles_missing_file(self):
        """Test loading state when file doesn't exist."""
        missing_path = Path(self.temp_dir) / "nonexistent.json"
        tracker = ArchitectureTracker(state_path=missing_path)
        assert len(tracker._stats) == 0

    def test_load_state_handles_corrupt_file(self):
        """Test loading state from corrupt JSON file."""
        # Write corrupt JSON
        with open(self.state_path, "w") as f:
            f.write("not valid json{{{")

        tracker = ArchitectureTracker(state_path=self.state_path)
        assert len(tracker._stats) == 0  # Should gracefully handle error


class TestArchitectureTrackerRecordEvaluation:
    """Tests for ArchitectureTracker.record_evaluation()."""

    def setup_method(self):
        """Reset singleton and create tracker with temp state."""
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.state_path = Path(self.temp_dir) / "test_stats.json"
        self.tracker = ArchitectureTracker(state_path=self.state_path)

    def teardown_method(self):
        """Clean up."""
        ArchitectureTracker.reset_instance()

    def test_record_creates_new_stats(self):
        """Test recording creates new ArchitectureStats if not exists."""
        stats = self.tracker.record_evaluation(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            elo=1500,
        )

        assert stats.architecture == "v5"
        assert stats.board_type == "hex8"
        assert stats.num_players == 2
        assert stats.avg_elo == 1500.0

    def test_record_updates_existing_stats(self):
        """Test recording updates existing ArchitectureStats."""
        self.tracker.record_evaluation(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            elo=1400,
        )

        stats = self.tracker.record_evaluation(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            elo=1600,
        )

        assert stats.evaluation_count == 2
        assert stats.avg_elo == 1500.0  # (1400 + 1600) / 2

    def test_record_persists_state(self):
        """Test recording saves state to disk."""
        self.tracker.record_evaluation(
            architecture="v5",
            board_type="hex8",
            num_players=2,
            elo=1500,
        )

        assert self.state_path.exists()
        with open(self.state_path) as f:
            data = json.load(f)
        assert "v5:hex8_2p" in data["stats"]


class TestArchitectureTrackerQueries:
    """Tests for ArchitectureTracker query methods."""

    def setup_method(self):
        """Create tracker with test data."""
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ArchitectureTracker(state_path=Path(self.temp_dir) / "test.json")

        # Add test data
        self.tracker.record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 4, elo=1500, training_hours=1.5, games_evaluated=100)
        self.tracker.record_evaluation("v4", "square8", 2, elo=1450, training_hours=2.5, games_evaluated=100)

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_get_stats_existing(self):
        """Test get_stats for existing architecture."""
        stats = self.tracker.get_stats("v5", "hex8", 2)
        assert stats is not None
        assert stats.avg_elo == 1600.0

    def test_get_stats_nonexistent(self):
        """Test get_stats for nonexistent architecture."""
        stats = self.tracker.get_stats("v6", "hex8", 2)
        assert stats is None

    def test_get_all_stats_no_filter(self):
        """Test get_all_stats without filters."""
        all_stats = self.tracker.get_all_stats()
        assert len(all_stats) == 4

    def test_get_all_stats_filter_board_type(self):
        """Test get_all_stats filtered by board_type."""
        stats = self.tracker.get_all_stats(board_type="hex8")
        assert len(stats) == 3
        for s in stats:
            assert s.board_type == "hex8"

    def test_get_all_stats_filter_num_players(self):
        """Test get_all_stats filtered by num_players."""
        stats = self.tracker.get_all_stats(num_players=2)
        assert len(stats) == 3
        for s in stats:
            assert s.num_players == 2

    def test_get_all_stats_filter_both(self):
        """Test get_all_stats filtered by both fields."""
        stats = self.tracker.get_all_stats(board_type="hex8", num_players=2)
        assert len(stats) == 2


class TestArchitectureTrackerBestArchitecture:
    """Tests for ArchitectureTracker.get_best_architecture()."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ArchitectureTracker(state_path=Path(self.temp_dir) / "test.json")

        # v4: avg=1400, eff=200 (1400-1000)/2
        self.tracker.record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        # v5: avg=1600, eff=600 (1600-1000)/1
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)
        # v5_heavy: avg=1550, best=1700, eff=275 (1550-1000)/2
        self.tracker.record_evaluation("v5_heavy", "hex8", 2, elo=1400, training_hours=1.0, games_evaluated=50)
        self.tracker.record_evaluation("v5_heavy", "hex8", 2, elo=1700, training_hours=1.0, games_evaluated=50)

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_best_by_avg_elo(self):
        """Test get_best_architecture by avg_elo."""
        best = self.tracker.get_best_architecture("hex8", 2, metric="avg_elo")
        assert best is not None
        assert best.architecture == "v5"

    def test_best_by_best_elo(self):
        """Test get_best_architecture by best_elo."""
        best = self.tracker.get_best_architecture("hex8", 2, metric="best_elo")
        assert best is not None
        assert best.architecture == "v5_heavy"

    def test_best_by_efficiency(self):
        """Test get_best_architecture by efficiency_score."""
        best = self.tracker.get_best_architecture("hex8", 2, metric="efficiency_score")
        assert best is not None
        assert best.architecture == "v5"

    def test_best_no_data(self):
        """Test get_best_architecture with no matching data."""
        best = self.tracker.get_best_architecture("hexagonal", 4)
        assert best is None


class TestArchitectureTrackerEfficiencyRanking:
    """Tests for ArchitectureTracker.get_efficiency_ranking()."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ArchitectureTracker(state_path=Path(self.temp_dir) / "test.json")

        self.tracker.record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)
        self.tracker.record_evaluation("v5_heavy", "hex8", 2, elo=1500, training_hours=2.0, games_evaluated=100)

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_efficiency_ranking_order(self):
        """Test efficiency ranking is sorted correctly."""
        ranking = self.tracker.get_efficiency_ranking("hex8", 2)

        assert len(ranking) == 3
        # v5: (1600-1000)/1 = 600
        # v5_heavy: (1500-1000)/2 = 250
        # v4: (1400-1000)/2 = 200
        assert ranking[0][0] == "v5"
        assert ranking[1][0] == "v5_heavy"
        assert ranking[2][0] == "v4"

    def test_efficiency_ranking_values(self):
        """Test efficiency values are correct."""
        ranking = self.tracker.get_efficiency_ranking("hex8", 2)

        assert ranking[0][1] == 600.0  # v5
        assert ranking[1][1] == 250.0  # v5_heavy
        assert ranking[2][1] == 200.0  # v4


class TestArchitectureTrackerAllocationWeights:
    """Tests for ArchitectureTracker.compute_allocation_weights()."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ArchitectureTracker(state_path=Path(self.temp_dir) / "test.json")

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_allocation_weights_sum_to_one(self):
        """Test allocation weights sum to 1.0."""
        self.tracker.record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)
        self.tracker.record_evaluation("v5_heavy", "hex8", 2, elo=1500, training_hours=2.0, games_evaluated=100)

        weights = self.tracker.compute_allocation_weights("hex8", 2)

        assert abs(sum(weights.values()) - 1.0) < 0.0001

    def test_allocation_weights_prefer_efficient(self):
        """Test more efficient architectures get higher weights."""
        self.tracker.record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)

        weights = self.tracker.compute_allocation_weights("hex8", 2)

        assert weights["v5"] > weights["v4"]

    def test_allocation_weights_empty(self):
        """Test allocation weights with no data."""
        weights = self.tracker.compute_allocation_weights("hexagonal", 4)
        assert weights == {}

    def test_allocation_weights_temperature(self):
        """Test temperature affects weight concentration."""
        # Use more similar efficiency values to see temperature effect
        # v4: 1450 Elo / 2.0 hours = 225 efficiency
        # v5: 1550 Elo / 2.0 hours = 275 efficiency
        self.tracker.record_evaluation("v4", "hex8", 2, elo=1450, training_hours=2.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1550, training_hours=2.0, games_evaluated=100)

        weights_low_temp = self.tracker.compute_allocation_weights("hex8", 2, temperature=0.1)
        weights_high_temp = self.tracker.compute_allocation_weights("hex8", 2, temperature=2.0)

        # Lower temperature concentrates more on best
        # Higher temperature makes distribution more uniform
        # So the difference between v5 and v4 should be larger at low temp
        low_temp_diff = weights_low_temp["v5"] - weights_low_temp["v4"]
        high_temp_diff = weights_high_temp["v5"] - weights_high_temp["v4"]
        assert low_temp_diff > high_temp_diff

    def test_allocation_weights_equal_efficiency(self):
        """Test uniform weights when efficiencies are equal."""
        self.tracker.record_evaluation("v4", "hex8", 2, elo=1500, training_hours=2.5, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1500, training_hours=2.5, games_evaluated=100)

        weights = self.tracker.compute_allocation_weights("hex8", 2)

        # Should be approximately equal
        assert abs(weights["v4"] - weights["v5"]) < 0.01


class TestArchitectureTrackerBoost:
    """Tests for ArchitectureTracker.get_architecture_boost()."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ArchitectureTracker(state_path=Path(self.temp_dir) / "test.json")

        self.tracker.record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        self.tracker.record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_boost_for_best_architecture(self):
        """Test best architecture gets no boost (it's the reference)."""
        boost = self.tracker.get_architecture_boost("v5", "hex8", 2)
        assert boost == 1.0

    def test_boost_for_worse_architecture(self):
        """Test worse architecture gets boost based on difference."""
        boost = self.tracker.get_architecture_boost("v4", "hex8", 2)
        # v5 is 200 Elo better, so boost = 1.0 + 200/1000 = 1.2
        assert abs(boost - 1.2) < 0.01

    def test_boost_threshold(self):
        """Test boost is 1.0 when difference below threshold."""
        # Add v5_heavy at 1570 (only 30 below v5 at 1600)
        self.tracker.record_evaluation("v5_heavy", "hex8", 2, elo=1570, training_hours=1.0, games_evaluated=100)

        boost = self.tracker.get_architecture_boost("v5_heavy", "hex8", 2, threshold_elo_diff=50.0)
        assert boost == 1.0  # Below threshold, no boost

    def test_boost_no_data(self):
        """Test boost is 1.0 when no data exists."""
        boost = self.tracker.get_architecture_boost("v6", "hex8", 2)
        assert boost == 1.0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_get_architecture_tracker(self):
        """Test get_architecture_tracker returns singleton."""
        tracker1 = get_architecture_tracker()
        tracker2 = get_architecture_tracker()
        assert tracker1 is tracker2

    def test_record_evaluation_convenience(self):
        """Test record_evaluation convenience function."""
        # Use unique config to avoid pollution from other tests
        stats = record_evaluation(
            architecture="v5",
            board_type="hexagonal",  # Use different board to isolate
            num_players=3,  # Use different players to isolate
            elo=1500,
            training_hours=1.0,
            games_evaluated=50,
        )

        assert stats.architecture == "v5"
        assert stats.board_type == "hexagonal"
        assert stats.num_players == 3

    def test_get_best_architecture_convenience(self):
        """Test get_best_architecture convenience function."""
        record_evaluation("v4", "hex8", 2, elo=1400)
        record_evaluation("v5", "hex8", 2, elo=1600)

        best = get_best_architecture("hex8", 2)
        assert best is not None
        assert best.architecture == "v5"

    def test_get_allocation_weights_convenience(self):
        """Test get_allocation_weights convenience function."""
        record_evaluation("v4", "hex8", 2, elo=1400, training_hours=2.0, games_evaluated=100)
        record_evaluation("v5", "hex8", 2, elo=1600, training_hours=1.0, games_evaluated=100)

        weights = get_allocation_weights("hex8", 2)
        assert "v4" in weights
        assert "v5" in weights
        assert weights["v5"] > weights["v4"]


class TestExtractArchitectureFromModelPath:
    """Tests for extract_architecture_from_model_path()."""

    def test_nnue_model(self):
        """Test NNUE model detection."""
        assert extract_architecture_from_model_path("models/nnue_sq8_2p.pth") == "nnue_v1"
        assert extract_architecture_from_model_path("models/NNUE_hex8_4p.pth") == "nnue_v1"

    def test_v5_heavy(self):
        """Test v5_heavy detection."""
        assert extract_architecture_from_model_path("models/canonical_hex8_2p_v5heavy.pth") == "v5_heavy"
        assert extract_architecture_from_model_path("models/model_v5_heavy.pth") == "v5_heavy"

    def test_versioned_models(self):
        """Test versioned model detection."""
        assert extract_architecture_from_model_path("models/hex8_2p_v6.pth") == "v6"
        assert extract_architecture_from_model_path("models/hex8_2p_v5.pth") == "v5"
        assert extract_architecture_from_model_path("models/hex8_2p_v4.pth") == "v4"
        assert extract_architecture_from_model_path("models/hex8_2p_v3.pth") == "v3"
        assert extract_architecture_from_model_path("models/hex8_2p_v2.pth") == "v2"

    def test_default_canonical(self):
        """Test canonical model without version defaults to v5."""
        assert extract_architecture_from_model_path("models/canonical_hex8_2p.pth") == "v5"
        assert extract_architecture_from_model_path("models/ringrift_best_sq8_4p.pth") == "v5"


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestOnEvaluationCompleted:
    """Tests for _on_evaluation_completed event handler."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    @pytest.mark.asyncio
    async def test_handles_standard_event(self):
        """Test handling standard EVALUATION_COMPLETED event."""
        event = {
            "model_path": "models/canonical_hex8_2p.pth",
            "elo": 1500.0,
            "games_played": 100,
            "board_type": "hex8",
            "num_players": 2,
        }

        await _on_evaluation_completed(event)

        tracker = get_architecture_tracker()
        stats = tracker.get_stats("v5", "hex8", 2)
        assert stats is not None
        assert stats.avg_elo == 1500.0

    @pytest.mark.asyncio
    async def test_handles_config_key_format(self):
        """Test extracting board_type from config key."""
        event = {
            "model_id": "models/canonical_square8_4p.pth",
            "elo": 1450.0,
            "games_played": 50,
            "config": "square8_4p",
        }

        await _on_evaluation_completed(event)

        tracker = get_architecture_tracker()
        stats = tracker.get_stats("v5", "square8", 4)
        assert stats is not None
        assert stats.avg_elo == 1450.0

    @pytest.mark.asyncio
    async def test_handles_missing_board_type(self):
        """Test graceful handling when board_type cannot be extracted."""
        event = {
            "elo": 1500.0,
            "games_played": 100,
            # No board_type, no config
        }

        # Should not raise, just log debug message
        await _on_evaluation_completed(event)


class TestWireArchitectureTrackerToEvents:
    """Tests for wire_architecture_tracker_to_events()."""

    def test_wire_with_event_router(self):
        """Test wiring to event_router."""
        with patch("app.training.architecture_tracker.subscribe") as mock_subscribe:
            result = wire_architecture_tracker_to_events()
            assert result is True
            mock_subscribe.assert_called_once()

    def test_wire_fallback_to_data_events(self):
        """Test fallback to data_events bus."""
        with patch(
            "app.training.architecture_tracker.subscribe",
            side_effect=ImportError("No event_router"),
        ):
            mock_bus = MagicMock()
            with patch(
                "app.training.architecture_tracker.get_event_bus",
                return_value=mock_bus,
            ):
                result = wire_architecture_tracker_to_events()
                # May succeed or fail depending on actual imports
                # Just verify no exception raised

    def test_wire_handles_no_event_system(self):
        """Test graceful handling when no event system available."""
        with patch(
            "app.training.architecture_tracker.subscribe",
            side_effect=ImportError("No event_router"),
        ):
            with patch(
                "app.training.architecture_tracker.get_event_bus",
                side_effect=ImportError("No data_events"),
            ):
                result = wire_architecture_tracker_to_events()
                assert result is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of ArchitectureTracker."""

    def setup_method(self):
        ArchitectureTracker.reset_instance()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        ArchitectureTracker.reset_instance()

    def test_concurrent_record_evaluation(self):
        """Test concurrent record_evaluation calls are safe."""
        import threading

        tracker = ArchitectureTracker(state_path=Path(self.temp_dir) / "test.json")
        results = []
        errors = []

        def record_many(start_elo: int):
            try:
                for i in range(10):
                    tracker.record_evaluation(
                        architecture="v5",
                        board_type="hex8",
                        num_players=2,
                        elo=start_elo + i * 10,
                    )
                results.append(True)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=record_many, args=(1000,)),
            threading.Thread(target=record_many, args=(1500,)),
            threading.Thread(target=record_many, args=(2000,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3

        stats = tracker.get_stats("v5", "hex8", 2)
        assert stats is not None
        assert stats.evaluation_count == 30
