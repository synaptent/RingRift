"""Tests for composite culling system."""

from unittest.mock import patch

import pytest

from app.tournament.composite_culling import (
    CullingConfig,
    CullingReport,
    HierarchicalCullingController,
)


class TestCullingConfig:
    """Tests for CullingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CullingConfig()

        assert config.min_games_for_cull == 30
        assert config.min_participants_keep == 25
        assert config.protect_baselines is True
        assert config.protect_best_per_algo is True
        assert config.diversity_min_algos == 3
        assert config.nn_cull_threshold == 0.5
        assert config.algo_keep_per_nn == 2
        assert config.standard_keep_fraction == 0.25

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CullingConfig(
            min_games_for_cull=50,
            min_participants_keep=10,
            protect_baselines=False,
        )

        assert config.min_games_for_cull == 50
        assert config.min_participants_keep == 10
        assert config.protect_baselines is False


class TestCullingReport:
    """Tests for CullingReport dataclass."""

    def test_report_creation(self):
        """Test culling report creation."""
        report = CullingReport(
            board_type="square8",
            num_players=2,
        )

        assert report.board_type == "square8"
        assert report.num_players == 2
        assert report.dry_run is True
        assert report.level1_nns_culled == 0
        assert report.level2_combinations_culled == 0
        assert report.level3_participants_culled == 0

    def test_report_with_culling_data(self):
        """Test culling report with data."""
        report = CullingReport(
            board_type="square8",
            num_players=2,
            dry_run=False,
            level1_nns_evaluated=10,
            level1_nns_culled=3,
            level1_culled_nn_ids=["nn1", "nn2", "nn3"],
        )

        assert report.level1_nns_evaluated == 10
        assert report.level1_nns_culled == 3
        assert len(report.level1_culled_nn_ids) == 3


class TestHierarchicalCullingController:
    """Tests for HierarchicalCullingController."""

    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = HierarchicalCullingController(
            board_type="square8",
            num_players=2,
        )

        assert controller.board_type == "square8"
        assert controller.num_players == 2
        assert controller.config is not None

    def test_controller_with_custom_config(self):
        """Test controller with custom config."""
        config = CullingConfig(min_games_for_cull=100)
        controller = HierarchicalCullingController(
            board_type="square8",
            num_players=2,
            config=config,
        )

        assert controller.config.min_games_for_cull == 100

    def test_dry_run_returns_report(self):
        """Test that dry run returns a valid report."""
        with patch("app.tournament.composite_culling.get_elo_service") as mock_get_elo:
            mock_get_elo.return_value.get_composite_leaderboard.return_value = []
            controller = HierarchicalCullingController(
                board_type="square8",
                num_players=2,
            )

            report = controller.run_culling(dry_run=True)

        assert isinstance(report, CullingReport)
        assert report.dry_run is True

    @pytest.mark.parametrize("board_type", ["square8", "square19", "hexagonal"])
    def test_different_board_types(self, board_type):
        """Test controller works with different board types."""
        controller = HierarchicalCullingController(
            board_type=board_type,
            num_players=2,
        )

        assert controller.board_type == board_type

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_different_player_counts(self, num_players):
        """Test controller works with different player counts."""
        controller = HierarchicalCullingController(
            board_type="square8",
            num_players=num_players,
        )

        assert controller.num_players == num_players
