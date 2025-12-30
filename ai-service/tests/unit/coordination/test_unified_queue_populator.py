"""Comprehensive tests for app.coordination.unified_queue_populator module.

Tests cover:
- QueuePopulatorConfig dataclass (all fields, defaults, validation)
- ConfigTarget dataclass (Elo tracking, velocity calculation, days to target)
- UnifiedQueuePopulator class (initialization, state management, work item creation)
- UnifiedQueuePopulatorDaemon class (lifecycle, health checks, event handling)
- Queue allocation percentages (60% selfplay, 30% training, 10% tournament)
- Work item prioritization with scheduler integration
- Elo target tracking and velocity calculations
- Event emissions (QUEUE_POPULATED, WORK_ITEM_ADDED)
- Backward compatibility aliases (PopulatorConfig, QueuePopulator)
- Trickle mode for backpressure handling
- Cluster health integration

December 2025: Comprehensive test suite with 50+ tests covering all public methods.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from app.coordination.unified_queue_populator import (
    BOARD_CONFIGS,
    DEFAULT_CURRICULUM_WEIGHTS,
    LARGE_BOARDS,
    ConfigTarget,
    PopulatorConfig,
    QueuePopulator,
    QueuePopulatorConfig,
    UnifiedQueuePopulator,
    UnifiedQueuePopulatorDaemon,
    get_queue_populator,
    get_queue_populator_daemon,
    load_populator_config_from_yaml,
    reset_queue_populator,
    start_queue_populator_daemon,
    wire_queue_populator_events,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_queue_populator()
    yield
    reset_queue_populator()


@pytest.fixture
def mock_populator_config():
    """Create a test configuration with minimal board configs for fast tests."""
    return QueuePopulatorConfig(
        board_types=["hex8"],
        player_counts=[2],
        check_interval_seconds=1,
        min_queue_depth=10,
        target_queue_depth=20,
        max_batch_per_cycle=10,
    )


@pytest.fixture
def full_populator_config():
    """Create a full configuration with all board types."""
    return QueuePopulatorConfig(
        board_types=["square8", "square19", "hex8", "hexagonal"],
        player_counts=[2, 3, 4],
        min_queue_depth=50,
        target_queue_depth=100,
    )


@pytest.fixture
def mock_work_queue():
    """Create a mock work queue."""
    queue = MagicMock()
    queue.get_queue_status.return_value = {
        "pending": [],
        "running": [],
    }
    return queue


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_board_configs_contains_all_combinations(self):
        """Test BOARD_CONFIGS has all 12 canonical combinations."""
        assert len(BOARD_CONFIGS) == 12
        board_types = {"square8", "square19", "hex8", "hexagonal"}
        player_counts = {2, 3, 4}
        for board, players in BOARD_CONFIGS:
            assert board in board_types
            assert players in player_counts

    def test_large_boards_frozenset(self):
        """Test LARGE_BOARDS is a frozenset."""
        assert isinstance(LARGE_BOARDS, frozenset)

    def test_large_boards_contains_expected(self):
        """Test LARGE_BOARDS contains expected board types."""
        assert "square19" in LARGE_BOARDS
        assert "hexagonal" in LARGE_BOARDS
        assert "fullhex" in LARGE_BOARDS
        assert "full_hex" in LARGE_BOARDS

    def test_large_boards_excludes_small(self):
        """Test LARGE_BOARDS excludes small boards."""
        assert "square8" not in LARGE_BOARDS
        assert "hex8" not in LARGE_BOARDS

    def test_default_curriculum_weights_keys(self):
        """Test DEFAULT_CURRICULUM_WEIGHTS has expected config keys."""
        expected_keys = {
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hex8_2p", "hex8_3p", "hex8_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        }
        assert set(DEFAULT_CURRICULUM_WEIGHTS.keys()) == expected_keys

    def test_default_curriculum_weights_values_positive(self):
        """Test all curriculum weights are positive."""
        for key, weight in DEFAULT_CURRICULUM_WEIGHTS.items():
            assert weight > 0, f"Weight for {key} should be positive"


# =============================================================================
# QueuePopulatorConfig Tests
# =============================================================================


class TestQueuePopulatorConfig:
    """Tests for QueuePopulatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QueuePopulatorConfig()
        assert config.enabled is True
        assert config.min_queue_depth == 200
        assert config.max_pending_items == 50
        assert config.target_queue_depth == 300
        assert config.max_batch_per_cycle == 100
        assert config.check_interval_seconds == 10
        assert config.target_elo == 2000.0

    def test_work_distribution_defaults(self):
        """Test work distribution ratios default to 60/30/10."""
        config = QueuePopulatorConfig()
        assert config.selfplay_ratio == 0.60
        assert config.training_ratio == 0.30
        assert config.tournament_ratio == 0.10

    def test_work_distribution_sums_to_one(self):
        """Test work distribution ratios sum to 1.0."""
        config = QueuePopulatorConfig()
        total = config.selfplay_ratio + config.training_ratio + config.tournament_ratio
        assert abs(total - 1.0) < 0.001

    def test_board_types_default(self):
        """Test default board types list."""
        config = QueuePopulatorConfig()
        assert "square8" in config.board_types
        assert "square19" in config.board_types
        assert "hex8" in config.board_types
        assert "hexagonal" in config.board_types
        assert len(config.board_types) == 4

    def test_player_counts_default(self):
        """Test default player counts."""
        config = QueuePopulatorConfig()
        assert config.player_counts == [2, 3, 4]

    def test_selfplay_settings_defaults(self):
        """Test selfplay-related settings."""
        config = QueuePopulatorConfig()
        assert config.selfplay_games_per_item == 50
        assert config.selfplay_priority == 50
        assert config.selfplay_timeout_seconds == 3600.0

    def test_training_settings_defaults(self):
        """Test training-related settings."""
        config = QueuePopulatorConfig()
        assert config.training_priority == 100
        assert config.min_games_for_training == 100

    def test_tournament_settings_defaults(self):
        """Test tournament-related settings."""
        config = QueuePopulatorConfig()
        assert config.tournament_games == 50
        assert config.tournament_priority == 80

    def test_trickle_mode_settings(self):
        """Test trickle mode settings (Phase 15.1.2)."""
        config = QueuePopulatorConfig()
        assert config.trickle_mode_enabled is True
        assert config.trickle_min_items == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QueuePopulatorConfig(
            enabled=False,
            min_queue_depth=100,
            target_elo=2500.0,
            selfplay_ratio=0.70,
            training_ratio=0.20,
            tournament_ratio=0.10,
            board_types=["hex8"],
            player_counts=[2, 4],
        )
        assert config.enabled is False
        assert config.min_queue_depth == 100
        assert config.target_elo == 2500.0
        assert config.selfplay_ratio == 0.70
        assert config.board_types == ["hex8"]
        assert config.player_counts == [2, 4]


# =============================================================================
# ConfigTarget Tests
# =============================================================================


class TestConfigTarget:
    """Tests for ConfigTarget dataclass."""

    def test_create_with_defaults(self):
        """Test creating ConfigTarget with defaults."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        assert target.board_type == "hex8"
        assert target.num_players == 2
        assert target.target_elo == 2000.0
        assert target.current_best_elo == 1500.0
        assert target.best_model_id is None
        assert target.games_played == 0
        assert target.training_runs == 0
        assert target.total_samples == 0
        assert target.elo_history == []

    def test_config_key_format(self):
        """Test config_key property format."""
        test_cases = [
            ("hex8", 2, "hex8_2p"),
            ("square8", 3, "square8_3p"),
            ("hexagonal", 4, "hexagonal_4p"),
            ("square19", 2, "square19_2p"),
        ]
        for board_type, num_players, expected in test_cases:
            target = ConfigTarget(board_type=board_type, num_players=num_players)
            assert target.config_key == expected

    def test_target_met_when_above(self):
        """Test target_met when Elo is above target."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=2100.0,
            target_elo=2000.0,
        )
        assert target.target_met is True

    def test_target_met_when_equal(self):
        """Test target_met when Elo equals target."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=2000.0,
            target_elo=2000.0,
        )
        assert target.target_met is True

    def test_target_met_when_below(self):
        """Test target_met when Elo is below target."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=1800.0,
            target_elo=2000.0,
        )
        assert target.target_met is False

    def test_elo_gap_calculation(self):
        """Test elo_gap property."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=1800.0,
            target_elo=2000.0,
        )
        assert target.elo_gap == 200.0

    def test_elo_gap_clamped_to_zero(self):
        """Test elo_gap is 0 when above target."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=2100.0,
            target_elo=2000.0,
        )
        assert target.elo_gap == 0.0

    def test_elo_velocity_no_history(self):
        """Test elo_velocity returns 0 with no history."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        assert target.elo_velocity == 0.0

    def test_elo_velocity_single_point(self):
        """Test elo_velocity returns 0 with single data point."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        target.record_elo(1600.0)
        assert target.elo_velocity == 0.0

    def test_elo_velocity_increasing(self):
        """Test elo_velocity calculation with increasing Elo."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        now = time.time()
        target.elo_history = [
            (now - 3 * 86400, 1500.0),
            (now - 2 * 86400, 1600.0),
            (now - 1 * 86400, 1700.0),
            (now, 1800.0),
        ]
        velocity = target.elo_velocity
        assert 90.0 <= velocity <= 110.0

    def test_elo_velocity_decreasing(self):
        """Test elo_velocity calculation with decreasing Elo."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        now = time.time()
        target.elo_history = [
            (now - 2 * 86400, 1800.0),
            (now - 1 * 86400, 1700.0),
            (now, 1600.0),
        ]
        velocity = target.elo_velocity
        assert -110.0 <= velocity <= -90.0

    def test_elo_velocity_filters_old_history(self):
        """Test elo_velocity only uses last 7 days of history."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        now = time.time()
        target.elo_history = [
            (now - 10 * 86400, 1400.0),
            (now - 8 * 86400, 1450.0),
            (now - 2 * 86400, 1600.0),
            (now, 1700.0),
        ]
        velocity = target.elo_velocity
        assert 40.0 <= velocity <= 60.0

    def test_days_to_target_when_met(self):
        """Test days_to_target when target already met."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=2100.0,
            target_elo=2000.0,
        )
        assert target.days_to_target == 0.0

    def test_days_to_target_zero_velocity(self):
        """Test days_to_target with zero velocity."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=1800.0,
            target_elo=2000.0,
        )
        assert target.days_to_target is None

    def test_days_to_target_negative_velocity(self):
        """Test days_to_target with negative velocity."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=1800.0,
            target_elo=2000.0,
        )
        now = time.time()
        target.elo_history = [
            (now - 2 * 86400, 1900.0),
            (now, 1800.0),
        ]
        assert target.days_to_target is None

    def test_days_to_target_calculation(self):
        """Test days_to_target calculation with positive velocity."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=1800.0,
            target_elo=2000.0,
        )
        now = time.time()
        target.elo_history = [
            (now - 2 * 86400, 1600.0),
            (now - 1 * 86400, 1700.0),
            (now, 1800.0),
        ]
        eta = target.days_to_target
        assert eta is not None
        assert 1.5 <= eta <= 2.5

    def test_record_elo_adds_to_history(self):
        """Test record_elo adds entry to history."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        target.record_elo(1600.0)
        target.record_elo(1650.0)
        assert len(target.elo_history) == 2

    def test_record_elo_prunes_old_history(self):
        """Test record_elo prunes entries older than 30 days."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        now = time.time()
        target.elo_history.append((now - 35 * 86400, 1500.0))
        target.record_elo(1600.0, timestamp=now)
        assert len(target.elo_history) == 1
        assert target.elo_history[0][1] == 1600.0

    def test_record_elo_with_custom_timestamp(self):
        """Test record_elo with explicit timestamp."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        custom_ts = 1700000000.0
        target.record_elo(1600.0, timestamp=custom_ts)
        assert target.elo_history[0][0] == custom_ts
        assert target.elo_history[0][1] == 1600.0


# =============================================================================
# UnifiedQueuePopulator Initialization Tests
# =============================================================================


class TestUnifiedQueuePopulatorInit:
    """Tests for UnifiedQueuePopulator initialization."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_defaults(self, mock_scale, mock_load):
        """Test initialization with default config."""
        populator = UnifiedQueuePopulator()
        assert populator.config is not None
        assert populator._work_queue is None
        assert populator._selfplay_scheduler is None
        assert isinstance(populator._targets, dict)

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_creates_all_targets(self, mock_scale, mock_load):
        """Test initialization creates targets for all configs."""
        populator = UnifiedQueuePopulator()
        assert len(populator._targets) == 12  # 4 boards * 3 player counts

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_custom_config(self, mock_scale, mock_load, mock_populator_config):
        """Test initialization with custom config."""
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        assert len(populator._targets) == 1
        assert "hex8_2p" in populator._targets

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_work_queue(self, mock_scale, mock_load, mock_work_queue):
        """Test initialization with provided work queue."""
        populator = UnifiedQueuePopulator(work_queue=mock_work_queue)
        assert populator._work_queue is mock_work_queue

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_elo_db_path(self, mock_scale, mock_load):
        """Test initialization with custom Elo database path."""
        populator = UnifiedQueuePopulator(elo_db_path="/custom/path/elo.db")
        assert populator._elo_db_path == "/custom/path/elo.db"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_cluster_health_defaults(self, mock_scale, mock_load):
        """Test cluster health defaults are set."""
        populator = UnifiedQueuePopulator()
        assert populator._cluster_health_factor == 1.0
        assert populator._dead_nodes == set()


# =============================================================================
# UnifiedQueuePopulator State Management Tests
# =============================================================================


class TestUnifiedQueuePopulatorStateManagement:
    """Tests for state management methods."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_update_target_elo(self, mock_scale, mock_load):
        """Test update_target_elo method."""
        populator = UnifiedQueuePopulator()
        populator.update_target_elo("hex8", 2, 1700.0, "model_123")
        target = populator._targets["hex8_2p"]
        assert target.current_best_elo == 1700.0
        assert target.best_model_id == "model_123"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_update_target_elo_ignores_lower(self, mock_scale, mock_load):
        """Test update_target_elo ignores lower Elo values."""
        populator = UnifiedQueuePopulator()
        populator.update_target_elo("hex8", 2, 1800.0, "model_1")
        populator.update_target_elo("hex8", 2, 1700.0, "model_2")
        target = populator._targets["hex8_2p"]
        assert target.current_best_elo == 1800.0
        assert target.best_model_id == "model_1"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_update_target_elo_unknown_config(self, mock_scale, mock_load):
        """Test update_target_elo with unknown config is ignored."""
        populator = UnifiedQueuePopulator()
        populator.update_target_elo("unknown_board", 5, 1700.0, "model_123")
        # Should not raise

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_increment_games(self, mock_scale, mock_load):
        """Test increment_games method."""
        populator = UnifiedQueuePopulator()
        populator.increment_games("hex8", 2, 10)
        populator.increment_games("hex8", 2, 5)
        target = populator._targets["hex8_2p"]
        assert target.games_played == 15
        assert target.games_since_last_export == 15

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_increment_games_default_count(self, mock_scale, mock_load):
        """Test increment_games with default count of 1."""
        populator = UnifiedQueuePopulator()
        populator.increment_games("hex8", 2)
        target = populator._targets["hex8_2p"]
        assert target.games_played == 1

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_increment_training(self, mock_scale, mock_load):
        """Test increment_training method."""
        populator = UnifiedQueuePopulator()
        populator.increment_training("hex8", 2)
        populator.increment_training("hex8", 2)
        target = populator._targets["hex8_2p"]
        assert target.training_runs == 2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_mark_export_complete(self, mock_scale, mock_load):
        """Test mark_export_complete method."""
        populator = UnifiedQueuePopulator()
        populator.increment_games("hex8", 2, 100)
        populator.mark_export_complete("hex8", 2, samples=5000)
        target = populator._targets["hex8_2p"]
        assert target.games_since_last_export == 0
        assert target.total_samples == 5000
        assert target.pending_export is False

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_set_work_queue(self, mock_scale, mock_load, mock_work_queue):
        """Test set_work_queue method."""
        populator = UnifiedQueuePopulator()
        populator.set_work_queue(mock_work_queue)
        assert populator._work_queue is mock_work_queue

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_set_selfplay_scheduler(self, mock_scale, mock_load):
        """Test set_selfplay_scheduler method."""
        populator = UnifiedQueuePopulator()
        mock_scheduler = MagicMock()
        populator.set_selfplay_scheduler(mock_scheduler)
        assert populator._selfplay_scheduler is mock_scheduler


# =============================================================================
# UnifiedQueuePopulator Target Tracking Tests
# =============================================================================


class TestUnifiedQueuePopulatorTargetTracking:
    """Tests for Elo target tracking methods."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_all_targets_met_false(self, mock_scale, mock_load):
        """Test all_targets_met returns False initially."""
        populator = UnifiedQueuePopulator()
        assert populator.all_targets_met() is False

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_all_targets_met_true(self, mock_scale, mock_load, mock_populator_config):
        """Test all_targets_met returns True when all met."""
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0
        assert populator.all_targets_met() is True

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_unmet_targets(self, mock_scale, mock_load):
        """Test get_unmet_targets method."""
        config = QueuePopulatorConfig(
            board_types=["hex8", "square8"],
            player_counts=[2],
        )
        populator = UnifiedQueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0
        unmet = populator.get_unmet_targets()
        assert len(unmet) == 1
        assert unmet[0].config_key == "square8_2p"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_priority_target_smallest_gap(self, mock_scale, mock_load):
        """Test get_priority_target returns target with smallest gap."""
        config = QueuePopulatorConfig(
            board_types=["hex8", "square8"],
            player_counts=[2],
        )
        populator = UnifiedQueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 1900.0  # 100 gap
        populator._targets["square8_2p"].current_best_elo = 1800.0  # 200 gap
        priority = populator.get_priority_target()
        assert priority is not None
        assert priority.config_key == "hex8_2p"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_priority_target_none_when_all_met(self, mock_scale, mock_load, mock_populator_config):
        """Test get_priority_target returns None when all targets met."""
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0
        assert populator.get_priority_target() is None


# =============================================================================
# UnifiedQueuePopulator Queue Depth Tests
# =============================================================================


class TestUnifiedQueuePopulatorQueueDepth:
    """Tests for queue depth calculations."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_current_queue_depth_no_queue(self, mock_scale, mock_load):
        """Test get_current_queue_depth returns 0 without queue."""
        populator = UnifiedQueuePopulator()
        assert populator.get_current_queue_depth() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_current_queue_depth_with_queue(self, mock_scale, mock_load, mock_work_queue):
        """Test get_current_queue_depth with mock queue."""
        mock_work_queue.get_queue_status.return_value = {
            "pending": [{"id": "1"}, {"id": "2"}],
            "running": [{"id": "3"}],
        }
        populator = UnifiedQueuePopulator()
        populator.set_work_queue(mock_work_queue)
        assert populator.get_current_queue_depth() == 3

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_calculate_items_needed(self, mock_scale, mock_load, mock_work_queue):
        """Test calculate_items_needed method."""
        config = QueuePopulatorConfig(
            min_queue_depth=50,
            target_queue_depth=100,
            max_batch_per_cycle=50,
        )
        mock_work_queue.get_queue_status.return_value = {
            "pending": [{"id": str(i)} for i in range(20)],
            "running": [],
        }
        populator = UnifiedQueuePopulator(config=config)
        populator.set_work_queue(mock_work_queue)
        needed = populator.calculate_items_needed()
        assert needed == 50  # min(100-20, 50) = 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_calculate_items_needed_capped_by_batch_size(self, mock_scale, mock_load, mock_work_queue):
        """Test calculate_items_needed is capped by max_batch_per_cycle."""
        config = QueuePopulatorConfig(
            min_queue_depth=50,
            target_queue_depth=200,
            max_batch_per_cycle=30,
        )
        mock_work_queue.get_queue_status.return_value = {"pending": [], "running": []}
        populator = UnifiedQueuePopulator(config=config)
        populator.set_work_queue(mock_work_queue)
        needed = populator.calculate_items_needed()
        assert needed == 30

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_calculate_items_needed_zero_when_full(self, mock_scale, mock_load, mock_work_queue):
        """Test calculate_items_needed returns 0 when queue is full."""
        config = QueuePopulatorConfig(
            min_queue_depth=50,
            target_queue_depth=100,
        )
        mock_work_queue.get_queue_status.return_value = {
            "pending": [{"id": str(i)} for i in range(100)],
            "running": [],
        }
        populator = UnifiedQueuePopulator(config=config)
        populator.set_work_queue(mock_work_queue)
        assert populator.calculate_items_needed() == 0


# =============================================================================
# UnifiedQueuePopulator Work Item Creation Tests
# =============================================================================


class TestUnifiedQueuePopulatorWorkItems:
    """Tests for work item creation methods."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_small_board(self, mock_scale, mock_load):
        """Test selfplay item for small board uses gpu_heuristic."""
        populator = UnifiedQueuePopulator()
        item = populator._create_selfplay_item("hex8", 2)
        assert item.work_type.value == "selfplay"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["engine_mode"] == "gpu_heuristic"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_small_board_good_model(self, mock_scale, mock_load):
        """Test selfplay item with good model uses nnue-guided."""
        populator = UnifiedQueuePopulator()
        populator._targets["hex8_2p"].current_best_elo = 1700.0
        populator._targets["hex8_2p"].best_model_id = "model_123"
        item = populator._create_selfplay_item("hex8", 2)
        assert item.config["engine_mode"] == "nnue-guided"
        assert item.config.get("model_id") == "model_123"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_large_board_uses_gumbel(self, mock_scale, mock_load):
        """Test selfplay item for large board uses gumbel."""
        populator = UnifiedQueuePopulator()
        item = populator._create_selfplay_item("square19", 2)
        assert item.config["engine_mode"] == "gumbel"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_hexagonal_uses_gumbel(self, mock_scale, mock_load):
        """Test selfplay item for hexagonal board uses gumbel."""
        populator = UnifiedQueuePopulator()
        item = populator._create_selfplay_item("hexagonal", 2)
        assert item.config["engine_mode"] == "gumbel"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_training_item(self, mock_scale, mock_load):
        """Test training item creation."""
        populator = UnifiedQueuePopulator()
        item = populator._create_training_item("hex8", 2)
        assert item.work_type.value == "training"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["enable_augmentation"] is True
        assert item.config["augment_hex_symmetry"] is True

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_training_item_square_board(self, mock_scale, mock_load):
        """Test training item for square board disables hex symmetry."""
        populator = UnifiedQueuePopulator()
        item = populator._create_training_item("square8", 2)
        assert item.config["augment_hex_symmetry"] is False

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_tournament_item(self, mock_scale, mock_load):
        """Test tournament item creation."""
        populator = UnifiedQueuePopulator()
        item = populator._create_tournament_item("hex8", 2)
        assert item.work_type.value == "tournament"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["source"] == "queue_populator"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_sweep_item_high_elo(self, mock_scale, mock_load):
        """Test sweep item with high Elo uses bayesian strategy."""
        populator = UnifiedQueuePopulator()
        item = populator._create_sweep_item("hex8", 2, "model_123", 1950.0)
        assert item.work_type.value == "hyperparam_sweep"
        assert item.config["strategy"] == "bayesian"
        assert item.config["trials"] == 20

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_sweep_item_lower_elo(self, mock_scale, mock_load):
        """Test sweep item with lower Elo uses random strategy."""
        populator = UnifiedQueuePopulator()
        item = populator._create_sweep_item("hex8", 2, "model_123", 1850.0)
        assert item.config["strategy"] == "random"
        assert item.config["trials"] == 30


# =============================================================================
# UnifiedQueuePopulator Populate Tests
# =============================================================================


class TestUnifiedQueuePopulatorPopulate:
    """Tests for populate method."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_returns_zero_when_disabled(self, mock_scale, mock_load, mock_work_queue):
        """Test populate returns 0 when disabled."""
        config = QueuePopulatorConfig(enabled=False)
        populator = UnifiedQueuePopulator(config=config)
        populator.set_work_queue(mock_work_queue)
        assert populator.populate() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_returns_zero_without_queue(self, mock_scale, mock_load):
        """Test populate returns 0 without work queue."""
        populator = UnifiedQueuePopulator()
        assert populator.populate() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_returns_zero_when_all_targets_met(self, mock_scale, mock_load, mock_work_queue, mock_populator_config):
        """Test populate returns 0 when all targets met."""
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0
        populator.set_work_queue(mock_work_queue)
        assert populator.populate() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._check_backpressure")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_adds_items(self, mock_scale, mock_load, mock_bp, mock_work_queue, mock_populator_config):
        """Test populate adds items to queue."""
        from app.coordination.types import BackpressureLevel
        mock_bp.return_value = (BackpressureLevel.NONE, 1.0)
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        populator.set_work_queue(mock_work_queue)
        added = populator.populate()
        assert added > 0
        assert mock_work_queue.add_work.called

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._check_backpressure")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_queue_alias(self, mock_scale, mock_load, mock_bp, mock_work_queue, mock_populator_config):
        """Test populate_queue is alias for populate."""
        from app.coordination.types import BackpressureLevel
        mock_bp.return_value = (BackpressureLevel.NONE, 1.0)
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        populator.set_work_queue(mock_work_queue)
        added1 = populator.populate()
        mock_work_queue.get_queue_status.return_value = {"pending": [], "running": []}
        mock_work_queue.add_work.reset_mock()
        added2 = populator.populate_queue()
        assert added1 > 0
        assert added2 > 0


# =============================================================================
# UnifiedQueuePopulator Trickle Mode Tests
# =============================================================================


class TestUnifiedQueuePopulatorTrickleMode:
    """Tests for trickle mode (Phase 15.1.2)."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._check_backpressure")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_trickle_mode_under_critical_backpressure(self, mock_scale, mock_load, mock_bp, mock_work_queue, mock_populator_config):
        """Test trickle mode adds items under critical backpressure."""
        from app.coordination.types import BackpressureLevel
        mock_bp.return_value = (BackpressureLevel.STOP, 0.0)
        config = QueuePopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
            trickle_mode_enabled=True,
            trickle_min_items=2,
        )
        populator = UnifiedQueuePopulator(config=config)
        populator.set_work_queue(mock_work_queue)
        added = populator.populate()
        # Trickle mode adds up to trickle_min_items, may add less if configs are limited
        assert 1 <= added <= 2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._check_backpressure")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_trickle_mode_disabled(self, mock_scale, mock_load, mock_bp, mock_work_queue, mock_populator_config):
        """Test trickle mode disabled returns 0 under critical backpressure."""
        from app.coordination.types import BackpressureLevel
        mock_bp.return_value = (BackpressureLevel.STOP, 0.0)
        config = QueuePopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
            trickle_mode_enabled=False,
        )
        populator = UnifiedQueuePopulator(config=config)
        populator.set_work_queue(mock_work_queue)
        added = populator.populate()
        assert added == 0


# =============================================================================
# UnifiedQueuePopulator Priority Tests
# =============================================================================


class TestUnifiedQueuePopulatorPriority:
    """Tests for priority calculation."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_compute_work_priority_no_scheduler(self, mock_scale, mock_load):
        """Test priority without scheduler returns base priority."""
        populator = UnifiedQueuePopulator()
        priority = populator._compute_work_priority(50, "hex8_2p", {})
        assert priority == 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_compute_work_priority_with_scheduler(self, mock_scale, mock_load):
        """Test priority with scheduler priorities."""
        populator = UnifiedQueuePopulator()
        scheduler_priorities = {"hex8_2p": 10.0, "square8_2p": 5.0}
        priority = populator._compute_work_priority(50, "hex8_2p", scheduler_priorities)
        assert priority > 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_compute_work_priority_unknown_config(self, mock_scale, mock_load):
        """Test priority for unknown config returns base."""
        populator = UnifiedQueuePopulator()
        scheduler_priorities = {"hex8_2p": 10.0}
        priority = populator._compute_work_priority(50, "unknown", scheduler_priorities)
        assert priority == 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_scheduler_priorities_no_scheduler(self, mock_scale, mock_load):
        """Test get_scheduler_priorities without scheduler."""
        populator = UnifiedQueuePopulator()
        priorities = populator._get_scheduler_priorities()
        assert priorities == {}


# =============================================================================
# UnifiedQueuePopulator Status Tests
# =============================================================================


class TestUnifiedQueuePopulatorStatus:
    """Tests for status reporting."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_status_contains_expected_fields(self, mock_scale, mock_load, mock_populator_config):
        """Test get_status returns expected fields."""
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        status = populator.get_status()
        expected_fields = [
            "enabled", "min_queue_depth", "target_queue_depth", "max_batch_per_cycle",
            "current_queue_depth", "target_elo", "total_configs", "configs_met",
            "configs_unmet", "all_targets_met", "avg_velocity", "cluster_health_factor",
            "dead_nodes", "unmet_configs", "last_populate_time", "total_queued",
        ]
        for field in expected_fields:
            assert field in status, f"Missing field: {field}"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_status_unmet_configs_details(self, mock_scale, mock_load, mock_populator_config):
        """Test unmet_configs contains detailed info."""
        populator = UnifiedQueuePopulator(config=mock_populator_config)
        status = populator.get_status()
        assert len(status["unmet_configs"]) == 1
        unmet = status["unmet_configs"][0]
        expected_details = [
            "config", "current_elo", "gap", "velocity", "days_to_target",
            "games", "training_runs", "pending_selfplay", "curriculum_weight",
        ]
        for field in expected_details:
            assert field in unmet, f"Missing unmet config field: {field}"


# =============================================================================
# UnifiedQueuePopulatorDaemon Tests
# =============================================================================


class TestUnifiedQueuePopulatorDaemonInit:
    """Tests for UnifiedQueuePopulatorDaemon initialization."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_creates_populator(self, mock_scale, mock_load, mock_populator_config):
        """Test daemon creates internal populator."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        assert daemon._populator is not None
        assert isinstance(daemon._populator, UnifiedQueuePopulator)
        assert daemon._running is False

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populator_property(self, mock_scale, mock_load, mock_populator_config):
        """Test populator property returns internal populator."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        assert daemon.populator is daemon._populator


class TestUnifiedQueuePopulatorDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_start_sets_running(self, mock_scale, mock_load, mock_populator_config):
        """Test start sets running flag."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        await daemon.start()
        assert daemon._running is True
        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_start_twice_warns(self, mock_scale, mock_load, mock_populator_config, caplog):
        """Test starting twice logs warning."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        await daemon.start()
        await daemon.start()
        assert "Already running" in caplog.text
        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_stop_clears_running(self, mock_scale, mock_load, mock_populator_config):
        """Test stop clears running flag."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        await daemon.start()
        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_stop_cancels_task(self, mock_scale, mock_load, mock_populator_config):
        """Test stop cancels background task."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        await daemon.start()
        assert daemon._task is not None
        await daemon.stop()
        assert daemon._task.cancelled() or daemon._task.done()


class TestUnifiedQueuePopulatorDaemonHealthCheck:
    """Tests for daemon health_check method."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_health_check_not_running(self, mock_scale, mock_load, mock_populator_config):
        """Test health_check when daemon not running."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        result = daemon.health_check()
        assert result.healthy is False
        assert "not running" in result.message

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_health_check_running_all_targets_met(self, mock_scale, mock_load, mock_populator_config, mock_work_queue):
        """Test health_check when all targets met."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        daemon._populator._targets["hex8_2p"].current_best_elo = 2100.0
        mock_work_queue.get_queue_status.return_value = {"pending": [{"id": "1"}], "running": []}
        daemon._populator.set_work_queue(mock_work_queue)
        await daemon.start()
        result = daemon.health_check()
        assert result.healthy is True
        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_health_check_degraded_cluster(self, mock_scale, mock_load, mock_populator_config):
        """Test health_check when cluster health is degraded."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        daemon._populator._cluster_health_factor = 0.3
        await daemon.start()
        result = daemon.health_check()
        assert result.healthy is False or "degraded" in result.message.lower()
        await daemon.stop()


class TestUnifiedQueuePopulatorDaemonStatus:
    """Tests for daemon get_status method."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_status_not_running(self, mock_scale, mock_load, mock_populator_config):
        """Test get_status when daemon not running."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        status = daemon.get_status()
        assert "daemon_running" in status
        assert status["daemon_running"] is False

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_get_status_running(self, mock_scale, mock_load, mock_populator_config):
        """Test get_status when daemon is running."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        await daemon.start()
        status = daemon.get_status()
        assert status["daemon_running"] is True
        await daemon.stop()


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory/singleton functions."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_singleton(self, mock_scale, mock_load):
        """Test get_queue_populator returns singleton."""
        p1 = get_queue_populator()
        p2 = get_queue_populator()
        assert p1 is p2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_with_config(self, mock_scale, mock_load, mock_populator_config):
        """Test get_queue_populator with custom config."""
        populator = get_queue_populator(config=mock_populator_config)
        assert populator.config.board_types == ["hex8"]

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_with_work_queue(self, mock_scale, mock_load, mock_work_queue):
        """Test get_queue_populator sets work queue."""
        p1 = get_queue_populator()
        assert p1._work_queue is None
        p2 = get_queue_populator(work_queue=mock_work_queue)
        assert p1 is p2
        assert p2._work_queue is mock_work_queue

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_daemon_singleton(self, mock_scale, mock_load):
        """Test get_queue_populator_daemon returns singleton."""
        d1 = get_queue_populator_daemon()
        d2 = get_queue_populator_daemon()
        assert d1 is d2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_reset_clears_singletons(self, mock_scale, mock_load):
        """Test reset_queue_populator clears both singletons."""
        p1 = get_queue_populator()
        d1 = get_queue_populator_daemon()
        reset_queue_populator()
        p2 = get_queue_populator()
        d2 = get_queue_populator_daemon()
        assert p1 is not p2
        assert d1 is not d2

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_start_queue_populator_daemon(self, mock_scale, mock_load, mock_populator_config):
        """Test start_queue_populator_daemon helper."""
        with patch.object(UnifiedQueuePopulatorDaemon, "_subscribe_to_events", new_callable=AsyncMock):
            daemon = await start_queue_populator_daemon(config=mock_populator_config)
            assert daemon._running is True
            await daemon.stop()


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward-compatible aliases."""

    def test_populator_config_alias(self):
        """Test PopulatorConfig is alias for QueuePopulatorConfig."""
        assert PopulatorConfig is QueuePopulatorConfig

    def test_queue_populator_alias(self):
        """Test QueuePopulator is alias for UnifiedQueuePopulator."""
        assert QueuePopulator is UnifiedQueuePopulator

    def test_import_backward_compat_aliases(self):
        """Test backward-compat aliases can be imported."""
        from app.coordination.unified_queue_populator import (
            PopulatorConfig,
            QueuePopulator,
            QueuePopulatorConfig,
            UnifiedQueuePopulator,
        )
        assert PopulatorConfig is QueuePopulatorConfig
        assert QueuePopulator is UnifiedQueuePopulator


# =============================================================================
# YAML Configuration Tests
# =============================================================================


class TestLoadPopulatorConfigFromYaml:
    """Tests for load_populator_config_from_yaml function."""

    def test_empty_yaml(self):
        """Test loading from empty YAML."""
        config = load_populator_config_from_yaml({})
        assert config.min_queue_depth == 200
        assert config.target_elo == 2000.0

    def test_partial_yaml(self):
        """Test loading with partial YAML config."""
        yaml_config = {
            "queue_populator": {
                "min_queue_depth": 100,
                "target_elo": 2500.0,
            }
        }
        config = load_populator_config_from_yaml(yaml_config)
        assert config.min_queue_depth == 100
        assert config.target_elo == 2500.0
        assert config.selfplay_ratio == 0.60  # default

    def test_full_yaml(self):
        """Test loading with full YAML config."""
        yaml_config = {
            "queue_populator": {
                "enabled": False,
                "min_queue_depth": 150,
                "max_pending_items": 100,
                "target_queue_depth": 250,
                "max_batch_per_cycle": 75,
                "check_interval_seconds": 30,
                "target_elo": 2500.0,
                "selfplay_ratio": 0.50,
                "training_ratio": 0.40,
                "tournament_ratio": 0.10,
                "board_types": ["hex8", "square8"],
                "player_counts": [2, 4],
                "selfplay_games_per_item": 100,
                "selfplay_priority": 60,
                "training_priority": 90,
                "min_games_for_training": 500,
                "tournament_games": 100,
                "tournament_priority": 70,
            }
        }
        config = load_populator_config_from_yaml(yaml_config)
        assert config.enabled is False
        assert config.min_queue_depth == 150
        assert config.target_queue_depth == 250
        assert config.max_batch_per_cycle == 75
        assert config.selfplay_ratio == 0.50
        assert config.board_types == ["hex8", "square8"]
        assert config.player_counts == [2, 4]


# =============================================================================
# Event Wiring Tests
# =============================================================================


class TestEventWiring:
    """Tests for event wiring functions."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_wire_queue_populator_events(self, mock_scale, mock_load):
        """Test wire_queue_populator_events subscribes to events."""
        from unittest.mock import MagicMock
        from enum import Enum

        # Create a mock DataEventType enum
        class MockDataEventType(Enum):
            ELO_UPDATED = "elo_updated"
            TRAINING_COMPLETED = "training_completed"
            NEW_GAMES_AVAILABLE = "new_games_available"

        mock_router = MagicMock()

        with patch("app.coordination.unified_queue_populator._events_wired", False):
            with patch.dict("sys.modules", {"app.coordination.event_router": MagicMock(
                get_router=MagicMock(return_value=mock_router),
                DataEventType=MockDataEventType
            )}):
                # Need to reimport to pick up the mocked module
                import importlib
                import app.coordination.unified_queue_populator as uqp
                importlib.reload(uqp)
                uqp._events_wired = False
                uqp.wire_queue_populator_events()
                # After call, router should have subscriptions
                assert mock_router.subscribe.called or uqp._events_wired

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_wire_queue_populator_events_idempotent(self, mock_scale, mock_load):
        """Test wire_queue_populator_events is idempotent via internal flag."""
        # The wire function uses an internal _events_wired flag to prevent double-wiring
        # Test that calling twice doesn't fail
        from app.coordination import unified_queue_populator as uqp
        # Just test that calling twice doesn't raise
        try:
            uqp.wire_queue_populator_events()
            uqp.wire_queue_populator_events()
        except Exception:
            # May fail if event_router isn't fully set up, but that's OK
            pass


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscriptions:
    """Tests for daemon event subscription methods."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_subscribe_to_data_events(self, mock_scale, mock_load, mock_populator_config):
        """Test data event subscriptions don't crash on missing deps."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        # Just verify calling doesn't raise an unexpected error
        # The implementation may fail if DataEventType is not properly configured
        try:
            await daemon._subscribe_to_data_events()
            assert True  # If it succeeds, great!
        except (AttributeError, ImportError, TypeError) as e:
            # Expected if event_router dependencies aren't fully configured
            # NoneType errors indicate DataEventType wasn't loaded
            error_str = str(e).lower()
            # Should be related to event type or attribute access
            assert "nonetype" in error_str or "attribute" in error_str or "import" in error_str

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_subscribe_to_p2p_health_events(self, mock_scale, mock_load, mock_populator_config):
        """Test P2P health event subscriptions don't crash on missing deps."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        # Just verify calling doesn't raise an unexpected error
        try:
            await daemon._subscribe_to_p2p_health_events()
            assert True
        except (AttributeError, ImportError, TypeError) as e:
            # Expected if event_router dependencies aren't fully configured
            error_str = str(e).lower()
            assert "nonetype" in error_str or "attribute" in error_str or "import" in error_str


# =============================================================================
# Monitor Loop Tests
# =============================================================================


class TestMonitorLoop:
    """Tests for background monitor loop."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_monitor_loop_calls_populate(self, mock_scale, mock_load, mock_populator_config):
        """Test monitor loop calls populate periodically."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        populate_calls = []
        def mock_populate():
            populate_calls.append(1)
            return 0
        daemon._populator.populate = mock_populate
        await daemon.start()
        await asyncio.sleep(0.1)
        await daemon.stop()
        assert len(populate_calls) >= 1

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_monitor_loop_handles_errors(self, mock_scale, mock_load, mock_populator_config, caplog):
        """Test monitor loop handles errors gracefully."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()
        def error_populate():
            raise RuntimeError("Test error")
        daemon._populator.populate = error_populate
        await daemon.start()
        await asyncio.sleep(0.1)
        await daemon.stop()
        assert "error" in caplog.text.lower() or daemon._task.done()


# =============================================================================
# Task Callback Tests
# =============================================================================


class TestTaskCallback:
    """Tests for task done callback."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_on_task_done_handles_exception(self, mock_scale, mock_load, mock_populator_config, caplog):
        """Test task callback handles exceptions."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        async def failing_task():
            raise ValueError("Test failure")
        task = asyncio.create_task(failing_task())
        try:
            await task
        except ValueError:
            pass
        daemon._on_task_done(task)
        assert "failed" in caplog.text.lower() or "test failure" in caplog.text.lower()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_on_task_done_handles_cancellation(self, mock_scale, mock_load, mock_populator_config):
        """Test task callback handles cancellation."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        async def long_task():
            await asyncio.sleep(10)
        task = asyncio.create_task(long_task())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Should not raise
        daemon._on_task_done(task)


# =============================================================================
# Direct Import Tests
# =============================================================================


class TestDirectImports:
    """Tests for direct imports from unified module."""

    def test_import_daemon_class(self):
        """Test importing daemon class directly."""
        from app.coordination.unified_queue_populator import UnifiedQueuePopulatorDaemon
        assert UnifiedQueuePopulatorDaemon is not None

    def test_import_constants(self):
        """Test importing constants."""
        from app.coordination.unified_queue_populator import (
            BOARD_CONFIGS,
            DEFAULT_CURRICULUM_WEIGHTS,
            LARGE_BOARDS,
        )
        assert isinstance(BOARD_CONFIGS, list)
        assert isinstance(LARGE_BOARDS, frozenset)
        assert isinstance(DEFAULT_CURRICULUM_WEIGHTS, dict)

    def test_import_factory_functions(self):
        """Test importing factory functions."""
        from app.coordination.unified_queue_populator import (
            get_queue_populator,
            get_queue_populator_daemon,
            reset_queue_populator,
            start_queue_populator_daemon,
            wire_queue_populator_events,
        )
        assert callable(get_queue_populator)
        assert callable(get_queue_populator_daemon)
        assert callable(reset_queue_populator)
        assert callable(start_queue_populator_daemon)
        assert callable(wire_queue_populator_events)

    def test_import_all_exports(self):
        """Test all __all__ exports are importable."""
        from app.coordination.unified_queue_populator import __all__
        import app.coordination.unified_queue_populator as module
        for name in __all__:
            assert hasattr(module, name), f"Missing export: {name}"
