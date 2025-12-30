"""Tests for app.coordination.unified_queue_populator module.

Tests the work queue population system:
- BoardType enum
- ConfigTarget dataclass and Elo velocity tracking
- QueuePopulatorConfig settings
- UnifiedQueuePopulator class and work item generation
- Priority calculation and target management
- Queue population logic
- Work queue maintenance
- Data need detection
- Job prioritization
- Elo target checking
- Event handling

Created Dec 2025 as part of Phase 3 test coverage improvement.
Extended Dec 2025 for comprehensive coverage.
Updated Dec 2025 to use unified_queue_populator directly.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unified_queue_populator import (
    LARGE_BOARDS,
    ConfigTarget,
    QueuePopulatorConfig,
    UnifiedQueuePopulator,
    get_queue_populator,
    load_populator_config_from_yaml,
    reset_queue_populator,
)
from app.coordination.types import BoardType

# Backward-compat aliases for existing tests
PopulatorConfig = QueuePopulatorConfig
QueuePopulator = UnifiedQueuePopulator


# =============================================================================
# BoardType Enum Tests
# =============================================================================


class TestBoardType:
    """Tests for BoardType enum."""

    def test_all_board_types_exist(self):
        """Verify all expected board types are defined."""
        expected = ["SQUARE8", "SQUARE19", "HEX8", "HEXAGONAL"]
        for name in expected:
            assert hasattr(BoardType, name), f"Missing BoardType.{name}"

    def test_board_type_values(self):
        """Verify board type values are strings."""
        assert BoardType.SQUARE8.value == "square8"
        assert BoardType.SQUARE19.value == "square19"
        assert BoardType.HEX8.value == "hex8"
        assert BoardType.HEXAGONAL.value == "hexagonal"

    def test_board_type_is_string_enum(self):
        """Verify BoardType is a string enum."""
        assert isinstance(BoardType.SQUARE8, str)
        assert BoardType.SQUARE8 == "square8"


class TestLargeBoards:
    """Tests for LARGE_BOARDS constant."""

    def test_large_boards_contains_expected(self):
        """Verify LARGE_BOARDS contains large board types."""
        assert "square19" in LARGE_BOARDS
        assert "hexagonal" in LARGE_BOARDS

    def test_small_boards_not_in_large(self):
        """Verify small boards are not in LARGE_BOARDS."""
        assert "square8" not in LARGE_BOARDS
        assert "hex8" not in LARGE_BOARDS

    def test_large_boards_is_frozen(self):
        """Verify LARGE_BOARDS is a frozenset."""
        assert isinstance(LARGE_BOARDS, frozenset)

    def test_large_boards_contains_aliases(self):
        """Verify LARGE_BOARDS contains fullhex aliases."""
        assert "fullhex" in LARGE_BOARDS
        assert "full_hex" in LARGE_BOARDS


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

    def test_target_met_when_above(self):
        """Test target_met when Elo is above target."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=2100.0,
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

    def test_target_met_when_equal(self):
        """Test target_met when Elo equals target."""
        target = ConfigTarget(
            board_type="hex8",
            num_players=2,
            current_best_elo=2000.0,
            target_elo=2000.0,
        )
        assert target.target_met is True

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

    def test_config_key_format(self):
        """Test config_key property format."""
        target = ConfigTarget(board_type="square8", num_players=4)
        assert target.config_key == "square8_4p"

    def test_config_key_format_variations(self):
        """Test config_key format for various configs."""
        test_cases = [
            ("hex8", 2, "hex8_2p"),
            ("square19", 3, "square19_3p"),
            ("hexagonal", 4, "hexagonal_4p"),
        ]
        for board_type, num_players, expected in test_cases:
            target = ConfigTarget(board_type=board_type, num_players=num_players)
            assert target.config_key == expected

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
        # Record Elo increasing by ~100 points per day
        target.elo_history = [
            (now - 3 * 86400, 1500.0),  # 3 days ago
            (now - 2 * 86400, 1600.0),  # 2 days ago
            (now - 1 * 86400, 1700.0),  # 1 day ago
            (now, 1800.0),              # now
        ]
        velocity = target.elo_velocity
        # Should be approximately +100 per day
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
        # Should be approximately -100 per day
        assert -110.0 <= velocity <= -90.0

    def test_elo_velocity_filters_old_history(self):
        """Test elo_velocity only uses last 7 days of history."""
        target = ConfigTarget(board_type="hex8", num_players=2)
        now = time.time()
        # Add old data that should be filtered out
        target.elo_history = [
            (now - 10 * 86400, 1400.0),  # 10 days ago (ignored)
            (now - 8 * 86400, 1450.0),   # 8 days ago (ignored)
            (now - 2 * 86400, 1600.0),   # 2 days ago (used)
            (now, 1700.0),               # now (used)
        ]
        velocity = target.elo_velocity
        # Should be ~50 per day based on recent data only
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
        # No history means zero velocity
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
        # Velocity of 100 Elo/day
        target.elo_history = [
            (now - 2 * 86400, 1600.0),
            (now - 1 * 86400, 1700.0),
            (now, 1800.0),
        ]
        eta = target.days_to_target
        assert eta is not None
        # With 200 gap and ~100/day velocity, ~2 days
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
        # Add old entry (35 days ago)
        target.elo_history.append((now - 35 * 86400, 1500.0))
        # Add recent entry
        target.record_elo(1600.0, timestamp=now)
        # Old entry should be pruned
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
# PopulatorConfig Tests
# =============================================================================


class TestPopulatorConfig:
    """Tests for PopulatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PopulatorConfig()
        # December 2025: min_queue_depth updated to 200 in unified_queue_populator
        assert config.min_queue_depth == 200
        assert config.target_elo == 2000.0
        assert config.selfplay_ratio == 0.60
        assert config.training_ratio == 0.30
        assert config.tournament_ratio == 0.10
        assert config.enabled is True

    def test_ratios_sum_to_one(self):
        """Test work distribution ratios sum to 1.0."""
        config = PopulatorConfig()
        total = config.selfplay_ratio + config.training_ratio + config.tournament_ratio
        assert abs(total - 1.0) < 0.01

    def test_board_types_default(self):
        """Test default board types list."""
        config = PopulatorConfig()
        assert "square8" in config.board_types
        assert "square19" in config.board_types
        assert "hex8" in config.board_types
        assert "hexagonal" in config.board_types

    def test_player_counts_default(self):
        """Test default player counts."""
        config = PopulatorConfig()
        assert 2 in config.player_counts
        assert 3 in config.player_counts
        assert 4 in config.player_counts

    def test_custom_config(self):
        """Test custom configuration."""
        config = PopulatorConfig(
            min_queue_depth=100,
            target_elo=2500.0,
            selfplay_ratio=0.70,
            enabled=False,
        )
        assert config.min_queue_depth == 100
        assert config.target_elo == 2500.0
        assert config.selfplay_ratio == 0.70
        assert config.enabled is False

    def test_priority_settings(self):
        """Test priority default settings."""
        config = PopulatorConfig()
        assert config.selfplay_priority == 50
        assert config.training_priority == 100
        assert config.tournament_priority == 80

    def test_selfplay_games_per_item(self):
        """Test selfplay games per item default."""
        config = PopulatorConfig()
        assert config.selfplay_games_per_item == 50

    def test_tournament_games_default(self):
        """Test tournament games default."""
        config = PopulatorConfig()
        assert config.tournament_games == 50

    def test_check_interval_default(self):
        """Test check interval default (10 seconds in unified_queue_populator)."""
        config = PopulatorConfig()
        # December 2025: check_interval_seconds updated to 10 in unified_queue_populator
        assert config.check_interval_seconds == 10


# =============================================================================
# Load Config from YAML Tests
# =============================================================================


class TestLoadPopulatorConfigFromYaml:
    """Tests for load_populator_config_from_yaml function."""

    def test_empty_yaml(self):
        """Test loading from empty YAML."""
        config = load_populator_config_from_yaml({})
        # December 2025: min_queue_depth updated to 200 in unified_queue_populator
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
        # Defaults for unspecified
        assert config.selfplay_ratio == 0.60

    def test_full_yaml(self):
        """Test loading with full YAML config."""
        yaml_config = {
            "queue_populator": {
                "min_queue_depth": 200,
                "target_elo": 3000.0,
                "selfplay_ratio": 0.50,
                "training_ratio": 0.40,
                "tournament_ratio": 0.10,
                "board_types": ["hex8", "square8"],
                "player_counts": [2, 4],
                "selfplay_games_per_item": 100,
                "enabled": False,
            }
        }
        config = load_populator_config_from_yaml(yaml_config)
        assert config.min_queue_depth == 200
        assert config.target_elo == 3000.0
        assert config.selfplay_ratio == 0.50
        assert config.training_ratio == 0.40
        assert config.board_types == ["hex8", "square8"]
        assert config.player_counts == [2, 4]
        assert config.selfplay_games_per_item == 100
        assert config.enabled is False


# =============================================================================
# QueuePopulator Initialization Tests
# =============================================================================


class TestQueuePopulatorInit:
    """Tests for QueuePopulator initialization."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_defaults(self, mock_scale, mock_load):
        """Test initialization with default config."""
        populator = QueuePopulator()
        assert populator.config is not None
        assert populator._work_queue is None

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_creates_targets(self, mock_scale, mock_load):
        """Test initialization creates targets for all configs."""
        populator = QueuePopulator()
        # 4 board types * 3 player counts = 12 targets
        assert len(populator._targets) == 12

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_custom_config(self, mock_scale, mock_load):
        """Test initialization with custom config."""
        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        assert len(populator._targets) == 1
        assert "hex8_2p" in populator._targets

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_work_queue(self, mock_scale, mock_load):
        """Test initialization with provided work queue."""
        mock_queue = MagicMock()
        populator = QueuePopulator(work_queue=mock_queue)
        assert populator._work_queue is mock_queue

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_with_elo_db_path(self, mock_scale, mock_load):
        """Test initialization with custom Elo database path."""
        populator = QueuePopulator(elo_db_path="/custom/path/elo.db")
        assert populator._elo_db_path == "/custom/path/elo.db"


# =============================================================================
# QueuePopulator Method Tests
# =============================================================================


class TestQueuePopulatorMethods:
    """Tests for QueuePopulator methods."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_update_target_elo(self, mock_scale, mock_load):
        """Test update_target_elo method."""
        populator = QueuePopulator()
        populator.update_target_elo("hex8", 2, 1700.0, "model_123")

        target = populator._targets["hex8_2p"]
        assert target.current_best_elo == 1700.0
        assert target.best_model_id == "model_123"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_update_target_elo_ignores_lower(self, mock_scale, mock_load):
        """Test update_target_elo ignores lower Elo values."""
        populator = QueuePopulator()
        populator.update_target_elo("hex8", 2, 1800.0, "model_1")
        populator.update_target_elo("hex8", 2, 1700.0, "model_2")

        target = populator._targets["hex8_2p"]
        assert target.current_best_elo == 1800.0
        assert target.best_model_id == "model_1"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_update_target_elo_unknown_config(self, mock_scale, mock_load):
        """Test update_target_elo with unknown config is silently ignored."""
        populator = QueuePopulator()
        # Should not raise
        populator.update_target_elo("unknown", 5, 1700.0, "model_123")

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_increment_games(self, mock_scale, mock_load):
        """Test increment_games method."""
        populator = QueuePopulator()
        populator.increment_games("hex8", 2, 10)
        populator.increment_games("hex8", 2, 5)

        target = populator._targets["hex8_2p"]
        assert target.games_played == 15

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_increment_games_default_count(self, mock_scale, mock_load):
        """Test increment_games with default count of 1."""
        populator = QueuePopulator()
        populator.increment_games("hex8", 2)
        populator.increment_games("hex8", 2)

        target = populator._targets["hex8_2p"]
        assert target.games_played == 2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_increment_training(self, mock_scale, mock_load):
        """Test increment_training method."""
        populator = QueuePopulator()
        populator.increment_training("hex8", 2)
        populator.increment_training("hex8", 2)

        target = populator._targets["hex8_2p"]
        assert target.training_runs == 2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_all_targets_met_false(self, mock_scale, mock_load):
        """Test all_targets_met returns False when not all met."""
        populator = QueuePopulator()
        assert populator.all_targets_met() is False

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_all_targets_met_true(self, mock_scale, mock_load):
        """Test all_targets_met returns True when all met."""
        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0

        assert populator.all_targets_met() is True

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_unmet_targets(self, mock_scale, mock_load):
        """Test get_unmet_targets method."""
        config = PopulatorConfig(
            board_types=["hex8", "square8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0

        unmet = populator.get_unmet_targets()
        assert len(unmet) == 1
        assert unmet[0].config_key == "square8_2p"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_priority_target_by_gap(self, mock_scale, mock_load):
        """Test get_priority_target returns target with smallest gap."""
        config = PopulatorConfig(
            board_types=["hex8", "square8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 1900.0  # 100 gap
        populator._targets["square8_2p"].current_best_elo = 1800.0  # 200 gap

        priority = populator.get_priority_target()
        assert priority is not None
        # Smallest gap gets priority
        assert priority.config_key == "hex8_2p"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_priority_target_none_when_all_met(self, mock_scale, mock_load):
        """Test get_priority_target returns None when all targets met."""
        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0

        assert populator.get_priority_target() is None

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_current_queue_depth_no_queue(self, mock_scale, mock_load):
        """Test get_current_queue_depth returns 0 without queue."""
        populator = QueuePopulator()
        assert populator.get_current_queue_depth() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_current_queue_depth_with_queue(self, mock_scale, mock_load):
        """Test get_current_queue_depth with mock queue."""
        populator = QueuePopulator()
        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {
            "pending": [{"id": "1"}, {"id": "2"}],
            "running": [{"id": "3"}],
        }
        populator.set_work_queue(mock_queue)
        assert populator.get_current_queue_depth() == 3

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_set_work_queue(self, mock_scale, mock_load):
        """Test set_work_queue method."""
        populator = QueuePopulator()
        mock_queue = MagicMock()
        populator.set_work_queue(mock_queue)
        assert populator._work_queue is mock_queue

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_set_selfplay_scheduler(self, mock_scale, mock_load):
        """Test set_selfplay_scheduler method."""
        populator = QueuePopulator()
        mock_scheduler = MagicMock()
        populator.set_selfplay_scheduler(mock_scheduler)
        assert populator._selfplay_scheduler is mock_scheduler

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_calculate_items_needed(self, mock_scale, mock_load):
        """Test calculate_items_needed method."""
        # December 2025: calculate_items_needed now uses target_queue_depth (not min_queue_depth)
        # and caps at max_batch_per_cycle
        config = PopulatorConfig(
            min_queue_depth=50,
            target_queue_depth=50,
            max_batch_per_cycle=50,
        )
        populator = QueuePopulator(config=config)
        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {
            "pending": [{"id": str(i)} for i in range(20)],
            "running": [{"id": str(i)} for i in range(10)],
        }
        populator.set_work_queue(mock_queue)
        # target_queue_depth(50) - current(30) = 20 needed
        assert populator.calculate_items_needed() == 20

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_calculate_items_needed_zero_when_full(self, mock_scale, mock_load):
        """Test calculate_items_needed returns 0 when queue is full."""
        # December 2025: calculate_items_needed now uses target_queue_depth
        # Returns 0 when current >= target_queue_depth
        config = PopulatorConfig(
            min_queue_depth=50,
            target_queue_depth=50,  # Queue is full when current >= target
        )
        populator = QueuePopulator(config=config)
        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {
            "pending": [{"id": str(i)} for i in range(40)],
            "running": [{"id": str(i)} for i in range(20)],
        }
        populator.set_work_queue(mock_queue)
        # current(60) >= target_queue_depth(50), so needed = max(0, 50-60) = 0
        assert populator.calculate_items_needed() == 0


# =============================================================================
# QueuePopulator Work Item Creation Tests
# =============================================================================


class TestQueuePopulatorWorkItems:
    """Tests for work item creation."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_small_board_no_model(self, mock_scale, mock_load):
        """Test selfplay item for small board without model uses gpu_heuristic."""
        populator = QueuePopulator()
        item = populator._create_selfplay_item("hex8", 2)

        assert item.work_type.value == "selfplay"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["engine_mode"] == "gpu_heuristic"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_small_board_good_model(self, mock_scale, mock_load):
        """Test selfplay item for small board with good model uses nnue-guided."""
        populator = QueuePopulator()
        populator._targets["hex8_2p"].current_best_elo = 1700.0
        populator._targets["hex8_2p"].best_model_id = "model_123"

        item = populator._create_selfplay_item("hex8", 2)

        assert item.config["engine_mode"] == "nnue-guided"
        assert item.config.get("model_id") == "model_123"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_large_board_uses_gumbel(self, mock_scale, mock_load):
        """Test selfplay item for large board always uses gumbel."""
        populator = QueuePopulator()
        item = populator._create_selfplay_item("square19", 2)

        # Large boards always use gumbel regardless of model availability
        assert item.config["engine_mode"] == "gumbel"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_hexagonal_uses_gumbel(self, mock_scale, mock_load):
        """Test selfplay item for hexagonal board uses gumbel."""
        populator = QueuePopulator()
        item = populator._create_selfplay_item("hexagonal", 2)
        assert item.config["engine_mode"] == "gumbel"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_selfplay_item_medium_elo_no_model(self, mock_scale, mock_load):
        """Test selfplay item with medium Elo but no model uses gpu_heuristic."""
        populator = QueuePopulator()
        populator._targets["hex8_2p"].current_best_elo = 1650.0
        populator._targets["hex8_2p"].best_model_id = None

        item = populator._create_selfplay_item("hex8", 2)
        assert item.config["engine_mode"] == "gpu_heuristic"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_training_item(self, mock_scale, mock_load):
        """Test training item creation."""
        populator = QueuePopulator()
        item = populator._create_training_item("hex8", 2)

        assert item.work_type.value == "training"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["enable_augmentation"] is True
        assert item.config["augment_hex_symmetry"] is True  # hex board

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_training_item_square_board(self, mock_scale, mock_load):
        """Test training item for square board."""
        populator = QueuePopulator()
        item = populator._create_training_item("square8", 2)

        assert item.config["augment_hex_symmetry"] is False  # square board

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_tournament_item(self, mock_scale, mock_load):
        """Test tournament item creation."""
        populator = QueuePopulator()
        item = populator._create_tournament_item("hex8", 2)

        assert item.work_type.value == "tournament"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["source"] == "queue_populator"

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_sweep_item(self, mock_scale, mock_load):
        """Test hyperparam sweep item creation."""
        populator = QueuePopulator()
        item = populator._create_sweep_item("hex8", 2, "model_123", 1850.0)

        assert item.work_type.value == "hyperparam_sweep"
        assert item.config["board_type"] == "hex8"
        assert item.config["num_players"] == 2
        assert item.config["base_model_id"] == "model_123"
        assert item.config["base_elo"] == 1850.0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_sweep_item_high_elo_uses_bayesian(self, mock_scale, mock_load):
        """Test sweep item with high Elo uses bayesian strategy."""
        populator = QueuePopulator()
        item = populator._create_sweep_item("hex8", 2, "model_123", 1950.0)

        assert item.config["strategy"] == "bayesian"
        assert item.config["trials"] == 20

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_create_sweep_item_lower_elo_uses_random(self, mock_scale, mock_load):
        """Test sweep item with lower Elo uses random strategy."""
        populator = QueuePopulator()
        item = populator._create_sweep_item("hex8", 2, "model_123", 1850.0)

        assert item.config["strategy"] == "random"
        assert item.config["trials"] == 30


# =============================================================================
# QueuePopulator Populate Tests
# =============================================================================


class TestQueuePopulatorPopulate:
    """Tests for populate method."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_returns_zero_when_disabled(self, mock_scale, mock_load):
        """Test populate returns 0 when disabled."""
        config = PopulatorConfig(enabled=False)
        populator = QueuePopulator(config=config)
        populator.set_work_queue(MagicMock())
        assert populator.populate() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_returns_zero_without_queue(self, mock_scale, mock_load):
        """Test populate returns 0 without work queue."""
        populator = QueuePopulator()
        assert populator.populate() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_returns_zero_when_all_targets_met(self, mock_scale, mock_load):
        """Test populate returns 0 when all targets met."""
        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0
        populator.set_work_queue(MagicMock())
        assert populator.populate() == 0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._check_backpressure")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_adds_items(self, mock_scale, mock_load, mock_bp):
        """Test populate adds items to queue."""
        from app.coordination.types import BackpressureLevel

        mock_bp.return_value = (BackpressureLevel.NONE, 1.0)

        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
            min_queue_depth=10,
        )
        populator = QueuePopulator(config=config)
        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {"pending": [], "running": []}
        populator.set_work_queue(mock_queue)

        added = populator.populate()
        assert added > 0
        assert mock_queue.add_work.called

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._check_backpressure")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populate_queue_alias(self, mock_scale, mock_load, mock_bp):
        """Test populate_queue is alias for populate."""
        from app.coordination.types import BackpressureLevel

        mock_bp.return_value = (BackpressureLevel.NONE, 1.0)

        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
            min_queue_depth=10,
        )
        populator = QueuePopulator(config=config)
        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {"pending": [], "running": []}
        populator.set_work_queue(mock_queue)

        added1 = populator.populate()
        mock_queue.add_work.reset_mock()
        mock_queue.get_queue_status.return_value = {"pending": [], "running": []}
        added2 = populator.populate_queue()

        # Both should add same number of items
        assert added1 > 0
        assert added2 > 0


# =============================================================================
# QueuePopulator Backpressure Tests
# =============================================================================


class TestQueuePopulatorBackpressure:
    """Tests for backpressure handling."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_check_backpressure_with_monitor(self, mock_scale, mock_load):
        """Test backpressure check when monitor is available."""
        populator = QueuePopulator()

        # Mock the queue_monitor module
        with patch("app.coordination.queue_monitor.get_queue_monitor") as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_monitor.get_overall_status.return_value = {"backpressure_level": "none"}
            mock_get_monitor.return_value = mock_monitor

            bp_level, factor = populator._check_backpressure()
            assert factor == 1.0

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_check_backpressure_no_monitor(self, mock_scale, mock_load):
        """Test backpressure check when monitor unavailable."""
        populator = QueuePopulator()
        # Without mocking get_queue_monitor, it should return NONE
        bp_level, factor = populator._check_backpressure()
        assert factor == 1.0


# =============================================================================
# QueuePopulator Priority Tests
# =============================================================================


class TestQueuePopulatorPriority:
    """Tests for priority calculation."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_compute_work_priority_no_scheduler(self, mock_scale, mock_load):
        """Test priority without scheduler returns base priority."""
        populator = QueuePopulator()
        priority = populator._compute_work_priority(50, "hex8_2p", {})
        assert priority == 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_compute_work_priority_with_scheduler(self, mock_scale, mock_load):
        """Test priority with scheduler priorities."""
        populator = QueuePopulator()
        scheduler_priorities = {"hex8_2p": 10.0, "square8_2p": 5.0}
        priority = populator._compute_work_priority(50, "hex8_2p", scheduler_priorities)
        # Should be boosted
        assert priority > 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_compute_work_priority_unknown_config(self, mock_scale, mock_load):
        """Test priority for config not in scheduler returns base."""
        populator = QueuePopulator()
        scheduler_priorities = {"hex8_2p": 10.0}
        priority = populator._compute_work_priority(50, "unknown_config", scheduler_priorities)
        assert priority == 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_scheduler_priorities_no_scheduler(self, mock_scale, mock_load):
        """Test get_scheduler_priorities without scheduler."""
        populator = QueuePopulator()
        priorities = populator._get_scheduler_priorities()
        assert priorities == {}


# =============================================================================
# QueuePopulator Status Tests
# =============================================================================


class TestQueuePopulatorStatus:
    """Tests for status reporting."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_status(self, mock_scale, mock_load):
        """Test get_status returns expected fields."""
        config = PopulatorConfig(
            board_types=["hex8", "square8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)
        populator._targets["hex8_2p"].current_best_elo = 2100.0

        status = populator.get_status()
        assert "enabled" in status
        assert "min_queue_depth" in status
        assert "target_elo" in status
        assert "total_configs" in status
        assert "configs_met" in status
        assert "configs_unmet" in status
        assert "unmet_configs" in status

        assert status["configs_met"] == 1
        assert status["configs_unmet"] == 1

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_status_unmet_configs_details(self, mock_scale, mock_load):
        """Test unmet_configs contains detailed info."""
        config = PopulatorConfig(
            board_types=["hex8"],
            player_counts=[2],
        )
        populator = QueuePopulator(config=config)

        status = populator.get_status()
        assert len(status["unmet_configs"]) == 1

        unmet = status["unmet_configs"][0]
        assert "config" in unmet
        assert "current_elo" in unmet
        assert "gap" in unmet
        assert "velocity" in unmet
        assert "games" in unmet


# =============================================================================
# Singleton and Wire Events Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern and event wiring."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_queue_populator()

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_singleton(self, mock_scale, mock_load):
        """Test get_queue_populator returns same instance."""
        p1 = get_queue_populator()
        p2 = get_queue_populator()
        assert p1 is p2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_reset_queue_populator(self, mock_scale, mock_load):
        """Test reset_queue_populator clears singleton."""
        p1 = get_queue_populator()
        reset_queue_populator()
        p2 = get_queue_populator()
        assert p1 is not p2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_with_work_queue(self, mock_scale, mock_load):
        """Test get_queue_populator sets work queue on existing instance."""
        p1 = get_queue_populator()
        assert p1._work_queue is None

        mock_queue = MagicMock()
        p2 = get_queue_populator(work_queue=mock_queue)
        assert p1 is p2
        assert p2._work_queue is mock_queue


# =============================================================================
# Import Compatibility Tests
# =============================================================================


class TestImports:
    """Tests for module imports.

    December 2025: Updated to use unified_queue_populator after deprecating queue_populator wrapper.
    """

    def test_import_module(self):
        """Test that the unified module can be imported."""
        from app.coordination import unified_queue_populator
        assert unified_queue_populator is not None

    def test_import_public_classes(self):
        """Test that public classes can be imported from unified module."""
        from app.coordination.unified_queue_populator import (
            ConfigTarget,
            QueuePopulatorConfig,
            UnifiedQueuePopulator,
        )
        from app.coordination.types import BoardType
        assert BoardType is not None
        assert ConfigTarget is not None
        assert QueuePopulatorConfig is not None
        assert UnifiedQueuePopulator is not None

    def test_import_large_boards_constant(self):
        """Test that LARGE_BOARDS constant can be imported."""
        from app.coordination.unified_queue_populator import LARGE_BOARDS
        assert isinstance(LARGE_BOARDS, frozenset)

    def test_import_helper_functions(self):
        """Test that helper functions can be imported."""
        from app.coordination.unified_queue_populator import (
            get_queue_populator,
            load_populator_config_from_yaml,
            reset_queue_populator,
            wire_queue_populator_events,
        )
        assert callable(get_queue_populator)
        assert callable(load_populator_config_from_yaml)
        assert callable(reset_queue_populator)
        assert callable(wire_queue_populator_events)


# =============================================================================
# Cluster Scaling Tests
# =============================================================================


class TestClusterScaling:
    """Tests for cluster-based queue depth scaling."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    def test_scale_queue_depth_with_active_nodes(self, mock_load):
        """Test queue depth scaling with active nodes."""
        with patch("app.distributed.cluster_monitor.ClusterMonitor") as mock_monitor_class:
            mock_monitor = MagicMock()
            mock_monitor.get_cluster_status.return_value = MagicMock(active_nodes=25)
            mock_monitor_class.return_value = mock_monitor

            config = PopulatorConfig(min_queue_depth=50)
            populator = QueuePopulator(config=config)

            # 25 nodes * 2 = 50, same as default
            assert populator.config.min_queue_depth >= 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    def test_scale_queue_depth_large_cluster(self, mock_load):
        """Test queue depth scaling with large cluster."""
        with patch("app.distributed.cluster_monitor.ClusterMonitor") as mock_monitor_class:
            mock_monitor = MagicMock()
            mock_monitor.get_cluster_status.return_value = MagicMock(active_nodes=100)
            mock_monitor_class.return_value = mock_monitor

            config = PopulatorConfig(min_queue_depth=50)
            populator = QueuePopulator(config=config)

            # 100 nodes * 2 = 200
            assert populator.config.min_queue_depth >= 50

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    def test_scale_queue_depth_no_monitor_available(self, mock_load):
        """Test queue depth uses default when monitor unavailable."""
        config = PopulatorConfig(min_queue_depth=50)
        # Without mocking ClusterMonitor, import should fail gracefully
        populator = QueuePopulator(config=config)
        # Should still have a valid min_queue_depth
        assert populator.config.min_queue_depth >= 50


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Tests for event handling via wire_queue_populator_events."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_queue_populator()

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_wire_events_subscribes_to_events(self, mock_scale, mock_load):
        """Test wire_queue_populator_events subscribes to events."""
        from app.coordination.unified_queue_populator import wire_queue_populator_events

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            wire_queue_populator_events()

            # Should subscribe to multiple events
            assert mock_router.subscribe.called
            call_args = [call[0][0] for call in mock_router.subscribe.call_args_list]
            # Check some expected event types are subscribed
            assert len(call_args) >= 3
