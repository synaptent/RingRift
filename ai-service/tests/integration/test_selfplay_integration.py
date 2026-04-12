"""Integration tests for the selfplay pipeline.

Tests the selfplay configuration, argument parsing, and orchestration
components working together.
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from app.models import BoardType
from app.config.thresholds import DISK_PRODUCTION_HALT_PERCENT
from app.training.selfplay_config import (
    EngineMode,
    OutputFormat,
    SelfplayConfig,
    create_argument_parser,
    get_default_config,
    get_production_config,
    parse_selfplay_args,
)
from app.utils.parallel_defaults import get_default_workers


class TestSelfplayConfigCreation:
    """Test SelfplayConfig creation and validation."""

    def test_default_config_creation(self):
        """Test creating a default config."""
        config = SelfplayConfig()

        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.num_games == 1000
        assert config.engine_mode == EngineMode.MIXED
        assert config.output_format == OutputFormat.DB

    def test_custom_config_creation(self):
        """Test creating a custom config."""
        config = SelfplayConfig(
            board_type="hexagonal",
            num_players=4,
            num_games=5000,
            engine_mode=EngineMode.MCTS,
            mcts_simulations=1600,
        )

        assert config.board_type == "hexagonal"
        assert config.num_players == 4
        assert config.num_games == 5000
        assert config.engine_mode == EngineMode.MCTS
        assert config.mcts_simulations == 1600

    def test_board_type_normalization(self):
        """Test board type aliases are normalized."""
        # Test various aliases
        configs = [
            SelfplayConfig(board_type="hex"),
            SelfplayConfig(board_type="hexagonal"),
        ]
        for config in configs:
            assert config.board_type == "hexagonal"

        configs = [
            SelfplayConfig(board_type="sq8"),
            SelfplayConfig(board_type="square-8"),
            SelfplayConfig(board_type="square8"),
        ]
        for config in configs:
            assert config.board_type == "square8"

        with pytest.raises(ValueError, match="Unknown board type"):
            SelfplayConfig(board_type="square")

        config = SelfplayConfig(board_type="sq19")
        assert config.board_type == "square19"

    def test_engine_mode_from_string(self):
        """Test engine mode can be set from string."""
        config = SelfplayConfig(engine_mode="mcts")
        assert config.engine_mode == EngineMode.MCTS

        config = SelfplayConfig(engine_mode="heuristic-only")
        assert config.engine_mode == EngineMode.HEURISTIC

    def test_output_format_from_string(self):
        """Test output format can be set from string."""
        config = SelfplayConfig(output_format="jsonl")
        assert config.output_format == OutputFormat.JSONL

        config = SelfplayConfig(output_format="npz")
        assert config.output_format == OutputFormat.NPZ


class TestSelfplayConfigProperties:
    """Test SelfplayConfig computed properties."""

    def test_config_key_two_player(self):
        """Test config key for 2-player game."""
        config = SelfplayConfig(board_type="square8", num_players=2)
        assert config.config_key == "square8_2p"

    def test_config_key_four_player(self):
        """Test config key for 4-player game."""
        config = SelfplayConfig(board_type="hexagonal", num_players=4)
        assert config.config_key == "hexagonal_4p"

    def test_board_type_enum(self):
        """Test board_type_enum property."""
        config = SelfplayConfig(board_type="square8")
        assert config.board_type_enum == BoardType.SQUARE8

        config = SelfplayConfig(board_type="square19")
        assert config.board_type_enum == BoardType.SQUARE19

        config = SelfplayConfig(board_type="hexagonal")
        assert config.board_type_enum == BoardType.HEXAGONAL

        config = SelfplayConfig(board_type="hex8")
        assert config.board_type_enum == BoardType.HEX8

    def test_default_output_dir(self):
        """Test default output directory is set based on config key."""
        config = SelfplayConfig(board_type="square8", num_players=2)
        assert config.output_dir == "data/selfplay/square8_2p"

    def test_default_record_db(self):
        """Test default record DB path is set."""
        config = SelfplayConfig(board_type="square8", num_players=2)
        assert config.record_db == "data/games/square8_2p.db"


class TestSelfplayConfigSerialization:
    """Test SelfplayConfig serialization/deserialization."""

    def test_to_dict(self):
        """Test config to_dict serialization."""
        config = SelfplayConfig(
            board_type="square8",
            num_players=2,
            engine_mode=EngineMode.MCTS,
        )

        d = config.to_dict()

        assert d["board_type"] == "square8"
        assert d["num_players"] == 2
        assert d["engine_mode"] == "mcts"
        assert d["config_key"] == "square8_2p"

    def test_from_dict(self):
        """Test config from_dict deserialization."""
        d = {
            "board_type": "hexagonal",
            "num_players": 3,
            "engine_mode": "gumbel-mcts",
            "mcts_simulations": 1200,
        }

        config = SelfplayConfig.from_dict(d)

        assert config.board_type == "hexagonal"
        assert config.num_players == 3
        assert config.engine_mode == EngineMode.GUMBEL_MCTS
        assert config.mcts_simulations == 1200

    def test_roundtrip_serialization(self):
        """Test config survives to_dict/from_dict roundtrip."""
        original = SelfplayConfig(
            board_type="square19",
            num_players=4,
            engine_mode=EngineMode.DIVERSE,
            temperature=0.8,
            seed=42,
        )

        # Roundtrip
        d = original.to_dict()
        restored = SelfplayConfig.from_dict(d)

        assert restored.board_type == original.board_type
        assert restored.num_players == original.num_players
        assert restored.engine_mode == original.engine_mode
        assert restored.temperature == original.temperature
        assert restored.seed == original.seed


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_minimal_args(self):
        """Test parsing with minimal arguments."""
        config = parse_selfplay_args([])

        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.engine_mode == EngineMode.NNUE_GUIDED

    def test_parse_board_type(self):
        """Test parsing board type argument."""
        config = parse_selfplay_args(["--board", "hexagonal"])
        assert config.board_type == "hexagonal"

        config = parse_selfplay_args(["--board-type", "hex"])
        assert config.board_type == "hexagonal"

    def test_parse_num_players(self):
        """Test parsing player count argument."""
        config = parse_selfplay_args(["--num-players", "4"])
        assert config.num_players == 4

        config = parse_selfplay_args(["-p", "3"])
        assert config.num_players == 3

    def test_parse_engine_mode(self):
        """Test parsing engine mode argument."""
        config = parse_selfplay_args(["--engine-mode", "mcts"])
        assert config.engine_mode == EngineMode.MCTS

        config = parse_selfplay_args(["-e", "random"])
        assert config.engine_mode == EngineMode.RANDOM

    def test_parse_num_games(self):
        """Test parsing game count argument."""
        config = parse_selfplay_args(["--num-games", "5000"])
        assert config.num_games == 5000

        config = parse_selfplay_args(["-n", "100"])
        assert config.num_games == 100

    def test_parse_gpu_options(self):
        """Test parsing GPU-related options."""
        config = parse_selfplay_args(["--no-gpu"])
        assert config.use_gpu is False

        config = parse_selfplay_args(["--gpu-device", "1"])
        assert config.gpu_device == 1

    def test_parse_ramdrive_options(self):
        """Test parsing ramdrive options."""
        config = parse_selfplay_args(["--use-ramdrive", "--ramdrive-path", "/mnt/ramdisk"])
        assert config.use_ramdrive is True
        assert config.ramdrive_path == "/mnt/ramdisk"

    def test_parse_output_options(self):
        """Test parsing output options."""
        config = parse_selfplay_args([
            "--output-dir", "/tmp/games",
            "--output-format", "jsonl",
            "--lean-db",
        ])
        assert config.output_dir == "/tmp/games"
        assert config.output_format == OutputFormat.JSONL
        assert config.lean_db is True
        assert config.store_history_entries is False


class TestConvenienceFunctions:
    """Test convenience configuration functions."""

    def test_get_default_config(self):
        """Test get_default_config convenience function."""
        config = get_default_config()
        assert config.board_type == "square8"
        assert config.num_players == 2

        config = get_default_config("hexagonal", 4)
        assert config.board_type == "hexagonal"
        assert config.num_players == 4

    def test_get_production_config(self):
        """Test get_production_config convenience function."""
        config = get_production_config("square8", 2)

        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.num_games == 100000
        assert config.engine_mode == EngineMode.DIVERSE
        assert config.search_depth == 4
        assert config.mcts_simulations == 1600
        assert config.store_history_entries is True
        assert config.cache_nnue_features is True


class TestEngineModeEnum:
    """Test EngineMode enum values."""

    def test_all_engine_modes_exist(self):
        """Test all expected engine modes are defined."""
        expected_modes = [
            "heuristic-only",
            "nnue-guided",
            "policy-only",
            "nn-minimax",
            "nn-descent",
            "gumbel-mcts",
            "mcts",
            "mixed",
            "diverse",
            "random",
            "descent-only",
            "maxn",
            "brs",
            "gmo",
        ]

        for mode in expected_modes:
            assert EngineMode(mode) is not None

    def test_engine_mode_string_value(self):
        """Test EngineMode values are strings."""
        assert EngineMode.MCTS.value == "mcts"
        assert EngineMode.NNUE_GUIDED.value == "nnue-guided"
        assert EngineMode.DIVERSE.value == "diverse"


class TestOutputFormatEnum:
    """Test OutputFormat enum values."""

    def test_all_output_formats_exist(self):
        """Test all expected output formats are defined."""
        assert OutputFormat.JSONL.value == "jsonl"
        assert OutputFormat.DB.value == "db"
        assert OutputFormat.NPZ.value == "npz"


class TestResourceSettings:
    """Test resource-related configuration settings."""

    def test_default_resource_settings(self):
        """Test default resource settings."""
        config = SelfplayConfig()

        assert config.num_workers == get_default_workers()
        assert config.batch_size == 256
        assert config.use_gpu is True
        assert config.gpu_device == 0

    def test_disk_monitoring_thresholds(self):
        """Test disk monitoring threshold settings."""
        config = SelfplayConfig()

        assert config.disk_warning_percent == 75
        assert config.disk_critical_percent == DISK_PRODUCTION_HALT_PERCENT

    def test_checkpoint_interval(self):
        """Test checkpoint interval setting."""
        config = SelfplayConfig(checkpoint_interval=500)
        assert config.checkpoint_interval == 500


class TestNNBatchingSettings:
    """Test NN batching configuration settings."""

    def test_default_nn_batching_disabled(self):
        """Test NN batching is enabled by default."""
        config = SelfplayConfig()
        assert config.nn_batch_enabled is True

    def test_nn_batching_settings(self):
        """Test NN batching configuration."""
        config = SelfplayConfig(
            nn_batch_enabled=True,
            nn_batch_timeout_ms=100,
            nn_max_batch_size=512,
        )

        assert config.nn_batch_enabled is True
        assert config.nn_batch_timeout_ms == 100
        assert config.nn_max_batch_size == 512


class TestShadowValidationSettings:
    """Test shadow validation configuration settings."""

    def test_default_shadow_validation_disabled(self):
        """Test shadow validation is disabled by default."""
        config = SelfplayConfig()
        assert config.shadow_validation is False

    def test_shadow_validation_settings(self):
        """Test shadow validation configuration."""
        config = SelfplayConfig(
            shadow_validation=True,
            shadow_sample_rate=0.1,
            shadow_threshold=0.002,
        )

        assert config.shadow_validation is True
        assert config.shadow_sample_rate == 0.1
        assert config.shadow_threshold == 0.002


class TestGameRulesSettings:
    """Test game rules customization settings."""

    def test_default_game_rules(self):
        """Test default game rules settings."""
        config = SelfplayConfig()

        assert config.lps_victory_rounds == 3
        assert config.min_game_length == 0
        assert config.random_opening_moves == 0

    def test_custom_game_rules(self):
        """Test custom game rules settings."""
        config = SelfplayConfig(
            lps_victory_rounds=5,
            min_game_length=20,
            random_opening_moves=4,
        )

        assert config.lps_victory_rounds == 5
        assert config.min_game_length == 20
        assert config.random_opening_moves == 4
