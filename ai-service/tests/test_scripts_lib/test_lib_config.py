"""Tests for scripts.lib.config module."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import json

from scripts.lib.config import (
    TrainingConfig,
    ModelConfig,
    BoardConfig,
    ConfigManager,
    get_config,
    get_board_config,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.num_filters == 192
        assert config.num_res_blocks == 12
        assert config.hidden_dim == 256
        assert config.num_hidden_layers == 2
        assert config.dropout == 0.1
        assert config.use_batch_norm is False
        assert config.use_spectral_norm is False

    def test_custom_values(self):
        config = ModelConfig(
            num_filters=256,
            num_res_blocks=16,
            hidden_dim=512,
        )
        assert config.num_filters == 256
        assert config.num_res_blocks == 16
        assert config.hidden_dim == 512


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.learning_rate == 0.0003
        assert config.batch_size == 256
        assert config.epochs == 50
        assert config.weight_decay == 0.0001
        assert config.early_stopping_patience == 15
        assert isinstance(config.model, ModelConfig)

    def test_from_dict_basic(self):
        data = {
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 100,
        }
        config = TrainingConfig.from_dict(data)
        assert config.learning_rate == 0.001
        assert config.batch_size == 128
        assert config.epochs == 100

    def test_from_dict_with_model(self):
        data = {
            "learning_rate": 0.001,
            "num_filters": 256,
            "num_res_blocks": 16,
        }
        config = TrainingConfig.from_dict(data)
        assert config.learning_rate == 0.001
        assert config.model.num_filters == 256
        assert config.model.num_res_blocks == 16

    def test_from_dict_with_nested_model(self):
        data = {
            "learning_rate": 0.001,
            "model": {
                "num_filters": 512,
                "hidden_dim": 768,
            }
        }
        config = TrainingConfig.from_dict(data)
        assert config.model.num_filters == 512
        assert config.model.hidden_dim == 768

    def test_to_dict(self):
        config = TrainingConfig(learning_rate=0.001, batch_size=128)
        result = config.to_dict()
        assert result["learning_rate"] == 0.001
        assert result["batch_size"] == 128
        # Model config should be flattened
        assert "num_filters" in result
        assert "model" not in result


class TestBoardConfig:
    """Tests for BoardConfig dataclass."""

    def test_config_key(self):
        config = BoardConfig(board_type="square8", num_players=2)
        assert config.config_key == "square8_2p"

    def test_from_config_key_square8_2p(self):
        config = BoardConfig.from_config_key("square8_2p")
        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.board_size == 8

    def test_from_config_key_square19_2p(self):
        config = BoardConfig.from_config_key("square19_2p")
        assert config.board_type == "square19"
        assert config.num_players == 2
        assert config.board_size == 19

    def test_from_config_key_hex8_3p(self):
        config = BoardConfig.from_config_key("hex8_3p")
        assert config.board_type == "hex8"
        assert config.num_players == 3

    def test_from_config_key_invalid(self):
        with pytest.raises(ValueError):
            BoardConfig.from_config_key("invalid")

    def test_from_config_key_no_underscore(self):
        with pytest.raises(ValueError):
            BoardConfig.from_config_key("square8")


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_get_config_defaults(self):
        manager = ConfigManager()
        # Mock the hp file not existing
        with patch.object(Path, 'exists', return_value=False):
            config = manager.get_config("square8_2p")
            assert isinstance(config, TrainingConfig)

    def test_get_config_with_override(self):
        manager = ConfigManager()
        with patch.object(Path, 'exists', return_value=False):
            config = manager.get_config(
                "square8_2p",
                override={"learning_rate": 0.01}
            )
            assert config.learning_rate == 0.01

    def test_get_defaults(self):
        manager = ConfigManager()
        with patch.object(Path, 'exists', return_value=False):
            defaults = manager.get_defaults()
            assert isinstance(defaults, TrainingConfig)

    def test_apply_env_overrides(self):
        manager = ConfigManager()
        config = {"learning_rate": 0.001}

        with patch.dict('os.environ', {'RINGRIFT_LEARNING_RATE': '0.01'}):
            result = manager._apply_env_overrides(config)
            assert result["learning_rate"] == 0.01

    def test_apply_env_overrides_batch_size(self):
        manager = ConfigManager()
        config = {"batch_size": 128}

        with patch.dict('os.environ', {'RINGRIFT_BATCH_SIZE': '256'}):
            result = manager._apply_env_overrides(config)
            assert result["batch_size"] == 256


class TestGetConfig:
    """Tests for get_config convenience function."""

    def test_get_config_returns_training_config(self):
        with patch.object(Path, 'exists', return_value=False):
            config = get_config("square8_2p")
            assert isinstance(config, TrainingConfig)

    def test_get_board_config(self):
        config = get_board_config("square8_2p")
        assert isinstance(config, BoardConfig)
        assert config.board_type == "square8"
        assert config.num_players == 2
