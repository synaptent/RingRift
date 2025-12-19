"""Tests for scripts.lib.validation module."""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.lib.validation import (
    ValidationResult,
    validate_npz_file,
    validate_jsonl_file,
    validate_model_file,
    validate_training_config,
    DataValidator,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_valid(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.info == {}

    def test_add_error(self):
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")
        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_warning(self):
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid is True  # Warnings don't affect validity
        assert "Test warning" in result.warnings

    def test_merge(self):
        result1 = ValidationResult(is_valid=True)
        result1.add_warning("Warning 1")
        result1.info["key1"] = "value1"

        result2 = ValidationResult(is_valid=True)
        result2.add_error("Error 1")
        result2.info["key2"] = "value2"

        result1.merge(result2)

        assert result1.is_valid is False
        assert "Warning 1" in result1.warnings
        assert "Error 1" in result1.errors
        assert result1.info["key1"] == "value1"
        assert result1.info["key2"] == "value2"

    def test_to_dict(self):
        result = ValidationResult(is_valid=True)
        result.add_warning("Warning")
        result.info["samples"] = 100

        data = result.to_dict()
        assert data["is_valid"] is True
        assert "Warning" in data["warnings"]
        assert data["info"]["samples"] == 100


class TestValidateNpzFile:
    """Tests for validate_npz_file function."""

    def test_file_not_found(self):
        result = validate_npz_file(Path("/nonexistent/file.npz"))
        assert result.is_valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_valid_npz_file(self, tmp_path):
        npz_path = tmp_path / "test.npz"

        # Create a valid NPZ file
        features = np.random.randn(100, 56, 8, 8).astype(np.float32)
        values = np.random.randn(100).astype(np.float32)
        policy_indices = np.array([np.array([0, 1, 2]) for _ in range(100)], dtype=object)
        policy_values = np.array([np.array([0.5, 0.3, 0.2]) for _ in range(100)], dtype=object)

        np.savez_compressed(
            npz_path,
            features=features,
            values=values,
            policy_indices=policy_indices,
            policy_values=policy_values,
        )

        result = validate_npz_file(npz_path)
        assert result.is_valid is True
        assert result.info["num_samples"] == 100
        assert result.info["feature_shape"] == [100, 56, 8, 8]

    def test_empty_npz_file(self, tmp_path):
        npz_path = tmp_path / "empty.npz"

        features = np.array([]).reshape(0, 56, 8, 8).astype(np.float32)
        values = np.array([]).astype(np.float32)
        policy_indices = np.array([], dtype=object)
        policy_values = np.array([], dtype=object)

        np.savez_compressed(
            npz_path,
            features=features,
            values=values,
            policy_indices=policy_indices,
            policy_values=policy_values,
        )

        result = validate_npz_file(npz_path)
        assert result.is_valid is False
        assert any("no samples" in e.lower() for e in result.errors)

    def test_missing_required_keys(self, tmp_path):
        npz_path = tmp_path / "incomplete.npz"

        # Missing values key
        features = np.random.randn(10, 56, 8, 8).astype(np.float32)
        np.savez_compressed(npz_path, features=features)

        result = validate_npz_file(npz_path)
        assert result.is_valid is False
        assert any("missing" in e.lower() for e in result.errors)


class TestValidateJsonlFile:
    """Tests for validate_jsonl_file function."""

    def test_file_not_found(self):
        result = validate_jsonl_file(Path("/nonexistent/file.jsonl"))
        assert result.is_valid is False

    def test_valid_jsonl_file(self, tmp_path):
        jsonl_path = tmp_path / "test.jsonl"

        games = [
            {"game_id": "1", "board_type": "square8", "num_players": 2, "moves": [{"type": "place_ring"}]},
            {"game_id": "2", "board_type": "square8", "num_players": 2, "moves": [{"type": "move_stack"}]},
        ]

        with open(jsonl_path, "w") as f:
            for game in games:
                f.write(json.dumps(game) + "\n")

        result = validate_jsonl_file(jsonl_path)
        assert result.is_valid is True
        assert result.info["valid_games"] == 2
        assert "square8" in result.info["board_types"]

    def test_empty_jsonl_file(self, tmp_path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.touch()

        result = validate_jsonl_file(jsonl_path)
        assert result.is_valid is False
        assert any("no valid games" in e.lower() for e in result.errors)

    def test_invalid_json_lines(self, tmp_path):
        jsonl_path = tmp_path / "mixed.jsonl"

        with open(jsonl_path, "w") as f:
            f.write('{"game_id": "1", "board_type": "square8", "num_players": 2, "moves": []}\n')
            f.write('invalid json line\n')
            f.write('{"game_id": "2", "board_type": "hex8", "num_players": 3, "moves": []}\n')

        result = validate_jsonl_file(jsonl_path)
        assert result.is_valid is True  # Still valid, just has warnings
        assert result.info["valid_games"] == 2
        assert len(result.warnings) > 0


class TestValidateModelFile:
    """Tests for validate_model_file function."""

    def test_file_not_found(self):
        result = validate_model_file(Path("/nonexistent/model.pt"))
        assert result.is_valid is False

    def test_valid_model_file(self, tmp_path):
        torch = pytest.importorskip("torch", reason="PyTorch not installed")

        model_path = tmp_path / "model.pt"

        # Create a simple checkpoint
        checkpoint = {
            "model_state_dict": {
                "layer1.weight": torch.randn(64, 32),
                "layer1.bias": torch.randn(64),
            },
            "epoch": 10,
            "best_val_loss": 0.5,
        }

        torch.save(checkpoint, model_path)

        result = validate_model_file(model_path)
        assert result.is_valid is True
        assert result.info["epoch"] == 10
        assert result.info["best_val_loss"] == 0.5
        assert result.info["total_parameters"] > 0


class TestValidateTrainingConfig:
    """Tests for validate_training_config function."""

    def test_valid_config(self):
        with patch('scripts.lib.config.get_config') as mock_get_config:
            from scripts.lib.config import TrainingConfig
            mock_get_config.return_value = TrainingConfig(
                learning_rate=0.001,
                batch_size=128,
                epochs=50,
            )

            result = validate_training_config("square8_2p")
            assert result.is_valid is True
            assert result.info["learning_rate"] == 0.001

    def test_invalid_config_key(self):
        result = validate_training_config("invalid_config")
        assert result.is_valid is False

    def test_high_learning_rate_warning(self):
        with patch('scripts.lib.config.get_config') as mock_get_config:
            from scripts.lib.config import TrainingConfig
            mock_get_config.return_value = TrainingConfig(
                learning_rate=0.5,  # Very high
                batch_size=128,
                epochs=50,
            )

            result = validate_training_config("square8_2p")
            assert any("high" in w.lower() for w in result.warnings)


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_training_setup(self, tmp_path):
        validator = DataValidator(base_dir=tmp_path)

        # Create data directories
        (tmp_path / "data" / "training").mkdir(parents=True)
        (tmp_path / "models" / "nnue").mkdir(parents=True)

        with patch('scripts.lib.validation.validate_training_config') as mock_config:
            mock_config.return_value = ValidationResult(is_valid=True)

            result = validator.validate_training_setup("square8_2p")
            assert isinstance(result, ValidationResult)

    def test_check_training_data_no_files(self, tmp_path):
        validator = DataValidator(base_dir=tmp_path)

        # No NPZ files
        result = validator._check_training_data("square8_2p")
        assert "npz_files" in result.info
        assert result.info["npz_files"] == 0
