"""
Validation and Health Check Utilities

Provides validation for:
- Training data files (JSONL, NPZ)
- Model files (checkpoint integrity)
- Configuration consistency
- Cluster health

Usage:
    from scripts.lib.validation import (
        validate_npz_file,
        validate_model_file,
        validate_training_config,
        DataValidator,
    )

    # Validate NPZ
    result = validate_npz_file(Path("data/training.npz"))
    if not result.is_valid:
        print(f"Invalid: {result.errors}")

    # Comprehensive validation
    validator = DataValidator()
    report = validator.validate_training_setup("square8_2p")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from scripts.lib.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def validate_npz_file(file_path: Path) -> ValidationResult:
    """Validate an NPZ training data file.

    Args:
        file_path: Path to NPZ file

    Returns:
        ValidationResult with details
    """
    result = ValidationResult(is_valid=True)

    if not file_path.exists():
        result.add_error(f"File not found: {file_path}")
        return result

    if not file_path.suffix == ".npz":
        result.add_warning(f"Unexpected extension: {file_path.suffix}")

    try:
        import numpy as np
        data = np.load(file_path, allow_pickle=True)

        # Check required keys
        required_keys = {"values", "policy_indices", "policy_values"}
        feature_keys = {"features", "boards", "states"}
        global_keys = {"globals", "global_features"}

        available_keys = set(data.keys())
        result.info["keys"] = list(available_keys)

        # Check for features
        has_features = bool(available_keys & feature_keys)
        if not has_features:
            result.add_error(f"Missing feature data (need one of: {feature_keys})")

        # Check for globals (optional but common)
        has_globals = bool(available_keys & global_keys)
        if not has_globals:
            result.add_warning("No global features found")

        # Check required keys
        missing = required_keys - available_keys
        if missing:
            result.add_error(f"Missing required keys: {missing}")

        # Validate shapes
        feature_key = (available_keys & feature_keys).pop() if has_features else None
        if feature_key:
            features = data[feature_key]
            result.info["num_samples"] = len(features)
            result.info["feature_shape"] = list(features.shape)

            if len(features) == 0:
                result.add_error("No samples in file")

            # Check values shape
            if "values" in data:
                values = data["values"]
                if len(values) != len(features):
                    result.add_error(
                        f"Shape mismatch: features={len(features)}, values={len(values)}"
                    )
                result.info["value_shape"] = list(values.shape)

            # Check policy shape
            if "policy_indices" in data and "policy_values" in data:
                policy_indices = data["policy_indices"]
                policy_values = data["policy_values"]
                if len(policy_indices) != len(features):
                    result.add_error(
                        f"Shape mismatch: features={len(features)}, "
                        f"policy_indices={len(policy_indices)}"
                    )
                if len(policy_values) != len(policy_indices):
                    result.add_error("policy_indices and policy_values length mismatch")

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        result.info["file_size_mb"] = round(file_size_mb, 2)

        if file_size_mb < 0.001:
            result.add_warning("File is very small, may be incomplete")

        data.close()

    except Exception as e:
        result.add_error(f"Failed to load NPZ file: {e}")

    return result


def validate_jsonl_file(
    file_path: Path,
    sample_size: int = 100,
) -> ValidationResult:
    """Validate a JSONL training data file.

    Args:
        file_path: Path to JSONL file
        sample_size: Number of lines to sample for validation

    Returns:
        ValidationResult with details
    """
    result = ValidationResult(is_valid=True)

    if not file_path.exists():
        result.add_error(f"File not found: {file_path}")
        return result

    try:
        line_count = 0
        valid_games = 0
        invalid_lines = []
        board_types: set[str] = set()
        num_players_set: set[int] = set()

        with open(file_path) as f:
            for i, line in enumerate(f):
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    game = json.loads(line)

                    # Validate game structure
                    if isinstance(game, dict):
                        valid_games += 1

                        # Track board types
                        if "board_type" in game:
                            board_types.add(game["board_type"])
                        if "num_players" in game:
                            num_players_set.add(game["num_players"])

                        # Sample validation
                        if i < sample_size:
                            _validate_game_record(game, result)

                except json.JSONDecodeError:
                    if len(invalid_lines) < 10:
                        invalid_lines.append(i + 1)

        result.info["total_lines"] = line_count
        result.info["valid_games"] = valid_games
        result.info["board_types"] = list(board_types)
        result.info["num_players"] = list(num_players_set)

        if invalid_lines:
            result.add_warning(
                f"Found {len(invalid_lines)} invalid JSON lines: {invalid_lines[:5]}..."
            )

        if valid_games == 0:
            result.add_error("No valid games found in file")

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        result.info["file_size_mb"] = round(file_size_mb, 2)

    except Exception as e:
        result.add_error(f"Failed to read JSONL file: {e}")

    return result


def _validate_game_record(game: dict[str, Any], result: ValidationResult) -> None:
    """Validate a single game record structure."""
    required_fields = ["board_type", "num_players", "moves"]
    recommended_fields = ["game_id", "winner", "victory_type"]

    for field_name in required_fields:
        if field_name not in game:
            result.add_warning(f"Game missing required field: {field_name}")

    for field_name in recommended_fields:
        if field_name not in game:
            pass  # Don't warn for recommended fields

    # Validate moves if present
    if "moves" in game:
        moves = game["moves"]
        if not isinstance(moves, list):
            result.add_warning("Moves field is not a list")
        elif len(moves) == 0:
            result.add_warning("Game has no moves")


def validate_model_file(file_path: Path) -> ValidationResult:
    """Validate a PyTorch model checkpoint.

    Args:
        file_path: Path to model file

    Returns:
        ValidationResult with details
    """
    result = ValidationResult(is_valid=True)

    if not file_path.exists():
        result.add_error(f"File not found: {file_path}")
        return result

    try:
        import torch

        # Load checkpoint
        checkpoint = torch.load(file_path, map_location="cpu")

        # Check for model state dict
        if isinstance(checkpoint, dict):
            result.info["checkpoint_keys"] = list(checkpoint.keys())

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the whole checkpoint is the state dict
                state_dict = checkpoint

            # Count parameters
            total_params = sum(
                p.numel() for p in state_dict.values()
                if isinstance(p, torch.Tensor)
            )
            result.info["total_parameters"] = total_params

            # Check for common keys
            if "epoch" in checkpoint:
                result.info["epoch"] = checkpoint["epoch"]
            if "best_val_loss" in checkpoint:
                result.info["best_val_loss"] = checkpoint["best_val_loss"]
            if "config" in checkpoint:
                result.info["config"] = checkpoint["config"]

        else:
            result.add_warning("Checkpoint is not a dictionary")

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        result.info["file_size_mb"] = round(file_size_mb, 2)

        if file_size_mb < 0.1:
            result.add_warning("Model file is very small")

    except Exception as e:
        result.add_error(f"Failed to load model: {e}")

    return result


def validate_training_config(config_key: str) -> ValidationResult:
    """Validate a training configuration.

    Args:
        config_key: Configuration key (e.g., "square8_2p")

    Returns:
        ValidationResult with details
    """
    result = ValidationResult(is_valid=True)

    try:
        from scripts.lib.config import get_config, BoardConfig

        # Validate board config parsing
        try:
            board_config = BoardConfig.from_config_key(config_key)
            result.info["board_type"] = board_config.board_type
            result.info["num_players"] = board_config.num_players
            result.info["board_size"] = board_config.board_size
        except ValueError as e:
            result.add_error(f"Invalid config key: {e}")
            return result

        # Get training config
        training_config = get_config(config_key)
        result.info["learning_rate"] = training_config.learning_rate
        result.info["batch_size"] = training_config.batch_size
        result.info["epochs"] = training_config.epochs
        result.info["num_filters"] = training_config.model.num_filters
        result.info["num_res_blocks"] = training_config.model.num_res_blocks

        # Validate hyperparameters
        if training_config.learning_rate <= 0:
            result.add_error("Learning rate must be positive")
        if training_config.learning_rate > 0.1:
            result.add_warning("Learning rate seems high (> 0.1)")

        if training_config.batch_size < 8:
            result.add_warning("Batch size is very small")
        if training_config.batch_size > 2048:
            result.add_warning("Batch size is very large")

        if training_config.epochs < 1:
            result.add_error("Epochs must be at least 1")
        if training_config.epochs > 1000:
            result.add_warning("Very high number of epochs (> 1000)")

    except ImportError as e:
        result.add_error(f"Missing dependency: {e}")
    except Exception as e:
        result.add_error(f"Configuration error: {e}")

    return result


class DataValidator:
    """Comprehensive data validation for training setup."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize validator.

        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = base_dir or Path(".")
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"

    def validate_training_setup(self, config_key: str) -> ValidationResult:
        """Validate complete training setup for a configuration.

        Args:
            config_key: Configuration key (e.g., "square8_2p")

        Returns:
            ValidationResult with comprehensive details
        """
        result = ValidationResult(is_valid=True)

        # Validate config
        logger.info(f"Validating config: {config_key}")
        config_result = validate_training_config(config_key)
        result.merge(config_result)

        # Check for training data
        logger.info("Checking training data...")
        data_result = self._check_training_data(config_key)
        result.merge(data_result)

        # Check for models
        logger.info("Checking models...")
        model_result = self._check_models(config_key)
        result.merge(model_result)

        return result

    def _check_training_data(self, config_key: str) -> ValidationResult:
        """Check for available training data."""
        result = ValidationResult(is_valid=True)

        training_dir = self.data_dir / "training"
        npz_files = list(training_dir.glob(f"*{config_key}*.npz")) if training_dir.exists() else []

        result.info["npz_files"] = len(npz_files)

        if not npz_files:
            result.add_warning(f"No NPZ files found for {config_key}")
        else:
            # Validate the most recent NPZ
            most_recent = max(npz_files, key=lambda p: p.stat().st_mtime)
            result.info["most_recent_npz"] = str(most_recent)

            npz_result = validate_npz_file(most_recent)
            if not npz_result.is_valid:
                result.add_warning(f"Issues with {most_recent.name}: {npz_result.errors}")
            else:
                result.info["samples_available"] = npz_result.info.get("num_samples", 0)

        return result

    def _check_models(self, config_key: str) -> ValidationResult:
        """Check for existing model files."""
        result = ValidationResult(is_valid=True)

        # Check common model locations
        model_patterns = [
            self.models_dir / "nnue" / f"{config_key}*.pt",
            self.models_dir / f"{config_key}*.pt",
        ]

        model_files = []
        for pattern in model_patterns:
            model_files.extend(pattern.parent.glob(pattern.name))

        result.info["model_files"] = len(model_files)

        if model_files:
            # Validate the most recent model
            most_recent = max(model_files, key=lambda p: p.stat().st_mtime)
            result.info["most_recent_model"] = str(most_recent)

            model_result = validate_model_file(most_recent)
            if not model_result.is_valid:
                result.add_warning(f"Issues with {most_recent.name}: {model_result.errors}")
            else:
                result.info["model_params"] = model_result.info.get("total_parameters", 0)

        return result

    def validate_all_configs(self) -> dict[str, ValidationResult]:
        """Validate all known configurations.

        Returns:
            Dictionary of config_key -> ValidationResult
        """
        configs = [
            "square8_2p",
            "square8_3p",
            "square8_4p",
            "hex8_2p",
            "hex8_3p",
            "square19_2p",
        ]

        results = {}
        for config_key in configs:
            logger.info(f"Validating {config_key}...")
            results[config_key] = self.validate_training_setup(config_key)

        return results


def validate_cluster_health(timeout_seconds: int = 30) -> ValidationResult:
    """Validate cluster health.

    Args:
        timeout_seconds: SSH timeout in seconds

    Returns:
        ValidationResult with cluster health info
    """
    result = ValidationResult(is_valid=True)

    try:
        from scripts.lib.cluster import ClusterManager

        manager = ClusterManager()
        healthy_nodes = manager.get_healthy_nodes(force_check=True)

        result.info["total_nodes"] = len(manager.nodes)
        result.info["healthy_nodes"] = len(healthy_nodes)
        result.info["healthy_node_names"] = [n.name for n in healthy_nodes]

        if not healthy_nodes:
            result.add_error("No healthy nodes available")
        elif len(healthy_nodes) < len(manager.nodes) / 2:
            result.add_warning(
                f"Less than half of nodes healthy: "
                f"{len(healthy_nodes)}/{len(manager.nodes)}"
            )

        # Collect detailed metrics
        for node in manager.nodes:
            health = node.check_health(force=True)
            if health.is_healthy:
                result.info[f"node_{node.name}"] = {
                    "gpu_utilization": health.gpu_utilization,
                    "memory_used_gb": health.memory_used_gb,
                    "memory_total_gb": health.memory_total_gb,
                    "temperature_c": health.temperature_c,
                }
            else:
                result.info[f"node_{node.name}"] = {
                    "error": health.error_message,
                }

    except ImportError as e:
        result.add_error(f"Cluster module not available: {e}")
    except Exception as e:
        result.add_error(f"Cluster health check failed: {e}")

    return result
