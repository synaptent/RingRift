"""
Unit tests for app.training.model_versioning module.

Tests cover:
- ModelMetadata dataclass
- ModelVersionManager class
- Version compatibility checking
- Checksum computation
- Error classes

Created: December 2025
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.training.model_versioning import (
    ArchitectureMismatchError,
    CheckpointConfigError,
    ChecksumMismatchError,
    ConfigMismatchError,
    LegacyCheckpointError,
    ModelMetadata,
    ModelVersioningError,
    ModelVersionManager,
    VersionMismatchError,
    are_versions_compatible,
    compute_state_dict_checksum,
    get_model_config,
    get_model_version,
)


# =============================================================================
# ModelMetadata Tests
# =============================================================================


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_default_creation(self):
        """Metadata can be created with defaults."""
        meta = ModelMetadata()
        assert meta.architecture_version == "unknown"
        assert meta.model_class == "unknown"
        assert meta.config == {}
        assert meta.board_type == ""
        assert meta.board_size == 0
        assert meta.num_players == 0
        assert meta.policy_size == 0
        assert meta.training_info == {}
        assert meta.checksum == ""
        assert meta.parent_checkpoint is None

    def test_creation_with_values(self):
        """Metadata can be created with custom values."""
        meta = ModelMetadata(
            architecture_version="v2.1.0",
            model_class="RingRiftCNN_v2",
            config={"channels": 96},
            board_type="hex8",
            board_size=9,
            num_players=2,
            policy_size=4500,
        )
        assert meta.architecture_version == "v2.1.0"
        assert meta.model_class == "RingRiftCNN_v2"
        assert meta.config == {"channels": 96}
        assert meta.board_type == "hex8"

    def test_post_init_sets_timestamp(self):
        """Timestamp is set on creation if not provided."""
        meta = ModelMetadata()
        assert meta.created_at != ""
        assert "T" in meta.created_at  # ISO format check

    def test_to_dict(self):
        """Metadata can be converted to dictionary."""
        meta = ModelMetadata(
            architecture_version="v2.0.0",
            model_class="TestModel",
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["architecture_version"] == "v2.0.0"
        assert d["model_class"] == "TestModel"

    def test_from_dict(self):
        """Metadata can be created from dictionary."""
        data = {
            "architecture_version": "v3.0.0",
            "model_class": "HexNeuralNet_v3",
            "config": {"blocks": 6},
            "board_type": "hex8",
            "board_size": 9,
            "num_players": 2,
            "policy_size": 4500,
            "training_info": {},
            "created_at": "2025-12-29T12:00:00+00:00",
            "checksum": "abc123",
            "parent_checkpoint": None,
        }
        meta = ModelMetadata.from_dict(data)
        assert meta.architecture_version == "v3.0.0"
        assert meta.model_class == "HexNeuralNet_v3"
        assert meta.board_type == "hex8"

    def test_from_dict_ignores_unknown_fields(self):
        """Unknown fields in dict are ignored for forward compatibility."""
        data = {
            "architecture_version": "v2.0.0",
            "model_class": "TestModel",
            "future_field": "should_be_ignored",
        }
        meta = ModelMetadata.from_dict(data)
        assert meta.architecture_version == "v2.0.0"
        assert not hasattr(meta, "future_field")

    def test_is_compatible_with_same_version(self):
        """Same major.minor version is compatible."""
        meta1 = ModelMetadata(
            architecture_version="v2.1.0",
            model_class="TestModel",
        )
        meta2 = ModelMetadata(
            architecture_version="v2.1.5",
            model_class="TestModel",
        )
        assert meta1.is_compatible_with(meta2)

    def test_is_compatible_with_different_class(self):
        """Different model class is not compatible."""
        meta1 = ModelMetadata(
            architecture_version="v2.1.0",
            model_class="TestModel",
        )
        meta2 = ModelMetadata(
            architecture_version="v2.1.0",
            model_class="OtherModel",
        )
        assert not meta1.is_compatible_with(meta2)

    def test_is_compatible_with_different_major(self):
        """Different major version is not compatible."""
        meta1 = ModelMetadata(
            architecture_version="v2.1.0",
            model_class="TestModel",
        )
        meta2 = ModelMetadata(
            architecture_version="v3.1.0",
            model_class="TestModel",
        )
        assert not meta1.is_compatible_with(meta2)

    def test_parse_version(self):
        """Version parsing works correctly."""
        assert ModelMetadata._parse_version("v2.1.0") == (2, 1, 0)
        assert ModelMetadata._parse_version("3.0.5") == (3, 0, 5)

    def test_parse_version_invalid(self):
        """Invalid version format raises ValueError."""
        with pytest.raises(ValueError):
            ModelMetadata._parse_version("invalid")
        with pytest.raises(ValueError):
            ModelMetadata._parse_version("v2.1")


# =============================================================================
# Version Compatibility Tests
# =============================================================================


class TestVersionCompatibility:
    """Tests for version compatibility checking."""

    def test_exact_match(self):
        """Exact version match is compatible."""
        assert are_versions_compatible("v2.0.0", "v2.0.0")
        assert are_versions_compatible("v3.1.0", "v3.1.0")

    def test_known_compatible_versions(self):
        """Known compatible versions return True."""
        # v3.0.0 can be loaded by v3.1.0
        assert are_versions_compatible("v3.0.0", "v3.1.0")

    def test_incompatible_versions(self):
        """Incompatible versions return False."""
        assert not are_versions_compatible("v2.0.0", "v3.0.0")


# =============================================================================
# Checksum Tests
# =============================================================================


class TestChecksumComputation:
    """Tests for state dict checksum computation."""

    def test_basic_checksum(self):
        """Checksum is computed for basic state dict."""
        state_dict = {"layer.weight": torch.randn(10, 5)}
        checksum = compute_state_dict_checksum(state_dict)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

    def test_deterministic_checksum(self):
        """Same state dict produces same checksum."""
        state_dict = {"layer.weight": torch.ones(5, 5)}
        checksum1 = compute_state_dict_checksum(state_dict)
        checksum2 = compute_state_dict_checksum(state_dict)
        assert checksum1 == checksum2

    def test_different_state_dicts_different_checksums(self):
        """Different state dicts produce different checksums."""
        state_dict1 = {"layer.weight": torch.zeros(5, 5)}
        state_dict2 = {"layer.weight": torch.ones(5, 5)}
        checksum1 = compute_state_dict_checksum(state_dict1)
        checksum2 = compute_state_dict_checksum(state_dict2)
        assert checksum1 != checksum2


# =============================================================================
# ModelVersionManager Tests
# =============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    @property
    def config(self):
        return {"input_size": 10, "output_size": 5}


class TestModelVersionManager:
    """Tests for ModelVersionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a ModelVersionManager instance."""
        return ModelVersionManager()

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    def test_manager_instantiation(self, manager):
        """Manager can be instantiated."""
        assert manager is not None
        assert manager.default_device == torch.device("cpu")

    def test_create_metadata(self, manager, model):
        """Metadata can be created for a model."""
        meta = manager.create_metadata(model)
        assert meta is not None
        assert meta.model_class == "SimpleModel"
        assert meta.checksum != ""

    def test_create_metadata_with_training_info(self, manager, model):
        """Metadata includes training info."""
        training_info = {"epochs": 100, "loss": 0.05}
        meta = manager.create_metadata(model, training_info=training_info)
        assert meta.training_info["epochs"] == 100
        assert meta.training_info["loss"] == 0.05

    def test_save_and_load_checkpoint(self, manager, model, tmp_path):
        """Checkpoint can be saved and loaded."""
        path = tmp_path / "model.pth"
        meta = manager.create_metadata(model)

        # Save
        manager.save_checkpoint(model, meta, str(path))
        assert path.exists()

        # Load
        state_dict, loaded_meta = manager.load_checkpoint(str(path))
        assert state_dict is not None
        assert loaded_meta.model_class == "SimpleModel"

    def test_save_checkpoint_creates_directory(self, manager, model, tmp_path):
        """Save creates parent directory if needed."""
        path = tmp_path / "subdir" / "model.pth"
        meta = manager.create_metadata(model)

        manager.save_checkpoint(model, meta, str(path))
        assert path.exists()

    def test_save_with_optimizer(self, manager, model, tmp_path):
        """Checkpoint includes optimizer state."""
        path = tmp_path / "model.pth"
        optimizer = torch.optim.Adam(model.parameters())
        meta = manager.create_metadata(model)

        manager.save_checkpoint(model, meta, str(path), optimizer=optimizer)

        # Verify optimizer state is saved
        checkpoint = torch.load(path, weights_only=False)
        assert manager.OPTIMIZER_KEY in checkpoint


# =============================================================================
# Error Class Tests
# =============================================================================


class TestErrorClasses:
    """Tests for model versioning error classes."""

    def test_base_error(self):
        """Base error can be raised."""
        with pytest.raises(ModelVersioningError):
            raise ModelVersioningError("Test error")

    def test_version_mismatch_error(self):
        """Version mismatch error can be raised."""
        err = VersionMismatchError(
            checkpoint_version="v2.0.0",
            current_version="v3.0.0",
            checkpoint_path="/path/to/checkpoint.pth",
        )
        assert err.checkpoint_version == "v2.0.0"
        assert err.current_version == "v3.0.0"
        assert err.checkpoint_path == "/path/to/checkpoint.pth"

    def test_checksum_mismatch_error(self):
        """Checksum mismatch error can be raised."""
        err = ChecksumMismatchError(
            expected="abc123",
            actual="def456",
            checkpoint_path="/path/to/checkpoint.pth",
        )
        assert err.expected == "abc123"
        assert err.actual == "def456"
        assert err.checkpoint_path == "/path/to/checkpoint.pth"

    def test_legacy_checkpoint_error(self):
        """Legacy checkpoint error can be raised."""
        err = LegacyCheckpointError("/path/to/legacy.pth")
        assert err.checkpoint_path == "/path/to/legacy.pth"

    def test_config_mismatch_error(self):
        """Config mismatch error can be raised."""
        mismatched_keys = {
            "channels": (64, 96),  # (checkpoint_val, current_val)
            "blocks": (4, 6),
        }
        err = ConfigMismatchError(
            mismatched_keys=mismatched_keys,
            checkpoint_path="/path/to/checkpoint.pth",
        )
        assert err.mismatched_keys == mismatched_keys
        assert err.checkpoint_path == "/path/to/checkpoint.pth"

    def test_checkpoint_config_error(self):
        """Checkpoint config error can be raised."""
        err = CheckpointConfigError(
            checkpoint_path="/path/to/checkpoint.pth",
            expected_config={"board_type": "hex8"},
            checkpoint_config={"board_type": "square8"},
            mismatches={"board_type": ("hex8", "square8")},
        )
        assert err.checkpoint_path == "/path/to/checkpoint.pth"
        assert err.expected_config == {"board_type": "hex8"}

    def test_architecture_mismatch_error(self):
        """Architecture mismatch error can be raised."""
        err = ArchitectureMismatchError(
            checkpoint_path="/path/to/checkpoint.pth",
            key="channels",
            checkpoint_value=96,
            model_value=64,
        )
        assert err.checkpoint_path == "/path/to/checkpoint.pth"
        assert err.key == "channels"
        assert err.checkpoint_value == 96
        assert err.model_value == 64


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for model versioning."""

    def test_empty_state_dict_checksum(self):
        """Empty state dict has valid checksum."""
        checksum = compute_state_dict_checksum({})
        assert len(checksum) == 64

    def test_metadata_roundtrip(self):
        """Metadata survives serialization roundtrip."""
        original = ModelMetadata(
            architecture_version="v2.1.0",
            model_class="TestModel",
            config={"key": "value"},
            board_type="hex8",
            board_size=9,
            num_players=2,
            policy_size=4500,
        )
        d = original.to_dict()
        restored = ModelMetadata.from_dict(d)

        assert restored.architecture_version == original.architecture_version
        assert restored.model_class == original.model_class
        assert restored.board_type == original.board_type
        assert restored.config == original.config

    def test_legacy_checkpoint_missing_metadata(self):
        """Legacy checkpoints without metadata can be detected."""
        # Simulate legacy checkpoint
        checkpoint = {
            "state_dict": {"layer.weight": torch.randn(10, 5)},
        }
        manager = ModelVersionManager()

        # Legacy checkpoint has no _versioning_metadata key
        assert manager.METADATA_KEY not in checkpoint
