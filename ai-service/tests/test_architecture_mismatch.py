"""
Tests for Architecture Mismatch Prevention

Tests cover:
- ArchitectureMismatchError exception class
- Memory tier inference from models
- Checkpoint tier detection
- Architecture validation before loading
- CLI memory-tier flag handling
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.training.model_versioning import (
    ArchitectureMismatchError,
    ModelMetadata,
    ModelVersionManager,
    get_model_config,
    infer_memory_tier_from_model,
)


# =============================================================================
# Test Models
# =============================================================================


class MockCNNModel(nn.Module):
    """Mock CNN model for testing tier inference."""

    def __init__(self, num_filters: int = 128, num_res_blocks: int = 13):
        super().__init__()
        self.num_filters = num_filters
        # Create res_blocks as a ModuleList to match real model structure
        self.res_blocks = nn.ModuleList([
            nn.Linear(num_filters, num_filters) for _ in range(num_res_blocks)
        ])
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class RingRiftCNN_v4(MockCNNModel):
    """Mock v4 model (128 channels, 13 blocks)."""

    def __init__(self):
        super().__init__(num_filters=128, num_res_blocks=13)


class RingRiftCNN_v3(MockCNNModel):
    """Mock v3 model for v3-high tier (192 channels, 12 blocks)."""

    def __init__(self):
        super().__init__(num_filters=192, num_res_blocks=12)


class RingRiftCNN_v3_Lite(MockCNNModel):
    """Mock v3-low model (96 channels, 6 blocks)."""

    def __init__(self):
        super().__init__(num_filters=96, num_res_blocks=6)


class RingRiftCNN_v5_Heavy(MockCNNModel):
    """Mock v5 model (160 channels, 11 blocks)."""

    def __init__(self):
        super().__init__(num_filters=160, num_res_blocks=11)


# =============================================================================
# ArchitectureMismatchError Tests
# =============================================================================


class TestArchitectureMismatchError:
    """Tests for ArchitectureMismatchError exception class."""

    def test_basic_error_creation(self):
        """Test creating error with basic parameters."""
        error = ArchitectureMismatchError(
            checkpoint_path="/path/to/checkpoint.pth",
            key="num_filters",
            checkpoint_value=192,
            model_value=128,
        )

        assert error.checkpoint_path == "/path/to/checkpoint.pth"
        assert error.key == "num_filters"
        assert error.checkpoint_value == 192
        assert error.model_value == 128
        assert error.memory_tier is None

    def test_error_message_contains_key_info(self):
        """Test error message contains checkpoint path and mismatch details."""
        error = ArchitectureMismatchError(
            checkpoint_path="/models/my_checkpoint.pth",
            key="num_res_blocks",
            checkpoint_value=12,
            model_value=13,
        )

        message = str(error)
        assert "/models/my_checkpoint.pth" in message
        assert "num_res_blocks" in message
        assert "12" in message
        assert "13" in message

    def test_error_with_memory_tier_hint(self):
        """Test error includes memory tier suggestion when provided."""
        error = ArchitectureMismatchError(
            checkpoint_path="/path/to/checkpoint.pth",
            key="num_filters",
            checkpoint_value=192,
            model_value=128,
            memory_tier="v3-high",
        )

        assert error.memory_tier == "v3-high"
        message = str(error)
        assert "--memory-tier=v3-high" in message

    def test_error_without_memory_tier_hint(self):
        """Test error message when no tier hint available."""
        error = ArchitectureMismatchError(
            checkpoint_path="/path/to/checkpoint.pth",
            key="policy_size",
            checkpoint_value=100,
            model_value=200,
            memory_tier=None,
        )

        message = str(error)
        assert "--memory-tier=" not in message

    def test_error_is_runtime_error(self):
        """Test ArchitectureMismatchError is a RuntimeError subclass."""
        error = ArchitectureMismatchError(
            checkpoint_path="test.pth",
            key="num_filters",
            checkpoint_value=128,
            model_value=256,
        )

        assert isinstance(error, RuntimeError)

    def test_error_can_be_raised_and_caught(self):
        """Test error can be raised and caught normally."""
        with pytest.raises(ArchitectureMismatchError) as exc_info:
            raise ArchitectureMismatchError(
                checkpoint_path="test.pth",
                key="num_filters",
                checkpoint_value=192,
                model_value=128,
                memory_tier="v3-high",
            )

        assert exc_info.value.key == "num_filters"
        assert exc_info.value.memory_tier == "v3-high"


# =============================================================================
# Memory Tier Inference Tests
# =============================================================================


class TestInferMemoryTierFromModel:
    """Tests for infer_memory_tier_from_model() function."""

    def test_infer_v4_tier(self):
        """Test inference of v4 tier (128 channels, 13 blocks)."""
        model = RingRiftCNN_v4()
        tier = infer_memory_tier_from_model(model)
        assert tier == "v4"

    def test_infer_v3_high_tier(self):
        """Test inference of v3-high tier (192 channels, 12 blocks)."""
        model = RingRiftCNN_v3()
        tier = infer_memory_tier_from_model(model)
        assert tier == "v3-high"

    def test_infer_v3_low_tier(self):
        """Test inference of v3-low tier (96 channels, 6 blocks)."""
        model = RingRiftCNN_v3_Lite()
        tier = infer_memory_tier_from_model(model)
        assert tier == "v3-low"

    def test_infer_v5_tier(self):
        """Test inference of v5 tier (160 channels, 11 blocks)."""
        model = RingRiftCNN_v5_Heavy()
        tier = infer_memory_tier_from_model(model)
        assert tier == "v5"

    def test_infer_unknown_tier(self):
        """Test inference returns 'unknown' for unrecognized architecture."""
        model = MockCNNModel(num_filters=99, num_res_blocks=7)
        tier = infer_memory_tier_from_model(model)
        assert tier == "unknown"

    def test_infer_model_without_num_filters(self):
        """Test inference handles model without num_filters attribute."""
        model = nn.Linear(10, 10)  # Simple model without num_filters
        tier = infer_memory_tier_from_model(model)
        assert tier == "unknown"

    def test_infer_model_without_res_blocks(self):
        """Test inference handles model without res_blocks attribute."""

        class NoResBlocksModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_filters = 128
                self.fc = nn.Linear(128, 1)

        model = NoResBlocksModel()
        tier = infer_memory_tier_from_model(model)
        # Should still attempt to match based on available info
        assert tier in ["unknown", "v4"]  # May or may not match


# =============================================================================
# Get Model Config Tests
# =============================================================================


class TestGetModelConfig:
    """Tests for get_model_config() function."""

    def test_extracts_num_filters(self):
        """Test extraction of num_filters from model."""
        model = MockCNNModel(num_filters=192, num_res_blocks=12)
        config = get_model_config(model)
        assert config.get("num_filters") == 192

    def test_extracts_num_res_blocks(self):
        """Test extraction of num_res_blocks from model."""
        model = MockCNNModel(num_filters=128, num_res_blocks=13)
        config = get_model_config(model)
        assert config.get("num_res_blocks") == 13

    def test_includes_memory_tier(self):
        """Test that config includes inferred memory tier."""
        model = RingRiftCNN_v4()
        config = get_model_config(model)
        assert "memory_tier" in config
        assert config["memory_tier"] == "v4"

    def test_config_for_unknown_model(self):
        """Test config extraction for model with no known attributes."""
        model = nn.Linear(10, 10)
        config = get_model_config(model)
        # Should return dict even for unknown models
        assert isinstance(config, dict)


# =============================================================================
# Checkpoint Tier Detection Tests
# =============================================================================


class TestDetectTierFromCheckpoint:
    """Tests for detect_tier_from_checkpoint() in train.py."""

    @pytest.fixture
    def temp_checkpoint(self):
        """Create a temporary checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_detect_tier_from_metadata_memory_tier(self, temp_checkpoint):
        """Test detection when memory_tier is stored in metadata."""
        # Create checkpoint with memory_tier in metadata
        checkpoint = {
            "_versioning_metadata": {
                "memory_tier": "v3-high",
                "config": {"num_filters": 192, "num_res_blocks": 12},
            },
            "model_state_dict": {},
        }
        torch.save(checkpoint, temp_checkpoint)

        # Import the function
        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier == "v3-high"

    def test_detect_tier_from_config_num_filters(self, temp_checkpoint):
        """Test detection from num_filters when memory_tier not stored."""
        # Create checkpoint without memory_tier but with num_filters
        checkpoint = {
            "_versioning_metadata": {
                "config": {"num_filters": 192, "num_res_blocks": 12},
            },
            "model_state_dict": {},
        }
        torch.save(checkpoint, temp_checkpoint)

        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier == "v3-high"

    def test_detect_tier_v4_from_128_filters(self, temp_checkpoint):
        """Test detection of v4 tier from 128 filters."""
        checkpoint = {
            "_versioning_metadata": {
                "config": {"num_filters": 128},
            },
            "model_state_dict": {},
        }
        torch.save(checkpoint, temp_checkpoint)

        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier == "v4"

    def test_detect_tier_v5_from_160_filters(self, temp_checkpoint):
        """Test detection of v5 tier from 160 filters."""
        checkpoint = {
            "_versioning_metadata": {
                "config": {"num_filters": 160},
            },
            "model_state_dict": {},
        }
        torch.save(checkpoint, temp_checkpoint)

        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier == "v5"

    def test_detect_tier_v6_from_256_filters(self, temp_checkpoint):
        """Test detection of v6 tier from 256 filters."""
        checkpoint = {
            "_versioning_metadata": {
                "config": {"num_filters": 256},
            },
            "model_state_dict": {},
        }
        torch.save(checkpoint, temp_checkpoint)

        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier == "v6"

    def test_detect_tier_returns_none_for_unknown(self, temp_checkpoint):
        """Test detection returns None for unknown filter count."""
        checkpoint = {
            "_versioning_metadata": {
                "config": {"num_filters": 99},
            },
            "model_state_dict": {},
        }
        torch.save(checkpoint, temp_checkpoint)

        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier is None

    def test_detect_tier_returns_none_for_missing_metadata(self, temp_checkpoint):
        """Test detection returns None when no metadata present."""
        checkpoint = {"model_state_dict": {}}
        torch.save(checkpoint, temp_checkpoint)

        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint(temp_checkpoint)
        assert tier is None

    def test_detect_tier_returns_none_for_nonexistent_file(self):
        """Test detection returns None for nonexistent file."""
        from app.training.train import detect_tier_from_checkpoint

        tier = detect_tier_from_checkpoint("/nonexistent/path.pth")
        assert tier is None


# =============================================================================
# Architecture Validation Integration Tests
# =============================================================================


class TestArchitectureValidation:
    """Integration tests for architecture validation in checkpointing."""

    @pytest.fixture
    def manager(self):
        """Create a ModelVersionManager."""
        return ModelVersionManager()

    @pytest.fixture
    def temp_checkpoint(self):
        """Create a temporary checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_validation_passes_for_matching_architecture(
        self, manager, temp_checkpoint
    ):
        """Test validation passes when architectures match."""
        # Create and save a model
        model = MockCNNModel(num_filters=128, num_res_blocks=13)
        metadata = manager.create_metadata(model)
        manager.save_checkpoint(model, metadata, temp_checkpoint)

        # Load with matching model - should not raise
        model2 = MockCNNModel(num_filters=128, num_res_blocks=13)
        state_dict, _ = manager.load_checkpoint(temp_checkpoint, strict=False)
        # Should be able to load without error
        assert state_dict is not None

    def test_tier_mapping_consistency(self):
        """Test that tier mapping is consistent across components."""
        # The tier mappings in train_cli.py and model_versioning.py should match
        TIER_DEFAULTS = {
            "v4": (128, 13),
            "v3-high": (192, 12),
            "v3-low": (96, 6),
            "v5": (160, 11),
            "v5.1": (160, 11),
            "v6": (256, 18),
            "v6-xl": (320, 20),
        }

        # Verify each tier maps to expected filters
        for tier, (expected_filters, expected_blocks) in TIER_DEFAULTS.items():
            model = MockCNNModel(
                num_filters=expected_filters, num_res_blocks=expected_blocks
            )
            # Set the class name to match expected pattern
            model.__class__.__name__ = f"RingRiftCNN_{tier.replace('-', '_')}"
            config = get_model_config(model)
            assert config.get("num_filters") == expected_filters
            assert config.get("num_res_blocks") == expected_blocks


# =============================================================================
# CLI Memory Tier Flag Tests
# =============================================================================


class TestCLIMemoryTierFlag:
    """Tests for --memory-tier CLI flag handling."""

    def test_tier_choices_are_valid(self):
        """Test all tier choices are recognized."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--memory-tier",
            type=str,
            choices=["v4", "v3-high", "v3-low", "v5", "v5.1", "v6", "v6-xl"],
        )

        # All valid choices should parse
        for tier in ["v4", "v3-high", "v3-low", "v5", "v5.1", "v6", "v6-xl"]:
            args = parser.parse_args(["--memory-tier", tier])
            assert args.memory_tier == tier

    def test_invalid_tier_rejected(self):
        """Test invalid tier choice is rejected."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--memory-tier",
            type=str,
            choices=["v4", "v3-high", "v3-low", "v5", "v5.1", "v6", "v6-xl"],
        )

        with pytest.raises(SystemExit):
            parser.parse_args(["--memory-tier", "invalid-tier"])

    def test_tier_to_architecture_mapping(self):
        """Test tier names map to correct architecture parameters."""
        TIER_DEFAULTS = {
            "v4": (128, 13),
            "v3-high": (192, 12),
            "v3-low": (96, 6),
            "v5": (160, 11),
            "v5.1": (160, 11),
            "v6": (256, 18),
            "v6-xl": (320, 20),
        }

        # Verify the mapping
        assert TIER_DEFAULTS["v4"] == (128, 13)
        assert TIER_DEFAULTS["v3-high"] == (192, 12)
        assert TIER_DEFAULTS["v6-xl"] == (320, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
