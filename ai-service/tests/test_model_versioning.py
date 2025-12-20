"""
Tests for Model Versioning System

Tests cover:
- Version validation and mismatch detection
- Metadata extraction and serialization
- Migration of legacy checkpoints
- Checksum validation
- Backwards compatibility
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from app.ai import neural_net as neural_net_mod
from app.ai.neural_net import RingRiftCNN_v2
from app.training.model_versioning import (
    RINGRIFT_CNN_V2_VERSION,
    ChecksumMismatchError,
    LegacyCheckpointError,
    ModelMetadata,
    ModelVersionManager,
    VersionMismatchError,
    compute_state_dict_checksum,
    get_model_config,
    get_model_version,
    load_model_with_validation,
    save_model_checkpoint,
)


class SimpleModel(nn.Module):
    """Simple test model for testing versioning."""

    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DifferentArchModel(nn.Module):
    """Model with different architecture for mismatch testing."""

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(self, input_size: int = 10, hidden_size: int = 30):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Extra layer
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# ModelMetadata Tests
# =============================================================================


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = ModelMetadata(
            architecture_version="v1.0.0",
            model_class="TestModel",
            config={"input_size": 10},
        )
        assert metadata.architecture_version == "v1.0.0"
        assert metadata.model_class == "TestModel"
        assert metadata.config == {"input_size": 10}
        assert metadata.created_at != ""  # Should have default timestamp

    def test_metadata_serialization_roundtrip(self):
        """Test metadata can be serialized and deserialized."""
        metadata = ModelMetadata(
            architecture_version="v1.2.3",
            model_class="RingRiftCNN_v2",
            config={"board_size": 8, "num_filters": 128},
            training_info={"epochs": 100, "loss": 0.05},
            checksum="abc123",
            parent_checkpoint="/path/to/parent.pth",
        )

        # Convert to dict and back
        data = metadata.to_dict()
        restored = ModelMetadata.from_dict(data)

        assert restored.architecture_version == metadata.architecture_version
        assert restored.model_class == metadata.model_class
        assert restored.config == metadata.config
        assert restored.training_info == metadata.training_info
        assert restored.checksum == metadata.checksum
        assert restored.parent_checkpoint == metadata.parent_checkpoint

    def test_version_compatibility_same_major_minor(self):
        """Test compatibility check with same major.minor version."""
        meta1 = ModelMetadata(
            architecture_version="v1.2.0",
            model_class="TestModel",
            config={},
        )
        meta2 = ModelMetadata(
            architecture_version="v1.2.5",
            model_class="TestModel",
            config={},
        )
        assert meta1.is_compatible_with(meta2)

    def test_version_compatibility_different_minor(self):
        """Test incompatibility with different minor version."""
        meta1 = ModelMetadata(
            architecture_version="v1.2.0",
            model_class="TestModel",
            config={},
        )
        meta2 = ModelMetadata(
            architecture_version="v1.3.0",
            model_class="TestModel",
            config={},
        )
        assert not meta1.is_compatible_with(meta2)

    def test_version_compatibility_different_class(self):
        """Test incompatibility with different model class."""
        meta1 = ModelMetadata(
            architecture_version="v1.0.0",
            model_class="ModelA",
            config={},
        )
        meta2 = ModelMetadata(
            architecture_version="v1.0.0",
            model_class="ModelB",
            config={},
        )
        assert not meta1.is_compatible_with(meta2)

    def test_version_parsing(self):
        """Test version string parsing."""
        assert ModelMetadata._parse_version("v1.2.3") == (1, 2, 3)
        assert ModelMetadata._parse_version("1.2.3") == (1, 2, 3)
        assert ModelMetadata._parse_version("v0.0.1") == (0, 0, 1)

    def test_version_parsing_invalid(self):
        """Test version parsing with invalid format."""
        with pytest.raises(ValueError):
            ModelMetadata._parse_version("1.2")
        with pytest.raises(ValueError):
            ModelMetadata._parse_version("v1")


# =============================================================================
# Checksum Tests
# =============================================================================


class TestChecksum:
    """Tests for checksum computation."""

    def test_checksum_deterministic(self):
        """Test checksum is deterministic for same state dict."""
        model = SimpleModel()
        state_dict = model.state_dict()

        checksum1 = compute_state_dict_checksum(state_dict)
        checksum2 = compute_state_dict_checksum(state_dict)

        assert checksum1 == checksum2

    def test_checksum_different_for_different_weights(self):
        """Test checksum differs for different weights."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Reinitialize model2 with different weights
        model2.fc1.weight.data.fill_(0.5)

        checksum1 = compute_state_dict_checksum(model1.state_dict())
        checksum2 = compute_state_dict_checksum(model2.state_dict())

        assert checksum1 != checksum2

    def test_checksum_format(self):
        """Test checksum is a valid hex string."""
        model = SimpleModel()
        checksum = compute_state_dict_checksum(model.state_dict())

        assert len(checksum) == 64  # SHA256 produces 64 hex chars
        assert all(c in '0123456789abcdef' for c in checksum)


# =============================================================================
# ModelVersionManager Tests
# =============================================================================


class TestModelVersionManager:
    """Tests for ModelVersionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        return ModelVersionManager()

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def temp_checkpoint_path(self):
        """Create a temporary file path for checkpoints."""
        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            path = f.name
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    def test_create_metadata(self, manager, simple_model):
        """Test metadata creation from model."""
        metadata = manager.create_metadata(
            simple_model,
            training_info={"epochs": 10},
        )

        assert metadata.architecture_version == "v1.0.0"
        assert metadata.model_class == "SimpleModel"
        assert metadata.checksum != ""
        assert "epochs" in metadata.training_info

    def test_save_and_load_checkpoint(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test saving and loading a versioned checkpoint."""
        # Save
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        assert os.path.exists(temp_checkpoint_path)

        # Load
        state_dict, loaded_meta = manager.load_checkpoint(
            temp_checkpoint_path,
            strict=False,
        )

        arch_version = loaded_meta.architecture_version
        assert arch_version == metadata.architecture_version
        assert loaded_meta.model_class == metadata.model_class
        assert loaded_meta.checksum == metadata.checksum

        # Verify state dict can be loaded into model
        new_model = SimpleModel()
        new_model.load_state_dict(state_dict)

    def test_save_with_optimizer_and_scheduler(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test saving checkpoint with optimizer and scheduler state."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Do a step to create optimizer state
        x = torch.randn(1, 10)
        loss = simple_model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(
            simple_model,
            metadata,
            temp_checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            loss=0.1,
        )

        # Load and verify
        checkpoint = torch.load(temp_checkpoint_path)
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert checkpoint['epoch'] == 5
        assert checkpoint['loss'] == 0.1

    def test_load_with_version_validation_strict(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test strict version validation on load."""
        # Save with version v1.0.0
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        # Load with different expected version should fail
        with pytest.raises(VersionMismatchError) as exc_info:
            manager.load_checkpoint(
                temp_checkpoint_path,
                strict=True,
                expected_version="v2.0.0",
            )

        assert exc_info.value.checkpoint_version == "v1.0.0"
        assert exc_info.value.current_version == "v2.0.0"

    def test_load_with_version_validation_non_strict(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test non-strict version validation logs warning but succeeds."""
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        # Should succeed with warning
        _state_dict, loaded_meta = manager.load_checkpoint(
            temp_checkpoint_path,
            strict=False,
            expected_version="v2.0.0",
        )

        assert loaded_meta.architecture_version == "v1.0.0"

    def test_load_with_class_validation(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test model class validation on load."""
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        with pytest.raises(VersionMismatchError) as exc_info:
            manager.load_checkpoint(
                temp_checkpoint_path,
                strict=True,
                expected_class="DifferentModel",
            )

        details = exc_info.value.details or ""
        assert "Model class mismatch" in details

    def test_checksum_validation(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test checksum validation detects corruption."""
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        # Corrupt the checkpoint by modifying the stored checksum
        checkpoint = torch.load(temp_checkpoint_path)
        checkpoint[manager.METADATA_KEY]['checksum'] = 'corrupted_checksum'
        torch.save(checkpoint, temp_checkpoint_path)

        with pytest.raises(ChecksumMismatchError):
            manager.load_checkpoint(
                temp_checkpoint_path,
                strict=True,
                verify_checksum=True,
            )

    def test_checksum_validation_disabled(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test that checksum validation can be disabled."""
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        # Corrupt checksum
        checkpoint = torch.load(temp_checkpoint_path)
        checkpoint[manager.METADATA_KEY]['checksum'] = 'corrupted'
        torch.save(checkpoint, temp_checkpoint_path)

        # Should succeed with verify_checksum=False
        state_dict, _ = manager.load_checkpoint(
            temp_checkpoint_path,
            strict=False,
            verify_checksum=False,
        )
        assert state_dict is not None

    def test_get_metadata_without_loading_weights(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test extracting metadata without loading full model."""
        metadata = manager.create_metadata(
            simple_model,
            training_info={"special_info": "test"},
        )
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        # Get metadata only
        loaded_meta = manager.get_metadata(temp_checkpoint_path)

        arch_ver = loaded_meta.architecture_version
        assert arch_ver == metadata.architecture_version
        assert loaded_meta.training_info["special_info"] == "test"

    def test_validate_compatibility(
        self, manager, simple_model, temp_checkpoint_path
    ):
        """Test compatibility validation."""
        metadata = manager.create_metadata(simple_model)
        manager.save_checkpoint(simple_model, metadata, temp_checkpoint_path)

        # Compatible config
        compatible_config = {"input_size": 10, "hidden_size": 20}
        assert manager.validate_compatibility(
            compatible_config, temp_checkpoint_path
        )

    def test_file_not_found_handling(self, manager):
        """Test proper error on missing file."""
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint("/nonexistent/path.pth")

        with pytest.raises(FileNotFoundError):
            manager.get_metadata("/nonexistent/path.pth")


# =============================================================================
# Legacy Checkpoint Migration Tests
# =============================================================================


class TestLegacyCheckpointMigration:
    """Tests for legacy checkpoint migration."""

    @pytest.fixture
    def manager(self):
        return ModelVersionManager()

    @pytest.fixture
    def legacy_checkpoint_path(self):
        """Create a legacy checkpoint without versioning metadata."""
        model = SimpleModel()
        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            path = f.name

        # Save legacy format (just state_dict)
        torch.save(model.state_dict(), path)
        yield path

        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def legacy_checkpoint_with_extras_path(self):
        """Create legacy checkpoint with optimizer state."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            path = f.name

        # Save legacy format with extras
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 50,
            'loss': 0.05,
        }
        torch.save(checkpoint, path)
        yield path

        if os.path.exists(path):
            os.unlink(path)

    def test_legacy_checkpoint_raises_error_strict(
        self, manager, legacy_checkpoint_path
    ):
        """Test that legacy checkpoint raises error in strict mode."""
        with pytest.raises(LegacyCheckpointError):
            manager.load_checkpoint(legacy_checkpoint_path, strict=True)

    def test_legacy_checkpoint_loads_non_strict(
        self, manager, legacy_checkpoint_path
    ):
        """Test legacy checkpoint loads with warning in non-strict mode."""
        state_dict, metadata = manager.load_checkpoint(
            legacy_checkpoint_path, strict=False
        )

        assert state_dict is not None
        assert metadata.architecture_version == "v0.0.0-legacy"
        assert metadata.model_class == "Unknown"

    def test_migrate_legacy_checkpoint(
        self, manager, legacy_checkpoint_path
    ):
        """Test migration of legacy checkpoint to versioned format."""
        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            output_path = f.name

        try:
            config = {"input_size": 10, "hidden_size": 20}
            metadata = manager.migrate_legacy_checkpoint(
                legacy_path=legacy_checkpoint_path,
                output_path=output_path,
                model_class="SimpleModel",
                config=config,
                architecture_version="v1.0.0",
            )

            assert metadata.architecture_version == "v1.0.0"
            assert metadata.model_class == "SimpleModel"
            assert metadata.config == config
            assert "migrated_from" in metadata.training_info

            # Verify the migrated checkpoint can be loaded normally
            _state_dict, loaded_meta = manager.load_checkpoint(
                output_path, strict=True
            )
            assert loaded_meta.architecture_version == "v1.0.0"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_migrate_preserves_optimizer_state(
        self, manager, legacy_checkpoint_with_extras_path
    ):
        """Test migration preserves optimizer and epoch info."""
        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            output_path = f.name

        try:
            manager.migrate_legacy_checkpoint(
                legacy_path=legacy_checkpoint_with_extras_path,
                output_path=output_path,
                model_class="SimpleModel",
                config={"input_size": 10},
            )

            checkpoint = torch.load(output_path)
            assert 'optimizer_state_dict' in checkpoint
            assert checkpoint['epoch'] == 50
            assert checkpoint['loss'] == 0.05

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# =============================================================================
# Integration Tests with RingRiftCNN_v2
# =============================================================================


class TestRingRiftCNN_v2Versioning:
    """Tests for versioning with actual RingRiftCNN_v2 model."""

    @pytest.fixture
    def manager(self):
        return ModelVersionManager()

    @pytest.fixture
    def ringrift_model(self):
        """Create a small RingRiftCNN_v2 for testing."""
        return RingRiftCNN_v2(
            board_size=8,
            in_channels=10,
            global_features=10,
            num_res_blocks=2,  # Small for fast testing
            num_filters=32,
            history_length=3,
        )

    @pytest.fixture
    def temp_path(self):
        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_ringrift_version_constant(self, ringrift_model):
        """Test RingRiftCNN_v2 has version constant."""
        assert hasattr(RingRiftCNN_v2, 'ARCHITECTURE_VERSION')
        assert RingRiftCNN_v2.ARCHITECTURE_VERSION == RINGRIFT_CNN_V2_VERSION

    def test_ringrift_get_model_version(self, ringrift_model):
        """Test get_model_version works with RingRiftCNN_v2."""
        version = get_model_version(ringrift_model)
        assert version == RingRiftCNN_v2.ARCHITECTURE_VERSION

    def test_ringrift_get_model_config(self, ringrift_model):
        """Test get_model_config extracts correct config."""
        config = get_model_config(ringrift_model)

        assert config['board_size'] == 8
        # V2 model stores total_in_channels which is in_channels * (history_length + 1)
        assert 'total_in_channels' in config
        assert config["global_features"] == 10
        assert config["history_length"] == 3
        assert config['num_filters'] == 32
        assert config['num_res_blocks'] == 2
        # Policy size depends on board_size - just verify it's present
        assert 'policy_size' in config

    def test_neural_netai_uses_checkpoint_metadata_to_build_matching_model(
        self,
        manager: ModelVersionManager,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        """NeuralNetAI should construct an architecture that matches the checkpoint.

        This guards against the class of failures where a checkpoint has
        ``num_filters=192`` (value_fc1 in_features=212) but the runtime
        accidentally constructs the historical 128-filter model
        (value_fc1 in_features=148), causing neural tiers to fall back to
        heuristics.
        """
        # Force CPU to make the test deterministic across environments.
        monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

        # Create an isolated ai-service root with a models/ directory.
        base_dir = tmp_path / "ai-service"
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create a tiny RingRiftCNN_v2 checkpoint with canonical channel counts.
        # Keep it small so the test is fast, but non-default so we assert that
        # metadata is actually being used.
        num_filters = 24
        global_features = 20
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=14,
            global_features=global_features,
            num_res_blocks=2,
            num_filters=num_filters,
            history_length=3,
            policy_size=123,
        )
        metadata = manager.create_metadata(model)

        # Verify metadata captures the constants we rely on for compatibility checks.
        assert metadata.config["global_features"] == global_features
        assert metadata.config["history_length"] == 3

        ckpt_path = models_dir / "test_model.pth"
        manager.save_checkpoint(model, metadata, str(ckpt_path))

        # Clear global model cache so we don't pick up a cached real model.
        neural_net_mod._MODEL_CACHE.clear()

        from app.ai.neural_net import NeuralNetAI
        from app.models import AIConfig, BoardType

        cfg = AIConfig(
            difficulty=6,
            randomness=0.0,
            think_time=0,
            rng_seed=1,
            use_neural_net=True,
            nn_model_id="test_model",
            allow_fresh_weights=False,
        )
        nn = NeuralNetAI(player_number=1, config=cfg)
        nn._base_dir = str(base_dir)
        nn._ensure_model_initialized(BoardType.SQUARE8)

        assert nn.model is not None
        assert nn.model.value_fc1.in_features == num_filters + global_features

    def test_ringrift_save_and_load(
        self, manager, ringrift_model, temp_path
    ):
        """Test saving and loading RingRiftCNN_v2 with versioning."""
        # Save
        metadata = manager.create_metadata(
            ringrift_model,
            training_info={"epochs": 100, "val_loss": 0.05},
        )
        manager.save_checkpoint(ringrift_model, metadata, temp_path)

        # Load into new model
        new_model = RingRiftCNN_v2(
            board_size=8,
            in_channels=10,
            global_features=10,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )

        state_dict, loaded_meta = manager.load_checkpoint(
            temp_path,
            strict=True,
            expected_version=RingRiftCNN_v2.ARCHITECTURE_VERSION,
            expected_class="RingRiftCNN_v2",
        )
        new_model.load_state_dict(state_dict)

        assert loaded_meta.model_class == "RingRiftCNN_v2"
        assert loaded_meta.training_info["epochs"] == 100

    def test_ringrift_architecture_mismatch_detection(
        self, manager, ringrift_model, temp_path
    ):
        """Test detection of architecture mismatch for RingRiftCNN_v2."""
        # Save model with specific config
        metadata = manager.create_metadata(ringrift_model)
        manager.save_checkpoint(ringrift_model, metadata, temp_path)

        # Try to load with different config
        different_model_config = {
            'board_size': 8,
            'num_filters': 64,  # Different!
            'num_res_blocks': 2,
        }

        # Compatibility check should fail
        assert not manager.validate_compatibility(
            different_model_config, temp_path
        )


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def temp_path(self):
        with tempfile.NamedTemporaryFile(
            suffix='.pth', delete=False
        ) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_save_model_checkpoint(self, temp_path):
        """Test save_model_checkpoint convenience function."""
        model = SimpleModel()

        metadata = save_model_checkpoint(
            model,
            temp_path,
            training_info={"test": True},
            epoch=10,
            loss=0.1,
        )

        assert metadata.architecture_version == "v1.0.0"
        assert os.path.exists(temp_path)

    def test_load_model_with_validation(self, temp_path):
        """Test load_model_with_validation convenience function."""
        model = SimpleModel()

        # Save
        save_model_checkpoint(model, temp_path)

        # Load
        new_model = SimpleModel()
        loaded_model, metadata = load_model_with_validation(
            new_model, temp_path, strict=False
        )

        assert loaded_model is new_model
        assert metadata.model_class == "SimpleModel"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def manager(self):
        return ModelVersionManager()

    def test_empty_config(self, manager):
        """Test handling of empty config."""
        metadata = ModelMetadata(
            architecture_version="v1.0.0",
            model_class="Test",
            config={},
        )
        assert metadata.config == {}

    def test_special_characters_in_path(self, manager):
        """Test handling of special characters in file path."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model with spaces.pth")

            metadata = manager.create_metadata(model)
            manager.save_checkpoint(model, metadata, path)

            state_dict, _ = manager.load_checkpoint(path, strict=False)
            assert state_dict is not None

    def test_nested_directory_creation(self, manager):
        """Test automatic directory creation for nested paths."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "model.pth")

            metadata = manager.create_metadata(model)
            manager.save_checkpoint(model, metadata, path)

            assert os.path.exists(path)

    def test_large_training_info(self, manager):
        """Test handling of large training info dict."""
        model = SimpleModel()

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name

        try:
            # Large training info
            training_info: dict = {
                f"metric_{i}": i * 0.1
                for i in range(1000)
            }
            training_info["history"] = list(range(10000))

            metadata = manager.create_metadata(
                model, training_info=training_info
            )
            manager.save_checkpoint(model, metadata, path)

            _, loaded_meta = manager.load_checkpoint(path, strict=False)
            assert len(loaded_meta.training_info["history"]) == 10000

        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
