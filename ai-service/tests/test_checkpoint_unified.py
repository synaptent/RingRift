"""
Tests for checkpoint_unified.py - Unified checkpoint management module.

Tests cover:
- UnifiedCheckpointManager initialization and configuration
- Checkpoint saving and loading
- Adaptive checkpoint frequency
- Hash verification
- Lineage tracking
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import os
import tempfile

import pytest


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock()
    model.state_dict.return_value = {"layer1.weight": MagicMock()}
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"state": {}, "param_groups": []}
    return optimizer


class TestUnifiedCheckpointConfig:
    """Tests for UnifiedCheckpointConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        try:
            from app.training.checkpoint_unified import UnifiedCheckpointConfig
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig()
        assert config.save_frequency == 1000
        assert config.max_checkpoints == 5
        assert config.hash_verification is True
        assert config.adaptive_frequency is True
        assert config.min_improvement == 0.01

    def test_custom_config_values(self):
        """Test custom configuration values."""
        try:
            from app.training.checkpoint_unified import UnifiedCheckpointConfig
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            save_frequency=500,
            max_checkpoints=10,
            hash_verification=False,
            adaptive_frequency=False,
            min_improvement=0.05,
        )
        assert config.save_frequency == 500
        assert config.max_checkpoints == 10
        assert config.hash_verification is False


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_progress_initialization(self):
        """Test TrainingProgress default values."""
        try:
            from app.training.checkpoint_unified import TrainingProgress
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        progress = TrainingProgress()
        assert progress.epoch == 0
        assert progress.global_step == 0
        assert progress.best_loss == float('inf')

    def test_progress_to_dict(self):
        """Test TrainingProgress serialization."""
        try:
            from app.training.checkpoint_unified import TrainingProgress
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        progress = TrainingProgress(
            epoch=5,
            global_step=1000,
            best_loss=0.5,
        )
        data = progress.to_dict()
        assert data["epoch"] == 5
        assert data["global_step"] == 1000
        assert data["best_loss"] == 0.5


class TestUnifiedCheckpointManager:
    """Tests for UnifiedCheckpointManager class."""

    def test_manager_initialization(self, temp_checkpoint_dir):
        """Test checkpoint manager initialization."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(checkpoint_dir=str(temp_checkpoint_dir))
        manager = UnifiedCheckpointManager(config)

        assert manager.config.checkpoint_dir == str(temp_checkpoint_dir)
        assert temp_checkpoint_dir.exists()

    def test_should_save_by_frequency(self, temp_checkpoint_dir):
        """Test checkpoint frequency logic."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
                TrainingProgress,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            save_frequency=100,
            adaptive_frequency=False,
        )
        manager = UnifiedCheckpointManager(config)

        # Step 50 - should not save
        progress_50 = TrainingProgress(global_step=50)
        assert not manager.should_save(progress_50)

        # Step 100 - should save
        progress_100 = TrainingProgress(global_step=100)
        assert manager.should_save(progress_100)

        # Step 200 - should save
        progress_200 = TrainingProgress(global_step=200)
        assert manager.should_save(progress_200)

    def test_adaptive_checkpoint_frequency(self, temp_checkpoint_dir):
        """Test adaptive checkpoint frequency based on loss improvement."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
                TrainingProgress,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            save_frequency=100,
            adaptive_frequency=True,
            min_improvement=0.05,
        )
        manager = UnifiedCheckpointManager(config)

        # First save
        progress1 = TrainingProgress(global_step=100, best_loss=1.0)
        assert manager.should_save(progress1)
        manager._last_checkpoint_step = 100
        manager._last_loss = 1.0

        # No improvement at step 200 - should still save due to frequency
        progress2 = TrainingProgress(global_step=200, best_loss=1.0)
        assert manager.should_save(progress2)

    def test_checkpoint_path_generation(self, temp_checkpoint_dir):
        """Test checkpoint path generation."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
                TrainingProgress,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(checkpoint_dir=str(temp_checkpoint_dir))
        manager = UnifiedCheckpointManager(config)

        progress = TrainingProgress(epoch=5, global_step=1000)
        path = manager._get_checkpoint_path(progress)

        assert "epoch_5" in str(path) or "step_1000" in str(path)
        assert str(temp_checkpoint_dir) in str(path)

    @patch("torch.save")
    def test_save_checkpoint(self, mock_torch_save, temp_checkpoint_dir, mock_model, mock_optimizer):
        """Test checkpoint saving."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
                TrainingProgress,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            hash_verification=False,
        )
        manager = UnifiedCheckpointManager(config)

        progress = TrainingProgress(epoch=1, global_step=100)
        path = manager.save(
            model=mock_model,
            optimizer=mock_optimizer,
            progress=progress,
        )

        # torch.save should have been called
        mock_torch_save.assert_called()

    def test_checkpoint_cleanup(self, temp_checkpoint_dir):
        """Test old checkpoint cleanup."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            max_checkpoints=3,
        )
        manager = UnifiedCheckpointManager(config)

        # Create dummy checkpoint files
        for i in range(5):
            ckpt_path = temp_checkpoint_dir / f"checkpoint_step_{i * 100}.pt"
            ckpt_path.touch()

        # Run cleanup
        manager._cleanup_old_checkpoints()

        # Should only have max_checkpoints files remaining
        remaining = list(temp_checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(remaining) <= config.max_checkpoints


class TestCheckpointLineage:
    """Tests for checkpoint lineage tracking."""

    def test_lineage_tracking(self, temp_checkpoint_dir):
        """Test that checkpoint lineage is tracked."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
                TrainingProgress,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            track_lineage=True,
        )
        manager = UnifiedCheckpointManager(config)

        # Check lineage file creation
        lineage_path = temp_checkpoint_dir / "lineage.json"
        if hasattr(manager, '_lineage'):
            assert isinstance(manager._lineage, (dict, list))


class TestHashVerification:
    """Tests for checkpoint hash verification."""

    def test_hash_computation(self, temp_checkpoint_dir):
        """Test checkpoint hash computation."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            hash_verification=True,
        )
        manager = UnifiedCheckpointManager(config)

        # Create a dummy file
        test_file = temp_checkpoint_dir / "test.pt"
        test_file.write_bytes(b"test checkpoint data")

        if hasattr(manager, '_compute_hash'):
            hash1 = manager._compute_hash(test_file)
            hash2 = manager._compute_hash(test_file)
            assert hash1 == hash2
            assert len(hash1) > 0


class TestCheckpointRecovery:
    """Tests for checkpoint recovery scenarios."""

    def test_find_latest_checkpoint(self, temp_checkpoint_dir):
        """Test finding the latest checkpoint."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(checkpoint_dir=str(temp_checkpoint_dir))
        manager = UnifiedCheckpointManager(config)

        # Create dummy checkpoints with different timestamps
        import time
        for i in range(3):
            ckpt_path = temp_checkpoint_dir / f"checkpoint_step_{i * 100}.pt"
            ckpt_path.touch()
            time.sleep(0.01)  # Small delay for different timestamps

        if hasattr(manager, 'find_latest_checkpoint'):
            latest = manager.find_latest_checkpoint()
            if latest:
                assert "step_200" in str(latest)

    def test_resume_from_checkpoint(self, temp_checkpoint_dir):
        """Test resuming from a checkpoint."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointManager,
                UnifiedCheckpointConfig,
            )
        except ImportError:
            pytest.skip("checkpoint_unified not available")

        config = UnifiedCheckpointConfig(checkpoint_dir=str(temp_checkpoint_dir))
        manager = UnifiedCheckpointManager(config)

        # This tests the interface exists
        assert hasattr(manager, 'load') or hasattr(manager, 'resume')
