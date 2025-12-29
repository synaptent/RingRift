"""Unit tests for app/training/checkpoint_unified.py module.

Tests cover:
- CheckpointType enum
- CheckpointMetadata dataclass
- TrainingProgress dataclass
- UnifiedCheckpointConfig dataclass
- create_checkpoint_manager factory function

December 2025: Created as part of training module test coverage initiative.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCheckpointType:
    """Tests for CheckpointType enum."""

    def test_checkpoint_types_exist(self):
        """All checkpoint types are defined."""
        from app.training.checkpoint_unified import CheckpointType

        assert CheckpointType.REGULAR.value == "regular"
        assert CheckpointType.EPOCH.value == "epoch"
        assert CheckpointType.BEST.value == "best"
        assert CheckpointType.EMERGENCY.value == "emergency"
        assert CheckpointType.RECOVERY.value == "recovery"

    def test_checkpoint_type_count(self):
        """Exactly 5 checkpoint types exist."""
        from app.training.checkpoint_unified import CheckpointType

        assert len(list(CheckpointType)) == 5


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_creation(self):
        """CheckpointMetadata can be created."""
        from app.training.checkpoint_unified import (
            CheckpointMetadata,
            CheckpointType,
        )

        now = datetime.now()
        metadata = CheckpointMetadata(
            checkpoint_id="ckpt-001",
            checkpoint_type=CheckpointType.EPOCH,
            epoch=5,
            global_step=1000,
            timestamp=now,
            metrics={"loss": 0.5, "accuracy": 0.9},
            training_config={"batch_size": 512},
            file_path="/path/to/checkpoint.pt",
            file_hash="abc123",
        )

        assert metadata.checkpoint_id == "ckpt-001"
        assert metadata.checkpoint_type == CheckpointType.EPOCH
        assert metadata.epoch == 5
        assert metadata.global_step == 1000
        assert metadata.timestamp == now
        assert metadata.metrics["loss"] == 0.5
        assert metadata.file_path == "/path/to/checkpoint.pt"
        assert metadata.parent_checkpoint is None

    def test_to_dict(self):
        """CheckpointMetadata.to_dict() serializes correctly."""
        from app.training.checkpoint_unified import (
            CheckpointMetadata,
            CheckpointType,
        )

        now = datetime.now()
        metadata = CheckpointMetadata(
            checkpoint_id="ckpt-002",
            checkpoint_type=CheckpointType.BEST,
            epoch=10,
            global_step=2000,
            timestamp=now,
            metrics={"loss": 0.3},
            training_config={},
            file_path="/path/to/best.pt",
            file_hash="def456",
        )

        d = metadata.to_dict()

        assert d["checkpoint_id"] == "ckpt-002"
        assert d["checkpoint_type"] == "best"  # String value, not enum
        assert d["epoch"] == 10
        assert d["timestamp"] == now.isoformat()

    def test_from_dict(self):
        """CheckpointMetadata.from_dict() deserializes correctly."""
        from app.training.checkpoint_unified import (
            CheckpointMetadata,
            CheckpointType,
        )

        now = datetime.now()
        d = {
            "checkpoint_id": "ckpt-003",
            "checkpoint_type": "emergency",
            "epoch": 15,
            "global_step": 3000,
            "timestamp": now.isoformat(),
            "metrics": {"loss": 0.2},
            "training_config": {"lr": 0.001},
            "file_path": "/path/to/emergency.pt",
            "file_hash": "ghi789",
        }

        metadata = CheckpointMetadata.from_dict(d)

        assert metadata.checkpoint_id == "ckpt-003"
        assert metadata.checkpoint_type == CheckpointType.EMERGENCY
        assert metadata.epoch == 15
        assert metadata.timestamp == now

    def test_round_trip(self):
        """CheckpointMetadata serializes and deserializes correctly."""
        from app.training.checkpoint_unified import (
            CheckpointMetadata,
            CheckpointType,
        )

        now = datetime.now()
        original = CheckpointMetadata(
            checkpoint_id="ckpt-roundtrip",
            checkpoint_type=CheckpointType.REGULAR,
            epoch=20,
            global_step=4000,
            timestamp=now,
            metrics={"loss": 0.1, "acc": 0.95},
            training_config={"batch_size": 1024},
            file_path="/path/to/checkpoint.pt",
            file_hash="roundtrip123",
            parent_checkpoint="ckpt-prev",
        )

        d = original.to_dict()
        restored = CheckpointMetadata.from_dict(d)

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.checkpoint_type == original.checkpoint_type
        assert restored.epoch == original.epoch
        assert restored.parent_checkpoint == original.parent_checkpoint


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_default_creation(self):
        """TrainingProgress has sensible defaults."""
        from app.training.checkpoint_unified import TrainingProgress

        progress = TrainingProgress()

        assert progress.epoch == 0
        assert progress.global_step == 0
        assert progress.batch_idx == 0
        assert progress.samples_seen == 0
        assert progress.best_metric is None
        assert progress.best_metric_name == "loss"
        assert progress.best_epoch == 0
        assert progress.total_epochs == 100
        assert progress.learning_rate == 0.001
        assert progress.optimizer_state is None
        assert progress.scheduler_state is None
        assert progress.random_state is None
        assert progress.extra_state == {}

    def test_custom_creation(self):
        """TrainingProgress can be created with custom values."""
        from app.training.checkpoint_unified import TrainingProgress

        progress = TrainingProgress(
            epoch=10,
            global_step=5000,
            samples_seen=50000,
            best_metric=0.85,
            best_metric_name="accuracy",
            learning_rate=0.0001,
        )

        assert progress.epoch == 10
        assert progress.global_step == 5000
        assert progress.best_metric == 0.85
        assert progress.learning_rate == 0.0001

    def test_to_dict(self):
        """TrainingProgress.to_dict() serializes correctly."""
        from app.training.checkpoint_unified import TrainingProgress

        progress = TrainingProgress(epoch=5, global_step=1000)
        d = progress.to_dict()

        assert d["epoch"] == 5
        assert d["global_step"] == 1000
        assert "optimizer_state" in d

    def test_from_dict(self):
        """TrainingProgress.from_dict() deserializes correctly."""
        from app.training.checkpoint_unified import TrainingProgress

        d = {
            "epoch": 15,
            "global_step": 3000,
            "batch_idx": 50,
            "samples_seen": 30000,
            "best_metric": 0.9,
            "best_metric_name": "accuracy",
            "best_epoch": 12,
            "total_epochs": 50,
            "learning_rate": 0.0005,
        }

        progress = TrainingProgress.from_dict(d)

        assert progress.epoch == 15
        assert progress.global_step == 3000
        assert progress.best_metric == 0.9

    def test_from_dict_extra_keys(self):
        """TrainingProgress.from_dict() ignores extra keys."""
        from app.training.checkpoint_unified import TrainingProgress

        d = {
            "epoch": 5,
            "global_step": 1000,
            "unknown_key": "should be ignored",
            "another_extra": 123,
        }

        # Should not raise
        progress = TrainingProgress.from_dict(d)
        assert progress.epoch == 5


class TestUnifiedCheckpointConfig:
    """Tests for UnifiedCheckpointConfig dataclass."""

    def test_default_values(self):
        """UnifiedCheckpointConfig has sensible defaults."""
        from app.training.checkpoint_unified import UnifiedCheckpointConfig

        config = UnifiedCheckpointConfig()

        assert config.checkpoint_dir == Path("checkpoints")
        assert config.max_checkpoints == 10
        assert config.keep_best == 3
        assert config.keep_every_n_epochs == 10
        assert config.adaptive_enabled is True
        assert config.min_interval_epochs == 1
        assert config.max_interval_epochs == 10
        assert config.improvement_threshold == 0.01
        assert config.checkpoint_interval_steps == 1000
        assert config.verify_hash is True
        assert config.async_save is False

    def test_custom_values(self):
        """UnifiedCheckpointConfig can be customized."""
        from app.training.checkpoint_unified import UnifiedCheckpointConfig

        config = UnifiedCheckpointConfig(
            checkpoint_dir=Path("/custom/path"),
            max_checkpoints=20,
            keep_best=5,
            adaptive_enabled=False,
            async_save=True,
        )

        assert config.checkpoint_dir == Path("/custom/path")
        assert config.max_checkpoints == 20
        assert config.keep_best == 5
        assert config.adaptive_enabled is False
        assert config.async_save is True


class TestCreateCheckpointManager:
    """Tests for create_checkpoint_manager factory function."""

    def test_creates_manager(self):
        """create_checkpoint_manager creates a manager instance."""
        from app.training.checkpoint_unified import create_checkpoint_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_checkpoint_manager(checkpoint_dir=tmpdir)
            assert manager is not None

    def test_with_custom_options(self):
        """create_checkpoint_manager accepts custom options."""
        from app.training.checkpoint_unified import create_checkpoint_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_checkpoint_manager(
                checkpoint_dir=tmpdir,
                keep_best=5,
                adaptive_enabled=False,
            )
            assert manager is not None
            assert manager.config.keep_best == 5
            assert manager.config.adaptive_enabled is False

    def test_creates_directory(self):
        """create_checkpoint_manager creates checkpoint directory."""
        from app.training.checkpoint_unified import create_checkpoint_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "nested" / "checkpoints"
            manager = create_checkpoint_manager(checkpoint_dir=str(ckpt_dir))

            # Directory should be created
            assert ckpt_dir.exists()


class TestUnifiedCheckpointManagerBasic:
    """Basic tests for UnifiedCheckpointManager class."""

    def test_initialization(self):
        """UnifiedCheckpointManager initializes correctly."""
        from app.training.checkpoint_unified import (
            UnifiedCheckpointConfig,
            UnifiedCheckpointManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UnifiedCheckpointConfig(checkpoint_dir=Path(tmpdir))
            manager = UnifiedCheckpointManager(config)

            assert manager.config == config
            assert manager.checkpoint_dir == Path(tmpdir)

    def test_checkpoint_dir_creation(self):
        """UnifiedCheckpointManager creates checkpoint directory."""
        from app.training.checkpoint_unified import (
            UnifiedCheckpointConfig,
            UnifiedCheckpointManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "new_checkpoints"
            config = UnifiedCheckpointConfig(checkpoint_dir=ckpt_dir)
            manager = UnifiedCheckpointManager(config)

            assert ckpt_dir.exists()

    def test_list_checkpoints_empty(self):
        """list_checkpoints returns empty list when no checkpoints exist."""
        from app.training.checkpoint_unified import (
            UnifiedCheckpointConfig,
            UnifiedCheckpointManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UnifiedCheckpointConfig(checkpoint_dir=Path(tmpdir))
            manager = UnifiedCheckpointManager(config)

            checkpoints = manager.list_checkpoints()
            assert checkpoints == []

    def test_get_latest_checkpoint_none(self):
        """get_latest_checkpoint returns None when no checkpoints exist."""
        from app.training.checkpoint_unified import (
            UnifiedCheckpointConfig,
            UnifiedCheckpointManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = UnifiedCheckpointConfig(checkpoint_dir=Path(tmpdir))
            manager = UnifiedCheckpointManager(config)

            latest = manager.get_latest_checkpoint()
            assert latest is None
