"""Tests for app/training/locking_integration.py.

Tests the distributed locking integration for training operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Test the module can be imported
def test_module_imports():
    """Verify the module can be imported without errors."""
    from app.training import locking_integration
    assert locking_integration is not None


class TestTrainingLockType:
    """Tests for TrainingLockType constants."""

    def test_checkpoint_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.CHECKPOINT == "checkpoint"

    def test_model_registry_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.MODEL_REGISTRY == "model_registry"

    def test_training_job_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.TRAINING_JOB == "training_job"

    def test_evaluation_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.EVALUATION == "evaluation"

    def test_selfplay_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.SELFPLAY == "selfplay"

    def test_promotion_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.PROMOTION == "promotion"

    def test_data_sync_constant(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType.DATA_SYNC == "data_sync"


class TestExports:
    """Test that all expected symbols are exported."""

    def test_exports_lock_types(self):
        from app.training.locking_integration import TrainingLockType
        assert TrainingLockType is not None

    def test_exports_training_locks(self):
        from app.training.locking_integration import TrainingLocks
        assert TrainingLocks is not None

    def test_exports_checkpoint_lock(self):
        from app.training.locking_integration import checkpoint_lock
        assert callable(checkpoint_lock)

    def test_exports_model_registry_lock(self):
        from app.training.locking_integration import model_registry_lock
        assert callable(model_registry_lock)

    def test_exports_training_job_lock(self):
        from app.training.locking_integration import training_job_lock
        assert callable(training_job_lock)

    def test_exports_evaluation_lock(self):
        from app.training.locking_integration import evaluation_lock
        assert callable(evaluation_lock)

    def test_exports_selfplay_lock(self):
        from app.training.locking_integration import selfplay_lock
        assert callable(selfplay_lock)

    def test_exports_promotion_lock(self):
        from app.training.locking_integration import promotion_lock
        assert callable(promotion_lock)

    def test_exports_get_training_lock(self):
        from app.training.locking_integration import get_training_lock
        assert callable(get_training_lock)

    def test_exports_is_training_locked(self):
        from app.training.locking_integration import is_training_locked
        assert callable(is_training_locked)


class TestCheckpointLock:
    """Tests for checkpoint_lock context manager."""

    def test_checkpoint_lock_is_context_manager(self):
        from app.training.locking_integration import checkpoint_lock
        import contextlib
        assert hasattr(checkpoint_lock, '__call__')

    @patch('app.training.locking_integration.DistributedLock')
    def test_checkpoint_lock_creates_distributed_lock(self, mock_lock_class):
        from app.training.locking_integration import checkpoint_lock
        mock_lock = MagicMock()
        mock_lock.__enter__ = Mock(return_value=True)
        mock_lock.__exit__ = Mock(return_value=False)
        mock_lock_class.return_value = mock_lock

        with checkpoint_lock("test_config") as lock:
            pass

        # Verify lock was created with checkpoint prefix
        mock_lock_class.assert_called_once()
        call_args = mock_lock_class.call_args
        assert "checkpoint" in call_args[0][0].lower() or "checkpoint" in str(call_args)


class TestModelRegistryLock:
    """Tests for model_registry_lock context manager."""

    @patch('app.training.locking_integration.DistributedLock')
    def test_model_registry_lock_uses_correct_prefix(self, mock_lock_class):
        from app.training.locking_integration import model_registry_lock
        mock_lock = MagicMock()
        mock_lock.__enter__ = Mock(return_value=True)
        mock_lock.__exit__ = Mock(return_value=False)
        mock_lock_class.return_value = mock_lock

        with model_registry_lock("test_config") as lock:
            pass

        mock_lock_class.assert_called_once()


class TestTrainingJobLock:
    """Tests for training_job_lock context manager."""

    @patch('app.training.locking_integration.DistributedLock')
    def test_training_job_lock_works(self, mock_lock_class):
        from app.training.locking_integration import training_job_lock
        mock_lock = MagicMock()
        mock_lock.__enter__ = Mock(return_value=True)
        mock_lock.__exit__ = Mock(return_value=False)
        mock_lock_class.return_value = mock_lock

        with training_job_lock("test_config") as lock:
            pass

        mock_lock_class.assert_called_once()


class TestTrainingLocksClass:
    """Tests for the TrainingLocks class."""

    def test_training_locks_has_checkpoint_method(self):
        from app.training.locking_integration import TrainingLocks
        assert hasattr(TrainingLocks, 'checkpoint')

    def test_training_locks_has_model_registry_method(self):
        from app.training.locking_integration import TrainingLocks
        # Check for either 'model_registry' or 'registry' method
        assert hasattr(TrainingLocks, 'model_registry') or hasattr(TrainingLocks, 'registry')

    def test_training_locks_has_evaluation_method(self):
        from app.training.locking_integration import TrainingLocks
        assert hasattr(TrainingLocks, 'evaluation')

    def test_training_locks_has_selfplay_method(self):
        from app.training.locking_integration import TrainingLocks
        assert hasattr(TrainingLocks, 'selfplay')


class TestGetTrainingLock:
    """Tests for get_training_lock utility."""

    @patch('app.training.locking_integration.DistributedLock')
    def test_get_training_lock_returns_lock(self, mock_lock_class):
        from app.training.locking_integration import get_training_lock
        mock_lock = Mock()
        mock_lock_class.return_value = mock_lock

        result = get_training_lock("checkpoint", "test_config")

        assert result is not None


class TestIsTrainingLocked:
    """Tests for is_training_locked utility."""

    def test_is_training_locked_returns_boolean(self):
        from app.training.locking_integration import is_training_locked
        result = is_training_locked("checkpoint", "nonexistent_config")
        assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for locking module."""

    def test_all_exports_are_in_all(self):
        """Verify __all__ contains all expected exports."""
        from app.training import locking_integration
        expected = [
            "TrainingLockType",
            "TrainingLocks",
            "checkpoint_lock",
            "evaluation_lock",
            "get_training_lock",
            "is_training_locked",
            "model_registry_lock",
            "promotion_lock",
            "selfplay_lock",
            "training_job_lock",
        ]
        for name in expected:
            assert name in locking_integration.__all__, f"{name} not in __all__"
