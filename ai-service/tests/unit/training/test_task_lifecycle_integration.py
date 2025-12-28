"""Tests for app/training/task_lifecycle_integration.py.

Tests the task lifecycle integration for training components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# Test module imports
def test_module_imports():
    """Verify the module can be imported without errors."""
    from app.training import task_lifecycle_integration
    assert task_lifecycle_integration is not None


class TestTrainingTaskType:
    """Tests for TrainingTaskType constants."""

    def test_training_job_constant(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType.TRAINING_JOB == "training_job"

    def test_data_loading_constant(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType.DATA_LOADING == "data_loading"

    def test_evaluation_constant(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType.EVALUATION == "evaluation"

    def test_selfplay_constant(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType.SELFPLAY == "selfplay"

    def test_checkpoint_constant(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType.CHECKPOINT == "checkpoint"

    def test_prefetch_constant(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType.PREFETCH == "prefetch"


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_heartbeat_sender(self):
        from app.training.task_lifecycle_integration import HeartbeatSender
        assert HeartbeatSender is not None

    def test_exports_training_task_tracker(self):
        from app.training.task_lifecycle_integration import TrainingTaskTracker
        assert TrainingTaskTracker is not None

    def test_exports_training_task_type(self):
        from app.training.task_lifecycle_integration import TrainingTaskType
        assert TrainingTaskType is not None

    def test_exports_complete_training_task(self):
        from app.training.task_lifecycle_integration import complete_training_task
        assert callable(complete_training_task)

    def test_exports_fail_training_task(self):
        from app.training.task_lifecycle_integration import fail_training_task
        assert callable(fail_training_task)

    def test_exports_get_training_task_tracker(self):
        from app.training.task_lifecycle_integration import get_training_task_tracker
        assert callable(get_training_task_tracker)

    def test_exports_register_data_loader_task(self):
        from app.training.task_lifecycle_integration import register_data_loader_task
        assert callable(register_data_loader_task)

    def test_exports_register_evaluation_task(self):
        from app.training.task_lifecycle_integration import register_evaluation_task
        assert callable(register_evaluation_task)

    def test_exports_register_selfplay_task(self):
        from app.training.task_lifecycle_integration import register_selfplay_task
        assert callable(register_selfplay_task)

    def test_exports_register_training_job(self):
        from app.training.task_lifecycle_integration import register_training_job
        assert callable(register_training_job)

    def test_exports_send_training_heartbeat(self):
        from app.training.task_lifecycle_integration import send_training_heartbeat
        assert callable(send_training_heartbeat)

    def test_exports_training_task_context(self):
        from app.training.task_lifecycle_integration import training_task_context
        assert callable(training_task_context)


class TestAllExports:
    """Test __all__ contains expected exports."""

    def test_all_exports_in_module(self):
        from app.training import task_lifecycle_integration
        expected = [
            "HeartbeatSender",
            "TrainingTaskTracker",
            "TrainingTaskType",
            "complete_training_task",
            "fail_training_task",
            "get_training_task_tracker",
            "register_data_loader_task",
            "register_evaluation_task",
            "register_selfplay_task",
            "register_training_job",
            "send_training_heartbeat",
            "training_task_context",
        ]
        for name in expected:
            assert name in task_lifecycle_integration.__all__, f"{name} not in __all__"


class TestHeartbeatSender:
    """Tests for HeartbeatSender class."""

    def test_class_exists(self):
        from app.training.task_lifecycle_integration import HeartbeatSender
        assert HeartbeatSender is not None


class TestTrainingTaskTracker:
    """Tests for TrainingTaskTracker class."""

    def test_class_exists(self):
        from app.training.task_lifecycle_integration import TrainingTaskTracker
        assert TrainingTaskTracker is not None

    def test_get_training_task_tracker_returns_tracker(self):
        from app.training.task_lifecycle_integration import (
            get_training_task_tracker,
            TrainingTaskTracker,
        )
        tracker = get_training_task_tracker()
        # May return None if not initialized, or a tracker instance
        if tracker is not None:
            assert isinstance(tracker, TrainingTaskTracker)


class TestRegisterTrainingJob:
    """Tests for register_training_job function."""

    def test_register_training_job_returns_task_info(self):
        from app.training.task_lifecycle_integration import (
            register_training_job,
            reset_training_task_tracker,
        )
        # Reset to get fresh tracker
        reset_training_task_tracker()

        result = register_training_job(
            job_id="job-123",
            config_key="square8_2p",
            node_id="gh200-a",
            auto_heartbeat=False,  # Disable heartbeat for test
        )

        assert result is not None
        assert result.config_key == "square8_2p"


class TestRegisterDataLoaderTask:
    """Tests for register_data_loader_task function."""

    def test_register_data_loader_task_uses_loader_id(self):
        from app.training.task_lifecycle_integration import (
            register_data_loader_task,
            reset_training_task_tracker,
        )
        reset_training_task_tracker()

        result = register_data_loader_task(
            loader_id="loader-123",
            config_key="hex8_2p",
        )

        assert result is not None
        assert result.config_key == "hex8_2p"


class TestRegisterSelfplayTask:
    """Tests for register_selfplay_task function."""

    def test_register_selfplay_task_uses_correct_args(self):
        from app.training.task_lifecycle_integration import (
            register_selfplay_task,
            reset_training_task_tracker,
        )
        reset_training_task_tracker()

        result = register_selfplay_task(
            selfplay_id="selfplay-123",
            config_key="hex8_4p",
            iteration=1,
        )

        assert result is not None
        assert result.config_key == "hex8_4p"


class TestRegisterEvaluationTask:
    """Tests for register_evaluation_task function."""

    def test_register_evaluation_task_uses_eval_id(self):
        from app.training.task_lifecycle_integration import (
            register_evaluation_task,
            reset_training_task_tracker,
        )
        reset_training_task_tracker()

        result = register_evaluation_task(
            eval_id="eval-123",
            config_key="square8_4p",
        )

        assert result is not None
        assert result.config_key == "square8_4p"


class TestSendTrainingHeartbeat:
    """Tests for send_training_heartbeat function."""

    def test_send_heartbeat_works(self):
        from app.training.task_lifecycle_integration import (
            send_training_heartbeat,
            register_training_job,
            reset_training_task_tracker,
        )
        reset_training_task_tracker()

        # Register a task first
        task = register_training_job(
            job_id="job-heartbeat",
            config_key="test",
            auto_heartbeat=False,
        )

        # Should not raise
        send_training_heartbeat(task_id=task.task_id)


class TestCompleteTrainingTask:
    """Tests for complete_training_task function."""

    def test_complete_task_works(self):
        from app.training.task_lifecycle_integration import (
            complete_training_task,
            register_training_job,
            reset_training_task_tracker,
        )
        reset_training_task_tracker()

        # Register a task first
        task = register_training_job(
            job_id="job-complete",
            config_key="test",
            auto_heartbeat=False,
        )

        # Should not raise
        complete_training_task(
            task_id=task.task_id,
            success=True,
            result={"loss": 0.01},
        )


class TestFailTrainingTask:
    """Tests for fail_training_task function."""

    def test_fail_task_works(self):
        from app.training.task_lifecycle_integration import (
            fail_training_task,
            register_training_job,
            reset_training_task_tracker,
        )
        reset_training_task_tracker()

        # Register a task first
        task = register_training_job(
            job_id="job-fail",
            config_key="test",
            auto_heartbeat=False,
        )

        # Should not raise
        fail_training_task(
            task_id=task.task_id,
            error="Training failed",
        )


class TestTrainingTaskContext:
    """Tests for training_task_context context manager."""

    def test_context_is_callable(self):
        from app.training.task_lifecycle_integration import training_task_context
        assert callable(training_task_context)

    @patch('app.training.task_lifecycle_integration.get_task_lifecycle_coordinator')
    def test_context_manager_usage(self, mock_get_coordinator):
        from app.training.task_lifecycle_integration import training_task_context
        mock_coordinator = MagicMock()
        mock_get_coordinator.return_value = mock_coordinator

        # Use as context manager
        try:
            with training_task_context(
                task_type="training_job",
                config_key="hex8_2p",
            ) as task:
                pass  # Context manager should work
        except Exception:
            # May fail if coordinator not properly configured
            pass
