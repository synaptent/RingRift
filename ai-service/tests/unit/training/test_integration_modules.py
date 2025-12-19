"""Tests for training integration modules.

Tests:
- lifecycle_integration.py
- model_state_machine.py
- thread_integration.py
- event_integration.py
- task_lifecycle_integration.py
- metrics_integration.py
- locking_integration.py
- exception_integration.py
"""

import asyncio
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock


# =============================================================================
# Model State Machine Tests
# =============================================================================

class TestModelStateMachine:
    """Tests for app.training.model_state_machine."""

    def test_model_state_enum(self):
        """Test ModelState enum values."""
        from app.training.model_state_machine import ModelState

        assert ModelState.TRAINING.value == "training"
        assert ModelState.PRODUCTION.value == "production"
        assert ModelState.ROLLED_BACK.value == "rolled_back"

    def test_lifecycle_singleton(self):
        """Test get_model_lifecycle returns singleton."""
        from app.training.model_state_machine import (
            get_model_lifecycle,
            reset_model_lifecycle,
        )

        reset_model_lifecycle()

        lifecycle1 = get_model_lifecycle()
        lifecycle2 = get_model_lifecycle()

        assert lifecycle1 is lifecycle2

        reset_model_lifecycle()

    def test_register_model(self):
        """Test registering a model."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()

        lifecycle.register_model("test_model_1", ModelState.TRAINING)
        record = lifecycle.get_record("test_model_1")

        assert record is not None
        assert record.model_id == "test_model_1"
        assert record.current_state == ModelState.TRAINING

    def test_valid_transition(self):
        """Test valid state transition."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_2", ModelState.TRAINING)

        # TRAINING -> TRAINED is valid
        result = lifecycle.transition("test_model_2", ModelState.TRAINED)
        assert result is True

        record = lifecycle.get_record("test_model_2")
        assert record.current_state == ModelState.TRAINED

    def test_invalid_transition(self):
        """Test invalid state transition raises exception."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
            InvalidTransitionError,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_3", ModelState.TRAINING)

        # TRAINING -> PRODUCTION is NOT valid (skips steps)
        with pytest.raises(InvalidTransitionError):
            lifecycle.transition("test_model_3", ModelState.PRODUCTION)

        # State should remain TRAINING
        record = lifecycle.get_record("test_model_3")
        assert record.current_state == ModelState.TRAINING

    def test_transition_history(self):
        """Test transition history is recorded."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_4", ModelState.TRAINING)
        lifecycle.transition("test_model_4", ModelState.TRAINED, reason="Training complete")

        record = lifecycle.get_record("test_model_4")
        assert len(record.history) >= 1

    def test_get_valid_transitions(self):
        """Test getting valid transitions from a state."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_5", ModelState.EVALUATED)

        valid = lifecycle.get_valid_transitions("test_model_5")
        # From EVALUATED, should be able to go to STAGING
        assert ModelState.STAGING in valid


# =============================================================================
# Lifecycle Integration Tests
# =============================================================================

class TestLifecycleIntegration:
    """Tests for app.training.lifecycle_integration."""

    def test_background_eval_service_init(self):
        """Test BackgroundEvalService initialization."""
        from app.training.lifecycle_integration import BackgroundEvalService

        service = BackgroundEvalService(
            model_getter=lambda: {"state_dict": {}},
            eval_interval=1000,
            games_per_eval=10,
        )

        assert service.name == "background_eval"
        assert service.dependencies == []  # No dependencies without real games

    def test_background_selfplay_service_init(self):
        """Test BackgroundSelfplayService initialization."""
        from app.training.lifecycle_integration import BackgroundSelfplayService

        service = BackgroundSelfplayService(
            config={"board": "square8", "players": 2},
        )

        assert service.name == "background_selfplay"
        assert service.dependencies == []

    def test_training_lifecycle_manager(self):
        """Test TrainingLifecycleManager registration."""
        from app.training.lifecycle_integration import TrainingLifecycleManager

        manager = TrainingLifecycleManager()

        eval_service = manager.register_eval_service(
            model_getter=lambda: {},
            eval_interval=500,
        )

        assert eval_service is not None
        assert "background_eval" in manager._services

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check on service."""
        from app.training.lifecycle_integration import BackgroundEvalService
        from app.core.health import HealthStatus

        service = BackgroundEvalService(
            model_getter=lambda: {},
        )

        # Before start, should be unhealthy
        status = await service.check_health()
        assert status.state.value != "healthy"


# =============================================================================
# Thread Integration Tests
# =============================================================================

class TestThreadIntegration:
    """Tests for app.training.thread_integration."""

    def test_spawn_eval_thread(self):
        """Test spawning evaluation thread."""
        from app.training.thread_integration import (
            spawn_eval_thread,
            get_training_thread_spawner,
            reset_training_thread_spawner,
        )

        reset_training_thread_spawner()

        completed = threading.Event()

        def eval_loop():
            completed.set()

        thread = spawn_eval_thread(
            target=eval_loop,
            name="test_eval_thread",
        )

        # Wait for completion
        completed.wait(timeout=2.0)
        assert completed.is_set()

        reset_training_thread_spawner()

    def test_training_thread_group(self):
        """Test TrainingThreadGroup constants."""
        from app.training.thread_integration import TrainingThreadGroup

        assert TrainingThreadGroup.EVALUATION == "evaluation"
        assert TrainingThreadGroup.DATA_LOADING == "data_loading"
        assert TrainingThreadGroup.CHECKPOINTING == "checkpointing"

    def test_training_thread_spawner_stats(self):
        """Test spawner statistics."""
        from app.training.thread_integration import (
            get_training_thread_spawner,
            reset_training_thread_spawner,
        )

        reset_training_thread_spawner()
        spawner = get_training_thread_spawner()

        stats = spawner.get_stats()
        assert "threads_spawned" in stats
        assert "threads_running" in stats

        reset_training_thread_spawner()


# =============================================================================
# Event Integration Tests
# =============================================================================

class TestEventIntegration:
    """Tests for app.training.event_integration."""

    def test_training_topics(self):
        """Test TrainingTopics constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.TRAINING_STARTED == "training.started"
        assert TrainingTopics.TRAINING_COMPLETED == "training.completed"
        assert TrainingTopics.EVAL_COMPLETED == "training.eval.completed"
        assert TrainingTopics.CHECKPOINT_SAVED == "training.checkpoint.saved"

    def test_training_event_dataclass(self):
        """Test TrainingEvent dataclass."""
        from app.training.event_integration import TrainingStartedEvent

        event = TrainingStartedEvent(
            topic="training.started",
            config_key="square8_2p",
            job_id="job-123",
            total_epochs=100,
            batch_size=256,
        )

        assert event.config_key == "square8_2p"
        assert event.total_epochs == 100

    def test_evaluation_event_dataclass(self):
        """Test EvaluationCompletedEvent dataclass."""
        from app.training.event_integration import EvaluationCompletedEvent

        event = EvaluationCompletedEvent(
            topic="training.eval.completed",
            config_key="square8_2p",
            elo=1650,
            win_rate=0.65,
            games_played=100,
            passes_gating=True,
        )

        assert event.elo == 1650
        assert event.passes_gating is True

    @pytest.mark.asyncio
    async def test_publish_training_started(self):
        """Test publishing training started event."""
        from app.training.event_integration import publish_training_started
        from app.core.event_bus import reset_event_bus

        reset_event_bus()

        count = await publish_training_started(
            config_key="test_config",
            job_id="test_job",
            total_epochs=50,
        )

        # Should return number of handlers notified (may be 0)
        assert count >= 0

        reset_event_bus()


# =============================================================================
# Task Lifecycle Integration Tests
# =============================================================================

class TestTaskLifecycleIntegration:
    """Tests for app.training.task_lifecycle_integration."""

    def test_training_task_type_constants(self):
        """Test TrainingTaskType constants."""
        from app.training.task_lifecycle_integration import TrainingTaskType

        assert TrainingTaskType.TRAINING_JOB == "training_job"
        assert TrainingTaskType.EVALUATION == "evaluation"
        assert TrainingTaskType.SELFPLAY == "selfplay"

    def test_register_training_job(self):
        """Test registering a training job."""
        from app.training.task_lifecycle_integration import (
            TrainingTaskTracker,
            TrainingTaskType,
        )

        tracker = TrainingTaskTracker(node_id="test_node")

        info = tracker.register_job(
            job_id="test_job_1",
            config_key="square8_2p",
            auto_heartbeat=False,  # Don't start heartbeat thread
        )

        assert info.task_id == "training:test_job_1"
        assert info.config_key == "square8_2p"
        assert info.task_type == TrainingTaskType.TRAINING_JOB

        tracker.shutdown()

    def test_heartbeat(self):
        """Test sending heartbeat."""
        from app.training.task_lifecycle_integration import TrainingTaskTracker

        tracker = TrainingTaskTracker(node_id="test_node")

        info = tracker.register_job(
            job_id="test_job_2",
            config_key="square8_2p",
            auto_heartbeat=False,
        )

        result = tracker.heartbeat(info.task_id, step=100)
        assert result is True

        task_info = tracker.get_task(info.task_id)
        assert task_info.step == 100

        tracker.shutdown()

    def test_complete_task(self):
        """Test completing a task."""
        from app.training.task_lifecycle_integration import TrainingTaskTracker

        tracker = TrainingTaskTracker(node_id="test_node")

        info = tracker.register_job(
            job_id="test_job_3",
            config_key="square8_2p",
            auto_heartbeat=False,
        )

        tracker.complete(info.task_id, success=True, result={"loss": 0.01})

        # Task should be removed from tracker
        assert tracker.get_task(info.task_id) is None

        tracker.shutdown()


# =============================================================================
# Metrics Integration Tests
# =============================================================================

class TestMetricsIntegration:
    """Tests for app.training.metrics_integration."""

    def test_training_metric_names(self):
        """Test TrainingMetricNames constants."""
        from app.training.metrics_integration import TrainingMetricNames

        assert TrainingMetricNames.STEPS_TOTAL == "training_steps_total"
        assert TrainingMetricNames.CURRENT_LOSS == "training_loss"
        assert TrainingMetricNames.CURRENT_ELO == "model_elo"

    def test_training_metrics_step(self):
        """Test TrainingMetrics.step() method."""
        from app.training.metrics_integration import TrainingMetrics
        from app.metrics.unified_publisher import reset_metrics_publisher

        reset_metrics_publisher()

        # Should not raise
        TrainingMetrics.step(
            config_key="square8_2p",
            step=1000,
            loss=0.01,
            learning_rate=0.001,
        )

        reset_metrics_publisher()

    def test_training_metrics_evaluation(self):
        """Test TrainingMetrics.evaluation() method."""
        from app.training.metrics_integration import TrainingMetrics
        from app.metrics.unified_publisher import reset_metrics_publisher

        reset_metrics_publisher()

        # Should not raise
        TrainingMetrics.evaluation(
            config_key="square8_2p",
            elo=1650,
            win_rate=0.65,
            games_played=100,
        )

        reset_metrics_publisher()

    def test_epoch_timer_context_manager(self):
        """Test TrainingMetrics.epoch_timer() context manager."""
        from app.training.metrics_integration import TrainingMetrics
        from app.metrics.unified_publisher import reset_metrics_publisher

        reset_metrics_publisher()

        with TrainingMetrics.epoch_timer("square8_2p", epoch=5):
            time.sleep(0.01)  # Small delay

        reset_metrics_publisher()


# =============================================================================
# Locking Integration Tests
# =============================================================================

class TestLockingIntegration:
    """Tests for app.training.locking_integration."""

    def test_training_lock_type_constants(self):
        """Test TrainingLockType constants."""
        from app.training.locking_integration import TrainingLockType

        assert TrainingLockType.CHECKPOINT == "checkpoint"
        assert TrainingLockType.PROMOTION == "promotion"
        assert TrainingLockType.SELFPLAY == "selfplay"

    def test_checkpoint_lock_context_manager(self):
        """Test checkpoint_lock context manager."""
        from app.training.locking_integration import checkpoint_lock

        with checkpoint_lock("test_config", timeout=1) as lock:
            # Should acquire lock
            assert lock is not None
            assert lock.is_held()

        # Lock should be released
        assert not lock.is_held()

    def test_training_locks_class(self):
        """Test TrainingLocks static methods."""
        from app.training.locking_integration import TrainingLocks

        with TrainingLocks.checkpoint("test_config_2", timeout=1) as lock:
            assert lock is not None

    def test_is_training_locked(self):
        """Test is_training_locked function."""
        from app.training.locking_integration import (
            TrainingLocks,
            TrainingLockType,
            is_training_locked,
        )

        # Should not be locked initially
        assert not is_training_locked(TrainingLockType.CHECKPOINT, "test_config_3")

        # Lock and check
        with TrainingLocks.checkpoint("test_config_3", timeout=1):
            assert is_training_locked(TrainingLockType.CHECKPOINT, "test_config_3")


# =============================================================================
# Exception Integration Tests
# =============================================================================

class TestExceptionIntegration:
    """Tests for app.training.exception_integration."""

    def test_training_exception_types(self):
        """Test training exception types."""
        from app.training.exception_integration import (
            TrainingError,
            CheckpointError,
            EvaluationError,
            SelfplayError,
            DataLoadError,
        )

        assert issubclass(TrainingError, Exception)
        assert issubclass(CheckpointError, TrainingError)
        assert issubclass(EvaluationError, TrainingError)

    def test_training_retry_policies(self):
        """Test TrainingRetryPolicies."""
        from app.training.exception_integration import TrainingRetryPolicies
        from app.core.error_handler import RetryStrategy

        assert TrainingRetryPolicies.CHECKPOINT_SAVE.max_attempts == 3
        assert TrainingRetryPolicies.DATA_LOAD.strategy == RetryStrategy.EXPONENTIAL_JITTER

    def test_retry_checkpoint_save_decorator(self):
        """Test retry_checkpoint_save decorator."""
        from app.training.exception_integration import retry_checkpoint_save

        call_count = 0

        @retry_checkpoint_save
        def flaky_save():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Transient error")
            return True

        result = flaky_save()
        assert result is True
        assert call_count == 2  # One failure, one success

    def test_safe_training_step(self):
        """Test safe_training_step wrapper."""
        from app.training.exception_integration import safe_training_step

        def failing_step():
            raise RuntimeError("GPU error")

        result = safe_training_step(failing_step, default=-1, log_errors=False)
        assert result == -1

    def test_training_error_aggregator(self):
        """Test TrainingErrorAggregator."""
        from app.training.exception_integration import TrainingErrorAggregator

        errors = TrainingErrorAggregator("test operation", max_errors_before_abort=3)

        errors.add(ValueError("Error 1"))
        errors.add(ValueError("Error 2"))

        assert errors.count == 2
        assert not errors.should_abort()

        errors.add(ValueError("Error 3"))
        assert errors.should_abort()

    def test_training_error_context(self):
        """Test training_error_context manager."""
        from app.training.exception_integration import (
            training_error_context,
            TrainingError,
        )

        with pytest.raises(TrainingError):
            with training_error_context("test operation"):
                raise ValueError("Something went wrong")


# =============================================================================
# Thread Spawner Core Tests
# =============================================================================

class TestThreadSpawnerCore:
    """Tests for app.core.thread_spawner."""

    def test_spawn_basic_thread(self):
        """Test spawning a basic thread."""
        from app.core.thread_spawner import ThreadSpawner, ThreadState

        spawner = ThreadSpawner()
        completed = threading.Event()

        def simple_task():
            completed.set()

        thread = spawner.spawn(target=simple_task, name="test_thread")

        completed.wait(timeout=2.0)
        assert completed.is_set()

        # Wait for thread to finish
        thread.join(timeout=1.0)
        assert thread.state in (ThreadState.COMPLETED, ThreadState.RUNNING)

        spawner.shutdown(timeout=2.0)

    def test_thread_restart_on_failure(self):
        """Test thread restarts on failure."""
        from app.core.thread_spawner import ThreadSpawner, RestartPolicy

        spawner = ThreadSpawner()
        call_count = 0
        success = threading.Event()

        def flaky_task():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient error")
            success.set()

        thread = spawner.spawn(
            target=flaky_task,
            name="flaky_thread",
            restart_policy=RestartPolicy.ON_FAILURE,
            max_restarts=3,
            restart_delay=0.1,
        )

        success.wait(timeout=5.0)
        assert success.is_set()
        assert call_count == 2

        spawner.shutdown(timeout=2.0)

    def test_spawner_health_check(self):
        """Test spawner health check."""
        from app.core.thread_spawner import ThreadSpawner

        spawner = ThreadSpawner()
        health = spawner.health_check()

        assert "healthy" in health
        assert "running" in health

        spawner.shutdown(timeout=1.0)


# =============================================================================
# Promotion Controller Distributed Lock Tests
# =============================================================================

class TestPromotionControllerDistributedLocking:
    """Tests for distributed locking in promotion_controller."""

    def test_has_distributed_locks_flag(self):
        """Test _HAS_DISTRIBUTED_LOCKS is set correctly."""
        from app.training.promotion_controller import _HAS_DISTRIBUTED_LOCKS

        # Should be True when locking_integration is available
        assert _HAS_DISTRIBUTED_LOCKS is True

    def test_extract_config_key_standard_format(self):
        """Test config key extraction from model ID."""
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()

        # Standard format: board_players_version
        assert controller._extract_config_key("square8_2p_v42") == "square8_2p"
        assert controller._extract_config_key("hex7_4p_v1") == "hex7_4p"
        assert controller._extract_config_key("triangle6_3p_v100") == "triangle6_3p"

    def test_extract_config_key_numeric_suffix(self):
        """Test config key extraction with numeric suffix."""
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()

        # Numeric suffix
        assert controller._extract_config_key("square8_2p_42") == "square8_2p"
        assert controller._extract_config_key("hex7_4p_1") == "hex7_4p"

    def test_extract_config_key_fallback(self):
        """Test config key extraction fallback for unusual formats."""
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()

        # No version suffix - returns as-is
        assert controller._extract_config_key("square8_2p") == "square8_2p"

    def test_execute_promotion_acquires_lock(self):
        """Test that execute_promotion uses distributed locking."""
        from app.training.promotion_controller import (
            PromotionController,
            PromotionDecision,
            PromotionType,
        )
        from unittest.mock import patch, MagicMock

        controller = PromotionController()

        decision = PromotionDecision(
            model_id="square8_2p_v42",
            promotion_type=PromotionType.STAGING,
            should_promote=True,
            reason="Test promotion",
        )

        # Mock TrainingLocks.promotion to verify it's called
        with patch('app.training.promotion_controller.TrainingLocks') as mock_locks:
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=MagicMock())
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_locks.promotion.return_value = mock_context

            # Also mock the locked method to prevent actual promotion
            with patch.object(controller, '_execute_promotion_locked', return_value=True):
                controller.execute_promotion(decision)

            # Verify promotion lock was requested with correct config key
            mock_locks.promotion.assert_called_once_with("square8_2p", timeout=60)

    def test_execute_promotion_fails_without_lock(self):
        """Test that execute_promotion fails gracefully when lock not acquired."""
        from app.training.promotion_controller import (
            PromotionController,
            PromotionDecision,
            PromotionType,
        )
        from unittest.mock import patch, MagicMock

        controller = PromotionController()

        decision = PromotionDecision(
            model_id="square8_2p_v42",
            promotion_type=PromotionType.STAGING,
            should_promote=True,
            reason="Test promotion",
        )

        # Mock TrainingLocks.promotion to return None (lock not acquired)
        with patch('app.training.promotion_controller.TrainingLocks') as mock_locks:
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=None)  # Lock not acquired
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_locks.promotion.return_value = mock_context

            result = controller.execute_promotion(decision)

            # Should return False when lock not acquired
            assert result is False


# =============================================================================
# Exception Consolidation Tests
# =============================================================================

class TestExceptionConsolidation:
    """Tests for exception consolidation in app.errors."""

    def test_exceptions_available_in_app_errors(self):
        """Test all training exceptions are available in app.errors."""
        from app.errors import (
            TrainingError,
            CheckpointError,
            EvaluationError,
            SelfplayError,
            DataLoadError,
            ModelVersioningError,
        )

        # All should be importable
        assert TrainingError is not None
        assert CheckpointError is not None
        assert EvaluationError is not None
        assert SelfplayError is not None
        assert DataLoadError is not None
        assert ModelVersioningError is not None

    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        from app.errors import (
            RingRiftError,
            TrainingError,
            CheckpointError,
            EvaluationError,
            SelfplayError,
            DataLoadError,
            ModelVersioningError,
        )

        # TrainingError inherits from RingRiftError
        assert issubclass(TrainingError, RingRiftError)

        # Training-specific errors inherit from TrainingError
        assert issubclass(CheckpointError, TrainingError)
        assert issubclass(EvaluationError, TrainingError)
        assert issubclass(SelfplayError, TrainingError)
        assert issubclass(DataLoadError, TrainingError)
        assert issubclass(ModelVersioningError, CheckpointError)

    def test_exceptions_reexported_from_exception_integration(self):
        """Test exceptions are re-exported from exception_integration."""
        from app.training.exception_integration import (
            TrainingError,
            CheckpointError,
            EvaluationError,
            SelfplayError,
            DataLoadError,
        )
        from app.errors import (
            TrainingError as TrainingErrorDirect,
            CheckpointError as CheckpointErrorDirect,
        )

        # Should be the same classes
        assert TrainingError is TrainingErrorDirect
        assert CheckpointError is CheckpointErrorDirect

    def test_exception_error_codes(self):
        """Test exception error codes are set correctly."""
        from app.errors import (
            TrainingError,
            EvaluationError,
            SelfplayError,
            DataLoadError,
        )

        assert TrainingError.code == "TRAINING_ERROR"
        assert EvaluationError.code == "EVALUATION_ERROR"
        assert SelfplayError.code == "SELFPLAY_ERROR"
        assert DataLoadError.code == "DATA_LOAD_ERROR"


# =============================================================================
# Deprecation Warning Tests
# =============================================================================

class TestDeprecationWarnings:
    """Tests for deprecation warnings in fault_tolerance."""

    def test_retry_with_backoff_deprecation_warning(self):
        """Test retry_with_backoff emits deprecation warning at decoration time."""
        import warnings

        # Capture warnings at decoration time (when decorator is applied)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import inside context to capture deprecation warning
            from app.training.fault_tolerance import retry_with_backoff

            @retry_with_backoff(max_retries=1, base_delay=0.01)
            def simple_func():
                return "success"

            # Should have deprecation warning from decoration
            deprecation_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "retry_with_backoff is deprecated" in str(x.message)
            ]
            assert len(deprecation_warnings) >= 1

        result = simple_func()
        assert result == "success"

    def test_async_retry_with_backoff_deprecation_warning(self):
        """Test async_retry_with_backoff emits deprecation warning at decoration time."""
        import warnings

        # Capture warnings at decoration time
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from app.training.fault_tolerance import async_retry_with_backoff

            @async_retry_with_backoff(max_retries=1, base_delay=0.01)
            async def simple_async_func():
                return "async_success"

            deprecation_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "async_retry_with_backoff is deprecated" in str(x.message)
            ]
            assert len(deprecation_warnings) >= 1

        result = asyncio.get_event_loop().run_until_complete(simple_async_func())
        assert result == "async_success"


# =============================================================================
# Unified Orchestrator Distributed Lock Tests
# =============================================================================

class TestUnifiedOrchestratorDistributedLocking:
    """Tests for distributed locking in unified_orchestrator."""

    def test_has_distributed_locks_flag(self):
        """Test _HAS_DISTRIBUTED_LOCKS is set correctly."""
        from app.training.unified_orchestrator import _HAS_DISTRIBUTED_LOCKS

        # Should be True when locking_integration is available
        assert _HAS_DISTRIBUTED_LOCKS is True

    def test_save_checkpoint_locked_method_exists(self):
        """Test _save_checkpoint_locked method exists."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        # Method should exist
        assert hasattr(UnifiedTrainingOrchestrator, '_save_checkpoint_locked')

    def test_training_metrics_flag(self):
        """Test _HAS_TRAINING_METRICS is set correctly."""
        from app.training.unified_orchestrator import _HAS_TRAINING_METRICS

        # Should be True when metrics_integration is available
        assert _HAS_TRAINING_METRICS is True

    def test_training_events_flag(self):
        """Test _HAS_TRAINING_EVENTS is set correctly."""
        from app.training.unified_orchestrator import _HAS_TRAINING_EVENTS

        # Should be True when event_integration is available
        assert _HAS_TRAINING_EVENTS is True


# =============================================================================
# Model Versioning Exception Tests
# =============================================================================

class TestModelVersioningExceptions:
    """Tests for model_versioning exception consolidation."""

    def test_model_versioning_error_in_app_errors(self):
        """Test ModelVersioningError is in app.errors."""
        from app.errors import ModelVersioningError, CheckpointError

        assert issubclass(ModelVersioningError, CheckpointError)
        assert ModelVersioningError.code == "MODEL_VERSIONING_ERROR"

    def test_subclasses_in_model_versioning(self):
        """Test specialized subclasses remain in model_versioning."""
        from app.training.model_versioning import (
            ModelVersioningError,
            VersionMismatchError,
            ChecksumMismatchError,
        )

        assert issubclass(VersionMismatchError, ModelVersioningError)
        assert issubclass(ChecksumMismatchError, ModelVersioningError)


# =============================================================================
# Regression Detector Tests
# =============================================================================

class TestRegressionDetector:
    """Tests for app.training.regression_detector."""

    def test_regression_severity_enum(self):
        """Test RegressionSeverity enum values."""
        from app.training.regression_detector import RegressionSeverity

        assert RegressionSeverity.MINOR.value == "minor"
        assert RegressionSeverity.MODERATE.value == "moderate"
        assert RegressionSeverity.SEVERE.value == "severe"
        assert RegressionSeverity.CRITICAL.value == "critical"

    def test_get_regression_detector_singleton(self):
        """Test get_regression_detector returns singleton."""
        from app.training.regression_detector import (
            get_regression_detector,
            _detector_instance,
        )

        # Reset singleton for test isolation
        import app.training.regression_detector as rd
        rd._detector_instance = None

        detector1 = get_regression_detector()
        detector2 = get_regression_detector()

        assert detector1 is detector2

        # Cleanup
        rd._detector_instance = None

    def test_check_regression_no_regression(self):
        """Test check_regression returns None when no regression."""
        from app.training.regression_detector import RegressionDetector

        detector = RegressionDetector()

        event = detector.check_regression(
            model_id="test_model",
            current_elo=1550,
            baseline_elo=1500,  # Improvement, not regression
            current_win_rate=0.55,
            baseline_win_rate=0.50,
            games_played=100,
        )

        assert event is None  # No regression detected

    def test_check_regression_minor(self):
        """Test check_regression detects minor regression."""
        from app.training.regression_detector import (
            RegressionDetector,
            RegressionSeverity,
        )

        detector = RegressionDetector()

        event = detector.check_regression(
            model_id="test_model",
            current_elo=1480,
            baseline_elo=1500,  # -20 Elo (minor)
            current_win_rate=0.48,
            baseline_win_rate=0.50,
            games_played=100,
        )

        assert event is not None
        assert event.severity == RegressionSeverity.MINOR

    def test_check_regression_critical(self):
        """Test check_regression detects critical regression."""
        from app.training.regression_detector import (
            RegressionDetector,
            RegressionConfig,
            RegressionSeverity,
        )

        config = RegressionConfig(
            elo_drop_minor=10,
            elo_drop_moderate=25,
            elo_drop_severe=50,
            elo_drop_critical=75,
        )
        detector = RegressionDetector(config)

        event = detector.check_regression(
            model_id="test_model",
            current_elo=1400,
            baseline_elo=1500,  # -100 Elo (critical)
            current_win_rate=0.35,
            baseline_win_rate=0.50,
            games_played=100,
        )

        assert event is not None
        assert event.severity == RegressionSeverity.CRITICAL

    def test_add_listener(self):
        """Test adding and notifying listeners."""
        from app.training.regression_detector import (
            RegressionDetector,
            RegressionListener,
            RegressionEvent,
        )

        class TestListener(RegressionListener):
            def __init__(self):
                self.events = []

            def on_regression(self, event: RegressionEvent) -> None:
                self.events.append(event)

        detector = RegressionDetector()
        listener = TestListener()
        detector.add_listener(listener)

        # Trigger a regression
        detector.check_regression(
            model_id="test_model",
            current_elo=1400,
            baseline_elo=1500,
            games_played=100,
        )

        assert len(listener.events) == 1
        assert listener.events[0].model_id == "test_model"


# =============================================================================
# Feedback Accelerator Tests
# =============================================================================

class TestFeedbackAccelerator:
    """Tests for app.training.feedback_accelerator."""

    def test_feedback_state_dataclass(self):
        """Test FeedbackState dataclass."""
        from app.training.feedback_accelerator import FeedbackState

        state = FeedbackState()
        assert state.consecutive_promotions == 0
        assert state.plateau_count == 0
        assert state.momentum == 0.0

    def test_get_feedback_accelerator_singleton(self):
        """Test get_feedback_accelerator returns singleton."""
        from app.training.feedback_accelerator import get_feedback_accelerator

        accel1 = get_feedback_accelerator()
        accel2 = get_feedback_accelerator()

        assert accel1 is accel2

    def test_record_promotion_updates_state(self):
        """Test recording a promotion updates feedback state."""
        from app.training.feedback_accelerator import get_feedback_accelerator

        accel = get_feedback_accelerator()

        initial_promotions = accel.get_state("square8_2p").consecutive_promotions

        accel.record_promotion("square8_2p", elo_gain=30.0)

        state = accel.get_state("square8_2p")
        assert state.consecutive_promotions == initial_promotions + 1
        assert state.momentum > 0

    def test_get_acceleration_factor(self):
        """Test acceleration factor calculation."""
        from app.training.feedback_accelerator import get_feedback_accelerator

        accel = get_feedback_accelerator()

        # Initial factor should be near 1.0
        factor = accel.get_acceleration_factor("square8_2p")
        assert 0.5 <= factor <= 2.0


# =============================================================================
# Quality Bridge Tests
# =============================================================================

class TestQualityBridge:
    """Tests for app.training.quality_bridge."""

    def test_quality_bridge_import(self):
        """Test quality_bridge can be imported."""
        from app.training.quality_bridge import (
            QualityBridge,
            get_quality_bridge,
        )

        assert QualityBridge is not None
        assert get_quality_bridge is not None

    def test_get_quality_bridge_singleton(self):
        """Test get_quality_bridge returns singleton."""
        from app.training.quality_bridge import get_quality_bridge, QualityBridge

        # Reset singleton for isolation
        QualityBridge.reset_instance()

        bridge1 = get_quality_bridge()
        bridge2 = get_quality_bridge()

        assert bridge1 is bridge2

        QualityBridge.reset_instance()

    def test_quality_score_range(self):
        """Test quality score is in valid range."""
        from app.training.quality_bridge import get_quality_bridge, QualityBridge

        QualityBridge.reset_instance()
        bridge = get_quality_bridge()

        # Get default quality score
        score = bridge.get_quality_score("square8_2p")

        # Should be between 0 and 1
        assert 0.0 <= score <= 1.0

        QualityBridge.reset_instance()


# =============================================================================
# Unified Signals Integration Tests
# =============================================================================

class TestUnifiedSignalsIntegration:
    """Integration tests for unified_signals with other modules."""

    def test_signal_computer_factory(self):
        """Test get_signal_computer factory."""
        from app.training.unified_signals import (
            get_signal_computer,
            reset_signal_computer,
        )

        reset_signal_computer()

        computer1 = get_signal_computer()
        computer2 = get_signal_computer()

        assert computer1 is computer2

        reset_signal_computer()

    def test_compute_signals_basic(self):
        """Test basic signal computation."""
        from app.training.unified_signals import get_signal_computer, reset_signal_computer

        reset_signal_computer()
        computer = get_signal_computer()

        signals = computer.compute_signals(
            current_games=100,
            current_elo=1500,
            config_key="square8_2p",
        )

        assert signals is not None
        assert hasattr(signals, 'urgency')
        assert hasattr(signals, 'elo_trend')

        reset_signal_computer()

    def test_urgency_levels(self):
        """Test TrainingUrgency enum values."""
        from app.training.unified_signals import TrainingUrgency

        assert TrainingUrgency.CRITICAL.value == "critical"
        assert TrainingUrgency.HIGH.value == "high"
        assert TrainingUrgency.NORMAL.value == "normal"
        assert TrainingUrgency.LOW.value == "low"
        assert TrainingUrgency.NONE.value == "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
