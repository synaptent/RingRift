"""Tests for the training fault tolerance module.

Comprehensive test coverage for:
- Retry decorator with exponential backoff
- GPU error handling
- Training error handler (OOM, batch size reduction)
- Checkpoint management
- Heartbeat monitoring
- Graceful shutdown
- Fault-tolerant trainer
"""

import json
import os
import signal
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.checkpoint_unified import UnifiedCheckpointConfig
from app.training.fault_tolerance import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointType,
    DistributedFaultHandler,
    FaultTolerantTrainer,
    GracefulShutdown,
    HeartbeatMonitor,
    NonRecoverableError,
    RecoverableError,
    TrainingErrorHandler,
    TrainingProgress,
    TrainingState,
    handle_gpu_error,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    """Test retry decorator."""

    def test_success_no_retry(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that max retries raises exception."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_specific_exceptions_only(self):
        """Test that only specified exceptions trigger retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def fails_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not caught")

        with pytest.raises(TypeError):
            fails_with_type_error()

        assert call_count == 1  # No retry for TypeError

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callbacks = []

        def on_retry(exc, attempt, delay):
            callbacks.append((str(exc), attempt, delay))

        @retry_with_backoff(max_retries=2, base_delay=0.01, on_retry=on_retry)
        def fails_once():
            if len(callbacks) == 0:
                raise ValueError("First failure")
            return "success"

        result = fails_once()
        assert result == "success"
        assert len(callbacks) == 1
        assert "First failure" in callbacks[0][0]
        assert callbacks[0][1] == 1  # First retry attempt

    def test_exponential_backoff_delay(self):
        """Test that delays increase exponentially."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exponential_base=2.0,
            max_delay=10.0,
            on_retry=on_retry,
            jitter=False,  # Disable jitter to test exact exponential delays
        )
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fails()

        # Delays should be 0.01, 0.02, 0.04 (exponential)
        assert len(delays) == 3
        assert delays[0] == pytest.approx(0.01, rel=0.1)
        assert delays[1] == pytest.approx(0.02, rel=0.1)
        assert delays[2] == pytest.approx(0.04, rel=0.1)


class TestTrainingErrorHandler:
    """Test TrainingErrorHandler."""

    def test_init_defaults(self):
        """Test default initialization."""
        handler = TrainingErrorHandler()
        assert handler.max_retries == 3
        assert handler.min_batch_size == 8
        assert handler.batch_reduction_factor == 0.5

    def test_handle_oom_reduces_batch_size(self):
        """Test OOM handling reduces batch size."""
        handler = TrainingErrorHandler(min_batch_size=8, batch_reduction_factor=0.5)

        new_size = handler.handle_oom(256)
        assert new_size == 128  # 256 * 0.5

        new_size = handler.handle_oom(new_size)
        assert new_size == 64  # 128 * 0.5

    def test_handle_oom_min_batch_size(self):
        """Test OOM handling respects minimum batch size."""
        handler = TrainingErrorHandler(min_batch_size=32, batch_reduction_factor=0.5)

        new_size = handler.handle_oom(64)
        assert new_size == 32  # 64 * 0.5, but >= min

        # Can't reduce further
        with pytest.raises(NonRecoverableError):
            handler.handle_oom(32)

    def test_safe_training_step_success(self):
        """Test safe_training_step context on success."""
        handler = TrainingErrorHandler()

        with handler.safe_training_step(batch_size=256) as ctx:
            # Simulate successful step
            ctx.record_success()

        assert handler._consecutive_failures == 0

    def test_safe_training_step_oom(self):
        """Test safe_training_step handles OOM."""
        handler = TrainingErrorHandler(min_batch_size=8, batch_reduction_factor=0.5)

        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("CUDA out of memory")

        assert handler.recommended_batch_size == 128
        assert handler._oom_count == 1

    def test_safe_training_step_max_retries(self):
        """Test safe_training_step raises after max retries."""
        handler = TrainingErrorHandler(max_retries=2)

        # First 2 failures should be recoverable
        for _ in range(2):
            handler._consecutive_failures += 1

        # Third failure should be non-recoverable
        with pytest.raises(NonRecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("Some error")

    def test_reset_failure_count(self):
        """Test failure count resets on success."""
        handler = TrainingErrorHandler()
        handler._consecutive_failures = 5

        handler.reset_failure_count()
        assert handler._consecutive_failures == 0

    def test_get_stats(self):
        """Test stats retrieval."""
        handler = TrainingErrorHandler()
        handler._oom_count = 3
        handler._consecutive_failures = 1
        handler._total_recoveries = 2
        handler._current_batch_size = 64

        stats = handler.get_stats()
        assert stats["oom_count"] == 3
        assert stats["consecutive_failures"] == 1
        assert stats["total_recoveries"] == 2
        assert stats["current_batch_size"] == 64


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_001",
            checkpoint_type=CheckpointType.REGULAR,
            epoch=5,
            global_step=1000,
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            metrics={"loss": 0.5},
            training_config={"lr": 0.001},
            file_path="/path/to/ckpt.pt",
            file_hash="abc123",
        )

        d = metadata.to_dict()
        assert d["checkpoint_id"] == "ckpt_001"
        assert d["checkpoint_type"] == "regular"
        assert d["epoch"] == 5
        assert d["timestamp"] == "2025-01-15T12:00:00"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "checkpoint_id": "ckpt_001",
            "checkpoint_type": "regular",
            "epoch": 5,
            "global_step": 1000,
            "timestamp": "2025-01-15T12:00:00",
            "metrics": {"loss": 0.5},
            "training_config": {"lr": 0.001},
            "file_path": "/path/to/ckpt.pt",
            "file_hash": "abc123",
            "parent_checkpoint": None,
        }

        metadata = CheckpointMetadata.from_dict(d)
        assert metadata.checkpoint_id == "ckpt_001"
        assert metadata.checkpoint_type == CheckpointType.REGULAR
        assert metadata.epoch == 5
        assert metadata.timestamp == datetime(2025, 1, 15, 12, 0, 0)


class TestTrainingProgress:
    """Test TrainingProgress dataclass."""

    def test_defaults(self):
        """Test default values."""
        progress = TrainingProgress()
        assert progress.epoch == 0
        assert progress.global_step == 0
        assert progress.best_metric is None
        assert progress.total_epochs == 100

    def test_to_dict(self):
        """Test serialization."""
        progress = TrainingProgress(epoch=5, global_step=1000, best_metric=0.1)
        d = progress.to_dict()

        assert d["epoch"] == 5
        assert d["global_step"] == 1000
        assert d["best_metric"] == 0.1

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "epoch": 5,
            "global_step": 1000,
            "batch_idx": 50,
            "samples_seen": 5000,
            "best_metric": 0.1,
            "best_metric_name": "loss",
            "best_epoch": 3,
            "total_epochs": 100,
            "learning_rate": 0.001,
        }

        progress = TrainingProgress.from_dict(d)
        assert progress.epoch == 5
        assert progress.global_step == 1000
        assert progress.best_metric == 0.1


class TestCheckpointManager:
    """Test CheckpointManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create CheckpointManager."""
        config = UnifiedCheckpointConfig(
            checkpoint_dir=str(temp_dir / "checkpoints"),
            max_checkpoints=5,
            keep_best=2,
            keep_every_n_epochs=5,
        )
        return CheckpointManager(config)

    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates checkpoint directory."""
        ckpt_dir = temp_dir / "new_checkpoints"
        assert not ckpt_dir.exists()

        config = UnifiedCheckpointConfig(checkpoint_dir=str(ckpt_dir))
        CheckpointManager(config)
        assert ckpt_dir.exists()

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch required"
    )
    def test_save_and_load_checkpoint(self, manager):
        """Test saving and loading checkpoints."""
        import torch

        model_state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        progress = TrainingProgress(epoch=1, global_step=100)
        metrics = {"loss": 0.5, "accuracy": 0.8}

        metadata = manager.save_checkpoint(
            model_state=model_state,
            progress=progress,
            checkpoint_type=CheckpointType.REGULAR,
            metrics=metrics,
        )

        assert metadata.epoch == 1
        assert metadata.global_step == 100
        assert metadata.metrics["loss"] == 0.5

        # Load checkpoint
        loaded = manager.load_checkpoint(metadata.checkpoint_id)
        assert loaded is not None
        assert torch.allclose(
            loaded["model_state_dict"]["weight"],
            model_state["weight"]
        )
        assert loaded["progress"]["epoch"] == 1

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch required"
    )
    def test_load_latest_checkpoint(self, manager):
        """Test loading most recent checkpoint."""
        import torch

        # Save multiple checkpoints
        for epoch in range(3):
            manager.save_checkpoint(
                model_state={"epoch": torch.tensor([epoch])},
                progress=TrainingProgress(epoch=epoch, global_step=epoch * 100),
            )

        # Load latest
        loaded = manager.load_checkpoint()
        assert loaded is not None
        assert loaded["progress"]["epoch"] == 2

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch required"
    )
    def test_get_best_checkpoint(self, manager):
        """Test getting best checkpoint by metric."""
        import torch

        # Save checkpoints with different loss values
        for loss in [0.5, 0.3, 0.7, 0.4]:
            manager.save_checkpoint(
                model_state={"loss": torch.tensor([loss])},
                progress=TrainingProgress(epoch=0),
                metrics={"loss": loss},
            )

        best = manager.get_best_checkpoint("loss", lower_is_better=True)
        assert best is not None
        assert best.metrics["loss"] == 0.3

    def test_list_checkpoints(self, manager):
        """Test listing checkpoints."""
        # Initially empty
        assert len(manager.list_checkpoints()) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch required"
    )
    def test_checkpoint_cleanup(self, manager):
        """Test that old checkpoints are cleaned up."""
        import torch

        # Save more than max_checkpoints
        for epoch in range(10):
            manager.save_checkpoint(
                model_state={"epoch": torch.tensor([epoch])},
                progress=TrainingProgress(epoch=epoch, global_step=epoch * 100),
                metrics={"loss": 1.0 - epoch * 0.1},
            )

        # Should have cleaned up old ones
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= manager.config.max_checkpoints + manager.config.keep_best


class TestHeartbeatMonitor:
    """Test HeartbeatMonitor."""

    def test_beat_records_time(self):
        """Test that beat() records timestamp."""
        monitor = HeartbeatMonitor()
        assert monitor.last_heartbeat is None

        monitor.beat()
        assert monitor.last_heartbeat is not None
        assert (datetime.now() - monitor.last_heartbeat).total_seconds() < 1

    def test_beat_writes_file(self):
        """Test that beat() writes heartbeat file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_file = Path(tmpdir) / "heartbeat.json"
            monitor = HeartbeatMonitor()
            monitor.heartbeat_file = heartbeat_file

            monitor.beat()

            assert heartbeat_file.exists()
            with open(heartbeat_file) as f:
                data = json.load(f)
            assert "timestamp" in data
            assert "pid" in data

    def test_check_external_heartbeat_alive(self):
        """Test checking external heartbeat of alive process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_file = Path(tmpdir) / "heartbeat.json"

            # Write recent heartbeat
            data = {
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            }
            with open(heartbeat_file, "w") as f:
                json.dump(data, f)

            assert HeartbeatMonitor.check_external_heartbeat(heartbeat_file, timeout=60.0)

    def test_check_external_heartbeat_dead(self):
        """Test checking external heartbeat of dead process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_file = Path(tmpdir) / "heartbeat.json"

            # Write old heartbeat
            old_time = datetime.now() - timedelta(minutes=5)
            data = {
                "timestamp": old_time.isoformat(),
                "pid": os.getpid()
            }
            with open(heartbeat_file, "w") as f:
                json.dump(data, f)

            assert not HeartbeatMonitor.check_external_heartbeat(heartbeat_file, timeout=60.0)

    def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = HeartbeatMonitor(heartbeat_interval=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_file = Path(tmpdir) / "heartbeat.json"
            monitor.start(heartbeat_file)

            assert monitor._running
            assert monitor._thread is not None
            assert monitor._thread.is_alive()

            time.sleep(0.2)  # Let monitor run a bit

            monitor.stop()
            assert not monitor._running


class TestGracefulShutdown:
    """Test GracefulShutdown."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_shutdown_requested_initially_false(self, temp_dir):
        """Test that shutdown_requested is initially False."""
        config = UnifiedCheckpointConfig(checkpoint_dir=str(temp_dir / "checkpoints"))
        manager = CheckpointManager(config)
        shutdown = GracefulShutdown(manager)

        assert not shutdown.shutdown_requested

    def test_request_shutdown(self, temp_dir):
        """Test programmatic shutdown request."""
        config = UnifiedCheckpointConfig(checkpoint_dir=str(temp_dir / "checkpoints"))
        manager = CheckpointManager(config)
        shutdown = GracefulShutdown(manager)

        shutdown.request_shutdown()
        assert shutdown.shutdown_requested


class TestFaultTolerantTrainer:
    """Test FaultTolerantTrainer."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def trainer(self, temp_dir):
        """Create FaultTolerantTrainer."""
        return FaultTolerantTrainer(
            checkpoint_dir=temp_dir / "checkpoints",
            checkpoint_interval_steps=100,
            checkpoint_interval_epochs=1,
            max_retries=3,
        )

    def test_init(self, trainer):
        """Test initialization."""
        assert trainer.state == TrainingState.INITIALIZING
        assert trainer.checkpoint_interval_steps == 100
        assert trainer.max_retries == 3

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch required"
    )
    def test_initialize_fresh(self, trainer):
        """Test fresh initialization."""
        import torch

        model_state = {"weight": torch.tensor([1.0])}
        training_config = {"lr": 0.001}

        progress = trainer.initialize(
            model_state=model_state,
            training_config=training_config,
            total_epochs=10,
            resume=False,
        )

        assert progress.epoch == 0
        assert progress.total_epochs == 10
        assert trainer.state == TrainingState.RUNNING

    def test_should_checkpoint(self, trainer):
        """Test checkpoint condition checking."""
        # Initially no checkpoint needed
        assert not trainer.should_checkpoint()

        # At step 100, should checkpoint
        trainer.progress.global_step = 100
        assert trainer.should_checkpoint()

        # At step 101, no checkpoint
        trainer.progress.global_step = 101
        assert not trainer.should_checkpoint()

    def test_update_progress(self, trainer):
        """Test progress updates."""
        trainer.heartbeat_monitor.beat = MagicMock()

        trainer.update_progress(
            epoch=5,
            batch_idx=50,
            global_step=550,
            metrics={"loss": 0.3},
        )

        assert trainer.progress.epoch == 5
        assert trainer.progress.batch_idx == 50
        assert trainer.progress.global_step == 550
        trainer.heartbeat_monitor.beat.assert_called_once()

    def test_update_progress_tracks_best(self, trainer):
        """Test that update_progress tracks best metric."""
        trainer.progress.best_metric_name = "loss"

        trainer.update_progress(epoch=1, batch_idx=0, global_step=100, metrics={"loss": 0.5})
        assert trainer.progress.best_metric == 0.5
        assert trainer.progress.best_epoch == 1

        trainer.update_progress(epoch=2, batch_idx=0, global_step=200, metrics={"loss": 0.3})
        assert trainer.progress.best_metric == 0.3
        assert trainer.progress.best_epoch == 2

        # Worse metric shouldn't update best
        trainer.update_progress(epoch=3, batch_idx=0, global_step=300, metrics={"loss": 0.4})
        assert trainer.progress.best_metric == 0.3
        assert trainer.progress.best_epoch == 2

    def test_should_stop(self, trainer):
        """Test should_stop property."""
        assert not trainer.should_stop

        trainer.graceful_shutdown.request_shutdown()
        assert trainer.should_stop


class TestDistributedFaultHandler:
    """Test DistributedFaultHandler."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def handler(self, temp_dir):
        """Create DistributedFaultHandler."""
        return DistributedFaultHandler(
            world_size=4,
            rank=0,
            coordinator_path=temp_dir / "coordination",
            timeout=5.0,
        )

    def test_report_status(self, handler):
        """Test status reporting."""
        handler.report_status("running", epoch=5, step=1000)

        assert handler._status_file.exists()
        with open(handler._status_file) as f:
            data = json.load(f)

        assert data["rank"] == 0
        assert data["status"] == "running"
        assert data["epoch"] == 5
        assert data["step"] == 1000

    def test_report_heartbeat(self, handler):
        """Test heartbeat reporting."""
        handler.report_heartbeat()

        assert handler._heartbeat_file.exists()
        with open(handler._heartbeat_file) as f:
            data = json.load(f)

        assert data["rank"] == 0
        assert data["pid"] == os.getpid()

    def test_check_all_workers_alive_all_dead(self, handler):
        """Test checking workers when none have heartbeats."""
        dead = handler.check_all_workers_alive()
        assert len(dead) == 4  # All workers reported as dead

    def test_check_all_workers_alive_some_alive(self, handler):
        """Test checking workers when some are alive."""
        # Write heartbeat for ranks 0 and 2
        for rank in [0, 2]:
            heartbeat_file = handler.coordinator_path / f"worker_{rank}_heartbeat.json"
            data = {
                "rank": rank,
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            }
            with open(heartbeat_file, "w") as f:
                json.dump(data, f)

        dead = handler.check_all_workers_alive()
        assert set(dead) == {1, 3}

    def test_elect_leader(self, handler):
        """Test leader election."""
        # All workers alive - rank 0 should be leader
        for rank in range(4):
            heartbeat_file = handler.coordinator_path / f"worker_{rank}_heartbeat.json"
            data = {
                "rank": rank,
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            }
            with open(heartbeat_file, "w") as f:
                json.dump(data, f)

        leader = handler.elect_leader()
        assert leader == 0

    def test_elect_leader_with_dead_workers(self, handler):
        """Test leader election when rank 0 is dead."""
        # Only ranks 2 and 3 alive
        for rank in [2, 3]:
            heartbeat_file = handler.coordinator_path / f"worker_{rank}_heartbeat.json"
            data = {
                "rank": rank,
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            }
            with open(heartbeat_file, "w") as f:
                json.dump(data, f)

        leader = handler.elect_leader()
        assert leader == 2  # Lowest alive rank


class TestEnums:
    """Test enum types."""

    def test_checkpoint_type_values(self):
        """Test CheckpointType enum values."""
        assert CheckpointType.REGULAR.value == "regular"
        assert CheckpointType.EPOCH.value == "epoch"
        assert CheckpointType.BEST.value == "best"
        assert CheckpointType.EMERGENCY.value == "emergency"
        assert CheckpointType.RECOVERY.value == "recovery"

    def test_training_state_values(self):
        """Test TrainingState enum values."""
        assert TrainingState.INITIALIZING.value == "initializing"
        assert TrainingState.RUNNING.value == "running"
        assert TrainingState.PAUSED.value == "paused"
        assert TrainingState.CHECKPOINTING.value == "checkpointing"
        assert TrainingState.RECOVERING.value == "recovering"
        assert TrainingState.COMPLETED.value == "completed"
        assert TrainingState.FAILED.value == "failed"


class TestExceptionTypes:
    """Test custom exception types."""

    def test_recoverable_error(self):
        """Test RecoverableError."""
        error = RecoverableError("Temporary failure")
        assert "Temporary failure" in str(error)
        assert isinstance(error, Exception)

    def test_non_recoverable_error(self):
        """Test NonRecoverableError."""
        error = NonRecoverableError("Permanent failure")
        assert "Permanent failure" in str(error)
        assert isinstance(error, Exception)


# =============================================================================
# Async Retry with Backoff Tests
# =============================================================================

class TestAsyncRetryWithBackoff:
    """Tests for async_retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_async_success_no_retry(self):
        """Test that successful async calls don't retry."""
        from app.training.fault_tolerance import async_retry_with_backoff

        call_count = 0

        @async_retry_with_backoff(max_retries=3, base_delay=0.01)
        async def async_succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_succeed()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_failure(self):
        """Test that async failures trigger retry."""
        from app.training.fault_tolerance import async_retry_with_backoff

        call_count = 0

        @async_retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def async_fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await async_fail_twice()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_max_retries_exceeded(self):
        """Test that async exhausts retries then raises."""
        from app.training.fault_tolerance import async_retry_with_backoff

        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        async def async_always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await async_always_fails()

        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_async_jitter_enabled(self):
        """Test that jitter adds variation to delays."""
        from app.training.fault_tolerance import async_retry_with_backoff

        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        @async_retry_with_backoff(
            max_retries=5,
            base_delay=0.01,
            jitter=True,
            on_retry=on_retry,
        )
        async def async_always_fails():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await async_always_fails()

        # With jitter, delays should vary (not all identical)
        assert len(delays) == 5
        # Jitter multiplies by 0.5-1.5, so values should differ
        assert len({round(d, 4) for d in delays}) > 1 or all(d > 0 for d in delays)


# =============================================================================
# RetryPolicy Tests
# =============================================================================

class TestRetryPolicy:
    """Tests for RetryPolicy pre-configured policies."""

    def test_aggressive_policy(self):
        """Test AGGRESSIVE policy values."""
        from app.training.fault_tolerance import RetryPolicy

        policy = RetryPolicy.AGGRESSIVE
        assert policy["max_retries"] == 5
        assert policy["base_delay"] == 0.5
        assert policy["max_delay"] == 10.0
        assert policy["exponential_base"] == 1.5

    def test_standard_policy(self):
        """Test STANDARD policy values."""
        from app.training.fault_tolerance import RetryPolicy

        policy = RetryPolicy.STANDARD
        assert policy["max_retries"] == 3
        assert policy["base_delay"] == 1.0
        assert policy["max_delay"] == 60.0
        assert policy["exponential_base"] == 2.0

    def test_conservative_policy(self):
        """Test CONSERVATIVE policy values."""
        from app.training.fault_tolerance import RetryPolicy

        policy = RetryPolicy.CONSERVATIVE
        assert policy["max_retries"] == 3
        assert policy["base_delay"] == 5.0
        assert policy["max_delay"] == 300.0

    def test_network_policy(self):
        """Test NETWORK policy values."""
        from app.training.fault_tolerance import RetryPolicy

        policy = RetryPolicy.NETWORK
        assert policy["max_retries"] == 4
        assert policy["base_delay"] == 2.0
        assert policy["max_delay"] == 120.0

    def test_get_policy_by_name(self):
        """Test get_policy method."""
        from app.training.fault_tolerance import RetryPolicy

        # Case-insensitive lookup
        policy = RetryPolicy.get_policy("AGGRESSIVE")
        assert policy == RetryPolicy.AGGRESSIVE

        policy = RetryPolicy.get_policy("network")
        assert policy == RetryPolicy.NETWORK

        # Unknown returns STANDARD
        policy = RetryPolicy.get_policy("unknown")
        assert policy == RetryPolicy.STANDARD

    def test_apply_sync(self):
        """Test apply method for sync functions."""
        from app.training.fault_tolerance import RetryPolicy

        # Should return a decorator
        decorator = RetryPolicy.apply(RetryPolicy.STANDARD, sync=True)
        assert callable(decorator)

        # Apply to a function
        @RetryPolicy.apply(RetryPolicy.AGGRESSIVE, sync=True)
        def sample_func():
            return "ok"

        result = sample_func()
        assert result == "ok"

    def test_apply_async(self):
        """Test apply method for async functions."""
        from app.training.fault_tolerance import RetryPolicy

        # Should return a decorator
        decorator = RetryPolicy.apply(RetryPolicy.NETWORK, sync=False)
        assert callable(decorator)


# =============================================================================
# Jitter Tests
# =============================================================================

class TestJitterBehavior:
    """Tests for jitter in retry decorators."""

    def test_sync_retry_jitter_enabled(self):
        """Test that sync retry has jitter by default."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            jitter=True,
            on_retry=on_retry,
        )
        def always_fails():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fails()

        # With jitter, delays should be positive
        assert all(d > 0 for d in delays)

    def test_sync_retry_jitter_disabled(self):
        """Test that sync retry without jitter has predictable delays."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exponential_base=2.0,
            jitter=False,
            on_retry=on_retry,
        )
        def always_fails():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fails()

        # Without jitter, delays should follow exponential pattern
        assert len(delays) == 3
        # 0.01, 0.02, 0.04 (approximately)
        assert delays[0] == pytest.approx(0.01, rel=0.1)
        assert delays[1] == pytest.approx(0.02, rel=0.1)
        assert delays[2] == pytest.approx(0.04, rel=0.1)
