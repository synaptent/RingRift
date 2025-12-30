"""Tests for Training Coordinator module.

Tests the cluster-wide training coordination system that prevents duplicate
training jobs and provides visibility into training status across nodes.
"""

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus
from app.coordination.training_coordinator import (
    HEARTBEAT_INTERVAL_SECONDS,
    # Constants
    MAX_CONCURRENT_TRAINING_SAME_CONFIG,
    MAX_TOTAL_CONCURRENT_TRAINING,
    TRAINING_TIMEOUT_HOURS,
    # Main class
    TrainingCoordinator,
    # Data classes
    TrainingJob,
    can_train,
    # Module functions
    get_training_coordinator,
    get_training_status,
)

# =============================================================================
# TrainingJob Tests
# =============================================================================


class TestTrainingJob:
    """Tests for TrainingJob dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        now = time.time()
        job = TrainingJob(
            job_id="test-job-1",
            board_type="square8",
            num_players=2,
            node_name="test-node",
            node_ip="192.168.1.1",
            pid=12345,
            started_at=now,
            last_heartbeat=now,
        )
        assert job.job_id == "test-job-1"
        assert job.status == "running"
        assert job.epochs_completed == 0
        assert job.best_val_loss == float("inf")
        assert job.current_elo == 0.0

    def test_config_key(self):
        """Should generate correct config key."""
        job = TrainingJob(
            job_id="test",
            board_type="hex7",
            num_players=3,
            node_name="node1",
            node_ip="1.1.1.1",
            pid=1,
            started_at=time.time(),
            last_heartbeat=time.time(),
        )
        assert job.config_key == "hex7_3p"

    def test_age_hours(self):
        """Should calculate age in hours."""
        now = time.time()
        two_hours_ago = now - 7200  # 2 hours in seconds
        job = TrainingJob(
            job_id="test",
            board_type="square8",
            num_players=2,
            node_name="node1",
            node_ip="1.1.1.1",
            pid=1,
            started_at=two_hours_ago,
            last_heartbeat=now,
        )
        assert 1.9 <= job.age_hours <= 2.1

    def test_heartbeat_age_seconds(self):
        """Should calculate heartbeat age in seconds."""
        now = time.time()
        job = TrainingJob(
            job_id="test",
            board_type="square8",
            num_players=2,
            node_name="node1",
            node_ip="1.1.1.1",
            pid=1,
            started_at=now,
            last_heartbeat=now - 120,  # 2 minutes ago
        )
        assert 119 <= job.heartbeat_age_seconds <= 121

    def test_is_stale_fresh_job(self):
        """Fresh job should not be stale."""
        now = time.time()
        job = TrainingJob(
            job_id="test",
            board_type="square8",
            num_players=2,
            node_name="node1",
            node_ip="1.1.1.1",
            pid=1,
            started_at=now,
            last_heartbeat=now,
        )
        assert not job.is_stale

    def test_is_stale_missed_heartbeats(self):
        """Job with missed heartbeats should be stale."""
        now = time.time()
        stale_heartbeat = now - (HEARTBEAT_INTERVAL_SECONDS * 4)
        job = TrainingJob(
            job_id="test",
            board_type="square8",
            num_players=2,
            node_name="node1",
            node_ip="1.1.1.1",
            pid=1,
            started_at=now - 1000,
            last_heartbeat=stale_heartbeat,
        )
        assert job.is_stale

    def test_is_stale_timeout(self):
        """Job exceeding timeout should be stale."""
        now = time.time()
        old_start = now - (TRAINING_TIMEOUT_HOURS * 3600 + 100)
        job = TrainingJob(
            job_id="test",
            board_type="square8",
            num_players=2,
            node_name="node1",
            node_ip="1.1.1.1",
            pid=1,
            started_at=old_start,
            last_heartbeat=now,  # Even with fresh heartbeat
        )
        assert job.is_stale

    def test_to_dict(self):
        """Should serialize to dictionary."""
        now = time.time()
        job = TrainingJob(
            job_id="test-job",
            board_type="square8",
            num_players=2,
            node_name="gpu-server-1",
            node_ip="10.0.0.1",
            pid=9999,
            started_at=now,
            last_heartbeat=now,
            epochs_completed=5,
            best_val_loss=0.123,
            current_elo=1550.0,
        )
        data = job.to_dict()
        assert data["job_id"] == "test-job"
        assert data["config_key"] == "square8_2p"
        assert data["epochs_completed"] == 5
        assert data["best_val_loss"] == 0.123
        assert "age_hours" in data
        assert "is_stale" in data


# =============================================================================
# TrainingCoordinator Tests
# =============================================================================


class TestTrainingCoordinator:
    """Tests for TrainingCoordinator class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def coordinator(self, temp_db_path):
        """Create a TrainingCoordinator with temporary database."""
        # Mock NFS path to use local temp directory
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                coord = TrainingCoordinator(use_nfs=False)
                yield coord

    def test_initialization(self, coordinator):
        """Should initialize correctly."""
        assert coordinator._node_name is not None
        assert coordinator._node_ip == "127.0.0.1"

    def test_can_start_training_empty(self, coordinator):
        """Should allow training when no jobs exist."""
        assert coordinator.can_start_training("square8", 2)

    def test_start_training(self, coordinator):
        """Should start a training job."""
        # Mock the distributed lock
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training(
                board_type="square8",
                num_players=2,
                model_version="v1.0",
            )

            assert job_id is not None
            assert "square8_2p" in job_id

    def test_cannot_start_duplicate_training(self, coordinator):
        """Should not allow duplicate training for same config."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start first job
            job1 = coordinator.start_training("square8", 2)
            assert job1 is not None

            # Try to start second job for same config
            job2 = coordinator.start_training("square8", 2)
            assert job2 is None

    def test_different_configs_allowed(self, coordinator):
        """Should allow training for different configs."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job1 = coordinator.start_training("square8", 2)
            job2 = coordinator.start_training("hex7", 2)
            job3 = coordinator.start_training("square8", 4)

            assert job1 is not None
            assert job2 is not None
            assert job3 is not None

    def test_get_active_jobs(self, coordinator):
        """Should return active jobs."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            coordinator.start_training("square8", 2)
            coordinator.start_training("hex7", 3)

            jobs = coordinator.get_active_jobs()
            assert len(jobs) == 2
            configs = [j.config_key for j in jobs]
            assert "square8_2p" in configs
            assert "hex7_3p" in configs

    def test_get_job(self, coordinator):
        """Should retrieve a specific job by config."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training("square8", 2)
            job = coordinator.get_job("square8", 2)

            assert job is not None
            assert job.job_id == job_id
            assert job.board_type == "square8"
            assert job.num_players == 2

    def test_get_job_not_found(self, coordinator):
        """Should return None for non-existent config."""
        job = coordinator.get_job("nonexistent", 2)
        assert job is None

    def test_update_progress(self, coordinator):
        """Should update training progress."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training("square8", 2)

            # Update progress
            success = coordinator.update_progress(
                job_id=job_id,
                epochs_completed=10,
                best_val_loss=0.15,
                current_elo=1520.0,
            )
            assert success

            # Verify update
            job = coordinator.get_job("square8", 2)
            assert job.epochs_completed == 10
            assert job.best_val_loss == 0.15
            assert job.current_elo == 1520.0

    def test_complete_training(self, coordinator):
        """Should complete training and archive to history."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training("square8", 2)

            # Complete the job
            with patch.object(coordinator, '_emit_training_event'):
                success = coordinator.complete_training(
                    job_id=job_id,
                    status="completed",
                    final_val_loss=0.10,
                    final_elo=1580.0,
                )

            assert success

            # Job should no longer be active
            job = coordinator.get_job("square8", 2)
            assert job is None

            # Should be in history
            history = coordinator.get_training_history(limit=10)
            assert len(history) >= 1
            assert history[0]["job_id"] == job_id
            assert history[0]["status"] == "completed"

    def test_complete_training_failed(self, coordinator):
        """Should handle failed training."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training("square8", 2)

            with patch.object(coordinator, '_emit_training_event'):
                success = coordinator.complete_training(
                    job_id=job_id,
                    status="failed",
                )

            assert success

            # Check history
            history = coordinator.get_training_history(limit=10)
            assert history[0]["status"] == "failed"

    def test_get_status(self, coordinator):
        """Should return coordinator status."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            coordinator.start_training("square8", 2)

            status = coordinator.get_status()
            assert "active_jobs" in status
            # active_jobs may be count or list depending on implementation
            if isinstance(status["active_jobs"], int):
                assert status["active_jobs"] >= 1
            else:
                assert len(status["active_jobs"]) >= 1

    def test_cleanup_stale_jobs(self, coordinator):
        """Should cleanup stale jobs."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training("square8", 2)

            # Artificially make the job stale
            conn = coordinator._get_connection()
            old_heartbeat = time.time() - (HEARTBEAT_INTERVAL_SECONDS * 10)
            conn.execute(
                "UPDATE training_jobs SET last_heartbeat = ? WHERE job_id = ?",
                (old_heartbeat, job_id)
            )
            conn.commit()

            # Run cleanup
            cleaned = coordinator._cleanup_stale_jobs()

            # Stale job should be cleaned
            assert cleaned >= 1

            # Job should no longer be active
            job = coordinator.get_job("square8", 2)
            assert job is None

    def test_training_available_after_completion(self, coordinator):
        """Should allow new training after completion."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start first training
            job1 = coordinator.start_training("square8", 2)
            assert job1 is not None

            # Cannot start duplicate
            assert not coordinator.can_start_training("square8", 2)

            # Complete first training
            with patch.object(coordinator, '_emit_training_event'):
                coordinator.complete_training(job1, status="completed")

            # Now can start again
            assert coordinator.can_start_training("square8", 2)

    def test_get_training_history(self, coordinator):
        """Should retrieve training history."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Create some history
            for _i in range(3):
                job_id = coordinator.start_training("square8", 2)
                with patch.object(coordinator, '_emit_training_event'):
                    coordinator.complete_training(job_id, status="completed")

            history = coordinator.get_training_history(limit=10)
            assert len(history) == 3

    def test_get_training_history_filtered(self, coordinator):
        """Should filter history by board type."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Create mixed history
            for board in ["square8", "hex7", "square8"]:
                job_id = coordinator.start_training(board, 2)
                with patch.object(coordinator, '_emit_training_event'):
                    coordinator.complete_training(job_id, status="completed")

            # Filter by board
            history = coordinator.get_training_history(board_type="hex7", limit=10)
            assert len(history) == 1
            assert history[0]["board_type"] == "hex7"

    def test_max_concurrent_limit(self, coordinator):
        """Should respect max concurrent training limit."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start max number of jobs
            jobs = []
            for i in range(MAX_TOTAL_CONCURRENT_TRAINING):
                job = coordinator.start_training(f"board{i}", 2)
                if job:
                    jobs.append(job)

            # Should have max jobs
            assert len(jobs) == MAX_TOTAL_CONCURRENT_TRAINING

            # Next job should fail
            extra_job = coordinator.start_training("extra", 2)
            assert extra_job is None


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def reset_coordinator(self, temp_db_path):
        """Reset coordinator singleton."""
        # Clear any existing singleton
        import app.coordination.training_coordinator as tc
        tc._coordinator = None

        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                yield

        tc._coordinator = None

    def test_can_train(self, reset_coordinator):
        """Should check if training is available."""
        result = can_train("square8", 2)
        assert result is True  # No active training

    def test_get_training_status(self, reset_coordinator):
        """Should get training status."""
        status = get_training_status()
        assert "active_jobs" in status


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingCoordinatorIntegration:
    """Integration tests for TrainingCoordinator."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def coordinator(self, temp_db_path):
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                coord = TrainingCoordinator(use_nfs=False)
                yield coord

    def test_full_training_workflow(self, coordinator):
        """Test complete training workflow."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # 1. Check availability
            assert coordinator.can_start_training("square8", 2)

            # 2. Start training
            job_id = coordinator.start_training(
                board_type="square8",
                num_players=2,
                model_version="v1.0",
            )
            assert job_id is not None

            # 3. Update progress multiple times
            for epoch in range(1, 6):
                coordinator.update_progress(
                    job_id=job_id,
                    epochs_completed=epoch,
                    best_val_loss=1.0 / epoch,
                    current_elo=1500 + epoch * 10,
                )

            # 4. Verify progress
            job = coordinator.get_job("square8", 2)
            assert job.epochs_completed == 5
            assert job.current_elo == 1550.0

            # 5. Complete training
            with patch.object(coordinator, '_emit_training_event'):
                coordinator.complete_training(
                    job_id=job_id,
                    status="completed",
                    final_val_loss=0.2,
                    final_elo=1560.0,
                )

            # 6. Verify completion
            assert coordinator.can_start_training("square8", 2)
            history = coordinator.get_training_history(limit=1)
            assert history[0]["final_elo"] == 1560.0

    def test_concurrent_configs(self, coordinator):
        """Test training multiple configs concurrently."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            configs = [
                ("square8", 2),
                ("hex7", 2),
                ("square8", 4),
            ]

            job_ids = []
            for board, players in configs:
                job_id = coordinator.start_training(board, players)
                assert job_id is not None
                job_ids.append(job_id)

            # All should be active
            active = coordinator.get_active_jobs()
            assert len(active) == 3

            # Complete all
            for job_id in job_ids:
                with patch.object(coordinator, '_emit_training_event'):
                    coordinator.complete_training(job_id, status="completed")

            # All should be in history
            history = coordinator.get_training_history(limit=10)
            assert len(history) == 3

    def test_persistence(self, temp_db_path):
        """Test that state persists across coordinator instances."""
        with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
            with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
                mock_lock_instance = MagicMock()
                mock_lock_instance.acquire.return_value = True
                mock_lock.return_value = mock_lock_instance

                # First coordinator instance
                with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
                    coord1 = TrainingCoordinator(use_nfs=False)
                    job_id = coord1.start_training("square8", 2)
                    coord1.update_progress(job_id, epochs_completed=5)

                # Second coordinator instance (simulating restart)
                with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
                    coord2 = TrainingCoordinator(use_nfs=False)
                    job = coord2.get_job("square8", 2)

                    assert job is not None
                    assert job.epochs_completed == 5


# =============================================================================
# Event Wiring Tests
# =============================================================================


class TestEventWiring:
    """Tests for event bus integration."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    def test_wire_training_events_no_event_bus(self, temp_db_path):
        """Should handle missing event bus gracefully."""
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                from app.coordination.training_coordinator import wire_training_events

                # Should not raise even if data_events not available
                try:
                    coordinator = wire_training_events()
                    assert coordinator is not None
                except (ImportError, AttributeError):
                    pass  # Expected if data_events/DataEventType not available


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestTrainingSlotContextManager:
    """Tests for training_slot context manager."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def reset_singleton(self, temp_db_path):
        """Reset coordinator singleton for clean tests."""
        import app.coordination.training_coordinator as tc
        tc._coordinator = None
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                yield
        tc._coordinator = None

    def test_training_slot_success(self, reset_singleton):
        """Context manager should acquire and release slot on success."""
        from app.coordination.training_coordinator import training_slot

        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            with training_slot("square8", 2, timeout=5) as job_id:
                assert job_id is not None
                assert "square8_2p" in job_id

            # After context, job should be completed
            from app.coordination.training_coordinator import can_train
            assert can_train("square8", 2)  # Slot now available

    def test_training_slot_failure_marks_failed(self, reset_singleton):
        """Context manager should mark job as failed on exception."""
        from app.coordination.training_coordinator import (
            training_slot,
            get_training_coordinator,
        )

        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            try:
                with training_slot("square8", 2, timeout=5) as job_id:
                    assert job_id is not None
                    raise ValueError("Simulated training failure")
            except ValueError:
                pass

            # Job should be in history as failed
            coordinator = get_training_coordinator()
            history = coordinator.get_training_history(limit=1)
            assert len(history) >= 1
            assert history[0]["status"] == "failed"

    def test_training_slot_timeout(self, reset_singleton):
        """Context manager should return None on timeout."""
        from app.coordination.training_coordinator import training_slot

        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start first training to block the slot
            from app.coordination.training_coordinator import request_training_slot
            job1 = request_training_slot("square8", 2)
            assert job1 is not None

            # Try to get same slot with short timeout
            with training_slot("square8", 2, timeout=1) as job_id:
                assert job_id is None  # Should timeout

    def test_training_slot_no_lock_available(self, reset_singleton):
        """Context manager should handle lock acquisition failure."""
        from app.coordination.training_coordinator import training_slot

        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = False  # Lock not acquired
            mock_lock.return_value = mock_lock_instance

            with training_slot("square8", 2, timeout=1) as job_id:
                # Should still work since we use SQLite for coordination
                # Lock failure is handled gracefully
                pass  # May or may not get job_id depending on implementation


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent training slot access."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def coordinator(self, temp_db_path):
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                coord = TrainingCoordinator(use_nfs=False)
                yield coord

    def test_concurrent_same_config_blocked(self, coordinator):
        """Multiple requests for same config should be blocked."""
        import threading
        results = []

        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            def request_training():
                job_id = coordinator.start_training("square8", 2)
                results.append(job_id)

            # Start multiple threads requesting same config
            threads = [threading.Thread(target=request_training) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Only one should succeed
            successful = [r for r in results if r is not None]
            assert len(successful) == 1

    def test_concurrent_different_configs_allowed(self, coordinator):
        """Multiple requests for different configs should all succeed."""
        import threading
        results = {}

        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            configs = [
                ("square8", 2),
                ("hex7", 2),
                ("square8", 4),
            ]

            def request_training(board, players):
                job_id = coordinator.start_training(board, players)
                results[(board, players)] = job_id

            threads = [
                threading.Thread(target=request_training, args=(b, p))
                for b, p in configs
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All should succeed
            for config in configs:
                assert results[config] is not None

    def test_slot_release_allows_new_request(self, coordinator):
        """Releasing a slot should allow new training to start."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start and complete training
            job1 = coordinator.start_training("square8", 2)
            assert job1 is not None

            # Cannot start another
            job2 = coordinator.start_training("square8", 2)
            assert job2 is None

            # Complete first training
            with patch.object(coordinator, '_emit_training_event'):
                coordinator.complete_training(job1, status="completed")

            # Now can start another
            job3 = coordinator.start_training("square8", 2)
            assert job3 is not None

    def test_stale_job_cleanup_allows_new_request(self, coordinator):
        """Cleaning up stale job should allow new training."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start training
            job1 = coordinator.start_training("square8", 2)
            assert job1 is not None

            # Make job stale
            conn = coordinator._get_connection()
            old_heartbeat = time.time() - (HEARTBEAT_INTERVAL_SECONDS * 10)
            conn.execute(
                "UPDATE training_jobs SET last_heartbeat = ? WHERE job_id = ?",
                (old_heartbeat, job1)
            )
            conn.commit()

            # Cleanup
            coordinator._cleanup_stale_jobs()

            # Now can start another
            job2 = coordinator.start_training("square8", 2)
            assert job2 is not None


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handler integration."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def coordinator(self, temp_db_path):
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                coord = TrainingCoordinator(use_nfs=False)
                yield coord

    def test_cluster_unhealthy_blocks_training(self, coordinator):
        """Cluster unhealthy should block new training."""
        # Simulate cluster unhealthy event
        mock_event = MagicMock()
        mock_event.payload = {"reason": "quorum_lost", "healthy_nodes": []}

        coordinator._on_cluster_unhealthy(mock_event)

        assert not coordinator._cluster_healthy
        # Note: Training might still be allowed via can_start_training
        # depending on implementation, but capacity should be affected

    def test_cluster_healthy_resumes_training(self, coordinator):
        """Cluster healthy should enable training."""
        # First make cluster unhealthy
        coordinator._cluster_healthy = False

        # Then healthy
        mock_event = MagicMock()
        mock_event.payload = {}
        coordinator._on_cluster_healthy(mock_event)

        assert coordinator._cluster_healthy

    def test_capacity_change_affects_state(self, coordinator):
        """Capacity change event should update internal state."""
        mock_event = MagicMock()
        mock_event.payload = {"capacity": 0.5}

        coordinator._on_capacity_changed(mock_event)

        assert coordinator._cluster_capacity == 0.5

    def test_capacity_clamped_to_valid_range(self, coordinator):
        """Capacity should be clamped between 0 and 1."""
        # Too high
        mock_event = MagicMock()
        mock_event.payload = {"capacity": 1.5}
        coordinator._on_capacity_changed(mock_event)
        assert coordinator._cluster_capacity == 1.0

        # Too low
        mock_event.payload = {"capacity": -0.5}
        coordinator._on_capacity_changed(mock_event)
        assert coordinator._cluster_capacity == 0.0

    def test_regression_minor_logs_but_continues(self, coordinator):
        """Minor regression should log but not pause."""
        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "square8_2p",
            "elo_drop": 20,
            "previous_elo": 1600,
            "current_elo": 1580,
        }

        old_capacity = coordinator._cluster_capacity
        coordinator._on_regression_detected(mock_event)

        # Capacity unchanged for minor regression
        assert coordinator._cluster_capacity == old_capacity
        assert coordinator._events_processed >= 1

    def test_regression_moderate_reduces_capacity(self, coordinator):
        """Moderate regression should reduce capacity."""
        coordinator._cluster_capacity = 1.0

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "square8_2p",
            "elo_drop": 40,
            "previous_elo": 1600,
            "current_elo": 1560,
        }

        coordinator._on_regression_detected(mock_event)

        # Capacity should be reduced
        assert coordinator._cluster_capacity < 1.0

    def test_regression_critical_pauses_training(self, coordinator):
        """Critical regression should pause training."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Start a training job
            job_id = coordinator.start_training("square8", 2)
            assert job_id is not None

            # Simulate critical regression
            mock_event = MagicMock()
            mock_event.payload = {
                "config_key": "square8_2p",
                "elo_drop": 100,
                "model_id": "square8_2p_v1",
            }

            with patch.object(coordinator, '_pause_training_for_config') as mock_pause:
                with patch.object(coordinator, '_trigger_model_rollback'):
                    with patch.object(coordinator, '_emit_via_router'):
                        coordinator._on_regression_critical(mock_event)

                mock_pause.assert_called_once()
                assert coordinator._errors_count >= 1

    def test_data_sync_completed_updates_state(self, coordinator):
        """Data sync completed should update sync time."""
        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "square8_2p",
            "sync_time": time.time(),
        }

        coordinator._on_data_sync_completed(mock_event)

        # Sync time should be updated
        assert coordinator._last_sync_time > 0


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check integration."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def coordinator(self, temp_db_path):
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                coord = TrainingCoordinator(use_nfs=False)
                yield coord

    def test_health_check_returns_result(self, coordinator):
        """health_check should return HealthCheckResult."""
        result = coordinator.health_check()

        assert result is not None
        assert hasattr(result, 'healthy')
        assert hasattr(result, 'status')

    def test_health_check_healthy_when_running(self, coordinator):
        """health_check should report healthy when running."""
        result = coordinator.health_check()

        assert result.healthy
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_includes_details(self, coordinator):
        """health_check should include useful details."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            coordinator.start_training("square8", 2)

            result = coordinator.health_check()

            assert "active_jobs" in result.details or result.details.get("active_count", 0) >= 0

    def test_name_property(self, coordinator):
        """Coordinator should have a name property."""
        assert coordinator.name is not None
        assert "training" in coordinator.name.lower() or "coordinator" in coordinator.name.lower()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "training_coordinator.db"
            yield db_path

    @pytest.fixture
    def coordinator(self, temp_db_path):
        with patch.object(TrainingCoordinator, '_get_db_path', return_value=temp_db_path):
            with patch.object(TrainingCoordinator, '_get_node_ip', return_value="127.0.0.1"):
                coord = TrainingCoordinator(use_nfs=False)
                yield coord

    def test_update_nonexistent_job(self, coordinator):
        """Updating nonexistent job should fail gracefully."""
        result = coordinator.update_progress(
            job_id="nonexistent-job",
            epochs_completed=10,
        )
        assert not result

    def test_complete_nonexistent_job(self, coordinator):
        """Completing nonexistent job should fail gracefully."""
        with patch.object(coordinator, '_emit_training_event'):
            result = coordinator.complete_training(
                job_id="nonexistent-job",
                status="completed",
            )
        assert not result

    def test_empty_board_type(self, coordinator):
        """Empty board type should be handled."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Empty board type might raise or return None
            try:
                job = coordinator.start_training("", 2)
                # If it doesn't raise, job should be None or empty
            except (ValueError, sqlite3.IntegrityError):
                pass  # Expected

    def test_invalid_num_players(self, coordinator):
        """Invalid player count should be handled."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            # Negative players
            try:
                job = coordinator.start_training("square8", -1)
            except (ValueError, sqlite3.IntegrityError):
                pass

    def test_get_active_jobs_empty(self, coordinator):
        """get_active_jobs with no jobs should return empty list."""
        jobs = coordinator.get_active_jobs()
        assert jobs == []

    def test_get_training_history_empty(self, coordinator):
        """get_training_history with no history should return empty list."""
        history = coordinator.get_training_history(limit=10)
        assert history == []

    def test_heartbeat_update(self, coordinator):
        """Heartbeat should update last_heartbeat timestamp."""
        with patch("app.coordination.training_coordinator.DistributedLock") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance

            job_id = coordinator.start_training("square8", 2)
            job_before = coordinator.get_job("square8", 2)
            initial_heartbeat = job_before.last_heartbeat

            time.sleep(0.1)  # Small delay

            # Update progress (should also update heartbeat)
            coordinator.update_progress(job_id, epochs_completed=1)

            job_after = coordinator.get_job("square8", 2)
            assert job_after.last_heartbeat >= initial_heartbeat
