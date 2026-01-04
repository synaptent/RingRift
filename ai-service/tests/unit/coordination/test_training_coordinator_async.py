"""Tests for TrainingCoordinator async wrappers added in Sprint 17.3.

Tests for async-safe training coordination operations that use asyncio.to_thread()
to prevent blocking the event loop.

January 4, 2026 (Sprint 17.3): SQLite async safety for training coordination.
"""

import asyncio
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.training_coordinator import (
    TrainingCoordinator,
    get_training_coordinator,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockDistributedLock:
    """Mock distributed lock for testing."""

    def __init__(self, name: str):
        self.name = name
        self._held = False

    def acquire(self, timeout: float = 30.0, blocking: bool = True) -> bool:
        self._held = True
        return True

    def release(self) -> None:
        self._held = False

    def is_held(self) -> bool:
        return self._held


@pytest.fixture
def training_coordinator(temp_db):
    """Create a training coordinator with temporary database."""
    # Mock the distributed lock to avoid actual lock acquisition
    with patch(
        "app.coordination.training_coordinator.DistributedLock",
        MockDistributedLock,
    ):
        with patch.object(
            TrainingCoordinator, "_get_db_path", return_value=temp_db / "training.db"
        ):
            coordinator = TrainingCoordinator(use_nfs=False)
            yield coordinator
            coordinator.close()


# =============================================================================
# can_start_training_async Tests
# =============================================================================


class TestCanStartTrainingAsync:
    """Tests for can_start_training_async method."""

    @pytest.mark.asyncio
    async def test_can_start_training_async_basic(self, training_coordinator):
        """Test basic async can_start_training check."""
        result = await training_coordinator.can_start_training_async("hex8", 2)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_can_start_training_async_runs_in_thread(self, training_coordinator):
        """Test that can_start_training_async runs in a separate thread."""
        main_thread_id = threading.current_thread().ident
        check_thread_ids = []

        original_can_start = training_coordinator.can_start_training

        def tracked_can_start(board_type, num_players):
            check_thread_ids.append(threading.current_thread().ident)
            return original_can_start(board_type, num_players)

        with patch.object(
            training_coordinator, "can_start_training", tracked_can_start
        ):
            await training_coordinator.can_start_training_async("hex8", 2)

        assert len(check_thread_ids) == 1
        assert check_thread_ids[0] != main_thread_id


# =============================================================================
# start_training_async Tests
# =============================================================================


class TestStartTrainingAsync:
    """Tests for start_training_async method."""

    @pytest.mark.asyncio
    async def test_start_training_async_basic(self, training_coordinator):
        """Test basic async training start."""
        job_id = await training_coordinator.start_training_async("hex8", 2)
        # May return None if slot not available, or job_id if started
        assert job_id is None or isinstance(job_id, str)

    @pytest.mark.asyncio
    async def test_start_training_async_with_metadata(self, training_coordinator):
        """Test async training start with metadata."""
        metadata = {"epochs": 50, "batch_size": 256}
        job_id = await training_coordinator.start_training_async(
            "hex8", 2, metadata=metadata
        )
        assert job_id is None or isinstance(job_id, str)


# =============================================================================
# complete_training_async Tests
# =============================================================================


class TestCompleteTrainingAsync:
    """Tests for complete_training_async method."""

    @pytest.mark.asyncio
    async def test_complete_training_async_not_found(self, training_coordinator):
        """Test completing nonexistent training returns False."""
        result = await training_coordinator.complete_training_async(
            "nonexistent-job-id"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_training_async_full_cycle(self, training_coordinator):
        """Test complete training cycle."""
        # Start training first
        job_id = await training_coordinator.start_training_async("hex8", 2)

        if job_id:
            # Complete it
            result = await training_coordinator.complete_training_async(
                job_id, status="completed", final_val_loss=0.05, final_elo=1450.0
            )
            assert result is True


# =============================================================================
# get_status_async Tests
# =============================================================================


class TestGetStatusAsync:
    """Tests for get_status_async method."""

    @pytest.mark.asyncio
    async def test_get_status_async_basic(self, training_coordinator):
        """Test basic async status retrieval."""
        status = await training_coordinator.get_status_async()

        assert isinstance(status, dict)
        assert "active_jobs" in status
        assert "max_concurrent" in status
        assert "slots_available" in status

    @pytest.mark.asyncio
    async def test_get_status_async_runs_in_thread(self, training_coordinator):
        """Test that get_status_async runs in a separate thread."""
        main_thread_id = threading.current_thread().ident
        get_thread_ids = []

        original_get_status = training_coordinator.get_status

        def tracked_get_status():
            get_thread_ids.append(threading.current_thread().ident)
            return original_get_status()

        with patch.object(training_coordinator, "get_status", tracked_get_status):
            await training_coordinator.get_status_async()

        assert len(get_thread_ids) == 1
        assert get_thread_ids[0] != main_thread_id


# =============================================================================
# health_check_async Tests
# =============================================================================


class TestHealthCheckAsync:
    """Tests for health_check_async method."""

    @pytest.mark.asyncio
    async def test_health_check_async_basic(self, training_coordinator):
        """Test basic async health check."""
        result = await training_coordinator.health_check_async()

        # Should return HealthCheckResult-like object
        assert hasattr(result, "healthy")
        assert hasattr(result, "status")

    @pytest.mark.asyncio
    async def test_health_check_async_is_healthy(self, training_coordinator):
        """Test that fresh coordinator is healthy."""
        result = await training_coordinator.health_check_async()
        assert result.healthy is True


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAsyncAccess:
    """Tests for concurrent async access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_can_start_checks(self, training_coordinator):
        """Test multiple concurrent can_start_training checks."""
        configs = [
            ("hex8", 2),
            ("hex8", 3),
            ("hex8", 4),
            ("square8", 2),
            ("square8", 3),
        ]

        async def check_config(board_type, num_players):
            return await training_coordinator.can_start_training_async(
                board_type, num_players
            )

        tasks = [check_config(bt, np) for bt, np in configs]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_status_and_health(self, training_coordinator):
        """Test concurrent status and health check calls."""
        tasks = [
            training_coordinator.get_status_async(),
            training_coordinator.health_check_async(),
            training_coordinator.get_status_async(),
            training_coordinator.health_check_async(),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert isinstance(results[0], dict)  # status
        assert hasattr(results[1], "healthy")  # health
        assert isinstance(results[2], dict)  # status
        assert hasattr(results[3], "healthy")  # health
