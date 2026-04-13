"""Tests for WorkQueue async wrappers added in Sprint 17.3.

Tests for async-safe work queue operations that use asyncio.to_thread()
to prevent blocking the event loop.

January 4, 2026 (Sprint 17.3): SQLite async safety for work queue.
"""

import asyncio
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.work_queue import (
    WorkItem,
    WorkQueue,
    WorkStatus,
    WorkType,
    get_work_queue,
    reset_work_queue,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield Path(db_path)
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def work_queue(temp_db):
    """Create a work queue with temporary database."""
    # Reset singleton before each test
    reset_work_queue()
    queue = WorkQueue(db_path=temp_db)
    yield queue
    queue.close()


@pytest.fixture
def sample_work_item():
    """Create a sample work item for testing."""
    return WorkItem(
        work_type=WorkType.SELFPLAY,
        config={
            "board_type": "hex8",
            "num_players": 2,
        },
        priority=100,
    )


# =============================================================================
# add_work_async Tests
# =============================================================================


class TestAddWorkAsync:
    """Tests for add_work_async method."""

    @pytest.mark.asyncio
    async def test_add_work_async_basic(self, work_queue, sample_work_item):
        """Test basic async work addition."""
        work_id = await work_queue.add_work_async(sample_work_item)
        assert work_id == sample_work_item.work_id
        assert work_id in work_queue.items

    @pytest.mark.asyncio
    async def test_add_work_async_runs_in_thread(self, work_queue, sample_work_item):
        """Test that add_work_async runs in a separate thread."""
        main_thread_id = threading.current_thread().ident
        add_thread_ids = []

        original_add = work_queue.add_work

        def tracked_add(item, force=False):
            add_thread_ids.append(threading.current_thread().ident)
            return original_add(item, force)

        with patch.object(work_queue, "add_work", tracked_add):
            await work_queue.add_work_async(sample_work_item)

        assert len(add_thread_ids) == 1
        # The operation should run in a different thread (thread pool)
        assert add_thread_ids[0] != main_thread_id

    @pytest.mark.asyncio
    async def test_add_work_async_with_force(self, work_queue, sample_work_item):
        """Test add_work_async with force=True."""
        work_id = await work_queue.add_work_async(sample_work_item, force=True)
        assert work_id == sample_work_item.work_id

    @pytest.mark.asyncio
    async def test_add_work_async_exception_propagates(self, work_queue):
        """Test that exceptions from add_work propagate correctly."""
        invalid_item = MagicMock()
        invalid_item.work_id = "test"
        invalid_item.work_type = WorkType.SELFPLAY
        invalid_item.config = {}
        invalid_item.to_dict.side_effect = ValueError("Invalid item")

        with pytest.raises(ValueError):
            await work_queue.add_work_async(invalid_item)


# =============================================================================
# add_work_batch_async Tests
# =============================================================================


class TestAddWorkBatchAsync:
    """Tests for add_work_batch_async method."""

    @pytest.mark.asyncio
    async def test_add_work_batch_async_basic(self, work_queue):
        """Test batch async work addition."""
        items = [
            WorkItem(
                work_type=WorkType.SELFPLAY,
                config={"board_type": "hex8", "num_players": 2},
                priority=i * 10,
            )
            for i in range(5)
        ]

        work_ids = await work_queue.add_work_batch_async(items)
        assert len(work_ids) == 5
        for item in items:
            assert item.work_id in work_queue.items


# =============================================================================
# claim_work_async Tests
# =============================================================================


class TestClaimWorkAsync:
    """Tests for claim_work_async method."""

    @pytest.mark.asyncio
    async def test_claim_work_async_basic(self, work_queue, sample_work_item):
        """Test basic async work claiming."""
        await work_queue.add_work_async(sample_work_item)

        claimed = await work_queue.claim_work_async(
            node_id="test-node-1",
            capabilities=["selfplay"],
        )

        assert claimed is not None
        assert claimed.work_id == sample_work_item.work_id
        assert claimed.claimed_by == "test-node-1"

    @pytest.mark.asyncio
    async def test_claim_work_async_empty_queue(self, work_queue):
        """Test claiming from empty queue returns None."""
        claimed = await work_queue.claim_work_async(
            node_id="test-node-1",
            capabilities=["selfplay"],
        )
        assert claimed is None

    @pytest.mark.asyncio
    async def test_claim_work_async_capability_filter(self, work_queue, sample_work_item):
        """Test that capability filter works."""
        await work_queue.add_work_async(sample_work_item)

        # Wrong capability should return None
        claimed = await work_queue.claim_work_async(
            node_id="test-node-1",
            capabilities=["training"],  # Not selfplay
        )
        assert claimed is None

        # Correct capability should work
        claimed = await work_queue.claim_work_async(
            node_id="test-node-1",
            capabilities=["selfplay"],
        )
        assert claimed is not None

    @pytest.mark.asyncio
    async def test_claim_work_async_respects_node_role_selfplay_gate(
        self,
        work_queue,
        sample_work_item,
    ):
        """Trainer-role nodes must not claim P2P selfplay work."""
        await work_queue.add_work_async(sample_work_item)

        with patch(
            "app.config.node_roles.node_allows_work_type",
            return_value=False,
        ) as mock_gate:
            claimed = await work_queue.claim_work_async(
                node_id="gh200-8",
                capabilities=["selfplay"],
            )

        assert claimed is None
        mock_gate.assert_called_once_with(
            "gh200-8",
            "selfplay",
            config_key="hex8_2p",
        )


# =============================================================================
# start_work_async Tests
# =============================================================================


class TestStartWorkAsync:
    """Tests for start_work_async method."""

    @pytest.mark.asyncio
    async def test_start_work_async_basic(self, work_queue, sample_work_item):
        """Test basic async work start."""
        await work_queue.add_work_async(sample_work_item)
        await work_queue.claim_work_async(node_id="test-node-1")

        result = await work_queue.start_work_async(sample_work_item.work_id)
        assert result is True

        # Verify status changed
        item = work_queue.items.get(sample_work_item.work_id)
        assert item.status == WorkStatus.RUNNING


# =============================================================================
# complete_work_async Tests
# =============================================================================


class TestCompleteWorkAsync:
    """Tests for complete_work_async method."""

    @pytest.mark.asyncio
    async def test_complete_work_async_basic(self, work_queue, sample_work_item):
        """Test basic async work completion."""
        await work_queue.add_work_async(sample_work_item)
        await work_queue.claim_work_async(node_id="test-node-1")
        await work_queue.start_work_async(sample_work_item.work_id)

        result = await work_queue.complete_work_async(
            sample_work_item.work_id,
            result={"games_generated": 100},
        )
        assert result is True

        # Verify status changed
        item = work_queue.items.get(sample_work_item.work_id)
        assert item.status == WorkStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_complete_work_async_nonexistent(self, work_queue):
        """Test completing nonexistent work returns False."""
        result = await work_queue.complete_work_async("nonexistent-id")
        assert result is False


# =============================================================================
# fail_work_async Tests
# =============================================================================


class TestFailWorkAsync:
    """Tests for fail_work_async method."""

    @pytest.mark.asyncio
    async def test_fail_work_async_basic(self, work_queue, sample_work_item):
        """Test basic async work failure."""
        sample_work_item.max_attempts = 1  # Will fail permanently
        await work_queue.add_work_async(sample_work_item)
        await work_queue.claim_work_async(node_id="test-node-1")

        result = await work_queue.fail_work_async(
            sample_work_item.work_id,
            error="GPU OOM",
        )
        assert result is True

        # Verify status changed
        item = work_queue.items.get(sample_work_item.work_id)
        assert item.status == WorkStatus.FAILED


# =============================================================================
# get_work_async Tests
# =============================================================================


class TestGetWorkAsync:
    """Tests for get_work_async method."""

    @pytest.mark.asyncio
    async def test_get_work_async_exists(self, work_queue, sample_work_item):
        """Test getting an existing work item."""
        await work_queue.add_work_async(sample_work_item)

        item = await work_queue.get_work_async(sample_work_item.work_id)
        assert item is not None
        assert item.work_id == sample_work_item.work_id

    @pytest.mark.asyncio
    async def test_get_work_async_not_exists(self, work_queue):
        """Test getting a nonexistent work item returns None."""
        item = await work_queue.get_work_async("nonexistent-id")
        assert item is None


# =============================================================================
# cancel_work_async Tests
# =============================================================================


class TestCancelWorkAsync:
    """Tests for cancel_work_async method."""

    @pytest.mark.asyncio
    async def test_cancel_work_async_basic(self, work_queue, sample_work_item):
        """Test basic async work cancellation."""
        await work_queue.add_work_async(sample_work_item)

        result = await work_queue.cancel_work_async(sample_work_item.work_id)
        assert result is True

        # Verify status changed
        item = work_queue.items.get(sample_work_item.work_id)
        assert item.status == WorkStatus.CANCELLED


# =============================================================================
# get_pending_work_async Tests
# =============================================================================


class TestGetPendingWorkAsync:
    """Tests for get_pending_work_async method."""

    @pytest.mark.asyncio
    async def test_get_pending_work_async_all(self, work_queue):
        """Test getting all pending work items."""
        items = [
            WorkItem(
                work_type=WorkType.SELFPLAY,
                config={"board_type": "hex8", "num_players": 2},
                priority=(i + 1) * 10,
            )
            for i in range(3)
        ]

        for item in items:
            await work_queue.add_work_async(item)

        pending = await work_queue.get_pending_work_async()
        assert len(pending) == 3
        # Should be sorted by priority (highest first)
        assert pending[0].priority == 30
        assert pending[1].priority == 20
        assert pending[2].priority == 10

    @pytest.mark.asyncio
    async def test_get_pending_work_async_filtered(self, work_queue):
        """Test getting pending work with type filter."""
        items = [
            WorkItem(
                work_type=WorkType.SELFPLAY,
                config={"board_type": "hex8", "num_players": 2},
                priority=10,
            ),
            WorkItem(
                work_type=WorkType.TRAINING,
                config={"board_type": "hex8", "num_players": 2},
                priority=20,
            ),
        ]

        for item in items:
            await work_queue.add_work_async(item)

        pending = await work_queue.get_pending_work_async(work_type=WorkType.SELFPLAY)
        assert len(pending) == 1
        assert pending[0].work_type == WorkType.SELFPLAY


# =============================================================================
# health_check_async Tests
# =============================================================================


class TestHealthCheckAsync:
    """Tests for health_check_async method."""

    @pytest.mark.asyncio
    async def test_health_check_async_basic(self, work_queue, sample_work_item):
        """Test async health check."""
        await work_queue.add_work_async(sample_work_item)

        result = await work_queue.health_check_async()
        assert result.healthy is True
        assert result.details["pending"] == 1
        assert result.details["total_items"] == 1


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAsyncAccess:
    """Tests for concurrent async access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_add_work(self, work_queue):
        """Test multiple concurrent add_work_async calls."""
        items = [
            WorkItem(
                work_type=WorkType.SELFPLAY,
                config={"board_type": "hex8", "num_players": 2},
                priority=i,
            )
            for i in range(10)
        ]

        # Add all items concurrently
        tasks = [work_queue.add_work_async(item) for item in items]
        work_ids = await asyncio.gather(*tasks)

        assert len(work_ids) == 10
        assert len(set(work_ids)) == 10  # All unique

    @pytest.mark.asyncio
    async def test_concurrent_claim_work(self, work_queue):
        """Test concurrent claiming doesn't cause race conditions."""
        # Add several work items
        items = [
            WorkItem(
                work_type=WorkType.SELFPLAY,
                config={"board_type": "hex8", "num_players": 2},
                priority=i,
            )
            for i in range(5)
        ]

        for item in items:
            await work_queue.add_work_async(item)

        # Try to claim concurrently from different "nodes"
        async def claim_from_node(node_id: str):
            return await work_queue.claim_work_async(
                node_id=node_id,
                capabilities=["selfplay"],
            )

        tasks = [claim_from_node(f"node-{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Should have 5 successful claims, 5 None results
        claimed = [r for r in results if r is not None]
        assert len(claimed) == 5

        # Each item should only be claimed once
        claimed_ids = [r.work_id for r in claimed]
        assert len(set(claimed_ids)) == 5
