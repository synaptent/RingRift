"""Tests for DeliveryRetryQueue.

Tests the retry queue with exponential backoff for failed deliveries.
December 2025: Created as part of Phase 3 infrastructure improvements.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.delivery_ledger import (
    DeliveryLedger,
    DeliveryRecord,
    DeliveryStatus,
)
from app.coordination.delivery_retry_queue import (
    DeliveryRetryQueue,
    RetryConfig,
    PendingRetry,
    get_delivery_retry_queue,
    reset_delivery_retry_queue,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_ledger.db"


@pytest.fixture
def ledger(temp_db):
    """Create a test ledger instance."""
    return DeliveryLedger(db_path=temp_db)


@pytest.fixture
def config():
    """Create a test retry config with short delays."""
    return RetryConfig(
        initial_delay_seconds=0.1,
        max_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_retries=4,
        process_interval_seconds=0.1,
        max_concurrent_retries=5,
    )


@pytest.fixture
def queue(ledger, config):
    """Create a test retry queue."""
    return DeliveryRetryQueue(ledger=ledger, config=config)


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.initial_delay_seconds == 30.0
        assert config.max_delay_seconds == 240.0
        assert config.backoff_multiplier == 2.0
        assert config.max_retries == 4

    def test_calculate_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay_seconds=30.0,
            backoff_multiplier=2.0,
            max_delay_seconds=240.0,
        )

        # First retry: 30 * 2^0 = 30
        assert config.calculate_delay(0) == 30.0

        # Second retry: 30 * 2^1 = 60
        assert config.calculate_delay(1) == 60.0

        # Third retry: 30 * 2^2 = 120
        assert config.calculate_delay(2) == 120.0

        # Fourth retry: 30 * 2^3 = 240 (at max)
        assert config.calculate_delay(3) == 240.0

        # Fifth retry: would be 30 * 2^4 = 480, but capped at 240
        assert config.calculate_delay(4) == 240.0

    def test_calculate_delay_caps_at_max(self):
        """Test that delay is capped at max_delay_seconds."""
        config = RetryConfig(
            initial_delay_seconds=100.0,
            backoff_multiplier=10.0,
            max_delay_seconds=200.0,
        )

        # Would be 1000 but capped at 200
        assert config.calculate_delay(1) == 200.0


class TestPendingRetry:
    """Test PendingRetry dataclass."""

    def test_ordering(self):
        """Test that PendingRetry orders by retry_at."""
        now = time.time()

        retry1 = PendingRetry(retry_at=now + 10, delivery_id="id1", retry_count=0)
        retry2 = PendingRetry(retry_at=now + 5, delivery_id="id2", retry_count=1)
        retry3 = PendingRetry(retry_at=now + 15, delivery_id="id3", retry_count=2)

        sorted_retries = sorted([retry1, retry2, retry3])

        assert sorted_retries[0].delivery_id == "id2"  # Earliest
        assert sorted_retries[1].delivery_id == "id1"
        assert sorted_retries[2].delivery_id == "id3"  # Latest


class TestDeliveryRetryQueue:
    """Test DeliveryRetryQueue functionality."""

    def test_initialization(self, queue):
        """Test queue initializes correctly."""
        assert queue._running is False
        assert len(queue._queue) == 0
        assert len(queue._active_retries) == 0

    def test_enqueue_retry(self, queue, ledger):
        """Test enqueuing a failed delivery for retry."""
        # Create a failed delivery
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )
        ledger.record_delivery_failed(record.delivery_id, "Connection timeout")

        # Get updated record with retry_count
        updated = ledger.get_delivery(record.delivery_id)

        # Enqueue for retry
        queue.enqueue_retry(updated)

        assert queue.get_queue_size() == 1
        assert record.delivery_id in queue._delivery_ids_in_queue

    def test_enqueue_retry_already_in_queue(self, queue, ledger):
        """Test that duplicate enqueues are ignored."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )
        ledger.record_delivery_failed(record.delivery_id, "Error")
        updated = ledger.get_delivery(record.delivery_id)

        queue.enqueue_retry(updated)
        queue.enqueue_retry(updated)

        assert queue.get_queue_size() == 1

    def test_enqueue_retry_max_retries_reached(self, queue, ledger):
        """Test that deliveries at max retries are not enqueued."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
            max_retries=1,
        )
        # Fail once to increment retry_count to 1, matching max_retries
        ledger.record_delivery_failed(record.delivery_id, "Error")

        updated = ledger.get_delivery(record.delivery_id)
        assert updated.retry_count == 1
        assert updated.max_retries == 1
        assert not updated.can_retry

        queue.enqueue_retry(updated)

        assert queue.get_queue_size() == 0
        assert queue._stats["retries_exhausted"] == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, queue):
        """Test starting and stopping the queue."""
        assert queue._running is False

        await queue.start()
        assert queue._running is True
        assert queue._processing_task is not None

        await queue.stop()
        assert queue._running is False

    @pytest.mark.asyncio
    async def test_start_loads_retryable_deliveries(self, ledger, config):
        """Test that starting loads existing retryable deliveries."""
        # Create some failed deliveries
        for i in range(3):
            record = ledger.record_delivery_started(
                data_type="model",
                data_path=f"/models/{i}.pth",
                target_node="test-node",
            )
            ledger.record_delivery_failed(record.delivery_id, "Error")

        # Create queue and start
        queue = DeliveryRetryQueue(ledger=ledger, config=config)
        await queue.start()

        try:
            assert queue.get_queue_size() == 3
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_process_due_retries(self, ledger, config):
        """Test processing of due retries."""
        # Create a mock handler
        handler_called = asyncio.Event()
        handler_record = None

        async def mock_handler(record):
            nonlocal handler_record
            handler_record = record
            handler_called.set()
            return True

        queue = DeliveryRetryQueue(
            ledger=ledger,
            config=config,
            retry_handler=mock_handler,
        )

        # Create and fail a delivery
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )
        ledger.record_delivery_failed(record.delivery_id, "Error")

        # Start queue
        await queue.start()

        try:
            # Wait for handler to be called (should be quick with short delays)
            await asyncio.wait_for(handler_called.wait(), timeout=2.0)

            assert handler_record is not None
            assert handler_record.delivery_id == record.delivery_id
            assert queue._stats["retries_attempted"] >= 1
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_failed_retry_reenqueues(self, ledger, config):
        """Test that failed retries are re-enqueued."""
        call_count = 0

        async def mock_handler(record):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return False  # Fail first attempt
            return True  # Succeed second attempt

        queue = DeliveryRetryQueue(
            ledger=ledger,
            config=config,
            retry_handler=mock_handler,
        )

        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )
        ledger.record_delivery_failed(record.delivery_id, "Initial error")

        await queue.start()

        try:
            # Wait for both retries to complete
            await asyncio.sleep(0.5)

            assert call_count >= 2
            assert queue._stats["retries_succeeded"] >= 1
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_max_concurrent_retries(self, ledger):
        """Test concurrent retry limiting."""
        config = RetryConfig(
            initial_delay_seconds=0.0,  # Immediate retry
            max_concurrent_retries=2,
            process_interval_seconds=0.05,
        )

        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_handler(record):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.2)  # Simulate work

            async with lock:
                concurrent_count -= 1

            return True

        queue = DeliveryRetryQueue(
            ledger=ledger,
            config=config,
            retry_handler=mock_handler,
        )

        # Create 5 failed deliveries
        for i in range(5):
            record = ledger.record_delivery_started(
                data_type="model",
                data_path=f"/models/{i}.pth",
                target_node="test-node",
            )
            ledger.record_delivery_failed(record.delivery_id, "Error")

        await queue.start()

        try:
            await asyncio.sleep(1.0)  # Let some retries process

            # Max concurrent should never exceed 2
            assert max_concurrent <= 2
        finally:
            await queue.stop()

    def test_get_stats(self, queue):
        """Test getting queue statistics."""
        stats = queue.get_stats()

        assert "queue_size" in stats
        assert "active_retries" in stats
        assert "running" in stats
        assert "retries_attempted" in stats
        assert "retries_succeeded" in stats
        assert "retries_failed" in stats
        assert "retries_exhausted" in stats

    def test_get_next_retry_time(self, queue, ledger):
        """Test getting next scheduled retry time."""
        assert queue.get_next_retry_time() is None

        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="test-node",
        )
        ledger.record_delivery_failed(record.delivery_id, "Error")
        updated = ledger.get_delivery(record.delivery_id)

        queue.enqueue_retry(updated)

        next_time = queue.get_next_retry_time()
        assert next_time is not None
        assert next_time > time.time()  # Should be in the future

    @pytest.mark.asyncio
    async def test_handler_exception_handled(self, ledger, config):
        """Test that handler exceptions are handled gracefully."""
        async def error_handler(record):
            raise RuntimeError("Handler error")

        queue = DeliveryRetryQueue(
            ledger=ledger,
            config=config,
            retry_handler=error_handler,
        )

        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="test-node",
        )
        ledger.record_delivery_failed(record.delivery_id, "Error")

        await queue.start()

        try:
            await asyncio.sleep(0.5)

            # Queue should still be running despite handler errors
            assert queue._running is True
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_no_handler_leaves_delivery_in_failed_state(self, ledger, config):
        """Test behavior when no retry handler is configured."""
        queue = DeliveryRetryQueue(
            ledger=ledger,
            config=config,
            retry_handler=None,  # No handler
        )

        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="test-node",
        )
        ledger.record_delivery_failed(record.delivery_id, "Error")

        await queue.start()

        try:
            await asyncio.sleep(0.3)

            # Delivery should still be in failed state
            updated = ledger.get_delivery(record.delivery_id)
            assert updated.status == DeliveryStatus.FAILED
        finally:
            await queue.stop()


class TestSingletonPattern:
    """Test singleton pattern for retry queue."""

    def test_get_delivery_retry_queue_returns_same_instance(self):
        """Test that get_delivery_retry_queue returns singleton."""
        reset_delivery_retry_queue()

        queue1 = get_delivery_retry_queue()
        queue2 = get_delivery_retry_queue()

        assert queue1 is queue2

        reset_delivery_retry_queue()

    def test_reset_delivery_retry_queue(self):
        """Test resetting the singleton instance."""
        reset_delivery_retry_queue()

        queue1 = get_delivery_retry_queue()
        reset_delivery_retry_queue()
        queue2 = get_delivery_retry_queue()

        assert queue1 is not queue2

        reset_delivery_retry_queue()


class TestExponentialBackoff:
    """Test exponential backoff behavior."""

    def test_backoff_sequence(self):
        """Test the expected backoff sequence."""
        config = RetryConfig(
            initial_delay_seconds=30.0,
            backoff_multiplier=2.0,
            max_delay_seconds=240.0,
        )

        expected_delays = [30.0, 60.0, 120.0, 240.0]

        for retry_count, expected in enumerate(expected_delays):
            actual = config.calculate_delay(retry_count)
            assert actual == expected, f"Retry {retry_count}: expected {expected}, got {actual}"

    def test_backoff_with_different_multiplier(self):
        """Test backoff with 1.5x multiplier."""
        config = RetryConfig(
            initial_delay_seconds=30.0,
            backoff_multiplier=1.5,
            max_delay_seconds=500.0,
        )

        assert config.calculate_delay(0) == 30.0
        assert config.calculate_delay(1) == 45.0  # 30 * 1.5
        assert config.calculate_delay(2) == 67.5  # 30 * 1.5^2
        assert config.calculate_delay(3) == pytest.approx(101.25)  # 30 * 1.5^3
