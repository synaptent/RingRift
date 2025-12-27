"""Delivery retry queue with exponential backoff.

Manages automatic retry of failed data deliveries with configurable
exponential backoff delays.

December 2025: Created as part of Phase 3 infrastructure improvements.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from app.coordination.delivery_ledger import (
    DeliveryLedger,
    DeliveryRecord,
    DeliveryStatus,
    get_delivery_ledger,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DeliveryRetryQueue",
    "RetryConfig",
    "get_delivery_retry_queue",
    "reset_delivery_retry_queue",
]

# Singleton instance
_retry_queue_instance: "DeliveryRetryQueue | None" = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Retry settings
    initial_delay_seconds: float = 30.0
    max_delay_seconds: float = 240.0
    backoff_multiplier: float = 2.0
    max_retries: int = 4

    # Queue processing
    process_interval_seconds: float = 10.0
    max_concurrent_retries: int = 5

    # Age limits
    max_retry_age_hours: float = 24.0

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay for a given retry count.

        Uses exponential backoff: delay = initial * (multiplier ^ retry_count)

        Args:
            retry_count: Number of previous retries

        Returns:
            Delay in seconds, capped at max_delay_seconds
        """
        delay = self.initial_delay_seconds * (self.backoff_multiplier ** retry_count)
        return min(delay, self.max_delay_seconds)


@dataclass(order=True)
class PendingRetry:
    """A retry scheduled in the queue.

    Ordered by retry_at for priority queue usage.
    """
    retry_at: float
    delivery_id: str = field(compare=False)
    retry_count: int = field(compare=False)


class DeliveryRetryQueue:
    """Manages automatic retry of failed deliveries with exponential backoff.

    Features:
    - Priority queue ordered by retry time
    - Exponential backoff delays (30s → 60s → 120s → 240s)
    - Concurrent retry processing
    - Automatic loading from persistent ledger on start
    - Integration with DeliveryLedger for state tracking
    """

    def __init__(
        self,
        ledger: DeliveryLedger | None = None,
        config: RetryConfig | None = None,
        retry_handler: Callable[[DeliveryRecord], Coroutine[Any, Any, bool]] | None = None,
    ):
        """Initialize the retry queue.

        Args:
            ledger: DeliveryLedger to use for persistence. Uses singleton if not specified.
            config: RetryConfig for retry behavior. Uses defaults if not specified.
            retry_handler: Async callable that performs the actual retry. Takes a
                DeliveryRecord and returns True on success, False on failure.
        """
        self.ledger = ledger or get_delivery_ledger()
        self.config = config or RetryConfig()
        self._retry_handler = retry_handler

        # Priority queue of pending retries
        self._queue: list[PendingRetry] = []
        self._delivery_ids_in_queue: set[str] = set()

        # State
        self._running = False
        self._processing_task: asyncio.Task | None = None
        self._active_retries: set[str] = set()
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "retries_attempted": 0,
            "retries_succeeded": 0,
            "retries_failed": 0,
            "retries_exhausted": 0,
        }

    async def start(self) -> None:
        """Start the retry queue processor.

        Loads existing retryable deliveries from the ledger and starts
        the background processing task.
        """
        if self._running:
            logger.debug("[DeliveryRetryQueue] Already running")
            return

        self._running = True

        # Load existing retryable deliveries
        await self._load_retryable_deliveries()

        # Start processing task
        self._processing_task = asyncio.create_task(
            self._processing_loop(),
            name="delivery_retry_processor",
        )

        logger.info(
            f"[DeliveryRetryQueue] Started with {len(self._queue)} pending retries"
        )

    async def stop(self) -> None:
        """Stop the retry queue processor gracefully."""
        if not self._running:
            return

        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        logger.info("[DeliveryRetryQueue] Stopped")

    def enqueue_retry(self, record: DeliveryRecord) -> None:
        """Enqueue a failed delivery for retry.

        Args:
            record: The failed DeliveryRecord to retry
        """
        if record.delivery_id in self._delivery_ids_in_queue:
            logger.debug(
                f"[DeliveryRetryQueue] Delivery {record.delivery_id[:8]} "
                "already in queue"
            )
            return

        if not record.can_retry:
            logger.debug(
                f"[DeliveryRetryQueue] Delivery {record.delivery_id[:8]} "
                f"cannot retry (count={record.retry_count}, max={record.max_retries})"
            )
            self._stats["retries_exhausted"] += 1
            return

        # Calculate retry time with exponential backoff
        delay = self.config.calculate_delay(record.retry_count)
        retry_at = time.time() + delay

        pending = PendingRetry(
            retry_at=retry_at,
            delivery_id=record.delivery_id,
            retry_count=record.retry_count,
        )

        heapq.heappush(self._queue, pending)
        self._delivery_ids_in_queue.add(record.delivery_id)

        logger.debug(
            f"[DeliveryRetryQueue] Enqueued {record.delivery_id[:8]} "
            f"for retry in {delay:.1f}s (attempt {record.retry_count + 1}/{record.max_retries})"
        )

    async def _load_retryable_deliveries(self) -> None:
        """Load existing retryable deliveries from the ledger."""
        retryable = self.ledger.get_retryable_deliveries(
            max_age_hours=self.config.max_retry_age_hours
        )

        for record in retryable:
            self.enqueue_retry(record)

        if retryable:
            logger.info(
                f"[DeliveryRetryQueue] Loaded {len(retryable)} retryable deliveries"
            )

    async def _processing_loop(self) -> None:
        """Main processing loop for retries."""
        while self._running:
            try:
                await self._process_due_retries()
                await asyncio.sleep(self.config.process_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[DeliveryRetryQueue] Error in processing loop: {e}")
                await asyncio.sleep(self.config.process_interval_seconds)

    async def _process_due_retries(self) -> None:
        """Process all retries that are due."""
        now = time.time()
        tasks = []

        async with self._lock:
            # Process up to max_concurrent_retries
            while (
                self._queue
                and self._queue[0].retry_at <= now
                and len(self._active_retries) < self.config.max_concurrent_retries
            ):
                pending = heapq.heappop(self._queue)

                if pending.delivery_id in self._active_retries:
                    # Already being processed
                    continue

                self._delivery_ids_in_queue.discard(pending.delivery_id)
                self._active_retries.add(pending.delivery_id)

                task = asyncio.create_task(
                    self._execute_retry(pending.delivery_id),
                    name=f"retry_{pending.delivery_id[:8]}",
                )
                tasks.append(task)

        # Wait for all retry tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_retry(self, delivery_id: str) -> None:
        """Execute a single retry attempt.

        Args:
            delivery_id: ID of the delivery to retry
        """
        try:
            # Get current record state
            record = self.ledger.get_delivery(delivery_id)
            if not record:
                logger.warning(
                    f"[DeliveryRetryQueue] Delivery {delivery_id[:8]} not found"
                )
                return

            if not record.can_retry:
                logger.debug(
                    f"[DeliveryRetryQueue] Delivery {delivery_id[:8]} "
                    "no longer retryable"
                )
                return

            # Mark as retrying
            self.ledger.record_retry_started(delivery_id)
            self._stats["retries_attempted"] += 1

            logger.info(
                f"[DeliveryRetryQueue] Retrying {delivery_id[:8]} "
                f"(attempt {record.retry_count + 1}/{record.max_retries}) "
                f"to {record.target_node}"
            )

            # Execute the retry if handler is available
            if self._retry_handler:
                try:
                    success = await self._retry_handler(record)
                except Exception as e:
                    logger.error(
                        f"[DeliveryRetryQueue] Retry handler failed for "
                        f"{delivery_id[:8]}: {e}"
                    )
                    success = False

                if success:
                    # Retry succeeded - ledger should be updated by handler
                    self._stats["retries_succeeded"] += 1
                    logger.info(
                        f"[DeliveryRetryQueue] Retry succeeded for {delivery_id[:8]}"
                    )
                else:
                    # Retry failed - update ledger and maybe re-enqueue
                    self.ledger.record_delivery_failed(
                        delivery_id,
                        "Retry attempt failed",
                        increment_retry=True,
                    )
                    self._stats["retries_failed"] += 1

                    # Check if can retry again
                    updated = self.ledger.get_delivery(delivery_id)
                    if updated and updated.can_retry:
                        self.enqueue_retry(updated)
                    elif updated:
                        self._stats["retries_exhausted"] += 1
                        logger.warning(
                            f"[DeliveryRetryQueue] Max retries exhausted for "
                            f"{delivery_id[:8]}"
                        )
            else:
                # No handler - just log and re-enqueue for external processing
                logger.debug(
                    f"[DeliveryRetryQueue] No retry handler configured, "
                    f"delivery {delivery_id[:8]} remains in failed state"
                )
                # Don't increment retry count since we didn't actually retry
                updated = self.ledger.get_delivery(delivery_id)
                if updated and updated.can_retry:
                    self.enqueue_retry(updated)

        finally:
            self._active_retries.discard(delivery_id)

    def get_queue_size(self) -> int:
        """Get number of deliveries pending retry.

        Returns:
            Number of deliveries in the retry queue
        """
        return len(self._queue)

    def get_active_retries(self) -> int:
        """Get number of retries currently in progress.

        Returns:
            Number of active retry attempts
        """
        return len(self._active_retries)

    def get_stats(self) -> dict[str, Any]:
        """Get retry queue statistics.

        Returns:
            Dictionary with retry statistics
        """
        return {
            "queue_size": len(self._queue),
            "active_retries": len(self._active_retries),
            "running": self._running,
            **self._stats,
        }

    def get_next_retry_time(self) -> float | None:
        """Get the time of the next scheduled retry.

        Returns:
            Unix timestamp of next retry, or None if queue is empty
        """
        if self._queue:
            return self._queue[0].retry_at
        return None


def get_delivery_retry_queue() -> DeliveryRetryQueue:
    """Get the singleton DeliveryRetryQueue instance.

    Returns:
        The global DeliveryRetryQueue instance
    """
    global _retry_queue_instance
    if _retry_queue_instance is None:
        _retry_queue_instance = DeliveryRetryQueue()
    return _retry_queue_instance


def reset_delivery_retry_queue() -> None:
    """Reset the singleton DeliveryRetryQueue instance.

    Used primarily for testing.
    """
    global _retry_queue_instance
    if _retry_queue_instance is not None:
        # Try to stop if running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_retry_queue_instance.stop())
        except RuntimeError:
            pass
    _retry_queue_instance = None
