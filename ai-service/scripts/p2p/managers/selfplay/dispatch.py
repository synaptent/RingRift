"""Dispatch tracking for SelfplayScheduler.

Extracted from selfplay_scheduler.py for modularity (Phase 2 decomposition).
Contains spawn verification, success/failure tracking, and dispatch metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DispatchTrackingMixin:
    """Mixin for job spawn verification and dispatch tracking.

    Provides:
    - register_pending_spawn(): Register a job for spawn verification
    - verify_pending_spawns(): Verify pending spawns and emit events
    - get_spawn_success_rate(): Get per-node spawn success rate
    - get_pending_spawn_count(): Get count of pending verifications

    Assumes the host class has attributes from SelfplayScheduler.__init__():
    - _pending_spawn_verification, _spawn_verification_lock
    - _spawn_success_count, _spawn_failure_count
    - _spawn_verification_timeout, _get_job_status_fn
    """

    def register_pending_spawn(
        self,
        job_id: str,
        node_id: str,
        config_key: str,
    ) -> None:
        """Register a job for spawn verification.

        January 2026 - Sprint 6: Added for job spawn verification loop.
        Call this when dispatching a job to track it for verification.

        Args:
            job_id: The job ID being spawned
            node_id: The node where the job is being spawned
            config_key: Board configuration key (e.g., "hex8_2p")
        """
        with self._spawn_verification_lock:
            self._pending_spawn_verification[job_id] = (
                node_id,
                config_key,
                time.time(),
            )
            logger.debug(f"Registered job {job_id} for spawn verification on {node_id}")

    async def verify_pending_spawns(self) -> dict[str, int]:
        """Verify pending job spawns and emit events.

        January 2026 - Sprint 6: Job spawn verification loop.
        Checks all pending spawns and verifies they are actually running.
        Emits JOB_SPAWN_VERIFIED or JOB_SPAWN_FAILED events.

        Returns:
            Dict with 'verified', 'failed', and 'pending' counts.
        """
        if self._get_job_status_fn is None:
            logger.debug("No job status callback set, skipping spawn verification")
            return {"verified": 0, "failed": 0, "pending": 0}

        now = time.time()
        to_verify: list[tuple[str, str, str, float]] = []
        verified_count = 0
        failed_count = 0

        # Collect jobs to verify
        with self._spawn_verification_lock:
            for job_id, (node_id, config_key, spawn_time) in list(
                self._pending_spawn_verification.items()
            ):
                elapsed = now - spawn_time
                to_verify.append((job_id, node_id, config_key, spawn_time))

        # Verify each job outside the lock
        for job_id, node_id, config_key, spawn_time in to_verify:
            elapsed = now - spawn_time
            try:
                job_status = self._get_job_status_fn(job_id)
                if job_status is not None:
                    status = job_status.get("status", "")
                    if status in ("running", "claimed", "started"):
                        # Job verified as running
                        verified_count += 1
                        verification_time = now - spawn_time
                        await self._emit_spawn_verified(
                            job_id, node_id, config_key, verification_time
                        )
                        # Update success count for node
                        self._spawn_success_count[node_id] = (
                            self._spawn_success_count.get(node_id, 0) + 1
                        )
                        # Remove from pending
                        with self._spawn_verification_lock:
                            self._pending_spawn_verification.pop(job_id, None)
                        logger.debug(
                            f"Job {job_id} verified running on {node_id} "
                            f"(took {verification_time:.1f}s)"
                        )
                        continue

                # Check timeout
                if elapsed >= self._spawn_verification_timeout:
                    # Spawn verification timed out
                    failed_count += 1
                    await self._emit_spawn_failed(
                        job_id, node_id, config_key,
                        self._spawn_verification_timeout,
                        reason="verification_timeout" if job_status is None else "status_not_running"
                    )
                    # Update failure count for node
                    self._spawn_failure_count[node_id] = (
                        self._spawn_failure_count.get(node_id, 0) + 1
                    )
                    # Remove from pending
                    with self._spawn_verification_lock:
                        self._pending_spawn_verification.pop(job_id, None)
                    logger.warning(
                        f"Job {job_id} spawn verification failed on {node_id} "
                        f"(timeout={self._spawn_verification_timeout}s)"
                    )
            except Exception as e:
                logger.debug(f"Error verifying job {job_id}: {e}")
                # On error, still respect timeout
                if elapsed >= self._spawn_verification_timeout:
                    failed_count += 1
                    with self._spawn_verification_lock:
                        self._pending_spawn_verification.pop(job_id, None)

        # Jan 2, 2026 - Sprint 9: Return dict with pending count for SpawnVerificationLoop
        with self._spawn_verification_lock:
            pending_count = len(self._pending_spawn_verification)
        return {"verified": verified_count, "failed": failed_count, "pending": pending_count}

    async def _emit_spawn_verified(
        self,
        job_id: str,
        node_id: str,
        config_key: str,
        verification_time: float,
    ) -> None:
        """Emit JOB_SPAWN_VERIFIED event."""
        try:
            from app.distributed.data_events import emit_job_spawn_verified
            await emit_job_spawn_verified(
                job_id=job_id,
                node_id=node_id,
                config_key=config_key,
                verification_time_seconds=verification_time,
                source="selfplay_scheduler",
            )
        except ImportError:
            logger.debug("data_events not available, skipping spawn verified event")
        except Exception as e:
            logger.debug(f"Failed to emit spawn verified event: {e}")

    async def _emit_spawn_failed(
        self,
        job_id: str,
        node_id: str,
        config_key: str,
        timeout: float,
        reason: str,
    ) -> None:
        """Emit JOB_SPAWN_FAILED event."""
        try:
            from app.distributed.data_events import emit_job_spawn_failed
            await emit_job_spawn_failed(
                job_id=job_id,
                node_id=node_id,
                config_key=config_key,
                timeout_seconds=timeout,
                reason=reason,
                source="selfplay_scheduler",
            )
        except ImportError:
            logger.debug("data_events not available, skipping spawn failed event")
        except Exception as e:
            logger.debug(f"Failed to emit spawn failed event: {e}")

    def get_spawn_success_rate(self, node_id: str) -> float:
        """Get the spawn success rate for a node.

        January 2026 - Sprint 6: Added for capacity estimation.
        Used to adjust job targets based on historical spawn success.

        Args:
            node_id: Node to get success rate for

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no data is available.
        """
        success = self._spawn_success_count.get(node_id, 0)
        failure = self._spawn_failure_count.get(node_id, 0)
        total = success + failure
        if total == 0:
            return 1.0  # Assume success if no data
        return success / total

    def get_pending_spawn_count(self) -> int:
        """Get the number of jobs pending spawn verification."""
        with self._spawn_verification_lock:
            return len(self._pending_spawn_verification)
