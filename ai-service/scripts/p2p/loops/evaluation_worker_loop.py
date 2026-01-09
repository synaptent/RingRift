"""Evaluation Worker Loop for Distributed Model Evaluation.

Jan 9, 2026: Phase 3 of cluster-wide model evaluation system.

This loop runs on GPU worker nodes to:
1. Claim evaluation jobs from the EvaluationScheduler via /work/claim_evaluation
2. Sync models to local disk if not already present
3. Execute MultiHarnessGauntlet.evaluate_model() for each claimed job
4. Report results back to the scheduler

The loop integrates with the existing P2P work distribution system and
supports distributed execution across the cluster.

Usage:
    from scripts.p2p.loops.evaluation_worker_loop import EvaluationWorkerLoop

    loop = EvaluationWorkerLoop(
        node_id="worker-1",
        claim_evaluation_callback=claim_callback,
        report_result_callback=report_callback,
    )
    await loop.start_background()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from scripts.p2p.loops.base import BaseLoop, BackoffConfig

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_INTERVAL = 30.0  # Check for work every 30 seconds
DEFAULT_CLAIM_TIMEOUT = 30.0  # Timeout for claim request
DEFAULT_EVALUATION_TIMEOUT = 3600.0  # 1 hour timeout for evaluation
DEFAULT_MODEL_SYNC_TIMEOUT = 300.0  # 5 minutes for model sync


@dataclass
class EvaluationWorkerConfig:
    """Configuration for the evaluation worker loop."""

    interval: float = DEFAULT_INTERVAL
    claim_timeout: float = DEFAULT_CLAIM_TIMEOUT
    evaluation_timeout: float = DEFAULT_EVALUATION_TIMEOUT
    model_sync_timeout: float = DEFAULT_MODEL_SYNC_TIMEOUT
    capabilities: list[str] = field(default_factory=lambda: ["gpu", "nn", "nnue"])
    models_dir: str = "models"
    max_consecutive_failures: int = 5  # Stop after this many consecutive failures
    skip_model_sync: bool = False  # For testing


class EvaluationWorkerLoop(BaseLoop):
    """Worker loop that claims and executes evaluation jobs.

    This loop runs on GPU worker nodes and:
    1. Claims evaluation jobs from the scheduler
    2. Ensures models are available locally (syncs if needed)
    3. Runs MultiHarnessGauntlet evaluation
    4. Reports results back to the scheduler

    The loop is designed to work with the distributed P2P system and
    can operate even when the cluster is in a degraded state.
    """

    def __init__(
        self,
        node_id: str,
        *,
        config: EvaluationWorkerConfig | None = None,
        claim_evaluation_callback: Callable[
            [str, list[str] | None], Coroutine[Any, Any, dict[str, Any] | None]
        ] | None = None,
        report_result_callback: Callable[
            [str, dict[str, Any]], Coroutine[Any, Any, bool]
        ] | None = None,
        report_failure_callback: Callable[
            [str, str], Coroutine[Any, Any, bool]
        ] | None = None,
        sync_model_callback: Callable[
            [str, str, str], Coroutine[Any, Any, Path | None]
        ] | None = None,
        enabled: bool = True,
    ):
        """Initialize the evaluation worker loop.

        Args:
            node_id: ID of this worker node
            config: Worker configuration
            claim_evaluation_callback: Async callback to claim work via HTTP
                (node_id, capabilities) -> job_dict or None
            report_result_callback: Async callback to report results
                (job_id, results) -> success
            report_failure_callback: Async callback to report failure
                (job_id, error_message) -> success
            sync_model_callback: Async callback to sync a model to local disk
                (model_path, source_node, local_dir) -> local_path or None
            enabled: Whether the loop is enabled
        """
        self.config = config or EvaluationWorkerConfig()

        super().__init__(
            name="evaluation_worker",
            interval=self.config.interval,
            backoff_config=BackoffConfig(
                initial_delay=10.0,
                max_delay=120.0,
                multiplier=1.5,
            ),
            enabled=enabled,
        )

        self.node_id = node_id

        # Callbacks for distributed operations
        self._claim_evaluation = claim_evaluation_callback
        self._report_result = report_result_callback
        self._report_failure = report_failure_callback
        self._sync_model = sync_model_callback

        # Statistics
        self._jobs_claimed = 0
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._consecutive_failures = 0
        self._last_claim_time: float = 0
        self._last_completion_time: float = 0
        self._total_evaluation_time: float = 0

        # Current job tracking
        self._current_job: dict[str, Any] | None = None
        self._current_job_start: float = 0

    async def _run_once(self) -> None:
        """Main loop iteration: claim and execute one evaluation job."""
        # Skip if we have too many consecutive failures
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            logger.warning(
                f"[EvalWorker] Too many consecutive failures "
                f"({self._consecutive_failures}), pausing..."
            )
            # Reset after waiting
            await asyncio.sleep(60.0)
            self._consecutive_failures = 0
            return

        # Try to claim an evaluation job
        job = await self._claim_job()
        if job is None:
            # No work available
            return

        # Process the job
        try:
            await self._process_job(job)
            self._jobs_completed += 1
            self._consecutive_failures = 0
            self._last_completion_time = time.time()
        except Exception as e:
            self._jobs_failed += 1
            self._consecutive_failures += 1
            logger.error(f"[EvalWorker] Job {job.get('job_id')} failed: {e}")

            # Report failure
            await self._report_job_failure(
                job.get("job_id", "unknown"),
                str(e),
            )

    async def _claim_job(self) -> dict[str, Any] | None:
        """Claim an evaluation job from the scheduler.

        Returns:
            Job dictionary or None if no work available
        """
        if not self._claim_evaluation:
            # Try direct HTTP claim if no callback
            return await self._claim_via_http()

        try:
            result = await asyncio.wait_for(
                self._claim_evaluation(self.node_id, self.config.capabilities),
                timeout=self.config.claim_timeout,
            )

            if result and result.get("status") == "claimed":
                job = result.get("job")
                if job:
                    self._jobs_claimed += 1
                    self._last_claim_time = time.time()
                    self._current_job = job
                    logger.info(
                        f"[EvalWorker] Claimed job {job.get('job_id')} "
                        f"for model {job.get('model_id')}"
                    )
                    return job

            return None

        except asyncio.TimeoutError:
            logger.debug("[EvalWorker] Claim request timed out")
            return None
        except Exception as e:
            logger.debug(f"[EvalWorker] Claim error: {e}")
            return None

    async def _claim_via_http(self) -> dict[str, Any] | None:
        """Claim work directly via HTTP when no callback is configured.

        This is the fallback when the loop is used standalone.
        """
        try:
            import aiohttp

            # Build URL to leader or localhost
            base_url = "http://localhost:8770"
            capabilities = ",".join(self.config.capabilities)
            url = f"{base_url}/work/claim_evaluation?node_id={self.node_id}&capabilities={capabilities}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.claim_timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "claimed":
                            job = data.get("job")
                            if job:
                                self._jobs_claimed += 1
                                self._last_claim_time = time.time()
                                self._current_job = job
                                logger.info(
                                    f"[EvalWorker] Claimed job {job.get('job_id')} via HTTP"
                                )
                                return job
                    return None

        except Exception as e:
            logger.debug(f"[EvalWorker] HTTP claim failed: {e}")
            return None

    async def _process_job(self, job: dict[str, Any]) -> None:
        """Process an evaluation job.

        Args:
            job: Job dictionary from scheduler
        """
        job_id = job.get("job_id", "unknown")
        model_path = job.get("model_path", "")
        model_type = job.get("model_type", "nn")
        board_type = job.get("board_type", "")
        num_players = job.get("num_players", 2)
        harnesses = job.get("harnesses", [])
        games_per_harness = job.get("games_per_harness", 50)
        source_node = job.get("node_id", "")

        logger.info(
            f"[EvalWorker] Processing job {job_id}: "
            f"{model_path} ({model_type}) on {board_type}_{num_players}p "
            f"with {len(harnesses)} harnesses"
        )

        self._current_job_start = time.time()

        # Step 1: Ensure model is available locally
        local_model_path = await self._ensure_model_available(
            model_path, source_node
        )
        if local_model_path is None:
            raise RuntimeError(f"Failed to sync model: {model_path}")

        # Step 2: Run evaluation
        results = await self._run_evaluation(
            local_model_path,
            model_type,
            board_type,
            num_players,
            harnesses,
            games_per_harness,
        )

        # Step 3: Report results
        evaluation_time = time.time() - self._current_job_start
        self._total_evaluation_time += evaluation_time

        results["evaluation_time_seconds"] = evaluation_time
        results["evaluated_by"] = self.node_id

        await self._report_job_result(job_id, results)

        logger.info(
            f"[EvalWorker] Completed job {job_id} in {evaluation_time:.1f}s"
        )

        self._current_job = None

    async def _ensure_model_available(
        self, model_path: str, source_node: str
    ) -> Path | None:
        """Ensure the model is available locally.

        Args:
            model_path: Path to the model on source node
            source_node: Node ID where model is located

        Returns:
            Local path to the model or None if sync failed
        """
        # Check if model is already local
        local_path = Path(self.config.models_dir) / Path(model_path).name
        if local_path.exists():
            logger.debug(f"[EvalWorker] Model already local: {local_path}")
            return local_path

        # Also check the original path (might be absolute)
        if Path(model_path).exists():
            return Path(model_path)

        # Skip sync if configured
        if self.config.skip_model_sync:
            logger.warning(f"[EvalWorker] Model not found and sync disabled: {model_path}")
            return None

        # Sync from source node
        if self._sync_model:
            try:
                synced_path = await asyncio.wait_for(
                    self._sync_model(model_path, source_node, self.config.models_dir),
                    timeout=self.config.model_sync_timeout,
                )
                if synced_path:
                    logger.info(f"[EvalWorker] Synced model to: {synced_path}")
                    return synced_path
            except asyncio.TimeoutError:
                logger.error(f"[EvalWorker] Model sync timed out: {model_path}")
            except Exception as e:
                logger.error(f"[EvalWorker] Model sync failed: {e}")

        return None

    async def _run_evaluation(
        self,
        model_path: Path,
        model_type: str,
        board_type: str,
        num_players: int,
        harnesses: list[str],
        games_per_harness: int,
    ) -> dict[str, Any]:
        """Run the actual evaluation using MultiHarnessGauntlet.

        Args:
            model_path: Local path to the model
            model_type: "nn" or "nnue"
            board_type: Board type for evaluation
            num_players: Number of players
            harnesses: List of harness type names
            games_per_harness: Games per harness

        Returns:
            Results dictionary with harness ratings and statistics
        """
        try:
            # Import evaluation infrastructure
            from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

            gauntlet = MultiHarnessGauntlet()

            # Run evaluation (async-wrapped since it may be blocking)
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    gauntlet.evaluate_model_sync,
                    model_path=str(model_path),
                    model_type=model_type,
                    board_type=board_type,
                    num_players=num_players,
                    harness_names=harnesses,
                    games_per_harness=games_per_harness,
                ),
                timeout=self.config.evaluation_timeout,
            )

            return results.to_dict() if hasattr(results, "to_dict") else results

        except asyncio.TimeoutError:
            logger.error(f"[EvalWorker] Evaluation timed out after {self.config.evaluation_timeout}s")
            raise RuntimeError("Evaluation timed out")
        except ImportError as e:
            logger.error(f"[EvalWorker] Missing evaluation infrastructure: {e}")
            raise RuntimeError(f"Missing evaluation module: {e}")
        except Exception as e:
            logger.error(f"[EvalWorker] Evaluation error: {e}")
            raise

    async def _report_job_result(self, job_id: str, results: dict[str, Any]) -> None:
        """Report job completion to the scheduler.

        Args:
            job_id: Job identifier
            results: Evaluation results
        """
        if self._report_result:
            try:
                success = await self._report_result(job_id, results)
                if success:
                    logger.info(f"[EvalWorker] Reported results for {job_id}")
                else:
                    logger.warning(f"[EvalWorker] Failed to report results for {job_id}")
            except Exception as e:
                logger.error(f"[EvalWorker] Error reporting results: {e}")
        else:
            # Try to report directly via HTTP
            await self._report_via_http(job_id, results, is_failure=False)

    async def _report_job_failure(self, job_id: str, error: str) -> None:
        """Report job failure to the scheduler.

        Args:
            job_id: Job identifier
            error: Error message
        """
        if self._report_failure:
            try:
                await self._report_failure(job_id, error)
                logger.info(f"[EvalWorker] Reported failure for {job_id}")
            except Exception as e:
                logger.error(f"[EvalWorker] Error reporting failure: {e}")
        else:
            # Try to report directly via HTTP
            await self._report_via_http(job_id, {"error": error}, is_failure=True)

    async def _report_via_http(
        self, job_id: str, data: dict[str, Any], is_failure: bool
    ) -> None:
        """Report job result/failure via HTTP.

        Args:
            job_id: Job identifier
            data: Results or error data
            is_failure: True if reporting a failure
        """
        try:
            import aiohttp

            base_url = "http://localhost:8770"
            endpoint = "/work/fail" if is_failure else "/work/complete"
            url = f"{base_url}{endpoint}"

            payload = {
                "work_id": job_id,
                **({"error": data.get("error", "Unknown error")} if is_failure else {"result": data}),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[EvalWorker] HTTP report failed: {resp.status}"
                        )

        except Exception as e:
            logger.debug(f"[EvalWorker] HTTP report error: {e}")

    def health_check(self) -> dict[str, Any]:
        """Return health check result for DaemonManager integration.

        Returns:
            Health check result dictionary
        """
        try:
            from app.coordination.health_check_protocol import (
                HealthCheckResult,
                HealthStatus,
            )

            if not self._running:
                return HealthCheckResult(
                    status=HealthStatus.STOPPED,
                    message="EvaluationWorkerLoop is stopped",
                    details={"enabled": self.enabled},
                ).to_dict()

            # Calculate success rate
            total_jobs = self._jobs_completed + self._jobs_failed
            success_rate = (
                (self._jobs_completed / total_jobs * 100) if total_jobs > 0 else 100.0
            )

            # Determine status based on consecutive failures and success rate
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                status = HealthStatus.ERROR
                message = f"Too many consecutive failures ({self._consecutive_failures})"
            elif success_rate < 50:
                status = HealthStatus.DEGRADED
                message = f"Low success rate: {success_rate:.1f}%"
            else:
                status = HealthStatus.RUNNING
                message = "EvaluationWorkerLoop healthy"

            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "node_id": self.node_id,
                    "enabled": self.enabled,
                    "jobs_claimed": self._jobs_claimed,
                    "jobs_completed": self._jobs_completed,
                    "jobs_failed": self._jobs_failed,
                    "consecutive_failures": self._consecutive_failures,
                    "success_rate": success_rate,
                    "avg_evaluation_time": (
                        self._total_evaluation_time / self._jobs_completed
                        if self._jobs_completed > 0
                        else 0
                    ),
                    "current_job": self._current_job.get("job_id") if self._current_job else None,
                    "stats": self._stats.to_dict(),
                },
            ).to_dict()

        except ImportError:
            # Fallback if protocol not available
            return {
                "status": "running" if self._running else "stopped",
                "jobs_claimed": self._jobs_claimed,
                "jobs_completed": self._jobs_completed,
                "jobs_failed": self._jobs_failed,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "node_id": self.node_id,
            "enabled": self.enabled,
            "running": self._running,
            "jobs_claimed": self._jobs_claimed,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "consecutive_failures": self._consecutive_failures,
            "last_claim_time": self._last_claim_time,
            "last_completion_time": self._last_completion_time,
            "total_evaluation_time": self._total_evaluation_time,
            "avg_evaluation_time": (
                self._total_evaluation_time / self._jobs_completed
                if self._jobs_completed > 0
                else 0
            ),
            "current_job": self._current_job.get("job_id") if self._current_job else None,
            "loop_stats": self._stats.to_dict(),
        }


# Factory function for easy instantiation
def create_evaluation_worker_loop(
    node_id: str,
    config: EvaluationWorkerConfig | None = None,
    **kwargs,
) -> EvaluationWorkerLoop:
    """Create an evaluation worker loop instance.

    Args:
        node_id: ID of this worker node
        config: Worker configuration
        **kwargs: Additional arguments for EvaluationWorkerLoop

    Returns:
        Configured EvaluationWorkerLoop instance
    """
    return EvaluationWorkerLoop(
        node_id=node_id,
        config=config,
        **kwargs,
    )
