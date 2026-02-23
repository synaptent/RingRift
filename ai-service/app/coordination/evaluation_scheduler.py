"""Evaluation Scheduler for Cluster-Wide Model Evaluation.

This module schedules evaluation jobs for models discovered by the
ClusterModelInventoryManager. It creates prioritized jobs that evaluate
models under various harnesses, producing fresh Elo rankings.

Key Features:
- Priority-based scheduling (4-player configs get 2x boost, stale Elo 1.5x, no Elo 3x)
- Harness-aware job creation (each model evaluated under all compatible harnesses)
- Job status tracking (pending, running, completed, failed)
- Integration with curriculum weights from SelfplayScheduler

Usage:
    from app.coordination.evaluation_scheduler import (
        get_evaluation_scheduler,
        EvaluationJob,
    )

    scheduler = get_evaluation_scheduler()

    # Schedule all pending evaluations
    jobs = await scheduler.schedule_all_evaluations()

    # Get next job for a worker
    job = scheduler.get_next_job(node_capabilities=["gpu", "nn"])

    # Mark job complete
    scheduler.complete_job(job.job_id, results={...})
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.ai.harness.base_harness import HarnessType

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_GAMES_PER_HARNESS = 50  # Games per harness evaluation
DEFAULT_GAUNTLET_GAMES = 50  # Games against baseline opponents (raised from 30, Feb 22)
DEFAULT_TOURNAMENT_GAMES = 10  # Games per matchup in round-robin


class EvaluationJobStatus(str, Enum):
    """Status of an evaluation job."""
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationType(str, Enum):
    """Type of evaluation to perform."""
    GAUNTLET = "gauntlet"  # Model vs baseline opponents under all harnesses
    TOURNAMENT = "tournament"  # Round-robin between top models
    SINGLE_HARNESS = "single_harness"  # Evaluation under a specific harness


@dataclass
class EvaluationJob:
    """An evaluation job for a model.

    Attributes:
        job_id: Unique job identifier
        model_id: ID of the model to evaluate
        model_path: Path to the model file
        model_type: "nn" or "nnue"
        board_type: Board type for evaluation
        num_players: Number of players
        harnesses: List of harness types to evaluate under
        priority: Job priority (higher = more important)
        status: Current job status
        evaluation_type: Type of evaluation
        games_per_harness: Number of games per harness
        node_id: Node where model is located (for sync)
        claimed_by: Node that claimed this job
        claimed_at: Timestamp when job was claimed
        started_at: Timestamp when evaluation started
        completed_at: Timestamp when evaluation completed
        results: Evaluation results (Elo per harness, etc.)
        error: Error message if failed
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts
    """

    job_id: str
    model_id: str
    model_path: str
    model_type: str
    board_type: str
    num_players: int
    harnesses: list[HarnessType] = field(default_factory=list)
    priority: float = 0.0
    status: EvaluationJobStatus = EvaluationJobStatus.PENDING
    evaluation_type: EvaluationType = EvaluationType.GAUNTLET
    games_per_harness: int = DEFAULT_GAMES_PER_HARNESS
    node_id: str = ""
    claimed_by: str | None = None
    claimed_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def get_config_key(self) -> str:
        """Get config key for this job."""
        return f"{self.board_type}_{self.num_players}p"

    def get_composite_id(self, harness: HarnessType) -> str:
        """Generate composite participant ID for a harness.

        Format: {model_id}:{harness}:{config_hash}
        """
        from app.training.composite_participant import normalize_nn_id
        nn_id = normalize_nn_id(self.model_id) or self.model_id
        config_str = f"{self.board_type}_{self.num_players}p"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        return f"{nn_id}:{harness.value}:{config_hash}"

    def is_claimable(self) -> bool:
        """Check if job can be claimed."""
        return self.status == EvaluationJobStatus.PENDING

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return (
            self.status == EvaluationJobStatus.FAILED
            and self.retry_count < self.max_retries
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "harnesses": [h.value for h in self.harnesses],
            "priority": self.priority,
            "status": self.status.value,
            "evaluation_type": self.evaluation_type.value,
            "games_per_harness": self.games_per_harness,
            "node_id": self.node_id,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": self.results,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class SchedulerConfig:
    """Configuration for the evaluation scheduler.

    Attributes:
        games_per_harness: Default games per harness evaluation
        priority_no_elo: Priority multiplier for models without Elo
        priority_stale_elo: Priority multiplier for stale Elo
        priority_4p: Priority multiplier for 4-player configs
        priority_3p: Priority multiplier for 3-player configs
        job_timeout: Timeout for claimed jobs (seconds)
        max_concurrent_jobs: Maximum jobs running simultaneously
    """

    games_per_harness: int = DEFAULT_GAMES_PER_HARNESS
    priority_no_elo: float = 3.0
    priority_stale_elo: float = 1.5
    priority_4p: float = 2.0
    priority_3p: float = 1.5
    job_timeout: float = 3600.0  # 1 hour
    max_concurrent_jobs: int = 50


class EvaluationScheduler:
    """Schedules and manages evaluation jobs across the cluster.

    This class:
    1. Creates evaluation jobs from the model inventory
    2. Prioritizes jobs based on curriculum weights and Elo status
    3. Tracks job status (pending, running, completed)
    4. Provides jobs to workers via claim mechanism
    """

    def __init__(self, config: SchedulerConfig | None = None):
        """Initialize the scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config or SchedulerConfig()

        # Job storage
        self._jobs: dict[str, EvaluationJob] = {}
        self._job_queue: list[str] = []  # Job IDs in priority order

        # Tracking
        self._last_schedule_time: float = 0
        self._jobs_created: int = 0
        self._jobs_completed: int = 0
        self._jobs_failed: int = 0

    async def schedule_all_evaluations(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        force_refresh: bool = False,
    ) -> list[EvaluationJob]:
        """Schedule evaluation jobs for all models needing evaluation.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by number of players
            force_refresh: Force inventory rebuild

        Returns:
            List of created evaluation jobs
        """
        from app.coordination.cluster_model_inventory import (
            get_cluster_model_inventory,
        )

        logger.info("Scheduling evaluation jobs...")

        # Build inventory
        inventory = get_cluster_model_inventory()
        await inventory.build_full_inventory(
            include_remote=True,
            force_refresh=force_refresh,
        )

        # Get models needing evaluation
        models = inventory.get_models_needing_evaluation(
            board_type=board_type,
            num_players=num_players,
        )

        logger.info(f"Found {len(models)} models needing evaluation")

        # Create jobs for each model
        new_jobs = []
        for model in models:
            # Skip if job already exists for this model
            existing_job = self._find_job_for_model(model.model_id, model.node_id)
            if existing_job and existing_job.status in (
                EvaluationJobStatus.PENDING,
                EvaluationJobStatus.CLAIMED,
                EvaluationJobStatus.RUNNING,
            ):
                continue

            # Create job
            job = self._create_job_for_model(model)
            if job:
                self._jobs[job.job_id] = job
                new_jobs.append(job)

        # Sort queue by priority
        self._rebuild_queue()

        self._last_schedule_time = time.time()
        self._jobs_created += len(new_jobs)

        logger.info(f"Created {len(new_jobs)} evaluation jobs")
        return new_jobs

    def _create_job_for_model(self, model) -> EvaluationJob | None:
        """Create an evaluation job for a model entry.

        Args:
            model: ClusterModelEntry from inventory

        Returns:
            EvaluationJob or None if no harnesses available
        """
        # Skip if no compatible harnesses
        if not model.compatible_harnesses:
            logger.debug(f"Skipping {model.model_id}: no compatible harnesses")
            return None

        # Calculate priority
        priority = self._calculate_priority(model)

        # Create job
        job = EvaluationJob(
            job_id=f"eval-{uuid.uuid4().hex[:12]}",
            model_id=model.model_id,
            model_path=model.path,
            model_type=model.model_type,
            board_type=model.board_type,
            num_players=model.num_players,
            harnesses=model.compatible_harnesses.copy(),
            priority=priority,
            status=EvaluationJobStatus.PENDING,
            evaluation_type=EvaluationType.GAUNTLET,
            games_per_harness=self.config.games_per_harness,
            node_id=model.node_id,
        )

        return job

    def _calculate_priority(self, model) -> float:
        """Calculate priority score for a model.

        Args:
            model: ClusterModelEntry

        Returns:
            Priority score (higher = more important)
        """
        priority = model.get_priority_score()

        # Apply config multipliers
        if not model.has_elo:
            priority *= self.config.priority_no_elo
        elif model.elo_stale:
            priority *= self.config.priority_stale_elo

        if model.num_players == 4:
            priority *= self.config.priority_4p
        elif model.num_players == 3:
            priority *= self.config.priority_3p

        # Try to get curriculum weight boost
        try:
            from app.config.coordination_defaults import (
                get_curriculum_weight,
            )
            curriculum_weight = get_curriculum_weight(
                model.board_type, model.num_players
            )
            priority *= (1.0 + curriculum_weight)
        except ImportError:
            pass

        return priority

    def _find_job_for_model(
        self,
        model_id: str,
        node_id: str,
    ) -> EvaluationJob | None:
        """Find existing job for a model."""
        for job in self._jobs.values():
            if job.model_id == model_id and job.node_id == node_id:
                return job
        return None

    def _rebuild_queue(self) -> None:
        """Rebuild the priority queue."""
        # Get pending and retryable jobs
        candidates = [
            job for job in self._jobs.values()
            if job.is_claimable() or job.can_retry()
        ]

        # Sort by priority (highest first)
        candidates.sort(key=lambda j: j.priority, reverse=True)

        # Reset retryable jobs to pending
        for job in candidates:
            if job.can_retry():
                job.status = EvaluationJobStatus.PENDING
                job.retry_count += 1
                job.error = None

        self._job_queue = [job.job_id for job in candidates]

    def get_next_job(
        self,
        node_id: str | None = None,
        capabilities: list[str] | None = None,
    ) -> EvaluationJob | None:
        """Get the next available job for a worker.

        Args:
            node_id: ID of the claiming node
            capabilities: List of node capabilities (e.g., ["gpu", "nn", "nnue"])

        Returns:
            Next claimable job or None
        """
        # Clean up stale claims first
        self._cleanup_stale_claims()

        for job_id in self._job_queue:
            job = self._jobs.get(job_id)
            if not job:
                continue

            if not job.is_claimable():
                continue

            # Check capabilities
            if capabilities:
                # Check if node can handle model type
                if job.model_type == "nnue" and "nnue" not in capabilities:
                    continue
                if job.model_type == "nn" and "nn" not in capabilities:
                    continue

            return job

        return None

    def claim_job(
        self,
        job_id: str,
        node_id: str,
    ) -> bool:
        """Claim a job for execution.

        Args:
            job_id: Job to claim
            node_id: Node claiming the job

        Returns:
            True if claim successful
        """
        job = self._jobs.get(job_id)
        if not job or not job.is_claimable():
            return False

        job.status = EvaluationJobStatus.CLAIMED
        job.claimed_by = node_id
        job.claimed_at = time.time()

        logger.info(f"Job {job_id} claimed by {node_id}")
        return True

    def start_job(self, job_id: str) -> bool:
        """Mark a job as started.

        Args:
            job_id: Job to start

        Returns:
            True if state change successful
        """
        job = self._jobs.get(job_id)
        if not job or job.status != EvaluationJobStatus.CLAIMED:
            return False

        job.status = EvaluationJobStatus.RUNNING
        job.started_at = time.time()

        logger.info(f"Job {job_id} started")
        return True

    def complete_job(
        self,
        job_id: str,
        results: dict[str, Any],
    ) -> bool:
        """Mark a job as completed with results.

        Args:
            job_id: Job to complete
            results: Evaluation results

        Returns:
            True if state change successful
        """
        job = self._jobs.get(job_id)
        if not job or job.status != EvaluationJobStatus.RUNNING:
            return False

        job.status = EvaluationJobStatus.COMPLETED
        job.completed_at = time.time()
        job.results = results

        self._jobs_completed += 1

        # Emit event for pipeline integration
        self._emit_evaluation_completed(job)

        logger.info(f"Job {job_id} completed")
        return True

    def fail_job(
        self,
        job_id: str,
        error: str,
    ) -> bool:
        """Mark a job as failed.

        Args:
            job_id: Job to fail
            error: Error message

        Returns:
            True if state change successful
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.status = EvaluationJobStatus.FAILED
        job.completed_at = time.time()
        job.error = error

        self._jobs_failed += 1

        # Rebuild queue to potentially retry
        if job.can_retry():
            self._rebuild_queue()

        logger.warning(f"Job {job_id} failed: {error}")
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job to cancel

        Returns:
            True if state change successful
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.status = EvaluationJobStatus.CANCELLED
        job.completed_at = time.time()

        logger.info(f"Job {job_id} cancelled")
        return True

    def _cleanup_stale_claims(self) -> None:
        """Reset jobs that have been claimed too long."""
        now = time.time()
        for job in self._jobs.values():
            if job.status == EvaluationJobStatus.CLAIMED:
                if job.claimed_at and (now - job.claimed_at) > self.config.job_timeout:
                    logger.warning(
                        f"Job {job.job_id} claim timed out "
                        f"(claimed by {job.claimed_by})"
                    )
                    job.status = EvaluationJobStatus.PENDING
                    job.claimed_by = None
                    job.claimed_at = None

    def _emit_evaluation_completed(self, job: EvaluationJob) -> None:
        """Emit EVALUATION_COMPLETED event for pipeline integration."""
        try:
            from app.distributed.data_events import (
                DataEventType,
                emit_data_event,
            )

            # Emit for each harness result
            harness_ratings = job.results.get("harness_ratings", {})
            for harness_name, rating_data in harness_ratings.items():
                emit_data_event(
                    DataEventType.EVALUATION_COMPLETED,
                    {
                        "model_id": job.model_id,
                        "model_path": job.model_path,
                        "board_type": job.board_type,
                        "num_players": job.num_players,
                        "config_key": job.get_config_key(),
                        "harness_type": harness_name,
                        "elo": rating_data.get("elo", 1500),
                        "win_rate": rating_data.get("win_rate", 0.5),
                        "games_played": rating_data.get("games_played", 0),
                        "composite_id": job.get_composite_id(
                            HarnessType(harness_name)
                        ),
                    },
                )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error emitting evaluation event: {e}")

    def get_job(self, job_id: str) -> EvaluationJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_pending_jobs(self) -> list[EvaluationJob]:
        """Get all pending jobs in priority order."""
        return [
            self._jobs[job_id]
            for job_id in self._job_queue
            if job_id in self._jobs
            and self._jobs[job_id].is_claimable()
        ]

    def get_running_jobs(self) -> list[EvaluationJob]:
        """Get all currently running jobs."""
        return [
            job for job in self._jobs.values()
            if job.status == EvaluationJobStatus.RUNNING
        ]

    def get_scheduler_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        by_status = {status.value: 0 for status in EvaluationJobStatus}
        for job in self._jobs.values():
            by_status[job.status.value] += 1

        return {
            "total_jobs": len(self._jobs),
            "queue_length": len(self._job_queue),
            "jobs_created": self._jobs_created,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "by_status": by_status,
            "last_schedule_time": self._last_schedule_time,
        }

    def clear(self) -> None:
        """Clear all jobs."""
        self._jobs.clear()
        self._job_queue.clear()


# Module-level singleton
_scheduler: EvaluationScheduler | None = None


def get_evaluation_scheduler() -> EvaluationScheduler:
    """Get the singleton EvaluationScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = EvaluationScheduler()
    return _scheduler


def reset_evaluation_scheduler() -> None:
    """Reset the singleton (for testing)."""
    global _scheduler
    _scheduler = None


# CLI for scheduler operations
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation Scheduler CLI")
    parser.add_argument(
        "--schedule", action="store_true", help="Schedule all pending evaluations"
    )
    parser.add_argument(
        "--pending", action="store_true", help="Show pending jobs"
    )
    parser.add_argument(
        "--running", action="store_true", help="Show running jobs"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show scheduler statistics"
    )
    parser.add_argument(
        "--board-type", type=str, help="Filter by board type"
    )
    parser.add_argument(
        "--num-players", type=int, help="Filter by number of players"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    async def main():
        scheduler = get_evaluation_scheduler()

        if args.schedule:
            jobs = await scheduler.schedule_all_evaluations(
                board_type=args.board_type,
                num_players=args.num_players,
                force_refresh=True,
            )
            print(f"\nScheduled {len(jobs)} evaluation jobs\n")

        if args.pending:
            pending = scheduler.get_pending_jobs()
            print(f"\n=== Pending Jobs ({len(pending)}) ===\n")
            print(f"{'Job ID':<20} {'Model':<30} {'Config':<12} {'Priority':<10}")
            print("-" * 75)
            for job in pending[:30]:
                print(
                    f"{job.job_id:<20} "
                    f"{job.model_id[:30]:<30} "
                    f"{job.get_config_key():<12} "
                    f"{job.priority:.1f}"
                )
            if len(pending) > 30:
                print(f"... and {len(pending) - 30} more")
            print()

        if args.running:
            running = scheduler.get_running_jobs()
            print(f"\n=== Running Jobs ({len(running)}) ===\n")
            for job in running:
                elapsed = time.time() - (job.started_at or 0)
                print(f"  {job.job_id}: {job.model_id} ({elapsed:.0f}s)")
            print()

        if args.stats:
            stats = scheduler.get_scheduler_stats()
            print("\n=== Scheduler Statistics ===\n")
            print(f"Total jobs:     {stats['total_jobs']}")
            print(f"Queue length:   {stats['queue_length']}")
            print(f"Jobs created:   {stats['jobs_created']}")
            print(f"Jobs completed: {stats['jobs_completed']}")
            print(f"Jobs failed:    {stats['jobs_failed']}")
            print("\nBy status:")
            for status, count in stats['by_status'].items():
                print(f"  {status}: {count}")
            print()

    asyncio.run(main())
