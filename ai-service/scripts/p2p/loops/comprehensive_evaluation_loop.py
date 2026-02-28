"""Comprehensive Evaluation Loop for automated model evaluation.

January 2026: Created as part of the comprehensive evaluation system.

This loop runs on the leader node every 6 hours to ensure all models across
the cluster are evaluated under all compatible harnesses. It prioritizes
unevaluated combinations first, then stale combinations (not evaluated in 7+ days).

Features:
- Enumerates ALL models across cluster via P2P endpoints
- Cross-references with EloService for evaluation status
- Prioritizes: unevaluated > stale > recently evaluated
- Dispatches evaluations to GPU nodes
- Records results with composite participant IDs
- Saves games for training

Usage:
    from scripts.p2p.loops.comprehensive_evaluation_loop import (
        ComprehensiveEvaluationLoop,
        ComprehensiveEvaluationConfig,
    )

    loop = ComprehensiveEvaluationLoop(
        get_role=lambda: NodeRole.LEADER,
        get_orchestrator=lambda: orchestrator,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .base import BaseLoop

if TYPE_CHECKING:
    from app.distributed.cluster_model_enumerator import (
        ClusterModelEnumerator,
        EvaluationCombination,
        ModelInfo,
    )
    from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveEvaluationConfig:
    """Configuration for comprehensive evaluation loop."""

    # Loop interval in seconds (default: 6 hours)
    interval: float = 6 * 3600

    # Maximum evaluations per cycle to prevent overload.
    # Feb 28, 2026: Reduced from 50 to 12 (one per canonical config).
    # With the corrected HARNESS_COMPATIBILITY matrix (1 harness per model
    # instead of 4-5), 12 is sufficient for a full cluster evaluation pass.
    max_evaluations_per_cycle: int = 12

    # Days after which an evaluation is considered stale
    stale_threshold_days: int = 7

    # Games to play per harness during evaluation
    games_per_harness: int = 50

    # Whether to save games for training
    save_games: bool = True

    # Whether to register results with EloService
    register_with_elo: bool = True

    # Timeout per evaluation in seconds (default: 30 minutes)
    evaluation_timeout: float = 1800.0

    # Whether loop is enabled
    enabled: bool = True


@dataclass
class EvaluationDispatchResult:
    """Result of dispatching an evaluation to a GPU node."""

    model_path: str
    harness: str
    config_key: str
    success: bool
    node_id: str | None = None
    error: str | None = None
    job_id: str | None = None


@dataclass
class CycleStats:
    """Statistics for a single evaluation cycle."""

    total_models: int = 0
    unevaluated_count: int = 0
    stale_count: int = 0
    queued_count: int = 0
    dispatched_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    cycle_duration: float = 0.0
    dispatch_results: list[EvaluationDispatchResult] = field(default_factory=list)


class ComprehensiveEvaluationLoop(BaseLoop):
    """Periodic comprehensive evaluation of all models under all harnesses.

    Runs on leader node every 6 hours. Ensures no (model, harness, config)
    combination goes more than 7 days without fresh evaluation.

    Evaluation Strategy:
    1. Enumerate all models across cluster via P2P /models/inventory endpoint
    2. Cross-reference with EloService to find unevaluated combinations
    3. Prioritize: unevaluated first, then stale by age
    4. Dispatch evaluations to GPU nodes via work queue
    5. Results recorded with composite IDs: {model_hash}:{harness}:{config}
    """

    def __init__(
        self,
        get_role: Callable[[], Any],
        get_orchestrator: Callable[[], Any] | None = None,
        config: ComprehensiveEvaluationConfig | None = None,
    ):
        """Initialize the comprehensive evaluation loop.

        Args:
            get_role: Callable returning current NodeRole (LEADER/FOLLOWER/VOTER)
            get_orchestrator: Optional callable to get P2P orchestrator for dispatch
            config: Loop configuration
        """
        self.config = config or ComprehensiveEvaluationConfig()

        super().__init__(
            name="comprehensive_evaluation",
            interval=self.config.interval,
            enabled=self.config.enabled,
        )

        self._get_role = get_role
        self._get_orchestrator = get_orchestrator

        # Lazy-loaded components
        self._enumerator: ClusterModelEnumerator | None = None
        self._gauntlet: MultiHarnessGauntlet | None = None

        # Statistics
        self._last_cycle_stats: CycleStats | None = None
        self._total_evaluations: int = 0
        self._total_failures: int = 0

    def _is_leader(self) -> bool:
        """Check if current node is the cluster leader."""
        try:
            role = self._get_role()
            # Support both string and enum
            role_str = role.value if hasattr(role, "value") else str(role)
            return role_str.lower() == "leader"
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get role: {e}")
            return False

    def _get_enumerator(self) -> "ClusterModelEnumerator":
        """Get or create the model enumerator."""
        if self._enumerator is None:
            from app.distributed.cluster_model_enumerator import ClusterModelEnumerator

            self._enumerator = ClusterModelEnumerator()
        return self._enumerator

    def _get_gauntlet(self) -> "MultiHarnessGauntlet":
        """Get or create the multi-harness gauntlet."""
        if self._gauntlet is None:
            from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

            self._gauntlet = MultiHarnessGauntlet(
                default_games_per_baseline=self.config.games_per_harness // 2,
            )
        return self._gauntlet

    def _is_coordinator(self) -> bool:
        """Check if this node is a coordinator (no local GPU evaluation)."""
        try:
            from app.config.env import env

            return env.is_coordinator
        except ImportError:
            return False

    async def _run_once(self) -> None:
        """Execute one iteration of the comprehensive evaluation loop."""
        # Only run on leader
        if not self._is_leader():
            logger.debug(f"[{self.name}] Not leader, skipping evaluation cycle")
            return

        # On coordinator nodes, we dispatch evaluations to GPU nodes via
        # the work queue. If dispatch fails for a specific evaluation, it's
        # handled per-item in _dispatch_evaluation() (which already has a
        # coordinator guard to skip local execution). We do NOT skip the
        # entire cycle here — even if dispatch capability is temporarily
        # unavailable, we still want to enumerate models and queue work
        # items so they'll be picked up when GPU nodes pull work.
        if self._is_coordinator() and not self._can_dispatch_to_cluster():
            logger.info(
                f"[{self.name}] Coordinator without direct dispatch - "
                "will queue evaluations for GPU nodes to pull"
            )

        logger.info(f"[{self.name}] Starting comprehensive evaluation cycle")
        cycle_start = time.time()
        stats = CycleStats()

        try:
            # 1. Enumerate all models across cluster
            enumerator = self._get_enumerator()
            all_models = await enumerator.enumerate_all_models(force_refresh=True)
            stats.total_models = len(all_models)
            logger.info(
                f"[{self.name}] Found {stats.total_models} models across cluster"
            )

            if not all_models:
                logger.info(f"[{self.name}] No models found, skipping evaluation")
                return

            # 2. Get combinations needing evaluation
            unevaluated = enumerator.get_unevaluated_combinations()
            stale = enumerator.get_stale_combinations(self.config.stale_threshold_days)

            stats.unevaluated_count = len(unevaluated)
            stats.stale_count = len(stale)

            logger.info(
                f"[{self.name}] Found {stats.unevaluated_count} unevaluated, "
                f"{stats.stale_count} stale combinations"
            )

            # 3. Prioritize: unevaluated first, then stale by age
            queue = list(unevaluated)
            # Sort stale by age (oldest first)
            stale_sorted = sorted(
                stale, key=lambda x: x.last_evaluated or 0.0
            )
            queue.extend(stale_sorted)

            # Limit to max evaluations per cycle
            queue = queue[: self.config.max_evaluations_per_cycle]
            stats.queued_count = len(queue)

            logger.info(
                f"[{self.name}] Queued {stats.queued_count} evaluations "
                f"({stats.unevaluated_count} unevaluated, "
                f"{min(stats.stale_count, len(queue) - stats.unevaluated_count)} stale)"
            )

            # 4. Dispatch evaluations
            for combo in queue:
                result = await self._dispatch_evaluation(combo)
                stats.dispatch_results.append(result)
                stats.dispatched_count += 1

                if result.success:
                    stats.success_count += 1
                    self._total_evaluations += 1
                else:
                    stats.failed_count += 1
                    self._total_failures += 1

                # Brief delay between dispatches to prevent overload
                await asyncio.sleep(1.0)

            # 5. Emit summary event
            self._emit_completion_event(stats)

        except Exception as e:
            logger.error(f"[{self.name}] Evaluation cycle failed: {e}", exc_info=True)
            raise

        finally:
            stats.cycle_duration = time.time() - cycle_start
            self._last_cycle_stats = stats

            logger.info(
                f"[{self.name}] Evaluation cycle complete: "
                f"{stats.success_count}/{stats.dispatched_count} succeeded "
                f"in {stats.cycle_duration:.1f}s"
            )

    async def _dispatch_evaluation(
        self, combo: "EvaluationCombination"
    ) -> EvaluationDispatchResult:
        """Dispatch a single evaluation to a GPU node.

        For now, runs evaluation locally via the gauntlet.
        Future: dispatch via work queue to GPU nodes.

        Args:
            combo: The evaluation combination to dispatch

        Returns:
            EvaluationDispatchResult with success/failure info
        """
        result = EvaluationDispatchResult(
            model_path=combo.model_path,
            harness=combo.harness,
            config_key=combo.config_key,
            success=False,
        )

        try:
            logger.info(
                f"[{self.name}] Dispatching evaluation: "
                f"{combo.model_path} under {combo.harness} for {combo.config_key}"
            )

            # Try to dispatch via work queue if orchestrator available
            orchestrator = self._get_orchestrator() if self._get_orchestrator else None
            if orchestrator is not None:
                success = await self._dispatch_via_work_queue(combo, orchestrator)
                if success:
                    result.success = True
                    result.job_id = f"eval_{combo.model_hash}_{combo.harness}"
                    return result

            # Coordinators must not run evaluations locally - they block the
            # event loop and ultimately fail anyway since game_gauntlet refuses
            # to run on coordinator nodes.
            try:
                from app.config.env import env

                if env.is_coordinator:
                    result.error = "Coordinator node - local evaluation skipped"
                    logger.debug(
                        f"[{self.name}] Skipping local evaluation on coordinator: "
                        f"{combo.model_path}"
                    )
                    return result
            except ImportError:
                pass

            # Fallback: run locally via gauntlet (non-coordinator GPU nodes only)
            gauntlet = self._get_gauntlet()

            # Extract board_type and num_players from config_key
            parts = combo.config_key.split("_")
            if len(parts) >= 2:
                board_type = parts[0]
                num_players = int(parts[1].rstrip("p"))
            else:
                board_type = "square8"
                num_players = 2

            # Run evaluation with timeout
            try:
                rating = await asyncio.wait_for(
                    gauntlet.evaluate_all_harnesses(
                        model_path=combo.model_path,
                        board_type=board_type,
                        num_players=num_players,
                        games_per_harness=self.config.games_per_harness,
                        save_games=self.config.save_games,
                        register_with_elo=self.config.register_with_elo,
                    ),
                    timeout=self.config.evaluation_timeout,
                )

                if rating:
                    result.success = True
                    logger.info(
                        f"[{self.name}] Evaluation complete: {combo.model_path} "
                        f"under {combo.harness} - {len(rating)} harnesses evaluated"
                    )

            except asyncio.TimeoutError:
                result.error = f"Timeout after {self.config.evaluation_timeout}s"
                logger.warning(
                    f"[{self.name}] Evaluation timed out: {combo.model_path}"
                )

        except Exception as e:
            result.error = str(e)
            logger.error(
                f"[{self.name}] Evaluation failed: {combo.model_path} - {e}"
            )

        return result

    def _can_dispatch_to_cluster(self) -> bool:
        """Check if we can dispatch evaluations to GPU nodes.

        Returns True if either the orchestrator work queue or the
        WorkDistributor is available for dispatching evaluation jobs.
        """
        # Check orchestrator work queue
        if self._get_orchestrator:
            orchestrator = self._get_orchestrator()
            if orchestrator and getattr(orchestrator, "work_queue", None) is not None:
                return True

        # Check WorkDistributor
        try:
            from app.coordination.work_distributor import get_work_distributor

            distributor = get_work_distributor()
            if distributor is not None:
                return True
        except ImportError:
            pass

        return False

    async def _dispatch_via_work_queue(
        self, combo: "EvaluationCombination", orchestrator: Any
    ) -> bool:
        """Dispatch evaluation via the work queue or WorkDistributor.

        Tries the orchestrator's work queue first, then falls back to
        WorkDistributor which can route evaluation jobs to GPU nodes.

        Args:
            combo: Evaluation combination to dispatch
            orchestrator: P2P orchestrator instance

        Returns:
            True if successfully queued, False otherwise
        """
        # Strategy 1: Try orchestrator work queue
        try:
            work_queue = getattr(orchestrator, "work_queue", None)
            if work_queue is not None and hasattr(work_queue, "submit_evaluation"):
                job_id = f"eval_{combo.model_hash}_{combo.harness}_{int(time.time())}"
                await work_queue.submit_evaluation(
                    job_id=job_id,
                    model_path=combo.model_path,
                    harness=combo.harness,
                    config_key=combo.config_key,
                    games_per_harness=self.config.games_per_harness,
                    save_games=self.config.save_games,
                )
                logger.info(f"[{self.name}] Queued evaluation job via work queue: {job_id}")
                return True
        except Exception as e:
            logger.debug(f"[{self.name}] Work queue dispatch failed: {e}")

        # Strategy 2: Try WorkDistributor (dispatches to GPU nodes)
        try:
            from app.coordination.work_distributor import get_work_distributor

            distributor = get_work_distributor()
            if distributor is not None:
                # Extract board_type and num_players from config_key
                parts = combo.config_key.split("_")
                if len(parts) >= 2:
                    board_type = parts[0]
                    num_players = int(parts[1].rstrip("p"))
                else:
                    board_type = "square8"
                    num_players = 2

                work_id = await distributor.submit_evaluation(
                    candidate_model=combo.model_path,
                    games=self.config.games_per_harness,
                    board=board_type,
                    num_players=num_players,
                    evaluation_type="gauntlet",
                )
                if work_id:
                    logger.info(
                        f"[{self.name}] Dispatched evaluation to cluster via "
                        f"WorkDistributor: {work_id}"
                    )
                    return True
        except ImportError:
            logger.debug(f"[{self.name}] WorkDistributor not available")
        except Exception as e:
            logger.debug(f"[{self.name}] WorkDistributor dispatch failed: {e}")

        return False

    def _emit_completion_event(self, stats: CycleStats) -> None:
        """Emit event when evaluation cycle completes."""
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.COMPREHENSIVE_EVALUATION_COMPLETED,
                {
                    "total_models": stats.total_models,
                    "unevaluated_count": stats.unevaluated_count,
                    "stale_count": stats.stale_count,
                    "queued_count": stats.queued_count,
                    "evaluated": stats.dispatched_count,
                    "success_count": stats.success_count,
                    "failed_count": stats.failed_count,
                    "remaining_unevaluated": max(
                        0, stats.unevaluated_count - stats.success_count
                    ),
                    "cycle_duration": stats.cycle_duration,
                },
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event system not available")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit completion event: {e}")

    def health_check(self) -> Any:
        """Return health check result for DaemonManager integration."""
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "total_evaluations": self._total_evaluations,
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="ComprehensiveEvaluationLoop is stopped",
            )

        # Check if we're leader
        if not self._is_leader():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.IDLE,
                message="Not leader - evaluation loop idle",
                details={"role": "follower"},
            )

        # Check error rate
        total = self._total_evaluations + self._total_failures
        if total > 0:
            success_rate = self._total_evaluations / total
            if success_rate < 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High failure rate: {(1-success_rate)*100:.1f}%",
                    details={
                        "total_evaluations": self._total_evaluations,
                        "total_failures": self._total_failures,
                        "success_rate": f"{success_rate*100:.1f}%",
                    },
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="ComprehensiveEvaluationLoop healthy",
            details={
                "total_evaluations": self._total_evaluations,
                "total_failures": self._total_failures,
                "last_cycle": (
                    self._last_cycle_stats.cycle_duration
                    if self._last_cycle_stats
                    else None
                ),
                "models_enumerated": (
                    self._last_cycle_stats.total_models
                    if self._last_cycle_stats
                    else 0
                ),
            },
        )

    def get_last_cycle_stats(self) -> CycleStats | None:
        """Get statistics from the last evaluation cycle."""
        return self._last_cycle_stats
