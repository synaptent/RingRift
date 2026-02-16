"""SelfplayScheduler: Priority-based selfplay configuration selection.

Extracted from p2p_orchestrator.py for better modularity.
Handles weighted config selection, job targeting, diversity tracking, and Elo-based priority.

February 2026: Decomposed into mixin modules in selfplay/ subpackage:
- engine_selection.py: Engine mode selection and diversity tracking
- job_targeting.py: Per-node job targeting and resource allocation
- event_handlers.py: Event subscription and handler methods
- priority.py: Priority calculation and architecture selection
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

# February 2026: Import mixin classes from decomposed subpackage
from scripts.p2p.managers.selfplay import (
    ArchitectureSelectionMixin,
    DiversityMetrics,
    EngineSelectionMixin,
    EventHandlersMixin,
    JobTargetingMixin,
    PriorityCalculatorMixin,
)

# Jan 2026: Import adaptive budget calculator for dynamic budget selection
try:
    from app.coordination.budget_calculator import (
        get_adaptive_budget_for_games,
        get_budget_tier_name,
        get_board_adjusted_budget,  # Jan 2026: Large board budget caps
    )
    BUDGET_CALCULATOR_AVAILABLE = True
except ImportError:
    BUDGET_CALCULATOR_AVAILABLE = False
    def get_adaptive_budget_for_games(game_count: int, elo: float) -> int:
        """Fallback budget when calculator not available."""
        if game_count < 100:
            return 64
        elif game_count < 500:
            return 150
        elif game_count < 1000:
            return 200
        elif elo >= 2000:
            return 3200
        elif elo >= 1800:
            return 1600
        else:
            return 800

    def get_board_adjusted_budget(board_type: str, budget: int, game_count: int, num_players: int = 2) -> int:
        """Fallback: no board adjustment when calculator not available."""
        return budget

    def get_budget_tier_name(budget: int) -> str:
        """Fallback tier name when calculator not available."""
        names = {64: "BOOTSTRAP_T1", 150: "BOOTSTRAP_T2", 200: "BOOTSTRAP_T3",
                 800: "STANDARD", 1600: "ULTIMATE", 3200: "MASTER"}
        return names.get(budget, f"CUSTOM({budget})")

logger = logging.getLogger(__name__)


# Import constants from canonical source to avoid duplication
try:
    from app.p2p.constants import (
        PROMOTION_PENALTY_DURATION_CRITICAL,
        PROMOTION_PENALTY_DURATION_MULTIPLE,
        PROMOTION_PENALTY_DURATION_SINGLE,
        PROMOTION_PENALTY_FACTOR_CRITICAL,
        PROMOTION_PENALTY_FACTOR_MULTIPLE,
        PROMOTION_PENALTY_FACTOR_SINGLE,
        TRAINING_BOOST_DURATION,
    )
except ImportError:
    # Fallback for testing/standalone use - match canonical values in app/p2p/constants.py
    TRAINING_BOOST_DURATION = 1800  # 30 minutes
    PROMOTION_PENALTY_DURATION_CRITICAL = 7200  # 2 hours
    PROMOTION_PENALTY_DURATION_MULTIPLE = 3600  # 1 hour
    PROMOTION_PENALTY_DURATION_SINGLE = 1800  # 30 min
    PROMOTION_PENALTY_FACTOR_CRITICAL = 0.3
    PROMOTION_PENALTY_FACTOR_MULTIPLE = 0.5
    PROMOTION_PENALTY_FACTOR_SINGLE = 0.7


class SelfplayScheduler(
    EngineSelectionMixin,
    ArchitectureSelectionMixin,
    PriorityCalculatorMixin,
    JobTargetingMixin,
    EventHandlersMixin,
    EventSubscriptionMixin,
):
    """Manages selfplay configuration selection and job targeting.

    Responsibilities:
    - Weighted config selection based on static priority, Elo performance, curriculum
    - Job targeting per node based on hardware capabilities and utilization
    - Diversity tracking for monitoring
    - Integration with backpressure and resource optimization

    Inherits from EventSubscriptionMixin for standardized event handling (Dec 2025).

    February 2026: Decomposed into mixin classes for better modularity:
    - EngineSelectionMixin: Engine mode selection and diversity tracking
    - ArchitectureSelectionMixin: Per-config architecture selection based on Elo
    - PriorityCalculatorMixin: Priority calculation with staleness, bootstrap, starvation
    - JobTargetingMixin: Per-node job targeting with memory/GPU awareness
    - EventHandlersMixin: Event subscription and 25+ handler methods

    Usage:
        scheduler = SelfplayScheduler(
            get_cluster_elo_fn=lambda: orchestrator._get_cluster_elo_summary(),
            verbose=True
        )

        # Pick a config for a node
        config = scheduler.pick_weighted_config(node)

        # Get job target for a node
        target = scheduler.get_target_jobs_for_node(node)

        # Track diversity
        scheduler.track_diversity(config)

        # Get metrics
        metrics = scheduler.get_diversity_metrics()
    """

    MIXIN_TYPE = "selfplay_scheduler"

    def __init__(
        self,
        get_cluster_elo_fn: Callable[[], dict[str, Any]] | None = None,
        load_curriculum_weights_fn: Callable[[], dict[str, float]] | None = None,
        get_board_priority_overrides_fn: Callable[[], dict[str, int]] | None = None,
        should_stop_production_fn: Callable[[Any], bool] | None = None,
        should_throttle_production_fn: Callable[[Any], bool] | None = None,
        get_throttle_factor_fn: Callable[[Any], float] | None = None,
        record_utilization_fn: Callable[[str, float, float, float, int], None]
        | None = None,
        get_host_targets_fn: Callable[[str], Any] | None = None,
        get_target_job_count_fn: Callable[[str, int, float, float], int] | None = None,
        should_scale_up_fn: Callable[[str, float, float, int], tuple[bool, str]]
        | None = None,
        should_scale_down_fn: Callable[
            [str, float, float, float], tuple[bool, int, str]
        ]
        | None = None,
        get_max_selfplay_for_node_fn: Callable[..., int] | None = None,
        get_hybrid_selfplay_limits_fn: Callable[..., dict[str, int]] | None = None,
        is_emergency_active_fn: Callable[[], bool] | None = None,
        get_active_configs_for_node_fn: Callable[[str], list[str]] | None = None,
        verbose: bool = False,
    ):
        """Initialize the SelfplayScheduler.

        Args:
            get_cluster_elo_fn: Function to get cluster-wide Elo summary
            load_curriculum_weights_fn: Function to load curriculum weights
            get_board_priority_overrides_fn: Function to load board priority overrides
            should_stop_production_fn: Function to check if production should stop (backpressure)
            should_throttle_production_fn: Function to check if production should throttle
            get_throttle_factor_fn: Function to get throttle factor
            record_utilization_fn: Function to record node utilization
            get_host_targets_fn: Function to get host-specific targets
            get_target_job_count_fn: Function to calculate target job count
            should_scale_up_fn: Function to check if scaling up is needed
            should_scale_down_fn: Function to check if scaling down is needed
            get_max_selfplay_for_node_fn: Function to get max selfplay jobs for node
            get_hybrid_selfplay_limits_fn: Function to get hybrid selfplay limits
            is_emergency_active_fn: Function to check if emergency safeguards are active
            get_active_configs_for_node_fn: Function to get active config keys running on a node
                (Session 17.34 - multi-config per node support)
            verbose: Enable verbose logging
        """
        self.get_cluster_elo = get_cluster_elo_fn or (lambda: {})
        self.load_curriculum_weights = load_curriculum_weights_fn or (lambda: {})
        self.get_board_priority_overrides = get_board_priority_overrides_fn or (
            lambda: {}
        )
        self.should_stop_production = should_stop_production_fn
        self.should_throttle_production = should_throttle_production_fn
        self.get_throttle_factor = get_throttle_factor_fn
        self.record_utilization = record_utilization_fn
        self.get_host_targets = get_host_targets_fn
        self.get_target_job_count = get_target_job_count_fn
        self.should_scale_up = should_scale_up_fn
        self.should_scale_down = should_scale_down_fn
        self.get_max_selfplay_for_node = get_max_selfplay_for_node_fn
        self.get_hybrid_selfplay_limits = get_hybrid_selfplay_limits_fn
        self.is_emergency_active = is_emergency_active_fn
        # Session 17.34: Multi-config per node support
        self.get_active_configs_for_node = get_active_configs_for_node_fn
        self.verbose = verbose

        # Diversity tracking
        self.diversity_metrics = DiversityMetrics()

        # Rate adjustment state (December 2025 - feedback loop integration)
        # Maps config_key -> current rate multiplier (1.0 = normal, >1 = boost, <1 = throttle)
        self._rate_multipliers: dict[str, float] = {}
        self._subscribed = False
        self._subscription_lock = threading.Lock()  # Dec 2025: Prevent race in subscribe_to_events

        # Track previous targets for change detection (P0.2 Dec 2025)
        self._previous_targets: dict[str, int] = {}
        self._previous_priorities: dict[str, int] = {}

        # Dec 2025: Initialize boost tracking dicts (fix for hasattr checks)
        # These track temporary boosts to selfplay rates for specific configs
        self._exploration_boosts: dict[str, tuple[float, float]] = {}  # config_key -> (boost_factor, expiry_time)
        self._training_complete_boosts: dict[str, float] = {}  # config_key -> expiry_time

        # Dec 2025 Phase 2B: Track configs currently in training pipeline
        # Used to reduce selfplay allocation while training is active
        self._configs_in_training_pipeline: set[str] = set()

        # Dec 2025: Curriculum weights per config (1.0 = normal)
        # Updated by _on_evaluation_completed based on model performance
        self._curriculum_weights: dict[str, float] = {}

        # Dec 2025 Phase 2: Feedback states for data starvation tracking
        # Maps config_key -> FeedbackState with games_since_last_training
        self._feedback_states: dict[str, Any] = {}

        # Dec 2025 Phase 4D: Track plateaued configs for priority reduction
        # Maps config_key -> expiration timestamp (epoch seconds)
        # Configs in plateau state get 50% priority reduction to avoid wasting resources
        self._plateaued_configs: dict[str, float] = {}

        # Dec 2025 Phase 5: Evaluation backpressure tracking
        # When evaluation queue is full, pause selfplay to prevent cascading backpressure
        self._evaluation_backpressure_active = False
        self._evaluation_backpressure_start: float = 0.0

        # Jan 5, 2026 Session 17.26: Architecture weight caching
        # Caches weights from ARCHITECTURE_WEIGHTS_UPDATED events to reduce DB queries
        # Format: {config_key: (weights_dict, timestamp)}
        self._architecture_weights_cache: dict[str, tuple[dict[str, float], float]] = {}
        self._architecture_weight_cache_ttl = 1800  # 30 minutes, matches emission interval

        # Jan 2026 Sprint 10: Quality-blocked training feedback
        # When training is blocked by quality, boost high-quality selfplay modes
        # Maps config_key -> (quality_boost_factor, expiry_timestamp)
        self._quality_blocked_configs: dict[str, tuple[float, float]] = {}

        # Jan 2026 Session 17.22: Immediate quality score tracking
        # Store quality scores as they're reported via QUALITY_SCORE_UPDATED events
        # Maps config_key -> (quality_score, games_assessed, timestamp)
        self._quality_scores: dict[str, tuple[float, int, float]] = {}

        # Jan 2026 Sprint 13: Force rebalance flag
        # Set when a peer reconnects to trigger immediate work rebalancing
        self._force_rebalance: bool = False

        # Jan 2026 Sprint 6: Job spawn verification tracking
        # Tracks pending jobs waiting for verification and spawn success/failure stats
        # Pending jobs: job_id -> (node_id, config_key, spawn_time)
        self._pending_spawn_verification: dict[str, tuple[str, str, float]] = {}
        self._spawn_verification_lock = threading.Lock()
        # Per-node spawn success/failure tracking for capacity estimation
        self._spawn_success_count: dict[str, int] = {}  # node_id -> count
        self._spawn_failure_count: dict[str, int] = {}  # node_id -> count
        # Jan 2026: Spawn verification timeout and callback
        self._spawn_verification_timeout: float = float(
            os.environ.get("RINGRIFT_SPAWN_VERIFICATION_TIMEOUT", "30.0")
        )
        # Callback to check if a job is running (set by P2P orchestrator)
        self._get_job_status_fn: Callable[[str], dict[str, Any] | None] | None = None

    # =========================================================================
    # Job Spawn Verification (January 2026 - Sprint 6)
    # =========================================================================

    def set_job_status_callback(
        self,
        get_job_status_fn: Callable[[str], dict[str, Any] | None]
    ) -> None:
        """Set the callback function to check job status.

        January 2026 - Sprint 6: Added for job spawn verification.
        This callback is used to verify that a job is actually running.

        Args:
            get_job_status_fn: Function that takes job_id and returns job status dict
                or None if job not found. The dict should have a 'status' key.
        """
        self._get_job_status_fn = get_job_status_fn

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

        Jan 2, 2026 - Sprint 9: Changed return type from tuple to dict for
        SpawnVerificationLoop compatibility.

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

    # =========================================================================
    # Job Selection
    # =========================================================================

    async def get_job_for_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a recommended selfplay job configuration for a specific node.

        January 2026 Sprint 6: Used by PredictiveScalingLoop for preemptive job spawning.
        Selects the best config based on priority-weighted allocation, with minimum
        allocation enforcement for underserved configs.

        Args:
            node_id: The target node identifier

        Returns:
            Dict with board_type, num_players, num_games if a job is recommended,
            or None if no job should be spawned.
        """
        try:
            # Check if we should force allocation to underserved config (minimum enforcement)
            config_key = self._get_enforced_minimum_allocation()

            if config_key is None:
                # No enforcement needed, use priority-weighted selection
                config_key = self._pick_simple_weighted_config()

            if config_key is None:
                return None

            # Parse config key
            parts = config_key.split("_")
            if len(parts) >= 2 and parts[-1].endswith("p"):
                board_type = "_".join(parts[:-1])  # Handle square19, hex8, etc.
                num_players = int(parts[-1].rstrip("p"))
            else:
                return None

            # Determine games based on node capabilities (conservative for preemptive)
            # Preemptive jobs should be smaller to allow quick reallocation
            num_games = 100

            return {
                "board_type": board_type,
                "num_players": num_players,
                "num_games": num_games,
                "config_key": config_key,
            }

        except Exception as e:
            logger.debug(f"Failed to get job for node {node_id}: {e}")
            return None

    # =========================================================================
    # Unified Game Count Aggregation
    # =========================================================================

    async def refresh_from_unified_aggregator(self) -> dict[str, int]:
        """Refresh game counts from all sources (LOCAL, CLUSTER, S3, OWC).

        January 14, 2026: Added to ensure selfplay allocation uses complete data
        visibility across all storage locations including OWC external drive.

        Returns:
            Dict mapping config_key -> total_game_count across all sources
        """
        try:
            from app.utils.unified_game_aggregator import get_unified_game_aggregator

            aggregator = get_unified_game_aggregator()
            counts = await aggregator.get_all_configs_counts()

            # Extract totals from AggregatedGameCount objects
            totals: dict[str, int] = {}
            for config_key, agg_count in counts.items():
                if hasattr(agg_count, "total_games"):
                    totals[config_key] = agg_count.total_games
                elif isinstance(agg_count, dict):
                    totals[config_key] = agg_count.get("total_games", 0)
                else:
                    totals[config_key] = int(agg_count) if agg_count else 0

            self._unified_game_counts = totals
            logger.info(
                f"[SelfplayScheduler] Unified aggregator refresh: "
                f"{sum(totals.values()):,} total games across {len(totals)} configs"
            )
            return totals

        except ImportError as e:
            logger.debug(f"[SelfplayScheduler] UnifiedGameAggregator not available: {e}")
            return {}
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Unified aggregator refresh failed: {e}")
            return {}

    # =========================================================================
    # Auto-Start Selfplay (January 2026 - Phase 18A)
    # =========================================================================

    async def auto_start_selfplay(self, peer: Any, idle_duration: float) -> None:
        """Auto-start diverse hybrid selfplay on an idle node.

        Works with both NodeInfo (P2P peers) and DiscoveredNode (unified inventory).
        Uses diverse profiles for high-quality training data:
        - Multiple engine modes (gumbel-mcts, nnue-guided, policy-only, mcts)
        - Multiple board types (hex8, square8, square19)
        - Multiple player counts (2, 3, 4)
        - Multiple heuristic profiles (balanced, aggressive, territorial, defensive)

        Jan 28, 2026: Phase 18A - Migrated from p2p_orchestrator.py.

        Args:
            peer: NodeInfo or DiscoveredNode representing the idle node
            idle_duration: How long the node has been idle (seconds)
        """
        # Check for GPU - works with both NodeInfo and DiscoveredNode
        gpu_name = getattr(peer, "gpu_name", "") or ""

        # Don't auto-start on nodes that aren't GPU nodes
        has_gpu = bool(gpu_name) or getattr(peer, "has_gpu", False)
        is_gpu_node_fn = getattr(peer, "is_gpu_node", lambda: has_gpu)
        is_gpu_node = is_gpu_node_fn() if callable(is_gpu_node_fn) else is_gpu_node_fn
        if not has_gpu and not is_gpu_node:
            return

        # GPU selfplay uses batch processing - scale based on GPU power
        # Jan 7, 2026: Reduced games_per_process to prevent 1-hour timeouts
        gpu_upper = gpu_name.upper()
        if "GH200" in gpu_upper or "H100" in gpu_upper or "H200" in gpu_upper:
            num_processes = 4
            games_per_process = 500
            gpu_tier = "high"
        elif "A100" in gpu_upper or "A40" in gpu_upper:
            num_processes = 3
            games_per_process = 300
            gpu_tier = "high"
        elif "4090" in gpu_upper or "5090" in gpu_upper:
            num_processes = 3
            games_per_process = 300
            gpu_tier = "mid"
        elif "4080" in gpu_upper or "5080" in gpu_upper or "5070" in gpu_upper:
            num_processes = 2
            games_per_process = 200
            gpu_tier = "mid"
        elif "3090" in gpu_upper or "4070" in gpu_upper:
            num_processes = 2
            games_per_process = 200
            gpu_tier = "mid"
        else:
            num_processes = 2
            games_per_process = 100
            gpu_tier = "low"

        peer_node_id = getattr(peer, "node_id", "unknown")
        self._log_info(
            f"Auto-starting {num_processes} diverse selfplay processes on idle node {peer_node_id} "
            f"(GPU={gpu_name}, tier={gpu_tier}, {games_per_process} games each, idle for {idle_duration:.0f}s)"
        )

        # Send parallel requests to /selfplay/start endpoint
        try:
            # Import profile selector
            try:
                from scripts.p2p.config.selfplay_job_configs import select_diverse_profiles
            except ImportError:
                # Fallback: use simple profile selection
                def select_diverse_profiles(k: int) -> list[dict]:
                    return [
                        {"board_type": "hex8", "num_players": 2, "engine_mode": "gumbel-mcts", "profile": "balanced", "description": "default"},
                    ] * k

            selected_profiles = select_diverse_profiles(k=num_processes)

            # Build job configs from selected profiles
            job_configs = []
            for i, profile in enumerate(selected_profiles):
                job_configs.append({
                    "board_type": profile["board_type"],
                    "num_players": profile["num_players"],
                    "num_games": games_per_process,
                    "engine_mode": profile["engine_mode"],
                    "heuristic_profile": profile.get("profile", "balanced"),
                    "auto_assigned": True,
                    "reason": f"auto_idle_{profile['engine_mode']}_{profile['board_type']}_{profile['num_players']}p_{int(idle_duration)}s",
                })
                self._log_debug(f"  Process {i}: {profile.get('description', profile['engine_mode'])}")

            # Feb 2026: If orchestrator is not available, fall back to work queue
            # instead of silently returning. This was causing 108+ selfplay dispatch
            # failures per day because _orchestrator was None after init.
            has_orchestrator = hasattr(self, "_orchestrator") and self._orchestrator
            if not has_orchestrator:
                self._log_info(
                    f"No orchestrator for HTTP push to {peer_node_id}, "
                    f"falling back to work queue for {len(job_configs)} selfplay jobs"
                )

            # NAT-blocked node detection and work queue routing
            # Feb 2026: Also route via work queue when orchestrator is unavailable
            is_nat_blocked = getattr(peer, "nat_blocked", False)
            if is_nat_blocked or not has_orchestrator:
                try:
                    from app.coordination.work_queue import WorkItem, WorkType, get_work_queue
                    wq = get_work_queue()
                    if wq is not None:
                        queued_count = 0
                        for cfg in job_configs:
                            work_item = WorkItem(
                                work_type=WorkType.SELFPLAY,
                                priority=50,  # Normal priority for auto-idle work
                                config={
                                    **cfg,
                                    "target_node": peer_node_id,
                                    "target_node_expires_at": time.time() + 600,  # 10 min expiration
                                    "nat_blocked_dispatch": True,
                                },
                                timeout_seconds=3600.0,
                            )
                            wq.add_work(work_item)
                            queued_count += 1

                        from collections import Counter
                        engine_counts = Counter(cfg["engine_mode"] for cfg in job_configs)
                        board_counts = Counter(cfg["board_type"] for cfg in job_configs)
                        profile_summary = ", ".join(f"{k}:{v}" for k, v in engine_counts.items())
                        board_summary = ", ".join(f"{k}:{v}" for k, v in board_counts.items())

                        self._log_info(
                            f"NAT-blocked node {peer_node_id}: Queued {queued_count} selfplay jobs "
                            f"[engines: {profile_summary}] [boards: {board_summary}] "
                            f"(idle for {idle_duration:.0f}s, WorkerPullLoop will claim)"
                        )
                        return
                    else:
                        self._log_warning(f"NAT-blocked node {peer_node_id}: Work queue unavailable, cannot dispatch")
                        return
                except Exception as e:
                    self._log_warning(f"NAT-blocked node {peer_node_id}: Failed to queue work: {e}")
                    return

            # Non-NAT-blocked nodes: Direct HTTP push to /selfplay/start endpoint
            try:
                from aiohttp import ClientTimeout
                from scripts.p2p.network import get_client_session
            except ImportError:
                self._log_warning("get_client_session not available for auto_start_selfplay")
                return

            url = self._orchestrator._url_for_peer(peer, "/selfplay/start")
            timeout = ClientTimeout(total=30)

            async def send_selfplay_request(session: Any, payload: dict) -> bool:
                """Send a single selfplay start request."""
                async with session.post(url, json=payload, headers=self._orchestrator._auth_headers()) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        body = await resp.text()
                        self._log_warning(f"Failed selfplay request on {peer_node_id}: {resp.status} {body[:100]}")
                        return False

            async with get_client_session(timeout) as session:
                import asyncio
                tasks = [send_selfplay_request(session, cfg) for cfg in job_configs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                started = 0
                failed = 0
                for r in results:
                    if isinstance(r, Exception):
                        failed += 1
                        self._log_debug(f"Selfplay request failed with exception: {r}")
                    elif r is True:
                        started += 1
                if failed > 0:
                    self._log_warning(f"Auto-start on {peer_node_id}: {failed}/{len(results)} requests failed")

                # Log profile distribution
                from collections import Counter
                engine_counts = Counter(cfg["engine_mode"] for cfg in job_configs)
                board_counts = Counter(cfg["board_type"] for cfg in job_configs)
                profile_summary = ", ".join(f"{k}:{v}" for k, v in engine_counts.items())
                board_summary = ", ".join(f"{k}:{v}" for k, v in board_counts.items())

                self._log_info(
                    f"Auto-started {started}/{num_processes} diverse selfplay on {peer_node_id} "
                    f"[engines: {profile_summary}] [boards: {board_summary}]"
                )

        except Exception as e:  # noqa: BLE001
            self._log_warning(f"Auto-start request failed for {peer_node_id}: {e}")

    # =========================================================================
    # Main Config Selection (pick_weighted_config)
    # =========================================================================

    def pick_weighted_config(self, node: NodeInfo) -> dict[str, Any] | None:
        """Pick a selfplay config weighted by priority and node capabilities.

        PRIORITY-BASED SCHEDULING: Combines static priority with dynamic
        ELO-based boosts to allocate more resources to high-performing configs.

        Args:
            node: Node information for capability filtering

        Returns:
            Config dict with board_type, num_players, engine_mode, or None if no valid config
        """
        # Get the selfplay configs - DIVERSE mode prioritized for high-quality training data
        # Uses "mixed" engine mode for varied AI matchups (NNUE, MCTS, heuristic combinations)
        selfplay_configs = [
            # Priority 8: Underrepresented hex/sq19 combos with diverse AI (highest priority)
            {
                "board_type": "hexagonal",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hexagonal",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "hex8",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 8,
            },
            {
                "board_type": "square19",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 8,
            },
            # Priority 7: Square8 multi-player with diverse AI
            {
                "board_type": "square8",
                "num_players": 3,
                "engine_mode": "mixed",
                "priority": 7,
            },
            {
                "board_type": "square8",
                "num_players": 4,
                "engine_mode": "mixed",
                "priority": 7,
            },
            # Priority 6: Cross-AI matches (specific matchup types)
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            {
                "board_type": "hexagonal",
                "num_players": 3,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            {
                "board_type": "square19",
                "num_players": 2,
                "engine_mode": "heuristic-vs-mcts",
                "priority": 6,
            },
            # Priority 5: Standard 2p square8 with diverse AI
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "mixed",
                "priority": 5,
            },
            # Priority 4: Tournament varied (for evaluation-style games)
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "tournament-varied",
                "priority": 4,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "tournament-varied",
                "priority": 4,
            },
            # Priority 3: CPU-bound specialized modes
            {
                "board_type": "square8",
                "num_players": 2,
                "engine_mode": "mcts-only",
                "priority": 3,
            },
            {
                "board_type": "hexagonal",
                "num_players": 2,
                "engine_mode": "gumbel-mcts",  # Jan 2026: High-quality GPU mode
                "priority": 3,
            },
            # Priority 2: Heterogeneous cross-AI games (December 2025)
            # Per-player AI configs for maximum training diversity
            # Neural net learning from diverse opponent behaviors
            {
                "board_type": "hex8",
                "num_players": 2,
                "engine_mode": "cross-ai",
                "priority": 2,
                "player_ai_configs": {
                    1: {"engine": "gumbel-mcts", "budget": 150},
                    2: {"engine": "heuristic", "difficulty": 5},
                },
            },
            {
                "board_type": "square8",
                "num_players": 3,
                "engine_mode": "cross-ai",
                "priority": 2,
                "player_ai_configs": {
                    1: {"engine": "gumbel-mcts", "budget": 150},
                    2: {"engine": "brs"},
                    3: {"engine": "maxn"},
                },
            },
            {
                "board_type": "hex8",
                "num_players": 4,
                "engine_mode": "cross-ai",
                "priority": 2,
                "player_ai_configs": {
                    1: {"engine": "gumbel-mcts", "budget": 200},
                    2: {"engine": "heuristic", "difficulty": 6},
                    3: {"engine": "brs"},
                    4: {"engine": "random"},
                },
            },
        ]

        # Filter by GPU VRAM (avoid large boards on low-VRAM GPUs)
        # NOTE: Use gpu_vram_gb (GPU memory), NOT memory_gb (system RAM)
        # Bug fix Dec 2025: Was incorrectly using system RAM which filtered out
        # Vast.ai nodes with 12-16GB VRAM but >48GB system RAM
        gpu_vram = int(
            getattr(node, "gpu_vram_gb", 0)
            or getattr(node, "gpu_memory_gb", 0)
            or 0
        )
        if gpu_vram and gpu_vram < 48:
            # Jan 12, 2026: Allow hex8 and square8 on smaller GPUs, only filter truly large boards
            # Previous filter was too restrictive (only square8), blocking hex8 which fits in 8GB
            selfplay_configs = [
                c for c in selfplay_configs
                if c.get("board_type") not in ("square19", "hexagonal")
            ]

        # December 2025: Filter by GPU capability
        # CPU-only nodes should only get CPU-compatible engine modes
        # This prevents wasting compute on GPU-required modes that will fall back to heuristic
        selfplay_configs = self._filter_configs_by_gpu(selfplay_configs, node)

        if not selfplay_configs:
            node_id = getattr(node, "node_id", "unknown")
            logger.warning(
                f"No compatible selfplay configs for node {node_id} "
                f"(has_gpu={self._node_has_gpu(node)}, gpu_vram={gpu_vram}GB)"
            )
            return None

        # PRIORITY-BASED SCHEDULING: Add ELO-based priority boosts
        # Phase 3.1: Also incorporate curriculum weights from unified AI loop
        curriculum_weights = {}
        try:
            curriculum_weights = self.load_curriculum_weights()
        except (OSError, ValueError, KeyError, ImportError):
            pass  # Use empty weights on error

        # Load board priority overrides from config (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
        board_priority_overrides = self.get_board_priority_overrides()

        for cfg in selfplay_configs:
            elo_boost = self.get_elo_based_priority_boost(
                cfg.get("board_type", ""),
                cfg.get("num_players", 2),
            )

            # Phase 3.1: Apply curriculum weight boost
            # Config keys are formatted as "board_type_Np" (e.g., "square8_2p")
            config_key = f"{cfg.get('board_type', '')}_{cfg.get('num_players', 2)}p"
            curriculum_weight = curriculum_weights.get(config_key, 1.0)
            # Convert weight (0.7-1.5) to priority boost (0-3)
            # weight 0.7 = -1 boost, weight 1.0 = 0 boost, weight 1.5 = +2 boost
            curriculum_boost = int((curriculum_weight - 1.0) * 4)
            curriculum_boost = max(-2, min(3, curriculum_boost))  # Clamp to -2..+3

            # Apply board priority overrides from config
            # 0=CRITICAL adds +6, 1=HIGH adds +4, 2=MEDIUM adds +2, 3=LOW adds 0
            board_priority = board_priority_overrides.get(
                config_key, 3
            )  # Default to LOW (3)
            board_priority_boost = (3 - board_priority) * 2  # 0->6, 1->4, 2->2, 3->0

            # Apply rate multiplier from feedback loop (December 2025)
            # Rate multiplier > 1 = boost priority, < 1 = reduce priority
            rate_multiplier = self.get_rate_multiplier(config_key)
            rate_boost = int((rate_multiplier - 1.0) * 5)  # ±5 priority max
            rate_boost = max(-3, min(5, rate_boost))  # Clamp to -3..+5

            # Dec 2025 Phase 2: Apply data starvation boost
            # Configs with few recent games get priority boost to ensure training data
            starvation_multiplier = self._get_data_starvation_boost(config_key)
            # Convert multiplier (0.5-2.0) to additive boost (-3 to +5)
            starvation_boost = int((starvation_multiplier - 1.0) * 5)
            starvation_boost = max(-3, min(5, starvation_boost))  # Clamp to -3..+5

            # Sprint 17.9 (Jan 2026): Apply bootstrap boost for critically underserved configs
            # This provides very aggressive priority for configs that need initial data collection
            bootstrap_boost = self._get_bootstrap_priority_boost(config_key)

            # Dec 2025 Phase 4D: Apply plateau penalty
            # Configs in plateau state (no Elo improvement) get 50% priority reduction
            # to redirect resources to configs making progress
            base_priority = (
                cfg.get("priority", 1)
                + elo_boost
                + curriculum_boost
                + board_priority_boost
                + rate_boost
                + starvation_boost
                + bootstrap_boost
            )

            if self._is_config_plateaued(config_key):
                # Apply 50% penalty for plateaued configs
                cfg["effective_priority"] = max(1, int(base_priority * 0.5))
            else:
                cfg["effective_priority"] = base_priority

        # Session 17.34: Apply multi-config preference for large GPUs
        # Boosts priority for configs NOT currently running on the node
        selfplay_configs = self._apply_multi_config_preference(selfplay_configs, node)

        # Build weighted list by effective priority
        weighted = []
        for cfg in selfplay_configs:
            # Ensure minimum priority of 1
            priority = max(1, cfg.get("effective_priority", 1))
            weighted.extend([cfg] * priority)

        selected = random.choice(weighted) if weighted else None

        # P0.2 (Dec 2025): Emit event when priority changes significantly
        if selected:
            config_key = f"{selected.get('board_type', '')}_{selected.get('num_players', 2)}p"
            new_priority = selected.get("effective_priority", 1)
            old_priority = self._previous_priorities.get(config_key, new_priority)

            # Emit event if priority changed by 2+ or rate multiplier applied
            rate_mult = self.get_rate_multiplier(config_key)
            priority_change = abs(new_priority - old_priority)
            if priority_change >= 2 or (rate_mult != 1.0 and priority_change >= 1):
                reason = "priority_boost" if new_priority > old_priority else "priority_reduced"
                self._emit_selfplay_target_updated(
                    config_key=config_key,
                    priority="high" if priority_change >= 3 else "normal",
                    reason=f"{reason}:{priority_change}",
                    effective_priority=new_priority,
                    exploration_boost=rate_mult if rate_mult != 1.0 else None,
                )
                self._previous_priorities[config_key] = new_priority

        # December 2025: Apply mixed-engine selection for ALL board types
        # Instead of returning "mixed" mode, select a specific engine from the mix
        # This ensures BRS/MaxN diversity in hex8/square8 as well as large boards
        if selected:
            board_type = selected.get("board_type", "")
            engine_mode = selected.get("engine_mode", "")
            num_players = selected.get("num_players", 0)
            config_key = f"{board_type}_{num_players}p"

            # January 2026: Check if config should use mixed opponent training
            # Weak configs benefit from diverse opponents to break weak-vs-weak cycles
            if self.should_use_mixed_opponents(config_key):
                # Override engine mode to use MixedOpponentSelfplayRunner
                # This provides diverse opponents (random, heuristic, mcts, minimax, etc.)
                selected["engine_mode"] = "mixed-opponents"
                logger.info(
                    f"[MixedOpponents] {config_key}: forcing mixed opponent mode for diverse training"
                )
                return selected

            # Apply engine mix for any board type with "mixed" or "diverse" mode
            if engine_mode in ("mixed", "diverse"):
                has_gpu = self._node_has_gpu(node)
                num_players = selected.get("num_players", 0)
                config_key = f"{board_type}_{num_players}p"

                # Jan 2026: Get adaptive budget based on game count and Elo
                # This replaces static budgets with dynamic calculation
                adaptive_budget = self.get_adaptive_selfplay_budget(config_key)
                # Feb 2026: Apply large board budget caps scaled by player count
                game_count = self._get_game_counts_per_config().get(config_key, 0)
                adaptive_budget = get_board_adjusted_budget(board_type, adaptive_budget, game_count, num_players)

                # Jan 2026 Sprint 10: Check for quality boost - forces high-quality modes
                quality_boost = self.get_quality_boost(config_key)

                if quality_boost > 1.0 and has_gpu:
                    # Quality boost active - force high-quality Gumbel MCTS
                    # Budget scales with boost, starting from adaptive budget
                    boosted_budget = int(adaptive_budget * quality_boost)
                    # Cap at MASTER tier (3200) to prevent excessive compute
                    boosted_budget = min(boosted_budget, 3200)

                    actual_engine = "gumbel-mcts"
                    extra_args = {"budget": boosted_budget}

                    logger.info(
                        f"Quality boost override: {config_key} using '{actual_engine}' "
                        f"with budget={boosted_budget} (adaptive={adaptive_budget}, boost={quality_boost:.2f}x)"
                    )
                else:
                    # Normal selection from engine mix
                    actual_engine, extra_args = self._select_board_engine(
                        has_gpu=has_gpu,
                        board_type=board_type,
                        num_players=num_players,
                    )

                    # Jan 2026: Override static budget with adaptive budget for gumbel-mcts
                    # This is the KEY FIX - configs with 1000+ games now get 800+ budget
                    if actual_engine == "gumbel-mcts" and extra_args:
                        old_budget = extra_args.get("budget", 0)
                        extra_args["budget"] = adaptive_budget
                        if old_budget != adaptive_budget:
                            logger.info(
                                f"[AdaptiveBudget] {config_key}: {old_budget} -> {adaptive_budget} "
                                f"({get_budget_tier_name(adaptive_budget)})"
                            )

                # Update the config with the selected engine
                selected["engine_mode"] = actual_engine
                if extra_args:
                    selected["engine_extra_args"] = extra_args

                # Determine board category for logging
                board_category = "large" if board_type in self.LARGE_BOARD_TYPES else "standard"
                if quality_boost <= 1.0:
                    logger.info(
                        f"{board_category.capitalize()} board engine mix: {board_type}_{num_players}p '{engine_mode}' -> "
                        f"'{actual_engine}' (gpu={has_gpu}, extra_args={extra_args})"
                    )

            # Jan 2026: Handle mode-specific mixes (minimax-only, mcts-only, descent-only)
            # These modes now resolve to specific engines (including BRS/MaxN) for diversity tracking
            elif engine_mode in self.MODE_SPECIFIC_MIXES:
                has_gpu = self._node_has_gpu(node)
                num_players = selected.get("num_players", 0)
                config_key = f"{board_type}_{num_players}p"

                # Get the appropriate mix based on GPU availability
                gpu_mix, cpu_mix = self.MODE_SPECIFIC_MIXES[engine_mode]
                engine_mix = gpu_mix if has_gpu else cpu_mix

                # Filter to available engines (respect GPU requirements)
                available_engines = [
                    (mode, weight, gpu_required, args)
                    for mode, weight, gpu_required, args in engine_mix
                    if not gpu_required or has_gpu
                ]

                if available_engines:
                    # Weighted random selection
                    weighted_engines = []
                    for mode, weight, _gpu, args in available_engines:
                        weighted_engines.extend([(mode, args)] * weight)

                    if weighted_engines:
                        actual_engine, extra_args = random.choice(weighted_engines)

                        # Update the config with the selected engine
                        selected["engine_mode"] = actual_engine
                        if extra_args:
                            selected["engine_extra_args"] = extra_args

                        logger.info(
                            f"Mode-specific engine mix: {config_key} '{engine_mode}' -> "
                            f"'{actual_engine}' (gpu={has_gpu}, extra_args={extra_args})"
                        )

        # Session 17.22: Add architecture selection based on Elo performance
        # This closes the feedback loop: better-performing architectures get more selfplay
        if selected:
            selected_arch = self._select_architecture_for_config(
                board_type=selected.get("board_type", ""),
                num_players=selected.get("num_players", 2),
            )
            selected["model_version"] = selected_arch

        return selected

    # =========================================================================
    # Training Completion and Exploration Boost
    # =========================================================================

    def on_training_complete(self, config_key: str) -> None:
        """Handle training completion for a config.

        Called by P2P orchestrator when TRAINING_COMPLETED event fires.
        Refreshes selfplay priorities to potentially increase allocation
        for the just-trained config (more data needed for next training cycle).

        P0.1 (Dec 2025): Added to fix missing method referenced at
        p2p_orchestrator.py:2338.

        Args:
            config_key: The config key (e.g., "hex8_2p") that completed training.
        """
        logger.info(
            f"[SelfplayScheduler] Training completed for {config_key}, "
            f"refreshing priorities"
        )

        # Boost selfplay rate for this config temporarily (just trained = needs more data)
        try:
            # Increase rate multiplier for 30 minutes after training
            boost_duration = TRAINING_BOOST_DURATION
            expiry = time.time() + boost_duration
            # Dec 2025: _training_complete_boosts initialized in __init__
            self._training_complete_boosts[config_key] = expiry

            logger.debug(
                f"[SelfplayScheduler] Boosting {config_key} selfplay priority "
                f"for {boost_duration}s after training completion"
            )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error boosting {config_key}: {e}")

    def set_exploration_boost(
        self, config_key: str, boost_factor: float, duration_seconds: float = 900
    ) -> None:
        """Set exploration boost for a config due to training anomaly.

        P0.1 (Dec 2025): Added to handle EXPLORATION_BOOST events from
        FeedbackLoopController when training anomalies are detected.

        When training has loss spikes or stalls, we boost exploration to
        generate more diverse training data.

        Args:
            config_key: The config key (e.g., "hex8_2p") to boost.
            boost_factor: Multiplicative boost factor (e.g., 1.3 for 30% more).
            duration_seconds: How long the boost should last.
        """
        try:
            expiry = time.time() + duration_seconds
            # Dec 2025: _exploration_boosts initialized in __init__
            self._exploration_boosts[config_key] = (boost_factor, expiry)

            logger.info(
                f"[SelfplayScheduler] Set exploration boost for {config_key}: "
                f"{boost_factor:.2f}x for {duration_seconds}s"
            )

            # P0.2 Dec 2025: Emit allocation updated event
            # Get current curriculum weights to include in the event
            curriculum_weights = {}
            try:
                curriculum_weights = self.load_curriculum_weights()
            except (OSError, ValueError, KeyError, ImportError):
                pass  # Config file read/parse/import errors - use empty weights
            self._emit_selfplay_allocation_updated(
                config_key, curriculum_weights, boost_factor, "exploration_boost"
            )
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error setting exploration boost: {e}")

    def get_exploration_boost(self, config_key: str) -> float:
        """Get current exploration boost factor for a config.

        Args:
            config_key: The config key to check.

        Returns:
            Boost factor (1.0 = no boost, >1.0 = boosted).
        """
        # Dec 2025: _exploration_boosts always initialized in __init__
        boost_info = self._exploration_boosts.get(config_key)
        if not boost_info:
            return 1.0

        boost_factor, expiry = boost_info
        if time.time() > expiry:
            # Boost expired, clean up
            del self._exploration_boosts[config_key]
            return 1.0

        return boost_factor

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self) -> "HealthCheckResult":
        """Check health status of SelfplayScheduler.

        Returns:
            HealthCheckResult with status, scheduling metrics, and error info
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        status = CoordinatorStatus.RUNNING
        is_healthy = True
        errors_count = 0
        last_error: str | None = None

        # Get diversity metrics
        diversity = self.get_diversity_metrics()

        # Check if subscribed to events
        if not self._subscribed:
            if is_healthy:
                status = CoordinatorStatus.DEGRADED
                last_error = "Not subscribed to events"

        # Check config coverage (all configs should have some targets)
        configs_with_targets = sum(
            1 for v in self._previous_targets.values() if v > 0
        )
        total_configs = len(self._previous_targets) if self._previous_targets else 0

        if total_configs > 0 and configs_with_targets == 0:
            status = CoordinatorStatus.DEGRADED
            last_error = "No configs have selfplay targets"
        elif total_configs > 0 and configs_with_targets < total_configs // 2:
            if is_healthy:
                status = CoordinatorStatus.DEGRADED
                last_error = f"Only {configs_with_targets}/{total_configs} configs have targets"

        # Count active exploration boosts
        # Dec 2025: _exploration_boosts always initialized in __init__
        now = time.time()
        active_boosts = sum(
            1 for _, expiry in self._exploration_boosts.values()
            if expiry > now
        )

        return HealthCheckResult(
            healthy=is_healthy,
            status=status if isinstance(status, str) else status,
            message=last_error or "SelfplayScheduler healthy",
            details={
                "operations_count": total_configs,
                "errors_count": errors_count,
                "subscribed": self._subscribed,
                "configs_tracked": total_configs,
                "configs_with_targets": configs_with_targets,
                "active_exploration_boosts": active_boosts,
                "diversity_metrics": diversity,
            },
        )

    def record_promotion_failure(self, config_key: str) -> None:
        """Record a promotion failure for curriculum feedback.

        December 27, 2025: Added to handle PROMOTION_FAILED events.
        When a model fails to promote (gauntlet failure), we temporarily
        reduce the selfplay priority for that config to avoid wasting
        resources on a potentially unstable training trajectory.

        Args:
            config_key: The configuration that failed promotion (e.g., "hex8_2p")
        """
        if not hasattr(self, "_promotion_failures"):
            self._promotion_failures: dict[str, list[float]] = {}

        # Track failure timestamps
        if config_key not in self._promotion_failures:
            self._promotion_failures[config_key] = []
        self._promotion_failures[config_key].append(time.time())

        # Keep only failures from last 24 hours
        cutoff = time.time() - 86400
        self._promotion_failures[config_key] = [
            t for t in self._promotion_failures[config_key] if t > cutoff
        ]

        failure_count = len(self._promotion_failures[config_key])
        logger.info(
            f"[SelfplayScheduler] Recorded promotion failure for {config_key} "
            f"({failure_count} failures in last 24h)"
        )

        # Apply temporary priority reduction based on failure count
        # More failures = longer penalty period
        if failure_count >= 3:
            # After 3 failures, significantly reduce priority for 2 hours
            penalty_duration = PROMOTION_PENALTY_DURATION_CRITICAL
            penalty_factor = PROMOTION_PENALTY_FACTOR_CRITICAL
        elif failure_count >= 2:
            # After 2 failures, reduce priority for 1 hour
            penalty_duration = PROMOTION_PENALTY_DURATION_MULTIPLE
            penalty_factor = PROMOTION_PENALTY_FACTOR_MULTIPLE
        else:
            # First failure, reduce priority for 30 minutes
            penalty_duration = PROMOTION_PENALTY_DURATION_SINGLE
            penalty_factor = PROMOTION_PENALTY_FACTOR_SINGLE

        # Store penalty in exploration boosts (negative boost = reduced priority)
        if not hasattr(self, "_promotion_penalties"):
            self._promotion_penalties: dict[str, tuple[float, float]] = {}
        self._promotion_penalties[config_key] = (penalty_factor, time.time() + penalty_duration)

        logger.info(
            f"[SelfplayScheduler] Applied {penalty_factor:.0%} priority penalty "
            f"to {config_key} for {penalty_duration}s"
        )
