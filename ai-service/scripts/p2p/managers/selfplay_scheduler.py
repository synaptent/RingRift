"""SelfplayScheduler: Priority-based selfplay configuration selection.

Extracted from p2p_orchestrator.py for better modularity.
Handles weighted config selection, job targeting, diversity tracking, and Elo-based priority.

February 2026: Decomposed into mixin modules in selfplay/ subpackage:
- engine_selection.py: Engine mode selection and diversity tracking
- job_targeting.py: Per-node job targeting and resource allocation
- event_handlers.py: Event subscription and handler methods
- priority.py: Priority calculation and architecture selection
- config_selection.py: Core config selection (pick_weighted_config) and boost management
- dispatch.py: Spawn verification and dispatch tracking
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

# February 2026: Import mixin classes from decomposed subpackage
from scripts.p2p.managers.selfplay import (
    ArchitectureSelectionMixin,
    ConfigSelectionMixin,
    DispatchTrackingMixin,
    DiversityMetrics,
    EngineSelectionMixin,
    EventHandlersMixin,
    JobTargetingMixin,
    PriorityCalculatorMixin,
)

logger = logging.getLogger(__name__)


class SelfplayScheduler(
    ConfigSelectionMixin,
    DispatchTrackingMixin,
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
    - ConfigSelectionMixin: Core config selection (pick_weighted_config) and boost management
    - DispatchTrackingMixin: Spawn verification and dispatch tracking
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

            # Feb 2026: Starvation-aware profile selection.
            forced_profiles: list[dict] = []
            try:
                underserved = self._get_underserved_configs()
                if underserved:
                    # Sort by game count (most starved first)
                    underserved.sort(key=lambda x: x[1])
                    # Force at least half the processes to target starved configs
                    num_forced = max(1, num_processes // 2)
                    for i in range(num_forced):
                        config_key, game_count = underserved[i % len(underserved)]
                        parts = config_key.split("_")
                        board_type = "_".join(parts[:-1])
                        n_players = int(parts[-1].rstrip("p"))
                        forced_profiles.append({
                            "engine_mode": "gumbel-mcts",
                            "board_type": board_type,
                            "num_players": n_players,
                            "profile": "balanced",
                            "description": f"Starvation-forced {config_key} ({game_count} games)",
                        })
                    if forced_profiles:
                        forced_desc = ", ".join(
                            f"{f['board_type']}_{f['num_players']}p" for f in forced_profiles
                        )
                        self._log_info(
                            f"Starvation override: {len(forced_profiles)}/{num_processes} slots "
                            f"forced to [{forced_desc}]"
                        )
            except Exception:
                pass  # Fall through to random selection

            remaining = num_processes - len(forced_profiles)
            random_profiles = select_diverse_profiles(k=remaining) if remaining > 0 else []
            selected_profiles = forced_profiles + random_profiles

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
            has_orchestrator = hasattr(self, "_orchestrator") and self._orchestrator
            if not has_orchestrator:
                self._log_info(
                    f"No orchestrator for HTTP push to {peer_node_id}, "
                    f"falling back to work queue for {len(job_configs)} selfplay jobs"
                )

            # NAT-blocked node detection and work queue routing
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
                for idx, r in enumerate(results):
                    if isinstance(r, Exception):
                        failed += 1
                        self._log_error(
                            f"Dispatch to {peer_node_id} request {idx}: "
                            f"{type(r).__name__}: {r}"
                        )
                    elif r is True:
                        started += 1
                    elif r is False:
                        failed += 1
                        self._log_error(
                            f"Dispatch to {peer_node_id} request {idx}: rejected (HTTP error)"
                        )
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
