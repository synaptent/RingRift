"""SelfplayScheduler: Priority-based selfplay configuration selection.

Extracted from p2p_orchestrator.py for better modularity.
Handles weighted config selection, job targeting, diversity tracking, and Elo-based priority.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

logger = logging.getLogger(__name__)


# Import constants from canonical source to avoid duplication
try:
    from scripts.p2p.constants import (
        DISK_WARNING_THRESHOLD,
        MEMORY_WARNING_THRESHOLD,
        MIN_MEMORY_GB_FOR_TASKS,
    )
    from app.p2p.constants import (
        CPU_ONLY_JOB_MIN_CPUS,
        EXPLORATION_BOOST_DEFAULT_DURATION,
        PLATEAU_CLEAR_WIN_RATE,
        PLATEAU_PENALTY_DEFAULT_DURATION,
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
    MIN_MEMORY_GB_FOR_TASKS = 64  # Must match app/p2p/constants.py:84
    DISK_WARNING_THRESHOLD = 65  # Conservative: match constants.py
    MEMORY_WARNING_THRESHOLD = 75  # Conservative: match constants.py
    # Selfplay scheduler constants - match app/p2p/constants.py
    CPU_ONLY_JOB_MIN_CPUS = 128
    EXPLORATION_BOOST_DEFAULT_DURATION = 900  # 15 minutes
    PLATEAU_CLEAR_WIN_RATE = 0.50
    PLATEAU_PENALTY_DEFAULT_DURATION = 1800  # 30 minutes
    TRAINING_BOOST_DURATION = 1800  # 30 minutes
    # Promotion penalty constants - match app/p2p/constants.py
    PROMOTION_PENALTY_DURATION_CRITICAL = 7200  # 2 hours
    PROMOTION_PENALTY_DURATION_MULTIPLE = 3600  # 1 hour
    PROMOTION_PENALTY_DURATION_SINGLE = 1800  # 30 min
    PROMOTION_PENALTY_FACTOR_CRITICAL = 0.3
    PROMOTION_PENALTY_FACTOR_MULTIPLE = 0.5
    PROMOTION_PENALTY_FACTOR_SINGLE = 0.7

# Session 17.22: Architecture selection feedback loop
# Import architecture tracker functions for per-(config, architecture) performance tracking
# and intelligent architecture selection based on Elo performance
try:
    from app.training.architecture_tracker import (
        record_evaluation as _record_architecture_eval,
        get_allocation_weights as _get_architecture_weights,
    )
except ImportError:
    _record_architecture_eval = None  # type: ignore
    _get_architecture_weights = None  # type: ignore

# Memory-aware job allocation constants (P1 - Sprint 6, Jan 2026)
# Job-type specific memory requirements in GB
JOB_MEMORY_REQUIREMENTS: dict[str, float] = {
    "gpu_gumbel": 8.0,  # High-quality Gumbel MCTS on GPU
    "gpu_policy": 6.0,  # Policy-only inference on GPU
    "cpu_heuristic": 2.0,  # CPU heuristic selfplay
    "cpu_gumbel": 4.0,  # CPU Gumbel MCTS
    "training": 16.0,  # Training job (needs extra headroom)
    "evaluation": 4.0,  # Evaluation/gauntlet job
    "default": 4.0,  # Default for unknown job types
}
# System reserved memory (OS, drivers, etc.)
SYSTEM_RESERVED_MEMORY_GB = 4.0
# Minimum free memory to maintain after job allocation
MIN_FREE_MEMORY_GB = 2.0


@dataclass
class DiversityMetrics:
    """Diversity tracking metrics for selfplay scheduling.

    January 2026 Sprint 10: Added opponent_types_by_config tracking for
    diversity maximization. Tracks which opponent types each config has
    played against to ensure variety in training data.
    """

    games_by_engine_mode: dict[str, int] = field(default_factory=dict)
    games_by_board_config: dict[str, int] = field(default_factory=dict)
    games_by_difficulty: dict[str, int] = field(default_factory=dict)
    asymmetric_games: int = 0
    symmetric_games: int = 0
    last_reset: float = field(default_factory=time.time)
    # January 2026 Sprint 10: Track opponent types per config for diversity
    # Key: config_key, Value: set of opponent types (harness names)
    opponent_types_by_config: dict[str, set[str]] = field(default_factory=dict)
    # Track total games per opponent type (cluster-wide)
    games_by_opponent_type: dict[str, int] = field(default_factory=dict)

    def get_diversity_score(self, config_key: str) -> float:
        """Get diversity score for a config (0-1, higher = more diverse).

        January 2026 Sprint 10: Computes diversity based on opponent variety.
        A config that has played against many different opponent types gets
        a higher score.
        """
        # Total possible opponent types (from AI_HARNESS_CONFIGS)
        # Approximate based on typical harness count
        MAX_OPPONENT_TYPES = 8  # random, heuristic, gumbel, minimax, maxn, brs, policy, descent

        opponents_seen = self.opponent_types_by_config.get(config_key, set())
        num_seen = len(opponents_seen)

        if num_seen == 0:
            return 0.0  # No opponents seen = needs diversity
        if num_seen >= MAX_OPPONENT_TYPES:
            return 1.0  # All opponent types seen

        return min(1.0, num_seen / MAX_OPPONENT_TYPES)

    def record_opponent(self, config_key: str, opponent_type: str) -> None:
        """Record that a config played against an opponent type.

        January 2026 Sprint 10: Tracks opponent variety for diversity scoring.
        """
        if config_key not in self.opponent_types_by_config:
            self.opponent_types_by_config[config_key] = set()
        self.opponent_types_by_config[config_key].add(opponent_type)

        # Also track total games by opponent type
        if opponent_type not in self.games_by_opponent_type:
            self.games_by_opponent_type[opponent_type] = 0
        self.games_by_opponent_type[opponent_type] += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed metrics."""
        total_games = self.asymmetric_games + self.symmetric_games
        asymmetric_ratio = (
            self.asymmetric_games / total_games if total_games > 0 else 0.0
        )

        engine_total = sum(self.games_by_engine_mode.values())
        engine_mode_distribution = (
            {k: v / engine_total for k, v in self.games_by_engine_mode.items()}
            if engine_total > 0
            else {}
        )

        # January 2026 Sprint 10: Include diversity scores per config
        diversity_scores = {
            config: self.get_diversity_score(config)
            for config in self.opponent_types_by_config.keys()
        }

        return {
            "games_by_engine_mode": dict(self.games_by_engine_mode),
            "games_by_board_config": dict(self.games_by_board_config),
            "games_by_difficulty": dict(self.games_by_difficulty),
            "asymmetric_games": self.asymmetric_games,
            "symmetric_games": self.symmetric_games,
            "asymmetric_ratio": asymmetric_ratio,
            "engine_mode_distribution": engine_mode_distribution,
            "uptime_seconds": time.time() - self.last_reset,
            "games_by_opponent_type": dict(self.games_by_opponent_type),
            "diversity_scores": diversity_scores,
        }


class SelfplayScheduler(EventSubscriptionMixin):
    """Manages selfplay configuration selection and job targeting.

    Responsibilities:
    - Weighted config selection based on static priority, Elo performance, curriculum
    - Job targeting per node based on hardware capabilities and utilization
    - Diversity tracking for monitoring
    - Integration with backpressure and resource optimization

    Inherits from EventSubscriptionMixin for standardized event handling (Dec 2025).

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

    # GPU-required engine modes (require CUDA or MPS) - December 2025
    # These modes use neural network inference and require GPU acceleration
    GPU_REQUIRED_ENGINE_MODES = {
        "gumbel-mcts", "mcts", "mcts-only", "nnue-guided", "policy-only",
        "nn-minimax", "nn-descent", "gnn", "hybrid",
        "gmo", "ebmo", "ig-gmo", "cage",
    }

    # CPU-compatible engine modes (can run on any node)
    CPU_COMPATIBLE_ENGINE_MODES = {
        "heuristic-only", "heuristic", "random", "random-only",
        "descent-only", "maxn", "brs", "mixed", "diverse",
        "tournament-varied", "heuristic-vs-mcts", "cross-ai",
    }

    # December 2025: Large board engine mix for square19 and hexagonal
    # Uses a weighted mix of engines optimized for large board selfplay:
    # - heuristic: Fast bootstrap, lowest quality but very fast (⚡⚡⚡⚡⚡, ⭐⭐)
    # - brs: Best Reply Search, good for 3-4 player (⚡⚡⚡, ⭐⭐⭐⭐)
    # - maxn: MaxN search, highest heuristic quality (⚡⚡, ⭐⭐⭐⭐)
    # - policy-only: Neural-guided, needs model (⚡⚡⚡⚡, ⭐⭐⭐) - GPU required
    # - gumbel-mcts: Balanced neural search with budget 64 (⚡⚡, ⭐⭐⭐⭐) - GPU required
    # Dec 31, 2025: Minimum 15% Gumbel MCTS for quality training data (48h autonomous operation)
    # Dec 31, 2025: Added MINIMAX (paranoid) to all engine mixes
    # Paranoid minimax assumes all opponents ally against current player - works for 2-4 players
    LARGE_BOARD_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: GPU Gumbel MCTS prioritized for high-quality training data
        ("random", 3, False, None),  # 3% - baseline diversity (vs pure random)
        ("heuristic", 10, False, None),  # 10% - fast bootstrap (reduced)
        ("minimax", 5, False, {"depth": 3}),  # 5% - paranoid search (works for 2-4p)
        ("brs", 7, False, None),  # 7% - good for multiplayer
        ("maxn", 5, False, None),  # 5% - highest heuristic quality
        ("nn-descent", 5, True, None),  # 5% - exploration via neural descent (GPU)
        ("policy-only", 15, True, None),  # 15% - neural guided (GPU)
        ("gumbel-mcts", 50, True, {"budget": 150}),  # 50% - HIGH QUALITY neural (GPU) - primary mode
    ]

    # CPU-only variant for nodes without GPU
    LARGE_BOARD_ENGINE_MIX_CPU = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: Full harness diversity for comprehensive training data
        ("random", 5, False, None),  # 5% - baseline diversity (vs pure random)
        ("heuristic", 35, False, None),  # 35% - fast bootstrap
        ("minimax", 15, False, {"depth": 3}),  # 15% - paranoid search (2-4p)
        ("brs", 25, False, None),  # 25% - good for multiplayer
        ("maxn", 20, False, None),  # 20% - highest heuristic quality
    ]

    # December 2025: Standard board engine mix for smaller boards (hex8, square8)
    # Higher neural network weight since games are faster on smaller boards
    # Jan 2026: GPU Gumbel MCTS heavily prioritized for high-quality training data
    STANDARD_BOARD_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: GPU Gumbel MCTS prioritized for high-quality training data
        ("random", 2, False, None),  # 2% - baseline diversity (vs pure random)
        ("heuristic", 8, False, None),  # 8% - fast bootstrap (reduced)
        ("minimax", 5, False, {"depth": 4}),  # 5% - paranoid search (2-4p, deeper on small boards)
        ("brs", 5, False, None),  # 5% - good for multiplayer diversity
        ("maxn", 5, False, None),  # 5% - highest heuristic quality
        ("nn-descent", 5, True, None),  # 5% - exploration via neural descent (GPU)
        ("policy-only", 10, True, None),  # 10% - neural guided (GPU)
        ("gumbel-mcts", 60, True, {"budget": 200}),  # 60% - HIGH QUALITY neural (GPU) - primary mode
    ]

    # CPU-only variant for standard boards
    STANDARD_BOARD_ENGINE_MIX_CPU = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: Full harness diversity for comprehensive training data
        ("random", 5, False, None),  # 5% - baseline diversity (vs pure random)
        ("heuristic", 25, False, None),  # 25% - fast bootstrap
        ("minimax", 25, False, {"depth": 4}),  # 25% - paranoid search (2-4p)
        ("brs", 25, False, None),  # 25% - good for multiplayer
        ("maxn", 20, False, None),  # 20% - highest heuristic quality
    ]

    # Large board types that should use the large board engine mix
    LARGE_BOARD_TYPES = {"square19", "hexagonal"}
    # Standard board types that should use the standard board engine mix
    STANDARD_BOARD_TYPES = {"hex8", "square8"}

    @classmethod
    def _select_board_engine(
        cls,
        has_gpu: bool,
        board_type: str,
        num_players: int = 0,  # Kept for API compatibility, not used
    ) -> tuple[str, dict[str, Any] | None]:
        """Select an engine mode from the appropriate engine mix for the board type.

        Uses weighted random selection from the engine mix matching the board type:
        - Large boards (square19, hexagonal): LARGE_BOARD_ENGINE_MIX
        - Standard boards (hex8, square8): STANDARD_BOARD_ENGINE_MIX

        All mixes include MINIMAX (paranoid search) which works for 2-4 players.
        Paranoid minimax assumes all opponents ally against the current player.

        GPU vs CPU variants are selected based on node capability.

        Args:
            has_gpu: Whether the node has GPU capability
            board_type: Board type to select engine for
            num_players: Kept for API compatibility (MINIMAX works for all player counts)

        Returns:
            Tuple of (engine_mode, extra_args) where extra_args may contain
            additional parameters like {"budget": 64} for gumbel-mcts.

        December 2025: Extended to support mixed-engine strategy for ALL board types,
        not just large boards. BRS and MaxN now available for hex8/square8 diversity.
        December 31, 2025: Added MINIMAX (paranoid) to all mixes - works for 2-4 players.
        """
        # Select appropriate engine mix based on board type and GPU availability
        if board_type in cls.LARGE_BOARD_TYPES:
            engine_mix = cls.LARGE_BOARD_ENGINE_MIX if has_gpu else cls.LARGE_BOARD_ENGINE_MIX_CPU
            board_category = "large"
        else:
            # Use standard board mix for all other boards (hex8, square8, etc.)
            engine_mix = cls.STANDARD_BOARD_ENGINE_MIX if has_gpu else cls.STANDARD_BOARD_ENGINE_MIX_CPU
            board_category = "standard"

        # Build weighted selection list
        weighted_engines: list[tuple[str, dict[str, Any] | None]] = []
        for engine_mode, weight, gpu_required, extra_args in engine_mix:
            # Skip GPU-required engines on CPU-only nodes
            if gpu_required and not has_gpu:
                continue
            # Add engine with its weight
            weighted_engines.extend([(engine_mode, extra_args)] * weight)

        if not weighted_engines:
            # Fallback to heuristic if no engines available
            logger.warning(
                f"No compatible engines for {board_category} board (gpu={has_gpu}), "
                f"falling back to heuristic"
            )
            return ("heuristic", None)

        # Random weighted selection
        selected = random.choice(weighted_engines)
        engine_mode, extra_args = selected

        logger.debug(
            f"Selected engine '{engine_mode}' for {board_category} board "
            f"(board={board_type}, gpu={has_gpu}, extra_args={extra_args})"
        )

        return (engine_mode, extra_args)

    @classmethod
    def _select_large_board_engine(
        cls,
        has_gpu: bool,
        board_type: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Legacy wrapper - use _select_board_engine() instead.

        Kept for backward compatibility.
        """
        return cls._select_board_engine(has_gpu, board_type or "square19")

    def _select_architecture_for_config(
        self,
        board_type: str,
        num_players: int,
    ) -> str:
        """Select architecture version based on Elo performance weights.

        Session 17.22: This closes the architecture selection feedback loop.
        The ArchitectureTracker records per-(config, architecture) Elo ratings,
        and compute_allocation_weights() returns softmax weights biased toward
        better-performing architectures.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Architecture version string (e.g., "v5", "v2", "v5-heavy")
        """
        # Default architecture if tracker unavailable or no weights
        default_arch = "v5"

        if _get_architecture_weights is None:
            return default_arch

        try:
            # Get Elo-based allocation weights from ArchitectureTracker
            weights = _get_architecture_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=0.5,  # Moderate exploration vs exploitation
            )

            if not weights:
                # Cold start: no evaluation data yet
                logger.debug(
                    f"No architecture weights for {board_type}_{num_players}p, "
                    f"using default: {default_arch}"
                )
                return default_arch

            # Weighted random selection based on Elo performance
            architectures = list(weights.keys())
            arch_weights = list(weights.values())

            # Use random.choices for weighted selection
            selected_arch = random.choices(architectures, weights=arch_weights, k=1)[0]

            logger.info(
                f"Architecture selection for {board_type}_{num_players}p: "
                f"{selected_arch} (weights: {weights})"
            )

            return selected_arch

        except (KeyError, ValueError, TypeError) as e:
            logger.debug(
                f"Error selecting architecture for {board_type}_{num_players}p: {e}, "
                f"using default: {default_arch}"
            )
            return default_arch

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

    def _pick_simple_weighted_config(self) -> str | None:
        """Pick a selfplay config key using simple priority-weighted selection.

        January 2026 Sprint 6: Simplified version of pick_weighted_config for
        use in preemptive job spawning where we don't have node info.

        Returns:
            Config key (e.g., "hex8_2p") or None if no valid config
        """
        # Standard configs with priority weights
        STANDARD_CONFIGS = [
            # High priority: Underserved/complex configs
            ("hexagonal_3p", 8),
            ("hexagonal_4p", 8),
            ("square19_3p", 7),
            ("square19_4p", 7),
            ("hex8_3p", 6),
            ("hex8_4p", 6),
            # Medium priority: Standard 2-player
            ("hex8_2p", 5),
            ("square8_2p", 5),
            ("hexagonal_2p", 5),
            # Lower priority: Well-covered configs
            ("square8_3p", 4),
            ("square8_4p", 4),
            ("square19_2p", 4),
        ]

        # Apply dynamic boosts from curriculum weights
        curriculum_weights = {}
        try:
            curriculum_weights = self.load_curriculum_weights()
        except Exception:
            pass

        weighted_configs = []
        for config_key, base_priority in STANDARD_CONFIGS:
            # Apply curriculum weight boost
            curriculum_mult = curriculum_weights.get(config_key, 1.0)

            # Apply staleness boost for underserved configs
            staleness_boost = self._get_staleness_boost(config_key)

            effective_priority = int(base_priority * curriculum_mult + staleness_boost)
            effective_priority = max(1, effective_priority)

            weighted_configs.extend([config_key] * effective_priority)

        if not weighted_configs:
            return None

        return random.choice(weighted_configs)

    def _get_staleness_boost(self, config_key: str) -> int:
        """Get staleness-based priority boost for a config.

        Configs that haven't had recent games get higher priority.
        """
        try:
            # Initialize tracking dict if needed
            if not hasattr(self, "_last_config_job_time"):
                self._last_config_job_time = {}

            # Check last job time for this config
            last_job_time = self._last_config_job_time.get(config_key, 0)
            if last_job_time == 0:
                return 3  # Never had a job, high boost

            age_hours = (time.time() - last_job_time) / 3600
            if age_hours > 24:
                return 3
            elif age_hours > 12:
                return 2
            elif age_hours > 6:
                return 1
            return 0
        except Exception:
            return 0

    def record_job_dispatched(self, config_key: str) -> None:
        """Record that a job was dispatched for a config.

        January 2026 Sprint 6: Used for staleness-based priority boosting.
        Configs that haven't had recent jobs get higher priority.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
        """
        if not hasattr(self, "_last_config_job_time"):
            self._last_config_job_time = {}
        self._last_config_job_time[config_key] = time.time()

    # =========================================================================
    # Minimum Allocation Enforcement (January 2026 Sprint 6)
    # =========================================================================

    # Minimum allocation constants
    MINIMUM_ALLOCATION_PERCENT = 0.20  # Reserve 20% of jobs for underserved
    UNDERSERVED_THRESHOLD = 5000  # Configs below this game count are "underserved"
    CRITICAL_THRESHOLD = 1000  # Configs below this get highest priority
    MINIMUM_ENFORCE_INTERVAL = 30.0  # Seconds between enforcement checks

    def _get_enforced_minimum_allocation(self) -> str | None:
        """Check if we should force allocation to an underserved config.

        January 2026 Sprint 6: Implements minimum allocation enforcement.
        Reserves 20% of cluster capacity for underserved configs to guarantee
        they receive games even when higher-priority configs dominate.

        Returns:
            Config key if enforcement is active, None otherwise
        """
        try:
            now = time.time()

            # Rate limit enforcement checks
            if hasattr(self, "_last_enforcement_check"):
                if now - self._last_enforcement_check < self.MINIMUM_ENFORCE_INTERVAL:
                    return None
            self._last_enforcement_check = now

            # Random 20% chance to enforce (simulates 20% allocation)
            if random.random() > self.MINIMUM_ALLOCATION_PERCENT:
                return None

            # Find underserved configs
            underserved = self._get_underserved_configs()
            if not underserved:
                return None

            # Sort by game count (lowest first = most critical)
            underserved.sort(key=lambda x: x[1])

            # Pick the most critical config (lowest game count)
            selected_config, game_count = underserved[0]

            logger.info(
                f"[MinimumAllocation] Enforcing allocation to {selected_config} "
                f"(only {game_count} games, threshold={self.UNDERSERVED_THRESHOLD})"
            )

            return selected_config

        except Exception as e:
            logger.debug(f"[MinimumAllocation] Enforcement check failed: {e}")
            return None

    def _get_underserved_configs(self) -> list[tuple[str, int]]:
        """Get list of configs below the underserved threshold.

        Returns:
            List of (config_key, game_count) tuples for underserved configs
        """
        try:
            # Get game counts from tracking or discovery
            game_counts = self._get_game_counts_per_config()

            underserved = []
            all_configs = [
                "hex8_2p", "hex8_3p", "hex8_4p",
                "square8_2p", "square8_3p", "square8_4p",
                "square19_2p", "square19_3p", "square19_4p",
                "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
            ]

            for config_key in all_configs:
                count = game_counts.get(config_key, 0)
                if count < self.UNDERSERVED_THRESHOLD:
                    underserved.append((config_key, count))

            return underserved

        except Exception as e:
            logger.debug(f"[MinimumAllocation] Failed to get underserved configs: {e}")
            return []

    def _get_game_counts_per_config(self) -> dict[str, int]:
        """Get current game counts for each config.

        Uses cached data from P2P manifest or local database discovery.

        Returns:
            Dict mapping config_key -> game_count
        """
        try:
            # First try to get counts from P2P manifest (cluster-wide view)
            if hasattr(self, "_p2p_game_counts") and self._p2p_game_counts:
                return dict(self._p2p_game_counts)

            # Fall back to local tracking
            if hasattr(self, "_config_game_counts"):
                return dict(self._config_game_counts)

            # Return empty dict if no data available
            return {}

        except Exception:
            return {}

    def update_config_game_count(self, config_key: str, game_count: int) -> None:
        """Update the tracked game count for a config.

        Called when game data is synced or generated.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            game_count: Current total game count
        """
        if not hasattr(self, "_config_game_counts"):
            self._config_game_counts = {}
        self._config_game_counts[config_key] = game_count

    def update_p2p_game_counts(self, counts: dict[str, int]) -> None:
        """Update game counts from P2P manifest data.

        Called by P2P orchestrator when manifest data is refreshed.

        Args:
            counts: Dict mapping config_key -> game_count
        """
        self._p2p_game_counts = dict(counts)

    # =========================================================================
    # Memory-Aware Job Allocation (P1 - Sprint 6, Jan 2026)
    # =========================================================================

    def _get_job_memory_requirement(self, job_type: str) -> float:
        """Get memory requirement in GB for a specific job type.

        Args:
            job_type: Type of job (gpu_gumbel, cpu_heuristic, training, etc.)

        Returns:
            Memory requirement in GB for this job type.
        """
        return JOB_MEMORY_REQUIREMENTS.get(
            job_type, JOB_MEMORY_REQUIREMENTS["default"]
        )

    def _check_memory_available(
        self, node: Any, job_type: str = "gpu_gumbel"
    ) -> bool:
        """Check if a node has enough memory for a specific job type.

        This method checks CURRENT memory usage, not just total memory.
        It accounts for:
        - Job-type specific memory requirements
        - System reserved memory (OS, drivers)
        - Minimum free memory buffer

        Args:
            node: Node info object with memory_gb and memory_percent attributes.
            job_type: Type of job being considered.

        Returns:
            True if node has sufficient memory for this job type.
        """
        try:
            # Get total and current memory usage
            total_memory_gb = float(getattr(node, "memory_gb", 0) or 0)
            memory_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)

            if total_memory_gb <= 0:
                # No memory info available - fall back to basic check
                return True

            # Calculate current memory usage
            current_usage_gb = (memory_percent / 100.0) * total_memory_gb

            # Calculate available memory after accounting for:
            # 1. Current usage
            # 2. System reserved memory
            # 3. Minimum free memory buffer
            available_gb = (
                total_memory_gb
                - current_usage_gb
                - SYSTEM_RESERVED_MEMORY_GB
                - MIN_FREE_MEMORY_GB
            )

            # Get job-specific memory requirement
            job_memory_gb = self._get_job_memory_requirement(job_type)

            # Check if we have enough available memory
            has_enough = available_gb >= job_memory_gb

            if not has_enough and self.verbose:
                node_id = getattr(node, "node_id", "unknown")
                logger.debug(
                    f"Memory check failed for {node_id}: "
                    f"available={available_gb:.1f}GB, "
                    f"needed={job_memory_gb:.1f}GB "
                    f"(total={total_memory_gb:.1f}GB, "
                    f"used={memory_percent:.1f}%)"
                )

            return has_enough

        except (TypeError, ValueError, AttributeError) as e:
            # On any error, fall back to allowing the job
            logger.debug(f"Memory check error: {e}")
            return True

    def _get_recommended_job_type(self, node: Any) -> str:
        """Get the recommended job type based on node capabilities and memory.

        Considers:
        - GPU availability
        - Current memory usage
        - GPU memory usage

        Args:
            node: Node info object with hardware attributes.

        Returns:
            Recommended job type string.
        """
        has_gpu = bool(getattr(node, "has_gpu", False))
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)

        if has_gpu:
            # GPU node - check GPU memory for job type
            if gpu_mem_percent < 50:
                # Plenty of GPU memory - can run high-quality Gumbel
                return "gpu_gumbel"
            elif gpu_mem_percent < 80:
                # Moderate GPU memory - use policy-only
                return "gpu_policy"
            else:
                # GPU memory constrained - fall back to CPU
                return "cpu_heuristic"
        else:
            # CPU-only node
            if self._check_memory_available(node, "cpu_gumbel"):
                return "cpu_gumbel"
            else:
                return "cpu_heuristic"

    def get_elo_based_priority_boost(self, board_type: str, num_players: int) -> int:
        """Get priority boost based on ELO performance for this config.

        PRIORITY-BASED SCHEDULING: Configs with high-performing models get
        priority boost to allocate more resources to promising configurations.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)

        Returns:
            Priority boost (0-5) based on:
            - Top model ELO for this config
            - Recent improvement rate
            - Data coverage (inverse - underrepresented get boost)
        """
        boost = 0

        try:
            cluster_elo = self.get_cluster_elo()
            top_models = cluster_elo.get("top_models", [])

            # Find best model for this board/player combo
            best_elo = 0
            for model in top_models:
                model_name = model.get("name", "")
                # Model names typically include board type and player count
                if board_type in model_name or str(num_players) in model_name:
                    best_elo = max(best_elo, model.get("elo", 0))

            # ELO-based boost (every 100 ELO above 1200 = +1 priority)
            if best_elo > 1200:
                boost += min(3, (best_elo - 1200) // 100)

            # Underrepresented config boost
            # (hex and square19 often have fewer games)
            if board_type in ("hexagonal", "square19"):
                boost += 1
            if num_players > 2:
                boost += 1

        except AttributeError:
            pass

        return min(5, boost)  # Cap at +5

    def _get_data_starvation_boost(self, config_key: str) -> float:
        """Get priority boost for configs that are data-starved for training.

        December 2025 Phase 2: Boost selfplay allocation for configs that have
        few recent games, ensuring training data availability across all configs.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Boost multiplier:
            - 0.5: Config is actively training (reduce selfplay)
            - 1.0: Normal priority (no boost)
            - 1.5: Moderate data starvation (<50 games since training)
            - 2.0: High data starvation (<25 games since training)
        """
        # If config is currently in training pipeline, reduce selfplay priority
        # to avoid wasting resources on a config that's actively learning
        if config_key in self._configs_in_training_pipeline:
            return 0.5

        # Check recent game count from feedback state
        state = self._feedback_states.get(config_key)
        if state is not None:
            # Try to get games_since_last_training attribute
            games = getattr(state, "games_since_last_training", None)
            if games is not None:
                if games < 25:
                    # Very data-starved - high boost to generate more data
                    return 2.0
                elif games < 50:
                    # Moderately data-starved
                    return 1.5

        # Also check if there's a training complete boost active
        # (recently trained configs need more data for next cycle)
        if config_key in self._training_complete_boosts:
            expiry = self._training_complete_boosts[config_key]
            if time.time() < expiry:
                return 1.3  # Slight boost after training completes

        return 1.0  # No boost

    # =========================================================================
    # GPU Capability Helpers (December 2025)
    # =========================================================================

    def _engine_mode_requires_gpu(self, engine_mode: str) -> bool:
        """Check if an engine mode requires GPU acceleration.

        Args:
            engine_mode: The engine mode string (e.g., "gumbel-mcts", "heuristic-only")

        Returns:
            True if the engine mode requires GPU (CUDA or MPS), False otherwise.

        December 2025: Added to ensure GPU-required selfplay is only assigned
        to GPU-capable nodes, preventing wasted compute.
        """
        if not engine_mode:
            return False
        mode_lower = engine_mode.lower().strip()
        return mode_lower in self.GPU_REQUIRED_ENGINE_MODES

    def _node_has_gpu(self, node: "NodeInfo") -> bool:
        """Check if a node has GPU capability.

        Args:
            node: Node information object

        Returns:
            True if the node has GPU (CUDA-capable), False otherwise.

        December 2025: Added for GPU-aware config selection.
        """
        # Try has_cuda_gpu() method first (NodeInfo from scripts/p2p/models.py)
        if hasattr(node, "has_cuda_gpu"):
            return node.has_cuda_gpu()

        # Try is_gpu_node() method
        if hasattr(node, "is_gpu_node"):
            return node.is_gpu_node()

        # Try has_gpu attribute
        if hasattr(node, "has_gpu"):
            return bool(node.has_gpu)

        # Try gpu_info attribute directly
        if hasattr(node, "gpu_info"):
            gpu_info = node.gpu_info
            if gpu_info is not None:
                gpu_count = getattr(gpu_info, "gpu_count", 0)
                return gpu_count > 0

        # Fallback: assume no GPU if we can't determine
        return False

    def _filter_configs_by_gpu(
        self,
        configs: list[dict[str, Any]],
        node: "NodeInfo",
    ) -> list[dict[str, Any]]:
        """Filter configs to only those compatible with node's GPU capability.

        For CPU-only nodes, removes configs that require GPU (e.g., gumbel-mcts).
        GPU nodes can run all configs.

        Args:
            configs: List of selfplay config dicts
            node: Node information object

        Returns:
            Filtered list of configs compatible with the node.

        December 2025: Core GPU-aware filtering for config selection.
        """
        if self._node_has_gpu(node):
            # GPU nodes can run any config
            return configs

        # CPU-only nodes: filter out GPU-required configs
        cpu_compatible = [
            cfg
            for cfg in configs
            if not self._engine_mode_requires_gpu(cfg.get("engine_mode", ""))
        ]

        if len(cpu_compatible) < len(configs):
            filtered_count = len(configs) - len(cpu_compatible)
            node_id = getattr(node, "node_id", "unknown")
            logger.info(
                f"Filtered {filtered_count} GPU-required configs for CPU-only node {node_id}"
            )

        return cpu_compatible

    # =========================================================================
    # Event Subscriptions (December 2025)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for EventSubscriptionMixin.

        Dec 28, 2025: Migrated to use EventSubscriptionMixin pattern.
        Dec 29, 2025 (Phase 2B): Added pipeline coordination events:
        - NPZ_EXPORT_COMPLETE: Temporarily boost priority for freshly exported config
        - TRAINING_STARTED: Mark config as in-training, reduce selfplay allocation
        - EVALUATION_COMPLETED: Update curriculum weights based on gauntlet results

        Returns:
            Dict mapping event names to handler methods
        """
        return {
            "SELFPLAY_RATE_CHANGED": self._on_selfplay_rate_changed,
            "EXPLORATION_BOOST": self._on_exploration_boost,
            "TRAINING_COMPLETED": self._on_training_completed,
            "ELO_VELOCITY_CHANGED": self._on_elo_velocity_changed,
            # December 2025 - Phase 2B: Pipeline coordination events
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
            "TRAINING_STARTED": self._on_training_started,
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            # December 2025 - Phase 4D: Plateau detection for resource balancing
            "PLATEAU_DETECTED": self._on_plateau_detected,
            # December 2025 - Phase 5: Evaluation backpressure handling
            "EVALUATION_BACKPRESSURE": self._on_evaluation_backpressure,
            "EVALUATION_BACKPRESSURE_RELEASED": self._on_evaluation_backpressure_released,
            # January 2026 - Sprint 10: Quality-blocked training feedback
            "TRAINING_BLOCKED_BY_QUALITY": self._on_training_blocked_by_quality,
            # January 2026 - Sprint 10: Hyperparameter updates affect selfplay quality
            "HYPERPARAMETER_UPDATED": self._on_hyperparameter_updated,
            # January 2026 - Sprint 13: Peer reconnection triggers work rebalancing
            "PEER_RECONNECTED": self._on_peer_reconnected,
            # January 2026 - Session 17.22: Immediate quality score application
            "QUALITY_SCORE_UPDATED": self._on_quality_score_updated,
        }

    async def _on_selfplay_rate_changed(self, event) -> None:
        """Handle SELFPLAY_RATE_CHANGED events from feedback loop.

        Adjusts rate multipliers for configs based on Elo velocity and performance.

        Args:
            event: Event with payload containing config_key, new_rate, reason
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        new_rate = payload.get("new_rate", 1.0)
        reason = payload.get("reason", "unknown")

        if not config_key:
            return

        old_rate = self._rate_multipliers.get(config_key, 1.0)
        self._rate_multipliers[config_key] = new_rate

        if abs(new_rate - old_rate) > 0.01:
            self._log_info(
                f"Rate changed for {config_key}: {old_rate:.2f} -> {new_rate:.2f} ({reason})"
            )

    def get_rate_multiplier(self, config_key: str) -> float:
        """Get current rate multiplier for a config.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Rate multiplier (1.0 = normal, >1 = boost, <1 = throttle)
        """
        return self._rate_multipliers.get(config_key, 1.0)

    async def _on_exploration_boost(self, event) -> None:
        """Handle EXPLORATION_BOOST events from FeedbackLoopController.

        Dec 2025: React to training anomalies (loss spikes, stalls) by increasing
        exploration to generate more diverse training data.

        Args:
            event: Event with payload containing config_key, boost_factor, reason
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        boost_factor = payload.get("boost_factor", 1.3)
        duration = payload.get("duration_seconds", EXPLORATION_BOOST_DEFAULT_DURATION)
        reason = payload.get("reason", "training_anomaly")

        if not config_key:
            return

        # Delegate to existing set_exploration_boost method
        self.set_exploration_boost(config_key, boost_factor, duration)

        self._log_info(
            f"Exploration boost from event: {config_key} "
            f"{boost_factor:.2f}x for {duration}s ({reason})"
        )

    async def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED events to boost selfplay after training.

        Dec 2025: When training completes, boost selfplay for that config to
        generate more data for the next training cycle.

        Args:
            event: Event with payload containing config_key
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")

        if not config_key:
            return

        # Delegate to existing on_training_complete method
        self.on_training_complete(config_key)

    async def _on_elo_velocity_changed(self, event) -> None:
        """Handle ELO_VELOCITY_CHANGED events for selfplay rate adjustment.

        Dec 2025: Adjusts selfplay rate based on Elo velocity trends.
        - accelerating: Increase selfplay to capitalize on momentum
        - decelerating: Reduce selfplay, shift focus to training quality
        - stable: Maintain current allocation

        Args:
            event: Event with payload containing config_key, velocity, trend
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        velocity = payload.get("velocity", 0.0)
        trend = payload.get("trend", "stable")

        if not config_key:
            return

        old_rate = self._rate_multipliers.get(config_key, 1.0)

        # Adjust rate based on velocity trend
        if trend == "accelerating":
            # Capitalize on positive momentum - increase selfplay rate
            new_rate = min(1.5, old_rate * 1.2)
        elif trend == "decelerating":
            # Slow down and focus on quality
            new_rate = max(0.6, old_rate * 0.85)
        else:  # stable
            # Slight adjustment toward 1.0
            if old_rate > 1.0:
                new_rate = max(1.0, old_rate * 0.95)
            elif old_rate < 1.0:
                new_rate = min(1.0, old_rate * 1.05)
            else:
                new_rate = 1.0

        self._rate_multipliers[config_key] = new_rate

        if abs(new_rate - old_rate) > 0.01:
            self._log_info(
                f"Elo velocity {trend} for {config_key}: "
                f"velocity={velocity:.1f}, rate {old_rate:.2f} -> {new_rate:.2f}"
            )
            # P0.2 Dec 2025: Emit rate change event for significant changes
            self._emit_selfplay_rate_changed(
                config_key, old_rate, new_rate, f"elo_velocity_{trend}"
            )

    # =========================================================================
    # Phase 2B Event Handlers (December 2025)
    # =========================================================================

    async def _on_npz_export_complete(self, event) -> None:
        """Handle NPZ_EXPORT_COMPLETE events - boost priority for freshly exported config.

        Dec 2025 Phase 2B: When NPZ export completes, temporarily boost the config's
        priority to generate more training data quickly while the exported data
        is being used for training.

        This ensures selfplay doesn't pile on extra games for a config that's about
        to enter training, but still keeps generating some data for the next cycle.

        Args:
            event: Event with payload containing config_key, samples, output_path
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")
        samples = payload.get("samples", 0)

        if not config_key:
            return

        # Temporarily reduce selfplay rate while training is consuming this export
        # This prevents wasting compute on a config that's actively training
        old_rate = self._rate_multipliers.get(config_key, 1.0)

        # Reduce to 70% of normal rate during training phase
        new_rate = max(0.5, old_rate * 0.7)
        self._rate_multipliers[config_key] = new_rate

        self._log_info(
            f"NPZ export complete for {config_key}: {samples} samples, "
            f"reducing selfplay rate {old_rate:.2f} -> {new_rate:.2f}"
        )

        # Track that this config is in "export->training" transition
        self._configs_in_training_pipeline.add(config_key)

    async def _on_training_started(self, event) -> None:
        """Handle TRAINING_STARTED events - mark config as in-training.

        Dec 2025 Phase 2B: When training starts for a config, mark it as
        "in training pipeline" and reduce selfplay allocation slightly.
        This prioritizes sync and evaluation over generating excess selfplay data.

        Args:
            event: Event with payload containing config_key, epochs, batch_size
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")

        if not config_key:
            return

        # Mark config as in training
        self._configs_in_training_pipeline.add(config_key)

        # Reduce selfplay rate while training is active
        old_rate = self._rate_multipliers.get(config_key, 1.0)
        new_rate = max(0.4, old_rate * 0.6)  # 60% of current rate
        self._rate_multipliers[config_key] = new_rate

        self._log_info(
            f"Training started for {config_key}, "
            f"reducing selfplay rate {old_rate:.2f} -> {new_rate:.2f}"
        )

    async def _on_evaluation_completed(self, event) -> None:
        """Handle EVALUATION_COMPLETED events - update curriculum weights.

        Dec 2025 Phase 2B: When gauntlet evaluation completes, update curriculum
        weights based on performance results. This enables real-time curriculum
        adjustments based on model strength.

        Session 17.22: Added architecture-specific Elo tracking. Records evaluation
        results to ArchitectureTracker for per-(config, architecture) performance.

        Curriculum weight updates:
        - High win rate (>75%): Reduce weight, model is strong
        - Low win rate (<50%): Increase weight, model needs more training data
        - Mid-range: Maintain current weight

        Args:
            event: Event with payload containing config_key, win_rate, elo, architecture
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")
        win_rate = payload.get("win_rate", 0.5)
        elo = payload.get("elo", 1500.0)
        architecture = payload.get("architecture", "")
        games_played = payload.get("games_played", 0)

        if not config_key:
            return

        # Remove from training pipeline (evaluation is final step)
        self._configs_in_training_pipeline.discard(config_key)

        # Dec 2025 Phase 4D: Clear plateau penalty on successful evaluation
        # If win rate is acceptable, the config is making progress
        if win_rate >= PLATEAU_CLEAR_WIN_RATE and config_key in self._plateaued_configs:
            del self._plateaued_configs[config_key]
            self._log_info(
                f"Plateau cleared for {config_key} (win_rate={win_rate:.1%}), "
                f"restoring normal priority"
            )

        # Restore normal selfplay rate after training cycle completes
        old_rate = self._rate_multipliers.get(config_key, 1.0)
        if old_rate < 0.8:
            # Restore to normal rate
            self._rate_multipliers[config_key] = 1.0
            self._log_info(
                f"Evaluation complete for {config_key}, "
                f"restoring selfplay rate {old_rate:.2f} -> 1.00"
            )

        # Update curriculum weight based on performance
        current_weight = self._curriculum_weights.get(config_key, 1.0)

        if win_rate > 0.75:
            # Strong model - reduce curriculum weight
            new_weight = max(0.3, current_weight * 0.8)
            reason = "strong_model"
        elif win_rate < 0.50:
            # Struggling model - increase curriculum weight
            new_weight = min(2.5, current_weight * 1.3)
            reason = "struggling_model"
        else:
            # Mid-range - slight adjustment toward 1.0
            if current_weight > 1.0:
                new_weight = max(1.0, current_weight * 0.95)
            elif current_weight < 1.0:
                new_weight = min(1.0, current_weight * 1.05)
            else:
                new_weight = 1.0
            reason = "stable_model"

        if abs(new_weight - current_weight) > 0.05:
            self._curriculum_weights[config_key] = new_weight
            self._log_info(
                f"Curriculum weight updated for {config_key} "
                f"(win_rate={win_rate:.1%}, elo={elo:.0f}): "
                f"{current_weight:.2f} -> {new_weight:.2f} ({reason})"
            )

        # Session 17.22: Record evaluation to ArchitectureTracker for per-(config, arch) Elo
        # This enables intelligent architecture allocation based on performance
        if architecture and _record_architecture_eval is not None:
            try:
                # Parse config_key to get board_type and num_players
                # Format: "hex8_2p" -> board_type="hex8", num_players=2
                parts = config_key.rsplit("_", 1)
                if len(parts) == 2 and parts[1].endswith("p"):
                    board_type = parts[0]
                    num_players = int(parts[1].rstrip("p"))
                    training_hours = payload.get("training_hours", 0.0)

                    _record_architecture_eval(
                        architecture=architecture,
                        board_type=board_type,
                        num_players=num_players,
                        elo=elo,
                        training_hours=training_hours,
                        games_evaluated=games_played,
                    )
                    self._log_debug(
                        f"Recorded evaluation for {architecture}:{config_key} "
                        f"(elo={elo:.0f}, games={games_played})"
                    )
            except (ValueError, TypeError) as e:
                self._log_debug(f"Could not parse config_key {config_key}: {e}")

    async def _on_plateau_detected(self, event) -> None:
        """Handle PLATEAU_DETECTED events - reduce priority for plateaued configs.

        Dec 2025 Phase 4D: When a config is detected as plateaued (no Elo improvement
        despite training), reduce its selfplay allocation to avoid wasting resources.
        The config will still receive some games for exploration, but healthy configs
        get priority.

        Plateau penalty expires after a duration (default 30 minutes) to allow
        recovery after hyperparameter adjustments or curriculum changes.

        Args:
            event: Event with payload containing config_key, reason, duration_seconds
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "") or payload.get("config", "")
        duration_seconds = payload.get("duration_seconds", PLATEAU_PENALTY_DEFAULT_DURATION)
        reason = payload.get("reason", "elo_stagnation")

        if not config_key:
            return

        # Calculate expiration timestamp
        expiry_time = time.time() + duration_seconds
        old_expiry = self._plateaued_configs.get(config_key)

        # Update or add plateau tracking
        self._plateaued_configs[config_key] = expiry_time

        if old_expiry is None:
            self._log_info(
                f"Plateau detected for {config_key} ({reason}), "
                f"reducing selfplay priority for {duration_seconds}s"
            )
        else:
            self._log_debug(
                f"Plateau extended for {config_key}, new expiry in {duration_seconds}s"
            )

    def _is_config_plateaued(self, config_key: str) -> bool:
        """Check if a config is currently in plateau state.

        Dec 2025 Phase 4D: Returns True if the config has an active plateau penalty.
        Automatically cleans up expired entries.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            True if config is plateaued and penalty is active, False otherwise.
        """
        expiry_time = self._plateaued_configs.get(config_key)
        if expiry_time is None:
            return False

        current_time = time.time()
        if current_time >= expiry_time:
            # Plateau expired - clean up and return False
            del self._plateaued_configs[config_key]
            self._log_debug(f"Plateau expired for {config_key}, restoring priority")
            return False

        return True

    async def _on_evaluation_backpressure(self, event) -> None:
        """Handle EVALUATION_BACKPRESSURE events - pause selfplay to let evaluations catch up.

        Dec 2025 Phase 5: When evaluation queue exceeds threshold (typically 70 pending),
        the evaluation daemon emits this event. Selfplay should pause to prevent
        cascading backpressure: selfplay → training → evaluation bottleneck.

        Without this handler, selfplay continues producing games even when downstream
        pipeline is saturated, leading to training loop deadlock.

        Args:
            event: Event with payload containing queue_depth, threshold
        """
        payload = self._extract_event_payload(event)
        queue_depth = payload.get("queue_depth", 0)
        threshold = payload.get("threshold", 70)

        self._evaluation_backpressure_active = True
        self._evaluation_backpressure_start = time.time()

        self._log_info(
            f"Evaluation backpressure ACTIVATED: queue_depth={queue_depth} > threshold={threshold}, "
            "pausing selfplay allocation until evaluation queue drains"
        )

    async def _on_evaluation_backpressure_released(self, event) -> None:
        """Handle EVALUATION_BACKPRESSURE_RELEASED events - resume selfplay.

        Dec 2025 Phase 5: When evaluation queue drops below release threshold
        (typically 35), resume selfplay allocation.

        Args:
            event: Event with payload containing queue_depth, release_threshold
        """
        payload = self._extract_event_payload(event)
        queue_depth = payload.get("queue_depth", 0)
        release_threshold = payload.get("release_threshold", 35)

        if self._evaluation_backpressure_active:
            duration = time.time() - self._evaluation_backpressure_start
            self._log_info(
                f"Evaluation backpressure RELEASED: queue_depth={queue_depth} < release_threshold={release_threshold}, "
                f"resuming selfplay allocation after {duration:.1f}s pause"
            )
        self._evaluation_backpressure_active = False
        self._evaluation_backpressure_start = 0.0

    async def _on_training_blocked_by_quality(self, event) -> None:
        """Handle TRAINING_BLOCKED_BY_QUALITY events - boost high-quality selfplay.

        Jan 2026 Sprint 10: When training is blocked due to low data quality,
        increase the proportion of high-quality selfplay modes (Gumbel MCTS with
        higher budget) and reduce total game volume to focus on quality over quantity.

        This closes the quality feedback loop:
        - Quality gate blocks training → Signal to selfplay scheduler
        - Scheduler boosts quality mode percentage (e.g., 50% → 80% Gumbel MCTS)
        - Higher quality data unblocks training → Better Elo gains

        Expected improvement: +2-5 Elo per config by generating higher quality training data.

        Args:
            event: Event with payload containing config_key, quality_score, threshold
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.0)
        threshold = payload.get("threshold", 0.7)
        reason = payload.get("reason", "low_data_quality")

        if not config_key:
            return

        # Calculate quality boost factor based on how far below threshold we are
        # Quality score 0.5 with threshold 0.7 → boost 1.5x
        # Quality score 0.3 with threshold 0.7 → boost 2.0x
        quality_deficit = max(0, threshold - quality_score)
        quality_boost = 1.0 + (quality_deficit * 2.5)  # 1.0 to ~2.5x boost
        quality_boost = min(quality_boost, 2.5)  # Cap at 2.5x

        # Set quality boost for 30 minutes (time to generate and train on better data)
        duration = 1800  # 30 minutes
        expiry = time.time() + duration
        self._quality_blocked_configs[config_key] = (quality_boost, expiry)

        self._log_info(
            f"Quality boost for {config_key}: {quality_boost:.2f}x for {duration}s "
            f"(quality={quality_score:.2f}, threshold={threshold:.2f}, reason={reason})"
        )

        # Also set exploration boost to diversify the data
        self.set_exploration_boost(config_key, min(1.3, quality_boost), duration)

    def get_quality_boost(self, config_key: str) -> float:
        """Get current quality boost factor for a config.

        Jan 2026 Sprint 10: Quality boost increases the proportion of high-quality
        selfplay modes when a config is blocked by the quality gate.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Quality boost factor (1.0 = normal, >1.0 = boosted high-quality mode %).
        """
        boost_info = self._quality_blocked_configs.get(config_key)
        if not boost_info:
            return 1.0

        boost_factor, expiry = boost_info
        if time.time() > expiry:
            # Boost expired, clean up
            del self._quality_blocked_configs[config_key]
            return 1.0

        return boost_factor

    async def _on_quality_score_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED events - apply quality score immediately.

        Session 17.22: Reduces latency from 0-60s (polling) to <1s (event-driven).
        When quality_monitor_daemon.py assesses data quality, this handler
        immediately stores the score so allocation decisions use fresh data.

        This enables:
        - Immediate allocation shift to higher-quality modes when quality drops
        - Faster response to quality improvements (unblock training sooner)
        - More responsive feedback loop (+8-15 Elo improvement expected)

        Args:
            event: Event with payload containing config_key, quality_score,
                   games_assessed, confidence
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.5)
        games_assessed = payload.get("games_assessed", 0)

        if not config_key:
            return

        # Store the quality score with timestamp
        self._quality_scores[config_key] = (
            quality_score,
            games_assessed,
            time.time(),
        )

        # If quality is low, proactively boost high-quality mode ratio
        # This happens BEFORE training is blocked, preventing quality-gate delays
        if quality_score < 0.5 and games_assessed >= 50:
            # Low quality detected - boost quality-focused selfplay modes
            quality_deficit = 0.5 - quality_score
            boost_factor = 1.0 + (quality_deficit * 3.0)  # 1.0 to 2.5x boost
            boost_factor = min(boost_factor, 2.0)  # Cap lower than blocking boost

            duration = 600  # 10 minutes - reassess on next quality event
            self._quality_blocked_configs[config_key] = (boost_factor, time.time() + duration)

            self._log_info(
                f"Quality preemptive boost for {config_key}: {boost_factor:.2f}x "
                f"(quality={quality_score:.2f}, games={games_assessed})"
            )
        elif quality_score >= 0.7:
            # Good quality - clear any existing boost to restore normal allocation
            if config_key in self._quality_blocked_configs:
                del self._quality_blocked_configs[config_key]
                self._log_info(
                    f"Quality restored for {config_key} (score={quality_score:.2f})"
                )

    def get_quality_score(self, config_key: str) -> tuple[float, int]:
        """Get current quality score and games assessed for a config.

        Session 17.22: Returns the most recent quality score from events.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            Tuple of (quality_score, games_assessed). Defaults to (0.5, 0) if unknown.
        """
        quality_info = self._quality_scores.get(config_key)
        if not quality_info:
            return 0.5, 0

        score, games, timestamp = quality_info
        # Quality scores older than 1 hour are considered stale - decay toward 0.5
        age_hours = (time.time() - timestamp) / 3600.0
        if age_hours > 1.0:
            # Decay toward 0.5 (neutral) with half-life of 1 hour
            decay_factor = 0.5 ** age_hours
            score = 0.5 + (score - 0.5) * decay_factor

        return score, games

    async def _on_hyperparameter_updated(self, event) -> None:
        """Handle HYPERPARAMETER_UPDATED events - adjust selfplay strategy.

        Jan 2026 Sprint 10: When training hyperparameters change (e.g., from
        GauntletFeedbackController), adjust selfplay quality and exploration
        to match the new training configuration.

        Key hyperparameters we respond to:
        - exploration_boost: Adjust temperature/exploration in selfplay
        - quality_threshold: Adjust Gumbel budget and quality mode ratio
        - learning_rate_multiplier: Higher LR = need more diverse data
        - batch_reduction: Smaller batches = can handle higher quality data

        Args:
            event: Event with payload containing param_name, new_value, config
        """
        payload = self._extract_event_payload(event)
        config_key = payload.get("config", payload.get("config_key", ""))
        param_name = payload.get("param_name", "")
        new_value = payload.get("new_value", 0.0)
        reason = payload.get("reason", "hyperparameter_update")

        if not config_key or not param_name:
            return

        self._log_info(
            f"Hyperparameter update for {config_key}: {param_name}={new_value} ({reason})"
        )

        # Handle specific hyperparameter changes
        if param_name == "exploration_boost":
            # Direct exploration boost from gauntlet feedback
            boost_duration = 1800  # 30 minutes
            self.set_exploration_boost(config_key, float(new_value), boost_duration)

        elif param_name == "quality_threshold":
            # Quality threshold raised = need higher quality selfplay
            threshold = float(new_value)
            if threshold >= 0.8:
                # High quality threshold - boost quality mode ratio
                quality_boost = 1.5
                expiry = time.time() + 1800
                self._quality_blocked_configs[config_key] = (quality_boost, expiry)
                self._log_info(
                    f"Quality boost {quality_boost:.2f}x for {config_key} "
                    f"(high threshold: {threshold:.2f})"
                )

        elif param_name == "learning_rate_multiplier":
            # Higher LR needs more diverse data to prevent overfitting
            lr_mult = float(new_value)
            if lr_mult > 1.2:
                # High LR - boost exploration slightly
                exploration = min(1.3, lr_mult * 0.9)
                self.set_exploration_boost(config_key, exploration, 1800)

        elif param_name == "batch_reduction":
            # Smaller batches can handle higher quality data
            batch_factor = float(new_value)
            if batch_factor < 0.8:
                # Significantly smaller batches - we can afford higher quality
                quality_boost = 1.0 + (1.0 - batch_factor)  # e.g., 0.7 batch → 1.3 boost
                expiry = time.time() + 1800
                self._quality_blocked_configs[config_key] = (quality_boost, expiry)

    async def _on_peer_reconnected(self, event: dict) -> None:
        """Handle PEER_RECONNECTED events - rebalance work when node rejoins.

        Jan 2026 Sprint 13: When a peer reconnects (via Tailscale discovery),
        trigger work rebalancing to take advantage of the newly available node.
        This ensures recovered nodes immediately receive selfplay allocations
        rather than waiting for the next scheduling cycle.
        """
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", payload.get("peer_id", "unknown"))

        logger.info(
            f"[SelfplayScheduler] Peer reconnected: {node_id}, "
            "triggering work rebalancing"
        )

        # Mark that we need to rebalance allocations
        # The next scheduling cycle will pick this up
        self._force_rebalance = True

        # Emit event for other components to react
        try:
            from app.coordination.event_router import publish_sync
            publish_sync("SELFPLAY_REBALANCE_TRIGGERED", {
                "reason": "peer_reconnected",
                "node_id": node_id,
                "source": "selfplay_scheduler",
            })
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Could not emit rebalance event: {e}")

    def _emit_selfplay_target_updated(
        self,
        config_key: str,
        priority: str,
        reason: str,
        *,
        target_jobs: int | None = None,
        effective_priority: int | None = None,
        exploration_boost: float | None = None,
    ) -> bool:
        """Emit SELFPLAY_TARGET_UPDATED event for feedback loop integration.

        P0.2 (December 2025): Enables pipeline coordination to respond to
        scheduling decisions. Events trigger:
        - DaemonManager workload scaling
        - FeedbackLoopController priority adjustments
        - Training pipeline data freshness checks

        Dec 2025 (P0-1 fix): Returns bool for caller to check success/failure.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            priority: Priority level ("urgent", "high", "normal")
            reason: Descriptive reason for the update
            target_jobs: Optional target job count
            effective_priority: Optional effective priority value
            exploration_boost: Optional exploration boost multiplier

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        try:
            from app.coordination.event_router import publish_sync

            payload: dict[str, Any] = {
                "config_key": config_key,
                "priority": priority,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            if target_jobs is not None:
                payload["target_jobs"] = target_jobs
            if effective_priority is not None:
                payload["effective_priority"] = effective_priority
            if exploration_boost is not None:
                payload["exploration_boost"] = exploration_boost

            publish_sync("SELFPLAY_TARGET_UPDATED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_TARGET_UPDATED: "
                    f"{config_key} priority={priority} reason={reason}"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for target updates")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit target update: {e}")
            return False

    def _emit_selfplay_rate_changed(
        self,
        config_key: str,
        old_rate: float,
        new_rate: float,
        reason: str,
    ) -> bool:
        """Emit SELFPLAY_RATE_CHANGED event when rate multiplier changes >20%.

        P0.2 (December 2025): Enables IdleResourceDaemon and other consumers
        to react to significant rate changes.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            old_rate: Previous rate multiplier
            new_rate: New rate multiplier
            reason: Descriptive reason for the change

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        # Only emit for significant changes (>20%)
        if old_rate > 0 and abs(new_rate - old_rate) / old_rate < 0.2:
            return False

        try:
            from app.coordination.event_router import publish_sync

            payload = {
                "config_key": config_key,
                "old_rate": old_rate,
                "new_rate": new_rate,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            publish_sync("SELFPLAY_RATE_CHANGED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_RATE_CHANGED: "
                    f"{config_key} {old_rate:.2f} -> {new_rate:.2f} ({reason})"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for rate changes")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit rate change: {e}")
            return False

    def _emit_selfplay_allocation_updated(
        self,
        config_key: str,
        allocation_weights: dict[str, float],
        exploration_boost: float,
        reason: str,
    ) -> bool:
        """Emit SELFPLAY_ALLOCATION_UPDATED event when allocation weights change.

        P0.2 (December 2025): Enables IdleResourceDaemon and SelfplayScheduler
        to react to curriculum weight changes and exploration boosts.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            allocation_weights: Current allocation weights by config
            exploration_boost: Current exploration boost factor
            reason: Descriptive reason for the change

        Returns:
            True if event was emitted successfully, False otherwise.
        """
        try:
            from app.coordination.event_router import publish_sync

            payload = {
                "config_key": config_key,
                "allocation_weights": allocation_weights,
                "exploration_boost": exploration_boost,
                "reason": reason,
                "source": "p2p_selfplay_scheduler",
            }
            publish_sync("SELFPLAY_ALLOCATION_UPDATED", payload)
            if self.verbose:
                logger.info(
                    f"[SelfplayScheduler] Emitted SELFPLAY_ALLOCATION_UPDATED: "
                    f"{config_key} boost={exploration_boost:.2f} ({reason})"
                )
            return True
        except ImportError:
            logger.debug("[SelfplayScheduler] Event router not available for allocation updates")
            return False
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit allocation update: {e}")
            return False

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
            selfplay_configs = [
                c for c in selfplay_configs if c.get("board_type") == "square8"
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
            )

            if self._is_config_plateaued(config_key):
                # Apply 50% penalty for plateaued configs
                cfg["effective_priority"] = max(1, int(base_priority * 0.5))
            else:
                cfg["effective_priority"] = base_priority

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

            # Apply engine mix for any board type with "mixed" or "diverse" mode
            if engine_mode in ("mixed", "diverse"):
                has_gpu = self._node_has_gpu(node)
                num_players = selected.get("num_players", 0)
                config_key = f"{board_type}_{num_players}p"

                # Jan 2026 Sprint 10: Check for quality boost - forces high-quality modes
                quality_boost = self.get_quality_boost(config_key)

                if quality_boost > 1.0 and has_gpu:
                    # Quality boost active - force high-quality Gumbel MCTS with higher budget
                    # Budget scales with boost: 1.5x boost → budget 200, 2.0x → budget 300
                    base_budget = 150
                    boosted_budget = int(base_budget * quality_boost)
                    boosted_budget = min(boosted_budget, 400)  # Cap at 400 for perf

                    actual_engine = "gumbel-mcts"
                    extra_args = {"budget": boosted_budget}

                    logger.info(
                        f"Quality boost override: {config_key} using '{actual_engine}' "
                        f"with budget={boosted_budget} (boost={quality_boost:.2f}x)"
                    )
                else:
                    # Normal selection from engine mix
                    actual_engine, extra_args = self._select_board_engine(
                        has_gpu=has_gpu,
                        board_type=board_type,
                        num_players=num_players,
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

        # Session 17.22: Add architecture selection based on Elo performance
        # This closes the feedback loop: better-performing architectures get more selfplay
        if selected:
            selected_arch = self._select_architecture_for_config(
                board_type=selected.get("board_type", ""),
                num_players=selected.get("num_players", 2),
            )
            selected["model_version"] = selected_arch

        return selected

    def get_target_jobs_for_node(self, node: NodeInfo) -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Args:
            node: Node information

        Returns:
            Target number of selfplay jobs (minimum 1)

        Target: 60-80% CPU/GPU utilization for optimal training throughput.
        """
        # Dec 29, 2025: Skip coordinator nodes (no selfplay capability)
        node_caps = getattr(node, "capabilities", None) or []
        if not node_caps or "selfplay" not in node_caps:
            return 0

        # Check safeguards first - halt all selfplay during emergency
        if self.is_emergency_active is not None:
            try:
                if self.is_emergency_active():
                    return 0
            except (TypeError, AttributeError, RuntimeError, KeyError):
                pass  # Ignore errors in safeguards callback (non-critical)

        # Dec 2025 Phase 5: Check evaluation backpressure - pause when eval queue full
        # Dec 31, 2025: Add minimum job floor for high-end GPUs during backpressure
        # High-end GPUs should never be completely idle to maximize cluster throughput
        if self._evaluation_backpressure_active:
            has_gpu = bool(getattr(node, "has_gpu", False))
            # Use gpu_power_score to identify high-end GPUs:
            # H100 80GB = 4000, A100 80GB = 6864, GH200 = 15000
            gpu_power = int(getattr(node, "gpu_power_score", 0) or 0)
            gpu_name = str(getattr(node, "gpu_name", "") or "")
            # High-end GPU: A100, H100, GH200, RTX 5090 (power_score >= 4000)
            is_high_end_gpu = has_gpu and gpu_power >= 4000

            if is_high_end_gpu:
                # Allow 25% capacity during backpressure for high-end GPUs
                # This prevents total starvation while still reducing throughput
                base_target = int(getattr(node, "max_selfplay_jobs", 4) or 4)
                min_jobs = max(1, base_target // 4)
                logger.info(
                    f"Backpressure active but allowing {min_jobs} jobs on high-end GPU "
                    f"{node.node_id} ({gpu_name}, power={gpu_power})"
                )
                return min_jobs

            logger.debug(
                f"Evaluation backpressure active, halting selfplay on {node.node_id}"
            )
            return 0

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if (
            self.should_stop_production is not None
            and self.should_throttle_production is not None
        ):
            try:
                # Import QueueType here to avoid circular imports
                try:
                    from app.coordination import QueueType

                    queue_type = QueueType.TRAINING_DATA
                except ImportError:
                    queue_type = "TRAINING_DATA"

                if self.should_stop_production(queue_type):
                    logger.info(
                        f"Backpressure STOP: training queue full, halting selfplay on {node.node_id}"
                    )
                    return 0
                if self.should_throttle_production(queue_type):
                    if self.get_throttle_factor is not None:
                        backpressure_factor = self.get_throttle_factor(queue_type)
                        logger.info(f"Backpressure throttle: factor={backpressure_factor:.2f}")
            except (TypeError, AttributeError, ValueError, RuntimeError) as e:
                logger.info(f"Backpressure check error: {e}")

        # Minimum memory requirement - skip low-memory machines to avoid OOM
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Memory-aware job allocation (P1 - Sprint 6, Jan 2026)
        # Check current memory usage, not just total memory
        has_gpu = bool(getattr(node, "has_gpu", False))
        recommended_job_type = self._get_recommended_job_type(node)
        if not self._check_memory_available(node, recommended_job_type):
            # Not enough memory for even the lightest job type
            if not self._check_memory_available(node, "cpu_heuristic"):
                if self.verbose:
                    node_id = getattr(node, "node_id", "unknown")
                    logger.debug(
                        f"Node {node_id} has insufficient memory even for cpu_heuristic, "
                        f"skipping job allocation"
                    )
                return 0

        # Extract node metrics
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)

        # Record utilization for adaptive feedback
        if self.record_utilization is not None:
            with contextlib.suppress(Exception):
                self.record_utilization(
                    node.node_id, cpu_percent, gpu_percent, mem_percent, current_jobs
                )

        # Use unified resource targets if available
        if (
            self.get_host_targets is not None
            and self.get_target_job_count is not None
        ):
            try:
                # Get host-specific targets adjusted for tier and backpressure
                host_targets = self.get_host_targets(node.node_id)

                # Use the unified target calculator
                target_selfplay = self.get_target_job_count(
                    node.node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Check if we should scale up (underutilized)
                if self.should_scale_up is not None:
                    scale_up, reason = self.should_scale_up(
                        node.node_id, cpu_percent, gpu_percent, current_jobs
                    )
                    if scale_up and current_jobs < target_selfplay:
                        # Controlled scale-up: Add 2-4 jobs at a time, not all at once
                        scale_up_increment = min(4, target_selfplay - current_jobs)
                        target_selfplay = current_jobs + scale_up_increment
                        if self.verbose:
                            logger.info(
                                f"Scale-up on {node.node_id}: {reason}, target={target_selfplay}"
                            )

                # Check if we should scale down (overloaded)
                if self.should_scale_down is not None:
                    scale_down, reduction, reason = self.should_scale_down(
                        node.node_id, cpu_percent, gpu_percent, mem_percent
                    )
                    if scale_down:
                        target_selfplay = max(1, current_jobs - reduction)
                        logger.info(
                            f"Scale-down on {node.node_id}: {reason}, target={target_selfplay}"
                        )

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                target_selfplay = min(target_selfplay, host_targets.max_selfplay)

                final_target = int(max(1, target_selfplay))

                # P0.2 (Dec 2025): Emit event when target changes significantly
                old_target = self._previous_targets.get(node.node_id, final_target)
                target_change = abs(final_target - old_target)
                relative_change = target_change / max(1, old_target)
                if target_change >= 3 or relative_change >= 0.5:
                    reason = "target_increased" if final_target > old_target else "target_decreased"
                    evt_priority = "high" if target_change >= 5 or relative_change >= 0.75 else "normal"
                    self._emit_selfplay_target_updated(
                        config_key=f"node:{node.node_id}",
                        priority=evt_priority,
                        reason=f"{reason}:{target_change}",
                        target_jobs=final_target,
                    )
                    self._previous_targets[node.node_id] = final_target

                return final_target

            except (TypeError, AttributeError, ValueError, KeyError, RuntimeError) as e:
                logger.info(f"Resource targets error, falling back to hardware-aware: {e}")

        # FALLBACK: Use unified hardware-aware limits from resource_optimizer
        # This ensures consistent limits across all orchestrators
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        if self.get_max_selfplay_for_node is not None:
            # Use single source of truth from resource_optimizer
            max_selfplay = self.get_max_selfplay_for_node(
                node_id=node.node_id,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                has_gpu=has_gpu,
            )
        else:
            # Minimal fallback when resource_optimizer unavailable
            # Values calibrated from observed workloads (GH200: 48 jobs at 70% GPU)
            if has_gpu:
                gpu_upper = gpu_name.upper()
                if any(g in gpu_upper for g in ["GH200"]):
                    # GH200 with unified 480GB memory - CPU is bottleneck
                    max_selfplay = int(cpu_count * 0.8) if cpu_count > 0 else 48
                elif any(g in gpu_upper for g in ["H100", "H200"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.5), 48) if cpu_count > 0 else 32
                    )
                elif any(g in gpu_upper for g in ["A100", "L40"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.4), 32) if cpu_count > 0 else 24
                    )
                elif any(g in gpu_upper for g in ["5090"]):
                    # RTX 5090 (32GB) - very high capacity
                    max_selfplay = (
                        min(int(cpu_count * 0.3), gpu_count * 12, 64)
                        if cpu_count > 0
                        else 48
                    )
                elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.3), 24) if cpu_count > 0 else 16
                    )
                elif any(g in gpu_upper for g in ["4080", "4070", "3080", "4060"]):
                    max_selfplay = (
                        min(int(cpu_count * 0.25), 12) if cpu_count > 0 else 8
                    )
                elif any(
                    g in gpu_upper for g in ["3070", "3060", "2060", "2070", "2080"]
                ):
                    max_selfplay = (
                        min(int(cpu_count * 0.2), 10) if cpu_count > 0 else 6
                    )
                else:
                    max_selfplay = min(int(cpu_count * 0.2), 8) if cpu_count > 0 else 6
            else:
                # CPU-only: ~0.3 jobs per core, capped at 32
                max_selfplay = min(int(cpu_count * 0.3), 32) if cpu_count > 0 else 8

        target_selfplay = max_selfplay

        # Utilization-aware adjustments (target 60-80%)
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 80
        gpu_has_headroom = gpu_percent < 60 and gpu_mem_percent < 75
        cpu_has_headroom = cpu_percent < 60

        # Scale DOWN if overloaded
        if gpu_overloaded:
            target_selfplay = max(2, target_selfplay - 2)
        if cpu_overloaded:
            target_selfplay = max(2, target_selfplay - 1)

        # Scale UP only if both resources have headroom (gradual)
        if (
            not gpu_overloaded
            and not cpu_overloaded
            and current_jobs > 0
            and (has_gpu and gpu_has_headroom and cpu_has_headroom)
        ) or ((not has_gpu and cpu_has_headroom) and current_jobs < target_selfplay):
            target_selfplay = min(target_selfplay, current_jobs + 2)

        # Resource pressure warnings
        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        final_target = int(max(1, target_selfplay))

        # P0.2 (Dec 2025): Emit event when target changes significantly
        node_id = getattr(node, "node_id", "unknown")
        old_target = self._previous_targets.get(node_id, final_target)
        target_change = abs(final_target - old_target)

        # Emit if target changed by 3+ jobs or 50%+ relative change
        relative_change = target_change / max(1, old_target)
        if target_change >= 3 or relative_change >= 0.5:
            reason = "target_increased" if final_target > old_target else "target_decreased"
            priority = "high" if target_change >= 5 or relative_change >= 0.75 else "normal"
            self._emit_selfplay_target_updated(
                config_key=f"node:{node_id}",
                priority=priority,
                reason=f"{reason}:{target_change}",
                target_jobs=final_target,
            )
            self._previous_targets[node_id] = final_target

        return final_target

    def get_hybrid_job_targets(self, node: NodeInfo) -> dict[str, int]:
        """Get separate GPU and CPU-only selfplay job targets for hybrid mode.

        For high-CPU nodes with limited GPU VRAM (like Vast hosts), this enables:
        - Running GPU jobs up to VRAM limit
        - Running additional CPU-only jobs to utilize excess CPU capacity

        Args:
            node: Node information

        Returns:
            Dict with 'gpu_jobs', 'cpu_only_jobs', 'total_jobs'
        """
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        # Use hybrid limits function if available
        if self.get_hybrid_selfplay_limits is not None:
            try:
                limits = self.get_hybrid_selfplay_limits(
                    node_id=node.node_id,
                    gpu_count=gpu_count,
                    gpu_name=gpu_name,
                    cpu_count=cpu_count,
                    memory_gb=memory_gb,
                    has_gpu=has_gpu,
                )
                return limits
            except (TypeError, AttributeError, ValueError, KeyError, RuntimeError) as e:
                logger.info(f"Hybrid limits error: {e}")

        # Fallback: No CPU-only jobs, use standard target
        gpu_jobs = self.get_target_jobs_for_node(node)
        return {"gpu_jobs": gpu_jobs, "cpu_only_jobs": 0, "total_jobs": gpu_jobs}

    def should_spawn_cpu_only_jobs(self, node: NodeInfo) -> bool:
        """Check if a node should spawn CPU-only jobs in addition to GPU jobs.

        CPU-only jobs are beneficial when:
        1. Node has many CPU cores (64+)
        2. Node has limited GPU VRAM (<=16GB per GPU)
        3. GPU jobs are already at capacity (VRAM-limited)

        Args:
            node: Node information

        Returns:
            True if CPU-only jobs should be spawned
        """
        if self.get_hybrid_selfplay_limits is None:
            return False

        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        has_gpu = bool(getattr(node, "has_gpu", False))
        gpu_name = (getattr(node, "gpu_name", "") or "").upper()

        # Must have significant CPU resources (64+ cores)
        if cpu_count < 64:
            return False

        # For GPU nodes, only spawn CPU-only if GPU has limited VRAM
        if has_gpu:
            # High-end datacenter GPUs don't need CPU-only jobs (plenty of VRAM)
            if any(g in gpu_name for g in ["GH200", "H100", "H200", "A100", "L40"]):
                return False
            # Consumer GPUs with limited VRAM benefit from CPU-only supplement
            if any(
                g in gpu_name
                for g in ["3070", "3060", "2060", "2070", "2080", "4060", "4070"]
            ):
                return True
            # 5090/4090 with 24-32GB might not need it unless very high CPU count
            if any(g in gpu_name for g in ["5090", "4090", "3090"]):
                return cpu_count >= CPU_ONLY_JOB_MIN_CPUS

        # CPU-only nodes always benefit from full CPU utilization
        return True

    def track_diversity(self, config: dict[str, Any]) -> None:
        """Track diversity metrics for a scheduled selfplay game.

        Args:
            config: Selfplay configuration dict with engine_mode, board_type, num_players, etc.

        January 2026 Sprint 10: Added opponent type tracking for diversity maximization.
        Records which opponent types each config has played against.
        """
        # Track engine mode
        engine_mode = config.get("engine_mode", "unknown")
        if engine_mode not in self.diversity_metrics.games_by_engine_mode:
            self.diversity_metrics.games_by_engine_mode[engine_mode] = 0
        self.diversity_metrics.games_by_engine_mode[engine_mode] += 1

        # Track board config
        board_key = (
            f"{config.get('board_type', 'unknown')}_{config.get('num_players', 0)}p"
        )
        if board_key not in self.diversity_metrics.games_by_board_config:
            self.diversity_metrics.games_by_board_config[board_key] = 0
        self.diversity_metrics.games_by_board_config[board_key] += 1

        # January 2026 Sprint 10: Track opponent type for diversity scoring
        # Record the engine_mode as the opponent type for this config
        self.diversity_metrics.record_opponent(board_key, engine_mode)

        # Track asymmetric vs symmetric
        if config.get("asymmetric"):
            self.diversity_metrics.asymmetric_games += 1
            strong = config.get("strong_config", {})
            weak = config.get("weak_config", {})
            logger.info(
                f"DIVERSE: Asymmetric game scheduled - "
                f"Strong({strong.get('engine_mode')}@D{strong.get('difficulty')}) vs "
                f"Weak({weak.get('engine_mode')}@D{weak.get('difficulty')}) "
                f"on {board_key}"
            )
            # Record both opponent types in asymmetric games
            if strong.get("engine_mode"):
                self.diversity_metrics.record_opponent(board_key, strong["engine_mode"])
            if weak.get("engine_mode"):
                self.diversity_metrics.record_opponent(board_key, weak["engine_mode"])
        else:
            self.diversity_metrics.symmetric_games += 1

        # Track difficulty if available
        difficulty = config.get("difficulty", config.get("difficulty_band"))
        if difficulty:
            diff_key = str(difficulty)
            if diff_key not in self.diversity_metrics.games_by_difficulty:
                self.diversity_metrics.games_by_difficulty[diff_key] = 0
            self.diversity_metrics.games_by_difficulty[diff_key] += 1

    def get_diversity_metrics(self) -> dict[str, Any]:
        """Get diversity tracking metrics for monitoring.

        Returns:
            Dictionary with diversity metrics including computed statistics
        """
        return self.diversity_metrics.to_dict()

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
            except (OSError, json.JSONDecodeError) as e:
                # Config file read/parse errors - use empty weights
                logger.debug(f"Could not load curriculum weights: {e}")
            except (ValueError, KeyError) as e:
                # Invalid weight values/structure - use empty weights
                logger.debug(f"Invalid curriculum weights format: {e}")
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
