"""Architecture Feedback Controller for RingRift.

Bridges evaluation results to selfplay allocation by tracking architecture
performance and adjusting weights accordingly. Ensures all architectures
maintain at least 10% allocation to prevent starvation.

Key Responsibilities:
    - Subscribe to EVALUATION_COMPLETED, TRAINING_COMPLETED events
    - Record architecture performance via ArchitectureTracker
    - Emit ARCHITECTURE_WEIGHTS_UPDATED events
    - Enforce 10% minimum allocation per architecture

Usage:
    from app.coordination.architecture_feedback_controller import (
        get_architecture_feedback_controller,
        ArchitectureFeedbackController,
    )

    # Get singleton
    controller = get_architecture_feedback_controller()
    await controller.start()

December 29, 2025 - Phase 2 of Unified AI Evaluation Architecture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureFeedbackConfig:
    """Configuration for architecture feedback controller.

    Attributes:
        min_allocation_per_arch: Minimum allocation for any architecture (10%)
        weight_update_interval: How often to emit weight updates (seconds)
        weight_temperature: Temperature for softmax allocation (higher = more uniform)
        supported_architectures: List of architectures to track
    """

    min_allocation_per_arch: float = 0.10  # 10% minimum per architecture
    weight_update_interval: float = 1800.0  # 30 minutes
    weight_temperature: float = 0.5  # Balance exploration vs exploitation

    # Architectures to track (NN + NNUE)
    supported_architectures: list[str] = field(
        default_factory=lambda: [
            "v2",  # Legacy baseline
            "v4",  # NAS-optimized
            "v5",  # Current production
            "v5_heavy",  # High-capacity variant
            "v6",  # Next generation
            "nnue_v1",  # NNUE value-only
            "nnue_v1_policy",  # NNUE with policy head
        ]
    )


@dataclass
class ArchitectureFeedbackState:
    """Persistent state for architecture feedback tracking."""

    # Last weight update time
    last_weight_update_time: float = 0.0

    # Cached weights per config_key
    cached_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    # Event counts for tracking
    evaluations_processed: int = 0
    trainings_processed: int = 0
    comparisons_processed: int = 0


class ArchitectureFeedbackController(HandlerBase):
    """Controller that bridges evaluation results to selfplay allocation.

    Subscribes to evaluation and training events, records architecture
    performance, and emits weight update events for SelfplayScheduler.

    Thread-safe singleton with state persistence.
    """

    _instance: ClassVar[ArchitectureFeedbackController | None] = None
    CONTROLLER_VERSION: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        config: ArchitectureFeedbackConfig | None = None,
    ):
        """Initialize architecture feedback controller.

        Args:
            config: Optional configuration override
        """
        super().__init__(
            name="architecture_feedback",
            cycle_interval=60.0,  # Check every minute
        )
        self._config = config or ArchitectureFeedbackConfig()
        self._state = ArchitectureFeedbackState()
        self._running = False

    @classmethod
    def get_instance(cls) -> ArchitectureFeedbackController:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for this handler."""
        return {
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            "TRAINING_COMPLETED": self._on_training_completed,
            "ARCHITECTURE_COMPARISON_COMPLETED": self._on_architecture_comparison_completed,
        }

    async def _on_evaluation_completed(self, event: dict[str, Any]) -> None:
        """Handle evaluation completed event.

        Records architecture performance based on Elo results.

        December 30, 2025: Updated to prefer explicit architecture field if
        available in event, falling back to extraction from model_path.

        Args:
            event: Event data with config_key, model_path, elo, architecture, etc.
        """
        try:
            from app.training.architecture_tracker import (
                extract_architecture_from_model_path,
                get_architecture_tracker,
            )
            from app.coordination.event_utils import extract_evaluation_data

            data = extract_evaluation_data(event)
            if not data.is_valid or not data.model_path:
                return

            # December 30, 2025: Prefer explicit architecture from event payload
            # This supports multi-architecture training where architecture is known
            # Check both metadata and direct payload (event_router puts it in payload)
            architecture = event.get("architecture") or event.get("metadata", {}).get("architecture")
            if not architecture:
                # Fallback to extraction from model path
                architecture = extract_architecture_from_model_path(data.model_path)

            # Record in tracker
            tracker = get_architecture_tracker()
            tracker.record_evaluation(
                architecture=architecture,
                board_type=data.board_type,
                num_players=data.num_players,
                elo=data.elo,
                games_evaluated=data.games_played,
            )

            self._state.evaluations_processed += 1
            logger.info(
                f"ArchitectureFeedback: Recorded eval for {architecture}/{data.config_key} "
                f"(Elo: {data.elo:.0f})"
            )

            # Check if we should emit weight update
            await self._maybe_emit_weight_update(data.config_key)

        except (ImportError, ValueError, KeyError) as e:
            logger.debug(f"ArchitectureFeedback: Error handling eval: {e}")

    async def _on_training_completed(self, event: dict[str, Any]) -> None:
        """Handle training completed event.

        Updates training hours for architecture efficiency calculation.

        Args:
            event: Event data with config_key, model_path, duration, etc.
        """
        try:
            from app.training.architecture_tracker import (
                extract_architecture_from_model_path,
                get_architecture_tracker,
            )
            from app.coordination.event_utils import extract_training_data

            data = extract_training_data(event)
            duration_seconds = event.get("duration_seconds", 0.0)

            if not data.is_valid or not data.model_path:
                return

            # Extract architecture
            architecture = extract_architecture_from_model_path(data.model_path)

            # Record training time (convert to hours)
            training_hours = duration_seconds / 3600.0

            tracker = get_architecture_tracker()
            # Use record_evaluation with just training hours
            # Elo=1000 (neutral) and games=0 to only update time
            tracker.record_evaluation(
                architecture=architecture,
                board_type=data.board_type,
                num_players=data.num_players,
                elo=1000.0,  # Neutral, no Elo update
                training_hours=training_hours,
                games_evaluated=0,
            )

            self._state.trainings_processed += 1
            logger.debug(
                f"ArchitectureFeedback: Recorded {training_hours:.2f}h for "
                f"{architecture}/{data.config_key}"
            )

        except (ImportError, ValueError, KeyError) as e:
            logger.debug(f"ArchitectureFeedback: Error handling training: {e}")

    async def _on_architecture_comparison_completed(self, event: dict[str, Any]) -> None:
        """Handle architecture comparison completed event.

        Updates architecture tracker with comparison results and emits weight
        updates based on the new Elo rankings.

        Args:
            event: Event data with elo_ratings, matchups, config, etc.
        """
        try:
            from app.training.architecture_tracker import get_architecture_tracker

            elo_ratings = event.get("elo_ratings", {})
            config_data = event.get("config", {})
            matchups = event.get("matchups", [])

            board_type = config_data.get("board_type", "")
            num_players = config_data.get("num_players", 2)
            harness = config_data.get("harness", "policy_only")

            if not elo_ratings or not board_type:
                logger.debug(
                    "ArchitectureFeedback: Missing elo_ratings or board_type in comparison"
                )
                return

            # Calculate total games per architecture from matchups
            games_per_arch: dict[str, int] = {}
            for matchup in matchups:
                arch_a = matchup.get("arch_a", "")
                arch_b = matchup.get("arch_b", "")
                games = matchup.get("games_played", 0)
                games_per_arch[arch_a] = games_per_arch.get(arch_a, 0) + games
                games_per_arch[arch_b] = games_per_arch.get(arch_b, 0) + games

            tracker = get_architecture_tracker()

            # Record each architecture's Elo from comparison
            for architecture, elo in elo_ratings.items():
                games = games_per_arch.get(architecture, 0)

                # Record in tracker with harness-specific Elo
                tracker.record_evaluation(
                    architecture=architecture,
                    board_type=board_type,
                    num_players=num_players,
                    elo=elo,
                    games_evaluated=games,
                )

                # Also record harness-specific Elo
                stats = tracker.get_stats(architecture, board_type, num_players)
                if stats:
                    stats.record_harness_evaluation(
                        harness=harness,
                        elo=elo,
                        games=games,
                    )

                logger.info(
                    f"ArchitectureFeedback: Recorded comparison for {architecture}/{board_type}_{num_players}p "
                    f"(Elo: {elo:.0f}, harness: {harness})"
                )

            self._state.comparisons_processed += 1

            # Emit weight update immediately after comparison
            config_key = make_config_key(board_type, num_players)
            await self._emit_architecture_weights_updated(config_key)

        except (ImportError, ValueError, KeyError) as e:
            logger.debug(f"ArchitectureFeedback: Error handling comparison: {e}")

    async def _maybe_emit_weight_update(self, config_key: str) -> None:
        """Emit weight update if enough time has passed.

        Args:
            config_key: Configuration key that triggered update check
        """
        now = time.time()
        if now - self._state.last_weight_update_time < self._config.weight_update_interval:
            return

        await self._emit_architecture_weights_updated(config_key)
        self._state.last_weight_update_time = now

    async def _emit_architecture_weights_updated(self, config_key: str) -> None:
        """Compute and emit architecture weights for a config.

        Weights are based on Elo performance with 10% minimum per architecture.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
        """
        try:
            from app.training.architecture_tracker import get_allocation_weights

            # Dec 30, 2025: Use consolidated parse_config_key utility
            parsed = parse_config_key(config_key)
            if not parsed:
                return

            board_type = parsed.board_type
            num_players = parsed.num_players

            # Get raw weights from tracker
            raw_weights = get_allocation_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=self._config.weight_temperature,
            )

            # Enforce minimum allocation
            weights = self._enforce_minimum_allocation(raw_weights)

            # Cache and emit
            self._state.cached_weights[config_key] = weights

            # Emit event
            self._emit_event(
                "ARCHITECTURE_WEIGHTS_UPDATED",
                {
                    "config_key": config_key,
                    "weights": weights,
                    "timestamp": time.time(),
                },
            )

            logger.info(
                f"ArchitectureFeedback: Emitted weights for {config_key}: "
                f"{list(weights.items())[:3]}..."
            )

        except (ImportError, ValueError, KeyError) as e:
            logger.debug(f"ArchitectureFeedback: Error emitting weights: {e}")

    def _enforce_minimum_allocation(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Enforce 10% minimum allocation per architecture.

        This ensures all architectures continue to receive training,
        preventing the system from getting stuck on one architecture.

        Args:
            weights: Raw allocation weights

        Returns:
            Adjusted weights with 10% minimum per architecture
        """
        if not weights:
            return weights

        min_alloc = self._config.min_allocation_per_arch
        num_archs = len(weights)

        # If minimum allocation would exceed 100%, use equal weights
        if num_archs * min_alloc > 1.0:
            return {arch: 1.0 / num_archs for arch in weights}

        # Apply minimum floor
        adjusted = {}
        for arch, weight in weights.items():
            adjusted[arch] = max(min_alloc, weight)

        # Renormalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {arch: w / total for arch, w in adjusted.items()}

        return adjusted

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to the event router."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.emit(event_type, data)
        except (ImportError, AttributeError) as e:
            logger.debug(f"ArchitectureFeedback: Could not emit {event_type}: {e}")

    async def _run_cycle(self) -> None:
        """Run periodic check cycle.

        Emits weight updates for all tracked configs periodically.
        """
        if not self._running:
            return

        # Periodic weight update for all tracked configs
        now = time.time()
        if now - self._state.last_weight_update_time >= self._config.weight_update_interval:
            for config_key in list(self._state.cached_weights.keys()):
                await self._emit_architecture_weights_updated(config_key)
            self._state.last_weight_update_time = now

    def health_check(self) -> HealthCheckResult:
        """Return health check result for this controller."""
        from app.coordination.contracts import CoordinatorStatus

        is_healthy = self._running

        details = {
            "evaluations_processed": self._state.evaluations_processed,
            "trainings_processed": self._state.trainings_processed,
            "comparisons_processed": self._state.comparisons_processed,
            "cached_configs": len(self._state.cached_weights),
            "version": self.CONTROLLER_VERSION,
        }

        return HealthCheckResult(
            status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED,
            message="Running" if is_healthy else "Not running",
            details=details,
        )


# Module-level accessor functions
def get_architecture_feedback_controller() -> ArchitectureFeedbackController:
    """Get singleton architecture feedback controller."""
    return ArchitectureFeedbackController.get_instance()


async def start_architecture_feedback_controller() -> ArchitectureFeedbackController:
    """Start the architecture feedback controller."""
    controller = get_architecture_feedback_controller()
    await controller.start()
    return controller
