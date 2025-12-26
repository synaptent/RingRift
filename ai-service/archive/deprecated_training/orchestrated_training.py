"""Training Orchestrator - Manager lifecycle coordination.

.. deprecated:: December 2025
    This module is deprecated. Use ``UnifiedTrainingOrchestrator`` from
    ``unified_orchestrator.py`` instead, which now includes all manager
    coordination features.

    See ``app/training/ORCHESTRATOR_GUIDE.md`` for migration instructions.

Migration:
    # Old
    from app.training.orchestrated_training import TrainingOrchestrator
    orchestrator = TrainingOrchestrator(config)
    await orchestrator.initialize()

    # New
    from app.training.unified_orchestrator import (
        UnifiedTrainingOrchestrator,
        OrchestratorConfig,
    )
    orchestrator = UnifiedTrainingOrchestrator(model, config)
    with orchestrator:
        for batch in orchestrator.get_dataloader():
            loss = orchestrator.train_step(batch)

This module provided a single entry point for coordinating all training-related
managers and services. These features are now built into UnifiedTrainingOrchestrator.

Managers that were wrapped (now in UnifiedTrainingOrchestrator):
- UnifiedCheckpointManager: Checkpoint saving/loading
- IntegratedTrainingManager: Training enhancements
- RollbackManager: Model rollback on regression
- PromotionController: Model promotion decisions
- TrainingCoordinator: Cluster-wide training coordination
- DataCoordinator: Training data management
- EloService: Elo rating updates
- CurriculumFeedback: Curriculum weight adjustments
"""

from __future__ import annotations

import logging
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

# Emit deprecation warning on import (December 2025)
warnings.warn(
    "orchestrated_training.py is deprecated. "
    "Use UnifiedTrainingOrchestrator from unified_orchestrator.py instead. "
    "See app/training/ORCHESTRATOR_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingOrchestratorConfig:
    """Configuration for TrainingOrchestrator."""
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_steps: int = 1000
    max_checkpoints: int = 5

    # Rollback
    enable_rollback: bool = True
    auto_rollback: bool = True

    # Promotion
    enable_promotion: bool = True
    promotion_cooldown_seconds: float = 900.0

    # Data coordination
    use_data_coordinator: bool = True
    prefetch_batches: int = 3

    # Curriculum
    enable_curriculum: bool = True
    curriculum_rebalance_interval: int = 5000


@dataclass
class TrainingOrchestratorState:
    """State tracking for the orchestrator."""
    initialized: bool = False
    current_step: int = 0
    current_epoch: int = 0
    last_checkpoint_step: int = 0
    last_evaluation_step: int = 0
    training_active: bool = False
    managers_loaded: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class TrainingOrchestrator:
    """Manager LIFECYCLE orchestrator for training infrastructure.

    .. note:: Orchestrator Hierarchy (2025-12)
        - **UnifiedTrainingOrchestrator** (unified_orchestrator.py): Step-level
          training operations (forward/backward pass, hot buffer, enhancements)
        - **TrainingOrchestrator** (this): Manager lifecycle coordination
          (checkpoint manager, rollback manager, data coordinator, Elo service)
        - **ModelSyncCoordinator** (model_lifecycle.py): Model registry sync
        - **P2P Coordinators** (p2p_integration.py): P2P cluster REST API wrappers

    Use this class when you need to coordinate multiple training managers.
    For step-level training operations, use UnifiedTrainingOrchestrator.

    This class coordinates multiple training-related managers, providing:
    - Unified initialization and shutdown
    - Coordinated checkpointing
    - Automatic rollback on regression
    - Promotion evaluation triggers
    - Data coordination
    """

    def __init__(self, config: TrainingOrchestratorConfig | None = None):
        self.config = config or TrainingOrchestratorConfig()
        self.state = TrainingOrchestratorState()

        # Manager instances (lazy loaded)
        self._checkpoint_manager = None
        self._rollback_manager = None
        self._promotion_controller = None
        self._training_coordinator = None
        self._data_coordinator = None
        self._elo_service = None
        self._curriculum_feedback = None

    async def initialize(self) -> bool:
        """Initialize all managers.

        Returns:
            True if all managers initialized successfully
        """
        if self.state.initialized:
            return True

        logger.info("[TrainingOrchestrator] Initializing managers...")
        start_time = time.time()
        managers_loaded = []

        # Load checkpoint manager
        try:
            from app.training.checkpoint_unified import UnifiedCheckpointManager
            self._checkpoint_manager = UnifiedCheckpointManager(
                checkpoint_dir=self.config.checkpoint_dir,
                max_checkpoints=self.config.max_checkpoints,
            )
            managers_loaded.append("checkpoint_manager")
        except ImportError as e:
            logger.warning(f"[TrainingOrchestrator] Could not load checkpoint_manager: {e}")
            self.state.errors.append(f"checkpoint_manager: {e}")

        # Load rollback manager and wire regression→rollback automation (December 2025)
        if self.config.enable_rollback:
            try:
                from app.training.model_registry import get_model_registry
                from app.training.rollback_manager import (
                    RollbackManager,
                    wire_regression_to_rollback,
                )

                # Get registry and create rollback manager
                registry = get_model_registry()
                self._rollback_manager = RollbackManager(registry)

                # Wire regression detector to auto-rollback handler
                # This enables automatic rollback on CRITICAL regressions
                # and pending rollbacks requiring approval for SEVERE regressions
                wire_regression_to_rollback(
                    registry=registry,
                    auto_rollback_enabled=self.config.auto_rollback,
                    require_approval_for_severe=True,
                    subscribe_to_events=True,
                )
                logger.info(
                    f"[TrainingOrchestrator] Regression→Rollback wired "
                    f"(auto={self.config.auto_rollback})"
                )
                managers_loaded.append("rollback_manager")
            except ImportError as e:
                logger.warning(f"[TrainingOrchestrator] Could not load rollback_manager: {e}")
                self.state.errors.append(f"rollback_manager: {e}")

        # Load promotion controller
        if self.config.enable_promotion:
            try:
                from app.training.promotion_controller import PromotionController
                self._promotion_controller = PromotionController()
                managers_loaded.append("promotion_controller")
            except ImportError as e:
                logger.warning(f"[TrainingOrchestrator] Could not load promotion_controller: {e}")
                self.state.errors.append(f"promotion_controller: {e}")

        # Load training coordinator
        try:
            from app.coordination.training_coordinator import (
                get_training_coordinator,
            )
            self._training_coordinator = get_training_coordinator()
            managers_loaded.append("training_coordinator")
        except ImportError as e:
            logger.warning(f"[TrainingOrchestrator] Could not load training_coordinator: {e}")
            self.state.errors.append(f"training_coordinator: {e}")

        # Load data coordinator
        if self.config.use_data_coordinator:
            try:
                from app.training.data_coordinator import DataCoordinator
                self._data_coordinator = DataCoordinator()
                managers_loaded.append("data_coordinator")
            except ImportError as e:
                logger.warning(f"[TrainingOrchestrator] Could not load data_coordinator: {e}")
                self.state.errors.append(f"data_coordinator: {e}")

        # Load curriculum feedback
        if self.config.enable_curriculum:
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback
                self._curriculum_feedback = get_curriculum_feedback()
                managers_loaded.append("curriculum_feedback")
            except ImportError as e:
                logger.warning(f"[TrainingOrchestrator] Could not load curriculum_feedback: {e}")
                self.state.errors.append(f"curriculum_feedback: {e}")

        # Load EloService (SSoT for Elo ratings in training pipeline)
        try:
            from app.training.elo_service import get_elo_service
            self._elo_service = get_elo_service()
            managers_loaded.append("elo_service")
        except ImportError as e:
            logger.debug(f"[TrainingOrchestrator] Could not load elo_service: {e}")
            # EloService is optional, don't add to errors

        self.state.managers_loaded = managers_loaded
        self.state.initialized = True

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[TrainingOrchestrator] Initialized {len(managers_loaded)} managers "
            f"in {duration_ms:.1f}ms: {', '.join(managers_loaded)}"
        )

        return len(self.state.errors) == 0

    async def shutdown(self) -> None:
        """Shutdown all managers gracefully."""
        if not self.state.initialized:
            return

        logger.info("[TrainingOrchestrator] Shutting down managers...")

        # Save final checkpoint if training was active
        if self.state.training_active and self._checkpoint_manager:
            try:
                self._checkpoint_manager.save_checkpoint(
                    step=self.state.current_step,
                    is_final=True,
                )
            except Exception as e:
                logger.error(f"[TrainingOrchestrator] Failed to save final checkpoint: {e}")

        self.state.initialized = False
        self.state.training_active = False
        logger.info("[TrainingOrchestrator] Shutdown complete")

    @asynccontextmanager
    async def training_context(self):
        """Context manager for a training session.

        Handles initialization, coordination lock, and cleanup.
        """
        if not self.state.initialized:
            await self.initialize()

        # Request training slot if coordinator is available
        if self._training_coordinator:
            try:
                from app.coordination.training_coordinator import request_training_slot
                slot = request_training_slot()
                if not slot:
                    logger.warning("[TrainingOrchestrator] Could not acquire training slot")
            except Exception as e:
                logger.warning(f"[TrainingOrchestrator] Training slot request failed: {e}")

        self.state.training_active = True
        try:
            yield self
        finally:
            self.state.training_active = False
            # Release training slot
            if self._training_coordinator:
                try:
                    from app.coordination.training_coordinator import release_training_slot
                    release_training_slot()
                except (ImportError, RuntimeError):
                    pass

    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint."""
        steps_since_checkpoint = self.state.current_step - self.state.last_checkpoint_step
        return steps_since_checkpoint >= self.config.checkpoint_interval_steps

    def record_step(self, step: int) -> None:
        """Record a completed training step."""
        self.state.current_step = step

    def record_epoch(self, epoch: int) -> None:
        """Record a completed epoch."""
        self.state.current_epoch = epoch

    def save_checkpoint(self, **kwargs) -> str | None:
        """Save a checkpoint using the checkpoint manager.

        Returns:
            Checkpoint path if successful, None otherwise
        """
        if not self._checkpoint_manager:
            return None

        try:
            path = self._checkpoint_manager.save_checkpoint(
                step=self.state.current_step,
                epoch=self.state.current_epoch,
                **kwargs,
            )
            self.state.last_checkpoint_step = self.state.current_step
            return path
        except Exception as e:
            logger.error(f"[TrainingOrchestrator] Checkpoint save failed: {e}")
            return None

    def get_latest_checkpoint(self) -> str | None:
        """Get path to latest checkpoint."""
        if not self._checkpoint_manager:
            return None
        return self._checkpoint_manager.get_latest_checkpoint()

    def get_state(self) -> dict[str, Any]:
        """Get orchestrator state for monitoring."""
        return {
            "initialized": self.state.initialized,
            "training_active": self.state.training_active,
            "current_step": self.state.current_step,
            "current_epoch": self.state.current_epoch,
            "managers_loaded": self.state.managers_loaded,
            "errors": self.state.errors,
            "config": {
                "enable_rollback": self.config.enable_rollback,
                "enable_promotion": self.config.enable_promotion,
                "use_data_coordinator": self.config.use_data_coordinator,
                "checkpoint_interval": self.config.checkpoint_interval_steps,
            },
        }


# Singleton instance
_training_orchestrator: TrainingOrchestrator | None = None


def get_training_orchestrator(
    config: TrainingOrchestratorConfig | None = None,
) -> TrainingOrchestrator:
    """Get the global training orchestrator singleton.

    Args:
        config: Configuration (only used on first call)

    Returns:
        TrainingOrchestrator instance
    """
    global _training_orchestrator
    if _training_orchestrator is None:
        _training_orchestrator = TrainingOrchestrator(config)
    return _training_orchestrator


def reset_training_orchestrator() -> None:
    """Reset the training orchestrator singleton (for testing)."""
    global _training_orchestrator
    _training_orchestrator = None
