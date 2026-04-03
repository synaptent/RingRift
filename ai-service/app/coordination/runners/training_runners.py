"""Training, Pipeline, Evaluation and Promotion daemon runners.

February 2026: Extracted from daemon_runners.py.

Contains runners for:
- Training & Pipeline Daemons (data_pipeline through architecture_feedback)
- Evaluation & Promotion Daemons (evaluation_daemon through backlog_evaluation)
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.coordination.runners import _wait_for_daemon

logger = logging.getLogger(__name__)


# =============================================================================
# Training & Pipeline Daemons
# =============================================================================


async def create_data_pipeline() -> None:
    """Create and run data pipeline orchestrator (December 2025)."""
    try:
        from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

        orchestrator = get_pipeline_orchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"DataPipelineOrchestrator not available: {e}")
        raise


async def create_continuous_training_loop() -> None:
    """Deprecated. This daemon type has been removed.

    No ImportError path is needed because this runner is now a no-op.
    """
    logger.debug("Deprecated daemon runner called (no-op): CONTINUOUS_TRAINING_LOOP")


async def create_selfplay_scheduler() -> None:
    """Backward-compatible alias for the scheduler daemon surface.

    ImportError handling delegates to create_selfplay_coordinator().
    """
    return await create_selfplay_coordinator()


async def create_selfplay_coordinator() -> None:
    """Initialize selfplay scheduler singleton (December 2025).

    Note: SelfplayScheduler is a utility class, not a daemon with a lifecycle.
    It provides priority-based selfplay scheduling decisions to other daemons
    like IdleResourceDaemon. This runner initializes the singleton and wires
    up its event subscriptions.
    """
    try:
        from app.coordination.selfplay_scheduler import get_selfplay_scheduler

        # Get singleton and wire up event subscriptions
        scheduler = get_selfplay_scheduler()
        logger.info(
            f"[SelfplayCoordinator] Initialized scheduler with {len(scheduler._config_priorities)} configs"
        )
        # SelfplayScheduler doesn't have a lifecycle - it's used by other daemons
        # Keep running indefinitely
        while True:
            await asyncio.sleep(3600)  # Sleep 1 hour, wake up to check for shutdown
    except ImportError as e:
        logger.error(f"SelfplayScheduler not available: {e}")
        raise


async def create_training_coordinator() -> None:
    """Create and run the TrainingCoordinator daemon wrapper."""
    try:
        from app.coordination.training_coordinator import (
            get_training_coordinator_daemon,
        )

        daemon = get_training_coordinator_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TrainingCoordinatorDaemon not available: {e}")
        raise


async def create_training_trigger() -> None:
    """Create and run training trigger daemon (December 2025)."""
    try:
        from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

        daemon = TrainingTriggerDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TrainingTriggerDaemon not available: {e}")
        raise


async def create_auto_export() -> None:
    """Create and run auto export daemon (December 2025)."""
    try:
        from app.coordination.auto_export_daemon import AutoExportDaemon

        daemon = AutoExportDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoExportDaemon not available: {e}")
        raise


async def create_tournament_daemon() -> None:
    """Create and run tournament daemon (December 2025)."""
    try:
        from app.coordination.tournament_daemon import TournamentDaemon

        daemon = TournamentDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TournamentDaemon not available: {e}")
        raise


async def create_nnue_training() -> None:
    """Create and run NNUE training daemon (December 2025).

    Automatically trains NNUE models when game thresholds are met.
    Per-config thresholds: hex8_2p=5000, hex8_4p=10000, square19_2p=2000.
    Subscribes to: NEW_GAMES_AVAILABLE, CONSOLIDATION_COMPLETE, DATA_SYNC_COMPLETED.
    """
    try:
        from app.coordination.nnue_training_daemon import NNUETrainingDaemon

        daemon = NNUETrainingDaemon.get_instance()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NNUETrainingDaemon not available: {e}")
        raise


async def create_architecture_feedback() -> None:
    """Create and run architecture feedback controller (December 2025).

    Bridges evaluation results to selfplay allocation by tracking architecture
    performance. Enforces 10% minimum allocation per architecture.
    Subscribes to: EVALUATION_COMPLETED, TRAINING_COMPLETED.
    Emits: ARCHITECTURE_WEIGHTS_UPDATED.
    """
    try:
        from app.coordination.architecture_feedback_controller import (
            ArchitectureFeedbackController,
        )

        controller = ArchitectureFeedbackController.get_instance()
        await controller.start()
        await _wait_for_daemon(controller)
    except ImportError as e:
        logger.error(f"ArchitectureFeedbackController not available: {e}")
        raise


async def create_parity_validation() -> None:
    """Create and run parity validation daemon (December 30, 2025).

    Runs on coordinator (which has Node.js) to validate TS/Python parity for
    canonical databases. Cluster nodes generate databases with "pending_gate"
    status because they lack npx. This daemon validates them and stores TS
    reference hashes, enabling hash-based validation on cluster nodes.

    Subscribes to: DATA_SYNC_COMPLETED (to validate newly synced databases)
    Emits: PARITY_VALIDATION_COMPLETED
    """
    try:
        from app.coordination.parity_validation_daemon import (
            get_parity_validation_daemon,
        )

        daemon = get_parity_validation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ParityValidationDaemon not available: {e}")


async def create_elo_progress() -> None:
    """Create and run Elo progress tracking daemon (December 31, 2025).

    Periodically snapshots the best model's Elo for each config to track
    improvement over time. Provides evidence of training loop effectiveness.

    - Takes snapshots hourly by default
    - Also triggers on EVALUATION_COMPLETED and MODEL_PROMOTED events
    - Stores data in elo_progress.db for trend analysis

    Subscribes to: EVALUATION_COMPLETED, MODEL_PROMOTED
    """
    try:
        from app.coordination.elo_progress_daemon import get_elo_progress_daemon

        daemon = get_elo_progress_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"EloProgressDaemon not available: {e}")
        raise


# =============================================================================
# Evaluation & Promotion Daemons
# =============================================================================


async def create_evaluation_daemon() -> None:
    """Create and run evaluation daemon (December 2025)."""
    try:
        from app.coordination.evaluation_daemon import get_evaluation_daemon

        daemon = get_evaluation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"EvaluationDaemon not available: {e}")
        raise


async def create_auto_promotion() -> None:
    """Create and run auto-promotion daemon (December 2025)."""
    try:
        from app.coordination.auto_promotion_daemon import AutoPromotionDaemon

        daemon = AutoPromotionDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoPromotionDaemon not available: {e}")
        raise


async def create_unified_promotion() -> None:
    """Create and run unified promotion daemon (December 2025).

    PromotionController is event-driven and sets up subscriptions in __init__.
    We just need to instantiate and keep it alive.
    """
    try:
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()
        # PromotionController subscribes to events in setup_event_subscriptions()
        # called from __init__ - no start() method needed
        logger.info("PromotionController initialized and subscribed to events")

        # Keep alive by waiting indefinitely
        while True:
            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"PromotionController not available: {e}")
        raise


async def create_gauntlet_feedback() -> None:
    """Create and run gauntlet feedback controller (December 2025)."""
    try:
        from app.coordination.gauntlet_feedback_controller import (
            GauntletFeedbackController,
        )

        controller = GauntletFeedbackController()
        await controller.start()
        await _wait_for_daemon(controller)
    except ImportError as e:
        logger.error(f"GauntletFeedbackController not available: {e}")
        raise


async def create_backlog_evaluation() -> None:
    """Create and run backlog evaluation daemon (Sprint 15 - Jan 3, 2026).

    Discovers OWC models and queues them for Elo evaluation.
    """
    try:
        from app.coordination.backlog_evaluation_daemon import (
            BacklogEvaluationDaemon,
        )

        daemon = BacklogEvaluationDaemon.get_instance()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"BacklogEvaluationDaemon not available: {e}")
        raise
