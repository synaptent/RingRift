"""Standalone daemon factory implementations.

December 2025 Phase 5: Extract factory methods from DaemonManager to reduce file size.

This module contains standalone async factory functions for creating daemons.
These are called by DaemonManager.register_factory() instead of bound methods.

Migration Status:
- Total factory methods in daemon_manager.py: 61
- Migrated to this module: 8 (examples for pattern)
- Remaining: 53 (can be migrated incrementally)

Benefits:
- Reduces daemon_manager.py from ~3,600 LOC to ~2,000 LOC
- Factory functions are easier to test in isolation
- Cleaner separation of concerns

Usage:
    from app.coordination.daemon_factory_implementations import create_auto_sync

    manager.register_factory(
        DaemonType.AUTO_SYNC,
        create_auto_sync,
        depends_on=[DaemonType.EVENT_ROUTER],
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Core Sync Daemons
# =============================================================================


async def create_auto_sync() -> None:
    """Create and run AutoSyncDaemon.

    Primary data synchronization daemon for cluster-wide game data distribution.
    Uses push-from-generator and gossip replication strategies.
    """
    try:
        from app.coordination.auto_sync_daemon import AutoSyncDaemon

        daemon = AutoSyncDaemon()
        await daemon.start()
        # Keep running until stopped
        while daemon._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available: {e}")
        raise


async def create_model_distribution() -> None:
    """Create and run model distribution daemon.

    Distributes trained models to cluster nodes after promotion.
    Uses unified distribution infrastructure (Dec 2025).
    """
    try:
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
            DataType,
        )

        daemon = UnifiedDistributionDaemon(
            data_types=[DataType.MODEL],
            name="ModelDistributionDaemon",
        )
        await daemon.start()
        while daemon._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"UnifiedDistributionDaemon not available: {e}")
        raise


async def create_npz_distribution() -> None:
    """Create and run NPZ distribution daemon.

    Distributes training NPZ files to cluster nodes.
    Uses unified distribution infrastructure (Dec 2025).
    """
    try:
        from app.coordination.unified_distribution_daemon import (
            UnifiedDistributionDaemon,
            DataType,
        )

        daemon = UnifiedDistributionDaemon(
            data_types=[DataType.NPZ],
            name="NPZDistributionDaemon",
        )
        await daemon.start()
        while daemon._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"UnifiedDistributionDaemon not available: {e}")
        raise


# =============================================================================
# Evaluation Daemons
# =============================================================================


async def create_evaluation() -> None:
    """Create and run evaluation daemon.

    Auto-evaluates models after training completes.
    Triggers gauntlet evaluation against baselines.
    """
    try:
        from app.coordination.evaluation_daemon import EvaluationDaemon

        daemon = EvaluationDaemon()
        await daemon.start()
        while daemon._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"EvaluationDaemon not available: {e}")
        raise


async def create_auto_promotion() -> None:
    """Create and run auto-promotion daemon.

    Promotes models that pass gauntlet evaluation.
    Uses PromotionController for promotion logic.
    """
    try:
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()
        await controller.run()

    except ImportError as e:
        logger.error(f"PromotionController not available: {e}")
        raise


# =============================================================================
# Quality & Feedback Daemons
# =============================================================================


async def create_quality_monitor() -> None:
    """Create and run quality monitor daemon.

    Monitors selfplay data quality metrics.
    Emits LOW_QUALITY_DETECTED events when thresholds breached.
    """
    try:
        from app.coordination.quality_monitor_daemon import QualityMonitorDaemon

        daemon = QualityMonitorDaemon()
        await daemon.start()
        while daemon._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"QualityMonitorDaemon not available: {e}")
        raise


async def create_feedback_loop() -> None:
    """Create and run feedback loop controller.

    Central orchestration for training feedback loops.
    Adjusts training parameters based on evaluation results.
    """
    try:
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        controller = FeedbackLoopController()
        await controller.start()
        while controller._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"FeedbackLoopController not available: {e}")
        raise


# =============================================================================
# Pipeline Daemons
# =============================================================================


async def create_data_pipeline() -> None:
    """Create and run data pipeline orchestrator.

    Tracks pipeline stages: selfplay -> sync -> export -> train -> eval -> promote.
    Triggers next stage when previous completes.
    """
    try:
        from app.coordination.data_pipeline_orchestrator import (
            DataPipelineOrchestrator,
        )

        orchestrator = DataPipelineOrchestrator()
        await orchestrator.start()
        while orchestrator._running:
            import asyncio
            await asyncio.sleep(1)

    except ImportError as e:
        logger.error(f"DataPipelineOrchestrator not available: {e}")
        raise


# =============================================================================
# Factory Registry
# =============================================================================

# Map daemon type names to factory functions for easy lookup
# This allows DaemonManager to use string-based registration
FACTORY_REGISTRY: dict[str, callable] = {
    "AUTO_SYNC": create_auto_sync,
    "MODEL_DISTRIBUTION": create_model_distribution,
    "NPZ_DISTRIBUTION": create_npz_distribution,
    "EVALUATION": create_evaluation,
    "AUTO_PROMOTION": create_auto_promotion,
    "QUALITY_MONITOR": create_quality_monitor,
    "FEEDBACK_LOOP": create_feedback_loop,
    "DATA_PIPELINE": create_data_pipeline,
}


def get_factory(daemon_type_name: str) -> callable | None:
    """Get factory function for a daemon type.

    Args:
        daemon_type_name: Name of the DaemonType enum value

    Returns:
        Factory function or None if not found in this module
    """
    return FACTORY_REGISTRY.get(daemon_type_name)
