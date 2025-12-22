"""Integration Package for RingRift AI Self-Improvement Loop.

This package provides integration components that connect training,
evaluation, and optimization into a cohesive self-improvement system.

Usage Patterns (2025-12):
    These modules are primarily used by:
    - scripts/unified_ai_loop.py - Main training loop
    - scripts/unified_loop/*.py - Loop components
    - app/training/promotion_controller.py - Model promotion

    Import examples::

        # Pipeline feedback (for curriculum adjustment)
        from app.integration.pipeline_feedback import (
            PipelineFeedbackController,
            FeedbackAction,
            FeedbackSignal,
        )

        # Model lifecycle (for promotion/registry)
        from app.integration.model_lifecycle import (
            ModelLifecycleManager,
            LifecycleConfig,
        )

        # P2P cluster integration
        from app.integration.p2p_integration import (
            P2PIntegrationManager,
            SelfplayCoordinator,
            TrainingCoordinator,
        )

Modules:
    - unified_loop_extensions: Extensions for the unified AI loop
    - pipeline_feedback: Feedback loops for the pipeline orchestrator
    - model_lifecycle: Model lifecycle management and ModelSyncCoordinator
    - p2p_integration: P2P cluster REST API wrappers (Selfplay/Training/EvaluationCoordinator)
    - evaluation_curriculum_bridge: Evaluation → curriculum feedback loop
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from .model_lifecycle import LifecycleConfig, ModelLifecycleManager
    from .p2p_integration import P2PIntegrationConfig, P2PIntegrationManager
    from .pipeline_feedback import (
        FeedbackAction,
        FeedbackSignal,
        FeedbackSignalRouter,
        OpponentWinRateTracker,
        PipelineFeedbackController,
        create_feedback_controller,
        create_feedback_router,
        create_opponent_tracker,
    )
    from .unified_loop_extensions import ExtensionConfig, UnifiedLoopExtensions

__all__ = [
    # Evaluation → Curriculum bridge
    "EvaluationCurriculumBridge",
    "ExtensionConfig",
    "FeedbackAction",
    "FeedbackSignal",
    "FeedbackSignalRouter",
    "LifecycleConfig",
    # Auto Elo integration
    "ModelEloIntegration",
    # Model lifecycle
    "ModelLifecycleManager",
    "OpponentWinRateTracker",
    "P2PIntegrationConfig",
    # P2P integration
    "P2PIntegrationManager",
    # Pipeline feedback
    "PipelineFeedbackController",
    # Unified loop extensions
    "UnifiedLoopExtensions",
    "connect_to_cluster",
    "create_evaluation_bridge",
    "create_feedback_controller",
    "create_feedback_router",
    "create_full_selfplay_training_loop",
    "create_lifecycle_manager",
    "create_opponent_tracker",
    # Factory functions for lazy loading
    "get_auto_elo_integration",
    "get_evaluation_curriculum_bridge",
    "get_feedback_signal_router",
    "get_model_lifecycle_manager",
    "get_opponent_win_rate_tracker",
    "get_p2p_integration_manager",
    "get_pipeline_feedback_controller",
    "get_unified_loop_extensions",
    "integrate_evaluation_with_curriculum",
    "integrate_extensions",
    "integrate_feedback_with_selfplay",
    "integrate_lifecycle_with_p2p",
    "integrate_selfplay_with_training",
    "register_model_for_elo",
]


def get_unified_loop_extensions():
    """Get UnifiedLoopExtensions class."""
    from .unified_loop_extensions import UnifiedLoopExtensions
    return UnifiedLoopExtensions


def get_pipeline_feedback_controller():
    """Get PipelineFeedbackController class."""
    from .pipeline_feedback import PipelineFeedbackController
    return PipelineFeedbackController


def get_feedback_signal_router():
    """Get FeedbackSignalRouter class."""
    from .pipeline_feedback import FeedbackSignalRouter
    return FeedbackSignalRouter


def get_opponent_win_rate_tracker():
    """Get OpponentWinRateTracker class."""
    from .pipeline_feedback import OpponentWinRateTracker
    return OpponentWinRateTracker


def get_model_lifecycle_manager():
    """Get ModelLifecycleManager class."""
    from .model_lifecycle import ModelLifecycleManager
    return ModelLifecycleManager


def get_p2p_integration_manager():
    """Get P2PIntegrationManager class."""
    from .p2p_integration import P2PIntegrationManager
    return P2PIntegrationManager


def integrate_extensions(unified_loop, config=None):
    """Integrate extensions into an existing UnifiedAILoop instance."""
    from .unified_loop_extensions import integrate_extensions as _integrate
    return _integrate(unified_loop, config)


async def create_lifecycle_manager(config=None, start=True):
    """Create and optionally start a lifecycle manager."""
    from .model_lifecycle import create_lifecycle_manager as _create
    return await _create(config, start)


async def connect_to_cluster(base_url="http://localhost:8770", auth_token=None):
    """Connect to P2P cluster and return integration manager."""
    from .p2p_integration import connect_to_cluster as _connect
    return await _connect(base_url, auth_token)


def integrate_lifecycle_with_p2p(lifecycle_manager, p2p_manager):
    """Integrate lifecycle manager with P2P cluster."""
    from .p2p_integration import integrate_lifecycle_with_p2p as _integrate
    return _integrate(lifecycle_manager, p2p_manager)


def integrate_feedback_with_selfplay(feedback_router, selfplay_coordinator):
    """Integrate feedback signals with selfplay coordinator."""
    from .p2p_integration import integrate_feedback_with_selfplay as _integrate
    return _integrate(feedback_router, selfplay_coordinator)


def integrate_selfplay_with_training(
    selfplay_coordinator,
    training_triggers=None,
    training_scheduler=None,
    auto_trigger=True
):
    """Integrate selfplay game completion with training triggers.

    Creates event-driven pipeline:
    1. Selfplay coordinator reports game completions
    2. TrainingTriggers updates game counts
    3. When threshold reached, emits TRAINING_THRESHOLD_REACHED
    4. If auto_trigger=True, training is automatically scheduled
    """
    from .p2p_integration import integrate_selfplay_with_training as _integrate
    return _integrate(selfplay_coordinator, training_triggers, training_scheduler, auto_trigger)


def create_full_selfplay_training_loop(
    p2p_manager,
    training_scheduler=None,
    feedback_controller=None,
    auto_trigger=True
):
    """Create fully integrated selfplay → training loop.

    Sets up:
    1. Selfplay coordinator with feedback integration
    2. Event-driven training triggers
    3. Curriculum weight adjustment
    4. Auto-training when thresholds met
    """
    from .p2p_integration import create_full_selfplay_training_loop as _create
    return _create(p2p_manager, training_scheduler, feedback_controller, auto_trigger)


def get_evaluation_curriculum_bridge():
    """Get EvaluationCurriculumBridge class."""
    from .evaluation_curriculum_bridge import EvaluationCurriculumBridge
    return EvaluationCurriculumBridge


def create_evaluation_bridge(feedback_controller=None, feedback_router=None, selfplay_coordinator=None):
    """Create evaluation-curriculum bridge."""
    from .evaluation_curriculum_bridge import create_evaluation_bridge as _create
    return _create(feedback_controller, feedback_router, selfplay_coordinator)


def integrate_evaluation_with_curriculum(feedback_controller, selfplay_coordinator):
    """One-line evaluation → curriculum integration."""
    from .evaluation_curriculum_bridge import integrate_evaluation_with_curriculum as _integrate
    return _integrate(feedback_controller, selfplay_coordinator)


def get_auto_elo_integration():
    """Get ModelEloIntegration class."""
    from .auto_elo_integration import ModelEloIntegration
    return ModelEloIntegration


def register_model_for_elo(model_path):
    """Register a model for automatic Elo evaluation."""
    from .auto_elo_integration import register_model_for_elo as _register
    return _register(model_path)
