"""
Integration Package for RingRift AI Self-Improvement Loop.

This package provides integration components that connect all the
training, evaluation, and optimization components into a cohesive
self-improvement system.

Modules:
- unified_loop_extensions: Extensions for the unified AI loop
- pipeline_feedback: Feedback loops for the pipeline orchestrator
- model_lifecycle: Model lifecycle management
- p2p_integration: P2P cluster integration
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from .unified_loop_extensions import UnifiedLoopExtensions, ExtensionConfig
    from .pipeline_feedback import PipelineFeedbackController, FeedbackAction
    from .model_lifecycle import ModelLifecycleManager, LifecycleConfig
    from .p2p_integration import P2PIntegrationManager, P2PIntegrationConfig

__all__ = [
    # Unified loop extensions
    "UnifiedLoopExtensions",
    "ExtensionConfig",
    "integrate_extensions",

    # Pipeline feedback
    "PipelineFeedbackController",
    "FeedbackAction",

    # Model lifecycle
    "ModelLifecycleManager",
    "LifecycleConfig",
    "create_lifecycle_manager",

    # P2P integration
    "P2PIntegrationManager",
    "P2PIntegrationConfig",
    "connect_to_cluster",
]


def get_unified_loop_extensions():
    """Get UnifiedLoopExtensions class."""
    from .unified_loop_extensions import UnifiedLoopExtensions
    return UnifiedLoopExtensions


def get_pipeline_feedback_controller():
    """Get PipelineFeedbackController class."""
    from .pipeline_feedback import PipelineFeedbackController
    return PipelineFeedbackController


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
