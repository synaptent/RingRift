"""Coordination Mixins - Reusable behavior components.

This package provides mixins that can be used to add common functionality
to coordinators, daemons, and managers without requiring inheritance from
a specific base class.

December 2025: Created as part of Phase 2 consolidation to reduce code
duplication across 76+ files implementing similar patterns.

Available mixins:
- HealthCheckMixin: Standard health check implementation (~600 LOC savings)
- LifecycleMixin: Async lifecycle management (start/stop/shutdown)
- EventSubscriptionMixin: Event subscription management

Base Classes (for mixin families):
- PipelineMixinBase: Base for DataPipelineOrchestrator mixins (4 mixins)
- SyncMixinBase: Base for AutoSyncDaemon mixins (4 mixins)
"""

from app.coordination.mixins.health_check_mixin import (
    HealthCheckMixin,
)
from app.coordination.mixins.lifecycle_mixin import (
    EventSubscriptionMixin,
    LifecycleMixin,
    LifecycleState,
)
from app.coordination.pipeline_mixin_base import (
    DataPipelineOrchestratorProtocol,
    PipelineMixinBase,
)
from app.coordination.sync_mixin_base import (
    AutoSyncDaemonProtocol,
    SyncMixinBase,
)

__all__ = [
    # Health check
    "HealthCheckMixin",
    # Lifecycle
    "EventSubscriptionMixin",
    "LifecycleMixin",
    "LifecycleState",
    # Pipeline mixins base
    "DataPipelineOrchestratorProtocol",
    "PipelineMixinBase",
    # Sync mixins base
    "AutoSyncDaemonProtocol",
    "SyncMixinBase",
]
