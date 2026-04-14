"""Coordination Mixins - Reusable behavior components.

This package provides mixins that can be used to add common functionality
to coordinators, daemons, and managers without requiring inheritance from
a specific base class.

December 2025: Created as part of Phase 2 consolidation to reduce code
duplication across 76+ files implementing similar patterns.

January 2026: Added ImportDaemonMixin for consolidated import/download logic.

Available mixins:
- HealthCheckMixin: Standard health check implementation (~600 LOC savings)
- LifecycleMixin: Async lifecycle management (start/stop/shutdown)
- EventSubscriptionMixin: Event subscription management
- ImportDaemonMixin: Consolidated import/download patterns (~200 LOC savings)

Base Classes (for mixin families):
- PipelineMixinBase: Base for DataPipelineOrchestrator mixins (4 mixins)
- SyncMixinBase: Base for AutoSyncDaemon mixins (4 mixins)
"""

from app.coordination.mixins.health_check_mixin import (
    HealthCheckMixin,
)
from app.coordination.mixins.import_mixin import (
    DownloadProgress,
    ImportDaemonMixin,
    ImportValidationResult,
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
    # Import/download
    "DownloadProgress",
    "ImportDaemonMixin",
    "ImportValidationResult",
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


def __dir__() -> list[str]:
    """Expose the intended package surface for discoverability and tests."""

    return sorted(set(globals()) | set(__all__))
