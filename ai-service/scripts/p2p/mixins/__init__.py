"""P2P Mixins Package.

January 2026 - Sprint 17: Shared mixin classes for P2P orchestrator components.

Mixins:
- HealthTrackingMixin: Per-entity failure tracking with backoff and health scoring
- LeadershipHealthMixin: Voter health and quorum monitoring

Usage:
    from scripts.p2p.mixins import HealthTrackingMixin, HealthTrackingConfig
    from scripts.p2p.mixins import LeadershipHealthMixin

    class MyLoop(BaseLoop, HealthTrackingMixin):
        def __init__(self):
            super().__init__()
            self.init_health_tracking(HealthTrackingConfig(failure_threshold=5))

    class P2POrchestrator(LeadershipHealthMixin, ...):
        pass
"""

from .health_tracking import (
    HealthTrackingMixin,
    HealthTrackingConfig,
    EntityHealthSummary,
    EntityHealthState,
)
from .leadership_health_mixin import LeadershipHealthMixin

__all__ = [
    "HealthTrackingMixin",
    "HealthTrackingConfig",
    "EntityHealthSummary",
    "EntityHealthState",
    "LeadershipHealthMixin",
]
