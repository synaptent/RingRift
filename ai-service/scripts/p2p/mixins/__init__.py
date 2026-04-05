"""P2P Mixins Package.

January 2026 - Sprint 17: Shared mixin classes for P2P orchestrator components.

Mixins:
- HealthTrackingMixin: Per-entity failure tracking with backoff and health scoring
- LeadershipHealthMixin: Voter health and quorum monitoring
- LeadershipTransitionsMixin: Step-down and state machine transitions
- AdvertiseValidationMixin: IP validation and advertise host management
- HeartbeatLoopMixin: Heartbeat loop and bootstrap methods (April 2026 - Target 4)
- TrainingPipelineMixin: AlphaZero-style training loop coordination (April 2026 - Target 3)

Usage:
    from scripts.p2p.mixins import HealthTrackingMixin, HealthTrackingConfig
    from scripts.p2p.mixins import LeadershipHealthMixin
    from scripts.p2p.mixins import AdvertiseValidationMixin
    from scripts.p2p.mixins import HeartbeatLoopMixin
    from scripts.p2p.mixins import TrainingPipelineMixin

    class MyLoop(BaseLoop, HealthTrackingMixin):
        def __init__(self):
            super().__init__()
            self.init_health_tracking(HealthTrackingConfig(failure_threshold=5))

    class P2POrchestrator(TrainingPipelineMixin, HeartbeatLoopMixin, LeadershipHealthMixin, AdvertiseValidationMixin, ...):
        pass
"""

from .health_tracking import (
    HealthTrackingMixin,
    HealthTrackingConfig,
    EntityHealthSummary,
    EntityHealthState,
)
from .heartbeat_loop_mixin import HeartbeatLoopMixin
from .leadership_health_mixin import LeadershipHealthMixin
from .leadership_transitions_mixin import LeadershipTransitionsMixin
from .advertise_validation_mixin import AdvertiseValidationMixin
from .training_pipeline_mixin import TrainingPipelineMixin

__all__ = [
    "HealthTrackingMixin",
    "HealthTrackingConfig",
    "EntityHealthSummary",
    "EntityHealthState",
    "HeartbeatLoopMixin",
    "LeadershipHealthMixin",
    "LeadershipTransitionsMixin",
    "AdvertiseValidationMixin",
    "TrainingPipelineMixin",
]
