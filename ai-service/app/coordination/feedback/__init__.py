"""Feedback loop controller mixins package.

January 2026 Sprint 17.9: Phase 4 decomposition of feedback_loop_controller.py.

This package contains mixin classes that extract focused functionality from
the main FeedbackLoopController (4,200 LOC) into smaller, testable modules.

Mixins:
- FeedbackClusterHealthMixin: Cluster health and capacity event handlers
- ExplorationBoostMixin: Exploration boost logic (anomaly, stall, decay)
- QualityFeedbackMixin: Quality assessment and quality -> intensity/curriculum feedback
- EloVelocityAdaptationMixin: Elo velocity tracking and adaptive training signals
- TrainingCurriculumFeedbackMixin: Training -> curriculum feedback and Elo recording
- LossMonitoringMixin: Loss anomaly detection and trend monitoring
- EvaluationFeedbackMixin: Evaluation triggering, gauntlet handling, retry logic
- RegressionHandlingMixin: Regression detection response and curriculum rollback
- SelfplayFeedbackMixin: Selfplay completion, rate changes, quality assessment
"""

from __future__ import annotations

from app.coordination.feedback.cluster_health_mixin import FeedbackClusterHealthMixin
from app.coordination.feedback.exploration_boost import ExplorationBoostMixin
from app.coordination.feedback.quality_feedback import QualityFeedbackMixin
from app.coordination.feedback.elo_velocity_mixin import EloVelocityAdaptationMixin
from app.coordination.feedback.training_curriculum_mixin import TrainingCurriculumFeedbackMixin
from app.coordination.feedback.loss_monitoring_mixin import LossMonitoringMixin
from app.coordination.feedback.evaluation_feedback_mixin import EvaluationFeedbackMixin
from app.coordination.feedback.regression_handling_mixin import RegressionHandlingMixin
from app.coordination.feedback.selfplay_feedback_mixin import SelfplayFeedbackMixin

__all__ = [
    "FeedbackClusterHealthMixin",
    "ExplorationBoostMixin",
    "QualityFeedbackMixin",
    "EloVelocityAdaptationMixin",
    "TrainingCurriculumFeedbackMixin",
    "LossMonitoringMixin",
    "EvaluationFeedbackMixin",
    "RegressionHandlingMixin",
    "SelfplayFeedbackMixin",
]


def __dir__() -> list[str]:
    """Expose the intended package surface for discoverability and tests."""

    return sorted(set(globals()) | set(__all__))
