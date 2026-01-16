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
"""

from __future__ import annotations

from app.coordination.feedback.cluster_health_mixin import FeedbackClusterHealthMixin
from app.coordination.feedback.exploration_boost import ExplorationBoostMixin
from app.coordination.feedback.quality_feedback import QualityFeedbackMixin
from app.coordination.feedback.elo_velocity_mixin import EloVelocityAdaptationMixin
from app.coordination.feedback.training_curriculum_mixin import TrainingCurriculumFeedbackMixin

__all__ = [
    "FeedbackClusterHealthMixin",
    "ExplorationBoostMixin",
    "QualityFeedbackMixin",
    "EloVelocityAdaptationMixin",
    "TrainingCurriculumFeedbackMixin",
]
