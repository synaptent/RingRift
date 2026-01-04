"""Feedback loop controller mixins package.

January 2026 Sprint 17.9: Phase 4 decomposition of feedback_loop_controller.py.

This package contains mixin classes that extract focused functionality from
the main FeedbackLoopController (4,200 LOC) into smaller, testable modules.

Mixins:
- FeedbackClusterHealthMixin: Cluster health and capacity event handlers
"""

from __future__ import annotations

from app.coordination.feedback.cluster_health_mixin import FeedbackClusterHealthMixin

__all__ = [
    "FeedbackClusterHealthMixin",
]
