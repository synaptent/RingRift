"""Unified Loop Curriculum Services.

This module contains curriculum-related services for the unified AI loop:
- AdaptiveCurriculum: Elo-weighted training curriculum with feedback integration

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).

NOTE: The load_curriculum_weights and export_curriculum_weights functions have been
moved to app.coordination.curriculum_weights. They are re-exported here for
backwards compatibility.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Optional

# Re-export from canonical location for backwards compatibility
from app.coordination.curriculum_weights import (
    CURRICULUM_WEIGHTS_PATH,
    export_curriculum_weights,
    load_curriculum_weights,
)

from .config import CurriculumConfig, DataEvent, DataEventType

if TYPE_CHECKING:
    from unified_ai_loop import EventBus, UnifiedLoopState

    from app.integration.pipeline_feedback import PipelineFeedbackController

# Optional ELO service import
try:
    from app.training.elo_service import get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None


class AdaptiveCurriculum:
    """Manages Elo-weighted training curriculum with feedback integration."""

    def __init__(self, config: CurriculumConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback: PipelineFeedbackController | None = None

    def set_feedback_controller(self, feedback: PipelineFeedbackController):
        """Set the feedback controller for curriculum adjustments."""
        self.feedback = feedback

    async def rebalance_weights(self) -> dict[str, float]:
        """Recompute training weights based on Elo performance."""
        if not self.config.adaptive:
            return {}

        try:
            # Query Elo by config via centralized service
            if get_elo_service is None:
                return {}

            elo_svc = get_elo_service()

            # Get best Elo for each config
            # Include both ringrift_* models and *_nn_baseline* models
            rows = elo_svc.execute_query("""
                SELECT board_type, num_players, MAX(rating) as best_elo
                FROM elo_ratings
                WHERE (participant_id LIKE 'ringrift_%' OR participant_id LIKE '%_nn_baseline%')
                GROUP BY board_type, num_players
            """)

            if not rows:
                return {}

            elo_by_config = {
                f"{row[0]}_{row[1]}p": row[2]
                for row in rows
            }

            # Compute weights based on deviation from median
            elos = list(elo_by_config.values())
            median_elo = statistics.median(elos)

            new_weights = {}
            for config_key, elo in elo_by_config.items():
                # Boost weight for underperforming configs based on Elo
                deficit = median_elo - elo
                elo_weight = 1.0 + (deficit / 200.0)

                # Merge with feedback controller weights if available
                if self.feedback:
                    feedback_weight = self.feedback.get_curriculum_weight(config_key)
                    # Average Elo-based and feedback-based weights
                    weight = (elo_weight + feedback_weight) / 2.0
                else:
                    weight = elo_weight

                # Clamp to configured range
                weight = max(self.config.min_weight_multiplier,
                           min(self.config.max_weight_multiplier, weight))

                new_weights[config_key] = weight

            # Update state
            self.state.curriculum_weights = new_weights
            self.state.last_curriculum_rebalance = time.time()

            # Update config states
            for config_key, weight in new_weights.items():
                if config_key in self.state.configs:
                    self.state.configs[config_key].training_weight = weight

            # Phase 3.1: Export weights to JSON for P2P orchestrator consumption
            export_curriculum_weights(new_weights)

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={"weights": new_weights}
            ))

            return new_weights

        except Exception as e:
            print(f"[AdaptiveCurriculum] Error rebalancing: {e}")
            return {}
