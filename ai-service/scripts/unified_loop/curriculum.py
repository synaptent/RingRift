"""Unified Loop Curriculum Services.

This module contains curriculum-related services for the unified AI loop:
- AdaptiveCurriculum: Elo-weighted training curriculum with feedback integration

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import json
import statistics
import time
from typing import TYPE_CHECKING, Dict, Optional

from .config import CurriculumConfig, DataEvent, DataEventType
from app.utils.paths import AI_SERVICE_ROOT

# Path for curriculum weights shared with P2P orchestrator (Phase 3.1)
CURRICULUM_WEIGHTS_PATH = AI_SERVICE_ROOT / "data" / "curriculum_weights.json"

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


def export_curriculum_weights(weights: Dict[str, float]) -> bool:
    """Export curriculum weights to JSON file for P2P orchestrator consumption.

    Phase 3.1: This allows the P2P orchestrator to prioritize selfplay jobs
    based on curriculum learning weights from the unified AI loop.

    Args:
        weights: Dictionary of config_key -> weight multiplier

    Returns:
        True if export succeeded, False otherwise
    """
    try:
        CURRICULUM_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": weights,
            "updated_at": time.time(),
            "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Atomic write: write to temp file then rename
        temp_path = CURRICULUM_WEIGHTS_PATH.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(CURRICULUM_WEIGHTS_PATH)
        return True
    except Exception as e:
        print(f"[Curriculum] Failed to export weights: {e}")
        return False


def load_curriculum_weights() -> Dict[str, float]:
    """Load curriculum weights from JSON file.

    Used by P2P orchestrator to read curriculum priorities.

    Returns:
        Dictionary of config_key -> weight multiplier, or empty dict if unavailable
    """
    try:
        if not CURRICULUM_WEIGHTS_PATH.exists():
            return {}
        with open(CURRICULUM_WEIGHTS_PATH) as f:
            data = json.load(f)
        # Check staleness (weights older than 2 hours are considered stale)
        updated_at = data.get("updated_at", 0)
        if time.time() - updated_at > 7200:
            return {}  # Stale weights
        return data.get("weights", {})
    except Exception:
        return {}


class AdaptiveCurriculum:
    """Manages Elo-weighted training curriculum with feedback integration."""

    def __init__(self, config: CurriculumConfig, state: "UnifiedLoopState", event_bus: "EventBus"):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback: Optional["PipelineFeedbackController"] = None

    def set_feedback_controller(self, feedback: "PipelineFeedbackController"):
        """Set the feedback controller for curriculum adjustments."""
        self.feedback = feedback

    async def rebalance_weights(self) -> Dict[str, float]:
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
