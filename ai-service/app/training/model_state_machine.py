"""Model Lifecycle State Machine for RingRift AI.

Provides formal state machine for model promotion lifecycle:
- Enforces valid state transitions
- Prevents invalid promotions/rollbacks
- Tracks transition history with audit trail
- Integrates with PromotionController

Usage:
    from app.training.model_state_machine import (
        ModelLifecycleStateMachine,
        ModelState,
        get_model_lifecycle,
    )

    lifecycle = get_model_lifecycle()

    # Register a new model
    lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

    # Transition through lifecycle
    lifecycle.transition("model_v42", ModelState.EVALUATING)
    lifecycle.transition("model_v42", ModelState.STAGING)
    lifecycle.transition("model_v42", ModelState.PRODUCTION)

    # Rollback if needed
    lifecycle.rollback("model_v42", "model_v41")
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.state_machine import InvalidTransitionError, State, Transition

logger = logging.getLogger(__name__)

__all__ = [
    "ModelLifecycleStateMachine",
    "ModelState",
    "ModelStateRecord",
    "get_model_lifecycle",
    "reset_model_lifecycle",
]


# =============================================================================
# Model States
# =============================================================================

class ModelState(Enum):
    """States in model lifecycle."""
    # Initial states
    TRAINING = "training"          # Model is being trained
    TRAINED = "trained"            # Training complete, awaiting evaluation

    # Evaluation states
    EVALUATING = "evaluating"      # Model under evaluation
    EVALUATED = "evaluated"        # Evaluation complete

    # Promotion states
    STAGING = "staging"            # In staging environment
    SHADOW = "shadow"              # Running in shadow mode (parallel with prod)
    PRODUCTION = "production"      # Active production model

    # Terminal states
    ARCHIVED = "archived"          # Retired/archived model
    ROLLED_BACK = "rolled_back"    # Rolled back from production
    FAILED = "failed"              # Training/evaluation failed


# Define valid state transitions
MODEL_TRANSITIONS = [
    # Training flow
    Transition(
        from_state=State(ModelState.TRAINING.value),
        to_state=State(ModelState.TRAINED.value),
        name="training_complete",
    ),
    Transition(
        from_state=State(ModelState.TRAINING.value),
        to_state=State(ModelState.FAILED.value),
        name="training_failed",
    ),

    # Evaluation flow
    Transition(
        from_state=State(ModelState.TRAINED.value),
        to_state=State(ModelState.EVALUATING.value),
        name="start_evaluation",
    ),
    Transition(
        from_state=State(ModelState.EVALUATING.value),
        to_state=State(ModelState.EVALUATED.value),
        name="evaluation_complete",
    ),
    Transition(
        from_state=State(ModelState.EVALUATING.value),
        to_state=State(ModelState.FAILED.value),
        name="evaluation_failed",
    ),

    # Promotion flow
    Transition(
        from_state=State(ModelState.EVALUATED.value),
        to_state=State(ModelState.STAGING.value),
        name="promote_to_staging",
    ),
    Transition(
        from_state=State(ModelState.EVALUATED.value),
        to_state=State(ModelState.ARCHIVED.value),
        name="archive_evaluated",
    ),
    Transition(
        from_state=State(ModelState.STAGING.value),
        to_state=State(ModelState.SHADOW.value),
        name="start_shadow",
    ),
    Transition(
        from_state=State(ModelState.STAGING.value),
        to_state=State(ModelState.PRODUCTION.value),
        name="promote_to_production",
    ),
    Transition(
        from_state=State(ModelState.STAGING.value),
        to_state=State(ModelState.ARCHIVED.value),
        name="archive_staging",
    ),
    Transition(
        from_state=State(ModelState.SHADOW.value),
        to_state=State(ModelState.PRODUCTION.value),
        name="shadow_to_production",
    ),
    Transition(
        from_state=State(ModelState.SHADOW.value),
        to_state=State(ModelState.ARCHIVED.value),
        name="archive_shadow",
    ),

    # Rollback/archive from production
    Transition(
        from_state=State(ModelState.PRODUCTION.value),
        to_state=State(ModelState.ROLLED_BACK.value),
        name="rollback",
    ),
    Transition(
        from_state=State(ModelState.PRODUCTION.value),
        to_state=State(ModelState.ARCHIVED.value),
        name="archive_production",
    ),

    # Recovery from failed state
    Transition(
        from_state=State(ModelState.FAILED.value),
        to_state=State(ModelState.TRAINING.value),
        name="retry_training",
    ),
    Transition(
        from_state=State(ModelState.FAILED.value),
        to_state=State(ModelState.ARCHIVED.value),
        name="archive_failed",
    ),

    # Re-promotion from rolled back
    Transition(
        from_state=State(ModelState.ROLLED_BACK.value),
        to_state=State(ModelState.EVALUATING.value),
        name="re_evaluate",
    ),
    Transition(
        from_state=State(ModelState.ROLLED_BACK.value),
        to_state=State(ModelState.ARCHIVED.value),
        name="archive_rolled_back",
    ),
]


# =============================================================================
# Model State Record
# =============================================================================

@dataclass
class ModelStateRecord:
    """Record of a model's state and history."""
    model_id: str
    current_state: ModelState
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_transition(
        self,
        from_state: ModelState,
        to_state: ModelState,
        reason: str = "",
        triggered_by: str = "",
    ) -> None:
        """Record a state transition."""
        self.history.append({
            "from_state": from_state.value,
            "to_state": to_state.value,
            "timestamp": time.time(),
            "reason": reason,
            "triggered_by": triggered_by,
        })
        self.current_state = to_state
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "current_state": self.current_state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": self.history,
            "metadata": self.metadata,
        }


# =============================================================================
# Model Lifecycle State Machine
# =============================================================================

class ModelLifecycleStateMachine:
    """State machine for managing model lifecycle transitions.

    Tracks model states and enforces valid transitions.

    Example:
        lifecycle = ModelLifecycleStateMachine()

        # Register model
        lifecycle.register_model("model_v42")

        # Training complete
        lifecycle.transition("model_v42", ModelState.TRAINED)

        # Start evaluation
        lifecycle.transition("model_v42", ModelState.EVALUATING)

        # Check current state
        state = lifecycle.get_state("model_v42")

        # Get valid next states
        valid = lifecycle.get_valid_transitions("model_v42")
    """

    def __init__(self):
        self._models: dict[str, ModelStateRecord] = {}
        self._lock = threading.RLock()
        self._transition_listeners: list[Callable] = []

        # Build transition map
        self._transitions: dict[str, list[tuple[str, str]]] = {}
        for t in MODEL_TRANSITIONS:
            from_state = t.from_state.name
            to_state = t.to_state.name
            if from_state not in self._transitions:
                self._transitions[from_state] = []
            self._transitions[from_state].append((to_state, t.name or ""))

    def register_model(
        self,
        model_id: str,
        initial_state: ModelState = ModelState.TRAINING,
        metadata: dict[str, Any] | None = None,
    ) -> ModelStateRecord:
        """Register a new model for lifecycle tracking.

        Args:
            model_id: Unique model identifier
            initial_state: Initial state (default: TRAINING)
            metadata: Optional metadata

        Returns:
            ModelStateRecord for the model

        Raises:
            ValueError: If model already registered
        """
        with self._lock:
            if model_id in self._models:
                raise ValueError(f"Model {model_id} already registered")

            record = ModelStateRecord(
                model_id=model_id,
                current_state=initial_state,
                metadata=metadata or {},
            )
            self._models[model_id] = record
            logger.info(f"[ModelLifecycle] Registered {model_id} in state {initial_state.value}")
            return record

    def get_or_register(
        self,
        model_id: str,
        initial_state: ModelState = ModelState.TRAINING,
    ) -> ModelStateRecord:
        """Get existing model or register new one."""
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            return self.register_model(model_id, initial_state)

    def get_state(self, model_id: str) -> ModelState | None:
        """Get current state of a model."""
        with self._lock:
            record = self._models.get(model_id)
            return record.current_state if record else None

    def get_record(self, model_id: str) -> ModelStateRecord | None:
        """Get full record for a model."""
        with self._lock:
            return self._models.get(model_id)

    def get_valid_transitions(self, model_id: str) -> list[ModelState]:
        """Get valid next states for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of valid target states
        """
        current = self.get_state(model_id)
        if current is None:
            return []

        valid_states = []
        for to_state, _ in self._transitions.get(current.value, []):
            with contextlib.suppress(ValueError):
                valid_states.append(ModelState(to_state))

        return valid_states

    def can_transition(
        self,
        model_id: str,
        target_state: ModelState,
    ) -> tuple[bool, str]:
        """Check if transition to target state is valid.

        Args:
            model_id: Model identifier
            target_state: Desired target state

        Returns:
            Tuple of (can_transition, reason)
        """
        current = self.get_state(model_id)
        if current is None:
            return False, f"Model {model_id} not registered"

        valid_targets = self._transitions.get(current.value, [])
        for to_state, name in valid_targets:
            if to_state == target_state.value:
                return True, name

        valid_state_names = [t[0] for t in valid_targets]
        return False, (
            f"Invalid transition from {current.value} to {target_state.value}. "
            f"Valid targets: {valid_state_names}"
        )

    def transition(
        self,
        model_id: str,
        target_state: ModelState,
        reason: str = "",
        triggered_by: str = "system",
        force: bool = False,
    ) -> bool:
        """Transition a model to a new state.

        Args:
            model_id: Model identifier
            target_state: Target state
            reason: Reason for transition
            triggered_by: Who/what triggered the transition
            force: Force transition even if invalid

        Returns:
            True if transition succeeded

        Raises:
            InvalidTransitionError: If transition invalid and force=False
        """
        with self._lock:
            record = self._models.get(model_id)
            if record is None:
                raise ValueError(f"Model {model_id} not registered")

            current_state = record.current_state

            # Check validity
            can_do, msg = self.can_transition(model_id, target_state)
            if not can_do and not force:
                raise InvalidTransitionError(
                    current_state.value,
                    target_state.value,
                    msg,
                )

            # Execute transition
            record.add_transition(
                from_state=current_state,
                to_state=target_state,
                reason=reason,
                triggered_by=triggered_by,
            )

            logger.info(
                f"[ModelLifecycle] {model_id}: {current_state.value} -> {target_state.value} "
                f"({reason or 'no reason'})"
            )

            # Notify listeners
            self._notify_transition(model_id, current_state, target_state, reason)

            return True

    def rollback(
        self,
        current_model_id: str,
        rollback_to_model_id: str,
        reason: str = "",
    ) -> bool:
        """Rollback a model and restore another.

        Args:
            current_model_id: Model to rollback
            rollback_to_model_id: Model to restore
            reason: Reason for rollback

        Returns:
            True if rollback succeeded
        """
        with self._lock:
            # Rollback current model
            current_record = self._models.get(current_model_id)
            if current_record and current_record.current_state == ModelState.PRODUCTION:
                self.transition(
                    current_model_id,
                    ModelState.ROLLED_BACK,
                    reason=reason,
                    triggered_by="rollback",
                )

            # Restore rollback target to production
            rollback_record = self._models.get(rollback_to_model_id)
            if rollback_record:
                # Force transition to production (may come from various states)
                rollback_record.add_transition(
                    from_state=rollback_record.current_state,
                    to_state=ModelState.PRODUCTION,
                    reason=f"Restored via rollback from {current_model_id}",
                    triggered_by="rollback",
                )
                logger.info(
                    f"[ModelLifecycle] Restored {rollback_to_model_id} to production "
                    f"(rollback from {current_model_id})"
                )
                return True

            return False

    def add_transition_listener(
        self,
        listener: Callable[[str, ModelState, ModelState, str], None],
    ) -> None:
        """Add a listener for state transitions.

        Args:
            listener: Callable(model_id, from_state, to_state, reason)
        """
        self._transition_listeners.append(listener)

    def _notify_transition(
        self,
        model_id: str,
        from_state: ModelState,
        to_state: ModelState,
        reason: str,
    ) -> None:
        """Notify listeners of a transition."""
        for listener in self._transition_listeners:
            try:
                listener(model_id, from_state, to_state, reason)
            except Exception as e:
                logger.warning(f"Transition listener error: {e}")

    def get_models_in_state(self, state: ModelState) -> list[str]:
        """Get all models in a particular state."""
        with self._lock:
            return [
                model_id
                for model_id, record in self._models.items()
                if record.current_state == state
            ]

    def get_production_model(self) -> str | None:
        """Get the current production model."""
        models = self.get_models_in_state(ModelState.PRODUCTION)
        return models[0] if models else None

    def get_staging_models(self) -> list[str]:
        """Get all models in staging."""
        return self.get_models_in_state(ModelState.STAGING)

    def get_all_models(self) -> dict[str, ModelStateRecord]:
        """Get all tracked models."""
        with self._lock:
            return dict(self._models)

    def get_model_history(
        self,
        model_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get transition history for a model."""
        with self._lock:
            record = self._models.get(model_id)
            if record is None:
                return []
            return record.history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get lifecycle statistics."""
        with self._lock:
            state_counts = {}
            for state in ModelState:
                state_counts[state.value] = len(self.get_models_in_state(state))

            total_transitions = sum(
                len(r.history) for r in self._models.values()
            )

            return {
                "total_models": len(self._models),
                "state_counts": state_counts,
                "total_transitions": total_transitions,
                "production_model": self.get_production_model(),
            }


# =============================================================================
# Integration with PromotionController
# =============================================================================

class PromotionControllerIntegration:
    """Integrates ModelLifecycleStateMachine with PromotionController.

    Automatically updates model states when promotions occur.

    Usage:
        from app.training.model_state_machine import PromotionControllerIntegration

        integration = PromotionControllerIntegration()
        integration.wire_promotion_controller(promotion_controller)
    """

    def __init__(self, lifecycle: ModelLifecycleStateMachine | None = None):
        self._lifecycle = lifecycle or get_model_lifecycle()

    def wire_promotion_controller(self, controller: Any) -> None:
        """Wire this integration to a PromotionController.

        Monkey-patches the controller to update state machine on promotions.
        """
        original_execute = controller.execute_promotion

        def wrapped_execute(decision, dry_run=False):
            result = original_execute(decision, dry_run)

            if result and not dry_run:
                self._update_state_from_decision(decision)

            return result

        controller.execute_promotion = wrapped_execute
        logger.info("[Integration] Wired state machine to PromotionController")

    def _update_state_from_decision(self, decision: Any) -> None:
        """Update model state based on promotion decision."""
        from app.training.promotion_controller import PromotionType

        model_id = decision.model_id

        # Ensure model is registered
        self._lifecycle.get_or_register(model_id, ModelState.TRAINED)

        # Map promotion type to target state
        type_to_state = {
            PromotionType.STAGING: ModelState.STAGING,
            PromotionType.PRODUCTION: ModelState.PRODUCTION,
            PromotionType.CHAMPION: ModelState.PRODUCTION,
            PromotionType.ROLLBACK: ModelState.ROLLED_BACK,
        }

        target_state = type_to_state.get(decision.promotion_type)
        if target_state:
            try:
                self._lifecycle.transition(
                    model_id,
                    target_state,
                    reason=decision.reason,
                    triggered_by="promotion_controller",
                    force=True,  # Force because controller already validated
                )
            except Exception as e:
                logger.warning(f"Failed to update state machine: {e}")


# =============================================================================
# Global Instance
# =============================================================================

_model_lifecycle: ModelLifecycleStateMachine | None = None
_lifecycle_lock = threading.Lock()


def get_model_lifecycle() -> ModelLifecycleStateMachine:
    """Get the global model lifecycle state machine."""
    global _model_lifecycle
    if _model_lifecycle is None:
        with _lifecycle_lock:
            if _model_lifecycle is None:
                _model_lifecycle = ModelLifecycleStateMachine()
    return _model_lifecycle


def reset_model_lifecycle() -> None:
    """Reset the global lifecycle (for testing)."""
    global _model_lifecycle
    with _lifecycle_lock:
        _model_lifecycle = None
