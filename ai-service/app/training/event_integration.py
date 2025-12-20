"""Event Integration for Training Components.

Provides event publishing and subscription utilities for training components
using the unified EventBus system.

Training Events:
- Training started/completed/failed
- Epoch started/completed
- Step completed
- Evaluation completed
- Checkpoint saved
- Model promoted
- Elo changed
- Selfplay started/completed

Usage:
    from app.training.event_integration import (
        publish_training_started,
        publish_training_completed,
        subscribe_to_training_events,
        TrainingEvent,
    )

    # Publish events
    await publish_training_started(
        config_key="square8_2p",
        job_id="job-123",
    )

    # Subscribe to events
    @subscribe_to_training_events("training.completed")
    async def on_training_done(event: TrainingCompletedEvent):
        print(f"Training done: {event.config_key}")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.event_bus import (
    Event,
    EventFilter,
    EventHandler,
    get_event_bus,
    publish,
    subscribe,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointEvent",
    "CheckpointSavedEvent",
    "EloChangedEvent",
    "EpochCompletedEvent",
    "EvaluationCompletedEvent",
    "EvaluationEvent",
    "ModelPromotedEvent",
    "SelfplayCompletedEvent",
    "SelfplayEvent",
    "SelfplayStartedEvent",
    "StepCompletedEvent",
    "TrainingCompletedEvent",
    # Base event types
    "TrainingEvent",
    "TrainingFailedEvent",
    # Specific events
    "TrainingStartedEvent",
    # Topics
    "TrainingTopics",
    "publish_checkpoint_saved",
    "publish_elo_changed",
    "publish_epoch_completed",
    "publish_evaluation_completed",
    "publish_model_promoted",
    "publish_selfplay_completed",
    "publish_selfplay_started",
    "publish_step_completed",
    "publish_training_completed",
    "publish_training_failed",
    # Publishers
    "publish_training_started",
    "subscribe_to_evaluation_events",
    # Subscriptions
    "subscribe_to_training_events",
    # Composite ELO Events (Sprint 5)
    "CompositeAlgorithmRankingEvent",
    "CompositeConsistencyCheckEvent",
    "CompositeEloEvent",
    "CompositeEloUpdatedEvent",
    "CompositeGauntletCompletedEvent",
    "CompositeNNCulledEvent",
    # Composite ELO Publishers
    "publish_composite_algorithm_ranking",
    "publish_composite_consistency_check",
    "publish_composite_elo_updated",
    "publish_composite_elo_updated_sync",
    "publish_composite_gauntlet_completed",
    "publish_composite_nn_culled",
    # Composite subscriptions
    "subscribe_to_composite_events",
]


# =============================================================================
# Topic Constants
# =============================================================================

class TrainingTopics:
    """Training event topic constants."""

    # Training lifecycle
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    EPOCH_COMPLETED = "training.epoch.completed"
    STEP_COMPLETED = "training.step.completed"

    # Evaluation
    EVAL_COMPLETED = "training.eval.completed"
    ELO_CHANGED = "training.eval.elo_changed"
    BASELINE_GATING_FAILED = "training.eval.gating_failed"

    # Checkpointing
    CHECKPOINT_SAVED = "training.checkpoint.saved"
    CHECKPOINT_LOADED = "training.checkpoint.loaded"
    CHECKPOINT_ERROR = "training.checkpoint.error"

    # Model promotion
    MODEL_PROMOTED = "training.model.promoted"
    MODEL_ROLLED_BACK = "training.model.rolled_back"

    # Selfplay
    SELFPLAY_STARTED = "training.selfplay.started"
    SELFPLAY_COMPLETED = "training.selfplay.completed"
    SELFPLAY_FAILED = "training.selfplay.failed"

    # Composite ELO System (Sprint 5)
    COMPOSITE_ELO_UPDATED = "training.composite.elo_updated"
    COMPOSITE_GAUNTLET_COMPLETED = "training.composite.gauntlet_completed"
    COMPOSITE_NN_CULLED = "training.composite.nn_culled"
    COMPOSITE_ALGORITHM_RANKING = "training.composite.algorithm_ranking"
    COMPOSITE_CONSISTENCY_CHECK = "training.composite.consistency_check"

    # Patterns for subscription
    ALL_TRAINING = "training.*"
    ALL_EVAL = "training.eval.*"
    ALL_CHECKPOINT = "training.checkpoint.*"
    ALL_MODEL = "training.model.*"
    ALL_SELFPLAY = "training.selfplay.*"
    ALL_COMPOSITE = "training.composite.*"


# =============================================================================
# Base Event Types
# =============================================================================

@dataclass
class TrainingEvent(Event):
    """Base event for training-related events."""
    config_key: str = ""
    job_id: str = ""


@dataclass
class EvaluationEvent(TrainingEvent):
    """Base event for evaluation-related events."""
    eval_step: int = 0
    elo: float = 0.0


@dataclass
class CheckpointEvent(TrainingEvent):
    """Base event for checkpoint-related events."""
    checkpoint_path: str = ""
    step: int = 0


@dataclass
class SelfplayEvent(TrainingEvent):
    """Base event for selfplay-related events."""
    iteration: int = 0
    games_count: int = 0


# =============================================================================
# Specific Events
# =============================================================================

@dataclass
class TrainingStartedEvent(TrainingEvent):
    """Published when training starts."""
    total_epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0


@dataclass
class TrainingCompletedEvent(TrainingEvent):
    """Published when training completes successfully."""
    epochs_completed: int = 0
    final_loss: float = 0.0
    final_elo: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class TrainingFailedEvent(TrainingEvent):
    """Published when training fails."""
    error_type: str = ""
    error_message: str = ""
    epoch: int = 0
    step: int = 0


@dataclass
class EpochCompletedEvent(TrainingEvent):
    """Published when an epoch completes."""
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    learning_rate: float = 0.0


@dataclass
class StepCompletedEvent(TrainingEvent):
    """Published when a training step completes."""
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    samples_per_second: float = 0.0


@dataclass
class EvaluationCompletedEvent(EvaluationEvent):
    """Published when evaluation completes."""
    games_played: int = 0
    win_rate: float = 0.0
    baseline_results: dict[str, float] = field(default_factory=dict)
    passes_gating: bool = True
    failed_baselines: list[str] = field(default_factory=list)


@dataclass
class EloChangedEvent(EvaluationEvent):
    """Published when Elo rating changes significantly."""
    old_elo: float = 0.0
    new_elo: float = 0.0
    elo_delta: float = 0.0
    is_improvement: bool = False
    is_drop: bool = False


@dataclass
class CheckpointSavedEvent(CheckpointEvent):
    """Published when checkpoint is saved."""
    is_best: bool = False
    elo_at_save: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelPromotedEvent(TrainingEvent):
    """Published when model is promoted."""
    model_id: str = ""
    from_state: str = ""
    to_state: str = ""
    promotion_type: str = ""
    elo: float = 0.0


@dataclass
class SelfplayStartedEvent(SelfplayEvent):
    """Published when selfplay starts."""
    engine: str = ""


@dataclass
class SelfplayCompletedEvent(SelfplayEvent):
    """Published when selfplay completes."""
    success: bool = True
    output_path: str = ""
    duration_seconds: float = 0.0


# =============================================================================
# Composite ELO Events (Sprint 5)
# =============================================================================

@dataclass
class CompositeEloEvent(Event):
    """Base event for composite ELO system events."""
    board_type: str = ""
    num_players: int = 2


@dataclass
class CompositeEloUpdatedEvent(CompositeEloEvent):
    """Published when a composite participant's Elo rating changes.

    Tracks (NN, Algorithm) combination rating updates.
    """
    nn_id: str = ""
    ai_type: str = ""
    config_hash: str = ""
    participant_id: str = ""
    old_elo: float = 0.0
    new_elo: float = 0.0
    elo_delta: float = 0.0
    games_played: int = 0
    is_improvement: bool = False


@dataclass
class CompositeGauntletCompletedEvent(CompositeEloEvent):
    """Published when a two-phase gauntlet completes."""
    phase1_nn_count: int = 0
    phase1_passed_count: int = 0
    phase2_participants: int = 0
    total_games_played: int = 0
    duration_seconds: float = 0.0
    top_nn_ids: list[str] = field(default_factory=list)
    top_algorithm: str = ""


@dataclass
class CompositeNNCulledEvent(CompositeEloEvent):
    """Published when an NN is culled from the system."""
    nn_id: str = ""
    reason: str = ""  # "underperforming", "redundant", "age_limit"
    final_elo: float = 0.0
    games_played: int = 0
    algorithms_tested: list[str] = field(default_factory=list)
    cull_level: int = 1  # 1=NN cull, 2=Combo cull, 3=Standard cull


@dataclass
class CompositeAlgorithmRankingEvent(CompositeEloEvent):
    """Published when algorithm rankings are updated."""
    rankings: list[dict] = field(default_factory=list)
    # Each entry: {"algorithm": str, "avg_elo": float, "games": int, "rank": int}
    expected_order_violations: int = 0
    top_algorithm: str = ""
    nn_count_evaluated: int = 0


@dataclass
class CompositeConsistencyCheckEvent(CompositeEloEvent):
    """Published after consistency checks run."""
    overall_healthy: bool = True
    checks_passed: int = 0
    checks_failed: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    check_results: dict[str, bool] = field(default_factory=dict)
    # Key checks: nn_consistency, algorithm_stability, transitivity, baseline_anchoring


# =============================================================================
# Publisher Functions
# =============================================================================

async def publish_training_started(
    config_key: str,
    job_id: str = "",
    total_epochs: int = 0,
    batch_size: int = 0,
    learning_rate: float = 0.0,
) -> int:
    """Publish training started event."""
    event = TrainingStartedEvent(
        topic=TrainingTopics.TRAINING_STARTED,
        config_key=config_key,
        job_id=job_id,
        total_epochs=total_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        source="training",
    )
    return await publish(event)


async def publish_training_completed(
    config_key: str,
    job_id: str = "",
    epochs_completed: int = 0,
    final_loss: float = 0.0,
    final_elo: float = 0.0,
    duration_seconds: float = 0.0,
) -> int:
    """Publish training completed event."""
    event = TrainingCompletedEvent(
        topic=TrainingTopics.TRAINING_COMPLETED,
        config_key=config_key,
        job_id=job_id,
        epochs_completed=epochs_completed,
        final_loss=final_loss,
        final_elo=final_elo,
        duration_seconds=duration_seconds,
        source="training",
    )
    return await publish(event)


async def publish_training_failed(
    config_key: str,
    error: Exception,
    job_id: str = "",
    epoch: int = 0,
    step: int = 0,
) -> int:
    """Publish training failed event."""
    event = TrainingFailedEvent(
        topic=TrainingTopics.TRAINING_FAILED,
        config_key=config_key,
        job_id=job_id,
        error_type=type(error).__name__,
        error_message=str(error),
        epoch=epoch,
        step=step,
        source="training",
    )
    return await publish(event)


async def publish_epoch_completed(
    config_key: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float | None = None,
    learning_rate: float = 0.0,
    job_id: str = "",
) -> int:
    """Publish epoch completed event."""
    event = EpochCompletedEvent(
        topic=TrainingTopics.EPOCH_COMPLETED,
        config_key=config_key,
        job_id=job_id,
        epoch=epoch,
        total_epochs=total_epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=learning_rate,
        source="training",
    )
    return await publish(event)


async def publish_step_completed(
    config_key: str,
    step: int,
    loss: float,
    learning_rate: float = 0.0,
    samples_per_second: float = 0.0,
    job_id: str = "",
) -> int:
    """Publish step completed event."""
    event = StepCompletedEvent(
        topic=TrainingTopics.STEP_COMPLETED,
        config_key=config_key,
        job_id=job_id,
        step=step,
        loss=loss,
        learning_rate=learning_rate,
        samples_per_second=samples_per_second,
        source="training",
    )
    return await publish(event)


async def publish_evaluation_completed(
    config_key: str,
    eval_step: int,
    elo: float,
    games_played: int,
    win_rate: float,
    baseline_results: dict[str, float] | None = None,
    passes_gating: bool = True,
    failed_baselines: list[str] | None = None,
    job_id: str = "",
) -> int:
    """Publish evaluation completed event."""
    event = EvaluationCompletedEvent(
        topic=TrainingTopics.EVAL_COMPLETED,
        config_key=config_key,
        job_id=job_id,
        eval_step=eval_step,
        elo=elo,
        games_played=games_played,
        win_rate=win_rate,
        baseline_results=baseline_results or {},
        passes_gating=passes_gating,
        failed_baselines=failed_baselines or [],
        source="evaluation",
    )
    return await publish(event)


async def publish_elo_changed(
    config_key: str,
    old_elo: float,
    new_elo: float,
    eval_step: int = 0,
    job_id: str = "",
) -> int:
    """Publish Elo changed event."""
    elo_delta = new_elo - old_elo
    event = EloChangedEvent(
        topic=TrainingTopics.ELO_CHANGED,
        config_key=config_key,
        job_id=job_id,
        eval_step=eval_step,
        elo=new_elo,
        old_elo=old_elo,
        new_elo=new_elo,
        elo_delta=elo_delta,
        is_improvement=elo_delta > 0,
        is_drop=elo_delta < -10,  # Significant drop
        source="evaluation",
    )
    return await publish(event)


async def publish_checkpoint_saved(
    config_key: str,
    checkpoint_path: str,
    step: int,
    is_best: bool = False,
    elo_at_save: float = 0.0,
    metrics: dict[str, float] | None = None,
    job_id: str = "",
) -> int:
    """Publish checkpoint saved event."""
    event = CheckpointSavedEvent(
        topic=TrainingTopics.CHECKPOINT_SAVED,
        config_key=config_key,
        job_id=job_id,
        checkpoint_path=checkpoint_path,
        step=step,
        is_best=is_best,
        elo_at_save=elo_at_save,
        metrics=metrics or {},
        source="checkpoint",
    )
    return await publish(event)


async def publish_model_promoted(
    config_key: str,
    model_id: str,
    from_state: str,
    to_state: str,
    promotion_type: str = "",
    elo: float = 0.0,
    job_id: str = "",
) -> int:
    """Publish model promoted event."""
    event = ModelPromotedEvent(
        topic=TrainingTopics.MODEL_PROMOTED,
        config_key=config_key,
        job_id=job_id,
        model_id=model_id,
        from_state=from_state,
        to_state=to_state,
        promotion_type=promotion_type,
        elo=elo,
        source="promotion",
    )
    return await publish(event)


async def publish_selfplay_started(
    config_key: str,
    iteration: int,
    games_count: int,
    engine: str = "",
    job_id: str = "",
) -> int:
    """Publish selfplay started event."""
    event = SelfplayStartedEvent(
        topic=TrainingTopics.SELFPLAY_STARTED,
        config_key=config_key,
        job_id=job_id,
        iteration=iteration,
        games_count=games_count,
        engine=engine,
        source="selfplay",
    )
    return await publish(event)


async def publish_selfplay_completed(
    config_key: str,
    iteration: int,
    games_count: int,
    success: bool = True,
    output_path: str = "",
    duration_seconds: float = 0.0,
    job_id: str = "",
) -> int:
    """Publish selfplay completed event."""
    event = SelfplayCompletedEvent(
        topic=TrainingTopics.SELFPLAY_COMPLETED,
        config_key=config_key,
        job_id=job_id,
        iteration=iteration,
        games_count=games_count,
        success=success,
        output_path=output_path,
        duration_seconds=duration_seconds,
        source="selfplay",
    )
    return await publish(event)


# =============================================================================
# Composite ELO Publishers (Sprint 5)
# =============================================================================

async def publish_composite_elo_updated(
    nn_id: str,
    ai_type: str,
    config_hash: str,
    participant_id: str,
    old_elo: float,
    new_elo: float,
    games_played: int,
    board_type: str = "square8",
    num_players: int = 2,
) -> int:
    """Publish composite Elo updated event."""
    elo_delta = new_elo - old_elo
    event = CompositeEloUpdatedEvent(
        topic=TrainingTopics.COMPOSITE_ELO_UPDATED,
        board_type=board_type,
        num_players=num_players,
        nn_id=nn_id,
        ai_type=ai_type,
        config_hash=config_hash,
        participant_id=participant_id,
        old_elo=old_elo,
        new_elo=new_elo,
        elo_delta=elo_delta,
        games_played=games_played,
        is_improvement=elo_delta > 0,
        source="composite_elo",
    )
    return await publish(event)


async def publish_composite_gauntlet_completed(
    board_type: str,
    num_players: int,
    phase1_nn_count: int,
    phase1_passed_count: int,
    phase2_participants: int,
    total_games_played: int,
    duration_seconds: float,
    top_nn_ids: list[str] | None = None,
    top_algorithm: str = "",
) -> int:
    """Publish composite gauntlet completed event."""
    event = CompositeGauntletCompletedEvent(
        topic=TrainingTopics.COMPOSITE_GAUNTLET_COMPLETED,
        board_type=board_type,
        num_players=num_players,
        phase1_nn_count=phase1_nn_count,
        phase1_passed_count=phase1_passed_count,
        phase2_participants=phase2_participants,
        total_games_played=total_games_played,
        duration_seconds=duration_seconds,
        top_nn_ids=top_nn_ids or [],
        top_algorithm=top_algorithm,
        source="composite_gauntlet",
    )
    return await publish(event)


async def publish_composite_nn_culled(
    nn_id: str,
    reason: str,
    final_elo: float,
    games_played: int,
    cull_level: int = 1,
    algorithms_tested: list[str] | None = None,
    board_type: str = "square8",
    num_players: int = 2,
) -> int:
    """Publish composite NN culled event."""
    event = CompositeNNCulledEvent(
        topic=TrainingTopics.COMPOSITE_NN_CULLED,
        board_type=board_type,
        num_players=num_players,
        nn_id=nn_id,
        reason=reason,
        final_elo=final_elo,
        games_played=games_played,
        cull_level=cull_level,
        algorithms_tested=algorithms_tested or [],
        source="composite_culling",
    )
    return await publish(event)


async def publish_composite_algorithm_ranking(
    rankings: list[dict],
    expected_order_violations: int,
    top_algorithm: str,
    nn_count_evaluated: int,
    board_type: str = "square8",
    num_players: int = 2,
) -> int:
    """Publish algorithm ranking update event."""
    event = CompositeAlgorithmRankingEvent(
        topic=TrainingTopics.COMPOSITE_ALGORITHM_RANKING,
        board_type=board_type,
        num_players=num_players,
        rankings=rankings,
        expected_order_violations=expected_order_violations,
        top_algorithm=top_algorithm,
        nn_count_evaluated=nn_count_evaluated,
        source="composite_ranking",
    )
    return await publish(event)


async def publish_composite_consistency_check(
    overall_healthy: bool,
    checks_passed: int,
    checks_failed: int,
    warnings_count: int,
    errors_count: int,
    check_results: dict[str, bool] | None = None,
    board_type: str = "square8",
    num_players: int = 2,
) -> int:
    """Publish consistency check result event."""
    event = CompositeConsistencyCheckEvent(
        topic=TrainingTopics.COMPOSITE_CONSISTENCY_CHECK,
        board_type=board_type,
        num_players=num_players,
        overall_healthy=overall_healthy,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        warnings_count=warnings_count,
        errors_count=errors_count,
        check_results=check_results or {},
        source="composite_consistency",
    )
    return await publish(event)


# Synchronous versions for non-async contexts
def publish_composite_elo_updated_sync(
    nn_id: str,
    ai_type: str,
    config_hash: str,
    participant_id: str,
    old_elo: float,
    new_elo: float,
    games_played: int,
    board_type: str = "square8",
    num_players: int = 2,
) -> int:
    """Synchronously publish composite Elo updated event."""
    elo_delta = new_elo - old_elo
    event = CompositeEloUpdatedEvent(
        topic=TrainingTopics.COMPOSITE_ELO_UPDATED,
        board_type=board_type,
        num_players=num_players,
        nn_id=nn_id,
        ai_type=ai_type,
        config_hash=config_hash,
        participant_id=participant_id,
        old_elo=old_elo,
        new_elo=new_elo,
        elo_delta=elo_delta,
        games_played=games_played,
        is_improvement=elo_delta > 0,
        source="composite_elo",
    )
    return get_event_bus().publish_sync(event)


# =============================================================================
# Synchronous Publishers (for non-async contexts)
# =============================================================================

def publish_training_started_sync(
    config_key: str,
    job_id: str = "",
    **kwargs: Any,
) -> int:
    """Synchronously publish training started event."""
    event = TrainingStartedEvent(
        topic=TrainingTopics.TRAINING_STARTED,
        config_key=config_key,
        job_id=job_id,
        source="training",
        **kwargs,
    )
    return get_event_bus().publish_sync(event)


def publish_step_completed_sync(
    config_key: str,
    step: int,
    loss: float,
    **kwargs: Any,
) -> int:
    """Synchronously publish step completed event."""
    event = StepCompletedEvent(
        topic=TrainingTopics.STEP_COMPLETED,
        config_key=config_key,
        step=step,
        loss=loss,
        source="training",
        **kwargs,
    )
    return get_event_bus().publish_sync(event)


def publish_checkpoint_saved_sync(
    config_key: str,
    checkpoint_path: str,
    step: int,
    **kwargs: Any,
) -> int:
    """Synchronously publish checkpoint saved event."""
    event = CheckpointSavedEvent(
        topic=TrainingTopics.CHECKPOINT_SAVED,
        config_key=config_key,
        checkpoint_path=checkpoint_path,
        step=step,
        source="checkpoint",
        **kwargs,
    )
    return get_event_bus().publish_sync(event)


# =============================================================================
# Subscription Helpers
# =============================================================================

def subscribe_to_training_events(
    topic_or_pattern: str = TrainingTopics.ALL_TRAINING,
    priority: int = 0,
) -> Callable[[EventHandler], EventHandler]:
    """Decorator to subscribe to training events.

    Args:
        topic_or_pattern: Topic or pattern to subscribe to
        priority: Handler priority (higher = called first)

    Returns:
        Decorator function

    Example:
        @subscribe_to_training_events("training.completed")
        async def on_training_done(event: TrainingCompletedEvent):
            print(f"Training done: {event.config_key}")

        @subscribe_to_training_events("training.*")
        async def on_any_training_event(event: TrainingEvent):
            logger.info(f"Training event: {event.topic}")
    """
    if "*" in topic_or_pattern:
        # Pattern subscription
        pattern = topic_or_pattern.replace(".", r"\.").replace("*", ".*")
        event_filter = EventFilter(topic_pattern=pattern)
        return subscribe(event_filter, priority=priority)
    else:
        # Exact topic subscription
        return subscribe(topic_or_pattern, priority=priority)


def subscribe_to_evaluation_events(
    priority: int = 0,
) -> Callable[[EventHandler], EventHandler]:
    """Decorator to subscribe to all evaluation events.

    Example:
        @subscribe_to_evaluation_events()
        async def on_eval_event(event: EvaluationEvent):
            print(f"Eval event: {event.topic}, elo={event.elo}")
    """
    pattern = TrainingTopics.ALL_EVAL.replace(".", r"\.").replace("*", ".*")
    event_filter = EventFilter(topic_pattern=pattern)
    return subscribe(event_filter, priority=priority)


def subscribe_to_composite_events(
    priority: int = 0,
) -> Callable[[EventHandler], EventHandler]:
    """Decorator to subscribe to all composite ELO events.

    Example:
        @subscribe_to_composite_events()
        async def on_composite_event(event: CompositeEloEvent):
            print(f"Composite event: {event.topic}, board={event.board_type}")

        @subscribe_to_composite_events(priority=10)
        async def high_priority_handler(event: CompositeEloEvent):
            # Handle with high priority
            pass
    """
    pattern = TrainingTopics.ALL_COMPOSITE.replace(".", r"\.").replace("*", ".*")
    event_filter = EventFilter(topic_pattern=pattern)
    return subscribe(event_filter, priority=priority)


# =============================================================================
# Convenience: Wire Existing Components to Events
# =============================================================================

def wire_background_evaluator_events(evaluator: Any) -> None:
    """Wire BackgroundEvaluator to publish events on evaluation.

    Args:
        evaluator: BackgroundEvaluator instance

    Example:
        from app.training.background_eval import BackgroundEvaluator
        from app.training.event_integration import wire_background_evaluator_events

        evaluator = BackgroundEvaluator(model_getter, config)
        wire_background_evaluator_events(evaluator)
        evaluator.start()
    """
    original_process = evaluator._process_result

    def process_with_events(result):
        # Call original
        original_process(result)

        # Publish event (sync since we're in a thread)
        bus = get_event_bus()

        event = EvaluationCompletedEvent(
            topic=TrainingTopics.EVAL_COMPLETED,
            config_key=getattr(evaluator, '_config_key', ''),
            eval_step=result.step,
            elo=result.elo_estimate,
            games_played=result.games_played,
            win_rate=result.win_rate,
            baseline_results=result.baseline_results,
            passes_gating=result.passes_baseline_gating,
            failed_baselines=result.failed_baselines,
            source="background_eval",
        )
        bus.publish_sync(event)

        # Also publish Elo changed event if significant
        if hasattr(evaluator, '_last_published_elo'):
            elo_delta = result.elo_estimate - evaluator._last_published_elo
            if abs(elo_delta) > 5:  # Only publish significant changes
                elo_event = EloChangedEvent(
                    topic=TrainingTopics.ELO_CHANGED,
                    config_key=getattr(evaluator, '_config_key', ''),
                    old_elo=evaluator._last_published_elo,
                    new_elo=result.elo_estimate,
                    elo_delta=elo_delta,
                    is_improvement=elo_delta > 0,
                    is_drop=elo_delta < -10,
                    source="background_eval",
                )
                bus.publish_sync(elo_event)
                evaluator._last_published_elo = result.elo_estimate
        else:
            evaluator._last_published_elo = result.elo_estimate

    evaluator._process_result = process_with_events
    logger.info("[EventIntegration] Wired BackgroundEvaluator to event bus")


def wire_checkpoint_manager_events(manager: Any, config_key: str = "") -> None:
    """Wire UnifiedCheckpointManager to publish events on save.

    Args:
        manager: UnifiedCheckpointManager instance
        config_key: Configuration key for events
    """
    original_save = manager.save_checkpoint

    def save_with_events(*args, **kwargs):
        result = original_save(*args, **kwargs)

        if result:
            publish_checkpoint_saved_sync(
                config_key=config_key,
                checkpoint_path=str(result.path) if hasattr(result, 'path') else "",
                step=result.step if hasattr(result, 'step') else 0,
                is_best=getattr(result, 'is_best', False),
            )

        return result

    manager.save_checkpoint = save_with_events
    logger.info("[EventIntegration] Wired UnifiedCheckpointManager to event bus")
