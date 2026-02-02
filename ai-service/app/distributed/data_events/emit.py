"""Emit Functions for Data Events.

This module contains all the emit_* convenience functions for publishing
events to the event bus. These functions provide type-safe, documented
interfaces for emitting events.

Usage:
    from app.distributed.data_events import emit_training_completed

    await emit_training_completed(
        config="hex8_2p",
        success=True,
        duration=3600.0,
        model_path="models/hex8_2p.pth",
    )

Note: The unified event router is preferred for new code:
    from app.coordination.event_router import get_router
"""

from __future__ import annotations

import time
from typing import Any

from .event_types import DataEventType, DataEvent
from .event_bus import get_event_bus


# Convenience functions for common events


async def emit_data_event(
    event_type: DataEventType | str,
    payload: dict[str, Any] | None = None,
    source: str = "",
    *,
    bridge_cross_process: bool = True,
) -> None:
    """Emit an arbitrary DataEvent.

    This is a thin helper used by the unified router and by quality/monitoring
    utilities that want to publish DataEventType values without depending on
    the full EventBus API.

    Args:
        event_type: Event type enum or its string value
        payload: Event payload
        source: Component that generated the event
        bridge_cross_process: Whether to bridge to cross-process queue
    """
    if isinstance(event_type, str):
        event_type = DataEventType(event_type)

    await get_event_bus().publish(
        DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        ),
        bridge_cross_process=bridge_cross_process,
    )


async def emit_new_games(host: str, new_games: int, total_games: int, source: str = "") -> None:
    """Emit a NEW_GAMES_AVAILABLE event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={
            "host": host,
            "new_games": new_games,
            "total_games": total_games,
        },
        source=source,
    ))


async def emit_training_threshold(config: str, games: int, source: str = "") -> None:
    """Emit a TRAINING_THRESHOLD_REACHED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
        payload={
            "config": config,
            "games": games,
        },
        source=source,
    ))


async def emit_training_completed(
    config: str,
    success: bool,
    duration: float,
    model_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_COMPLETED,
        payload={
            "config": config,
            "success": success,
            "duration": duration,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_evaluation_completed(
    config: str,
    elo: float,
    games_played: int,
    win_rate: float,
    source: str = "",
) -> None:
    """Emit an EVALUATION_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_COMPLETED,
        payload={
            "config": config,
            "elo": elo,
            "games_played": games_played,
            "win_rate": win_rate,
        },
        source=source,
    ))


async def emit_model_promoted(
    model_id: str,
    config: str,
    elo: float,
    elo_gain: float,
    source: str = "",
) -> None:
    """Emit a MODEL_PROMOTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_PROMOTED,
        payload={
            "model_id": model_id,
            "config": config,
            "elo": elo,
            "elo_gain": elo_gain,
        },
        source=source,
    ))


async def emit_error(
    component: str,
    error: str,
    details: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit an ERROR event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ERROR,
        payload={
            "component": component,
            "error": error,
            "details": details or {},
        },
        source=source,
    ))


async def emit_elo_updated(
    config: str,
    model_id: str,
    new_elo: float,
    old_elo: float,
    games_played: int,
    source: str = "",
) -> None:
    """Emit an ELO_UPDATED event and check for significant changes.

    If the Elo change exceeds the threshold from unified config,
    also emits an ELO_SIGNIFICANT_CHANGE event to trigger curriculum rebalancing.
    """
    elo_change = new_elo - old_elo

    # Emit the standard Elo update event
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ELO_UPDATED,
        payload={
            "config": config,
            "model_id": model_id,
            "new_elo": new_elo,
            "old_elo": old_elo,
            "elo_change": elo_change,
            "games_played": games_played,
        },
        source=source,
    ))

    # Check for significant change (threshold from unified config)
    try:
        from app.config.unified_config import get_config
        config_obj = get_config()
        threshold = config_obj.curriculum.elo_change_threshold
        should_rebalance = config_obj.curriculum.rebalance_on_elo_change
    except ImportError:
        threshold = 20  # December 2025: Lowered from 50 to enable faster curriculum adaptation
        should_rebalance = True

    if should_rebalance and abs(elo_change) >= threshold:
        await get_event_bus().publish(DataEvent(
            event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
            payload={
                "config": config,
                "model_id": model_id,
                "elo_change": elo_change,
                "new_elo": new_elo,
                "threshold": threshold,
            },
            source=source,
        ))


def emit_elo_updated_sync(
    config: str,
    model_id: str,
    new_elo: float,
    old_elo: float,
    games_played: int,
    source: str = "",
) -> None:
    """Synchronous version of emit_elo_updated.

    Feb 2026: Use this in sync contexts (gauntlet evaluation, tournament daemon)
    where no async event loop is running. This ensures ELO_UPDATED events are
    always emitted, enabling elo_progression tracking.

    Args:
        config: Config key (e.g., "hex8_2p")
        model_id: Participant ID
        new_elo: New Elo rating
        old_elo: Previous Elo rating
        games_played: Number of games played
        source: Component emitting this event
    """
    elo_change = new_elo - old_elo

    try:
        bus = get_event_bus()
        if bus:
            bus.publish_sync(DataEvent(
                event_type=DataEventType.ELO_UPDATED,
                payload={
                    "config": config,
                    "model_id": model_id,
                    "new_elo": new_elo,
                    "old_elo": old_elo,
                    "elo_change": elo_change,
                    "games_played": games_played,
                },
                source=source,
            ))
    except (AttributeError, RuntimeError):
        # Event bus not available - non-critical
        pass

    # Check for significant change (threshold from unified config)
    try:
        from app.config.unified_config import get_config
        config_obj = get_config()
        threshold = config_obj.curriculum.elo_change_threshold
        should_rebalance = config_obj.curriculum.rebalance_on_elo_change
    except ImportError:
        threshold = 20
        should_rebalance = True

    if should_rebalance and abs(elo_change) >= threshold:
        try:
            bus = get_event_bus()
            if bus:
                bus.publish_sync(DataEvent(
                    event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
                    payload={
                        "config": config,
                        "model_id": model_id,
                        "elo_change": elo_change,
                        "new_elo": new_elo,
                        "threshold": threshold,
                    },
                    source=source,
                ))
        except (AttributeError, RuntimeError):
            pass


async def emit_elo_velocity_changed(
    config_key: str,
    velocity: float,
    previous_velocity: float,
    trend: str,
    source: str = "queue_populator",
) -> None:
    """Emit an ELO_VELOCITY_CHANGED event.

    P10-LOOP-3 (Dec 2025): Signals when Elo improvement velocity changes significantly.
    This triggers selfplay rate adjustments:
    - Accelerating velocity → increase selfplay to capitalize on momentum
    - Decelerating velocity → reduce selfplay, focus on quality
    - Negative velocity → boost exploration to escape local minimum

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        velocity: Current Elo points per day
        previous_velocity: Previous velocity measurement
        trend: Velocity trend ("accelerating", "stable", "decelerating")
        source: Component emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ELO_VELOCITY_CHANGED,
        payload={
            "config_key": config_key,
            "velocity": velocity,
            "previous_velocity": previous_velocity,
            "velocity_change": velocity - previous_velocity,
            "trend": trend,
        },
        source=source,
    ))


async def emit_quality_score_updated(
    game_id: str,
    quality_score: float,
    quality_category: str,
    training_weight: float,
    game_length: int = 0,
    is_decisive: bool = False,
    source: str = "",
) -> None:
    """Emit a QUALITY_SCORE_UPDATED event.

    Args:
        game_id: Unique game identifier
        quality_score: Computed quality score (0-1)
        quality_category: Category (excellent/good/adequate/poor/unusable)
        training_weight: Weight for training sample selection
        game_length: Number of moves in the game
        is_decisive: Whether game had a clear winner
        source: Component that computed the quality
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_SCORE_UPDATED,
        payload={
            "game_id": game_id,
            "quality_score": quality_score,
            "quality_category": quality_category,
            "training_weight": training_weight,
            "game_length": game_length,
            "is_decisive": is_decisive,
        },
        source=source,
    ))


async def emit_quality_degraded(
    config_key: str,
    quality_score: float,
    threshold: float = 0.6,
    previous_score: float = 0.0,
    source: str = "quality_monitor",
) -> None:
    """Emit a QUALITY_DEGRADED event when quality drops below threshold.

    Phase 5 (Dec 2025): Enables feedback loop to reduce selfplay allocation
    for configs producing low-quality games.

    Args:
        config_key: Board configuration (e.g., "hex8_2p")
        quality_score: Current quality score (0-1)
        threshold: Threshold that was crossed
        previous_score: Previous quality score for comparison
        source: Component that detected the degradation
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_DEGRADED,
        payload={
            "config_key": config_key,
            "quality_score": quality_score,
            "threshold": threshold,
            "previous_score": previous_score,
            "degradation_delta": previous_score - quality_score if previous_score else 0.0,
        },
        source=source,
    ))


async def emit_quality_check_requested(
    config_key: str,
    reason: str,
    source: str = "feedback_loop_controller",
    priority: str = "normal",
) -> None:
    """Emit a QUALITY_CHECK_REQUESTED event to trigger on-demand quality check.

    Phase 9 (Dec 2025): Closes the feedback loop between training loss anomalies
    and data quality verification. When training loss spikes, this requests the
    QualityMonitorDaemon to perform an immediate quality check.

    Args:
        config_key: Board configuration (e.g., "hex8_2p")
        reason: Reason for the quality check request
        source: Component requesting the check
        priority: Priority level ("normal", "high", "urgent")
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_CHECK_REQUESTED,
        payload={
            "config_key": config_key,
            "reason": reason,
            "priority": priority,
        },
        source=source,
    ))


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    trigger: str = "scheduled",
    source: str = "",
) -> None:
    """Emit a CURRICULUM_REBALANCED event.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_weights: Previous curriculum weights
        new_weights: New curriculum weights
        trigger: What triggered the rebalance ("scheduled", "elo_change", "manual")
        source: Component that triggered the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_REBALANCED,
        payload={
            "config": config,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "trigger": trigger,
        },
        source=source,
    ))


async def emit_plateau_detected(
    config: str,
    current_elo: float,
    plateau_duration_games: int,
    plateau_duration_seconds: float,
    source: str = "",
) -> None:
    """Emit a PLATEAU_DETECTED event.

    Args:
        config: Board configuration
        current_elo: Current Elo rating
        plateau_duration_games: Number of games in plateau
        plateau_duration_seconds: Time duration of plateau
        source: Component that detected the plateau
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PLATEAU_DETECTED,
        payload={
            "config": config,
            "current_elo": current_elo,
            "plateau_duration_games": plateau_duration_games,
            "plateau_duration_seconds": plateau_duration_seconds,
        },
        source=source,
    ))


async def emit_cmaes_triggered(
    config: str,
    reason: str,
    current_params: dict[str, Any],
    source: str = "",
) -> None:
    """Emit a CMAES_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why CMA-ES was triggered (e.g., "plateau_detected", "manual")
        current_params: Current hyperparameters before optimization
        source: Component that triggered CMA-ES
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CMAES_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "current_params": current_params,
        },
        source=source,
    ))


async def emit_nas_triggered(
    config: str,
    reason: str,
    search_space: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit a NAS_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why NAS was triggered
        search_space: Optional architecture search space configuration
        source: Component that triggered NAS
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NAS_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "search_space": search_space or {},
        },
        source=source,
    ))


async def emit_hyperparameter_updated(
    config: str,
    param_name: str,
    old_value: Any,
    new_value: Any,
    optimizer: str = "manual",
    source: str = "",
) -> None:
    """Emit a HYPERPARAMETER_UPDATED event.

    Args:
        config: Board configuration
        param_name: Name of the parameter that changed
        old_value: Previous value
        new_value: New value
        optimizer: What triggered the update ("cmaes", "nas", "manual")
        source: Component that updated the parameter
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HYPERPARAMETER_UPDATED,
        payload={
            "config": config,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "optimizer": optimizer,
        },
        source=source,
    ))


async def emit_exploration_boost(
    config_key: str,
    boost_factor: float,
    reason: str,
    anomaly_count: int = 0,
    source: str = "feedback_loop_controller",
) -> None:
    """Emit an EXPLORATION_BOOST event.

    P11-CRITICAL-1 (Dec 2025): Signals when exploration should be boosted due to
    training anomalies (loss spikes, divergence, stalls). This closes the feedback
    loop from training issues back to selfplay exploration rates.

    Subscribers (e.g., SelfplayScheduler, TemperatureScheduler) should react by:
    - Increasing MCTS exploration temperature
    - Adding more Dirichlet noise at root
    - Prioritizing diverse game openings

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        boost_factor: Exploration boost multiplier (1.0 = no boost, 2.0 = double)
        reason: Why exploration is being boosted ("loss_anomaly", "stall", "divergence")
        anomaly_count: Number of consecutive anomalies that triggered this boost
        source: Component emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EXPLORATION_BOOST,
        payload={
            "config_key": config_key,
            "boost_factor": boost_factor,
            "reason": reason,
            "anomaly_count": anomaly_count,
        },
        source=source,
    ))


# =============================================================================
# Data Sync Events
# =============================================================================

async def emit_data_sync_started(
    host: str,
    sync_type: str = "incremental",
    source: str = "",
) -> None:
    """Emit a DATA_SYNC_STARTED event.

    Args:
        host: Host being synced
        sync_type: Type of sync (incremental, full)
        source: Component initiating the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_STARTED,
        payload={
            "host": host,
            "sync_type": sync_type,
        },
        source=source,
    ))


async def emit_data_sync_completed(
    host: str,
    games_synced: int,
    duration: float,
    bytes_transferred: int = 0,
    source: str = "",
    avg_quality_score: float = 0.0,
    high_quality_count: int = 0,
    config: str = "",
) -> None:
    """Emit a DATA_SYNC_COMPLETED event with quality metrics.

    Args:
        host: Host that was synced
        games_synced: Number of games transferred
        duration: Sync duration in seconds
        bytes_transferred: Bytes transferred (if known)
        source: Component that performed the sync
        avg_quality_score: Average quality score of synced games (0-1)
        high_quality_count: Number of games with quality >= 0.7
        config: Configuration key (e.g., "square8_2p")
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_COMPLETED,
        payload={
            "host": host,
            "games_synced": games_synced,
            "duration": duration,
            "bytes_transferred": bytes_transferred,
            "avg_quality_score": avg_quality_score,
            "high_quality_count": high_quality_count,
            "config": config,
        },
        source=source,
    ))


async def emit_data_sync_failed(
    host: str,
    error: str,
    retry_count: int = 0,
    source: str = "",
) -> None:
    """Emit a DATA_SYNC_FAILED event.

    Args:
        host: Host that failed to sync
        error: Error message
        retry_count: Number of retries attempted
        source: Component that attempted the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_FAILED,
        payload={
            "host": host,
            "error": error,
            "retry_count": retry_count,
        },
        source=source,
    ))


# =============================================================================
# Model Distribution Events (December 2025)
# =============================================================================

async def emit_model_distribution_started(
    total_models: int,
    target_hosts: int,
    source: str = "",
) -> None:
    """Emit a MODEL_DISTRIBUTION_STARTED event.

    December 2025: Enables tracking of model sync operations.

    Args:
        total_models: Number of models being distributed
        target_hosts: Number of target hosts
        source: Component initiating distribution
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_DISTRIBUTION_STARTED,
        payload={
            "total_models": total_models,
            "target_hosts": target_hosts,
        },
        source=source,
    ))


async def emit_model_distribution_complete(
    models_collected: int,
    models_distributed: int,
    target_hosts: int,
    duration: float,
    source: str = "",
) -> None:
    """Emit a MODEL_DISTRIBUTION_COMPLETE event.

    December 2025: Enables tracking of successful model sync completion.

    Args:
        models_collected: Number of models collected from cluster
        models_distributed: Number of models distributed to cluster
        target_hosts: Number of target hosts
        duration: Distribution duration in seconds
        source: Component that completed distribution
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_DISTRIBUTION_COMPLETE,
        payload={
            "models_collected": models_collected,
            "models_distributed": models_distributed,
            "target_hosts": target_hosts,
            "duration": duration,
        },
        source=source,
    ))


async def emit_model_distribution_failed(
    error: str,
    partial_models: int = 0,
    source: str = "",
) -> None:
    """Emit a MODEL_DISTRIBUTION_FAILED event.

    December 2025: Enables tracking of failed model sync operations.

    Args:
        error: Error message
        partial_models: Number of models partially synced before failure
        source: Component where distribution failed
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_DISTRIBUTION_FAILED,
        payload={
            "error": error,
            "partial_models": partial_models,
        },
        source=source,
    ))


# =============================================================================
# Host Status Events
# =============================================================================

async def emit_host_online(
    host: str,
    capabilities: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit a HOST_ONLINE event.

    Args:
        host: Host that came online
        capabilities: List of host capabilities (gpu, cpu, etc.)
        source: Component that detected the host
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HOST_ONLINE,
        payload={
            "host": host,
            "capabilities": capabilities or [],
        },
        source=source,
    ))


async def emit_host_offline(
    host: str,
    reason: str = "",
    last_seen: float | None = None,
    source: str = "",
) -> None:
    """Emit a HOST_OFFLINE event.

    Args:
        host: Host that went offline
        reason: Reason for going offline (timeout, error, etc.)
        last_seen: Timestamp when host was last seen
        source: Component that detected the offline status
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HOST_OFFLINE,
        payload={
            "host": host,
            "reason": reason,
            "last_seen": last_seen,
        },
        source=source,
    ))


# =============================================================================
# Daemon Lifecycle Events
# =============================================================================

async def emit_daemon_started(
    daemon_name: str,
    hostname: str,
    pid: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_STARTED event.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        pid: Process ID
        source: Component starting the daemon
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STARTED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "pid": pid,
        },
        source=source,
    ))


async def emit_daemon_stopped(
    daemon_name: str,
    hostname: str,
    reason: str = "normal",
    source: str = "",
) -> None:
    """Emit a DAEMON_STOPPED event.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        reason: Reason for stopping (normal, error, signal)
        source: Component reporting the stop
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STOPPED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "reason": reason,
        },
        source=source,
    ))


async def emit_daemon_permanently_failed(
    daemon_name: str,
    hostname: str,
    restart_count: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_PERMANENTLY_FAILED event.

    December 2025: Emitted when a daemon has exceeded its hourly restart limit
    and is marked as permanently failed. This requires manual intervention to
    clear the failure state and allow the daemon to restart.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        restart_count: Number of restarts in the last hour
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_PERMANENTLY_FAILED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "restart_count": restart_count,
            "requires_intervention": True,
        },
        source=source,
    ))


async def emit_daemon_crash_loop_detected(
    daemon_name: str,
    hostname: str,
    restart_count: int,
    window_minutes: int,
    max_restarts: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_CRASH_LOOP_DETECTED event (early warning before permanent failure).

    December 2025: Emitted when a daemon has restarted multiple times within a short
    window, indicating a crash loop. This is an early warning before the daemon
    reaches the permanent failure threshold (MAX_RESTARTS_PER_HOUR).

    This event enables:
    - Dashboard alerts for crash loops
    - Proactive investigation before permanent failure
    - Notification to operators

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        restart_count: Number of restarts in the window
        window_minutes: Time window in minutes
        max_restarts: Max restarts before permanent failure
        source: Component reporting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_CRASH_LOOP_DETECTED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "restart_count": restart_count,
            "window_minutes": window_minutes,
            "max_restarts": max_restarts,
            "severity": "warning",
            "message": f"{daemon_name} is crash looping ({restart_count} restarts in {window_minutes}min)",
        },
        source=source,
    ))


# =============================================================================
# Orphan Detection Events (December 2025)
# =============================================================================

async def emit_orphan_games_detected(
    host: str,
    orphan_count: int,
    orphan_paths: list[str],
    total_games: int = 0,
    source: str = "",
) -> None:
    """Emit an ORPHAN_GAMES_DETECTED event.

    Dec 2025: Emitted when unregistered game databases are found on a node.
    These are databases that exist on disk but aren't tracked in the manifest.

    Args:
        host: Host where orphans were detected
        orphan_count: Number of orphan databases found
        orphan_paths: List of paths to orphan databases
        total_games: Total game count in orphan databases
        source: Component that detected the orphans
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ORPHAN_GAMES_DETECTED,
        payload={
            "host": host,
            "orphan_count": orphan_count,
            "orphan_paths": orphan_paths[:10],  # Limit to 10 paths to avoid huge payloads
            "total_games": total_games,
        },
        source=source,
    ))


async def emit_orphan_games_registered(
    host: str,
    registered_count: int,
    registered_paths: list[str],
    games_recovered: int = 0,
    source: str = "",
) -> None:
    """Emit an ORPHAN_GAMES_REGISTERED event.

    Dec 2025: Emitted when orphan databases have been auto-registered
    into the manifest after detection.

    Args:
        host: Host where orphans were registered
        registered_count: Number of databases registered
        registered_paths: List of paths that were registered
        games_recovered: Total game count recovered from orphans
        source: Component that performed registration
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ORPHAN_GAMES_REGISTERED,
        payload={
            "host": host,
            "registered_count": registered_count,
            "registered_paths": registered_paths[:10],  # Limit paths
            "games_recovered": games_recovered,
        },
        source=source,
    ))


# =============================================================================
# Data Integrity Events (January 2026)
# =============================================================================


async def emit_game_save_failed(
    game_id: str,
    error: str,
    error_type: str,
    config_key: str,
    board_type: str | None = None,
    num_players: int | None = None,
    db_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a GAME_SAVE_FAILED event.

    January 2026: Tracks failed game saves for monitoring.
    Previously these failures were silently swallowed (P0 issue).

    Args:
        game_id: The game ID that failed to save
        error: Error message
        error_type: Exception class name
        config_key: Board configuration key (e.g., "hex8_2p")
        board_type: Board type
        num_players: Number of players
        db_path: Path to target database (if available)
        source: Component where the failure occurred
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.GAME_SAVE_FAILED,
        payload={
            "game_id": game_id,
            "error": error,
            "error_type": error_type,
            "config_key": config_key,
            "board_type": board_type,
            "num_players": num_players,
            "db_path": db_path,
        },
        source=source,
    ))


async def emit_orphan_game_prevented(
    game_id: str,
    move_count: int,
    min_required: int,
    source: str = "",
) -> None:
    """Emit an ORPHAN_GAME_PREVENTED event.

    January 2026: Tracks when orphan games are prevented by validation.
    This happens when post-insert validation detects insufficient moves.

    Args:
        game_id: The game ID that was prevented
        move_count: Actual move count found
        min_required: Minimum moves required (typically 5)
        source: Component that prevented the orphan
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ORPHAN_GAME_PREVENTED,
        payload={
            "game_id": game_id,
            "move_count": move_count,
            "min_required": min_required,
        },
        source=source,
    ))


# =============================================================================
# Training Failure Events
# =============================================================================

async def emit_training_started(
    config: str,
    model_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_STARTED event.

    Args:
        config: Board configuration
        model_path: Path to model being trained
        source: Component starting the training
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_STARTED,
        payload={
            "config": config,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_training_failed(
    config: str,
    error: str,
    duration: float = 0,
    source: str = "",
) -> None:
    """Emit a TRAINING_FAILED event.

    Args:
        config: Board configuration
        error: Error message
        duration: How long training ran before failing
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_FAILED,
        payload={
            "config": config,
            "error": error,
            "duration": duration,
        },
        source=source,
    ))


# =============================================================================
# Evaluation Failure Events
# =============================================================================

async def emit_evaluation_started(
    config: str,
    model_id: str,
    games_planned: int = 0,
    source: str = "",
) -> None:
    """Emit an EVALUATION_STARTED event.

    Args:
        config: Board configuration
        model_id: Model being evaluated
        games_planned: Number of evaluation games planned
        source: Component starting the evaluation
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_STARTED,
        payload={
            "config": config,
            "model_id": model_id,
            "games_planned": games_planned,
        },
        source=source,
    ))


async def emit_evaluation_failed(
    config: str,
    model_id: str,
    error: str,
    source: str = "",
) -> None:
    """Emit an EVALUATION_FAILED event.

    Args:
        config: Board configuration
        model_id: Model being evaluated
        error: Error message
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_FAILED,
        payload={
            "config": config,
            "model_id": model_id,
            "error": error,
        },
        source=source,
    ))


# =============================================================================
# Promotion Events
# =============================================================================

async def emit_promotion_candidate(
    model_id: str,
    board_type: str,
    num_players: int,
    win_rate_vs_heuristic: float,
    source: str = "",
) -> None:
    """Emit a PROMOTION_CANDIDATE event.

    Args:
        model_id: Model that is a promotion candidate
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        win_rate_vs_heuristic: Win rate against heuristic baseline
        source: Component emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_CANDIDATE,
        payload={
            "model_id": model_id,
            "board_type": board_type,
            "num_players": num_players,
            "config_key": f"{board_type}_{num_players}p",
            "win_rate_vs_heuristic": win_rate_vs_heuristic,
        },
        source=source,
    ))


async def emit_promotion_started(
    config: str,
    model_id: str,
    source: str = "",
) -> None:
    """Emit a PROMOTION_STARTED event.

    Args:
        config: Board configuration
        model_id: Model being considered for promotion
        source: Component starting the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_STARTED,
        payload={
            "config": config,
            "model_id": model_id,
        },
        source=source,
    ))


async def emit_promotion_failed(
    config: str,
    model_id: str,
    error: str,
    source: str = "",
) -> None:
    """Emit a PROMOTION_FAILED event.

    Args:
        config: Board configuration
        model_id: Model that failed promotion
        error: Error message
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_FAILED,
        payload={
            "config": config,
            "model_id": model_id,
            "error": error,
        },
        source=source,
    ))


async def emit_promotion_rejected(
    config: str,
    model_id: str,
    reason: str,
    elo_improvement: float = 0,
    source: str = "",
) -> None:
    """Emit a PROMOTION_REJECTED event.

    Args:
        config: Board configuration
        model_id: Model that was rejected
        reason: Reason for rejection
        elo_improvement: Elo improvement achieved (if any)
        source: Component rejecting the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_REJECTED,
        payload={
            "config": config,
            "model_id": model_id,
            "reason": reason,
            "elo_improvement": elo_improvement,
        },
        source=source,
    ))


# =============================================================================
# Tier Promotion Events
# =============================================================================
# STATUS: RESERVED - Emitter exists, no subscribers yet (Dec 2025)
#
# Purpose: Track model promotion through difficulty ladder tiers (D1→D2→D3, etc.)
#
# Intended subscribers (to be implemented Q1 2026):
#   - CurriculumIntegration: Adjust training curriculum when tier changes
#   - TierProgressTracker: Monitor promotion velocity for dashboards
#   - ModelRegistry: Update model metadata with tier information
#
# The emit function is wired from app.training.promotion_controller._execute_tier_promotion()
# when a model passes threshold evaluation at its current tier.
# =============================================================================

async def emit_tier_promotion(
    config: str,
    old_tier: str,
    new_tier: str,
    model_id: str = "",
    win_rate: float = 0.0,
    elo: float = 0.0,
    games_played: int = 0,
    source: str = "",
) -> None:
    """Emit a TIER_PROMOTION event for difficulty ladder progression.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_tier: Previous tier (e.g., "D4")
        new_tier: New tier after promotion (e.g., "D5")
        model_id: ID of the model being promoted
        win_rate: Win rate that triggered promotion
        elo: Current Elo rating
        games_played: Number of games played at current tier
        source: Component that triggered the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TIER_PROMOTION,
        payload={
            "config": config,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "model_id": model_id,
            "win_rate": win_rate,
            "elo": elo,
            "games_played": games_played,
        },
        source=source,
    ))


async def emit_deadlock_detected(
    resources: list[str],
    holders: list[str],
    source: str = "",
) -> None:
    """Emit a DEADLOCK_DETECTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DEADLOCK_DETECTED,
        payload={
            "resources": resources,
            "holders": holders,
        },
        source=source,
    ))


# =============================================================================
# Checkpoint Events (December 2025)
# =============================================================================

async def emit_checkpoint_saved(
    config: str,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    source: str = "",
) -> None:
    """Emit a CHECKPOINT_SAVED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CHECKPOINT_SAVED,
        payload={
            "config": config,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        },
        source=source,
    ))


async def emit_checkpoint_loaded(
    config: str,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    source: str = "",
) -> None:
    """Emit a CHECKPOINT_LOADED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CHECKPOINT_LOADED,
        payload={
            "config": config,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
        },
        source=source,
    ))


# =============================================================================
# Task Lifecycle Events (December 2025)
# =============================================================================

async def emit_task_spawned(
    task_id: str,
    task_type: str,
    node_id: str,
    config: str = "",
    priority: int = 0,
    source: str = "",
) -> None:
    """Emit a TASK_SPAWNED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_SPAWNED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "config": config,
            "priority": priority,
        },
        source=source,
    ))


async def emit_task_heartbeat(
    task_id: str,
    node_id: str,
    progress: float = 0.0,
    status: str = "running",
    source: str = "",
) -> None:
    """Emit a TASK_HEARTBEAT event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_HEARTBEAT,
        payload={
            "task_id": task_id,
            "node_id": node_id,
            "progress": progress,
            "status": status,
        },
        source=source,
    ))


async def emit_task_completed(
    task_id: str,
    task_type: str,
    node_id: str,
    duration_seconds: float = 0.0,
    result: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit a TASK_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_COMPLETED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "duration_seconds": duration_seconds,
            "result": result or {},
        },
        source=source,
    ))


async def emit_task_failed(
    task_id: str,
    task_type: str,
    node_id: str,
    error: str,
    duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a TASK_FAILED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_FAILED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "error": error,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


async def emit_task_orphaned(
    task_id: str,
    task_type: str,
    node_id: str,
    last_heartbeat_seconds_ago: float = 0.0,
    source: str = "",
) -> None:
    """Emit a TASK_ORPHANED event for tasks that stopped sending heartbeats."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_ORPHANED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "last_heartbeat_seconds_ago": last_heartbeat_seconds_ago,
        },
        source=source,
    ))


async def emit_task_abandoned(
    work_id: str,
    reason: str,
    component: str = "p2p_orchestrator",
    node_id: str = "",
    timestamp: float | None = None,
    source: str = "",
) -> None:
    """Emit a TASK_ABANDONED event for intentionally cancelled tasks.

    December 2025: Added to distinguish intentional cancellations from failures.
    Use this when a task is deliberately stopped (e.g., user request, superseded
    by newer version, resource reallocation) rather than failing unexpectedly.

    Args:
        work_id: The work item ID that was abandoned
        reason: Why the task was abandoned (e.g., "superseded", "user_request")
        component: Which component triggered the abandonment
        node_id: Optional node where the task was running
        timestamp: When abandonment occurred (defaults to now)
        source: Event source identifier
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_ABANDONED,
        payload={
            "work_id": work_id,
            "reason": reason,
            "component": component,
            "node_id": node_id,
            "timestamp": timestamp or time.time(),
        },
        source=source,
    ))


# =============================================================================
# Job Spawn Verification Events (January 2026 - Sprint 6)
# =============================================================================


async def emit_job_spawn_verified(
    job_id: str,
    node_id: str,
    config_key: str,
    verification_time_seconds: float,
    source: str = "",
) -> None:
    """Emit a JOB_SPAWN_VERIFIED event when a job is confirmed running.

    January 2026 - Sprint 6: Added for job spawn verification loop.
    Used by SelfplayScheduler to track spawn success rates and adjust capacity estimates.

    Args:
        job_id: The job ID that was verified
        node_id: Node where the job is running
        config_key: Board configuration key (e.g., "hex8_2p")
        verification_time_seconds: Time taken to verify the spawn
        source: Event source identifier
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.JOB_SPAWN_VERIFIED,
        payload={
            "job_id": job_id,
            "node_id": node_id,
            "config_key": config_key,
            "verification_time_seconds": verification_time_seconds,
            "timestamp": time.time(),
        },
        source=source,
    ))


async def emit_job_spawn_failed(
    job_id: str,
    node_id: str,
    config_key: str,
    timeout_seconds: float,
    reason: str = "verification_timeout",
    source: str = "",
) -> None:
    """Emit a JOB_SPAWN_FAILED event when job spawn verification fails.

    January 2026 - Sprint 6: Added for job spawn verification loop.
    Used by SelfplayScheduler to detect spawn failures and adjust node capacity.

    Args:
        job_id: The job ID that failed to spawn
        node_id: Node where spawn was attempted
        config_key: Board configuration key (e.g., "hex8_2p")
        timeout_seconds: Verification timeout that was exceeded
        reason: Why verification failed (e.g., "verification_timeout", "job_not_found")
        source: Event source identifier
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.JOB_SPAWN_FAILED,
        payload={
            "job_id": job_id,
            "node_id": node_id,
            "config_key": config_key,
            "timeout_seconds": timeout_seconds,
            "reason": reason,
            "timestamp": time.time(),
        },
        source=source,
    ))


# =============================================================================
# Capacity/Backpressure Events (December 2025)
# =============================================================================

async def emit_cluster_capacity_changed(
    total_gpus: int,
    available_gpus: int,
    total_nodes: int,
    healthy_nodes: int,
    source: str = "",
) -> None:
    """Emit a CLUSTER_CAPACITY_CHANGED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CLUSTER_CAPACITY_CHANGED,
        payload={
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
        },
        source=source,
    ))


async def emit_backpressure_activated(
    reason: str,
    queue_depth: int = 0,
    utilization_percent: float = 0.0,
    source: str = "",
) -> None:
    """Emit a BACKPRESSURE_ACTIVATED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BACKPRESSURE_ACTIVATED,
        payload={
            "reason": reason,
            "queue_depth": queue_depth,
            "utilization_percent": utilization_percent,
        },
        source=source,
    ))


async def emit_backpressure_released(
    reason: str,
    duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a BACKPRESSURE_RELEASED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BACKPRESSURE_RELEASED,
        payload={
            "reason": reason,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


# =============================================================================
# Leader Election Events (December 2025)
# =============================================================================

async def emit_leader_elected(
    leader_id: str,
    term: int = 0,
    source: str = "",
) -> None:
    """Emit a LEADER_ELECTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_ELECTED,
        payload={
            "leader_id": leader_id,
            "term": term,
        },
        source=source,
    ))


async def emit_leader_lost(
    old_leader_id: str,
    reason: str = "",
    source: str = "",
) -> None:
    """Emit a LEADER_LOST event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_LOST,
        payload={
            "old_leader_id": old_leader_id,
            "reason": reason,
        },
        source=source,
    ))


async def emit_leader_heartbeat_missing(
    leader_id: str,
    last_heartbeat: float,
    expected_interval: float,
    delay_seconds: float,
    source: str = "",
) -> None:
    """Emit a LEADER_HEARTBEAT_MISSING event for monitoring.

    This event is emitted when the leader hasn't sent a heartbeat for longer
    than expected. It's a warning signal before the lease actually expires.

    Args:
        leader_id: ID of the leader node
        last_heartbeat: Timestamp of last known heartbeat
        expected_interval: Expected heartbeat interval in seconds
        delay_seconds: How many seconds the heartbeat is delayed
        source: Node emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_HEARTBEAT_MISSING,
        payload={
            "leader_id": leader_id,
            "last_heartbeat": last_heartbeat,
            "expected_interval": expected_interval,
            "delay_seconds": delay_seconds,
        },
        source=source,
    ))


async def emit_leader_lease_expired(
    leader_id: str,
    lease_expiry_time: float,
    current_time: float,
    grace_seconds: float = 30.0,
    source: str = "",
) -> None:
    """Emit a LEADER_LEASE_EXPIRED event when leader lease expires without stepdown.

    This event is emitted when a leader's lease expires but the leader hasn't
    voluntarily stepped down. This indicates a potential stuck or unresponsive
    leader that needs to be replaced via election.

    January 2, 2026: Added for stale leader alerting in Sprint 3.

    Args:
        leader_id: ID of the leader node whose lease expired
        lease_expiry_time: Unix timestamp when lease expired
        current_time: Current Unix timestamp
        grace_seconds: Grace period that was allowed beyond expiry
        source: Node emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_LEASE_EXPIRED,
        payload={
            "leader_id": leader_id,
            "lease_expiry_time": lease_expiry_time,
            "current_time": current_time,
            "grace_seconds": grace_seconds,
            "expired_by_seconds": current_time - lease_expiry_time,
        },
        source=source,
    ))


async def emit_training_loss_anomaly(
    config_key: str,
    current_loss: float,
    avg_loss: float,
    epoch: int,
    anomaly_ratio: float = 0.0,
    source: str = "training",
) -> None:
    """Emit a TRAINING_LOSS_ANOMALY event when loss spikes above threshold.

    This triggers quality checks and potential data investigation.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_LOSS_ANOMALY,
        payload={
            "config_key": config_key,
            "current_loss": current_loss,
            "avg_loss": avg_loss,
            "epoch": epoch,
            "anomaly_ratio": anomaly_ratio,
        },
        source=source,
    ))


async def emit_training_loss_trend(
    config_key: str,
    trend: str,  # "improving", "stalled", "degrading"
    epoch: int,
    current_loss: float,
    previous_loss: float,
    improvement_rate: float = 0.0,
    source: str = "training",
) -> None:
    """Emit a TRAINING_LOSS_TREND event to guide data collection.

    Used by selfplay to adjust exploration/exploitation based on training needs.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_LOSS_TREND,
        payload={
            "config_key": config_key,
            "trend": trend,
            "epoch": epoch,
            "current_loss": current_loss,
            "previous_loss": previous_loss,
            "improvement_rate": improvement_rate,
        },
        source=source,
    ))


async def emit_training_early_stopped(
    config_key: str,
    epoch: int,
    best_loss: float,
    final_loss: float,
    best_elo: float | None = None,
    reason: str = "loss_stagnation",
    epochs_without_improvement: int = 0,
    source: str = "training",
) -> None:
    """Emit a TRAINING_EARLY_STOPPED event when training stops early.

    This triggers curriculum boost for the config to accelerate training recovery.
    The curriculum feedback system should increase training allocation for
    configs that are stuck.

    Args:
        config_key: Board/player config (e.g., "hex8_2p")
        epoch: Epoch at which training stopped
        best_loss: Best validation loss achieved
        final_loss: Final validation loss before stopping
        best_elo: Best Elo achieved (if Elo tracking enabled)
        reason: Why early stopping triggered ("loss_stagnation", "elo_stagnation", "regression")
        epochs_without_improvement: How many epochs passed without improvement
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_EARLY_STOPPED,
        payload={
            "config_key": config_key,
            "epoch": epoch,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "best_elo": best_elo,
            "reason": reason,
            "epochs_without_improvement": epochs_without_improvement,
        },
        source=source,
    ))


async def emit_sync_stalled(
    source_host: str,
    target_host: str,
    data_type: str,
    timeout_seconds: float,
    retry_count: int = 0,
    source: str = "sync_coordinator",
) -> None:
    """Emit a SYNC_STALLED event when sync times out.

    This triggers failover to alternative sync sources.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SYNC_STALLED,
        payload={
            "source_host": source_host,
            "target_host": target_host,
            "data_type": data_type,
            "timeout_seconds": timeout_seconds,
            "retry_count": retry_count,
        },
        source=source,
    ))


async def emit_node_overloaded(
    host: str,
    cpu_percent: float,
    gpu_percent: float = 0.0,
    memory_percent: float = 0.0,
    resource_type: str = "cpu",
    source: str = "health_manager",
) -> None:
    """Emit a NODE_OVERLOADED event for job redistribution.

    This triggers the job scheduler to migrate jobs away from overloaded nodes.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_OVERLOADED,
        payload={
            "host": host,
            "cpu_percent": cpu_percent,
            "gpu_percent": gpu_percent,
            "memory_percent": memory_percent,
            "resource_type": resource_type,
        },
        source=source,
    ))


async def emit_selfplay_target_updated(
    config_key: str,
    target_games: int,
    reason: str,
    priority: int = 5,
    source: str = "feedback_controller",
) -> None:
    """Emit a SELFPLAY_TARGET_UPDATED event to scale selfplay.

    Used to request more selfplay games when training needs data.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_TARGET_UPDATED,
        payload={
            "config_key": config_key,
            "target_games": target_games,
            "reason": reason,
            "priority": priority,
        },
        source=source,
    ))


async def emit_selfplay_complete(
    config_key: str,
    games_played: int,
    db_path: str = "",
    duration_seconds: float = 0.0,
    engine: str = "unknown",
    source: str = "selfplay_runner",
) -> None:
    """Emit a SELFPLAY_COMPLETE event when a selfplay batch finishes.

    P0.2 Dec 2025: Bridges StageEvent.SELFPLAY_COMPLETE to DataEventBus.
    This enables components subscribed to DataEventType.SELFPLAY_COMPLETE
    (e.g., feedback_loop_controller, master_loop) to receive the event.

    Args:
        config_key: Board config (e.g., "hex8_2p")
        games_played: Number of games completed
        db_path: Path to database with games
        duration_seconds: Total selfplay duration
        engine: Selfplay engine used (e.g., "gumbel", "heuristic")
        source: Component that ran selfplay
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_COMPLETE,
        payload={
            "config_key": config_key,
            "games_played": games_played,
            "db_path": db_path,
            "duration_seconds": duration_seconds,
            "engine": engine,
        },
        source=source,
    ))


async def emit_idle_resource_detected(
    node_id: str,
    host: str,
    gpu_utilization: float,
    gpu_memory_gb: float,
    idle_duration_seconds: float,
    recommended_config: str = "",
    source: str = "idle_resource_daemon",
) -> None:
    """Emit an IDLE_RESOURCE_DETECTED event for job scheduling.

    Used by IdleResourceDaemon to signal idle GPU resources that can be used.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.IDLE_RESOURCE_DETECTED,
        payload={
            "node_id": node_id,
            "host": host,
            "gpu_utilization": gpu_utilization,
            "gpu_memory_gb": gpu_memory_gb,
            "idle_duration_seconds": idle_duration_seconds,
            "recommended_config": recommended_config,
        },
        source=source,
    ))


# =============================================================================
# Batch Scheduling Events (December 27, 2025)
# =============================================================================
# P0 fix for pipeline coordination - emits when batches are scheduled/dispatched


async def emit_batch_scheduled(
    batch_id: str,
    batch_type: str,
    config_key: str,
    job_count: int,
    target_nodes: list[str],
    reason: str = "normal",
    source: str = "selfplay_scheduler",
) -> None:
    """Emit a BATCH_SCHEDULED event when a batch of jobs is scheduled.

    Used by SelfplayScheduler to signal when a batch of jobs has been selected
    for dispatch. Enables pipeline coordination to track batch operations.

    Args:
        batch_id: Unique batch identifier
        batch_type: Type of batch ("selfplay", "training", "tournament")
        config_key: Configuration key (e.g., "hex8_2p")
        job_count: Number of jobs in the batch
        target_nodes: List of node IDs selected for dispatch
        reason: Why the batch was scheduled (e.g., "priority_change", "idle_detection")
        source: Component that scheduled the batch
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BATCH_SCHEDULED,
        payload={
            "batch_id": batch_id,
            "batch_type": batch_type,
            "config_key": config_key,
            "job_count": job_count,
            "target_nodes": target_nodes,
            "reason": reason,
            "timestamp": time.time(),
        },
        source=source,
    ))


async def emit_batch_dispatched(
    batch_id: str,
    batch_type: str,
    config_key: str,
    jobs_dispatched: int,
    jobs_failed: int = 0,
    target_nodes: list[str] | None = None,
    source: str = "job_dispatcher",
) -> None:
    """Emit a BATCH_DISPATCHED event when batch jobs are sent to nodes.

    Used after jobs have been dispatched to signal the pipeline that work
    is in progress.

    Args:
        batch_id: Unique batch identifier (same as BATCH_SCHEDULED)
        batch_type: Type of batch ("selfplay", "training", "tournament")
        config_key: Configuration key (e.g., "hex8_2p")
        jobs_dispatched: Number of jobs successfully dispatched
        jobs_failed: Number of jobs that failed to dispatch
        target_nodes: List of node IDs that received jobs
        source: Component that dispatched the batch
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BATCH_DISPATCHED,
        payload={
            "batch_id": batch_id,
            "batch_type": batch_type,
            "config_key": config_key,
            "jobs_dispatched": jobs_dispatched,
            "jobs_failed": jobs_failed,
            "target_nodes": target_nodes or [],
            "timestamp": time.time(),
        },
        source=source,
    ))


# =============================================================================
# P0.5 Missing Event Emitters (December 2025)
# =============================================================================
# These were identified as "ghost events" - event types defined but never emitted.


async def emit_data_stale(
    config: str,
    data_age_hours: float,
    last_sync_time: str,
    threshold_hours: float = 1.0,
    source: str = "training_freshness",
) -> None:
    """Emit a DATA_STALE event when training data is older than threshold.

    P0.5 Dec 2025: Closes feedback loop for data freshness monitoring.
    This enables training_freshness.py and sync daemons to respond to stale data.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        data_age_hours: How old the data is in hours
        last_sync_time: ISO timestamp of last data sync
        threshold_hours: Threshold at which data is considered stale
        source: Component that detected staleness
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_STALE,
        payload={
            "config": config,
            "data_age_hours": data_age_hours,
            "last_sync_time": last_sync_time,
            "threshold_hours": threshold_hours,
        },
        source=source,
    ))


async def emit_curriculum_advanced(
    config: str,
    old_tier: str,
    new_tier: str,
    trigger_reason: str,
    elo: float = 0.0,
    win_rate: float = 0.0,
    games_at_tier: int = 0,
    source: str = "curriculum_integration",
) -> None:
    """Emit a CURRICULUM_ADVANCED event when curriculum tier progresses.

    P0.5 Dec 2025: Closes curriculum feedback loop.
    This enables SelfplayScheduler to adjust priorities based on curriculum progression.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        old_tier: Previous curriculum tier (e.g., "BEGINNER")
        new_tier: New curriculum tier (e.g., "INTERMEDIATE")
        trigger_reason: What triggered the advancement (elo, winrate, games)
        elo: Current Elo rating
        win_rate: Current win rate
        games_at_tier: Number of games played at old tier
        source: Component that triggered advancement
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_ADVANCED,
        payload={
            "config": config,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "trigger_reason": trigger_reason,
            "elo": elo,
            "win_rate": win_rate,
            "games_at_tier": games_at_tier,
        },
        source=source,
    ))


async def emit_curriculum_rollback_completed(
    config_key: str,
    old_weight: float,
    new_weight: float,
    elo_delta: float,
    trigger_reason: str = "regression_detected",
    source: str = "curriculum_integration",
) -> None:
    """Emit a CURRICULUM_ROLLBACK_COMPLETED event when curriculum weight is reduced.

    Sprint 16.1 (Jan 3, 2026): Confirmation event for observability when curriculum
    weight is rolled back due to regression or other quality issues. Enables:
    - Monitoring dashboards to track rollback frequency
    - SelfplayScheduler to adjust priorities after rollback
    - Alert systems to notify on repeated rollbacks

    Args:
        config_key: Board configuration (e.g., "hex8_2p")
        old_weight: Previous curriculum weight (0.0-1.0)
        new_weight: New reduced weight (0.0-1.0)
        elo_delta: Elo change that triggered rollback (negative for regression)
        trigger_reason: What triggered the rollback (e.g., "regression_detected")
        source: Component that triggered rollback
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_ROLLBACK_COMPLETED,
        payload={
            "config_key": config_key,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "elo_delta": elo_delta,
            "trigger_reason": trigger_reason,
            "weight_reduction_pct": (1 - new_weight / old_weight) * 100 if old_weight > 0 else 0,
        },
        source=source,
    ))


async def emit_daemon_status_changed(
    daemon_name: str,
    hostname: str,
    old_status: str,
    new_status: str,
    reason: str = "",
    error: str | None = None,
    source: str = "daemon_manager",
) -> None:
    """Emit a DAEMON_STATUS_CHANGED event for daemon health transitions.

    P0.5 Dec 2025: Enables watchdog to react to daemon health state changes.
    Useful for detecting stuck, crashed, or restarted daemons.

    Args:
        daemon_name: Name of the daemon (e.g., "selfplay_scheduler")
        hostname: Host running the daemon
        old_status: Previous status (running, stopped, error)
        new_status: New status (running, stopped, error, stuck)
        reason: Why the status changed (timeout, exception, signal, restart)
        error: Error message if transition was due to error
        source: Component reporting the change
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STATUS_CHANGED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason,
            "error": error,
        },
        source=source,
    ))


async def emit_data_fresh(
    config: str,
    data_age_hours: float,
    last_sync_time: str,
    threshold_hours: float = 1.0,
    source: str = "training_freshness",
) -> None:
    """Emit a DATA_FRESH event when training data meets freshness requirements.

    P0.5 Dec 2025: Closes feedback loop for data freshness monitoring.
    Counterpart to DATA_STALE - signals that sync is NOT needed.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        data_age_hours: How old the data is in hours
        last_sync_time: ISO timestamp of last data sync
        threshold_hours: Threshold at which data is considered stale
        source: Component that checked freshness
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_FRESH,
        payload={
            "config": config,
            "data_age_hours": data_age_hours,
            "last_sync_time": last_sync_time,
            "threshold_hours": threshold_hours,
        },
        source=source,
    ))


async def emit_sync_triggered(
    config: str,
    reason: str,
    source_host: str = "",
    target_hosts: list[str] | None = None,
    data_age_hours: float = 0.0,
    source: str = "auto_sync_daemon",
) -> None:
    """Emit a SYNC_TRIGGERED event when data sync is initiated due to staleness.

    P0.5 Dec 2025: Closes feedback loop for sync coordination.
    Enables components to track when syncs are triggered and why.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        reason: Why sync was triggered (stale_data, manual_request, new_games)
        source_host: Host that initiated the sync
        target_hosts: List of hosts being synced to
        data_age_hours: How stale the data was when sync was triggered
        source: Component that triggered the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SYNC_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "source_host": source_host,
            "target_hosts": target_hosts or [],
            "data_age_hours": data_age_hours,
        },
        source=source,
    ))


# =============================================================================
# Curriculum Weight Events
# =============================================================================
# STATUS: RESERVED - Emitter exists, no subscribers yet (Dec 2025)
#
# Purpose: Track changes to curriculum/opponent weights for auditing and debugging.
# When curriculum integration adjusts weights (exploration, opponent strength, etc.),
# this event provides a trail for understanding why weights changed.
#
# Intended subscribers (to be implemented Q1 2026):
#   - WeightHistoryTracker: Store weight change history for analysis
#   - CurriculumDashboard: Visualize weight evolution over time
#   - SelfplayScheduler: Could adjust priorities based on weight changes
#
# The emit function is intended to be called from curriculum integration components
# when they adjust training weights.
# =============================================================================

async def emit_weight_updated(
    config: str,
    component: str,
    old_weight: float,
    new_weight: float,
    reason: str,
    trigger_event: str = "",
    source: str = "curriculum_integration",
) -> None:
    """Emit a WEIGHT_UPDATED event when curriculum/opponent weights change.

    P0.5 Dec 2025: Closes curriculum feedback loop.
    Enables tracking of weight changes for debugging and auditing.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        component: What weight changed (opponent, curriculum_tier, exploration)
        old_weight: Previous weight value
        new_weight: New weight value
        reason: Why the weight changed (elo_velocity, win_rate, manual)
        trigger_event: Event that triggered this change (if any)
        source: Component that changed the weight
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.WEIGHT_UPDATED,
        payload={
            "config": config,
            "component": component,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "reason": reason,
            "trigger_event": trigger_event,
        },
        source=source,
    ))


# =============================================================================
# Cluster Health Event Emitters (December 2025 - Phase 21)
# =============================================================================
# These enable key daemons to emit cluster health events for visibility.


async def emit_node_unhealthy(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    gpu_utilization: float | None = None,
    disk_used_percent: float | None = None,
    consecutive_failures: int = 0,
    source: str = "",
) -> None:
    """Emit NODE_UNHEALTHY event when a node is detected as unhealthy.

    Args:
        node_id: Identifier for the node
        reason: Why the node is unhealthy
        node_ip: Node IP address
        gpu_utilization: GPU utilization percentage (0-100)
        disk_used_percent: Disk usage percentage (0-100)
        consecutive_failures: Number of consecutive health check failures
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_UNHEALTHY,
        payload={
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "gpu_utilization": gpu_utilization,
            "disk_used_percent": disk_used_percent,
            "consecutive_failures": consecutive_failures,
        },
        source=source,
    ))


async def emit_node_recovered(
    node_id: str,
    *,
    node_ip: str = "",
    recovery_time_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit NODE_RECOVERED event when a node recovers to healthy state.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        recovery_time_seconds: How long recovery took
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_RECOVERED,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "recovery_time_seconds": recovery_time_seconds,
        },
        source=source,
    ))


async def emit_node_suspect(
    node_id: str,
    *,
    node_ip: str = "",
    last_seen: float | None = None,
    seconds_since_heartbeat: float = 0.0,
    source: str = "",
) -> None:
    """Emit NODE_SUSPECT event when a node enters SUSPECT state.

    December 2025: Added for peer state transition tracking.
    SUSPECT is a grace period between ALIVE and DEAD to reduce false positives.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        last_seen: Timestamp when node was last seen
        seconds_since_heartbeat: Seconds since last heartbeat
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_SUSPECT,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "last_seen": last_seen,
            "seconds_since_heartbeat": seconds_since_heartbeat,
        },
        source=source,
    ))


async def emit_node_retired(
    node_id: str,
    *,
    node_ip: str = "",
    reason: str = "manual",
    last_seen: float | None = None,
    total_uptime_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit NODE_RETIRED event when a node is retired from the cluster.

    December 2025: Added for peer state transition tracking.
    Retired nodes are excluded from job allocation but may be recovered later.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        reason: Why the node was retired (manual, timeout, error, capacity)
        last_seen: Timestamp when node was last seen
        total_uptime_seconds: Total uptime before retirement
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_RETIRED,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "reason": reason,
            "last_seen": last_seen,
            "total_uptime_seconds": total_uptime_seconds,
        },
        source=source,
    ))


async def emit_node_incompatible_with_workload(
    node_id: str,
    *,
    node_ip: str = "",
    gpu_vram_gb: float = 0.0,
    has_gpu: bool = False,
    reason: str = "",
    compatible_configs: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit NODE_INCOMPATIBLE_WITH_WORKLOAD when a node has no compatible selfplay configs.

    December 2025: Added to track nodes that are idle but cannot run any selfplay
    due to GPU/memory constraints. This prevents wasted evaluation cycles.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        gpu_vram_gb: GPU VRAM in GB
        has_gpu: Whether node has a GPU
        reason: Why the node is incompatible (e.g., "no_compatible_configs", "gpu_too_small")
        compatible_configs: List of configs node could run (empty if none)
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_INCOMPATIBLE_WITH_WORKLOAD,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "gpu_vram_gb": gpu_vram_gb,
            "has_gpu": has_gpu,
            "reason": reason,
            "compatible_configs": compatible_configs or [],
        },
        source=source,
    ))


async def emit_health_check_passed(
    node_id: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    latency_ms: float | None = None,
    source: str = "",
) -> None:
    """Emit HEALTH_CHECK_PASSED event after successful health check.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        check_type: Type of health check (ssh, p2p, gpu, general)
        latency_ms: Health check latency in milliseconds
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HEALTH_CHECK_PASSED,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "check_type": check_type,
            "latency_ms": latency_ms,
        },
        source=source,
    ))


async def emit_health_check_failed(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    error: str = "",
    source: str = "",
) -> None:
    """Emit HEALTH_CHECK_FAILED event after failed health check.

    Args:
        node_id: Identifier for the node
        reason: Why the health check failed
        node_ip: Node IP address
        check_type: Type of health check (ssh, p2p, gpu, general)
        error: Error message if any
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HEALTH_CHECK_FAILED,
        payload={
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "check_type": check_type,
            "error": error,
        },
        source=source,
    ))


async def emit_p2p_cluster_healthy(
    healthy_nodes: int,
    node_count: int,
    *,
    source: str = "",
) -> None:
    """Emit P2P_CLUSTER_HEALTHY event when cluster becomes healthy.

    Args:
        healthy_nodes: Number of healthy nodes
        node_count: Total node count
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.P2P_CLUSTER_HEALTHY,
        payload={
            "healthy": True,
            "healthy_nodes": healthy_nodes,
            "node_count": node_count,
        },
        source=source,
    ))


async def emit_p2p_cluster_unhealthy(
    healthy_nodes: int,
    node_count: int,
    *,
    alerts: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit P2P_CLUSTER_UNHEALTHY event when cluster becomes unhealthy.

    Args:
        healthy_nodes: Number of healthy nodes
        node_count: Total node count
        alerts: List of alert messages
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.P2P_CLUSTER_UNHEALTHY,
        payload={
            "healthy": False,
            "healthy_nodes": healthy_nodes,
            "node_count": node_count,
            "alerts": alerts or [],
        },
        source=source,
    ))


async def emit_p2p_restarted(
    *,
    trigger: str = "recovery_daemon",
    restart_count: int = 0,
    previous_state: str = "",
    source: str = "",
) -> None:
    """Emit P2P_RESTARTED event when P2P orchestrator successfully restarts.

    Dec 30, 2025: Added for event subscription resilience. Subscribers can
    use this to ensure they resubscribe after P2P mesh recovery.

    Args:
        trigger: What triggered the restart (recovery_daemon, manual, etc.)
        restart_count: Total number of restarts since daemon start
        previous_state: State before restart (healthy, unhealthy, partitioned)
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.P2P_RESTARTED,
        payload={
            "trigger": trigger,
            "restart_count": restart_count,
            "previous_state": previous_state,
            "timestamp": time.time(),
        },
        source=source,
    ))


# =============================================================================
# Disk Space Events (December 2025)
# =============================================================================


async def emit_disk_space_low(
    host: str,
    usage_percent: float,
    free_gb: float,
    *,
    threshold: float = 70.0,
    source: str = "",
) -> None:
    """Emit DISK_SPACE_LOW event when disk usage exceeds threshold.

    Args:
        host: Hostname/node ID
        usage_percent: Current disk usage percentage (0-100)
        free_gb: Free disk space in GB
        threshold: The threshold that was exceeded
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DISK_SPACE_LOW,
        payload={
            "host": host,
            "usage_percent": usage_percent,
            "free_gb": free_gb,
            "threshold": threshold,
        },
        source=source,
    ))


# =============================================================================
# Availability/Provisioning Events (December 28, 2025)
# =============================================================================


async def emit_capacity_low(
    current_gpus: int,
    min_gpus: int,
    provider: str = "",
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit CAPACITY_LOW event when GPU capacity drops below threshold.

    Args:
        current_gpus: Current number of active GPUs
        min_gpus: Minimum GPU threshold
        provider: Provider with low capacity (optional, empty for cluster-wide)
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CAPACITY_LOW,
        payload={
            "current_gpus": current_gpus,
            "min_gpus": min_gpus,
            "provider": provider,
        },
        source=source,
    ))


async def emit_capacity_restored(
    current_gpus: int,
    min_gpus: int,
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit CAPACITY_RESTORED event when capacity is back above threshold.

    Args:
        current_gpus: Current number of active GPUs
        min_gpus: Minimum GPU threshold
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CAPACITY_RESTORED,
        payload={
            "current_gpus": current_gpus,
            "min_gpus": min_gpus,
        },
        source=source,
    ))


async def emit_node_provisioned(
    node_id: str,
    provider: str,
    gpu_type: str,
    ip_address: str = "",
    *,
    source: str = "Provisioner",
) -> None:
    """Emit NODE_PROVISIONED event when a new node is created.

    Args:
        node_id: Unique identifier for the new node
        provider: Cloud provider name (vast, lambda, runpod, etc.)
        gpu_type: GPU type on the node (e.g., "GH200", "H100")
        ip_address: Node IP address if available
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_PROVISIONED,
        payload={
            "node_id": node_id,
            "provider": provider,
            "gpu_type": gpu_type,
            "ip_address": ip_address,
        },
        source=source,
    ))


async def emit_node_provision_failed(
    provider: str,
    gpu_type: str,
    error: str,
    *,
    source: str = "Provisioner",
) -> None:
    """Emit NODE_PROVISION_FAILED event when node provisioning fails.

    Args:
        provider: Cloud provider that failed
        gpu_type: GPU type requested
        error: Error message
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_PROVISION_FAILED,
        payload={
            "provider": provider,
            "gpu_type": gpu_type,
            "error": error,
        },
        source=source,
    ))


async def emit_budget_exceeded(
    budget_type: str,
    limit: float,
    current: float,
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit BUDGET_EXCEEDED event when spending exceeds limit.

    Args:
        budget_type: Type of budget exceeded ("hourly" or "daily")
        limit: Budget limit in USD
        current: Current spending in USD
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BUDGET_EXCEEDED,
        payload={
            "budget_type": budget_type,
            "limit_usd": limit,
            "current_usd": current,
        },
        source=source,
    ))


async def emit_budget_alert(
    budget_type: str,
    limit: float,
    current: float,
    threshold_percent: float = 80.0,
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit BUDGET_ALERT event when approaching budget threshold.

    Args:
        budget_type: Type of budget ("hourly" or "daily")
        limit: Budget limit in USD
        current: Current spending in USD
        threshold_percent: Threshold percentage that triggered alert
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BUDGET_ALERT,
        payload={
            "budget_type": budget_type,
            "limit_usd": limit,
            "current_usd": current,
            "threshold_percent": threshold_percent,
        },
        source=source,
    ))


# =============================================================================
# Selfplay Coordination Events (December 2025)
# =============================================================================
# P0 fix for autonomous training loop - enables SelfplayScheduler feedback


async def emit_selfplay_rate_changed(
    config_key: str,
    old_rate: float,
    new_rate: float,
    reason: str,
    source: str = "selfplay_scheduler",
) -> None:
    """Emit a SELFPLAY_RATE_CHANGED event when rate multiplier changes >20%.

    Used to notify IdleResourceDaemon and other consumers of significant
    changes to selfplay rate.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_RATE_CHANGED,
        payload={
            "config_key": config_key,
            "old_rate": old_rate,
            "new_rate": new_rate,
            "reason": reason,
        },
        source=source,
    ))


async def emit_selfplay_allocation_updated(
    config_key: str,
    allocation_weights: dict[str, float],
    reason: str,
    exploration_boost: float = 1.0,
    source: str = "selfplay_scheduler",
) -> None:
    """Emit a SELFPLAY_ALLOCATION_UPDATED event when allocation weights change.

    Used to notify IdleResourceDaemon and SelfplayScheduler of changes to
    curriculum weights, exploration boosts, or priority adjustments.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_ALLOCATION_UPDATED,
        payload={
            "config_key": config_key,
            "allocation_weights": allocation_weights,
            "exploration_boost": exploration_boost,
            "reason": reason,
        },
        source=source,
    ))


async def emit_node_capacity_updated(
    node_id: str,
    gpu_memory_gb: float,
    gpu_utilization: float,
    cpu_utilization: float,
    available_slots: int,
    reason: str = "capacity_update",
    source: str = "health_check_orchestrator",
    *,
    queue_depth: int = 0,
) -> None:
    """Emit a NODE_CAPACITY_UPDATED event when node capacity changes.

    Used by SelfplayScheduler and ResourceMonitoringCoordinator to track
    available capacity for job scheduling.

    Sprint 10 (Jan 3, 2026): Unified emitter for NODE_CAPACITY_UPDATED.
    All emission locations should use this function (or its sync variant)
    to ensure consistent payloads.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_CAPACITY_UPDATED,
        payload={
            "node_id": node_id,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_utilization": gpu_utilization,
            "cpu_utilization": cpu_utilization,
            "available_slots": available_slots,
            "queue_depth": queue_depth,
            "reason": reason,
        },
        source=source,
    ))


def emit_node_capacity_updated_sync(
    node_id: str,
    gpu_memory_gb: float = 0.0,
    gpu_utilization: float = 0.0,
    cpu_utilization: float = 0.0,
    available_slots: int = 0,
    reason: str = "capacity_update",
    source: str = "health_check_orchestrator",
    *,
    queue_depth: int = 0,
) -> None:
    """Synchronous version of emit_node_capacity_updated.

    Sprint 10 (Jan 3, 2026): Unified synchronous emitter for NODE_CAPACITY_UPDATED.
    Use this in sync contexts (heartbeat loops, health checks).

    All callers should use this function to ensure consistent payloads:
    - health_check_orchestrator.py
    - sync_router.py
    - p2p_orchestrator.py

    Args:
        node_id: Node identifier
        gpu_memory_gb: Available GPU memory in GB
        gpu_utilization: GPU utilization percentage (0-100)
        cpu_utilization: CPU utilization percentage (0-100)
        available_slots: Number of available job slots
        reason: Reason for capacity update
        source: Source component emitting the event
        queue_depth: Current work queue depth
    """
    try:
        bus = get_event_bus()
        if bus:
            bus.publish_sync(DataEvent(
                event_type=DataEventType.NODE_CAPACITY_UPDATED,
                payload={
                    "node_id": node_id,
                    "gpu_memory_gb": gpu_memory_gb,
                    "gpu_utilization": gpu_utilization,
                    "cpu_utilization": cpu_utilization,
                    "available_slots": available_slots,
                    "queue_depth": queue_depth,
                    "reason": reason,
                },
                source=source,
            ))
    except (AttributeError, RuntimeError):
        # Event bus not available - non-critical
        pass


# =============================================================================
# Parity Validation Events (December 29, 2025)
# =============================================================================


async def emit_parity_failure_rate_changed(
    new_rate: float,
    old_rate: float,
    board_type: str = "",
    num_players: int = 0,
    total_checked: int = 0,
    *,
    source: str = "ParityValidator",
) -> None:
    """Emit PARITY_FAILURE_RATE_CHANGED event when parity failure rate changes significantly.

    Args:
        new_rate: New parity failure rate (0.0-1.0)
        old_rate: Previous parity failure rate (0.0-1.0)
        board_type: Board type being validated (if config-specific)
        num_players: Number of players (if config-specific)
        total_checked: Total games checked in this batch
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PARITY_FAILURE_RATE_CHANGED,
        payload={
            "new_rate": new_rate,
            "old_rate": old_rate,
            "board_type": board_type,
            "num_players": num_players,
            "total_checked": total_checked,
            "rate_change": new_rate - old_rate,
        },
        source=source,
    ))

