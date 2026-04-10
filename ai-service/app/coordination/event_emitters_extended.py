"""Late-added event emitter wrappers split out from ``event_emitters``."""

from __future__ import annotations

import time
from typing import Any

from app.coordination.event_router import DataEventType


async def _emit_data_event(event_type: DataEventType, payload: dict[str, Any], **kwargs: Any) -> bool:
    from app.coordination.event_emitters import _emit_data_event as emit_data_event

    return await emit_data_event(event_type, payload, **kwargs)


async def emit_curriculum_updated(
    config_key: str,
    new_weight: float,
    trigger: str = "automatic",
    all_weights: dict[str, float] | None = None,
    **metadata,
) -> bool:
    """Emit CURRICULUM_REBALANCED event for a single config update.

    This is a convenience wrapper for emit_curriculum_rebalanced when
    updating a single config's weight. Used by the curriculum feedback
    system to notify the selfplay orchestrator of weight changes.

    December 2025: Added for Phase 1 self-improvement feedback loop.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        new_weight: New weight for this config
        trigger: What triggered the update (promotion, elo_change, plateau, etc.)
        all_weights: Optional dict of all current weights
        **metadata: Additional event metadata

    Returns:
        True if event was emitted successfully

    Example:
        await emit_curriculum_updated("square8_2p", 1.3, trigger="promotion")
    """
    from app.coordination import event_emitters

    return await event_emitters.emit_curriculum_rebalanced(
        config=config_key,
        old_weights={},  # Old weights often not available for single updates
        new_weights=all_weights or {config_key: new_weight},
        reason=f"config_update_{config_key}",
        trigger=trigger,
        config_key=config_key,
        new_weight=new_weight,
        **metadata,
    )


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: dict,
    new_weights: dict,
    reason: str,
    trigger: str = "automatic",
    **metadata,
) -> bool:
    """Emit CURRICULUM_REBALANCED event."""
    return await _emit_data_event(
        DataEventType.CURRICULUM_REBALANCED,
        {
            "config": config,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "reason": reason,
            "trigger": trigger,
            **metadata,
        },
        log_message=f"Emitted curriculum_rebalanced for {config}: {reason}",
        log_level="info",
    )


async def emit_training_triggered(
    config: str,
    job_id: str,
    trigger_reason: str,
    game_count: int = 0,
    threshold: int = 0,
    priority: str = "normal",
    **metadata,
) -> bool:
    """Emit event when training is triggered (before it starts)."""
    return await _emit_data_event(
        DataEventType.TRAINING_THRESHOLD_REACHED,
        {
            "config": config,
            "job_id": job_id,
            "trigger_reason": trigger_reason,
            "games": game_count,
            "threshold": threshold,
            "priority": priority,
            "event_subtype": "training_triggered",
            **metadata,
        },
        log_message=f"Emitted training_triggered for {config}: {trigger_reason}",
        log_level="info",
    )


# =============================================================================
# Cluster Health Events (December 2025)
# =============================================================================
# These emitters consolidate the try/except boilerplate from:
# - cluster_watchdog_daemon.py
# - unified_node_health_daemon.py
# - node_recovery_daemon.py
# - unified_health_manager.py


async def emit_node_unhealthy(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    gpu_utilization: float | None = None,
    disk_used_percent: float | None = None,
    consecutive_failures: int = 0,
    source: str = "",
) -> bool:
    """Emit NODE_UNHEALTHY event when a node is detected as unhealthy."""
    return await _emit_data_event(
        DataEventType.NODE_UNHEALTHY,
        {
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "gpu_utilization": gpu_utilization,
            "disk_used_percent": disk_used_percent,
            "consecutive_failures": consecutive_failures,
        },
        source=source or "event_emitters",
        log_message=f"Emitted node_unhealthy for {node_id}: {reason}",
        log_level="warning",
    )


async def emit_health_check_passed(
    node_id: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    latency_ms: float | None = None,
    source: str = "",
) -> bool:
    """Emit HEALTH_CHECK_PASSED event after successful health check."""
    return await _emit_data_event(
        DataEventType.HEALTH_CHECK_PASSED,
        {
            "node_id": node_id,
            "node_ip": node_ip,
            "check_type": check_type,
            "latency_ms": latency_ms,
        },
        source=source or "event_emitters",
        log_message=f"Emitted health_check_passed for {node_id}",
    )


async def emit_health_check_failed(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    error: str = "",
    source: str = "",
) -> bool:
    """Emit HEALTH_CHECK_FAILED event after failed health check."""
    return await _emit_data_event(
        DataEventType.HEALTH_CHECK_FAILED,
        {
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "check_type": check_type,
            "error": error,
        },
        source=source or "event_emitters",
        log_message=f"Emitted health_check_failed for {node_id}: {reason}",
        log_level="warning",
    )


async def emit_p2p_cluster_healthy(
    healthy_nodes: int,
    node_count: int,
    *,
    source: str = "",
) -> bool:
    """Emit P2P_CLUSTER_HEALTHY event when cluster becomes healthy."""
    return await _emit_data_event(
        DataEventType.P2P_CLUSTER_HEALTHY,
        {"healthy": True, "healthy_nodes": healthy_nodes, "node_count": node_count},
        source=source or "event_emitters",
        log_message=f"Emitted p2p_cluster_healthy: {healthy_nodes}/{node_count} nodes",
        log_level="info",
    )


async def emit_p2p_cluster_unhealthy(
    healthy_nodes: int,
    node_count: int,
    *,
    alerts: list[str] | None = None,
    source: str = "",
) -> bool:
    """Emit P2P_CLUSTER_UNHEALTHY event when cluster becomes unhealthy."""
    return await _emit_data_event(
        DataEventType.P2P_CLUSTER_UNHEALTHY,
        {
            "healthy": False,
            "healthy_nodes": healthy_nodes,
            "node_count": node_count,
            "alerts": alerts or [],
        },
        source=source or "event_emitters",
        log_message=f"Emitted p2p_cluster_unhealthy: {healthy_nodes}/{node_count} nodes",
        log_level="warning",
    )


async def emit_split_brain_detected(
    leaders_seen: list[str],
    *,
    severity: str = "warning",
    voter_count: int = 0,
    resolution_action: str = "step_down",
    source: str = "",
) -> bool:
    """Emit SPLIT_BRAIN_DETECTED event when multiple leaders are detected.

    December 2025: Critical for cluster coordination - indicates P2P split-brain
    condition where multiple nodes believe they are the leader. This triggers:
    - AlertManager: Send critical alert
    - UnifiedHealthManager: Track cluster degradation
    - LeadershipCoordinator: Initiate resolution

    Args:
        leaders_seen: List of node IDs claiming leadership
        severity: "warning" (2 leaders) or "critical" (3+ leaders)
        voter_count: Number of voter nodes in quorum
        resolution_action: Action taken (step_down, force_election, wait)
        source: Source component emitting the event
    """
    return await _emit_data_event(
        DataEventType.SPLIT_BRAIN_DETECTED,
        {
            "leaders_seen": leaders_seen,
            "leader_count": len(leaders_seen),
            "severity": severity,
            "voter_count": voter_count,
            "resolution_action": resolution_action,
        },
        source=source or "event_emitters",
        log_message=f"Emitted split_brain_detected: {len(leaders_seen)} leaders ({severity})",
        log_level="error" if severity == "critical" else "warning",
    )


async def emit_p2p_node_dead(
    node_id: str,
    *,
    reason: str = "timeout",
    last_seen: float | None = None,
    offline_duration_seconds: float = 0.0,
    source: str = "",
) -> bool:
    """Emit P2P_NODE_DEAD event when a node is confirmed dead.

    December 2025: Distinct from HOST_OFFLINE - this indicates a node that
    has been confirmed dead (not just temporarily offline) and requires
    work reassignment. Subscribers use this for:
    - SelfplayScheduler: Mark node as unavailable for selfplay allocation
    - UnifiedQueuePopulator: Reassign jobs from dead node
    """
    return await _emit_data_event(
        DataEventType.P2P_NODE_DEAD,
        {
            "node_id": node_id,
            "reason": reason,
            "last_seen": last_seen or time.time(),
            "offline_duration_seconds": offline_duration_seconds,
        },
        source=source or "event_emitters",
        log_message=f"Emitted p2p_node_dead for {node_id}: {reason}",
        log_level="warning",
    )


# =============================================================================
# Replication Repair Events (December 2025)
# =============================================================================


async def emit_repair_completed(
    game_id: str,
    source_nodes: list[str],
    target_nodes: list[str],
    duration_seconds: float,
    new_replica_count: int,
    source: str = "",
    **metadata,
) -> bool:
    """Emit REPAIR_COMPLETED event when a replication repair succeeds.

    December 2025: Wired into unified_replication_daemon.py for pipeline
    coordination of successful repair operations.

    Args:
        game_id: ID of the game that was repaired
        source_nodes: Nodes that provided the data
        target_nodes: Nodes that received the data
        duration_seconds: How long the repair took
        new_replica_count: Current replica count after repair
        source: Event source identifier
        **metadata: Additional event metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.REPAIR_COMPLETED,
        {
            "game_id": game_id,
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "duration_seconds": duration_seconds,
            "new_replica_count": new_replica_count,
            **metadata,
        },
        source=source or "unified_replication_daemon",
        log_message=f"Emitted repair_completed for {game_id}",
        log_level="info",
    )


async def emit_repair_failed(
    game_id: str,
    source_nodes: list[str],
    target_nodes: list[str],
    error: str,
    duration_seconds: float = 0.0,
    current_replica_count: int = 0,
    source: str = "",
    **metadata,
) -> bool:
    """Emit REPAIR_FAILED event when a replication repair fails.

    December 2025: Wired into unified_replication_daemon.py for pipeline
    coordination of failed repair operations.

    Args:
        game_id: ID of the game that failed repair
        source_nodes: Nodes that were supposed to provide data
        target_nodes: Nodes that were supposed to receive data
        error: Error message describing failure
        duration_seconds: How long before failure
        current_replica_count: Current replica count (unchanged)
        source: Event source identifier
        **metadata: Additional event metadata

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        DataEventType.REPAIR_FAILED,
        {
            "game_id": game_id,
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "error": error,
            "duration_seconds": duration_seconds,
            "current_replica_count": current_replica_count,
            **metadata,
        },
        source=source or "unified_replication_daemon",
        log_message=f"Emitted repair_failed for {game_id}: {error}",
        log_level="warning",
    )


async def emit_generic_event(
    event_type: DataEventType,
    payload: dict,
    source: str = "",
    log_message: str = "",
    log_level: str = "debug",
) -> bool:
    """Emit a generic event with custom payload.

    December 2025: Added for flexible event emission from daemons that
    need to emit events not covered by typed emitters.

    Args:
        event_type: The DataEventType to emit
        payload: Event payload dictionary
        source: Event source identifier
        log_message: Optional log message
        log_level: Logging level ("debug", "info", "warning")

    Returns:
        True if emitted successfully
    """
    return await _emit_data_event(
        event_type,
        payload,
        source=source or "generic",
        log_message=log_message or f"Emitted {event_type.name}",
        log_level=log_level,
    )
