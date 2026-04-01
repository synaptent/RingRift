"""Allocation event emission helpers for selfplay scheduling.

January 2026: Extracted from selfplay_scheduler.py as part of code quality
decomposition. These functions handle event emission for allocation updates,
starvation alerts, and related notifications.

Usage:
    from app.coordination.selfplay.allocation_events import (
        emit_allocation_updated,
        emit_starvation_alert,
        emit_idle_node_work_injected,
    )

    # Emit allocation update event
    emit_allocation_updated(
        allocation={"hex8_2p": {"node1": 100}},
        total_games=100,
        trigger="allocate_batch",
    )
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.selfplay_priority_types import ConfigPriority

logger = logging.getLogger(__name__)

# Import starvation thresholds
try:
    from app.config.coordination_defaults import (
        DATA_STARVATION_ULTRA_THRESHOLD,
        DATA_STARVATION_ULTRA_MULTIPLIER,
    )
except ImportError:
    DATA_STARVATION_ULTRA_THRESHOLD = 20
    DATA_STARVATION_ULTRA_MULTIPLIER = 10.0

__all__ = [
    "emit_allocation_updated",
    "emit_starvation_alert",
    "emit_idle_node_work_injected",
]


def emit_allocation_updated(
    allocation: dict[str, dict[str, int]] | None,
    total_games: int,
    trigger: str,
    config_key: str | None = None,
    config_priorities: dict[str, "ConfigPriority"] | None = None,
) -> None:
    """Emit SELFPLAY_ALLOCATION_UPDATED event.

    December 2025: Notifies downstream consumers (IdleResourceDaemon, feedback
    loops) when selfplay allocation has changed. This enables:
    - IdleResourceDaemon to know which configs are prioritized
    - Feedback loops to track allocation changes from their signals
    - Monitoring to track allocation patterns over time

    Args:
        allocation: Dict of config_key -> {node_id: games} for batch allocations
        total_games: Total games in this allocation
        trigger: What caused this allocation (e.g., "allocate_batch", "exploration_boost")
        config_key: Specific config that changed (for single-config updates)
        config_priorities: Optional priority dict for including exploration/curriculum data
    """
    try:
        from app.coordination.event_router import DataEventType, publish_sync

        # Build allocation summary
        if allocation:
            configs_allocated = list(allocation.keys())
            nodes_involved = set()
            for node_games in allocation.values():
                nodes_involved.update(node_games.keys())
        else:
            configs_allocated = [config_key] if config_key else []
            nodes_involved = set()

        payload = {
            "trigger": trigger,
            "total_games": total_games,
            "configs": configs_allocated,
            "node_count": len(nodes_involved),
            "timestamp": time.time(),
        }

        # Include exploration boosts for tracking feedback loop efficacy
        if config_key and config_priorities and config_key in config_priorities:
            priority = config_priorities[config_key]
            payload["exploration_boost"] = priority.exploration_boost
            payload["curriculum_weight"] = priority.curriculum_weight

        publish_sync(
            DataEventType.SELFPLAY_ALLOCATION_UPDATED,
            payload,
            source="allocation_events",
        )
        logger.debug(
            f"[allocation_events] Emitted SELFPLAY_ALLOCATION_UPDATED: "
            f"trigger={trigger}, games={total_games}, configs={len(configs_allocated)}"
        )

    except ImportError:
        pass  # Event system not available
    except Exception as e:
        logger.debug(f"[allocation_events] Failed to emit allocation update: {e}")


def emit_starvation_alert(
    config_key: str,
    game_count: int,
    tier: str,
) -> None:
    """Emit DATA_STARVATION_CRITICAL event to trigger priority dispatch.

    Jan 5, 2026: Added for automatic starvation response. When ULTRA starvation
    is detected (<20 games), this event enables QueuePopulatorLoop to auto-submit
    priority selfplay jobs without manual intervention.

    Args:
        config_key: Config with starvation (e.g., "square19_3p")
        game_count: Current game count for this config
        tier: Starvation tier ("ULTRA", "EMERGENCY", "CRITICAL")
    """
    try:
        from app.coordination.event_router import DataEventType, publish_sync

        payload = {
            "config_key": config_key,
            "game_count": game_count,
            "tier": tier,
            "threshold": DATA_STARVATION_ULTRA_THRESHOLD,
            "multiplier": DATA_STARVATION_ULTRA_MULTIPLIER,
            "timestamp": time.time(),
        }

        publish_sync(
            DataEventType.DATA_STARVATION_CRITICAL,
            payload,
            source="allocation_events",
        )
        logger.info(
            f"[allocation_events] Emitted DATA_STARVATION_CRITICAL: "
            f"{config_key} ({tier} tier, {game_count} games)"
        )

    except ImportError:
        pass  # Event system not available
    except Exception as e:
        logger.debug(f"[allocation_events] Failed to emit starvation alert: {e}")


def emit_idle_node_work_injected(
    node_id: str,
    config_key: str,
    games: int,
    reason: str = "idle_threshold_exceeded",
) -> None:
    """Emit event when work is injected to an idle node.

    Jan 2026: Tracks automatic work injection for idle nodes.
    Useful for monitoring cluster utilization patterns.

    Args:
        node_id: The node receiving work
        config_key: Config the work is for
        games: Number of games assigned
        reason: Why work was injected
    """
    try:
        from app.coordination.event_router import DataEventType, publish_sync

        payload = {
            "node_id": node_id,
            "config_key": config_key,
            "games": games,
            "reason": reason,
            "timestamp": time.time(),
        }

        publish_sync(
            DataEventType.IDLE_NODE_WORK_INJECTED,
            payload,
            source="allocation_events",
        )
        logger.debug(
            f"[allocation_events] Emitted IDLE_NODE_WORK_INJECTED: "
            f"node={node_id}, config={config_key}, games={games}"
        )

    except ImportError:
        pass  # Event system not available
    except Exception as e:
        logger.debug(f"[allocation_events] Failed to emit idle work injection: {e}")
