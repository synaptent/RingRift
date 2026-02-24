"""P2P Event Bridge - Wires P2P orchestrator events to the coordination EventRouter.

This module provides a bridge between P2P HTTP handlers and the central event system,
enabling distributed coordination across the cluster.

Usage:
    from scripts.p2p.p2p_event_bridge import (
        emit_p2p_work_completed,
        emit_p2p_gauntlet_completed,
        emit_p2p_leader_changed,
        emit_p2p_node_online,
        emit_p2p_node_offline,
    )

    # From work_queue handler when work completes
    await emit_p2p_work_completed(
        work_id="abc123",
        work_type="selfplay",
        config_key="hex8_2p",
        result={"games_generated": 500},
        node_id="runpod-h100",
    )

Events emitted:
    - SELFPLAY_COMPLETE: When selfplay work finishes
    - TRAINING_COMPLETED: When training work finishes
    - EVALUATION_COMPLETED: When gauntlet/evaluation work finishes
    - HOST_ONLINE: When a node joins the cluster
    - HOST_OFFLINE: When a node leaves the cluster
    - ELO_UPDATED: When Elo ratings are synced
    - P2P_CLUSTER_HEALTHY: When cluster reaches quorum
    - P2P_CLUSTER_UNHEALTHY: When cluster loses quorum

Created: December 2025
"""

from __future__ import annotations

import asyncio

from app.core.async_context import safe_create_task
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Cross-Node Event Forwarding (Jan 2026)
# =============================================================================

# Coordinator URL for forwarding events (gauntlet runs on P2P leader, auto_promotion on coordinator)
_COORDINATOR_EVENT_URL: str | None = None


def _get_coordinator_event_url() -> str | None:
    """Get the coordinator's /event endpoint URL.

    Returns URL like 'http://mac-studio:8790/event' or None if not configured.
    Uses RINGRIFT_COORDINATOR_HOST env var, or tries cluster config.
    """
    global _COORDINATOR_EVENT_URL

    if _COORDINATOR_EVENT_URL is not None:
        return _COORDINATOR_EVENT_URL if _COORDINATOR_EVENT_URL else None

    # Check env var first
    host = os.environ.get("RINGRIFT_COORDINATOR_HOST", "").strip()
    if not host:
        # Try cluster config
        try:
            from app.config.cluster_config import get_coordinator_node
            node = get_coordinator_node()
            if node:
                host = node.best_ip or node.tailscale_ip or node.ssh_host
        except Exception:
            pass

    if not host:
        _COORDINATOR_EVENT_URL = ""  # Cache negative result
        return None

    port = int(os.environ.get("RINGRIFT_HEALTH_PORT", "8790"))
    _COORDINATOR_EVENT_URL = f"http://{host}:{port}/event"
    logger.info(f"[P2PEventBridge] Coordinator event URL: {_COORDINATOR_EVENT_URL}")
    return _COORDINATOR_EVENT_URL


async def _forward_event_to_coordinator(
    event_type: str,
    payload: dict[str, Any],
    source: str = "p2p_remote",
) -> bool:
    """Forward event to coordinator's /event endpoint.

    This enables cross-node event propagation for scenarios where:
    - Gauntlet runs on P2P leader (e.g., lambda-gh200-5)
    - auto_promotion_daemon subscribes on coordinator (e.g., mac-studio)

    Returns True if forwarded successfully, False otherwise.
    """
    url = _get_coordinator_event_url()
    if not url:
        logger.debug("[P2PEventBridge] No coordinator URL configured, skipping forward")
        return False

    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "event_type": event_type,
                    "payload": payload,
                    "source": source,
                },
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status == 200:
                    logger.info(f"[P2PEventBridge] Forwarded {event_type} to coordinator")
                    return True
                else:
                    text = await resp.text()
                    logger.warning(
                        f"[P2PEventBridge] Coordinator rejected event: {resp.status} {text}"
                    )
                    return False

    except ImportError:
        logger.debug("[P2PEventBridge] aiohttp not available, cannot forward events")
        return False
    except asyncio.TimeoutError:
        logger.warning(f"[P2PEventBridge] Timeout forwarding {event_type} to coordinator")
        return False
    except Exception as e:
        logger.warning(f"[P2PEventBridge] Failed to forward {event_type}: {e}")
        return False

# Import canonical config key parser (December 2025 - DRY consolidation)
try:
    from app.utils.canonical_naming import parse_config_key as _canonical_parse_config_key
    HAS_CANONICAL_PARSING = True
except ImportError:
    HAS_CANONICAL_PARSING = False


def _parse_config_key_safe(config_key: str) -> tuple[str, int]:
    """Parse config_key into (board_type, num_players) with fallback.

    Uses canonical parse_config_key when available, with inline fallback
    for standalone P2P mode.

    Returns:
        (board_type, num_players) tuple. Returns ("", 0) if parsing fails.
    """
    if not config_key or "_" not in config_key:
        return "", 0

    # Try canonical parser first (December 2025)
    if HAS_CANONICAL_PARSING:
        try:
            return _canonical_parse_config_key(config_key)
        except ValueError:
            return "", 0

    # Fallback for standalone P2P mode
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        return "", 0
    board_type = parts[0]
    try:
        num_players = int(parts[1].rstrip("p"))
        return board_type, num_players
    except ValueError:
        return "", 0

# =============================================================================
# Event Router Import (with fallback for standalone P2P mode)
# =============================================================================

try:
    from app.coordination.event_router import (
        publish,
        publish_sync,
        get_router,
        DataEventType,
    )
    HAS_EVENT_ROUTER = True
except ImportError:
    HAS_EVENT_ROUTER = False
    logger.info("[P2PEventBridge] Event router not available, events will be logged only")

    async def publish(*args, **kwargs):
        pass

    def publish_sync(*args, **kwargs):
        pass

    def get_router():
        return None

    DataEventType = None


# =============================================================================
# Config Key Validation (December 2025)
# =============================================================================

# Valid board types for config_key validation
_VALID_BOARD_TYPES = {"hex8", "hexagonal", "square8", "square19"}


def _validate_config_key(config_key: str, context: str = "") -> bool:
    """Validate config_key format before event emission.

    Valid formats:
    - Standard: {board_type}_{num_players}p (e.g., "hex8_2p", "square19_4p")
    - Special: "all" (used by elo_sync for cluster-wide operations)
    - Node-specific: "node:{node_id}" (used by selfplay_scheduler for per-node tracking)

    Returns True if valid, False otherwise (logs warning if invalid).
    """
    if not config_key:
        logger.warning(f"[P2PEventBridge] {context}: Empty config_key")
        return False

    # Allow special "all" config for cluster-wide operations
    if config_key == "all":
        return True

    # Allow node-specific config keys
    if config_key.startswith("node:"):
        return True

    # Validate standard format: {board_type}_{num_players}p
    if "_" not in config_key:
        logger.warning(f"[P2PEventBridge] {context}: Invalid config_key format '{config_key}' (missing underscore)")
        return False

    parts = config_key.rsplit("_", 1)
    if len(parts) != 2:
        logger.warning(f"[P2PEventBridge] {context}: Invalid config_key format '{config_key}'")
        return False

    board_type, players_str = parts

    # Validate board_type
    if board_type not in _VALID_BOARD_TYPES:
        logger.warning(f"[P2PEventBridge] {context}: Unknown board_type '{board_type}' in config_key '{config_key}'")
        return False

    # Validate num_players format (should be Np where N is 2-4)
    if not players_str.endswith("p"):
        logger.warning(f"[P2PEventBridge] {context}: Invalid players format in config_key '{config_key}' (should end with 'p')")
        return False

    try:
        num_players = int(players_str.rstrip("p"))
        if num_players not in (2, 3, 4):
            logger.warning(f"[P2PEventBridge] {context}: Invalid num_players {num_players} in config_key '{config_key}' (must be 2, 3, or 4)")
            return False
    except ValueError:
        logger.warning(f"[P2PEventBridge] {context}: Cannot parse num_players from config_key '{config_key}'")
        return False

    return True


# =============================================================================
# Work Queue Event Emitters
# =============================================================================

async def emit_p2p_work_completed(
    work_id: str,
    work_type: str,
    config_key: str,
    result: dict[str, Any],
    node_id: str,
    duration_seconds: float = 0.0,
) -> None:
    """Emit event when P2P work queue item completes.

    Routes to appropriate event type based on work_type:
    - selfplay -> SELFPLAY_COMPLETE
    - training -> TRAINING_COMPLETED
    - tournament/gauntlet -> EVALUATION_COMPLETED
    """
    if not HAS_EVENT_ROUTER:
        logger.info(f"[P2PEventBridge] Work completed: {work_type} {config_key} on {node_id}")
        return

    # Dec 2025: Validate config_key before emission (logs warning if invalid)
    _validate_config_key(config_key, f"emit_p2p_work_completed({work_type})")

    # Parse config_key using canonical parser (December 2025 - DRY consolidation)
    board_type, num_players = _parse_config_key_safe(config_key)

    timestamp = datetime.now().isoformat()

    try:
        if work_type == "selfplay":
            await publish(
                event_type="SELFPLAY_COMPLETE",
                payload={
                    "task_id": work_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "config_key": config_key,
                    "games_generated": result.get("games_generated", 0),
                    "success": True,
                    "node_id": node_id,
                    "duration_seconds": duration_seconds,
                    "selfplay_type": result.get("selfplay_type", "standard"),
                    "iteration": result.get("iteration", 0),
                    "timestamp": timestamp,
                },
                source="p2p_work_queue",
            )
            logger.debug(f"[P2PEventBridge] Emitted SELFPLAY_COMPLETE for {config_key}")

        elif work_type == "training":
            await publish(
                event_type="TRAINING_COMPLETED",
                payload={
                    "config_key": config_key,
                    "model_id": result.get("model_id", ""),
                    "model_path": result.get("model_path", ""),
                    "val_loss": result.get("val_loss", 0.0),
                    "train_loss": result.get("train_loss", 0.0),
                    "epochs": result.get("epochs", 0),
                    "success": True,
                    "node_id": node_id,
                    "duration_seconds": duration_seconds,
                    "timestamp": timestamp,
                },
                source="p2p_work_queue",
            )
            logger.debug(f"[P2PEventBridge] Emitted TRAINING_COMPLETED for {config_key}")

        elif work_type in ("tournament", "gauntlet"):
            await publish(
                event_type="EVALUATION_COMPLETED",
                payload={
                    "model_id": result.get("best_model") or result.get("model_id", ""),
                    "board_type": board_type,
                    "num_players": num_players,
                    "elo": result.get("best_elo") or result.get("elo", 0.0),
                    "win_rate": result.get("win_rate", 0.0),
                    "games_played": result.get("games_played", 0),
                    "elo_delta": result.get("elo_delta", 0.0),
                    "node_id": node_id,
                    "timestamp": timestamp,
                },
                source="p2p_work_queue",
            )
            logger.debug(f"[P2PEventBridge] Emitted EVALUATION_COMPLETED for {config_key}")

        else:
            # Generic work completion (no specific event type)
            logger.debug(f"[P2PEventBridge] Work completed: {work_type} (no event emitted)")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit work completed event: {e}")


async def emit_p2p_work_failed(
    work_id: str,
    work_type: str,
    config_key: str,
    error: str,
    node_id: str,
) -> None:
    """Emit event when P2P work queue item fails."""
    if not HAS_EVENT_ROUTER:
        logger.warning(f"[P2PEventBridge] Work failed: {work_type} {config_key}: {error}")
        return

    # Dec 2025: Validate config_key before emission (logs warning if invalid)
    _validate_config_key(config_key, f"emit_p2p_work_failed({work_type})")

    timestamp = datetime.now().isoformat()

    try:
        if work_type == "training":
            await publish(
                event_type="TRAINING_FAILED",
                payload={
                    "config_key": config_key,
                    "error": error,
                    "error_details": error,
                    "node_id": node_id,
                    "success": False,
                    "timestamp": timestamp,
                },
                source="p2p_work_queue",
            )
            logger.debug(f"[P2PEventBridge] Emitted TRAINING_FAILED for {config_key}")
        else:
            # For other work types, just log
            logger.warning(f"[P2PEventBridge] Work failed: {work_type} {config_key}: {error}")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit work failed event: {e}")


# =============================================================================
# Gauntlet Event Emitters
# =============================================================================

async def emit_p2p_gauntlet_completed(
    model_id: str,
    baseline_id: str,
    config_key: str,
    wins: int,
    total_games: int,
    win_rate: float,
    passed: bool,
    node_id: str,
) -> None:
    """Emit EVALUATION_COMPLETED event when gauntlet finishes.

    Jan 2026: Also forwards event to coordinator for cross-node propagation.
    This ensures auto_promotion_daemon on coordinator receives the event even
    when gauntlet runs on P2P leader (a different node).
    """
    # Dec 2025: Validate config_key before emission (logs warning if invalid)
    _validate_config_key(config_key, "emit_p2p_gauntlet_completed")

    # Parse config_key using canonical parser (December 2025 - DRY consolidation)
    board_type, num_players = _parse_config_key_safe(config_key)

    # Build payload once for both local and remote emission
    payload = {
        "model_id": model_id,
        "board_type": board_type,
        "num_players": num_players,
        "elo": 0.0,  # Gauntlet doesn't compute Elo directly
        "win_rate": win_rate,
        "games_played": total_games,
        "elo_delta": 0.0,
        "opponents": [baseline_id],
        "passed": passed,
        "node_id": node_id,
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Emit locally (for any local subscribers)
    if HAS_EVENT_ROUTER:
        try:
            await publish(
                event_type="EVALUATION_COMPLETED",
                payload=payload,
                source="p2p_gauntlet",
            )
            logger.debug(f"[P2PEventBridge] Emitted EVALUATION_COMPLETED for {model_id}")
        except Exception as e:
            logger.error(f"[P2PEventBridge] Failed to emit gauntlet completed event: {e}")
    else:
        logger.info(
            f"[P2PEventBridge] Gauntlet completed: {model_id} vs {baseline_id} "
            f"({wins}/{total_games}, {win_rate:.1%}, passed={passed})"
        )

    # 2. Forward to coordinator (Jan 2026 - cross-node event propagation)
    # This ensures auto_promotion_daemon on coordinator receives the event
    await _forward_event_to_coordinator(
        event_type="EVALUATION_COMPLETED",
        payload=payload,
        source="p2p_gauntlet",
    )


# =============================================================================
# Node Lifecycle Event Emitters
# =============================================================================

async def emit_p2p_node_online(
    node_id: str,
    host_type: str = "",
    capabilities: dict[str, Any] | None = None,
) -> None:
    """Emit HOST_ONLINE event when a node joins the cluster."""
    if not HAS_EVENT_ROUTER:
        logger.info(f"[P2PEventBridge] Node online: {node_id} ({host_type})")
        return

    try:
        await publish(
            event_type="HOST_ONLINE",
            payload={
                "node_id": node_id,
                "host_id": node_id,  # Alias for compatibility
                "host_type": host_type,
                "capabilities": capabilities or {},
                "timestamp": datetime.now().isoformat(),
            },
            source="p2p_gossip",
        )
        logger.debug(f"[P2PEventBridge] Emitted HOST_ONLINE for {node_id}")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit node online event: {e}")


async def emit_p2p_node_offline(
    node_id: str,
    reason: str = "unreachable",
) -> None:
    """Emit HOST_OFFLINE event when a node leaves the cluster."""
    if not HAS_EVENT_ROUTER:
        logger.info(f"[P2PEventBridge] Node offline: {node_id} ({reason})")
        return

    try:
        await publish(
            event_type="HOST_OFFLINE",
            payload={
                "node_id": node_id,
                "host_id": node_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            },
            source="p2p_gossip",
        )
        logger.debug(f"[P2PEventBridge] Emitted HOST_OFFLINE for {node_id}")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit node offline event: {e}")


# =============================================================================
# Cluster Health Event Emitters
# =============================================================================

async def emit_p2p_cluster_healthy(
    alive_peers: int,
    total_peers: int,
    leader_id: str,
    quorum: bool = True,
) -> None:
    """Emit P2P_CLUSTER_HEALTHY event when cluster is healthy."""
    if not HAS_EVENT_ROUTER:
        logger.info(f"[P2PEventBridge] Cluster healthy: {alive_peers}/{total_peers} peers, leader={leader_id}")
        return

    try:
        await publish(
            event_type="P2P_CLUSTER_HEALTHY",
            payload={
                "alive_peers": alive_peers,
                "total_peers": total_peers,
                "leader_id": leader_id,
                "quorum": quorum,
                "timestamp": datetime.now().isoformat(),
            },
            source="p2p_election",
        )
        logger.debug(f"[P2PEventBridge] Emitted P2P_CLUSTER_HEALTHY")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit cluster healthy event: {e}")


async def emit_p2p_cluster_unhealthy(
    alive_peers: int,
    total_peers: int,
    reason: str = "quorum_lost",
) -> None:
    """Emit P2P_CLUSTER_UNHEALTHY event when cluster loses quorum or leader."""
    if not HAS_EVENT_ROUTER:
        logger.warning(f"[P2PEventBridge] Cluster unhealthy: {alive_peers}/{total_peers} peers, reason={reason}")
        return

    try:
        await publish(
            event_type="P2P_CLUSTER_UNHEALTHY",
            payload={
                "alive_peers": alive_peers,
                "total_peers": total_peers,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            },
            source="p2p_election",
        )
        logger.debug(f"[P2PEventBridge] Emitted P2P_CLUSTER_UNHEALTHY")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit cluster unhealthy event: {e}")


# =============================================================================
# Elo Sync Event Emitters
# =============================================================================

async def emit_p2p_elo_updated(
    model_id: str,
    config_key: str,
    old_elo: float,
    new_elo: float,
    games_played: int = 0,
) -> None:
    """Emit ELO_UPDATED event when Elo ratings are synced."""
    if not HAS_EVENT_ROUTER:
        logger.info(f"[P2PEventBridge] Elo updated: {model_id} {old_elo:.0f} -> {new_elo:.0f}")
        return

    # Dec 2025: Validate config_key before emission (logs warning if invalid)
    _validate_config_key(config_key, "emit_p2p_elo_updated")

    try:
        await publish(
            event_type="ELO_UPDATED",
            payload={
                "model_id": model_id,
                "config_key": config_key,
                "old_elo": old_elo,
                "new_elo": new_elo,
                "elo_delta": new_elo - old_elo,
                "games_played": games_played,
                "timestamp": datetime.now().isoformat(),
            },
            source="p2p_elo_sync",
        )
        logger.debug(f"[P2PEventBridge] Emitted ELO_UPDATED for {model_id}")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit elo updated event: {e}")


# =============================================================================
# Leader Election Event Emitters
# =============================================================================

async def emit_p2p_leader_changed(
    new_leader_id: str,
    old_leader_id: str,
    term: int = 0,
) -> None:
    """Emit LEADER_CHANGED event when P2P leader changes.

    Note: This is a P2P-specific event, not in the standard catalog.
    Consumers can subscribe to monitor cluster leadership.
    """
    if not HAS_EVENT_ROUTER:
        logger.info(f"[P2PEventBridge] Leader changed: {old_leader_id} -> {new_leader_id}")
        return

    try:
        await publish(
            event_type="LEADER_CHANGED",
            payload={
                "new_leader_id": new_leader_id,
                "old_leader_id": old_leader_id,
                "term": term,
                "timestamp": datetime.now().isoformat(),
            },
            source="p2p_election",
        )
        logger.debug(f"[P2PEventBridge] Emitted LEADER_CHANGED: {old_leader_id} -> {new_leader_id}")

    except Exception as e:
        logger.error(f"[P2PEventBridge] Failed to emit leader changed event: {e}")


# =============================================================================
# Sync versions for non-async contexts
# =============================================================================

def emit_p2p_work_completed_sync(
    work_id: str,
    work_type: str,
    config_key: str,
    result: dict[str, Any],
    node_id: str,
    duration_seconds: float = 0.0,
) -> None:
    """Synchronous version of emit_p2p_work_completed."""
    try:
        asyncio.get_running_loop()
        # If we're in an async context, schedule it
        safe_create_task(
            emit_p2p_work_completed(
                work_id, work_type, config_key, result, node_id, duration_seconds
            ),
            name=f"bridge-work-completed-{work_id}",
        )
    except RuntimeError:
        # No running loop - run synchronously
        asyncio.run(
            emit_p2p_work_completed(
                work_id, work_type, config_key, result, node_id, duration_seconds
            )
        )


def emit_p2p_node_online_sync(
    node_id: str,
    host_type: str = "",
    capabilities: dict[str, Any] | None = None,
) -> None:
    """Synchronous version of emit_p2p_node_online."""
    try:
        asyncio.get_running_loop()
        safe_create_task(emit_p2p_node_online(node_id, host_type, capabilities), name=f"bridge-node-online-{node_id}")
    except RuntimeError:
        asyncio.run(emit_p2p_node_online(node_id, host_type, capabilities))


def emit_p2p_node_offline_sync(node_id: str, reason: str = "unreachable") -> None:
    """Synchronous version of emit_p2p_node_offline."""
    try:
        asyncio.get_running_loop()
        safe_create_task(emit_p2p_node_offline(node_id, reason), name=f"bridge-node-offline-{node_id}")
    except RuntimeError:
        asyncio.run(emit_p2p_node_offline(node_id, reason))


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Work queue events
    "emit_p2p_work_completed",
    "emit_p2p_work_completed_sync",
    "emit_p2p_work_failed",
    # Gauntlet events
    "emit_p2p_gauntlet_completed",
    # Node lifecycle events
    "emit_p2p_node_online",
    "emit_p2p_node_online_sync",
    "emit_p2p_node_offline",
    "emit_p2p_node_offline_sync",
    # Cluster health events
    "emit_p2p_cluster_healthy",
    "emit_p2p_cluster_unhealthy",
    # Elo sync events
    "emit_p2p_elo_updated",
    # Leader election events
    "emit_p2p_leader_changed",
    # Constants
    "HAS_EVENT_ROUTER",
]
