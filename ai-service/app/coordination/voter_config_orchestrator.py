"""Voter Configuration Orchestrator.

Jan 20, 2026: Coordinates voter configuration synchronization across the cluster.
Integrates with VoterConfigSyncLoop and VoterConfigChangeProtocol.

This orchestrator:
1. Subscribes to config drift and sync events
2. Monitors cluster health for config-related issues
3. Triggers emergency syncs when drift is detected
4. Coordinates with the change protocol for safe voter list updates

Usage:
    orchestrator = VoterConfigOrchestrator.get_instance()
    await orchestrator.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.coordination.handler_base import HandlerBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class VoterConfigOrchestratorConfig:
    """Configuration for VoterConfigOrchestrator."""
    # How often to check cluster config health
    health_check_interval_seconds: float = 60.0
    # Maximum version delta before triggering emergency sync
    max_version_delta: int = 3
    # Time to wait after drift detection before triggering sync
    drift_debounce_seconds: float = 10.0
    # Maximum time to wait for cluster to converge after change
    convergence_timeout_seconds: float = 120.0


@dataclass
class ConfigHealthStatus:
    """Health status of voter config across cluster."""
    local_version: int
    local_hash: str
    peer_versions: dict[str, int]
    highest_version: int
    lowest_version: int
    version_spread: int
    peers_at_highest: int
    peers_behind: int
    is_healthy: bool
    health_reason: str
    timestamp: float = field(default_factory=time.time)


class VoterConfigOrchestrator(HandlerBase):
    """Orchestrator for voter configuration synchronization.

    Coordinates between:
    - VoterConfigSyncLoop (P2P background drift detection)
    - VoterConfigChangeProtocol (joint consensus for changes)
    - VoterConfigManager (local config storage)

    Subscribes to events:
    - VOTER_CONFIG_DRIFT_DETECTED: Triggers emergency sync
    - VOTER_CONFIG_SYNC_COMPLETED: Updates health metrics
    - VOTER_CONFIG_CHANGE_COMMITTED: Broadcasts to all peers
    - VOTER_CONFIG_CHANGE_FAILED: Logs and alerts
    - LEADER_ELECTED: Triggers config validation
    """

    _instance: "VoterConfigOrchestrator | None" = None

    def __init__(
        self,
        config: VoterConfigOrchestratorConfig | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration options
        """
        super().__init__(
            name="voter_config_orchestrator",
            cycle_interval=config.health_check_interval_seconds if config else 60.0,
        )
        self._config = config or VoterConfigOrchestratorConfig()
        self._last_drift_time: float = 0.0
        self._last_sync_time: float = 0.0
        self._consecutive_drift_events: int = 0
        self._health_status: ConfigHealthStatus | None = None
        self._p2p_orchestrator: Any = None

    @classmethod
    def get_instance(cls) -> "VoterConfigOrchestrator":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def set_p2p_orchestrator(self, orchestrator: Any) -> None:
        """Set the P2P orchestrator reference.

        Args:
            orchestrator: P2POrchestrator instance
        """
        self._p2p_orchestrator = orchestrator

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this orchestrator."""
        return {
            "VOTER_CONFIG_DRIFT_DETECTED": self._on_drift_detected,
            "VOTER_CONFIG_SYNC_COMPLETED": self._on_sync_completed,
            "VOTER_CONFIG_CHANGE_COMMITTED": self._on_change_committed,
            "VOTER_CONFIG_CHANGE_FAILED": self._on_change_failed,
            "LEADER_ELECTED": self._on_leader_elected,
        }

    async def _run_cycle(self) -> None:
        """Main orchestrator cycle - check config health."""
        try:
            health = await self._check_config_health()
            self._health_status = health

            if not health.is_healthy:
                logger.warning(
                    f"[VoterConfigOrchestrator] Unhealthy config state: "
                    f"{health.health_reason}"
                )

                # Trigger sync if we're behind
                if health.version_spread > self._config.max_version_delta:
                    await self._trigger_emergency_sync(health)

        except Exception as e:
            logger.error(f"[VoterConfigOrchestrator] Health check error: {e}")

    async def _check_config_health(self) -> ConfigHealthStatus:
        """Check the health of voter config across the cluster.

        Returns:
            ConfigHealthStatus with current state
        """
        try:
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager

            manager = get_voter_config_manager()
            local_config = manager.get_current()

            local_version = local_config.version if local_config else 0
            local_hash = local_config.sha256_hash[:16] if local_config else ""

            # Collect peer versions
            peer_versions: dict[str, int] = {}

            if self._p2p_orchestrator:
                peers = getattr(self._p2p_orchestrator, "peers", {})
                for peer_id, peer_info in peers.items():
                    version = getattr(peer_info, "voter_config_version", None)
                    if version is not None:
                        peer_versions[peer_id] = version

            if not peer_versions:
                return ConfigHealthStatus(
                    local_version=local_version,
                    local_hash=local_hash,
                    peer_versions={},
                    highest_version=local_version,
                    lowest_version=local_version,
                    version_spread=0,
                    peers_at_highest=0,
                    peers_behind=0,
                    is_healthy=True,
                    health_reason="no_peer_data",
                )

            highest_version = max(peer_versions.values())
            lowest_version = min(peer_versions.values())
            version_spread = highest_version - lowest_version

            peers_at_highest = sum(1 for v in peer_versions.values() if v == highest_version)
            peers_behind = sum(1 for v in peer_versions.values() if v < highest_version)

            # Determine health
            is_healthy = True
            health_reason = "ok"

            if local_version < highest_version:
                is_healthy = False
                health_reason = f"local_behind: v{local_version} < v{highest_version}"
            elif version_spread > self._config.max_version_delta:
                is_healthy = False
                health_reason = f"high_spread: {version_spread} versions"
            elif peers_behind > len(peer_versions) // 2:
                is_healthy = False
                health_reason = f"many_peers_behind: {peers_behind}/{len(peer_versions)}"

            return ConfigHealthStatus(
                local_version=local_version,
                local_hash=local_hash,
                peer_versions=peer_versions,
                highest_version=highest_version,
                lowest_version=lowest_version,
                version_spread=version_spread,
                peers_at_highest=peers_at_highest,
                peers_behind=peers_behind,
                is_healthy=is_healthy,
                health_reason=health_reason,
            )

        except Exception as e:
            logger.error(f"[VoterConfigOrchestrator] Health check failed: {e}")
            return ConfigHealthStatus(
                local_version=0,
                local_hash="",
                peer_versions={},
                highest_version=0,
                lowest_version=0,
                version_spread=0,
                peers_at_highest=0,
                peers_behind=0,
                is_healthy=False,
                health_reason=f"error: {e}",
            )

    async def _trigger_emergency_sync(self, health: ConfigHealthStatus) -> None:
        """Trigger an emergency sync when drift is detected.

        Args:
            health: Current health status
        """
        logger.info(
            f"[VoterConfigOrchestrator] Triggering emergency sync: "
            f"local v{health.local_version}, highest v{health.highest_version}"
        )

        # Find a peer with the highest version
        target_peer = None
        for peer_id, version in health.peer_versions.items():
            if version == health.highest_version:
                target_peer = peer_id
                break

        if target_peer is None:
            logger.warning("[VoterConfigOrchestrator] No peer with highest version found")
            return

        # Get the sync loop and trigger a pull
        if self._p2p_orchestrator:
            sync_loop = getattr(self._p2p_orchestrator, "_voter_config_sync_loop", None)
            if sync_loop and hasattr(sync_loop, "_pull_from_peer"):
                success = await sync_loop._pull_from_peer(target_peer)
                if success:
                    logger.info(
                        f"[VoterConfigOrchestrator] Emergency sync from {target_peer} succeeded"
                    )
                else:
                    logger.warning(
                        f"[VoterConfigOrchestrator] Emergency sync from {target_peer} failed"
                    )

    def _on_drift_detected(self, event_data: dict[str, Any]) -> None:
        """Handle VOTER_CONFIG_DRIFT_DETECTED event.

        Args:
            event_data: Event payload
        """
        now = time.time()

        # Debounce
        if now - self._last_drift_time < self._config.drift_debounce_seconds:
            return

        self._last_drift_time = now
        self._consecutive_drift_events += 1

        local_version = event_data.get("local_version", 0)
        highest_version = event_data.get("highest_version", 0)
        description = event_data.get("description", "")

        logger.info(
            f"[VoterConfigOrchestrator] Drift detected: "
            f"v{local_version} < v{highest_version} ({description})"
        )

        # If too many consecutive drift events, force a health check
        if self._consecutive_drift_events >= 3:
            logger.warning(
                f"[VoterConfigOrchestrator] {self._consecutive_drift_events} "
                f"consecutive drift events, forcing health check"
            )
            asyncio.create_task(self._run_cycle())

    def _on_sync_completed(self, event_data: dict[str, Any]) -> None:
        """Handle VOTER_CONFIG_SYNC_COMPLETED event.

        Args:
            event_data: Event payload
        """
        self._last_sync_time = time.time()
        self._consecutive_drift_events = 0

        version = event_data.get("version", 0)
        duration = event_data.get("sync_duration_seconds", 0)

        logger.info(
            f"[VoterConfigOrchestrator] Sync completed: "
            f"v{version} in {duration:.1f}s"
        )

    def _on_change_committed(self, event_data: dict[str, Any]) -> None:
        """Handle VOTER_CONFIG_CHANGE_COMMITTED event.

        Args:
            event_data: Event payload
        """
        old_version = event_data.get("old_version", 0)
        new_version = event_data.get("new_version", 0)
        old_voters = event_data.get("old_voters", [])
        new_voters = event_data.get("new_voters", [])

        logger.info(
            f"[VoterConfigOrchestrator] Change committed: "
            f"v{old_version} -> v{new_version}, "
            f"voters: {len(old_voters)} -> {len(new_voters)}"
        )

        # Reset drift tracking
        self._consecutive_drift_events = 0

    def _on_change_failed(self, event_data: dict[str, Any]) -> None:
        """Handle VOTER_CONFIG_CHANGE_FAILED event.

        Args:
            event_data: Event payload
        """
        proposal_id = event_data.get("proposal_id", "")
        reason = event_data.get("reason", "unknown")
        phase_reached = event_data.get("phase_reached", "")

        logger.warning(
            f"[VoterConfigOrchestrator] Change failed: "
            f"{proposal_id} - {reason} (phase: {phase_reached})"
        )

    def _on_leader_elected(self, event_data: dict[str, Any]) -> None:
        """Handle LEADER_ELECTED event.

        Args:
            event_data: Event payload
        """
        new_leader = event_data.get("leader_id", "")
        logger.info(
            f"[VoterConfigOrchestrator] New leader elected: {new_leader}, "
            f"triggering config validation"
        )

        # Trigger a health check when leadership changes
        asyncio.create_task(self._run_cycle())

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status.

        Returns:
            Dict with health status
        """
        if self._health_status is None:
            return {
                "status": "unknown",
                "last_check": 0,
            }

        return {
            "status": "healthy" if self._health_status.is_healthy else "unhealthy",
            "local_version": self._health_status.local_version,
            "local_hash": self._health_status.local_hash,
            "highest_version": self._health_status.highest_version,
            "version_spread": self._health_status.version_spread,
            "peers_at_highest": self._health_status.peers_at_highest,
            "peers_behind": self._health_status.peers_behind,
            "health_reason": self._health_status.health_reason,
            "last_check": self._health_status.timestamp,
            "consecutive_drift_events": self._consecutive_drift_events,
            "last_drift_time": self._last_drift_time,
            "last_sync_time": self._last_sync_time,
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check data for DaemonManager integration.

        Returns:
            Dict with health status and metrics
        """
        base_health = super().health_check()
        status = self.get_health_status()

        return {
            **base_health,
            "config_health": status.get("status", "unknown"),
            "local_version": status.get("local_version", 0),
            "version_spread": status.get("version_spread", 0),
            "consecutive_drift_events": status.get("consecutive_drift_events", 0),
        }


# Module-level accessor
def get_voter_config_orchestrator() -> VoterConfigOrchestrator:
    """Get the VoterConfigOrchestrator singleton.

    Returns:
        VoterConfigOrchestrator instance
    """
    return VoterConfigOrchestrator.get_instance()
