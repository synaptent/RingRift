"""Voter Configuration Synchronization Loop.

Jan 20, 2026: Implements automated voter config drift detection and synchronization.
Part of the consensus-safe voter configuration synchronization system.

This loop:
1. Periodically checks for config drift across cluster peers
2. Pulls newer configs from peers that have higher versions
3. Emits events when drift is detected or resolved

Usage:
    loop = VoterConfigSyncLoop(orchestrator)
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

from scripts.p2p.loops.base import BaseLoop, BackoffConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Configuration via environment variables
VOTER_CONFIG_SYNC_INTERVAL = int(
    os.environ.get("RINGRIFT_VOTER_CONFIG_SYNC_INTERVAL", "30")
)
VOTER_CONFIG_PULL_TIMEOUT = float(
    os.environ.get("RINGRIFT_VOTER_CONFIG_PULL_TIMEOUT", "10.0")
)
VOTER_CONFIG_MAX_VERSION_DELTA = int(
    os.environ.get("RINGRIFT_VOTER_CONFIG_MAX_VERSION_DELTA", "3")
)


@dataclass
class PeerConfigInfo:
    """Config info received from a peer via heartbeat or status."""
    node_id: str
    version: int
    hash_short: str
    updated_at: float = field(default_factory=time.time)


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    has_drift: bool
    local_version: int
    local_hash: str
    peer_versions: dict[str, int]  # node_id -> version
    highest_version: int
    highest_version_peers: list[str]
    drift_description: str = ""


class VoterConfigSyncLoop(BaseLoop):
    """Background loop for voter configuration synchronization.

    Detects config drift across the cluster and pulls newer configs
    from peers with higher version numbers.

    Attributes:
        orchestrator: Reference to P2POrchestrator for accessing peers and config
    """

    def __init__(self, orchestrator: Any):
        """Initialize the voter config sync loop.

        Args:
            orchestrator: P2POrchestrator instance (passed by reference)
        """
        super().__init__(
            name="voter_config_sync",
            interval=float(VOTER_CONFIG_SYNC_INTERVAL),
            backoff_config=BackoffConfig(
                initial_delay=5.0,
                max_delay=120.0,
                multiplier=1.5,
            ),
        )
        self._orchestrator = orchestrator
        self._peer_configs: dict[str, PeerConfigInfo] = {}
        self._last_drift_detected: float = 0.0
        self._drift_resolved_time: float = 0.0
        self._sync_in_progress: bool = False
        self._consecutive_sync_failures: int = 0

    async def _run_once(self) -> None:
        """Main sync loop iteration.

        1. Detect drift by comparing local config with peers
        2. If drift detected, attempt to pull from highest-version peer
        3. Emit events for monitoring
        """
        try:
            # Get local config info
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager
            manager = get_voter_config_manager()
            local_config = manager.get_current()

            local_version = local_config.version if local_config else 0
            local_hash = local_config.sha256_hash[:16] if local_config else ""

            # Collect peer config versions from recent heartbeats/status
            peer_versions = self._collect_peer_config_versions()

            # Analyze drift
            drift_result = self._analyze_drift(local_version, local_hash, peer_versions)

            if drift_result.has_drift:
                logger.info(
                    f"[VoterConfigSync] Drift detected: {drift_result.drift_description}"
                )
                self._last_drift_detected = time.time()
                await self._emit_drift_event(drift_result)

                # Attempt to pull from highest-version peer
                if drift_result.highest_version > local_version:
                    await self._pull_from_peer(drift_result.highest_version_peers[0])

            elif self._last_drift_detected > 0 and self._drift_resolved_time == 0:
                # Drift was previously detected but now resolved
                self._drift_resolved_time = time.time()
                logger.info(
                    f"[VoterConfigSync] Drift resolved: all peers at v{local_version}"
                )
                await self._emit_sync_completed_event(local_version)

        except Exception as e:
            logger.error(f"[VoterConfigSync] Error in sync loop: {e}")
            raise

    def _collect_peer_config_versions(self) -> dict[str, int]:
        """Collect voter config versions from all known peers.

        Uses peer info from status endpoint or heartbeat data.

        Returns:
            Dict mapping node_id to config version
        """
        versions: dict[str, int] = {}

        # Get peer info from orchestrator
        peers = getattr(self._orchestrator, "peers", {})
        for peer_id, peer_info in peers.items():
            # Try to get voter_config_version from peer info
            version = getattr(peer_info, "voter_config_version", None)
            if version is not None:
                versions[peer_id] = version
                # Update cache
                hash_short = getattr(peer_info, "voter_config_hash", "")
                self._peer_configs[peer_id] = PeerConfigInfo(
                    node_id=peer_id,
                    version=version,
                    hash_short=hash_short,
                )

        # Also check cached configs for peers we've heard from recently
        cutoff = time.time() - 120  # 2 minute staleness window
        for node_id, info in self._peer_configs.items():
            if node_id not in versions and info.updated_at > cutoff:
                versions[node_id] = info.version

        return versions

    def _analyze_drift(
        self,
        local_version: int,
        local_hash: str,
        peer_versions: dict[str, int],
    ) -> DriftResult:
        """Analyze config drift across the cluster.

        Args:
            local_version: Our current config version
            local_hash: Our current config hash (first 16 chars)
            peer_versions: Map of peer node_id to their config version

        Returns:
            DriftResult with analysis
        """
        if not peer_versions:
            return DriftResult(
                has_drift=False,
                local_version=local_version,
                local_hash=local_hash,
                peer_versions={},
                highest_version=local_version,
                highest_version_peers=[],
                drift_description="No peer config info available",
            )

        # Find highest version and which peers have it
        highest_version = max(peer_versions.values())
        highest_version_peers = [
            node_id for node_id, v in peer_versions.items() if v == highest_version
        ]

        # Check for version drift
        has_version_drift = highest_version > local_version

        # Check for hash drift at same version
        has_hash_drift = False
        for node_id, info in self._peer_configs.items():
            if info.version == local_version and info.hash_short and info.hash_short != local_hash:
                has_hash_drift = True
                break

        has_drift = has_version_drift or has_hash_drift

        # Build description
        if has_version_drift:
            description = (
                f"Local v{local_version} < remote v{highest_version} "
                f"(peers: {', '.join(highest_version_peers[:3])})"
            )
        elif has_hash_drift:
            description = f"Hash mismatch at v{local_version}"
        else:
            description = "No drift"

        return DriftResult(
            has_drift=has_drift,
            local_version=local_version,
            local_hash=local_hash,
            peer_versions=peer_versions,
            highest_version=highest_version,
            highest_version_peers=highest_version_peers,
            drift_description=description,
        )

    async def _pull_from_peer(self, peer_id: str) -> bool:
        """Pull voter config from a specific peer.

        Args:
            peer_id: Node ID of the peer to pull from

        Returns:
            True if config was successfully pulled and applied
        """
        if self._sync_in_progress:
            logger.debug(f"[VoterConfigSync] Sync already in progress, skipping pull")
            return False

        self._sync_in_progress = True
        try:
            # Get peer URL
            peers = getattr(self._orchestrator, "peers", {})
            peer_info = peers.get(peer_id)
            if not peer_info:
                logger.warning(f"[VoterConfigSync] Peer {peer_id} not found")
                return False

            # Build URL - try to get from peer info
            url = self._get_peer_url(peer_id, peer_info, "/voter-config")
            if not url:
                logger.warning(f"[VoterConfigSync] Could not build URL for {peer_id}")
                return False

            logger.info(f"[VoterConfigSync] Pulling config from {peer_id}")

            # Fetch config
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=VOTER_CONFIG_PULL_TIMEOUT),
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"[VoterConfigSync] Pull from {peer_id} failed: HTTP {response.status}"
                        )
                        self._consecutive_sync_failures += 1
                        return False

                    data = await response.json()

            # Validate response
            if "version" not in data or data.get("version", 0) == 0:
                logger.warning(f"[VoterConfigSync] Peer {peer_id} has no config")
                return False

            # Apply remote config
            from app.coordination.voter_config_types import VoterConfigVersion
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager

            remote_config = VoterConfigVersion.from_dict(data)
            manager = get_voter_config_manager()
            success, reason = manager.apply_remote_config(
                remote_config, source=f"sync_from_{peer_id}"
            )

            if success:
                logger.info(
                    f"[VoterConfigSync] Applied config v{remote_config.version} from {peer_id}"
                )
                self._consecutive_sync_failures = 0
                return True
            else:
                logger.debug(
                    f"[VoterConfigSync] Config from {peer_id} not applied: {reason}"
                )
                return False

        except asyncio.TimeoutError:
            logger.warning(f"[VoterConfigSync] Pull from {peer_id} timed out")
            self._consecutive_sync_failures += 1
            return False
        except Exception as e:
            logger.error(f"[VoterConfigSync] Pull from {peer_id} failed: {e}")
            self._consecutive_sync_failures += 1
            return False
        finally:
            self._sync_in_progress = False

    def _get_peer_url(self, peer_id: str, peer_info: Any, path: str) -> str | None:
        """Build URL for a peer endpoint.

        Args:
            peer_id: Node ID
            peer_info: NodeInfo object
            path: URL path (e.g., "/voter-config")

        Returns:
            Full URL or None if can't determine
        """
        # Try Tailscale IP first
        tailscale_ip = getattr(peer_info, "tailscale_ip", None)
        if tailscale_ip:
            port = getattr(peer_info, "port", 8770)
            return f"http://{tailscale_ip}:{port}{path}"

        # Try endpoint
        endpoint = getattr(peer_info, "endpoint", None)
        if endpoint:
            # Endpoint might include port
            if ":" in endpoint:
                return f"http://{endpoint}{path}"
            return f"http://{endpoint}:8770{path}"

        # Try IP address
        ip = getattr(peer_info, "ip", None)
        if ip:
            port = getattr(peer_info, "port", 8770)
            return f"http://{ip}:{port}{path}"

        return None

    async def _emit_drift_event(self, drift_result: DriftResult) -> None:
        """Emit VOTER_CONFIG_DRIFT_DETECTED event.

        Args:
            drift_result: Drift analysis result
        """
        try:
            # Use orchestrator's event emission if available
            emit_fn = getattr(self._orchestrator, "_safe_emit_event", None)
            if emit_fn:
                await asyncio.to_thread(
                    emit_fn,
                    "VOTER_CONFIG_DRIFT_DETECTED",
                    {
                        "local_version": drift_result.local_version,
                        "highest_version": drift_result.highest_version,
                        "highest_version_peers": drift_result.highest_version_peers[:5],
                        "peer_count": len(drift_result.peer_versions),
                        "description": drift_result.drift_description,
                        "timestamp": time.time(),
                    },
                )
        except Exception as e:
            logger.debug(f"[VoterConfigSync] Failed to emit drift event: {e}")

    async def _emit_sync_completed_event(self, version: int) -> None:
        """Emit VOTER_CONFIG_SYNC_COMPLETED event.

        Args:
            version: Config version that is now synced
        """
        try:
            emit_fn = getattr(self._orchestrator, "_safe_emit_event", None)
            if emit_fn:
                await asyncio.to_thread(
                    emit_fn,
                    "VOTER_CONFIG_SYNC_COMPLETED",
                    {
                        "version": version,
                        "sync_duration_seconds": time.time() - self._last_drift_detected,
                        "timestamp": time.time(),
                    },
                )
        except Exception as e:
            logger.debug(f"[VoterConfigSync] Failed to emit sync completed event: {e}")

    def update_peer_config(self, node_id: str, version: int, hash_short: str) -> None:
        """Update cached peer config info (called from heartbeat handler).

        Args:
            node_id: Peer node ID
            version: Peer's config version
            hash_short: First 16 chars of peer's config hash
        """
        self._peer_configs[node_id] = PeerConfigInfo(
            node_id=node_id,
            version=version,
            hash_short=hash_short,
        )

    def health_check(self) -> dict[str, Any]:
        """Return health check data for DaemonManager integration.

        Returns:
            Dict with health status and metrics
        """
        from scripts.p2p.managers.voter_config_manager import get_voter_config_manager

        try:
            manager = get_voter_config_manager()
            config = manager.get_current()
            local_version = config.version if config else 0
        except Exception:
            local_version = 0

        return {
            "name": self.name,
            "running": self._running,
            "enabled": self.enabled,
            "local_version": local_version,
            "tracked_peers": len(self._peer_configs),
            "consecutive_sync_failures": self._consecutive_sync_failures,
            "last_drift_detected": self._last_drift_detected,
            "drift_resolved_time": self._drift_resolved_time,
            "sync_in_progress": self._sync_in_progress,
            "stats": self._stats.to_dict(),
        }
