"""Automated config synchronization daemon.

Monitors distributed_hosts.yaml for changes on the coordinator and
propagates updates to all cluster nodes via CONFIG_UPDATED events.

Design principles:
- Only coordinator runs active file monitoring (mtime polling)
- Workers subscribe to CONFIG_UPDATED events and pull when needed
- Never expose config content in events (only hash + timestamp)
- Use mtime polling (no watchdog/inotify for cross-platform)
- Integrate with existing ClusterConfigCache versioning

January 2026 - Created to fix P2P voter config drift issue.
Root cause: distributed_hosts.yaml is in .gitignore and never syncs via git pull.
Each node has a local copy that can drift, causing voter mismatches.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from app.config.cluster_config import (
    ConfigVersion,
    clear_cluster_config_cache,
    get_config_version,
    load_cluster_config,
)
from app.config.env import env
from app.coordination.contracts import HealthCheckResult
from app.coordination.handler_base import HandlerBase
from app.distributed.data_events import DataEventType

logger = logging.getLogger(__name__)


class ConfigSyncDaemon(HandlerBase):
    """Daemon for automatic config synchronization across cluster.

    On coordinator:
    - Polls config file mtime every 30s (configurable)
    - Detects changes via ClusterConfigCache.get_version()
    - Emits CONFIG_UPDATED event with hash + timestamp
    - Tracks nodes that haven't synced and pushes after timeout

    On workers:
    - Subscribes to CONFIG_UPDATED event
    - Compares event version against local version
    - Pulls newer config via rsync/SSH if stale
    - Reloads ClusterConfigCache after pull
    """

    _event_source = "ConfigSyncDaemon"

    # Default config file path
    DEFAULT_CONFIG_PATH = "config/distributed_hosts.yaml"

    def __init__(
        self,
        check_interval: float = 30.0,
        sync_timeout: float = 120.0,
        push_delay: float = 60.0,
        max_retries: int = 3,
        config_path: str | None = None,
    ):
        """Initialize ConfigSyncDaemon.

        Args:
            check_interval: Seconds between config file checks (coordinator)
            sync_timeout: Timeout for rsync operations
            push_delay: Seconds to wait before active push to stale nodes
            max_retries: Max retries for failed sync operations
            config_path: Path to distributed_hosts.yaml (relative to ai-service)
        """
        super().__init__(
            name="config_sync",
            cycle_interval=check_interval,
        )
        self._sync_timeout = sync_timeout
        self._push_delay = push_delay
        self._max_retries = max_retries

        # Config file path
        self._config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        if not self._config_path.is_absolute():
            # Make relative to ai-service root
            ai_service_root = Path(__file__).parent.parent.parent
            self._config_path = ai_service_root / self._config_path

        # Version tracking
        self._last_known_version: ConfigVersion | None = None
        self._pending_sync_nodes: dict[str, float] = {}  # node_id -> first_seen_stale
        self._sync_failures: dict[str, int] = {}  # node_id -> failure_count

        # Coordinator detection
        self._is_coordinator = env.is_coordinator

        # Stats
        self._changes_detected = 0
        self._syncs_pushed = 0
        self._syncs_pulled = 0

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Subscribe to CONFIG_UPDATED for pull-based sync."""
        return {
            "CONFIG_UPDATED": self._on_config_updated,
            "config_updated": self._on_config_updated,  # Lowercase variant
        }

    async def _run_cycle(self) -> None:
        """Main cycle: detect changes and orchestrate sync."""
        if self._is_coordinator:
            await self._coordinator_cycle()
        else:
            await self._worker_cycle()

    async def _coordinator_cycle(self) -> None:
        """Coordinator: detect file changes and emit events."""
        try:
            # Check for config changes
            current_version = self._get_current_version()
            if current_version is None:
                logger.warning(f"[ConfigSync] Config file not found: {self._config_path}")
                return

            if self._has_version_changed(current_version):
                self._changes_detected += 1
                logger.info(
                    f"[ConfigSync] Detected config change: "
                    f"hash={current_version.content_hash[:8]}... "
                    f"(change #{self._changes_detected})"
                )

                # Emit CONFIG_UPDATED event
                await self._emit_config_update(current_version)
                self._last_known_version = current_version

            # Check for nodes that haven't synced (optional push)
            await self._push_to_stale_nodes()

        except Exception as e:
            logger.error(f"[ConfigSync] Coordinator cycle error: {e}")
            self._record_error(str(e))

    async def _worker_cycle(self) -> None:
        """Worker: periodic freshness check against coordinator.

        Workers mostly react to CONFIG_UPDATED events, but this provides
        a fallback health check in case events are missed.
        """
        # Workers are primarily event-driven via _on_config_updated
        # This cycle just logs status for debugging
        if self._stats.cycles_completed % 10 == 0:  # Every 10th cycle (~5 min)
            version = self._get_current_version()
            if version:
                logger.debug(
                    f"[ConfigSync] Worker config status: hash={version.content_hash[:8]}... "
                    f"mtime={time.strftime('%H:%M:%S', time.localtime(version.timestamp))}"
                )

    def _get_current_version(self) -> ConfigVersion | None:
        """Get current config version from ClusterConfigCache."""
        try:
            return get_config_version()
        except Exception as e:
            logger.warning(f"[ConfigSync] Failed to get config version: {e}")
            return None

    def _has_version_changed(self, current: ConfigVersion) -> bool:
        """Check if config version has changed since last check."""
        if self._last_known_version is None:
            # First check - initialize but don't emit
            self._last_known_version = current
            return False

        # Compare content hashes
        return current.content_hash != self._last_known_version.content_hash

    async def _emit_config_update(self, version: ConfigVersion) -> None:
        """Emit CONFIG_UPDATED event to notify all nodes."""
        payload = {
            "source_node": env.node_id,
            "hash": version.content_hash,
            "timestamp": version.timestamp,
            "config_path": str(self._config_path),
        }

        # Use safe event emission from HandlerBase
        success = await self._safe_emit_event_async(
            DataEventType.CONFIG_UPDATED.value,
            payload,
        )

        if success:
            logger.info(
                f"[ConfigSync] Emitted CONFIG_UPDATED: "
                f"hash={version.content_hash[:8]}... source={env.node_id}"
            )
        else:
            logger.warning("[ConfigSync] Failed to emit CONFIG_UPDATED event")

    async def _on_config_updated(self, event: Any) -> None:
        """Handle CONFIG_UPDATED event (worker nodes).

        Compares event version against local version and pulls if stale.
        """
        try:
            payload = self._get_payload(event)

            # Extract version info from event
            event_hash = payload.get("hash", "")
            event_timestamp = payload.get("timestamp", 0)
            source_node = payload.get("source_node", "unknown")

            # Skip if we're the source
            if source_node == env.node_id:
                return

            # Skip if coordinator (we're the source of truth)
            if self._is_coordinator:
                return

            # Compare with local version
            local_version = self._get_current_version()
            if local_version is None:
                logger.warning("[ConfigSync] Local config version unavailable")
                return

            # Check if event version is newer
            if event_timestamp <= local_version.timestamp:
                logger.debug(
                    f"[ConfigSync] Local config is up to date "
                    f"(local={local_version.timestamp:.0f}, event={event_timestamp:.0f})"
                )
                return

            # Check if hash already matches (already synced)
            if event_hash == local_version.content_hash:
                logger.debug("[ConfigSync] Local config hash already matches event")
                return

            # Pull newer config from source
            logger.info(
                f"[ConfigSync] Config update detected from {source_node}, "
                f"pulling new version (hash={event_hash[:8]}...)"
            )
            success = await self._pull_config(source_node)

            if success:
                self._syncs_pulled += 1
                logger.info(f"[ConfigSync] Successfully pulled config from {source_node}")
            else:
                logger.warning(f"[ConfigSync] Failed to pull config from {source_node}")

        except Exception as e:
            logger.error(f"[ConfigSync] Error handling CONFIG_UPDATED: {e}")
            self._record_error(str(e))

    async def _pull_config(self, source_node: str) -> bool:
        """Pull config from source node via rsync/SSH.

        Uses Tailscale IP if available, falls back to SSH host.

        Args:
            source_node: Node ID to pull from

        Returns:
            True if config was successfully pulled and reloaded
        """
        try:
            # Get source node config
            config = load_cluster_config()
            hosts = config.get("hosts", {})
            node_config = hosts.get(source_node, {})

            if not node_config:
                logger.warning(f"[ConfigSync] Unknown source node: {source_node}")
                return False

            # Get IP (prefer Tailscale)
            ip = node_config.get("tailscale_ip") or node_config.get("ssh_host")
            if not ip:
                logger.warning(f"[ConfigSync] No IP for source node: {source_node}")
                return False

            # Get remote path
            remote_user = node_config.get("ssh_user", "ubuntu")
            remote_path = f"{remote_user}@{ip}:ringrift/ai-service/config/distributed_hosts.yaml"
            local_path = str(self._config_path)

            # rsync with checksum verification
            cmd = f"rsync -az --checksum {remote_path} {local_path}"

            logger.debug(f"[ConfigSync] Running: {cmd}")

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._sync_timeout,
            )

            if proc.returncode != 0:
                logger.error(
                    f"[ConfigSync] rsync failed: {stderr.decode().strip()}"
                )
                return False

            # Clear cache to force reload
            clear_cluster_config_cache()

            # Verify new version loaded
            new_version = self._get_current_version()
            if new_version:
                logger.info(
                    f"[ConfigSync] Config reloaded: hash={new_version.content_hash[:8]}..."
                )

            return True

        except asyncio.TimeoutError:
            logger.error(f"[ConfigSync] Timeout pulling config from {source_node}")
            return False
        except Exception as e:
            logger.error(f"[ConfigSync] Error pulling config: {e}")
            return False

    async def _push_to_stale_nodes(self) -> None:
        """Push config to nodes that haven't pulled within timeout.

        This is a coordinator fallback for nodes that miss the event.
        """
        # TODO: Implement active push for nodes that don't pull
        # This requires tracking which nodes have synced via gossip state
        pass

    def _record_error(self, error: str) -> None:
        """Record an error in stats."""
        self._stats.errors_count += 1
        self._stats.last_error = error
        self._stats.last_error_time = time.time()

    def health_check(self) -> HealthCheckResult:
        """Return health check result."""
        is_healthy = self._running and self._stats.errors_count < 10

        details = {
            "is_coordinator": self._is_coordinator,
            "config_path": str(self._config_path),
            "changes_detected": self._changes_detected,
            "syncs_pushed": self._syncs_pushed,
            "syncs_pulled": self._syncs_pulled,
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
            "last_error": self._stats.last_error,
        }

        if self._last_known_version:
            details["last_hash"] = self._last_known_version.content_hash[:8]
            details["last_mtime"] = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(self._last_known_version.timestamp),
            )

        return HealthCheckResult(
            healthy=is_healthy,
            message=f"ConfigSync {'coordinator' if self._is_coordinator else 'worker'} "
                    f"({self._changes_detected} changes, {self._syncs_pulled} pulls)",
            details=details,
        )


# Singleton accessor
_config_sync_instance: ConfigSyncDaemon | None = None


def get_config_sync_daemon() -> ConfigSyncDaemon:
    """Get or create the ConfigSyncDaemon singleton."""
    global _config_sync_instance
    if _config_sync_instance is None:
        _config_sync_instance = ConfigSyncDaemon()
    return _config_sync_instance
