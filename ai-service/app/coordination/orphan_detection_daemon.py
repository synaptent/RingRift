"""Orphan Detection Daemon - Detect and handle orphaned game data.

This daemon periodically scans for game databases that exist on disk but are
not registered in the ClusterManifest. These "orphaned" games can occur when:
- Sync events fail to fire
- Nodes terminate before sync completes
- Manual database copies

Architecture:
    1. Periodically scans all .db files in data/games/
    2. Compares against ClusterManifest registry
    3. Auto-registers orphaned games if valid
    4. Emits ORPHAN_GAMES_DETECTED event for monitoring
    5. Optionally triggers cleanup of old orphans

Usage:
    # As standalone daemon
    python -m app.coordination.orphan_detection_daemon

    # Via DaemonManager
    manager.register_factory(DaemonType.ORPHAN_DETECTION, daemon.run)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Add parent to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class OrphanDetectionConfig:
    """Configuration for orphan detection daemon."""

    # Scan settings (Phase 3B.3: Dec 2025 - reduced from 30 min to 5 min)
    scan_interval_seconds: float = 300.0  # 5 minutes
    games_dir: str = "data/games"

    # Auto-registration
    auto_register_orphans: bool = True
    min_games_to_register: int = 1  # Minimum games in DB to auto-register

    # Cleanup settings
    cleanup_enabled: bool = False  # Don't delete by default
    min_age_before_cleanup_hours: float = 24.0
    max_orphan_count_before_alert: int = 100

    # Event settings
    emit_detection_event: bool = True


@dataclass
class OrphanInfo:
    """Information about an orphaned database."""
    path: Path
    game_count: int
    file_size_bytes: int
    modified_time: float
    board_type: str | None = None
    num_players: int | None = None


class OrphanDetectionDaemon:
    """Daemon that detects and handles orphaned game databases.

    Scans for .db files not registered in ClusterManifest and either:
    - Auto-registers them (if valid game databases)
    - Reports them for manual review
    - Optionally cleans up old orphans

    This solves the gap where games generated on nodes may not be tracked
    in the central manifest, leading to "invisible" training data.
    """

    def __init__(self, config: OrphanDetectionConfig | None = None):
        self.config = config or OrphanDetectionConfig()
        self._running = False
        self._last_scan_time: float = 0.0
        self._orphan_history: list[dict[str, Any]] = []
        self._event_subscription = None

    async def start(self) -> None:
        """Start the daemon."""
        logger.info("OrphanDetectionDaemon starting...")
        self._running = True

        # Phase 4A.3: Subscribe to DATABASE_CREATED events for immediate registration
        await self._subscribe_to_database_events()

        while self._running:
            try:
                # Run scan if interval has passed
                if time.time() - self._last_scan_time > self.config.scan_interval_seconds:
                    await self._run_scan()
                    self._last_scan_time = time.time()

                await asyncio.sleep(60.0)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orphan detection loop: {e}")
                await asyncio.sleep(60.0)

        logger.info("OrphanDetectionDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully.

        December 2025: Added proper cleanup to unsubscribe from events.
        """
        self._running = False

        # Dec 2025: Unsubscribe from events on shutdown
        if self._event_subscription is not None:
            try:
                from app.coordination.event_router import unsubscribe, DataEventType
                unsubscribe(DataEventType.DATABASE_CREATED, self._event_subscription)
                logger.debug("[OrphanDetection] Unsubscribed from DATABASE_CREATED events")
            except Exception as e:
                logger.debug(f"[OrphanDetection] Failed to unsubscribe: {e}")
            self._event_subscription = None

    async def _subscribe_to_database_events(self) -> None:
        """Subscribe to DATABASE_CREATED events for immediate registration.

        Phase 4A.3 (December 2025): Enables immediate database visibility
        instead of waiting for the 5-minute periodic scan.
        """
        try:
            from app.coordination.event_router import subscribe, DataEventType

            def on_database_created(event):
                """Handle DATABASE_CREATED event - register immediately."""
                try:
                    payload = event.payload if hasattr(event, "payload") else event
                    db_path = payload.get("db_path")
                    node_id = payload.get("node_id")
                    config_key = payload.get("config_key")
                    board_type = payload.get("board_type")
                    num_players = payload.get("num_players")
                    engine_mode = payload.get("engine_mode")

                    if db_path and node_id:
                        # Register in ClusterManifest
                        asyncio.create_task(
                            self._register_database_from_event(
                                db_path, node_id, config_key, board_type, num_players, engine_mode
                            )
                        )
                        logger.info(f"[OrphanDetection] Immediate registration: {db_path}")
                except Exception as e:
                    logger.debug(f"[OrphanDetection] Failed to handle DATABASE_CREATED: {e}")

            subscribe(DataEventType.DATABASE_CREATED, on_database_created)
            # Dec 2025: Store callback reference for cleanup in stop()
            self._event_subscription = on_database_created
            logger.info("[OrphanDetection] Subscribed to DATABASE_CREATED events")

        except ImportError:
            logger.debug("[OrphanDetection] Event system not available")
        except Exception as e:
            logger.warning(f"[OrphanDetection] Failed to subscribe to events: {e}")

    async def _register_database_from_event(
        self,
        db_path: str,
        node_id: str,
        config_key: str | None,
        board_type: str | None,
        num_players: int | None,
        engine_mode: str | None,
    ) -> None:
        """Register database from event in ClusterManifest."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            manifest.register_database(
                db_path=db_path,
                node_id=node_id,
                board_type=board_type,
                num_players=num_players,
                config_key=config_key,
                engine_mode=engine_mode,
            )
            logger.debug(f"[OrphanDetection] Registered database: {db_path} on {node_id}")

        except Exception as e:
            logger.debug(f"[OrphanDetection] Could not register database: {e}")

    async def _run_scan(self) -> list[OrphanInfo]:
        """Scan for orphaned databases."""
        logger.info("Starting orphan detection scan...")

        games_dir = ROOT / self.config.games_dir
        if not games_dir.exists():
            logger.debug(f"Games directory not found: {games_dir}")
            return []

        # Get all .db files
        db_files = list(games_dir.glob("*.db"))
        if not db_files:
            logger.debug("No database files found")
            return []

        # Get registered databases from manifest
        registered_dbs = await self._get_registered_databases()

        # Find orphans
        orphans: list[OrphanInfo] = []
        for db_path in db_files:
            # Skip if registered
            if str(db_path) in registered_dbs or db_path.name in registered_dbs:
                continue

            # Analyze the database
            orphan_info = await self._analyze_database(db_path)
            if orphan_info and orphan_info.game_count >= self.config.min_games_to_register:
                orphans.append(orphan_info)

        if orphans:
            logger.info(f"Found {len(orphans)} orphaned databases")

            # Auto-register if enabled
            if self.config.auto_register_orphans:
                await self._register_orphans(orphans)

            # Emit event
            if self.config.emit_detection_event:
                await self._emit_detection_event(orphans)

            # Check if we need to alert
            if len(orphans) >= self.config.max_orphan_count_before_alert:
                logger.warning(
                    f"High orphan count detected: {len(orphans)} databases. "
                    "Consider investigating sync issues."
                )

        else:
            logger.info("No orphaned databases found")

        return orphans

    async def _get_registered_databases(self) -> set[str]:
        """Get set of registered database paths from ClusterManifest."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            registered = set()

            # Get all registered game locations
            if hasattr(manifest, "get_all_game_locations"):
                locations = manifest.get_all_game_locations()
                for loc in locations:
                    registered.add(loc.get("path", ""))
                    registered.add(Path(loc.get("path", "")).name)

            return registered

        except ImportError:
            logger.debug("ClusterManifest not available")
            return set()
        except Exception as e:
            logger.error(f"Failed to get registered databases: {e}")
            return set()

    async def _analyze_database(self, db_path: Path) -> OrphanInfo | None:
        """Analyze a database file to get orphan info."""
        try:
            # Get file stats
            stat = db_path.stat()

            # Try to get game count
            game_count = 0
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM games")
                    game_count = cursor.fetchone()[0]
            except (sqlite3.Error, TypeError):
                # Not a valid game database (sqlite3.Error for DB issues, TypeError if fetchone() is None)
                return None

            # Parse board type and num_players from filename
            # Expected format: canonical_hex8_2p.db or hex8_2p.db
            board_type = None
            num_players = None
            name = db_path.stem.replace("canonical_", "")

            for bt in ["hex8", "hexagonal", "square8", "square19"]:
                if bt in name:
                    board_type = bt
                    break

            for np in ["2p", "3p", "4p"]:
                if np in name:
                    num_players = int(np[0])
                    break

            return OrphanInfo(
                path=db_path,
                game_count=game_count,
                file_size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
                board_type=board_type,
                num_players=num_players,
            )

        except Exception as e:
            logger.error(f"Failed to analyze database {db_path}: {e}")
            return None

    async def _register_orphans(self, orphans: list[OrphanInfo]) -> int:
        """Register orphaned databases in ClusterManifest."""
        registered_count = 0

        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()

            for orphan in orphans:
                try:
                    # Get local node ID
                    node_id = os.environ.get("RINGRIFT_NODE_ID", "local")

                    # Register the database
                    if hasattr(manifest, "register_database"):
                        manifest.register_database(
                            node_id=node_id,
                            db_path=str(orphan.path),
                            board_type=orphan.board_type,
                            num_players=orphan.num_players,
                            game_count=orphan.game_count,
                        )
                        registered_count += 1
                        logger.info(
                            f"Registered orphan: {orphan.path.name} "
                            f"({orphan.game_count} games)"
                        )

                except Exception as e:
                    logger.error(f"Failed to register orphan {orphan.path}: {e}")

            # Emit registration event if any orphans were registered
            if registered_count > 0:
                await self._emit_registration_event(orphans[:registered_count])

        except ImportError:
            logger.debug("ClusterManifest not available for registration")
        except Exception as e:
            logger.error(f"Failed to register orphans: {e}")

        return registered_count

    async def _emit_registration_event(self, registered: list[OrphanInfo]) -> None:
        """Emit ORPHAN_GAMES_REGISTERED event."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                return

            total_games = sum(o.game_count for o in registered)

            await router.publish(
                DataEventType.ORPHAN_GAMES_REGISTERED,
                {
                    "registered_count": len(registered),
                    "total_games": total_games,
                    "registered_paths": [str(o.path) for o in registered],
                    "timestamp": time.time(),
                },
            )
            logger.info(f"Emitted ORPHAN_GAMES_REGISTERED: {len(registered)} databases")

        except ImportError:
            pass  # Event system not available - expected in some contexts
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.warning(f"Failed to emit orphan registration event: {e}")

    async def _emit_detection_event(self, orphans: list[OrphanInfo]) -> None:
        """Emit ORPHAN_GAMES_DETECTED event."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                logger.debug("Event router not available")
                return

            total_games = sum(o.game_count for o in orphans)
            total_bytes = sum(o.file_size_bytes for o in orphans)

            await router.publish(
                DataEventType.ORPHAN_GAMES_DETECTED,
                {
                    "orphan_count": len(orphans),
                    "total_games": total_games,
                    "total_bytes": total_bytes,
                    "orphan_paths": [str(o.path) for o in orphans],
                    "board_types": list({o.board_type for o in orphans if o.board_type}),
                    "timestamp": time.time(),
                },
            )
            logger.info(
                f"Emitted ORPHAN_GAMES_DETECTED: {len(orphans)} orphans, "
                f"{total_games} total games"
            )

        except ImportError as e:
            logger.debug(f"Event system not available: {e}")
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

    async def force_scan(self) -> list[OrphanInfo]:
        """Force an immediate orphan scan."""
        return await self._run_scan()

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status.

        December 2025: Added to satisfy CoordinatorProtocol for unified health monitoring.
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Orphan detection daemon not running",
            )

        # Check if scans are happening
        now = time.time()
        if self._last_scan_time > 0:
            hours_since_scan = (now - self._last_scan_time) / 3600
            # Warning if no scan in 2x the interval
            if hours_since_scan > (self.config.scan_interval_seconds / 3600) * 2:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Orphan scan overdue ({hours_since_scan:.1f}h since last scan)",
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Orphan detection daemon running (last scan: {self._last_scan_time:.0f}s ago)" if self._last_scan_time else "Orphan detection daemon running (no scans yet)",
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring.

        December 2025: Added for DaemonManager status reporting.

        Returns:
            Status dict with daemon state and metrics
        """
        now = time.time()
        uptime = now - self._last_scan_time if self._last_scan_time > 0 else 0.0

        return {
            "daemon": "OrphanDetectionDaemon",
            "running": self._running,
            "last_scan_time": self._last_scan_time,
            "seconds_since_scan": now - self._last_scan_time if self._last_scan_time > 0 else None,
            "orphan_history_count": len(self._orphan_history),
            "recent_orphans": self._orphan_history[-5:] if self._orphan_history else [],
            "config": {
                "scan_interval_seconds": self.config.scan_interval_seconds,
                "games_dir": self.config.games_dir,
                "auto_register_orphans": self.config.auto_register_orphans,
                "emit_detection_event": self.config.emit_detection_event,
            },
        }


async def run() -> None:
    """Run the daemon (entry point for DaemonManager)."""
    daemon = OrphanDetectionDaemon()
    await daemon.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run())
