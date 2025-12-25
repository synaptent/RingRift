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
from dataclasses import dataclass, field
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

    # Scan settings
    scan_interval_seconds: float = 1800.0  # 30 minutes
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

    async def start(self) -> None:
        """Start the daemon."""
        logger.info("OrphanDetectionDaemon starting...")
        self._running = True

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
        """Stop the daemon gracefully."""
        self._running = False

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
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("SELECT COUNT(*) FROM games")
                game_count = cursor.fetchone()[0]
                conn.close()
            except Exception:
                # Not a valid game database
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

        except ImportError:
            logger.debug("ClusterManifest not available for registration")
        except Exception as e:
            logger.error(f"Failed to register orphans: {e}")

        return registered_count

    async def _emit_detection_event(self, orphans: list[OrphanInfo]) -> None:
        """Emit ORPHAN_GAMES_DETECTED event."""
        try:
            from app.coordination.event_router import emit

            total_games = sum(o.game_count for o in orphans)
            total_bytes = sum(o.file_size_bytes for o in orphans)

            await emit(
                "ORPHAN_GAMES_DETECTED",
                {
                    "orphan_count": len(orphans),
                    "total_games": total_games,
                    "total_bytes": total_bytes,
                    "orphan_paths": [str(o.path) for o in orphans],
                    "timestamp": time.time(),
                },
            )
            logger.info("Emitted ORPHAN_GAMES_DETECTED event")

        except ImportError:
            logger.debug("event_router not available, skipping event emission")
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

    async def force_scan(self) -> list[OrphanInfo]:
        """Force an immediate orphan scan."""
        return await self._run_scan()


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
