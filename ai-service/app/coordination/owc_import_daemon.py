"""OWC Import Daemon - Automatic data import from external OWC drive.

This daemon monitors the OWC external drive (mac-studio:/Volumes/RingRift-Data)
for new training data and automatically imports it into the training pipeline.

Workflow:
1. Periodically scan OWC for databases with games for underserved configs
2. Sync databases to local staging area
3. Emit NEW_GAMES_AVAILABLE events to trigger DataConsolidationDaemon
4. Emit DATA_SYNC_COMPLETED to trigger downstream pipeline

This integrates with the existing daemon infrastructure:
- DataConsolidationDaemon: Handles the actual merge into canonical databases
- DataPipelineOrchestrator: Handles NPZ export and training triggers
- AutoSyncDaemon: Handles cluster-wide data distribution

Environment Variables:
    OWC_HOST: OWC host (default: mac-studio)
    OWC_USER: SSH user for OWC host
    OWC_BASE_PATH: OWC mount path (default: /Volumes/RingRift-Data)
    OWC_SSH_KEY: Path to SSH key (default: ~/.ssh/id_ed25519)
    RINGRIFT_OWC_IMPORT_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_OWC_IMPORT_INTERVAL: Check interval in seconds (default: 3600)
    RINGRIFT_OWC_IMPORT_MIN_GAMES: Minimum games to trigger import (default: 50)
    RINGRIFT_OWC_UNDERSERVED_THRESHOLD: Local game count below which to import (default: 100000)
        Jan 2026: Increased from 500 to enable comprehensive import from OWC drive

December 2025: Created as part of the training data pipeline infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.event_utils import parse_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.mixins.import_mixin import ImportDaemonMixin
from app.coordination.protocols import CoordinatorStatus
from app.core.ssh import SSHClient, SSHConfig, SSHResult
from app.config.coordination_defaults import build_ssh_options


def _is_running_on_owc_host(owc_host: str) -> bool:
    """Check if we're running on the OWC host itself.

    Dec 30, 2025: Added to enable local file access when running on mac-studio,
    avoiding unnecessary SSH overhead and auth issues.
    """
    hostname = socket.gethostname().lower()
    owc_host_lower = owc_host.lower()

    # Check various hostname patterns
    local_patterns = [
        owc_host_lower,
        f"{owc_host_lower}.local",
        owc_host_lower.replace("-", ""),  # mac-studio -> macstudio
        owc_host_lower.replace("-", "").replace(".", ""),
    ]

    hostname_normalized = hostname.replace("-", "").replace(".", "").replace("_", "")

    for pattern in local_patterns:
        pattern_normalized = pattern.replace("-", "").replace(".", "").replace("_", "")
        if hostname_normalized.startswith(pattern_normalized):
            return True

    # Also check if owc_host is localhost
    if owc_host_lower in ("localhost", "127.0.0.1", "::1"):
        return True

    return False

logger = logging.getLogger(__name__)

__all__ = [
    "OWCImportDaemon",
    "OWCImportConfig",
    "get_owc_import_daemon",
    "reset_owc_import_daemon",
]


# ============================================================================
# Configuration
# ============================================================================

OWC_HOST = os.getenv("OWC_HOST", "mac-studio")
OWC_USER = os.getenv("OWC_USER", "armand")
OWC_BASE_PATH = os.getenv("OWC_BASE_PATH", "/Volumes/RingRift-Data")
OWC_SSH_KEY = os.getenv("OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

# Critical configs that need more data
# January 2026: Increased from 500 to 100,000 to enable comprehensive data import
# from OWC. With 8.5M games on OWC and only 84K locally, 500 was far too low.
# Set via RINGRIFT_OWC_UNDERSERVED_THRESHOLD env var if needed.
UNDERSERVED_THRESHOLD = int(os.getenv("RINGRIFT_OWC_UNDERSERVED_THRESHOLD", "100000"))

# Known good OWC database paths
OWC_SOURCE_DATABASES = [
    "selfplay_repository/consolidated_archives/synced_20251213_182629_vast_2x5090_selfplay.db",
    "selfplay_repository/consolidated_archives/synced_20251213_172954_vast-5090-quad_selfplay.db",
    "selfplay_repository/consolidated_archives/synced_20251213_182629_lambda-h100_selfplay.db",
    "training_data/coordinator_backup/sq19_4p_selfplay.db",
    "training_data/coordinator_backup/sq19_2p_selfplay.db",
]


@dataclass
class OWCImportConfig:
    """Configuration for OWC Import daemon.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.
    """

    # Check interval (passed to HandlerBase as cycle_interval)
    check_interval_seconds: int = 3600

    # Daemon control
    enabled: bool = True

    # Minimum games on OWC to trigger import
    min_games_for_import: int = 50

    # Local staging directory
    staging_dir: Path = field(default_factory=lambda: Path("data/games/owc_imports"))

    # OWC connection
    owc_host: str = OWC_HOST
    owc_user: str = OWC_USER
    owc_base_path: str = OWC_BASE_PATH
    owc_ssh_key: str = OWC_SSH_KEY

    # Timeout for OWC operations
    ssh_timeout: int = 60
    rsync_timeout: int = 600

    @classmethod
    def from_env(cls) -> "OWCImportConfig":
        """Load configuration from environment."""
        return cls(
            enabled=os.getenv("RINGRIFT_OWC_IMPORT_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.getenv("RINGRIFT_OWC_IMPORT_INTERVAL", "3600")),
            min_games_for_import=int(os.getenv("RINGRIFT_OWC_IMPORT_MIN_GAMES", "50")),
        )


@dataclass
class OWCDatabaseInfo:
    """Information about a database on OWC."""
    path: str
    configs: dict[str, int] = field(default_factory=dict)
    synced: bool = False


@dataclass
class ImportStats:
    """Statistics for an import cycle."""
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    databases_scanned: int = 0
    databases_synced: int = 0
    games_imported: int = 0
    configs_updated: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.cycle_end - self.cycle_start


# ============================================================================
# OWC Import Daemon
# ============================================================================


class OWCImportDaemon(HandlerBase, ImportDaemonMixin):
    """Daemon that imports training data from OWC external drive.

    This daemon runs on the coordinator and periodically checks the OWC drive
    for databases containing games for underserved configurations.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking

    January 2026: Inherits from ImportDaemonMixin for file validation
    functionality.
    """

    # ImportDaemonMixin configuration
    IMPORT_LOG_PREFIX = "[OWCImport]"
    IMPORT_VERIFY_CHECKSUMS = True

    def __init__(self, config: OWCImportConfig | None = None):
        self._daemon_config = config or OWCImportConfig.from_env()

        super().__init__(
            name="OWCImportDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._last_import: dict[str, float] = {}  # config_key -> last import time
        self._import_history: list[ImportStats] = []
        self._total_games_imported = 0
        self._owc_available = True

        # Dec 30, 2025: Detect if running on OWC host for local file access
        self._is_local = _is_running_on_owc_host(self.config.owc_host)
        if self._is_local:
            logger.info(
                f"[OWCImport] Running on OWC host '{self.config.owc_host}', "
                f"using local file access"
            )

        # Dec 29, 2025: Use canonical SSHClient for remote SSH operations
        self._ssh_client: SSHClient | None = None
        if not self._is_local:
            self._ssh_client = SSHClient(SSHConfig(
                host=self.config.owc_host,
                user=self.config.owc_user,
                key_path=self.config.owc_ssh_key,
                connect_timeout=10,
                command_timeout=self.config.ssh_timeout,
            ))

    @property
    def config(self) -> OWCImportConfig:
        """Get daemon configuration."""
        return self._daemon_config

    # =========================================================================
    # OWC Operations (Dec 29, 2025: Uses canonical SSHClient from app/core/ssh)
    # =========================================================================

    async def _run_command(self, command: str) -> tuple[bool, str]:
        """Run command on OWC host (locally or via SSH).

        Dec 30, 2025: Added local execution support when running on OWC host.
        Uses direct subprocess for local execution, SSH for remote.
        """
        if self._is_local:
            # Run locally using subprocess
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.ssh_timeout,
                )
                if result.returncode == 0:
                    return True, result.stdout.strip()
                else:
                    return False, result.stderr.strip() or "Command failed"
            except subprocess.TimeoutExpired:
                return False, "Command timed out"
            except Exception as e:
                return False, str(e)
        else:
            # Run remotely via SSH
            if self._ssh_client is None:
                return False, "SSH client not initialized"
            result = await self._ssh_client.run_async(
                command, timeout=self.config.ssh_timeout
            )
            if result.success:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip() or result.error or "Unknown error"

    # Backward compat alias
    async def _run_ssh_command(self, command: str) -> tuple[bool, str]:
        """Alias for _run_command (backward compatibility)."""
        return await self._run_command(command)

    async def _check_owc_available(self) -> bool:
        """Check if OWC drive is accessible."""
        if self._is_local:
            # Check locally using Path
            owc_path = Path(self.config.owc_base_path)
            return owc_path.exists() and owc_path.is_dir()

        # Check via SSH
        # Jan 2026: Use shlex.quote() for proper path escaping
        success, output = await self._run_command(
            f"ls -d {shlex.quote(self.config.owc_base_path)} 2>/dev/null"
        )
        return success

    async def _discover_owc_databases(self) -> list[str]:
        """Dynamically discover databases on OWC drive.

        Dec 30, 2025: Added to replace hardcoded OWC_SOURCE_DATABASES list.
        Searches for .db files in common locations on the OWC drive.
        Falls back to static list if discovery fails.
        Jan 2026: Use shlex.quote() for proper path escaping.
        """
        quoted_path = shlex.quote(self.config.owc_base_path)
        find_cmd = (
            f"find {quoted_path} "
            f"\\( -path '*/selfplay_repository/*' -o -path '*/training_data/*' -o -path '*/data/games/*' \\) "
            f"-name '*.db' -type f 2>/dev/null"
        )
        success, output = await self._run_ssh_command(find_cmd)

        if not success or not output.strip():
            logger.debug("[OWCImport] Dynamic discovery failed, using static list")
            return OWC_SOURCE_DATABASES

        databases = []
        base_path = self.config.owc_base_path
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.endswith(".db"):
                # Convert absolute path to relative path
                if line.startswith(base_path):
                    rel_path = line[len(base_path):].lstrip("/")
                    databases.append(rel_path)
                else:
                    databases.append(line)

        if not databases:
            logger.debug("[OWCImport] No databases found, using static list")
            return OWC_SOURCE_DATABASES

        logger.info(f"[OWCImport] Discovered {len(databases)} databases on OWC")
        return databases

    async def _scan_owc_database(self, rel_path: str) -> OWCDatabaseInfo | None:
        """Scan a database on OWC for game counts by config."""
        full_path = f"{self.config.owc_base_path}/{rel_path}"

        query = """
            SELECT board_type || '_' || num_players || 'p' as config, COUNT(*)
            FROM games
            WHERE winner IS NOT NULL
            GROUP BY board_type, num_players
        """

        success, output = await self._run_ssh_command(
            f"sqlite3 '{full_path}' \"{query}\""
        )

        if not success:
            return None

        info = OWCDatabaseInfo(path=rel_path)

        for line in output.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 2:
                    try:
                        info.configs[parts[0]] = int(parts[1])
                    except ValueError:
                        pass

        return info if info.configs else None

    async def _sync_database(self, rel_path: str) -> Path | None:
        """Sync a database from OWC to local staging.

        Dec 30, 2025: Added local mode support - uses shutil.copy when
        running on OWC host, rsync over SSH when remote.
        """
        import shutil

        self.config.staging_dir.mkdir(parents=True, exist_ok=True)

        local_name = rel_path.replace("/", "_")
        local_path = self.config.staging_dir / local_name
        source_path = Path(self.config.owc_base_path) / rel_path

        if self._is_local:
            # Local mode: direct file copy
            try:
                if not source_path.exists():
                    logger.warning(f"[OWCImport] Source not found: {source_path}")
                    return None

                await asyncio.to_thread(shutil.copy2, source_path, local_path)

                # Validate synced database using ImportDaemonMixin
                validation = await self._validate_import(local_path, expected_type="db")
                if not validation.valid:
                    logger.warning(
                        f"[OWCImport] Synced file failed validation: {validation.error}"
                    )
                    local_path.unlink(missing_ok=True)
                    return None

                logger.info(
                    f"[OWCImport] Copied {rel_path} "
                    f"({local_path.stat().st_size / 1024 / 1024:.1f} MB, validated)"
                )
                return local_path
            except Exception as e:
                logger.warning(f"[OWCImport] Copy error for {rel_path}: {e}")
                return None

        # Remote mode: rsync over SSH
        remote_path = f"{self.config.owc_user}@{self.config.owc_host}:{self.config.owc_base_path}/{rel_path}"

        # Dec 30, 2025: Use centralized SSH config for consistent timeouts
        ssh_opts = build_ssh_options(
            key_path=self.config.owc_ssh_key,
            include_keepalive=False,  # rsync has its own timeout
        )
        rsync_cmd = [
            "rsync", "-avz",
            "-e", ssh_opts,
            remote_path,
            str(local_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.rsync_timeout,
            )

            if proc.returncode == 0 and local_path.exists():
                # Validate synced database using ImportDaemonMixin
                validation = await self._validate_import(local_path, expected_type="db")
                if not validation.valid:
                    logger.warning(
                        f"[OWCImport] Synced file failed validation: {validation.error}"
                    )
                    local_path.unlink(missing_ok=True)
                    return None

                logger.info(
                    f"[OWCImport] Synced {rel_path} "
                    f"({local_path.stat().st_size / 1024 / 1024:.1f} MB, validated)"
                )
                return local_path
            else:
                return None

        except asyncio.TimeoutError:
            logger.warning(f"[OWCImport] Rsync timed out for {rel_path}")
            return None
        except Exception as e:
            logger.warning(f"[OWCImport] Rsync error for {rel_path}: {e}")
            return None

    # =========================================================================
    # Local Operations
    # =========================================================================

    def _get_local_game_count(self, config_key: str) -> int:
        """Get current game count in local canonical database."""
        parsed = parse_config_key(config_key)
        if not parsed:
            return 0

        board_type = parsed.board_type
        num_players = parsed.num_players

        canonical_path = Path("data/games") / f"canonical_{board_type}_{num_players}p.db"

        if not canonical_path.exists():
            return 0

        try:
            import sqlite3
            with sqlite3.connect(str(canonical_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM games WHERE winner IS NOT NULL")
                return cursor.fetchone()[0]
        except (sqlite3.Error, OSError, TypeError) as e:
            # Dec 29, 2025: Narrowed from bare Exception
            # sqlite3.Error: database errors (locked, corrupted, missing table)
            # OSError: file system errors (permissions, disk full)
            # TypeError: fetchone returns None (schema mismatch)
            logger.debug(f"[OWCImport] Error reading game count: {e}")
            return 0

    def _get_underserved_configs(self) -> list[str]:
        """Get list of configs that need more data."""
        from app.coordination.progress_watchdog_daemon import CANONICAL_CONFIGS

        underserved = []
        for config in CANONICAL_CONFIGS:
            local_count = self._get_local_game_count(config)
            if local_count < UNDERSERVED_THRESHOLD:
                underserved.append(config)

        return underserved

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _emit_new_games_available(self, config_key: str, games_added: int, source: str) -> None:
        """Emit NEW_GAMES_AVAILABLE event to trigger consolidation."""
        safe_emit_event(
            "new_games_available",
            {
                "config_key": config_key,
                "new_games": games_added,
                "source": source,
                "trigger": "owc_import",
            },
            context="OWCImport",
        )
        logger.debug(f"[OWCImport] Emitted NEW_GAMES_AVAILABLE for {config_key}")

    def _emit_data_sync_completed(self, configs_updated: list[str], games_imported: int) -> None:
        """Emit DATA_SYNC_COMPLETED event to trigger pipeline."""
        safe_emit_event(
            "data_sync_completed",
            {
                "sync_type": "owc_import",
                "configs_updated": configs_updated,
                "games_imported": games_imported,
                "source": "OWCImportDaemon",
            },
            context="OWCImport",
        )

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main import cycle."""
        # February 2026: Block when coordinator is low on RAM/disk
        from app.utils.resource_guard import coordinator_resource_gate
        if not coordinator_resource_gate("OWC_IMPORT"):
            return

        stats = ImportStats(cycle_start=time.time())

        # Check OWC availability
        if not await self._check_owc_available():
            if self._owc_available:  # Log only on state change
                if self._is_local:
                    logger.warning(f"[OWCImport] OWC drive not available at {self.config.owc_base_path}")
                else:
                    logger.warning(f"[OWCImport] OWC drive not available at {self.config.owc_host}:{self.config.owc_base_path}")
            self._owc_available = False
            return

        if not self._owc_available:
            logger.info(f"[OWCImport] OWC drive is now available")
        self._owc_available = True

        # Get underserved configs
        # December 30, 2025: Wrap blocking SQLite calls with asyncio.to_thread
        underserved = await asyncio.to_thread(self._get_underserved_configs)
        if not underserved:
            logger.debug("[OWCImport] All configs have sufficient data")
            return

        logger.info(f"[OWCImport] Checking OWC for underserved configs: {underserved}")

        # Scan OWC databases - Dec 30, 2025: Use dynamic discovery
        databases_to_sync: list[OWCDatabaseInfo] = []
        owc_databases = await self._discover_owc_databases()

        for rel_path in owc_databases:
            info = await self._scan_owc_database(rel_path)
            if info:
                stats.databases_scanned += 1

                # Check if database has games for underserved configs
                has_needed_games = any(
                    config in underserved and count >= self.config.min_games_for_import
                    for config, count in info.configs.items()
                )

                if has_needed_games:
                    databases_to_sync.append(info)

        # Sync relevant databases
        synced_paths: list[Path] = []
        for db_info in databases_to_sync:
            local_path = await self._sync_database(db_info.path)
            if local_path:
                synced_paths.append(local_path)
                db_info.synced = True
                stats.databases_synced += 1

        if not synced_paths:
            logger.debug("[OWCImport] No databases needed syncing")
            return

        # Emit events for each underserved config that has new data
        for db_info in databases_to_sync:
            if not db_info.synced:
                continue

            for config_key, count in db_info.configs.items():
                if config_key in underserved and count >= self.config.min_games_for_import:
                    self._emit_new_games_available(config_key, count, f"owc:{db_info.path}")
                    stats.games_imported += count

                    if config_key not in stats.configs_updated:
                        stats.configs_updated.append(config_key)

        # Emit completion event
        if stats.configs_updated:
            self._emit_data_sync_completed(stats.configs_updated, stats.games_imported)

        stats.cycle_end = time.time()
        self._import_history.append(stats)
        self._total_games_imported += stats.games_imported

        # Trim history
        if len(self._import_history) > 50:
            self._import_history = self._import_history[-50:]

        logger.info(
            f"[OWCImport] Cycle complete: synced {stats.databases_synced} DBs, "
            f"~{stats.games_imported} games for {stats.configs_updated}"
        )

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="OWCImport not running",
            )

        if not self._owc_available:
            return HealthCheckResult(
                healthy=True,  # Still healthy, just OWC unavailable
                status=CoordinatorStatus.RUNNING,
                message="OWC drive not available",
                details={
                    "owc_host": self.config.owc_host,
                    "is_local": self._is_local,
                },
            )

        mode = "local" if self._is_local else "remote"
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"OWC import active ({mode}), {self._total_games_imported} games imported",
            details={
                "cycles_completed": self._stats.cycles_completed,
                "total_games_imported": self._total_games_imported,
                "owc_available": self._owc_available,
                "is_local": self._is_local,
                "errors_count": self._stats.errors_count,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Return detailed status."""
        return {
            "running": self._running,
            "config": {
                "enabled": self.config.enabled,
                "check_interval_seconds": self.config.check_interval_seconds,
                "min_games_for_import": self.config.min_games_for_import,
            },
            "stats": {
                "cycles_completed": self._stats.cycles_completed,
                "last_activity": self._stats.last_activity,
                "started_at": self._stats.started_at,
                "errors_count": self._stats.errors_count,
            },
            "owc_host": self.config.owc_host,
            "owc_available": self._owc_available,
            "is_local": self._is_local,
            "total_games_imported": self._total_games_imported,
            "recent_imports": [
                {
                    "configs": s.configs_updated,
                    "games": s.games_imported,
                    "duration_seconds": s.duration_seconds,
                }
                for s in self._import_history[-5:]
            ],
        }


# ============================================================================
# Singleton Accessors (using HandlerBase class methods)
# ============================================================================


def get_owc_import_daemon() -> OWCImportDaemon:
    """Get or create the singleton OWCImportDaemon instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return OWCImportDaemon.get_instance()


def reset_owc_import_daemon() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    OWCImportDaemon.reset_instance()
