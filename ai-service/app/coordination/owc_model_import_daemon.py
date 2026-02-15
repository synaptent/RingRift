"""OWCModelImportDaemon - Import trained models from OWC external drive.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

This daemon differs from OWCImportDaemon (which imports game databases):
- OWCImportDaemon: Imports .db game databases for training data
- OWCModelImportDaemon: Imports .pth model files for Elo evaluation

The OWC drive has 1000s of trained models from historical training runs that have
never been evaluated for Elo ratings. This daemon:
1. Discovers model files on OWC drive
2. Cross-references with EloService to skip models with existing ratings
3. Imports unevaluated models to local storage
4. Emits MODEL_IMPORTED events for downstream evaluation

Environment Variables:
    OWC_HOST: OWC host (default: mac-studio)
    OWC_USER: SSH user for OWC host
    OWC_BASE_PATH: OWC mount path (default: /Volumes/RingRift-Data)
    OWC_SSH_KEY: Path to SSH key (default: ~/.ssh/id_ed25519)
    RINGRIFT_OWC_MODEL_IMPORT_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_OWC_MODEL_IMPORT_INTERVAL: Check interval in seconds (default: 7200)
    RINGRIFT_OWC_MODEL_IMPORT_MAX_PER_CYCLE: Max models to import per cycle (default: 10)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.event_router import DataEventType, safe_emit_event
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.mixins.import_mixin import ImportDaemonMixin
from app.core.ssh import SSHClient, SSHConfig

logger = logging.getLogger(__name__)

__all__ = [
    "OWCModelImportConfig",
    "OWCModelImportDaemon",
    "OWCModelInfo",
    "get_owc_model_import_daemon",
    "reset_owc_model_import_daemon",
]


# ============================================================================
# Configuration
# ============================================================================

OWC_HOST = os.getenv("OWC_HOST", "mac-studio")
OWC_USER = os.getenv("OWC_USER", "armand")
OWC_BASE_PATH = os.getenv("OWC_BASE_PATH", "/Volumes/RingRift-Data")
OWC_SSH_KEY = os.getenv("OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

# Model directories to scan on OWC drive
OWC_MODEL_PATHS = [
    "models/archived",
    "models/training_runs",
    "models/checkpoints",
    "selfplay_repository/models",
    "training_data/models",
]

# Minimum model file size (1MB) - smaller files are likely corrupt
MIN_MODEL_SIZE_BYTES = 1_000_000


def _is_running_on_owc_host(owc_host: str) -> bool:
    """Check if we're running on the OWC host itself."""
    hostname = socket.gethostname().lower()
    owc_host_lower = owc_host.lower()

    local_patterns = [
        owc_host_lower,
        f"{owc_host_lower}.local",
        owc_host_lower.replace("-", ""),
    ]

    hostname_normalized = hostname.replace("-", "").replace(".", "").replace("_", "")

    for pattern in local_patterns:
        pattern_normalized = pattern.replace("-", "").replace(".", "").replace("_", "")
        if hostname_normalized.startswith(pattern_normalized):
            return True

    if owc_host_lower in ("localhost", "127.0.0.1", "::1"):
        return True

    return False


@dataclass
class OWCModelImportConfig:
    """Configuration for OWC Model Import daemon."""

    # Check interval (2 hours default)
    check_interval_seconds: int = 7200

    # Daemon control
    enabled: bool = True

    # Maximum models to import per cycle
    max_models_per_cycle: int = 10

    # Minimum model file size
    min_model_size_bytes: int = MIN_MODEL_SIZE_BYTES

    # Local import directory
    import_dir: Path = field(default_factory=lambda: Path("models/owc_imports"))

    # OWC connection
    owc_host: str = OWC_HOST
    owc_user: str = OWC_USER
    owc_base_path: str = OWC_BASE_PATH
    owc_ssh_key: str = OWC_SSH_KEY

    # Timeout for OWC operations
    ssh_timeout: int = 60
    rsync_timeout: int = 600

    @classmethod
    def from_env(cls) -> "OWCModelImportConfig":
        """Load configuration from environment."""
        return cls(
            enabled=os.getenv("RINGRIFT_OWC_MODEL_IMPORT_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.getenv("RINGRIFT_OWC_MODEL_IMPORT_INTERVAL", "7200")),
            max_models_per_cycle=int(os.getenv("RINGRIFT_OWC_MODEL_IMPORT_MAX_PER_CYCLE", "10")),
        )


@dataclass
class OWCModelInfo:
    """Model discovered on OWC drive."""

    path: str                      # Relative path on OWC
    file_name: str                 # Just the filename
    board_type: str | None         # Extracted from filename
    num_players: int | None        # Extracted from filename
    architecture_version: str | None  # Extracted from filename
    file_size: int                 # File size in bytes
    has_elo: bool = False          # Cross-referenced with EloService

    @property
    def config_key(self) -> str | None:
        """Get config key if board_type and num_players are known."""
        if self.board_type and self.num_players:
            return f"{self.board_type}_{self.num_players}p"
        return None


@dataclass
class ModelImportStats:
    """Statistics for model import operations."""

    cycle_count: int = 0
    models_discovered: int = 0
    models_imported: int = 0
    models_skipped_has_elo: int = 0
    models_skipped_invalid: int = 0
    import_errors: int = 0
    last_cycle_time: float = 0.0
    last_cycle_duration: float = 0.0


# ============================================================================
# Model Info Extraction
# ============================================================================

# Pattern to extract board_type, num_players, and version from model filenames
# Examples: canonical_hex8_2p.pth, hex8_4p_v5heavy.pth, square8_2p_20251225.pth
MODEL_NAME_PATTERN = re.compile(
    r"(?:canonical_)?(?:ringrift_)?(?:best_)?"
    r"(hex8|hexagonal|square8|square19)"
    r"[_-]?"
    r"([234])p"
    r"(?:[_-](v\d+\w*))?"
    r"(?:[_-](\d{8}))?"
    r"\.pth$",
    re.IGNORECASE,
)


def extract_model_info(file_path: str, file_size: int) -> OWCModelInfo | None:
    """Extract model information from file path.

    Args:
        file_path: Path to model file (relative or absolute)
        file_size: Size of the file in bytes

    Returns:
        OWCModelInfo if parsing successful, None otherwise
    """
    file_name = Path(file_path).name

    match = MODEL_NAME_PATTERN.search(file_name)
    if match:
        board_type = match.group(1).lower()
        num_players = int(match.group(2))
        version = match.group(3)

        return OWCModelInfo(
            path=file_path,
            file_name=file_name,
            board_type=board_type,
            num_players=num_players,
            architecture_version=version,
            file_size=file_size,
        )

    # Fallback: try to extract at least partial info
    file_name_lower = file_name.lower()

    board_type = None
    for bt in ["hexagonal", "hex8", "square19", "square8"]:
        if bt in file_name_lower:
            board_type = bt
            break

    num_players = None
    for np in [4, 3, 2]:
        if f"{np}p" in file_name_lower:
            num_players = np
            break

    return OWCModelInfo(
        path=file_path,
        file_name=file_name,
        board_type=board_type,
        num_players=num_players,
        architecture_version=None,
        file_size=file_size,
    )


# ============================================================================
# OWC Model Import Daemon
# ============================================================================


class OWCModelImportDaemon(HandlerBase, ImportDaemonMixin):
    """Daemon that imports trained models from OWC external drive.

    This daemon runs on the coordinator and periodically scans the OWC drive
    for model files that haven't been evaluated for Elo ratings.
    """

    IMPORT_LOG_PREFIX = "[OWCModelImport]"
    IMPORT_VERIFY_CHECKSUMS = True

    def __init__(self, config: OWCModelImportConfig | None = None):
        self._daemon_config = config or OWCModelImportConfig.from_env()

        super().__init__(
            name="OWCModelImportDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._stats = ModelImportStats()
        self._owc_available = True
        self._imported_models: set[str] = set()  # Track what we've imported

        # Detect if running on OWC host
        self._is_local = _is_running_on_owc_host(self._daemon_config.owc_host)
        if self._is_local:
            logger.info(
                f"[OWCModelImport] Running on OWC host '{self._daemon_config.owc_host}', "
                f"using local file access"
            )

        # SSH client for remote operations
        self._ssh_client: SSHClient | None = None
        if not self._is_local:
            self._ssh_client = SSHClient(SSHConfig(
                host=self._daemon_config.owc_host,
                user=self._daemon_config.owc_user,
                key_path=self._daemon_config.owc_ssh_key,
                connect_timeout=10,
                command_timeout=self._daemon_config.ssh_timeout,
            ))

    @property
    def config(self) -> OWCModelImportConfig:
        """Get daemon configuration."""
        return self._daemon_config

    # =========================================================================
    # OWC Operations
    # =========================================================================

    async def _run_command(self, command: str) -> tuple[bool, str]:
        """Run command on OWC host (locally or via SSH)."""
        if self._is_local:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self._daemon_config.ssh_timeout,
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
            if self._ssh_client is None:
                return False, "SSH client not initialized"
            result = await self._ssh_client.run_async(
                command, timeout=self._daemon_config.ssh_timeout
            )
            if result.success:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip() or result.error or "Unknown error"

    async def _check_owc_available(self) -> bool:
        """Check if OWC drive is accessible."""
        if self._is_local:
            owc_path = Path(self._daemon_config.owc_base_path)
            return owc_path.exists() and owc_path.is_dir()

        success, _ = await self._run_command(
            f"ls -d {shlex.quote(self._daemon_config.owc_base_path)} 2>/dev/null"
        )
        return success

    async def _discover_owc_models(self) -> list[OWCModelInfo]:
        """Find all model files on OWC drive.

        Returns:
            List of OWCModelInfo for discovered models
        """
        models = []
        base_path = self._daemon_config.owc_base_path

        # Build find command for all model directories
        search_paths = " ".join(
            shlex.quote(f"{base_path}/{p}") for p in OWC_MODEL_PATHS
        )

        # Find all .pth files with size info
        find_cmd = (
            f"find {search_paths} -name '*.pth' -type f "
            f"-size +{self._daemon_config.min_model_size_bytes // 1024}k "
            f"-exec stat -f '%z %N' {{}} \\; 2>/dev/null || "
            f"find {search_paths} -name '*.pth' -type f "
            f"-size +{self._daemon_config.min_model_size_bytes // 1024}k "
            f"-exec stat --format='%s %n' {{}} \\; 2>/dev/null"
        )

        success, output = await self._run_command(find_cmd)

        if not success or not output.strip():
            logger.debug(f"[OWCModelImport] No models found or discovery failed")
            return models

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse size and path
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue

            try:
                file_size = int(parts[0])
                file_path = parts[1]

                # Convert absolute path to relative
                if file_path.startswith(base_path):
                    rel_path = file_path[len(base_path):].lstrip("/")
                else:
                    rel_path = file_path

                # Extract model info
                model_info = extract_model_info(rel_path, file_size)
                if model_info:
                    models.append(model_info)

            except (ValueError, IndexError):
                continue

        logger.info(f"[OWCModelImport] Discovered {len(models)} models on OWC")
        self._stats.models_discovered += len(models)

        return models

    async def _get_models_without_elo(
        self, models: list[OWCModelInfo]
    ) -> list[OWCModelInfo]:
        """Filter to models that have no Elo rating.

        Args:
            models: List of discovered models

        Returns:
            Filtered list of models without Elo ratings
        """
        try:
            from app.training.elo_service import EloService

            elo_service = EloService.get_instance()
            unevaluated = []

            for model in models:
                if model.config_key is None:
                    # Can't check Elo without knowing config
                    unevaluated.append(model)
                    continue

                # Check if model has Elo rating
                # Note: EloService uses model filename as participant ID
                try:
                    rating = elo_service.get_rating(model.file_name, model.config_key)
                    if rating is None:
                        model.has_elo = False
                        unevaluated.append(model)
                    else:
                        model.has_elo = True
                        self._stats.models_skipped_has_elo += 1
                except Exception:
                    # If we can't check, assume no Elo
                    unevaluated.append(model)

            return unevaluated

        except ImportError:
            logger.warning("[OWCModelImport] EloService not available, returning all models")
            return models

    async def _import_model(self, model: OWCModelInfo) -> Path | None:
        """Import a model from OWC to local storage.

        Args:
            model: Model info to import

        Returns:
            Local path if successful, None otherwise
        """
        self._daemon_config.import_dir.mkdir(parents=True, exist_ok=True)

        # Generate local filename (preserve original name)
        local_path = self._daemon_config.import_dir / model.file_name
        source_path = Path(self._daemon_config.owc_base_path) / model.path

        try:
            if self._is_local:
                # Local mode: direct file copy
                if not source_path.exists():
                    logger.warning(f"[OWCModelImport] Source not found: {source_path}")
                    return None

                await asyncio.to_thread(shutil.copy2, source_path, local_path)
            else:
                # Remote mode: rsync
                rsync_cmd = [
                    "rsync",
                    "-az",
                    "--progress",
                    "-e",
                    f"ssh -i {self._daemon_config.owc_ssh_key} -o StrictHostKeyChecking=no",
                    f"{self._daemon_config.owc_user}@{self._daemon_config.owc_host}:{source_path}",
                    str(local_path),
                ]

                result = await asyncio.to_thread(
                    subprocess.run,
                    rsync_cmd,
                    capture_output=True,
                    timeout=self._daemon_config.rsync_timeout,
                )

                if result.returncode != 0:
                    logger.error(
                        f"[OWCModelImport] rsync failed for {model.file_name}: "
                        f"{result.stderr.decode()}"
                    )
                    return None

            # Validate imported file using ImportDaemonMixin
            validation = await self._validate_import(local_path, expected_type="pth")
            if not validation.valid:
                logger.warning(
                    f"[OWCModelImport] Imported file failed validation: {validation.error}"
                )
                local_path.unlink(missing_ok=True)
                self._stats.models_skipped_invalid += 1
                return None

            logger.info(f"[OWCModelImport] Imported: {model.file_name}")
            self._stats.models_imported += 1
            self._imported_models.add(model.file_name)

            return local_path

        except Exception as e:
            logger.error(f"[OWCModelImport] Failed to import {model.file_name}: {e}")
            self._stats.import_errors += 1
            return None

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Run one import cycle."""
        # February 2026: Block when coordinator is low on RAM/disk
        from app.utils.resource_guard import coordinator_resource_gate
        if not coordinator_resource_gate("OWC_MODEL_IMPORT"):
            return

        cycle_start = time.time()
        self._stats.cycle_count += 1

        if not self._daemon_config.enabled:
            logger.debug("[OWCModelImport] Daemon disabled, skipping cycle")
            return

        # Check OWC availability
        if not await self._check_owc_available():
            self._owc_available = False
            logger.warning("[OWCModelImport] OWC drive not accessible")
            return

        self._owc_available = True

        # Discover models
        all_models = await self._discover_owc_models()
        if not all_models:
            logger.debug("[OWCModelImport] No models discovered")
            return

        # Filter to models without Elo
        unevaluated = await self._get_models_without_elo(all_models)
        if not unevaluated:
            logger.debug("[OWCModelImport] All discovered models have Elo ratings")
            return

        logger.info(
            f"[OWCModelImport] Found {len(unevaluated)} models without Elo ratings"
        )

        # Emit discovery event for observability
        await self._emit_discovery_event(all_models, unevaluated)

        # Import up to max_per_cycle models
        imported_count = 0
        for model in unevaluated[:self._daemon_config.max_models_per_cycle]:
            # Skip if already imported this session
            if model.file_name in self._imported_models:
                continue

            local_path = await self._import_model(model)
            if local_path:
                imported_count += 1
                await self._emit_model_imported_event(model, local_path)

        cycle_duration = time.time() - cycle_start
        self._stats.last_cycle_time = time.time()
        self._stats.last_cycle_duration = cycle_duration

        logger.info(
            f"[OWCModelImport] Cycle complete: imported {imported_count} models "
            f"in {cycle_duration:.1f}s"
        )

    async def _emit_discovery_event(
        self, all_models: list[OWCModelInfo], unevaluated: list[OWCModelInfo]
    ) -> None:
        """Emit OWC_MODELS_DISCOVERED event for observability."""
        # Count by config
        config_counts: dict[str, int] = {}
        for model in unevaluated:
            key = model.config_key or "unknown"
            config_counts[key] = config_counts.get(key, 0) + 1

        payload = {
            "total_discovered": len(all_models),
            "unevaluated": len(unevaluated),
            "by_config": config_counts,
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.OWC_MODELS_DISCOVERED, payload)
        except Exception as e:
            logger.debug(f"[OWCModelImport] Failed to emit discovery event: {e}")

    async def _emit_model_imported_event(
        self, model: OWCModelInfo, local_path: Path
    ) -> None:
        """Emit MODEL_IMPORTED event after successful import."""
        payload = {
            "model_path": str(local_path),
            "original_path": model.path,
            "file_name": model.file_name,
            "board_type": model.board_type,
            "num_players": model.num_players,
            "config_key": model.config_key,
            "architecture_version": model.architecture_version,
            "file_size": model.file_size,
            "source": "owc_import",
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.MODEL_IMPORTED, payload)
        except Exception as e:
            logger.debug(f"[OWCModelImport] Failed to emit import event: {e}")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        details = {
            "enabled": self._daemon_config.enabled,
            "owc_available": self._owc_available,
            "is_local": self._is_local,
            "cycle_count": self._stats.cycle_count,
            "models_discovered": self._stats.models_discovered,
            "models_imported": self._stats.models_imported,
            "models_skipped_has_elo": self._stats.models_skipped_has_elo,
            "import_errors": self._stats.import_errors,
            "last_cycle_time": self._stats.last_cycle_time,
            "last_cycle_duration": self._stats.last_cycle_duration,
        }

        if not self._daemon_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status="disabled",
                message="OWC model import daemon is disabled",
                details=details,
            )

        if not self._owc_available:
            return HealthCheckResult(
                healthy=False,
                status="degraded",
                message="OWC drive not accessible",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status="healthy",
            message=f"Imported {self._stats.models_imported} models",
            details=details,
        )


# ============================================================================
# Singleton Access
# ============================================================================

_daemon_instance: OWCModelImportDaemon | None = None


def get_owc_model_import_daemon(
    config: OWCModelImportConfig | None = None,
) -> OWCModelImportDaemon:
    """Get the singleton OWCModelImportDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton daemon instance
    """
    global _daemon_instance

    if _daemon_instance is None:
        _daemon_instance = OWCModelImportDaemon(config)
    return _daemon_instance


def reset_owc_model_import_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None
