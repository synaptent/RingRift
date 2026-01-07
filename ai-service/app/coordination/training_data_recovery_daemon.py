"""Training Data Recovery Daemon.

Monitors TRAINING_FAILED events and automatically recovers from data corruption
by re-exporting training NPZ files from raw game databases.

Key features:
- Subscribes to TRAINING_FAILED events
- Detects data corruption failure patterns
- Triggers NPZ re-export from canonical databases
- Emits TRAINING_DATA_RECOVERED or TRAINING_DATA_RECOVERY_FAILED events

January 3, 2026: Created for Sprint 13.3 P2P stability improvements.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.coordination.singleton_mixin import SingletonMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingDataRecoveryConfig:
    """Configuration for Training Data Recovery daemon.

    Attributes:
        enabled: Whether daemon should run
        check_interval_seconds: How often to check for recovery opportunities
        cooldown_per_config_seconds: Minimum time between recovery attempts per config
        max_retries_per_config: Maximum recovery attempts before giving up
        corruption_patterns: Error patterns that indicate data corruption
        export_script_path: Path to the export replay dataset script
        max_concurrent_recoveries: Maximum parallel recovery operations
        recovery_timeout_seconds: Timeout for each recovery operation
    """

    enabled: bool = True
    check_interval_seconds: int = 30
    cooldown_per_config_seconds: int = 300  # 5 minutes between retries per config
    max_retries_per_config: int = 3
    corruption_patterns: list[str] = field(default_factory=lambda: [
        "corrupt",
        "invalid data",
        "shape mismatch",
        "KeyError",
        "truncated",
        "cannot read",
        "npz file",
        "bad numpy",
        "checksum",
        "decode error",
        "unexpected end",
    ])
    export_script_path: str = "scripts/export_replay_dataset.py"
    max_concurrent_recoveries: int = 2
    recovery_timeout_seconds: int = 600  # 10 minutes

    @classmethod
    def from_env(cls) -> "TrainingDataRecoveryConfig":
        """Load config from environment variables."""
        config = cls()

        if os.environ.get("RINGRIFT_DATA_RECOVERY_ENABLED"):
            config.enabled = os.environ.get("RINGRIFT_DATA_RECOVERY_ENABLED", "1") == "1"

        if os.environ.get("RINGRIFT_DATA_RECOVERY_INTERVAL"):
            try:
                config.check_interval_seconds = int(
                    os.environ.get("RINGRIFT_DATA_RECOVERY_INTERVAL", "30")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_DATA_RECOVERY_COOLDOWN"):
            try:
                config.cooldown_per_config_seconds = int(
                    os.environ.get("RINGRIFT_DATA_RECOVERY_COOLDOWN", "300")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_DATA_RECOVERY_MAX_RETRIES"):
            try:
                config.max_retries_per_config = int(
                    os.environ.get("RINGRIFT_DATA_RECOVERY_MAX_RETRIES", "3")
                )
            except ValueError:
                pass

        return config


# =============================================================================
# Daemon Implementation
# =============================================================================


class TrainingDataRecoveryDaemon(SingletonMixin, HandlerBase):
    """Daemon that recovers from training data corruption automatically.

    Subscribes to TRAINING_FAILED events and detects corruption-related failures.
    When corruption is detected, triggers NPZ re-export from canonical databases.

    January 2026: Migrated to use SingletonMixin for consistency.
    """

    def __init__(self, config: TrainingDataRecoveryConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Optional configuration. If None, loads from environment.
        """
        super().__init__(
            name="training_data_recovery",
            cycle_interval=30.0,  # Check every 30s
        )
        self.config = config or TrainingDataRecoveryConfig.from_env()

        # Track recovery attempts per config
        self._recovery_attempts: dict[str, int] = {}  # config_key -> attempt count
        self._last_recovery_time: dict[str, float] = {}  # config_key -> timestamp
        self._pending_recoveries: set[str] = set()  # config_keys awaiting recovery
        self._active_recoveries: set[str] = set()  # config_keys currently recovering

        # Stats
        self._recoveries_triggered: int = 0
        self._recoveries_succeeded: int = 0
        self._recoveries_failed: int = 0
        self._corruptions_detected: int = 0

        logger.info(
            f"TrainingDataRecoveryDaemon initialized: "
            f"enabled={self.config.enabled}, "
            f"cooldown={self.config.cooldown_per_config_seconds}s, "
            f"max_retries={self.config.max_retries_per_config}"
        )

    # -------------------------------------------------------------------------
    # HandlerBase Interface
    # -------------------------------------------------------------------------

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to training failure events."""
        return {
            "training_failed": self._on_training_failed,
        }

    async def _run_cycle(self) -> None:
        """Process pending recoveries each cycle."""
        if not self.config.enabled:
            return

        # Process pending recoveries up to concurrency limit
        while (
            self._pending_recoveries
            and len(self._active_recoveries) < self.config.max_concurrent_recoveries
        ):
            config_key = self._pending_recoveries.pop()

            # Check cooldown
            last_time = self._last_recovery_time.get(config_key, 0)
            if time.time() - last_time < self.config.cooldown_per_config_seconds:
                # Still in cooldown, skip this one
                continue

            # Check retry limit
            attempts = self._recovery_attempts.get(config_key, 0)
            if attempts >= self.config.max_retries_per_config:
                logger.warning(
                    f"[DataRecovery] Skipping {config_key}: max retries exceeded "
                    f"({attempts}/{self.config.max_retries_per_config})"
                )
                continue

            # Start recovery
            self._active_recoveries.add(config_key)
            asyncio.create_task(self._recover_config(config_key))

    async def _on_start(self) -> None:
        """Called when daemon starts."""
        logger.info("[DataRecovery] Starting TrainingDataRecoveryDaemon")

    async def _on_stop(self) -> None:
        """Called when daemon stops."""
        logger.info(
            f"[DataRecovery] Stopping. Stats: "
            f"corruptions={self._corruptions_detected}, "
            f"recoveries_triggered={self._recoveries_triggered}, "
            f"succeeded={self._recoveries_succeeded}, "
            f"failed={self._recoveries_failed}"
        )

    def health_check(self) -> HealthCheckResult:
        """Return health check status."""
        is_healthy = self.config.enabled and self._running

        # Calculate success rate
        total = self._recoveries_succeeded + self._recoveries_failed
        success_rate = (
            self._recoveries_succeeded / total if total > 0 else 1.0
        )

        return HealthCheckResult(
            status=CoordinatorStatus.HEALTHY if is_healthy else CoordinatorStatus.DEGRADED,
            message="Running" if is_healthy else "Not running",
            details={
                "enabled": self.config.enabled,
                "corruptions_detected": self._corruptions_detected,
                "recoveries_triggered": self._recoveries_triggered,
                "recoveries_succeeded": self._recoveries_succeeded,
                "recoveries_failed": self._recoveries_failed,
                "success_rate": round(success_rate, 3),
                "pending_recoveries": len(self._pending_recoveries),
                "active_recoveries": len(self._active_recoveries),
            },
        )

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    async def _on_training_failed(self, event: dict[str, Any]) -> None:
        """Handle TRAINING_FAILED event.

        Detects if failure is due to data corruption and queues recovery.
        """
        if not self.config.enabled:
            return

        config_key = event.get("config_key", "")
        error_message = str(event.get("error", "") or event.get("reason", ""))

        if not config_key:
            logger.debug("[DataRecovery] TRAINING_FAILED without config_key, skipping")
            return

        # Check if this looks like data corruption
        if not self._is_corruption_error(error_message):
            logger.debug(
                f"[DataRecovery] {config_key} failure not corruption-related: {error_message[:100]}"
            )
            return

        self._corruptions_detected += 1
        logger.warning(
            f"[DataRecovery] Detected data corruption for {config_key}: {error_message[:200]}"
        )

        # Queue for recovery if not already pending/active
        if config_key not in self._pending_recoveries and config_key not in self._active_recoveries:
            self._pending_recoveries.add(config_key)
            logger.info(f"[DataRecovery] Queued {config_key} for recovery")

    def _is_corruption_error(self, error_message: str) -> bool:
        """Check if error message indicates data corruption."""
        error_lower = error_message.lower()
        return any(
            pattern.lower() in error_lower
            for pattern in self.config.corruption_patterns
        )

    # -------------------------------------------------------------------------
    # Recovery Logic
    # -------------------------------------------------------------------------

    async def _recover_config(self, config_key: str) -> None:
        """Attempt to recover training data for a config.

        Re-exports NPZ from canonical database.
        """
        try:
            self._recoveries_triggered += 1
            self._recovery_attempts[config_key] = self._recovery_attempts.get(config_key, 0) + 1
            self._last_recovery_time[config_key] = time.time()

            logger.info(
                f"[DataRecovery] Starting recovery for {config_key} "
                f"(attempt {self._recovery_attempts[config_key]})"
            )

            # Parse config_key (e.g., "hex8_2p" -> board_type="hex8", num_players=2)
            board_type, num_players = self._parse_config_key(config_key)
            if not board_type or not num_players:
                logger.error(f"[DataRecovery] Invalid config_key format: {config_key}")
                self._recoveries_failed += 1
                await self._emit_recovery_failed(config_key, "Invalid config_key format")
                return

            # Run export script
            success = await self._run_export(config_key, board_type, num_players)

            if success:
                self._recoveries_succeeded += 1
                self._recovery_attempts[config_key] = 0  # Reset counter on success
                logger.info(f"[DataRecovery] Successfully recovered {config_key}")
                await self._emit_recovery_success(config_key, board_type, num_players)
            else:
                self._recoveries_failed += 1
                logger.error(f"[DataRecovery] Failed to recover {config_key}")
                await self._emit_recovery_failed(config_key, "Export script failed")

        except Exception as e:
            self._recoveries_failed += 1
            logger.error(f"[DataRecovery] Exception recovering {config_key}: {e}")
            await self._emit_recovery_failed(config_key, str(e))

        finally:
            self._active_recoveries.discard(config_key)

    def _parse_config_key(self, config_key: str) -> tuple[str | None, int | None]:
        """Parse config_key into board_type and num_players.

        Args:
            config_key: e.g., "hex8_2p", "square8_4p"

        Returns:
            Tuple of (board_type, num_players) or (None, None) on parse error.
        """
        try:
            # Handle format: {board_type}_{n}p
            if "_" not in config_key or not config_key.endswith("p"):
                return None, None

            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            players_part = parts[1]  # e.g., "2p"
            num_players = int(players_part[:-1])  # Remove 'p' suffix

            if num_players < 2 or num_players > 4:
                return None, None

            return board_type, num_players
        except (ValueError, IndexError):
            return None, None

    async def _run_export(
        self, config_key: str, board_type: str, num_players: int
    ) -> bool:
        """Run export script to regenerate NPZ.

        Returns:
            True if export succeeded, False otherwise.
        """
        # Build output path
        output_path = Path("data/training") / f"{config_key}.npz"

        # Build command
        cmd = [
            "python",
            self.config.export_script_path,
            "--use-discovery",
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--output", str(output_path),
            "--overwrite",  # Overwrite existing corrupt file
        ]

        logger.info(f"[DataRecovery] Running: {' '.join(cmd)}")

        try:
            # Run in thread to not block event loop
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent,  # ai-service root
                ),
                timeout=self.config.recovery_timeout_seconds,
            )

            if result.returncode == 0:
                logger.info(f"[DataRecovery] Export succeeded for {config_key}")
                return True
            else:
                logger.error(
                    f"[DataRecovery] Export failed for {config_key}: "
                    f"returncode={result.returncode}, stderr={result.stderr[:500] if result.stderr else ''}"
                )
                return False

        except asyncio.TimeoutError:
            logger.error(
                f"[DataRecovery] Export timed out for {config_key} "
                f"after {self.config.recovery_timeout_seconds}s"
            )
            return False
        except Exception as e:
            logger.error(f"[DataRecovery] Export exception for {config_key}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    async def _emit_recovery_success(
        self, config_key: str, board_type: str, num_players: int
    ) -> None:
        """Emit TRAINING_DATA_RECOVERED event."""
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                DataEventType.TRAINING_DATA_RECOVERED,
                {
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "recovery_time": time.time(),
                    "output_path": f"data/training/{config_key}.npz",
                },
                context="DataRecovery",
                log_after=f"Emitted TRAINING_DATA_RECOVERED for {config_key}",
            )
        except ImportError:
            logger.debug("[DataRecovery] Event emission unavailable")

    async def _emit_recovery_failed(self, config_key: str, reason: str) -> None:
        """Emit TRAINING_DATA_RECOVERY_FAILED event."""
        try:
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                DataEventType.TRAINING_DATA_RECOVERY_FAILED,
                {
                    "config_key": config_key,
                    "reason": reason,
                    "attempts": self._recovery_attempts.get(config_key, 0),
                    "max_retries": self.config.max_retries_per_config,
                    "failure_time": time.time(),
                },
                context="DataRecovery",
                log_after=f"Emitted TRAINING_DATA_RECOVERY_FAILED for {config_key}",
            )
        except ImportError:
            logger.debug("[DataRecovery] Event emission unavailable")


# =============================================================================
# Factory Functions
# =============================================================================


def get_training_data_recovery_daemon() -> TrainingDataRecoveryDaemon:
    """Get the singleton TrainingDataRecoveryDaemon instance."""
    return TrainingDataRecoveryDaemon.get_instance()


def create_training_data_recovery_daemon(
    config: TrainingDataRecoveryConfig | None = None,
) -> TrainingDataRecoveryDaemon:
    """Create a new TrainingDataRecoveryDaemon instance.

    Note: Use get_training_data_recovery_daemon() for singleton access.
    """
    return TrainingDataRecoveryDaemon(config=config)
