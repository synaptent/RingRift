"""Online Model Merge Daemon.

Periodically validates shadow models that have received online learning
updates and merges them into canonical models if they show improvement.

January 2026 - Human game training enhancement.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

# Default paths
SHADOW_MODEL_DIR = Path("models/shadow")
CANONICAL_MODEL_DIR = Path("models")
BACKUP_MODEL_DIR = Path("models/backup")


@dataclass
class OnlineMergeConfig:
    """Configuration for online merge daemon."""

    # Validation settings
    min_games_before_validation: int = 10  # Minimum games in shadow buffer before validating
    validation_games: int = 20  # Games to play in mini-gauntlet
    win_rate_threshold: float = 0.55  # Shadow must win >= 55% to merge

    # Timing
    check_interval_seconds: float = 3600.0  # Check every hour
    min_hours_since_last_merge: float = 24.0  # Don't merge more than once per day

    # Safety
    create_backup: bool = True  # Backup canonical before merge
    max_backup_count: int = 5  # Keep last N backups


class OnlineMergeDaemon(HandlerBase):
    """Daemon that validates and merges shadow models into canonical models.

    The shadow model receives online learning updates from human games.
    This daemon periodically:
    1. Checks if shadow models exist and have received updates
    2. Runs a mini-gauntlet between shadow and canonical models
    3. If shadow wins >= 55%, merges it into canonical
    4. Backs up the old canonical model before replacement

    This provides a safe way to incorporate human game learning into
    the production models without risking catastrophic forgetting.
    """

    def __init__(self, config: OnlineMergeConfig | None = None) -> None:
        """Initialize the merge daemon.

        Args:
            config: Daemon configuration
        """
        resolved_config = config or OnlineMergeConfig()
        super().__init__(
            name="online_merge_daemon",
            config=resolved_config,
            cycle_interval=resolved_config.check_interval_seconds,
        )

        # Track last merge times per config
        self._last_merge_times: dict[str, float] = {}

        # Track validation results
        self._validation_results: dict[str, dict[str, Any]] = {}

    async def _run_cycle(self) -> None:
        """Main cycle: check and merge shadow models."""
        try:
            shadow_models = self._find_shadow_models()
            if not shadow_models:
                logger.debug("No shadow models found")
                return

            for shadow_path in shadow_models:
                config_key = self._extract_config_key(shadow_path)
                if not config_key:
                    continue

                # Check if enough time has passed since last merge
                if not self._can_merge(config_key):
                    logger.debug(f"Skipping {config_key}: too soon since last merge")
                    continue

                # Run validation
                result = await self._validate_shadow_model(shadow_path, config_key)
                self._validation_results[config_key] = result

                # Merge if validation passed
                if result.get("should_merge", False):
                    await self._merge_shadow_to_canonical(shadow_path, config_key)

        except Exception as e:
            logger.error(f"Error in merge cycle: {e}")

    def _find_shadow_models(self) -> list[Path]:
        """Find all shadow models that may need merging."""
        if not SHADOW_MODEL_DIR.exists():
            return []

        return list(SHADOW_MODEL_DIR.glob("*_online.pth"))

    def _extract_config_key(self, shadow_path: Path) -> str | None:
        """Extract config key (e.g., 'hex8_2p') from shadow model path."""
        # Path like: models/shadow/hex8_2p_online.pth -> hex8_2p
        name = shadow_path.stem  # hex8_2p_online
        if name.endswith("_online"):
            return name[:-7]  # Remove '_online'
        return None

    def _get_canonical_path(self, config_key: str) -> Path:
        """Get canonical model path for config."""
        return CANONICAL_MODEL_DIR / f"canonical_{config_key}.pth"

    def _can_merge(self, config_key: str) -> bool:
        """Check if enough time has passed since last merge."""
        import time

        last_merge = self._last_merge_times.get(config_key, 0)
        hours_since = (time.time() - last_merge) / 3600
        return hours_since >= self.config.min_hours_since_last_merge

    async def _validate_shadow_model(
        self, shadow_path: Path, config_key: str
    ) -> dict[str, Any]:
        """Run mini-gauntlet to validate shadow model.

        Args:
            shadow_path: Path to shadow model
            config_key: Config key (e.g., 'hex8_2p')

        Returns:
            Validation results including win rate and merge recommendation
        """
        canonical_path = self._get_canonical_path(config_key)

        if not canonical_path.exists():
            logger.warning(f"Canonical model not found: {canonical_path}")
            return {"error": "canonical_not_found", "should_merge": False}

        def _run_gauntlet() -> dict[str, Any]:
            """Run gauntlet in thread."""
            try:
                from app.training.game_gauntlet import run_gauntlet_evaluation

                # Parse config key for board type and num players
                parts = config_key.rsplit("_", 1)
                if len(parts) != 2:
                    return {"error": "invalid_config_key", "should_merge": False}

                board_type = parts[0]
                num_players = int(parts[1].rstrip("p"))

                results = run_gauntlet_evaluation(
                    model_path=str(shadow_path),
                    board_type=board_type,
                    num_players=num_players,
                    opponent_model=str(canonical_path),
                    num_games=self.config.validation_games,
                )

                shadow_wins = results.get("wins", 0)
                total = results.get("total", self.config.validation_games)
                win_rate = shadow_wins / total if total > 0 else 0.0

                return {
                    "shadow_wins": shadow_wins,
                    "total_games": total,
                    "win_rate": round(win_rate, 3),
                    "should_merge": win_rate >= self.config.win_rate_threshold,
                    "threshold": self.config.win_rate_threshold,
                }

            except ImportError as e:
                logger.warning(f"Gauntlet module not available: {e}")
                return {"error": "gauntlet_unavailable", "should_merge": False}
            except Exception as e:
                logger.error(f"Gauntlet validation failed: {e}")
                return {"error": str(e), "should_merge": False}

        return await asyncio.to_thread(_run_gauntlet)

    async def _merge_shadow_to_canonical(
        self, shadow_path: Path, config_key: str
    ) -> bool:
        """Merge shadow model into canonical.

        Args:
            shadow_path: Path to shadow model
            config_key: Config key (e.g., 'hex8_2p')

        Returns:
            True if merge succeeded
        """
        canonical_path = self._get_canonical_path(config_key)

        def _do_merge() -> bool:
            try:
                # Create backup of canonical
                if self.config.create_backup and canonical_path.exists():
                    BACKUP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    import time

                    timestamp = int(time.time())
                    backup_path = BACKUP_MODEL_DIR / f"canonical_{config_key}_{timestamp}.pth"
                    shutil.copy2(canonical_path, backup_path)
                    logger.info(f"Backed up canonical model: {backup_path}")

                    # Clean old backups
                    self._cleanup_old_backups(config_key)

                # Copy shadow to canonical
                shutil.copy2(shadow_path, canonical_path)
                logger.info(f"Merged shadow model to canonical: {config_key}")

                # Update symlink if it exists
                symlink_path = CANONICAL_MODEL_DIR / f"ringrift_best_{config_key}.pth"
                if symlink_path.is_symlink():
                    symlink_path.unlink()
                    symlink_path.symlink_to(canonical_path.name)

                # Delete shadow model after successful merge
                shadow_path.unlink()
                logger.info(f"Removed merged shadow model: {shadow_path}")

                return True

            except Exception as e:
                logger.error(f"Failed to merge shadow model: {e}")
                return False

        success = await asyncio.to_thread(_do_merge)

        if success:
            import time

            self._last_merge_times[config_key] = time.time()

            # Emit event for model promotion
            try:
                from app.coordination.safe_event_emitter import safe_emit_event

                safe_emit_event(
                    "ONLINE_MODEL_MERGED",
                    {
                        "config_key": config_key,
                        "shadow_path": str(shadow_path),
                        "canonical_path": str(canonical_path),
                        "validation_results": self._validation_results.get(config_key, {}),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to emit ONLINE_MODEL_MERGED event: {e}")

        return success

    def _cleanup_old_backups(self, config_key: str) -> None:
        """Remove old backups, keeping only the most recent N."""
        if not BACKUP_MODEL_DIR.exists():
            return

        pattern = f"canonical_{config_key}_*.pth"
        backups = sorted(BACKUP_MODEL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)

        # Remove oldest backups if we have too many
        while len(backups) > self.config.max_backup_count:
            old_backup = backups.pop(0)
            try:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health check status."""
        shadow_count = len(self._find_shadow_models())
        return HealthCheckResult(
            healthy=True,
            details={
                "shadow_models": shadow_count,
                "validation_results": self._validation_results,
                "last_merge_times": self._last_merge_times,
            },
        )

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """No event subscriptions - runs on timer only."""
        return {}


# Singleton instance
_daemon_instance: OnlineMergeDaemon | None = None


def get_online_merge_daemon() -> OnlineMergeDaemon:
    """Get or create the singleton merge daemon."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = OnlineMergeDaemon()
    return _daemon_instance


def reset_online_merge_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None
