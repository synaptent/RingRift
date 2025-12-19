"""Model Retention Manager - file-level model lifecycle management (December 2025).

This module provides automated model FILE management including:
- Model organization with canonical naming (board_type_Np format)
- Periodic cleanup based on retention policies
- Archival and deletion of old model files
- Integration with ModelRegistry for stage tracking

Architecture Note:
    This module handles FILE-LEVEL operations (archival, deletion, disk space).
    For FULL LIFECYCLE orchestration (training, promotion, P2P sync), use:
    - app.integration.model_lifecycle.ModelLifecycleManager

    The two modules are complementary:
    - ModelRetentionManager (this): Manages model files on disk
    - ModelLifecycleManager (integration): Orchestrates full model lifecycle

RETENTION POLICY (configurable):
    - Keep latest N production models per config
    - Keep top M models by Elo per config
    - Archive models older than X days with poor performance
    - Delete archived models after Y days

Usage:
    from app.training.model_lifecycle import ModelRetentionManager

    manager = ModelRetentionManager()

    # Run full lifecycle maintenance
    result = manager.run_maintenance()
    print(f"Archived: {result.archived}, Deleted: {result.deleted}")

    # Check specific config
    manager.check_config("square8_2p")
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.utils.canonical_naming import (
    normalize_board_type,
    make_config_key,
    parse_config_key,
    CANONICAL_CONFIG_KEYS,
)

# Unified signals integration - defer maintenance when training is urgent
try:
    from app.training.unified_signals import (
        get_signal_computer,
        TrainingUrgency,
        TrainingSignals,
    )
    HAS_UNIFIED_SIGNALS = True
except ImportError:
    HAS_UNIFIED_SIGNALS = False
    get_signal_computer = None
    TrainingUrgency = None
    TrainingSignals = None

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Model retention policy configuration."""

    # Per-config limits
    max_models_per_config: int = 100  # Trigger culling above this
    keep_top_by_elo: int = 25  # Always keep top N by Elo
    keep_latest_production: int = 5  # Keep N most recent production models

    # Time-based retention
    archive_after_days: int = 30  # Archive old low-performing models
    delete_archived_after_days: int = 90  # Delete archived models

    # Performance thresholds
    min_elo_for_retention: float = -100  # Archive models below this Elo delta
    min_games_for_archival: int = 20  # Don't archive models with few games

    # Safety limits
    min_models_to_keep: int = 10  # Never go below this per config
    cooldown_hours: float = 1.0  # Hours between maintenance runs per config


@dataclass
class MaintenanceResult:
    """Result of a maintenance run."""

    config_key: str
    models_before: int
    models_after: int
    archived: int
    deleted: int
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class FullMaintenanceResult:
    """Result of full maintenance across all configs."""

    total_archived: int = 0
    total_deleted: int = 0
    total_errors: int = 0
    per_config_results: Dict[str, MaintenanceResult] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ModelRetentionManager:
    """Manages model file retention with canonical naming and policies.

    Handles FILE-LEVEL operations (archival, deletion, disk space).
    For full lifecycle orchestration, use integration.model_lifecycle.

    Integrates with:
    - app/utils/canonical_naming for consistent naming
    - app/tournament/model_culling for Elo-based culling
    - app/training/model_registry for stage tracking
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        elo_db_path: Optional[Path] = None,
        policy: Optional[RetentionPolicy] = None,
    ):
        """Initialize lifecycle manager.

        Args:
            model_dir: Directory containing model files (default: models/)
            elo_db_path: Path to Elo database (default: data/unified_elo.db)
            policy: Retention policy (default: RetentionPolicy())
        """
        self.model_dir = Path(model_dir) if model_dir else self._default_model_dir()
        self.elo_db_path = Path(elo_db_path) if elo_db_path else self._default_elo_db()
        self.policy = policy or RetentionPolicy()

        # Track last maintenance time per config
        self._last_maintenance: Dict[str, float] = {}

        # Unified signals integration - check training urgency before maintenance
        self._signal_computer = get_signal_computer() if HAS_UNIFIED_SIGNALS else None

        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "archived").mkdir(exist_ok=True)

    def _default_model_dir(self) -> Path:
        """Get default model directory."""
        # Try environment variable first
        env_path = os.environ.get("RINGRIFT_MODEL_DIR")
        if env_path:
            return Path(env_path)

        # Default relative to ai-service root
        ai_service_root = Path(__file__).parent.parent.parent
        return ai_service_root / "models"

    def _default_elo_db(self) -> Path:
        """Get default Elo database path."""
        env_path = os.environ.get("RINGRIFT_ELO_DB")
        if env_path:
            return Path(env_path)

        ai_service_root = Path(__file__).parent.parent.parent
        return ai_service_root / "data" / "unified_elo.db"

    def _get_culler(self):
        """Get or create ModelCullingController."""
        try:
            from app.tournament.model_culling import ModelCullingController

            return ModelCullingController(
                elo_db_path=self.elo_db_path,
                model_dir=self.model_dir,
                cull_threshold=self.policy.max_models_per_config,
                keep_fraction=self.policy.keep_top_by_elo / self.policy.max_models_per_config,
            )
        except ImportError:
            logger.warning("ModelCullingController not available")
            return None

    def _needs_maintenance(self, config_key: str) -> bool:
        """Check if config needs maintenance based on cooldown."""
        last = self._last_maintenance.get(config_key, 0)
        cooldown_seconds = self.policy.cooldown_hours * 3600
        return time.time() - last >= cooldown_seconds

    def _should_defer_for_training(self, config_key: str) -> Tuple[bool, str]:
        """Check if maintenance should be deferred due to training urgency.

        When training is CRITICAL or HIGH urgency, defer non-essential maintenance
        to avoid I/O contention and let training proceed uninterrupted.

        Args:
            config_key: Config to check

        Returns:
            Tuple of (should_defer, reason)
        """
        if self._signal_computer is None:
            return False, "unified_signals_not_available"

        try:
            signals = self._signal_computer.compute_signals(
                current_games=0,  # We don't have game count here
                current_elo=None,
                config_key=config_key,
            )

            # Defer maintenance during high-priority training
            if signals.urgency in (TrainingUrgency.CRITICAL, TrainingUrgency.HIGH):
                return True, f"training_urgency_{signals.urgency.value}"

            return False, "training_not_urgent"
        except Exception as e:
            logger.warning(f"Failed to check training urgency for {config_key}: {e}")
            return False, f"error: {e}"

    def get_training_signals(self, config_key: str) -> Optional["TrainingSignals"]:
        """Get current training signals for a config.

        Args:
            config_key: Config to check

        Returns:
            TrainingSignals or None if unavailable
        """
        if self._signal_computer is None:
            return None

        try:
            return self._signal_computer.compute_signals(
                current_games=0,
                current_elo=None,
                config_key=config_key,
            )
        except Exception as e:
            logger.warning(f"Failed to get training signals for {config_key}: {e}")
            return None

    def get_models_for_config(self, config_key: str) -> List[Path]:
        """Get all model files for a config.

        Args:
            config_key: Canonical config key (e.g., "square8_2p")

        Returns:
            List of model file paths
        """
        board_type, num_players = parse_config_key(config_key)

        models = []
        patterns = [
            f"*{board_type}*{num_players}p*.pth",
            f"*{board_type}_{num_players}p*.pth",
            f"*{config_key}*.pth",
        ]

        for pattern in patterns:
            models.extend(self.model_dir.glob(pattern))

        # Deduplicate
        return list(set(models))

    def get_archived_models(self, config_key: str) -> List[Path]:
        """Get archived models for a config."""
        archive_dir = self.model_dir / "archived" / config_key
        if not archive_dir.exists():
            return []
        return list(archive_dir.glob("*.pth"))

    def archive_model(self, model_path: Path, config_key: str, reason: str = "retention_policy") -> bool:
        """Archive a model file.

        Args:
            model_path: Path to model file
            config_key: Config key for organization
            reason: Reason for archival

        Returns:
            True if successful
        """
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False

        archive_dir = self.model_dir / "archived" / config_key
        archive_dir.mkdir(parents=True, exist_ok=True)

        dest = archive_dir / model_path.name
        try:
            shutil.move(str(model_path), str(dest))
            logger.info(f"Archived {model_path.name} -> {dest} (reason: {reason})")
            return True
        except Exception as e:
            logger.error(f"Failed to archive {model_path}: {e}")
            return False

    def delete_archived_model(self, model_path: Path) -> bool:
        """Permanently delete an archived model.

        Args:
            model_path: Path to archived model

        Returns:
            True if successful
        """
        if not model_path.exists():
            return True

        try:
            model_path.unlink()
            logger.info(f"Deleted archived model: {model_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_path}: {e}")
            return False

    def cleanup_old_archives(self, config_key: str) -> int:
        """Delete archived models older than retention period.

        Args:
            config_key: Config to clean

        Returns:
            Number of models deleted
        """
        archived = self.get_archived_models(config_key)
        cutoff = datetime.now() - timedelta(days=self.policy.delete_archived_after_days)

        deleted = 0
        for model_path in archived:
            try:
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                if mtime < cutoff:
                    if self.delete_archived_model(model_path):
                        deleted += 1
            except Exception as e:
                logger.warning(f"Error checking {model_path}: {e}")

        if deleted > 0:
            logger.info(f"[{config_key}] Deleted {deleted} archived models older than {self.policy.delete_archived_after_days} days")

        return deleted

    def check_config(self, config_key: str, force: bool = False) -> MaintenanceResult:
        """Run maintenance for a single config.

        Args:
            config_key: Config to maintain
            force: Skip cooldown check

        Returns:
            MaintenanceResult with operation details
        """
        # Normalize config key
        board_type, num_players = parse_config_key(config_key)
        config_key = make_config_key(board_type, num_players)

        models = self.get_models_for_config(config_key)
        models_before = len(models)
        errors = []

        # Check cooldown
        if not force and not self._needs_maintenance(config_key):
            return MaintenanceResult(
                config_key=config_key,
                models_before=models_before,
                models_after=models_before,
                archived=0,
                deleted=0,
            )

        # Check if training is urgent - defer maintenance to avoid I/O contention
        if not force:
            should_defer, reason = self._should_defer_for_training(config_key)
            if should_defer:
                logger.info(f"[{config_key}] Deferring maintenance: {reason}")
                return MaintenanceResult(
                    config_key=config_key,
                    models_before=models_before,
                    models_after=models_before,
                    archived=0,
                    deleted=0,
                    errors=[f"deferred: {reason}"],
                )

        archived = 0
        deleted = 0

        # Step 1: Use ModelCullingController for Elo-based culling
        culler = self._get_culler()
        if culler and models_before > self.policy.max_models_per_config:
            try:
                result = culler.check_and_cull(config_key)
                archived += result.culled
            except Exception as e:
                errors.append(f"Culling error: {e}")
                logger.error(f"[{config_key}] Culling failed: {e}")

        # Step 2: Clean up old archived models
        try:
            deleted = self.cleanup_old_archives(config_key)
        except Exception as e:
            errors.append(f"Archive cleanup error: {e}")
            logger.error(f"[{config_key}] Archive cleanup failed: {e}")

        # Update maintenance timestamp
        self._last_maintenance[config_key] = time.time()

        # Count remaining models
        models_after = len(self.get_models_for_config(config_key))

        return MaintenanceResult(
            config_key=config_key,
            models_before=models_before,
            models_after=models_after,
            archived=archived,
            deleted=deleted,
            errors=errors,
        )

    def run_maintenance(self, configs: Optional[List[str]] = None, force: bool = False) -> FullMaintenanceResult:
        """Run maintenance across all configs.

        Args:
            configs: Specific configs to maintain (default: all canonical configs)
            force: Skip cooldown checks

        Returns:
            FullMaintenanceResult with aggregated results
        """
        start_time = time.time()

        # Use all canonical configs if not specified
        if configs is None:
            configs = CANONICAL_CONFIG_KEYS

        result = FullMaintenanceResult()

        for config_key in configs:
            try:
                config_result = self.check_config(config_key, force=force)
                result.per_config_results[config_key] = config_result
                result.total_archived += config_result.archived
                result.total_deleted += config_result.deleted
                result.total_errors += len(config_result.errors)
            except Exception as e:
                logger.error(f"Maintenance failed for {config_key}: {e}")
                result.total_errors += 1

        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Maintenance complete: archived={result.total_archived}, "
            f"deleted={result.total_deleted}, errors={result.total_errors}, "
            f"duration={result.duration_seconds:.1f}s"
        )

        return result

    def get_status(self) -> Dict[str, Dict]:
        """Get status summary for all configs.

        Returns:
            Dict mapping config_key to status info
        """
        status = {}
        for config_key in CANONICAL_CONFIG_KEYS:
            models = self.get_models_for_config(config_key)
            archived = self.get_archived_models(config_key)
            last_maint = self._last_maintenance.get(config_key, 0)

            status[config_key] = {
                "active_models": len(models),
                "archived_models": len(archived),
                "last_maintenance": datetime.fromtimestamp(last_maint).isoformat() if last_maint else None,
                "needs_culling": len(models) > self.policy.max_models_per_config,
            }

        return status


# Convenience function for scripts
def run_model_maintenance(
    model_dir: Optional[str] = None,
    elo_db: Optional[str] = None,
    force: bool = False,
) -> FullMaintenanceResult:
    """Run model maintenance with default settings.

    Args:
        model_dir: Override model directory
        elo_db: Override Elo database path
        force: Force maintenance even within cooldown

    Returns:
        FullMaintenanceResult
    """
    manager = ModelRetentionManager(
        model_dir=Path(model_dir) if model_dir else None,
        elo_db_path=Path(elo_db) if elo_db else None,
    )
    return manager.run_maintenance(force=force)


# Backward-compatible alias (December 2025)
# Use ModelRetentionManager for new code - this alias prevents import breakage
ModelLifecycleManager = ModelRetentionManager
