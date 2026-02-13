"""UnevaluatedModelScannerDaemon - Scan for models without Elo ratings.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

This daemon scans all model sources for trained models that lack Elo ratings:
1. Local models/ directory
2. OWC imported models (models/owc_imports)
3. Cluster nodes (via P2P discovery)
4. ModelRegistry entries

For each unevaluated model, it:
1. Cross-references with EloService (unified_elo.db)
2. Computes curriculum-aware priority
3. Emits EVALUATION_REQUESTED events
4. Adds to PersistentEvaluationQueue

Priority Computation:
- Underserved configs (4-player, hexagonal): +50
- Recent training (within 24h): +30
- Canonical model: +20
- Diversity bonus (underrepresented arch): +10

Environment Variables:
    RINGRIFT_SCANNER_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_SCANNER_INTERVAL: Scan interval in seconds (default: 3600)
    RINGRIFT_SCANNER_MAX_QUEUE_PER_CYCLE: Max models to queue per cycle (default: 20)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.event_router import DataEventType, get_event_payload, safe_emit_event
from app.coordination.event_utils import make_config_key
from app.coordination.evaluation_queue import (
    PersistentEvaluationQueue,
    get_evaluation_queue,
)
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.models.discovery import ModelInfo, discover_models, MODELS_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "UnevaluatedModel",
    "UnevaluatedModelScannerConfig",
    "UnevaluatedModelScannerDaemon",
    "get_unevaluated_model_scanner_daemon",
    "reset_unevaluated_model_scanner_daemon",
]


# ============================================================================
# Configuration
# ============================================================================

# Curriculum weights for priority computation
# These match the existing CURRICULUM_WEIGHTS in selfplay_scheduler
CURRICULUM_WEIGHTS = {
    "hex8_2p": 1.0,
    "hex8_3p": 1.5,
    "hex8_4p": 2.0,
    "square8_2p": 1.0,
    "square8_3p": 1.5,
    "square8_4p": 2.0,
    "square19_2p": 1.0,
    "square19_3p": 1.2,
    "square19_4p": 1.5,
    "hexagonal_2p": 1.2,
    "hexagonal_3p": 1.5,
    "hexagonal_4p": 2.0,
}

# Priority bonuses
PRIORITY_BOOST_4_PLAYER = 30
PRIORITY_BOOST_UNDERSERVED = 50
PRIORITY_BOOST_CANONICAL = 20
PRIORITY_BOOST_RECENT = 30
PRIORITY_BOOST_DIVERSITY = 10

# Recent model threshold (24 hours)
RECENT_THRESHOLD_SECONDS = 86400

# Directories to scan for models
MODEL_SCAN_PATHS = [
    "models",
    "models/owc_imports",
    "models/archived",
    "models/training_runs",
]


@dataclass
class UnevaluatedModelScannerConfig:
    """Configuration for unevaluated model scanner daemon."""

    # Scan interval (1 hour default)
    scan_interval_seconds: int = 3600

    # Daemon control
    enabled: bool = True

    # Maximum models to queue per cycle
    max_queue_per_cycle: int = 20

    # Base priority for new requests
    base_priority: int = 50

    # Model directories to scan
    scan_paths: list[str] = field(default_factory=lambda: MODEL_SCAN_PATHS.copy())

    @classmethod
    def from_env(cls) -> "UnevaluatedModelScannerConfig":
        """Load configuration from environment."""
        return cls(
            enabled=os.getenv("RINGRIFT_SCANNER_ENABLED", "true").lower() == "true",
            scan_interval_seconds=int(os.getenv("RINGRIFT_SCANNER_INTERVAL", "3600")),
            max_queue_per_cycle=int(os.getenv("RINGRIFT_SCANNER_MAX_QUEUE_PER_CYCLE", "20")),
        )


@dataclass
class UnevaluatedModel:
    """Model that needs Elo evaluation."""

    path: str                      # Full path to model
    source: str                    # "local", "owc", "cluster", "registry"
    board_type: str | None         # Extracted from model
    num_players: int | None        # Extracted from model
    architecture_version: str | None
    priority: int = 50             # Computed priority
    discovered_at: float = field(default_factory=time.time)
    file_size: int = 0
    is_canonical: bool = False

    @property
    def config_key(self) -> str | None:
        """Get config key if board_type and num_players are known."""
        if self.board_type and self.num_players:
            return make_config_key(self.board_type, self.num_players)
        return None


@dataclass
class ScannerStats:
    """Statistics for scanner operations."""

    scan_count: int = 0
    models_found: int = 0
    models_without_elo: int = 0
    models_queued: int = 0
    models_skipped_has_elo: int = 0
    models_skipped_invalid: int = 0
    last_scan_time: float = 0.0
    last_scan_duration: float = 0.0


# ============================================================================
# Unevaluated Model Scanner Daemon
# ============================================================================


class UnevaluatedModelScannerDaemon(HandlerBase):
    """Daemon that scans for models without Elo ratings.

    This daemon:
    1. Scans local model directories
    2. Cross-references with EloService
    3. Computes curriculum-aware priorities
    4. Emits EVALUATION_REQUESTED events
    5. Adds to PersistentEvaluationQueue
    """

    def __init__(self, config: UnevaluatedModelScannerConfig | None = None):
        self._daemon_config = config or UnevaluatedModelScannerConfig.from_env()

        super().__init__(
            name="UnevaluatedModelScannerDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.scan_interval_seconds),
        )

        self._stats = ScannerStats()
        self._eval_queue: PersistentEvaluationQueue | None = None
        self._elo_service: Any = None  # Lazy loaded

        # Track architecture distribution for diversity bonus
        self._arch_counts: dict[str, int] = {}

    @property
    def config(self) -> UnevaluatedModelScannerConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _get_eval_queue(self) -> PersistentEvaluationQueue:
        """Get or create the evaluation queue."""
        if self._eval_queue is None:
            self._eval_queue = get_evaluation_queue()
        return self._eval_queue

    def _get_elo_service(self) -> Any:
        """Get or create the EloService."""
        if self._elo_service is None:
            try:
                from app.training.elo_service import EloService
                self._elo_service = EloService.get_instance()
            except ImportError:
                logger.warning("[Scanner] EloService not available")
        return self._elo_service

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _get_subscriptions(self) -> dict[Any, Any]:
        """Return event subscriptions for this daemon.

        Subscribes to MODEL_IMPORTED to immediately queue newly imported models
        for evaluation instead of waiting for the next scan cycle.
        """
        return {
            DataEventType.MODEL_IMPORTED: self._on_model_imported,
        }

    async def _on_model_imported(self, event: dict[str, Any]) -> None:
        """Handle MODEL_IMPORTED events from OWCModelImportDaemon.

        When a model is imported from OWC, immediately queue it for evaluation
        instead of waiting for the next periodic scan cycle.

        Args:
            event: Event payload with model_path, board_type, num_players, etc.
        """
        try:
            # Feb 2026: Extract payload from RouterEvent (was crashing with AttributeError)
            payload = get_event_payload(event)
            model_path = payload.get("model_path")
            if not model_path:
                logger.warning("[Scanner] MODEL_IMPORTED event missing model_path")
                return

            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            config_key = payload.get("config_key")

            if not board_type or not num_players:
                logger.debug(
                    f"[Scanner] MODEL_IMPORTED missing config: {model_path}"
                )
                return

            # Skip if already has Elo
            elo_service = self._get_elo_service()
            if elo_service:
                file_name = Path(model_path).name
                rating = elo_service.get_rating(file_name, config_key or f"{board_type}_{num_players}p")
                if rating is not None:
                    logger.debug(f"[Scanner] Skipping {file_name} - already has Elo")
                    self._stats.models_skipped_has_elo += 1
                    return

            # Compute priority with OWC import bonus
            priority = self._daemon_config.base_priority

            # 4-player bonus
            if num_players == 4:
                priority += PRIORITY_BOOST_4_PLAYER

            # Underserved config bonus
            if board_type in ("hexagonal",) or num_players == 4:
                priority += PRIORITY_BOOST_UNDERSERVED

            # Recent import bonus (always applies for MODEL_IMPORTED)
            priority += PRIORITY_BOOST_RECENT

            # Add to evaluation queue
            queue = self._get_eval_queue()
            request_id = queue.add_request(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                priority=priority,
                source="owc_import_event",
            )

            if request_id:
                self._stats.models_queued += 1

                # Emit EVALUATION_REQUESTED for EvaluationDaemon
                await self._emit_evaluation_requested_from_import(
                    model_path, board_type, num_players, config_key, priority, request_id
                )

                logger.info(
                    f"[Scanner] Queued imported model for evaluation: "
                    f"{Path(model_path).name} (priority={priority})"
                )

        except Exception as e:
            logger.error(f"[Scanner] Error handling MODEL_IMPORTED: {e}")

    async def _emit_evaluation_requested_from_import(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        config_key: str | None,
        priority: int,
        request_id: str,
    ) -> None:
        """Emit EVALUATION_REQUESTED event for an imported model."""
        payload = {
            "request_id": request_id,
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "config_key": config_key or make_config_key(board_type, num_players),
            "priority": priority,
            "source": "owc_import_event",
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.EVALUATION_REQUESTED, payload)
        except Exception as e:
            logger.debug(f"[Scanner] Failed to emit eval request: {e}")

    # =========================================================================
    # Model Discovery
    # =========================================================================

    async def _scan_local_models(self) -> list[UnevaluatedModel]:
        """Scan local model directories for models."""
        models = []

        for scan_path in self._daemon_config.scan_paths:
            path = Path(scan_path)
            if not path.is_absolute():
                # Try relative to MODELS_DIR and cwd
                candidates = [
                    MODELS_DIR.parent / scan_path,
                    Path.cwd() / scan_path,
                ]
            else:
                candidates = [path]

            for candidate in candidates:
                if not candidate.exists() or not candidate.is_dir():
                    continue

                # Find all .pth files
                for pth_file in candidate.rglob("*.pth"):
                    try:
                        model = self._parse_model_file(pth_file)
                        if model:
                            models.append(model)
                    except Exception as e:
                        logger.debug(f"[Scanner] Error parsing {pth_file}: {e}")

        return models

    def _parse_model_file(self, path: Path) -> UnevaluatedModel | None:
        """Parse a model file and extract information."""
        file_name = path.name
        file_size = path.stat().st_size if path.exists() else 0

        # Skip very small files (likely corrupt)
        if file_size < 100_000:  # 100KB minimum
            return None

        # Extract board type and num_players from filename
        board_type, num_players = self._extract_config_from_name(file_name)

        # Determine source from path
        path_str = str(path)
        if "owc_imports" in path_str:
            source = "owc"
        elif "archived" in path_str:
            source = "archived"
        else:
            source = "local"

        # Check if canonical
        is_canonical = "canonical_" in file_name.lower()

        # Extract architecture version
        arch_version = self._extract_architecture_version(file_name)

        return UnevaluatedModel(
            path=str(path),
            source=source,
            board_type=board_type,
            num_players=num_players,
            architecture_version=arch_version,
            file_size=file_size,
            is_canonical=is_canonical,
        )

    def _extract_config_from_name(self, name: str) -> tuple[str | None, int | None]:
        """Extract board_type and num_players from model filename."""
        name_lower = name.lower()

        # Board type patterns
        board_type = None
        for bt in ["hexagonal", "hex8", "square19", "square8"]:
            if bt in name_lower:
                board_type = bt
                break

        # Num players pattern
        num_players = None
        for np in [4, 3, 2]:
            if f"{np}p" in name_lower or f"_{np}p" in name_lower:
                num_players = np
                break

        return board_type, num_players

    def _extract_architecture_version(self, name: str) -> str | None:
        """Extract architecture version from filename."""
        # Pattern: v2, v4, v5, v5_heavy, v5-heavy-large, etc.
        match = re.search(r"(v\d+(?:[_-]?\w+)?)", name.lower())
        if match:
            return match.group(1)
        return None

    # =========================================================================
    # Elo Cross-Reference
    # =========================================================================

    async def _filter_models_without_elo(
        self, models: list[UnevaluatedModel]
    ) -> list[UnevaluatedModel]:
        """Filter to models that have no Elo rating.

        Args:
            models: List of discovered models

        Returns:
            Filtered list of models without Elo ratings
        """
        elo_service = self._get_elo_service()
        if elo_service is None:
            # No EloService, return all models
            return models

        unevaluated = []

        for model in models:
            if model.config_key is None:
                # Can't check Elo without config key, include anyway
                unevaluated.append(model)
                continue

            try:
                # Use model filename as participant ID
                model_name = Path(model.path).name
                rating = elo_service.get_rating(model_name, model.config_key)
                if rating is None:
                    unevaluated.append(model)
                else:
                    self._stats.models_skipped_has_elo += 1
            except Exception:
                # If we can't check, include it
                unevaluated.append(model)

        return unevaluated

    # =========================================================================
    # Priority Computation
    # =========================================================================

    def _compute_priority(self, model: UnevaluatedModel) -> int:
        """Compute curriculum-aware priority for a model.

        Priority factors:
        - Underserved configs (4-player, hexagonal): +50
        - 4-player models: +30
        - Canonical model: +20
        - Recent model (within 24h): +30
        - Diversity bonus (underrepresented arch): +10

        Args:
            model: Model to compute priority for

        Returns:
            Priority value (higher = evaluated sooner)
        """
        priority = self._daemon_config.base_priority

        if model.config_key:
            # Apply curriculum weight
            curriculum_weight = CURRICULUM_WEIGHTS.get(model.config_key, 1.0)
            priority = int(priority * curriculum_weight)

        # 4-player bonus
        if model.num_players == 4:
            priority += PRIORITY_BOOST_4_PLAYER

        # Underserved config bonus
        if model.board_type in ("hexagonal",) or model.num_players == 4:
            priority += PRIORITY_BOOST_UNDERSERVED

        # Canonical model bonus
        if model.is_canonical:
            priority += PRIORITY_BOOST_CANONICAL

        # Recent model bonus (if we can determine modification time)
        try:
            mtime = Path(model.path).stat().st_mtime
            if time.time() - mtime < RECENT_THRESHOLD_SECONDS:
                priority += PRIORITY_BOOST_RECENT
        except Exception:
            pass

        # Diversity bonus for underrepresented architectures
        if model.architecture_version:
            arch_count = self._arch_counts.get(model.architecture_version, 0)
            if arch_count < 5:  # Underrepresented
                priority += PRIORITY_BOOST_DIVERSITY

        model.priority = priority
        return priority

    def _update_arch_counts(self, models: list[UnevaluatedModel]) -> None:
        """Update architecture distribution for diversity computation."""
        self._arch_counts.clear()
        for model in models:
            if model.architecture_version:
                self._arch_counts[model.architecture_version] = (
                    self._arch_counts.get(model.architecture_version, 0) + 1
                )

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Run one scan cycle."""
        cycle_start = time.time()
        self._stats.scan_count += 1

        if not self._daemon_config.enabled:
            logger.debug("[Scanner] Daemon disabled, skipping cycle")
            return

        # Scan local models
        all_models = await self._scan_local_models()
        self._stats.models_found += len(all_models)

        if not all_models:
            logger.debug("[Scanner] No models found in scan paths")
            return

        # Update architecture counts for diversity
        self._update_arch_counts(all_models)

        # Filter to models without Elo
        unevaluated = await self._filter_models_without_elo(all_models)
        self._stats.models_without_elo += len(unevaluated)

        if not unevaluated:
            logger.debug("[Scanner] All discovered models have Elo ratings")
            return

        # Compute priorities
        for model in unevaluated:
            self._compute_priority(model)

        # Sort by priority (highest first)
        unevaluated.sort(key=lambda m: m.priority, reverse=True)

        logger.info(
            f"[Scanner] Found {len(unevaluated)} models without Elo ratings"
        )

        # Emit summary event for observability
        await self._emit_summary_event(all_models, unevaluated)

        # Queue top models for evaluation
        queue = self._get_eval_queue()
        queued_count = 0

        for model in unevaluated[:self._daemon_config.max_queue_per_cycle]:
            if model.board_type is None or model.num_players is None:
                self._stats.models_skipped_invalid += 1
                continue

            request_id = queue.add_request(
                model_path=model.path,
                board_type=model.board_type,
                num_players=model.num_players,
                priority=model.priority,
                source=f"scanner_{model.source}",
            )

            if request_id:
                queued_count += 1
                self._stats.models_queued += 1
                await self._emit_evaluation_requested(model, request_id)

        cycle_duration = time.time() - cycle_start
        self._stats.last_scan_time = time.time()
        self._stats.last_scan_duration = cycle_duration

        logger.info(
            f"[Scanner] Cycle complete: queued {queued_count}/{len(unevaluated)} models "
            f"in {cycle_duration:.1f}s"
        )

    async def _emit_summary_event(
        self, all_models: list[UnevaluatedModel], unevaluated: list[UnevaluatedModel]
    ) -> None:
        """Emit UNEVALUATED_MODELS_FOUND event for observability."""
        # Count by config
        config_counts: dict[str, int] = {}
        for model in unevaluated:
            key = model.config_key or "unknown"
            config_counts[key] = config_counts.get(key, 0) + 1

        payload = {
            "total_scanned": len(all_models),
            "unevaluated": len(unevaluated),
            "by_config": config_counts,
            "by_source": self._count_by_source(unevaluated),
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.UNEVALUATED_MODELS_FOUND, payload)
        except Exception as e:
            logger.debug(f"[Scanner] Failed to emit summary event: {e}")

    async def _emit_evaluation_requested(
        self, model: UnevaluatedModel, request_id: str
    ) -> None:
        """Emit EVALUATION_REQUESTED event."""
        payload = {
            "request_id": request_id,
            "model_path": model.path,
            "board_type": model.board_type,
            "num_players": model.num_players,
            "config_key": model.config_key,
            "priority": model.priority,
            "source": f"scanner_{model.source}",
            "architecture_version": model.architecture_version,
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.EVALUATION_REQUESTED, payload)
        except Exception as e:
            logger.debug(f"[Scanner] Failed to emit eval request event: {e}")

    def _count_by_source(self, models: list[UnevaluatedModel]) -> dict[str, int]:
        """Count models by source."""
        counts: dict[str, int] = {}
        for model in models:
            counts[model.source] = counts.get(model.source, 0) + 1
        return counts

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        details = {
            "enabled": self._daemon_config.enabled,
            "scan_count": self._stats.scan_count,
            "models_found": self._stats.models_found,
            "models_without_elo": self._stats.models_without_elo,
            "models_queued": self._stats.models_queued,
            "models_skipped_has_elo": self._stats.models_skipped_has_elo,
            "last_scan_time": self._stats.last_scan_time,
            "last_scan_duration": self._stats.last_scan_duration,
        }

        if not self._daemon_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status="disabled",
                message="Unevaluated model scanner is disabled",
                details=details,
            )

        # Check queue status
        try:
            queue = self._get_eval_queue()
            queue_status = queue.get_queue_status()
            details["queue_pending"] = queue_status.get("pending", 0)
            details["queue_running"] = queue_status.get("running", 0)
        except Exception as e:
            details["queue_error"] = str(e)

        return HealthCheckResult(
            healthy=True,
            status="healthy",
            message=f"Queued {self._stats.models_queued} models for evaluation",
            details=details,
        )


# ============================================================================
# Singleton Access
# ============================================================================

_daemon_instance: UnevaluatedModelScannerDaemon | None = None


def get_unevaluated_model_scanner_daemon(
    config: UnevaluatedModelScannerConfig | None = None,
) -> UnevaluatedModelScannerDaemon:
    """Get the singleton UnevaluatedModelScannerDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton daemon instance
    """
    global _daemon_instance

    if _daemon_instance is None:
        _daemon_instance = UnevaluatedModelScannerDaemon(config)
    return _daemon_instance


def reset_unevaluated_model_scanner_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None
