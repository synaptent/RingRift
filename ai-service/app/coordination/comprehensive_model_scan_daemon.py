"""ComprehensiveModelScanDaemon - Scan ALL model sources and queue multi-harness evaluations.

Sprint 17.9 (January 9, 2026): Comprehensive model evaluation automation.

This daemon discovers ALL models across:
1. Local models/ directory (NN and NNUE)
2. ClusterManifest - models registered on other nodes
3. ClusterModelDiscovery - SSH queries to training nodes
4. OWCModelDiscovery - external drive with archived models

For each discovered model, it:
1. Determines compatible harnesses (BRS, MaxN, Descent, Gumbel MCTS, etc.)
2. Checks EvaluationStatusTracker for what's already been evaluated
3. Queues unevaluated model+harness combinations to PersistentEvaluationQueue
4. Emits MULTI_HARNESS_EVALUATION_QUEUED events for observability

The goal is to ensure ALL models get fresh Elo ratings under multiple harnesses,
with all evaluation games saved for training.

Environment Variables:
    RINGRIFT_MODEL_SCAN_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_MODEL_SCAN_INTERVAL: Scan interval in seconds (default: 1800 = 30 min)
    RINGRIFT_MODEL_SCAN_MAX_QUEUE_PER_CYCLE: Max models to queue per cycle (default: 50)
    RINGRIFT_MODEL_SCAN_INCLUDE_OWC: Include OWC drive models (default: true)
    RINGRIFT_MODEL_SCAN_INCLUDE_CLUSTER: Include cluster node models (default: true)
    RINGRIFT_MODEL_SCAN_HARNESSES: Comma-separated harness types (default: brs,maxn,descent,gumbel_mcts)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ai.harness.base_harness import HarnessType, ModelType
from app.ai.harness.harness_registry import (
    get_harnesses_for_model_and_players,
    is_harness_valid_for_player_count,
)
from app.coordination.contracts import HealthCheckResult
from app.coordination.event_router import DataEventType, get_event_payload, safe_emit_event
from app.coordination.event_utils import make_config_key
from app.coordination.evaluation_queue import (
    PersistentEvaluationQueue,
    get_evaluation_queue,
)
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

__all__ = [
    "ComprehensiveModelScanConfig",
    "ComprehensiveModelScanDaemon",
    "DiscoveredModelInfo",
    "ScanStats",
    "get_comprehensive_model_scan_daemon",
    "reset_comprehensive_model_scan_daemon",
]


# ============================================================================
# Configuration
# ============================================================================

# Default harnesses to evaluate under (user requested: BRS, MaxN, Descent, Gumbel MCTS)
DEFAULT_HARNESSES = [
    HarnessType.BRS,
    HarnessType.MAXN,
    HarnessType.DESCENT,
    HarnessType.GUMBEL_MCTS,
]

# Priority bonuses for queue ordering
PRIORITY_BOOST_4_PLAYER = 30
PRIORITY_BOOST_UNDERSERVED = 50
PRIORITY_BOOST_CANONICAL = 20
PRIORITY_BOOST_RECENT = 30
PRIORITY_BOOST_OWC = 10  # Slight boost for OWC models (historical archive)

# Directories to scan for models
MODEL_SCAN_PATHS = [
    "models",
    "models/owc_imports",
    "models/archived",
    "models/training_runs",
    "models/synced",
    "models/nnue",
]


@dataclass
class ComprehensiveModelScanConfig:
    """Configuration for comprehensive model scan daemon."""

    # Scan interval (30 minutes default)
    scan_interval_seconds: int = 1800

    # Daemon control
    enabled: bool = True

    # Maximum models to queue per cycle
    max_queue_per_cycle: int = 50

    # Base priority for new requests
    base_priority: int = 50

    # Model directories to scan
    scan_paths: list[str] = field(default_factory=lambda: MODEL_SCAN_PATHS.copy())

    # Include remote sources
    include_owc: bool = True
    include_cluster: bool = True

    # Harness types to evaluate under
    harness_types: list[HarnessType] = field(
        default_factory=lambda: DEFAULT_HARNESSES.copy()
    )

    @classmethod
    def from_env(cls) -> "ComprehensiveModelScanConfig":
        """Load configuration from environment."""
        # Parse harness types from env
        harness_str = os.getenv("RINGRIFT_MODEL_SCAN_HARNESSES", "")
        if harness_str:
            harness_types = []
            for h in harness_str.split(","):
                h = h.strip().lower()
                try:
                    harness_types.append(HarnessType(h))
                except ValueError:
                    logger.warning(f"[ComprehensiveModelScan] Unknown harness: {h}")
            if not harness_types:
                harness_types = DEFAULT_HARNESSES.copy()
        else:
            harness_types = DEFAULT_HARNESSES.copy()

        return cls(
            enabled=os.getenv("RINGRIFT_MODEL_SCAN_ENABLED", "true").lower() == "true",
            scan_interval_seconds=int(
                os.getenv("RINGRIFT_MODEL_SCAN_INTERVAL", "1800")
            ),
            max_queue_per_cycle=int(
                os.getenv("RINGRIFT_MODEL_SCAN_MAX_QUEUE_PER_CYCLE", "50")
            ),
            include_owc=os.getenv("RINGRIFT_MODEL_SCAN_INCLUDE_OWC", "true").lower()
            == "true",
            include_cluster=os.getenv(
                "RINGRIFT_MODEL_SCAN_INCLUDE_CLUSTER", "true"
            ).lower()
            == "true",
            harness_types=harness_types,
        )


@dataclass
class DiscoveredModelInfo:
    """Model discovered from any source."""

    path: str
    name: str
    board_type: str | None
    num_players: int | None
    model_type: str  # "nn" or "nnue"
    architecture_version: str | None
    source: str  # "local", "cluster", "owc", "manifest"
    file_size: int = 0
    sha256: str | None = None
    is_canonical: bool = False
    discovered_at: float = field(default_factory=time.time)

    @property
    def config_key(self) -> str | None:
        """Get config key if board_type and num_players are known."""
        if self.board_type and self.num_players:
            return make_config_key(self.board_type, self.num_players)
        return None


@dataclass
class ScanStats:
    """Statistics for scanner operations."""

    scan_count: int = 0
    models_found: int = 0
    models_queued: int = 0
    harness_combinations_queued: int = 0
    models_skipped_already_evaluated: int = 0
    models_skipped_invalid: int = 0
    models_skipped_no_compatible_harness: int = 0
    owc_models_found: int = 0
    cluster_models_found: int = 0
    local_models_found: int = 0
    last_scan_time: float = 0.0
    last_scan_duration: float = 0.0
    errors: list[str] = field(default_factory=list)


# ============================================================================
# Comprehensive Model Scan Daemon
# ============================================================================


class ComprehensiveModelScanDaemon(HandlerBase):
    """Daemon that discovers ALL models and queues multi-harness evaluations.

    This daemon aggregates models from:
    1. Local models/ directory
    2. ClusterManifest - centrally registered models
    3. ClusterModelDiscovery - SSH queries to cluster nodes
    4. OWCModelDiscovery - external OWC drive

    For each model, it determines compatible harnesses and queues
    evaluation requests for any (model, harness) combination not yet evaluated.
    """

    def __init__(self, config: ComprehensiveModelScanConfig | None = None):
        self._scan_config = config or ComprehensiveModelScanConfig.from_env()

        super().__init__(
            name="ComprehensiveModelScanDaemon",
            config=self._scan_config,
            cycle_interval=float(self._scan_config.scan_interval_seconds),
        )

        self._stats = ScanStats()
        self._eval_queue: PersistentEvaluationQueue | None = None
        self._elo_service: Any = None  # Lazy loaded
        self._evaluation_tracker: Any = None  # Lazy loaded

        # Track unique models by SHA256 for deduplication
        self._seen_hashes: set[str] = set()

    @property
    def config(self) -> ComprehensiveModelScanConfig:
        """Get daemon configuration."""
        return self._scan_config

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
                logger.warning("[ComprehensiveModelScan] EloService not available")
        return self._elo_service

    def _get_evaluation_tracker(self) -> Any:
        """Get or create the EvaluationStatusTracker."""
        if self._evaluation_tracker is None:
            try:
                from app.training.evaluation_status import (
                    get_evaluation_status_tracker,
                )

                self._evaluation_tracker = get_evaluation_status_tracker()
            except ImportError:
                logger.warning(
                    "[ComprehensiveModelScan] EvaluationStatusTracker not available"
                )
        return self._evaluation_tracker

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _get_subscriptions(self) -> dict[Any, Any]:
        """Return event subscriptions for this daemon."""
        return {
            DataEventType.MODEL_IMPORTED: self._on_model_imported,
            DataEventType.MODEL_PROMOTED: self._on_model_promoted,
        }

    async def _on_model_imported(self, event: dict[str, Any]) -> None:
        """Handle MODEL_IMPORTED events - immediately queue for evaluation."""
        try:
            # Feb 2026: Extract payload from RouterEvent (was crashing with AttributeError)
            payload = get_event_payload(event)
            model_path = payload.get("model_path")
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")

            if not model_path or not board_type or not num_players:
                return

            model = DiscoveredModelInfo(
                path=model_path,
                name=Path(model_path).stem,
                board_type=board_type,
                num_players=num_players,
                model_type="nn",  # Assume NN for imported models
                architecture_version=payload.get("architecture_version"),
                source="import_event",
                is_canonical="canonical" in model_path.lower(),
            )

            await self._queue_model_for_all_harnesses(
                model, priority_boost=PRIORITY_BOOST_RECENT
            )

        except Exception as e:
            logger.error(f"[ComprehensiveModelScan] Error handling MODEL_IMPORTED: {e}")

    async def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Handle MODEL_PROMOTED events - queue promoted model for evaluation."""
        try:
            # Feb 2026: Extract payload from RouterEvent (was crashing with AttributeError)
            payload = get_event_payload(event)
            model_path = payload.get("model_path")
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")

            if not model_path or not board_type or not num_players:
                return

            model = DiscoveredModelInfo(
                path=model_path,
                name=Path(model_path).stem,
                board_type=board_type,
                num_players=num_players,
                model_type="nn",
                architecture_version=payload.get("model_version"),
                source="promotion_event",
                is_canonical="canonical" in model_path.lower(),
            )

            # Promoted models get highest priority
            await self._queue_model_for_all_harnesses(
                model, priority_boost=PRIORITY_BOOST_CANONICAL + PRIORITY_BOOST_RECENT
            )

        except Exception as e:
            logger.error(f"[ComprehensiveModelScan] Error handling MODEL_PROMOTED: {e}")

    # =========================================================================
    # Model Discovery from All Sources
    # =========================================================================

    async def _discover_all_models(self) -> list[DiscoveredModelInfo]:
        """Aggregate models from all sources."""
        all_models: list[DiscoveredModelInfo] = []
        self._seen_hashes.clear()

        # 1. Discover local models
        local_models = await self._discover_local_models()
        all_models.extend(local_models)
        self._stats.local_models_found = len(local_models)

        # 2. Discover cluster models (if enabled)
        if self._scan_config.include_cluster:
            try:
                cluster_models = await self._discover_cluster_models()
                # Deduplicate by hash
                for cm in cluster_models:
                    if cm.sha256 and cm.sha256 in self._seen_hashes:
                        continue
                    if cm.sha256:
                        self._seen_hashes.add(cm.sha256)
                    all_models.append(cm)
                self._stats.cluster_models_found = len(cluster_models)
            except Exception as e:
                logger.warning(f"[ComprehensiveModelScan] Cluster discovery failed: {e}")
                self._stats.errors.append(f"cluster_discovery: {e}")

        # 3. Discover OWC models (if enabled)
        if self._scan_config.include_owc:
            try:
                owc_models = await self._discover_owc_models()
                # Deduplicate by hash
                for om in owc_models:
                    if om.sha256 and om.sha256 in self._seen_hashes:
                        continue
                    if om.sha256:
                        self._seen_hashes.add(om.sha256)
                    all_models.append(om)
                self._stats.owc_models_found = len(owc_models)
            except Exception as e:
                logger.warning(f"[ComprehensiveModelScan] OWC discovery failed: {e}")
                self._stats.errors.append(f"owc_discovery: {e}")

        return all_models

    async def _discover_local_models(self) -> list[DiscoveredModelInfo]:
        """Discover models in local directories."""
        from app.models.discovery import discover_models, ModelInfo

        models: list[DiscoveredModelInfo] = []

        # Get all models (no filter by board/players)
        try:
            local_models: list[ModelInfo] = await asyncio.to_thread(discover_models)

            for m in local_models:
                model_type = "nnue" if "nnue" in m.path.lower() else "nn"
                dm = DiscoveredModelInfo(
                    path=m.path,
                    name=m.name,
                    board_type=m.board_type,
                    num_players=m.num_players,
                    model_type=model_type,
                    architecture_version=m.architecture_version,
                    source="local",
                    file_size=m.size_bytes,
                    is_canonical="canonical" in m.name.lower(),
                )
                models.append(dm)
                if m.sha256:
                    self._seen_hashes.add(m.sha256)
                    dm.sha256 = m.sha256

        except Exception as e:
            logger.error(f"[ComprehensiveModelScan] Local discovery error: {e}")
            self._stats.errors.append(f"local_discovery: {e}")

        return models

    async def _discover_cluster_models(self) -> list[DiscoveredModelInfo]:
        """Discover models on cluster nodes via ClusterModelDiscovery."""
        models: list[DiscoveredModelInfo] = []

        try:
            from app.models.cluster_discovery import get_cluster_model_discovery

            discovery = get_cluster_model_discovery()

            # Query all configs
            for board_type in ["hex8", "square8", "square19", "hexagonal"]:
                for num_players in [2, 3, 4]:
                    try:
                        remote_models = await asyncio.to_thread(
                            discovery.discover_cluster_models,
                            board_type=board_type,
                            num_players=num_players,
                            include_local=False,
                            include_remote=True,
                            max_remote_nodes=10,
                            timeout=30.0,
                        )

                        for rm in remote_models:
                            model_type = (
                                "nnue" if "nnue" in rm.remote_path.lower() else "nn"
                            )
                            dm = DiscoveredModelInfo(
                                path=rm.remote_path,
                                name=rm.model_info.name,
                                board_type=rm.model_info.board_type,
                                num_players=rm.model_info.num_players,
                                model_type=model_type,
                                architecture_version=rm.model_info.architecture_version,
                                source=f"cluster:{rm.node_id}",
                                file_size=rm.model_info.size_bytes,
                                is_canonical="canonical" in rm.model_info.name.lower(),
                            )
                            models.append(dm)

                    except Exception as e:
                        logger.debug(
                            f"[ComprehensiveModelScan] Error querying {board_type}_{num_players}p: {e}"
                        )

        except ImportError:
            logger.debug("[ComprehensiveModelScan] ClusterModelDiscovery not available")

        return models

    async def _discover_owc_models(self) -> list[DiscoveredModelInfo]:
        """Discover models on OWC external drive."""
        models: list[DiscoveredModelInfo] = []

        try:
            from app.models.owc_discovery import get_owc_discovery, DiscoveredModel

            discovery = get_owc_discovery()

            # Check if OWC is available
            if not await discovery.check_available():
                logger.debug("[ComprehensiveModelScan] OWC drive not available")
                return models

            owc_models: list[DiscoveredModel] = await discovery.discover_all_models(
                force_refresh=False, include_hashes=True
            )

            for om in owc_models:
                model_type = "nnue" if "nnue" in om.path.lower() else "nn"
                dm = DiscoveredModelInfo(
                    path=om.path,
                    name=om.file_name,
                    board_type=om.board_type,
                    num_players=om.num_players,
                    model_type=model_type,
                    architecture_version=om.architecture_version,
                    source="owc",
                    file_size=om.file_size,
                    sha256=om.sha256,
                    is_canonical=om.is_canonical,
                )
                models.append(dm)

        except ImportError:
            logger.debug("[ComprehensiveModelScan] OWCModelDiscovery not available")
        except Exception as e:
            logger.warning(f"[ComprehensiveModelScan] OWC discovery error: {e}")

        return models

    # =========================================================================
    # Harness Compatibility and Queue Logic
    # =========================================================================

    def _get_compatible_harnesses(
        self, model: DiscoveredModelInfo
    ) -> list[HarnessType]:
        """Get harnesses compatible with a model.

        Considers:
        - Model type (NN vs NNUE)
        - Number of players
        - Configured harness types to evaluate
        """
        if model.board_type is None or model.num_players is None:
            return []

        # Determine model type enum
        model_type_enum = (
            ModelType.NNUE
            if model.model_type == "nnue"
            else ModelType.NEURAL_NET
        )

        # Get harnesses valid for this model type AND player count
        compatible = get_harnesses_for_model_and_players(
            model_type_enum, model.num_players
        )

        # Filter to configured harnesses
        result = [h for h in compatible if h in self._scan_config.harness_types]

        return result

    def _needs_evaluation(
        self, model: DiscoveredModelInfo, harness_type: HarnessType
    ) -> bool:
        """Check if a model+harness combination needs evaluation.

        Uses EloService to check if we have an Elo rating for this combination.
        """
        if model.config_key is None:
            return False

        elo_service = self._get_elo_service()
        if elo_service is None:
            # If no Elo service, assume needs evaluation
            return True

        try:
            # Build composite participant ID
            # Format: model_name:harness_type:config_hash
            from app.training.composite_participant import normalize_nn_id
            model_name = normalize_nn_id(Path(model.path).stem)
            participant_id = f"{model_name}:{harness_type.value}"

            rating = elo_service.get_rating(participant_id, model.config_key)
            return rating is None

        except Exception:
            # If check fails, assume needs evaluation
            return True

    def _compute_priority(self, model: DiscoveredModelInfo) -> int:
        """Compute priority for a model evaluation request."""
        priority = self._scan_config.base_priority

        # 4-player bonus
        if model.num_players == 4:
            priority += PRIORITY_BOOST_4_PLAYER

        # Underserved config bonus
        if model.board_type in ("hexagonal",) or model.num_players == 4:
            priority += PRIORITY_BOOST_UNDERSERVED

        # Canonical model bonus
        if model.is_canonical:
            priority += PRIORITY_BOOST_CANONICAL

        # OWC bonus (historical models worth evaluating)
        if model.source == "owc":
            priority += PRIORITY_BOOST_OWC

        # Recent model bonus (check modification time if available)
        try:
            path = Path(model.path)
            if path.exists():
                mtime = path.stat().st_mtime
                if time.time() - mtime < 86400:  # 24 hours
                    priority += PRIORITY_BOOST_RECENT
        except Exception:
            pass

        return priority

    async def _queue_model_for_all_harnesses(
        self, model: DiscoveredModelInfo, priority_boost: int = 0
    ) -> int:
        """Queue a model for evaluation under all compatible harnesses.

        Returns:
            Number of harness combinations queued
        """
        if model.config_key is None:
            self._stats.models_skipped_invalid += 1
            return 0

        compatible_harnesses = self._get_compatible_harnesses(model)
        if not compatible_harnesses:
            self._stats.models_skipped_no_compatible_harness += 1
            return 0

        queue = self._get_eval_queue()
        base_priority = self._compute_priority(model) + priority_boost
        queued = 0

        for harness in compatible_harnesses:
            if not self._needs_evaluation(model, harness):
                self._stats.models_skipped_already_evaluated += 1
                continue

            request_id = queue.add_request(
                model_path=model.path,
                board_type=model.board_type,
                num_players=model.num_players,
                priority=base_priority,
                source=f"comprehensive_scan_{model.source}",
                harness_type=harness.value,
            )

            if request_id:
                queued += 1

        return queued

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Run one comprehensive scan cycle."""
        cycle_start = time.time()
        self._stats.scan_count += 1
        self._stats.errors.clear()

        if not self._scan_config.enabled:
            logger.debug("[ComprehensiveModelScan] Daemon disabled, skipping cycle")
            return

        # Discover all models from all sources
        all_models = await self._discover_all_models()
        self._stats.models_found = len(all_models)

        if not all_models:
            logger.info("[ComprehensiveModelScan] No models found across all sources")
            return

        logger.info(
            f"[ComprehensiveModelScan] Discovered {len(all_models)} models "
            f"(local: {self._stats.local_models_found}, "
            f"cluster: {self._stats.cluster_models_found}, "
            f"owc: {self._stats.owc_models_found})"
        )

        # Sort by priority (highest first)
        models_with_priority = [
            (self._compute_priority(m), m) for m in all_models
        ]
        models_with_priority.sort(key=lambda x: -x[0])

        # Queue top models for multi-harness evaluation
        total_combinations_queued = 0
        models_queued = 0

        for priority, model in models_with_priority:
            if models_queued >= self._scan_config.max_queue_per_cycle:
                break

            combinations = await self._queue_model_for_all_harnesses(model)
            if combinations > 0:
                total_combinations_queued += combinations
                models_queued += 1

        self._stats.models_queued = models_queued
        self._stats.harness_combinations_queued = total_combinations_queued

        cycle_duration = time.time() - cycle_start
        self._stats.last_scan_time = time.time()
        self._stats.last_scan_duration = cycle_duration

        logger.info(
            f"[ComprehensiveModelScan] Cycle complete: queued {total_combinations_queued} "
            f"harness combinations for {models_queued} models in {cycle_duration:.1f}s"
        )

        # Emit summary event
        await self._emit_scan_summary()

    async def _emit_scan_summary(self) -> None:
        """Emit MULTI_HARNESS_EVALUATION_QUEUED event for observability."""
        payload = {
            "total_models_found": self._stats.models_found,
            "models_queued": self._stats.models_queued,
            "harness_combinations_queued": self._stats.harness_combinations_queued,
            "by_source": {
                "local": self._stats.local_models_found,
                "cluster": self._stats.cluster_models_found,
                "owc": self._stats.owc_models_found,
            },
            "harnesses_evaluated": [h.value for h in self._scan_config.harness_types],
            "skipped_already_evaluated": self._stats.models_skipped_already_evaluated,
            "skipped_invalid": self._stats.models_skipped_invalid,
            "scan_duration": self._stats.last_scan_duration,
            "timestamp": time.time(),
        }

        try:
            safe_emit_event(DataEventType.MULTI_HARNESS_EVALUATION_QUEUED, payload)
        except Exception as e:
            logger.debug(f"[ComprehensiveModelScan] Failed to emit summary event: {e}")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Check daemon health."""
        details = {
            "enabled": self._scan_config.enabled,
            "scan_count": self._stats.scan_count,
            "models_found": self._stats.models_found,
            "models_queued": self._stats.models_queued,
            "harness_combinations_queued": self._stats.harness_combinations_queued,
            "local_models_found": self._stats.local_models_found,
            "cluster_models_found": self._stats.cluster_models_found,
            "owc_models_found": self._stats.owc_models_found,
            "skipped_already_evaluated": self._stats.models_skipped_already_evaluated,
            "last_scan_time": self._stats.last_scan_time,
            "last_scan_duration": self._stats.last_scan_duration,
            "harness_types": [h.value for h in self._scan_config.harness_types],
            "errors": self._stats.errors[-5:],  # Last 5 errors
        }

        if not self._scan_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status="disabled",
                message="Comprehensive model scan is disabled",
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
            message=f"Queued {self._stats.harness_combinations_queued} combinations from {self._stats.models_found} models",
            details=details,
        )


# ============================================================================
# Singleton Access
# ============================================================================

_daemon_instance: ComprehensiveModelScanDaemon | None = None


def get_comprehensive_model_scan_daemon(
    config: ComprehensiveModelScanConfig | None = None,
) -> ComprehensiveModelScanDaemon:
    """Get the singleton ComprehensiveModelScanDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton daemon instance
    """
    global _daemon_instance

    if _daemon_instance is None:
        _daemon_instance = ComprehensiveModelScanDaemon(config)
    return _daemon_instance


def reset_comprehensive_model_scan_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None
