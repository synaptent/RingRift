"""Data Management Loops for P2P Orchestrator.

December 2025: Background loops for data processing and synchronization.

Loops:
- ModelSyncLoop: Synchronizes models across cluster nodes
- DataAggregationLoop: Aggregates training data from nodes

Usage:
    from scripts.p2p.loops import ModelSyncLoop, DataAggregationLoop

    sync = ModelSyncLoop(
        get_model_versions=lambda: orchestrator.model_versions,
        sync_model=orchestrator.sync_model_to_node,
    )
    await sync.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class ModelSyncConfig:
    """Configuration for model sync loop."""

    check_interval_seconds: float = 120.0  # 2 minutes
    max_sync_operations_per_cycle: int = 5
    sync_timeout_seconds: float = 300.0  # 5 minutes
    priority_configs: list[str] = field(default_factory=lambda: ["hex8_2p", "square8_2p"])

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.max_sync_operations_per_cycle <= 0:
            raise ValueError("max_sync_operations_per_cycle must be > 0")
        if self.sync_timeout_seconds <= 0:
            raise ValueError("sync_timeout_seconds must be > 0")


class ModelSyncLoop(BaseLoop):
    """Background loop that synchronizes models across cluster nodes.

    Ensures all nodes have the latest models for their assigned configurations.
    Prioritizes recently promoted models and high-traffic configurations.
    """

    def __init__(
        self,
        get_model_versions: Callable[[], dict[str, dict[str, str]]],
        get_node_models: Callable[[str], Coroutine[Any, Any, dict[str, str]]],
        sync_model: Callable[[str, str, str], Coroutine[Any, Any, bool]],
        get_active_nodes: Callable[[], list[str]],
        config: ModelSyncConfig | None = None,
    ):
        """Initialize model sync loop.

        Args:
            get_model_versions: Callback returning config -> {version, path, checksum}
            get_node_models: Async callback to get node's model versions
            sync_model: Async callback to sync model to node (node_id, config, source_path)
            get_active_nodes: Callback returning list of active node IDs
            config: Sync configuration
        """
        self.config = config or ModelSyncConfig()
        super().__init__(
            name="model_sync",
            interval=self.config.check_interval_seconds,
        )
        self._get_model_versions = get_model_versions
        self._get_node_models = get_node_models
        self._sync_model = sync_model
        self._get_active_nodes = get_active_nodes
        self._sync_stats = {
            "models_synced": 0,
            "sync_failures": 0,
            "bytes_transferred": 0,
        }

    async def _run_once(self) -> None:
        """Check for and sync outdated models."""
        current_versions = self._get_model_versions()
        if not current_versions:
            return

        active_nodes = self._get_active_nodes()
        if not active_nodes:
            return

        sync_tasks: list[tuple[str, str, str]] = []  # (node_id, config, path)

        # Check each node's model versions
        for node_id in active_nodes:
            try:
                node_models = await self._get_node_models(node_id)

                for config, version_info in current_versions.items():
                    current_version = version_info.get("version", "")
                    node_version = node_models.get(config, "")

                    if node_version != current_version:
                        sync_tasks.append((
                            node_id,
                            config,
                            version_info.get("path", ""),
                        ))

            except Exception as e:
                logger.debug(f"[ModelSync] Failed to check {node_id}: {e}")

        # Sort by priority configs first
        def sync_priority(task: tuple[str, str, str]) -> int:
            _, config, _ = task
            if config in self.config.priority_configs:
                return 0
            return 1

        sync_tasks.sort(key=sync_priority)

        # Execute sync operations
        synced_count = 0
        for node_id, config, path in sync_tasks[:self.config.max_sync_operations_per_cycle]:
            try:
                success = await asyncio.wait_for(
                    self._sync_model(node_id, config, path),
                    timeout=self.config.sync_timeout_seconds,
                )
                if success:
                    synced_count += 1
                    self._sync_stats["models_synced"] += 1
                    logger.info(f"[ModelSync] Synced {config} to {node_id}")
                else:
                    self._sync_stats["sync_failures"] += 1
            except asyncio.TimeoutError:
                logger.warning(f"[ModelSync] Timeout syncing {config} to {node_id}")
                self._sync_stats["sync_failures"] += 1
            except Exception as e:
                logger.warning(f"[ModelSync] Failed to sync {config} to {node_id}: {e}")
                self._sync_stats["sync_failures"] += 1

        if synced_count > 0:
            logger.info(f"[ModelSync] Synced {synced_count} models this cycle")

    def get_sync_stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        return {
            **self._sync_stats,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        sync_stats = self.get_sync_stats()
        total_syncs = sync_stats.get("models_synced", 0)
        failures = sync_stats.get("sync_failures", 0)

        # Calculate success rate
        total_ops = total_syncs + failures
        success_rate = total_syncs / max(1, total_ops)

        if not self.running:
            status = "ERROR"
            message = "Model sync loop not running"
        elif failures > 10 and success_rate < 0.5:
            status = "DEGRADED"
            message = f"High failure rate: {success_rate:.0%}"
        else:
            status = "HEALTHY"
            message = f"Synced {total_syncs} models, {success_rate:.0%} success rate"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "models_synced": total_syncs,
                "sync_failures": failures,
                "success_rate": success_rate,
                "run_count": self.stats.total_runs,
            },
        }


@dataclass
class DataAggregationConfig:
    """Configuration for data aggregation loop."""

    check_interval_seconds: float = 300.0  # 5 minutes
    min_games_to_aggregate: int = 100
    max_nodes_per_cycle: int = 10
    aggregation_timeout_seconds: float = 600.0  # 10 minutes

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.min_games_to_aggregate <= 0:
            raise ValueError("min_games_to_aggregate must be > 0")
        if self.max_nodes_per_cycle <= 0:
            raise ValueError("max_nodes_per_cycle must be > 0")
        if self.aggregation_timeout_seconds <= 0:
            raise ValueError("aggregation_timeout_seconds must be > 0")


class DataAggregationLoop(BaseLoop):
    """Background loop that aggregates training data from cluster nodes.

    Collects selfplay game databases from distributed nodes and consolidates
    them to central storage for training.
    """

    def __init__(
        self,
        get_node_game_counts: Callable[[], dict[str, int]],
        aggregate_from_node: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        config: DataAggregationConfig | None = None,
    ):
        """Initialize data aggregation loop.

        Args:
            get_node_game_counts: Callback returning node_id -> game_count
            aggregate_from_node: Async callback to aggregate data from a node
            config: Aggregation configuration
        """
        self.config = config or DataAggregationConfig()
        super().__init__(
            name="data_aggregation",
            interval=self.config.check_interval_seconds,
        )
        self._get_node_game_counts = get_node_game_counts
        self._aggregate_from_node = aggregate_from_node
        self._aggregation_stats = {
            "total_games_aggregated": 0,
            "total_bytes_transferred": 0,
            "aggregation_failures": 0,
        }

    async def _run_once(self) -> None:
        """Check for nodes with data to aggregate."""
        node_counts = self._get_node_game_counts()
        if not node_counts:
            return

        # Find nodes with enough games to aggregate
        nodes_to_aggregate = [
            (node_id, count)
            for node_id, count in node_counts.items()
            if count >= self.config.min_games_to_aggregate
        ]

        # Sort by game count descending (highest priority first)
        nodes_to_aggregate.sort(key=lambda x: x[1], reverse=True)

        # Aggregate from top nodes
        aggregated_count = 0
        for node_id, game_count in nodes_to_aggregate[:self.config.max_nodes_per_cycle]:
            try:
                result = await asyncio.wait_for(
                    self._aggregate_from_node(node_id),
                    timeout=self.config.aggregation_timeout_seconds,
                )

                games = result.get("games_aggregated", 0)
                bytes_transferred = result.get("bytes_transferred", 0)

                if games > 0:
                    aggregated_count += games
                    self._aggregation_stats["total_games_aggregated"] += games
                    self._aggregation_stats["total_bytes_transferred"] += bytes_transferred
                    logger.info(
                        f"[DataAggregation] Aggregated {games} games from {node_id} "
                        f"({bytes_transferred / (1024**2):.1f} MB)"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"[DataAggregation] Timeout aggregating from {node_id}")
                self._aggregation_stats["aggregation_failures"] += 1
            except Exception as e:
                logger.warning(f"[DataAggregation] Failed to aggregate from {node_id}: {e}")
                self._aggregation_stats["aggregation_failures"] += 1

        if aggregated_count > 0:
            logger.info(f"[DataAggregation] Aggregated {aggregated_count} total games this cycle")

    def get_aggregation_stats(self) -> dict[str, Any]:
        """Get aggregation statistics."""
        return {
            **self._aggregation_stats,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        agg_stats = self.get_aggregation_stats()
        total_games = agg_stats.get("total_games_aggregated", 0)
        failures = agg_stats.get("aggregation_failures", 0)

        # Calculate success rate
        total_ops = (total_games > 0) + failures  # 1 if any games, plus failures
        success_rate = 1.0 if failures == 0 else (1.0 if total_games > 0 else 0.0)

        if not self.running:
            status = "ERROR"
            message = "Data aggregation loop not running"
        elif failures > 5:
            status = "DEGRADED"
            message = f"High failure rate: {failures} failures"
        else:
            status = "HEALTHY"
            message = f"Aggregated {total_games} games"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "total_games_aggregated": total_games,
                "aggregation_failures": failures,
                "bytes_transferred": agg_stats.get("total_bytes_transferred", 0),
                "run_count": self.stats.total_runs,
            },
        }


@dataclass
class DataManagementConfig:
    """Configuration for data management loop.

    December 2025: Extracted from inline _data_management_loop.
    """

    # Timing
    interval_seconds: float = 300.0  # 5 minutes
    initial_delay_seconds: float = 30.0  # Wait before first run

    # Disk thresholds
    disk_warning_percent: float = 80.0
    disk_critical_percent: float = 85.0

    # Export thresholds
    db_export_threshold_mb: float = 100.0
    max_concurrent_exports: int = 1  # Reduced from 3: concurrent exports cause I/O contention with P2P
    export_timeout_seconds: float = 3600.0  # 1 hour

    # Training thresholds
    auto_training_threshold_mb: float = 500.0

    # Integrity check frequency (every N cycles)
    db_integrity_check_frequency: int = 6  # ~30 minutes at 5-min interval

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.disk_warning_percent >= self.disk_critical_percent:
            raise ValueError("disk_warning_percent must be < disk_critical_percent")
        if self.max_concurrent_exports <= 0:
            raise ValueError("max_concurrent_exports must be > 0")


class DataManagementLoop(BaseLoop):
    """Background loop for automatic data management.

    December 2025: Extracted from inline _data_management_loop in p2p_orchestrator.py.

    This loop handles:
    - Local operations (runs on ALL nodes):
      - Disk usage check and cleanup
      - JSONL→DB conversion
      - JSONL→NPZ conversion
    - Leader operations (runs on LEADER only):
      - Database integrity check
      - Trigger exports when databases exceed threshold
      - Auto-trigger training when enough data
      - Request data sync from peers
    """

    def __init__(
        self,
        # Role check
        is_leader: Callable[[], bool],
        # Disk operations
        check_disk_capacity: Callable[[float], tuple[bool, float]],  # threshold -> (has_capacity, percent)
        cleanup_disk: Callable[[], Coroutine[Any, Any, None]],
        # Conversion operations
        convert_jsonl_to_db: Callable[[Path, Path], Coroutine[Any, Any, int]],  # data_dir, games_dir -> count
        convert_jsonl_to_npz: Callable[[Path, Path], Coroutine[Any, Any, int]],  # data_dir, training_dir -> count
        # Leader operations
        check_db_integrity: Callable[[Path], Coroutine[Any, Any, dict[str, int]]] | None = None,
        trigger_export: Callable[[Path, Path, str], Coroutine[Any, Any, bool]] | None = None,  # db, output, board_type
        start_training: Callable[[Path], Coroutine[Any, Any, None]] | None = None,  # npz_path
        request_peer_data: Callable[[], Coroutine[Any, Any, None]] | None = None,
        # Directory providers
        get_data_dir: Callable[[], Path] | None = None,
        get_games_dir: Callable[[], Path] | None = None,
        get_training_dir: Callable[[], Path] | None = None,
        # GPU check for training
        is_gpu_node: Callable[[], bool] | None = None,
        has_training_jobs: Callable[[], bool] | None = None,
        # Configuration
        config: DataManagementConfig | None = None,
    ):
        """Initialize data management loop.

        Args:
            is_leader: Callback returning True if this node is leader
            check_disk_capacity: Callback to check disk capacity at threshold
            cleanup_disk: Async callback to cleanup disk space
            convert_jsonl_to_db: Async callback for JSONL→DB conversion
            convert_jsonl_to_npz: Async callback for JSONL→NPZ conversion
            check_db_integrity: Optional async callback to check DB integrity
            trigger_export: Optional async callback to trigger export job
            start_training: Optional async callback to start training
            request_peer_data: Optional async callback to request data from peers
            get_data_dir: Callback returning data directory path
            get_games_dir: Callback returning games directory path
            get_training_dir: Callback returning training directory path
            is_gpu_node: Callback returning True if this is a GPU node
            has_training_jobs: Callback returning True if training is running
            config: Loop configuration
        """
        self.config = config or DataManagementConfig()
        super().__init__(
            name="data_management",
            interval=self.config.interval_seconds,
        )

        # Required callbacks
        self._is_leader = is_leader
        self._check_disk_capacity = check_disk_capacity
        self._cleanup_disk = cleanup_disk
        self._convert_jsonl_to_db = convert_jsonl_to_db
        self._convert_jsonl_to_npz = convert_jsonl_to_npz

        # Optional leader callbacks
        self._check_db_integrity = check_db_integrity
        self._trigger_export = trigger_export
        self._start_training = start_training
        self._request_peer_data = request_peer_data

        # Directory providers (default to current dir if not provided)
        self._get_data_dir = get_data_dir or (lambda: Path("data"))
        self._get_games_dir = get_games_dir or (lambda: Path("data/games"))
        self._get_training_dir = get_training_dir or (lambda: Path("data/training"))

        # Optional node info
        self._is_gpu_node = is_gpu_node or (lambda: False)
        self._has_training_jobs = has_training_jobs or (lambda: False)

        # Track state
        self._active_exports: dict[str, float] = {}  # path -> start_time
        self._db_integrity_counter = 0

        # Statistics (renamed from _stats to avoid conflict with BaseLoop._stats)
        self._data_stats = {
            "disk_cleanups": 0,
            "jsonl_to_db_conversions": 0,
            "jsonl_to_npz_conversions": 0,
            "exports_triggered": 0,
            "training_triggered": 0,
            "integrity_checks": 0,
        }

    async def _on_start(self) -> None:
        """Wait for initial delay before starting loop."""
        if self.config.initial_delay_seconds > 0:
            logger.info(
                f"[{self.name}] Starting (first run in {self.config.initial_delay_seconds}s)"
            )
            await asyncio.sleep(self.config.initial_delay_seconds)

    async def _run_once(self) -> None:
        """Execute one cycle of data management."""
        # === LOCAL NODE OPERATIONS (run on ALL nodes) ===

        # 1. Check disk usage and cleanup if needed
        has_capacity, disk_pct = self._check_disk_capacity(self.config.disk_warning_percent)
        if not has_capacity:
            logger.info(
                f"[{self.name}] Disk at {disk_pct:.1f}% "
                f"(warning: {self.config.disk_warning_percent}%), triggering cleanup"
            )
            try:
                await self._cleanup_disk()
                self._data_stats["disk_cleanups"] += 1
            except Exception as e:
                logger.debug(f"[{self.name}] Cleanup error: {e}")
            # Re-check after cleanup
            has_capacity, disk_pct = self._check_disk_capacity(self.config.disk_critical_percent)

        # 2. Ensure directories exist
        data_dir = self._get_data_dir()
        games_dir = self._get_games_dir()
        training_dir = self._get_training_dir()
        games_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        # 3. JSONL→DB conversion
        # Jan 28, 2026: Uses data_pipeline_manager directly
        try:
            converted = await self.data_pipeline_manager.convert_jsonl_to_db(data_dir, games_dir)
            if converted > 0:
                self._data_stats["jsonl_to_db_conversions"] += converted
                logger.debug(f"[{self.name}] JSONL→DB: {converted} games converted")
        except Exception as e:
            logger.debug(f"[{self.name}] JSONL→DB error: {e}")

        # 4. JSONL→NPZ conversion
        try:
            npz_created = await self._convert_jsonl_to_npz(data_dir, training_dir)
            if npz_created > 0:
                self._data_stats["jsonl_to_npz_conversions"] += npz_created
                logger.debug(f"[{self.name}] JSONL→NPZ: {npz_created} files created")
        except Exception as e:
            logger.debug(f"[{self.name}] JSONL→NPZ error: {e}")

        # === LEADER-ONLY OPERATIONS ===
        if not self._is_leader():
            return

        logger.debug(f"[{self.name}] Running leader operations...")

        # Skip if disk still critical after cleanup
        if not has_capacity:
            logger.info(f"[{self.name}] Disk at {disk_pct:.1f}%, skipping leader operations")
            return

        # 5. Database integrity check (every N cycles)
        self._db_integrity_counter += 1
        if (
            self._check_db_integrity is not None
            and self._db_integrity_counter % self.config.db_integrity_check_frequency == 0
        ):
            try:
                result = await self._check_db_integrity(games_dir)
                self._data_stats["integrity_checks"] += 1
                if result.get("corrupted", 0) > 0:
                    logger.info(
                        f"[{self.name}] DB integrity: {result.get('checked', 0)} checked, "
                        f"{result.get('corrupted', 0)} corrupted"
                    )
            except Exception as e:
                logger.debug(f"[{self.name}] DB integrity check error: {e}")

        # 6. Check databases and trigger exports
        await self._check_and_trigger_exports(games_dir, training_dir)

        # 7. Calculate training data and auto-trigger training
        await self._check_and_trigger_training(training_dir)

        # 8. Request data from peers
        if self._request_peer_data is not None:
            try:
                await self._request_peer_data()
            except Exception as e:
                logger.debug(f"[{self.name}] Peer data request error: {e}")

    async def _check_and_trigger_exports(self, games_dir: Path, training_dir: Path) -> None:
        """Check database sizes and trigger exports if thresholds exceeded."""
        if self._trigger_export is None:
            return

        # Clean up stale export tracking
        now = time.time()
        self._active_exports = {
            p: t for p, t in self._active_exports.items()
            if now - t < self.config.export_timeout_seconds
        }

        current_exports = len(self._active_exports)
        if current_exports >= self.config.max_concurrent_exports:
            return

        if not games_dir.exists():
            return

        for db_file in games_dir.glob("*.db"):
            if current_exports >= self.config.max_concurrent_exports:
                break

            export_key = str(db_file)
            if export_key in self._active_exports:
                continue

            try:
                db_size_mb = db_file.stat().st_size / (1024 * 1024)
            except OSError:
                continue

            if db_size_mb < self.config.db_export_threshold_mb:
                continue

            # Determine board type from filename
            board_type = self._infer_board_type(db_file.name)

            # Trigger export
            output_path = training_dir / f"auto_{db_file.stem}_{int(time.time())}.npz"
            try:
                success = await self._trigger_export(db_file, output_path, board_type)
                if success:
                    self._active_exports[export_key] = time.time()
                    self._data_stats["exports_triggered"] += 1
                    current_exports += 1
                    logger.info(f"[{self.name}] Triggered export for {db_file.name}")
            except Exception as e:
                logger.debug(f"[{self.name}] Export trigger error for {db_file.name}: {e}")

    async def _check_and_trigger_training(self, training_dir: Path) -> None:
        """Check training data size and auto-trigger training if threshold exceeded."""
        if self._start_training is None:
            return

        if not self._is_gpu_node():
            return

        if self._has_training_jobs():
            return

        if not training_dir.exists():
            return

        # Calculate total training data
        total_mb = 0.0
        largest_npz: Path | None = None
        largest_size = 0

        for npz_file in training_dir.glob("*.npz"):
            try:
                size = npz_file.stat().st_size
                total_mb += size / (1024 * 1024)
                if size > largest_size:
                    largest_size = size
                    largest_npz = npz_file
            except OSError:
                continue

        logger.debug(f"[{self.name}] Training data: {total_mb:.1f}MB")

        if total_mb >= self.config.auto_training_threshold_mb and largest_npz:
            logger.info(f"[{self.name}] Auto-triggering training ({total_mb:.1f}MB available)")
            try:
                await self._start_training(largest_npz)
                self._data_stats["training_triggered"] += 1
            except Exception as e:
                logger.debug(f"[{self.name}] Training trigger error: {e}")

    @staticmethod
    def _infer_board_type(filename: str) -> str:
        """Infer board type from database filename."""
        name_lower = filename.lower()
        if "hex" in name_lower:
            return "hexagonal"
        elif "square19" in name_lower or "sq19" in name_lower:
            return "square19"
        return "square8"

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status with statistics."""
        status = super().get_status()
        status["data_management_stats"] = {
            **self._data_stats,
            "active_exports": len(self._active_exports),
            "integrity_check_counter": self._db_integrity_counter,
        }
        return status

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        cleanups = self._data_stats.get("disk_cleanups", 0)
        db_converts = self._data_stats.get("db_conversions", 0)
        npz_converts = self._data_stats.get("npz_conversions", 0)
        export_triggers = self._data_stats.get("export_triggers", 0)
        training_triggers = self._data_stats.get("training_triggers", 0)

        if not self.running:
            status = "ERROR"
            message = "Data management loop not running"
        else:
            status = "HEALTHY"
            message = (
                f"Cleanups: {cleanups}, DB conversions: {db_converts}, "
                f"NPZ conversions: {npz_converts}"
            )

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "is_leader": self._is_leader(),
                "disk_cleanups": cleanups,
                "db_conversions": db_converts,
                "npz_conversions": npz_converts,
                "export_triggers": export_triggers,
                "training_triggers": training_triggers,
                "active_exports": len(self._active_exports),
                "run_count": self.stats.total_runs,
            },
        }


# =============================================================================
# Model Fetch Loop (December 2025)
# =============================================================================


@dataclass
class ModelFetchConfig:
    """Configuration for model fetch loop.

    December 2025: Fetch models from training nodes to coordinator.
    This is the reverse of ModelSyncLoop - it pulls models FROM remote
    training nodes TO the coordinator for evaluation/promotion.
    """

    check_interval_seconds: float = 60.0  # Check every minute
    fetch_timeout_seconds: float = 180.0  # 3 minutes per fetch
    max_fetch_retries: int = 3
    retry_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.fetch_timeout_seconds <= 0:
            raise ValueError("fetch_timeout_seconds must be > 0")


class ModelFetchLoop(BaseLoop):
    """Background loop that fetches trained models from remote nodes.

    December 2025: Critical fix for model distribution. Training nodes
    (nebius-h100-*, lambda-gh200-*) produce models that need to be
    fetched to the coordinator BEFORE gauntlet evaluation can run.

    This loop acts as a safety net - if handle_training_job_completion()
    fails to fetch for any reason, this loop will retry.
    """

    def __init__(
        self,
        is_leader: Callable[[], bool],
        get_completed_training_jobs: Callable[[], list[Any]],
        fetch_model: Callable[[Any], Coroutine[Any, Any, bool]],
        mark_model_fetched: Callable[[str], None],
        is_model_fetched: Callable[[str], bool],
        config: ModelFetchConfig | None = None,
    ):
        """Initialize model fetch loop.

        Args:
            is_leader: Callback returning True if this node is leader
            get_completed_training_jobs: Callback returning completed training jobs
            fetch_model: Async callback to fetch model (takes job, returns success)
            mark_model_fetched: Callback to mark a job's model as fetched
            is_model_fetched: Callback to check if job's model is already fetched
            config: Loop configuration
        """
        self.config = config or ModelFetchConfig()
        super().__init__(
            name="model_fetch",
            interval=self.config.check_interval_seconds,
        )
        self._is_leader = is_leader
        self._get_completed_training_jobs = get_completed_training_jobs
        self._fetch_model = fetch_model
        self._mark_model_fetched = mark_model_fetched
        self._is_model_fetched = is_model_fetched
        self._fetch_stats = {
            "models_fetched": 0,
            "fetch_failures": 0,
            "retry_count": 0,
        }
        # Track retry counts per job
        self._job_retries: dict[str, int] = {}

    async def _run_once(self) -> None:
        """Check for completed training jobs and fetch unfetched models."""
        # Only run on coordinator (leader)
        if not self._is_leader():
            return

        try:
            completed_jobs = self._get_completed_training_jobs()
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"[{self.name}] Failed to get completed jobs: {e}")
            return

        if not completed_jobs:
            return

        for job in completed_jobs:
            job_id = getattr(job, "job_id", None)
            if not job_id:
                continue

            # Skip if already fetched
            if self._is_model_fetched(job_id):
                continue

            # Check retry limit
            retries = self._job_retries.get(job_id, 0)
            if retries >= self.config.max_fetch_retries:
                logger.debug(
                    f"[{self.name}] Max retries ({retries}) reached for {job_id}"
                )
                continue

            # Skip if no output model path
            if not getattr(job, "output_model_path", None):
                continue

            # Skip if no worker node
            if not getattr(job, "worker_node", None):
                continue

            logger.info(
                f"[{self.name}] Fetching model for {job_id} from {job.worker_node} "
                f"(attempt {retries + 1}/{self.config.max_fetch_retries})"
            )

            try:
                success = await asyncio.wait_for(
                    self._fetch_model(job),
                    timeout=self.config.fetch_timeout_seconds,
                )

                if success:
                    self._mark_model_fetched(job_id)
                    self._fetch_stats["models_fetched"] += 1
                    # Clear retry count on success
                    self._job_retries.pop(job_id, None)
                    logger.info(f"[{self.name}] Successfully fetched model for {job_id}")
                else:
                    self._job_retries[job_id] = retries + 1
                    self._fetch_stats["fetch_failures"] += 1
                    self._fetch_stats["retry_count"] += 1

            except asyncio.TimeoutError:
                self._job_retries[job_id] = retries + 1
                self._fetch_stats["fetch_failures"] += 1
                logger.warning(f"[{self.name}] Timeout fetching model for {job_id}")

            except (OSError, RuntimeError) as e:
                self._job_retries[job_id] = retries + 1
                self._fetch_stats["fetch_failures"] += 1
                logger.debug(f"[{self.name}] Failed to fetch model for {job_id}: {e}")

            # Small delay between fetches to avoid overwhelming network
            await asyncio.sleep(5.0)

    def get_status(self) -> dict[str, Any]:
        """Get extended loop status with fetch statistics."""
        status = super().get_status()
        status["model_fetch_stats"] = {
            **self._fetch_stats,
            "pending_retries": len(self._job_retries),
        }
        return status

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        models_fetched = self._fetch_stats.get("models_fetched", 0)
        failures = self._fetch_stats.get("fetch_failures", 0)
        pending_retries = len(self._job_retries)

        # Calculate success rate
        total_ops = models_fetched + failures
        success_rate = models_fetched / max(1, total_ops)

        if not self.running:
            status = "ERROR"
            message = "Model fetch loop not running"
        elif failures > 10 and success_rate < 0.5:
            status = "DEGRADED"
            message = f"High failure rate: {success_rate:.0%}"
        elif pending_retries > 5:
            status = "DEGRADED"
            message = f"{pending_retries} models pending retry"
        else:
            status = "HEALTHY"
            message = f"Fetched {models_fetched} models, {success_rate:.0%} success rate"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "is_leader": self._is_leader(),
                "models_fetched": models_fetched,
                "fetch_failures": failures,
                "pending_retries": pending_retries,
                "success_rate": success_rate,
                "run_count": self.stats.total_runs,
            },
        }


__all__ = [
    "DataAggregationConfig",
    "DataAggregationLoop",
    "DataManagementConfig",
    "DataManagementLoop",
    "ModelFetchConfig",
    "ModelFetchLoop",
    "ModelSyncConfig",
    "ModelSyncLoop",
]
