"""
Model Lifecycle Manager for RingRift AI.

Integrates model registry, training pipeline, P2P orchestrator, and evaluation
systems into a cohesive lifecycle management system.

Components:
- LifecycleConfig: Configuration for lifecycle management
- ModelLifecycleManager: Main lifecycle management orchestration
- PromotionGate: Evaluation criteria for model promotion
- ModelSyncCoordinator: Distributed model synchronization
- TrainingTrigger: Automatic training triggering based on data
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    from aiohttp import ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    ClientSession = None

logger = logging.getLogger(__name__)


# Import canonical config values
try:
    from app.config.unified_config import (
        get_min_elo_improvement,
        get_training_threshold,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False
    def get_min_elo_improvement():
        return 25.0  # Fallback default
    def get_training_threshold():
        return 500  # Fallback default

# Import centralized thresholds (single source of truth)
try:
    from app.config.thresholds import (
        ELO_DROP_ROLLBACK,
        ELO_IMPROVEMENT_PROMOTE,
        MIN_GAMES_PROMOTE,
        MIN_GAMES_REGRESSION,
        MIN_WIN_RATE_PROMOTE,
        TRAINING_STALENESS_HOURS,
        TRAINING_TRIGGER_GAMES,
    )
    HAS_THRESHOLDS = True
except ImportError:
    HAS_THRESHOLDS = False
    # Fallback defaults matching thresholds.py
    ELO_IMPROVEMENT_PROMOTE = 20
    MIN_GAMES_PROMOTE = 100
    MIN_WIN_RATE_PROMOTE = 0.45
    TRAINING_TRIGGER_GAMES = 500
    TRAINING_STALENESS_HOURS = 6.0
    ELO_DROP_ROLLBACK = 50
    MIN_GAMES_REGRESSION = 50

# Import PromotionCriteria for unified thresholds
try:
    from app.training.promotion_controller import PromotionCriteria as UnifiedCriteria
    HAS_PROMOTION_CRITERIA = True
except ImportError:
    HAS_PROMOTION_CRITERIA = False
    UnifiedCriteria = None

# Import unified signals for cross-system consistency
try:
    from app.training.unified_signals import (
        TrainingUrgency,
        get_signal_computer,
    )
    HAS_UNIFIED_SIGNALS = True
except ImportError:
    HAS_UNIFIED_SIGNALS = False
    get_signal_computer = None
    TrainingUrgency = None


# ============================================
# Configuration
# ============================================

@dataclass
class LifecycleConfig:
    """Configuration for model lifecycle management.

    Defaults are sourced from app.config.thresholds (single source of truth).
    Override at instantiation if needed.
    """
    # Registry paths
    registry_dir: str = "data/model_registry"
    model_storage_dir: str = "data/models"

    # Promotion criteria - from thresholds.py
    min_elo_improvement: float = field(default_factory=lambda: float(ELO_IMPROVEMENT_PROMOTE))
    min_games_for_staging: int = field(default_factory=lambda: MIN_GAMES_REGRESSION)
    min_games_for_production: int = field(default_factory=lambda: MIN_GAMES_PROMOTE * 2)
    min_win_rate_vs_production: float = field(default_factory=lambda: MIN_WIN_RATE_PROMOTE)
    max_value_mse_degradation: float = 0.05

    # Training triggers - from thresholds.py
    min_games_for_training: int = field(default_factory=lambda: TRAINING_TRIGGER_GAMES)
    training_data_staleness_hours: float = field(default_factory=lambda: TRAINING_STALENESS_HOURS)
    auto_train_on_data_threshold: bool = True

    # Sync settings
    sync_interval_seconds: float = 300.0
    p2p_api_base: str = "http://localhost:8770"
    sync_timeout_seconds: float = 60.0

    # Rollback settings - from thresholds.py
    auto_rollback_on_regression: bool = True
    regression_threshold_elo: float = field(default_factory=lambda: -float(ELO_DROP_ROLLBACK))

    # Calibration settings
    recalibrate_on_promotion: bool = True
    calibration_games_required: int = field(default_factory=lambda: MIN_GAMES_PROMOTE)

    # Lifecycle hooks
    on_promotion_webhook: str | None = None
    on_training_webhook: str | None = None


class LifecycleStage(Enum):
    """Extended lifecycle stages with intermediate states."""
    TRAINING = "training"           # Currently being trained
    CALIBRATING = "calibrating"     # Value head calibration
    EVALUATING = "evaluating"       # Under evaluation
    STAGING = "staging"             # Passed initial evaluation
    PRODUCTION = "production"       # Active production model
    ROLLBACK_CANDIDATE = "rollback" # Marked for potential rollback
    ARCHIVED = "archived"           # Retired model
    REJECTED = "rejected"           # Failed evaluation


class PromotionDecision(Enum):
    """Result of promotion evaluation."""
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"
    ROLLBACK = "rollback"


# ============================================
# Promotion Gate
# ============================================

@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    model_id: str
    version: int
    elo: float | None = None
    elo_uncertainty: float | None = None
    games_played: int = 0
    win_rate: float | None = None
    draw_rate: float | None = None
    value_mse: float | None = None
    policy_accuracy: float | None = None

    # Comparison vs production
    elo_vs_production: float | None = None
    win_rate_vs_production: float | None = None
    games_vs_production: int = 0

    # Additional metrics
    avg_game_length: float | None = None
    calibration_error: float | None = None
    inference_time_ms: float | None = None


class PromotionGate:
    """
    Evaluates whether a model should be promoted.

    Implements multi-criteria decision making for model promotion.
    Uses PromotionCriteria from promotion_controller.py for unified thresholds
    when available, falling back to LifecycleConfig values.
    """

    def __init__(
        self,
        config: LifecycleConfig,
        criteria: Optional["UnifiedCriteria"] = None
    ):
        self.config = config
        self._evaluation_history: dict[str, list[EvaluationResult]] = {}

        # Use provided criteria or load from PromotionCriteria for unified thresholds
        if criteria is not None:
            self._criteria = criteria
        elif HAS_PROMOTION_CRITERIA and UnifiedCriteria is not None:
            self._criteria = UnifiedCriteria(
                min_elo_improvement=config.min_elo_improvement,
                min_games_played=config.min_games_for_production,
                min_win_rate=config.min_win_rate_vs_production,
            )
        else:
            self._criteria = None

    @property
    def min_elo_improvement(self) -> float:
        """Get unified Elo improvement threshold."""
        if self._criteria is not None:
            return self._criteria.min_elo_improvement
        return self.config.min_elo_improvement

    @property
    def min_games_for_production(self) -> int:
        """Get unified games threshold for production."""
        if self._criteria is not None:
            return self._criteria.min_games_played
        return self.config.min_games_for_production

    @property
    def min_win_rate(self) -> float:
        """Get unified win rate threshold."""
        if self._criteria is not None:
            return self._criteria.min_win_rate
        return self.config.min_win_rate_vs_production

    def evaluate_for_staging(self, result: EvaluationResult) -> tuple[PromotionDecision, str]:
        """
        Evaluate if model should be promoted to staging.

        Returns: (decision, reason)
        """
        key = f"{result.model_id}:{result.version}"

        # Store in history
        if key not in self._evaluation_history:
            self._evaluation_history[key] = []
        self._evaluation_history[key].append(result)

        # Check minimum games
        if result.games_played < self.config.min_games_for_staging:
            return (
                PromotionDecision.HOLD,
                f"Insufficient games: {result.games_played}/{self.config.min_games_for_staging}"
            )

        # Check if Elo is available
        if result.elo is None:
            return (PromotionDecision.HOLD, "No Elo rating available")

        # Check for obvious regression (much worse than baseline)
        if result.elo < 1400:  # Minimum viable Elo threshold
            return (
                PromotionDecision.REJECT,
                f"Elo too low: {result.elo:.0f} < 1400"
            )

        # Check policy accuracy if available
        if result.policy_accuracy is not None and result.policy_accuracy < 0.3:
            return (
                PromotionDecision.REJECT,
                f"Policy accuracy too low: {result.policy_accuracy:.2%}"
            )

        return (PromotionDecision.PROMOTE, "Meets staging criteria")

    def evaluate_for_production(
        self,
        result: EvaluationResult,
        production_result: EvaluationResult | None = None
    ) -> tuple[PromotionDecision, str]:
        """
        Evaluate if model should be promoted to production.

        Returns: (decision, reason)
        """
        # Check minimum games (use unified property)
        min_games = self.min_games_for_production
        if result.games_played < min_games:
            return (
                PromotionDecision.HOLD,
                f"Insufficient games: {result.games_played}/{min_games}"
            )

        # If no production model, promote if staging passed
        if production_result is None:
            return (PromotionDecision.PROMOTE, "No current production model")

        # Check head-to-head results
        if result.games_vs_production < 50:
            return (
                PromotionDecision.HOLD,
                f"Insufficient head-to-head games: {result.games_vs_production}/50"
            )

        # Check Elo improvement (use unified property)
        min_elo = self.min_elo_improvement
        if result.elo_vs_production is not None and result.elo_vs_production < min_elo:
            if result.elo_vs_production < self.config.regression_threshold_elo:
                return (
                    PromotionDecision.REJECT,
                    f"Significant regression: {result.elo_vs_production:+.0f} Elo"
                )
            return (
                PromotionDecision.HOLD,
                f"Insufficient Elo improvement: {result.elo_vs_production:+.0f} < {min_elo}"
            )

        # Check win rate (use unified property)
        min_wr = self.min_win_rate
        if result.win_rate_vs_production is not None and result.win_rate_vs_production < min_wr:
            return (
                PromotionDecision.HOLD,
                f"Win rate too low: {result.win_rate_vs_production:.1%} < {min_wr:.1%}"
            )

        # Check for value MSE regression
        if (result.value_mse is not None and
            production_result.value_mse is not None):
            mse_change = result.value_mse - production_result.value_mse
            if mse_change > self.config.max_value_mse_degradation:
                return (
                    PromotionDecision.HOLD,
                    f"Value MSE regression: +{mse_change:.4f}"
                )

        # All checks passed
        improvement = result.elo_vs_production if result.elo_vs_production else 0
        return (
            PromotionDecision.PROMOTE,
            f"Elo improved by {improvement:+.0f}"
        )

    def check_rollback(
        self,
        current: EvaluationResult,
        previous: EvaluationResult
    ) -> tuple[bool, str]:
        """
        Check if we should rollback to previous model.

        Returns: (should_rollback, reason)
        """
        if not self.config.auto_rollback_on_regression:
            return (False, "Auto-rollback disabled")

        # Check Elo regression
        if current.elo is not None and previous.elo is not None:
            elo_diff = current.elo - previous.elo
            if elo_diff < self.config.regression_threshold_elo:
                return (True, f"Elo regression: {elo_diff:+.0f}")

        # Check win rate collapse
        if (current.win_rate is not None and
            previous.win_rate is not None and
            previous.win_rate > 0.45):  # Only if previous was decent
            win_rate_diff = current.win_rate - previous.win_rate
            if win_rate_diff < -0.15:  # More than 15% drop
                return (True, f"Win rate collapsed: {win_rate_diff:+.1%}")

        return (False, "No rollback needed")


# ============================================
# Training Trigger
# ============================================

@dataclass
class TrainingConditions:
    """Conditions that can trigger training."""
    new_games_count: int = 0
    hours_since_last_training: float = 0.0
    data_quality_score: float = 1.0
    elo_plateau_detected: bool = False
    curriculum_stage_ready: bool = False
    manual_trigger: bool = False


class TrainingTrigger:
    """
    Determines when to trigger new training runs.

    Monitors data accumulation, staleness, and quality.
    Uses UnifiedSignalComputer for base decision logic.
    """

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._last_training_time: datetime | None = None
        self._last_games_count: int = 0
        # Use unified signal computer if available
        self._signal_computer = get_signal_computer() if HAS_UNIFIED_SIGNALS else None

    def should_trigger_training(
        self,
        conditions: TrainingConditions,
        config_key: str = "",
        current_elo: float = 1500.0,
    ) -> tuple[bool, str]:
        """
        Determine if training should be triggered.

        Uses UnifiedSignalComputer for base decision, with additional
        lifecycle-specific checks.

        Returns: (should_train, reason)
        """
        # Always allow manual triggers
        if conditions.manual_trigger:
            return (True, "Manual training trigger")

        # Check if auto-training is enabled
        if not self.config.auto_train_on_data_threshold:
            return (False, "Auto-training disabled")

        # Check data quality (lifecycle-specific check not in unified signals)
        if conditions.data_quality_score < 0.5:
            return (False, f"Data quality too low: {conditions.data_quality_score:.2f}")

        # Check curriculum stage readiness (lifecycle-specific)
        if conditions.curriculum_stage_ready:
            return (True, "Curriculum stage ready for advancement")

        # Use unified signals if available
        if self._signal_computer is not None:
            signals = self._signal_computer.compute_signals(
                current_games=conditions.new_games_count + self._last_games_count,
                current_elo=current_elo,
                config_key=config_key,
            )

            # Override with Elo plateau check
            if conditions.elo_plateau_detected and conditions.new_games_count >= 100:
                return (True, "Elo plateau detected, trying new training")

            if signals.should_train:
                return (True, signals.reason)
            else:
                return (False, signals.reason)

        # Fallback to legacy logic if unified signals not available
        # Check if enough new games
        if conditions.new_games_count >= self.config.min_games_for_training:
            return (True, f"Sufficient new games: {conditions.new_games_count}")

        # Check staleness
        if conditions.hours_since_last_training >= self.config.training_data_staleness_hours:
            if conditions.new_games_count >= self.config.min_games_for_training // 2:
                return (
                    True,
                    f"Data staleness ({conditions.hours_since_last_training:.1f}h) with {conditions.new_games_count} games"
                )

        # Check for Elo plateau
        if conditions.elo_plateau_detected and conditions.new_games_count >= 100:
            return (True, "Elo plateau detected, trying new training")

        return (False, f"Waiting for more data: {conditions.new_games_count}/{self.config.min_games_for_training}")

    def record_training_started(self, games_count: int, config_key: str = "") -> None:
        """Record that training has started."""
        self._last_training_time = datetime.now()
        self._last_games_count = games_count
        if self._signal_computer:
            self._signal_computer.record_training_started(games_count, config_key)

    def record_training_completed(
        self,
        new_elo: float | None = None,
        config_key: str = "",
        model_id: str | None = None,
        auto_trigger_evaluation: bool = True,
    ) -> None:
        """Record that training has completed and optionally trigger evaluation.

        Args:
            new_elo: New Elo rating after training
            config_key: Configuration key (e.g., "square8_2p")
            model_id: ID of the newly trained model
            auto_trigger_evaluation: If True, automatically trigger evaluation (Dec 2025)
        """
        if self._signal_computer:
            self._signal_computer.record_training_completed(new_elo, config_key)

        # Auto-trigger evaluation for the new model (December 2025)
        if auto_trigger_evaluation and model_id:
            self._trigger_evaluation_for_model(model_id, config_key)

    def _trigger_evaluation_for_model(self, model_id: str, config_key: str = "") -> bool:
        """Trigger evaluation for a newly trained model.

        Adds evaluation work to the work queue so that the new model
        gets evaluated against baselines automatically.

        Args:
            model_id: ID of the model to evaluate
            config_key: Configuration key for context

        Returns:
            True if evaluation was triggered successfully
        """
        try:
            # Try work queue approach first
            from app.coordination.work_queue import WorkItem, WorkType, get_work_queue
            queue = get_work_queue()

            # Parse board type from model_id or config_key
            board_type = "square8"
            num_players = 2
            if config_key:
                parts = config_key.replace("_", " ").replace("p", "").split()
                if len(parts) >= 1:
                    board_type = parts[0]
                if len(parts) >= 2 and parts[1].isdigit():
                    num_players = int(parts[1])

            # Create evaluation work item with high priority
            work = WorkItem(
                work_type=WorkType.VALIDATION,
                priority=85,  # High priority for new models
                config={
                    "model_id": model_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "games_per_opponent": 100,
                    "auto_triggered": True,
                },
                timeout_seconds=3600.0,  # 1 hour for evaluation
            )
            queue.add_work(work)
            logger.info(f"Auto-triggered evaluation for model {model_id}")
            return True

        except ImportError:
            logger.debug("WorkQueue not available, trying event bus")

        try:
            # Fallback to event bus
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.EVALUATION_STARTED,
                payload={
                    "model_id": model_id,
                    "config_key": config_key,
                    "auto_triggered": True,
                    "source": "training_complete",
                },
                source="model_lifecycle",
            )

            bus = get_event_bus()
            import asyncio
            try:
                asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                if hasattr(bus, 'publish_sync'):
                    bus.publish_sync(event)

            logger.info(f"Published evaluation event for model {model_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to trigger evaluation for {model_id}: {e}")
            return False

    def get_unified_urgency(self, config_key: str, current_games: int, current_elo: float) -> Optional["TrainingUrgency"]:
        """Get unified training urgency."""
        if self._signal_computer is None:
            return None
        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=current_elo,
            config_key=config_key,
        )
        return signals.urgency

    def record_training(self, games_count: int) -> None:
        """Record that training was performed."""
        self._last_training_time = datetime.now()
        self._last_games_count = games_count

    def get_conditions(
        self,
        current_games: int,
        data_quality: float = 1.0,
        elo_plateau: bool = False,
        curriculum_ready: bool = False
    ) -> TrainingConditions:
        """Get current training conditions."""
        hours_since = 0.0
        if self._last_training_time:
            delta = datetime.now() - self._last_training_time
            hours_since = delta.total_seconds() / 3600

        new_games = max(0, current_games - self._last_games_count)

        return TrainingConditions(
            new_games_count=new_games,
            hours_since_last_training=hours_since,
            data_quality_score=data_quality,
            elo_plateau_detected=elo_plateau,
            curriculum_stage_ready=curriculum_ready
        )


# ============================================
# Model Sync Coordinator
# ============================================

class ModelSyncCoordinator:
    """Coordinates MODEL REGISTRY synchronization across P2P cluster.

    .. note:: Orchestrator Hierarchy (2025-12)
        - **UnifiedTrainingOrchestrator** (unified_orchestrator.py): Step-level
          training operations (forward/backward pass, hot buffer, enhancements)
        - **TrainingOrchestrator** (orchestrated_training.py): Manager lifecycle
          coordination (checkpoint manager, rollback manager, data coordinator)
        - **ModelSyncCoordinator** (this): Model registry sync operations
        - **P2P Coordinators** (p2p_integration.py): P2P cluster REST API wrappers

    Use this class for syncing models across cluster nodes.
    For training operations, use UnifiedTrainingOrchestrator or TrainingOrchestrator.

    Handles:
    - Push new models to cluster nodes (via aria2 or HTTP)
    - Pull latest production model
    - Sync model registry state

    Integrates with distributed SyncCoordinator for optimal transport selection.
    """

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._sync_state: dict[str, Any] = {}
        self._last_sync: dict[str, float] = {}
        self._distributed_coordinator = None

    def _get_distributed_coordinator(self):
        """Lazily get the distributed SyncCoordinator."""
        if self._distributed_coordinator is None:
            try:
                from app.distributed.sync_coordinator import SyncCoordinator
                self._distributed_coordinator = SyncCoordinator.get_instance()
            except ImportError:
                pass
        return self._distributed_coordinator

    async def get_cluster_status(self) -> dict[str, Any]:
        """Get current cluster status from P2P orchestrator."""
        if not HAS_AIOHTTP:
            return {"error": "aiohttp not available"}

        try:
            timeout = ClientTimeout(total=self.config.sync_timeout_seconds)
            async with ClientSession(timeout=timeout) as session, session.get(
                f"{self.config.p2p_api_base}/api/cluster/status"
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"error": str(e)}

    async def push_model(
        self,
        model_path: Path,
        model_id: str,
        version: int,
        target_nodes: list[str] | None = None
    ) -> dict[str, bool]:
        """
        Push model to cluster nodes.

        Uses distributed SyncCoordinator for optimal transport (aria2, SSH, P2P)
        when available, with fallback to direct HTTP push.

        Returns: {node_id: success}
        """
        results = {}

        # Try distributed SyncCoordinator first (has aria2, SSH, P2P support)
        dist_coord = self._get_distributed_coordinator()
        if dist_coord:
            try:
                sync_stats = await dist_coord.sync_models(
                    model_ids=[f"{model_id}_v{version}"],
                )
                if sync_stats.files_synced > 0:
                    logger.info(
                        f"Model {model_id}:v{version} synced via {sync_stats.transport_used} "
                        f"({sync_stats.bytes_transferred / (1024*1024):.1f}MB)"
                    )
                    # Record metrics
                    try:
                        from app.metrics.orchestrator import record_sync_coordinator_op
                        record_sync_coordinator_op(
                            category="models",
                            transport=sync_stats.transport_used,
                            files_synced=sync_stats.files_synced,
                            bytes_transferred=sync_stats.bytes_transferred,
                            duration_seconds=sync_stats.duration_seconds,
                        )
                    except ImportError:
                        pass

                    self._last_sync[model_id] = time.time()
                    return {"distributed_sync": True}
            except Exception as e:
                logger.warning(f"Distributed sync failed, falling back to HTTP: {e}")

        # Fallback to direct HTTP push
        if not HAS_AIOHTTP:
            return {}

        # Get target nodes if not specified
        if target_nodes is None:
            status = await self.get_cluster_status()
            if "nodes" in status:
                # Prioritize GPU nodes for model sync
                target_nodes = [
                    n["node_id"] for n in status["nodes"]
                    if n.get("has_gpu") and n.get("is_alive", False)
                ]

        if not target_nodes:
            logger.warning("No target nodes for model sync")
            return {}

        # Sync to each node
        timeout = ClientTimeout(total=self.config.sync_timeout_seconds)

        for node_id in target_nodes:
            try:
                # Get node endpoint
                node = await self._get_node_endpoint(node_id)
                if not node:
                    results[node_id] = False
                    continue

                # Push model via sync endpoint
                async with ClientSession(timeout=timeout) as session:
                    with open(model_path, 'rb') as f:
                        data = {
                            'model_id': model_id,
                            'version': version,
                            'model_data': f.read()
                        }
                        async with session.post(
                            f"{node}/sync/file",
                            data=data
                        ) as resp:
                            results[node_id] = resp.status == 200

            except Exception as e:
                logger.error(f"Failed to push model to {node_id}: {e}")
                results[node_id] = False

        self._last_sync[model_id] = time.time()
        return results

    async def pull_production_model(
        self,
        dest_path: Path
    ) -> dict[str, Any] | None:
        """
        Pull the current production model from cluster.

        Returns: model metadata or None
        """
        if not HAS_AIOHTTP:
            return None

        try:
            # Get cluster status
            status = await self.get_cluster_status()
            leader = status.get("leader")

            if not leader:
                logger.warning("No cluster leader for model pull")
                return None

            # Get production model info from leader
            timeout = ClientTimeout(total=self.config.sync_timeout_seconds)
            async with ClientSession(timeout=timeout) as session, session.get(
                f"{self.config.p2p_api_base}/api/model/production"
            ) as resp:
                if resp.status != 200:
                    return None

                metadata = await resp.json()
                model_url = metadata.get("file_url")

                if model_url:
                    # Download model file
                    async with session.get(model_url) as file_resp:
                        if file_resp.status == 200:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(dest_path, 'wb') as f:
                                f.write(await file_resp.read())
                            return metadata

                return None

        except Exception as e:
            logger.error(f"Failed to pull production model: {e}")
            return None

    async def broadcast_promotion(
        self,
        model_id: str,
        version: int,
        stage: str,
        reason: str
    ) -> None:
        """Broadcast model promotion to cluster."""
        if not HAS_AIOHTTP:
            return

        try:
            timeout = ClientTimeout(total=10.0)
            async with ClientSession(timeout=timeout) as session:
                await session.post(
                    f"{self.config.p2p_api_base}/api/model/promotion",
                    json={
                        "model_id": model_id,
                        "version": version,
                        "stage": stage,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Failed to broadcast promotion: {e}")

    async def _get_node_endpoint(self, node_id: str) -> str | None:
        """Get endpoint URL for a node."""
        status = await self.get_cluster_status()
        for node in status.get("nodes", []):
            if node["node_id"] == node_id:
                scheme = node.get("scheme", "http")
                host = node.get("host", "")
                port = node.get("port", 8770)
                return f"{scheme}://{host}:{port}"
        return None


# ============================================
# Model Lifecycle Manager
# ============================================

class ModelLifecycleManager:
    """
    Main orchestrator for model lifecycle management.

    Coordinates:
    - Model registration and versioning
    - Promotion decisions and transitions
    - Training triggers
    - Cluster synchronization
    - Rollback handling
    """

    def __init__(self, config: LifecycleConfig | None = None):
        self.config = config or LifecycleConfig()

        # Initialize components
        self.promotion_gate = PromotionGate(self.config)
        self.training_trigger = TrainingTrigger(self.config)
        self.sync_coordinator = ModelSyncCoordinator(self.config)

        # State
        self._lifecycle_state: dict[str, LifecycleStage] = {}
        self._evaluation_queue: list[tuple[str, int]] = []
        self._pending_promotions: list[dict[str, Any]] = []
        self._callbacks: dict[str, list[Callable]] = {}

        # Registry reference (lazy loaded)
        self._registry = None
        self._calibrator = None

        # Background task handles
        self._tasks: list[asyncio.Task] = []
        self._running = False

    @property
    def registry(self):
        """Lazy load model registry."""
        if self._registry is None:
            try:
                from app.training.model_registry import ModelRegistry
                self._registry = ModelRegistry(
                    Path(self.config.registry_dir),
                    Path(self.config.model_storage_dir)
                )
            except ImportError:
                logger.warning("Model registry not available")
        return self._registry

    @property
    def calibrator(self):
        """Lazy load value calibrator."""
        if self._calibrator is None:
            try:
                from app.training.value_calibration import ValueCalibrator
                self._calibrator = ValueCalibrator()
            except ImportError:
                logger.warning("Value calibrator not available")
        return self._calibrator

    async def start(self) -> None:
        """Start lifecycle management background tasks."""
        self._running = True

        # Start background loops
        self._tasks.append(asyncio.create_task(self._promotion_loop()))
        self._tasks.append(asyncio.create_task(self._sync_loop()))
        self._tasks.append(asyncio.create_task(self._training_check_loop()))

        logger.info("Model lifecycle manager started")

    async def stop(self) -> None:
        """Stop lifecycle management."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._tasks.clear()
        logger.info("Model lifecycle manager stopped")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for lifecycle events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _fire_callbacks(self, event: str, **kwargs) -> None:
        """Fire callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(**kwargs)
                else:
                    callback(**kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    # ==========================================
    # Model Registration
    # ==========================================

    async def register_model(
        self,
        name: str,
        model_path: Path,
        training_config: dict[str, Any] | None = None,
        tags: list[str] | None = None
    ) -> tuple[str, int]:
        """
        Register a new model for lifecycle management.

        Returns: (model_id, version)
        """
        if not self.registry:
            raise RuntimeError("Model registry not available")

        from app.training.model_registry import ModelStage, ModelType, TrainingConfig

        # Register with registry
        model_id, version = self.registry.register_model(
            name=name,
            model_path=model_path,
            model_type=ModelType.POLICY_VALUE,
            training_config=TrainingConfig.from_dict(training_config or {}),
            tags=tags or [],
            initial_stage=ModelStage.DEVELOPMENT
        )

        # Set lifecycle state
        key = f"{model_id}:{version}"
        self._lifecycle_state[key] = LifecycleStage.TRAINING

        # Queue for evaluation
        self._evaluation_queue.append((model_id, version))

        await self._fire_callbacks(
            "model_registered",
            model_id=model_id,
            version=version
        )

        logger.info(f"Registered model {model_id}:v{version}")
        return model_id, version

    # ==========================================
    # Evaluation & Promotion
    # ==========================================

    async def submit_evaluation(
        self,
        model_id: str,
        version: int,
        result: EvaluationResult
    ) -> None:
        """Submit evaluation results for a model."""
        key = f"{model_id}:{version}"
        current_stage = self._lifecycle_state.get(key, LifecycleStage.EVALUATING)

        # Update registry metrics
        if self.registry:
            from app.training.model_registry import ModelMetrics
            self.registry.update_metrics(
                model_id, version,
                ModelMetrics(
                    elo=result.elo,
                    elo_uncertainty=result.elo_uncertainty,
                    win_rate=result.win_rate,
                    draw_rate=result.draw_rate,
                    games_played=result.games_played,
                    policy_accuracy=result.policy_accuracy,
                    value_mse=result.value_mse
                )
            )

        # Determine promotion
        if current_stage in (LifecycleStage.TRAINING, LifecycleStage.EVALUATING):
            decision, reason = self.promotion_gate.evaluate_for_staging(result)
            await self._handle_promotion_decision(
                model_id, version, decision, reason,
                LifecycleStage.STAGING
            )

        elif current_stage == LifecycleStage.STAGING:
            # Get production model for comparison
            production_result = await self._get_production_result()
            decision, reason = self.promotion_gate.evaluate_for_production(
                result, production_result
            )
            await self._handle_promotion_decision(
                model_id, version, decision, reason,
                LifecycleStage.PRODUCTION
            )

    async def _handle_promotion_decision(
        self,
        model_id: str,
        version: int,
        decision: PromotionDecision,
        reason: str,
        target_stage: LifecycleStage
    ) -> None:
        """Handle a promotion decision."""
        key = f"{model_id}:{version}"

        if decision == PromotionDecision.PROMOTE:
            # Calibrate value head before production
            if (target_stage == LifecycleStage.PRODUCTION and
                self.config.recalibrate_on_promotion):
                await self._calibrate_model(model_id, version)

            # Update lifecycle state
            self._lifecycle_state[key] = target_stage

            # Update registry stage
            if self.registry:
                from app.training.model_registry import ModelStage
                stage_map = {
                    LifecycleStage.STAGING: ModelStage.STAGING,
                    LifecycleStage.PRODUCTION: ModelStage.PRODUCTION
                }
                if target_stage in stage_map:
                    self.registry.promote(
                        model_id, version,
                        stage_map[target_stage],
                        reason
                    )

            # Sync to cluster if production
            if target_stage == LifecycleStage.PRODUCTION:
                await self._sync_production_model(model_id, version)
                await self.sync_coordinator.broadcast_promotion(
                    model_id, version, target_stage.value, reason
                )

            await self._fire_callbacks(
                "model_promoted",
                model_id=model_id,
                version=version,
                stage=target_stage.value,
                reason=reason
            )

            logger.info(f"Promoted {model_id}:v{version} to {target_stage.value}: {reason}")

        elif decision == PromotionDecision.REJECT:
            self._lifecycle_state[key] = LifecycleStage.REJECTED

            if self.registry:
                from app.training.model_registry import ModelStage
                self.registry.promote(
                    model_id, version,
                    ModelStage.REJECTED,
                    reason
                )

            logger.warning(f"Rejected {model_id}:v{version}: {reason}")

        else:  # HOLD
            logger.debug(f"Holding {model_id}:v{version}: {reason}")

    async def _get_production_result(self) -> EvaluationResult | None:
        """Get evaluation result for current production model."""
        if not self.registry:
            return None

        prod = self.registry.get_production_model()
        if not prod:
            return None

        return EvaluationResult(
            model_id=prod.model_id,
            version=prod.version,
            elo=prod.metrics.elo,
            elo_uncertainty=prod.metrics.elo_uncertainty,
            games_played=prod.metrics.games_played,
            win_rate=prod.metrics.win_rate,
            value_mse=prod.metrics.value_mse,
            policy_accuracy=prod.metrics.policy_accuracy
        )

    async def _calibrate_model(self, model_id: str, version: int) -> None:
        """Calibrate value head for model."""
        if not self.calibrator:
            return

        logger.info(f"Calibrating value head for {model_id}:v{version}")
        # Calibration would be done here using stored validation games
        # This is a placeholder for integration with the calibrator

    async def _sync_production_model(self, model_id: str, version: int) -> None:
        """Sync production model to cluster."""
        if not self.registry:
            return

        model = self.registry.get_model(model_id, version)
        if not model:
            return

        model_path = Path(model.file_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return

        results = await self.sync_coordinator.push_model(
            model_path, model_id, version
        )

        if results.get("distributed_sync"):
            logger.info(
                "Triggered distributed sync for %s:v%s",
                model_id,
                version,
            )
            return

        success_count = sum(1 for v in results.values() if v)
        logger.info(
            f"Synced {model_id}:v{version} to {success_count}/{len(results)} nodes"
        )

    # ==========================================
    # Rollback
    # ==========================================

    async def check_rollback(
        self,
        current_id: str,
        current_version: int,
        previous_id: str,
        previous_version: int
    ) -> bool:
        """
        Check if rollback is needed and execute if so.

        Returns: True if rollback was executed
        """
        current_result = await self._get_evaluation_result(current_id, current_version)
        previous_result = await self._get_evaluation_result(previous_id, previous_version)

        if not current_result or not previous_result:
            return False

        should_rollback, reason = self.promotion_gate.check_rollback(
            current_result, previous_result
        )

        if should_rollback:
            await self.execute_rollback(previous_id, previous_version, reason)
            return True

        return False

    async def execute_rollback(
        self,
        target_id: str,
        target_version: int,
        reason: str
    ) -> None:
        """Execute rollback to a previous model."""
        logger.warning(f"Executing rollback to {target_id}:v{target_version}: {reason}")

        # Mark current production for archival
        if self.registry:
            current_prod = self.registry.get_production_model()
            if current_prod:
                from app.training.model_registry import ModelStage
                self.registry.promote(
                    current_prod.model_id,
                    current_prod.version,
                    ModelStage.ARCHIVED,
                    f"Replaced by rollback to {target_id}:v{target_version}"
                )

        # Promote rollback target to production
        if self.registry:
            from app.training.model_registry import ModelStage
            self.registry.promote(
                target_id, target_version,
                ModelStage.PRODUCTION,
                f"Rollback: {reason}"
            )

        # Sync to cluster
        await self._sync_production_model(target_id, target_version)
        await self.sync_coordinator.broadcast_promotion(
            target_id, target_version, "production", f"Rollback: {reason}"
        )

        await self._fire_callbacks(
            "model_rollback",
            target_id=target_id,
            target_version=target_version,
            reason=reason
        )

    async def _get_evaluation_result(
        self,
        model_id: str,
        version: int
    ) -> EvaluationResult | None:
        """Get evaluation result for a model."""
        if not self.registry:
            return None

        model = self.registry.get_model(model_id, version)
        if not model:
            return None

        return EvaluationResult(
            model_id=model_id,
            version=version,
            elo=model.metrics.elo,
            games_played=model.metrics.games_played,
            win_rate=model.metrics.win_rate,
            value_mse=model.metrics.value_mse
        )

    # ==========================================
    # Training Triggers
    # ==========================================

    async def check_training_trigger(
        self,
        current_games: int,
        data_quality: float = 1.0
    ) -> tuple[bool, str]:
        """
        Check if training should be triggered.

        Returns: (should_train, reason)
        """
        # Check for Elo plateau from metrics
        elo_plateau = await self._detect_elo_plateau()

        conditions = self.training_trigger.get_conditions(
            current_games,
            data_quality,
            elo_plateau
        )

        should_train, reason = self.training_trigger.should_trigger_training(conditions)

        if should_train:
            await self._fire_callbacks(
                "training_triggered",
                reason=reason,
                conditions=asdict(conditions)
            )

        return should_train, reason

    async def _detect_elo_plateau(self) -> bool:
        """Detect if Elo has plateaued."""
        # This would analyze recent Elo history
        # Placeholder for integration with dashboard metrics
        return False

    def record_training_complete(self, games_used: int) -> None:
        """Record that training was completed."""
        self.training_trigger.record_training(games_used)

    # ==========================================
    # Background Loops
    # ==========================================

    async def _promotion_loop(self) -> None:
        """Background loop for processing promotion queue."""
        while self._running:
            try:
                # Process pending promotions
                while self._pending_promotions:
                    self._pending_promotions.pop(0)
                    # Process promotion

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Promotion loop error: {e}")
                await asyncio.sleep(30)

    async def _sync_loop(self) -> None:
        """Background loop for cluster synchronization."""
        while self._running:
            try:
                # Get cluster status
                status = await self.sync_coordinator.get_cluster_status()

                if "error" not in status:
                    # Check for model sync needs
                    pass

                await asyncio.sleep(self.config.sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(60)

    async def _training_check_loop(self) -> None:
        """Background loop for checking training triggers."""
        while self._running:
            try:
                # Get current games count
                # This would integrate with data pipeline

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training check loop error: {e}")
                await asyncio.sleep(60)

    # ==========================================
    # Status & Reporting
    # ==========================================

    def get_status(self) -> dict[str, Any]:
        """Get current lifecycle manager status."""
        return {
            "running": self._running,
            "lifecycle_states": {
                k: v.value for k, v in self._lifecycle_state.items()
            },
            "evaluation_queue_size": len(self._evaluation_queue),
            "pending_promotions": len(self._pending_promotions),
            "config": asdict(self.config)
        }

    def get_model_lifecycle(self, model_id: str, version: int) -> dict[str, Any]:
        """Get lifecycle information for a specific model."""
        key = f"{model_id}:{version}"

        result = {
            "model_id": model_id,
            "version": version,
            "lifecycle_stage": self._lifecycle_state.get(key, LifecycleStage.TRAINING).value,
            "in_evaluation_queue": (model_id, version) in self._evaluation_queue
        }

        if self.registry:
            model = self.registry.get_model(model_id, version)
            if model:
                result["registry_stage"] = model.stage.value
                result["metrics"] = model.metrics.to_dict()
                result["history"] = self.registry.get_stage_history(model_id, version)

        return result


# ============================================
# Integration Functions
# ============================================

async def create_lifecycle_manager(
    config: LifecycleConfig | None = None,
    start: bool = True
) -> ModelLifecycleManager:
    """Create and optionally start a lifecycle manager."""
    manager = ModelLifecycleManager(config)

    if start:
        await manager.start()

    return manager


def integrate_with_p2p(
    manager: ModelLifecycleManager,
    p2p_base_url: str = "http://localhost:8770"
) -> None:
    """Configure lifecycle manager for P2P integration."""
    manager.config.p2p_api_base = p2p_base_url


def integrate_with_pipeline(
    manager: ModelLifecycleManager,
    _pipeline_orchestrator: Any
) -> None:
    """
    Integrate lifecycle manager with pipeline orchestrator.

    Sets up callbacks for training completion and evaluation results.
    """
    # Register for training completion
    manager.register_callback(
        "training_triggered",
        lambda **kw: logger.info(f"Training triggered: {kw.get('reason')}")
    )

    # Register for promotion events
    manager.register_callback(
        "model_promoted",
        lambda **kw: logger.info(
            f"Model promoted: {kw.get('model_id')}:v{kw.get('version')} "
            f"to {kw.get('stage')}"
        )
    )


# ============================================
# Main
# ============================================

async def main():
    """Example usage of model lifecycle manager."""
    # Create config
    config = LifecycleConfig(
        min_elo_improvement=15.0,
        min_games_for_staging=30,
        min_games_for_production=100
    )

    # Create manager
    manager = ModelLifecycleManager(config)

    # Test promotion gate
    gate = PromotionGate(config)

    # Test staging evaluation
    result = EvaluationResult(
        model_id="test_model",
        version=1,
        elo=1520,
        games_played=50,
        win_rate=0.52
    )

    decision, reason = gate.evaluate_for_staging(result)
    print(f"Staging decision: {decision.value} - {reason}")

    # Test production evaluation with comparison
    prod_result = EvaluationResult(
        model_id="prod_model",
        version=5,
        elo=1500,
        games_played=1000,
        win_rate=0.50
    )

    result.elo_vs_production = 20
    result.win_rate_vs_production = 0.54
    result.games_vs_production = 100
    result.games_played = 150

    decision, reason = gate.evaluate_for_production(result, prod_result)
    print(f"Production decision: {decision.value} - {reason}")

    # Test training trigger
    trigger = TrainingTrigger(config)

    conditions = trigger.get_conditions(
        current_games=600,
        data_quality=0.9
    )

    should_train, reason = trigger.should_trigger_training(conditions)
    print(f"Training trigger: {should_train} - {reason}")

    # Print status
    print("\nManager status:")
    print(json.dumps(manager.get_status(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
