"""Optimized Training Data Pipeline.

Combines all training loop improvements into a unified pipeline:
- Export cache (skip unchanged data)
- Parallel export (multi-core encoding)
- Dynamic export settings (auto-compute based on data)
- Distributed locks (prevent concurrent training)
- Curriculum feedback (adaptive weights)
- Health monitoring (track status and alerts)
- Model registry (full traceability)

Usage:
    from app.training.optimized_pipeline import OptimizedTrainingPipeline

    pipeline = OptimizedTrainingPipeline()

    # Run optimized training for a config
    result = pipeline.run_training("square8_2p", db_paths=["data/games/selfplay.db"])

    # Get pipeline status
    status = pipeline.get_status()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Import all optimizations
try:
    from .export_cache import ExportCache
    HAS_EXPORT_CACHE = True
except ImportError:
    HAS_EXPORT_CACHE = False
    ExportCache = None

try:
    from .dynamic_export import get_export_settings, ExportSettings
    HAS_DYNAMIC_EXPORT = True
except ImportError:
    HAS_DYNAMIC_EXPORT = False
    get_export_settings = None
    ExportSettings = None

try:
    from .curriculum_feedback import get_curriculum_feedback, CurriculumFeedback
    HAS_CURRICULUM_FEEDBACK = True
except ImportError:
    HAS_CURRICULUM_FEEDBACK = False
    get_curriculum_feedback = None
    CurriculumFeedback = None

try:
    from .training_triggers import TrainingTriggers, TriggerConfig
    HAS_TRAINING_TRIGGERS = True
except ImportError:
    HAS_TRAINING_TRIGGERS = False
    TrainingTriggers = None
    TriggerConfig = None

try:
    from .training_health import get_training_health_monitor, TrainingHealthMonitor
    HAS_HEALTH_MONITOR = True
except ImportError:
    HAS_HEALTH_MONITOR = False
    get_training_health_monitor = None
    TrainingHealthMonitor = None

try:
    from .training_registry import register_trained_model
    HAS_MODEL_REGISTRY = True
except ImportError:
    HAS_MODEL_REGISTRY = False
    register_trained_model = None

try:
    from app.coordination.distributed_lock import DistributedLock
    HAS_DISTRIBUTED_LOCK = True
except ImportError:
    HAS_DISTRIBUTED_LOCK = False
    DistributedLock = None

# Training enhancements
try:
    from .training_enhancements import (
        CheckpointAverager,
        GradientAccumulator,
        DataQualityScorer,
        AdaptiveLRScheduler,
        EnhancedEarlyStopping,
        EWCRegularizer,
        ModelEnsemble,
        CalibrationAutomation,
        create_training_enhancements,
    )
    HAS_TRAINING_ENHANCEMENTS = True
except ImportError:
    HAS_TRAINING_ENHANCEMENTS = False
    CheckpointAverager = None
    GradientAccumulator = None
    DataQualityScorer = None
    AdaptiveLRScheduler = None
    EnhancedEarlyStopping = None
    EWCRegularizer = None
    ModelEnsemble = None
    CalibrationAutomation = None
    create_training_enhancements = None

try:
    from .multi_task_learning import (
        MultiTaskConfig,
        MultiTaskHead,
        MultiTaskLoss,
        MultiTaskModelWrapper,
    )
    HAS_MULTI_TASK = True
except ImportError:
    HAS_MULTI_TASK = False
    MultiTaskConfig = None
    MultiTaskHead = None
    MultiTaskLoss = None
    MultiTaskModelWrapper = None

try:
    from .distributed_training import (
        DistributedTrainer,
        DistributedConfig,
        GradientCompressor,
    )
    HAS_DISTRIBUTED_TRAINING = True
except ImportError:
    HAS_DISTRIBUTED_TRAINING = False
    DistributedTrainer = None
    DistributedConfig = None
    GradientCompressor = None

try:
    from .advanced_training import (
        LRFinder,
        LRFinderResult,
        GradientCheckpointing,
        PFSPOpponentPool,
        OpponentStats,
        CMAESAutoTuner,
        PlateauConfig,
        create_advanced_training_suite,
    )
    HAS_ADVANCED_TRAINING = True
except ImportError:
    HAS_ADVANCED_TRAINING = False
    LRFinder = None
    LRFinderResult = None
    GradientCheckpointing = None
    PFSPOpponentPool = None
    OpponentStats = None
    CMAESAutoTuner = None
    PlateauConfig = None
    create_advanced_training_suite = None


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    config_key: str
    success: bool
    message: str
    export_time: float = 0
    training_time: float = 0
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class PipelineStatus:
    """Status of the pipeline."""
    available_features: Dict[str, bool]
    active_training: List[str]
    health_status: str
    curriculum_weights: Dict[str, float]
    recent_results: List[PipelineResult]


class OptimizedTrainingPipeline:
    """Unified optimized training pipeline."""

    def __init__(self):
        # Initialize components
        self._export_cache = ExportCache() if HAS_EXPORT_CACHE else None
        self._curriculum = get_curriculum_feedback() if HAS_CURRICULUM_FEEDBACK else None
        self._health = get_training_health_monitor() if HAS_HEALTH_MONITOR else None
        self._triggers = TrainingTriggers(TriggerConfig()) if HAS_TRAINING_TRIGGERS else None

        # Track active training and recent results
        self._active_locks: Dict[str, Any] = {}
        self._recent_results: List[PipelineResult] = []
        self._max_recent = 50

        # Training enhancements
        self._checkpoint_averager = CheckpointAverager() if HAS_TRAINING_ENHANCEMENTS else None
        self._quality_scorer = DataQualityScorer() if HAS_TRAINING_ENHANCEMENTS else None
        self._calibration = CalibrationAutomation() if HAS_TRAINING_ENHANCEMENTS else None

        # Advanced training utilities
        self._pfsp_pool = PFSPOpponentPool() if HAS_ADVANCED_TRAINING else None
        self._auto_tuner = None  # Initialized on demand with model-specific settings
        self._gradient_checkpointing = None  # Initialized with model

        logger.info(f"OptimizedTrainingPipeline initialized with: "
                    f"cache={HAS_EXPORT_CACHE}, dynamic={HAS_DYNAMIC_EXPORT}, "
                    f"curriculum={HAS_CURRICULUM_FEEDBACK}, locks={HAS_DISTRIBUTED_LOCK}, "
                    f"health={HAS_HEALTH_MONITOR}, registry={HAS_MODEL_REGISTRY}, "
                    f"enhancements={HAS_TRAINING_ENHANCEMENTS}, "
                    f"multi_task={HAS_MULTI_TASK}, distributed={HAS_DISTRIBUTED_TRAINING}, "
                    f"advanced={HAS_ADVANCED_TRAINING}")

    def should_train(self, config_key: str, games_since_training: int = 0) -> Tuple[bool, str]:
        """Check if training should run for a config.

        Returns:
            Tuple of (should_train, reason)
        """
        if self._triggers:
            decision = self._triggers.should_train(config_key)
            return decision.should_train, decision.reason

        # Fallback to simple threshold
        threshold = 500
        should = games_since_training >= threshold
        return should, f"games={games_since_training} >= threshold={threshold}"

    def get_export_settings(
        self,
        config_key: str,
        db_paths: List[str],
    ) -> Dict[str, Any]:
        """Get optimal export settings for a config.

        Returns:
            Dict with max_games, sample_every, epochs, batch_size
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        if HAS_DYNAMIC_EXPORT and db_paths:
            try:
                settings = get_export_settings(db_paths, board_type, num_players)
                return {
                    "max_games": settings.max_games,
                    "sample_every": settings.sample_every,
                    "epochs": settings.epochs,
                    "batch_size": settings.batch_size,
                    "data_tier": settings.data_tier,
                    "estimated_samples": settings.estimated_samples,
                }
            except Exception as e:
                logger.warning(f"Could not compute dynamic settings: {e}")

        # Default settings
        return {
            "max_games": 50000,
            "sample_every": 2,
            "epochs": 50,
            "batch_size": 256,
            "data_tier": "unknown",
            "estimated_samples": 0,
        }

    def needs_export(
        self,
        config_key: str,
        db_paths: List[str],
        output_path: str,
    ) -> bool:
        """Check if export is needed (or can use cached data)."""
        if not HAS_EXPORT_CACHE or self._export_cache is None:
            return True

        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        return self._export_cache.needs_export(
            db_paths=db_paths,
            output_path=output_path,
            board_type=board_type,
            num_players=num_players,
        )

    def acquire_lock(self, config_key: str, timeout: int = 60) -> bool:
        """Acquire distributed lock for training.

        Returns:
            True if lock acquired, False otherwise
        """
        if not HAS_DISTRIBUTED_LOCK:
            return True

        if config_key in self._active_locks:
            return True  # Already have lock

        lock = DistributedLock(f"training:{config_key}", lock_timeout=7200)
        if lock.acquire(timeout=timeout, blocking=True):
            self._active_locks[config_key] = lock
            logger.info(f"Acquired training lock for {config_key}")
            return True

        logger.warning(f"Could not acquire lock for {config_key}")
        return False

    def release_lock(self, config_key: str) -> None:
        """Release distributed lock."""
        if config_key in self._active_locks:
            self._active_locks[config_key].release()
            del self._active_locks[config_key]
            logger.info(f"Released training lock for {config_key}")

    def get_curriculum_weight(self, config_key: str) -> float:
        """Get curriculum weight for a config (0.5 to 2.0)."""
        if self._curriculum:
            weights = self._curriculum.get_curriculum_weights()
            return weights.get(config_key, 1.0)
        return 1.0

    def run_export(
        self,
        config_key: str,
        db_paths: List[str],
        output_path: str,
        parallel: bool = True,
        workers: int = 4,
    ) -> Tuple[bool, float]:
        """Run data export with all optimizations.

        Returns:
            Tuple of (success, elapsed_seconds)
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Get optimal settings
        settings = self.get_export_settings(config_key, db_paths)

        # Check cache
        if not self.needs_export(config_key, db_paths, output_path):
            logger.info(f"Export cache valid for {config_key}, skipping export")
            return True, 0

        # Build export command
        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "export_replay_dataset.py"),
            "--output", output_path,
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--sample-every", str(settings["sample_every"]),
            "--require-completed",
            "--use-cache",
        ]

        if settings["max_games"]:
            cmd.extend(["--max-games", str(settings["max_games"])])

        if parallel:
            cmd.extend(["--parallel", "--workers", str(workers)])

        for db_path in db_paths:
            cmd.extend(["--db", db_path])

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                logger.info(f"Export complete for {config_key} in {elapsed:.1f}s")
                return True, elapsed
            else:
                logger.error(f"Export failed for {config_key}: {result.stderr[:500]}")
                return False, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.error(f"Export timeout for {config_key} after {elapsed:.1f}s")
            return False, elapsed
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Export error for {config_key}: {e}")
            return False, elapsed

    def run_training(
        self,
        config_key: str,
        db_paths: Optional[List[str]] = None,
        npz_path: Optional[str] = None,
        skip_export: bool = False,
    ) -> PipelineResult:
        """Run full optimized training pipeline.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            db_paths: Database paths for export
            npz_path: Pre-exported NPZ path (skips export)
            skip_export: Skip export step entirely

        Returns:
            PipelineResult with status and metrics
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = PipelineResult(config_key=config_key, success=False, message="")

        # Acquire lock
        if not self.acquire_lock(config_key):
            result.message = "Could not acquire training lock"
            self._add_result(result)
            return result

        try:
            # Record health start
            if self._health:
                self._health.record_training_start(config_key)

            # Export step
            if npz_path is None:
                npz_path = str(AI_SERVICE_ROOT / "data" / "training" / f"{config_key}_{timestamp}.npz")

            if not skip_export and db_paths:
                export_success, export_time = self.run_export(
                    config_key, db_paths, npz_path
                )
                result.export_time = export_time

                if not export_success:
                    result.message = "Export failed"
                    if self._health:
                        self._health.record_training_complete(config_key, False, error_message=result.message)
                    self._add_result(result)
                    return result

            # Check NPZ exists
            if not Path(npz_path).exists():
                result.message = f"Training data not found: {npz_path}"
                if self._health:
                    self._health.record_training_complete(config_key, False, error_message=result.message)
                self._add_result(result)
                return result

            # Get settings
            settings = self.get_export_settings(config_key, db_paths or [])

            # Build training command
            run_dir = AI_SERVICE_ROOT / "runs" / f"{config_key}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_nn_training_baseline.py"),
                "--board", board_type,
                "--num-players", str(num_players),
                "--data-path", npz_path,
                "--run-dir", str(run_dir),
                "--epochs", str(settings["epochs"]),
                "--batch-size", str(settings["batch_size"]),
                "--model-version", "v3",
                "--use-optimized-hyperparams",
            ]

            start = time.time()
            try:
                train_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                    env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
                )
                result.training_time = time.time() - start

                if train_result.returncode == 0:
                    result.success = True
                    result.message = "Training complete"
                    result.model_path = str(run_dir / "best_model.pt")

                    # Register model
                    if HAS_MODEL_REGISTRY and Path(result.model_path).exists():
                        try:
                            result.model_id = register_trained_model(
                                model_path=result.model_path,
                                board_type=board_type,
                                num_players=num_players,
                                training_config=settings,
                                source_data_paths=db_paths or [npz_path],
                            )
                        except Exception as e:
                            logger.warning(f"Could not register model: {e}")

                    if self._health:
                        self._health.record_training_complete(config_key, True, metrics=result.metrics)
                else:
                    result.message = f"Training failed: {train_result.stderr[:200]}"
                    if self._health:
                        self._health.record_training_complete(config_key, False, error_message=result.message)

            except subprocess.TimeoutExpired:
                result.training_time = time.time() - start
                result.message = "Training timeout"
                if self._health:
                    self._health.record_training_complete(config_key, False, error_message=result.message)

        finally:
            self.release_lock(config_key)
            self._add_result(result)

        return result

    def _add_result(self, result: PipelineResult) -> None:
        """Add result to recent history."""
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent:]

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status."""
        health_status = "unknown"
        if self._health:
            report = self._health.get_health_status()
            health_status = report.status.value

        curriculum_weights = {}
        if self._curriculum:
            curriculum_weights = self._curriculum.get_curriculum_weights()

        return PipelineStatus(
            available_features={
                "export_cache": HAS_EXPORT_CACHE,
                "dynamic_export": HAS_DYNAMIC_EXPORT,
                "curriculum_feedback": HAS_CURRICULUM_FEEDBACK,
                "distributed_lock": HAS_DISTRIBUTED_LOCK,
                "health_monitor": HAS_HEALTH_MONITOR,
                "model_registry": HAS_MODEL_REGISTRY,
                "training_triggers": HAS_TRAINING_TRIGGERS,
                "training_enhancements": HAS_TRAINING_ENHANCEMENTS,
                "multi_task_learning": HAS_MULTI_TASK,
                "distributed_training": HAS_DISTRIBUTED_TRAINING,
                "advanced_training": HAS_ADVANCED_TRAINING,
            },
            active_training=list(self._active_locks.keys()),
            health_status=health_status,
            curriculum_weights=curriculum_weights,
            recent_results=self._recent_results[-10:],
        )

    def get_training_enhancements(
        self,
        model: Any,
        optimizer: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get training enhancement objects for a training run.

        Args:
            model: Model being trained
            optimizer: Optimizer being used
            config: Optional configuration overrides

        Returns:
            Dictionary of enhancement objects or None if not available
        """
        if not HAS_TRAINING_ENHANCEMENTS or create_training_enhancements is None:
            return None

        return create_training_enhancements(model, optimizer, config)

    def get_checkpoint_averager(self, num_checkpoints: int = 5) -> Optional[Any]:
        """Get a checkpoint averager for model weight averaging.

        Args:
            num_checkpoints: Number of checkpoints to average

        Returns:
            CheckpointAverager or None if not available
        """
        if not HAS_TRAINING_ENHANCEMENTS or CheckpointAverager is None:
            return None
        return CheckpointAverager(num_checkpoints=num_checkpoints)

    def get_gradient_accumulator(
        self,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
    ) -> Optional[Any]:
        """Get a gradient accumulator for larger effective batch sizes.

        Args:
            accumulation_steps: Number of steps to accumulate
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            GradientAccumulator or None if not available
        """
        if not HAS_TRAINING_ENHANCEMENTS or GradientAccumulator is None:
            return None
        return GradientAccumulator(
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
        )

    def get_model_ensemble(
        self,
        model_class: type,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get a model ensemble for diverse self-play opponents.

        Args:
            model_class: Class to instantiate models from
            model_kwargs: Arguments for model constructor

        Returns:
            ModelEnsemble or None if not available
        """
        if not HAS_TRAINING_ENHANCEMENTS or ModelEnsemble is None:
            return None
        return ModelEnsemble(model_class=model_class, model_kwargs=model_kwargs)

    def should_recalibrate(self, epoch: int = 0) -> bool:
        """Check if value head recalibration is needed.

        Args:
            epoch: Current epoch number

        Returns:
            True if recalibration is recommended
        """
        if self._calibration:
            return self._calibration.should_recalibrate(epoch)
        return False

    def add_calibration_samples(
        self,
        predictions: Any,
        outcomes: Any,
    ) -> None:
        """Add samples for calibration monitoring.

        Args:
            predictions: Model value predictions
            outcomes: Actual game outcomes
        """
        if self._calibration:
            self._calibration.add_samples(predictions, outcomes)

    # =========================================================================
    # Advanced Training Utilities
    # =========================================================================

    def get_lr_finder(
        self,
        model: Any,
        optimizer: Any,
        criterion: Any,
    ) -> Optional[Any]:
        """Get a learning rate finder for optimal LR detection.

        Args:
            model: Model to find LR for
            optimizer: Optimizer to use
            criterion: Loss function

        Returns:
            LRFinder instance or None if not available
        """
        if not HAS_ADVANCED_TRAINING or LRFinder is None:
            return None

        return LRFinder(model, optimizer, criterion)

    def get_gradient_checkpointing(self, model: Any) -> Optional[Any]:
        """Get gradient checkpointing for memory-efficient training.

        Args:
            model: Model to apply checkpointing to

        Returns:
            GradientCheckpointing instance or None if not available
        """
        if not HAS_ADVANCED_TRAINING or GradientCheckpointing is None:
            return None

        self._gradient_checkpointing = GradientCheckpointing(model)
        return self._gradient_checkpointing

    def get_pfsp_pool(self) -> Optional[Any]:
        """Get the PFSP opponent pool for self-play.

        Returns:
            PFSPOpponentPool instance or None if not available
        """
        return self._pfsp_pool

    def get_auto_tuner(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        plateau_patience: int = 10,
    ) -> Optional[Any]:
        """Get CMA-ES auto-tuner for hyperparameter optimization on plateau.

        Args:
            board_type: Board type for optimization
            num_players: Number of players
            plateau_patience: Epochs without improvement before triggering

        Returns:
            CMAESAutoTuner instance or None if not available
        """
        if not HAS_ADVANCED_TRAINING or CMAESAutoTuner is None:
            return None

        if self._auto_tuner is None:
            plateau_cfg = PlateauConfig(patience=plateau_patience)
            self._auto_tuner = CMAESAutoTuner(
                board_type=board_type,
                num_players=num_players,
                plateau_config=plateau_cfg,
            )

        return self._auto_tuner

    def get_advanced_training_suite(
        self,
        model: Any,
        optimizer: Any,
        criterion: Any,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """Get a complete suite of advanced training utilities.

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            board_type: Board type
            num_players: Number of players

        Returns:
            Dictionary of utility objects or None if not available
        """
        if not HAS_ADVANCED_TRAINING or create_advanced_training_suite is None:
            return None

        return create_advanced_training_suite(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            board_type=board_type,
            num_players=num_players,
        )


# Singleton instance
_pipeline_instance: Optional[OptimizedTrainingPipeline] = None


def get_optimized_pipeline() -> OptimizedTrainingPipeline:
    """Get the global optimized training pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = OptimizedTrainingPipeline()
    return _pipeline_instance
