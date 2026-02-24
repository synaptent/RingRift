"""Training script for RingRift Neural Network AI.

Includes validation split, checkpointing, early stopping, LR warmup,
and distributed training support via PyTorch DDP.

Recommended Usage (December 2025):
    For unified data management and training coordination, consider using:
    - TrainingDataCoordinator: app.training.data_coordinator
    - UnifiedTrainingOrchestrator: app.training.unified_orchestrator

    For modular training step/epoch logic (December 2025 extraction):
    - train_step: Core batch-level training (forward, loss, backward, step)
    - train_epoch: Epoch-level training loop with validation and early stopping

    Example using data coordinator and orchestrator:
        from app.training.data_coordinator import get_data_coordinator
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        # Use data coordinator for quality-aware data loading
        coordinator = get_data_coordinator()
        await coordinator.prepare_for_training(board_type="square8", num_players=2)

        # Use unified training orchestrator for lifecycle management
        orchestrator = UnifiedTrainingOrchestrator.from_config(config)
        await orchestrator.initialize()
        # Run training with orchestrator...

    Example using modular train_step/train_epoch:
        from app.training.train_step import TrainStepContext, TrainStepConfig, run_training_step
        from app.training.train_epoch import EpochContext, EpochConfig, run_all_epochs

        # Create contexts
        step_config = TrainStepConfig(use_mixed_precision=True)
        epoch_config = EpochConfig(epochs=20, patience=10)
        epoch_context = EpochContext(
            model=model, optimizer=optimizer, train_loader=train_loader,
            val_loader=val_loader, device=device, config=epoch_config,
        )

        # Run all epochs with early stopping
        results = run_all_epochs(epoch_context)
"""
from __future__ import annotations


import contextlib
import glob
import logging
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from app.utils.numpy_utils import safe_load_npz
from app.utils.torch_utils import safe_load_checkpoint

# Training control thresholds (December 2025)
from app.config.thresholds import (
    EARLY_STOPPING_PATIENCE,
    ELO_PATIENCE,
    MIN_TRAINING_EPOCHS,
    TRAINING_RETRY_SLEEP_SECONDS,
)

# Training metrics extracted to dedicated module (December 2025)
from app.training.train_metrics import (
    ANOMALY_DETECTIONS,
    BATCH_SIZE,
    CALIBRATION_ECE,
    CALIBRATION_MCE,
    CIRCUIT_BREAKER_STATE,
    GRADIENT_CLIP_NORM,
    GRADIENT_NORM,
    HAS_METRICS_COLLECTOR,
    HAS_PROMETHEUS,
    TRAINING_DURATION,
    TRAINING_EPOCHS,
    TRAINING_LOSS,
    TRAINING_SAMPLES,
    MetricsCollector,
)

from app.ai.neural_losses import (
    build_rank_targets,
    detect_masked_policy_output,
    masked_log_softmax,
    masked_policy_kl,
    uses_spatial_policy_head,
    validate_hex_policy_indices,
)
from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    MAX_PLAYERS,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Flat,  # V3 with flat policy heads (training compatible, Dec 2025)
    HexNeuralNet_v4,
    HexNeuralNet_v5_Heavy,
    RingRiftCNN_v2,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Flat,  # V3 with flat policy heads (training compatible, Dec 2025)
    get_policy_size_for_board,
    multi_player_value_loss,
)
from app.models import BoardType
from app.utils.canonical_naming import normalize_board_type
from app.training.config import TrainConfig
# December 2025: Structured config objects for train_model() parameter reduction
from app.training.train_config import (
    FullTrainingConfig,
    TrainingDataConfig,
    DistributedConfig,
    CheckpointConfig,
    EnhancementConfig,
    config_from_legacy_params,
)
from app.training.data_loader import (
    StreamingDataLoader,
    WeightedStreamingDataLoader,
    get_sample_count,
    prefetch_loader,
)
from app.training.datasets import RingRiftDataset, WeightedRingRiftDataset
from app.training.distributed import (
    DistributedMetrics,
    cleanup_distributed,
    get_distributed_sampler,
    get_rank,
    get_world_size,
    is_main_process,
    scale_learning_rate,
    seed_everything,
    setup_distributed,
    wrap_model_ddp,
)
from app.training.fault_tolerance import HeartbeatMonitor
from app.training.gradient_surgery import GradientSurgeon, GradientSurgeryConfig
from app.training.model_versioning import (
    save_model_checkpoint,
)
from app.training.seed_utils import seed_all
from app.training.train_setup import (
    FaultToleranceConfig,
    TrainingState,
    setup_fault_tolerance,
)
from app.training.value_calibration import CalibrationTracker

# December 2025: Modular training step/epoch logic
# These modules extract core training logic for testability and reuse
from app.training.train_step import (
    BatchData,
    LossComponents,
    TrainStepConfig,
    TrainStepContext,
    TrainStepResult,
    parse_batch,
    run_training_step,
    transfer_batch_to_device,
)
from app.training.train_epoch import (
    EarlyStopState,
    EpochConfig,
    EpochContext,
    EpochResult,
    run_all_epochs,
    run_training_epoch,
    run_validation_loop,
)
from app.training.train_components import (
    resolve_train_config,
)

# February 2026: Extracted modules from train_model() for maintainability
from app.training.train_pre_validation import run_pre_training_validation
from app.training.train_dataset_inference import (
    DatasetInferenceResult,
    infer_dataset_metadata,
)
from app.training.train_model_factory import create_training_model

# Data validation (2025-12) - use unified module
try:
    from app.training.unified_data_validator import (
        DataValidator,
        DataValidatorConfig,
        validate_npz_file,
    )
    HAS_DATA_VALIDATION = True
except ImportError:
    HAS_DATA_VALIDATION = False
    DataValidator = None
    DataValidatorConfig = None
    validate_npz_file = None

# Checksum verification for data integrity (December 2025)
try:
    from app.training.data_quality import verify_npz_checksums
    HAS_CHECKSUM_VERIFICATION = True
except ImportError:
    verify_npz_checksums = None
    HAS_CHECKSUM_VERIFICATION = False

# NPZ structure validation for corruption detection (December 2025)
# Catches issues like rsync --partial creating files with unreasonable dimensions
try:
    from app.coordination.npz_validation import (
        validate_npz_structure,
        NPZValidationResult,
    )
    HAS_NPZ_STRUCTURE_VALIDATION = True
except ImportError:
    validate_npz_structure = None
    NPZValidationResult = None
    HAS_NPZ_STRUCTURE_VALIDATION = False

# December 2025: Extracted validation utilities
try:
    from app.training.train_validation import (
        validate_training_data_freshness,
        validate_training_data_files,
        validate_data_checksums,
        FreshnessResult,
    )
    HAS_TRAIN_VALIDATION = True
except ImportError:
    HAS_TRAIN_VALIDATION = False
    validate_training_data_freshness = None
    validate_training_data_files = None
    validate_data_checksums = None
    FreshnessResult = None

# Hot data buffer for priority experience replay (2024-12)
try:
    from app.training.hot_data_buffer import HotDataBuffer
    HAS_HOT_DATA_BUFFER = True
except ImportError:
    HotDataBuffer = None
    HAS_HOT_DATA_BUFFER = False

# Quality bridge for quality-aware data selection (2025-12)
try:
    from app.training.quality_bridge import (
        QualityBridge,
        get_quality_bridge,
    )
    HAS_QUALITY_BRIDGE = True
except ImportError:
    HAS_QUALITY_BRIDGE = False
    get_quality_bridge = None
    QualityBridge = None

# Integrated enhancements (2024-12)
try:
    from app.training.integrated_enhancements import (
        IntegratedEnhancementsConfig,
        IntegratedTrainingManager,
    )
    HAS_INTEGRATED_ENHANCEMENTS = True
except ImportError:
    IntegratedTrainingManager = None
    IntegratedEnhancementsConfig = None
    HAS_INTEGRATED_ENHANCEMENTS = False

# Circuit breaker for training fault tolerance (2025-12)
try:
    from app.distributed.circuit_breaker import CircuitState, get_training_breaker
    from app.coordination.event_router import get_router
    from app.coordination.event_router import DataEvent, DataEventType
    HAS_CIRCUIT_BREAKER = True
    HAS_EVENT_BUS = True
except ImportError:
    get_training_breaker = None
    CircuitState = None
    get_router = None
    DataEvent = None
    DataEventType = None
    HAS_CIRCUIT_BREAKER = False
    HAS_EVENT_BUS = False

# Event emission for training feedback loops (Phase 21.2 - Dec 2025)
try:
    from app.coordination.event_router import (
        emit_training_loss_anomaly,
        emit_training_loss_trend,
    )
    HAS_TRAINING_EVENTS = True
except ImportError:
    emit_training_loss_anomaly = None
    emit_training_loss_trend = None
    HAS_TRAINING_EVENTS = False

# Epoch event emission for curriculum feedback (December 2025)
try:
    from app.training.event_integration import publish_epoch_completed
    HAS_EPOCH_EVENTS = True
except ImportError:
    publish_epoch_completed = None
    HAS_EPOCH_EVENTS = False

# Regression detection for training quality monitoring (2025-12)
try:
    from app.training.regression_detector import (
        RegressionSeverity,
        get_regression_detector,
    )
    HAS_REGRESSION_DETECTOR = True
except ImportError:
    get_regression_detector = None
    RegressionSeverity = None
    HAS_REGRESSION_DETECTOR = False

# Training data freshness checking (2025-12)
try:
    from app.coordination.training_freshness import (
        check_freshness_sync,
        FreshnessConfig,
        FreshnessResult,
    )
    HAS_FRESHNESS_CHECK = True
except ImportError:
    check_freshness_sync = None
    FreshnessConfig = None
    FreshnessResult = None
    HAS_FRESHNESS_CHECK = False

# Training stale data fallback (December 2025)
# Part of 48-hour autonomous operation plan - allows training with stale data
# after configurable sync failures or timeout
try:
    from app.coordination.stale_fallback import (
        get_training_fallback_controller,
        should_allow_stale_training,
    )
    HAS_STALE_FALLBACK = True
except ImportError:
    get_training_fallback_controller = None
    should_allow_stale_training = None
    HAS_STALE_FALLBACK = False

# Training anomaly detection and enhancements (2025-12)
try:
    from app.training.training_enhancements import (
        AdaptiveGradientClipper,
        CheckpointAverager,
        EvaluationFeedbackHandler,
        TrainingAnomalyDetector,
    )
    HAS_TRAINING_ENHANCEMENTS = True
except ImportError:
    TrainingAnomalyDetector = None
    CheckpointAverager = None
    AdaptiveGradientClipper = None
    EvaluationFeedbackHandler = None
    HAS_TRAINING_ENHANCEMENTS = False

# Hard example mining for curriculum learning (2025-12)
try:
    from app.training.enhancements.hard_example_mining import HardExampleMiner
    from app.training.enhancements.per_sample_loss import compute_per_sample_loss
    HAS_HARD_EXAMPLE_MINING = True
except ImportError:
    HardExampleMiner = None
    compute_per_sample_loss = None
    HAS_HARD_EXAMPLE_MINING = False

# Unified training enhancements facade (2025-12)
# Consolidates: hard example mining, per-sample loss, curriculum LR, freshness weighting
try:
    from app.training.enhancements.training_facade import (
        FacadeConfig,
        TrainingEnhancementsFacade,
    )
    HAS_TRAINING_FACADE = True
except ImportError:
    FacadeConfig = None
    TrainingEnhancementsFacade = None
    HAS_TRAINING_FACADE = False

# DataCatalog for cluster-wide training data discovery (2025-12)
try:
    from app.distributed.data_catalog import DataCatalog, get_data_catalog
    HAS_DATA_CATALOG = True
except ImportError:
    DataCatalog = None
    get_data_catalog = None
    HAS_DATA_CATALOG = False

# Quality-weighted training (2025-12) - resurrected from ebmo_network.py
try:
    from app.training.quality_weighted_loss import (
        QualityWeightedTrainer,
        compute_quality_weights,
        quality_weighted_policy_loss,
        ranking_loss_from_quality,
    )
    HAS_QUALITY_WEIGHTING = True
except ImportError:
    QualityWeightedTrainer = None
    compute_quality_weights = None
    quality_weighted_policy_loss = None
    ranking_loss_from_quality = None
    HAS_QUALITY_WEIGHTING = False

# Auto-streaming threshold: datasets larger than this will automatically use
# StreamingDataLoader to avoid OOM. Default 5GB.
AUTO_STREAMING_THRESHOLD_BYTES = int(os.environ.get(
    "RINGRIFT_AUTO_STREAMING_THRESHOLD_GB", "5"
)) * 1024 * 1024 * 1024

from app.ai.heuristic_weights import (
    HEURISTIC_WEIGHT_KEYS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.training.eval_pools import run_heuristic_tier_eval
from app.training.tier_eval_config import (
    HEURISTIC_TIER_SPECS,
    HeuristicTierSpec,
)

# Heuristic tuning utilities (extracted Dec 2025)
# NOTE: Some helper functions are re-exported for backwards compatibility with
# integration tests and older tooling.
#
# IMPORTANT: Unit tests monkeypatch
# [`app.training.train.evaluate_heuristic_candidate`](ai-service/app/training/train.py:1)
# to avoid running the expensive eval-pool harness. To keep that patching
# surface stable after the refactor to `heuristic_tuning.py`, we expose thin
# wrappers here and pass the wrapper into the underlying implementation.
from app.training.heuristic_tuning import (
    _flatten_heuristic_weights,
    _reconstruct_heuristic_profile,
    evaluate_heuristic_candidate as _evaluate_heuristic_candidate,
    run_cmaes_heuristic_optimization as _run_cmaes_heuristic_optimization,
    temporary_heuristic_profile,
)


# =============================================================================
# LossMonitor - Detect learning stalls during training (Jan 2026 fix)
# =============================================================================

class LossMonitor:
    """Track loss curves and detect learning failures during training.

    This class monitors training progress and emits events when the model
    stops learning (loss not decreasing for consecutive epochs).

    Usage:
        monitor = LossMonitor(patience=5, config_key="hex8_2p")
        for epoch in range(epochs):
            train_loss = train_epoch(...)
            val_loss = validate(...)

            if not monitor.record(epoch, train_loss, val_loss):
                logger.error("Training stalled - stopping early")
                break
    """

    def __init__(self, patience: int = 5, config_key: str = "unknown"):
        """Initialize the loss monitor.

        Args:
            patience: Number of epochs without improvement before signaling stall.
            config_key: Configuration key for event emission (e.g., "hex8_2p").
        """
        self.patience = patience
        self.config_key = config_key
        self.history: list[dict[str, float]] = []
        self.best_loss = float('inf')
        self.stale_epochs = 0
        self._logger = logging.getLogger(__name__)

    def record(self, epoch: int, train_loss: float, val_loss: float) -> bool:
        """Record epoch losses and check for learning stall.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.

        Returns:
            True if training should continue, False if stalled.
        """
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        # Check for improvement (1% threshold)
        if val_loss < self.best_loss * 0.99:
            self.best_loss = val_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1

        # Emit warning if approaching stall
        if self.stale_epochs >= self.patience:
            self._logger.warning(
                f"[LossMonitor] Loss not decreasing for {self.patience} epochs! "
                f"Best: {self.best_loss:.4f}, Current: {val_loss:.4f}"
            )

            # Emit anomaly event if available
            if HAS_TRAINING_EVENTS and emit_training_loss_anomaly is not None:
                try:
                    emit_training_loss_anomaly(
                        config_key=self.config_key,
                        anomaly_type="learning_stall",
                        epochs_stale=self.stale_epochs,
                        best_loss=self.best_loss,
                        current_loss=val_loss,
                    )
                except (RuntimeError, TypeError) as e:
                    self._logger.debug(f"Failed to emit anomaly event: {e}")

            return False  # Signal to stop

        # Emit trend info periodically
        if epoch % 5 == 0 and HAS_TRAINING_EVENTS and emit_training_loss_trend is not None:
            try:
                emit_training_loss_trend(
                    config_key=self.config_key,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
            except (RuntimeError, TypeError):
                pass  # Non-critical, don't log

        return True  # Continue training

    def get_summary(self) -> dict[str, Any]:
        """Get summary of loss monitoring.

        Returns:
            Dictionary with monitoring summary.
        """
        return {
            'config_key': self.config_key,
            'epochs_recorded': len(self.history),
            'best_loss': self.best_loss,
            'stale_epochs': self.stale_epochs,
            'is_stalled': self.stale_epochs >= self.patience,
        }


def evaluate_heuristic_candidate(
    tier_spec: HeuristicTierSpec,
    base_profile_id: str,
    keys: Sequence[str],
    candidate_vector: Sequence[float],
    rng_seed: int,
    games_per_candidate: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a single heuristic weight candidate for CMA-ES optimization.

    Wrapper that runs selfplay games with a modified heuristic profile
    and returns the win rate as fitness score.

    Args:
        tier_spec: Heuristic tier specification for evaluation
        base_profile_id: Base heuristic profile to modify
        keys: Weight parameter names to optimize
        candidate_vector: Weight values for this candidate
        rng_seed: Random seed for reproducibility
        games_per_candidate: Number of games per evaluation (default from tier_spec)

    Returns:
        Tuple of (win_rate, detailed_stats)
    """
    return _evaluate_heuristic_candidate(
        tier_spec=tier_spec,
        base_profile_id=base_profile_id,
        keys=keys,
        candidate_vector=candidate_vector,
        rng_seed=rng_seed,
        games_per_candidate=games_per_candidate,
    )


def run_cmaes_heuristic_optimization(
    tier_id: str,
    base_profile_id: str,
    generations: int = 5,
    population_size: int = 8,
    rng_seed: int = 1,
    games_per_candidate: int | None = None,
) -> dict[str, Any]:
    """Run CMA-ES optimization to tune heuristic evaluation weights.

    Uses evolutionary strategy to find optimal weight configurations
    by running selfplay tournaments and selecting winning candidates.

    Args:
        tier_id: Heuristic tier to optimize (e.g., 'basic', 'advanced')
        base_profile_id: Starting profile for weight initialization
        generations: Number of CMA-ES generations
        population_size: Candidates per generation
        rng_seed: Random seed for reproducibility
        games_per_candidate: Games per candidate evaluation

    Returns:
        Dict with optimized weights and optimization history
    """
    return _run_cmaes_heuristic_optimization(
        tier_id=tier_id,
        base_profile_id=base_profile_id,
        generations=generations,
        population_size=population_size,
        rng_seed=rng_seed,
        games_per_candidate=games_per_candidate,
        evaluate_fn=evaluate_heuristic_candidate,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Backwards-compatible alias that forwards to the shared training
# seeding utility so that existing callers importing seed_all from this
# module continue to work.
def seed_all_legacy(seed: int = 42) -> None:
    """Seed all random number generators for reproducibility.

    Backward-compatible alias for seed_all() from training_enhancements.
    Sets seeds for Python random, NumPy, and PyTorch (CPU/CUDA).

    Args:
        seed: Random seed value (default: 42)
    """
    seed_all(seed)


# EarlyStopping is now imported from training_enhancements for consolidation
# The EnhancedEarlyStopping class provides backwards compatibility via __call__ method
from app.training.training_enhancements import EarlyStopping

# Checkpointing utilities - use unified module (2025-12)
HAS_UNIFIED_CHECKPOINT = True

# Legacy checkpointing functions (still available for backward compatibility)
# Migrated to import from checkpoint_unified (December 2025)
from app.training.checkpoint_unified import (
    AsyncCheckpointer,
    GracefulShutdownHandler,
    load_checkpoint,
    save_checkpoint,
)

# LR scheduler utilities extracted to dedicated module (2025-12)
from app.training.schedulers import create_lr_scheduler

# Dataset classes extracted to dedicated module (2025-12)
# RingRiftDataset and WeightedRingRiftDataset are imported from app.training.datasets


def _validate_training_compatibility(
    model: nn.Module,
    dataset: Any,
    config: "TrainConfig",
) -> None:
    """Phase 6: Validate model and dataset are compatible before training.

    This function catches common issues early to prevent wasted GPU hours:
    - Policy size mismatches between model and data
    - Board type incompatibility
    - Invalid sample data

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    dataset : Any
        The training dataset (RingRiftDataset or similar).
    config : TrainConfig
        Training configuration.

    Raises
    ------
    ValueError
        If model/dataset are incompatible or data validation fails.
    """
    logger.info("Running training compatibility validation...")

    # 1. Policy size compatibility
    model_policy_size = getattr(model, 'policy_size', None)
    dataset_policy_size = getattr(dataset, 'policy_size', None)

    if model_policy_size is not None and dataset_policy_size is not None:
        if dataset_policy_size > model_policy_size:
            raise ValueError(
                f"Dataset policy_size ({dataset_policy_size}) > model policy_size ({model_policy_size}). "
                f"Dataset contains indices the model cannot predict. "
                f"Check board type settings and encoder version."
            )
        elif dataset_policy_size < model_policy_size:
            logger.info(
                f"Dataset policy_size ({dataset_policy_size}) < model policy_size ({model_policy_size}). "
                f"Policy targets will be zero-padded (this is normal)."
            )

    # 2. Board type compatibility (if available)
    model_board_type = getattr(model, 'board_type', None)
    dataset_board_type = getattr(dataset, 'board_type', None)

    if model_board_type is not None and dataset_board_type is not None:
        if model_board_type != dataset_board_type:
            # Dec 28, 2025: Changed from warning to error to prevent cross-config contamination
            # Training hex8 model with square8 data (or vice versa) produces garbage models
            raise ValueError(
                f"[CROSS-CONFIG CONTAMINATION] Board type mismatch: "
                f"model expects '{model_board_type}', dataset contains '{dataset_board_type}'. "
                f"This would produce a garbage model. "
                f"Use --board-type to specify the correct board type, or regenerate training data."
            )

    # 3. Sample validation - check first few samples
    num_samples_to_check = min(10, len(dataset))
    policy_size = model_policy_size or dataset_policy_size or 4500

    invalid_samples = []
    for i in range(num_samples_to_check):
        try:
            sample = dataset[i]
            # Handle different return formats
            if isinstance(sample, tuple) and len(sample) >= 4:
                _, _, _, policy = sample[:4]
            else:
                continue

            # Check policy vector
            if hasattr(policy, 'sum'):
                policy_sum = policy.sum().item()
                if policy_sum > 0 and not (0.5 < policy_sum < 1.5):
                    invalid_samples.append((i, f"policy_sum={policy_sum:.4f}"))

                # Check for NaN
                if hasattr(policy, 'isnan') and policy.isnan().any():
                    invalid_samples.append((i, "contains NaN"))

        except (KeyError, IndexError, ValueError, AttributeError) as e:
            # KeyError: missing data keys, IndexError: array access,
            # ValueError: data validation, AttributeError: missing methods
            invalid_samples.append((i, f"error: {str(e)[:50]}"))

    if invalid_samples:
        logger.warning(
            f"Found {len(invalid_samples)} potentially invalid samples in first {num_samples_to_check}: "
            f"{invalid_samples[:5]}"
        )
        if len(invalid_samples) > num_samples_to_check // 2:
            raise ValueError(
                f"More than half of checked samples are invalid. "
                f"Dataset may be corrupted or incompatible. "
                f"Issues: {invalid_samples[:5]}"
            )

    logger.info("Training compatibility validation passed")


def detect_tier_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[str, str, int, int] | None:
    """Detect memory tier and architecture from a checkpoint.

    This function inspects a checkpoint file to extract the memory tier
    and architecture parameters. Used when resuming training to ensure
    the model architecture matches the checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the checkpoint file.
    device : str
        Device to load checkpoint onto (default: cpu).

    Returns
    -------
    tuple[str, str, int, int] or None
        (memory_tier, model_version, num_filters, num_res_blocks) if detected,
        None if checkpoint doesn't exist or can't be analyzed.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = safe_load_checkpoint(str(checkpoint_path), map_location=device)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Could not load checkpoint for tier detection: {e}")
        return None

    # Extract versioning metadata
    metadata = checkpoint.get("_versioning_metadata", {})
    config = metadata.get("config", {})

    # Check if memory_tier is directly stored
    memory_tier = metadata.get("memory_tier") or config.get("memory_tier", "")

    # Extract architecture parameters
    num_filters = config.get("num_filters")
    num_res_blocks = config.get("num_res_blocks")

    # If no direct tier, infer from config
    if not memory_tier and num_filters is not None:
        from app.training.model_versioning import infer_memory_tier_from_config
        memory_tier = infer_memory_tier_from_config(config)

    if not memory_tier:
        return None

    # Map tier to model_version
    tier_to_version = {
        "v4": "v4",
        "v3-high": "v3",
        "v3-low": "v3",
        "v5": "v5",
        "v5.1": "v5-heavy",
        "v5-heavy-large": "v5-heavy",
        "v5-heavy-xl": "v5-heavy",
        # Deprecated aliases (use v5-heavy-large, v5-heavy-xl instead)
        "v6": "v5-heavy",
        "v6-xl": "v5-heavy",
        "v2": "v2",
        "v2-lite": "v2",
        "gnn": "gnn",
        "hybrid": "hybrid",
    }
    model_version = tier_to_version.get(memory_tier, "v2")

    # Ensure we have architecture parameters
    if num_filters is None or num_res_blocks is None:
        # Use tier defaults
        tier_defaults = {
            "v4": (128, 13),
            "v3-high": (192, 12),
            "v3-low": (96, 6),
            "v5": (160, 11),
            "v5.1": (160, 11),
            "v5-heavy-large": (256, 18),
            "v5-heavy-xl": (320, 20),
            # Deprecated aliases
            "v6": (256, 18),
            "v6-xl": (320, 20),
            "v2": (96, 6),
            "v2-lite": (64, 6),
        }
        defaults = tier_defaults.get(memory_tier, (96, 6))
        num_filters = num_filters or defaults[0]
        num_res_blocks = num_res_blocks or defaults[1]

    logger.info(
        f"Detected checkpoint architecture: tier={memory_tier}, "
        f"version={model_version}, filters={num_filters}, blocks={num_res_blocks}"
    )

    return (memory_tier, model_version, num_filters, num_res_blocks)


def validate_training_data(npz_path: Path, board_type: str, num_players: int) -> None:
    """Validate NPZ file before training. Raises ValueError on issues.

    Pre-flight check that catches corrupt, empty, or mismatched data files
    before we spend time creating models and optimizers.

    Args:
        npz_path: Path to the NPZ training data file.
        board_type: Expected board type (e.g. "hex8", "square8").
        num_players: Expected number of players (2, 3, or 4).

    Raises:
        ValueError: If the NPZ file is invalid or too small.
        FileNotFoundError: If the NPZ file does not exist.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Training data not found: {npz_path}")

    file_size = npz_path.stat().st_size
    if file_size < 1024:
        raise ValueError(
            f"Training data file too small ({file_size} bytes): {npz_path}. "
            f"Likely corrupt or incomplete transfer."
        )

    data = np.load(npz_path, allow_pickle=True)
    required_keys = ["features", "values"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"NPZ missing required key: '{key}' in {npz_path}")

    sample_count = data["features"].shape[0]
    if sample_count < 100:
        raise ValueError(
            f"Too few samples in {npz_path}: {sample_count} (minimum 100). "
            f"Collect more selfplay data before training."
        )

    # Cross-check board_type metadata if present
    if "board_type" in data:
        raw_bt = data["board_type"]
        meta_bt = str(raw_bt.item() if hasattr(raw_bt, "item") else raw_bt).lower().strip()
        expected_bt = board_type.lower().strip()
        if meta_bt != expected_bt:
            raise ValueError(
                f"Board type mismatch in {npz_path}: "
                f"file has '{meta_bt}', expected '{expected_bt}'"
            )

    logger.info(
        f"Training data validated: {npz_path} "
        f"({sample_count} samples, board_type={board_type}, num_players={num_players})"
    )


def train_model(
    config: TrainConfig,
    data_path: str | list[str],
    save_path: str,
    early_stopping_patience: int | None = None,
    elo_early_stopping_patience: int | None = None,
    elo_min_improvement: float | None = None,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_interval: int = 5,
    _save_all_epochs: bool = True,  # Save every epoch for Elo-based selection
    warmup_epochs: int | None = None,
    lr_scheduler: str | None = None,
    lr_min: float | None = None,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
    resume_path: str | None = None,
    init_weights_path: str | None = None,
    init_weights_strict: bool = False,
    freeze_policy: bool = False,
    augment_hex_symmetry: bool = False,
    distributed: bool = False,
    local_rank: int = -1,
    scale_lr: bool = False,
    lr_scale_mode: str = 'linear',
    find_unused_parameters: bool = False,
    use_streaming: bool = False,
    data_dir: str | None = None,
    sampling_weights: str = 'uniform',
    multi_player: bool = False,
    num_players: int = 2,
    model_version: str = 'v2',
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
    heartbeat_file: str | None = None,
    heartbeat_interval: float = 30.0,
    # 2024-12 Training Improvements (accept but log for now)
    spectral_norm: bool = False,
    cyclic_lr: bool = False,
    cyclic_lr_period: int = 5,
    mixed_precision: bool = False,
    amp_dtype: str = 'bfloat16',
    value_whitening: bool = False,
    value_whitening_momentum: float = 0.99,
    ema: bool = False,
    ema_decay: float = 0.999,
    stochastic_depth: bool = False,
    stochastic_depth_prob: float = 0.1,
    adaptive_warmup: bool = False,
    # Dec 28, 2025: Enabled by default to reduce overfitting and reach 2000+ Elo
    hard_example_mining: bool = True,  # Focus on difficult examples
    hard_example_top_k: float = 0.3,
    # Outcome-weighted policy loss (2025-12)
    # Weights policy loss by game outcome: winner's moves → higher weight, loser's → lower
    # Inspired by EBMO outcome-contrastive loss for improved move quality learning
    # Dec 28, 2025: Enabled by default to improve move quality learning
    enable_outcome_weighted_policy: bool = True,  # Learn from winning moves
    outcome_weight_scale: float = 0.5,  # How much to scale by outcome (0=no effect, 1=full)
    auto_tune_batch_size: bool = True,  # Enabled by default for 15-30% better throughput
    # January 2026: Conservative memory targeting (50% default, 35% safe mode)
    target_memory_fraction: float | None = None,  # None = use config default (50% or 35% safe mode)
    safe_mode: bool = False,  # Extra conservative batch sizing (35% memory target)
    track_calibration: bool = False,
    # 2024-12 Hot Data Buffer and Integrated Enhancements
    use_hot_data_buffer: bool = False,
    hot_buffer_size: int = 10000,
    hot_buffer_mix_ratio: float = 0.3,
    external_hot_buffer: Any | None = None,  # Pre-populated HotDataBuffer from caller
    use_integrated_enhancements: bool = True,  # December 2025: Enable by default for Elo improvement
    # Dec 28, 2025: Enabled curriculum and augmentation by default to reduce overfitting
    enable_curriculum: bool = True,  # Progressive difficulty during training
    enable_augmentation: bool = True,  # Board symmetry augmentation
    enable_elo_weighting: bool = True,  # December 2025: Enable for sample prioritization (+20-35 Elo)
    enable_auxiliary_tasks: bool = True,  # December 2025: Enable for multi-task learning (+5-15 Elo)
    enable_batch_scheduling: bool = False,
    enable_background_eval: bool = True,  # December 2025: Enable for real-time Elo feedback (+30-50 Elo)
    # Policy label smoothing (2025-12)
    policy_label_smoothing: float = 0.0,
    # Data validation (2025-12)
    validate_data: bool = True,
    fail_on_invalid_data: bool = False,
    # Fault tolerance (2025-12)
    enable_circuit_breaker: bool = True,
    enable_anomaly_detection: bool = True,
    gradient_clip_mode: str = 'adaptive',
    gradient_clip_max_norm: float = 1.0,
    anomaly_spike_threshold: float = 3.0,
    anomaly_gradient_threshold: float = 100.0,
    enable_graceful_shutdown: bool = True,
    # Regularization (2025-12)
    dropout: float = 0.08,
    # Quality-aware data discovery (2025-12)
    discover_synced_data: bool = False,
    min_quality_score: float = 0.0,
    _include_local_data: bool = True,
    _include_nfs_data: bool = True,
    # Learning rate finder (2025-12)
    find_lr: bool = False,
    lr_finder_min: float = 1e-7,
    lr_finder_max: float = 1.0,
    lr_finder_iterations: int = 100,
    # GNN support (2025-12)
    model_type: str = "cnn",  # "cnn", "gnn", or "hybrid"
    # Training data freshness check (2025-12)
    # MANDATORY BY DEFAULT - prevents 95% of stale data training incidents
    # Phase 1.5 of improvement plan: fail early if data is stale
    skip_freshness_check: bool = False,  # Default: check IS enabled
    max_data_age_hours: float = 2000.0,  # Default: data must be <2000 hours old (relaxed)
    allow_stale_data: bool = False,      # Default: FAIL on stale data (not warn)
    # Stale fallback for 48-hour autonomous operation (December 2025)
    # Allows training to proceed with stale data after sync failures or timeout
    disable_stale_fallback: bool = False,  # If True, no automatic fallback
    max_sync_failures: int = 5,            # Failures before fallback allowed
    max_sync_duration: float = 2700.0,     # Seconds (45 min) before fallback
    # Checkpoint averaging (2025-12)
    # Averages last N checkpoints at end of training for +10-20 Elo improvement
    enable_checkpoint_averaging: bool = True,
    num_checkpoints_to_average: int = 5,
    # Best checkpoint selection on overfitting (January 2026)
    # When overfitting detected (val_loss/train_loss > threshold), use best checkpoint
    # instead of averaged. This prevents averaged overfit checkpoints from degrading quality.
    prefer_best_on_overfit: bool = True,
    overfit_divergence_threshold: float = 0.5,  # 50% divergence triggers best checkpoint
    # Quality-weighted training (2025-12) - resurrected from ebmo_network.py
    # December 2025: Enabled by default to improve training signal quality
    # Quality weighting focuses learning on high-quality MCTS-derived moves
    enable_quality_weighting: bool = True,
    quality_weight_blend: float = 0.5,
    quality_ranking_weight: float = 0.1,
    # Auto-promotion after training (January 2026)
    # Runs gauntlet evaluation and promotes if criteria met (Elo parity OR win rate floors)
    auto_promote: bool = False,
    auto_promote_games: int = 30,
    auto_promote_sync: bool = True,
    # Gradient checkpointing (January 2026)
    # Trades compute for memory - recomputes activations during backward pass
    # Enables training large models (e.g., hexagonal) on memory-constrained GPUs
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    """
    Train the RingRift neural network model.

    Args:
        config: Training configuration
        data_path: Path(s) to training data (.npz file or list of files)
        save_path: Path to save the best model weights
        early_stopping_patience: Number of epochs without loss improvement before
            stopping (0 to disable early stopping)
        elo_early_stopping_patience: Number of epochs without Elo improvement
            before stopping (works in conjunction with loss patience when both
            are tracked; 0 to disable Elo-based early stopping)
        elo_min_improvement: Minimum Elo improvement (default 5.0) to reset
            the Elo patience counter
        checkpoint_dir: Directory for saving periodic checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        warmup_epochs: Number of epochs for LR warmup (0 to disable)
        lr_scheduler: Type of LR scheduler:
            - 'none': No scheduling (constant LR after warmup)
            - 'step': Step decay by 0.5 every 10 epochs
            - 'cosine': CosineAnnealingLR over remaining epochs
            - 'cosine-warm-restarts': CosineAnnealingWarmRestarts
        lr_min: Minimum learning rate for cosine annealing (default: 1e-6)
        lr_t0: T_0 for CosineAnnealingWarmRestarts (initial restart period)
        lr_t_mult: T_mult for CosineAnnealingWarmRestarts (period multiplier)
        resume_path: Path to checkpoint to resume training from
        augment_hex_symmetry: Enable D6 symmetry augmentation for hex boards
        distributed: Enable distributed training with DDP
        local_rank: Local rank for distributed training (set by torchrun)
        scale_lr: Whether to scale learning rate with world size
        lr_scale_mode: LR scaling mode ('linear' or 'sqrt')
        find_unused_parameters: Enable find_unused_parameters for DDP
        use_streaming: Use StreamingDataLoader for large datasets
        data_dir: Directory containing multiple .npz files (for streaming)
        sampling_weights: Position sampling strategy for non-streaming data:
            'uniform', 'late_game', 'phase_emphasis', or 'combined'
        use_integrated_enhancements: Enable IntegratedTrainingManager for advanced features
        enable_curriculum: Enable curriculum learning (difficulty progression)
        enable_augmentation: Enable data augmentation (symmetry transforms)
        enable_elo_weighting: Enable Elo-based sample weighting
        enable_auxiliary_tasks: Enable auxiliary prediction tasks (outcome classification)
            Requires model support for return_features=True
        enable_batch_scheduling: Enable dynamic batch size scheduling (linear ramp-up)
        enable_background_eval: Enable background Elo evaluation during training
            Provides early stopping based on Elo tracking
        find_lr: Run learning rate finder before training to find optimal LR
        lr_finder_min: Minimum LR for range test (default 1e-7)
        lr_finder_max: Maximum LR for range test (default 1.0)
        lr_finder_iterations: Number of iterations for LR range test (default 100)
    """
    # Resolve optional parameters using TrainConfigResolver (December 2025)
    # Provides consistent precedence: explicit param > config attr > default
    resolved = resolve_train_config(
        config=config,
        early_stopping_patience=early_stopping_patience,
        elo_early_stopping_patience=elo_early_stopping_patience,
        elo_min_improvement=elo_min_improvement,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        distributed=distributed,
        local_rank=local_rank,
        num_players=num_players,
    )

    # Extract resolved values for backward compatibility
    early_stopping_patience = resolved.early_stopping_patience
    elo_early_stopping_patience = resolved.elo_early_stopping_patience
    elo_min_improvement = resolved.elo_min_improvement
    warmup_epochs = resolved.warmup_epochs
    lr_scheduler = resolved.lr_scheduler
    lr_min = resolved.lr_min

    # Set up distributed training if enabled
    if distributed:
        # Setup distributed process group
        setup_distributed(local_rank)
        world_size = get_world_size()

        # Seed with rank offset for different random state per process
        seed_everything(config.seed, rank_offset=True)

        # Scale learning rate if requested
        if scale_lr:
            config.learning_rate = scale_learning_rate(
                config.learning_rate, world_size, scale_type=lr_scale_mode
            )
            if is_main_process():
                logger.info(
                    f"Scaled learning rate to {config.learning_rate:.6f} "
                    f"({lr_scale_mode} scaling with world_size={world_size})"
                )
    else:
        seed_all(config.seed)

    # ==========================================================================
    # Pre-Training Validation (extracted to train_pre_validation.py)
    # ==========================================================================
    run_pre_training_validation(
        data_path=data_path,
        config=config,
        num_players=num_players,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        skip_freshness_check=skip_freshness_check,
        max_data_age_hours=max_data_age_hours,
        allow_stale_data=allow_stale_data,
        disable_stale_fallback=disable_stale_fallback,
        max_sync_failures=max_sync_failures,
        max_sync_duration=max_sync_duration,
        validate_data=validate_data,
        fail_on_invalid_data=fail_on_invalid_data,
        use_streaming=use_streaming,
        check_freshness_sync=check_freshness_sync,
        validate_npz_structure_fn=validate_npz_structure,
        validate_npz_file_fn=validate_npz_file,
        verify_npz_checksums_fn=verify_npz_checksums,
        should_allow_stale_training_fn=should_allow_stale_training,
        HAS_FRESHNESS_CHECK=HAS_FRESHNESS_CHECK,
        HAS_NPZ_STRUCTURE_VALIDATION=HAS_NPZ_STRUCTURE_VALIDATION,
        HAS_DATA_VALIDATION=HAS_DATA_VALIDATION,
        HAS_CHECKSUM_VERIFICATION=HAS_CHECKSUM_VERIFICATION,
        HAS_STALE_FALLBACK=HAS_STALE_FALLBACK,
        DataEventType=DataEventType,
    )

    # ==========================================================================
    # Mandatory NPZ validation (Feb 2026)
    # Catches empty, corrupt, or mismatched data before model/optimizer creation
    # ==========================================================================
    if not use_streaming:
        npz_paths = data_path if isinstance(data_path, list) else [data_path]
        board_type_str = (
            config.board_type.value
            if hasattr(config.board_type, "value")
            else str(config.board_type)
        )
        for p in npz_paths:
            if p and os.path.exists(p):
                try:
                    validate_training_data(Path(p), board_type_str, num_players)
                except (ValueError, FileNotFoundError) as e:
                    logger.error(f"Training data validation failed: {e}")
                    raise

    # Device configuration
    if distributed:
        # In distributed mode, use the local_rank device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        if is_main_process():
            logger.info(
                f"Distributed training on device: {device} "
                f"(rank {get_rank()}/{get_world_size()})"
            )
    else:
        # Standard single-device selection
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

    # Log 2024-12 Training Improvements status
    improvements_enabled = []
    if spectral_norm:
        improvements_enabled.append("spectral_norm")
    if cyclic_lr:
        improvements_enabled.append(f"cyclic_lr(period={cyclic_lr_period})")
    if mixed_precision:
        improvements_enabled.append(f"mixed_precision({amp_dtype})")
    if value_whitening:
        improvements_enabled.append("value_whitening")
    if ema:
        improvements_enabled.append(f"ema(decay={ema_decay})")
    if stochastic_depth:
        improvements_enabled.append(f"stochastic_depth(p={stochastic_depth_prob})")
    if adaptive_warmup:
        improvements_enabled.append("adaptive_warmup")
    if hard_example_mining:
        improvements_enabled.append(f"hard_example_mining(top_k={hard_example_top_k})")

    if improvements_enabled:
        logger.info(f"2024-12 Training Improvements enabled: {', '.join(improvements_enabled)}")

    # Initialize hot data buffer if requested
    # NOTE (2025-12): HotDataBuffer enables mixing fresh selfplay games with static
    # training data for online/streaming training. For the buffer to receive games:
    # 1. Pass external_hot_buffer (pre-populated buffer from unified_orchestrator), OR
    # 2. Subscribe to EventBus NEW_GAME_AVAILABLE events and call hot_buffer.add_game()
    hot_buffer = None
    if external_hot_buffer is not None:
        # Use externally-provided buffer (e.g., from unified_orchestrator)
        hot_buffer = external_hot_buffer
        current_samples = getattr(hot_buffer, 'total_samples', 0)
        logger.info(
            f"Using external hot data buffer with {current_samples} samples "
            f"(mix_ratio={hot_buffer_mix_ratio})"
        )
    elif use_hot_data_buffer and HAS_HOT_DATA_BUFFER:
        hot_buffer = HotDataBuffer(
            max_size=hot_buffer_size,
            training_threshold=config.batch_size * 5,
        )
        logger.info(f"Hot data buffer enabled (size={hot_buffer_size}, mix_ratio={hot_buffer_mix_ratio})")
        logger.info(
            "Note: Hot buffer requires external game population via add_game() "
            "or event bus subscription to receive selfplay games"
        )
    elif use_hot_data_buffer and not HAS_HOT_DATA_BUFFER:
        logger.warning("Hot data buffer requested but not available (import failed)")

    # Configure quality bridge for quality-aware data selection (2025-12)
    # This integrates quality scores from the sync system with training data loading
    if HAS_QUALITY_BRIDGE:
        try:
            quality_bridge = get_quality_bridge()
            num_refreshed = quality_bridge.refresh(force=True)
            logger.info(f"Quality bridge initialized with {num_refreshed} game quality scores")

            # Configure hot buffer with quality lookups if available
            if hot_buffer is not None:
                configured = quality_bridge.configure_hot_data_buffer(hot_buffer)
                if configured > 0:
                    logger.info(f"Hot buffer configured with {configured} quality scores")
        except (ImportError, AttributeError, RuntimeError, ValueError) as e:
            # ImportError: quality bridge module dependencies missing
            # AttributeError: API changes or missing methods
            # RuntimeError: quality bridge initialization errors
            # ValueError: invalid quality score data
            logger.warning(f"Failed to initialize quality bridge: {e}")

    # Initialize integrated enhancements if requested
    enhancements_manager = None
    if use_integrated_enhancements and HAS_INTEGRATED_ENHANCEMENTS:
        enh_config = IntegratedEnhancementsConfig(
            curriculum_enabled=enable_curriculum,
            augmentation_enabled=enable_augmentation,
            elo_weighting_enabled=enable_elo_weighting,
            auxiliary_tasks_enabled=enable_auxiliary_tasks,
            batch_scheduling_enabled=enable_batch_scheduling,
            background_eval_enabled=enable_background_eval,
            # December 2025: Enable real game evaluation for accurate Elo tracking
            eval_use_real_games=enable_background_eval,  # Use real games when bg eval is on
            eval_board_type=config.board_type,
        )
        enhancements_manager = IntegratedTrainingManager(
            config=enh_config,
            model=None,  # Will be set after model creation
            board_type=config.board_type.value,
        )
        logger.info(
            f"Integrated enhancements enabled: "
            f"curriculum={enable_curriculum}, augmentation={enable_augmentation}, "
            f"elo_weighting={enable_elo_weighting}, auxiliary_tasks={enable_auxiliary_tasks}, "
            f"batch_scheduling={enable_batch_scheduling}, background_eval={enable_background_eval}"
        )
    elif use_integrated_enhancements and not HAS_INTEGRATED_ENHANCEMENTS:
        logger.warning("Integrated enhancements requested but not available (import failed)")

    # Initialize checkpoint averager for end-of-training averaging (2025-12)
    # Averages last N checkpoints for +10-20 Elo improvement with reduced variance
    checkpoint_averager = None
    if enable_checkpoint_averaging and CheckpointAverager is not None:
        checkpoint_averager = CheckpointAverager(
            num_checkpoints=num_checkpoints_to_average,
            checkpoint_dir=Path(checkpoint_dir),
            keep_on_disk=True,  # Save memory by keeping checkpoints on disk
        )
        logger.info(
            f"[Checkpoint Averaging] Enabled: will average last {num_checkpoints_to_average} checkpoints at end of training"
        )
    elif enable_checkpoint_averaging and CheckpointAverager is None:
        logger.warning("[Checkpoint Averaging] Requested but CheckpointAverager not available (import failed)")

    # Initialize hard example miner for curriculum learning (2025-12)
    # Tracks per-sample losses to focus training on difficult examples
    hard_example_miner: HardExampleMiner | None = None
    if hard_example_mining and HAS_HARD_EXAMPLE_MINING:
        hard_example_miner = HardExampleMiner(
            buffer_size=50000,  # Track up to 50K examples
            hard_fraction=hard_example_top_k,  # Fraction of batch that should be hard examples
            loss_threshold_percentile=80.0,  # Top 20% by loss are "hard"
            uncertainty_weight=0.3,  # 30% weight on policy uncertainty
            decay_rate=0.99,  # Decay old hardness scores
            min_samples_before_mining=5000,  # Need 5K samples before mining starts
            max_times_sampled=10,  # Cap oversampling of any single example
        )
        logger.info(
            f"[Hard Example Mining] Enabled: hard_fraction={hard_example_top_k}, "
            f"buffer_size=50000, min_samples_before_mining=5000"
        )
    elif hard_example_mining and not HAS_HARD_EXAMPLE_MINING:
        logger.warning("[Hard Example Mining] Requested but HardExampleMiner not available (import failed)")

    # Initialize unified training enhancements facade (2025-12)
    # Consolidates hard example mining, per-sample loss, curriculum LR, freshness weighting
    # This is the recommended way to use all training enhancements together (+80-165 Elo)
    training_facade: TrainingEnhancementsFacade | None = None
    if HAS_TRAINING_FACADE and TrainingEnhancementsFacade is not None:
        facade_config = FacadeConfig(
            enable_hard_mining=hard_example_mining,
            hard_fraction=hard_example_top_k,
            hard_buffer_size=50000,
            hard_min_samples_before_mining=5000,
            track_per_sample_loss=True,
            enable_curriculum_lr=enable_curriculum,
            curriculum_lr_min_scale=0.8,
            curriculum_lr_max_scale=1.2,
            enable_freshness_weighting=enable_elo_weighting,
            freshness_decay_hours=24.0,
            policy_weight=config.policy_weight,
        )
        training_facade = TrainingEnhancementsFacade(config=facade_config)
        training_facade.set_total_epochs(config.epochs_per_iter)
        logger.info(
            f"[Training Facade] Enabled: hard_mining={hard_example_mining}, "
            f"curriculum_lr={enable_curriculum}, freshness={enable_elo_weighting}"
        )
    elif hard_example_mining and not HAS_TRAINING_FACADE:
        logger.info("[Training Facade] Not available, falling back to standalone HardExampleMiner")

    # Quality-weighted training (2025-12) - resurrected from ebmo_network.py
    # Weights samples by MCTS visit counts for better learning signal
    quality_trainer: QualityWeightedTrainer | None = None
    if enable_quality_weighting and HAS_QUALITY_WEIGHTING:
        quality_trainer = QualityWeightedTrainer(
            quality_weight=quality_weight_blend,
            ranking_weight=quality_ranking_weight,
            ranking_margin=0.5,  # Default margin for ranking loss
            min_quality_weight=0.1,
            temperature=1.0,
        )
        logger.info(
            f"[Quality Weighting] Enabled: blend={quality_weight_blend:.2f}, "
            f"ranking_weight={quality_ranking_weight:.2f}"
        )
    elif enable_quality_weighting and not HAS_QUALITY_WEIGHTING:
        logger.warning("[Quality Weighting] Requested but module not available")

    # Mixed precision setup (CUDA-only for now)
    amp_enabled = bool(mixed_precision and device.type == 'cuda')
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
    amp_torch_dtype = dtype_map.get(amp_dtype, torch.bfloat16)
    use_grad_scaler = bool(amp_enabled and amp_torch_dtype == torch.float16)
    # GradScaler API changed in PyTorch 2.4+: torch.amp.GradScaler vs torch.cuda.amp.GradScaler
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    if amp_enabled:
        logger.info(f"Mixed precision training enabled with {amp_dtype}")

    # Gradient surgery for multi-task learning (2025-12)
    # Projects conflicting gradients between value/policy heads to prevent oscillation
    gradient_surgeon: GradientSurgeon | None = None
    use_gradient_surgery = getattr(config, 'enable_gradient_surgery', False)
    if use_gradient_surgery:
        if use_grad_scaler:
            logger.warning(
                "Gradient surgery disabled: incompatible with FP16 GradScaler. "
                "Use bfloat16 mixed precision or disable mixed precision."
            )
            use_gradient_surgery = False
        elif getattr(config, 'gradient_accumulation_steps', 1) > 1:
            logger.warning(
                "Gradient surgery disabled: incompatible with gradient accumulation. "
                "Set gradient_accumulation_steps=1 to use gradient surgery."
            )
            use_gradient_surgery = False
        else:
            gradient_surgeon = GradientSurgeon(GradientSurgeryConfig(
                enabled=True,
                method="pcgrad",
                conflict_threshold=0.0,
            ))
            logger.info("Gradient surgery (PCGrad) enabled for multi-task learning")

    # Initialize dashboard metrics collector for persistent metric storage (2025-12)
    metrics_collector = None
    if HAS_METRICS_COLLECTOR and (not distributed or is_main_process()):
        try:
            metrics_collector = MetricsCollector()
            logger.info("Dashboard metrics collector initialized")
        except (ImportError, RuntimeError, OSError) as e:
            # ImportError: metrics collector dependencies missing
            # RuntimeError: initialization failures
            # OSError: file/database access errors
            logger.warning(f"Could not initialize metrics collector: {e}")

    # ==========================================================================
    # Dataset Metadata Inference (extracted to train_dataset_inference.py)
    # ==========================================================================
    _ds_result = infer_dataset_metadata(
        data_path=data_path,
        config=config,
        num_players=num_players,
        model_version=model_version,
        multi_player=multi_player,
        use_streaming=use_streaming,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        resume_path=resume_path,
        num_filters=num_filters,
        num_res_blocks=num_res_blocks,
        device=device,
        BoardType=BoardType,
        HEX_BOARD_SIZE=HEX_BOARD_SIZE,
        HEX8_BOARD_SIZE=HEX8_BOARD_SIZE,
        MAX_PLAYERS=MAX_PLAYERS,
        get_policy_size_for_board=get_policy_size_for_board,
        normalize_board_type=normalize_board_type,
        validate_hex_policy_indices=validate_hex_policy_indices,
        detect_tier_from_checkpoint=detect_tier_from_checkpoint,
    )
    board_size = _ds_result.board_size
    policy_size = _ds_result.policy_size
    hex_in_channels = _ds_result.hex_in_channels
    hex_num_players = _ds_result.hex_num_players
    use_hex_model = _ds_result.use_hex_model
    use_hex_v3 = _ds_result.use_hex_v3
    use_hex_v4 = _ds_result.use_hex_v4
    use_hex_v5 = _ds_result.use_hex_v5
    use_hex_v5_large = _ds_result.use_hex_v5_large
    detected_num_heuristics = _ds_result.detected_num_heuristics
    config_feature_version = _ds_result.config_feature_version
    hex_radius = _ds_result.hex_radius

    # (Dataset metadata inference handled above by infer_dataset_metadata)

    # ===================================================================
    # Architecture-Data Compatibility Validation (December 2025)
    # Validates that the training data is compatible with the selected
    # model architecture, especially for heuristic-dependent models.
    # ===================================================================
    def _validate_architecture_data_compatibility() -> None:
        """Validate training data is compatible with selected architecture.

        This catches errors early before expensive model initialization:
        - V5-heavy requires at least 21 heuristic features (fast heuristics)
        - V5-heavy-large/xl require all 49 heuristic features (full heuristics)

        Raises:
            ValueError: If data is incompatible with selected architecture
        """
        nonlocal detected_num_heuristics

        # Only validate for architectures that require heuristics
        v5_heavy_versions = ('v5', 'v5-gnn', 'v5-heavy', 'v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl')
        if not (use_hex_v5 or model_version in v5_heavy_versions):
            return

        # Import encoder registry to get requirements
        try:
            from app.training.encoder_registry import get_encoder_config
            board_type_name = config.board_type.name if hasattr(config.board_type, 'name') else str(config.board_type)
            # Map to encoder registry key
            if model_version in ('v5-heavy-large', 'v6'):
                version_key = "v5-heavy"  # Uses same encoder as v5-heavy
            else:
                version_key = "v5-heavy"
            encoder_config = get_encoder_config(board_type_name, version_key)
        except (ValueError, ImportError):
            # Registry doesn't have this config, skip validation
            return

        # Check if architecture requires heuristics
        if not encoder_config.requires_heuristics:
            return

        min_required = encoder_config.min_heuristic_features
        actual_heuristics = detected_num_heuristics or 0

        if actual_heuristics < min_required:
            # Map model version to human-readable name
            version_names = {
                "v6": "V5-Heavy-Large (deprecated alias)",
                "v6-xl": "V5-Heavy-XL (deprecated alias)",
                "v5-heavy-large": "V5-Heavy-Large",
                "v5-heavy-xl": "V5-Heavy-XL",
            }
            version_name = version_names.get(model_version, "V5-Heavy")
            raise ValueError(
                f"\n{'='*70}\n"
                f"ARCHITECTURE-DATA COMPATIBILITY ERROR\n"
                f"{'='*70}\n\n"
                f"Model: {version_name} (--model-version {model_version})\n"
                f"  - Requires at least {min_required} heuristic features\n\n"
                f"Dataset: {data_path_str if isinstance(data_path, str) else data_path[0] if data_path else 'unknown'}\n"
                f"  - Has {actual_heuristics} heuristic features\n\n"
                f"SOLUTIONS:\n"
                f"  1. Re-export data with --full-heuristics flag:\n"
                f"     python scripts/export_replay_dataset.py --full-heuristics ...\n"
                f"  2. Use a different architecture that doesn't require heuristics:\n"
                f"     --model-version v2 or --model-version v4\n"
                f"{'='*70}"
            )

        if not distributed or is_main_process():
            logger.info(
                f"Architecture validation passed: {model_version} requires {min_required} "
                f"heuristics, dataset has {actual_heuristics}"
            )

    # Run architecture-data compatibility check
    if use_hex_model or use_hex_v5 or model_version in ('v5', 'v5-gnn', 'v5-heavy', 'v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'):
        _validate_architecture_data_compatibility()

    if not distributed or is_main_process():
        if use_hex_model or use_hex_v5_large:
            if use_hex_v5_large:
                hex_model_name = "HexNeuralNet_v5_Heavy (large)"
            elif use_hex_v5:
                hex_model_name = "HexNeuralNet_v5_Heavy"
            elif use_hex_v4:
                hex_model_name = "HexNeuralNet_v4"
            elif use_hex_v3:
                if model_version == 'v3-flat':
                    hex_model_name = "HexNeuralNet_v3_Flat"
                else:
                    hex_model_name = "HexNeuralNet_v3 (spatial policy)"
            else:
                hex_model_name = "HexNeuralNet_v2"
            logger.info(
                f"Initializing {hex_model_name} with board_size={board_size}, "
                f"policy_size={policy_size}, in_channels={hex_in_channels}, "
                f"num_players={hex_num_players}"
            )
        else:
            logger.info(
                f"Initializing RingRiftCNN with board_size={board_size}, "
                f"policy_size={policy_size}"
            )

    # Determine model architecture size (allow CLI override for scaling up)
    # Default: 11 blocks / 160 filters for v5, 13 blocks / 128 filters for v4,
    # 12 blocks / 192 filters for v3/hex, 6 blocks / 96 filters for v2
    # Note: v5-heavy-large/xl use factory defaults from v5_heavy_large.py
    if use_hex_v5 or model_version in ('v5', 'v5-gnn', 'v5-heavy'):
        effective_blocks = num_res_blocks if num_res_blocks is not None else 11  # 6 SE + 5 attention
        effective_filters = num_filters if num_filters is not None else 160  # v5 default
    elif model_version in ('v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'):
        # v5-heavy-large/xl use configs from v5_heavy_large.py (256-320 filters)
        # Don't override effective_blocks/filters - factory handles defaults
        effective_blocks = num_res_blocks if num_res_blocks is not None else 20  # 10 SE + 10 attention
        effective_filters = num_filters if num_filters is not None else 256  # Large default
    elif use_hex_v4 or model_version == 'v4':
        effective_blocks = num_res_blocks if num_res_blocks is not None else 13  # NAS optimal
        effective_filters = num_filters if num_filters is not None else 128  # NAS optimal
    elif model_version == 'v3' or use_hex_model:
        effective_blocks = num_res_blocks if num_res_blocks is not None else 12
        effective_filters = num_filters if num_filters is not None else 192
    else:
        effective_blocks = num_res_blocks if num_res_blocks is not None else 6
        effective_filters = num_filters if num_filters is not None else 96

    # Log architecture size if non-default
    if (num_res_blocks is not None or num_filters is not None) and (not distributed or is_main_process()):
        logger.info(
            f"Using custom architecture: {effective_blocks} residual blocks, "
            f"{effective_filters} filters"
        )

    # Value head dimension validation helper (December 2025)
    # Prevents training with wrong output dimensions for multiplayer models
    def _validate_model_value_head(model: nn.Module, expected_players: int, context: str = "") -> None:
        """Validate model value head matches expected player count.

        This prevents training with mismatched value head dimensions, which was
        a root cause of cluster model failures (hex8_4p, square19_3p regressions).

        Args:
            model: Neural network model to validate
            expected_players: Expected number of players (2, 3, or 4)
            context: Description of when validation is happening (for error messages)

        Raises:
            ValueError: If model value head doesn't match expected player count
        """
        ctx = f" ({context})" if context else ""

        # Check model's num_players attribute if present
        if hasattr(model, 'num_players'):
            model_players = model.num_players
            if model_players != expected_players:
                raise ValueError(
                    f"Model value head mismatch{ctx}: model.num_players={model_players} "
                    f"but training expects {expected_players} players. "
                    f"Use transfer_2p_to_4p.py to resize value head."
                )

        # Check value head output dimension
        # v4/v5-heavy use 3-layer value head (fc1 → fc2 → fc3), others use 2-layer (fc1 → fc2)
        # Check the final layer that outputs to num_players
        final_value_layer = None
        if hasattr(model, 'value_fc3'):
            # v4/v5-heavy: value_fc3 is the final output layer
            final_value_layer = model.value_fc3
        elif hasattr(model, 'value_fc2'):
            # v2/v3: value_fc2 is the final output layer
            final_value_layer = model.value_fc2

        if final_value_layer is not None:
            out_features = final_value_layer.out_features
            if out_features != expected_players:
                layer_name = 'value_fc3' if hasattr(model, 'value_fc3') else 'value_fc2'
                raise ValueError(
                    f"{layer_name} output mismatch{ctx}: out_features={out_features} "
                    f"but training expects {expected_players} players. "
                    f"Use transfer_2p_to_4p.py to resize value head."
                )

        # Check value_head output dimension (used in some architectures)
        if hasattr(model, 'value_head'):
            # value_head might be a Sequential or Linear
            value_head = model.value_head
            if hasattr(value_head, 'out_features'):
                out_features = value_head.out_features
                if out_features != expected_players:
                    raise ValueError(
                        f"value_head output mismatch{ctx}: out_features={out_features} "
                        f"but training expects {expected_players} players."
                    )
            elif isinstance(value_head, nn.Sequential):
                # Check last layer of Sequential
                last_layer = list(value_head.modules())[-1]
                if hasattr(last_layer, 'out_features'):
                    out_features = last_layer.out_features
                    if out_features != expected_players:
                        raise ValueError(
                            f"value_head output mismatch{ctx}: last layer out_features={out_features} "
                            f"but training expects {expected_players} players."
                        )

    # ==========================================================================
    # Model Creation (extracted to train_model_factory.py)
    # ==========================================================================
    model = create_training_model(
        config=config,
        model_version=model_version,
        model_type=model_type,
        board_size=board_size,
        policy_size=policy_size,
        num_players=num_players,
        hex_in_channels=hex_in_channels,
        hex_radius=hex_radius,
        hex_num_players=hex_num_players,
        use_hex_model=use_hex_model,
        use_hex_v3=use_hex_v3,
        use_hex_v4=use_hex_v4,
        use_hex_v5=use_hex_v5,
        use_hex_v5_large=use_hex_v5_large,
        detected_num_heuristics=detected_num_heuristics,
        effective_blocks=effective_blocks,
        effective_filters=effective_filters,
        multi_player=multi_player,
        dropout=dropout,
        config_feature_version=config_feature_version,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        HexNeuralNet_v2=HexNeuralNet_v2,
        HexNeuralNet_v3=HexNeuralNet_v3,
        HexNeuralNet_v3_Flat=HexNeuralNet_v3_Flat,
        HexNeuralNet_v4=HexNeuralNet_v4,
        HexNeuralNet_v5_Heavy=HexNeuralNet_v5_Heavy,
        RingRiftCNN_v2=RingRiftCNN_v2,
        RingRiftCNN_v3=RingRiftCNN_v3,
        RingRiftCNN_v3_Flat=RingRiftCNN_v3_Flat,
        MAX_PLAYERS=MAX_PLAYERS,
    )
    model.to(device)

    # Enable gradient checkpointing for memory-efficient training (January 2026)
    # Trades ~20-30% compute overhead for ~40-60% memory savings
    if gradient_checkpointing:
        try:
            from app.training.gradient_checkpointing import GradientCheckpointing
            gc_manager = GradientCheckpointing(model)
            gc_manager.enable()
            if not distributed or is_main_process():
                logger.info("[GradientCheckpointing] Enabled - trading compute for memory")
        except ImportError as e:
            logger.warning(f"[GradientCheckpointing] Failed to enable: {e}")

    # Validate value head dimension after model creation (December 2025)
    # This catches mismatches early before any training starts
    _validate_model_value_head(model, num_players, "after model creation")

    # Initialize enhancements manager with model reference
    if enhancements_manager is not None:
        enhancements_manager.model = model
        enhancements_manager.initialize_all()

    # Auto-tune batch size if requested (overrides config.batch_size)
    # January 2026: Use fast GPU memory heuristic by default for optimal batch size
    # Conservative memory targeting: 50% default, 35% safe mode (was 70%)
    if auto_tune_batch_size and str(device).startswith('cuda'):
        try:
            from app.training.config import (
                get_optimal_batch_size_from_gpu_memory,
                get_gpu_scaling_config,
            )
            original_batch = config.batch_size

            # Count model parameters for memory estimation
            model_params = sum(p.numel() for p in model.parameters())

            # Get feature channels from model or use defaults
            try:
                feature_channels = model.in_channels if hasattr(model, 'in_channels') else 56
            except Exception:
                feature_channels = 56

            # Determine effective memory fraction
            gpu_config = get_gpu_scaling_config()
            effective_memory_fraction = target_memory_fraction
            if effective_memory_fraction is None:
                if safe_mode:
                    effective_memory_fraction = gpu_config.safe_mode_memory_fraction
                # else: get_optimal_batch_size_from_gpu_memory uses config defaults (50% or 35%)

            mode_str = "[SAFE MODE]" if safe_mode else ""
            logger.info(f"[AutoBatchSize]{mode_str} Calculating optimal batch size from GPU memory...")
            logger.info(f"[AutoBatchSize] Model params: {model_params:,}, board_size: {board_size}, num_players: {num_players}")
            if effective_memory_fraction:
                logger.info(f"[AutoBatchSize] Memory target: {effective_memory_fraction*100:.0f}%")

            config.batch_size = get_optimal_batch_size_from_gpu_memory(
                model_params=model_params,
                feature_channels=feature_channels,
                board_size=board_size,
                num_players=num_players,
                target_memory_fraction=effective_memory_fraction,  # None = use config (50% or 35% safe mode)
                min_batch=64,
                max_batch=4096,  # Reduced from 8192 for safety
                config=gpu_config,
            )
            logger.info(f"[AutoBatchSize] Auto-tuned batch size: {config.batch_size} (was {original_batch})")
        except (RuntimeError, ValueError, ImportError) as e:
            # RuntimeError: CUDA/GPU errors during tuning
            # ValueError: invalid batch size values
            # ImportError: auto-tuning module missing
            logger.warning(f"[AutoBatchSize] Batch size auto-tuning failed: {e}. Using original batch size.")

    # Auto-detect canonical model for iterative training (January 2026)
    # If no init_weights specified and not resuming, use the existing canonical model
    # This enables continuous self-improvement: train on new data, improving existing model
    # IMPORTANT: Only auto-select if encoder versions are compatible (Jan 2026 fix)
    if init_weights_path is None and not os.path.exists(save_path):
        board_type_str = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)
        canonical_path = f"models/canonical_{board_type_str}_{num_players}p.pth"
        if os.path.exists(canonical_path):
            # Check encoder compatibility BEFORE auto-selecting
            try:
                from app.ai.neural_net.architecture_registry import get_encoder_version_from_checkpoint
                canonical_encoder = get_encoder_version_from_checkpoint(canonical_path)

                # Determine data encoder version
                data_encoder = None
                if hex_in_channels == 40:
                    data_encoder = "v2"
                elif hex_in_channels == 64:
                    data_encoder = "v3"
                elif hex_in_channels == 56:
                    data_encoder = "v3"  # V5-heavy compatible with v3

                if canonical_encoder and data_encoder and canonical_encoder == data_encoder:
                    init_weights_path = canonical_path
                    if not distributed or is_main_process():
                        logger.info(f"[AutoInitWeights] Using canonical model as starting point: {canonical_path}")
                else:
                    if not distributed or is_main_process():
                        logger.info(
                            f"[AutoInitWeights] Canonical model {canonical_path} has encoder {canonical_encoder}, "
                            f"but data uses {data_encoder}. Training from scratch instead."
                        )
            except Exception as e:
                if not distributed or is_main_process():
                    logger.warning(f"[AutoInitWeights] Could not check canonical model compatibility: {e}. Training from scratch.")
        else:
            if not distributed or is_main_process():
                logger.info(f"[AutoInitWeights] No canonical model found at {canonical_path}, training from scratch")

    # January 2026: FAIL-FAST architecture validation for init_weights
    # Prevents training with incompatible init_weights and dataset
    if init_weights_path is not None and os.path.exists(init_weights_path):
        try:
            from app.ai.neural_net.architecture_registry import (
                get_encoder_version_from_checkpoint,
                get_model_version_from_checkpoint,
            )
            init_encoder_version = get_encoder_version_from_checkpoint(init_weights_path)
            init_model_version = get_model_version_from_checkpoint(init_weights_path)

            # Determine what encoder version the training data expects
            data_encoder = None
            if hex_in_channels == 40:
                data_encoder = "v2"
            elif hex_in_channels == 64:
                data_encoder = "v3"
            elif hex_in_channels == 56:
                data_encoder = "v3"  # V5-heavy compatible with v3

            # Check encoder compatibility
            if init_encoder_version and data_encoder and init_encoder_version != data_encoder:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"ENCODER MISMATCH DETECTED (FAIL-FAST)\n"
                    f"{'='*70}\n\n"
                    f"Init weights: {init_weights_path}\n"
                    f"  - Encoder: {init_encoder_version} ({40 if init_encoder_version == 'v2' else 64} channels)\n\n"
                    f"Training data: {data_path_str}\n"
                    f"  - Encoder: {data_encoder} ({hex_in_channels} channels)\n\n"
                    f"PROBLEM: Cannot train {data_encoder} data with {init_encoder_version} model weights.\n\n"
                    f"SOLUTIONS:\n"
                    f"  1. Re-export training data with --encoder-version {init_encoder_version}\n"
                    f"  2. Use a different init_weights file matching {data_encoder}\n"
                    f"  3. Train from scratch without --init-weights\n"
                    f"{'='*70}"
                )
                if not distributed or is_main_process():
                    logger.error(error_msg)
                raise ValueError(f"Encoder mismatch: init_weights={init_encoder_version}, data={data_encoder}")

            # Check model version and auto-adapt if needed
            if init_model_version:
                if model_version != init_model_version:
                    if not distributed or is_main_process():
                        logger.warning(
                            f"[ArchValidation] Model version mismatch detected!\n"
                            f"  Init weights uses: {init_model_version}\n"
                            f"  Training configured for: {model_version}\n"
                            f"  Auto-adapting to use: {init_model_version}"
                        )
                    # Auto-adapt to match init_weights architecture
                    model_version = init_model_version
                else:
                    if not distributed or is_main_process():
                        logger.info(f"[ArchValidation] Architecture validated: encoder={init_encoder_version}, model={init_model_version}")
        except ImportError:
            pass  # architecture_registry not available, skip validation
        except FileNotFoundError:
            pass  # init_weights file doesn't exist yet, will be caught later

    # Load initial weights for transfer learning (before save_path check)
    # This allows starting from a pre-trained model (e.g., 2p->4p transfer)
    # If save_path exists, it will override these weights (resume takes priority)
    if init_weights_path is not None and os.path.exists(init_weights_path):
        if not os.path.exists(save_path):  # Only if not resuming existing training
            try:
                from app.training.checkpointing import load_weights_only
                load_result = load_weights_only(
                    init_weights_path,
                    model,
                    device=device,
                    strict=init_weights_strict,
                )
                if not distributed or is_main_process():
                    logger.info(f"Loaded initial weights from {init_weights_path}")
                    if load_result.get('missing_keys'):
                        logger.info(f"  Missing keys (will be randomly initialized): {len(load_result['missing_keys'])}")
                    if load_result.get('unexpected_keys'):
                        logger.info(f"  Unexpected keys (ignored): {len(load_result['unexpected_keys'])}")
                # Validate value head after loading init weights (catches 2p->4p transfer issues)
                _validate_model_value_head(model, num_players, "after loading init_weights")
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                # OSError: file I/O errors reading checkpoint
                # RuntimeError: PyTorch loading errors, incompatible models
                # ValueError: invalid checkpoint format
                # KeyError: missing required checkpoint keys
                if not distributed or is_main_process():
                    logger.warning(f"Could not load init weights from {init_weights_path}: {e}. Starting fresh.")
        else:
            if not distributed or is_main_process():
                logger.info(f"Skipping init_weights_path (save_path {save_path} exists, resuming instead)")

    # Load existing weights if available to continue training
    if os.path.exists(save_path):
        try:
            # Use safe_load_checkpoint for secure loading with fallback
            checkpoint = safe_load_checkpoint(save_path, map_location=device, warn_on_unsafe=False)
            # Handle both raw state_dict and checkpoint dict formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            if not distributed or is_main_process():
                logger.info(f"Loaded existing model weights from {save_path}")
            # Validate value head after loading checkpoint (catches resumed training with wrong config)
            _validate_model_value_head(model, num_players, "after loading checkpoint")
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            # OSError: file I/O errors reading checkpoint
            # RuntimeError: PyTorch loading errors, incompatible models
            # ValueError: invalid checkpoint format
            # KeyError: missing model_state_dict key
            if not distributed or is_main_process():
                logger.warning(
                    f"Could not load existing weights: {e}. Starting fresh."
                )

    # Wrap model with DDP if using distributed training
    if distributed:
        model = wrap_model_ddp(
            model, device,
            find_unused_parameters=find_unused_parameters
        )
        if is_main_process():
            logger.info("Model wrapped with DistributedDataParallel")

    # Loss functions
    # For multi-player mode, we use multi_player_value_loss instead of MSELoss
    # which properly masks inactive player slots
    value_criterion = nn.MSELoss()  # Used for scalar mode; multi-player uses function
    nn.KLDivLoss(reduction='batchmean')
    # HexNeuralNet_v2 supports multi-player outputs, so enable multi-player loss for all boards
    use_multi_player_loss = multi_player
    # Note: masked_policy_kl and build_rank_targets are imported from app.ai.neural_losses

    # Handle freeze_policy: only train value head when enabled
    if freeze_policy:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only value head parameters
        value_head_params = []
        for name, param in model.named_parameters():
            # Value head layers are typically named value_fc1, value_fc2, value_head, etc.
            if any(x in name.lower() for x in ['value_fc', 'value_head', 'value_conv', 'value_bn']):
                param.requires_grad = True
                value_head_params.append(param)
                logger.info(f"[freeze_policy] Unfreezing: {name}")

        if not value_head_params:
            logger.warning(
                "[freeze_policy] No value head parameters found! "
                "Check model architecture. Training all parameters."
            )
            for param in model.parameters():
                param.requires_grad = True
            optimizer_params = model.parameters()
        else:
            logger.info(f"[freeze_policy] Training only {len(value_head_params)} value head parameters")
            optimizer_params = value_head_params
    else:
        optimizer_params = model.parameters()

    optimizer = optim.Adam(
        optimizer_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with optional warmup
    # Use the new create_lr_scheduler for advanced cosine options
    epoch_scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type=lr_scheduler,
        total_epochs=config.epochs_per_iter,
        warmup_epochs=warmup_epochs,
        lr_min=lr_min,
        lr_t0=lr_t0,
        lr_t_mult=lr_t_mult,
    )

    # ReduceLROnPlateau as fallback if no scheduler configured
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    ) if epoch_scheduler is None else None

    # Evaluation feedback handler - adjusts LR based on Elo trends (December 2025)
    eval_feedback_handler: EvaluationFeedbackHandler | None = None
    if HAS_TRAINING_ENHANCEMENTS and EvaluationFeedbackHandler is not None:
        config_key = f"{config.board_type.value}_{num_players}p"
        eval_feedback_handler = EvaluationFeedbackHandler(
            optimizer=optimizer,
            config_key=config_key,
            min_lr=lr_min or 1e-6,
            max_lr=config.learning_rate * 2,  # Allow 2x initial LR
        )
        if eval_feedback_handler.subscribe():
            if not distributed or is_main_process():
                logger.info(
                    f"[EvaluationFeedbackHandler] Enabled for {config_key} "
                    "(LR adjusted based on Elo trends)"
                )
        else:
            eval_feedback_handler = None

    # Early stopping (supports both loss-based and Elo-based criteria)
    # Also emits PLATEAU_DETECTED events for curriculum feedback
    early_stopper: EarlyStopping | None = None
    if early_stopping_patience > 0 or elo_early_stopping_patience > 0:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience if early_stopping_patience > 0 else 999999,
            min_delta=0.0001,
            elo_patience=elo_early_stopping_patience if elo_early_stopping_patience > 0 else None,
            elo_min_improvement=elo_min_improvement,
            config_name=config_key,  # For PLATEAU_DETECTED event emission
        )

    # Track starting epoch for resume
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_path is not None and os.path.exists(resume_path):
        # For DDP, load into the underlying model
        model_to_load = cast(
            nn.Module,
            model.module if distributed else model,
        )
        start_epoch, _ = load_checkpoint(
            resume_path,
            model_to_load,
            optimizer,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
            device=device,
        )
        start_epoch += 1  # Start from next epoch
        if not distributed or is_main_process():
            logger.info(f"Resuming training from epoch {start_epoch}")

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize async checkpointer for non-blocking checkpoint I/O (5-10% speedup)
    use_async_checkpoint = getattr(config, 'use_async_checkpoint', True)
    async_checkpointer: AsyncCheckpointer | None = None
    if use_async_checkpoint:
        async_checkpointer = AsyncCheckpointer(max_pending=2)
        if not distributed or is_main_process():
            logger.info("Async checkpointing enabled (non-blocking I/O)")

    # Value calibration tracker for monitoring value head quality
    calibration_tracker: CalibrationTracker | None = None
    if track_calibration:
        calibration_tracker = CalibrationTracker(window_size=5000)
        if not distributed or is_main_process():
            logger.info("Value calibration tracking enabled")

    # Mixed precision scaler configured above (GradScaler only for float16)

    train_streaming_loader: StreamingDataLoader | None = None
    val_streaming_loader: StreamingDataLoader | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    train_sampler = None
    val_sampler = None
    allow_empty_policies = bool(getattr(config, "allow_empty_policies", False))
    filter_empty_policies = not allow_empty_policies
    value_only_training = False  # Set True if dataset has no policy data

    # Auto-detect large datasets and switch to streaming mode to prevent OOM
    if not use_streaming:
        # Calculate total data size
        total_data_size = 0
        paths_to_check: list[str] = []

        if data_dir is not None:
            npz_pattern = os.path.join(data_dir, "*.npz")
            paths_to_check = glob.glob(npz_pattern)
        elif isinstance(data_path, list):
            paths_to_check = data_path
        elif data_path:
            paths_to_check = [data_path]

        for p in paths_to_check:
            if os.path.exists(p):
                total_data_size += os.path.getsize(p)

        if total_data_size > AUTO_STREAMING_THRESHOLD_BYTES:
            size_gb = total_data_size / (1024 ** 3)
            threshold_gb = AUTO_STREAMING_THRESHOLD_BYTES / (1024 ** 3)
            if not distributed or is_main_process():
                logger.warning(
                    f"Auto-enabling streaming mode: dataset size {size_gb:.1f}GB "
                    f"exceeds threshold {threshold_gb:.0f}GB. "
                    f"Set RINGRIFT_AUTO_STREAMING_THRESHOLD_GB to adjust or "
                    f"use --use-streaming explicitly."
                )
            use_streaming = True

    # Collect data paths for streaming mode
    data_paths: list[str] = []

    # Quality-aware data discovery from synced sources
    if discover_synced_data and HAS_DATA_CATALOG:
        try:
            catalog = get_data_catalog()
            discovered_paths = catalog.get_recommended_training_sources(
                target_games=100000,
                board_type=config.board_type.value if hasattr(config, 'board_type') else None,
                num_players=num_players,
            )
            if discovered_paths:
                # Convert discovered .db paths to training data
                # Note: These are SQLite databases that need to be processed
                # by the streaming loader or converted to .npz format
                data_paths.extend([str(p) for p in discovered_paths])
                if not distributed or is_main_process():
                    stats = catalog.get_stats()
                    logger.info(
                        f"DataCatalog discovered {len(discovered_paths)} sources "
                        f"with {stats.total_games} total games "
                        f"(avg quality: {stats.avg_quality_score:.3f})"
                    )
        except (ImportError, AttributeError, OSError, ConnectionError) as e:
            # Module unavailable, missing attributes, file I/O errors, or network issues
            if not distributed or is_main_process():
                logger.warning(f"DataCatalog discovery failed: {e}")

    if use_streaming:
        # Use streaming data loader for large datasets
        if data_dir is not None:
            # Collect all .npz files from directory
            npz_pattern = os.path.join(data_dir, "*.npz")
            data_paths.extend(sorted(glob.glob(npz_pattern)))
            if not distributed or is_main_process():
                logger.info(
                    f"Found {len(data_paths)} .npz files in {data_dir}"
                )
        elif isinstance(data_path, list):
            data_paths.extend(data_path)
        elif data_path:
            data_paths.append(data_path)

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for p in data_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        data_paths = unique_paths

        if not data_paths:
            if not distributed or is_main_process():
                logger.warning("No data files found for streaming; skipping.")
            if distributed:
                cleanup_distributed()
            return

        # Best-effort metadata check on the first file to validate history_length
        # and policy_encoding expectations.
        first_path = data_paths[0]
        dataset_history_length: int | None = None
        policy_encoding: str | None = None
        dataset_feature_version: int | None = None
        dataset_in_channels: int | None = None
        dataset_globals_dim: int | None = None
        # New encoder metadata fields (added 2025-12)
        dataset_encoder_type: str | None = None
        dataset_base_channels: int | None = None
        dataset_board_type_meta: str | None = None
        # V2.1 encoder metadata (added 2025-12 for V3 encoder fix)
        dataset_encoder_version: str | None = None
        dataset_in_channels_meta: int | None = None
        is_npz = bool(first_path and first_path.endswith(".npz"))
        try:
            if first_path and os.path.exists(first_path):
                with safe_load_npz(first_path, mmap_mode="r") as d:
                    if "features" in d:
                        feat_shape = d["features"].shape
                        if len(feat_shape) >= 2:
                            dataset_in_channels = int(feat_shape[1])
                    if "globals" in d:
                        glob_shape = d["globals"].shape
                        if len(glob_shape) >= 2:
                            dataset_globals_dim = int(glob_shape[1])
                    if "policy_encoding" in d:
                        try:
                            policy_encoding = str(np.asarray(d["policy_encoding"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            policy_encoding = None
                    if "history_length" in d:
                        try:
                            dataset_history_length = int(np.asarray(d["history_length"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_history_length = None
                    if "feature_version" in d:
                        try:
                            dataset_feature_version = int(np.asarray(d["feature_version"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_feature_version = None
                    # Read new encoder metadata fields (added 2025-12)
                    if "encoder_type" in d:
                        try:
                            dataset_encoder_type = str(np.asarray(d["encoder_type"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_encoder_type = None
                    if "base_channels" in d:
                        try:
                            dataset_base_channels = int(np.asarray(d["base_channels"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_base_channels = None
                    if "board_type" in d:
                        try:
                            dataset_board_type_meta = str(np.asarray(d["board_type"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_board_type_meta = None
                    # Read V2.1 encoder metadata (added 2025-12 for V3 encoder fix)
                    if "encoder_version" in d:
                        try:
                            dataset_encoder_version = str(np.asarray(d["encoder_version"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_encoder_version = None
                    if "in_channels" in d:
                        try:
                            dataset_in_channels_meta = int(np.asarray(d["in_channels"]).item())
                        except (ValueError, TypeError, AttributeError):
                            # Metadata field missing, wrong type, or empty array
                            dataset_in_channels_meta = None
        except (OSError, KeyError, ValueError) as exc:
            # File I/O errors, missing keys, or data access failures
            if not distributed or is_main_process():
                logger.warning(
                    "Failed to read dataset metadata from %s: %s",
                    first_path,
                    exc,
                )

        if dataset_history_length is not None and dataset_history_length != config.history_length:
            raise ValueError(
                "Training history_length does not match dataset metadata.\n"
                f"  dataset={first_path}\n"
                f"  dataset_history_length={dataset_history_length}\n"
                f"  config.history_length={config.history_length}\n"
                "Regenerate the dataset with matching --history-length or "
                "update the training config."
            )
        elif (dataset_history_length is None and config.history_length != 3
              and (not distributed or is_main_process())):
            logger.warning(
                "Dataset %s missing history_length metadata; using "
                "config.history_length=%d. Ensure the dataset was built "
                "with matching history frames.",
                first_path,
                config.history_length,
            )

        if dataset_feature_version is not None and dataset_feature_version != config_feature_version:
            raise ValueError(
                "Training feature_version does not match dataset metadata.\n"
                f"  dataset={first_path}\n"
                f"  dataset_feature_version={dataset_feature_version}\n"
                f"  config_feature_version={config_feature_version}\n"
                "Regenerate the dataset with matching --feature-version or "
                "update the training config."
            )
        elif dataset_feature_version is None:
            if config_feature_version != 1:
                # Check for autonomous mode - fallback to v1 instead of failing
                autonomous_mode = os.environ.get("RINGRIFT_AUTONOMOUS_MODE", "").lower() in ("1", "true")
                if autonomous_mode:
                    if not distributed or is_main_process():
                        logger.warning(
                            "[AUTONOMOUS] Dataset %s missing feature_version metadata. "
                            "Config requested v%d but falling back to v1 for compatibility.",
                            first_path,
                            config_feature_version,
                        )
                    config_feature_version = 1
                else:
                    raise ValueError(
                        "Dataset is missing feature_version metadata but training "
                        "was configured for feature_version="
                        f"{config_feature_version}.\n"
                        f"  dataset={first_path}\n"
                        "Regenerate the dataset with --feature-version or "
                        "set feature_version=1 to use legacy features."
                    )
            if not distributed or is_main_process():
                logger.warning(
                    "Dataset %s missing feature_version metadata; assuming legacy "
                    "feature_version=1.",
                    first_path,
                )

        if dataset_globals_dim is None:
            if is_npz:
                raise ValueError(
                    "Dataset is missing globals features required for training.\n"
                    f"  dataset={first_path}\n"
                    "Regenerate the dataset with scripts/export_replay_dataset.py."
                )
        elif dataset_globals_dim != 20:
            raise ValueError(
                "Dataset globals feature dimension does not match the CNN encoder.\n"
                f"  dataset={first_path}\n"
                f"  dataset_globals_dim={dataset_globals_dim}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py "
                "to produce 20 global features."
            )

        if dataset_in_channels is not None:
            if use_hex_model:
                hex_base = 16 if use_hex_v3 else 10
                expected_in_channels = hex_base * (config.history_length + 1)
                expected_encoder = "hex_v3" if use_hex_v3 else "hex_v2"
            else:
                expected_in_channels = 14 * (config.history_length + 1)
                expected_encoder = "square"

            # Log encoder metadata if available
            if dataset_encoder_type and (not distributed or is_main_process()):
                logger.info(
                    "Dataset encoder metadata: type=%s, base_channels=%s, "
                    "in_channels=%s, board_type=%s",
                    dataset_encoder_type,
                    dataset_base_channels,
                    dataset_in_channels,
                    dataset_board_type_meta,
                )

            # Log V2.1 encoder metadata if available (Dec 2025)
            if dataset_encoder_version and (not distributed or is_main_process()):
                logger.info(
                    "Dataset V2.1 metadata: encoder_version=%s, in_channels_meta=%s",
                    dataset_encoder_version,
                    dataset_in_channels_meta,
                )

            # Cross-validate in_channels from metadata against actual feature shape
            # This catches cases where export script updated metadata but not actual encoder
            if dataset_in_channels_meta is not None and dataset_in_channels is not None:
                if dataset_in_channels_meta != dataset_in_channels:
                    raise ValueError(
                        "========================================\n"
                        "DATA INTEGRITY ERROR - METADATA MISMATCH\n"
                        "========================================\n"
                        f"Dataset in_channels metadata: {dataset_in_channels_meta}\n"
                        f"Actual feature shape:         {dataset_in_channels} channels\n"
                        f"Dataset:                      {first_path}\n"
                        "\n"
                        "The export script recorded a channel count that doesn't match\n"
                        "the actual feature tensor shape. This indicates a bug in the\n"
                        "export pipeline.\n"
                        "\n"
                        "SOLUTION: Re-export the data with a fixed export script.\n"
                        "========================================"
                    )

            # HARD ERROR: Encoder type must match model version (Dec 2025)
            if dataset_encoder_type and dataset_encoder_type != expected_encoder:
                raise ValueError(
                    "========================================\n"
                    "ENCODER TYPE MISMATCH - CANNOT TRAIN\n"
                    "========================================\n"
                    f"Dataset encoded with: {dataset_encoder_type}\n"
                    f"Model expects:        {expected_encoder}\n"
                    f"Model version:        {model_version}\n"
                    f"Dataset:              {first_path}\n"
                    "\n"
                    "SOLUTION: Re-export data with --encoder-version matching model version\n"
                    f"  For v3 model: use --encoder-version v3\n"
                    f"  For v2 model: use --encoder-version v2\n"
                    "========================================"
                )

            # HARD ERROR: Board type must match (Dec 2025)
            if dataset_board_type_meta:
                dataset_board_upper = dataset_board_type_meta.upper()
                config_board_name = config.board_type.name
                if dataset_board_upper != config_board_name:
                    raise ValueError(
                        "========================================\n"
                        "BOARD TYPE MISMATCH - CANNOT TRAIN\n"
                        "========================================\n"
                        f"Dataset board type:   {dataset_board_type_meta}\n"
                        f"Training board type:  {config.board_type.name}\n"
                        f"Dataset:              {first_path}\n"
                        "\n"
                        "Dataset and training board types must match.\n"
                        "========================================"
                    )

            if dataset_in_channels != expected_in_channels:
                # Build enhanced error message with encoder metadata if available
                encoder_info = ""
                if dataset_encoder_type:
                    encoder_info = f"  dataset_encoder_type={dataset_encoder_type}\n"
                    encoder_info += f"  dataset_base_channels={dataset_base_channels}\n"
                    if dataset_board_type_meta:
                        encoder_info += f"  dataset_board_type={dataset_board_type_meta}\n"

                raise ValueError(
                    "Dataset feature channels do not match the expected encoder.\n"
                    f"  dataset={first_path}\n"
                    f"  dataset_in_channels={dataset_in_channels}\n"
                    f"  expected_in_channels={expected_in_channels} ({expected_encoder})\n"
                    f"{encoder_info}"
                    f"Model expects {expected_encoder} encoder ({expected_in_channels} channels).\n"
                    "Solutions:\n"
                    "  1. Regenerate dataset with matching encoder version:\n"
                    f"     --encoder-version {'v3' if use_hex_v3 else 'v2'}\n"
                    "  2. Or use matching model version for your data:\n"
                    f"     --model-version {'v2' if dataset_in_channels == 40 else 'v3' if dataset_in_channels == 64 else 'unknown'}"
                )
        elif is_npz:
            raise ValueError(
                "Dataset is missing features required for training.\n"
                f"  dataset={first_path}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py."
            )

        if model_version in ('v3', 'v4'):
            if policy_encoding == "legacy_max_n":
                raise ValueError(
                    f"Dataset uses legacy MAX_N policy encoding but --model-version={model_version} "
                    "requires board-aware policy encoding.\n"
                    f"  dataset={first_path}\n"
                    "Regenerate the dataset with --board-aware-encoding."
                )
            if policy_encoding is None and (not distributed or is_main_process()):
                logger.warning(
                    "Dataset %s missing policy_encoding metadata; assuming board-aware "
                    "encoding for %s. If this dataset was exported with legacy MAX_N, "
                    "regenerate with --board-aware-encoding.",
                    first_path,
                    model_version,
                )

        # Get total sample count across all files
        total_samples = sum(
            get_sample_count(p) for p in data_paths if os.path.exists(p)
        )
        _total_samples = total_samples
        _num_data_files = len(data_paths)

        if total_samples == 0:
            if not distributed or is_main_process():
                logger.warning("No samples found in data files; skipping.")
            if distributed:
                cleanup_distributed()
            return

        if not distributed or is_main_process():
            logger.info(
                f"StreamingDataLoader: {total_samples} total samples "
                f"across {len(data_paths)} files"
            )

        # Create streaming data loaders (80/20 split approximated by files)
        # For simplicity, we use all data for training in streaming mode
        # and compute validation on a subset
        val_split = 0.2
        val_samples = int(total_samples * val_split)
        train_samples = total_samples - val_samples

        # Determine rank/world_size for distributed data sharding
        if distributed:
            stream_rank = get_rank()
            stream_world_size = get_world_size()
        else:
            stream_rank = 0
            stream_world_size = 1

        # Use weighted streaming loader when sampling_weights != 'uniform'
        if sampling_weights != 'uniform':
            train_streaming_loader = WeightedStreamingDataLoader(
                data_paths=data_paths,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed,
                drop_last=False,
                policy_size=policy_size,
                rank=stream_rank,
                world_size=stream_world_size,
                filter_empty_policies=filter_empty_policies,
                sampling_weights=sampling_weights,
            )
            if not distributed or is_main_process():
                logger.info(
                    f"Using WeightedStreamingDataLoader with "
                    f"sampling_weights={sampling_weights}"
                )
        else:
            train_streaming_loader = StreamingDataLoader(
                data_paths=data_paths,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed,
                drop_last=False,
                policy_size=policy_size,
                rank=stream_rank,
                world_size=stream_world_size,
                filter_empty_policies=filter_empty_policies,
            )

        # For validation, always use uniform sampling
        val_streaming_loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed + 1000,
            drop_last=False,
            policy_size=policy_size,
            rank=stream_rank,
            world_size=stream_world_size,
            filter_empty_policies=filter_empty_policies,
        )

        # Auto-detect multi-player values from streaming data
        # If data has multi-player values but --multi-player wasn't specified,
        # log a suggestion to the user
        if (train_streaming_loader.has_multi_player_values and not multi_player
                and (not distributed or is_main_process())):
            logger.info(
                "Dataset contains multi-player value vectors (values_mp). "
                "Consider using --multi-player flag for multi-player training."
            )
        # If multi-player training was requested but streaming data does not
        # include vector value targets, fail fast to avoid silent shape issues.
        if (multi_player and not train_streaming_loader.has_multi_player_values
                and (not distributed or is_main_process())):
            logger.error(
                "multi_player=True but streaming dataset does not contain "
                "'values_mp' / 'num_players'. Regenerate data with "
                "multi-player value targets or disable --multi-player."
            )
            if distributed:
                cleanup_distributed()
            raise ValueError(
                "Multi-player training requested but streaming dataset lacks values_mp."
            )

        # Check for value-only training (no policy data)
        if not train_streaming_loader.has_policy:
            if not distributed or is_main_process():
                logger.info(
                    "Dataset has no policy data - enabling value-only training mode "
                    "(policy_weight=0). Policy head will not be trained."
                )
            config.policy_weight = 0.0
            value_only_training = True
        else:
            value_only_training = False

        train_sampler = None
        train_size = train_samples
        val_size = val_samples

    else:
        # Legacy single-file loading with RingRiftDataset or WeightedRingRiftDataset
        if isinstance(data_path, list):
            data_path_str = data_path[0] if data_path else ""
        else:
            data_path_str = data_path

        # V5-heavy models use heuristic features if available in the data
        use_heuristics = model_version in ('v5', 'v5-gnn', 'v5-heavy')
        if sampling_weights == 'uniform':
            full_dataset = RingRiftDataset(
                data_path_str,
                board_type=config.board_type,
                augment_hex=augment_hex_symmetry,
                use_multi_player_values=multi_player,
                filter_empty_policies=filter_empty_policies,
                return_num_players=multi_player,
                return_heuristics=use_heuristics,
            )
            use_weighted_sampling = False
        else:
            full_dataset = WeightedRingRiftDataset(
                data_path_str,
                board_type=config.board_type,
                augment_hex=augment_hex_symmetry,
                weighting=sampling_weights,
                use_multi_player_values=multi_player,
                filter_empty_policies=filter_empty_policies,
                return_num_players=multi_player,
                return_heuristics=use_heuristics,
            )
            use_weighted_sampling = True

        # Phase 5b: Load ELO weights if available (December 2025)
        # This strengthens the self-improvement loop by weighting samples from
        # games against stronger opponents more heavily
        elo_sample_weights: np.ndarray | None = None
        if enable_elo_weighting and data_path_str and os.path.exists(data_path_str):
            try:
                with safe_load_npz(data_path_str, mmap_mode="r") as npz_data:
                    if "opponent_elo" in npz_data:
                        from app.training.elo_weighting import compute_elo_weights
                        opponent_elos = np.array(npz_data["opponent_elo"])
                        # Use model_elo=1500 as baseline reference
                        elo_sample_weights = compute_elo_weights(
                            opponent_elos,
                            model_elo=1500.0,
                            elo_scale=400.0,
                            min_weight=0.2,
                            max_weight=3.0,
                        )
                        if not distributed or is_main_process():
                            logger.info(
                                f"ELO weighting enabled: {len(elo_sample_weights)} samples, "
                                f"weight range [{elo_sample_weights.min():.3f}, {elo_sample_weights.max():.3f}]"
                            )
                    else:
                        if not distributed or is_main_process():
                            logger.info(
                                "ELO weighting requested but dataset lacks 'opponent_elo' field. "
                                "Regenerate with export_replay_dataset.py to include opponent ELO data."
                            )
            except (OSError, KeyError, ValueError) as e:
                # File I/O errors, missing keys, or data type issues
                if not distributed or is_main_process():
                    logger.warning(f"Failed to load ELO weights: {e}")

        # Phase 5c: Load quality scores if available (December 2025)
        # This strengthens the self-improvement loop by weighting samples from
        # higher-quality games more heavily
        quality_sample_weights: np.ndarray | None = None
        if data_path_str and os.path.exists(data_path_str):
            try:
                with safe_load_npz(data_path_str, mmap_mode="r") as npz_data:
                    if "quality_score" in npz_data:
                        quality_scores = np.array(npz_data["quality_score"])
                        # Apply min_quality_score filter as a mask
                        if min_quality_score > 0.0:
                            quality_mask = quality_scores >= min_quality_score
                            num_filtered = np.sum(~quality_mask)
                            if not distributed or is_main_process():
                                logger.info(
                                    f"Quality filtering: {num_filtered} samples below threshold "
                                    f"({min_quality_score:.2f}) will be weighted to 0"
                                )
                            # Zero out weights for low-quality samples
                            quality_sample_weights = np.where(quality_mask, quality_scores, 0.0)
                        else:
                            # Use quality scores directly as weights
                            quality_sample_weights = quality_scores
                        if not distributed or is_main_process():
                            nonzero = quality_sample_weights[quality_sample_weights > 0]
                            if len(nonzero) > 0:
                                logger.info(
                                    f"Quality weighting enabled: {len(quality_sample_weights)} samples, "
                                    f"weight range [{nonzero.min():.3f}, {nonzero.max():.3f}]"
                                )
                    else:
                        if not distributed or is_main_process():
                            logger.debug(
                                "Dataset lacks 'quality_score' field - quality weighting disabled. "
                                "Regenerate with export_replay_dataset.py to include quality data."
                            )
            except (OSError, KeyError, ValueError) as e:
                # File I/O errors, missing keys, or data type issues
                if not distributed or is_main_process():
                    logger.warning(f"Failed to load quality scores: {e}")

        # Phase 5d: Load generator Elo weights if available (January 2026)
        # This implements quality-weighted sampling: games generated by stronger
        # models (higher Elo) get higher weight during training. This creates a
        # positive feedback loop where each generation learns more from better data.
        generator_elo_weights: np.ndarray | None = None
        if enable_elo_weighting and data_path_str and os.path.exists(data_path_str):
            try:
                with safe_load_npz(data_path_str, mmap_mode="r") as npz_data:
                    if "generator_elo" in npz_data:
                        from app.training.elo_weighting import compute_generator_elo_weights
                        generator_elos = np.array(npz_data["generator_elo"])
                        # Compute weights: higher generator Elo = higher weight
                        generator_elo_weights = compute_generator_elo_weights(
                            generator_elos,
                            baseline_elo=1000.0,  # Center point for sigmoid
                            elo_scale=200.0,  # Steepness (lower = steeper)
                            min_weight=0.3,  # Minimum weight for weak generators
                            max_weight=3.0,  # Maximum weight for strong generators
                        )
                        if not distributed or is_main_process():
                            logger.info(
                                f"Generator Elo weighting enabled: {len(generator_elo_weights)} samples, "
                                f"weight range [{generator_elo_weights.min():.3f}, {generator_elo_weights.max():.3f}]"
                            )
                    else:
                        if not distributed or is_main_process():
                            logger.debug(
                                "Dataset lacks 'generator_elo' field - generator Elo weighting disabled. "
                                "Regenerate with export_replay_dataset.py to include generator Elo data."
                            )
            except (OSError, KeyError, ValueError) as e:
                # File I/O errors, missing keys, or data type issues
                if not distributed or is_main_process():
                    logger.warning(f"Failed to load generator Elo weights: {e}")

        # Phase 6: Record data quality in ImprovementOptimizer for feedback loop
        # This enables the self-improvement loop to adapt training parameters
        # based on data quality metrics
        if quality_sample_weights is not None and (not distributed or is_main_process()):
            try:
                from app.training.improvement_optimizer import get_improvement_optimizer
                avg_quality = float(np.mean(quality_sample_weights[quality_sample_weights > 0]))
                optimizer_instance = get_improvement_optimizer()
                # Parity rate is assumed 1.0 since we passed validation; could be
                # enhanced to track actual parity test results
                rec = optimizer_instance.record_data_quality(
                    parity_success_rate=1.0,  # Assume passed if we got here
                    data_quality_score=avg_quality,
                )
                logger.info(
                    f"[ImprovementOptimizer] Recorded data quality: {avg_quality:.3f} "
                    f"(signal: {rec.signal.name}, threshold_adj: {rec.threshold_adjustment:.2f})"
                )
            except ImportError:
                pass  # Improvement optimizer not available
            except (AttributeError, TypeError) as e:
                # Missing attributes or type errors
                logger.debug(f"[ImprovementOptimizer] Failed to record data quality: {e}")

        if len(full_dataset) == 0:
            if not distributed or is_main_process():
                logger.warning(
                    "Training dataset at %s is empty; skipping.",
                    data_path_str,
                )
            if distributed:
                cleanup_distributed()
            return

        # If multi-player mode was requested but the dataset does not provide
        # vector value targets, fail fast to avoid silent shape mismatches.
        if (
            multi_player
            and not getattr(full_dataset, "has_multi_player_values", False)
        ):
            if not distributed or is_main_process():
                logger.error(
                    "multi_player=True but dataset %s does not contain "
                    "'values_mp' / 'num_players'. Regenerate data with "
                    "multi-player value targets or disable --multi-player.",
                    data_path_str,
                )
            if distributed:
                cleanup_distributed()
            raise ValueError(
                "Multi-player training requested but dataset lacks values_mp."
            )

        # Check for value-only training (no policy data)
        if not getattr(full_dataset, "has_policy", True):
            if not distributed or is_main_process():
                logger.info(
                    "Dataset has no policy data - enabling value-only training mode "
                    "(policy_weight=0). Policy head will not be trained."
                )
            config.policy_weight = 0.0
            value_only_training = True
        else:
            value_only_training = False

        # Log spatial shape if available
        shape = getattr(full_dataset, "spatial_shape", None)
        if shape is not None and (not distributed or is_main_process()):
            h, w = shape
            logger.info(
                "Dataset spatial feature shape inferred as %dx%d.",
                h,
                w,
            )

        # Split into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Create data loaders with distributed samplers if needed
        if distributed:
            train_sampler = get_distributed_sampler(
                train_dataset,
                shuffle=True,
            )
            val_sampler = get_distributed_sampler(
                val_dataset,
                shuffle=False,
            )
            # Note: num_workers=0 required on macOS - memory-mapped NPZ files
            # contain BufferedReader objects that can't be pickled for multiprocessing.
            # Also required on Linux with mmap mode to avoid DataLoader hangs.
            # Platform-aware default: 0 on macOS/mmap, else min(4, cpu_count//2) on Linux
            env_workers = os.environ.get("RINGRIFT_DATALOADER_WORKERS")
            if env_workers is not None:
                num_loader_workers = int(env_workers)
            elif sys.platform == "darwin":
                num_loader_workers = 0  # macOS: mmap incompatible with multiprocessing
            else:
                # Linux/Windows: use moderate parallelism for non-mmap data loading
                # Default to 0 for safety (mmap commonly used), allow override via env
                import multiprocessing
                num_loader_workers = min(4, multiprocessing.cpu_count() // 2) if not use_streaming else 0
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,  # Sampler handles shuffling
                sampler=train_sampler,
                num_workers=num_loader_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_loader_workers,
                pin_memory=True,
            )
        else:
            # Non-distributed: optionally use weighted sampling for training.
            # December 2025: Combine position, ELO, and quality weights
            # January 2026: Added generator Elo weights for quality-weighted sampling
            use_any_weighting = (
                use_weighted_sampling
                or (elo_sample_weights is not None)
                or (quality_sample_weights is not None)
                or (generator_elo_weights is not None)
            )

            if use_any_weighting and isinstance(train_dataset, torch.utils.data.Subset):
                subset_indices = np.array(train_dataset.indices, dtype=np.int64)

                # Start with position-based weights if available
                if use_weighted_sampling:
                    base_dataset = cast(WeightedRingRiftDataset, train_dataset.dataset)
                    if base_dataset.sample_weights is None:
                        train_weights_np = np.ones(len(train_dataset), dtype=np.float32)
                    else:
                        train_weights_np = base_dataset.sample_weights[subset_indices].astype(np.float32)
                else:
                    train_weights_np = np.ones(len(train_dataset), dtype=np.float32)

                # Apply ELO weights multiplicatively if available
                if elo_sample_weights is not None:
                    elo_weights_subset = elo_sample_weights[subset_indices].astype(np.float32)
                    train_weights_np = train_weights_np * elo_weights_subset

                # Apply quality weights multiplicatively if available
                if quality_sample_weights is not None:
                    quality_weights_subset = quality_sample_weights[subset_indices].astype(np.float32)
                    train_weights_np = train_weights_np * quality_weights_subset

                # Apply generator Elo weights multiplicatively if available (January 2026)
                # Games from stronger generating models get higher weight
                if generator_elo_weights is not None:
                    generator_weights_subset = generator_elo_weights[subset_indices].astype(np.float32)
                    train_weights_np = train_weights_np * generator_weights_subset

                # Log final combined weights
                if not distributed or is_main_process():
                    weight_sources = []
                    if use_weighted_sampling:
                        weight_sources.append("position")
                    if elo_sample_weights is not None:
                        weight_sources.append("ELO")
                    if quality_sample_weights is not None:
                        weight_sources.append("quality")
                    if generator_elo_weights is not None:
                        weight_sources.append("generator_elo")
                    nonzero = train_weights_np[train_weights_np > 0]
                    if len(nonzero) > 0:
                        logger.info(
                            f"Combined weights ({' * '.join(weight_sources)}): "
                            f"{len(nonzero)}/{len(train_weights_np)} samples with weight > 0, "
                            f"range [{nonzero.min():.3f}, {nonzero.max():.3f}]"
                        )

                train_weights = torch.from_numpy(train_weights_np)
                train_sampler = WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_dataset),
                    replacement=True,
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    sampler=train_sampler,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                )
            else:
                train_sampler = None
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                )

    # Phase 6: Validate training compatibility before starting
    if not distributed or is_main_process():
        try:
            _validate_training_compatibility(model, full_dataset, config)
        except ValueError as e:
            logger.error(f"Training compatibility validation failed: {e}")
            if fail_on_invalid_data:
                raise
            else:
                logger.warning("Continuing despite validation failure (fail_on_invalid_data=False)")

    if not distributed or is_main_process():
        logger.info(
            f"Starting training for {config.epochs_per_iter} epochs..."
        )
        logger.info(f"Train size: {train_size}, Val size: {val_size}")
        if use_streaming:
            logger.info("Using StreamingDataLoader for memory-efficient data")
            if distributed:
                logger.info(
                    f"  Data sharding: rank {get_rank()}/{get_world_size()}, "
                    f"~{train_size // get_world_size()} samples per rank"
                )
        if distributed:
            logger.info(
                f"Distributed training with {get_world_size()} processes"
            )
        if early_stopper is not None:
            elo_info = ""
            if elo_early_stopping_patience > 0:
                elo_info = f", Elo patience: {elo_early_stopping_patience} (min improvement: {elo_min_improvement})"
            logger.info(
                f"Early stopping enabled with loss patience: "
                f"{early_stopping_patience}{elo_info}"
            )
        if warmup_epochs > 0:
            logger.info(f"LR warmup enabled for {warmup_epochs} epochs")
        if lr_scheduler in ('cosine', 'cosine-warm-restarts'):
            logger.info(
                f"LR scheduler: {lr_scheduler} (min_lr={lr_min})"
            )
            if lr_scheduler == 'cosine-warm-restarts':
                logger.info(f"  T_0={lr_t0}, T_mult={lr_t_mult}")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Initialize distributed metrics tracker
    dist_metrics = DistributedMetrics() if distributed else None

    # Initialize heartbeat monitor for fault tolerance
    heartbeat_monitor: HeartbeatMonitor | None = None
    if heartbeat_file and is_main_process():
        heartbeat_path = Path(heartbeat_file)
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_monitor = HeartbeatMonitor(
            heartbeat_interval=heartbeat_interval,
            timeout_threshold=heartbeat_interval * 4,  # 4 missed beats = timeout
        )
        heartbeat_monitor.start(heartbeat_path)
        logger.info(f"Heartbeat monitor started: {heartbeat_file} (interval={heartbeat_interval}s)")

    best_val_loss = float('inf')
    best_train_loss_at_best_val = float('inf')  # Track train loss at best val for overfitting detection
    avg_val_loss = float('inf')  # Initialize for final checkpoint
    avg_train_loss = float('inf')  # Track for return value
    avg_policy_accuracy: float | None = None  # Jan 2026: Initialize for TRAINING_COMPLETED event

    # Track per-epoch losses for downstream analysis
    epoch_losses: list[dict[str, float]] = []
    epochs_completed = 0

    # Hardened event emission tracking (December 2025)
    # These flags ensure TRAINING_COMPLETED or TRAINING_FAILED always fires in finally block
    _training_completed_normally = False
    _training_exception: Exception | None = None
    _training_start_time = time.time()
    _final_checkpoint_path: str | None = None  # Track for event emission
    _total_samples: int = 0  # Track for generation tracking
    _num_data_files: int = 0  # Track for generation tracking

    # Define config_label unconditionally (used for metrics and event logging)
    config_label = f"{config.board_type.value}_{num_players}p"

    # Initialize LossMonitor for early learning stall detection (Jan 2026 fix)
    loss_monitor = LossMonitor(patience=5, config_key=config_label)

    # Report batch size metric at start of training
    if HAS_PROMETHEUS and (not distributed or is_main_process()):
        BATCH_SIZE.labels(config=config_label).set(config.batch_size)

    # Start integrated enhancements background services (evaluation, etc.)
    if enhancements_manager is not None:
        enhancements_manager.start_background_services()
        logger.info("Integrated enhancements background services started")

    # Initialize fault tolerance components via factory (2025-12, refactored)
    ft_config = FaultToleranceConfig(
        enable_circuit_breaker=enable_circuit_breaker,
        enable_anomaly_detection=enable_anomaly_detection,
        enable_graceful_shutdown=enable_graceful_shutdown,
        gradient_clip_mode=gradient_clip_mode,
        gradient_clip_max_norm=gradient_clip_max_norm,
        anomaly_spike_threshold=anomaly_spike_threshold,
        anomaly_gradient_threshold=anomaly_gradient_threshold,
    )
    ft_components = setup_fault_tolerance(
        ft_config,
        distributed=distributed,
        is_main_process_fn=is_main_process if distributed else None,
    )

    # Extract components for use in training loop
    training_breaker = ft_components.training_breaker
    anomaly_detector = ft_components.anomaly_detector
    adaptive_clipper = ft_components.adaptive_clipper
    fixed_clip_norm = ft_components.fixed_clip_norm
    gradient_clip_mode = ft_components.gradient_clip_mode
    anomaly_step = 0  # Track step for anomaly detection

    # Training state for checkpoint tracking and rollback (2025-12, refactored)
    training_state = TrainingState(
        epoch=start_epoch,
        best_val_loss=float('inf'),
        avg_val_loss=float('inf'),
    )

    # Setup graceful shutdown handler for emergency checkpoints (2025-12)
    shutdown_handler: GracefulShutdownHandler | None = None
    if enable_graceful_shutdown and (not distributed or is_main_process()):
        def _emergency_checkpoint_callback():
            """Save emergency checkpoint on signal."""
            model_to_save = model.module if distributed else model
            emergency_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_emergency_epoch_{training_state.epoch}.pth",
            )
            save_checkpoint(
                model_to_save,
                optimizer,
                training_state.epoch,
                training_state.avg_val_loss,
                emergency_path,
                scheduler=epoch_scheduler,
                early_stopping=early_stopper,
            )

        shutdown_handler = GracefulShutdownHandler()
        shutdown_handler.setup(_emergency_checkpoint_callback)

    # Wire REGRESSION_DETECTED → automatic rollback (December 2025)
    # This ensures that if model performance regresses during training,
    # we can automatically roll back to the last good checkpoint.
    rollback_handler = None
    try:
        from app.training.rollback_manager import wire_regression_to_rollback
        from app.training.model_registry import get_model_registry

        registry = get_model_registry()
        rollback_handler = wire_regression_to_rollback(
            registry=registry,
            auto_rollback_enabled=True,  # Auto-rollback on CRITICAL regressions
            require_approval_for_severe=True,  # Prompt for SEVERE regressions
            subscribe_to_events=True,  # Subscribe to event bus
        )
        if not distributed or is_main_process():
            logger.info("[train_model] Regression → rollback wiring activated")
    except ImportError:
        pass  # Rollback manager not available
    except (AttributeError, TypeError, RuntimeError) as e:
        # Missing attributes, type errors, or initialization failures
        if not distributed or is_main_process():
            logger.debug(f"[train_model] Rollback wiring not available: {e}")

    # Aliases for backwards compatibility with existing loop code
    _last_good_checkpoint_path = training_state.last_good_checkpoint_path
    _last_good_epoch = training_state.last_good_epoch
    _circuit_breaker_rollbacks = training_state.circuit_breaker_rollbacks
    _max_circuit_breaker_rollbacks = training_state.max_circuit_breaker_rollbacks

    # Publish training started event (2025-12)
    if HAS_EVENT_BUS and get_router is not None and DataEventType is not None and (not distributed or is_main_process()):
        try:
            router = get_router()
            # config.model_dir may be a str or Path, so use Path() for safety
            model_path = Path(config.model_dir) / f"model_{num_players}p.pth"
            router.publish_sync(DataEvent(
                event_type=DataEventType.TRAINING_STARTED,
                payload={
                    "total_epochs": config.epochs_per_iter,
                    "start_epoch": start_epoch,
                    "config": f"{config.board_type.value}_{num_players}p",
                    "model_path": str(model_path),
                },
                source="train",
            ))
        except (RuntimeError, ConnectionError, TimeoutError, TypeError) as e:
            # Event emission can fail due to async runtime, network issues, or type mismatches
            logger.debug(f"Failed to publish training started event: {e}")

    # Learning rate finder (2025-12)
    # Runs a range test to find optimal learning rate before training
    if find_lr and (not distributed or is_main_process()):
        try:
            from app.training.advanced_training import LRFinder

            logger.info(
                f"[LR Finder] Running learning rate range test "
                f"(min={lr_finder_min:.1e}, max={lr_finder_max:.1e}, iters={lr_finder_iterations})"
            )

            # Create a simple combined loss for LR finding
            def combined_criterion(outputs: Any, targets: Any) -> torch.Tensor:
                """Combine value and policy losses for LR range test."""
                if isinstance(outputs, tuple):
                    value_out, policy_out = outputs[:2]
                    if isinstance(targets, tuple):
                        value_target, policy_target = targets[:2]
                    else:
                        value_target = targets
                        policy_target = None
                    value_loss = nn.functional.mse_loss(value_out.squeeze(), value_target.squeeze())
                    if policy_target is not None:
                        policy_loss = nn.functional.cross_entropy(policy_out, policy_target)
                        return value_loss + policy_loss
                    return value_loss
                return nn.functional.mse_loss(outputs, targets)

            lr_finder = LRFinder(
                model=model.module if distributed else model,
                optimizer=optimizer,
                criterion=combined_criterion,
                device=device,
            )

            # Use train_loader for LR range test
            lr_result = lr_finder.range_test(
                train_loader,
                min_lr=lr_finder_min,
                max_lr=lr_finder_max,
                num_iter=lr_finder_iterations,
            )

            logger.info(
                f"[LR Finder] Results: suggested_lr={lr_result.suggested_lr:.2e}, "
                f"steepest_lr={lr_result.steepest_lr:.2e}, best_lr={lr_result.best_lr:.2e}"
            )

            # Apply suggested learning rate
            old_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_result.suggested_lr
            logger.info(f"[LR Finder] Updated learning rate: {old_lr:.2e} -> {lr_result.suggested_lr:.2e}")

        except (RuntimeError, ValueError, OSError) as e:
            # CUDA OOM, invalid data, or file I/O errors during LR finding
            logger.warning(f"[LR Finder] Failed: {e}. Continuing with configured LR.")

    try:
        for epoch in range(start_epoch, config.epochs_per_iter):
            # Circuit breaker check - skip training if circuit is open (2025-12)
            if training_breaker and not training_breaker.can_execute("training_epoch"):
                logger.warning(f"Training circuit OPEN - skipping epoch {epoch} (recovering from failures)")
                # Update circuit breaker state metric (1=open)
                if HAS_PROMETHEUS and CIRCUIT_BREAKER_STATE and (not distributed or is_main_process()):
                    CIRCUIT_BREAKER_STATE.labels(config=config_label, operation='training_epoch').set(1)

                # Attempt checkpoint rollback if we have a good checkpoint (2025-12)
                if _last_good_checkpoint_path and _circuit_breaker_rollbacks < _max_circuit_breaker_rollbacks:
                    _circuit_breaker_rollbacks += 1
                    logger.warning(
                        f"Circuit breaker rollback {_circuit_breaker_rollbacks}/{_max_circuit_breaker_rollbacks}: "
                        f"restoring checkpoint from epoch {_last_good_epoch}"
                    )
                    try:
                        # Load the last good checkpoint
                        loaded_epoch, loaded_loss = load_checkpoint(
                            _last_good_checkpoint_path, model, optimizer,
                            scheduler=epoch_scheduler, device=device
                        )
                        logger.info(f"Rollback successful: restored to epoch {loaded_epoch}, loss {loaded_loss:.4f}")

                        # Reduce learning rate by 50% to stabilize training
                        for param_group in optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = old_lr * 0.5
                            logger.info(f"Reduced learning rate: {old_lr:.2e} -> {param_group['lr']:.2e}")

                        # Reset circuit breaker to allow retry
                        if training_breaker:
                            training_breaker.record_success("training_epoch")
                    except (OSError, RuntimeError, AttributeError) as e:
                        # File I/O errors, state restoration failures, or missing attributes
                        logger.error(f"Rollback failed: {e}")

                time.sleep(TRAINING_RETRY_SLEEP_SECONDS)  # Configurable pause before retry
                continue

            # Update circuit breaker state metric (0=closed, training can proceed)
            if HAS_PROMETHEUS and CIRCUIT_BREAKER_STATE and training_breaker and (not distributed or is_main_process()):
                CIRCUIT_BREAKER_STATE.labels(config=config_label, operation='training_epoch').set(0)

            # Circuit breaker: Check resources at the start of each epoch
            # This prevents training from overwhelming the system when resources are constrained
            if epoch % 5 == 0:  # Check every 5 epochs to minimize overhead
                try:
                    from app.utils.resource_guard import can_proceed, get_resource_status, wait_for_resources
                    if not can_proceed(check_disk=True, check_mem=True, check_cpu_load=False):
                        status = get_resource_status()
                        logger.warning(
                            f"Resource pressure detected at epoch {epoch}: "
                            f"CPU={status['cpu']['used_percent']:.0f}%, "
                            f"Memory={status['memory']['used_percent']:.0f}%, "
                            f"Disk={status['disk']['used_percent']:.0f}%. "
                            f"Waiting for resources..."
                        )
                        if not wait_for_resources(timeout=300.0, mem_required_gb=2.0):
                            logger.warning("Resources still constrained after 5 min wait, continuing anyway")
                except ImportError:
                    pass  # resource_guard not available

            # Log scheduled batch size if batch scheduling is enabled
            if enhancements_manager is not None and enhancements_manager._batch_scheduler is not None:
                scheduled_batch = enhancements_manager.get_batch_size()
                if not distributed or is_main_process():
                    logger.info(f"Epoch {epoch+1}: scheduled batch size = {scheduled_batch}")

            # Set epoch for distributed sampler or streaming loader
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if use_streaming:
                assert train_streaming_loader is not None
                assert val_streaming_loader is not None
                train_streaming_loader.set_epoch(epoch)
                val_streaming_loader.set_epoch(epoch)

            # Track epoch failure state for circuit breaker (2025-12)

            # Phase 2 Feedback Loop: Check improvement optimizer for training adjustments
            # December 2025: Training now responds to evaluation signals (promotion streaks, regressions)
            # Note: HYPERPARAMETER_UPDATED events are handled in real-time by EvaluationFeedbackHandler
            # (see subscribe() call above). This polling is a fallback for cross-process updates.
            # Check EVERY epoch for pending hyperparameter updates from gauntlet feedback
            try:
                from app.coordination.gauntlet_feedback_controller import get_pending_hyperparameter_updates
                pending_updates = get_pending_hyperparameter_updates(config_label)
                if pending_updates and (not distributed or is_main_process()):
                    logger.info(f"[GauntletFeedback] Applying {len(pending_updates)} hyperparameter update(s) at epoch {epoch}")

                for param, update in pending_updates.items():
                    value = update.get("value")
                    reason = update.get("reason", "gauntlet_feedback")

                    # Learning rate adjustments
                    if param == "learning_rate" and isinstance(value, (int, float)):
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            param_group["lr"] = float(value)
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] LR adjusted: {old_lr:.2e} -> {value:.2e} (reason: {reason})"
                            )
                    elif param == "lr_multiplier" and isinstance(value, (int, float)):
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            param_group["lr"] = old_lr * float(value)
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] LR scaled: {old_lr:.2e} * {value:.2f} (reason: {reason})"
                            )

                    # Temperature scale (exploration reduction for strong models)
                    elif param == "temperature_scale" and isinstance(value, (int, float)):
                        # Temperature affects selfplay, not training directly
                        # Log the update for awareness - actual application happens in selfplay
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] Temperature scale updated: {value:.2f} (reason: {reason})"
                            )
                            logger.info(
                                f"  Note: Temperature affects selfplay data generation, not training directly"
                            )

                    # Quality threshold boost (raise quality bar for strong models)
                    elif param == "quality_threshold_boost" and isinstance(value, (int, float)):
                        # Quality threshold affects data filtering in selfplay/training data generation
                        # Store for next training iteration
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] Quality threshold boost: +{value:.3f} (reason: {reason})"
                            )
                            logger.info(
                                f"  Note: Quality threshold affects data filtering in future training iterations"
                            )

                    # Epoch multiplier (extend training for weak models)
                    elif param == "epoch_multiplier" and isinstance(value, (int, float)):
                        # Calculate how many additional epochs to run
                        multiplier = float(value)
                        original_epochs = config.epochs_per_iter
                        new_total_epochs = int(original_epochs * multiplier)
                        additional_epochs = new_total_epochs - original_epochs

                        if additional_epochs > 0 and (not distributed or is_main_process()):
                            logger.info(
                                f"[GauntletFeedback] Epoch extension requested: {multiplier:.1f}x "
                                f"({original_epochs} -> {new_total_epochs} epochs, +{additional_epochs}) "
                                f"(reason: {reason})"
                            )
                            logger.info(
                                f"  Note: Epoch extension will be applied in the next training run. "
                                f"Current run continues to {original_epochs} epochs."
                            )

                    # Unknown parameter - log for debugging
                    else:
                        if not distributed or is_main_process():
                            logger.debug(
                                f"[GauntletFeedback] Unknown parameter '{param}' = {value} (reason: {reason})"
                            )
            except ImportError:
                pass  # Gauntlet feedback not available
            except (AttributeError, TypeError, OSError, ConnectionError) as e:
                # Missing attributes, type errors, file I/O, or network issues
                if not distributed or is_main_process():
                    logger.debug(f"[GauntletFeedback] Failed to check updates: {e}")

            # Check improvement optimizer every 5 epochs (less frequent than gauntlet feedback)
            if epoch % 5 == 0:
                try:
                    from app.training.improvement_optimizer import get_training_adjustment

                    adjustment = get_training_adjustment(config_label)
                    if adjustment.get("lr_multiplier", 1.0) != 1.0:
                        lr_mult = adjustment["lr_multiplier"]
                        reason = adjustment.get("reason", "unknown")
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            param_group["lr"] = old_lr * lr_mult
                        if not distributed or is_main_process():
                            logger.info(
                                f"[ImprovementOptimizer] LR adjustment: {lr_mult:.2f}x (reason: {reason})"
                            )

                    if adjustment.get("regularization_boost", 0.0) > 0:
                        # Add extra weight decay for overfit mitigation
                        reg_boost = adjustment["regularization_boost"]
                        for param_group in optimizer.param_groups:
                            param_group["weight_decay"] = param_group.get("weight_decay", 0) + reg_boost
                        if not distributed or is_main_process():
                            logger.info(f"[ImprovementOptimizer] Regularization boost: +{reg_boost:.4f}")

                except ImportError:
                    pass  # Improvement optimizer not available
                except (AttributeError, TypeError, ValueError) as e:
                    # Missing attributes, type errors, or invalid values
                    if not distributed or is_main_process():
                        logger.debug(f"[ImprovementOptimizer] Check failed: {e}")

            # Training
            model.train()
            train_loss = torch.tensor(0.0, device=device)  # Accumulate on GPU to avoid per-batch .item() sync
            train_batches = 0
            if dist_metrics is not None:
                dist_metrics.reset()

            # Select appropriate data source
            # For multi-player mode with streaming, use iter_with_mp() to get
            # per-sample num_players from the batch.
            use_mp_iter = use_multi_player_loss and use_streaming and train_streaming_loader.has_multi_player_values
            if use_streaming:
                assert train_streaming_loader is not None
                # Use prefetch_loader for background prefetching if enabled
                use_prefetch = getattr(config, 'use_prefetch', True)
                pin_memory = getattr(config, 'pin_memory', True) and device.type == 'cuda'
                prefetch_count = getattr(config, 'prefetch_count', 2)
                # Enable async GPU transfer in prefetch thread (10-20% speedup)
                prefetch_to_device = getattr(config, 'prefetch_to_device', True) and device.type == 'cuda'

                if use_prefetch:
                    train_data_iter = prefetch_loader(
                        train_streaming_loader,
                        prefetch_count=prefetch_count,
                        pin_memory=pin_memory,
                        use_mp=use_mp_iter,
                        transfer_to_device=device if prefetch_to_device else None,
                    )
                elif use_mp_iter:
                    train_data_iter = train_streaming_loader.iter_with_mp()
                else:
                    train_data_iter = iter(train_streaming_loader)
            else:
                assert train_loader is not None
                train_data_iter = iter(train_loader)

            for i, batch_data in enumerate(train_data_iter):
                # Circuit breaker check: skip batches if circuit is open (2025-12)
                if training_breaker and not training_breaker.can_execute("batch_processing"):
                    if i % 100 == 0:  # Log every 100th skipped batch
                        logger.debug(f"Batch {i} skipped: circuit breaker open for batch_processing")
                    continue

                # Handle streaming, streaming with multi-player, and legacy batch formats
                batch_num_players = None  # Per-sample num_players or None
                batch_heuristics = None  # Heuristic features for v5 (if available)
                if use_streaming:
                    if use_multi_player_loss and train_streaming_loader.has_multi_player_values:
                        # Streaming with multi-player values
                        (
                            (features, globals_vec),
                            (value_targets, policy_targets),
                            values_mp_batch,
                            batch_num_players,
                        ) = batch_data
                        # Use values_mp as the value targets for multi-player loss
                        if values_mp_batch is not None:
                            value_targets = values_mp_batch
                    else:
                        (
                            (features, globals_vec),
                            (value_targets, policy_targets),
                        ) = batch_data
                else:
                    # Non-streaming mode: batch structure varies based on dataset config
                    # 4 elems: (features, globals, value, policy)
                    # 5 elems: (features, globals, value, policy, num_players) OR (... , heuristics)
                    # 6 elems: (features, globals, value, policy, num_players, heuristics)
                    batch_len = len(batch_data) if isinstance(batch_data, (list, tuple)) else 0
                    if batch_len == 6:
                        # Full: with num_players and heuristics
                        (
                            features,
                            globals_vec,
                            value_targets,
                            policy_targets,
                            batch_num_players,
                            batch_heuristics,
                        ) = batch_data
                    elif batch_len == 5:
                        # Check if 5th element is num_players (int/long tensor) or heuristics (float)
                        fifth_elem = batch_data[4]
                        if fifth_elem.dtype in (torch.int64, torch.int32, torch.long):
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                                batch_num_players,
                            ) = batch_data
                        else:
                            # Heuristics without num_players
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                                batch_heuristics,
                            ) = batch_data
                    else:
                        (
                            features,
                            globals_vec,
                            value_targets,
                            policy_targets,
                        ) = batch_data

                # Data quality metrics (every 500 batches to minimize GPU sync overhead)
                if i % 500 == 0 and i > 0:
                    # Value target distribution: check for P1/P2 balance
                    # Positive values typically indicate P1 advantage, negative P2
                    if value_targets.dim() == 1:
                        mean_val = value_targets.mean().item()
                        pos_ratio = (value_targets > 0).float().mean().item()
                        if abs(mean_val) > 0.15 or abs(pos_ratio - 0.5) > 0.15:
                            logger.debug(
                                f"Data quality: value_mean={mean_val:.3f}, "
                                f"positive_ratio={pos_ratio:.2%} (batch {i})"
                            )
                    # Policy entropy: measure diversity of targets
                    # Low entropy indicates concentrated/biased policy targets
                    policy_sums = policy_targets.sum(dim=1)
                    valid_policy = policy_sums > 0
                    if torch.any(valid_policy):
                        policy_probs = policy_targets[valid_policy] + 1e-8  # Avoid log(0)
                        policy_entropy = -(policy_probs * policy_probs.log()).sum(dim=1).mean().item()
                        if policy_entropy < 1.0:  # Very low entropy indicates potential issue
                            logger.debug(
                                f"Data quality: low policy entropy={policy_entropy:.3f} (batch {i})"
                            )

                # Transfer to device if not already there (prefetch may have done this)
                if features.device != device:
                    features = features.to(device, non_blocking=True)
                    globals_vec = globals_vec.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    policy_targets = policy_targets.to(device, non_blocking=True)
                if batch_num_players is not None and batch_num_players.device != device:
                    batch_num_players = batch_num_players.to(device, non_blocking=True)
                if batch_heuristics is not None and batch_heuristics.device != device:
                    batch_heuristics = batch_heuristics.to(device, non_blocking=True)

                # Hot data buffer mixing: replace portion of batch with hot buffer samples (2025-12)
                if hot_buffer is not None and hot_buffer.total_samples >= config.batch_size:
                    try:
                        # Compute how many samples to replace
                        n_hot = int(features.size(0) * hot_buffer_mix_ratio)
                        if n_hot > 0:
                            # Get samples from hot buffer
                            hot_board, hot_global, hot_policy, hot_value = hot_buffer.get_training_batch(
                                batch_size=n_hot, shuffle=True
                            )
                            if len(hot_board) > 0:
                                # Convert to tensors and transfer to device
                                hot_board_t = torch.from_numpy(hot_board).to(device, non_blocking=True)
                                hot_global_t = torch.from_numpy(hot_global).to(device, non_blocking=True)
                                hot_policy_t = torch.from_numpy(hot_policy).to(device, non_blocking=True)
                                hot_value_t = torch.from_numpy(hot_value).to(device, non_blocking=True)

                                # Replace last n_hot samples in the batch with hot buffer samples
                                actual_n_hot = min(n_hot, len(hot_board_t), features.size(0))
                                if actual_n_hot > 0:
                                    features[-actual_n_hot:] = hot_board_t[:actual_n_hot]
                                    globals_vec[-actual_n_hot:] = hot_global_t[:actual_n_hot]
                                    policy_targets[-actual_n_hot:] = hot_policy_t[:actual_n_hot]
                                    # Handle scalar vs vector value targets
                                    if value_targets.dim() == 1:
                                        value_targets[-actual_n_hot:] = hot_value_t[:actual_n_hot]
                                    else:
                                        # Vector values - broadcast hot buffer scalar to first element
                                        value_targets[-actual_n_hot:, 0] = hot_value_t[:actual_n_hot]
                    except (RuntimeError, ValueError, IndexError, AttributeError) as e:
                        # Tensor operation errors, invalid values, index errors, or missing attributes
                        # Don't fail training on hot buffer errors
                        if i % 100 == 0:
                            logger.debug(f"Hot buffer mixing skipped: {e}")

                # Data augmentation: apply random symmetry transforms (2025-12)
                if enhancements_manager is not None and enhancements_manager._augmentor is not None:
                    try:
                        features, policy_targets = enhancements_manager.augment_batch_dense(
                            features, policy_targets
                        )
                    except (RuntimeError, ValueError, AttributeError) as e:
                        # Tensor operation errors, invalid values, or missing attributes
                        # Don't fail training on augmentation errors
                        if i % 100 == 0:
                            logger.debug(f"Data augmentation skipped: {e}")

                # Pad policy targets if smaller than model policy_size (e.g., dataset
                # was generated with a smaller policy space than the model supports)
                if hasattr(model, 'policy_size') and policy_targets.size(1) < model.policy_size:
                    pad_size = model.policy_size - policy_targets.size(1)
                    policy_targets = torch.nn.functional.pad(
                        policy_targets, (0, pad_size), value=0.0
                    )

                policy_valid_mask = policy_targets.sum(dim=1) > 0

                # Phase 1 Diagnostics: Validate policy target normalization
                if torch.any(policy_valid_mask):
                    target_sums = policy_targets[policy_valid_mask].sum(dim=1)
                    if not torch.allclose(target_sums, torch.ones_like(target_sums), atol=1e-4):
                        bad_sums = target_sums[~torch.isclose(target_sums, torch.ones_like(target_sums), atol=1e-4)]
                        logger.error(
                            f"Policy targets not normalized at batch {i}! "
                            f"Expected sum=1.0, got: min={target_sums.min():.6f}, "
                            f"max={target_sums.max():.6f}, "
                            f"num_bad={len(bad_sums)}/{len(target_sums)}"
                        )
                        if target_sums.min() < 0.5 or target_sums.max() > 1.5:
                            raise ValueError(
                                f"Policy targets severely denormalized at batch {i}. "
                                f"Check data export pipeline."
                            )

                # Apply label smoothing to policy targets if configured
                # smoothed = (1 - eps) * target + eps * uniform
                # IMPORTANT: For V3/V4 spatial policy heads, only smooth over positions
                # where the original target > 0. This prevents adding probability mass
                # to invalid hex corners (indices 0-11 for hex8) that the model
                # correctly assigns -1e9 logits to via scatter initialization.
                if config.policy_label_smoothing > 0 and torch.any(policy_valid_mask):
                    eps = config.policy_label_smoothing
                    policy_targets = policy_targets.clone()

                    # Create mask of valid action positions (non-zero in original targets)
                    action_mask = policy_targets > 0  # [B, policy_size]

                    # Count valid actions per sample for proper uniform distribution
                    num_valid_per_sample = action_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

                    # Create per-sample uniform distribution over valid actions only
                    # For positions where target=0, uniform stays 0 (preserves zeros)
                    uniform_over_valid = action_mask.float() / num_valid_per_sample

                    # Apply smoothing: (1-eps)*target + eps*uniform_over_valid
                    policy_targets = (1 - eps) * policy_targets + eps * uniform_over_valid

                # Gradient accumulation: only zero grad at start of accumulation window
                # Dynamic batch scheduling: calculate accumulation steps from batch scheduler
                base_accumulation = getattr(config, 'gradient_accumulation_steps', 1)
                if enhancements_manager is not None and enhancements_manager._batch_scheduler is not None:
                    # Get target batch size from scheduler
                    target_batch_size = enhancements_manager.get_batch_size()
                    actual_batch_size = config.batch_size
                    # Calculate accumulation steps to achieve target effective batch size
                    # accumulation_steps = target / actual (minimum 1)
                    scheduler_accumulation = max(1, target_batch_size // actual_batch_size)
                    accumulation_steps = max(base_accumulation, scheduler_accumulation)
                else:
                    accumulation_steps = base_accumulation
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()

                # Autocast for mixed precision (CUDA only for now).
                with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_torch_dtype):
                    # Check if auxiliary tasks are enabled and model supports return_features
                    use_aux_tasks = (
                        enhancements_manager is not None
                        and enhancements_manager.config.auxiliary_tasks_enabled
                        and enhancements_manager._auxiliary_module is not None
                    )

                    # V5 models accept heuristics parameter
                    model_accepts_heuristics = model_version in ('v5', 'v5-gnn', 'v5-heavy')

                    # Forward pass with optional backbone feature extraction
                    if use_aux_tasks:
                        try:
                            # Jan 10, 2026: Try return_features, fall back if legacy model
                            if model_accepts_heuristics:
                                out = model(features, globals_vec, heuristics=batch_heuristics, return_features=True)
                            else:
                                out = model(features, globals_vec, return_features=True)
                        except TypeError as e:
                            # Legacy checkpoints don't support return_features parameter
                            if "return_features" in str(e):
                                logger.warning(
                                    "Model doesn't support return_features - disabling aux tasks"
                                )
                                use_aux_tasks = False
                                if model_accepts_heuristics:
                                    out = model(features, globals_vec, heuristics=batch_heuristics)
                                else:
                                    out = model(features, globals_vec)
                            else:
                                raise
                        # V3+ models with features return (values, policy, rank_dist, features)
                        if use_aux_tasks and isinstance(out, tuple) and len(out) == 4:
                            value_pred, policy_pred, rank_dist_pred, backbone_features = out
                        elif use_aux_tasks and isinstance(out, tuple) and len(out) == 3:
                            # V2 models with features return (values, policy, features)
                            value_pred, policy_pred, backbone_features = out
                            rank_dist_pred = None
                        else:
                            # Fallback: model doesn't support return_features or aux disabled
                            if isinstance(out, tuple) and len(out) >= 3:
                                value_pred, policy_pred, rank_dist_pred = out[:3]
                            else:
                                value_pred, policy_pred = out[:2]
                                rank_dist_pred = None
                            backbone_features = None
                            use_aux_tasks = False
                    else:
                        if model_accepts_heuristics:
                            out = model(features, globals_vec, heuristics=batch_heuristics)
                        else:
                            out = model(features, globals_vec)
                        # V3 models return (values, policy_logits, rank_dist). We
                        # ignore the rank distribution for v1/v2 training losses.
                        if isinstance(out, tuple) and len(out) == 3:
                            value_pred, policy_pred, rank_dist_pred = out
                        else:
                            value_pred, policy_pred = out
                            rank_dist_pred = None
                        backbone_features = None

                    # Phase 1 Diagnostics: Detect numerical issues in policy predictions
                    if torch.any(torch.isnan(policy_pred)) or torch.any(torch.isinf(policy_pred)):
                        nan_count = torch.isnan(policy_pred).sum().item()
                        inf_count = torch.isinf(policy_pred).sum().item()
                        logger.error(
                            f"NaN/Inf detected in policy_pred! "
                            f"NaNs: {nan_count}, Infs: {inf_count}, "
                            f"Range: [{policy_pred[~torch.isnan(policy_pred)].min():.2e}, "
                            f"{policy_pred[~torch.isnan(policy_pred)].max():.2e}]"
                        )
                        raise ValueError(
                            f"Model produced NaN/Inf in policy predictions at batch {i}. "
                            f"Check model weights and learning rate."
                        )

                    # Check for extreme logits, excluding intentional -1e9 masking for invalid hex cells
                    valid_logits_mask = policy_pred > -1e8  # -1e9 is intentional masking
                    if torch.any(valid_logits_mask):
                        valid_logits = policy_pred[valid_logits_mask]
                        policy_pred_max = valid_logits.abs().max().item()
                        if policy_pred_max > 1e6:
                            logger.warning(
                                f"Extreme policy logits detected at batch {i}: "
                                f"max_abs={policy_pred_max:.2e}, "
                                f"valid_range=[{valid_logits.min():.2e}, {valid_logits.max():.2e}]"
                            )

                    # Apply log_softmax to policy prediction for KLDivLoss
                    # For spatial policy heads (V3/V4), use masked_log_softmax to avoid
                    # numerical instability from -1e4/-1e9 masked positions
                    if detect_masked_policy_output(policy_pred):
                        # Valid positions are either: (1) target distribution > 0, or
                        # (2) model logits > -1e3 (not masked by spatial scatter)
                        valid_mask = (policy_targets > 0) | (policy_pred > -1e3)
                        policy_log_probs = masked_log_softmax(policy_pred, valid_mask)
                    else:
                        # Flat policy heads (V2, V3_Flat) - use standard log_softmax
                        policy_log_probs = torch.log_softmax(policy_pred, dim=1)

                    # Use multi-player value loss for vector value targets
                    if use_multi_player_loss:
                        # Use per-sample num_players from batch if available,
                        # otherwise fall back to the fixed num_players argument
                        effective_num_players = (
                            batch_num_players if batch_num_players is not None
                            else num_players
                        )
                        value_loss = multi_player_value_loss(
                            value_pred, value_targets, effective_num_players
                        )
                    else:
                        # Scalar training uses only the first value head,
                        # matching NeuralNetAI.evaluate_batch behaviour.
                        if value_pred.ndim == 2:
                            value_pred_scalar = value_pred[:, 0]
                        else:
                            value_pred_scalar = value_pred
                        value_loss = value_criterion(
                            value_pred_scalar.reshape(-1),
                            value_targets.reshape(-1),
                        )

                    policy_loss = masked_policy_kl(
                        policy_log_probs,
                        policy_targets,
                    )

                    # Outcome-weighted policy loss (2025-12)
                    # Weight policy loss by game outcome: winner's moves get higher weight
                    # This focuses learning on moves that lead to winning outcomes
                    if enable_outcome_weighted_policy and outcome_weight_scale > 0:
                        # Compute per-sample outcome weights from value targets
                        # value_targets > 0 → winning position → weight > 1
                        # value_targets < 0 → losing position → weight < 1
                        with torch.no_grad():
                            if value_targets.ndim == 2:
                                # Multi-player: use mean value per sample
                                outcome_signal = value_targets.mean(dim=1)
                            else:
                                outcome_signal = value_targets.reshape(-1)

                            # Compute weights: 1 + outcome_weight_scale * sign(outcome)
                            # Winners: 1 + scale, Losers: 1 - scale
                            outcome_weights = 1.0 + outcome_weight_scale * outcome_signal.sign()
                            outcome_weights = outcome_weights.clamp(min=0.1)  # Prevent zero/negative weights

                        # Compute per-sample policy loss and apply weights
                        # NOTE: Use torch.where to avoid 0 * -inf = NaN when policy_log_probs
                        # has -inf values from masked_log_softmax (V3/V4 spatial policy heads)
                        per_sample_policy = -torch.where(
                            policy_targets > 0,
                            policy_targets * policy_log_probs,
                            torch.zeros_like(policy_log_probs)
                        ).sum(dim=1)
                        valid_mask = policy_targets.sum(dim=1) > 0
                        if valid_mask.any():
                            weighted_policy = (per_sample_policy[valid_mask] * outcome_weights[valid_mask]).mean()
                            policy_loss = weighted_policy

                    # Quality-weighted training (2025-12) - resurrected from ebmo_network.py
                    # Weights samples by MCTS visit counts to focus on high-quality moves
                    quality_ranking_loss = torch.tensor(0.0, device=device)
                    if quality_trainer is not None:
                        # Use policy targets as quality proxy (MCTS visit-derived probabilities)
                        # Higher entropy in targets = less certain position = lower quality
                        with torch.no_grad():
                            target_entropy = -(policy_targets * (policy_targets + 1e-8).log()).sum(dim=1)
                            # Invert: low entropy = high quality
                            quality_scores = 1.0 / (1.0 + target_entropy)
                            # Normalize to [0, 1]
                            quality_scores = (quality_scores - quality_scores.min()) / (
                                quality_scores.max() - quality_scores.min() + 1e-8
                            )

                        # Compute ranking loss to enforce quality ordering
                        if quality_trainer.ranking_weight > 0:
                            quality_ranking_loss = ranking_loss_from_quality(
                                policy_log_probs,
                                quality_scores,
                                margin=quality_trainer.ranking_margin,
                            )
                            quality_trainer.quality_stats["ranking_loss"] = quality_ranking_loss.item()

                        # January 2026 Sprint 10: Apply quality weights to per-sample losses
                        # Higher quality samples (sharper policy targets) contribute more to loss
                        # Expected improvement: +25-40 Elo by focusing learning on decisive positions
                        if quality_trainer.quality_weight > 0:
                            # Compute quality weights with minimum floor
                            quality_weights = torch.clamp(quality_scores, min=quality_trainer.min_quality_weight)
                            # Normalize to mean 1.0 (preserves effective batch size)
                            quality_weights = quality_weights / quality_weights.mean()
                            # Blend with uniform weights
                            blend = quality_trainer.quality_weight
                            uniform_weights = torch.ones_like(quality_weights)
                            final_weights = blend * quality_weights + (1.0 - blend) * uniform_weights

                            # Apply to policy loss (per-sample then weighted mean)
                            # NOTE: Use torch.where to avoid 0 * -inf = NaN when policy_log_probs
                            # has -inf values from masked_log_softmax (V3/V4 spatial policy heads)
                            per_sample_policy_loss = -torch.where(
                                policy_targets > 0,
                                policy_targets * policy_log_probs,
                                torch.zeros_like(policy_log_probs)
                            ).sum(dim=1)
                            valid_mask = policy_targets.sum(dim=1) > 0
                            if valid_mask.any():
                                policy_loss = (per_sample_policy_loss[valid_mask] * final_weights[valid_mask]).mean()

                            # Apply to value loss (per-sample then weighted mean)
                            if value_pred.ndim == 2:
                                per_sample_value_loss = ((value_pred - value_targets) ** 2).mean(dim=1)
                            else:
                                per_sample_value_loss = (value_pred_scalar.reshape(-1) - value_targets.reshape(-1)) ** 2
                            value_loss = (per_sample_value_loss * final_weights).mean()

                            # Track statistics
                            quality_trainer.quality_stats["mean_weight"] = final_weights.mean().item()
                            quality_trainer.quality_stats["std_weight"] = final_weights.std().item()

                    # Entropy regularization to prevent policy collapse
                    # H(p) = -sum(p * log(p)); higher entropy = more exploration
                    # We add -entropy_weight * H to encourage exploration
                    entropy_bonus = torch.tensor(0.0, device=device)
                    if config.entropy_weight > 0:
                        policy_probs = policy_log_probs.exp()
                        # Entropy: -sum(p * log(p)), clamping log for numerical stability
                        policy_entropy = -(policy_probs * policy_log_probs.clamp(min=-20)).sum(dim=1).mean()
                        # Subtract entropy (maximize entropy = minimize negative entropy)
                        entropy_bonus = -config.entropy_weight * policy_entropy

                    # Collect individual losses for gradient surgery
                    # Jan 2026: Apply value_weight to balance value vs policy learning
                    task_losses: dict[str, torch.Tensor] = {
                        "value": config.value_weight * value_loss,
                        "policy": config.policy_weight * policy_loss + entropy_bonus,
                    }

                    # Rank distribution loss (V3+ multi-player head)
                    rank_loss = None
                    if (
                        rank_dist_pred is not None
                        and use_multi_player_loss
                        and value_targets.ndim == 2
                    ):
                        rank_targets, rank_mask = build_rank_targets(
                            value_targets,
                            effective_num_players,
                        )
                        rank_log_probs = torch.log(
                            rank_dist_pred.clamp_min(1e-8)
                        )
                        per_player_loss = -(
                            rank_targets * rank_log_probs
                        ).sum(dim=-1)
                        if torch.any(rank_mask):
                            rank_loss = per_player_loss[rank_mask].mean()
                            task_losses["rank"] = config.rank_dist_weight * rank_loss

                    # Add quality ranking loss if enabled (2025-12)
                    if quality_trainer is not None and quality_ranking_loss.item() > 0:
                        task_losses["quality_ranking"] = quality_trainer.ranking_weight * quality_ranking_loss

                    # Auxiliary task loss (outcome prediction from value targets)
                    aux_loss = None
                    if use_aux_tasks and backbone_features is not None:
                        # Derive outcome class from value targets:
                        # value > 0.3 → Win (2), value < -0.3 → Loss (0), else Draw (1)
                        # For multi-player games, use mean value per sample (not per player)
                        if value_targets.dim() == 2:
                            # Multi-player: value_targets is (batch, num_players)
                            value_flat = value_targets.mean(dim=1)
                        else:
                            value_flat = value_targets.reshape(-1)
                        outcome_targets = torch.where(
                            value_flat > 0.3,
                            torch.tensor(2, device=device, dtype=torch.long),
                            torch.where(
                                value_flat < -0.3,
                                torch.tensor(0, device=device, dtype=torch.long),
                                torch.tensor(1, device=device, dtype=torch.long),
                            ),
                        )
                        aux_targets = {"outcome": outcome_targets}
                        aux_loss, _aux_breakdown = enhancements_manager.compute_auxiliary_loss(
                            backbone_features, aux_targets
                        )
                        task_losses["aux"] = aux_loss

                    # Compute combined loss for metrics (always needed)
                    loss = sum(task_losses.values())

                    # Training facade: per-sample loss, hard example mining, weighted loss (2025-12)
                    # Uses unified facade when available for +80-165 Elo improvement
                    if training_facade is not None:
                        try:
                            # Create batch indices: batch_idx * batch_size + sample_idx
                            batch_size = features.size(0)
                            batch_indices = torch.arange(
                                i * config.batch_size,
                                i * config.batch_size + batch_size,
                                device=device,
                            )

                            # Compute per-sample losses for mining
                            with torch.no_grad():
                                per_sample_losses = training_facade.compute_per_sample_loss(
                                    policy_logits=policy_pred,
                                    policy_targets=policy_targets,
                                    value_pred=value_pred[:, 0] if value_pred.ndim == 2 else value_pred,
                                    value_targets=value_targets[:, 0] if value_targets.ndim == 2 else value_targets,
                                    reduction="none",
                                )

                                # Compute uncertainty from policy entropy
                                policy_probs = torch.softmax(policy_pred, dim=1)
                                policy_entropy = -(policy_probs * (policy_probs + 1e-8).log()).sum(dim=1)

                            # Record batch and get weighted loss
                            training_facade.record_batch(
                                batch_indices=batch_indices,
                                per_sample_losses=per_sample_losses,
                                uncertainties=policy_entropy,
                            )

                            # Apply hard example weighting to loss (upweights difficult samples)
                            # This focuses training on samples the model struggles with
                            if training_facade.is_mining_active:
                                loss = training_facade.get_weighted_loss(
                                    per_sample_losses=per_sample_losses,
                                    batch_indices=batch_indices,
                                )
                                # Add auxiliary losses back (they use original weighting)
                                for key in ['aux', 'rank']:
                                    if key in task_losses:
                                        loss = loss + task_losses[key]
                        except (RuntimeError, ValueError) as e:
                            # Don't fail training on facade errors
                            if i % 500 == 0:
                                logger.debug(f"[Training Facade] Batch {i} skipped: {e}")

                    # Fallback: standalone hard example miner (2025-12)
                    # Used when training facade is not available
                    elif hard_example_miner is not None and compute_per_sample_loss is not None:
                        try:
                            # Compute per-sample losses (no reduction)
                            with torch.no_grad():
                                per_sample_losses = compute_per_sample_loss(
                                    policy_logits=policy_pred,
                                    policy_targets=policy_targets,
                                    value_pred=value_pred[:, 0] if value_pred.ndim == 2 else value_pred,
                                    value_targets=value_targets[:, 0] if value_targets.ndim == 2 else value_targets,
                                    policy_weight=config.policy_weight,
                                    reduction="none",
                                )

                                # Compute uncertainty from policy entropy (higher entropy = more uncertain)
                                policy_probs = torch.softmax(policy_pred, dim=1)
                                policy_entropy = -(policy_probs * (policy_probs + 1e-8).log()).sum(dim=1)

                                # Create batch indices: batch_idx * batch_size + sample_idx
                                batch_size = features.size(0)
                                batch_indices = torch.arange(
                                    i * config.batch_size,
                                    i * config.batch_size + batch_size,
                                    device=device,
                                )

                                # Record to miner
                                hard_example_miner.record_batch(
                                    indices=batch_indices,
                                    losses=per_sample_losses,
                                    uncertainties=policy_entropy,
                                )
                        except (RuntimeError, ValueError) as e:
                            # Don't fail training on mining errors
                            if i % 500 == 0:
                                logger.debug(f"[Hard Example Mining] Batch {i} skipped: {e}")

                    # Scale loss for gradient accumulation to maintain gradient magnitude
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps
                        # Also scale individual losses for gradient surgery
                        task_losses = {k: v / accumulation_steps for k, v in task_losses.items()}

                # Circuit breaker protection for backward pass (2025-12)
                # Catches CUDA errors, OOM, and other runtime exceptions
                try:
                    if use_gradient_surgery and gradient_surgeon is not None:
                        # Use gradient surgery to project conflicting gradients
                        # Note: apply_surgery handles model.zero_grad and sets gradients
                        gradient_surgeon.apply_surgery(model, task_losses)
                    elif use_grad_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    error_msg = str(e).lower()
                    is_cuda_error = (
                        'cuda' in error_msg or 'out of memory' in error_msg or
                        'cublas' in error_msg or 'cudnn' in error_msg
                    )
                    if is_cuda_error:
                        logger.warning(f"CUDA error in batch {i}: {e}")
                        if training_breaker:
                            training_breaker.record_failure("batch_processing", e)
                        # Clear gradients and memory
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue  # Skip to next batch
                    else:
                        raise  # Re-raise non-CUDA errors

                # Only step optimizer after accumulating gradients
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_iter):
                    # Gradient clipping (adaptive or fixed) (2025-12)
                    if use_grad_scaler:
                        scaler.unscale_(optimizer)
                    if adaptive_clipper is not None:
                        grad_norm = adaptive_clipper.update_and_clip(model.parameters())
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=fixed_clip_norm,
                        )

                    # Circuit breaker protection for optimizer step (2025-12)
                    try:
                        if use_grad_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        # Record successful batch processing
                        if training_breaker:
                            training_breaker.record_success("batch_processing")
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        error_msg = str(e).lower()
                        is_cuda_error = (
                            'cuda' in error_msg or 'out of memory' in error_msg or
                            'cublas' in error_msg or 'cudnn' in error_msg
                        )
                        if is_cuda_error:
                            logger.warning(f"CUDA error in optimizer step at batch {i}: {e}")
                            if training_breaker:
                                training_breaker.record_failure("batch_processing", e)
                            optimizer.zero_grad(set_to_none=True)
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                    # Update gradient metrics (every 100 batches to minimize overhead)
                    if i % 100 == 0 and HAS_PROMETHEUS and (not distributed or is_main_process()):
                        if GRADIENT_NORM:
                            GRADIENT_NORM.labels(config=config_label).set(
                                grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                            )
                        if adaptive_clipper is not None and GRADIENT_CLIP_NORM:
                            GRADIENT_CLIP_NORM.labels(config=config_label).set(
                                adaptive_clipper.current_max_norm
                            )

                    # Update integrated enhancements step counter
                    if enhancements_manager is not None:
                        enhancements_manager.update_step()

                        # Check if reanalysis should be triggered (2025-12)
                        # Reanalyzes historical data with current model for improved targets
                        if enhancements_manager.should_reanalyze():
                            if not distributed or is_main_process():
                                logger.info(
                                    "[Reanalysis] Triggering MuZero-style reanalysis of training data"
                                )
                                reanalyzed_path = enhancements_manager.process_reanalysis(
                                    data_path_str if data_path_str else None
                                )
                                if reanalyzed_path:
                                    logger.info(f"[Reanalysis] Complete: {reanalyzed_path}")
                                    # Note: Reanalyzed data is saved for next training run
                                    # or can be loaded via ReanalyzedDataset for mixing

                # Anomaly detection: check for NaN/Inf in loss (2025-12)
                if anomaly_detector is not None:
                    loss_val = loss.detach().item()
                    anomaly_step += 1
                    if anomaly_detector.check_loss(loss_val, anomaly_step):
                        anomaly_summary = anomaly_detector.get_summary()
                        consecutive = anomaly_summary.get('consecutive_anomalies', 0)
                        # Dec 29, 2025: Detect NaN/Inf explicitly for event emission
                        is_nan = loss_val != loss_val  # NaN != NaN is True
                        is_inf = not is_nan and (loss_val == float('inf') or loss_val == float('-inf'))
                        anomaly_type = 'nan' if is_nan else ('inf' if is_inf else 'spike')
                        logger.warning(
                            f"Training anomaly detected at batch {i}: type={anomaly_type}, "
                            f"total={anomaly_summary.get('total_anomalies', 0)}, "
                            f"consecutive={consecutive}"
                        )
                        # Update Prometheus anomaly counter
                        if HAS_PROMETHEUS and ANOMALY_DETECTIONS and (not distributed or is_main_process()):
                            ANOMALY_DETECTIONS.labels(config=config_label, type=anomaly_type).inc()

                        # Dec 29, 2025: Emit event for batch-level NaN/Inf (critical anomalies)
                        if (is_nan or is_inf) and HAS_TRAINING_EVENTS and (not distributed or is_main_process()):
                            try:
                                import asyncio
                                config_key = f"{config.board_type.value}_{num_players}p"
                                loop = asyncio.get_running_loop()
                                asyncio.ensure_future(emit_training_loss_anomaly(
                                    config_key=config_key,
                                    current_loss=0.0 if is_nan else loss_val,
                                    avg_loss=0.0,
                                    epoch=epoch + 1,
                                    anomaly_ratio=float('inf'),
                                    source="train.py",
                                    anomaly_type=anomaly_type,
                                    batch=i,
                                ))
                            except RuntimeError:
                                pass  # No event loop - OK in non-async context

                        # Auto-reduce learning rate on repeated anomalies (2025-12)
                        # Reduce by 30% after 3 consecutive anomalies (before circuit breaker)
                        if consecutive >= 3 and consecutive % 3 == 0:
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                new_lr = old_lr * 0.7
                                param_group['lr'] = new_lr
                                if not distributed or is_main_process():
                                    logger.warning(
                                        f"Auto-reduced LR due to {consecutive} consecutive anomalies: "
                                        f"{old_lr:.2e} -> {new_lr:.2e}"
                                    )

                        # Record failure with circuit breaker
                        if training_breaker:
                            training_breaker.record_failure("training_epoch")
                        # Skip this batch to avoid corrupting gradients
                        optimizer.zero_grad()
                        continue

                # Accumulate loss without .item() to avoid GPU sync per batch
                # Detach to prevent gradient accumulation, but keep on GPU
                train_loss += loss.detach()
                train_batches += 1

                # Track metrics for distributed reduction (uses detached tensor)
                if dist_metrics is not None:
                    dist_metrics.add(
                        'train_loss',
                        loss.detach(),
                        features.size(0),
                    )

                # Logging every 50 batches - reduced from 10 to minimize GPU sync overhead
                if i % 50 == 0 and (not distributed or is_main_process()):
                    # Only call .item() for logging, not accumulation
                    logger.info(
                        f"Epoch {epoch+1}, Batch {i}: "
                        f"Loss={loss.detach().item():.4f} "
                        f"(Val={value_loss.detach().item():.4f}, "
                        f"Pol={policy_loss.detach().item():.4f})"
                    )

            # Compute average training loss - call .item() only at end of epoch
            if distributed and dist_metrics is not None:
                # Synchronize metrics across all processes
                train_metrics = dist_metrics.reduce_and_reset(device=device)
                avg_train_loss = train_metrics.get('train_loss', 0.0)
            elif train_batches > 0:
                # Single .item() call at end of epoch for accumulated tensor
                avg_train_loss = (train_loss / train_batches).item()
            else:
                avg_train_loss = 0.0

            # Validation
            model.eval()
            val_loss = torch.tensor(0.0, device=device)  # Accumulate on GPU
            val_batches = 0
            val_policy_correct = 0  # Policy accuracy tracking
            val_policy_total = 0
            if dist_metrics is not None:
                dist_metrics.reset()

            # Select appropriate validation data source
            # For multi-player mode with streaming, use iter_with_mp()
            use_val_mp_iter = use_multi_player_loss and use_streaming and val_streaming_loader.has_multi_player_values
            if use_streaming:
                assert val_streaming_loader is not None
                # Use prefetch_loader for background prefetching if enabled
                if use_prefetch:
                    val_data_iter = prefetch_loader(
                        val_streaming_loader,
                        prefetch_count=prefetch_count,
                        pin_memory=pin_memory,
                        use_mp=use_val_mp_iter,
                        transfer_to_device=device if prefetch_to_device else None,
                    )
                elif use_val_mp_iter:
                    val_data_iter = val_streaming_loader.iter_with_mp()
                else:
                    val_data_iter = iter(val_streaming_loader)
                # Limit validation to ~20% of batches for streaming
                max_val_batches = max(
                    1,
                    len(val_streaming_loader) // 5,
                )
            else:
                assert val_loader is not None
                val_data_iter = iter(val_loader)
                max_val_batches = float('inf')

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_data_iter):
                    if val_batch_idx >= max_val_batches:
                        break

                    # Handle streaming, streaming with multi-player, and legacy batch formats
                    val_batch_num_players = None
                    val_batch_heuristics = None
                    if use_streaming:
                        if use_multi_player_loss and val_streaming_loader.has_multi_player_values:
                            (
                                (features, globals_vec),
                                (value_targets, policy_targets),
                                values_mp_batch,
                                val_batch_num_players,
                            ) = val_batch
                            if values_mp_batch is not None:
                                value_targets = values_mp_batch
                        else:
                            (
                                (features, globals_vec),
                                (value_targets, policy_targets),
                            ) = val_batch
                    else:
                        # Non-streaming: batch structure varies based on dataset config
                        val_batch_len = len(val_batch) if isinstance(val_batch, (list, tuple)) else 0
                        if val_batch_len == 6:
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                                val_batch_num_players,
                                val_batch_heuristics,
                            ) = val_batch
                        elif val_batch_len == 5:
                            fifth_elem = val_batch[4]
                            if fifth_elem.dtype in (torch.int64, torch.int32, torch.long):
                                (
                                    features,
                                    globals_vec,
                                    value_targets,
                                    policy_targets,
                                    val_batch_num_players,
                                ) = val_batch
                            else:
                                (
                                    features,
                                    globals_vec,
                                    value_targets,
                                    policy_targets,
                                    val_batch_heuristics,
                                ) = val_batch
                        else:
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                            ) = val_batch

                    # Transfer to device if not already there (prefetch may have done this)
                    if features.device != device:
                        features = features.to(device, non_blocking=True)
                        globals_vec = globals_vec.to(device, non_blocking=True)
                        value_targets = value_targets.to(device, non_blocking=True)
                        policy_targets = policy_targets.to(device, non_blocking=True)
                    if val_batch_num_players is not None and val_batch_num_players.device != device:
                        val_batch_num_players = val_batch_num_players.to(device, non_blocking=True)
                    if val_batch_heuristics is not None and val_batch_heuristics.device != device:
                        val_batch_heuristics = val_batch_heuristics.to(device, non_blocking=True)

                    # Pad policy targets if smaller than model policy_size
                    if hasattr(model, 'policy_size') and policy_targets.size(1) < model.policy_size:
                        pad_size = model.policy_size - policy_targets.size(1)
                        policy_targets = torch.nn.functional.pad(
                            policy_targets, (0, pad_size), value=0.0
                        )

                    # Autocast for mixed precision validation (matches training)
                    with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_torch_dtype):
                        # For DDP, forward through the wrapped model
                        # V5 models accept heuristics parameter
                        if model_version in ('v5', 'v5-gnn', 'v5-heavy'):
                            out = model(features, globals_vec, heuristics=val_batch_heuristics)
                        else:
                            out = model(features, globals_vec)
                        if isinstance(out, tuple) and len(out) == 3:
                            value_pred, policy_pred, rank_dist_pred = out
                        else:
                            value_pred, policy_pred = out
                            rank_dist_pred = None

                        # Use masked log_softmax for spatial policy heads
                        if detect_masked_policy_output(policy_pred):
                            valid_mask = (policy_targets > 0) | (policy_pred > -1e3)
                            policy_log_probs = masked_log_softmax(policy_pred, valid_mask)
                        else:
                            policy_log_probs = torch.log_softmax(policy_pred, dim=1)

                        # Policy accuracy: compare predicted move vs target move
                        pred_move = policy_pred.argmax(dim=1)
                        target_move = policy_targets.argmax(dim=1)
                        val_policy_correct += (pred_move == target_move).sum().item()
                        val_policy_total += pred_move.size(0)

                        # Use multi-player value loss for validation too
                        if use_multi_player_loss:
                            effective_val_num_players = (
                                val_batch_num_players if val_batch_num_players is not None
                                else num_players
                            )
                            v_loss = multi_player_value_loss(
                                value_pred, value_targets, effective_val_num_players
                            )
                        else:
                            if value_pred.ndim == 2:
                                value_pred_scalar = value_pred[:, 0]
                            else:
                                value_pred_scalar = value_pred
                            v_loss = value_criterion(
                                value_pred_scalar.reshape(-1),
                                value_targets.reshape(-1),
                            )
                        p_loss = masked_policy_kl(
                            policy_log_probs, policy_targets
                        )
                        # Jan 2026: Apply value_weight for consistency with training
                        loss = (config.value_weight * v_loss) + (config.policy_weight * p_loss)

                        # Rank distribution loss (V3+ multi-player head)
                        if (
                            rank_dist_pred is not None
                            and use_multi_player_loss
                            and value_targets.ndim == 2
                        ):
                            rank_targets, rank_mask = build_rank_targets(
                                value_targets,
                                effective_val_num_players,
                            )
                            rank_log_probs = torch.log(
                                rank_dist_pred.clamp_min(1e-8)
                            )
                            per_player_loss = -(
                                rank_targets * rank_log_probs
                            ).sum(dim=-1)
                            if torch.any(rank_mask):
                                rank_loss = per_player_loss[rank_mask].mean()
                                loss = loss + (config.rank_dist_weight * rank_loss)
                    # Accumulate on GPU without .item() sync
                    val_loss += loss.detach()
                    val_batches += 1

                    # Track metrics for distributed reduction (detached tensor)
                    if dist_metrics is not None:
                        dist_metrics.add(
                            'val_loss', loss.detach(), features.size(0)
                        )

                    # Collect calibration samples (value predictions vs actual outcomes)
                    if calibration_tracker is not None and not use_multi_player_loss:
                        # Get scalar predictions and targets for calibration
                        preds_cpu = value_pred_scalar.detach().cpu().numpy().flatten()
                        targets_cpu = value_targets.detach().cpu().numpy().flatten()
                        # Sample subset to avoid too much overhead
                        sample_size = min(len(preds_cpu), 100)
                        for i in range(sample_size):
                            calibration_tracker.add_sample(
                                float(preds_cpu[i]),
                                float(targets_cpu[i])
                            )

            # Compute average validation loss - single .item() at end
            if distributed and dist_metrics is not None:
                val_metrics = dist_metrics.reduce_and_reset(device=device)
                avg_val_loss = val_metrics.get('val_loss', 0.0)
            elif val_batches > 0:
                avg_val_loss = (val_loss / val_batches).item()
            else:
                avg_val_loss = 0.0

            # Compute policy accuracy
            avg_policy_accuracy = (
                val_policy_correct / val_policy_total if val_policy_total > 0 else 0.0
            )

            # Update training state for emergency checkpoints (2025-12)
            training_state.epoch = epoch
            training_state.avg_val_loss = avg_val_loss
            if avg_val_loss < training_state.best_val_loss:
                training_state.best_val_loss = avg_val_loss

            # Update scheduler at end of epoch
            if epoch_scheduler is not None:
                epoch_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_val_loss)

            # Apply curriculum LR scaling from training facade (December 2025)
            # Scales LR based on training progress: warmup → 1.0 → max_scale
            if training_facade is not None and training_facade.config.enable_curriculum_lr:
                try:
                    curriculum_scale = training_facade.get_curriculum_lr_scale()
                    if abs(curriculum_scale - 1.0) > 0.01:  # Only apply if meaningfully different
                        base_lr = optimizer.param_groups[0]['lr']
                        adjusted_lr = base_lr * curriculum_scale
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = adjusted_lr
                        if (epoch + 1) % 5 == 0:  # Log every 5 epochs
                            logger.debug(
                                f"[Curriculum LR] Epoch {epoch+1}: scale={curriculum_scale:.3f}, "
                                f"lr={adjusted_lr:.2e}"
                            )
                except (AttributeError, ValueError) as e:
                    logger.debug(f"[Curriculum LR] Failed to apply: {e}")

            # Apply evaluation feedback LR adjustment (December 2025)
            # This responds to EVALUATION_COMPLETED events and adjusts LR based on Elo trends
            if eval_feedback_handler is not None and eval_feedback_handler.should_adjust_lr():
                new_lr = eval_feedback_handler.apply_lr_adjustment(current_epoch=epoch)
                if new_lr is not None and (not distributed or is_main_process()):
                    logger.info(
                        f"[EvaluationFeedback] LR adjusted to {new_lr:.2e} based on Elo trend"
                    )

            # Always log current learning rate
            if not distributed or is_main_process():
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"  Current LR: {current_lr:.6f}")

            if not distributed or is_main_process():
                # Log epoch statistics with hot buffer info
                epoch_log = (
                    f"Epoch [{epoch+1}/{config.epochs_per_iter}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Policy Acc: {avg_policy_accuracy:.1%}"
                )
                if hot_buffer is not None:
                    hot_stats = hot_buffer.get_statistics()
                    epoch_log += f", Hot Buffer: {hot_stats['game_count']}/{hot_stats['max_size']} games"
                logger.info(epoch_log)

                # Publish training progress event to EventBus (2025-12)
                if HAS_EVENT_BUS and get_router is not None:
                    try:
                        router = get_router()
                        event_payload = {
                            "epoch": epoch + 1,
                            "total_epochs": config.epochs_per_iter,
                            "train_loss": float(avg_train_loss),
                            "val_loss": float(avg_val_loss),
                            "policy_accuracy": float(avg_policy_accuracy),
                            "lr": float(optimizer.param_groups[0]['lr']),
                            "config": f"{config.board_type.value}_{num_players}p",
                        }
                        # Add hot buffer stats if available
                        if hot_buffer is not None:
                            event_payload["hot_buffer"] = hot_buffer.get_statistics()
                        router.publish_sync(DataEvent(
                            event_type=DataEventType.TRAINING_PROGRESS,
                            payload=event_payload,
                            source="train",
                        ))
                    except (RuntimeError, ConnectionError, TimeoutError) as e:
                        # Event emission can fail due to async runtime or network issues
                        logger.debug(f"Failed to publish training progress event: {e}")

            # LossMonitor: Check for learning stall (Jan 2026 fix)
            if not loss_monitor.record(epoch, avg_train_loss, avg_val_loss):
                logger.warning(
                    f"[LossMonitor] Training stalled - loss not improving. "
                    f"Consider checking data quality or model architecture. "
                    f"Summary: {loss_monitor.get_summary()}"
                )
                # Don't break early - let existing early stopping handle it
                # This is just for observability/alerting

            # Overfitting detection: warn if validation diverges significantly from train
            if avg_train_loss > 0 and epoch >= 3:
                overfitting_ratio = (avg_val_loss - avg_train_loss) / avg_train_loss
                if overfitting_ratio > 0.25 and (not distributed or is_main_process()):
                    logger.warning(
                        f"Overfitting detected: {overfitting_ratio*100:.1f}% divergence "
                        f"(train={avg_train_loss:.4f}, val={avg_val_loss:.4f})"
                    )

            # Regression detection: check if validation loss has regressed (2025-12)
            # Uses unified RegressionDetector for consistent detection across modules
            # January 2026: Enhanced to skip checkpoint saving on significant regression
            skip_checkpoint_on_regression = False
            if (HAS_REGRESSION_DETECTOR and get_regression_detector is not None
                    and epoch >= 2 and (not distributed or is_main_process())):
                try:
                    from app.training.regression_detector import RegressionSeverity
                    regression_detector = get_regression_detector(connect_event_bus=True)
                    model_id = f"{config.board_type.value}_{num_players}p"

                    # Set baseline on first check
                    if epoch == 2:
                        regression_detector.set_baseline(
                            model_id=model_id,
                            elo=best_val_loss * -1000,  # Convert loss to pseudo-Elo
                        )

                    # Check for regression (using inverted loss as pseudo-Elo)
                    regression_event = regression_detector.check_regression(
                        model_id=model_id,
                        current_elo=avg_val_loss * -1000,
                        games_played=epoch + 1,
                    )

                    if regression_event is not None:
                        logger.warning(
                            f"[RegressionDetector] {regression_event.severity.value.upper()} regression: "
                            f"val_loss {avg_val_loss:.4f} vs best {best_val_loss:.4f} "
                            f"({regression_event.reason})"
                        )
                        # January 2026: Skip checkpoint on MODERATE or worse regression
                        # This prevents polluting the model pool with regressing checkpoints
                        if regression_event.severity in (
                            RegressionSeverity.MODERATE,
                            RegressionSeverity.SEVERE,
                            RegressionSeverity.CRITICAL,
                        ):
                            skip_checkpoint_on_regression = True
                            logger.warning(
                                f"[RegressionDetector] Skipping checkpoint save due to "
                                f"{regression_event.severity.value} regression"
                            )
                except (AttributeError, ValueError, TypeError, ImportError) as e:
                    # Missing attributes, invalid values, type errors, or import issues
                    logger.debug(f"Regression detection error: {e}")

            # Record per-epoch losses for downstream analysis
            epochs_completed = epoch + 1
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'val_loss': float(avg_val_loss),
                'policy_accuracy': float(avg_policy_accuracy),
                'lr': float(optimizer.param_groups[0]['lr']),
            }

            # Compute and log calibration metrics every 5 epochs
            if calibration_tracker is not None and (epoch + 1) % 5 == 0:
                calibration_report = calibration_tracker.compute_current_calibration()
                if calibration_report is not None:
                    epoch_record['calibration_ece'] = calibration_report.ece
                    epoch_record['calibration_mce'] = calibration_report.mce
                    epoch_record['calibration_overconfidence'] = calibration_report.overconfidence
                    if not distributed or is_main_process():
                        logger.info(
                            f"  Calibration: ECE={calibration_report.ece:.4f}, "
                            f"MCE={calibration_report.mce:.4f}, "
                            f"Overconfidence={calibration_report.overconfidence:.4f}"
                        )
                        if calibration_report.optimal_temperature is not None:
                            logger.info(
                                f"  Optimal temperature: {calibration_report.optimal_temperature:.3f}"
                            )

            epoch_losses.append(epoch_record)

            # Training facade: log statistics and prepare for next epoch (2025-12)
            # Provides unified stats for hard mining, curriculum LR, and freshness weighting
            if training_facade is not None and (not distributed or is_main_process()):
                try:
                    facade_stats = training_facade.on_epoch_end()
                    if facade_stats.get('mining_active', False):
                        logger.info(
                            f"  [Training Facade] "
                            f"tracked={facade_stats.get('tracked_samples', 0)}, "
                            f"hard_frac={facade_stats.get('hard_examples_fraction', 0):.1%}, "
                            f"mean_loss={facade_stats.get('mean_per_sample_loss', 0):.4f}, "
                            f"lr_scale={facade_stats.get('curriculum_lr_scale', 1.0):.3f}"
                        )
                    # Add to epoch record for analysis
                    epoch_record['facade_mean_loss'] = facade_stats.get('mean_loss', 0)
                    epoch_record['facade_hard_fraction'] = facade_stats.get('hard_examples_fraction', 0)
                    epoch_record['facade_curriculum_lr_scale'] = facade_stats.get('curriculum_lr_scale', 1.0)
                    epoch_record['facade_mining_active'] = facade_stats.get('mining_active', False)
                except (AttributeError, ValueError) as e:
                    logger.debug(f"[Training Facade] on_epoch_end error: {e}")

            # Fallback: Log hard example mining statistics (2025-12)
            elif hard_example_miner is not None and (not distributed or is_main_process()):
                mining_stats = hard_example_miner.get_statistics()
                if mining_stats.get('mining_active', False):
                    logger.info(
                        f"  [Hard Example Mining] "
                        f"tracked={mining_stats.get('tracked_examples', 0)}, "
                        f"mean_loss={mining_stats.get('mean_loss', 0):.4f}, "
                        f"loss_p90={mining_stats.get('loss_p90', 0):.4f}"
                    )
                    # Add to epoch record for analysis
                    epoch_record['hard_mining_mean_loss'] = mining_stats.get('mean_loss', 0)
                    epoch_record['hard_mining_p90_loss'] = mining_stats.get('loss_p90', 0)
                    epoch_record['hard_mining_tracked'] = mining_stats.get('tracked_examples', 0)

            # Emit epoch completed event for curriculum feedback (December 2025)
            # This enables mid-training curriculum updates based on epoch progress
            if HAS_EPOCH_EVENTS and publish_epoch_completed and (not distributed or is_main_process()):
                try:
                    import asyncio
                    config_key = f"{config.board_type.value}_{num_players}p"
                    try:
                        # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                        loop = asyncio.get_running_loop()
                        asyncio.ensure_future(publish_epoch_completed(
                                config_key=config_key,
                                epoch=epoch + 1,
                                total_epochs=config.epochs_per_iter,
                                train_loss=avg_train_loss,
                                val_loss=avg_val_loss,
                                learning_rate=optimizer.param_groups[0]['lr'],
                            ))
                    except RuntimeError:
                        # No event loop running - use fire-and-forget
                        pass
                except (RuntimeError, ConnectionError, TimeoutError) as e:
                    # Event emission can fail due to async runtime or network issues
                    logger.debug(f"Failed to emit epoch completed event: {e}")

            # Emit training loss events for feedback loops (Phase 21.2 - Dec 2025)
            # Only emit on main process to avoid duplicates in DDP
            if HAS_TRAINING_EVENTS and (not distributed or is_main_process()):
                try:
                    import asyncio
                    config_key = f"{config.board_type.value}_{num_players}p"

                    # Calculate average loss over recent epochs (last 5 or all available)
                    recent_losses = [e.get('avg_val_loss', e.get('avg_train_loss', 0.0))
                                     for e in epoch_losses[-5:] if e]
                    if recent_losses:
                        avg_recent_loss = sum(recent_losses) / len(recent_losses)

                        # Detect anomaly: current loss > 2x average (significant spike)
                        if avg_val_loss > avg_recent_loss * 2.0 and len(epoch_losses) > 2:
                            anomaly_ratio = avg_val_loss / avg_recent_loss if avg_recent_loss > 0 else 0.0
                            logger.warning(
                                f"[TRAINING ANOMALY] Loss spike detected: {avg_val_loss:.4f} vs avg {avg_recent_loss:.4f} "
                                f"(ratio: {anomaly_ratio:.2f}x)"
                            )
                            # Fire-and-forget async emission
                            try:
                                # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                                loop = asyncio.get_running_loop()
                                asyncio.ensure_future(emit_training_loss_anomaly(
                                    config_key=config_key,
                                    current_loss=avg_val_loss,
                                    avg_loss=avg_recent_loss,
                                    epoch=epoch + 1,
                                    anomaly_ratio=anomaly_ratio,
                                    source="train.py",
                                ))
                            except RuntimeError:
                                # No event loop - skip emission (OK in non-async context)
                                pass

                        # Emit trend every 5 epochs
                        if (epoch + 1) % 5 == 0 and len(epoch_losses) >= 5:
                            # Compare last 5 epochs to previous 5 epochs
                            current_avg = sum(recent_losses) / len(recent_losses)
                            older_losses = [e.get('avg_val_loss', e.get('avg_train_loss', 0.0))
                                            for e in epoch_losses[-10:-5] if e]
                            if older_losses:
                                previous_avg = sum(older_losses) / len(older_losses)
                                improvement_rate = (previous_avg - current_avg) / previous_avg if previous_avg > 0 else 0.0

                                # Classify trend: >5% improvement = improving, <-5% = degrading, else stalled
                                if improvement_rate > 0.05:
                                    trend = "improving"
                                elif improvement_rate < -0.05:
                                    trend = "degrading"
                                else:
                                    trend = "stalled"

                                logger.info(
                                    f"[TRAINING TREND] {trend} (epoch {epoch+1}): "
                                    f"current_avg={current_avg:.4f}, previous_avg={previous_avg:.4f}, "
                                    f"improvement_rate={improvement_rate:.2%}"
                                )
                                try:
                                    # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                                    loop = asyncio.get_running_loop()
                                    asyncio.ensure_future(emit_training_loss_trend(
                                        config_key=config_key,
                                        trend=trend,
                                        epoch=epoch + 1,
                                        current_loss=current_avg,
                                        previous_loss=previous_avg,
                                        improvement_rate=improvement_rate,
                                        source="train.py",
                                    ))
                                except RuntimeError:
                                    pass

                        # Dec 29, 2025: Stricter plateau detection (0.1% over 10 epochs)
                        # This catches subtle plateaus that the 5-epoch/5% threshold misses
                        if (epoch + 1) % 10 == 0 and len(epoch_losses) >= 10:
                            last_10_losses = [e.get('avg_val_loss', e.get('avg_train_loss', 0.0))
                                              for e in epoch_losses[-10:] if e]
                            prev_10_losses = [e.get('avg_val_loss', e.get('avg_train_loss', 0.0))
                                              for e in epoch_losses[-20:-10] if e]
                            if len(last_10_losses) >= 10 and len(prev_10_losses) >= 5:
                                last_10_avg = sum(last_10_losses) / len(last_10_losses)
                                prev_10_avg = sum(prev_10_losses) / len(prev_10_losses)
                                long_term_improvement = (prev_10_avg - last_10_avg) / prev_10_avg if prev_10_avg > 0 else 0.0

                                # Plateau: < 0.1% improvement over 10 epochs
                                if abs(long_term_improvement) < 0.001:
                                    # Dec 29, 2025: Plateau type analysis (overfitting vs data limitation)
                                    # Train/val gap indicates overfitting, small gap indicates data limitation
                                    last_10_train = [e.get('avg_train_loss', 0.0) for e in epoch_losses[-10:] if e]
                                    last_10_train_avg = sum(last_10_train) / len(last_10_train) if last_10_train else 0.0
                                    train_val_gap = last_10_avg - last_10_train_avg  # val - train

                                    if train_val_gap > 0.05:
                                        plateau_type = "overfitting"
                                        recommendation = "reduce_epochs"
                                        exploration_boost = 1.5  # Higher boost to diversify data
                                    else:
                                        plateau_type = "data_limitation"
                                        recommendation = "more_games"
                                        exploration_boost = 1.3  # Moderate boost

                                    logger.warning(
                                        f"[TRAINING PLATEAU] Detected at epoch {epoch+1}: "
                                        f"<0.1% improvement over 10 epochs "
                                        f"(last_10={last_10_avg:.5f}, prev_10={prev_10_avg:.5f}, "
                                        f"type={plateau_type}, gap={train_val_gap:.4f})"
                                    )
                                    try:
                                        loop = asyncio.get_running_loop()
                                        # Emit TRAINING_LOSS_TREND for backward compatibility
                                        asyncio.ensure_future(emit_training_loss_trend(
                                            config_key=config_key,
                                            trend="plateau",
                                            epoch=epoch + 1,
                                            current_loss=last_10_avg,
                                            previous_loss=prev_10_avg,
                                            improvement_rate=long_term_improvement,
                                            source="train.py",
                                            window_size=10,
                                        ))
                                        # Dec 29, 2025: Emit PLATEAU_DETECTED with type analysis
                                        # January 2026 - migrated to event_router
                                        from app.coordination.event_emission_helpers import safe_emit_event
                                        safe_emit_event(
                                            "PLATEAU_DETECTED",
                                            {
                                                "metric_name": "validation_loss",
                                                "current_value": last_10_avg,
                                                "best_value": prev_10_avg,
                                                "epochs_since_improvement": 10,
                                                "plateau_type": plateau_type,  # "overfitting" or "data_limitation"
                                                "config_key": config_key,
                                                "epoch": epoch + 1,
                                                "recommendation": recommendation,
                                                "exploration_boost": exploration_boost,
                                                "train_val_gap": train_val_gap,
                                                "source": "train.py",
                                            },
                                            context="train.py",
                                        )
                                    except RuntimeError:
                                        pass

                except (RuntimeError, ConnectionError, TimeoutError, AttributeError) as e:
                    # Event emission failures, network issues, or missing attributes
                    logger.debug(f"Failed to emit training events: {e}")

            # Update Prometheus metrics (only on main process)
            if HAS_PROMETHEUS and (not distributed or is_main_process()):
                config_label = f"{config.board_type.value}_{num_players}p"
                TRAINING_EPOCHS.labels(config=config_label).inc()
                TRAINING_LOSS.labels(config=config_label, loss_type='train').set(avg_train_loss)
                TRAINING_LOSS.labels(config=config_label, loss_type='val').set(avg_val_loss)
                TRAINING_DURATION.labels(config=config_label).observe(
                    epoch_record.get('epoch_duration', 0.0)
                )
                if 'calibration_ece' in epoch_record:
                    CALIBRATION_ECE.labels(config=config_label).set(epoch_record['calibration_ece'])
                    CALIBRATION_MCE.labels(config=config_label).set(epoch_record['calibration_mce'])

            # Record to dashboard metrics collector for persistent storage (2025-12)
            if metrics_collector is not None and (not distributed or is_main_process()):
                try:
                    # Get GPU memory usage if available
                    gpu_memory_mb = 0.0
                    if device.type == 'cuda':
                        gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

                    metrics_collector.record_training_step(
                        epoch=epoch + 1,
                        step=epoch_record.get('train_batches', 0),
                        loss=avg_val_loss,  # Use validation loss as primary
                        policy_loss=epoch_record.get('avg_policy_loss', 0.0),
                        value_loss=epoch_record.get('avg_value_loss', 0.0),
                        accuracy=avg_policy_accuracy,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        batch_size=config.batch_size,
                        samples_per_second=epoch_record.get('samples_per_second', 0.0),
                        gpu_memory_mb=gpu_memory_mb,
                        model_id=config.model_id,
                    )
                except (OSError, RuntimeError, AttributeError) as e:
                    # File I/O errors, runtime errors, or missing attributes
                    logger.debug(f"Failed to record metrics to dashboard: {e}")

            # Check early stopping (only on main process for DDP)
            # Get model for checkpointing (unwrap DDP if needed)
            model_to_save = cast(
                nn.Module,
                model.module if distributed else model,
            )

            # Check integrated enhancements early stopping (based on Elo tracking)
            if enhancements_manager is not None and enhancements_manager.should_early_stop():
                if not distributed or is_main_process():
                    logger.info(
                        f"Enhancements manager triggered early stop at epoch {epoch+1} "
                        "(Elo regression detected)"
                    )
                break

            # Check baseline gating - warn if model failing against basic baselines
            if enhancements_manager is not None:
                passes_gating, failed_baselines, consecutive_failures = (
                    enhancements_manager.get_baseline_gating_status()
                )
                if not passes_gating and (not distributed or is_main_process()):
                    logger.warning(
                        f"[BASELINE GATING] Epoch {epoch+1}: Model failed baseline thresholds "
                        f"({', '.join(failed_baselines)}). Consecutive failures: {consecutive_failures}"
                    )
                    if consecutive_failures >= 5:
                        logger.error(
                            f"[BASELINE GATING] {consecutive_failures} consecutive failures! "
                            "Model may be overfitting to neural-vs-neural play. "
                            "Consider: more diverse training data, regularization, or early stopping."
                        )

            if early_stopper is not None:
                # Get current Elo from enhancements manager if available
                current_elo = None
                if enhancements_manager is not None:
                    current_elo = enhancements_manager.get_current_elo()

                # Use should_stop() with Elo support instead of __call__
                should_stop = early_stopper.should_stop(
                    val_loss=avg_val_loss,
                    current_elo=current_elo,
                    model=model_to_save,
                    epoch=epoch,
                )
                # December 2025: Enforce minimum training epochs before early stopping
                # This prevents stopping at 3-5 epochs when 15-20+ are needed for 2000+ Elo
                if should_stop and epoch + 1 < MIN_TRAINING_EPOCHS:
                    if not distributed or is_main_process():
                        logger.info(
                            f"Early stopping suppressed at epoch {epoch+1} (minimum: {MIN_TRAINING_EPOCHS})"
                        )
                    should_stop = False
                if should_stop:
                    if not distributed or is_main_process():
                        elo_info = f", best Elo: {early_stopper.best_elo:.1f}" if early_stopper.best_elo > float('-inf') else ""
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1} "
                            f"(best loss: {early_stopper.best_loss:.4f}{elo_info})"
                        )

                        # Emit TRAINING_EARLY_STOPPED event (December 2025 - feedback loop)
                        # This triggers curriculum boost for this config
                        try:
                            import asyncio
                            from app.coordination.event_router import emit_training_early_stopped

                            config_key = f"{config.board_type}_{num_players}p"
                            best_elo = early_stopper.best_elo if early_stopper.best_elo > float('-inf') else None
                            epochs_without_improvement = early_stopper.counter if hasattr(early_stopper, 'counter') else 0

                            # Use fire-and-forget emit via event loop
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(emit_training_early_stopped(
                                    config_key=config_key,
                                    epoch=epoch + 1,
                                    best_loss=float(early_stopper.best_loss),
                                    final_loss=float(avg_val_loss),
                                    best_elo=best_elo,
                                    reason="loss_stagnation",
                                    epochs_without_improvement=epochs_without_improvement,
                                ))
                            except RuntimeError:
                                # No running loop - create one for sync emit
                                asyncio.run(emit_training_early_stopped(
                                    config_key=config_key,
                                    epoch=epoch + 1,
                                    best_loss=float(early_stopper.best_loss),
                                    final_loss=float(avg_val_loss),
                                    best_elo=best_elo,
                                    reason="loss_stagnation",
                                    epochs_without_improvement=epochs_without_improvement,
                                ))

                            logger.info(f"[train] Emitted TRAINING_EARLY_STOPPED for {config_key}")
                        except (RuntimeError, ConnectionError, TimeoutError) as e:
                            # Event emission can fail due to async runtime or network issues
                            logger.warning(f"Failed to emit TRAINING_EARLY_STOPPED: {e}")
                        # Restore best weights
                        early_stopper.restore_best_weights(model_to_save)
                        # Save final checkpoint with best weights
                        final_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"checkpoint_early_stop_epoch_{epoch+1}.pth",
                        )
                        _final_checkpoint_path = final_checkpoint_path  # Track for event emission
                        if async_checkpointer is not None:
                            async_checkpointer.save_async(
                                model_to_save,
                                optimizer,
                                epoch,
                                early_stopper.best_loss,
                                final_checkpoint_path,
                                scheduler=epoch_scheduler,
                                early_stopping=early_stopper,
                            )
                        else:
                            save_checkpoint(
                                model_to_save,
                                optimizer,
                                epoch,
                                early_stopper.best_loss,
                                final_checkpoint_path,
                                scheduler=epoch_scheduler,
                                early_stopping=early_stopper,
                            )
                        # Save best model with versioning and config validation
                        save_model_checkpoint(
                            model_to_save,
                            save_path,
                            training_info={
                                'epoch': epoch,
                                'loss': float(early_stopper.best_loss),
                                'early_stopped': True,
                            },
                            board_type=config.board_type,
                            num_players=num_players,
                        )
                        logger.info("Best model saved to %s", save_path)
                    # Mark early stopping as successful completion (for hardened event emission)
                    _training_completed_normally = True
                    break

            # Checkpoint at intervals (only on main process)
            if (
                checkpoint_interval > 0
                and (epoch + 1) % checkpoint_interval == 0
            ) and (not distributed or is_main_process()):
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_epoch_{epoch+1}.pth",
                )
                if async_checkpointer is not None:
                    async_checkpointer.save_async(
                        model_to_save,
                        optimizer,
                        epoch,
                        avg_val_loss,
                        checkpoint_path,
                        scheduler=epoch_scheduler,
                        early_stopping=early_stopper,
                    )
                else:
                    save_checkpoint(
                        model_to_save,
                        optimizer,
                        epoch,
                        avg_val_loss,
                        checkpoint_path,
                        scheduler=epoch_scheduler,
                        early_stopping=early_stopper,
                    )
                # Track for circuit breaker rollback (2025-12)
                _last_good_checkpoint_path = checkpoint_path
                _last_good_epoch = epoch

            # Save best model (only on main process)
            # January 2026: Skip saving if significant regression detected
            if avg_val_loss < best_val_loss and not skip_checkpoint_on_regression:
                best_val_loss = avg_val_loss
                best_train_loss_at_best_val = avg_train_loss  # Track for overfitting detection
                if not distributed or is_main_process():
                    # Save with versioning metadata and config validation
                    save_model_checkpoint(
                        model_to_save,
                        save_path,
                        training_info={
                            'epoch': epoch + 1,
                            'samples_seen': train_size * (epoch + 1),
                            'val_loss': float(avg_val_loss),
                            'train_loss': float(avg_train_loss),
                        },
                        board_type=config.board_type,
                        num_players=num_players,
                    )
                    logger.info(
                        "  New best model saved (Val Loss: %.4f)",
                        avg_val_loss,
                    )

                    # Collect checkpoint for averaging (2025-12)
                    if checkpoint_averager is not None:
                        checkpoint_averager.add_checkpoint(
                            model_to_save.state_dict(),
                            epoch=epoch,
                        )

                    # Save timestamped checkpoint for history tracking
                    from datetime import datetime as dt
                    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                    version_path = save_path.replace(
                        ".pth",
                        f"_{timestamp}.pth",
                    )
                    save_model_checkpoint(
                        model_to_save,
                        version_path,
                        training_info={
                            'epoch': epoch + 1,
                            'samples_seen': train_size * (epoch + 1),
                            'val_loss': float(avg_val_loss),
                            'train_loss': float(avg_train_loss),
                            'timestamp': timestamp,
                        },
                        board_type=config.board_type,
                        num_players=num_players,
                    )
                    logger.info(
                        "  Versioned checkpoint saved: %s",
                        version_path,
                    )

            # Knowledge distillation check (2025-12)
            # Distills ensemble knowledge from best checkpoints into current model
            if enhancements_manager is not None:
                # Set checkpoint directory so distillation can find teacher models
                enhancements_manager.set_checkpoint_dir(checkpoint_dir)

                if enhancements_manager.should_distill(epoch + 1):
                    if not distributed or is_main_process():
                        logger.info(
                            f"[Distillation] Triggering ensemble distillation at epoch {epoch+1}"
                        )
                        # Use the training dataloader for distillation
                        distillation_success = enhancements_manager.run_distillation(
                            current_epoch=epoch + 1,
                            dataloader=train_loader,
                        )
                        if distillation_success:
                            logger.info(
                                f"[Distillation] Epoch {epoch+1}: Successfully distilled "
                                "ensemble knowledge into model"
                            )

            # Beat heartbeat at end of each epoch to signal health
            if heartbeat_monitor is not None:
                heartbeat_monitor.beat()

            # Record successful epoch completion with circuit breaker (2025-12)
            if training_breaker:
                training_breaker.record_success("training_epoch")
        else:
            # Final checkpoint at end of training (if not early stopped).
            # This else clause is for the for-loop and executes if no break
            # occurred.
            if not distributed or is_main_process():
                model_to_save_final = cast(
                    nn.Module,
                    model.module if distributed else model,
                )
                final_checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_final_epoch_{config.epochs_per_iter}.pth",
                )
                _final_checkpoint_path = final_checkpoint_path  # Track for event emission
                if async_checkpointer is not None:
                    async_checkpointer.save_async(
                        model_to_save_final,
                        optimizer,
                        config.epochs_per_iter - 1,
                        avg_val_loss,
                        final_checkpoint_path,
                        scheduler=epoch_scheduler,
                        early_stopping=early_stopper,
                    )
                else:
                    save_checkpoint(
                        model_to_save_final,
                        optimizer,
                        config.epochs_per_iter - 1,
                        avg_val_loss,
                        final_checkpoint_path,
                        scheduler=epoch_scheduler,
                        early_stopping=early_stopper,
                    )
                logger.info("Training completed. Final checkpoint saved.")

                # Apply checkpoint averaging if enabled (2025-12), with overfitting detection (January 2026)
                # Check for overfitting: if val_loss diverged significantly from train_loss, keep best checkpoint
                overfitting_detected = False
                if best_train_loss_at_best_val > 0 and best_val_loss != float('inf'):
                    divergence = (best_val_loss - best_train_loss_at_best_val) / best_train_loss_at_best_val
                    if divergence > overfit_divergence_threshold:
                        overfitting_detected = True
                        logger.warning(
                            f"[Overfitting Detected] Val/Train divergence: {divergence:.1%} > {overfit_divergence_threshold:.0%} threshold "
                            f"(train={best_train_loss_at_best_val:.4f}, val={best_val_loss:.4f})"
                        )

                # Skip averaging if overfitting detected and prefer_best_on_overfit is enabled
                skip_averaging = prefer_best_on_overfit and overfitting_detected
                if skip_averaging:
                    logger.info(
                        "[Best Checkpoint Selection] Keeping best validation loss checkpoint (skipping averaging due to overfitting)"
                    )
                    # Jan 12, 2026 FIX: Actually restore best weights when overfitting detected
                    # Previously we just skipped averaging but still saved final epoch weights
                    if early_stopper is not None and hasattr(early_stopper, 'restore_best_weights'):
                        logger.info(
                            f"[Auto-Restore Best] Restoring weights from best epoch (val_loss={early_stopper.best_loss:.4f})"
                        )
                        early_stopper.restore_best_weights(model_to_save_final)
                    if checkpoint_averager is not None:
                        checkpoint_averager.cleanup()
                elif checkpoint_averager is not None and checkpoint_averager.num_stored >= 2:
                    logger.info(
                        f"[Checkpoint Averaging] Averaging {checkpoint_averager.num_stored} checkpoints..."
                    )
                    try:
                        averaged_state_dict = checkpoint_averager.get_averaged_state_dict()

                        # Save averaged model separately
                        averaged_path = save_path.replace(".pth", "_averaged.pth")
                        model_to_save_final.load_state_dict(averaged_state_dict)
                        save_model_checkpoint(
                            model_to_save_final,
                            averaged_path,
                            training_info={
                                'epoch': config.epochs_per_iter,
                                'averaged_checkpoints': checkpoint_averager.num_stored,
                                'checkpoint_averaging': True,
                            },
                            board_type=config.board_type,
                            num_players=num_players,
                        )

                        # Overwrite main save_path with averaged weights (typically better)
                        save_model_checkpoint(
                            model_to_save_final,
                            save_path,
                            training_info={
                                'epoch': config.epochs_per_iter,
                                'averaged_checkpoints': checkpoint_averager.num_stored,
                                'checkpoint_averaging': True,
                            },
                            board_type=config.board_type,
                            num_players=num_players,
                        )
                        logger.info(
                            f"[Checkpoint Averaging] Saved averaged model ({checkpoint_averager.num_stored} checkpoints) to {save_path}"
                        )
                    except (OSError, RuntimeError, ValueError, TypeError, MemoryError) as e:
                        logger.warning(f"[Checkpoint Averaging] Failed to average checkpoints: {e}")
                    finally:
                        checkpoint_averager.cleanup()
                elif checkpoint_averager is not None:
                    logger.info(
                        f"[Checkpoint Averaging] Skipped: only {checkpoint_averager.num_stored} checkpoint(s) available (need >= 2)"
                    )
                    checkpoint_averager.cleanup()

                # Jan 12, 2026 FIX: Always ensure best weights are restored before final save
                # This handles the case where training completes without early stopping or
                # overfitting detection, but the final epoch is worse than the best epoch.
                if early_stopper is not None and not skip_averaging:
                    final_val_loss = avg_val_loss if 'avg_val_loss' in dir() else float('inf')
                    if final_val_loss > early_stopper.best_loss * 1.05:  # 5% tolerance
                        logger.info(
                            f"[Auto-Restore Best] Final loss ({final_val_loss:.4f}) > best loss ({early_stopper.best_loss:.4f}). "
                            f"Restoring best weights before final save."
                        )
                        early_stopper.restore_best_weights(model_to_save_final)
                        # Re-save with best weights
                        save_model_checkpoint(
                            model_to_save_final,
                            save_path,
                            training_info={
                                'epoch': early_stopper.best_epoch if hasattr(early_stopper, 'best_epoch') else config.epochs_per_iter,
                                'best_val_loss': early_stopper.best_loss,
                                'auto_restored': True,
                            },
                            board_type=config.board_type,
                            num_players=num_players,
                        )

                # Log reanalysis summary if enabled (2025-12)
                if enhancements_manager is not None:
                    reanalysis_stats = enhancements_manager.get_reanalysis_stats()
                    if reanalysis_stats.get("enabled") and reanalysis_stats.get("positions_reanalyzed", 0) > 0:
                        logger.info(
                            f"[Reanalysis Summary] "
                            f"Positions: {reanalysis_stats['positions_reanalyzed']}, "
                            f"Games: {reanalysis_stats['games_reanalyzed']}, "
                            f"Blend ratio: {reanalysis_stats['blend_ratio']:.2f}"
                        )

                    # Log distillation summary if enabled (2025-12)
                    distillation_stats = enhancements_manager.get_distillation_stats()
                    if distillation_stats.get("enabled") and distillation_stats.get("last_distillation_epoch", 0) > 0:
                        logger.info(
                            f"[Distillation Summary] "
                            f"Last epoch: {distillation_stats['last_distillation_epoch']}, "
                            f"Teachers: {distillation_stats['available_teachers']}, "
                            f"Temperature: {distillation_stats['temperature']:.1f}"
                        )

                # Publish training completed event (2025-12)
                if HAS_EVENT_BUS and get_router is not None:
                    try:
                        router = get_router()
                        event_payload = {
                            "epochs_completed": epochs_completed,
                            "best_val_loss": float(best_val_loss),
                            "final_train_loss": float(avg_train_loss),
                            "final_val_loss": float(avg_val_loss),
                            "config": f"{config.board_type.value}_{num_players}p",
                            "config_key": f"{config.board_type.value}_{num_players}p",
                            "checkpoint_path": str(final_checkpoint_path),
                            "trigger_evaluation": True,  # Trigger automatic evaluation
                            # model_path for FeedbackLoopController (Dec 2025 integration fix)
                            "model_path": str(save_path),
                            # policy_accuracy for evaluation trigger threshold check
                            "policy_accuracy": float(avg_policy_accuracy),
                            # Feb 2026: Include training data stats for generation tracking
                            "training_samples": _total_samples,
                            "training_games": _num_data_files,
                        }
                        # Add reanalysis and distillation stats to event payload
                        if enhancements_manager is not None:
                            reanalysis_stats = enhancements_manager.get_reanalysis_stats()
                            if reanalysis_stats.get("enabled"):
                                event_payload["reanalysis"] = reanalysis_stats
                            distillation_stats = enhancements_manager.get_distillation_stats()
                            if distillation_stats.get("enabled"):
                                event_payload["distillation"] = distillation_stats
                        router.publish_sync(DataEvent(
                            event_type=DataEventType.TRAINING_COMPLETED,
                            payload=event_payload,
                            source="train",
                        ))
                    except (RuntimeError, ConnectionError, TimeoutError) as e:
                        # Event emission can fail due to async runtime or network issues
                        logger.debug(f"Failed to publish training completed event: {e}")

                # Emit curriculum update event (December 2025)
                # Triggers curriculum reweighting when policy accuracy crosses threshold
                # January 2026 - migrated to event_router
                try:
                    from app.coordination.event_emission_helpers import safe_emit_event

                    config_key = f"{config.board_type.value}_{num_players}p"
                    policy_accuracy_threshold = 0.75

                    # Check if this config should have its curriculum weight increased
                    # High policy accuracy indicates strong learning - boost priority
                    trigger_reweight = avg_policy_accuracy >= policy_accuracy_threshold

                    if trigger_reweight:
                        # Increase curriculum weight for well-performing configs
                        new_weight = 1.0 + (avg_policy_accuracy - 0.5) * 0.5  # 0.75 acc → 1.125 weight
                        safe_emit_event(
                            "CURRICULUM_UPDATED",
                            {
                                "config_key": config_key,
                                "new_weight": new_weight,
                                "trigger": "training_complete",
                                "policy_accuracy": avg_policy_accuracy,
                                "value_loss": avg_val_loss,
                            },
                            context="train.py",
                        )
                        logger.info(
                            f"[Curriculum] Triggered reweight for {config_key}: "
                            f"policy_acc={avg_policy_accuracy:.1%} → weight={new_weight:.3f}"
                        )
                except ImportError:
                    pass  # Event emitters not available
                except (RuntimeError, ConnectionError, TimeoutError, AttributeError) as e:
                    # Event emission failures, network issues, or missing attributes
                    logger.debug(f"Failed to emit curriculum update: {e}")

                # Mark training as completed successfully (for hardened event emission)
                _training_completed_normally = True

    except (RuntimeError, ValueError, OSError, KeyError) as e:
        # RuntimeError: CUDA/tensor operations, training loop errors
        # ValueError: invalid training parameters or data
        # OSError: checkpoint save/load failures
        # KeyError: missing required data or config keys
        # Capture exception for hardened event emission in finally block
        _training_exception = e
        raise  # Re-raise after capturing
    finally:
        # ==========================================================================
        # Hardened Event Emission (December 2025)
        # ==========================================================================
        # ALWAYS emit training completion/failure event, even if training crashed.
        # This ensures the feedback loop (Training→Evaluation→Curriculum) never breaks.
        if HAS_EVENT_BUS and get_router is not None and (not distributed or is_main_process()):
            try:
                _training_duration = time.time() - _training_start_time
                _config_key = f"{config.board_type.value}_{num_players}p"

                if _training_completed_normally:
                    # Training succeeded - emit TRAINING_COMPLETED (may be duplicate, but guaranteed)
                    router = get_router()
                    payload = {
                        "epochs_completed": epochs_completed,
                        "best_val_loss": float(best_val_loss),
                        "final_train_loss": float(avg_train_loss),
                        "final_val_loss": float(avg_val_loss),
                        "config": _config_key,
                        "config_key": _config_key,
                        "board_type": config.board_type.value,
                        "num_players": num_players,
                        "duration_seconds": _training_duration,
                        "hardened_emit": True,  # Flag indicating this came from finally block
                        "trigger_evaluation": True,  # Trigger automatic evaluation
                        # model_path for FeedbackLoopController (Dec 2025 integration fix)
                        "model_path": str(save_path),
                        # policy_accuracy for evaluation trigger threshold check
                        # Jan 2026: Fixed - use proper None check instead of 'in dir()'
                        "policy_accuracy": float(avg_policy_accuracy) if avg_policy_accuracy is not None else 0.0,
                        # Feb 2026: Include training data stats for generation tracking
                        "training_samples": _total_samples,
                        "training_games": _num_data_files,
                    }
                    # Include checkpoint_path if available (for auto-evaluation)
                    if _final_checkpoint_path:
                        payload["checkpoint_path"] = str(_final_checkpoint_path)
                    router.publish_sync(DataEvent(
                        event_type=DataEventType.TRAINING_COMPLETED,
                        payload=payload,
                        source="train_finally",
                    ))
                    logger.info(f"[train] Hardened TRAINING_COMPLETED emitted for {_config_key}")
                else:
                    # Training failed - emit TRAINING_FAILED
                    router = get_router()
                    error_msg = str(_training_exception) if _training_exception else "Unknown error"
                    router.publish_sync(DataEvent(
                        event_type=DataEventType.TRAINING_FAILED,
                        payload={
                            "config": _config_key,
                            "error": error_msg,
                            "epochs_completed": epochs_completed,
                            "duration_seconds": _training_duration,
                            "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
                        },
                        source="train_finally",
                    ))
                    logger.warning(f"[train] Hardened TRAINING_FAILED emitted for {_config_key}: {error_msg}")
            except (RuntimeError, ConnectionError, TimeoutError, AttributeError, NameError) as e:
                # Event emission failures, network issues, missing attributes, or undefined vars in finally block
                logger.debug(f"Failed to emit hardened training event: {e}")

        # Shutdown async checkpointer and wait for pending saves
        if async_checkpointer is not None:
            async_checkpointer.shutdown()
            logger.info("Async checkpointer shutdown complete")

        # Stop heartbeat monitor
        if heartbeat_monitor is not None:
            heartbeat_monitor.stop()
            logger.info("Heartbeat monitor stopped")

        # Teardown graceful shutdown handler (2025-12)
        if shutdown_handler is not None:
            shutdown_handler.teardown()
            logger.debug("Graceful shutdown handler teardown complete")

        # Stop integrated enhancements background services
        if enhancements_manager is not None:
            enhancements_manager.stop_background_services()
            logger.info("Integrated enhancements background services stopped")

        # Clean up distributed process group
        if distributed:
            cleanup_distributed()

        # Explicitly shutdown DataLoader workers to prevent process hang.
        # On Linux with num_workers>0, worker processes prevent the main
        # process from exiting (GH200 nodes stuck for 12+ hours).
        for loader in (train_loader, val_loader):
            if loader is not None and hasattr(loader, '_workers'):
                try:
                    loader._iterator = None
                    # Force shutdown of any active worker processes
                    if hasattr(loader, '_shutdown_workers'):
                        loader._shutdown_workers()
                except Exception:
                    pass
        # Delete references to trigger __del__ cleanup
        del train_loader, val_loader

    # ==========================================================================
    # Auto-Promotion Hook (January 2026)
    # ==========================================================================
    # If auto-promote is enabled and training completed successfully,
    # run gauntlet evaluation and promote if criteria met.
    if auto_promote and _training_completed_normally:
        logger.info("[AutoPromotion] Starting automated promotion evaluation...")
        try:
            import asyncio
            from app.training.auto_promotion import evaluate_and_promote

            async def _run_auto_promote():
                result = await evaluate_and_promote(
                    model_path=save_path,
                    board_type=config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type),
                    num_players=num_players,
                    games=auto_promote_games,
                    sync_to_cluster=auto_promote_sync,
                )
                return result

            # Run async promotion in event loop
            try:
                loop = asyncio.get_running_loop()
                promotion_result = asyncio.ensure_future(_run_auto_promote())
            except RuntimeError:
                # No running loop - create one
                promotion_result = asyncio.run(_run_auto_promote())

            if hasattr(promotion_result, 'approved') and promotion_result.approved:
                logger.info(f"[AutoPromotion] Model promoted: {promotion_result.reason}")
            elif hasattr(promotion_result, 'reason'):
                logger.info(f"[AutoPromotion] Promotion rejected: {promotion_result.reason}")

        except ImportError as e:
            logger.warning(f"[AutoPromotion] Auto-promotion module not available: {e}")
        except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[AutoPromotion] Auto-promotion failed: {e}")

    # Return structured training result for downstream analysis
    return {
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(avg_train_loss),
        'final_val_loss': float(avg_val_loss),
        'epochs_completed': epochs_completed,
        'epoch_losses': epoch_losses,
    }


def train_with_config(full_config: FullTrainingConfig) -> dict[str, Any]:
    """
    Train using a unified FullTrainingConfig object.

    This is the recommended way to call training - instead of passing 91+
    individual parameters to train_model(), use this function with a single
    FullTrainingConfig object that groups related parameters logically.

    Args:
        full_config: Complete training configuration with all sub-configs.

    Returns:
        dict: Training results with keys:
            - best_val_loss: Best validation loss achieved
            - final_train_loss: Final epoch training loss
            - final_val_loss: Final epoch validation loss
            - epochs_completed: Number of epochs completed
            - epoch_losses: List of per-epoch loss values

    Example:
        from app.training.train_config import FullTrainingConfig, EnhancementConfig

        config = FullTrainingConfig(
            board_type="hex8",
            num_players=2,
            epochs=50,
            batch_size=512,
            enhancements=EnhancementConfig(
                enable_curriculum=True,
                enable_elo_weighting=True,
            ),
        )
        config.data.data_path = "data/training/hex8_2p.npz"
        config.checkpoint.save_path = "models/hex8_2p.pth"

        result = train_with_config(config)
        print(f"Best validation loss: {result['best_val_loss']:.4f}")
    """
    # Create TrainConfig from full_config core settings
    # Note: TrainConfig uses epochs_per_iter instead of epochs
    train_cfg = TrainConfig(
        board_type=full_config.board_type,
        num_players=full_config.num_players,
        epochs_per_iter=full_config.epochs,
        batch_size=full_config.batch_size,
        learning_rate=full_config.learning_rate,
    )

    # Call train_model with unpacked parameters from FullTrainingConfig
    return train_model(
        config=train_cfg,
        data_path=full_config.data.data_path,
        save_path=full_config.checkpoint.save_path,
        # Early stopping
        early_stopping_patience=full_config.early_stopping.patience,
        elo_early_stopping_patience=full_config.early_stopping.elo_patience,
        elo_min_improvement=full_config.early_stopping.elo_min_improvement,
        # Checkpointing
        checkpoint_dir=full_config.checkpoint.checkpoint_dir,
        checkpoint_interval=full_config.checkpoint.checkpoint_interval,
        _save_all_epochs=full_config.checkpoint.save_all_epochs,
        resume_path=full_config.checkpoint.resume_path,
        init_weights_path=full_config.checkpoint.init_weights_path,
        init_weights_strict=full_config.checkpoint.init_weights_strict,
        enable_checkpoint_averaging=full_config.checkpoint.enable_checkpoint_averaging,
        num_checkpoints_to_average=full_config.checkpoint.num_checkpoints_to_average,
        # Learning rate scheduling
        warmup_epochs=full_config.lr.warmup_epochs,
        lr_scheduler=full_config.lr.lr_scheduler,
        lr_min=full_config.lr.lr_min,
        lr_t0=full_config.lr.lr_t0,
        lr_t_mult=full_config.lr.lr_t_mult,
        cyclic_lr=full_config.lr.cyclic_lr,
        cyclic_lr_period=full_config.lr.cyclic_lr_period,
        find_lr=full_config.lr.find_lr,
        lr_finder_min=full_config.lr.lr_finder_min,
        lr_finder_max=full_config.lr.lr_finder_max,
        lr_finder_iterations=full_config.lr.lr_finder_iterations,
        # Distributed training
        distributed=full_config.distributed.distributed,
        local_rank=full_config.distributed.local_rank,
        scale_lr=full_config.distributed.scale_lr,
        lr_scale_mode=full_config.distributed.lr_scale_mode,
        find_unused_parameters=full_config.distributed.find_unused_parameters,
        # Data config
        use_streaming=full_config.data.use_streaming,
        data_dir=full_config.data.data_dir,
        sampling_weights=full_config.data.sampling_weights,
        validate_data=full_config.data.validate_data,
        fail_on_invalid_data=full_config.data.fail_on_invalid_data,
        skip_freshness_check=full_config.data.skip_freshness_check,
        max_data_age_hours=full_config.data.max_data_age_hours,
        allow_stale_data=full_config.data.allow_stale_data,
        discover_synced_data=full_config.data.discover_synced_data,
        min_quality_score=full_config.data.min_quality_score,
        _include_local_data=full_config.data.include_local_data,
        _include_nfs_data=full_config.data.include_nfs_data,
        # Model architecture
        model_version=full_config.model.model_version,
        model_type=full_config.model.model_type,
        num_res_blocks=full_config.model.num_res_blocks,
        num_filters=full_config.model.num_filters,
        dropout=full_config.model.dropout,
        freeze_policy=full_config.model.freeze_policy,
        spectral_norm=full_config.model.spectral_norm,
        stochastic_depth=full_config.model.stochastic_depth,
        stochastic_depth_prob=full_config.model.stochastic_depth_prob,
        # Multi-player settings
        multi_player=full_config.multi_player,
        num_players=full_config.num_players,
        # Enhancements
        use_integrated_enhancements=full_config.enhancements.use_integrated_enhancements,
        enable_curriculum=full_config.enhancements.enable_curriculum,
        enable_augmentation=full_config.enhancements.enable_augmentation,
        enable_elo_weighting=full_config.enhancements.enable_elo_weighting,
        enable_auxiliary_tasks=full_config.enhancements.enable_auxiliary_tasks,
        enable_batch_scheduling=full_config.enhancements.enable_batch_scheduling,
        enable_background_eval=full_config.enhancements.enable_background_eval,
        use_hot_data_buffer=full_config.enhancements.use_hot_data_buffer,
        hot_buffer_size=full_config.enhancements.hot_buffer_size,
        hot_buffer_mix_ratio=full_config.enhancements.hot_buffer_mix_ratio,
        external_hot_buffer=full_config.enhancements.external_hot_buffer,
        enable_quality_weighting=full_config.enhancements.enable_quality_weighting,
        quality_weight_blend=full_config.enhancements.quality_weight_blend,
        quality_ranking_weight=full_config.enhancements.quality_ranking_weight,
        enable_outcome_weighted_policy=full_config.enhancements.enable_outcome_weighted_policy,
        outcome_weight_scale=full_config.enhancements.outcome_weight_scale,
        # Fault tolerance
        enable_circuit_breaker=full_config.fault_tolerance.enable_circuit_breaker,
        enable_anomaly_detection=full_config.fault_tolerance.enable_anomaly_detection,
        gradient_clip_mode=full_config.fault_tolerance.gradient_clip_mode,
        gradient_clip_max_norm=full_config.fault_tolerance.gradient_clip_max_norm,
        anomaly_spike_threshold=full_config.fault_tolerance.anomaly_spike_threshold,
        anomaly_gradient_threshold=full_config.fault_tolerance.anomaly_gradient_threshold,
        enable_graceful_shutdown=full_config.fault_tolerance.enable_graceful_shutdown,
        # Mixed precision
        mixed_precision=full_config.mixed_precision.enabled,
        amp_dtype=full_config.mixed_precision.amp_dtype,
        # Augmentation
        augment_hex_symmetry=full_config.augmentation.augment_hex_symmetry,
        policy_label_smoothing=full_config.augmentation.policy_label_smoothing,
        # Heartbeat
        heartbeat_file=full_config.heartbeat.heartbeat_file,
        heartbeat_interval=full_config.heartbeat.heartbeat_interval,
        # Value whitening
        value_whitening=full_config.value_whitening,
        value_whitening_momentum=full_config.value_whitening_momentum,
        # EMA
        ema=full_config.ema,
        ema_decay=full_config.ema_decay,
        # Misc
        adaptive_warmup=full_config.adaptive_warmup,
        hard_example_mining=full_config.hard_example_mining,
        hard_example_top_k=full_config.hard_example_top_k,
        auto_tune_batch_size=full_config.auto_tune_batch_size,
        track_calibration=full_config.track_calibration,
    )


def train_from_file(
    data_path: str,
    output_path: str,
    config: TrainConfig | None = None,
    initial_model_path: str | None = None,
) -> dict[str, float]:
    """
    Simplified training function for curriculum training.

    This is a convenience wrapper around train_model that:
    - Returns loss values as a dict
    - Handles initial model loading
    - Uses sensible defaults

    Parameters
    ----------
    data_path : str
        Path to training data (.npz file).
    output_path : str
        Path to save the trained model.
    config : Optional[TrainConfig]
        Training configuration. If None, uses defaults.
    initial_model_path : Optional[str]
        Path to initial model weights to start from.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'total', 'policy', 'value' containing
        the final training losses.
    """
    if config is None:
        config = TrainConfig()

    # Create checkpoint directory
    checkpoint_dir = Path(output_path).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # train_model now extracts scheduler/stopping params from config automatically
        # Only pass params that need explicit override
        result = train_model(
            config=config,
            data_path=data_path,
            save_path=output_path,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_interval=config.epochs_per_iter,
            resume_path=initial_model_path,
        )

        # Extract losses from training result
        # Note: train_model returns combined loss; we use val_loss as proxy for total
        # Policy/value breakdown would require further instrumentation
        final_loss = result.get('best_val_loss', 0.0) if result else 0.0
        return {
            "total": final_loss,
            "policy": final_loss * config.policy_weight,  # Approximate split
            "value": final_loss * (1 - config.policy_weight),
            "epochs_completed": result.get('epochs_completed', 0) if result else 0,
            "epoch_losses": result.get('epoch_losses', []) if result else [],
        }

    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as e:
        # RuntimeError: CUDA/training errors
        # ValueError: invalid config/parameters
        # OSError: file I/O failures
        # KeyError: missing config keys
        # ImportError: missing dependencies
        logger.error("Training failed: %s", e)
        return {
            "total": float('inf'),
            "policy": float('inf'),
            "value": float('inf'),
            "epochs_completed": 0,
            "epoch_losses": [],
        }


# Re-export CLI functions for backwards compatibility
# The actual implementations are in train_cli.py
from app.training.train_cli import main, parse_args

if __name__ == "__main__":
    main()
