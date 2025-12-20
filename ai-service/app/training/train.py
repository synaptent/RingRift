"""Training script for RingRift Neural Network AI.

Includes validation split, checkpointing, early stopping, LR warmup,
and distributed training support via PyTorch DDP.

Recommended Usage (December 2025):
    For unified data management and training coordination, consider using:
    - TrainingDataCoordinator: app.training.data_coordinator
    - TrainingOrchestrator: app.training.orchestrated_training

    Example:
        from app.training.data_coordinator import get_data_coordinator
        from app.training.orchestrated_training import get_training_orchestrator

        # Use data coordinator for quality-aware data loading
        coordinator = get_data_coordinator()
        await coordinator.prepare_for_training(board_type="square8", num_players=2)

        # Use training orchestrator for unified lifecycle management
        orchestrator = get_training_orchestrator()
        await orchestrator.initialize()
        async with orchestrator.training_context():
            # Run training...
            pass
"""

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
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from app.utils.torch_utils import safe_load_checkpoint

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROMETHEUS = True

    # December 2025: Use centralized metric registry
    from app.metrics.registry import safe_metric as _safe_metric

    TRAINING_EPOCHS = _safe_metric(Counter, 'ringrift_training_epochs_total', 'Total training epochs completed', labelnames=['config'])
    TRAINING_LOSS = _safe_metric(Gauge, 'ringrift_training_loss', 'Current training loss', labelnames=['config', 'loss_type'])
    TRAINING_SAMPLES = _safe_metric(Counter, 'ringrift_training_samples_total', 'Total samples processed', labelnames=['config'])
    TRAINING_DURATION = _safe_metric(Histogram, 'ringrift_training_epoch_duration_seconds', 'Training epoch duration', labelnames=['config'], buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
    CALIBRATION_ECE = _safe_metric(Gauge, 'ringrift_calibration_ece', 'Expected Calibration Error', labelnames=['config'])
    CALIBRATION_MCE = _safe_metric(Gauge, 'ringrift_calibration_mce', 'Maximum Calibration Error', labelnames=['config'])
    BATCH_SIZE = _safe_metric(Gauge, 'ringrift_training_batch_size', 'Current training batch size', labelnames=['config'])

    # Fault tolerance metrics (2025-12)
    CIRCUIT_BREAKER_STATE = _safe_metric(Gauge, 'ringrift_circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open, 2=half-open)', labelnames=['config', 'operation'])
    ANOMALY_DETECTIONS = _safe_metric(Counter, 'ringrift_training_anomalies_total', 'Total training anomalies detected', labelnames=['config', 'type'])
    GRADIENT_CLIP_NORM = _safe_metric(Gauge, 'ringrift_gradient_clip_norm', 'Current gradient clipping threshold', labelnames=['config'])
    GRADIENT_NORM = _safe_metric(Gauge, 'ringrift_gradient_norm', 'Recent gradient norm', labelnames=['config'])
except ImportError:
    HAS_PROMETHEUS = False
    TRAINING_EPOCHS = None
    TRAINING_LOSS = None
    TRAINING_SAMPLES = None
    TRAINING_DURATION = None
    CALIBRATION_ECE = None
    CALIBRATION_MCE = None
    BATCH_SIZE = None
    CIRCUIT_BREAKER_STATE = None
    ANOMALY_DETECTIONS = None
    GRADIENT_CLIP_NORM = None
    GRADIENT_NORM = None

# Optional: Dashboard metrics collector for persistent storage (2025-12)
try:
    from app.monitoring.training_dashboard import MetricsCollector
    HAS_METRICS_COLLECTOR = True
except ImportError:
    HAS_METRICS_COLLECTOR = False
    MetricsCollector = None  # type: ignore

from app.ai.neural_losses import (
    build_rank_targets,
    masked_policy_kl,
)
from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    MAX_PLAYERS,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    RingRiftCNN_v2,
    RingRiftCNN_v3,
    get_policy_size_for_board,
    multi_player_value_loss,
)
from app.models import BoardType
from app.training.config import TrainConfig
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
    from app.distributed.data_events import DataEvent, DataEventType, get_event_bus
    HAS_CIRCUIT_BREAKER = True
    HAS_EVENT_BUS = True
except ImportError:
    get_training_breaker = None
    CircuitState = None
    get_event_bus = None
    DataEvent = None
    DataEventType = None
    HAS_CIRCUIT_BREAKER = False
    HAS_EVENT_BUS = False

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

# Training anomaly detection and enhancements (2025-12)
try:
    from app.training.training_enhancements import (
        AdaptiveGradientClipper,
        CheckpointAverager,
        TrainingAnomalyDetector,
    )
    HAS_TRAINING_ENHANCEMENTS = True
except ImportError:
    TrainingAnomalyDetector = None
    CheckpointAverager = None
    AdaptiveGradientClipper = None
    HAS_TRAINING_ENHANCEMENTS = False

# DataCatalog for cluster-wide training data discovery (2025-12)
try:
    from app.distributed.data_catalog import DataCatalog, get_data_catalog
    HAS_DATA_CATALOG = True
except ImportError:
    DataCatalog = None
    get_data_catalog = None
    HAS_DATA_CATALOG = False

# Auto-streaming threshold: datasets larger than this will automatically use
# StreamingDataLoader to avoid OOM. Default 5GB.
AUTO_STREAMING_THRESHOLD_BYTES = int(os.environ.get(
    "RINGRIFT_AUTO_STREAMING_THRESHOLD_GB", "5"
)) * 1024 * 1024 * 1024

from app.ai.heuristic_weights import (  # noqa: E402
    HEURISTIC_WEIGHT_KEYS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.training.eval_pools import run_heuristic_tier_eval  # noqa: E402
from app.training.tier_eval_config import (  # noqa: E402
    HEURISTIC_TIER_SPECS,
    HeuristicTierSpec,
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
    seed_all(seed)


def _flatten_heuristic_weights(
    profile: Mapping[str, float],
) -> tuple[list[str], list[float]]:
    """
    Deterministically flatten a heuristic weight profile into (keys, values).

    Keys are ordered according to HEURISTIC_WEIGHT_KEYS so that both CMA-ES
    and reconstruction remain stable across runs and consistent with other
    heuristic-training tooling.
    """
    keys: list[str] = list(HEURISTIC_WEIGHT_KEYS)
    values: list[float] = []
    for k in keys:
        try:
            values.append(float(profile[k]))
        except KeyError as exc:
            raise KeyError(
                f"Missing heuristic weight {k!r} in profile; all profiles "
                "used for optimisation must define the full "
                "HEURISTIC_WEIGHT_KEYS set."
            ) from exc
    return keys, values


def _reconstruct_heuristic_profile(
    keys: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    """Reconstruct a heuristic weight mapping from (keys, values)."""
    if len(keys) != len(values):
        raise ValueError(
            "Length mismatch reconstructing heuristic profile: "
            f"{len(keys)} keys vs {len(values)} values."
        )
    return {k: float(v) for k, v in zip(keys, values, strict=False)}


@contextlib.contextmanager
def temporary_heuristic_profile(
    profile_id: str,
    weights: Mapping[str, float],
):
    """
    Temporarily register a heuristic weight profile in the
    HEURISTIC_WEIGHT_PROFILES registry.

    This helper is intended for offline training/evaluation only (e.g.
    CMA-ES or search jobs) and must not be used on production code paths.
    """
    had_existing = profile_id in HEURISTIC_WEIGHT_PROFILES
    old_value = HEURISTIC_WEIGHT_PROFILES.get(profile_id)
    HEURISTIC_WEIGHT_PROFILES[profile_id] = dict(weights)
    try:
        yield
    finally:
        if had_existing:
            assert old_value is not None
            HEURISTIC_WEIGHT_PROFILES[profile_id] = old_value
        else:
            HEURISTIC_WEIGHT_PROFILES.pop(profile_id, None)


def _get_heuristic_tier_by_id(tier_id: str) -> HeuristicTierSpec:
    """Return the HeuristicTierSpec with the given id or raise ValueError."""
    for spec in HEURISTIC_TIER_SPECS:
        if spec.id == tier_id:
            return spec
    available = ", ".join(sorted(s.id for s in HEURISTIC_TIER_SPECS))
    raise ValueError(
        f"Unknown heuristic tier_id {tier_id!r}. "
        f"Available heuristic tiers: {available}"
    )


def evaluate_heuristic_candidate(
    tier_spec: HeuristicTierSpec,
    base_profile_id: str,
    keys: Sequence[str],
    candidate_vector: Sequence[float],
    rng_seed: int,
    games_per_candidate: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate a heuristic weight candidate via run_heuristic_tier_eval.

    Returns (fitness, raw_result_dict) where fitness is a scalar with
    higher values representing better performance.
    """
    # Rebuild mapping and register under a temporary candidate profile id.
    candidate_weights = _reconstruct_heuristic_profile(keys, candidate_vector)
    candidate_profile_id = f"cmaes_candidate_{tier_spec.id}"

    # Derive the concrete tier spec used for this evaluation. We keep all
    # structural fields but swap in the candidate/baseline profile ids so
    # that the tier harness routes AIs via the appropriate weight profiles.
    eval_tier = HeuristicTierSpec(
        id=tier_spec.id,
        name=tier_spec.name,
        board_type=tier_spec.board_type,
        num_players=tier_spec.num_players,
        eval_pool_id=tier_spec.eval_pool_id,
        num_games=tier_spec.num_games,
        candidate_profile_id=candidate_profile_id,
        baseline_profile_id=base_profile_id,
        description=tier_spec.description,
    )

    max_games = games_per_candidate or tier_spec.num_games

    with temporary_heuristic_profile(candidate_profile_id, candidate_weights):
        result = run_heuristic_tier_eval(
            tier_spec=eval_tier,
            rng_seed=rng_seed,
            max_games=max_games,
        )

    # Compute a simple scalar fitness from win/draw/loss and margins.
    games_played_raw = result.get("games_played") or 0
    games_played = max(1, int(games_played_raw))
    results = result.get("results") or {}
    wins = float(results.get("wins", 0.0))
    draws = float(results.get("draws", 0.0))
    win_rate = (wins + 0.5 * draws) / games_played

    margins = result.get("margins") or {}
    ring_margin = float(margins.get("ring_margin_mean") or 0.0)
    territory_margin = float(margins.get("territory_margin_mean") or 0.0)
    # Ring margin is primary; territory margin is down-weighted to avoid
    # over-optimising purely for space leads.
    margin_score = ring_margin + 0.25 * territory_margin

    fitness = float(win_rate + 0.01 * margin_score)
    return fitness, result


def run_cmaes_heuristic_optimization(
    tier_id: str,
    base_profile_id: str,
    generations: int = 5,
    population_size: int = 8,
    rng_seed: int = 1,
    games_per_candidate: int | None = None,
) -> dict[str, Any]:
    """
    Run a small CMA-ES-style optimisation loop over heuristic weights.

    The optimisation is offline-only and uses the heuristic eval-pool
    harness as its fitness function. It adapts the mean of a Gaussian
    search distribution over the heuristic weight vector while keeping a
    simple isotropic covariance (no full CMA matrix) for robustness.
    """
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    seed_all(rng_seed)
    py_rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed + 1)

    tier_spec = _get_heuristic_tier_by_id(tier_id)

    if base_profile_id not in HEURISTIC_WEIGHT_PROFILES:
        available = ", ".join(sorted(HEURISTIC_WEIGHT_PROFILES.keys()))
        raise ValueError(
            f"Unknown heuristic base_profile_id {base_profile_id!r}. "
            f"Available: {available}"
        )

    base_profile = HEURISTIC_WEIGHT_PROFILES[base_profile_id]
    keys, base_vector = _flatten_heuristic_weights(base_profile)

    dim = len(base_vector)
    mean = np.asarray(base_vector, dtype=float)
    # Initial step size chosen as a small fraction of the typical weight
    # magnitude so early generations explore but do not explode.
    sigma = 0.5

    history: list[dict[str, Any]] = []
    best_overall: dict[str, Any] | None = None
    best_overall_fitness = -float("inf")

    for gen in range(generations):
        candidates: list[dict[str, Any]] = []

        for _ in range(population_size):
            # Sample from an isotropic Gaussian around the current mean.
            perturbation = np_rng.standard_normal(dim)
            arr = mean + sigma * perturbation
            # Force a plain Python list[float] for downstream type-checkers.
            tmp = cast(Sequence[float], arr.tolist())
            vector: list[float] = [float(x) for x in tmp]

            eval_seed = py_rng.randint(1, 2**31 - 1)
            fitness, raw = evaluate_heuristic_candidate(
                tier_spec=tier_spec,
                base_profile_id=base_profile_id,
                keys=keys,
                candidate_vector=vector,
                rng_seed=eval_seed,
                games_per_candidate=games_per_candidate,
            )
            fitness = float(fitness)
            candidate_entry = {
                "vector": vector,
                "fitness": fitness,
                "raw": raw,
            }
            candidates.append(candidate_entry)

            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_overall = {
                    "generation": gen,
                    "vector": vector,
                    "fitness": fitness,
                    "raw": raw,
                }

        # Sort population and update mean via weighted recombination of top μ.
        candidates.sort(key=lambda c: c["fitness"], reverse=True)
        mu = max(1, population_size // 2)
        top = candidates[:mu]

        weights_arr = np.array(
            [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)],
            dtype=float,
        )
        weights_arr /= weights_arr.sum()

        new_mean = np.zeros(dim, dtype=float)
        for w, cand in zip(weights_arr, top, strict=False):
            new_mean += w * np.asarray(cand["vector"], dtype=float)
        mean = new_mean

        mean_fitness = float(
            sum(c["fitness"] for c in candidates) / len(candidates)
        )
        history.append(
            {
                "generation": gen,
                "best_fitness": float(top[0]["fitness"]),
                "mean_fitness": mean_fitness,
            }
        )

        # Simple geometric decay of sigma to encourage convergence while
        # still leaving some exploration in later generations.
        sigma *= 0.9

    report: dict[str, Any] = {
        "run_type": "heuristic_cmaes_square8",
        "tier_id": tier_id,
        "base_profile_id": base_profile_id,
        "generations": generations,
        "population_size": population_size,
        "rng_seed": rng_seed,
        "games_per_candidate": games_per_candidate,
        "dimension": dim,
        "keys": list(keys),
        "history": history,
        "best": best_overall,
    }
    return report


# EarlyStopping is now imported from training_enhancements for consolidation
# The EnhancedEarlyStopping class provides backwards compatibility via __call__ method
from app.training.training_enhancements import EarlyStopping

# Checkpointing utilities - use unified module (2025-12)
HAS_UNIFIED_CHECKPOINT = False

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




def train_model(
    config: TrainConfig,
    data_path: Union[str, list[str]],
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
    hard_example_mining: bool = False,
    hard_example_top_k: float = 0.3,
    auto_tune_batch_size: bool = True,  # Enabled by default for 15-30% better throughput
    track_calibration: bool = False,
    # 2024-12 Hot Data Buffer and Integrated Enhancements
    use_hot_data_buffer: bool = False,
    hot_buffer_size: int = 10000,
    hot_buffer_mix_ratio: float = 0.3,
    external_hot_buffer: Any | None = None,  # Pre-populated HotDataBuffer from caller
    use_integrated_enhancements: bool = False,
    enable_curriculum: bool = False,
    enable_augmentation: bool = False,
    enable_elo_weighting: bool = False,
    enable_auxiliary_tasks: bool = False,
    enable_batch_scheduling: bool = False,
    enable_background_eval: bool = False,
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
):
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
    """
    # Resolve optional parameters from config (use config as source of truth)
    # This ensures callers that don't pass these params get the config defaults
    # rather than arbitrary function defaults
    if early_stopping_patience is None:
        early_stopping_patience = getattr(config, 'early_stopping_patience', 5)
    if elo_early_stopping_patience is None:
        elo_early_stopping_patience = getattr(config, 'elo_early_stopping_patience', 10)
    if elo_min_improvement is None:
        elo_min_improvement = getattr(config, 'elo_min_improvement', 5.0)
    if warmup_epochs is None:
        warmup_epochs = getattr(config, 'warmup_epochs', 1)
    if lr_scheduler is None:
        lr_scheduler = getattr(config, 'lr_scheduler', 'cosine')
    if lr_min is None:
        lr_min = getattr(config, 'lr_min', 1e-6)

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
    # Data Validation (2025-12)
    # ==========================================================================
    # Validate training data before loading to catch corruption early
    if validate_data and HAS_DATA_VALIDATION and not use_streaming:
        data_paths_to_validate = []
        if isinstance(data_path, list):
            data_paths_to_validate = [p for p in data_path if p and os.path.exists(p)]
        elif data_path and os.path.exists(data_path):
            data_paths_to_validate = [data_path]

        if data_paths_to_validate:
            if not distributed or is_main_process():
                logger.info(f"Validating {len(data_paths_to_validate)} training data file(s)...")

            validation_failed = False
            for path in data_paths_to_validate:
                result = validate_npz_file(path)
                if not distributed or is_main_process():
                    if result.valid:
                        logger.info(f"  ✓ {path}: {result.total_samples} samples OK")
                    else:
                        logger.warning(
                            f"  ✗ {path}: {len(result.issues)} issues in "
                            f"{result.samples_with_issues}/{result.total_samples} samples"
                        )
                        # Log first few issues
                        for issue in result.issues[:5]:
                            logger.warning(f"    - {issue}")
                        if len(result.issues) > 5:
                            logger.warning(f"    ... and {len(result.issues) - 5} more issues")
                        validation_failed = True

            if validation_failed:
                if fail_on_invalid_data:
                    raise ValueError(
                        "Training data validation failed. Set fail_on_invalid_data=False "
                        "to proceed despite validation issues (not recommended)."
                    )
                else:
                    if not distributed or is_main_process():
                        logger.warning(
                            "Proceeding with training despite validation issues. "
                            "Set fail_on_invalid_data=True to enforce data quality."
                        )
    elif validate_data and not HAS_DATA_VALIDATION:
        if not distributed or is_main_process():
            logger.warning("Data validation requested but module not available")

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
        except Exception as e:
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

    # Mixed precision setup (CUDA-only for now)
    amp_enabled = bool(mixed_precision and device.type == 'cuda')
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
    amp_torch_dtype = dtype_map.get(amp_dtype, torch.bfloat16)
    use_grad_scaler = bool(amp_enabled and amp_torch_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
    if amp_enabled:
        logger.info(f"Mixed precision training enabled with {amp_dtype}")

    # Initialize dashboard metrics collector for persistent metric storage (2025-12)
    metrics_collector = None
    if HAS_METRICS_COLLECTOR and (not distributed or is_main_process()):
        try:
            metrics_collector = MetricsCollector()
            logger.info("Dashboard metrics collector initialized")
        except Exception as e:
            logger.warning(f"Could not initialize metrics collector: {e}")

    # Determine canonical spatial board_size for the CNN from config.
    if config.board_type == BoardType.SQUARE19:
        board_size = 19
    elif config.board_type == BoardType.HEXAGONAL:
        # For hex boards we use the canonical 2 * radius + 1 mapping used by
        # the feature encoder. With the default size parameter 13
        # (see create_initial_state), this yields a 25×25 grid.
        board_size = HEX_BOARD_SIZE  # 25
    elif config.board_type == BoardType.HEX8:
        # For hex8 (radius-4), the bounding box is 2*4+1 = 9.
        board_size = HEX8_BOARD_SIZE  # 9
    else:
        # Default to 8×8 (square8).
        board_size = 8

    # Determine whether to use HexNeuralNet for hexagonal boards (including hex8)
    use_hex_model = config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    config_feature_version = int(getattr(config, "feature_version", 1) or 1)

    # Validate model_id matches board_type to prevent architecture mismatch errors (P0)
    # A hex model saved with "sq8" in the name causes runtime failures when loading
    if use_hex_model and "sq8" in config.model_id.lower():
        raise ValueError(
            f"Model ID '{config.model_id}' contains 'sq8' but board_type is "
            f"{config.board_type.name} which uses HexNeuralNet architecture. "
            "Use a model ID that reflects the hex board type (e.g., 'ringrift_hex_2p')."
        )
    if not use_hex_model and ("hex" in config.model_id.lower() and "sq" not in config.model_id.lower()):
        raise ValueError(
            f"Model ID '{config.model_id}' appears to be for hex but board_type is "
            f"{config.board_type.name}. Use a model ID that matches the board type."
        )

    # Determine effective policy head size.
    policy_size: int
    if not use_hex_model and not use_streaming:
        # Non-hex, non-streaming: infer from the NPZ file if possible.
        if isinstance(data_path, list):
            data_path_str = data_path[0] if data_path else ""
        else:
            data_path_str = data_path

        inferred_size: int | None = None
        policy_encoding: str | None = None
        dataset_history_length: int | None = None
        dataset_feature_version: int | None = None
        dataset_in_channels: int | None = None
        dataset_globals_dim: int | None = None
        if data_path_str:
            try:
                if os.path.exists(data_path_str):
                    with np.load(
                        data_path_str,
                        mmap_mode="r",
                        allow_pickle=True,
                    ) as d:
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
                            except Exception:
                                policy_encoding = None
                        if "history_length" in d:
                            try:
                                dataset_history_length = int(np.asarray(d["history_length"]).item())
                            except Exception:
                                dataset_history_length = None
                        if "feature_version" in d:
                            try:
                                dataset_feature_version = int(np.asarray(d["feature_version"]).item())
                            except Exception:
                                dataset_feature_version = None
                        if "policy_indices" in d:
                            pi = d["policy_indices"]
                            max_idx = -1
                            for i in range(len(pi)):
                                arr = np.asarray(pi[i])
                                if arr.size == 0:
                                    continue
                                local_max = int(np.asarray(arr).max())
                                if local_max > max_idx:
                                    max_idx = local_max
                            if max_idx >= 0:
                                inferred_size = max_idx + 1
            except Exception as exc:
                if not distributed or is_main_process():
                    logger.warning(
                        "Failed to infer policy_size from %s: %s",
                        data_path_str,
                        exc,
                    )

        if dataset_history_length is not None and dataset_history_length != config.history_length:
            raise ValueError(
                "Training history_length does not match dataset metadata.\n"
                f"  dataset={data_path_str}\n"
                f"  dataset_history_length={dataset_history_length}\n"
                f"  config.history_length={config.history_length}\n"
                "Regenerate the dataset with matching --history-length or "
                "update the training config."
            )
        elif dataset_history_length is None and config.history_length != 3:
            if not distributed or is_main_process():
                logger.warning(
                    "Dataset %s missing history_length metadata; using "
                    "config.history_length=%d. Ensure the dataset was built "
                    "with matching history frames.",
                    data_path_str,
                    config.history_length,
                )

        if dataset_feature_version is not None and dataset_feature_version != config_feature_version:
            raise ValueError(
                "Training feature_version does not match dataset metadata.\n"
                f"  dataset={data_path_str}\n"
                f"  dataset_feature_version={dataset_feature_version}\n"
                f"  config_feature_version={config_feature_version}\n"
                "Regenerate the dataset with matching --feature-version or "
                "update the training config."
            )
        elif dataset_feature_version is None:
            if config_feature_version != 1:
                raise ValueError(
                    "Dataset is missing feature_version metadata but training "
                    "was configured for feature_version="
                    f"{config_feature_version}.\n"
                    f"  dataset={data_path_str}\n"
                    "Regenerate the dataset with --feature-version or "
                    "set feature_version=1 to use legacy features."
                )
            if not distributed or is_main_process():
                logger.warning(
                    "Dataset %s missing feature_version metadata; assuming legacy "
                    "feature_version=1.",
                    data_path_str,
                )

        expected_in_channels = 14 * (config.history_length + 1)
        if dataset_in_channels is not None and dataset_in_channels != expected_in_channels:
            raise ValueError(
                "Dataset feature channels do not match the square-board encoder.\n"
                f"  dataset={data_path_str}\n"
                f"  dataset_in_channels={dataset_in_channels}\n"
                f"  expected_in_channels={expected_in_channels}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py "
                "or app.training.generate_data using the default CNN encoder."
            )
        if dataset_globals_dim is None:
            raise ValueError(
                "Dataset is missing globals features required for training.\n"
                f"  dataset={data_path_str}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py."
            )
        if dataset_globals_dim != 20:
            raise ValueError(
                "Dataset globals feature dimension does not match the CNN encoder.\n"
                f"  dataset={data_path_str}\n"
                f"  dataset_globals_dim={dataset_globals_dim}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py "
                "to produce 20 global features."
            )

        if model_version in ('v3', 'v4'):
            if policy_encoding == "legacy_max_n":
                raise ValueError(
                    f"Dataset uses legacy MAX_N policy encoding but --model-version={model_version} "
                    "requires board-aware policy encoding.\n"
                    f"  dataset={data_path_str}\n"
                    "Regenerate the dataset with --board-aware-encoding."
                )
            if policy_encoding is None and (not distributed or is_main_process()):
                logger.warning(
                    "Dataset %s missing policy_encoding metadata; assuming board-aware "
                    "encoding for %s. If this dataset was exported with legacy MAX_N, "
                    "regenerate with --board-aware-encoding.",
                    data_path_str,
                    model_version,
                )

        if inferred_size is not None:
            board_default_size = get_policy_size_for_board(config.board_type)
            if model_version in ('v3', 'v4'):
                # V3/V4 models use spatial policy heads with fixed index mappings
                # (encode_move_for_board). Training must therefore use a dataset
                # whose policy indices were encoded with the same board-aware
                # mapping; otherwise the network learns probabilities for the
                # wrong action IDs and neural tiers degrade toward heuristic
                # rollouts.
                if policy_encoding == "legacy_max_n":
                    raise ValueError(
                        f"Dataset uses legacy MAX_N policy encoding but --model-version={model_version} "
                        "requires board-aware policy encoding.\n"
                        f"  dataset={data_path_str}\n"
                        f"  inferred_policy_size={inferred_size}\n"
                        f"  board_default_policy_size={board_default_size}\n\n"
                        "Regenerate the dataset with:\n"
                        "  PYTHONPATH=. python scripts/export_replay_dataset.py "
                        "--db <canonical_db> --board-type <square8|square19> "
                        "--num-players <2|3|4> --board-aware-encoding --output <path>.npz\n"
                    )
                if inferred_size > board_default_size:
                    raise ValueError(
                        f"Dataset policy indices exceed the {model_version} board-aware policy space. "
                        "This usually means the dataset was exported with legacy MAX_N encoding.\n"
                        f"  dataset={data_path_str}\n"
                        f"  inferred_policy_size={inferred_size}\n"
                        f"  board_default_policy_size={board_default_size}\n\n"
                        "Regenerate the dataset with --board-aware-encoding (see scripts/export_replay_dataset.py).\n"
                    )
                policy_size = board_default_size
                if not distributed or is_main_process():
                    logger.info(
                        "%s model requires board-aware policy space; using "
                        "board-default policy_size=%d (dataset max index implies %d)",
                        model_version.upper(),
                        policy_size,
                        inferred_size,
                    )
            else:
                policy_size = inferred_size
                if not distributed or is_main_process():
                    logger.info(
                        "Using inferred policy_size=%d from dataset %s",
                        policy_size,
                        data_path_str,
                    )
        else:
            policy_size = get_policy_size_for_board(config.board_type)
            if not distributed or is_main_process():
                logger.info(
                    "Using board-default policy_size=%d for board_type=%s",
                    policy_size,
                    config.board_type.name,
                )
    else:
        # Hex or streaming: try to infer from data, fall back to board defaults.
        inferred_hex_size: int | None = None
        dataset_history_length: int | None = None
        policy_encoding: str | None = None
        dataset_feature_version: int | None = None
        dataset_globals_dim: int | None = None
        if use_hex_model and not use_streaming:
            # Try to infer policy_size from hex data
            if isinstance(data_path, list):
                data_path_str = data_path[0] if data_path else ""
            else:
                data_path_str = data_path
            if data_path_str:
                try:
                    if os.path.exists(data_path_str):
                        with np.load(data_path_str, mmap_mode="r", allow_pickle=True) as d:
                            if "globals" in d:
                                glob_shape = d["globals"].shape
                                if len(glob_shape) >= 2:
                                    dataset_globals_dim = int(glob_shape[1])
                            if "policy_encoding" in d:
                                try:
                                    policy_encoding = str(np.asarray(d["policy_encoding"]).item())
                                except Exception:
                                    policy_encoding = None
                            if "history_length" in d:
                                try:
                                    dataset_history_length = int(np.asarray(d["history_length"]).item())
                                except Exception:
                                    dataset_history_length = None
                            if "feature_version" in d:
                                try:
                                    dataset_feature_version = int(np.asarray(d["feature_version"]).item())
                                except Exception:
                                    dataset_feature_version = None
                            if "policy_indices" in d:
                                pi = d["policy_indices"]
                                max_idx = -1
                                for i in range(len(pi)):
                                    arr = np.asarray(pi[i])
                                    if arr.size == 0:
                                        continue
                                    local_max = int(np.asarray(arr).max())
                                    if local_max > max_idx:
                                        max_idx = local_max
                                if max_idx >= 0:
                                    inferred_hex_size = max_idx + 1
                except Exception as exc:
                    if not distributed or is_main_process():
                        logger.warning(
                            "Failed to infer hex policy_size from %s: %s",
                            data_path_str,
                            exc,
                        )

            if dataset_history_length is not None and dataset_history_length != config.history_length:
                raise ValueError(
                    "Training history_length does not match dataset metadata.\n"
                    f"  dataset={data_path_str}\n"
                    f"  dataset_history_length={dataset_history_length}\n"
                    f"  config.history_length={config.history_length}\n"
                    "Regenerate the dataset with matching --history-length or "
                    "update the training config."
                )
            elif dataset_history_length is None and config.history_length != 3:
                if not distributed or is_main_process():
                    logger.warning(
                        "Dataset %s missing history_length metadata; using "
                        "config.history_length=%d. Ensure the dataset was built "
                        "with matching history frames.",
                        data_path_str,
                        config.history_length,
                    )

            if dataset_feature_version is not None and dataset_feature_version != config_feature_version:
                raise ValueError(
                    "Training feature_version does not match dataset metadata.\n"
                    f"  dataset={data_path_str}\n"
                    f"  dataset_feature_version={dataset_feature_version}\n"
                    f"  config_feature_version={config_feature_version}\n"
                    "Regenerate the dataset with matching --feature-version or "
                    "update the training config."
                )
            elif dataset_feature_version is None:
                if config_feature_version != 1:
                    raise ValueError(
                        "Dataset is missing feature_version metadata but training "
                        "was configured for feature_version="
                        f"{config_feature_version}.\n"
                        f"  dataset={data_path_str}\n"
                        "Regenerate the dataset with --feature-version or "
                        "set feature_version=1 to use legacy features."
                    )
                if not distributed or is_main_process():
                    logger.warning(
                        "Dataset %s missing feature_version metadata; assuming legacy "
                        "feature_version=1.",
                        data_path_str,
                    )

            if dataset_globals_dim is None:
                raise ValueError(
                    "Dataset is missing globals features required for training.\n"
                    f"  dataset={data_path_str}\n"
                    "Regenerate the dataset with scripts/export_replay_dataset.py."
                )
            if dataset_globals_dim != 20:
                raise ValueError(
                    "Dataset globals feature dimension does not match the CNN encoder.\n"
                    f"  dataset={data_path_str}\n"
                    f"  dataset_globals_dim={dataset_globals_dim}\n"
                    "Regenerate the dataset with scripts/export_replay_dataset.py "
                    "to produce 20 global features."
                )

            if model_version in ('v3', 'v4'):
                if policy_encoding == "legacy_max_n":
                    raise ValueError(
                        f"Dataset uses legacy MAX_N policy encoding but --model-version={model_version} "
                        "requires board-aware policy encoding.\n"
                        f"  dataset={data_path_str}\n"
                        "Regenerate the dataset with --board-aware-encoding."
                    )
                if policy_encoding is None and (not distributed or is_main_process()):
                    logger.warning(
                        "Dataset %s missing policy_encoding metadata; assuming board-aware "
                        "encoding for %s. If this dataset was exported with legacy MAX_N, "
                        "regenerate with --board-aware-encoding.",
                        data_path_str,
                        model_version,
                    )

        if inferred_hex_size is not None:
            if model_version in ('v3', 'v4'):
                board_default_size = get_policy_size_for_board(config.board_type)
                if inferred_hex_size > board_default_size:
                    raise ValueError(
                        f"Dataset policy indices exceed the {model_version} board-aware policy space. "
                        "This usually means the dataset was exported with legacy MAX_N encoding.\n"
                        f"  dataset={data_path_str}\n"
                        f"  inferred_policy_size={inferred_hex_size}\n"
                        f"  board_default_policy_size={board_default_size}\n\n"
                        "Regenerate the dataset with --board-aware-encoding (see scripts/export_replay_dataset.py).\n"
                    )
                policy_size = board_default_size
                if not distributed or is_main_process():
                    logger.info(
                        "%s model requires board-aware policy space; using "
                        "board-default policy_size=%d (dataset max index implies %d)",
                        model_version.upper(),
                        policy_size,
                        inferred_hex_size,
                    )
            else:
                policy_size = inferred_hex_size
                if not distributed or is_main_process():
                    logger.info(
                        "Using inferred hex policy_size=%d from dataset %s",
                        policy_size,
                        data_path_str,
                    )
        elif use_hex_model:
            # Use board-specific policy size (4500 for HEX8, 91876 for HEXAGONAL)
            policy_size = get_policy_size_for_board(config.board_type)
            if not distributed or is_main_process():
                logger.info(
                    "Using board-default hex policy_size=%d for board_type=%s",
                    policy_size,
                    config.board_type.name,
                )
        else:
            policy_size = get_policy_size_for_board(config.board_type)
            if not distributed or is_main_process():
                logger.info(
                    "Using board-default policy_size=%d for board_type=%s "
                    "(streaming path)",
                    policy_size,
                    config.board_type.name,
                )

    hex_in_channels = 0
    hex_num_players = num_players
    # HexNeuralNet_v3 uses spatial policy heads that assume board-aware (P_HEX)
    # indices. Enforce board-aware encoding for v3 via dataset metadata checks.
    use_hex_v3 = bool(use_hex_model and model_version == 'v3')
    if use_hex_model:
        # Try to infer in_channels from the dataset's feature shape
        inferred_in_channels = None
        if isinstance(data_path, list):
            data_path_str = data_path[0] if data_path else ""
        else:
            data_path_str = data_path
        if data_path_str and os.path.exists(data_path_str):
            try:
                with np.load(data_path_str, mmap_mode="r", allow_pickle=True) as d:
                    if "features" in d:
                        feat_shape = d["features"].shape
                        if len(feat_shape) >= 2:
                            inferred_in_channels = feat_shape[1]  # (N, C, H, W)
            except Exception as exc:
                if not distributed or is_main_process():
                    logger.warning(
                        "Failed to infer hex in_channels from %s: %s",
                        data_path_str,
                        exc,
                    )

        hex_base_channels = 16 if use_hex_v3 else 10
        expected_in_channels = hex_base_channels * (config.history_length + 1)

        if inferred_in_channels is not None:
            if inferred_in_channels != expected_in_channels:
                raise ValueError(
                    "Hex dataset feature channels do not match the selected model version.\n"
                    f"  dataset={data_path_str}\n"
                    f"  inferred_in_channels={inferred_in_channels}\n"
                    f"  expected_in_channels={expected_in_channels}\n"
                    f"  model_version={model_version}\n"
                    "Regenerate the dataset with the matching encoder version "
                    "or adjust --model-version."
                )
            hex_in_channels = inferred_in_channels
            if not distributed or is_main_process():
                logger.info(
                    "Using inferred hex in_channels=%d from dataset %s",
                    hex_in_channels,
                    data_path_str,
                )
        else:
            # Fallback to computed value
            hex_in_channels = expected_in_channels
        hex_num_players = MAX_PLAYERS if multi_player else num_players

    if not distributed or is_main_process():
        if use_hex_model:
            hex_model_name = "HexNeuralNet_v3" if use_hex_v3 else "HexNeuralNet_v2"
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
    # Default: 12 blocks / 192 filters for v3/hex, 6 blocks / 96 filters for v2
    if model_version == 'v3' or use_hex_model:
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

    # Initialize model based on board type and multi-player mode
    if use_hex_v3:
        # HexNeuralNet_v3 for hexagonal boards with spatial policy heads
        # V3 uses 16 base channels * (history_length + 1) frames = 64 channels
        model = HexNeuralNet_v3(
            in_channels=hex_in_channels,
            global_features=20,  # V3 encoder provides 20 global features
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=board_size,
            policy_size=policy_size,
            num_players=hex_num_players,
        )
    elif use_hex_model:
        # HexNeuralNet_v2 for hexagonal boards with multi-player support
        # V2 uses 10 base channels * (history_length + 1) frames = 40 channels
        model = HexNeuralNet_v2(
            in_channels=hex_in_channels,
            global_features=20,  # Must match _extract_features() which returns 20 globals
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=board_size,
            policy_size=policy_size,
            num_players=hex_num_players,
        )
    elif model_version == 'v3':
        # V3 architecture with spatial policy heads and rank distribution output
        v3_num_players = MAX_PLAYERS if multi_player else num_players
        model = RingRiftCNN_v3(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_players=v3_num_players,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
        )
        if not distributed or is_main_process():
            logger.info(
                f"Initializing RingRiftCNN_v3 with board_size={board_size}, "
                f"policy_size={policy_size}, num_players={v3_num_players}, "
                f"blocks={effective_blocks}, filters={effective_filters}"
            )
    elif model_version == 'v4':
        # V4 NAS-optimized architecture with multi-head attention (square boards only)
        from app.ai.neural_net import RingRiftCNN_v4
        v4_num_players = MAX_PLAYERS if multi_player else num_players
        # V4 uses NAS-discovered defaults: 13 blocks, 128 filters, 4-head attention
        v4_blocks = num_res_blocks if num_res_blocks is not None else 13
        v4_filters = num_filters if num_filters is not None else 128
        model = RingRiftCNN_v4(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_players=v4_num_players,
            num_res_blocks=v4_blocks,
            num_filters=v4_filters,
            num_attention_heads=4,  # NAS optimal
            dropout=dropout,  # Configurable, default NAS optimal 0.08
            initial_kernel_size=5,  # NAS optimal
        )
        if not distributed or is_main_process():
            logger.info(
                f"Initializing RingRiftCNN_v4 (NAS) with board_size={board_size}, "
                f"policy_size={policy_size}, num_players={v4_num_players}, "
                f"blocks={v4_blocks}, filters={v4_filters}, attention_heads=4"
            )
    elif multi_player:
        # Multi-player mode: use RingRiftCNN_v2 with multi-player value loss
        # (dedicated RingRiftCNN_MultiPlayer not yet implemented)
        model = RingRiftCNN_v2(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
        )
        if not distributed or is_main_process():
            logger.warning(
                "Multi-player value head not yet implemented for RingRiftCNN. "
                f"Using standard RingRiftCNN_v2 with multi_player_value_loss for {MAX_PLAYERS} players."
            )
    else:
        # RingRiftCNN_v2 for square boards
        model = RingRiftCNN_v2(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
        )
    with contextlib.suppress(Exception):
        model.feature_version = config_feature_version
    model.to(device)

    # Initialize enhancements manager with model reference
    if enhancements_manager is not None:
        enhancements_manager.model = model
        enhancements_manager.initialize_all()

    # Auto-tune batch size if requested (overrides config.batch_size)
    if auto_tune_batch_size and str(device).startswith('cuda'):
        try:
            from app.training.config import auto_tune_batch_size as tune_batch_fn
            original_batch = config.batch_size
            # Get feature shape from model if possible, otherwise use defaults
            feature_shape = (14 * (config.history_length + 1), board_size, board_size)
            globals_shape = (20,)  # 20 global features

            logger.info(f"Auto-tuning batch size (original: {original_batch})...")
            config.batch_size = tune_batch_fn(
                model=model,
                device=device,
                feature_shape=feature_shape,
                globals_shape=globals_shape,
                policy_size=policy_size,
                min_batch=max(32, original_batch // 4),
                max_batch=min(8192, original_batch * 8),
            )
            logger.info(f"Auto-tuned batch size: {config.batch_size} (was {original_batch})")
        except Exception as e:
            logger.warning(f"Batch size auto-tuning failed: {e}. Using original batch size.")

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
        except Exception as e:
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

    optimizer = optim.Adam(
        model.parameters(),
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

    # Early stopping (supports both loss-based and Elo-based criteria)
    early_stopper: EarlyStopping | None = None
    if early_stopping_patience > 0 or elo_early_stopping_patience > 0:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience if early_stopping_patience > 0 else 999999,
            min_delta=0.0001,
            elo_patience=elo_early_stopping_patience if elo_early_stopping_patience > 0 else None,
            elo_min_improvement=elo_min_improvement,
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
        except Exception as e:
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
        is_npz = bool(first_path and first_path.endswith(".npz"))
        try:
            if first_path and os.path.exists(first_path):
                with np.load(first_path, mmap_mode="r", allow_pickle=True) as d:
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
                        except Exception:
                            policy_encoding = None
                    if "history_length" in d:
                        try:
                            dataset_history_length = int(np.asarray(d["history_length"]).item())
                        except Exception:
                            dataset_history_length = None
                    if "feature_version" in d:
                        try:
                            dataset_feature_version = int(np.asarray(d["feature_version"]).item())
                        except Exception:
                            dataset_feature_version = None
        except Exception as exc:
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
        elif dataset_history_length is None and config.history_length != 3:
            if not distributed or is_main_process():
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
            else:
                expected_in_channels = 14 * (config.history_length + 1)
            if dataset_in_channels != expected_in_channels:
                raise ValueError(
                    "Dataset feature channels do not match the expected encoder.\n"
                    f"  dataset={first_path}\n"
                    f"  dataset_in_channels={dataset_in_channels}\n"
                    f"  expected_in_channels={expected_in_channels}\n"
                    "Regenerate the dataset with scripts/export_replay_dataset.py "
                    "or app.training.generate_data using the matching encoder."
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
        if train_streaming_loader.has_multi_player_values and not multi_player:
            if not distributed or is_main_process():
                logger.info(
                    "Dataset contains multi-player value vectors (values_mp). "
                    "Consider using --multi-player flag for multi-player training."
                )
        # If multi-player training was requested but streaming data does not
        # include vector value targets, fail fast to avoid silent shape issues.
        if multi_player and not train_streaming_loader.has_multi_player_values:
            if not distributed or is_main_process():
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

        train_sampler = None
        train_size = train_samples
        val_size = val_samples

    else:
        # Legacy single-file loading with RingRiftDataset or WeightedRingRiftDataset
        if isinstance(data_path, list):
            data_path_str = data_path[0] if data_path else ""
        else:
            data_path_str = data_path

        if sampling_weights == 'uniform':
            full_dataset = RingRiftDataset(
                data_path_str,
                board_type=config.board_type,
                augment_hex=augment_hex_symmetry,
                use_multi_player_values=multi_player,
                filter_empty_policies=filter_empty_policies,
                return_num_players=multi_player,
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
            )
            use_weighted_sampling = True

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
            if use_weighted_sampling and isinstance(train_dataset, torch.utils.data.Subset):
                base_dataset = cast(WeightedRingRiftDataset, train_dataset.dataset)
                if base_dataset.sample_weights is None:
                    train_weights = torch.ones(len(train_dataset), dtype=torch.float32)
                else:
                    subset_indices = np.array(train_dataset.indices, dtype=np.int64)
                    train_weights_np = base_dataset.sample_weights[subset_indices]
                    train_weights = torch.from_numpy(
                        train_weights_np.astype(np.float32)
                    )

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
    avg_val_loss = float('inf')  # Initialize for final checkpoint
    avg_train_loss = float('inf')  # Track for return value

    # Track per-epoch losses for downstream analysis
    epoch_losses: list[dict[str, float]] = []
    epochs_completed = 0

    # Report batch size metric at start of training
    if HAS_PROMETHEUS and (not distributed or is_main_process()):
        config_label = f"{config.board_type.value}_{num_players}p"
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

    # Aliases for backwards compatibility with existing loop code
    _last_good_checkpoint_path = training_state.last_good_checkpoint_path
    _last_good_epoch = training_state.last_good_epoch
    _circuit_breaker_rollbacks = training_state.circuit_breaker_rollbacks
    _max_circuit_breaker_rollbacks = training_state.max_circuit_breaker_rollbacks

    # Publish training started event (2025-12)
    if HAS_EVENT_BUS and get_event_bus is not None and (not distributed or is_main_process()):
        try:
            event_bus = get_event_bus()
            event_bus.publish_sync(DataEvent(
                event_type=DataEventType.TRAINING_STARTED,
                payload={
                    "total_epochs": config.epochs_per_iter,
                    "start_epoch": start_epoch,
                    "config": f"{config.board_type.value}_{num_players}p",
                    "model_path": str(config.model_dir / f"model_{num_players}p.pth"),
                },
                source="train",
            ))
        except Exception as e:
            logger.debug(f"Failed to publish training started event: {e}")

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
                    except Exception as e:
                        logger.error(f"Rollback failed: {e}")

                time.sleep(10.0)  # Brief pause before retry
                continue

            # Update circuit breaker state metric (0=closed, training can proceed)
            if HAS_PROMETHEUS and CIRCUIT_BREAKER_STATE and training_breaker and (not distributed or is_main_process()):
                CIRCUIT_BREAKER_STATE.labels(config=config_label, operation='training_epoch').set(0)

            # Circuit breaker: Check resources at the start of each epoch
            # This prevents training from overwhelming the system when resources are constrained
            if epoch % 5 == 0:  # Check every 5 epochs to minimize overhead
                try:
                    from app.utils.resource_guard import can_proceed, get_resource_status, wait_for_resources
                    if not can_proceed(check_disk=True, check_mem=True, check_cpu_load=True):
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
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 5:
                        (
                            features,
                            globals_vec,
                            value_targets,
                            policy_targets,
                            batch_num_players,
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
                    except Exception as e:
                        # Don't fail training on hot buffer errors
                        if i % 100 == 0:
                            logger.debug(f"Hot buffer mixing skipped: {e}")

                # Data augmentation: apply random symmetry transforms (2025-12)
                if enhancements_manager is not None and enhancements_manager._augmentor is not None:
                    try:
                        features, policy_targets = enhancements_manager.augment_batch_dense(
                            features, policy_targets
                        )
                    except Exception as e:
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

                # Apply label smoothing to policy targets if configured
                # smoothed = (1 - eps) * target + eps * uniform
                if config.policy_label_smoothing > 0 and torch.any(policy_valid_mask):
                    eps = config.policy_label_smoothing
                    policy_size = policy_targets.size(1)
                    uniform = 1.0 / policy_size
                    policy_targets = policy_targets.clone()
                    policy_targets[policy_valid_mask] = (
                        (1 - eps) * policy_targets[policy_valid_mask]
                        + eps * uniform
                    )

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

                    # Forward pass with optional backbone feature extraction
                    if use_aux_tasks:
                        out = model(features, globals_vec, return_features=True)
                        # V3+ models with features return (values, policy, rank_dist, features)
                        if isinstance(out, tuple) and len(out) == 4:
                            value_pred, policy_pred, rank_dist_pred, backbone_features = out
                        else:
                            # Fallback: model doesn't support return_features
                            value_pred, policy_pred = out[:2]
                            rank_dist_pred = None
                            backbone_features = None
                            use_aux_tasks = False
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

                    # Apply log_softmax to policy prediction for KLDivLoss
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
                    loss = value_loss + (config.policy_weight * policy_loss)

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
                            loss = loss + (config.rank_dist_weight * rank_loss)

                    # Auxiliary task loss (outcome prediction from value targets)
                    if use_aux_tasks and backbone_features is not None:
                        # Derive outcome class from value targets:
                        # value > 0.3 → Win (2), value < -0.3 → Loss (0), else Draw (1)
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
                        loss = loss + aux_loss

                    # Scale loss for gradient accumulation to maintain gradient magnitude
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps

                # Circuit breaker protection for backward pass (2025-12)
                # Catches CUDA errors, OOM, and other runtime exceptions
                try:
                    if use_grad_scaler:
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

                # Anomaly detection: check for NaN/Inf in loss (2025-12)
                if anomaly_detector is not None:
                    loss_val = loss.detach().item()
                    anomaly_step += 1
                    if anomaly_detector.check_loss(loss_val, anomaly_step):
                        anomaly_summary = anomaly_detector.get_summary()
                        consecutive = anomaly_summary.get('consecutive_anomalies', 0)
                        logger.warning(
                            f"Training anomaly detected at batch {i}: "
                            f"total={anomaly_summary.get('total_anomalies', 0)}, "
                            f"consecutive={consecutive}"
                        )
                        # Update Prometheus anomaly counter
                        if HAS_PROMETHEUS and ANOMALY_DETECTIONS and (not distributed or is_main_process()):
                            anomaly_type = 'nan' if loss_val != loss_val else 'spike'  # NaN != NaN
                            ANOMALY_DETECTIONS.labels(config=config_label, type=anomaly_type).inc()

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
                        if isinstance(val_batch, (list, tuple)) and len(val_batch) == 5:
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
                            ) = val_batch

                    # Transfer to device if not already there (prefetch may have done this)
                    if features.device != device:
                        features = features.to(device, non_blocking=True)
                        globals_vec = globals_vec.to(device, non_blocking=True)
                        value_targets = value_targets.to(device, non_blocking=True)
                        policy_targets = policy_targets.to(device, non_blocking=True)
                    if val_batch_num_players is not None and val_batch_num_players.device != device:
                        val_batch_num_players = val_batch_num_players.to(device, non_blocking=True)

                    # Pad policy targets if smaller than model policy_size
                    if hasattr(model, 'policy_size') and policy_targets.size(1) < model.policy_size:
                        pad_size = model.policy_size - policy_targets.size(1)
                        policy_targets = torch.nn.functional.pad(
                            policy_targets, (0, pad_size), value=0.0
                        )

                    # Autocast for mixed precision validation (matches training)
                    with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_torch_dtype):
                        # For DDP, forward through the wrapped model
                        out = model(features, globals_vec)
                        if isinstance(out, tuple) and len(out) == 3:
                            value_pred, policy_pred, rank_dist_pred = out
                        else:
                            value_pred, policy_pred = out
                            rank_dist_pred = None

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
                        loss = v_loss + (config.policy_weight * p_loss)

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
                if HAS_EVENT_BUS and get_event_bus is not None:
                    try:
                        event_bus = get_event_bus()
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
                        event_bus.publish_sync(DataEvent(
                            event_type=DataEventType.TRAINING_PROGRESS,
                            payload=event_payload,
                            source="train",
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to publish training progress event: {e}")

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
            if HAS_REGRESSION_DETECTOR and get_regression_detector is not None and epoch >= 2:
                if not distributed or is_main_process():
                    try:
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
                            # Record in epoch record
                    except Exception as e:
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
                except Exception as e:
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
                if should_stop:
                    if not distributed or is_main_process():
                        elo_info = f", best Elo: {early_stopper.best_elo:.1f}" if early_stopper.best_elo > float('-inf') else ""
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1} "
                            f"(best loss: {early_stopper.best_loss:.4f}{elo_info})"
                        )
                        # Restore best weights
                        early_stopper.restore_best_weights(model_to_save)
                        # Save final checkpoint with best weights
                        final_checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f"checkpoint_early_stop_epoch_{epoch+1}.pth",
                        )
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
                        # Save best model with versioning
                        save_model_checkpoint(
                            model_to_save,
                            save_path,
                            training_info={
                                'epoch': epoch,
                                'loss': float(early_stopper.best_loss),
                                'early_stopped': True,
                            },
                        )
                        logger.info("Best model saved to %s", save_path)
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
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not distributed or is_main_process():
                    # Save with versioning metadata
                    save_model_checkpoint(
                        model_to_save,
                        save_path,
                        training_info={
                            'epoch': epoch + 1,
                            'samples_seen': train_size * (epoch + 1),
                            'val_loss': float(avg_val_loss),
                            'train_loss': float(avg_train_loss),
                        },
                    )
                    logger.info(
                        "  New best model saved (Val Loss: %.4f)",
                        avg_val_loss,
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
                    )
                    logger.info(
                        "  Versioned checkpoint saved: %s",
                        version_path,
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

                # Publish training completed event (2025-12)
                if HAS_EVENT_BUS and get_event_bus is not None:
                    try:
                        event_bus = get_event_bus()
                        event_bus.publish_sync(DataEvent(
                            event_type=DataEventType.TRAINING_COMPLETED,
                            payload={
                                "epochs_completed": epochs_completed,
                                "best_val_loss": float(best_val_loss),
                                "final_train_loss": float(avg_train_loss),
                                "final_val_loss": float(avg_val_loss),
                                "config": f"{config.board_type.value}_{num_players}p",
                                "checkpoint_path": str(final_checkpoint_path),
                            },
                            source="train",
                        ))
                    except Exception as e:
                        logger.debug(f"Failed to publish training completed event: {e}")
    finally:
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

    # Return structured training result for downstream analysis
    return {
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(avg_train_loss),
        'final_val_loss': float(avg_val_loss),
        'epochs_completed': epochs_completed,
        'epoch_losses': epoch_losses,
    }


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

    except Exception as e:
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
from app.training.train_cli import main, parse_args  # noqa: E402, F401

if __name__ == "__main__":
    main()
