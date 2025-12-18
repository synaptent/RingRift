"""
Training script for RingRift Neural Network AI
Includes validation split, checkpointing, early stopping, LR warmup,
and distributed training support via PyTorch DDP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import random
import os
import copy
import argparse
import glob
import math
import contextlib
import json
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import logging

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY
    HAS_PROMETHEUS = True

    # Training metrics - avoid duplicate registration
    def _safe_metric(metric_class, name, doc, **kwargs):
        """Create metric or get existing one."""
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return metric_class(name, doc, **kwargs)

    TRAINING_EPOCHS = _safe_metric(Counter, 'ringrift_training_epochs_total', 'Total training epochs completed', labelnames=['config'])
    TRAINING_LOSS = _safe_metric(Gauge, 'ringrift_training_loss', 'Current training loss', labelnames=['config', 'loss_type'])
    TRAINING_SAMPLES = _safe_metric(Counter, 'ringrift_training_samples_total', 'Total samples processed', labelnames=['config'])
    TRAINING_DURATION = _safe_metric(Histogram, 'ringrift_training_epoch_duration_seconds', 'Training epoch duration', labelnames=['config'], buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
    CALIBRATION_ECE = _safe_metric(Gauge, 'ringrift_calibration_ece', 'Expected Calibration Error', labelnames=['config'])
    CALIBRATION_MCE = _safe_metric(Gauge, 'ringrift_calibration_mce', 'Maximum Calibration Error', labelnames=['config'])
    BATCH_SIZE = _safe_metric(Gauge, 'ringrift_training_batch_size', 'Current training batch size', labelnames=['config'])
except ImportError:
    HAS_PROMETHEUS = False
    TRAINING_EPOCHS = None
    TRAINING_LOSS = None
    TRAINING_SAMPLES = None
    TRAINING_DURATION = None
    CALIBRATION_ECE = None
    CALIBRATION_MCE = None
    BATCH_SIZE = None

from app.ai.neural_net import (
    RingRiftCNN_v2,
    RingRiftCNN_v3,
    HexNeuralNet,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    HEX_BOARD_SIZE,
    HEX8_BOARD_SIZE,
    P_HEX,
    MAX_PLAYERS,
    multi_player_value_loss,
    get_policy_size_for_board,
)
from app.training.config import TrainConfig
from app.models import BoardType
from app.utils.resource_guard import check_disk_space, get_disk_usage, LIMITS
from app.training.hex_augmentation import HexSymmetryTransform
from app.training.distributed import (  # noqa: E402
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_distributed_sampler,
    wrap_model_ddp,
    seed_everything,
    scale_learning_rate,
    DistributedMetrics,
)
from app.training.data_loader import (  # noqa: E402
    StreamingDataLoader,
    WeightedStreamingDataLoader,
    get_sample_count,
    prefetch_loader,
)
from app.training.model_versioning import (  # noqa: E402
    ModelVersionManager,
    save_model_checkpoint,
    VersionMismatchError,
    LegacyCheckpointError,
)
from app.training.seed_utils import seed_all
from app.training.fault_tolerance import HeartbeatMonitor  # noqa: E402
from app.training.value_calibration import CalibrationTracker  # noqa: E402

# Data validation (2025-12)
try:
    from app.training.data_validation import (
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

# Integrated enhancements (2024-12)
try:
    from app.training.integrated_enhancements import (
        IntegratedTrainingManager,
        IntegratedEnhancementsConfig,
    )
    HAS_INTEGRATED_ENHANCEMENTS = True
except ImportError:
    IntegratedTrainingManager = None
    IntegratedEnhancementsConfig = None
    HAS_INTEGRATED_ENHANCEMENTS = False

# Circuit breaker for training fault tolerance (2025-12)
try:
    from app.distributed.circuit_breaker import get_training_breaker, CircuitState
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    get_training_breaker = None
    CircuitState = None
    HAS_CIRCUIT_BREAKER = False

# Training anomaly detection and enhancements (2025-12)
try:
    from app.training.training_enhancements import (
        TrainingAnomalyDetector,
        CheckpointAverager,
        AdaptiveGradientClipper,
    )
    HAS_TRAINING_ENHANCEMENTS = True
except ImportError:
    TrainingAnomalyDetector = None
    CheckpointAverager = None
    AdaptiveGradientClipper = None
    HAS_TRAINING_ENHANCEMENTS = False

# Auto-streaming threshold: datasets larger than this will automatically use
# StreamingDataLoader to avoid OOM. Default 5GB.
AUTO_STREAMING_THRESHOLD_BYTES = int(os.environ.get(
    "RINGRIFT_AUTO_STREAMING_THRESHOLD_GB", "5"
)) * 1024 * 1024 * 1024

from app.ai.heuristic_weights import (  # noqa: E402
    HEURISTIC_WEIGHT_KEYS,
    HEURISTIC_WEIGHT_PROFILES,
)
from app.training.tier_eval_config import (  # noqa: E402
    HEURISTIC_TIER_SPECS,
    HeuristicTierSpec,
)
from app.training.eval_pools import run_heuristic_tier_eval  # noqa: E402

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
) -> Tuple[List[str], List[float]]:
    """
    Deterministically flatten a heuristic weight profile into (keys, values).

    Keys are ordered according to HEURISTIC_WEIGHT_KEYS so that both CMA-ES
    and reconstruction remain stable across runs and consistent with other
    heuristic-training tooling.
    """
    keys: List[str] = list(HEURISTIC_WEIGHT_KEYS)
    values: List[float] = []
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
) -> Dict[str, float]:
    """Reconstruct a heuristic weight mapping from (keys, values)."""
    if len(keys) != len(values):
        raise ValueError(
            "Length mismatch reconstructing heuristic profile: "
            f"{len(keys)} keys vs {len(values)} values."
        )
    return {k: float(v) for k, v in zip(keys, values)}


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
    games_per_candidate: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
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
    games_per_candidate: Optional[int] = None,
) -> Dict[str, Any]:
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

    history: List[Dict[str, Any]] = []
    best_overall: Optional[Dict[str, Any]] = None
    best_overall_fitness = -float("inf")

    for gen in range(generations):
        candidates: List[Dict[str, Any]] = []

        for _ in range(population_size):
            # Sample from an isotropic Gaussian around the current mean.
            perturbation = np_rng.standard_normal(dim)
            arr = mean + sigma * perturbation
            # Force a plain Python list[float] for downstream type-checkers.
            tmp = cast(Sequence[float], arr.tolist())
            vector: List[float] = [float(x) for x in tmp]

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
        for w, cand in zip(weights_arr, top):
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

    report: Dict[str, Any] = {
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
from app.training.training_enhancements import EarlyStopping, EnhancedEarlyStopping


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    use_versioning: bool = True,
) -> None:
    """
    Save a training checkpoint with optional versioning metadata.

    Args:
        model: The model to save
        optimizer: The optimizer to save state from
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint to
        scheduler: Optional LR scheduler to save state from
        early_stopping: Optional early stopping tracker to save state from
        use_versioning: Whether to include versioning metadata (default True)

    Raises:
        IOError: If disk space is insufficient (>70% used or <1GB available)
    """
    # Check disk space before saving (checkpoints can be 50-200MB)
    dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
    if not check_disk_space(required_gb=1.0, path=dir_path, log_warning=False):
        disk_pct, available_gb, _ = get_disk_usage(dir_path)
        raise IOError(
            f"Insufficient disk space to save checkpoint: "
            f"{disk_pct:.1f}% used (limit: {LIMITS.DISK_MAX_PERCENT}%), "
            f"{available_gb:.1f}GB available. Path: {path}"
        )

    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)

    if use_versioning:
        # Use versioned checkpoint format
        manager = ModelVersionManager()
        training_info = {
            'epoch': epoch,
            'loss': float(loss),
        }
        if early_stopping is not None:
            training_info['early_stopping'] = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
            }

        metadata = manager.create_metadata(
            model,
            training_info=training_info,
        )

        manager.save_checkpoint(
            model,
            metadata,
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss,
        )

        # Also save early stopping best_state if needed for resume
        if (
            early_stopping is not None
            and early_stopping.best_state is not None
        ):
            # Save early stopping state separately so it survives reloading
            # Use atomic save pattern to prevent corruption
            es_path = Path(path.replace('.pth', '_early_stopping.pth'))
            es_temp_path = es_path.with_suffix('.pth.tmp')
            try:
                torch.save(
                    {
                        'best_loss': early_stopping.best_loss,
                        'counter': early_stopping.counter,
                        'best_state': early_stopping.best_state,
                    },
                    es_temp_path,
                )
                es_temp_path.rename(es_path)
            except Exception as e:
                es_temp_path.unlink(missing_ok=True)
                logger.warning(f"Failed to save early stopping state: {e}")

        logger.info(
            f"Saved versioned checkpoint to {path} "
            f"(epoch {epoch}, loss {loss:.4f}, "
            f"version {metadata.architecture_version})"
        )
    else:
        # Legacy format for backwards compatibility
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if early_stopping is not None:
            checkpoint['early_stopping'] = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'best_state': early_stopping.best_state,
            }

        # Use atomic save pattern to prevent corruption
        path_obj = Path(path)
        temp_path = path_obj.with_suffix('.pth.tmp')
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(path_obj)
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save checkpoint: {e}")

        logger.info(
            "Saved legacy checkpoint to %s (epoch %d, loss %.4f)",
            path,
            epoch,
            loss,
        )


class AsyncCheckpointer:
    """
    Background checkpoint saver for non-blocking checkpoint I/O.

    Saves checkpoints in a background thread to avoid blocking the training loop.
    Provides 5-10% speedup by overlapping checkpoint I/O with GPU computation.

    Usage:
        checkpointer = AsyncCheckpointer(max_pending=2)

        # In training loop:
        checkpointer.save_async(model, optimizer, epoch, loss, path, ...)

        # At training end:
        checkpointer.wait_for_pending()
        checkpointer.shutdown()
    """

    def __init__(self, max_pending: int = 2):
        """
        Initialize the async checkpointer.

        Args:
            max_pending: Maximum number of pending checkpoint saves.
                Older pending saves will be waited on before new ones start.
        """
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
        self._pending: deque = deque(maxlen=max_pending)
        self._max_pending = max_pending

    def save_async(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        path: str,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        use_versioning: bool = True,
    ) -> None:
        """
        Queue a checkpoint for background saving.

        Makes a deep copy of model/optimizer state to avoid mutation during save.
        """
        import copy

        # Wait for oldest pending save if at capacity
        if len(self._pending) >= self._max_pending:
            oldest_path, oldest_future = self._pending.popleft()
            try:
                oldest_future.result(timeout=120)
            except Exception as e:
                logger.error(f"Async checkpoint save failed for {oldest_path}: {e}")

        # Deep copy state dicts to prevent mutation during background save
        # Move tensors to CPU to reduce GPU memory and enable background copy
        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler else None
        early_stopping_state = None
        if early_stopping is not None:
            early_stopping_state = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'best_state': {k: v.cpu().clone() for k, v in early_stopping.best_state.items()}
                if early_stopping.best_state else None,
            }

        # Submit to background thread
        future = self._executor.submit(
            self._save_worker,
            model_state,
            optimizer_state,
            epoch,
            loss,
            path,
            scheduler_state,
            early_stopping_state,
            use_versioning,
        )
        self._pending.append((path, future))
        logger.debug(f"Queued async checkpoint save: {path}")

    def _save_worker(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        loss: float,
        path: str,
        scheduler_state: Optional[dict],
        early_stopping_state: Optional[dict],
        use_versioning: bool,
    ) -> None:
        """Background worker that performs the actual save."""
        from pathlib import Path

        dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
        os.makedirs(dir_path, exist_ok=True)

        # Build checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'loss': loss,
        }
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        if early_stopping_state is not None:
            checkpoint['early_stopping'] = early_stopping_state

        # Atomic save with temp file
        path_obj = Path(path)
        temp_path = path_obj.with_suffix('.pth.tmp')
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(path_obj)
            logger.info(f"Async checkpoint saved: {path} (epoch {epoch}, loss {loss:.4f})")
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save async checkpoint: {e}")

    def wait_for_pending(self, timeout: float = 120) -> None:
        """Wait for all pending checkpoint saves to complete."""
        for path, future in list(self._pending):
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Async checkpoint save failed for {path}: {e}")
        self._pending.clear()

    def shutdown(self) -> None:
        """Shutdown the executor and wait for pending saves."""
        self.wait_for_pending()
        self._executor.shutdown(wait=True)


class GracefulShutdownHandler:
    """
    Handles graceful shutdown on SIGTERM/SIGINT signals.

    Saves an emergency checkpoint when the process receives a shutdown signal,
    preventing loss of training progress. (2025-12)
    """

    def __init__(self):
        self._shutdown_requested = False
        self._original_handlers: Dict[int, Any] = {}
        self._checkpoint_callback: Optional[Callable[[], None]] = None

    def setup(self, checkpoint_callback: Callable[[], None]) -> None:
        """
        Setup signal handlers for graceful shutdown.

        Args:
            checkpoint_callback: Function to call to save a checkpoint when shutdown is requested
        """
        self._checkpoint_callback = checkpoint_callback

        # Install signal handlers (only on main thread)
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
            logger.info("Graceful shutdown handlers installed")
        except ValueError:
            # Signal handling can only be set in main thread
            logger.debug("Signal handlers not installed (not main thread)")

    def teardown(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # May fail if not in main thread
        self._original_handlers.clear()

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True

        # Save emergency checkpoint
        if self._checkpoint_callback:
            try:
                logger.info("Saving emergency checkpoint before shutdown...")
                self._checkpoint_callback()
                logger.info("Emergency checkpoint saved successfully")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

        # Re-raise the signal to allow normal termination after checkpoint
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: Optional[torch.device] = None,
    strict_versioning: bool = False,
) -> Tuple[int, float]:
    """
    Load a training checkpoint with optional version validation.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional LR scheduler to load state into
        early_stopping: Optional early stopping tracker to restore state into
        device: Device to map checkpoint tensors to
        strict_versioning: If True, fail on version mismatch. If False,
            log warnings but continue (default: False for backwards compat)

    Returns:
        Tuple of (epoch, loss) from the checkpoint

    Raises:
        VersionMismatchError: If strict_versioning and version mismatch
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Check if this is a versioned checkpoint
    manager = ModelVersionManager(default_device=device)
    if manager.METADATA_KEY in checkpoint:
        # Versioned checkpoint
        try:
            state_dict, metadata = manager.load_checkpoint(
                path,
                strict=strict_versioning,
                verify_checksum=True,
                device=device,
            )
            model.load_state_dict(state_dict)
            logger.info(
                f"Loaded versioned checkpoint from {path} "
                f"(version {metadata.architecture_version})"
            )

            # Extract epoch/loss from metadata or checkpoint
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))

        except (VersionMismatchError, LegacyCheckpointError) as e:
            if strict_versioning:
                raise
            logger.warning(f"Version issue loading checkpoint: {e}")
            # Fall back to direct loading
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
    else:
        # Legacy checkpoint format
        logger.info(f"Loading legacy checkpoint from {path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load early stopping state
    if early_stopping is not None:
        if 'early_stopping' in checkpoint:
            es_state = checkpoint['early_stopping']
            early_stopping.best_loss = es_state['best_loss']
            early_stopping.counter = es_state['counter']
            early_stopping.best_state = es_state.get('best_state')
        else:
            # Check for separate early stopping file
            es_path = path.replace('.pth', '_early_stopping.pth')
            if os.path.exists(es_path):
                es_state = torch.load(es_path, map_location=device)
                early_stopping.best_loss = es_state['best_loss']
                early_stopping.counter = es_state['counter']
                early_stopping.best_state = es_state.get('best_state')

    logger.info(
        f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})"
    )
    return epoch, loss


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    scheduler_type: str = 'none',
) -> Optional[Any]:
    """
    Create a learning rate scheduler with optional warmup.

    This is the legacy warmup scheduler that uses LambdaLR for simple
    scheduling. For advanced cosine annealing, use create_lr_scheduler()
    instead.

    Args:
        optimizer: The optimizer to schedule
        warmup_epochs: Number of epochs for linear warmup (0 to disable)
        total_epochs: Total number of training epochs
        scheduler_type: Type of scheduler after warmup
            ('none', 'step', 'cosine')

    Returns:
        LR scheduler or None if no scheduling requested
    """
    if warmup_epochs == 0 and scheduler_type == 'none':
        return None

    def lr_lambda(epoch: int) -> float:
        # Linear warmup phase
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        # Post-warmup phase
        if scheduler_type == 'none':
            return 1.0
        elif scheduler_type == 'step':
            # Step decay: reduce by 0.5 every 10 epochs after warmup
            steps = (epoch - warmup_epochs) // 10
            return 0.5 ** steps
        elif scheduler_type == 'cosine':
            # Cosine annealing after warmup
            remaining = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / remaining
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    lr_min: float = 1e-6,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create a learning rate scheduler with PyTorch's native implementations.

    Supports cosine annealing with optional warmup using SequentialLR to chain
    a linear warmup scheduler with the main scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler:
            - 'none': No scheduling (returns None)
            - 'step': Step decay (legacy, uses LambdaLR)
            - 'cosine': CosineAnnealingLR to lr_min over total_epochs
            - 'cosine-warm-restarts': CosineAnnealingWarmRestarts with
              T_0, T_mult
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for linear warmup (0 to disable)
        lr_min: Minimum learning rate for cosine annealing (eta_min)
        lr_t0: T_0 parameter for CosineAnnealingWarmRestarts
            (initial restart period)
        lr_t_mult: T_mult parameter for CosineAnnealingWarmRestarts
            (period multiplier)

    Returns:
        LR scheduler or None if scheduler_type is 'none' and warmup_epochs is 0
    """
    # For legacy 'step' scheduler or 'none' with warmup, use the old function
    if scheduler_type in ('none', 'step'):
        return get_warmup_scheduler(
            optimizer, warmup_epochs, total_epochs, scheduler_type
        )

    # Create the main scheduler based on type
    if scheduler_type == 'cosine':
        # Calculate T_max: epochs for cosine annealing (after warmup)
        t_max = max(1, total_epochs - warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=lr_min
        )
    elif scheduler_type == 'cosine-warm-restarts':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=lr_t0, T_mult=lr_t_mult, eta_min=lr_min
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
        return None

    # If no warmup, return the main scheduler directly
    if warmup_epochs == 0:
        return main_scheduler

    # Create warmup scheduler using LinearLR
    # LinearLR scales the learning rate from start_factor to end_factor
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(1, warmup_epochs),  # Start at lr/warmup_epochs
        end_factor=1.0,  # End at full lr
        total_iters=warmup_epochs,
    )

    # Chain warmup and main scheduler using SequentialLR
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return combined_scheduler


class RingRiftDataset(Dataset):
    """
    Dataset of self-play positions for a single board geometry.

    Each .npz file is expected to be *homogeneous* in board type/size so that
    mini-batches drawn by a vanilla DataLoader contain only one spatial shape.
    This keeps the training input compatible with the CNN and with
    NeuralNetAI.evaluate_batch, which enforces same-board-per-batch semantics.

    Future multi-board runs can either:
      - use separate datasets per board type/size, or
      - introduce a higher-level sampler/collate_fn that groups samples by
        geometry before feeding them to the network.

    Note: Terminal states (samples with empty policy arrays) are automatically
    filtered out during loading to prevent NaN losses when using KLDivLoss.
    Empty policy targets would otherwise cause the loss to become undefined.

    Args:
        data_path: Path to the .npz training data file
        board_type: Board geometry type (for augmentation)
        augment_hex: Enable D6 symmetry augmentation for hex boards
    """

    def __init__(
        self,
        data_path: str,
        board_type: BoardType = BoardType.SQUARE8,
        augment_hex: bool = False,
        use_multi_player_values: bool = False,
    ):
        self.data_path = data_path
        self.board_type = board_type
        self.augment_hex = augment_hex and board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
        # When True and the underlying dataset provides 'values_mp' and
        # 'num_players', __getitem__ will surface vector value targets
        # suitable for multi-player value heads.
        self.use_multi_player_values = use_multi_player_values
        self.hex_transform: Optional[HexSymmetryTransform] = None

        # Initialize hex transform if augmentation enabled
        if self.augment_hex:
            # Set board_size based on board_type: hex8 uses 9x9, hexagonal uses 25x25
            hex_board_size = 9 if board_type == BoardType.HEX8 else 25
            self.hex_transform = HexSymmetryTransform(board_size=hex_board_size)
            logger.info(f"Hex symmetry augmentation enabled (D6 group, board_size={hex_board_size})")

        self.length = 0
        # Memory-mapped file object (np.lib.npyio.NpzFile) or in-memory dict
        self.data = None
        # Optional metadata inferred from the underlying npz file to aid
        # future multi-board training tooling.
        self.spatial_shape = None  # (H, W) of feature maps, if known
        self.board_type_meta = None
        self.board_size_meta = None
        # List of valid sample indices (those with non-empty policies)
        self.valid_indices = None
        # Multi-player value support metadata
        self.has_multi_player_values = False
        self.num_players_arr: Optional[np.ndarray] = None
        # Effective dense policy vector length inferred from data
        self.policy_size: int = 0

        if os.path.exists(data_path):
            try:
                # Load data and cache arrays in memory. NPZ files don't support
                # true mmap (they're zip files), so each array access would
                # re-read from disk. For training efficiency, we load everything
                # into RAM upfront. For very large datasets, consider using
                # HDF5 or raw .npy files instead.
                npz_data = np.load(data_path, allow_pickle=True)
                # Convert to dict to force-load all arrays into memory
                self.data = {k: np.asarray(v) for k, v in npz_data.items()}

                if 'features' in self.data:
                    total_samples = len(self.data['values'])

                    # Filter out samples with empty policies (terminal states)
                    # These would cause NaN when computing KLDivLoss
                    policy_indices_arr = self.data['policy_indices']
                    self.valid_indices = [
                        i for i in range(total_samples)
                        if len(policy_indices_arr[i]) > 0
                    ]

                    filtered_count = total_samples - len(self.valid_indices)
                    if filtered_count > 0:
                        logger.info(
                            f"Filtered {filtered_count} terminal states "
                            f"with empty policies out of {total_samples} "
                            f"total samples"
                        )

                    self.length = len(self.valid_indices)

                    if self.length == 0:
                        logger.warning(
                            f"All {total_samples} samples in {data_path} "
                            f"have empty policies (terminal states). "
                            f"Dataset is empty."
                        )
                    else:
                        logger.info(
                            f"Loaded {self.length} valid training samples "
                            f"from {data_path} (in-memory)"
                        )

                    # Optional per-dataset metadata for multi-board training.
                    # Newer datasets may include scalar or per-sample arrays
                    # named 'board_type' and/or 'board_size'. Older datasets
                    # will simply omit these keys.
                    available_keys = set(self.data.keys())
                    if "board_type" in available_keys:
                        self.board_type_meta = self.data["board_type"]
                    if "board_size" in available_keys:
                        self.board_size_meta = self.data["board_size"]

                    # Multi-player value targets: optional 'values_mp'
                    # (N, MAX_PLAYERS) and 'num_players' (N,) arrays.
                    if "values_mp" in available_keys and "num_players" in available_keys:
                        self.has_multi_player_values = True
                        self.num_players_arr = np.asarray(
                            self.data["num_players"],
                            dtype=np.int32,
                        )

                    # Infer the canonical spatial shape (H, W) once so that
                    # callers can route samples into same-board batches if
                    # mixed-geometry datasets are ever introduced.
                    try:
                        # Use first valid sample if available
                        if self.valid_indices:
                            first_valid = self.valid_indices[0]
                            sample = self.data["features"][first_valid]
                        else:
                            sample = self.data["features"][0]
                        if sample.ndim >= 3:
                            self.spatial_shape = tuple(sample.shape[-2:])
                    except Exception:
                        # Best-effort only; training will still work as long
                        # as individual samples are well-formed.
                        self.spatial_shape = None

                    # Infer effective policy_size from sparse indices.
                    try:
                        max_index = -1
                        for i in self.valid_indices or []:
                            indices = np.asarray(
                                policy_indices_arr[i],
                                dtype=np.int64,
                            )
                            if indices.size == 0:
                                continue
                            local_max = int(indices.max())
                            if local_max > max_index:
                                max_index = local_max
                        if max_index >= 0:
                            self.policy_size = max_index + 1
                            logger.info(
                                "Inferred policy_size=%d from %s",
                                self.policy_size,
                                data_path,
                            )
                        else:
                            # Fallback to board-default if no non-empty policies
                            self.policy_size = get_policy_size_for_board(
                                self.board_type
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to infer policy_size from %s (%s); "
                            "falling back to board default.",
                            data_path,
                            e,
                        )
                        self.policy_size = get_policy_size_for_board(
                            self.board_type
                        )
                else:
                    print("Invalid data format in npz")
                    self.length = 0
            except Exception as e:
                print(f"Error loading data: {e}")
                self.length = 0
        else:
            print(f"Data file {data_path} not found, generating dummy data")
            # Generate dummy data in memory for testing
            # Ensure all dummy samples have non-empty policies
            dummy_count = 100
            # Use board-appropriate policy size instead of hardcoded value
            dummy_policy_size = get_policy_size_for_board(self.board_type)
            # Keep demo dummy data structurally compatible with the default
            # model constructors in train_model():
            # - Square boards: 14 base channels × (history_length+1=4) = 56
            # - Hex boards:    10 base channels × (history_length+1=4) = 40
            dummy_input_channels = 40 if self.board_type in (BoardType.HEXAGONAL, BoardType.HEX8) else 56
            # Model expects 20 global features (see neural_net.py global_features default)
            dummy_global_features = 20
            if self.board_type == BoardType.SQUARE19:
                dummy_h = 19
                dummy_w = 19
            elif self.board_type == BoardType.HEXAGONAL:
                dummy_h = HEX_BOARD_SIZE
                dummy_w = HEX_BOARD_SIZE
            elif self.board_type == BoardType.HEX8:
                dummy_h = HEX8_BOARD_SIZE
                dummy_w = HEX8_BOARD_SIZE
            else:
                dummy_h = 8
                dummy_w = 8
            self.data = {
                'features': np.random.rand(
                    dummy_count, dummy_input_channels, dummy_h, dummy_w
                ).astype(np.float32),
                'globals': np.random.rand(dummy_count, dummy_global_features).astype(np.float32),
                'values': np.random.choice(
                    [1.0, 0.0, -1.0],
                    size=dummy_count,
                ).astype(np.float32),
                'policy_indices': np.array([
                    np.random.choice(dummy_policy_size, 5, replace=False).astype(np.int32)
                    for _ in range(dummy_count)
                ], dtype=object),
                'policy_values': np.array([
                    np.random.rand(5).astype(np.float32)
                    for _ in range(dummy_count)
                ], dtype=object),
            }
            # Use board-appropriate policy size
            self.policy_size = dummy_policy_size
            self.valid_indices = list(range(dummy_count))
            self.length = dummy_count
            self.spatial_shape = (dummy_h, dummy_w)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.length == 0:
            raise IndexError("Dataset is empty")

        if self.data is None:
            raise RuntimeError(
                "RingRiftDataset backing store is not initialised. "
                "This usually indicates a failed load."
            )

        # Map through valid_indices to get actual data index
        # This skips terminal states with empty policies
        if self.valid_indices is not None:
            actual_idx = self.valid_indices[idx]
        else:
            actual_idx = idx

        # Access data from memory-mapped arrays. We copy to ensure we have a
        # writable tensor if needed, and to detach from the mmap backing
        # store.
        features = np.array(self.data['features'][actual_idx])
        globals_vec = np.array(self.data['globals'][actual_idx])
        value = np.array(self.data['values'][actual_idx])

        # Policy is stored as object array of arrays (sparse). mmap does not
        # support object arrays directly, so these may be fully loaded into
        # memory depending on how the npz was written. For very large datasets
        # a CSR-style encoding would be preferable, but for now we assume the
        # object array fits in memory or is handled by OS paging.
        policy_indices = self.data['policy_indices'][actual_idx]
        policy_values = self.data['policy_values'][actual_idx]

        # Apply hex symmetry augmentation on-the-fly if enabled
        # This expands effective dataset size by 12x without extra memory
        if self.augment_hex and self.hex_transform is not None:
            # Pick a random transformation from the D6 group (0-11)
            transform_id = random.randint(0, 11)

            if transform_id != 0:  # 0 is identity, skip for efficiency
                # Transform the feature tensor
                features = self.hex_transform.transform_board(
                    features, transform_id
                )

                # Transform sparse policy
                indices_arr = np.asarray(policy_indices, dtype=np.int32)
                values_arr = np.asarray(policy_values, dtype=np.float32)
                policy_indices, policy_values = (
                    self.hex_transform.transform_sparse_policy(
                        indices_arr, values_arr, transform_id
                    )
                )

        # Reconstruct dense policy vector on-the-fly
        # Since we filter for non-empty policies, this should always have data
        if self.policy_size <= 0:
            # Defensive fallback; should not normally happen
            self.policy_size = get_policy_size_for_board(self.board_type)
        policy_vector = torch.zeros(self.policy_size, dtype=torch.float32)

        if len(policy_indices) > 0:
            # Convert to proper numpy arrays with correct dtype
            # The object array may contain arrays that need explicit casting
            indices_arr = np.asarray(policy_indices, dtype=np.int64)
            values_arr = np.asarray(policy_values, dtype=np.float32)
            policy_vector[indices_arr] = torch.from_numpy(values_arr)

        # Scalar vs multi-player value targets:
        # - Scalar: shape (1,) tensor containing float
        # - Multi-player: shape (MAX_PLAYERS,) tensor from 'values_mp'
        if self.use_multi_player_values and self.has_multi_player_values:
            values_mp = np.asarray(
                self.data["values_mp"][actual_idx],
                dtype=np.float32,
            )
            value_tensor = torch.from_numpy(values_mp)
        else:
            value_tensor = torch.tensor(
                [value.item()],
                dtype=torch.float32,
            )

        return (
            torch.from_numpy(features),
            torch.from_numpy(globals_vec),
            value_tensor,
            policy_vector,
        )


class WeightedRingRiftDataset(RingRiftDataset):
    """
    Dataset with position-weighted sampling for curriculum learning.

    Extends RingRiftDataset to compute per-sample weights based on:
    - Game progress (late-game positions weighted higher)
    - Game phase (territory/line decisions weighted higher)

    The weights can be used with torch.utils.data.WeightedRandomSampler
    for biased sampling during training.

    Args:
        data_path: Path to the .npz training data file
        board_type: Board geometry type (for augmentation)
        augment_hex: Enable D6 symmetry augmentation for hex boards
        weighting: Weighting strategy - one of:
            - 'uniform': No weighting (weight = 1.0 for all)
            - 'late_game': Higher weight for late-game positions
            - 'phase_emphasis': Higher weight for decision phases
            - 'combined': Combines late_game and phase_emphasis
    """

    # Phase weights for phase_emphasis and combined strategies
    PHASE_WEIGHTS = {
        # Canonical GamePhase values (snake_case)
        'ring_placement': 0.8,
        'movement': 1.0,
        'capture': 1.2,
        'chain_capture': 1.3,
        'line_processing': 1.5,
        'territory_processing': 1.5,
        # Final cleanup phase when a player is blocked with stacks but has
        # no legal placements, movements, or captures. We weight this in the
        # same band as other decision/cleanup phases so that forced-elimination
        # samples participate normally in phase-emphasis curricula.
        'forced_elimination': 1.5,
        # Legacy / alias names (for backwards compatibility)
        'RING_PLACEMENT': 0.8,
        'MOVEMENT': 1.0,
        'CAPTURE': 1.2,
        'CHAIN_CAPTURE': 1.3,
        'LINE_DECISION': 1.5,
        'TERRITORY_DECISION': 1.5,
        'FORCED_ELIMINATION': 1.5,
        'ring_movement': 1.0,
        'line_decision': 1.5,
        'territory_decision': 1.5,
    }

    def __init__(
        self,
        data_path: str,
        board_type: BoardType = BoardType.SQUARE8,
        augment_hex: bool = False,
        weighting: str = 'late_game',
        use_multi_player_values: bool = False,
    ):
        super().__init__(
            data_path,
            board_type,
            augment_hex,
            use_multi_player_values=use_multi_player_values,
        )

        self.weighting = weighting
        self.sample_weights: Optional[np.ndarray] = None

        if self.length > 0:
            self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute per-sample weights based on weighting strategy."""
        weights = np.ones(self.length, dtype=np.float32)

        # Load metadata if available
        move_numbers = None
        total_game_moves = None
        phases = None

        if self.data is not None:
            if 'move_numbers' in self.data:
                move_numbers = self.data['move_numbers']
            if 'total_game_moves' in self.data:
                total_game_moves = self.data['total_game_moves']
            if 'phases' in self.data:
                phases = self.data['phases']

        if self.weighting == 'uniform':
            # No weighting
            pass

        elif self.weighting == 'late_game':
            # Weight positions higher toward end of game
            # w = 0.5 + 0.5 * (move_num / total_moves)
            if move_numbers is not None and total_game_moves is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weights[i] = 0.5 + 0.5 * progress
            else:
                logger.warning(
                    "late_game weighting requested but move_numbers/total_game_moves "
                    "not in dataset. Using uniform weights."
                )

        elif self.weighting == 'phase_emphasis':
            # Boost territory/line decision phases
            if phases is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    phase = str(phases[orig_idx])
                    weights[i] = self.PHASE_WEIGHTS.get(phase, 1.0)
            else:
                logger.warning(
                    "phase_emphasis weighting requested but phases not in dataset. "
                    "Using uniform weights."
                )

        elif self.weighting == 'combined':
            # Combine late_game and phase_emphasis
            late_game_available = (
                move_numbers is not None and total_game_moves is not None
            )
            phase_available = phases is not None

            for i, orig_idx in enumerate(self.valid_indices):
                weight = 1.0

                # Late game factor
                if late_game_available:
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weight *= (0.5 + 0.5 * progress)

                # Phase factor
                if phase_available:
                    phase = str(phases[orig_idx])
                    weight *= self.PHASE_WEIGHTS.get(phase, 1.0)

                weights[i] = weight

        else:
            logger.warning(
                f"Unknown weighting strategy '{self.weighting}'. Using uniform."
            )

        # Normalize weights to sum to length (maintains expected gradient scale)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights * (self.length / weight_sum)

        self.sample_weights = weights
        logger.info(
            f"Computed {self.weighting} weights: "
            f"min={weights.min():.3f}, max={weights.max():.3f}, "
            f"mean={weights.mean():.3f}"
        )

    def get_sampler(self) -> 'torch.utils.data.WeightedRandomSampler':
        """
        Get a WeightedRandomSampler using the computed weights.

        Returns
        -------
        WeightedRandomSampler
            Sampler that samples indices according to computed weights.
        """
        from torch.utils.data import WeightedRandomSampler

        if self.sample_weights is None:
            weights = torch.ones(self.length)
        else:
            weights = torch.from_numpy(self.sample_weights)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=self.length,
            replacement=True,
        )


def train_model(
    config: TrainConfig,
    data_path: Union[str, List[str]],
    save_path: str,
    early_stopping_patience: int = 10,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_interval: int = 5,
    save_all_epochs: bool = True,  # Save every epoch for Elo-based selection
    warmup_epochs: int = 0,
    lr_scheduler: str = 'none',
    lr_min: float = 1e-6,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
    resume_path: Optional[str] = None,
    augment_hex_symmetry: bool = False,
    distributed: bool = False,
    local_rank: int = -1,
    scale_lr: bool = False,
    lr_scale_mode: str = 'linear',
    find_unused_parameters: bool = False,
    use_streaming: bool = False,
    data_dir: Optional[str] = None,
    sampling_weights: str = 'uniform',
    multi_player: bool = False,
    num_players: int = 2,
    model_version: str = 'v2',
    num_res_blocks: Optional[int] = None,
    num_filters: Optional[int] = None,
    heartbeat_file: Optional[str] = None,
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
    use_integrated_enhancements: bool = False,
    enable_curriculum: bool = False,
    enable_augmentation: bool = False,
    enable_elo_weighting: bool = False,
    # Policy label smoothing (2025-12)
    policy_label_smoothing: float = 0.0,
    # Data validation (2025-12)
    validate_data: bool = True,
    fail_on_invalid_data: bool = False,
):
    """
    Train the RingRift neural network model.

    Args:
        config: Training configuration
        data_path: Path(s) to training data (.npz file or list of files)
        save_path: Path to save the best model weights
        early_stopping_patience: Number of epochs without improvement before
            stopping (0 to disable early stopping)
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
    """
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
    hot_buffer = None
    if use_hot_data_buffer and HAS_HOT_DATA_BUFFER:
        hot_buffer = HotDataBuffer(
            max_size=hot_buffer_size,
            training_threshold=config.batch_size * 5,
        )
        logger.info(f"Hot data buffer enabled (size={hot_buffer_size}, mix_ratio={hot_buffer_mix_ratio})")
    elif use_hot_data_buffer and not HAS_HOT_DATA_BUFFER:
        logger.warning("Hot data buffer requested but not available (import failed)")

    # Initialize integrated enhancements if requested
    enhancements_manager = None
    if use_integrated_enhancements and HAS_INTEGRATED_ENHANCEMENTS:
        enh_config = IntegratedEnhancementsConfig(
            curriculum_enabled=enable_curriculum,
            augmentation_enabled=enable_augmentation,
            elo_weighting_enabled=enable_elo_weighting,
        )
        enhancements_manager = IntegratedTrainingManager(
            config=enh_config,
            model=None,  # Will be set after model creation
            board_type=config.board_type.value,
        )
        logger.info(
            f"Integrated enhancements enabled: "
            f"curriculum={enable_curriculum}, augmentation={enable_augmentation}, "
            f"elo_weighting={enable_elo_weighting}"
        )
    elif use_integrated_enhancements and not HAS_INTEGRATED_ENHANCEMENTS:
        logger.warning("Integrated enhancements requested but not available (import failed)")

    # Mixed precision setup
    use_amp = mixed_precision and device.type in ('cuda', 'mps')
    scaler = None
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
        amp_torch_dtype = dtype_map.get(amp_dtype, torch.bfloat16)
        logger.info(f"Mixed precision training enabled with {amp_dtype}")

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

        inferred_size: Optional[int] = None
        policy_encoding: Optional[str] = None
        if data_path_str:
            try:
                if os.path.exists(data_path_str):
                    with np.load(
                        data_path_str,
                        mmap_mode="r",
                        allow_pickle=True,
                    ) as d:
                        if "policy_encoding" in d:
                            try:
                                policy_encoding = str(np.asarray(d["policy_encoding"]).item())
                            except Exception:
                                policy_encoding = None
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
        inferred_hex_size: Optional[int] = None
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

        if inferred_hex_size is not None:
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
    # NOTE: HexNeuralNet_v3 has hardcoded spatial policy indices that assume P_HEX encoding.
    # If the dataset uses a different policy encoding, v3 will fail with scatter index OOB.
    # For now, disable v3 and always use v2 which has a flexible FC policy head.
    use_hex_v3 = False  # Disabled: use_hex_model and model_version == 'v3'
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

        if inferred_in_channels is not None:
            hex_in_channels = inferred_in_channels
            if not distributed or is_main_process():
                logger.info(
                    "Using inferred hex in_channels=%d from dataset %s",
                    hex_in_channels,
                    data_path_str,
                )
        else:
            # Fallback to computed value
            hex_base_channels = 16 if use_hex_v3 else 10
            hex_in_channels = hex_base_channels * (config.history_length + 1)
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
            dropout=0.08,  # NAS optimal
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
            feature_shape = (14 * config.history_length, board_size, board_size)
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
            model.load_state_dict(
                torch.load(save_path, map_location=device, weights_only=True)
            )
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
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    # HexNeuralNet_v2 supports multi-player outputs, so enable multi-player loss for all boards
    use_multi_player_loss = multi_player

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

    # Early stopping
    early_stopper: Optional[EarlyStopping] = None
    if early_stopping_patience > 0:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.0001,
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
    async_checkpointer: Optional[AsyncCheckpointer] = None
    if use_async_checkpoint:
        async_checkpointer = AsyncCheckpointer(max_pending=2)
        if not distributed or is_main_process():
            logger.info("Async checkpointing enabled (non-blocking I/O)")

    # Value calibration tracker for monitoring value head quality
    calibration_tracker: Optional[CalibrationTracker] = None
    if track_calibration:
        calibration_tracker = CalibrationTracker(window_size=5000)
        if not distributed or is_main_process():
            logger.info("Value calibration tracking enabled")

    # Mixed precision scaler
    # Note: GradScaler is primarily for CUDA.
    # For MPS, mixed precision support is evolving.
    # We'll enable it only for CUDA for now to be safe.
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    train_streaming_loader: Optional[StreamingDataLoader] = None
    val_streaming_loader: Optional[StreamingDataLoader] = None
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    train_sampler = None
    val_sampler = None

    # Auto-detect large datasets and switch to streaming mode to prevent OOM
    if not use_streaming:
        # Calculate total data size
        total_data_size = 0
        paths_to_check: List[str] = []

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
    data_paths: List[str] = []
    if use_streaming:
        # Use streaming data loader for large datasets
        if data_dir is not None:
            # Collect all .npz files from directory
            npz_pattern = os.path.join(data_dir, "*.npz")
            data_paths = sorted(glob.glob(npz_pattern))
            if not distributed or is_main_process():
                logger.info(
                    f"Found {len(data_paths)} .npz files in {data_dir}"
                )
        elif isinstance(data_path, list):
            data_paths = data_path
        else:
            data_paths = [data_path]

        if not data_paths:
            if not distributed or is_main_process():
                logger.warning("No data files found for streaming; skipping.")
            if distributed:
                cleanup_distributed()
            return

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
            )
            use_weighted_sampling = False
        else:
            full_dataset = WeightedRingRiftDataset(
                data_path_str,
                board_type=config.board_type,
                augment_hex=augment_hex_symmetry,
                weighting=sampling_weights,
                use_multi_player_values=multi_player,
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
            logger.info(
                f"Early stopping enabled with patience: "
                f"{early_stopping_patience}"
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
    heartbeat_monitor: Optional[HeartbeatMonitor] = None
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
    epoch_losses: List[Dict[str, float]] = []
    epochs_completed = 0

    # Report batch size metric at start of training
    if HAS_PROMETHEUS and (not distributed or is_main_process()):
        config_label = f"{config.board_type.value}_{num_players}p"
        BATCH_SIZE.labels(config=config_label).set(config.batch_size)

    # Start integrated enhancements background services (evaluation, etc.)
    if enhancements_manager is not None:
        enhancements_manager.start_background_services()
        logger.info("Integrated enhancements background services started")

    # Initialize training circuit breaker for fault tolerance (2025-12)
    training_breaker = None
    if HAS_CIRCUIT_BREAKER and get_training_breaker:
        training_breaker = get_training_breaker()
        logger.info("Training circuit breaker enabled for fault tolerance")

    # Initialize anomaly detector for NaN/Inf detection (2025-12)
    anomaly_detector = None
    if HAS_TRAINING_ENHANCEMENTS and TrainingAnomalyDetector:
        anomaly_detector = TrainingAnomalyDetector(
            nan_threshold=0.01,  # 1% NaN tolerance
            loss_spike_threshold=10.0,  # 10x loss spike detection
            gradient_threshold=1e6,  # Gradient explosion detection
        )
        logger.info("Training anomaly detector enabled")

    # Initialize adaptive gradient clipper (2025-12)
    adaptive_clipper = None
    if HAS_TRAINING_ENHANCEMENTS and AdaptiveGradientClipper:
        adaptive_clipper = AdaptiveGradientClipper(
            initial_max_norm=1.0,
            percentile=90.0,
            history_size=100,
            min_clip=0.1,
            max_clip=10.0,
        )
        logger.info("Adaptive gradient clipping enabled")

    # Mutable training state for graceful shutdown checkpoint (2025-12)
    _training_state = {
        'epoch': start_epoch,
        'best_val_loss': float('inf'),
        'avg_val_loss': float('inf'),
    }

    # Setup graceful shutdown handler for emergency checkpoints (2025-12)
    shutdown_handler: Optional[GracefulShutdownHandler] = None
    if not distributed or is_main_process():
        def _emergency_checkpoint_callback():
            """Save emergency checkpoint on signal."""
            model_to_save = model.module if distributed else model
            emergency_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_emergency_epoch_{_training_state['epoch']}.pth",
            )
            save_checkpoint(
                model_to_save,
                optimizer,
                _training_state['epoch'],
                _training_state['avg_val_loss'],
                emergency_path,
                scheduler=epoch_scheduler,
                early_stopping=early_stopper,
            )

        shutdown_handler = GracefulShutdownHandler()
        shutdown_handler.setup(_emergency_checkpoint_callback)

    try:
        for epoch in range(start_epoch, config.epochs_per_iter):
            # Circuit breaker check - skip training if circuit is open (2025-12)
            if training_breaker and not training_breaker.can_execute("training_epoch"):
                logger.warning(f"Training circuit OPEN - skipping epoch {epoch} (recovering from failures)")
                time.sleep(10.0)  # Brief pause before retry
                continue

            # Circuit breaker: Check resources at the start of each epoch
            # This prevents training from overwhelming the system when resources are constrained
            if epoch % 5 == 0:  # Check every 5 epochs to minimize overhead
                try:
                    from app.utils.resource_guard import can_proceed, wait_for_resources, get_resource_status
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

            # Set epoch for distributed sampler or streaming loader
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if use_streaming:
                assert train_streaming_loader is not None
                assert val_streaming_loader is not None
                train_streaming_loader.set_epoch(epoch)
                val_streaming_loader.set_epoch(epoch)

            # Track epoch failure state for circuit breaker (2025-12)
            epoch_failed = False

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
                    policy_probs = policy_targets + 1e-8  # Avoid log(0)
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

                # Pad policy targets if smaller than model policy_size (e.g., dataset
                # was generated with a smaller policy space than the model supports)
                if hasattr(model, 'policy_size') and policy_targets.size(1) < model.policy_size:
                    pad_size = model.policy_size - policy_targets.size(1)
                    policy_targets = torch.nn.functional.pad(
                        policy_targets, (0, pad_size), value=0.0
                    )

                # Apply label smoothing to policy targets if configured
                # smoothed = (1 - eps) * target + eps * uniform
                if config.policy_label_smoothing > 0:
                    eps = config.policy_label_smoothing
                    policy_size = policy_targets.size(1)
                    uniform = 1.0 / policy_size
                    policy_targets = (1 - eps) * policy_targets + eps * uniform

                # Gradient accumulation: only zero grad at start of accumulation window
                accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()

                # Autocast for mixed precision (CUDA only usually).
                # For MPS, we might need to check torch.amp.autocast with
                # device_type="mps", but it is safer to stick to float32
                # on MPS if unsure.
                use_amp = device.type == 'cuda'

                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(features, globals_vec)
                    # V3 models return (values, policy_logits, rank_dist). We
                    # ignore the rank distribution for v1/v2 training losses.
                    if isinstance(out, tuple) and len(out) == 3:
                        value_pred, policy_pred, _rank_dist_pred = out
                    else:
                        value_pred, policy_pred = out

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

                    policy_loss = policy_criterion(
                        policy_log_probs,
                        policy_targets,
                    )
                    loss = value_loss + (config.policy_weight * policy_loss)

                    # Scale loss for gradient accumulation to maintain gradient magnitude
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                # Only step optimizer after accumulating gradients
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_iter):
                    # Gradient clipping (adaptive or fixed) (2025-12)
                    scaler.unscale_(optimizer)
                    if adaptive_clipper is not None:
                        grad_norm = adaptive_clipper.update_and_clip(model.parameters())
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=1.0,
                        )

                    scaler.step(optimizer)
                    scaler.update()

                    # Update integrated enhancements step counter
                    if enhancements_manager is not None:
                        enhancements_manager.update_step()

                # Anomaly detection: check for NaN/Inf in loss (2025-12)
                if anomaly_detector is not None:
                    loss_val = loss.detach().item()
                    if anomaly_detector.check_loss(loss_val):
                        anomaly_result = anomaly_detector.get_anomaly_report()
                        logger.warning(
                            f"Training anomaly detected at batch {i}: {anomaly_result}"
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

                    # For DDP, forward through the wrapped model
                    out = model(features, globals_vec)
                    if isinstance(out, tuple) and len(out) == 3:
                        value_pred, policy_pred, _rank_dist_pred = out
                    else:
                        value_pred, policy_pred = out

                    policy_log_probs = torch.log_softmax(policy_pred, dim=1)

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
                    p_loss = policy_criterion(
                        policy_log_probs, policy_targets
                    )
                    loss = v_loss + (config.policy_weight * p_loss)
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

            # Update training state for emergency checkpoints (2025-12)
            _training_state['epoch'] = epoch
            _training_state['avg_val_loss'] = avg_val_loss
            if avg_val_loss < _training_state['best_val_loss']:
                _training_state['best_val_loss'] = avg_val_loss

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
                logger.info(
                    f"Epoch [{epoch+1}/{config.epochs_per_iter}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

            # Record per-epoch losses for downstream analysis
            epochs_completed = epoch + 1
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'val_loss': float(avg_val_loss),
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

            if early_stopper is not None:
                should_stop = early_stopper(avg_val_loss, model_to_save)
                if should_stop:
                    if not distributed or is_main_process():
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1} "
                            f"(best loss: {early_stopper.best_loss:.4f})"
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
            ):
                if not distributed or is_main_process():
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
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    config: Optional[TrainConfig] = None,
    initial_model_path: Optional[str] = None,
) -> Dict[str, float]:
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
        result = train_model(
            config=config,
            data_path=data_path,
            save_path=output_path,
            early_stopping_patience=config.early_stopping_patience,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_interval=config.epochs_per_iter,
            warmup_epochs=config.warmup_epochs,
            lr_scheduler=config.lr_scheduler,
            lr_min=config.lr_min,
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


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of argument strings. If None, uses sys.argv.
              Useful for testing.
    """
    parser = argparse.ArgumentParser(
        description='Train RingRift Neural Network AI'
    )

    # Config file (overrides individual arguments)
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to TrainingPipelineConfig YAML/JSON file. Overrides individual arguments.'
    )

    # Data and model paths
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to training data (.npz file)'
    )
    parser.add_argument(
        '--save-path', type=str, default=None,
        help='Path to save best model weights'
    )

    # Training configuration
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--auto-tune-batch-size', action='store_true',
        help='Auto-tune batch size via profiling (15-30%% faster, overrides --batch-size)'
    )
    parser.add_argument(
        '--track-calibration', action='store_true',
        help='Track value head calibration metrics during training'
    )

    # 2024-12 Hot Data Buffer and Integrated Enhancements
    parser.add_argument(
        '--use-hot-data-buffer', action='store_true',
        help='Enable hot data buffer for priority experience replay'
    )
    parser.add_argument(
        '--hot-buffer-size', type=int, default=10000,
        help='Size of hot data buffer (default: 10000)'
    )
    parser.add_argument(
        '--hot-buffer-mix-ratio', type=float, default=0.3,
        help='Ratio of samples from hot buffer vs regular data (default: 0.3)'
    )
    parser.add_argument(
        '--use-integrated-enhancements', action='store_true',
        help='Enable integrated training enhancements (curriculum, augmentation, etc.)'
    )
    parser.add_argument(
        '--enable-curriculum', action='store_true',
        help='Enable curriculum learning (progressive difficulty)'
    )
    parser.add_argument(
        '--enable-augmentation', action='store_true',
        help='Enable data augmentation (symmetry transforms)'
    )
    parser.add_argument(
        '--enable-elo-weighting', action='store_true',
        help='Enable ELO-based sample weighting'
    )

    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )

    # Early stopping
    parser.add_argument(
        '--early-stopping-patience', type=int, default=10,
        help='Early stopping patience (0 to disable)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for saving checkpoints'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int, default=5,
        help='Save checkpoint every N epochs'
    )

    # Learning rate scheduling
    # Defaults match TrainConfig for better convergence
    parser.add_argument(
        '--warmup-epochs', type=int, default=1,
        help='Number of warmup epochs (default: 1 for training stability)'
    )
    parser.add_argument(
        '--lr-scheduler', type=str, default='cosine',
        choices=['none', 'step', 'cosine', 'cosine-warm-restarts'],
        help='Learning rate scheduler type (default: cosine for better convergence)'
    )
    parser.add_argument(
        '--lr-min', type=float, default=1e-6,
        help='Minimum learning rate for cosine annealing (default: 1e-6)'
    )
    parser.add_argument(
        '--lr-t0', type=int, default=10,
        help='T_0 for CosineAnnealingWarmRestarts (initial restart period)'
    )
    parser.add_argument(
        '--lr-t-mult', type=int, default=2,
        help='T_mult for CosineAnnealingWarmRestarts (period multiplier)'
    )

    # Resume training
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )

    # Sampling weights for non-streaming datasets
    parser.add_argument(
        '--sampling-weights',
        type=str,
        default='uniform',
        choices=['uniform', 'late_game', 'phase_emphasis', 'combined'],
        help=(
            "Position sampling strategy for non-streaming data: "
            "'uniform', 'late_game', 'phase_emphasis', or 'combined'."
        ),
    )

    # Board type
    parser.add_argument(
        '--board-type', type=str, default=None,
        choices=['square8', 'square19', 'hex8', 'hexagonal'],
        help='Board type for training'
    )

    # Model architecture version
    parser.add_argument(
        '--model-version', type=str, default=None,
        choices=['v2', 'v3', 'v4', 'hex'],
        help=(
            'Model architecture version: v2 (flat policy), v3 (spatial policy '
            'heads with rank distribution), v4 (NAS-optimized with attention), '
            'or hex (HexNeuralNet). '
            'Default: board-aware (square8→v3, square19→v2, hexagonal→hex).'
        ),
    )

    # Model architecture size (for scaling up models)
    parser.add_argument(
        '--num-res-blocks', type=int, default=None,
        help=(
            'Number of SE-Residual blocks in the model. Default: 12 for v3/hex, '
            '6 for v2. Increase to 20+ for stronger models (AlphaZero uses 19-20).'
        ),
    )
    parser.add_argument(
        '--num-filters', type=int, default=None,
        help=(
            'Number of filters/channels in residual blocks. Default: 192 for v3/hex, '
            '96 for v2. Increase to 256 for stronger models (AlphaZero uses 256).'
        ),
    )

    # Multi-player value head
    parser.add_argument(
        '--multi-player',
        action='store_true',
        help=(
            'Use multi-player value head (RingRiftCNN_MultiPlayer) that '
            'outputs values for all players simultaneously. Requires vector '
            'value targets in training data. Recommended for 3-4 player games.'
        ),
    )
    parser.add_argument(
        '--num-players',
        type=int,
        default=2,
        choices=[2, 3, 4],
        help='Number of players in training games (default: 2). Used for '
             'multi-player value loss masking.',
    )

    # Hex symmetry augmentation
    parser.add_argument(
        '--augment-hex-symmetry', action='store_true',
        help='Enable D6 symmetry augmentation for hex boards (12x dataset)'
    )

    # Policy label smoothing for regularization
    parser.add_argument(
        '--policy-label-smoothing', type=float, default=0.0,
        help='Policy label smoothing factor (0=disabled, typical: 0.05-0.1). '
             'Mixes target with uniform distribution for regularization.'
    )

    # Distributed training arguments
    parser.add_argument(
        '--distributed', action='store_true',
        help='Enable distributed training with PyTorch DDP'
    )
    parser.add_argument(
        '--local-rank', type=int, default=-1,
        help='Local rank for distributed training (set by torchrun)'
    )
    parser.add_argument(
        '--scale-lr', action='store_true',
        help='Scale learning rate based on world size'
    )
    parser.add_argument(
        '--lr-scale-mode', type=str, default='linear',
        choices=['linear', 'sqrt'],
        help='LR scaling mode: linear (lr * world_size) or sqrt'
    )
    parser.add_argument(
        '--find-unused-parameters',
        action='store_true',
        help=(
            'Enable find_unused_parameters in DDP (slower but handles '
            'unused params)'
        ),
    )

    # Heuristic CMA-ES optimisation (offline, eval-pool based).
    parser.add_argument(
        '--cmaes-heuristic',
        action='store_true',
        help=(
            'Run a CMA-ES-style optimisation over heuristic weights using '
            'the eval-pool based heuristic tier harness instead of neural '
            'network training.'
        ),
    )
    parser.add_argument(
        '--cmaes-tier-id',
        type=str,
        default='sq8_heuristic_baseline_v1',
        help='HeuristicTierSpec.id to use as the evaluation environment.',
    )
    parser.add_argument(
        '--cmaes-base-profile-id',
        type=str,
        default='heuristic_v1_balanced',
        help=(
            'Base heuristic profile id (key in HEURISTIC_WEIGHT_PROFILES) '
            'to optimise around.'
        ),
    )
    parser.add_argument(
        '--cmaes-generations',
        type=int,
        default=3,
        help='Number of CMA-ES generations to run.',
    )
    parser.add_argument(
        '--cmaes-population-size',
        type=int,
        default=8,
        help='Population size (candidates per generation) for CMA-ES.',
    )
    parser.add_argument(
        '--cmaes-seed',
        type=int,
        default=1,
        help='Random seed for CMA-ES optimisation.',
    )
    parser.add_argument(
        '--cmaes-games-per-candidate',
        type=int,
        default=None,
        help=(
            'Optional override for number of games per candidate. '
            'If omitted, the tier specification num_games is used.'
        ),
    )

    # Curriculum training mode
    parser.add_argument(
        '--curriculum',
        action='store_true',
        help=(
            'Enable curriculum training mode: iterative self-play with '
            'model promotion. Ignores standard training arguments and uses '
            'curriculum-specific settings instead.'
        ),
    )
    parser.add_argument(
        '--curriculum-generations',
        type=int,
        default=10,
        help='Number of curriculum generations to run (default: 10)',
    )
    parser.add_argument(
        '--curriculum-games-per-gen',
        type=int,
        default=1000,
        help='Self-play games per generation (default: 1000)',
    )
    parser.add_argument(
        '--curriculum-eval-games',
        type=int,
        default=100,
        help='Evaluation games for promotion decisions (default: 100)',
    )
    parser.add_argument(
        '--curriculum-training-epochs',
        type=int,
        default=20,
        help='Training epochs per generation (default: 20)',
    )
    parser.add_argument(
        '--curriculum-promotion-threshold',
        type=float,
        default=0.55,
        help='Win rate threshold for model promotion (default: 0.55)',
    )
    parser.add_argument(
        '--curriculum-data-retention',
        type=int,
        default=3,
        help='Number of past generations of data to retain (default: 3)',
    )
    parser.add_argument(
        '--curriculum-num-players',
        type=int,
        default=2,
        choices=[2, 3, 4],
        help='Number of players for self-play games (default: 2)',
    )
    parser.add_argument(
        '--curriculum-engine',
        type=str,
        default='descent',
        choices=['descent', 'mcts'],
        help='Engine type for self-play data generation (default: descent)',
    )
    parser.add_argument(
        '--curriculum-engine-mix',
        type=str,
        default='single',
        choices=['single', 'per_game', 'per_player'],
        help=(
            'Engine mixing strategy: single (one engine), per_game (random '
            'engine per game), per_player (random per player). Default: single'
        ),
    )
    parser.add_argument(
        '--curriculum-engine-ratio',
        type=float,
        default=0.5,
        help='MCTS ratio when engine-mix != single (0.0-1.0, default: 0.5)',
    )
    parser.add_argument(
        '--curriculum-output-dir',
        type=str,
        default='curriculum_runs',
        help='Output directory for curriculum artifacts (default: curriculum_runs)',
    )
    parser.add_argument(
        '--curriculum-base-model',
        type=str,
        default=None,
        help='Path to initial model checkpoint for curriculum training',
    )

    return parser.parse_args(args)


def main():
    """Main entry point for training."""
    args = parse_args()

    # Load config file if provided (overrides individual arguments)
    if args.config:
        from app.training.config import TrainingPipelineConfig
        try:
            pipeline_config = TrainingPipelineConfig.load(args.config)
            logger.info(f"Loaded config from {args.config}")

            # Apply config values to args (config takes precedence if args not set)
            if args.data_path is None:
                args.data_path = pipeline_config.data.data_dir
            if args.epochs is None:
                args.epochs = pipeline_config.train.epochs_per_iter
            if args.batch_size is None:
                args.batch_size = pipeline_config.train.batch_size
            if args.learning_rate is None:
                args.learning_rate = pipeline_config.train.learning_rate
            if args.checkpoint_dir is None or args.checkpoint_dir == 'checkpoints':
                args.checkpoint_dir = pipeline_config.checkpoint.checkpoint_dir
            if args.board_type is None:
                args.board_type = pipeline_config.train.board_type.value

            # Log config summary
            logger.info(f"  Board type: {pipeline_config.train.board_type.value}")
            logger.info(f"  Learning rate: {pipeline_config.train.learning_rate}")
            logger.info(f"  Batch size: {pipeline_config.train.batch_size}")

        except Exception as e:
            logger.error(f"Failed to load config from {args.config}: {e}")
            raise

    # Curriculum training mode: iterative self-play with model promotion
    if getattr(args, "curriculum", False):
        from app.training.curriculum import CurriculumConfig, CurriculumTrainer

        # Map board type string to enum
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hex8': BoardType.HEX8,
            'hexagonal': BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(
            args.board_type or 'square8',
            BoardType.SQUARE8,
        )

        config = CurriculumConfig(
            board_type=board_type,
            generations=args.curriculum_generations,
            games_per_generation=args.curriculum_games_per_gen,
            training_epochs=args.curriculum_training_epochs,
            eval_games=args.curriculum_eval_games,
            promotion_threshold=args.curriculum_promotion_threshold,
            data_retention=args.curriculum_data_retention,
            num_players=args.curriculum_num_players,
            # Learning rate from standard args if provided
            learning_rate=args.learning_rate or 1e-3,
            batch_size=args.batch_size or 32,
            base_seed=args.seed or 42,
            output_dir=args.curriculum_output_dir,
            # Engine configuration
            engine=args.curriculum_engine,
            engine_mix=args.curriculum_engine_mix,
            engine_ratio=args.curriculum_engine_ratio,
        )

        trainer = CurriculumTrainer(config, args.curriculum_base_model)
        results = trainer.run()

        # Print summary
        print("\n" + "=" * 60)
        print("CURRICULUM TRAINING COMPLETE")
        print("=" * 60)
        promoted_count = sum(1 for r in results if r.promoted)
        print(f"Total generations: {len(results)}")
        print(f"Promotions: {promoted_count}")
        print(f"Output directory: {trainer.run_dir}")
        print()
        for r in results:
            status = "PROMOTED" if r.promoted else "skipped"
            print(
                f"Gen {r.generation}: {status} (win={r.win_rate:.1%}, "
                f"loss={r.training_loss:.4f})"
            )
        return

    # Offline heuristic CMA-ES optimisation mode is explicitly opt-in and
    # does not affect the neural-network training path.
    if getattr(args, "cmaes_heuristic", False):
        report = run_cmaes_heuristic_optimization(
            tier_id=args.cmaes_tier_id,
            base_profile_id=args.cmaes_base_profile_id,
            generations=args.cmaes_generations,
            population_size=args.cmaes_population_size,
            rng_seed=args.cmaes_seed,
            games_per_candidate=args.cmaes_games_per_candidate,
        )
        out_dir = Path("results") / "ai_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"cmaes_heuristic_square8_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote CMA-ES heuristic report to {out_path}")
        return

    # Create config
    config = TrainConfig()

    # Override config from CLI args
    if args.epochs is not None:
        config.epochs_per_iter = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed
    if args.policy_label_smoothing > 0:
        config.policy_label_smoothing = args.policy_label_smoothing
    if args.board_type is not None:
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hex8': BoardType.HEX8,
            'hexagonal': BoardType.HEXAGONAL,
        }
        config.board_type = board_type_map[args.board_type]

    # Determine paths
    data_path = args.data_path or os.path.join(config.data_dir, "dataset.npz")
    save_path = args.save_path or os.path.join(
        config.model_dir,
        f"{config.model_id}.pth",
    )
    # Board-aware default model version.
    # The training CLI is frequently invoked without specifying --model-version,
    # and we want square8 runs to default to the preferred v3 architecture.
    model_version = args.model_version
    if model_version is None:
        if config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            model_version = "hex"
        elif config.board_type == BoardType.SQUARE8:
            model_version = "v3"
        else:
            model_version = "v2"
    # Run training
    train_model(
        config=config,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_t0=args.lr_t0,
        lr_t_mult=args.lr_t_mult,
        resume_path=args.resume,
        augment_hex_symmetry=args.augment_hex_symmetry,
        distributed=args.distributed,
        local_rank=args.local_rank,
        scale_lr=args.scale_lr,
        lr_scale_mode=args.lr_scale_mode,
        find_unused_parameters=args.find_unused_parameters,
        sampling_weights=args.sampling_weights,
        multi_player=args.multi_player,
        num_players=args.num_players,
        model_version=model_version,
        num_res_blocks=getattr(args, 'num_res_blocks', None),
        num_filters=getattr(args, 'num_filters', None),
        auto_tune_batch_size=getattr(args, 'auto_tune_batch_size', False),
        track_calibration=getattr(args, 'track_calibration', False),
        # 2024-12 Hot Data Buffer and Integrated Enhancements
        use_hot_data_buffer=getattr(args, 'use_hot_data_buffer', False),
        hot_buffer_size=getattr(args, 'hot_buffer_size', 10000),
        hot_buffer_mix_ratio=getattr(args, 'hot_buffer_mix_ratio', 0.3),
        use_integrated_enhancements=getattr(args, 'use_integrated_enhancements', False),
        enable_curriculum=getattr(args, 'enable_curriculum', False),
        enable_augmentation=getattr(args, 'enable_augmentation', False),
        enable_elo_weighting=getattr(args, 'enable_elo_weighting', False),
    )


if __name__ == "__main__":
    main()
