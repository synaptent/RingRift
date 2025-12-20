"""Unified Training Orchestrator for RingRift AI.

Central orchestration module that integrates ALL training components:
- Distributed training (multi-GPU/multi-node)
- Hot data buffer (priority experience replay)
- Integrated enhancements (auxiliary tasks, gradient surgery, etc.)
- Checkpoint management (fault tolerance)
- Background evaluation (continuous Elo tracking)
- Adaptive controllers (learning rate, batch size)

This provides a single entry point for advanced training with all features
properly integrated and coordinated.

Related Modules:
    - orchestrated_training.py: High-level manager coordination (rollback,
      promotion, curriculum). Use when you need service orchestration.
    - This module (unified_orchestrator.py): Low-level training execution
      (GPU training, data loading, distributed). Use for actual training runs.

Usage:
    from app.training.unified_orchestrator import (
        UnifiedTrainingOrchestrator,
        OrchestratorConfig,
    )

    config = OrchestratorConfig(
        board_type="square8",
        num_players=2,
        enable_distributed=True,
        enable_hot_buffer=True,
        enable_background_eval=True,
    )

    orchestrator = UnifiedTrainingOrchestrator(model, config)

    with orchestrator:
        for epoch in range(epochs):
            for batch in orchestrator.get_dataloader():
                loss = orchestrator.train_step(batch)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Use shared lazy torch import to prevent OOM in orchestrator processes
from app.training.utils import get_torch

# Training metrics publishing (December 2025 consolidation)
try:
    from app.training.metrics_integration import TrainingMetrics
    _HAS_TRAINING_METRICS = True
except ImportError:
    _HAS_TRAINING_METRICS = False
    TrainingMetrics = None

# Training event publishing (December 2025 consolidation)
try:
    from app.training.event_integration import (
        publish_checkpoint_saved_sync,
        publish_step_completed_sync,
    )
    _HAS_TRAINING_EVENTS = True
except ImportError:
    _HAS_TRAINING_EVENTS = False
    publish_step_completed_sync = None
    publish_checkpoint_saved_sync = None

# Distributed locking for cross-node coordination (December 2025)
try:
    from app.training.locking_integration import TrainingLocks
    _HAS_DISTRIBUTED_LOCKS = True
except ImportError:
    _HAS_DISTRIBUTED_LOCKS = False
    TrainingLocks = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Unified configuration for training orchestration.

    Combines all training subsystem configurations into a single config.
    """
    # Basic training settings
    board_type: str = "square8"
    num_players: int = 2
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    auto_tune_batch_size: bool = True  # Auto-tune batch size via profiling

    # Data settings
    data_path: str | None = None
    train_val_split: float = 0.1

    # =========================================================================
    # Feature Flags
    # =========================================================================

    # Distributed training
    enable_distributed: bool = False
    world_size: int = 1
    compress_gradients: bool = False
    use_amp: bool = True

    # Hot data buffer (priority experience replay)
    enable_hot_buffer: bool = True
    hot_buffer_size: int = 10000
    hot_buffer_priority_alpha: float = 0.6

    # Integrated enhancements
    enable_enhancements: bool = True
    enable_auxiliary_tasks: bool = False
    enable_gradient_surgery: bool = False
    enable_batch_scheduling: bool = False
    enable_elo_weighting: bool = True
    enable_curriculum: bool = True
    enable_augmentation: bool = True

    # Background evaluation (enabled by default for continuous Elo tracking)
    enable_background_eval: bool = True
    eval_interval_steps: int = 1000
    eval_games_per_check: int = 20

    # Checkpoint settings
    checkpoint_dir: str = "data/checkpoints"
    checkpoint_interval: int = 1000
    keep_top_k_checkpoints: int = 3
    auto_resume: bool = True

    # Adaptive settings
    enable_adaptive_lr: bool = True
    enable_adaptive_batch: bool = False

    # Logging
    log_interval: int = 100
    verbose: bool = False


# =============================================================================
# Component Wrappers
# =============================================================================

class HotBufferWrapper:
    """Wrapper for hot data buffer integration."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._buffer = None

    def initialize(self):
        """Initialize hot data buffer if enabled."""
        if not self.config.enable_hot_buffer:
            return

        try:
            from app.training.hot_data_buffer import HotDataBuffer
            self._buffer = HotDataBuffer(
                max_size=self.config.hot_buffer_size,
                training_threshold=self.config.batch_size * 10,
            )
            logger.info(f"[Orchestrator] HotDataBuffer initialized (size={self.config.hot_buffer_size})")
        except ImportError:
            logger.warning("[Orchestrator] HotDataBuffer not available")

    def add_game(self, game_record: Any):
        """Add a game to the buffer."""
        if self._buffer is not None:
            self._buffer.add_game(game_record)

    def get_batch(self, batch_size: int) -> tuple | None:
        """Get a training batch from the buffer."""
        if self._buffer is not None:
            return self._buffer.get_training_batch(batch_size)
        return None

    @property
    def available(self) -> bool:
        return self._buffer is not None


class EnhancementsWrapper:
    """Wrapper for integrated enhancements."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._manager = None

    def initialize(self, model: Any = None):
        """Initialize enhancements manager if enabled."""
        if not self.config.enable_enhancements:
            return

        try:
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

            enh_config = IntegratedEnhancementsConfig(
                auxiliary_tasks_enabled=self.config.enable_auxiliary_tasks,
                gradient_surgery_enabled=self.config.enable_gradient_surgery,
                batch_scheduling_enabled=self.config.enable_batch_scheduling,
                elo_weighting_enabled=self.config.enable_elo_weighting,
                curriculum_learning_enabled=self.config.enable_curriculum,
                augmentation_enabled=self.config.enable_augmentation,
            )

            self._manager = IntegratedTrainingManager(
                config=enh_config,
                model=model,
                board_type=self.config.board_type,
            )
            self._manager.initialize_all()
            logger.info("[Orchestrator] IntegratedEnhancements initialized")
        except ImportError as e:
            logger.warning(f"[Orchestrator] IntegratedEnhancements not available: {e}")

    def get_batch_size(self) -> int:
        """Get current batch size (may be dynamic)."""
        if self._manager is not None:
            return self._manager.get_batch_size()
        return self.config.batch_size

    def compute_sample_weights(self, opponent_elos: np.ndarray) -> np.ndarray:
        """Compute sample weights based on opponent Elo."""
        if self._manager is not None:
            return self._manager.compute_sample_weights(opponent_elos)
        return np.ones(len(opponent_elos))

    def augment_batch(self, features, policy_indices, policy_values):
        """Apply data augmentation to batch."""
        if self._manager is not None:
            return self._manager.augment_batch(features, policy_indices, policy_values)
        return features, policy_indices, policy_values

    def update_step(self, game_won: bool | None = None):
        """Update step counter and related components."""
        if self._manager is not None:
            self._manager.update_step(game_won)

    def get_curriculum_params(self) -> dict[str, Any]:
        """Get current curriculum stage parameters."""
        if self._manager is not None:
            return self._manager.get_curriculum_parameters()
        return {}

    @property
    def available(self) -> bool:
        return self._manager is not None


class DistributedWrapper:
    """Wrapper for distributed training."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._trainer = None

    def initialize(self, model: Any, optimizer: Any = None):
        """Initialize distributed trainer if enabled."""
        if not self.config.enable_distributed or self.config.world_size <= 1:
            return

        try:
            from app.training.distributed_unified import (
                UnifiedDistributedConfig,
                UnifiedDistributedTrainer,
            )

            dist_config = UnifiedDistributedConfig(
                world_size=self.config.world_size,
                compress_gradients=self.config.compress_gradients,
                use_amp=self.config.use_amp,
                checkpoint_dir=self.config.checkpoint_dir,
                checkpoint_interval=self.config.checkpoint_interval,
            )

            self._trainer = UnifiedDistributedTrainer(model, dist_config, optimizer)
            self._trainer.setup()
            logger.info(f"[Orchestrator] Distributed training initialized (world_size={self.config.world_size})")
        except ImportError as e:
            logger.warning(f"[Orchestrator] Distributed training not available: {e}")

    def wrap_model(self, model: Any) -> Any:
        """Wrap model with DDP if distributed is enabled."""
        if self._trainer is not None:
            return self._trainer.model
        return model

    def train_step(self, batch: tuple, loss_fn: Callable) -> float:
        """Execute distributed training step."""
        if self._trainer is not None:
            return self._trainer.train_step(batch, loss_fn)
        return 0.0

    def barrier(self):
        """Synchronization barrier."""
        if self._trainer is not None:
            self._trainer.barrier()

    def cleanup(self):
        """Cleanup distributed resources."""
        if self._trainer is not None:
            self._trainer.cleanup()

    @property
    def is_main_process(self) -> bool:
        if self._trainer is not None:
            return self._trainer.is_main_process
        return True

    @property
    def available(self) -> bool:
        return self._trainer is not None


class BackgroundEvalWrapper:
    """Wrapper for background evaluation."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._evaluator = None

    def initialize(self, model_getter: Callable):
        """Initialize background evaluator if enabled."""
        if not self.config.enable_background_eval:
            return

        try:
            from app.training.background_eval import (
                BackgroundEvaluator,
                EvalConfig,
            )

            eval_config = EvalConfig(
                eval_interval_steps=self.config.eval_interval_steps,
                games_per_eval=self.config.eval_games_per_check,
            )

            self._evaluator = BackgroundEvaluator(model_getter, eval_config)
            logger.info("[Orchestrator] BackgroundEvaluator initialized")
        except ImportError as e:
            logger.warning(f"[Orchestrator] BackgroundEvaluator not available: {e}")

    def start(self):
        """Start background evaluation thread."""
        if self._evaluator is not None:
            self._evaluator.start()

    def stop(self):
        """Stop background evaluation thread."""
        if self._evaluator is not None:
            self._evaluator.stop()

    def update_step(self, step: int):
        """Update current training step."""
        if self._evaluator is not None:
            self._evaluator.update_step(step)

    def get_current_elo(self) -> float:
        """Get current Elo estimate."""
        if self._evaluator is not None:
            return self._evaluator.get_current_elo()
        # Use canonical INITIAL_ELO_RATING from app.config.thresholds
        from app.config.thresholds import INITIAL_ELO_RATING
        return INITIAL_ELO_RATING

    def should_early_stop(self) -> bool:
        """Check if training should early stop."""
        if self._evaluator is not None:
            return self._evaluator.should_early_stop()
        return False

    @property
    def available(self) -> bool:
        return self._evaluator is not None


class CheckpointWrapper:
    """Wrapper for unified checkpoint management.

    Uses checkpoint_unified.py which consolidates:
    - fault_tolerance.py: Comprehensive metadata, hash verification, types
    - advanced_training.py SmartCheckpointManager: Adaptive frequency
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._manager = None

    def initialize(self):
        """Initialize unified checkpoint manager."""
        try:
            from app.training.checkpoint_unified import (
                UnifiedCheckpointConfig,
                UnifiedCheckpointManager,
            )

            ckpt_config = UnifiedCheckpointConfig(
                checkpoint_dir=Path(self.config.checkpoint_dir),
                max_checkpoints=self.config.keep_top_k_checkpoints + 5,
                keep_best=self.config.keep_top_k_checkpoints,
                checkpoint_interval_steps=self.config.checkpoint_interval,
                adaptive_enabled=True,
            )

            self._manager = UnifiedCheckpointManager(ckpt_config)
            logger.info(f"[Orchestrator] UnifiedCheckpointManager initialized (dir={self.config.checkpoint_dir})")
        except ImportError as e:
            logger.warning(f"[Orchestrator] Checkpoint manager not available: {e}")

    def should_save(self, epoch: int, loss: float, step: int | None = None) -> bool:
        """Check if checkpoint should be saved using adaptive logic."""
        if self._manager is None:
            return False
        if hasattr(self._manager, 'should_save'):
            return self._manager.should_save(epoch=epoch, loss=loss, step=step)
        # Fallback for non-unified manager
        return step is not None and step % self.config.checkpoint_interval == 0

    def save(self, model_state: dict, progress: Any, metrics: dict | None = None):
        """Save checkpoint."""
        if self._manager is not None:
            # Import from canonical source (fault_tolerance re-exports from checkpoint_unified)
            from app.training.fault_tolerance import CheckpointType

            self._manager.save_checkpoint(
                model_state=model_state,
                progress=progress,
                checkpoint_type=CheckpointType.REGULAR,
                metrics=metrics,
            )

    def save_best(self, model_state: dict, progress: Any, metric_name: str, metric_value: float):
        """Save checkpoint if this is the best by metric."""
        if self._manager is None:
            return
        if hasattr(self._manager, 'save_best_if_improved'):
            self._manager.save_best_if_improved(
                model_state=model_state,
                progress=progress,
                metric_name=metric_name,
                metric_value=metric_value,
            )

    def load_latest(self) -> dict | None:
        """Load latest checkpoint."""
        if self._manager is not None:
            return self._manager.load_checkpoint()
        return None

    def load_best(self, metric_name: str = 'loss') -> dict | None:
        """Load best checkpoint by metric."""
        if self._manager is not None and hasattr(self._manager, 'load_checkpoint'):
            return self._manager.load_checkpoint(best_by_metric=metric_name)
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get checkpoint manager statistics."""
        if self._manager is not None and hasattr(self._manager, 'get_stats'):
            return self._manager.get_stats()
        return {}

    @property
    def available(self) -> bool:
        return self._manager is not None


# =============================================================================
# Unified Orchestrator
# =============================================================================

class UnifiedTrainingOrchestrator:
    """Unified training orchestrator for STEP-LEVEL training operations.

    .. note:: Orchestrator Hierarchy (2025-12)
        - **UnifiedTrainingOrchestrator** (this): Step-level training operations
          (forward/backward pass, hot buffer, checkpoints, enhancements)
        - **TrainingOrchestrator** (orchestrated_training.py): Manager lifecycle
          coordination (checkpoint manager, rollback manager, data coordinator)
        - **ModelSyncCoordinator** (model_lifecycle.py): Model registry sync
        - **P2P Coordinators** (p2p_integration.py): P2P cluster REST API wrappers

    Use this class when you need to run training steps with advanced features.
    For higher-level pipeline orchestration, use TrainingOrchestrator.

    Provides a single entry point for training with:
    - Distributed training support
    - Priority experience replay (hot buffer)
    - Integrated enhancements (auxiliary tasks, gradient surgery, etc.)
    - Background Elo evaluation
    - Fault-tolerant checkpointing
    - Adaptive learning rate and batch size

    All components are optional and can be enabled/disabled via config.
    """

    def __init__(
        self,
        model: Any,
        config: OrchestratorConfig | None = None,
        optimizer: Any | None = None,
        loss_fn: Callable | None = None,
    ):
        """Initialize unified training orchestrator.

        Args:
            model: PyTorch model to train
            config: Orchestrator configuration
            optimizer: Optional optimizer (created if not provided)
            loss_fn: Optional loss function
        """
        self.config = config or OrchestratorConfig()
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn

        # State
        self._step = 0
        self._epoch = 0
        self._initialized = False

        # Loss history for training quality feedback (plateau/overfit detection)
        self._loss_history: list[float] = []
        self._val_loss_history: list[float] = []
        self._loss_history_maxlen = 100  # Keep last 100 steps

        # Component wrappers
        self._hot_buffer = HotBufferWrapper(self.config)
        self._enhancements = EnhancementsWrapper(self.config)
        self._distributed = DistributedWrapper(self.config)
        self._background_eval = BackgroundEvalWrapper(self.config)
        self._checkpoint = CheckpointWrapper(self.config)

        logger.info(f"[Orchestrator] Created for {self.config.board_type}_{self.config.num_players}p")

    def initialize(self):
        """Initialize all enabled components with health tracking."""
        if self._initialized:
            return

        import time as _time
        torch = get_torch()

        # Track component health and initialization times
        component_health: dict[str, dict[str, Any]] = {}
        total_start = _time.perf_counter()

        # Create optimizer if not provided
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self.config.learning_rate,
            )

        # Auto-tune batch size if enabled and on GPU
        if self.config.auto_tune_batch_size and torch.cuda.is_available():
            self._auto_tune_batch_size(torch)

        # Initialize components in order with timing and error tracking
        def _init_component(name: str, init_fn, *args) -> bool:
            """Initialize a component with health tracking."""
            start = _time.perf_counter()
            try:
                init_fn(*args)
                elapsed_ms = (_time.perf_counter() - start) * 1000
                component_health[name] = {
                    "status": "ok",
                    "init_time_ms": round(elapsed_ms, 2),
                    "error": None,
                }
                logger.debug(f"[Orchestrator] {name} initialized in {elapsed_ms:.1f}ms")
                return True
            except Exception as e:
                elapsed_ms = (_time.perf_counter() - start) * 1000
                component_health[name] = {
                    "status": "failed",
                    "init_time_ms": round(elapsed_ms, 2),
                    "error": str(e),
                }
                logger.warning(f"[Orchestrator] {name} initialization failed: {e}")
                return False

        _init_component("Checkpoint", self._checkpoint.initialize)
        _init_component("HotBuffer", self._hot_buffer.initialize)
        _init_component("Enhancements", self._enhancements.initialize, self._model)
        _init_component("Distributed", self._distributed.initialize, self._model, self._optimizer)
        _init_component("BackgroundEval", self._background_eval.initialize, lambda: self._model)

        # Wrap model if distributed
        if self._distributed.available:
            self._model = self._distributed.wrap_model(self._model)

        # Try to resume from checkpoint
        if self.config.auto_resume:
            self._try_resume()

        self._initialized = True
        total_elapsed = (_time.perf_counter() - total_start) * 1000

        # Build component status summary
        active = []
        failed = []
        for name, health in component_health.items():
            wrapper = getattr(self, f"_{name.lower()}", None)
            if wrapper and getattr(wrapper, 'available', False):
                active.append(name)
            elif health["status"] == "failed":
                failed.append(name)

        # Store health status for monitoring
        self._component_health = component_health
        self._init_time_ms = round(total_elapsed, 2)

        # Log summary
        if failed:
            logger.warning(
                f"[Orchestrator] Initialized in {total_elapsed:.0f}ms. "
                f"Active: [{', '.join(active) or 'none'}], "
                f"Failed: [{', '.join(failed)}]"
            )
        else:
            logger.info(
                f"[Orchestrator] Initialized in {total_elapsed:.0f}ms. "
                f"Active: [{', '.join(active) or 'basic'}]"
            )

        # Log detailed timing in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            for name, health in component_health.items():
                logger.debug(
                    f"[Orchestrator] Component {name}: "
                    f"status={health['status']}, time={health['init_time_ms']}ms"
                )

    def get_health_status(self) -> dict[str, Any]:
        """Get component health status for monitoring.

        Returns:
            Dict with component health info including:
            - initialized: Whether orchestrator is initialized
            - init_time_ms: Total initialization time
            - components: Per-component health details
        """
        return {
            "initialized": self._initialized,
            "init_time_ms": getattr(self, '_init_time_ms', 0),
            "components": getattr(self, '_component_health', {}),
            "active_components": [
                name for name, health in getattr(self, '_component_health', {}).items()
                if health.get("status") == "ok"
            ],
        }

    def _try_resume(self):
        """Try to resume from checkpoint."""
        if not self._checkpoint.available:
            return

        checkpoint = self._checkpoint.load_latest()
        if checkpoint is not None:
            get_torch()
            self._model.load_state_dict(checkpoint.get("model_state_dict", {}))
            if self._optimizer and "optimizer_state_dict" in checkpoint:
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            progress = checkpoint.get("progress", {})
            self._step = progress.get("global_step", 0)
            self._epoch = progress.get("epoch", 0)
            logger.info(f"[Orchestrator] Resumed from step {self._step}, epoch {self._epoch}")

    def _auto_tune_batch_size(self, _torch_module) -> None:
        """Auto-tune batch size via GPU profiling.

        Uses binary search with actual forward/backward passes to find the
        largest batch size that fits in GPU memory. Updates config.batch_size
        with the tuned value.
        """
        try:
            from app.training.config import auto_tune_batch_size as tune_batch_fn

            original_batch = self.config.batch_size
            device = next(self._model.parameters()).device

            # Determine feature shape based on board type
            board_size_map = {
                "square8": 8,
                "square19": 19,
                "hex8": 15,  # Hex8 uses 15x15 grid
                "hexagonal": 25,
            }
            board_size = board_size_map.get(self.config.board_type, 8)
            history_length = 3  # Default history length

            # Feature channels: 14 base * history_length (typical)
            feature_shape = (14 * history_length, board_size, board_size)
            globals_shape = (20,)  # 20 global features
            policy_size = board_size * board_size * 15  # Approximate policy size

            logger.info(f"[Orchestrator] Auto-tuning batch size (current: {original_batch})...")

            tuned_batch = tune_batch_fn(
                model=self._model,
                device=device,
                feature_shape=feature_shape,
                globals_shape=globals_shape,
                policy_size=policy_size,
                min_batch=max(32, original_batch // 4),
                max_batch=min(8192, original_batch * 8),
            )

            # Update config with tuned value (create a new dataclass if frozen)
            if tuned_batch != original_batch:
                # OrchestratorConfig is a dataclass, update via object.__setattr__
                object.__setattr__(self.config, 'batch_size', tuned_batch)
                logger.info(f"[Orchestrator] Auto-tuned batch size: {tuned_batch} (was {original_batch})")
            else:
                logger.info(f"[Orchestrator] Batch size unchanged after auto-tune: {tuned_batch}")

        except Exception as e:
            logger.warning(f"[Orchestrator] Batch size auto-tuning failed: {e}. Using original.")

    def train_step(self, batch: tuple[Any, ...]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Input batch (features, policy_targets, value_targets, ...)

        Returns:
            Dictionary of losses and metrics
        """
        torch = get_torch()

        if not self._initialized:
            self.initialize()

        # Get dynamic batch size from enhancements
        if self._enhancements.available and self.config.enable_batch_scheduling:
            self._enhancements.get_batch_size()
        else:
            pass

        # Apply data augmentation
        if self._enhancements.available and self.config.enable_augmentation:
            features = batch[0]
            if len(batch) > 2:
                # Sparse policy format
                policy_indices = batch[1] if isinstance(batch[1], list) else [batch[1]]
                policy_values = batch[2] if isinstance(batch[2], list) else [batch[2]]
                features, policy_indices, policy_values = self._enhancements.augment_batch(
                    features, policy_indices, policy_values
                )

        # Forward pass
        self._model.train()

        if self._distributed.available:
            loss = self._distributed.train_step(batch, self._loss_fn)
            metrics = {"loss": loss}
        else:
            # Standard training step
            inputs = batch[0]
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            self._optimizer.zero_grad()
            outputs = self._model(inputs)

            if self._loss_fn is not None:
                targets = batch[1] if len(batch) > 1 else None
                if targets is not None and torch.cuda.is_available():
                    targets = targets.cuda()
                loss = self._loss_fn(outputs, targets)
            else:
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

            loss.backward()
            self._optimizer.step()

            metrics = {"loss": loss.item()}

        # Update step counters
        self._step += 1
        self._enhancements.update_step()
        self._background_eval.update_step(self._step)

        # Publish training metrics
        config_key = f"{self.config.board_type}_{self.config.num_players}p"
        if _HAS_TRAINING_METRICS and TrainingMetrics is not None:
            lr = self._optimizer.param_groups[0].get("lr", 0.0) if self._optimizer else 0.0
            TrainingMetrics.step(
                config_key=config_key,
                step=self._step,
                loss=metrics.get("loss", 0.0),
                learning_rate=lr,
            )

        # Publish step completed event (every N steps to avoid event flood)
        if _HAS_TRAINING_EVENTS and self._step % self.config.log_interval == 0:
            try:
                publish_step_completed_sync(
                    config_key=config_key,
                    step=self._step,
                    loss=metrics.get("loss", 0.0),
                )
            except Exception as e:
                logger.debug(f"[Orchestrator] Event publish failed: {e}")

        # Periodic checkpointing
        if self._step % self.config.checkpoint_interval == 0:
            self._save_checkpoint(metrics)

        # Log progress
        if self.config.verbose and self._step % self.config.log_interval == 0:
            self._log_progress(metrics)

        # Track loss history for training quality feedback
        if "loss" in metrics:
            self._loss_history.append(metrics["loss"])
            if len(self._loss_history) > self._loss_history_maxlen:
                self._loss_history.pop(0)
        if "val_loss" in metrics:
            self._val_loss_history.append(metrics["val_loss"])
            if len(self._val_loss_history) > self._loss_history_maxlen:
                self._val_loss_history.pop(0)

        return metrics

    def _save_checkpoint(self, metrics: dict[str, float]):
        """Save training checkpoint."""
        if not self._checkpoint.available:
            return

        if not self._distributed.is_main_process:
            return

        config_key = f"{self.config.board_type}_{self.config.num_players}p"

        # Use distributed lock for cross-node coordination (December 2025)
        if _HAS_DISTRIBUTED_LOCKS and TrainingLocks is not None:
            with TrainingLocks.checkpoint_save(config_key, timeout=120) as lock:
                if not lock:
                    logger.warning(
                        f"[Orchestrator] Could not acquire checkpoint lock for {config_key}, "
                        f"another node may be saving"
                    )
                    return
                self._save_checkpoint_locked(metrics, config_key)
        else:
            # Fallback to unlocked save if distributed locks unavailable
            self._save_checkpoint_locked(metrics, config_key)

    def _save_checkpoint_locked(self, metrics: dict[str, float], config_key: str):
        """Save checkpoint while holding the distributed lock.

        Args:
            metrics: Training metrics to save
            config_key: Configuration key for event publishing
        """
        try:
            # Import from canonical source (fault_tolerance re-exports from checkpoint_unified)
            from app.training.fault_tolerance import TrainingProgress

            progress = TrainingProgress(
                epoch=self._epoch,
                global_step=self._step,
                learning_rate=self.config.learning_rate,
            )

            self._checkpoint.save(
                model_state=self._model.state_dict(),
                progress=progress,
                metrics=metrics,
            )

            # Publish checkpoint metric and event
            if _HAS_TRAINING_METRICS and TrainingMetrics is not None:
                TrainingMetrics.checkpoint(
                    config_key=config_key,
                    step=self._step,
                    is_best=False,  # Best checkpoint tracking handled separately
                )
            if _HAS_TRAINING_EVENTS and publish_checkpoint_saved_sync is not None:
                try:
                    checkpoint_path = str(self._checkpoint._manager._checkpoint_dir) if hasattr(self._checkpoint, '_manager') else ""
                    publish_checkpoint_saved_sync(
                        config_key=config_key,
                        checkpoint_path=checkpoint_path,
                        step=self._step,
                    )
                except Exception as evt_e:
                    logger.debug(f"[Orchestrator] Checkpoint event publish failed: {evt_e}")
        except Exception as e:
            logger.warning(f"[Orchestrator] Checkpoint save failed: {e}")

    def _log_progress(self, metrics: dict[str, float]):
        """Log training progress."""
        elo = self._background_eval.get_current_elo()
        curriculum = self._enhancements.get_curriculum_params()
        stage = curriculum.get("name", "default")

        logger.info(
            f"[Orchestrator] Step {self._step}: "
            f"loss={metrics.get('loss', 0):.4f}, "
            f"elo={elo:.0f}, "
            f"stage={stage}"
        )

    def start_background_services(self):
        """Start background services (evaluation, etc.)."""
        self._background_eval.start()

    def stop_background_services(self):
        """Stop background services."""
        self._background_eval.stop()

    def cleanup(self):
        """Cleanup all resources."""
        self.stop_background_services()
        self._distributed.cleanup()
        logger.info("[Orchestrator] Cleanup complete")

    @property
    def model(self) -> Any:
        """Get the model (possibly DDP-wrapped)."""
        return self._model

    @property
    def step(self) -> int:
        """Current training step."""
        return self._step

    @property
    def epoch(self) -> int:
        """Current epoch."""
        return self._epoch

    def set_epoch(self, epoch: int):
        """Set current epoch."""
        self._epoch = epoch

    def should_stop(self) -> bool:
        """Check if training should stop (early stopping)."""
        return self._background_eval.should_early_stop()

    def get_training_quality(self, config_key: str = "") -> dict[str, Any]:
        """Get training quality metrics for selfplay feedback loop.

        Analyzes recent loss history to detect plateau and overfitting,
        which signals that selfplay should generate more diverse data.

        Args:
            config_key: Config identifier (unused, for API compatibility)

        Returns:
            Dict with training quality indicators:
            - loss_plateau: True if loss not improving
            - overfit_detected: True if train/val loss diverging
            - last_loss: Most recent loss value
            - loss_trend: Slope of recent losses (negative = improving)
            - train_val_gap: Gap between train and val loss
        """
        result = {
            "loss_plateau": False,
            "overfit_detected": False,
            "last_loss": None,
            "loss_trend": 0.0,
            "train_val_gap": 0.0,
        }

        # Need at least 10 samples for meaningful analysis
        if len(self._loss_history) < 10:
            return result

        # Get recent loss values
        recent = self._loss_history[-20:]  # Last 20 steps
        result["last_loss"] = recent[-1]

        # Detect plateau: loss not decreasing significantly
        # Compare first half to second half of recent history
        first_half = sum(recent[:len(recent)//2]) / max(1, len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / max(1, len(recent) - len(recent)//2)
        improvement = (first_half - second_half) / max(first_half, 1e-6)

        # Plateau if improvement < 1% over recent window
        result["loss_plateau"] = improvement < 0.01
        result["loss_trend"] = -improvement  # Negative = improving

        # Detect overfitting: train loss decreasing but val loss increasing
        if len(self._val_loss_history) >= 10:
            val_recent = self._val_loss_history[-20:]
            val_first_half = sum(val_recent[:len(val_recent)//2]) / max(1, len(val_recent)//2)
            val_second_half = sum(val_recent[len(val_recent)//2:]) / max(1, len(val_recent) - len(val_recent)//2)
            val_improvement = (val_first_half - val_second_half) / max(val_first_half, 1e-6)

            # Overfit if train improving but val getting worse
            if improvement > 0.02 and val_improvement < -0.02:
                result["overfit_detected"] = True

            result["train_val_gap"] = val_recent[-1] - recent[-1] if val_recent else 0.0

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics from all components."""
        metrics = {
            "step": self._step,
            "epoch": self._epoch,
            "elo": self._background_eval.get_current_elo(),
        }

        if self._enhancements.available:
            metrics["curriculum"] = self._enhancements.get_curriculum_params()

        return metrics

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        self.start_background_services()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# =============================================================================
# Factory Functions
# =============================================================================

def create_orchestrator(
    model: Any,
    board_type: str = "square8",
    num_players: int = 2,
    **kwargs,
) -> UnifiedTrainingOrchestrator:
    """Factory function to create a training orchestrator.

    Args:
        model: PyTorch model
        board_type: Board type string
        num_players: Number of players
        **kwargs: Additional config options

    Returns:
        Configured UnifiedTrainingOrchestrator
    """
    config = OrchestratorConfig(
        board_type=board_type,
        num_players=num_players,
        **kwargs,
    )
    return UnifiedTrainingOrchestrator(model, config)
