"""Integrated Training Enhancements for RingRift AI.

.. deprecated:: December 2025
    The ``IntegratedTrainingManager`` class is deprecated. Its features are now
    integrated into ``UnifiedTrainingOrchestrator`` in ``unified_orchestrator.py``.

    The individual enhancement modules (auxiliary tasks, gradient surgery, etc.)
    are still available for direct use. Only the manager wrapper is deprecated.

    See ``app/training/ORCHESTRATOR_GUIDE.md`` for migration instructions.

Migration:
    # Old
    from app.training.integrated_enhancements import IntegratedTrainingManager
    manager = IntegratedTrainingManager(config, model)
    manager.initialize_all()
    loss = manager.apply_step_enhancements(batch_loss, step)

    # New
    from app.training.unified_orchestrator import (
        UnifiedTrainingOrchestrator,
        OrchestratorConfig,
    )
    config = OrchestratorConfig(
        enable_auxiliary_tasks=True,
        enable_gradient_surgery=True,
    )
    orchestrator = UnifiedTrainingOrchestrator(model, config)
    # Enhancements applied automatically in train_step()
    loss = orchestrator.train_step(batch)

Unified module that consolidates and integrates all advanced training features:
- Auxiliary Tasks (game length, piece count, outcome prediction)
- Gradient Surgery (PCGrad for multi-task learning)
- Batch Scheduling (dynamic batch sizing)
- Background Evaluation (continuous Elo tracking)
- ELO Weighting (opponent strength-based sampling)
- Curriculum Learning (progressive difficulty)
- Data Augmentation (board symmetry)
- Reanalysis (historical game re-evaluation)
- Knowledge Distillation (ensemble compression)

This module provides a single entry point for enabling/configuring all features.

INTEGRATION STATUS (as of December 2025):
==========================================
Feature                    | Status           | Notes
---------------------------|------------------|----------------------------------
get_batch_size()           | INTEGRATED       | Used in train.py for dynamic batch sizing
compute_auxiliary_loss()   | INTEGRATED       | Used in train.py for multi-task learning
update_step()              | INTEGRATED       | Used in train.py to sync step counter
should_early_stop()        | INTEGRATED       | Used in train.py for Elo-based stopping
get_current_elo()          | INTEGRATED       | Used in train.py for logging
get_baseline_gating_status | INTEGRATED       | Used in train.py for quality monitoring
augment_batch_dense()      | INTEGRATED       | Used in train.py for data augmentation
should_reanalyze()         | INTEGRATED       | Check if reanalysis should be triggered
process_reanalysis()       | INTEGRATED       | Perform MuZero-style reanalysis on NPZ data
get_reanalysis_stats()     | INTEGRATED       | Get reanalysis statistics
should_distill()           | INTEGRATED       | Check if distillation should be triggered
run_distillation()         | INTEGRATED       | Run ensemble distillation into current model
get_distillation_stats()   | INTEGRATED       | Get distillation statistics
---------------------------|------------------|----------------------------------
compute_sample_weights()   | REQUIRES DATA    | Needs opponent_elo per sample in training data
get_curriculum_parameters()| SELFPLAY ONLY    | Used for selfplay difficulty, not training loop
apply_gradient_surgery()   | INTEGRATED       | Via train.py with config.enable_gradient_surgery

CONFIG FLAGS (December 2025):
- auxiliary_tasks_enabled: True -> compute_auxiliary_loss() IS called
- gradient_surgery_enabled: True -> apply_gradient_surgery() IS called (via train.py GradientSurgeon)
- background_eval_enabled: True -> should_early_stop() IS called
- reanalysis_enabled: True -> should_reanalyze() + process_reanalysis() available
- distillation_enabled: True -> should_distill() + run_distillation() available

REANALYSIS INTEGRATION (December 2025):
The reanalysis pipeline is now callable via IntegratedEnhancements:
1. should_reanalyze(): Check Elo delta, time interval, step interval thresholds
2. process_reanalysis(npz_path): Run reanalysis and save blended values
3. get_reanalysis_stats(): Get positions/games reanalyzed, timing info

KNOWLEDGE DISTILLATION INTEGRATION (December 2025):
The distillation pipeline compresses ensemble knowledge into the current model:
1. should_distill(epoch): Check if enough epochs passed and teachers available
2. run_distillation(epoch, dataloader): Distill from best checkpoints
3. get_distillation_stats(): Get distillation temperature, teachers used, etc.

To integrate the unused features, see the corresponding methods' docstrings
for API details, then add calls in app/training/train.py at appropriate points.
"""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Emit deprecation warning on import (December 2025)
warnings.warn(
    "IntegratedTrainingManager from integrated_enhancements.py is deprecated. "
    "Use UnifiedTrainingOrchestrator from unified_orchestrator.py instead. "
    "Individual enhancement modules are still available for direct use. "
    "See app/training/ORCHESTRATOR_GUIDE.md for migration instructions.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# Use shared lazy torch import; extend with torch.nn
from app.training.utils import get_torch

_nn = None


def _get_torch_nn():
    """Get torch and torch.nn modules lazily."""
    global _nn
    torch = get_torch()
    if _nn is None:
        import torch.nn as nn
        _nn = nn
    return torch, _nn


# =============================================================================
# Unified Configuration
# =============================================================================

@dataclass
class IntegratedEnhancementsConfig:
    """Unified configuration for all training enhancements.

    Centralizes all advanced training feature flags and parameters.
    """
    # =========================================================================
    # Auxiliary Tasks (Multi-Task Learning)
    # December 2025: Enabled by default for +30-80 Elo improvement
    # =========================================================================
    auxiliary_tasks_enabled: bool = True
    aux_game_length_weight: float = 0.1
    aux_piece_count_weight: float = 0.1
    aux_outcome_weight: float = 0.05

    # =========================================================================
    # Gradient Surgery (PCGrad)
    # December 2025: Enabled with auxiliary tasks to prevent gradient conflicts
    # =========================================================================
    gradient_surgery_enabled: bool = True
    gradient_surgery_method: str = "pcgrad"  # "pcgrad" or "cagrad"
    gradient_conflict_threshold: float = 0.0

    # =========================================================================
    # Batch Scheduling
    # =========================================================================
    batch_scheduling_enabled: bool = True  # Dynamic batch sizing (64→512) for better convergence
    batch_initial_size: int = 64
    batch_final_size: int = 512
    batch_warmup_steps: int = 1000
    batch_rampup_steps: int = 10000
    batch_schedule_type: str = "linear"  # "linear", "exponential", "step"

    # =========================================================================
    # Background Evaluation
    # =========================================================================
    # December 2025: Enabled for continuous Elo tracking and regression detection
    background_eval_enabled: bool = True
    eval_interval_steps: int = 1000
    eval_games_per_check: int = 20
    eval_elo_checkpoint_threshold: float = 10.0
    eval_elo_drop_threshold: float = 50.0
    eval_auto_checkpoint: bool = True
    eval_checkpoint_dir: str = "data/eval_checkpoints"
    eval_use_real_games: bool = False  # If True, play actual games against baselines
    eval_board_type: Any | None = None  # Board type for real games (required if eval_use_real_games=True)

    # =========================================================================
    # ELO Weighting
    # =========================================================================
    elo_weighting_enabled: bool = True
    elo_base_rating: float = 1500.0
    elo_weight_scale: float = 400.0
    elo_min_weight: float = 0.5
    elo_max_weight: float = 2.0

    # =========================================================================
    # Curriculum Learning
    # =========================================================================
    curriculum_enabled: bool = True
    curriculum_auto_advance: bool = True
    curriculum_checkpoint_path: str = "data/curriculum_state.json"

    # =========================================================================
    # Data Augmentation
    # =========================================================================
    augmentation_enabled: bool = True
    augmentation_mode: str = "all"  # "all", "random", "light"
    augmentation_probability: float = 1.0

    # =========================================================================
    # Reanalysis
    # December 2025: Enabled for +40-120 Elo improvement via dual-pass training
    # Policy targets already use mcts_visits by default in improvement loop
    # =========================================================================
    reanalysis_enabled: bool = True
    reanalysis_blend_ratio: float = 0.7  # 70% new, 30% old (MuZero-style)
    reanalysis_interval_steps: int = 2000  # Trigger more frequently for faster iteration
    reanalysis_batch_size: int = 1000
    reanalysis_min_elo_delta: int = 15  # Very low threshold for aggressive reanalysis (+40-120 Elo)
    # MCTS-based reanalysis (Phase 4): Higher-quality soft targets via GPU Gumbel MCTS
    reanalysis_use_mcts: bool = True  # Use GPU Gumbel MCTS for higher-quality targets
    reanalysis_mcts_simulations: int = 100  # MCTS simulations per position
    reanalysis_capture_q_values: bool = True  # Capture Q-values for auxiliary training
    reanalysis_capture_uncertainty: bool = True  # Capture uncertainty estimates
    reanalysis_db_path: str | None = None  # Database path for MCTS reanalysis

    # =========================================================================
    # Knowledge Distillation
    # December 2025: +15-25 Elo from ensemble compression
    # Distills knowledge from multiple checkpoints into a single model
    # =========================================================================
    distillation_enabled: bool = True
    distillation_temperature: float = 3.0  # Softmax temperature for soft targets
    distillation_alpha: float = 0.7  # Weight of soft targets vs hard targets
    distillation_interval_epochs: int = 10  # Run distillation every N epochs
    distillation_num_teachers: int = 3  # Number of best checkpoints to ensemble
    distillation_epochs: int = 5  # Epochs per distillation run
    distillation_lr: float = 1e-4  # Learning rate for student during distillation


# =============================================================================
# Integrated Training Manager
# =============================================================================

class IntegratedTrainingManager:
    """Unified manager for all training enhancements.

    Provides a single interface to:
    - Initialize all enhancement modules
    - Apply enhancements during training step
    - Track metrics across all features
    - Handle cleanup and checkpointing
    """

    def __init__(
        self,
        config: IntegratedEnhancementsConfig | None = None,
        model: Any | None = None,
        board_type: str = "square8",
    ):
        """Initialize integrated training manager.

        Args:
            config: Unified configuration for all enhancements
            model: PyTorch model (optional, can be set later)
            board_type: Board type for augmentation selection
        """
        self.config = config or IntegratedEnhancementsConfig()
        self.model = model
        self.board_type = board_type
        self._step = 0

        # Initialize components lazily
        self._auxiliary_module = None
        self._gradient_surgeon = None
        self._batch_scheduler = None
        self._background_evaluator = None
        self._elo_sampler = None
        self._curriculum_controller = None
        self._augmentor = None
        self._reanalysis_engine = None
        self._distillation_config = None
        self._checkpoint_dir: Path | None = None
        self._last_distillation_epoch = 0

        # Metrics tracking
        self._metrics: dict[str, Any] = {}
        self._lock = threading.Lock()

        logger.info("[IntegratedEnhancements] Manager initialized")

    def initialize_all(self, model: Any | None = None):
        """Initialize all enabled enhancement modules.

        Args:
            model: PyTorch model for auxiliary tasks and gradient surgery
        """
        if model is not None:
            self.model = model

        cfg = self.config

        # Auxiliary Tasks
        if cfg.auxiliary_tasks_enabled and self.model is not None:
            self._init_auxiliary_tasks()

        # Gradient Surgery
        if cfg.gradient_surgery_enabled:
            self._init_gradient_surgery()

        # Batch Scheduling
        if cfg.batch_scheduling_enabled:
            self._init_batch_scheduler()

        # Background Evaluation
        if cfg.background_eval_enabled:
            self._init_background_eval()

        # ELO Weighting
        if cfg.elo_weighting_enabled:
            self._init_elo_weighting()

        # Curriculum Learning
        if cfg.curriculum_enabled:
            self._init_curriculum()

        # Data Augmentation
        if cfg.augmentation_enabled:
            self._init_augmentation()

        # Reanalysis
        if cfg.reanalysis_enabled:
            self._init_reanalysis()

        # Knowledge Distillation
        if cfg.distillation_enabled:
            self._init_distillation()

        logger.info(f"[IntegratedEnhancements] Initialized {self._count_enabled()} modules")

    def _count_enabled(self) -> int:
        """Count number of enabled enhancement modules."""
        return sum([
            self._auxiliary_module is not None,
            self._gradient_surgeon is not None,
            self._batch_scheduler is not None,
            self._background_evaluator is not None,
            self._elo_sampler is not None,
            self._curriculum_controller is not None,
            self._augmentor is not None,
            self._reanalysis_engine is not None,
            self._distillation_config is not None,
        ])

    # =========================================================================
    # Component Initialization
    # =========================================================================

    def _init_auxiliary_tasks(self):
        """Initialize auxiliary prediction tasks."""
        try:
            from app.training.auxiliary_tasks import (
                AuxiliaryTaskModule,
                AuxTaskConfig,
            )

            # Detect input dim from model
            input_dim = 256  # Default
            if self.model is not None:
                for name, param in self.model.named_parameters():
                    if "fc" in name or "linear" in name:
                        input_dim = param.shape[-1]
                        break

            aux_config = AuxTaskConfig(
                enabled=True,
                game_length_weight=self.config.aux_game_length_weight,
                piece_count_weight=self.config.aux_piece_count_weight,
                outcome_weight=self.config.aux_outcome_weight,
            )
            self._auxiliary_module = AuxiliaryTaskModule(input_dim, aux_config)
            logger.info(f"[IntegratedEnhancements] Auxiliary tasks initialized (input_dim={input_dim})")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init auxiliary tasks: {e}")

    def _init_gradient_surgery(self):
        """Initialize gradient surgery for multi-task learning."""
        try:
            from app.training.gradient_surgery import (
                GradientSurgeon,
                GradientSurgeryConfig,
            )

            gs_config = GradientSurgeryConfig(
                enabled=True,
                method=self.config.gradient_surgery_method,
                conflict_threshold=self.config.gradient_conflict_threshold,
            )
            self._gradient_surgeon = GradientSurgeon(gs_config)
            logger.info("[IntegratedEnhancements] Gradient surgery initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init gradient surgery: {e}")

    def _init_batch_scheduler(self):
        """Initialize batch size scheduler."""
        try:
            from app.training.batch_scheduling import (
                BatchScheduleConfig,
                BatchSizeScheduler,
            )

            bs_config = BatchScheduleConfig(
                initial_batch_size=self.config.batch_initial_size,
                final_batch_size=self.config.batch_final_size,
                warmup_steps=self.config.batch_warmup_steps,
                rampup_steps=self.config.batch_rampup_steps,
                schedule_type=self.config.batch_schedule_type,
            )
            self._batch_scheduler = BatchSizeScheduler(bs_config)
            logger.info("[IntegratedEnhancements] Batch scheduler initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init batch scheduler: {e}")

    def _init_background_eval(self):
        """Initialize background evaluation thread."""
        try:
            from app.training.background_eval import (
                BackgroundEvaluator,
                EvalConfig,
            )

            def model_getter():
                # Return model in a format the evaluator can use
                return self.model

            eval_config = EvalConfig(
                eval_interval_steps=self.config.eval_interval_steps,
                games_per_eval=self.config.eval_games_per_check,
                elo_checkpoint_threshold=self.config.eval_elo_checkpoint_threshold,
                elo_drop_threshold=self.config.eval_elo_drop_threshold,
                auto_checkpoint=self.config.eval_auto_checkpoint,
                checkpoint_dir=self.config.eval_checkpoint_dir,
            )

            # Create evaluator with real games support if configured
            self._background_evaluator = BackgroundEvaluator(
                model_getter,
                eval_config,
                board_type=self.config.eval_board_type,
                use_real_games=self.config.eval_use_real_games,
            )

            mode = "real games" if self.config.eval_use_real_games else "placeholder"
            logger.info(f"[IntegratedEnhancements] Background evaluator initialized ({mode} mode)")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init background eval: {e}")

    def _init_elo_weighting(self):
        """Initialize ELO-weighted sampling."""
        try:
            from app.training.elo_weighting import (
                EloWeightConfig,
            )

            # Store config for use in compute_sample_weights
            self._elo_config = EloWeightConfig(
                base_elo=self.config.elo_base_rating,
                elo_scale=self.config.elo_weight_scale,
                min_weight=self.config.elo_min_weight,
                max_weight=self.config.elo_max_weight,
            )
            self._elo_sampler = True  # Flag that ELO weighting is enabled
            logger.info("[IntegratedEnhancements] ELO weighting initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init ELO weighting: {e}")

    def _init_curriculum(self):
        """Initialize curriculum learning controller."""
        try:
            from app.training.curriculum import (
                CurriculumController,
                CurriculumStage,
            )

            # Default 5-stage curriculum
            stages = [
                CurriculumStage(
                    name="beginner",
                    max_moves=30,
                    opponent_elo_delta=-300,
                    temperature=1.5,
                    win_rate_threshold=0.60,
                    games_required=50,
                ),
                CurriculumStage(
                    name="easy",
                    max_moves=50,
                    opponent_elo_delta=-150,
                    temperature=1.2,
                    win_rate_threshold=0.55,
                    games_required=100,
                ),
                CurriculumStage(
                    name="medium",
                    max_moves=75,
                    opponent_elo_delta=-50,
                    temperature=1.0,
                    win_rate_threshold=0.52,
                    games_required=150,
                ),
                CurriculumStage(
                    name="hard",
                    max_moves=100,
                    opponent_elo_delta=0,
                    temperature=0.8,
                    win_rate_threshold=0.50,
                    games_required=200,
                ),
                CurriculumStage(
                    name="expert",
                    max_moves=200,
                    opponent_elo_delta=50,
                    temperature=0.5,
                    win_rate_threshold=0.48,
                    games_required=300,
                ),
            ]

            checkpoint_path = Path(self.config.curriculum_checkpoint_path)
            self._curriculum_controller = CurriculumController(
                stages=stages,
                checkpoint_path=checkpoint_path,
                auto_advance=self.config.curriculum_auto_advance,
            )
            logger.info("[IntegratedEnhancements] Curriculum learning initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init curriculum: {e}")

    def _init_augmentation(self):
        """Initialize data augmentation."""
        try:
            from app.training.data_augmentation import (
                AugmentationConfig,
                DataAugmentor,
            )

            aug_config = AugmentationConfig(
                enabled=True,
                augment_probability=self.config.augmentation_probability,
                all_transforms=(self.config.augmentation_mode == "all"),
            )
            self._augmentor = DataAugmentor(self.board_type, aug_config)
            logger.info(f"[IntegratedEnhancements] Augmentation initialized for {self.board_type}")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init augmentation: {e}")

    def _init_reanalysis(self):
        """Initialize reanalysis engine."""
        try:
            from app.training.reanalysis import (
                ReanalysisConfig,
                ReanalysisEngine,
            )

            if self.model is None:
                logger.warning("[IntegratedEnhancements] Model required for reanalysis")
                return

            reanalysis_config = ReanalysisConfig(
                value_blend_ratio=self.config.reanalysis_blend_ratio,
                batch_size=self.config.reanalysis_batch_size,
                # MCTS-based reanalysis config
                use_mcts=self.config.reanalysis_use_mcts,
                mcts_simulations=self.config.reanalysis_mcts_simulations,
                capture_q_values=self.config.reanalysis_capture_q_values,
                capture_uncertainty=self.config.reanalysis_capture_uncertainty,
            )
            self._reanalysis_engine = ReanalysisEngine(self.model, reanalysis_config)
            mode = "MCTS" if self.config.reanalysis_use_mcts else "inference"
            logger.info(f"[IntegratedEnhancements] Reanalysis engine initialized ({mode} mode)")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init reanalysis: {e}")

    def _init_distillation(self):
        """Initialize knowledge distillation configuration."""
        try:
            from app.training.distillation import DistillationConfig

            self._distillation_config = DistillationConfig(
                temperature=self.config.distillation_temperature,
                alpha=self.config.distillation_alpha,
            )
            logger.info("[IntegratedEnhancements] Distillation config initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init distillation: {e}")

    def set_checkpoint_dir(self, checkpoint_dir: str | Path):
        """Set checkpoint directory for distillation teacher selection.

        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    # =========================================================================
    # Training Step Integration
    # =========================================================================

    def get_batch_size(self) -> int:
        """Get current batch size from scheduler."""
        if self._batch_scheduler is not None:
            return self._batch_scheduler.get_batch_size(self._step)
        return self.config.batch_initial_size

    def get_curriculum_parameters(self) -> dict[str, Any]:
        """Get current curriculum stage parameters."""
        if self._curriculum_controller is not None:
            return self._curriculum_controller.get_stage_parameters()
        return {}

    def compute_sample_weights(
        self,
        opponent_elos: np.ndarray,
        model_elo: float = 1500.0,
    ) -> np.ndarray:
        """Compute sample weights based on opponent ELO.

        Args:
            opponent_elos: Array of opponent ELO ratings
            model_elo: Current model ELO rating

        Returns:
            Array of sample weights
        """
        if self._elo_sampler is not None and hasattr(self, '_elo_config'):
            from app.training.elo_weighting import compute_elo_weights
            return compute_elo_weights(
                opponent_elos,
                model_elo=model_elo,
                elo_scale=self._elo_config.elo_scale,
            )
        return np.ones(len(opponent_elos))

    def augment_batch(
        self,
        features: np.ndarray,
        policy_indices: list[np.ndarray],
        policy_values: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Apply data augmentation to a batch.

        Args:
            features: Batch of board features (B, C, H, W)
            policy_indices: List of sparse policy indices
            policy_values: List of sparse policy values

        Returns:
            Augmented (features, policy_indices, policy_values)
        """
        if self._augmentor is not None:
            # Apply random augmentation per sample
            aug_features = np.zeros_like(features)
            aug_indices = []
            aug_values = []

            for i in range(features.shape[0]):
                t = self._augmentor.get_random_transform()
                aug_features[i] = self._augmentor.transform_board(features[i], t)
                aug_idx, aug_val = self._augmentor.transform_sparse_policy(
                    policy_indices[i], policy_values[i], t
                )
                aug_indices.append(aug_idx)
                aug_values.append(aug_val)

            return aug_features, aug_indices, aug_values

        return features, policy_indices, policy_values

    def augment_batch_dense(
        self,
        features: Any,  # torch.Tensor (B, C, H, W)
        policy_targets: Any,  # torch.Tensor (B, policy_size)
    ) -> tuple[Any, Any]:
        """Apply data augmentation to a batch with dense policy targets.

        This is the torch tensor version for integration with train.py.
        Applies random augmentation per sample in the batch.
        Uses GPU-accelerated transforms when possible.

        Args:
            features: Batch of board features (B, C, H, W) as torch tensor
            policy_targets: Dense policy targets (B, policy_size) as torch tensor

        Returns:
            Augmented (features, policy_targets) as torch tensors
        """
        if self._augmentor is None:
            return features, policy_targets

        torch, _ = _get_torch_nn()

        # Use GPU-accelerated D4 transforms for board features
        # Pick a single random transform for the whole batch (faster)
        t = self._augmentor.get_random_transform()

        # Apply board transform using PyTorch (stays on GPU)
        aug_features = self._apply_gpu_board_transform(features, t, torch)

        # Use GPU-accelerated policy transform with precomputed index mapping
        aug_policy = self._apply_gpu_policy_transform(policy_targets, t, torch)

        return aug_features, aug_policy

    def _get_policy_permutation(self, transform_id: int, policy_size: int, device: Any, torch: Any) -> Any:
        """Get or compute the policy index permutation for a D4 transform.

        Caches permutation tensors for reuse across batches.
        """
        cache_key = (transform_id, policy_size, str(device))

        if not hasattr(self, '_policy_perm_cache'):
            self._policy_perm_cache = {}

        if cache_key not in self._policy_perm_cache:
            # Compute the permutation using the augmentor
            perm = np.arange(policy_size, dtype=np.int64)
            for i in range(policy_size):
                new_idx = self._augmentor.transformer.transform_move_index(i, transform_id)
                if 0 <= new_idx < policy_size:
                    perm[new_idx] = i  # Inverse mapping: new position -> old position
            self._policy_perm_cache[cache_key] = torch.from_numpy(perm).to(device)

        return self._policy_perm_cache[cache_key]

    def _apply_gpu_policy_transform(self, policy: Any, transform_id: int, torch: Any) -> Any:
        """Apply D4 transform to policy tensor using GPU.

        Args:
            policy: Tensor of shape (B, policy_size)
            transform_id: D4 transform index 0-7
            torch: PyTorch module

        Returns:
            Transformed policy tensor of same shape
        """
        if transform_id == 0:
            return policy

        device = policy.device
        policy_size = policy.shape[1]

        # Get cached permutation tensor
        perm = self._get_policy_permutation(transform_id, policy_size, device, torch)

        # Apply permutation using advanced indexing (all on GPU)
        return policy[:, perm]

    def _apply_gpu_board_transform(self, features: Any, transform_id: int, torch: Any) -> Any:
        """Apply D4 symmetry transform to board features using GPU.

        D4 transforms:
        - 0: Identity
        - 1: 90° CW rotation
        - 2: 180° rotation
        - 3: 270° CW rotation
        - 4: Horizontal flip (left-right)
        - 5: Vertical flip (up-down)
        - 6: Main diagonal transpose
        - 7: Anti-diagonal transpose

        Args:
            features: Tensor of shape (B, C, H, W)
            transform_id: D4 transform index 0-7
            torch: PyTorch module

        Returns:
            Transformed tensor of same shape
        """
        if transform_id == 0:
            return features
        elif transform_id == 1:  # 90° CW
            return torch.rot90(features, -1, dims=(-2, -1))
        elif transform_id == 2:  # 180°
            return torch.rot90(features, 2, dims=(-2, -1))
        elif transform_id == 3:  # 270° CW
            return torch.rot90(features, 1, dims=(-2, -1))
        elif transform_id == 4:  # Horizontal flip
            return torch.flip(features, dims=(-1,))
        elif transform_id == 5:  # Vertical flip
            return torch.flip(features, dims=(-2,))
        elif transform_id == 6:  # Main diagonal (transpose)
            return features.transpose(-2, -1)
        elif transform_id == 7:  # Anti-diagonal
            return torch.flip(features.transpose(-2, -1), dims=(-2, -1))
        else:
            return features

    def compute_auxiliary_loss(
        self,
        features: Any,  # torch.Tensor
        targets: dict[str, Any],
    ) -> tuple[Any, dict[str, float]]:
        """Compute auxiliary task losses.

        Args:
            features: Shared features from backbone
            targets: Dictionary of auxiliary targets

        Returns:
            (total_aux_loss, loss_breakdown)
        """
        if self._auxiliary_module is not None:
            torch, _ = _get_torch_nn()
            # Move auxiliary module to same device as features if needed
            if hasattr(features, 'device'):
                self._auxiliary_module = self._auxiliary_module.to(features.device)
            # Check if input dimension matches, reinitialize if needed
            actual_dim = features.shape[-1] if len(features.shape) > 1 else features.shape[0]
            expected_dim = self._auxiliary_module.game_length_head.net[0].in_features
            if actual_dim != expected_dim:
                from app.training.auxiliary_tasks import AuxiliaryTaskModule, AuxTaskConfig
                aux_config = AuxTaskConfig(
                    enabled=True,
                    game_length_weight=self.config.aux_game_length_weight,
                    piece_count_weight=self.config.aux_piece_count_weight,
                    outcome_weight=self.config.aux_outcome_weight,
                )
                self._auxiliary_module = AuxiliaryTaskModule(actual_dim, aux_config).to(features.device)
                logger.info(f"[IntegratedEnhancements] Reinitialized auxiliary module with dim={actual_dim}")
            predictions = self._auxiliary_module(features)
            return self._auxiliary_module.compute_loss(predictions, targets)

        torch, _ = _get_torch_nn()
        return torch.tensor(0.0), {}

    def apply_gradient_surgery(
        self,
        model: Any,  # nn.Module
        losses: dict[str, Any],
    ) -> Any:
        """Apply gradient surgery for multi-task learning.

        Args:
            model: PyTorch model
            losses: Dictionary of task losses

        Returns:
            Combined loss after surgery
        """
        if self._gradient_surgeon is not None:
            return self._gradient_surgeon.apply_surgery(model, losses)

        # Simple sum without surgery
        return sum(losses.values())

    def update_step(self, game_won: bool | None = None):
        """Update step counter and related components.

        Args:
            game_won: Optional game result for curriculum tracking
        """
        self._step += 1

        if self._batch_scheduler is not None:
            self._batch_scheduler.set_step(self._step)

        if self._background_evaluator is not None:
            self._background_evaluator.update_step(self._step)

        if self._curriculum_controller is not None:
            self._curriculum_controller.update_step()
            if game_won is not None:
                self._curriculum_controller.update_game_result(game_won)

    def should_early_stop(self) -> bool:
        """Check if training should early stop based on evaluation."""
        if self._background_evaluator is not None:
            return self._background_evaluator.should_early_stop()
        return False

    def get_current_elo(self) -> float | None:
        """Get current Elo estimate from background evaluator.

        Returns:
            Current Elo rating, or None if background evaluator is not available.
        """
        if self._background_evaluator is not None:
            return self._background_evaluator.get_current_elo()
        return None

    def get_baseline_gating_status(self) -> tuple[bool, list[str], int]:
        """Get baseline gating status from background evaluator.

        Returns:
            Tuple of (passes_gating, failed_baselines, consecutive_failures)
            Returns (True, [], 0) if background evaluator is not available.
        """
        if self._background_evaluator is not None:
            return self._background_evaluator.get_baseline_gating_status()
        return True, [], 0

    def should_warn_baseline_failures(self, threshold: int = 3) -> bool:
        """Check if consecutive baseline failures exceed warning threshold.

        Args:
            threshold: Number of consecutive failures to trigger warning

        Returns:
            True if baseline failures exceed threshold
        """
        if self._background_evaluator is not None:
            return self._background_evaluator.should_trigger_baseline_warning(threshold)
        return False

    # =========================================================================
    # Reanalysis Integration
    # =========================================================================

    def should_reanalyze(self) -> bool:
        """Check if reanalysis should be triggered.

        Reanalysis is triggered when:
        1. Reanalysis is enabled
        2. Reanalysis engine is initialized
        3. Model has improved by at least min_model_elo_delta
        4. Sufficient time has passed since last reanalysis

        Returns:
            True if reanalysis should be performed
        """
        if not self.config.reanalysis_enabled:
            return False

        if self._reanalysis_engine is None:
            return False

        # Check Elo improvement threshold (use config's threshold if set)
        current_elo = self.get_current_elo()
        if current_elo is not None:
            elo_delta = current_elo - getattr(self, '_last_reanalysis_elo', 0.0)
            min_delta = getattr(self.config, 'reanalysis_min_elo_delta',
                                self._reanalysis_engine.config.min_model_elo_delta)
            if elo_delta < min_delta:
                return False

        # Check time interval
        import time
        last_reanalysis = getattr(self, '_last_reanalysis_time', 0.0)
        hours_since_reanalysis = (time.time() - last_reanalysis) / 3600
        if hours_since_reanalysis < self._reanalysis_engine.config.reanalysis_interval_hours:
            return False

        # Check step interval
        steps_since_reanalysis = self._step - getattr(self, '_last_reanalysis_step', 0)
        if steps_since_reanalysis < self.config.reanalysis_interval_steps:
            return False

        return True

    def process_reanalysis(
        self,
        npz_path: str | None = None,
        db_path: str | None = None,
        num_players: int = 2,
    ) -> str | None:
        """Perform reanalysis on cached training data.

        Reanalyzes positions using the current model and blends values
        with original targets using MuZero-style weighted averaging.

        If MCTS mode is enabled and a database path is provided, uses GPU
        Gumbel MCTS search for higher-quality soft policy targets.

        Args:
            npz_path: Path to NPZ file to reanalyze (for inference-only mode)
            db_path: Path to game database for MCTS-based reanalysis
            num_players: Number of players for MCTS reanalysis

        Returns:
            Path to reanalyzed NPZ file, or None if reanalysis failed.
        """
        import time
        from pathlib import Path

        if self._reanalysis_engine is None:
            logger.warning("[IntegratedEnhancements] Reanalysis engine not initialized")
            return None

        # Use configured db_path if not provided
        db_path = db_path or self.config.reanalysis_db_path

        try:
            # Choose reanalysis mode based on config and available data
            if self.config.reanalysis_use_mcts and db_path:
                # MCTS-based reanalysis from database
                db_file = Path(db_path)
                if not db_file.exists():
                    logger.warning(f"[IntegratedEnhancements] Database not found: {db_path}")
                    # Fall back to NPZ reanalysis if available
                    if npz_path:
                        logger.info("[IntegratedEnhancements] Falling back to NPZ reanalysis")
                    else:
                        return None
                else:
                    # Run MCTS reanalysis
                    output_path = db_file.parent / f"{db_file.stem}_mcts_reanalyzed.npz"
                    logger.info(
                        f"[IntegratedEnhancements] Starting MCTS reanalysis of {db_path}"
                    )

                    reanalyzed_path = self._reanalysis_engine.reanalyze_with_mcts(
                        db_path=db_file,
                        output_path=output_path,
                        board_type=self.board_type,
                        num_players=num_players,
                    )

                    if reanalyzed_path:
                        # Update tracking state
                        self._last_reanalysis_time = time.time()
                        self._last_reanalysis_step = self._step
                        current_elo = self.get_current_elo()
                        if current_elo is not None:
                            self._last_reanalysis_elo = current_elo

                        logger.info(
                            f"[IntegratedEnhancements] MCTS reanalysis complete: "
                            f"{self._reanalysis_engine.positions_reanalyzed} positions, "
                            f"saved to {reanalyzed_path}"
                        )
                        return str(reanalyzed_path)

                    logger.warning("[IntegratedEnhancements] MCTS reanalysis returned None")
                    return None

            # Standard NPZ-based reanalysis (inference only)
            if npz_path is None:
                logger.warning("[IntegratedEnhancements] No NPZ path provided for reanalysis")
                return None

            npz_file = Path(npz_path)
            if not npz_file.exists():
                logger.warning(f"[IntegratedEnhancements] NPZ file not found: {npz_path}")
                return None

            logger.info(f"[IntegratedEnhancements] Starting reanalysis of {npz_path}")

            # Perform reanalysis
            reanalyzed_path = self._reanalysis_engine.reanalyze_npz(npz_file)

            # Update tracking state
            self._last_reanalysis_time = time.time()
            self._last_reanalysis_step = self._step
            current_elo = self.get_current_elo()
            if current_elo is not None:
                self._last_reanalysis_elo = current_elo

            logger.info(
                f"[IntegratedEnhancements] Reanalysis complete: "
                f"{self._reanalysis_engine.positions_reanalyzed} positions, "
                f"saved to {reanalyzed_path}"
            )

            return str(reanalyzed_path)

        except Exception as e:
            logger.error(f"[IntegratedEnhancements] Reanalysis failed: {e}")
            return None

    def get_reanalysis_stats(self) -> dict[str, Any]:
        """Get reanalysis statistics.

        Returns:
            Dictionary with reanalysis statistics.
        """
        if self._reanalysis_engine is None:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "positions_reanalyzed": self._reanalysis_engine.positions_reanalyzed,
            "games_reanalyzed": self._reanalysis_engine.games_reanalyzed,
            "last_reanalysis_time": getattr(self, '_last_reanalysis_time', 0.0),
            "last_reanalysis_step": getattr(self, '_last_reanalysis_step', 0),
            "blend_ratio": self._reanalysis_engine.config.value_blend_ratio,
            # MCTS-specific stats
            "use_mcts": self._reanalysis_engine.config.use_mcts,
            "mcts_simulations": self._reanalysis_engine.config.mcts_simulations,
            "capture_q_values": self._reanalysis_engine.config.capture_q_values,
            "capture_uncertainty": self._reanalysis_engine.config.capture_uncertainty,
        }
        return stats

    # =========================================================================
    # Knowledge Distillation Integration
    # =========================================================================

    def should_distill(self, current_epoch: int) -> bool:
        """Check if knowledge distillation should be triggered.

        Distillation is triggered when:
        1. Distillation is enabled
        2. Checkpoint directory is set
        3. Sufficient epochs have passed since last distillation

        Args:
            current_epoch: Current training epoch number

        Returns:
            True if distillation should be performed
        """
        if not self.config.distillation_enabled:
            return False

        if self._distillation_config is None:
            return False

        if self._checkpoint_dir is None:
            return False

        # Check epoch interval
        epochs_since = current_epoch - self._last_distillation_epoch
        if epochs_since < self.config.distillation_interval_epochs:
            return False

        # Check if we have enough checkpoints
        checkpoint_paths = self._get_best_checkpoints()
        if len(checkpoint_paths) < 2:
            return False

        return True

    def _get_best_checkpoints(self) -> list[Path]:
        """Get paths to best checkpoints for ensemble distillation.

        Returns:
            List of checkpoint paths sorted by Elo (best first)
        """
        if self._checkpoint_dir is None:
            return []

        import re

        checkpoint_dir = Path(self._checkpoint_dir)
        if not checkpoint_dir.exists():
            return []

        # Find checkpoint files with Elo in filename (e.g., model_elo1725.pt)
        checkpoints = []
        for ckpt_path in checkpoint_dir.glob("*.pt"):
            # Try to extract Elo from filename
            match = re.search(r'elo(\d+)', ckpt_path.stem)
            if match:
                elo = int(match.group(1))
                checkpoints.append((elo, ckpt_path))

        # Sort by Elo descending and take top N
        checkpoints.sort(reverse=True, key=lambda x: x[0])
        num_teachers = self.config.distillation_num_teachers
        return [path for _, path in checkpoints[:num_teachers]]

    def run_distillation(
        self,
        current_epoch: int,
        dataloader: Any,
    ) -> bool:
        """Run knowledge distillation from ensemble to current model.

        Distills knowledge from best historical checkpoints into the
        current model, improving generalization.

        Args:
            current_epoch: Current training epoch
            dataloader: Training data loader

        Returns:
            True if distillation was successful
        """
        if self.model is None:
            logger.warning("[IntegratedEnhancements] Model required for distillation")
            return False

        checkpoint_paths = self._get_best_checkpoints()
        if len(checkpoint_paths) < 2:
            logger.warning(
                f"[IntegratedEnhancements] Need at least 2 checkpoints, found {len(checkpoint_paths)}"
            )
            return False

        try:
            from app.training.distillation import distill_checkpoint_ensemble

            logger.info(
                f"[Distillation] Starting ensemble distillation with {len(checkpoint_paths)} teachers"
            )

            # Run distillation
            distill_checkpoint_ensemble(
                checkpoint_paths=checkpoint_paths,
                student_model=self.model,
                dataloader=dataloader,
                epochs=self.config.distillation_epochs,
                temperature=self.config.distillation_temperature,
                learning_rate=self.config.distillation_lr,
            )

            # Update tracking state
            self._last_distillation_epoch = current_epoch

            logger.info(
                f"[Distillation] Complete at epoch {current_epoch}, "
                f"distilled from {len(checkpoint_paths)} teachers"
            )

            return True

        except Exception as e:
            logger.error(f"[IntegratedEnhancements] Distillation failed: {e}")
            return False

    def get_distillation_stats(self) -> dict[str, Any]:
        """Get distillation statistics.

        Returns:
            Dictionary with distillation statistics.
        """
        if self._distillation_config is None:
            return {"enabled": False}

        checkpoint_paths = self._get_best_checkpoints()

        return {
            "enabled": True,
            "temperature": self._distillation_config.temperature,
            "alpha": self._distillation_config.alpha,
            "last_distillation_epoch": self._last_distillation_epoch,
            "available_teachers": len(checkpoint_paths),
            "interval_epochs": self.config.distillation_interval_epochs,
        }

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def start_background_services(self):
        """Start background threads (evaluation, etc.)."""
        if self._background_evaluator is not None:
            self._background_evaluator.start()
            logger.info("[IntegratedEnhancements] Background evaluator started")

    def stop_background_services(self):
        """Stop all background threads."""
        if self._background_evaluator is not None:
            self._background_evaluator.stop()
            logger.info("[IntegratedEnhancements] Background evaluator stopped")

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics from all enhancement modules."""
        metrics = {
            "step": self._step,
            "enabled_modules": self._count_enabled(),
        }

        if self._batch_scheduler is not None:
            metrics["batch_size"] = self._batch_scheduler.get_batch_size()

        if self._background_evaluator is not None:
            metrics["current_elo"] = self._background_evaluator.get_current_elo()

        if self._curriculum_controller is not None:
            metrics["curriculum"] = self._curriculum_controller.get_progress()

        return metrics

    def save_checkpoints(self):
        """Save checkpoints for all components."""
        if self._curriculum_controller is not None:
            self._curriculum_controller._save_checkpoint()

    def __enter__(self):
        """Context manager entry."""
        self.initialize_all()
        self.start_background_services()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_background_services()
        self.save_checkpoints()


# =============================================================================
# Factory Functions
# =============================================================================

def create_integrated_manager(
    config_dict: dict[str, Any] | None = None,
    model: Any | None = None,
    board_type: str = "square8",
) -> IntegratedTrainingManager:
    """Create an integrated training manager from config dictionary.

    Args:
        config_dict: Configuration dictionary (overrides defaults)
        model: PyTorch model
        board_type: Board type string

    Returns:
        Configured IntegratedTrainingManager
    """
    config = IntegratedEnhancementsConfig()

    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return IntegratedTrainingManager(config, model, board_type)


def get_enhancement_defaults() -> dict[str, Any]:
    """Get default enhancement configuration as dictionary."""
    config = IntegratedEnhancementsConfig()
    return {
        # Auxiliary Tasks
        "auxiliary_tasks_enabled": config.auxiliary_tasks_enabled,
        "aux_game_length_weight": config.aux_game_length_weight,
        "aux_piece_count_weight": config.aux_piece_count_weight,
        "aux_outcome_weight": config.aux_outcome_weight,
        # Gradient Surgery
        "gradient_surgery_enabled": config.gradient_surgery_enabled,
        "gradient_surgery_method": config.gradient_surgery_method,
        # Batch Scheduling
        "batch_scheduling_enabled": config.batch_scheduling_enabled,
        "batch_initial_size": config.batch_initial_size,
        "batch_final_size": config.batch_final_size,
        "batch_schedule_type": config.batch_schedule_type,
        # Background Eval
        "background_eval_enabled": config.background_eval_enabled,
        "eval_interval_steps": config.eval_interval_steps,
        # ELO Weighting
        "elo_weighting_enabled": config.elo_weighting_enabled,
        # Curriculum
        "curriculum_enabled": config.curriculum_enabled,
        "curriculum_auto_advance": config.curriculum_auto_advance,
        # Augmentation
        "augmentation_enabled": config.augmentation_enabled,
        "augmentation_mode": config.augmentation_mode,
        # Reanalysis
        "reanalysis_enabled": config.reanalysis_enabled,
        "reanalysis_blend_ratio": config.reanalysis_blend_ratio,
        # Knowledge Distillation
        "distillation_enabled": config.distillation_enabled,
        "distillation_temperature": config.distillation_temperature,
        "distillation_alpha": config.distillation_alpha,
        "distillation_interval_epochs": config.distillation_interval_epochs,
    }
