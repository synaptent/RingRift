"""Integrated Training Enhancements for RingRift AI.

Unified module that consolidates and integrates all advanced training features:
- Auxiliary Tasks (game length, piece count, outcome prediction)
- Gradient Surgery (PCGrad for multi-task learning)
- Batch Scheduling (dynamic batch sizing)
- Background Evaluation (continuous Elo tracking)
- ELO Weighting (opponent strength-based sampling)
- Curriculum Learning (progressive difficulty)
- Data Augmentation (board symmetry)
- Reanalysis (historical game re-evaluation)

This module provides a single entry point for enabling/configuring all features.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for PyTorch to prevent OOM in orchestrator
_torch = None
_nn = None


def _get_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
    return _torch, _nn


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
    # =========================================================================
    auxiliary_tasks_enabled: bool = False
    aux_game_length_weight: float = 0.1
    aux_piece_count_weight: float = 0.1
    aux_outcome_weight: float = 0.05

    # =========================================================================
    # Gradient Surgery (PCGrad)
    # =========================================================================
    gradient_surgery_enabled: bool = False
    gradient_surgery_method: str = "pcgrad"  # "pcgrad" or "cagrad"
    gradient_conflict_threshold: float = 0.0

    # =========================================================================
    # Batch Scheduling
    # =========================================================================
    batch_scheduling_enabled: bool = False
    batch_initial_size: int = 64
    batch_final_size: int = 512
    batch_warmup_steps: int = 1000
    batch_rampup_steps: int = 10000
    batch_schedule_type: str = "linear"  # "linear", "exponential", "step"

    # =========================================================================
    # Background Evaluation
    # =========================================================================
    background_eval_enabled: bool = False
    eval_interval_steps: int = 1000
    eval_games_per_check: int = 20
    eval_elo_checkpoint_threshold: float = 10.0
    eval_elo_drop_threshold: float = 50.0
    eval_auto_checkpoint: bool = True
    eval_checkpoint_dir: str = "data/eval_checkpoints"

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
    # =========================================================================
    reanalysis_enabled: bool = False
    reanalysis_blend_ratio: float = 0.5
    reanalysis_interval_steps: int = 5000
    reanalysis_batch_size: int = 1000


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
        config: Optional[IntegratedEnhancementsConfig] = None,
        model: Optional[Any] = None,
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

        # Metrics tracking
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

        logger.info("[IntegratedEnhancements] Manager initialized")

    def initialize_all(self, model: Optional[Any] = None):
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
                BatchSizeScheduler,
                BatchScheduleConfig,
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
                return self.model

            eval_config = EvalConfig(
                eval_interval_steps=self.config.eval_interval_steps,
                games_per_eval=self.config.eval_games_per_check,
                elo_checkpoint_threshold=self.config.eval_elo_checkpoint_threshold,
                elo_drop_threshold=self.config.eval_elo_drop_threshold,
                auto_checkpoint=self.config.eval_auto_checkpoint,
                checkpoint_dir=self.config.eval_checkpoint_dir,
            )
            self._background_evaluator = BackgroundEvaluator(model_getter, eval_config)
            logger.info("[IntegratedEnhancements] Background evaluator initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init background eval: {e}")

    def _init_elo_weighting(self):
        """Initialize ELO-weighted sampling."""
        try:
            from app.training.elo_weighting import (
                EloWeightConfig,
                compute_elo_weights,
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
                DataAugmentor,
                AugmentationConfig,
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
                ReanalysisEngine,
                ReanalysisConfig,
            )

            if self.model is None:
                logger.warning("[IntegratedEnhancements] Model required for reanalysis")
                return

            reanalysis_config = ReanalysisConfig(
                value_blend_ratio=self.config.reanalysis_blend_ratio,
                batch_size=self.config.reanalysis_batch_size,
            )
            self._reanalysis_engine = ReanalysisEngine(self.model, reanalysis_config)
            logger.info("[IntegratedEnhancements] Reanalysis engine initialized")
        except Exception as e:
            logger.warning(f"[IntegratedEnhancements] Failed to init reanalysis: {e}")

    # =========================================================================
    # Training Step Integration
    # =========================================================================

    def get_batch_size(self) -> int:
        """Get current batch size from scheduler."""
        if self._batch_scheduler is not None:
            return self._batch_scheduler.get_batch_size(self._step)
        return self.config.batch_initial_size

    def get_curriculum_parameters(self) -> Dict[str, Any]:
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
        policy_indices: List[np.ndarray],
        policy_values: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
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

    def compute_auxiliary_loss(
        self,
        features: Any,  # torch.Tensor
        targets: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, float]]:
        """Compute auxiliary task losses.

        Args:
            features: Shared features from backbone
            targets: Dictionary of auxiliary targets

        Returns:
            (total_aux_loss, loss_breakdown)
        """
        if self._auxiliary_module is not None:
            torch, _ = _get_torch()
            predictions = self._auxiliary_module(features)
            return self._auxiliary_module.compute_loss(predictions, targets)

        torch, _ = _get_torch()
        return torch.tensor(0.0), {}

    def apply_gradient_surgery(
        self,
        model: Any,  # nn.Module
        losses: Dict[str, Any],
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

    def update_step(self, game_won: Optional[bool] = None):
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

    def get_metrics(self) -> Dict[str, Any]:
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
    config_dict: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
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


def get_enhancement_defaults() -> Dict[str, Any]:
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
    }
