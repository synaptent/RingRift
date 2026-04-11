"""Startup helpers for non-core training runtime services."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingRuntimeSetup:
    """Auxiliary services and mixed-precision state for a training run."""

    hot_buffer: Any | None
    enhancements_manager: Any | None
    checkpoint_averager: Any | None
    hard_example_miner: Any | None
    training_facade: Any | None
    quality_trainer: Any | None
    amp_enabled: bool
    amp_torch_dtype: torch.dtype
    use_grad_scaler: bool
    scaler: Any
    gradient_surgeon: Any | None
    use_gradient_surgery: bool
    metrics_collector: Any | None


def initialize_training_runtime_setup(
    *,
    config: Any,
    device: torch.device,
    checkpoint_dir: str,
    distributed: bool,
    is_main: bool,
    spectral_norm: bool,
    cyclic_lr: bool,
    cyclic_lr_period: int,
    mixed_precision: bool,
    amp_dtype: str,
    value_whitening: bool,
    ema: bool,
    ema_decay: float,
    stochastic_depth: bool,
    stochastic_depth_prob: float,
    adaptive_warmup: bool,
    hard_example_mining: bool,
    hard_example_top_k: float,
    use_hot_data_buffer: bool,
    hot_buffer_size: int,
    hot_buffer_mix_ratio: float,
    external_hot_buffer: Any | None,
    use_integrated_enhancements: bool,
    enable_curriculum: bool,
    enable_augmentation: bool,
    enable_elo_weighting: bool,
    enable_auxiliary_tasks: bool,
    enable_batch_scheduling: bool,
    enable_background_eval: bool,
    enable_checkpoint_averaging: bool,
    num_checkpoints_to_average: int,
    enable_quality_weighting: bool,
    quality_weight_blend: float,
    quality_ranking_weight: float,
    has_hot_data_buffer: bool,
    hot_data_buffer_cls: Any,
    has_quality_bridge: bool,
    get_quality_bridge: Any,
    has_integrated_enhancements: bool,
    integrated_enhancements_config_cls: Any,
    integrated_training_manager_cls: Any,
    checkpoint_averager_cls: Any,
    has_hard_example_mining: bool,
    hard_example_miner_cls: Any,
    has_training_facade: bool,
    training_facade_cls: Any,
    facade_config_cls: Any,
    has_quality_weighting: bool,
    quality_weighted_trainer_cls: Any,
    gradient_surgeon_cls: Any,
    gradient_surgery_config_cls: Any,
    has_metrics_collector: bool,
    metrics_collector_cls: Any,
) -> TrainingRuntimeSetup:
    """Create runtime helpers that are orthogonal to the core training loop."""
    improvements_enabled: list[str] = []
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
        logger.info("2024-12 Training Improvements enabled: %s", ", ".join(improvements_enabled))

    hot_buffer = None
    if external_hot_buffer is not None:
        hot_buffer = external_hot_buffer
        current_samples = getattr(hot_buffer, "total_samples", 0)
        logger.info(
            "Using external hot data buffer with %s samples (mix_ratio=%s)",
            current_samples,
            hot_buffer_mix_ratio,
        )
    elif use_hot_data_buffer and has_hot_data_buffer:
        hot_buffer = hot_data_buffer_cls(
            max_size=hot_buffer_size,
            training_threshold=config.batch_size * 5,
        )
        logger.info(
            "Hot data buffer enabled (size=%s, mix_ratio=%s)",
            hot_buffer_size,
            hot_buffer_mix_ratio,
        )
        logger.info(
            "Note: Hot buffer requires external game population via add_game() "
            "or event bus subscription to receive selfplay games"
        )
    elif use_hot_data_buffer and not has_hot_data_buffer:
        logger.warning("Hot data buffer requested but not available (import failed)")

    if has_quality_bridge:
        try:
            quality_bridge = get_quality_bridge()
            num_refreshed = quality_bridge.refresh(force=True)
            logger.info("Quality bridge initialized with %s game quality scores", num_refreshed)
            if hot_buffer is not None:
                configured = quality_bridge.configure_hot_data_buffer(hot_buffer)
                if configured > 0:
                    logger.info("Hot buffer configured with %s quality scores", configured)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to initialize quality bridge: %s", exc)

    enhancements_manager = None
    if use_integrated_enhancements and has_integrated_enhancements:
        enh_config = integrated_enhancements_config_cls(
            curriculum_enabled=enable_curriculum,
            augmentation_enabled=enable_augmentation,
            elo_weighting_enabled=enable_elo_weighting,
            auxiliary_tasks_enabled=enable_auxiliary_tasks,
            batch_scheduling_enabled=enable_batch_scheduling,
            background_eval_enabled=enable_background_eval,
            eval_use_real_games=enable_background_eval,
            eval_board_type=config.board_type,
        )
        enhancements_manager = integrated_training_manager_cls(
            config=enh_config,
            model=None,
            board_type=config.board_type.value,
        )
        logger.info(
            "Integrated enhancements enabled: curriculum=%s, augmentation=%s, "
            "elo_weighting=%s, auxiliary_tasks=%s, batch_scheduling=%s, background_eval=%s",
            enable_curriculum,
            enable_augmentation,
            enable_elo_weighting,
            enable_auxiliary_tasks,
            enable_batch_scheduling,
            enable_background_eval,
        )
    elif use_integrated_enhancements and not has_integrated_enhancements:
        logger.warning("Integrated enhancements requested but not available (import failed)")

    checkpoint_averager = None
    if enable_checkpoint_averaging and checkpoint_averager_cls is not None:
        checkpoint_averager = checkpoint_averager_cls(
            num_checkpoints=num_checkpoints_to_average,
            checkpoint_dir=Path(checkpoint_dir),
            keep_on_disk=True,
        )
        logger.info(
            "[Checkpoint Averaging] Enabled: will average last %s checkpoints at end of training",
            num_checkpoints_to_average,
        )
    elif enable_checkpoint_averaging and checkpoint_averager_cls is None:
        logger.warning("[Checkpoint Averaging] Requested but CheckpointAverager not available (import failed)")

    hard_example_miner = None
    if hard_example_mining and has_hard_example_mining:
        hard_example_miner = hard_example_miner_cls(
            buffer_size=50000,
            hard_fraction=hard_example_top_k,
            loss_threshold_percentile=80.0,
            uncertainty_weight=0.3,
            decay_rate=0.99,
            min_samples_before_mining=5000,
            max_times_sampled=10,
        )
        logger.info(
            "[Hard Example Mining] Enabled: hard_fraction=%s, buffer_size=50000, min_samples_before_mining=5000",
            hard_example_top_k,
        )
    elif hard_example_mining and not has_hard_example_mining:
        logger.warning("[Hard Example Mining] Requested but HardExampleMiner not available (import failed)")

    training_facade = None
    if has_training_facade and training_facade_cls is not None:
        facade_config = facade_config_cls(
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
        training_facade = training_facade_cls(config=facade_config)
        training_facade.set_total_epochs(config.epochs_per_iter)
        logger.info(
            "[Training Facade] Enabled: hard_mining=%s, curriculum_lr=%s, freshness=%s",
            hard_example_mining,
            enable_curriculum,
            enable_elo_weighting,
        )
    elif hard_example_mining and not has_training_facade:
        logger.info("[Training Facade] Not available, falling back to standalone HardExampleMiner")

    quality_trainer = None
    if enable_quality_weighting and has_quality_weighting:
        quality_trainer = quality_weighted_trainer_cls(
            quality_weight=quality_weight_blend,
            ranking_weight=quality_ranking_weight,
            ranking_margin=0.5,
            min_quality_weight=0.1,
            temperature=1.0,
        )
        logger.info(
            "[Quality Weighting] Enabled: blend=%.2f, ranking_weight=%.2f",
            quality_weight_blend,
            quality_ranking_weight,
        )
    elif enable_quality_weighting and not has_quality_weighting:
        logger.warning("[Quality Weighting] Requested but module not available")

    amp_enabled = bool(mixed_precision and device.type == "cuda")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    amp_torch_dtype = dtype_map.get(amp_dtype, torch.bfloat16)
    use_grad_scaler = bool(amp_enabled and amp_torch_dtype == torch.float16)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    if amp_enabled:
        logger.info("Mixed precision training enabled with %s", amp_dtype)

    gradient_surgeon = None
    use_gradient_surgery = bool(getattr(config, "enable_gradient_surgery", False))
    if use_gradient_surgery:
        if use_grad_scaler:
            logger.warning(
                "Gradient surgery disabled: incompatible with FP16 GradScaler. "
                "Use bfloat16 mixed precision or disable mixed precision."
            )
            use_gradient_surgery = False
        elif getattr(config, "gradient_accumulation_steps", 1) > 1:
            logger.warning(
                "Gradient surgery disabled: incompatible with gradient accumulation. "
                "Set gradient_accumulation_steps=1 to use gradient surgery."
            )
            use_gradient_surgery = False
        else:
            gradient_surgeon = gradient_surgeon_cls(
                gradient_surgery_config_cls(
                    enabled=True,
                    method="pcgrad",
                    conflict_threshold=0.0,
                )
            )
            logger.info("Gradient surgery (PCGrad) enabled for multi-task learning")

    metrics_collector = None
    if has_metrics_collector and is_main:
        try:
            metrics_collector = metrics_collector_cls()
            logger.info("Dashboard metrics collector initialized")
        except (ImportError, RuntimeError, OSError) as exc:
            logger.warning("Could not initialize metrics collector: %s", exc)

    return TrainingRuntimeSetup(
        hot_buffer=hot_buffer,
        enhancements_manager=enhancements_manager,
        checkpoint_averager=checkpoint_averager,
        hard_example_miner=hard_example_miner,
        training_facade=training_facade,
        quality_trainer=quality_trainer,
        amp_enabled=amp_enabled,
        amp_torch_dtype=amp_torch_dtype,
        use_grad_scaler=use_grad_scaler,
        scaler=scaler,
        gradient_surgeon=gradient_surgeon,
        use_gradient_surgery=use_gradient_surgery,
        metrics_collector=metrics_collector,
    )


__all__ = ["TrainingRuntimeSetup", "initialize_training_runtime_setup"]
