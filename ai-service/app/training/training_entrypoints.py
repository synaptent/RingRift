"""Thin entrypoints that wrap :func:`app.training.train.train_model`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.training.config import TrainConfig
from app.training.train_config import FullTrainingConfig

logger = logging.getLogger(__name__)


def train_with_config(full_config: FullTrainingConfig) -> dict[str, Any]:
    """Train using a unified ``FullTrainingConfig`` object."""
    from app.training.train import train_model

    train_cfg = TrainConfig(
        board_type=full_config.board_type,
        num_players=full_config.num_players,
        epochs_per_iter=full_config.epochs,
        batch_size=full_config.batch_size,
        learning_rate=full_config.learning_rate,
    )

    return train_model(
        config=train_cfg,
        data_path=full_config.data.data_path,
        save_path=full_config.checkpoint.save_path,
        early_stopping_patience=full_config.early_stopping.patience,
        elo_early_stopping_patience=full_config.early_stopping.elo_patience,
        elo_min_improvement=full_config.early_stopping.elo_min_improvement,
        checkpoint_dir=full_config.checkpoint.checkpoint_dir,
        checkpoint_interval=full_config.checkpoint.checkpoint_interval,
        _save_all_epochs=full_config.checkpoint.save_all_epochs,
        resume_path=full_config.checkpoint.resume_path,
        init_weights_path=full_config.checkpoint.init_weights_path,
        init_weights_strict=full_config.checkpoint.init_weights_strict,
        enable_checkpoint_averaging=full_config.checkpoint.enable_checkpoint_averaging,
        num_checkpoints_to_average=full_config.checkpoint.num_checkpoints_to_average,
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
        distributed=full_config.distributed.distributed,
        local_rank=full_config.distributed.local_rank,
        scale_lr=full_config.distributed.scale_lr,
        lr_scale_mode=full_config.distributed.lr_scale_mode,
        find_unused_parameters=full_config.distributed.find_unused_parameters,
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
        model_version=full_config.model.model_version,
        model_type=full_config.model.model_type,
        num_res_blocks=full_config.model.num_res_blocks,
        num_filters=full_config.model.num_filters,
        dropout=full_config.model.dropout,
        freeze_policy=full_config.model.freeze_policy,
        spectral_norm=full_config.model.spectral_norm,
        stochastic_depth=full_config.model.stochastic_depth,
        stochastic_depth_prob=full_config.model.stochastic_depth_prob,
        multi_player=full_config.multi_player,
        num_players=full_config.num_players,
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
        enable_circuit_breaker=full_config.fault_tolerance.enable_circuit_breaker,
        enable_anomaly_detection=full_config.fault_tolerance.enable_anomaly_detection,
        gradient_clip_mode=full_config.fault_tolerance.gradient_clip_mode,
        gradient_clip_max_norm=full_config.fault_tolerance.gradient_clip_max_norm,
        anomaly_spike_threshold=full_config.fault_tolerance.anomaly_spike_threshold,
        anomaly_gradient_threshold=full_config.fault_tolerance.anomaly_gradient_threshold,
        enable_graceful_shutdown=full_config.fault_tolerance.enable_graceful_shutdown,
        mixed_precision=full_config.mixed_precision.enabled,
        amp_dtype=full_config.mixed_precision.amp_dtype,
        augment_hex_symmetry=full_config.augmentation.augment_hex_symmetry,
        policy_label_smoothing=full_config.augmentation.policy_label_smoothing,
        heartbeat_file=full_config.heartbeat.heartbeat_file,
        heartbeat_interval=full_config.heartbeat.heartbeat_interval,
        value_whitening=full_config.value_whitening,
        value_whitening_momentum=full_config.value_whitening_momentum,
        ema=full_config.ema,
        ema_decay=full_config.ema_decay,
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
    """Simplified curriculum-training wrapper around ``train_model``."""
    from app.training.train import train_model

    if config is None:
        config = TrainConfig()

    checkpoint_dir = Path(output_path).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = train_model(
            config=config,
            data_path=data_path,
            save_path=output_path,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_interval=config.epochs_per_iter,
            resume_path=initial_model_path,
        )
        final_loss = result.get("best_val_loss", 0.0) if result else 0.0
        return {
            "total": final_loss,
            "policy": final_loss * config.policy_weight,
            "value": final_loss * (1 - config.policy_weight),
            "epochs_completed": result.get("epochs_completed", 0) if result else 0,
            "epoch_losses": result.get("epoch_losses", []) if result else [],
        }
    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as exc:
        logger.error("Training failed: %s", exc)
        return {
            "total": float("inf"),
            "policy": float("inf"),
            "value": float("inf"),
            "epochs_completed": 0,
            "epoch_losses": [],
        }


__all__ = ["train_from_file", "train_with_config"]
