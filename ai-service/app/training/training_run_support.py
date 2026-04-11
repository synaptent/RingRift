"""Helpers for initializing per-run training state."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingRunSupport:
    """State and helper services initialized once before the epoch loop."""

    dist_metrics: Any
    heartbeat_monitor: Any | None
    best_val_loss: float
    best_train_loss_at_best_val: float
    avg_val_loss: float
    avg_train_loss: float
    avg_policy_accuracy: float | None
    epoch_losses: list[dict[str, float]] = field(default_factory=list)
    epochs_completed: int = 0
    training_completed_normally: bool = False
    training_exception: Exception | None = None
    training_start_time: float = 0.0
    final_checkpoint_path: str | None = None
    total_samples: int = 0
    num_data_files: int = 0
    config_label: str = ""
    loss_monitor: Any | None = None
    training_breaker: Any | None = None
    anomaly_detector: Any | None = None
    adaptive_clipper: Any | None = None
    fixed_clip_norm: float | None = None
    gradient_clip_mode: str = "adaptive"
    anomaly_step: int = 0
    training_state: Any | None = None
    shutdown_handler: Any | None = None
    rollback_handler: Any | None = None
    last_good_checkpoint_path: str | None = None
    last_good_epoch: int = 0
    circuit_breaker_rollbacks: int = 0
    max_circuit_breaker_rollbacks: int = 3


def initialize_training_run_support(
    *,
    config: Any,
    num_players: int,
    batch_size_metric: Any,
    has_prometheus: bool,
    distributed: bool,
    is_main: bool,
    heartbeat_file: str | None,
    heartbeat_interval: float,
    start_epoch: int,
    checkpoint_dir: str,
    enable_graceful_shutdown: bool,
    enable_circuit_breaker: bool,
    enable_anomaly_detection: bool,
    gradient_clip_mode: str,
    gradient_clip_max_norm: float,
    anomaly_spike_threshold: float,
    anomaly_gradient_threshold: float,
    model: nn.Module,
    optimizer: Any,
    epoch_scheduler: Any,
    early_stopper: Any,
    enhancements_manager: Any,
    distributed_metrics_cls: Any,
    heartbeat_monitor_cls: Any,
    loss_monitor_cls: Any,
    fault_tolerance_config_cls: Any,
    setup_fault_tolerance_fn: Any,
    training_state_cls: Any,
    graceful_shutdown_handler_cls: Any,
    save_checkpoint_fn: Any,
    has_event_bus: bool,
    get_router_fn: Any,
    data_event_cls: Any,
    data_event_type: Any,
    time_module: Any,
) -> TrainingRunSupport:
    """Initialize monitors, fault-tolerance helpers, and per-run counters."""
    dist_metrics = distributed_metrics_cls() if distributed else None

    heartbeat_monitor = None
    if heartbeat_file and is_main:
        heartbeat_path = Path(heartbeat_file)
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_monitor = heartbeat_monitor_cls(
            heartbeat_interval=heartbeat_interval,
            timeout_threshold=heartbeat_interval * 4,
        )
        heartbeat_monitor.start(heartbeat_path)
        logger.info(
            "Heartbeat monitor started: %s (interval=%ss)",
            heartbeat_file,
            heartbeat_interval,
        )

    config_label = f"{config.board_type.value}_{num_players}p"
    loss_monitor = loss_monitor_cls(patience=5, config_key=config_label)

    if has_prometheus and batch_size_metric is not None and is_main:
        batch_size_metric.labels(config=config_label).set(config.batch_size)

    if enhancements_manager is not None:
        enhancements_manager.start_background_services()
        logger.info("Integrated enhancements background services started")

    ft_config = fault_tolerance_config_cls(
        enable_circuit_breaker=enable_circuit_breaker,
        enable_anomaly_detection=enable_anomaly_detection,
        enable_graceful_shutdown=enable_graceful_shutdown,
        gradient_clip_mode=gradient_clip_mode,
        gradient_clip_max_norm=gradient_clip_max_norm,
        anomaly_spike_threshold=anomaly_spike_threshold,
        anomaly_gradient_threshold=anomaly_gradient_threshold,
    )
    ft_components = setup_fault_tolerance_fn(
        ft_config,
        distributed=distributed,
        is_main_process_fn=(lambda: is_main) if distributed else None,
    )

    training_state = training_state_cls(
        epoch=start_epoch,
        best_val_loss=float("inf"),
        avg_val_loss=float("inf"),
    )

    shutdown_handler = None
    if enable_graceful_shutdown and is_main:

        def _emergency_checkpoint_callback() -> None:
            model_to_save = model.module if distributed else model
            emergency_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_emergency_epoch_{training_state.epoch}.pth",
            )
            save_checkpoint_fn(
                model_to_save,
                optimizer,
                training_state.epoch,
                training_state.avg_val_loss,
                emergency_path,
                scheduler=epoch_scheduler,
                early_stopping=early_stopper,
            )

        shutdown_handler = graceful_shutdown_handler_cls()
        shutdown_handler.setup(_emergency_checkpoint_callback)

    rollback_handler = None
    try:
        from app.training.model_registry import get_model_registry
        from app.training.rollback_manager import wire_regression_to_rollback

        rollback_handler = wire_regression_to_rollback(
            registry=get_model_registry(),
            auto_rollback_enabled=True,
            require_approval_for_severe=True,
            subscribe_to_events=True,
        )
        if is_main:
            logger.info("[train_model] Regression → rollback wiring activated")
    except ImportError:
        pass
    except (AttributeError, TypeError, RuntimeError) as exc:
        if is_main:
            logger.debug("[train_model] Rollback wiring not available: %s", exc)

    if has_event_bus and get_router_fn is not None and data_event_type is not None and is_main:
        try:
            router = get_router_fn()
            model_path = Path(config.model_dir) / f"model_{num_players}p.pth"
            router.publish_sync(
                data_event_cls(
                    event_type=data_event_type.TRAINING_STARTED,
                    payload={
                        "total_epochs": config.epochs_per_iter,
                        "start_epoch": start_epoch,
                        "config": config_label,
                        "model_path": str(model_path),
                    },
                    source="train",
                )
            )
        except (RuntimeError, ConnectionError, TimeoutError, TypeError) as exc:
            logger.debug("Failed to publish training started event: %s", exc)

    return TrainingRunSupport(
        dist_metrics=dist_metrics,
        heartbeat_monitor=heartbeat_monitor,
        best_val_loss=float("inf"),
        best_train_loss_at_best_val=float("inf"),
        avg_val_loss=float("inf"),
        avg_train_loss=float("inf"),
        avg_policy_accuracy=None,
        training_completed_normally=False,
        training_exception=None,
        training_start_time=time_module.time(),
        final_checkpoint_path=None,
        total_samples=0,
        num_data_files=0,
        config_label=config_label,
        loss_monitor=loss_monitor,
        training_breaker=ft_components.training_breaker,
        anomaly_detector=ft_components.anomaly_detector,
        adaptive_clipper=ft_components.adaptive_clipper,
        fixed_clip_norm=ft_components.fixed_clip_norm,
        gradient_clip_mode=ft_components.gradient_clip_mode,
        anomaly_step=0,
        training_state=training_state,
        shutdown_handler=shutdown_handler,
        rollback_handler=rollback_handler,
        last_good_checkpoint_path=training_state.last_good_checkpoint_path,
        last_good_epoch=training_state.last_good_epoch,
        circuit_breaker_rollbacks=training_state.circuit_breaker_rollbacks,
        max_circuit_breaker_rollbacks=training_state.max_circuit_breaker_rollbacks,
    )


def maybe_run_lr_finder(
    *,
    find_lr: bool,
    is_main: bool,
    distributed: bool,
    model: nn.Module,
    optimizer: Any,
    train_loader: Any,
    device: Any,
    lr_finder_min: float,
    lr_finder_max: float,
    lr_finder_iterations: int,
) -> None:
    """Run the optional LR finder and update the optimizer in place."""
    if not find_lr or not is_main:
        return

    try:
        from app.training.advanced_training import LRFinder

        logger.info(
            "[LR Finder] Running learning rate range test (min=%.1e, max=%.1e, iters=%s)",
            lr_finder_min,
            lr_finder_max,
            lr_finder_iterations,
        )

        def combined_criterion(outputs: Any, targets: Any) -> Any:
            if isinstance(outputs, tuple):
                value_out, policy_out = outputs[:2]
                if isinstance(targets, tuple):
                    value_target, policy_target = targets[:2]
                else:
                    value_target = targets
                    policy_target = None
                import torch.nn.functional as F

                value_loss = F.mse_loss(value_out.squeeze(), value_target.squeeze())
                if policy_target is not None:
                    policy_loss = F.cross_entropy(policy_out, policy_target)
                    return value_loss + policy_loss
                return value_loss
            import torch.nn.functional as F

            return F.mse_loss(outputs, targets)

        lr_finder = LRFinder(
            model=model.module if distributed else model,
            optimizer=optimizer,
            criterion=combined_criterion,
            device=device,
        )
        lr_result = lr_finder.range_test(
            train_loader,
            min_lr=lr_finder_min,
            max_lr=lr_finder_max,
            num_iter=lr_finder_iterations,
        )
        logger.info(
            "[LR Finder] Results: suggested_lr=%.2e, steepest_lr=%.2e, best_lr=%.2e",
            lr_result.suggested_lr,
            lr_result.steepest_lr,
            lr_result.best_lr,
        )
        old_lr = optimizer.param_groups[0]["lr"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_result.suggested_lr
        logger.info(
            "[LR Finder] Updated learning rate: %.2e -> %.2e",
            old_lr,
            lr_result.suggested_lr,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("[LR Finder] Failed: %s. Continuing with configured LR.", exc)


__all__ = [
    "TrainingRunSupport",
    "initialize_training_run_support",
    "maybe_run_lr_finder",
]
