"""Epoch-level reporting and observability helpers for training."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class EpochReportingResult:
    """Outputs produced by epoch reporting hooks."""

    skip_checkpoint_on_regression: bool
    epochs_completed: int
    epoch_record: dict[str, Any]


def handle_epoch_reporting_and_feedback(
    *,
    epoch: int,
    avg_train_loss: float,
    avg_val_loss: float,
    avg_policy_accuracy: float | None,
    optimizer: Any,
    best_val_loss: float,
    config: Any,
    num_players: int,
    distributed: bool,
    is_main: bool,
    device: torch.device,
    calibration_tracker: Any,
    epoch_losses: list[dict[str, Any]],
    loss_monitor: Any,
    training_facade: Any,
    hard_example_miner: Any,
    metrics_collector: Any,
    has_regression_detector: bool,
    get_regression_detector: Any,
    regression_severity: Any,
    has_epoch_events: bool,
    publish_epoch_completed: Any,
    has_training_events: bool,
    emit_training_loss_anomaly: Any,
    emit_training_loss_trend: Any,
    has_prometheus: bool,
    training_epochs_metric: Any,
    training_loss_metric: Any,
    training_duration_metric: Any,
    calibration_ece_metric: Any,
    calibration_mce_metric: Any,
) -> EpochReportingResult:
    """Run epoch-complete logging, anomaly detection, and metric emission."""
    if not loss_monitor.record(epoch, avg_train_loss, avg_val_loss):
        logger.warning(
            "[LossMonitor] Training stalled - loss not improving. "
            "Consider checking data quality or model architecture. Summary: %s",
            loss_monitor.get_summary(),
        )

    if avg_train_loss > 0 and epoch >= 3:
        overfitting_ratio = (avg_val_loss - avg_train_loss) / avg_train_loss
        if overfitting_ratio > 0.25 and is_main:
            logger.warning(
                "Overfitting detected: %.1f%% divergence (train=%.4f, val=%.4f)",
                overfitting_ratio * 100,
                avg_train_loss,
                avg_val_loss,
            )

    skip_checkpoint_on_regression = False
    if has_regression_detector and get_regression_detector is not None and epoch >= 2 and is_main:
        try:
            regression_detector = get_regression_detector(connect_event_bus=True)
            model_id = f"{config.board_type.value}_{num_players}p"
            if epoch == 2:
                regression_detector.set_baseline(
                    model_id=model_id,
                    elo=best_val_loss * -1000,
                )

            regression_event = regression_detector.check_regression(
                model_id=model_id,
                current_elo=avg_val_loss * -1000,
                games_played=epoch + 1,
            )
            if regression_event is not None:
                logger.warning(
                    "[RegressionDetector] %s regression: val_loss %.4f vs best %.4f (%s)",
                    regression_event.severity.value.upper(),
                    avg_val_loss,
                    best_val_loss,
                    regression_event.reason,
                )
                if regression_event.severity in (
                    regression_severity.MODERATE,
                    regression_severity.SEVERE,
                    regression_severity.CRITICAL,
                ):
                    skip_checkpoint_on_regression = True
                    logger.warning(
                        "[RegressionDetector] Skipping checkpoint save due to %s regression",
                        regression_event.severity.value,
                    )
        except (AttributeError, ValueError, TypeError, ImportError) as exc:
            logger.debug("Regression detection error: %s", exc)

    epochs_completed = epoch + 1
    epoch_record: dict[str, Any] = {
        "epoch": epoch + 1,
        "train_loss": float(avg_train_loss),
        "val_loss": float(avg_val_loss),
        "policy_accuracy": float(avg_policy_accuracy),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }

    if calibration_tracker is not None and (epoch + 1) % 5 == 0:
        calibration_report = calibration_tracker.compute_current_calibration()
        if calibration_report is not None:
            epoch_record["calibration_ece"] = calibration_report.ece
            epoch_record["calibration_mce"] = calibration_report.mce
            epoch_record["calibration_overconfidence"] = calibration_report.overconfidence
            if is_main:
                logger.info(
                    "  Calibration: ECE=%.4f, MCE=%.4f, Overconfidence=%.4f",
                    calibration_report.ece,
                    calibration_report.mce,
                    calibration_report.overconfidence,
                )
                if calibration_report.optimal_temperature is not None:
                    logger.info(
                        "  Optimal temperature: %.3f",
                        calibration_report.optimal_temperature,
                    )

    epoch_losses.append(epoch_record)

    if training_facade is not None and is_main:
        try:
            facade_stats = training_facade.on_epoch_end()
            if facade_stats.get("mining_active", False):
                logger.info(
                    "  [Training Facade] tracked=%s, hard_frac=%.1f%%, mean_loss=%.4f, lr_scale=%.3f",
                    facade_stats.get("tracked_samples", 0),
                    facade_stats.get("hard_examples_fraction", 0) * 100,
                    facade_stats.get("mean_per_sample_loss", 0),
                    facade_stats.get("curriculum_lr_scale", 1.0),
                )
            epoch_record["facade_mean_loss"] = facade_stats.get("mean_loss", 0)
            epoch_record["facade_hard_fraction"] = facade_stats.get("hard_examples_fraction", 0)
            epoch_record["facade_curriculum_lr_scale"] = facade_stats.get("curriculum_lr_scale", 1.0)
            epoch_record["facade_mining_active"] = facade_stats.get("mining_active", False)
        except (AttributeError, ValueError) as exc:
            logger.debug("[Training Facade] on_epoch_end error: %s", exc)
    elif hard_example_miner is not None and is_main:
        mining_stats = hard_example_miner.get_statistics()
        if mining_stats.get("mining_active", False):
            logger.info(
                "  [Hard Example Mining] tracked=%s, mean_loss=%.4f, loss_p90=%.4f",
                mining_stats.get("tracked_examples", 0),
                mining_stats.get("mean_loss", 0),
                mining_stats.get("loss_p90", 0),
            )
            epoch_record["hard_mining_mean_loss"] = mining_stats.get("mean_loss", 0)
            epoch_record["hard_mining_p90_loss"] = mining_stats.get("loss_p90", 0)
            epoch_record["hard_mining_tracked"] = mining_stats.get("tracked_examples", 0)

    if has_epoch_events and publish_epoch_completed and is_main:
        try:
            config_key = f"{config.board_type.value}_{num_players}p"
            try:
                asyncio.get_running_loop()
                asyncio.ensure_future(
                    publish_epoch_completed(
                        config_key=config_key,
                        epoch=epoch + 1,
                        total_epochs=config.epochs_per_iter,
                        train_loss=avg_train_loss,
                        val_loss=avg_val_loss,
                        learning_rate=optimizer.param_groups[0]["lr"],
                    )
                )
            except RuntimeError:
                pass
        except (RuntimeError, ConnectionError, TimeoutError) as exc:
            logger.debug("Failed to emit epoch completed event: %s", exc)

    if has_training_events and is_main:
        try:
            config_key = f"{config.board_type.value}_{num_players}p"
            recent_losses = [
                entry.get("avg_val_loss", entry.get("avg_train_loss", 0.0))
                for entry in epoch_losses[-5:]
                if entry
            ]
            if recent_losses:
                avg_recent_loss = sum(recent_losses) / len(recent_losses)
                if avg_val_loss > avg_recent_loss * 2.0 and len(epoch_losses) > 2:
                    anomaly_ratio = avg_val_loss / avg_recent_loss if avg_recent_loss > 0 else 0.0
                    logger.warning(
                        "[TRAINING ANOMALY] Loss spike detected: %.4f vs avg %.4f (ratio: %.2fx)",
                        avg_val_loss,
                        avg_recent_loss,
                        anomaly_ratio,
                    )
                    try:
                        asyncio.get_running_loop()
                        asyncio.ensure_future(
                            emit_training_loss_anomaly(
                                config_key=config_key,
                                current_loss=avg_val_loss,
                                avg_loss=avg_recent_loss,
                                epoch=epoch + 1,
                                anomaly_ratio=anomaly_ratio,
                                source="train.py",
                            )
                        )
                    except RuntimeError:
                        pass

                if (epoch + 1) % 5 == 0 and len(epoch_losses) >= 5:
                    current_avg = sum(recent_losses) / len(recent_losses)
                    older_losses = [
                        entry.get("avg_val_loss", entry.get("avg_train_loss", 0.0))
                        for entry in epoch_losses[-10:-5]
                        if entry
                    ]
                    if older_losses:
                        previous_avg = sum(older_losses) / len(older_losses)
                        improvement_rate = (previous_avg - current_avg) / previous_avg if previous_avg > 0 else 0.0
                        if improvement_rate > 0.05:
                            trend = "improving"
                        elif improvement_rate < -0.05:
                            trend = "degrading"
                        else:
                            trend = "stalled"

                        logger.info(
                            "[TRAINING TREND] %s (epoch %s): current_avg=%.4f, previous_avg=%.4f, improvement_rate=%.2f%%",
                            trend,
                            epoch + 1,
                            current_avg,
                            previous_avg,
                            improvement_rate * 100,
                        )
                        try:
                            asyncio.get_running_loop()
                            asyncio.ensure_future(
                                emit_training_loss_trend(
                                    config_key=config_key,
                                    trend=trend,
                                    epoch=epoch + 1,
                                    current_loss=current_avg,
                                    previous_loss=previous_avg,
                                    improvement_rate=improvement_rate,
                                    source="train.py",
                                )
                            )
                        except RuntimeError:
                            pass

                if (epoch + 1) % 10 == 0 and len(epoch_losses) >= 10:
                    last_10_losses = [
                        entry.get("avg_val_loss", entry.get("avg_train_loss", 0.0))
                        for entry in epoch_losses[-10:]
                        if entry
                    ]
                    prev_10_losses = [
                        entry.get("avg_val_loss", entry.get("avg_train_loss", 0.0))
                        for entry in epoch_losses[-20:-10]
                        if entry
                    ]
                    if len(last_10_losses) >= 10 and len(prev_10_losses) >= 5:
                        last_10_avg = sum(last_10_losses) / len(last_10_losses)
                        prev_10_avg = sum(prev_10_losses) / len(prev_10_losses)
                        long_term_improvement = (prev_10_avg - last_10_avg) / prev_10_avg if prev_10_avg > 0 else 0.0
                        if abs(long_term_improvement) < 0.001:
                            last_10_train = [entry.get("avg_train_loss", 0.0) for entry in epoch_losses[-10:] if entry]
                            last_10_train_avg = sum(last_10_train) / len(last_10_train) if last_10_train else 0.0
                            train_val_gap = last_10_avg - last_10_train_avg

                            if train_val_gap > 0.05:
                                plateau_type = "overfitting"
                                recommendation = "reduce_epochs"
                                exploration_boost = 1.5
                            else:
                                plateau_type = "data_limitation"
                                recommendation = "more_games"
                                exploration_boost = 1.3

                            logger.warning(
                                "[TRAINING PLATEAU] Detected at epoch %s: <0.1%% improvement over 10 epochs "
                                "(last_10=%.5f, prev_10=%.5f, type=%s, gap=%.4f)",
                                epoch + 1,
                                last_10_avg,
                                prev_10_avg,
                                plateau_type,
                                train_val_gap,
                            )
                            try:
                                asyncio.get_running_loop()
                                asyncio.ensure_future(
                                    emit_training_loss_trend(
                                        config_key=config_key,
                                        trend="plateau",
                                        epoch=epoch + 1,
                                        current_loss=last_10_avg,
                                        previous_loss=prev_10_avg,
                                        improvement_rate=long_term_improvement,
                                        source="train.py",
                                        window_size=10,
                                    )
                                )
                                from app.coordination.event_emission_helpers import safe_emit_event

                                safe_emit_event(
                                    "PLATEAU_DETECTED",
                                    {
                                        "metric_name": "validation_loss",
                                        "current_value": last_10_avg,
                                        "best_value": prev_10_avg,
                                        "epochs_since_improvement": 10,
                                        "plateau_type": plateau_type,
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
        except (RuntimeError, ConnectionError, TimeoutError, AttributeError) as exc:
            logger.debug("Failed to emit training events: %s", exc)

    if has_prometheus and is_main:
        config_label = f"{config.board_type.value}_{num_players}p"
        training_epochs_metric.labels(config=config_label).inc()
        training_loss_metric.labels(config=config_label, loss_type="train").set(avg_train_loss)
        training_loss_metric.labels(config=config_label, loss_type="val").set(avg_val_loss)
        training_duration_metric.labels(config=config_label).observe(
            epoch_record.get("epoch_duration", 0.0)
        )
        if "calibration_ece" in epoch_record:
            calibration_ece_metric.labels(config=config_label).set(epoch_record["calibration_ece"])
            calibration_mce_metric.labels(config=config_label).set(epoch_record["calibration_mce"])

    if metrics_collector is not None and is_main:
        try:
            gpu_memory_mb = 0.0
            if device.type == "cuda":
                gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

            metrics_collector.record_training_step(
                epoch=epoch + 1,
                step=epoch_record.get("train_batches", 0),
                loss=avg_val_loss,
                policy_loss=epoch_record.get("avg_policy_loss", 0.0),
                value_loss=epoch_record.get("avg_value_loss", 0.0),
                accuracy=avg_policy_accuracy,
                learning_rate=optimizer.param_groups[0]["lr"],
                batch_size=config.batch_size,
                samples_per_second=epoch_record.get("samples_per_second", 0.0),
                gpu_memory_mb=gpu_memory_mb,
                model_id=config.model_id,
            )
        except (OSError, RuntimeError, AttributeError) as exc:
            logger.debug("Failed to record metrics to dashboard: %s", exc)

    return EpochReportingResult(
        skip_checkpoint_on_regression=skip_checkpoint_on_regression,
        epochs_completed=epochs_completed,
        epoch_record=epoch_record,
    )


__all__ = ["EpochReportingResult", "handle_epoch_reporting_and_feedback"]
