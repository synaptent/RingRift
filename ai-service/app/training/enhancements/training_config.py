"""
Training Configuration for RingRift AI.

This module provides the consolidated TrainingConfig dataclass for all training
enhancements.

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    """
    Consolidated configuration for all training enhancements.

    This dataclass provides a single point of configuration for:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Data quality scoring
    - Hard example mining
    - Anomaly detection
    - Validation intervals
    - Seed management

    Usage:
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=256,
            use_mixed_precision=True,
            validation_interval_steps=500,
        )

        # Convert to dict for create_training_enhancements
        enhancements = create_training_enhancements(model, optimizer, config.to_dict())
    """

    # === Core Training ===
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    weight_decay: float = 0.0001
    seed: int | None = None

    # === Mixed Precision ===
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # "float16" or "bfloat16"

    # === Gradient Accumulation ===
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # === Learning Rate Schedule ===
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "warm_restarts"
    warmup_epochs: int = 5
    warmup_steps: int = 0
    min_lr: float = 1e-6
    max_lr: float = 0.01

    # Warm restarts (cosine annealing with restarts)
    warm_restart_t0: int = 10  # Initial restart period
    warm_restart_t_mult: int = 2  # Period multiplier after each restart
    warm_restart_eta_min: float = 1e-6

    # === Early Stopping === (December 2025: use centralized thresholds module)
    early_stopping_patience: int = 10  # See app.config.thresholds.EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = 1e-4
    elo_patience: int = 10  # See app.config.thresholds.ELO_PATIENCE
    elo_min_improvement: float = 5.0

    # === Data Quality Scoring ===
    freshness_decay_hours: float = 24.0
    freshness_weight: float = 0.2
    sample_temperature: float = 1.0

    # === Hard Example Mining ===
    use_hard_example_mining: bool = True
    hard_example_buffer_size: int = 10000
    hard_example_fraction: float = 0.3
    hard_example_percentile: float = 80.0
    min_samples_before_mining: int = 1000

    # === Anomaly Detection ===
    loss_spike_threshold: float = 3.0
    gradient_norm_threshold: float = 100.0
    halt_on_nan: bool = True
    halt_on_spike: bool = False
    max_consecutive_anomalies: int = 5

    # === Validation ===
    validation_interval_steps: int | None = 1000
    validation_interval_epochs: float | None = None
    validation_subset_size: float = 1.0
    adaptive_validation_interval: bool = False

    # === Checkpointing ===
    checkpoint_interval_epochs: int = 1
    avg_checkpoints: int = 5
    keep_checkpoints_on_disk: bool = False

    # === EWC (Elastic Weight Consolidation) ===
    use_ewc: bool = False
    lambda_ewc: float = 1000.0
    ewc_num_samples: int = 1000

    # === Calibration ===
    calibration_threshold: float = 0.05
    calibration_check_interval: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for backward compatibility."""
        return {
            # Core
            'base_lr': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'seed': self.seed,

            # Mixed precision
            'use_mixed_precision': self.use_mixed_precision,
            'mixed_precision_dtype': self.mixed_precision_dtype,

            # Gradient accumulation
            'accumulation_steps': self.accumulation_steps,
            'max_grad_norm': self.max_grad_norm,

            # LR schedule
            'lr_scheduler': self.lr_scheduler,
            'warmup_epochs': self.warmup_epochs,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warm_restart_t0': self.warm_restart_t0,
            'warm_restart_t_mult': self.warm_restart_t_mult,
            'warm_restart_eta_min': self.warm_restart_eta_min,

            # Early stopping
            'patience': self.early_stopping_patience,
            'min_delta': self.early_stopping_min_delta,
            'elo_patience': self.elo_patience,
            'elo_min_improvement': self.elo_min_improvement,

            # Data quality
            'freshness_decay_hours': self.freshness_decay_hours,
            'freshness_weight': self.freshness_weight,
            'sample_temperature': self.sample_temperature,

            # Hard example mining
            'use_hard_example_mining': self.use_hard_example_mining,
            'hard_example_buffer_size': self.hard_example_buffer_size,
            'hard_example_fraction': self.hard_example_fraction,
            'hard_example_percentile': self.hard_example_percentile,
            'min_samples_before_mining': self.min_samples_before_mining,

            # Anomaly detection
            'loss_spike_threshold': self.loss_spike_threshold,
            'gradient_norm_threshold': self.gradient_norm_threshold,
            'halt_on_nan': self.halt_on_nan,
            'halt_on_spike': self.halt_on_spike,
            'max_consecutive_anomalies': self.max_consecutive_anomalies,

            # Validation
            'validation_interval_steps': self.validation_interval_steps,
            'validation_interval_epochs': self.validation_interval_epochs,
            'validation_subset_size': self.validation_subset_size,
            'adaptive_validation_interval': self.adaptive_validation_interval,

            # Checkpointing
            'checkpoint_interval_epochs': self.checkpoint_interval_epochs,
            'avg_checkpoints': self.avg_checkpoints,
            'keep_checkpoints_on_disk': self.keep_checkpoints_on_disk,

            # EWC
            'use_ewc': self.use_ewc,
            'lambda_ewc': self.lambda_ewc,
            'ewc_num_samples': self.ewc_num_samples,

            # Calibration
            'calibration_threshold': self.calibration_threshold,
            'calibration_check_interval': self.calibration_check_interval,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TrainingConfig:
        """Create config from dictionary."""
        # Map dictionary keys to dataclass fields
        mapping = {
            'base_lr': 'learning_rate',
            'patience': 'early_stopping_patience',
            'min_delta': 'early_stopping_min_delta',
        }

        kwargs = {}
        for field_info in cls.__dataclass_fields__.values():
            name = field_info.name

            # Check for mapped name first
            dict_key = None
            for k, v in mapping.items():
                if v == name:
                    dict_key = k
                    break

            if dict_key and dict_key in config:
                kwargs[name] = config[dict_key]
            elif name in config:
                kwargs[name] = config[name]

        return cls(**kwargs)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.learning_rate > self.max_lr:
            warnings.append(f"learning_rate ({self.learning_rate}) > max_lr ({self.max_lr})")

        if self.learning_rate < self.min_lr:
            warnings.append(f"learning_rate ({self.learning_rate}) < min_lr ({self.min_lr})")

        if self.accumulation_steps < 1:
            warnings.append("accumulation_steps must be >= 1")

        if self.hard_example_fraction < 0 or self.hard_example_fraction > 1:
            warnings.append("hard_example_fraction must be between 0 and 1")

        if self.validation_subset_size < 0.01 or self.validation_subset_size > 1:
            warnings.append("validation_subset_size must be between 0.01 and 1")

        if self.lr_scheduler == "warm_restarts" and self.warm_restart_t0 < 1:
            warnings.append("warm_restart_t0 must be >= 1")

        return warnings

    def get_effective_batch_size(self) -> int:
        """Get effective batch size with gradient accumulation."""
        return self.batch_size * self.accumulation_steps

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainingConfig:"]
        for field_info in self.__dataclass_fields__.values():
            name = field_info.name
            value = getattr(self, name)
            lines.append(f"  {name}: {value}")
        return "\n".join(lines)
