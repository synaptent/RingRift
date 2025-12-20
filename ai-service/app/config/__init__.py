"""Configuration module for training runs.

Provides typed dataclasses for:
- CMA-ES optimization (CMAESConfig)
- Neural network training (NeuralNetConfig)
- Self-play data generation (SelfPlayConfig)
- Configuration validation (ConfigValidator)
- Unified configuration (UnifiedConfig, get_config) - December 2025
- Threshold constants (thresholds module)

Usage:
    # Legacy configs
    from app.config import CMAESConfig, NeuralNetConfig

    # Unified config (canonical)
    from app.config import get_config, UnifiedConfig

    # Threshold constants
    from app.config.thresholds import TRAINING_TRIGGER_GAMES, ELO_DROP_ROLLBACK
"""

__all__ = [
    "CMAESConfig",
    "ConfigLoadError",
    "ConfigLoader",
    "ConfigSource",
    "ConfigValidator",
    "DataLoadingConfig",
    "ELO_DROP_ROLLBACK",
    "ELO_IMPROVEMENT_PROMOTE",
    "ELO_K_FACTOR",
    "HAS_UNIFIED_CONFIG",
    "INITIAL_ELO_RATING",
    "IntegratedEnhancementsConfig",
    "MIN_GAMES_FOR_ELO",
    "MIN_GAMES_PROMOTE",
    "MIN_GAMES_REGRESSION",
    "NeuralNetConfig",
    "QualityConfig",
    "SelfPlayConfig",
    "TRAINING_MAX_CONCURRENT",
    "TRAINING_MIN_INTERVAL_SECONDS",
    "TRAINING_STALENESS_HOURS",
    "TRAINING_TRIGGER_GAMES",
    "UnifiedConfig",
    "ValidationResult",
    "WIN_RATE_DROP_ROLLBACK",
    "create_training_manager",
    "env_override",
    "get_config",
    "get_training_threshold",
    "load_config",
    "merge_configs",
    "save_config",
    "validate_all_configs",
    "validate_config",
    "validate_startup",
]

from app.config.config_validator import (
    ConfigValidator,
    ValidationResult,
    validate_all_configs,
    validate_startup,
)

# Unified config loader (December 2025)
from app.config.loader import (
    ConfigLoader,
    ConfigLoadError,
    ConfigSource,
    env_override,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from app.config.training_config import (
    CMAESConfig,
    NeuralNetConfig,
    SelfPlayConfig,
)

# Unified configuration (December 2025)
try:
    from app.config.unified_config import (
        DataLoadingConfig,
        IntegratedEnhancementsConfig,
        QualityConfig,
        UnifiedConfig,
        create_training_manager,
        get_config,
        get_training_threshold,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False
    DataLoadingConfig = None
    QualityConfig = None

# Re-export key threshold constants for convenience
from app.config.thresholds import (
    # Rollback thresholds
    ELO_DROP_ROLLBACK,
    # Promotion thresholds
    ELO_IMPROVEMENT_PROMOTE,
    ELO_K_FACTOR,
    # Elo system
    INITIAL_ELO_RATING,
    MIN_GAMES_FOR_ELO,
    MIN_GAMES_PROMOTE,
    MIN_GAMES_REGRESSION,
    TRAINING_MAX_CONCURRENT,
    TRAINING_MIN_INTERVAL_SECONDS,
    TRAINING_STALENESS_HOURS,
    # Training thresholds
    TRAINING_TRIGGER_GAMES,
    WIN_RATE_DROP_ROLLBACK,
)
