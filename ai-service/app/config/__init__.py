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

from app.config.training_config import (
    CMAESConfig,
    NeuralNetConfig,
    SelfPlayConfig,
)
from app.config.config_validator import (
    ConfigValidator,
    ValidationResult,
    validate_all_configs,
    validate_startup,
)

# Unified config loader (December 2025)
from app.config.loader import (
    load_config,
    save_config,
    ConfigLoader,
    ConfigSource,
    env_override,
    merge_configs,
    validate_config,
    ConfigLoadError,
)

# Unified configuration (December 2025)
try:
    from app.config.unified_config import (
        UnifiedConfig,
        get_config,
        get_training_threshold,
        IntegratedEnhancementsConfig,
        create_training_manager,
        DataLoadingConfig,
        QualityConfig,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False
    DataLoadingConfig = None
    QualityConfig = None

__all__ = [
    # Training configs
    "CMAESConfig",
    "NeuralNetConfig",
    "SelfPlayConfig",
    # Validation
    "ConfigValidator",
    "ValidationResult",
    "validate_all_configs",
    "validate_startup",
    "HAS_UNIFIED_CONFIG",
    # Config loader (December 2025)
    "load_config",
    "save_config",
    "ConfigLoader",
    "ConfigSource",
    "env_override",
    "merge_configs",
    "validate_config",
    "ConfigLoadError",
]

if HAS_UNIFIED_CONFIG:
    __all__.extend([
        "UnifiedConfig",
        "get_config",
        "get_training_threshold",
        "IntegratedEnhancementsConfig",
        "create_training_manager",
        "DataLoadingConfig",
        "QualityConfig",
    ])

# Re-export key threshold constants for convenience
from app.config.thresholds import (
    # Training thresholds
    TRAINING_TRIGGER_GAMES,
    TRAINING_MIN_INTERVAL_SECONDS,
    TRAINING_STALENESS_HOURS,
    TRAINING_MAX_CONCURRENT,
    # Rollback thresholds
    ELO_DROP_ROLLBACK,
    WIN_RATE_DROP_ROLLBACK,
    MIN_GAMES_REGRESSION,
    # Promotion thresholds
    ELO_IMPROVEMENT_PROMOTE,
    MIN_GAMES_PROMOTE,
    # Elo system
    INITIAL_ELO_RATING,
    ELO_K_FACTOR,
    MIN_GAMES_FOR_ELO,
)

__all__.extend([
    # Threshold constants
    "TRAINING_TRIGGER_GAMES",
    "TRAINING_MIN_INTERVAL_SECONDS",
    "TRAINING_STALENESS_HOURS",
    "TRAINING_MAX_CONCURRENT",
    "ELO_DROP_ROLLBACK",
    "WIN_RATE_DROP_ROLLBACK",
    "MIN_GAMES_REGRESSION",
    "ELO_IMPROVEMENT_PROMOTE",
    "MIN_GAMES_PROMOTE",
    "INITIAL_ELO_RATING",
    "ELO_K_FACTOR",
    "MIN_GAMES_FOR_ELO",
])
