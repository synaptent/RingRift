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

# Unified configuration (December 2025)
try:
    from app.config.unified_config import (
        UnifiedConfig,
        get_config,
        get_training_threshold,
        IntegratedEnhancementsConfig,
        create_training_manager,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False

__all__ = [
    "CMAESConfig",
    "NeuralNetConfig",
    "SelfPlayConfig",
    "ConfigValidator",
    "ValidationResult",
    "validate_all_configs",
    "validate_startup",
    "HAS_UNIFIED_CONFIG",
]

if HAS_UNIFIED_CONFIG:
    __all__.extend([
        "UnifiedConfig",
        "get_config",
        "get_training_threshold",
        "IntegratedEnhancementsConfig",
        "create_training_manager",
    ])
