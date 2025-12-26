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

    # Port configuration (fastest, no heavy deps)
    from app.config.ports import P2P_DEFAULT_PORT

Note (Dec 2025): This module uses 100% lazy imports.
All submodules (including ports and thresholds) are loaded on first access.
Direct submodule imports like `from app.config.ports import X` skip this __init__.py.
"""

from __future__ import annotations

__all__ = [
    # Port configuration (centralized)
    "DATA_SERVER_PORT",
    "GOSSIP_PORT",
    "HEALTH_CHECK_PORT",
    "METRICS_PORT",
    "P2P_DEFAULT_PORT",
    "get_data_server_url",
    "get_health_check_url",
    "get_p2p_base_url",
    "get_p2p_status_url",
    # Threshold constants
    "ELO_DROP_ROLLBACK",
    "ELO_IMPROVEMENT_PROMOTE",
    "ELO_K_FACTOR",
    "HAS_UNIFIED_CONFIG",
    "INITIAL_ELO_RATING",
    "MIN_GAMES_FOR_ELO",
    "MIN_GAMES_PROMOTE",
    "MIN_GAMES_REGRESSION",
    "TRAINING_MAX_CONCURRENT",
    "TRAINING_MIN_INTERVAL_SECONDS",
    "TRAINING_STALENESS_HOURS",
    "TRAINING_TRIGGER_GAMES",
    "WIN_RATE_DROP_ROLLBACK",
    # Config classes
    "CMAESConfig",
    "ConfigLoadError",
    "ConfigLoader",
    "ConfigSource",
    "ConfigValidator",
    "DataLoadingConfig",
    "IntegratedEnhancementsConfig",
    "NeuralNetConfig",
    "QualityConfig",
    "SelfPlayConfig",
    "UnifiedConfig",
    "ValidationResult",
    # Functions
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

# =============================================================================
# 100% Lazy Loading (December 2025)
# =============================================================================

# Cache for lazy-loaded modules
_lazy_cache: dict = {}

# Flag for unified config availability (checked on first access)
HAS_UNIFIED_CONFIG = True


def __getattr__(name: str):
    """Lazy import for all config exports."""
    global HAS_UNIFIED_CONFIG

    # Port configuration (lightweight)
    if name in ("DATA_SERVER_PORT", "GOSSIP_PORT", "HEALTH_CHECK_PORT", "METRICS_PORT",
                "P2P_DEFAULT_PORT", "get_data_server_url", "get_health_check_url",
                "get_p2p_base_url", "get_p2p_status_url"):
        if "ports" not in _lazy_cache:
            from app.config.ports import (
                DATA_SERVER_PORT as _DSP,
                GOSSIP_PORT as _GP,
                HEALTH_CHECK_PORT as _HCP,
                METRICS_PORT as _MP,
                P2P_DEFAULT_PORT as _PDP,
                get_data_server_url as _gdsu,
                get_health_check_url as _ghcu,
                get_p2p_base_url as _gpbu,
                get_p2p_status_url as _gpsu,
            )
            _lazy_cache["ports"] = {
                "DATA_SERVER_PORT": _DSP,
                "GOSSIP_PORT": _GP,
                "HEALTH_CHECK_PORT": _HCP,
                "METRICS_PORT": _MP,
                "P2P_DEFAULT_PORT": _PDP,
                "get_data_server_url": _gdsu,
                "get_health_check_url": _ghcu,
                "get_p2p_base_url": _gpbu,
                "get_p2p_status_url": _gpsu,
            }
        return _lazy_cache["ports"][name]

    # Threshold constants (lightweight)
    if name in ("ELO_DROP_ROLLBACK", "ELO_IMPROVEMENT_PROMOTE", "ELO_K_FACTOR",
                "INITIAL_ELO_RATING", "MIN_GAMES_FOR_ELO", "MIN_GAMES_PROMOTE",
                "MIN_GAMES_REGRESSION", "TRAINING_MAX_CONCURRENT",
                "TRAINING_MIN_INTERVAL_SECONDS", "TRAINING_STALENESS_HOURS",
                "TRAINING_TRIGGER_GAMES", "WIN_RATE_DROP_ROLLBACK"):
        if "thresholds" not in _lazy_cache:
            from app.config.thresholds import (
                ELO_DROP_ROLLBACK as _EDR,
                ELO_IMPROVEMENT_PROMOTE as _EIP,
                ELO_K_FACTOR as _EKF,
                INITIAL_ELO_RATING as _IER,
                MIN_GAMES_FOR_ELO as _MGFE,
                MIN_GAMES_PROMOTE as _MGP,
                MIN_GAMES_REGRESSION as _MGR,
                TRAINING_MAX_CONCURRENT as _TMC,
                TRAINING_MIN_INTERVAL_SECONDS as _TMIS,
                TRAINING_STALENESS_HOURS as _TSH,
                TRAINING_TRIGGER_GAMES as _TTG,
                WIN_RATE_DROP_ROLLBACK as _WRDR,
            )
            _lazy_cache["thresholds"] = {
                "ELO_DROP_ROLLBACK": _EDR,
                "ELO_IMPROVEMENT_PROMOTE": _EIP,
                "ELO_K_FACTOR": _EKF,
                "INITIAL_ELO_RATING": _IER,
                "MIN_GAMES_FOR_ELO": _MGFE,
                "MIN_GAMES_PROMOTE": _MGP,
                "MIN_GAMES_REGRESSION": _MGR,
                "TRAINING_MAX_CONCURRENT": _TMC,
                "TRAINING_MIN_INTERVAL_SECONDS": _TMIS,
                "TRAINING_STALENESS_HOURS": _TSH,
                "TRAINING_TRIGGER_GAMES": _TTG,
                "WIN_RATE_DROP_ROLLBACK": _WRDR,
            }
        return _lazy_cache["thresholds"][name]

    # Config validator (lightweight)
    if name in ("ConfigValidator", "ValidationResult", "validate_all_configs", "validate_startup"):
        if "config_validator" not in _lazy_cache:
            from app.config.config_validator import (
                ConfigValidator as _CV,
                ValidationResult as _VR,
                validate_all_configs as _vac,
                validate_startup as _vs,
            )
            _lazy_cache["config_validator"] = {
                "ConfigValidator": _CV,
                "ValidationResult": _VR,
                "validate_all_configs": _vac,
                "validate_startup": _vs,
            }
        return _lazy_cache["config_validator"][name]

    # Config loader (lightweight)
    if name in ("ConfigLoader", "ConfigLoadError", "ConfigSource", "env_override",
                "load_config", "merge_configs", "save_config", "validate_config"):
        if "loader" not in _lazy_cache:
            from app.config.loader import (
                ConfigLoader as _CL,
                ConfigLoadError as _CLE,
                ConfigSource as _CS,
                env_override as _eo,
                load_config as _lc,
                merge_configs as _mc,
                save_config as _sc,
                validate_config as _vc,
            )
            _lazy_cache["loader"] = {
                "ConfigLoader": _CL,
                "ConfigLoadError": _CLE,
                "ConfigSource": _CS,
                "env_override": _eo,
                "load_config": _lc,
                "merge_configs": _mc,
                "save_config": _sc,
                "validate_config": _vc,
            }
        return _lazy_cache["loader"][name]

    # Training config (HEAVY - imports torch via utils.checksum_utils)
    if name in ("CMAESConfig", "NeuralNetConfig", "SelfPlayConfig"):
        if "training_config" not in _lazy_cache:
            from app.config.training_config import (
                CMAESConfig as _CMAES,
                NeuralNetConfig as _NN,
                SelfPlayConfig as _SP,
            )
            _lazy_cache["training_config"] = {
                "CMAESConfig": _CMAES,
                "NeuralNetConfig": _NN,
                "SelfPlayConfig": _SP,
            }
        return _lazy_cache["training_config"][name]

    # Unified config (HEAVY - may import torch)
    if name in ("DataLoadingConfig", "IntegratedEnhancementsConfig", "QualityConfig",
                "UnifiedConfig", "create_training_manager", "get_config", "get_training_threshold"):
        if "unified_config" not in _lazy_cache:
            try:
                from app.config.unified_config import (
                    DataLoadingConfig as _DLC,
                    IntegratedEnhancementsConfig as _IEC,
                    QualityConfig as _QC,
                    UnifiedConfig as _UC,
                    create_training_manager as _ctm,
                    get_config as _gc,
                    get_training_threshold as _gtt,
                )
                _lazy_cache["unified_config"] = {
                    "DataLoadingConfig": _DLC,
                    "IntegratedEnhancementsConfig": _IEC,
                    "QualityConfig": _QC,
                    "UnifiedConfig": _UC,
                    "create_training_manager": _ctm,
                    "get_config": _gc,
                    "get_training_threshold": _gtt,
                }
            except ImportError:
                HAS_UNIFIED_CONFIG = False
                _lazy_cache["unified_config"] = {
                    "DataLoadingConfig": None,
                    "IntegratedEnhancementsConfig": None,
                    "QualityConfig": None,
                    "UnifiedConfig": None,
                    "create_training_manager": None,
                    "get_config": None,
                    "get_training_threshold": None,
                }
        return _lazy_cache["unified_config"][name]

    raise AttributeError(f"module 'app.config' has no attribute {name!r}")
