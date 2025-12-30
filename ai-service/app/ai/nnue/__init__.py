"""NNUE (Efficiently Updatable Neural Network) model management.

This package provides:
- registry: Canonical path management for NNUE models
"""

from .registry import (
    CANONICAL_CONFIGS,
    NNUEModelInfo,
    NNUERegistryStats,
    get_nnue_canonical_path,
    get_nnue_config_key,
    get_nnue_model_info,
    get_all_nnue_paths,
    get_existing_nnue_models,
    get_missing_nnue_models,
    get_nnue_registry_stats,
    get_nnue_output_path,
    promote_nnue_model,
    print_nnue_registry_status,
)

__all__ = [
    "CANONICAL_CONFIGS",
    "NNUEModelInfo",
    "NNUERegistryStats",
    "get_nnue_canonical_path",
    "get_nnue_config_key",
    "get_nnue_model_info",
    "get_all_nnue_paths",
    "get_existing_nnue_models",
    "get_missing_nnue_models",
    "get_nnue_registry_stats",
    "get_nnue_output_path",
    "promote_nnue_model",
    "print_nnue_registry_status",
]
