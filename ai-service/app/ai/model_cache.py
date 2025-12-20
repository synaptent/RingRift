"""Model cache for memory-efficient sharing of neural network instances.

This module implements a singleton LRU-style cache that allows multiple
NeuralNetAI instances to share loaded model weights. This prevents OOM
issues in long soak tests and selfplay runs where many games are played
in sequence.

Key features:
- TTL-based eviction for stale models (default: 1 hour)
- Max size limit with LRU eviction (default: 10 models)
- Automatic GPU/MPS cache clearing on eviction
- Thread-safe for typical read/write patterns

Usage
-----
The cache is used automatically by :class:`NeuralNetAI`. External callers
may use :func:`clear_model_cache` to explicitly release memory between
games or soak batches::

    from app.ai.model_cache import clear_model_cache

    # After a batch of games...
    clear_model_cache()
"""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from typing import Any

import torch
import torch.nn as nn

from app.utils.time_constants import ONE_HOUR

logger = logging.getLogger(__name__)

# =============================================================================
# Model Cache for Memory Efficiency
# =============================================================================
#
# Singleton cache to share model instances across NeuralNetAI instances.
# Key: (architecture_type, device_str, model_path, checkpoint_signature, board_type)
# Value: (loaded model instance, creation_timestamp, last_access_timestamp)
_MODEL_CACHE: dict[tuple[str, str, str, Any, str], tuple[nn.Module, float, float]] = {}

# Cache configuration
MODEL_CACHE_TTL_SECONDS = ONE_HOUR  # 1 hour TTL for cached models
MODEL_CACHE_MAX_SIZE = 10  # Maximum number of models to keep in cache


def _clear_gpu_caches() -> None:
    """Clear GPU and MPS caches after model eviction."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear MPS cache if available (PyTorch 2.0+)
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()


def evict_stale_models() -> int:
    """Evict models older than TTL or when cache exceeds max size.

    Returns the number of models evicted.
    """
    global _MODEL_CACHE

    now = time.time()
    evicted = 0
    keys_to_remove = []

    # First pass: remove models older than TTL
    for key, (model, _created_at, last_access) in _MODEL_CACHE.items():
        if now - last_access > MODEL_CACHE_TTL_SECONDS:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        model, _, _ = _MODEL_CACHE.pop(key)
        with contextlib.suppress(Exception):
            model.cpu()
        evicted += 1

    # Second pass: if still over limit, remove least recently used
    if len(_MODEL_CACHE) > MODEL_CACHE_MAX_SIZE:
        # Sort by last_access timestamp (oldest first)
        sorted_items = sorted(
            _MODEL_CACHE.items(),
            key=lambda x: x[1][2]  # last_access timestamp
        )
        # Remove oldest until under limit
        while len(_MODEL_CACHE) > MODEL_CACHE_MAX_SIZE and sorted_items:
            key, (model, _, _) = sorted_items.pop(0)
            if key in _MODEL_CACHE:
                del _MODEL_CACHE[key]
                with contextlib.suppress(Exception):
                    model.cpu()
                evicted += 1

    if evicted > 0:
        _clear_gpu_caches()
        gc.collect()
        logger.debug(f"Evicted {evicted} stale models from cache")

    return evicted


def clear_model_cache() -> None:
    """Clear the model cache and release GPU/MPS memory.

    Call this function between games or soak batches to prevent OOM issues.
    This is especially important for MPS where memory management is more
    aggressive than CUDA.
    """
    global _MODEL_CACHE
    cache_size = len(_MODEL_CACHE)

    # Move models to CPU before clearing to release GPU memory
    for model, _, _ in _MODEL_CACHE.values():
        with contextlib.suppress(Exception):
            model.cpu()

    _MODEL_CACHE.clear()
    _clear_gpu_caches()

    # Force garbage collection
    gc.collect()

    if cache_size > 0:
        logger.info(f"Cleared model cache ({cache_size} models)")


def get_cached_model_count() -> int:
    """Return the number of models currently in the cache."""
    return len(_MODEL_CACHE)


def get_cache_ref() -> dict[tuple[str, str, str, Any, str], tuple[nn.Module, float, float]]:
    """Return a reference to the internal cache dict.

    This is used by NeuralNetAI for direct cache access. External callers
    should use the public functions instead.
    """
    return _MODEL_CACHE


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize a PyTorch state_dict by stripping a leading ``module.`` prefix.

    Some training jobs save checkpoints from DistributedDataParallel (DDP),
    which prefixes all parameter keys with ``module.``. Runtime inference
    expects the non-prefixed keys.

    Args:
        state_dict: PyTorch state_dict to normalize

    Returns:
        State dict with ``module.`` prefix removed from keys if present
    """
    if not state_dict:
        return state_dict
    if not any(isinstance(k, str) and k.startswith("module.") for k in state_dict):
        return state_dict
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped
