"""Unified Model Loader for RingRift AI service.

This module provides a centralized interface for loading AI models,
abstracting away the details of model registry, versioning, and caching.
All orchestrators should use this module for model loading to ensure
consistent behavior.

Usage:
    from app.models.loader import ModelLoader, get_model, get_latest_model

    # Get a model by ID
    model, metadata = get_model("model_123")

    # Get the latest production model
    model, metadata = get_latest_model(
        board_type="square8",
        num_players=2,
        stage="production",
    )

    # Use the loader directly for more control
    loader = ModelLoader()
    model = loader.load_nnue("square8", 2)
    policy = loader.load_policy("square8", 2)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from app.training.model_registry import ModelRecord, ModelStage

logger = logging.getLogger(__name__)


# =============================================================================
# Model metadata
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model_type: str  # "nnue", "policy", "value", "policy_value"
    board_type: str
    num_players: int
    path: str
    version: Optional[str] = None
    elo: Optional[float] = None
    stage: Optional[str] = None
    architecture: Optional[str] = None
    loaded_from_cache: bool = False


# =============================================================================
# Model cache
# =============================================================================


class ModelCache:
    """Thread-safe cache for loaded models.

    This singleton cache prevents loading the same model multiple times
    and manages memory by limiting cache size.
    """

    _instance: Optional["ModelCache"] = None
    _lock = threading.RLock()

    # Default max cache sizes
    MAX_NNUE_MODELS = 4
    MAX_POLICY_MODELS = 2
    MAX_VALUE_MODELS = 2

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._nnue_cache: Dict[str, Tuple[nn.Module, ModelInfo]] = {}
        self._policy_cache: Dict[str, Tuple[nn.Module, ModelInfo]] = {}
        self._value_cache: Dict[str, Tuple[nn.Module, ModelInfo]] = {}
        self._cache_lock = threading.RLock()

    def _make_key(self, board_type: str, num_players: int, model_id: Optional[str] = None) -> str:
        """Create a cache key."""
        if model_id:
            return f"{board_type}_{num_players}p_{model_id}"
        return f"{board_type}_{num_players}p_default"

    def get_nnue(self, key: str) -> Optional[Tuple[nn.Module, ModelInfo]]:
        """Get a cached NNUE model."""
        with self._cache_lock:
            return self._nnue_cache.get(key)

    def put_nnue(self, key: str, model: nn.Module, info: ModelInfo) -> None:
        """Cache a NNUE model, evicting oldest if at capacity."""
        with self._cache_lock:
            if len(self._nnue_cache) >= self.MAX_NNUE_MODELS:
                # Evict oldest (first inserted)
                oldest_key = next(iter(self._nnue_cache))
                self._evict_model(self._nnue_cache, oldest_key)
            self._nnue_cache[key] = (model, info)

    def get_policy(self, key: str) -> Optional[Tuple[nn.Module, ModelInfo]]:
        """Get a cached policy model."""
        with self._cache_lock:
            return self._policy_cache.get(key)

    def put_policy(self, key: str, model: nn.Module, info: ModelInfo) -> None:
        """Cache a policy model."""
        with self._cache_lock:
            if len(self._policy_cache) >= self.MAX_POLICY_MODELS:
                oldest_key = next(iter(self._policy_cache))
                self._evict_model(self._policy_cache, oldest_key)
            self._policy_cache[key] = (model, info)

    def get_value(self, key: str) -> Optional[Tuple[nn.Module, ModelInfo]]:
        """Get a cached value model."""
        with self._cache_lock:
            return self._value_cache.get(key)

    def put_value(self, key: str, model: nn.Module, info: ModelInfo) -> None:
        """Cache a value model."""
        with self._cache_lock:
            if len(self._value_cache) >= self.MAX_VALUE_MODELS:
                oldest_key = next(iter(self._value_cache))
                self._evict_model(self._value_cache, oldest_key)
            self._value_cache[key] = (model, info)

    def _evict_model(self, cache: Dict, key: str) -> None:
        """Evict a model from cache and free memory."""
        if key in cache:
            model, _ = cache.pop(key)
            try:
                model.cpu()
            except Exception:
                pass
            del model
            gc.collect()
            logger.debug(f"Evicted model from cache: {key}")

    def clear(self) -> None:
        """Clear all caches and free memory."""
        with self._cache_lock:
            for cache in [self._nnue_cache, self._policy_cache, self._value_cache]:
                for key in list(cache.keys()):
                    self._evict_model(cache, key)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            gc.collect()
            logger.info("Cleared all model caches")

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "nnue_models": len(self._nnue_cache),
                "policy_models": len(self._policy_cache),
                "value_models": len(self._value_cache),
            }


# =============================================================================
# Model Loader
# =============================================================================


class ModelLoader:
    """Unified model loading interface.

    This class provides a consistent way to load models across the codebase,
    with automatic caching, version validation, and error handling.
    """

    # Default model directories (relative to ai-service root)
    DEFAULT_NNUE_DIR = "models/nnue"
    DEFAULT_POLICY_DIR = "models/policy"
    DEFAULT_VALUE_DIR = "models/value"

    def __init__(
        self,
        base_path: Optional[Path] = None,
        use_cache: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the model loader.

        Args:
            base_path: Base path for model directories (auto-detected if None)
            use_cache: Whether to use the model cache
            device: Device to load models on (auto-detected if None)
        """
        self._base_path = self._detect_base_path(base_path)
        self._use_cache = use_cache
        self._cache = ModelCache() if use_cache else None
        self._device = device or self._detect_device()
        self._registry = None

    def _detect_base_path(self, base_path: Optional[Path]) -> Path:
        """Detect the ai-service base path."""
        if base_path:
            return Path(base_path)

        candidates = [
            Path(__file__).parent.parent.parent,
            Path.cwd() / "ai-service",
            Path.cwd(),
            Path(os.getenv("RINGRIFT_AI_SERVICE_PATH", "")),
        ]

        for candidate in candidates:
            if candidate and (candidate / "models").exists():
                return candidate

        return Path.cwd()

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def registry(self):
        """Lazy-load the model registry."""
        if self._registry is None:
            try:
                from app.training.model_registry import ModelRegistry
                self._registry = ModelRegistry()
            except ImportError:
                logger.warning("ModelRegistry not available")
                self._registry = None
        return self._registry

    def load_nnue(
        self,
        board_type: str,
        num_players: int,
        model_id: Optional[str] = None,
        stage: str = "production",
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load a NNUE model.

        Args:
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players
            model_id: Specific model ID (uses latest if None)
            stage: Model stage to use if model_id is None

        Returns:
            Tuple of (model, info)

        Raises:
            FileNotFoundError: If no model is found
        """
        cache_key = self._cache._make_key(board_type, num_players, model_id) if self._cache else None

        # Check cache
        if self._cache and cache_key:
            cached = self._cache.get_nnue(cache_key)
            if cached:
                model, info = cached
                info.loaded_from_cache = True
                logger.debug(f"NNUE model loaded from cache: {cache_key}")
                return model, info

        # Find model path
        path = self._find_nnue_path(board_type, num_players, model_id, stage)

        # Load model
        from app.ai.nnue import NNUENetwork

        model = NNUENetwork(board_type, num_players)
        state_dict = torch.load(path, map_location=self._device, weights_only=True)

        # Handle wrapped state dict
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        model.to(self._device)
        model.eval()

        info = ModelInfo(
            model_id=model_id or f"nnue_{board_type}_{num_players}p",
            model_type="nnue",
            board_type=board_type,
            num_players=num_players,
            path=str(path),
        )

        # Cache model
        if self._cache and cache_key:
            self._cache.put_nnue(cache_key, model, info)

        logger.info(f"Loaded NNUE model: {path}")
        return model, info

    def _find_nnue_path(
        self,
        board_type: str,
        num_players: int,
        model_id: Optional[str],
        stage: str,
    ) -> Path:
        """Find the path to a NNUE model."""
        nnue_dir = self._base_path / self.DEFAULT_NNUE_DIR

        # Try model_id specific path
        if model_id:
            path = nnue_dir / f"{model_id}.pth"
            if path.exists():
                return path

        # Try standard naming convention
        patterns = [
            f"nnue_{board_type}_{num_players}p_{stage}.pth",
            f"nnue_{board_type}_{num_players}p.pth",
            f"{board_type}_{num_players}p_nnue.pth",
            f"{board_type}_{num_players}p.pth",
        ]

        for pattern in patterns:
            path = nnue_dir / pattern
            if path.exists():
                return path

        # Try to find from registry
        if self.registry:
            try:
                from app.training.model_registry import ModelStage
                stage_enum = getattr(ModelStage, stage.upper(), ModelStage.PRODUCTION)
                records = self.registry.get_models_by_stage(stage_enum)
                for record in records:
                    if record.board_type == board_type and record.num_players == num_players:
                        if record.checkpoint_path and Path(record.checkpoint_path).exists():
                            return Path(record.checkpoint_path)
            except Exception as e:
                logger.debug(f"Registry lookup failed: {e}")

        raise FileNotFoundError(
            f"No NNUE model found for {board_type}_{num_players}p (stage={stage})"
        )

    def load_policy(
        self,
        board_type: str,
        num_players: int,
        model_id: Optional[str] = None,
    ) -> Tuple[nn.Module, ModelInfo]:
        """Load a policy model.

        Args:
            board_type: Board type
            num_players: Number of players
            model_id: Specific model ID

        Returns:
            Tuple of (model, info)
        """
        cache_key = self._cache._make_key(board_type, num_players, model_id) if self._cache else None

        if self._cache and cache_key:
            cached = self._cache.get_policy(cache_key)
            if cached:
                model, info = cached
                info.loaded_from_cache = True
                return model, info

        # Find and load policy model
        policy_dir = self._base_path / self.DEFAULT_POLICY_DIR
        patterns = [
            f"policy_{board_type}_{num_players}p.pth",
            f"{board_type}_{num_players}p_policy.pth",
        ]

        path = None
        for pattern in patterns:
            candidate = policy_dir / pattern
            if candidate.exists():
                path = candidate
                break

        if not path:
            raise FileNotFoundError(
                f"No policy model found for {board_type}_{num_players}p"
            )

        # Load the model
        from app.ai.nnue_policy import PolicyNetwork

        model = PolicyNetwork(board_type, num_players)
        state_dict = torch.load(path, map_location=self._device, weights_only=True)

        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        model.to(self._device)
        model.eval()

        info = ModelInfo(
            model_id=model_id or f"policy_{board_type}_{num_players}p",
            model_type="policy",
            board_type=board_type,
            num_players=num_players,
            path=str(path),
        )

        if self._cache and cache_key:
            self._cache.put_policy(cache_key, model, info)

        logger.info(f"Loaded policy model: {path}")
        return model, info

    def get_available_models(
        self,
        board_type: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> Dict[str, list]:
        """List available models.

        Args:
            board_type: Filter by board type
            model_type: Filter by model type (nnue, policy, value)

        Returns:
            Dict of model_type -> list of available models
        """
        result: Dict[str, list] = {"nnue": [], "policy": [], "value": []}

        # Scan model directories
        for mtype, mdir in [
            ("nnue", self.DEFAULT_NNUE_DIR),
            ("policy", self.DEFAULT_POLICY_DIR),
            ("value", self.DEFAULT_VALUE_DIR),
        ]:
            if model_type and mtype != model_type:
                continue

            dir_path = self._base_path / mdir
            if not dir_path.exists():
                continue

            for f in dir_path.glob("*.pth"):
                name = f.stem
                # Extract board type if possible
                if board_type and board_type not in name:
                    continue
                result[mtype].append({
                    "name": name,
                    "path": str(f),
                    "size_mb": f.stat().st_size / (1024 * 1024),
                })

        return result

    def clear_cache(self) -> None:
        """Clear the model cache."""
        if self._cache:
            self._cache.clear()


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_loader: Optional[ModelLoader] = None


def get_loader() -> ModelLoader:
    """Get the default model loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ModelLoader()
    return _default_loader


def get_model(
    model_id: str,
    model_type: str = "nnue",
    board_type: str = "square8",
    num_players: int = 2,
) -> Tuple[nn.Module, ModelInfo]:
    """Load a model by ID.

    Args:
        model_id: Model identifier
        model_type: Type of model (nnue, policy, value)
        board_type: Board type
        num_players: Number of players

    Returns:
        Tuple of (model, info)
    """
    loader = get_loader()

    if model_type == "nnue":
        return loader.load_nnue(board_type, num_players, model_id)
    elif model_type == "policy":
        return loader.load_policy(board_type, num_players, model_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_latest_model(
    board_type: str = "square8",
    num_players: int = 2,
    model_type: str = "nnue",
    stage: str = "production",
) -> Tuple[nn.Module, ModelInfo]:
    """Load the latest model for a configuration.

    Args:
        board_type: Board type
        num_players: Number of players
        model_type: Type of model
        stage: Model stage (production, staging, development)

    Returns:
        Tuple of (model, info)
    """
    loader = get_loader()

    if model_type == "nnue":
        return loader.load_nnue(board_type, num_players, stage=stage)
    elif model_type == "policy":
        return loader.load_policy(board_type, num_players)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def clear_model_cache() -> None:
    """Clear the model cache and free memory."""
    if _default_loader:
        _default_loader.clear_cache()
