"""UnifiedModelStore - Consolidated facade for model storage operations (December 2025).

This module provides a single entry point for all model storage needs,
consolidating multiple components into a unified interface:

- ModelRegistry: Lifecycle and stage management
- Model versioning: Checkpoint integrity
- Model loading: Runtime model access
- Event emission: Model lifecycle events

Benefits:
- Single import for all model operations
- Consistent patterns across the system
- Automatic event emission for lifecycle changes
- Graceful fallback when components unavailable

Usage:
    from app.training.unified_model_store import (
        UnifiedModelStore,
        get_model_store,
        register_model,
        get_production_model,
    )

    # Get singleton store
    store = get_model_store()

    # Register a new model
    model_id, version = store.register(
        name="square8_2p_v42",
        model_path="models/trained.pt",
        elo=1650,
    )

    # Get production model
    model = store.get_production("square8_2p")

    # Promote to staging
    store.promote(model_id, "staging")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ModelStoreStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLBACK = "rollback"


class ModelStoreType(Enum):
    """Types of models."""
    POLICY_VALUE = "policy_value"
    NNUE = "nnue"
    HEURISTIC = "heuristic"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_id: str
    version: int
    name: str
    model_type: ModelStoreType
    stage: ModelStoreStage
    model_path: str
    elo: Optional[float] = None
    win_rate: Optional[float] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "name": self.name,
            "model_type": self.model_type.value,
            "stage": self.stage.value,
            "model_path": self.model_path,
            "elo": self.elo,
            "win_rate": self.win_rate,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class UnifiedModelStore:
    """Unified facade for model storage operations.

    Provides a single entry point that delegates to appropriate components
    (ModelRegistry, versioning, loading) with consistent patterns.
    """

    _instance: Optional["UnifiedModelStore"] = None

    def __init__(self, registry_dir: Optional[Path] = None):
        """Initialize the unified model store.

        Args:
            registry_dir: Directory for model registry (uses default if None)
        """
        self.registry_dir = registry_dir
        self._registry = None
        self._loader = None

        # Statistics
        self._models_registered = 0
        self._models_promoted = 0
        self._models_loaded = 0

    @classmethod
    def get_instance(cls, registry_dir: Optional[Path] = None) -> "UnifiedModelStore":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(registry_dir)
        return cls._instance

    def _get_registry(self):
        """Lazy-load the model registry."""
        if self._registry is None:
            try:
                from app.training.model_registry import ModelRegistry
                self._registry = ModelRegistry(self.registry_dir)
            except ImportError:
                logger.debug("ModelRegistry not available")
        return self._registry

    def _get_loader(self):
        """Lazy-load the model loader."""
        if self._loader is None:
            try:
                from app.models.loader import ModelLoader
                self._loader = ModelLoader.get_instance()
            except ImportError:
                logger.debug("ModelLoader not available")
        return self._loader

    def register(
        self,
        name: str,
        model_path: Union[str, Path],
        model_type: ModelStoreType = ModelStoreType.POLICY_VALUE,
        elo: Optional[float] = None,
        win_rate: Optional[float] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        initial_stage: ModelStoreStage = ModelStoreStage.DEVELOPMENT,
        **metadata,
    ) -> Tuple[str, int]:
        """Register a new model or version.

        Args:
            name: Model name (used to generate ID)
            model_path: Path to the model file
            model_type: Type of model
            elo: Elo rating if available
            win_rate: Win rate if available
            description: Model description
            tags: Optional tags
            initial_stage: Initial lifecycle stage
            **metadata: Additional metadata

        Returns:
            Tuple of (model_id, version)
        """
        model_path = Path(model_path)

        registry = self._get_registry()
        if registry is None:
            raise RuntimeError("ModelRegistry not available")

        try:
            # Map stage to registry enum
            from app.training.model_registry import ModelStage, ModelType, ModelMetrics

            stage_map = {
                ModelStoreStage.DEVELOPMENT: ModelStage.DEVELOPMENT,
                ModelStoreStage.STAGING: ModelStage.STAGING,
                ModelStoreStage.PRODUCTION: ModelStage.PRODUCTION,
                ModelStoreStage.ARCHIVED: ModelStage.ARCHIVED,
                ModelStoreStage.ROLLBACK: ModelStage.ROLLBACK,
            }

            type_map = {
                ModelStoreType.POLICY_VALUE: ModelType.POLICY_VALUE,
                ModelStoreType.NNUE: ModelType.NNUE,
                ModelStoreType.HEURISTIC: ModelType.HEURISTIC,
            }

            # Build metrics if available
            metrics = None
            if elo is not None or win_rate is not None:
                metrics = ModelMetrics(
                    elo=elo or 1500.0,
                    win_rate=win_rate or 0.5,
                )

            model_id, version = registry.register_model(
                name=name,
                model_path=model_path,
                model_type=type_map.get(model_type, ModelType.POLICY_VALUE),
                description=description,
                metrics=metrics,
                tags=tags,
                initial_stage=stage_map.get(initial_stage, ModelStage.DEVELOPMENT),
            )

            self._models_registered += 1

            # Emit registration event
            self._emit_model_event("registered", model_id, version, {
                "name": name,
                "stage": initial_stage.value,
                "elo": elo,
            })

            logger.info(f"[UnifiedModelStore] Registered {model_id} v{version}")
            return model_id, version

        except Exception as e:
            logger.error(f"[UnifiedModelStore] Registration failed: {e}")
            raise

    def get(
        self,
        model_id: str,
        version: Optional[int] = None,
    ) -> Optional[ModelInfo]:
        """Get model information.

        Args:
            model_id: Model identifier
            version: Specific version (latest if None)

        Returns:
            ModelInfo or None if not found
        """
        registry = self._get_registry()
        if registry is None:
            return None

        try:
            model = registry.get_model(model_id, version)
            if model is None:
                return None

            return ModelInfo(
                model_id=model.model_id,
                version=model.version,
                name=model.name,
                model_type=self._map_model_type(model.model_type),
                stage=self._map_stage(model.stage),
                model_path=str(model.storage_path) if model.storage_path else "",
                elo=model.metrics.elo if model.metrics else None,
                win_rate=model.metrics.win_rate if model.metrics else None,
                created_at=model.created_at,
            )
        except Exception as e:
            logger.debug(f"[UnifiedModelStore] Error getting model: {e}")
            return None

    def get_production(
        self,
        config_key: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """Get the current production model.

        Args:
            config_key: Optional config filter (e.g., "square8_2p")

        Returns:
            ModelInfo for production model or None
        """
        registry = self._get_registry()
        if registry is None:
            return None

        try:
            model = registry.get_production_model()
            if model is None:
                return None

            return ModelInfo(
                model_id=model.model_id,
                version=model.version,
                name=model.name,
                model_type=self._map_model_type(model.model_type),
                stage=ModelStoreStage.PRODUCTION,
                model_path=str(model.storage_path) if model.storage_path else "",
                elo=model.metrics.elo if model.metrics else None,
                win_rate=model.metrics.win_rate if model.metrics else None,
            )
        except Exception as e:
            logger.debug(f"[UnifiedModelStore] Error getting production model: {e}")
            return None

    def promote(
        self,
        model_id: str,
        target_stage: Union[str, ModelStoreStage],
        version: Optional[int] = None,
    ) -> bool:
        """Promote a model to a new stage.

        Args:
            model_id: Model identifier
            target_stage: Target stage (can be string or enum)
            version: Specific version (latest if None)

        Returns:
            True if promotion succeeded
        """
        registry = self._get_registry()
        if registry is None:
            return False

        try:
            from app.training.model_registry import ModelStage

            # Map stage
            if isinstance(target_stage, str):
                target_stage = ModelStoreStage(target_stage)

            stage_map = {
                ModelStoreStage.DEVELOPMENT: ModelStage.DEVELOPMENT,
                ModelStoreStage.STAGING: ModelStage.STAGING,
                ModelStoreStage.PRODUCTION: ModelStage.PRODUCTION,
                ModelStoreStage.ARCHIVED: ModelStage.ARCHIVED,
                ModelStoreStage.ROLLBACK: ModelStage.ROLLBACK,
            }

            registry.promote_model(model_id, stage_map[target_stage], version)

            self._models_promoted += 1

            # Emit promotion event
            self._emit_model_event("promoted", model_id, version or 0, {
                "stage": target_stage.value,
            })

            logger.info(f"[UnifiedModelStore] Promoted {model_id} to {target_stage.value}")
            return True

        except Exception as e:
            logger.error(f"[UnifiedModelStore] Promotion failed: {e}")
            return False

    def list_models(
        self,
        stage: Optional[ModelStoreStage] = None,
        model_type: Optional[ModelStoreType] = None,
        limit: int = 100,
    ) -> List[ModelInfo]:
        """List registered models.

        Args:
            stage: Filter by stage
            model_type: Filter by type
            limit: Maximum number to return

        Returns:
            List of ModelInfo
        """
        registry = self._get_registry()
        if registry is None:
            return []

        try:
            from app.training.model_registry import ModelStage, ModelType

            stage_filter = None
            type_filter = None

            if stage:
                stage_map = {
                    ModelStoreStage.DEVELOPMENT: ModelStage.DEVELOPMENT,
                    ModelStoreStage.STAGING: ModelStage.STAGING,
                    ModelStoreStage.PRODUCTION: ModelStage.PRODUCTION,
                    ModelStoreStage.ARCHIVED: ModelStage.ARCHIVED,
                }
                stage_filter = stage_map.get(stage)

            if model_type:
                type_map = {
                    ModelStoreType.POLICY_VALUE: ModelType.POLICY_VALUE,
                    ModelStoreType.NNUE: ModelType.NNUE,
                    ModelStoreType.HEURISTIC: ModelType.HEURISTIC,
                }
                type_filter = type_map.get(model_type)

            models = registry.list_models(stage=stage_filter, model_type=type_filter)

            return [
                ModelInfo(
                    model_id=m.model_id,
                    version=m.version,
                    name=m.name,
                    model_type=self._map_model_type(m.model_type),
                    stage=self._map_stage(m.stage),
                    model_path=str(m.storage_path) if m.storage_path else "",
                    elo=m.metrics.elo if m.metrics else None,
                )
                for m in models[:limit]
            ]

        except Exception as e:
            logger.debug(f"[UnifiedModelStore] Error listing models: {e}")
            return []

    def load_model(
        self,
        model_id: str,
        version: Optional[int] = None,
        device: str = "cpu",
    ) -> Optional[Any]:
        """Load a model for inference.

        Args:
            model_id: Model identifier
            version: Specific version (latest if None)
            device: Device to load on ("cpu", "cuda", etc.)

        Returns:
            Loaded model or None
        """
        loader = self._get_loader()
        if loader is None:
            # Fallback: try direct loading from registry
            model_info = self.get(model_id, version)
            if model_info and model_info.model_path:
                try:
                    import torch
                    self._models_loaded += 1
                    return torch.load(model_info.model_path, map_location=device)
                except Exception as e:
                    logger.error(f"[UnifiedModelStore] Direct load failed: {e}")
            return None

        try:
            model = loader.load(model_id, version, device=device)
            self._models_loaded += 1
            return model
        except Exception as e:
            logger.error(f"[UnifiedModelStore] Load failed: {e}")
            return None

    def _map_stage(self, stage) -> ModelStoreStage:
        """Map internal stage enum to unified stage."""
        try:
            from app.training.model_registry import ModelStage
            mapping = {
                ModelStage.DEVELOPMENT: ModelStoreStage.DEVELOPMENT,
                ModelStage.STAGING: ModelStoreStage.STAGING,
                ModelStage.PRODUCTION: ModelStoreStage.PRODUCTION,
                ModelStage.ARCHIVED: ModelStoreStage.ARCHIVED,
                ModelStage.ROLLBACK: ModelStoreStage.ROLLBACK,
            }
            return mapping.get(stage, ModelStoreStage.DEVELOPMENT)
        except ImportError:
            return ModelStoreStage.DEVELOPMENT

    def _map_model_type(self, model_type) -> ModelStoreType:
        """Map internal type enum to unified type."""
        try:
            from app.training.model_registry import ModelType
            mapping = {
                ModelType.POLICY_VALUE: ModelStoreType.POLICY_VALUE,
                ModelType.NNUE: ModelStoreType.NNUE,
                ModelType.HEURISTIC: ModelStoreType.HEURISTIC,
            }
            return mapping.get(model_type, ModelStoreType.POLICY_VALUE)
        except ImportError:
            return ModelStoreType.POLICY_VALUE

    def _emit_model_event(
        self,
        action: str,
        model_id: str,
        version: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit model lifecycle event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.REGISTRY_UPDATED,
                payload={
                    "action": action,
                    "model_id": model_id,
                    "version": version,
                    "timestamp": time.time(),
                    **(extra or {}),
                },
                source="unified_model_store",
            )

            bus = get_event_bus()
            bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to emit model event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "models_registered": self._models_registered,
            "models_promoted": self._models_promoted,
            "models_loaded": self._models_loaded,
            "registry_available": self._registry is not None,
            "loader_available": self._loader is not None,
        }


# Singleton access
_store: Optional[UnifiedModelStore] = None


def get_model_store(registry_dir: Optional[Path] = None) -> UnifiedModelStore:
    """Get the global UnifiedModelStore singleton."""
    global _store
    if _store is None:
        _store = UnifiedModelStore.get_instance(registry_dir)
    return _store


# Convenience functions

def register_model(
    name: str,
    model_path: Union[str, Path],
    **kwargs,
) -> Tuple[str, int]:
    """Register a model."""
    return get_model_store().register(name, model_path, **kwargs)


def get_production_model(config_key: Optional[str] = None) -> Optional[ModelInfo]:
    """Get the production model."""
    return get_model_store().get_production(config_key)


def promote_model(
    model_id: str,
    target_stage: Union[str, ModelStoreStage],
    version: Optional[int] = None,
) -> bool:
    """Promote a model."""
    return get_model_store().promote(model_id, target_stage, version)


__all__ = [
    "UnifiedModelStore",
    "ModelInfo",
    "ModelStoreStage",
    "ModelStoreType",
    "get_model_store",
    "register_model",
    "get_production_model",
    "promote_model",
]
