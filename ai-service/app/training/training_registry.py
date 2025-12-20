"""Training Registry Integration.

Utility functions for registering trained models in the model registry.
Provides a simple API for training scripts to register their outputs.

Note:
    This module provides training-specific convenience functions that wrap
    the unified model store. For general model operations, prefer using
    :mod:`app.training.unified_model_store` directly::

        from app.training import get_model_store, register_model

Usage:
    from app.training.training_registry import register_trained_model

    # After training completes:
    model_id = register_trained_model(
        model_path="models/square8_2p_v3.pth",
        board_type="square8",
        num_players=2,
        training_config={
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 0.001,
        },
        metrics={
            "final_loss": 0.42,
            "policy_accuracy": 0.65,
        },
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_registry = None


def _get_registry():
    """Get or create the model registry singleton."""
    global _registry
    if _registry is None:
        try:
            from app.training.model_registry import ModelRegistry
            _registry = ModelRegistry()
        except Exception as e:
            logger.warning(f"Could not initialize model registry: {e}")
            return None
    return _registry


def register_trained_model(
    model_path: str,
    board_type: str,
    num_players: int,
    training_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    model_name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
    source: str = "training",
    parent_model_id: str | None = None,
    data_path: str | None = None,
) -> str | None:
    """Register a trained model in the model registry.

    Args:
        model_path: Path to the saved model file (.pth)
        board_type: Board type (e.g., "square8", "hexagonal")
        num_players: Number of players (2, 3, or 4)
        training_config: Training configuration dict
        metrics: Performance metrics dict
        model_name: Optional custom name (defaults to filename)
        description: Optional description
        tags: Optional list of tags
        source: Source of the model (e.g., "training", "gauntlet", "curriculum")
        parent_model_id: ID of parent model (for fine-tuning lineage)
        data_path: Path to training data used

    Returns:
        model_id if registration succeeded, None otherwise
    """
    registry = _get_registry()
    if registry is None:
        logger.warning("Model registry not available - skipping registration")
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None

    # Generate model name from filename if not provided
    if model_name is None:
        model_name = model_path.stem

    # Build tags
    all_tags = [board_type, f"{num_players}p", source]
    if tags:
        all_tags.extend(tags)

    # Build training config
    try:
        from app.training.model_registry import ModelMetrics, ModelStage, ModelType, TrainingConfig

        tc = TrainingConfig(
            learning_rate=training_config.get("learning_rate", 0.001) if training_config else 0.001,
            batch_size=training_config.get("batch_size", 256) if training_config else 256,
            epochs=training_config.get("epochs", training_config.get("epochs_per_iter", 100)) if training_config else 100,
            optimizer=training_config.get("optimizer", "adam") if training_config else "adam",
            architecture=training_config.get("model_version", "v3") if training_config else "v3",
            num_residual_blocks=training_config.get("num_residual_blocks", 6) if training_config else 6,
            num_filters=training_config.get("num_filters", 96) if training_config else 96,
            parent_model_id=parent_model_id,
            training_data_hash=_compute_data_hash(data_path) if data_path else None,
            extra_config={
                "board_type": board_type,
                "num_players": num_players,
                "source": source,
                **(training_config or {}),
            },
        )

        mm = ModelMetrics(
            policy_accuracy=metrics.get("policy_accuracy") if metrics else None,
            value_mse=metrics.get("value_mse", metrics.get("final_loss")) if metrics else None,
        )

        model_id, version = registry.register_model(
            name=model_name,
            model_path=model_path,
            model_type=ModelType.POLICY_VALUE,
            description=description or f"Trained model for {board_type} {num_players}p",
            metrics=mm,
            training_config=tc,
            tags=all_tags,
            initial_stage=ModelStage.DEVELOPMENT,
        )

        logger.info(f"Registered model {model_id} v{version} in registry")
        return model_id

    except Exception as e:
        logger.warning(f"Failed to register model: {e}")
        return None


def _compute_data_hash(data_path: str) -> str | None:
    """Compute a hash of the training data file."""
    if not data_path or not os.path.exists(data_path):
        return None

    import hashlib
    try:
        # Hash first 10MB + file size for speed
        h = hashlib.md5()
        size = os.path.getsize(data_path)
        h.update(str(size).encode())

        with open(data_path, "rb") as f:
            chunk = f.read(10 * 1024 * 1024)  # 10MB
            h.update(chunk)

        return h.hexdigest()[:16]
    except Exception:
        return None


def get_model_lineage(model_id: str) -> list[dict[str, Any]]:
    """Get the lineage (parent chain) for a model."""
    registry = _get_registry()
    if registry is None:
        return []

    lineage = []
    current_id = model_id

    while current_id:
        try:
            model = registry.get_model(current_id)
            if model is None:
                break

            lineage.append({
                "model_id": model.model_id,
                "version": model.version,
                "name": model.name,
                "created_at": model.created_at.isoformat(),
                "stage": model.stage.value,
            })

            # Get parent from training config
            parent_id = model.training_config.parent_model_id
            current_id = parent_id
        except Exception:
            break

    return lineage


def find_best_model(
    board_type: str,
    num_players: int,
    metric: str = "elo",
) -> dict[str, Any] | None:
    """Find the best model for a board config by metric.

    Args:
        board_type: Board type to search for
        num_players: Number of players
        metric: Metric to sort by ("elo", "win_rate", "policy_accuracy")

    Returns:
        Model info dict or None if not found
    """
    registry = _get_registry()
    if registry is None:
        return None

    try:
        # Query models with matching tags
        models = registry.list_models(
            tags=[board_type, f"{num_players}p"],
        )

        if not models:
            return None

        # Sort by metric
        def get_metric_value(m):
            if metric == "elo":
                return m.metrics.elo or 0
            elif metric == "win_rate":
                return m.metrics.win_rate or 0
            elif metric == "policy_accuracy":
                return m.metrics.policy_accuracy or 0
            return 0

        best = max(models, key=get_metric_value)
        return {
            "model_id": best.model_id,
            "version": best.version,
            "name": best.name,
            "file_path": best.file_path,
            "stage": best.stage.value,
            metric: get_metric_value(best),
        }

    except Exception as e:
        logger.warning(f"Error finding best model: {e}")
        return None
