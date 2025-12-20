# This file exposes models from app.models module
# It is used to allow imports like `from app.models import GameState`
# where `app.models` refers to the package `app/models/` but the actual
# definitions are in `app/models.py`.
#
# However, since we have `app/models.py` AND `app/models/` directory,
# Python treats `app.models` as the package (the directory).
# The file `app/models.py` is shadowed by the directory `app/models/`.
#
# To fix this, we should move `app/models.py` into `app/models/__init__.py`
# OR rename `app/models.py` to something else and import it here.
#
# Given the constraints, the best approach is to rename `app/models.py`
# to `app/models/core.py` and import everything here.

from .core import (
    AIConfig,
    AIType,
    BoardState,
    BoardType,
    CaptureDirectionChoiceOption,
    CaptureDirectionChoiceRequest,
    CaptureDirectionChoiceResponse,
    ChainCaptureSegment,
    ChainCaptureState,
    GamePhase,
    GameState,
    GameStatus,
    LineInfo,
    LineOrderChoiceLine,
    LineOrderChoiceRequest,
    LineOrderChoiceResponse,
    LineRewardChoiceOption,
    LineRewardChoiceRequest,
    LineRewardChoiceResponse,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    ProgressSnapshot,
    RegionOrderChoiceOption,
    RegionOrderChoiceRequest,
    RegionOrderChoiceResponse,
    RingEliminationChoiceOption,
    RingEliminationChoiceRequest,
    RingEliminationChoiceResponse,
    RingStack,
    Territory,
    TimeControl,
)

# Multi-task learning heads for auxiliary predictions
# These are imported lazily to avoid loading torch when not needed
# Import directly from .multitask_heads when needed for training
_multitask_heads = None


def _get_multitask_heads():
    """Lazy import of multitask_heads module."""
    global _multitask_heads
    if _multitask_heads is None:
        from . import multitask_heads as mth
        _multitask_heads = mth
    return _multitask_heads


# Provide lazy accessors for torch-dependent types
def __getattr__(name):
    """Lazy attribute access for multitask_heads and transformer types."""
    # Multitask heads
    if name in ('AuxiliaryTask', 'TaskConfig', 'MultiTaskConfig', 'create_default_multitask_config'):
        return getattr(_get_multitask_heads(), name)

    # Transformer types
    if name == 'TransformerConfig':
        mod = _get_transformer()
        return mod.TransformerConfig if mod else None
    if name == 'create_transformer_model':
        mod = _get_transformer()
        return mod.create_model if mod else None
    if name == 'TRANSFORMER_AVAILABLE':
        return _get_transformer() is not False

    raise AttributeError(f"module 'app.models' has no attribute '{name}'")

# Transformer model architectures (loaded lazily to avoid importing torch)
# Use __getattr__ to access TransformerConfig, create_transformer_model, TRANSFORMER_AVAILABLE
_transformer_module = None


def _get_transformer():
    """Lazy import of transformer_model module."""
    global _transformer_module
    if _transformer_module is None:
        try:
            from . import transformer_model
            _transformer_module = transformer_model
        except (ImportError, NameError):
            _transformer_module = False  # Mark as unavailable
    return _transformer_module

__all__ = [
    "AIConfig",
    "AIType",
    # Multi-task learning
    "AuxiliaryTask",
    "BoardState",
    # Core game models
    "BoardType",
    "CaptureDirectionChoiceOption",
    "CaptureDirectionChoiceRequest",
    "CaptureDirectionChoiceResponse",
    "ChainCaptureSegment",
    "ChainCaptureState",
    "GamePhase",
    "GameState",
    "GameStatus",
    "LineInfo",
    "LineOrderChoiceLine",
    "LineOrderChoiceRequest",
    "LineOrderChoiceResponse",
    "LineRewardChoiceOption",
    "LineRewardChoiceRequest",
    "LineRewardChoiceResponse",
    "MarkerInfo",
    "Move",
    "MoveType",
    "MultiTaskConfig",
    "Player",
    "Position",
    "ProgressSnapshot",
    "RegionOrderChoiceOption",
    "RegionOrderChoiceRequest",
    "RegionOrderChoiceResponse",
    "RingEliminationChoiceOption",
    "RingEliminationChoiceRequest",
    "RingEliminationChoiceResponse",
    "RingStack",
    "TaskConfig",
    "Territory",
    "TimeControl",
    # Transformer models
    "TransformerConfig",
    "create_default_multitask_config",
    "create_transformer_model",
]
