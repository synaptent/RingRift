"""Public package surface for the canonical Python model types.

``app.models`` is the import boundary used throughout the AI service for the
core game-state, move, board, and configuration models. The concrete
definitions live in ``app.models.core`` because the historical single-file
``app/models.py`` layout was split into a package; this ``__init__`` preserves
the stable import path while keeping torch-heavy helpers lazy.
"""

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


def __dir__() -> list[str]:
    """Expose the intended model surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
