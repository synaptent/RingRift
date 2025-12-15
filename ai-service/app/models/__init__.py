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
    BoardType,
    GamePhase,
    GameStatus,
    MoveType,
    AIType,
    Position,
    LineInfo,
    Territory,
    RingStack,
    MarkerInfo,
    Player,
    TimeControl,
    Move,
    BoardState,
    ChainCaptureSegment,
    ChainCaptureState,
    GameState,
    AIConfig,
    LineRewardChoiceOption,
    LineRewardChoiceRequest,
    LineRewardChoiceResponse,
    RingEliminationChoiceOption,
    RingEliminationChoiceRequest,
    RingEliminationChoiceResponse,
    RegionOrderChoiceOption,
    ProgressSnapshot,
    RegionOrderChoiceRequest,
    RegionOrderChoiceResponse,
    LineOrderChoiceLine,
    LineOrderChoiceRequest,
    LineOrderChoiceResponse,
    CaptureDirectionChoiceOption,
    CaptureDirectionChoiceRequest,
    CaptureDirectionChoiceResponse,
)

# Multi-task learning heads for auxiliary predictions
from .multitask_heads import (
    AuxiliaryTask,
    TaskConfig,
    MultiTaskConfig,
    create_default_multitask_config,
)

# Transformer model architectures
from .transformer_model import (
    TransformerConfig,
    create_model as create_transformer_model,
)

__all__ = [
    # Core game models
    "BoardType",
    "GamePhase",
    "GameStatus",
    "MoveType",
    "AIType",
    "Position",
    "LineInfo",
    "Territory",
    "RingStack",
    "MarkerInfo",
    "Player",
    "TimeControl",
    "Move",
    "BoardState",
    "ChainCaptureSegment",
    "ChainCaptureState",
    "GameState",
    "AIConfig",
    "LineRewardChoiceOption",
    "LineRewardChoiceRequest",
    "LineRewardChoiceResponse",
    "RingEliminationChoiceOption",
    "RingEliminationChoiceRequest",
    "RingEliminationChoiceResponse",
    "RegionOrderChoiceOption",
    "ProgressSnapshot",
    "RegionOrderChoiceRequest",
    "RegionOrderChoiceResponse",
    "LineOrderChoiceLine",
    "LineOrderChoiceRequest",
    "LineOrderChoiceResponse",
    "CaptureDirectionChoiceOption",
    "CaptureDirectionChoiceRequest",
    "CaptureDirectionChoiceResponse",
    # Multi-task learning
    "AuxiliaryTask",
    "TaskConfig",
    "MultiTaskConfig",
    "create_default_multitask_config",
    # Transformer models
    "TransformerConfig",
    "create_transformer_model",
]
