"""Focused tests for app.models package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_model_surface() -> None:
    module = importlib.import_module("app.models")

    expected = [
        "AIConfig",
        "AIType",
        "AuxiliaryTask",
        "BoardState",
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
        "TransformerConfig",
        "create_default_multitask_config",
        "create_transformer_model",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
