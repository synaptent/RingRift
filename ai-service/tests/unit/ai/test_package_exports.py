"""Focused tests for app.ai package exports."""

from __future__ import annotations

import importlib
import warnings


def test_package_dir_lists_declared_ai_surface() -> None:
    module = importlib.import_module("app.ai")

    expected = [
        "CANONICAL_DIFFICULTY_PROFILES",
        "DIFFICULTY_DESCRIPTIONS",
        "AIFactory",
        "AIType",
        "DifficultyProfile",
        "create_ai",
        "create_ai_from_difficulty",
        "create_tournament_ai",
        "get_all_difficulties",
        "get_difficulty_description",
        "get_difficulty_profile",
        "get_randomness_for_difficulty",
        "get_think_time_for_difficulty",
        "select_ai_type",
        "uses_neural_net",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for name in expected:
            assert hasattr(module, name)
            assert name in dir(module)


def test_implementation_classes_remain_lazy_compatibility_aliases() -> None:
    module = importlib.import_module("app.ai")

    assert "HeuristicAI" not in module.__all__
    assert "HeuristicAI" not in dir(module)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", category=DeprecationWarning)
        heuristic_cls = module.HeuristicAI

    assert heuristic_cls.__name__ == "HeuristicAI"
    assert any("compatibility alias" in str(warning.message) for warning in captured)
