"""Focused coverage for the private TrainingOrchestrator compatibility module."""

from __future__ import annotations

import importlib
import warnings


def _load_deprecated_orchestrator_module():
    """Import or reload the module so its deprecation warning is observable."""

    module_name = "app.training._deprecated_orchestrated_training"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.reload(importlib.import_module(module_name))

    warning_text = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert any("orchestrated_training.py is deprecated" in text for text in warning_text)
    return module


def test_deprecated_training_orchestrator_surface_resolves() -> None:
    module = _load_deprecated_orchestrator_module()

    assert module.TrainingOrchestrator.__name__ == "TrainingOrchestrator"
    assert module.TrainingOrchestratorConfig.__name__ == "TrainingOrchestratorConfig"
    assert module.TrainingOrchestratorState.__name__ == "TrainingOrchestratorState"
    assert module.get_training_orchestrator().__class__ is module.TrainingOrchestrator
