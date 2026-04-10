"""Smoke imports for training modules without dedicated stem-matched tests.

These modules are still part of the active training surface, but their behavior
is exercised through broader orchestration/integration tests rather than a
same-stem unit test file. This smoke suite gives each of them an explicit,
named ownership point for the test-infrastructure contract.
"""

from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.timeout(30)


MODULES = [
    "app.training.auxiliary_tasks",
    "app.training.checkpointing",
    "app.training.distillation",
    "app.training.lr_finder",
    "app.training.opening_book",
    "app.training.pbt",
    "app.training.thread_integration",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_training_module_import_smoke(module_name: str) -> None:
    """Each active training helper module should be importable."""
    module = importlib.import_module(module_name)
    assert module is not None
