"""Focused tests for app.cli package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_cli_surface() -> None:
    module = importlib.import_module("app.cli")

    expected = [
        "ProgressBar",
        "ScriptRunner",
        "add_common_args",
        "print_error",
        "print_progress",
        "print_status",
        "print_success",
        "print_table",
        "setup_script",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
