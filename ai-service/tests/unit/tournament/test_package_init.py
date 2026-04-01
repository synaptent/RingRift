"""Tests for app.tournament package import behavior."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_app_tournament_import_avoids_unified_elo_deprecation_warning():
    """Importing app.tournament should not warn just for legacy re-exports."""
    sys.modules.pop("app.tournament", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("app.tournament")

    deprecation_warnings = [
        warning for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert not deprecation_warnings
