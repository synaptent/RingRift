"""Focused tests for scripts.lib import compatibility."""

from __future__ import annotations

import importlib
import warnings


def test_scripts_lib_import_avoids_deprecated_submodule_warnings():
    """Importing scripts.lib should not warn about its deprecated compatibility exports."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("scripts.lib")
        importlib.reload(module)

    warning_text = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
    ]
    assert not any("scripts.lib.data_quality is deprecated" in msg for msg in warning_text)
    assert not any("scripts.lib.retry is deprecated" in msg for msg in warning_text)
