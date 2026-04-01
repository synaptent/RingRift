"""Focused tests for app.distributed package imports."""

from __future__ import annotations

import importlib
import warnings


def test_app_distributed_import_avoids_compat_deprecation_warnings():
    """Importing app.distributed should not emit compatibility deprecation noise."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("app.distributed")
        importlib.reload(module)

    warning_text = [
        str(w.message)
        for w in caught
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]
    assert not any("SyncOrchestrator may be deprecated" in msg for msg in warning_text)
    assert not any("UnifiedDataSyncService is deprecated" in msg for msg in warning_text)
