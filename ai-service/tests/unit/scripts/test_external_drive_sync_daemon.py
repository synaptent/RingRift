"""Focused tests for external drive sync daemon safety checks."""

from __future__ import annotations

import shutil


def test_check_disk_has_capacity_fails_closed_when_usage_unavailable(monkeypatch, tmp_path):
    """Sync should block if the daemon cannot verify disk capacity."""
    from scripts import external_drive_sync_daemon

    monkeypatch.setattr(external_drive_sync_daemon, "HAS_RESOURCE_GUARD", False)
    monkeypatch.setattr(external_drive_sync_daemon, "unified_get_disk_usage", None)

    def raise_disk_error(_path):
        raise FileNotFoundError("disk not mounted")

    monkeypatch.setattr(shutil, "disk_usage", raise_disk_error)

    has_capacity, usage_percent = external_drive_sync_daemon.check_disk_has_capacity(tmp_path)

    assert has_capacity is False
    assert usage_percent == 100.0
