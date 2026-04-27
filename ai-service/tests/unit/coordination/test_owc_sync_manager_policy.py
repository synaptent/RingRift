from __future__ import annotations

from app.coordination.owc_sync_manager import OWCSyncConfig, OWCSyncManager


def test_owc_sync_cache_is_bounded() -> None:
    manager = OWCSyncManager(
        OWCSyncConfig(
            enable_pull=False,
            max_cache_entries=2,
        )
    )
    manager._file_mtimes = {"old": 1.0, "new": 3.0, "mid": 2.0}
    manager._file_checksums = {"old": "a", "new": "c", "mid": "b"}

    manager._prune_file_cache()

    assert manager._file_mtimes == {"new": 3.0, "mid": 2.0}
    assert manager._file_checksums == {"new": "c", "mid": "b"}
