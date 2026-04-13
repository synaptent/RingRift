"""Import-hygiene checks for coordination facade consumers."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

RATCHET_FILES = (
    "ai-service/app/distributed/unified_data_sync.py",
    "ai-service/app/distributed/cluster_coordinator.py",
    "ai-service/app/distributed/event_helpers.py",
    "ai-service/app/training/background_selfplay.py",
    "ai-service/app/execution/backends.py",
    "ai-service/scripts/unified_loop/data_collection.py",
    "ai-service/scripts/model_promotion_manager.py",
)


def test_selected_runtime_modules_avoid_top_level_coordination_facade() -> None:
    for relative_path in RATCHET_FILES:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "from app.coordination import" not in text, relative_path
        assert "import app.coordination as" not in text, relative_path
