"""Import-hygiene checks for metrics facade consumers."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

RATCHET_FILES = (
    "ai-service/app/tournament/orchestrator.py",
    "ai-service/app/training/elo_reconciliation.py",
    "ai-service/app/training/promotion_controller.py",
    "ai-service/scripts/run_self_play_soak.py",
    "ai-service/scripts/unified_loop/promotion.py",
)


def test_selected_runtime_modules_avoid_top_level_metrics_facade() -> None:
    for relative_path in RATCHET_FILES:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.metrics":
                raise AssertionError(relative_path)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "app.metrics":
                        raise AssertionError(relative_path)


def test_runtime_metrics_facade_import_sites_are_explicitly_inventory_controlled() -> None:
    allowed: set[str] = set()

    actual: set[str] = set()
    for root in (REPO_ROOT / "ai-service" / "app", REPO_ROOT / "ai-service" / "scripts"):
        for path in root.rglob("*.py"):
            if path == REPO_ROOT / "ai-service" / "app" / "metrics" / "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "app.metrics":
                    actual.add(str(path.relative_to(REPO_ROOT)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "app.metrics":
                            actual.add(str(path.relative_to(REPO_ROOT)))

    assert actual == allowed
