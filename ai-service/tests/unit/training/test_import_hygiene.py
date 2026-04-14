"""Import-hygiene checks for training facade consumers."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

RATCHET_FILES = (
    "ai-service/app/training/auto_promotion.py",
    "ai-service/scripts/training_preflight_check.py",
)


def test_selected_runtime_modules_avoid_top_level_training_facade() -> None:
    for relative_path in RATCHET_FILES:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.training":
                raise AssertionError(relative_path)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "app.training":
                        raise AssertionError(relative_path)


def test_runtime_training_facade_import_sites_are_explicitly_inventory_controlled() -> None:
    allowed: set[str] = set()

    actual: set[str] = set()
    for root in (REPO_ROOT / "ai-service" / "app", REPO_ROOT / "ai-service" / "scripts"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "app.training":
                    actual.add(str(path.relative_to(REPO_ROOT)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "app.training":
                            actual.add(str(path.relative_to(REPO_ROOT)))

    assert actual == allowed
