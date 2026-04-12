"""Validate DataEventType references stay in sync with the enum definition."""

from __future__ import annotations

import ast
from pathlib import Path

from app.distributed.data_events import DataEventType

ROOT = Path(__file__).resolve().parents[1]


def _iter_python_files(root: Path) -> list[Path]:
    skip_dirs = {
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        files.append(path)
    return files


def _load_data_event_types() -> set[str]:
    return set(DataEventType.__members__.keys())


def _find_data_event_type_references(text: str) -> list[str]:
    tree = ast.parse(text)
    names: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "DataEventType":
            continue
        if node.attr.startswith("__"):
            continue
        names.append(node.attr)

    return names


def test_data_event_type_references_are_defined() -> None:
    enum_names = _load_data_event_types()
    unknown: dict[str, list[str]] = {}

    for path in _iter_python_files(ROOT):
        text = path.read_text()
        for name in _find_data_event_type_references(text):
            if name in enum_names:
                continue
            unknown.setdefault(name, []).append(str(path.relative_to(ROOT)))

    assert not unknown, f"Unknown DataEventType references: {unknown}"
