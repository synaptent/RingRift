"""Import-hygiene contracts for supported app modules."""

from __future__ import annotations

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2] / "app"


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_app_modules_do_not_use_wildcard_imports() -> None:
    """Supported app modules should avoid wildcard imports."""
    violations: list[str] = []

    for path in _python_files(APP_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                violations.append(str(path.relative_to(APP_ROOT.parent)))

    assert not violations, f"Wildcard imports found: {violations}"


def test_app_tree_does_not_embed_test_directories() -> None:
    """Pytest-style test directories should live under ai-service/tests/."""
    embedded_tests = sorted(
        str(path.relative_to(APP_ROOT.parent))
        for path in APP_ROOT.rglob("tests")
        if path.is_dir()
    )

    assert not embedded_tests, f"Embedded test directories found under app/: {embedded_tests}"
