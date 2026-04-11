"""Contract tests for supported-path AI and DB module file-size budgets."""

from __future__ import annotations

from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
MAX_MODULE_LINES = 3500
SIZE_BUDGET_ROOTS = ("ai", "db")


def _supported_python_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SIZE_BUDGET_ROOTS:
        root = APP_ROOT / root_name
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or "archive" in path.parts:
                continue
            if path.name.endswith("_legacy.py"):
                continue
            files.append(path)
    return files


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=str(path.relative_to(APP_ROOT)))
        for path in _supported_python_files()
    ],
)
def test_supported_ai_and_db_modules_stay_under_size_budget(path: Path) -> None:
    """Keep supported-path AI/DB modules small enough to stay reviewable."""
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count <= MAX_MODULE_LINES, (
        f"{path.relative_to(APP_ROOT)} has {line_count} lines; "
        f"limit is {MAX_MODULE_LINES}. Extract a helper module instead of "
        "growing the file."
    )
