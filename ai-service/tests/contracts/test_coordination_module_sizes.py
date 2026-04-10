"""Contract tests for coordination module file-size budgets."""

from __future__ import annotations

from pathlib import Path

import pytest

COORDINATION_ROOT = Path(__file__).resolve().parents[2] / "app" / "coordination"
MAX_COORDINATION_FILE_LINES = 2500


def _coordination_python_files() -> list[Path]:
    return sorted(
        path
        for path in COORDINATION_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=str(path.relative_to(COORDINATION_ROOT)))
        for path in _coordination_python_files()
    ],
)
def test_coordination_modules_stay_under_size_budget(path: Path) -> None:
    """Keep coordination modules small enough to remain reviewable."""
    line_count = len(path.read_text(encoding="utf-8").splitlines())

    assert line_count <= MAX_COORDINATION_FILE_LINES, (
        f"{path.relative_to(COORDINATION_ROOT)} has {line_count} lines; "
        f"limit is {MAX_COORDINATION_FILE_LINES}. Extract a helper module "
        "instead of growing the file."
    )
