"""Contract tests for training module file-size budgets."""

from __future__ import annotations

from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[2] / "app" / "training"
MAX_TRAINING_MODULE_LINES = 3500
PHASE15_SIZE_TARGETS = {
    "train.py": 3000,
    "elo_service.py": 2000,
}


def _training_python_files() -> list[Path]:
    return sorted(
        path
        for path in TRAINING_ROOT.glob("*.py")
        if "__pycache__" not in path.parts
    )


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=path.name)
        for path in _training_python_files()
    ],
)
def test_training_modules_stay_under_global_size_budget(path: Path) -> None:
    """Keep training modules reviewable instead of allowing new monoliths."""
    line_count = len(path.read_text(encoding="utf-8").splitlines())

    assert line_count <= MAX_TRAINING_MODULE_LINES, (
        f"{path.name} has {line_count} lines; limit is "
        f"{MAX_TRAINING_MODULE_LINES}. Extract a helper module instead of "
        "growing the file."
    )


@pytest.mark.parametrize(
    ("filename", "max_lines"),
    [
        pytest.param(filename, max_lines, id=filename)
        for filename, max_lines in PHASE15_SIZE_TARGETS.items()
    ],
)
def test_phase15_training_refactors_stay_under_target(filename: str, max_lines: int) -> None:
    """Lock in the explicit size targets reached during the Phase 15 split."""
    path = TRAINING_ROOT / filename
    line_count = len(path.read_text(encoding="utf-8").splitlines())

    assert line_count <= max_lines, (
        f"{filename} has {line_count} lines; Phase 15 limit is {max_lines}. "
        "Keep the extracted helpers in separate modules."
    )
