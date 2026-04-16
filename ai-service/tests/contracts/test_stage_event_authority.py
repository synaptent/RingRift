"""Contract tests for stage event authority boundaries."""

from __future__ import annotations

import re
from pathlib import Path


_DEPRECATED_STAGE_EVENT_IMPORTS = (
    re.compile(r"from\s+app\.events\s+import\s+StageEvent\b"),
    re.compile(r"from\s+app\.events\.types\s+import\s+StageEvent\b"),
)

_ALLOWED_FILES = {
    Path("app/events/__init__.py"),
    Path("app/events/types.py"),
}


def test_app_code_uses_canonical_stage_event_enum() -> None:
    """Application code should depend on the canonical coordination StageEvent enum."""
    offenders: list[str] = []
    app_root = Path("app")

    for path in app_root.rglob("*.py"):
        relative = path.relative_to(Path("."))
        if relative in _ALLOWED_FILES:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _DEPRECATED_STAGE_EVENT_IMPORTS:
            if pattern.search(text):
                offenders.append(str(relative))
                break

    assert not offenders, (
        "Use app.coordination.stage_events.StageEvent as the canonical runtime enum; "
        "deprecated app.events StageEvent imports found in: "
        + ", ".join(sorted(offenders))
    )
