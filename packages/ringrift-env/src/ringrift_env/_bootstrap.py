"""Locate the RingRift Python rules engine and make it importable.

The rules engine lives in the RingRift repository under ``ai-service/app``
and is imported as the top-level package ``app`` (the ai-service convention
is ``PYTHONPATH=ai-service``). This module resolves that path once, in
priority order:

1. ``RINGRIFT_AI_SERVICE_PATH`` environment variable (points at the
   ``ai-service`` directory) — required for use outside the repository.
2. Repository-relative discovery: walk up from this file looking for an
   ``ai-service/app/game_engine`` directory. This makes an in-repo editable
   install (``pip install -e packages/ringrift-env``) work with no setup.
3. Current-working-directory discovery, for running from a repo checkout.

If ``app.game_engine`` is already importable, the existing path wins.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_ENV_VAR = "RINGRIFT_AI_SERVICE_PATH"


def _is_ai_service_dir(path: Path) -> bool:
    return (path / "app" / "game_engine").is_dir()


def _discover_ai_service() -> Path | None:
    override = os.environ.get(_ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if _is_ai_service_dir(candidate):
            return candidate
        raise ImportError(
            f"{_ENV_VAR}={override!r} does not look like a RingRift "
            "ai-service directory (missing app/game_engine)."
        )

    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [start, *start.parents]:
            candidate = parent / "ai-service"
            if _is_ai_service_dir(candidate):
                return candidate
    return None


def _engine_spec_present() -> bool:
    try:
        return importlib.util.find_spec("app.game_engine") is not None
    except ModuleNotFoundError:
        # find_spec on a dotted name imports the parent package; a missing
        # top-level "app" surfaces as ModuleNotFoundError rather than None.
        return False


def ensure_engine_importable() -> None:
    """Make the ``app`` rules-engine package importable, or raise with help."""
    if _engine_spec_present():
        return

    ai_service = _discover_ai_service()
    if ai_service is None:
        raise ImportError(
            "ringrift-env could not locate the RingRift rules engine. "
            "Either run from a RingRift repository checkout, or set "
            f"{_ENV_VAR} to the path of the ai-service directory "
            "(e.g. /path/to/RingRift/ai-service)."
        )
    sys.path.insert(0, str(ai_service))
    if not _engine_spec_present():
        raise ImportError(
            f"Found {ai_service}, but 'app.game_engine' is still not "
            "importable from it. The checkout may be incomplete."
        )
