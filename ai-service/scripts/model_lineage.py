#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for app.training.model_lineage."""

from __future__ import annotations

import sys
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.model_lineage import *  # noqa: F401,F403
from app.model_lineage import main


if __name__ == "__main__":
    sys.exit(main())
