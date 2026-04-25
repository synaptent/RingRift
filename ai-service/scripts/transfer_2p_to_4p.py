#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for player-count model transfer."""

from __future__ import annotations

import sys
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.player_count_transfer import *  # noqa: E402,F401,F403
from app.training.player_count_transfer import main  # noqa: E402


if __name__ == "__main__":
    main()
