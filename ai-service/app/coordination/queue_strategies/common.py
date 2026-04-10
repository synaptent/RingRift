"""Shared constants for queue population strategies."""

from __future__ import annotations

import logging
import os as _os

logger = logging.getLogger(__name__)


# All supported board configurations
BOARD_CONFIGS: list[tuple[str, int]] = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Large boards require specialized engine selection (Gumbel MCTS)
# These have many more cells (361-469) and benefit from quality over quantity
LARGE_BOARDS = frozenset({"square19", "hexagonal", "fullhex", "full_hex"})

# Minimum exploration constants (Phase 1.2 - Jan 2026)
# Ensures cluster never idles completely, even when Elo targets are met
MINIMUM_EXPLORATION_GAMES = 100  # Minimum pending games per config
EXPLORATION_CONFIGS_PER_CYCLE = 3  # Number of configs to explore each cycle
EXPLORATION_STALE_THRESHOLD_HOURS = 4.0  # Config is "stale" if no games in this many hours

# Default curriculum weights (priority multipliers)
# Mar 30, 2026: FOCUSED on hex8_2p to prove AlphaZero loop works on one config
# before spreading resources across 12. All other configs get minimal allocation.
# Previous: balanced across all 12, resulting in thin data everywhere.
# Set RINGRIFT_FOCUS_CONFIG=hex8_2p to enable focus mode (default: all configs).
_FOCUS_CONFIG = _os.environ.get("RINGRIFT_FOCUS_CONFIG", "")

if _FOCUS_CONFIG:
    # Focus mode: give 90% weight to focus config, 10% to others
    DEFAULT_CURRICULUM_WEIGHTS: dict[str, float] = {
        k: (5.0 if k == _FOCUS_CONFIG else 0.1)
        for k in [
            "square8_2p", "square8_3p", "square8_4p",
            "square19_2p", "square19_3p", "square19_4p",
            "hex8_2p", "hex8_3p", "hex8_4p",
            "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
        ]
    }
    logger.info(f"[QueuePopulator] FOCUS MODE: {_FOCUS_CONFIG} (5.0x weight)")
else:
    DEFAULT_CURRICULUM_WEIGHTS: dict[str, float] = {
        # Square8: Good 2p data, need more 3p/4p
        "square8_2p": 0.6, "square8_3p": 1.4, "square8_4p": 1.5,
        # Square19: All configs need more data
        "square19_2p": 0.8, "square19_3p": 1.3, "square19_4p": 1.4,
        # Hex8: 2p has some data, 3p/4p critical (especially 4p at 30% win rate)
        "hex8_2p": 0.5, "hex8_3p": 1.5, "hex8_4p": 1.6,
        # Hexagonal: All configs starved
        "hexagonal_2p": 1.0, "hexagonal_3p": 1.4, "hexagonal_4p": 1.3,
    }

