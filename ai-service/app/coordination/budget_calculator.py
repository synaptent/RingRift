"""Budget and target games calculation for selfplay scheduling.

December 2025: Extracted from selfplay_scheduler.py (~3900 LOC) to reduce module size
and make budget calculation reusable across the codebase.

This module provides pure functions for:
- Gumbel MCTS budget selection based on Elo tier
- Bootstrap budget tiers for data-starved configs
- Dynamic target games calculation based on board/player complexity

Usage:
    from app.coordination.budget_calculator import (
        get_adaptive_budget_for_elo,
        get_adaptive_budget_for_games,
        compute_target_games,
    )

    # Get budget for selfplay based on current Elo
    budget = get_adaptive_budget_for_elo(1650)  # Returns GUMBEL_BUDGET_QUALITY

    # Get budget accounting for bootstrap phase
    budget = get_adaptive_budget_for_games(game_count=50, elo=1200)  # Returns 64 (bootstrap)

    # Compute target games needed
    target = compute_target_games("hex8_2p", current_elo=1450)  # Returns ~225000
"""

from __future__ import annotations

import logging

from app.config.thresholds import (
    # Elo-based budget tiers
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_ULTIMATE,
    GUMBEL_BUDGET_MASTER,
    # Bootstrap budget tiers for starved configs
    GUMBEL_BUDGET_BOOTSTRAP_TIER1,
    GUMBEL_BUDGET_BOOTSTRAP_TIER2,
    GUMBEL_BUDGET_BOOTSTRAP_TIER3,
    BOOTSTRAP_TIER1_GAME_THRESHOLD,
    BOOTSTRAP_TIER2_GAME_THRESHOLD,
    BOOTSTRAP_TIER3_GAME_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Board and Player Complexity Multipliers
# =============================================================================

# Board difficulty multipliers for target games calculation
# Larger boards need more games to reach the same Elo
BOARD_DIFFICULTY_MULTIPLIERS: dict[str, float] = {
    "hex8": 1.0,       # Smallest - baseline (61 cells)
    "square8": 1.2,    # 64 cells, more complex than hex8
    "square19": 2.0,   # Go-sized (361 cells) - much harder
    "hexagonal": 2.5,  # Largest (469 cells) - hardest
}

# Player count multipliers for target games calculation
# More players = more diverse games needed
PLAYER_COUNT_MULTIPLIERS: dict[int, float] = {
    2: 1.0,   # 2-player baseline
    3: 1.5,   # 3-player more complex game tree
    4: 2.5,   # 4-player exponentially more complex
}

# Elo tier thresholds for budget selection
# Feb 2026: Lowered MASTER from 2000 to 1800 - configs at 1835-1851 Elo
# need maximum quality selfplay data (3200 sims) to break through to 2000.
# The previous 1600-sim ULTIMATE tier wasn't producing enough signal for
# late-stage training improvements.
ELO_TIER_MASTER = 1800     # 1800+ Elo: master tier (3200 sims)
ELO_TIER_ULTIMATE = 1600   # 1600+ Elo: ultimate tier (1600 sims)
ELO_TIER_QUALITY = 1400    # 1400+ Elo: quality tier (800 sims)

# Target Elo for training completion
TARGET_ELO_THRESHOLD = 1900

# Games per Elo point (empirical from training data)
GAMES_PER_ELO_POINT = 500

# Maximum target games cap
MAX_TARGET_GAMES = 500_000


# =============================================================================
# Large Board Budget Caps (February 2026)
# =============================================================================
# High-budget MCTS is too slow on large boards, especially with 3-4 players.
# Cap budgets based on board size AND player count to ensure games complete
# in reasonable time (~5-10 min target).
#
# Feb 2026: Lowered caps dramatically and added player scaling.
# Previous caps (300/400) matched bootstrap tier2 (300), providing no reduction.
# hexagonal_4p at budget 300 took 28+ min/game and games were killed mid-play,
# producing 159 empty run directories on gh200-3.
#
# New approach: base cap per board, scaled down by player count.
# More players = longer games = need lower budget to keep game duration sane.

LARGE_BOARD_BUDGET_CAPS: dict[str, dict[str, int]] = {
    # Feb 2026: Raised square19 caps (200/500 → 400/800). Previous caps produced
    # only 130 effective sims for square19_4p (200 * 0.65), which is near-random
    # play on a 361-cell board. This caused square19_4p to regress from training
    # on garbage data. New caps give 260 effective sims at 4p bootstrap, 520 mature.
    "square19": {"bootstrap": 400, "mature": 800},   # 361 cells (raised from 200/500)
    "hexagonal": {"bootstrap": 200, "mature": 400},   # 469 cells (raised: 64/200 was near-random)
}

# Player count scaling for budget caps.
# More players = exponentially more moves per game = need lower budget.
# 4p hexagonal averages 1233 moves vs ~300 for 2p.
# Feb 2026: Raised 4p from 0.65→0.80. Even at 0.65, square19_4p only got
# 130 effective sims (200 * 0.65) which produced near-random training data
# and caused Elo regression. At 0.80 with new caps: 320 bootstrap, 640 mature.
PLAYER_BUDGET_SCALING: dict[int, float] = {
    2: 1.0,    # 2-player: baseline
    3: 0.80,   # 3-player: ~50% more moves per game (raised from 0.67)
    4: 0.80,   # 4-player: ~4x more moves per game (raised from 0.65)
}

# Game count threshold for "mature" vs "bootstrap" phase
# Feb 2026: Lowered from 2000→500. square19 configs were permanently stuck
# in bootstrap mode because they couldn't accumulate 2000 games at low budgets.
LARGE_BOARD_MATURITY_THRESHOLD = 500


def get_board_adjusted_budget(
    board_type: str,
    base_budget: int,
    game_count: int,
    num_players: int = 2,
) -> int:
    """Apply board-specific budget caps for large boards, scaled by player count.

    Large boards (square19, hexagonal) are too slow with high budgets,
    especially in 3-4 player games where move counts are 2-4x higher.
    This caps the budget to keep game duration at ~5-10 minutes.

    Args:
        board_type: Board type string (e.g., "hex8", "square19", "hexagonal")
        base_budget: Budget from Elo-based or game-count-based selection
        game_count: Current number of games for this config
        num_players: Number of players (2, 3, or 4). More players = lower cap.

    Returns:
        Budget capped to board-appropriate level, or original if no cap applies

    Examples:
        >>> get_board_adjusted_budget("hexagonal", 300, 150, 2)  # 2p bootstrap
        200
        >>> get_board_adjusted_budget("hexagonal", 300, 150, 4)  # 4p bootstrap
        160
        >>> get_board_adjusted_budget("hexagonal", 800, 3000, 2)  # 2p mature
        400
        >>> get_board_adjusted_budget("hexagonal", 800, 3000, 4)  # 4p mature
        320
        >>> get_board_adjusted_budget("hex8", 3200, 150, 4)  # Small board, no cap
        3200
    """
    caps = LARGE_BOARD_BUDGET_CAPS.get(board_type)
    if caps is None:
        # No cap for small boards (hex8, square8)
        return base_budget

    # Select base cap based on maturity
    if game_count >= LARGE_BOARD_MATURITY_THRESHOLD:
        cap = caps["mature"]
    else:
        cap = caps["bootstrap"]

    # Scale cap by player count
    player_scale = PLAYER_BUDGET_SCALING.get(num_players, 1.0)
    cap = int(cap * player_scale)

    # Floor at 16 to maintain minimum viable search
    cap = max(16, cap)

    return min(base_budget, cap)


# =============================================================================
# Budget Calculation Functions
# =============================================================================

def get_adaptive_budget_for_elo(elo: float) -> int:
    """Get Gumbel MCTS budget based on current Elo tier.

    Higher Elo models benefit from deeper search. Scale budget with
    Elo tier to maximize training data quality.

    Tiers (Feb 2026 - lowered thresholds for late-stage training):
    - 1800+ Elo: MASTER tier (3200 budget) - maximum quality for 2000 push
    - 1600+ Elo: ULTIMATE tier (1600 budget) - high quality
    - 1400+ Elo: QUALITY tier (800 budget) - standard quality
    - <1400 Elo: STANDARD tier (800 budget) - baseline

    Args:
        elo: Current Elo rating for the config

    Returns:
        Gumbel budget for selfplay (one of the tier constants)

    Examples:
        >>> get_adaptive_budget_for_elo(2100)
        3200
        >>> get_adaptive_budget_for_elo(1850)
        3200
        >>> get_adaptive_budget_for_elo(1650)
        1600
    """
    if elo >= ELO_TIER_MASTER:
        return GUMBEL_BUDGET_MASTER  # 3200 - master tier
    elif elo >= ELO_TIER_ULTIMATE:
        return GUMBEL_BUDGET_ULTIMATE  # 1600 - ultimate tier
    elif elo >= ELO_TIER_QUALITY:
        return GUMBEL_BUDGET_QUALITY  # 800 - quality tier
    else:
        return GUMBEL_BUDGET_STANDARD  # 800 - standard tier


def get_adaptive_budget_for_games(game_count: int, elo: float) -> int:
    """Get Gumbel MCTS budget based on game count (prioritizes bootstrapping).

    When a config has very few games, prioritize QUANTITY over QUALITY
    to rapidly bootstrap the training dataset. As game count grows,
    transition to Elo-based quality budgets.

    Bootstrap tiers (game_count < threshold):
    - <100 games:  64 budget (THROUGHPUT - max speed for rapid bootstrap)
    - <500 games:  150 budget (faster iteration, acceptable quality)
    - <1000 games: 200 budget (balanced speed/quality)
    - >=1000 games: Use Elo-based adaptive budget (STANDARD/QUALITY/ULTIMATE/MASTER)

    Args:
        game_count: Number of games in the database for this config
        elo: Current Elo rating for the config (used when game_count >= 1000)

    Returns:
        Gumbel budget for selfplay

    Examples:
        >>> get_adaptive_budget_for_games(50, 1200)   # Bootstrap tier 1
        64
        >>> get_adaptive_budget_for_games(300, 1400)  # Bootstrap tier 2
        150
        >>> get_adaptive_budget_for_games(750, 1450)  # Bootstrap tier 3
        200
        >>> get_adaptive_budget_for_games(2000, 1800) # Mature - uses Elo-based
        3200
    """
    # Feb 2026: If Elo is already high (>1600), skip bootstrap tiers.
    # This handles configs that have good models but were data-starved
    # (e.g., after cluster downtime or DB loss). Low-budget bootstrap data
    # is counterproductive when the model already plays at high level.
    if elo >= 1600:
        return get_adaptive_budget_for_elo(elo)

    # Bootstrap phase: prioritize game generation speed
    if game_count < BOOTSTRAP_TIER1_GAME_THRESHOLD:
        # Very starved (<100 games): maximum throughput
        return GUMBEL_BUDGET_BOOTSTRAP_TIER1  # 64

    if game_count < BOOTSTRAP_TIER2_GAME_THRESHOLD:
        # Moderately starved (<500 games): fast iteration
        return GUMBEL_BUDGET_BOOTSTRAP_TIER2  # 150

    if game_count < BOOTSTRAP_TIER3_GAME_THRESHOLD:
        # Somewhat starved (<1000 games): balanced
        return GUMBEL_BUDGET_BOOTSTRAP_TIER3  # 200

    # Mature phase (>=1000 games): use Elo-based quality budget
    return get_adaptive_budget_for_elo(elo)


# =============================================================================
# Intensity-Coupled Budget (Sprint 10 - January 2026)
# =============================================================================

# Intensity multipliers for Gumbel budget
# Higher intensity → higher budget (more quality games)
# Feb 2026: Added "intensive" and "high" (set by EloVelocityMixin)
INTENSITY_BUDGET_MULTIPLIERS: dict[str, float] = {
    "intensive": 1.75,    # Stalled configs need highest quality data
    "hot_path": 1.5,      # Fast iteration needs highest quality data
    "high": 1.5,          # Plateau response, boost quality
    "accelerated": 1.25,  # Accelerated training benefits from quality
    "normal": 1.0,        # Baseline
    "reduced": 0.75,      # Lower intensity can use faster games
    "paused": 0.5,        # Minimal budget for paused configs
}


def get_budget_with_intensity(
    game_count: int,
    elo: float,
    training_intensity: str = "normal",
) -> int:
    """Get Gumbel budget factoring in BOTH data state AND training intensity.

    January 2026 Sprint 10: Couples training intensity to Gumbel budget.
    When training intensity is high (hot_path, accelerated), selfplay should
    produce higher quality games. When intensity is low (reduced, paused),
    prioritize throughput over quality.

    Expected improvement: +20-30 Elo from better intensity/budget alignment.

    Args:
        game_count: Number of games for this config
        elo: Current Elo rating
        training_intensity: One of "hot_path", "accelerated", "normal",
                           "reduced", "paused"

    Returns:
        Adjusted Gumbel budget

    Examples:
        >>> get_budget_with_intensity(500, 1400, "hot_path")
        225  # 150 * 1.5 = 225
        >>> get_budget_with_intensity(5000, 1700, "accelerated")
        1000  # 800 * 1.25 = 1000
        >>> get_budget_with_intensity(5000, 1700, "reduced")
        600  # 800 * 0.75 = 600
    """
    # Get base budget from games/Elo
    base_budget = get_adaptive_budget_for_games(game_count, elo)

    # Apply intensity multiplier
    multiplier = INTENSITY_BUDGET_MULTIPLIERS.get(training_intensity, 1.0)
    adjusted = int(base_budget * multiplier)

    # Clamp to reasonable bounds
    # Min: 32 (minimum viable search depth)
    # Max: 4800 (MASTER * 1.5)
    return max(32, min(4800, adjusted))


def parse_config_key(config: str) -> tuple[str, int]:
    """Parse a config key into board type and player count.

    December 2025: Now delegates to event_utils.parse_config_key() for
    consistent parsing across the codebase.

    Args:
        config: Config key (e.g., "hex8_2p", "square19_4p")

    Returns:
        Tuple of (board_type, num_players)

    Examples:
        >>> parse_config_key("hex8_2p")
        ("hex8", 2)
        >>> parse_config_key("square19_4p")
        ("square19", 4)
    """
    from app.coordination.event_utils import parse_config_key as _parse

    parsed = _parse(config)
    if parsed is None:
        return "hex8", 2  # Default fallback
    return parsed.board_type, parsed.num_players


def compute_target_games(config: str, current_elo: float) -> int:
    """Compute dynamic target games needed based on Elo gap and board difficulty.

    Calculates how many games are needed to reach target Elo (1900) based on:
    - Elo gap to target (larger gap = more games needed)
    - Board difficulty (larger boards need more games)
    - Player count (multiplayer needs more games per sample)

    Args:
        config: Config key (e.g., "hex8_2p", "square19_4p")
        current_elo: Current Elo rating for the config

    Returns:
        Target games needed to reach 1900 Elo (0 if already at target)

    Examples:
        >>> compute_target_games("hex8_2p", 1900)  # Already at target
        0
        >>> compute_target_games("hex8_2p", 1500)  # 400 Elo gap
        200000  # 400 * 500 * 1.0 (board) * 1.0 (players)
        >>> compute_target_games("square19_4p", 1500)  # Hard config
        1000000  # 400 * 500 * 2.0 (board) * 2.5 (players), capped at 500K
    """
    elo_gap = max(0, TARGET_ELO_THRESHOLD - current_elo)

    # No more games needed if already at target
    if elo_gap <= 0:
        return 0

    # Base: ~500 games per Elo point needed (empirical from training data)
    base_target = float(elo_gap * GAMES_PER_ELO_POINT)

    # Parse config for board type and player count
    board, players = parse_config_key(config)

    # Apply board difficulty multiplier
    base_target *= BOARD_DIFFICULTY_MULTIPLIERS.get(board, 1.0)

    # Apply player count multiplier
    base_target *= PLAYER_COUNT_MULTIPLIERS.get(players, 1.0)

    # Cap to reasonable maximum
    return min(int(base_target), MAX_TARGET_GAMES)


def get_budget_tier_name(budget: int) -> str:
    """Get human-readable name for a budget tier.

    Args:
        budget: The Gumbel budget value

    Returns:
        Tier name string

    Examples:
        >>> get_budget_tier_name(3200)
        "MASTER"
        >>> get_budget_tier_name(64)
        "BOOTSTRAP_TIER1"
    """
    budget_names = {
        GUMBEL_BUDGET_MASTER: "MASTER",
        GUMBEL_BUDGET_ULTIMATE: "ULTIMATE",
        GUMBEL_BUDGET_QUALITY: "QUALITY",
        GUMBEL_BUDGET_STANDARD: "STANDARD",
        GUMBEL_BUDGET_BOOTSTRAP_TIER1: "BOOTSTRAP_TIER1",
        GUMBEL_BUDGET_BOOTSTRAP_TIER2: "BOOTSTRAP_TIER2",
        GUMBEL_BUDGET_BOOTSTRAP_TIER3: "BOOTSTRAP_TIER3",
    }
    return budget_names.get(budget, f"CUSTOM({budget})")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Budget calculation
    "get_adaptive_budget_for_elo",
    "get_adaptive_budget_for_games",
    "get_budget_with_intensity",  # Sprint 10: Intensity-coupled budget
    "get_board_adjusted_budget",  # Jan 2026: Large board budget caps
    "get_budget_tier_name",
    # Target games calculation
    "compute_target_games",
    "parse_config_key",
    # Constants (for reference)
    "BOARD_DIFFICULTY_MULTIPLIERS",
    "PLAYER_COUNT_MULTIPLIERS",
    "INTENSITY_BUDGET_MULTIPLIERS",  # Sprint 10: Intensity multipliers
    "LARGE_BOARD_BUDGET_CAPS",  # Jan 2026: Large board caps
    "PLAYER_BUDGET_SCALING",  # Feb 2026: Player count scaling for budget caps
    "LARGE_BOARD_MATURITY_THRESHOLD",  # Jan 2026: Game count for mature phase
    "ELO_TIER_MASTER",
    "ELO_TIER_ULTIMATE",
    "ELO_TIER_QUALITY",
    "TARGET_ELO_THRESHOLD",
    "GAMES_PER_ELO_POINT",
    "MAX_TARGET_GAMES",
]
