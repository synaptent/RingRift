"""Dynamic Export Settings for Training Data.

Automatically computes optimal export settings based on available data:
- max_games: Cap on games to export (based on dataset size)
- sample_every: Subsampling rate (based on data volume)
- epochs: Training epochs (based on dataset size)

This replaces hardcoded per-config dicts with intelligent defaults
that adapt to the actual data available.

Usage:
    from app.training.dynamic_export import get_export_settings

    settings = get_export_settings(
        db_paths=["data/games/consolidated.db"],
        board_type="square8",
        num_players=2,
    )

    print(f"max_games={settings.max_games}, sample_every={settings.sample_every}")
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)


@dataclass
class ExportSettings:
    """Computed export settings for a config."""
    max_games: Optional[int]  # None = no limit
    sample_every: int  # Sample every Nth move
    epochs: int  # Recommended training epochs
    batch_size: int  # Recommended batch size
    estimated_samples: int  # Estimated output samples
    data_tier: str  # "small", "medium", "large", "xlarge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_games": self.max_games,
            "sample_every": self.sample_every,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "estimated_samples": self.estimated_samples,
            "data_tier": self.data_tier,
        }


# Data tier thresholds
TIER_THRESHOLDS = {
    "small": 10_000,      # < 10k games
    "medium": 50_000,     # 10k - 50k games
    "large": 200_000,     # 50k - 200k games
    "xlarge": float("inf"),  # > 200k games
}

# Default settings per tier
DEFAULT_SETTINGS = {
    "small": {
        "max_games": None,
        "sample_every": 1,
        "epochs": 100,
        "batch_size": 128,
    },
    "medium": {
        "max_games": 50_000,
        "sample_every": 1,
        "epochs": 75,
        "batch_size": 256,
    },
    "large": {
        "max_games": 100_000,
        "sample_every": 2,
        "epochs": 50,
        "batch_size": 512,
    },
    "xlarge": {
        "max_games": 150_000,
        "sample_every": 3,
        "epochs": 30,
        "batch_size": 1024,
    },
}

# Board-specific adjustments (moves per game varies by board)
BOARD_ADJUSTMENTS = {
    "square8": {"moves_per_game": 60, "sample_factor": 1.0},
    "square19": {"moves_per_game": 200, "sample_factor": 1.5},
    "hexagonal": {"moves_per_game": 80, "sample_factor": 1.2},
    "hex8": {"moves_per_game": 40, "sample_factor": 0.8},
}


def count_games_in_db(db_path: str, board_type: Optional[str] = None, num_players: Optional[int] = None) -> int:
    """Count games in a database, optionally filtered by board/players."""
    if not os.path.exists(db_path):
        return 0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM games WHERE 1=1"
        params = []

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)

        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"Error counting games in {db_path}: {e}")
        return 0


def count_games_in_dbs(
    db_paths: List[str],
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> int:
    """Count total games across multiple databases."""
    total = 0
    for db_path in db_paths:
        total += count_games_in_db(db_path, board_type, num_players)
    return total


def get_data_tier(game_count: int) -> str:
    """Determine data tier based on game count."""
    for tier, threshold in TIER_THRESHOLDS.items():
        if game_count < threshold:
            return tier
    return "xlarge"


def estimate_samples(
    game_count: int,
    board_type: str,
    sample_every: int,
    max_games: Optional[int] = None,
) -> int:
    """Estimate number of training samples from games."""
    effective_games = min(game_count, max_games) if max_games else game_count
    moves_per_game: int = BOARD_ADJUSTMENTS.get(board_type, {}).get("moves_per_game", 60)
    total_moves = effective_games * moves_per_game
    return int(total_moves // sample_every)


def get_export_settings(
    db_paths: List[str],
    board_type: str,
    num_players: int,
    target_samples: Optional[int] = None,
    target_epochs: Optional[int] = None,
) -> ExportSettings:
    """Compute optimal export settings based on available data.

    Args:
        db_paths: List of database paths to export from
        board_type: Board type (e.g., "square8")
        num_players: Number of players (2, 3, or 4)
        target_samples: Optional target sample count (overrides auto)
        target_epochs: Optional target epochs (overrides auto)

    Returns:
        ExportSettings with computed optimal parameters
    """
    # Count available games
    game_count = count_games_in_dbs(db_paths, board_type, num_players)

    # Determine data tier
    tier = get_data_tier(game_count)

    # Get base settings for tier
    settings = DEFAULT_SETTINGS[tier].copy()

    # Apply board-specific adjustments
    board_adj = BOARD_ADJUSTMENTS.get(board_type, {})
    sample_factor = board_adj.get("sample_factor", 1.0)

    # Adjust sample_every based on board type
    if sample_factor > 1.0 and settings["sample_every"] == 1:
        # Larger boards generate more samples per game
        settings["sample_every"] = max(1, int(settings["sample_every"] * sample_factor))

    # Override with targets if provided
    if target_epochs is not None:
        settings["epochs"] = target_epochs

    if target_samples is not None:
        # Adjust max_games to hit target samples
        moves_per_game = board_adj.get("moves_per_game", 60)
        required_games = (target_samples * settings["sample_every"]) // moves_per_game
        settings["max_games"] = min(required_games, game_count)

    # Estimate output samples
    estimated_samples = estimate_samples(
        game_count,
        board_type,
        settings["sample_every"],
        settings["max_games"],
    )

    # Adjust batch size based on estimated samples
    if estimated_samples < 10_000:
        settings["batch_size"] = min(settings["batch_size"], 64)
    elif estimated_samples < 50_000:
        settings["batch_size"] = min(settings["batch_size"], 128)

    return ExportSettings(
        max_games=settings["max_games"],
        sample_every=settings["sample_every"],
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        estimated_samples=estimated_samples,
        data_tier=tier,
    )


def get_config_export_settings(
    config_key: str,
    db_paths: Optional[List[str]] = None,
) -> ExportSettings:
    """Get export settings for a config key (e.g., "square8_2p").

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        db_paths: Optional list of DB paths (uses default if not provided)

    Returns:
        ExportSettings with computed parameters
    """
    # Parse config key
    parts = config_key.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid config key: {config_key}")

    board_type = parts[0]
    num_players = int(parts[1].replace("p", ""))

    # Use default DB paths if not provided
    if db_paths is None:
        default_db = AI_SERVICE_ROOT / "data" / "games" / f"consolidated_{board_type}_{num_players}p.db"
        if default_db.exists():
            db_paths = [str(default_db)]
        else:
            # Try generic consolidated DB
            generic_db = AI_SERVICE_ROOT / "data" / "games" / "consolidated.db"
            db_paths = [str(generic_db)] if generic_db.exists() else []

    return get_export_settings(db_paths, board_type, num_players)


# Legacy compatibility: dict of settings per config
def get_legacy_export_dict(configs: List[str], db_paths: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Get export settings as a dict (legacy format for multi_config_training_loop).

    Args:
        configs: List of config keys (e.g., ["square8_2p", "hexagonal_2p"])
        db_paths: Optional shared DB paths

    Returns:
        Dict mapping config_key -> {max_games, sample_every, epochs}
    """
    result = {}
    for config_key in configs:
        try:
            settings = get_config_export_settings(config_key, db_paths)
            result[config_key] = {
                "max_games": settings.max_games,
                "sample_every": settings.sample_every,
                "epochs": settings.epochs,
            }
        except Exception as e:
            logger.warning(f"Could not compute settings for {config_key}: {e}")
            # Use small tier defaults
            result[config_key] = DEFAULT_SETTINGS["small"].copy()

    return result
