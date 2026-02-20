"""Quality metadata extractor for game databases.

This module provides functions to extract quality scores from game databases
during sync operations. Quality scores are used to prioritize high-quality
training data across the cluster.

Note (December 2025):
    For quality computation logic, prefer using app.quality.unified_quality:

        from app.quality.unified_quality import compute_game_quality

    This module (quality_extractor) focuses on database extraction during sync,
    while unified_quality provides the canonical quality computation formulas.

Usage:
    from app.distributed.quality_extractor import (
        extract_game_quality,
        extract_batch_quality,
        QualityExtractorConfig,
    )

    # Extract quality for a single game
    quality = extract_game_quality(game_row, elo_lookup)

    # Extract quality for a batch of games from a database
    qualities = extract_batch_quality(db_path, game_ids, elo_lookup)
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.distributed.unified_manifest import GameQualityMetadata

logger = logging.getLogger(__name__)


@dataclass
class QualityExtractorConfig:
    """Configuration for quality score extraction.

    Weights must sum to 1.0 for proper normalization.
    """
    # Component weights
    elo_weight: float = 0.4
    length_weight: float = 0.3
    decisive_weight: float = 0.3

    # Elo normalization bounds
    min_elo: float = 1200.0
    max_elo: float = 2400.0
    default_elo: float = 1500.0

    # Game length normalization bounds
    min_length: int = 10
    max_length: int = 200

    # Decisive outcome settings
    draw_credit: float = 0.3  # Credit for draws (vs 1.0 for decisive)

    # Model recency bonus (newer models get higher scores)
    recency_bonus_weight: float = 0.0  # Set > 0 to enable
    recency_half_life_days: float = 7.0

    def __post_init__(self):
        """Validate configuration."""
        total = self.elo_weight + self.length_weight + self.decisive_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Quality weights sum to {total}, expected 1.0. "
                "Scores may not be properly normalized."
            )


# Default configuration
DEFAULT_CONFIG = QualityExtractorConfig()


def extract_game_quality(
    game_row: dict[str, Any],
    elo_lookup: Callable[[str], float] | None = None,
    config: QualityExtractorConfig = DEFAULT_CONFIG,
) -> GameQualityMetadata:
    """Extract quality metadata from a game database row.

    Args:
        game_row: Dictionary with game data from games table
        elo_lookup: Optional function to look up Elo by model/player ID
        config: Quality extraction configuration

    Returns:
        GameQualityMetadata with computed quality score
    """
    game_id = game_row.get("game_id", "")
    game_length = game_row.get("total_moves", 0) or game_row.get("move_count", 0) or 0
    winner = game_row.get("winner")
    termination_reason = game_row.get("termination_reason", "")
    source = game_row.get("source", "")
    created_at = game_row.get("created_at", 0.0)

    # Determine if game was decisive
    is_decisive = winner is not None and winner != -1

    # Extract model version from source or metadata
    model_version = ""
    metadata_json = game_row.get("metadata_json", "")
    if metadata_json:
        try:
            import json
            metadata = json.loads(metadata_json)
            model_version = metadata.get("model_version", "") or metadata.get("model", "")
        except (json.JSONDecodeError, TypeError):
            pass
    # Try to extract from source field (e.g., "selfplay_v42")
    if not model_version and source and "_v" in source:
        model_version = source.split("_v")[-1]

    # Look up Elo ratings for players
    avg_elo = config.default_elo
    min_elo = config.default_elo
    max_elo = config.default_elo
    elo_difference = 0.0

    if elo_lookup and model_version:
        try:
            player_elo = elo_lookup(model_version)
            if player_elo > 0:
                avg_elo = player_elo
                min_elo = player_elo
                max_elo = player_elo
        except Exception as e:
            logger.debug(f"Elo lookup failed for {model_version}: {e}")

    # Check if game already has pre-computed quality score (v9 schema)
    # These are computed by game_quality_scorer at game finalization
    pre_computed_quality = game_row.get("quality_score")
    if pre_computed_quality is not None and pre_computed_quality > 0:
        # Use pre-computed training data quality score
        quality_score = pre_computed_quality
    else:
        # Fall back to Elo-based quality computation for older DBs
        quality_score = GameQualityMetadata.compute_quality_score(
            avg_elo=avg_elo,
            game_length=game_length,
            is_decisive=is_decisive,
            elo_weight=config.elo_weight,
            length_weight=config.length_weight,
            decisive_weight=config.decisive_weight,
            min_elo=config.min_elo,
            max_elo=config.max_elo,
            min_length=config.min_length,
            max_length=config.max_length,
        )

    return GameQualityMetadata(
        game_id=game_id,
        avg_player_elo=avg_elo,
        min_player_elo=min_elo,
        max_player_elo=max_elo,
        elo_difference=elo_difference,
        game_length=game_length,
        is_decisive=is_decisive,
        termination_reason=termination_reason,
        model_version=model_version,
        quality_score=quality_score,
        created_at=created_at if isinstance(created_at, (float, int)) else 0.0,
    )


def extract_batch_quality(
    db_path: Path,
    game_ids: list[str] | None = None,
    elo_lookup: Callable[[str], float] | None = None,
    config: QualityExtractorConfig = DEFAULT_CONFIG,
    limit: int = 10000,
) -> list[GameQualityMetadata]:
    """Extract quality metadata for a batch of games from a database.

    Args:
        db_path: Path to the game database
        game_ids: Optional list of specific game IDs to extract
        elo_lookup: Optional function to look up Elo by model/player ID
        config: Quality extraction configuration
        limit: Maximum number of games to process

    Returns:
        List of GameQualityMetadata for each game
    """
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query - try with quality_score for v9+ schemas first
        base_columns = "game_id, total_moves, winner, termination_reason, source, created_at, metadata_json"
        has_quality_column = True

        # Check if quality_score column exists (v9+ schema)
        try:
            cursor.execute("SELECT quality_score FROM games LIMIT 1")
        except sqlite3.OperationalError:
            has_quality_column = False

        columns = f"{base_columns}, quality_score" if has_quality_column else base_columns
        query = f"SELECT {columns} FROM games"
        params: list[Any] = []

        if game_ids:
            placeholders = ",".join("?" * len(game_ids))
            query += f" WHERE game_id IN ({placeholders})"
            params.extend(game_ids)

        query += " LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Extract quality for each game
        qualities = []
        for row in rows:
            game_row = dict(row)
            quality = extract_game_quality(game_row, elo_lookup, config)
            qualities.append(quality)

        return qualities

    except sqlite3.Error as e:
        logger.error(f"Failed to extract quality from {db_path}: {e}")
        return []
    finally:
        if conn is not None:
            conn.close()


def extract_quality_from_synced_db(
    local_dir: Path,
    elo_lookup: Callable[[str], float] | None = None,
    config: QualityExtractorConfig = DEFAULT_CONFIG,
) -> dict[str, list[GameQualityMetadata]]:
    """Extract quality metadata from all databases in a synced directory.

    Args:
        local_dir: Directory containing synced .db files
        elo_lookup: Optional function to look up Elo by model/player ID
        config: Quality extraction configuration

    Returns:
        Dict mapping database filename to list of GameQualityMetadata
    """
    results: dict[str, list[GameQualityMetadata]] = {}

    if not local_dir.exists():
        logger.warning(f"Sync directory not found: {local_dir}")
        return results

    for db_file in local_dir.glob("*.db"):
        try:
            qualities = extract_batch_quality(db_file, None, elo_lookup, config)
            if qualities:
                results[db_file.name] = qualities
                logger.debug(f"Extracted quality for {len(qualities)} games from {db_file.name}")
        except Exception as e:
            logger.warning(f"Failed to extract quality from {db_file.name}: {e}")

    return results


def get_elo_lookup_from_service() -> Callable[[str], float] | None:
    """Get an Elo lookup function from the EloService.

    Returns:
        Function that looks up Elo by model ID, or None if service unavailable
    """
    try:
        from app.training.elo_service import get_elo_service

        elo_service = get_elo_service()

        def lookup(model_id: str) -> float:
            rating = elo_service.get_rating(model_id)
            if rating:
                return rating.rating
            return 1500.0

        return lookup

    except ImportError:
        logger.debug("EloService not available")
        return None
    except Exception as e:
        logger.debug(f"Failed to create Elo lookup: {e}")
        return None


def compute_priority_score(
    quality: GameQualityMetadata,
    urgency_weight: float = 0.2,
    hours_since_created: float = 0.0,
) -> float:
    """Compute a priority score for sync ordering.

    This combines quality score with urgency (older unsynced games get priority).

    Args:
        quality: GameQualityMetadata for the game
        urgency_weight: Weight for urgency component (0-1)
        hours_since_created: Hours since the game was created

    Returns:
        Priority score (higher = sync first)
    """
    # Base priority from quality
    base_priority = quality.quality_score * (1.0 - urgency_weight)

    # Urgency bonus: games waiting longer get higher priority
    # Caps at 1.0 after 24 hours
    urgency_normalized = min(1.0, hours_since_created / 24.0)
    urgency_bonus = urgency_normalized * urgency_weight

    return base_priority + urgency_bonus
