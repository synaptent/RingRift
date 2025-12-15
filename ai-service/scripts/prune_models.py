#!/usr/bin/env python3
"""Automatic model pruning based on Elo ratings.

Removes low-performing models to save disk space and speed up tournaments.
Preserves:
- Models above Elo threshold
- Models with insufficient games (need more evaluation)
- Canonical/best models for each config
- Models younger than age threshold (give them time to be evaluated)

Usage:
    # Dry run - show what would be pruned
    python scripts/prune_models.py --dry-run

    # Prune models with Elo < 1400 and > 30 games
    python scripts/prune_models.py --min-elo 1400 --min-games 30

    # Prune and archive (move to archive dir instead of delete)
    python scripts/prune_models.py --archive

    # Show statistics only
    python scripts/prune_models.py --stats
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

MODELS_DIR = AI_SERVICE_ROOT / "models"
ARCHIVE_DIR = AI_SERVICE_ROOT / "models" / "archive"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"  # Canonical Elo database

# Default thresholds
DEFAULT_MIN_ELO = 1400  # Prune models below this Elo
DEFAULT_MIN_GAMES = 30  # Only prune if model has played this many games
DEFAULT_MIN_AGE_DAYS = 7  # Don't prune models younger than this


@dataclass
class ModelStats:
    """Statistics for a model."""
    path: str
    filename: str
    elo: float
    games: int
    wins: int
    losses: int
    draws: int
    board_type: str
    num_players: int
    created_at: Optional[datetime]
    is_canonical: bool
    size_mb: float


def get_model_elo_stats(db_path: Path) -> Dict[str, ModelStats]:
    """Get Elo statistics for all models from the unified database."""
    if not db_path.exists():
        print(f"Warning: Elo database not found at {db_path}")
        return {}

    stats = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get all NN model ratings from elo_ratings table
        # Schema: model_id, board_type, num_players, rating, games_played, wins, losses, draws, last_update
        cursor.execute("""
            SELECT
                model_id,
                board_type,
                num_players,
                rating,
                games_played,
                wins,
                losses,
                draws,
                last_update
            FROM elo_ratings
            WHERE model_id LIKE 'nn:%' OR model_id LIKE '%.pth%'
               OR model_id LIKE 'ringrift_%' OR model_id LIKE 'sq%'
               OR model_id LIKE 'hex%' OR model_id LIKE 'v%'
        """)

        for row in cursor.fetchall():
            model_id, board_type, num_players, elo, games, wins, losses, draws, last_played = row

            # Extract model filename from model_id
            # Format is typically: nn:path/to/model.pth or just model_name
            if model_id.startswith("nn:"):
                model_path = model_id[3:]
                filename = Path(model_path).name
            else:
                filename = model_id
                model_path = str(MODELS_DIR / filename)

            # Check if file exists (try with and without .pth extension)
            full_path = MODELS_DIR / filename
            if not full_path.exists():
                # Try adding .pth extension
                full_path = MODELS_DIR / f"{filename}.pth"
                if not full_path.exists():
                    continue
                filename = f"{filename}.pth"

            # Get file stats
            try:
                file_stat = full_path.stat()
                size_mb = file_stat.st_size / (1024 * 1024)
                created_at = datetime.fromtimestamp(file_stat.st_ctime)
            except Exception:
                size_mb = 0
                created_at = None

            stats[filename] = ModelStats(
                path=str(full_path),
                filename=filename,
                elo=elo or 1500,
                games=games or 0,
                wins=wins or 0,
                losses=losses or 0,
                draws=draws or 0,
                board_type=board_type,
                num_players=num_players,
                created_at=created_at,
                is_canonical=False,  # Will be updated below
                size_mb=size_mb,
            )

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

    return stats


def get_canonical_models() -> Set[str]:
    """Get set of canonical model filenames that should never be pruned."""
    canonical = set()

    # Check for symlinks or known canonical names
    canonical_patterns = [
        "ringrift_best_*.pth",
        "*_canonical.pth",
        "*_production.pth",
    ]

    for pattern in canonical_patterns:
        for path in MODELS_DIR.glob(pattern):
            canonical.add(path.name)

    # Also check the unified Elo database for top-ranked models per config
    if ELO_DB_PATH.exists():
        conn = sqlite3.connect(ELO_DB_PATH)
        cursor = conn.cursor()
        try:
            # Get top model for each board_type/num_players combo
            cursor.execute("""
                SELECT model_id, board_type, num_players, rating as elo
                FROM elo_ratings
                WHERE games_played >= 10
                ORDER BY board_type, num_players, rating DESC
            """)

            seen_configs = set()
            for row in cursor.fetchall():
                model_id, board_type, num_players, elo = row
                config_key = f"{board_type}_{num_players}p"
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    # Extract filename
                    if model_id.startswith("nn:"):
                        filename = Path(model_id[3:]).name
                    else:
                        filename = model_id
                    canonical.add(filename)
                    print(f"  Canonical for {config_key}: {filename} (Elo: {elo:.0f})")

        except sqlite3.Error:
            pass
        finally:
            conn.close()

    return canonical


def get_models_to_prune(
    stats: Dict[str, ModelStats],
    canonical: Set[str],
    min_elo: float = DEFAULT_MIN_ELO,
    min_games: int = DEFAULT_MIN_GAMES,
    min_age_days: int = DEFAULT_MIN_AGE_DAYS,
) -> List[ModelStats]:
    """Determine which models should be pruned."""
    to_prune = []
    now = datetime.now()
    min_age = timedelta(days=min_age_days)

    for filename, model in stats.items():
        # Never prune canonical models
        if filename in canonical or model.is_canonical:
            continue

        # Don't prune models with insufficient games
        if model.games < min_games:
            continue

        # Don't prune young models
        if model.created_at and (now - model.created_at) < min_age:
            continue

        # Prune if below Elo threshold
        if model.elo < min_elo:
            to_prune.append(model)

    # Sort by Elo ascending (worst first)
    to_prune.sort(key=lambda m: m.elo)

    return to_prune


def prune_models(
    models: List[ModelStats],
    archive: bool = False,
    dry_run: bool = False,
) -> Tuple[int, float]:
    """Prune the specified models.

    Returns (count_pruned, space_freed_mb).
    """
    if archive and not dry_run:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    space_mb = 0.0

    for model in models:
        path = Path(model.path)
        if not path.exists():
            continue

        if dry_run:
            action = "Would archive" if archive else "Would delete"
            print(f"  {action}: {model.filename} (Elo: {model.elo:.0f}, games: {model.games}, size: {model.size_mb:.1f}MB)")
        else:
            if archive:
                dest = ARCHIVE_DIR / model.filename
                shutil.move(str(path), str(dest))
                print(f"  Archived: {model.filename}")
            else:
                path.unlink()
                print(f"  Deleted: {model.filename}")

        count += 1
        space_mb += model.size_mb

    return count, space_mb


def print_stats(stats: Dict[str, ModelStats], canonical: Set[str]):
    """Print model statistics."""
    if not stats:
        print("No model statistics available.")
        return

    total_models = len(stats)
    total_size = sum(m.size_mb for m in stats.values())
    total_games = sum(m.games for m in stats.values())

    # Elo distribution
    elos = [m.elo for m in stats.values() if m.games > 0]
    if elos:
        avg_elo = sum(elos) / len(elos)
        min_elo = min(elos)
        max_elo = max(elos)
    else:
        avg_elo = min_elo = max_elo = 1500

    print("\n" + "=" * 60)
    print("MODEL INVENTORY STATISTICS")
    print("=" * 60)
    print(f"\nTotal models: {total_models}")
    print(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"Total games played: {total_games}")
    print(f"Canonical models: {len(canonical)}")
    print(f"\nElo distribution:")
    print(f"  Min: {min_elo:.0f}")
    print(f"  Avg: {avg_elo:.0f}")
    print(f"  Max: {max_elo:.0f}")

    # Distribution by Elo bands
    bands = [
        (0, 1200, "< 1200 (weak)"),
        (1200, 1400, "1200-1400 (below avg)"),
        (1400, 1500, "1400-1500 (average)"),
        (1500, 1600, "1500-1600 (above avg)"),
        (1600, 1800, "1600-1800 (strong)"),
        (1800, 9999, "> 1800 (elite)"),
    ]

    print("\nElo bands:")
    for low, high, label in bands:
        count = sum(1 for m in stats.values() if low <= m.elo < high and m.games > 0)
        size = sum(m.size_mb for m in stats.values() if low <= m.elo < high and m.games > 0)
        print(f"  {label}: {count} models ({size:.1f} MB)")

    # Models needing evaluation
    unevaluated = sum(1 for m in stats.values() if m.games < 10)
    print(f"\nModels needing evaluation (< 10 games): {unevaluated}")


def main():
    parser = argparse.ArgumentParser(description="Prune low-performing models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned without doing it")
    parser.add_argument("--archive", action="store_true", help="Move to archive instead of deleting")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--min-elo", type=float, default=DEFAULT_MIN_ELO, help=f"Minimum Elo to keep (default: {DEFAULT_MIN_ELO})")
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES, help=f"Minimum games before pruning (default: {DEFAULT_MIN_GAMES})")
    parser.add_argument("--min-age-days", type=int, default=DEFAULT_MIN_AGE_DAYS, help=f"Minimum age in days (default: {DEFAULT_MIN_AGE_DAYS})")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    print("Loading model statistics from Elo database...")
    stats = get_model_elo_stats(ELO_DB_PATH)

    # Also add models not in database
    for pth in MODELS_DIR.glob("*.pth"):
        if pth.name not in stats:
            try:
                file_stat = pth.stat()
                stats[pth.name] = ModelStats(
                    path=str(pth),
                    filename=pth.name,
                    elo=1500,  # Default
                    games=0,
                    wins=0,
                    losses=0,
                    draws=0,
                    board_type="unknown",
                    num_players=0,
                    created_at=datetime.fromtimestamp(file_stat.st_ctime),
                    is_canonical=False,
                    size_mb=file_stat.st_size / (1024 * 1024),
                )
            except Exception:
                pass

    print(f"Found {len(stats)} models")

    print("\nIdentifying canonical models...")
    canonical = get_canonical_models()

    if args.stats:
        print_stats(stats, canonical)
        return

    print_stats(stats, canonical)

    # Determine what to prune
    print(f"\nPruning criteria:")
    print(f"  Min Elo: {args.min_elo}")
    print(f"  Min games: {args.min_games}")
    print(f"  Min age: {args.min_age_days} days")

    to_prune = get_models_to_prune(
        stats, canonical,
        min_elo=args.min_elo,
        min_games=args.min_games,
        min_age_days=args.min_age_days,
    )

    if not to_prune:
        print("\nNo models to prune.")
        return

    space_to_free = sum(m.size_mb for m in to_prune)
    print(f"\nModels to prune: {len(to_prune)} ({space_to_free:.1f} MB)")

    if args.dry_run:
        print("\n[DRY RUN]")
        prune_models(to_prune, archive=args.archive, dry_run=True)
        print(f"\nWould free {space_to_free:.1f} MB")
        return

    # Confirmation
    if not args.force:
        action = "archive" if args.archive else "DELETE"
        confirm = input(f"\n{action} {len(to_prune)} models? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    print(f"\n{'Archiving' if args.archive else 'Deleting'} models...")
    count, space = prune_models(to_prune, archive=args.archive, dry_run=False)
    print(f"\nPruned {count} models, freed {space:.1f} MB")


if __name__ == "__main__":
    main()
