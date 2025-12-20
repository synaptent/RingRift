#!/usr/bin/env python3
"""Per-config model culling - keep top quartile when model count exceeds threshold.

Automatically archives low-performing models to reduce clutter and focus
training resources on the best models.

Usage:
    # Check status (no changes)
    python scripts/model_culling.py --status

    # Dry run (show what would be archived)
    python scripts/model_culling.py --dry-run

    # Execute culling
    python scripts/model_culling.py --cull

    # Cull specific config only
    python scripts/model_culling.py --cull --config square8_2p

    # Custom threshold
    python scripts/model_culling.py --cull --threshold 50 --keep-fraction 0.5
"""

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Add project root
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.tournament.unified_elo_db import EloDatabase
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("model_culling")

# Default paths
MODELS_DIR = AI_SERVICE_ROOT / "models"
ARCHIVE_DIR = MODELS_DIR / "archived"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"

# Culling parameters
DEFAULT_CULL_THRESHOLD = 100  # Cull when > 100 models per config
DEFAULT_KEEP_FRACTION = 0.25  # Keep top 25%
MIN_KEEP_COUNT = 25  # Always keep at least 25 models

# All 9 game configurations
CONFIG_KEYS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


@dataclass
class CullResult:
    """Result of a culling operation."""
    config_key: str
    total_models: int
    kept: int
    archived: int
    archived_paths: list[str]
    threshold: int
    keep_fraction: float


@dataclass
class ModelInfo:
    """Info about a model for culling decisions."""
    participant_id: str
    path: Path
    board_type: str
    num_players: int
    elo: float
    games_played: int


def discover_models_for_config(
    models_dir: Path,
    board_type: str,
    num_players: int,
) -> list[ModelInfo]:
    """Discover all model files for a specific configuration.

    Matches patterns like:
    - sq8_2p_*, square8_2p_*, etc.
    - Checks both filename and sidecar JSON for config
    """
    models = []

    # Pattern variations
    board_patterns = {
        "square8": ["sq8", "square8"],
        "square19": ["sq19", "square19"],
        "hexagonal": ["hex", "hexagonal"],
    }

    patterns = board_patterns.get(board_type, [board_type])
    player_pattern = f"{num_players}p"

    for pth_file in models_dir.glob("*.pth"):
        name_lower = pth_file.stem.lower()

        # Check if model matches this config
        matches_board = any(p in name_lower for p in patterns)
        matches_players = player_pattern in name_lower

        if matches_board and matches_players:
            # Create participant ID
            participant_id = f"model_{pth_file.stem}"

            models.append(ModelInfo(
                participant_id=participant_id,
                path=pth_file,
                board_type=board_type,
                num_players=num_players,
                elo=1500.0,  # Default, will be updated from DB
                games_played=0,
            ))

    return models


def get_models_with_elo(
    models: list[ModelInfo],
    db: EloDatabase,
) -> list[ModelInfo]:
    """Enrich model info with Elo ratings from database."""
    for model in models:
        rating = db.get_rating(
            model.participant_id,
            model.board_type,
            model.num_players
        )
        model.elo = rating.rating
        model.games_played = rating.games_played

    return models


def check_config_status(
    models_dir: Path,
    db: EloDatabase,
    board_type: str,
    num_players: int,
    cull_threshold: int = DEFAULT_CULL_THRESHOLD,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
) -> dict[str, Any]:
    """Check culling status for a specific config."""
    config_key = f"{board_type}_{num_players}p"

    # Discover models
    models = discover_models_for_config(models_dir, board_type, num_players)
    models = get_models_with_elo(models, db)

    total = len(models)
    needs_culling = total > cull_threshold

    # Calculate how many would be kept
    keep_count = max(MIN_KEEP_COUNT, int(total * keep_fraction))
    would_archive = max(0, total - keep_count) if needs_culling else 0

    # Sort by Elo to show top and bottom
    sorted_models = sorted(models, key=lambda m: m.elo, reverse=True)

    return {
        "config_key": config_key,
        "board_type": board_type,
        "num_players": num_players,
        "total_models": total,
        "cull_threshold": cull_threshold,
        "needs_culling": needs_culling,
        "keep_count": keep_count if needs_culling else total,
        "would_archive": would_archive,
        "top_5": [
            {"name": m.path.stem, "elo": m.elo, "games": m.games_played}
            for m in sorted_models[:5]
        ],
        "bottom_5": [
            {"name": m.path.stem, "elo": m.elo, "games": m.games_played}
            for m in sorted_models[-5:]
        ] if len(sorted_models) >= 5 else [],
    }


def archive_model(
    model: ModelInfo,
    archive_dir: Path,
    reason: str = "low_elo_cull",
) -> bool:
    """Move model file to archive directory.

    Also moves associated files like:
    - .json sidecar
    - .onnx export
    - _metadata.json
    """
    config_archive = archive_dir / f"{model.board_type}_{model.num_players}p"
    config_archive.mkdir(parents=True, exist_ok=True)

    try:
        # Move main .pth file
        dest = config_archive / model.path.name
        shutil.move(str(model.path), str(dest))
        logger.info(f"Archived {model.path.name} -> {dest}")

        # Move associated files
        base = model.path.stem
        for suffix in [".json", ".onnx", "_metadata.json"]:
            associated = model.path.parent / f"{base}{suffix}"
            if associated.exists():
                assoc_dest = config_archive / f"{base}{suffix}"
                shutil.move(str(associated), str(assoc_dest))
                logger.debug(f"  Also archived {associated.name}")

        return True

    except Exception as e:
        logger.error(f"Failed to archive {model.path.name}: {e}")
        return False


def cull_config(
    models_dir: Path,
    archive_dir: Path,
    db: EloDatabase,
    board_type: str,
    num_players: int,
    cull_threshold: int = DEFAULT_CULL_THRESHOLD,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    dry_run: bool = False,
) -> CullResult:
    """Cull models for a specific config, keeping top performers."""
    config_key = f"{board_type}_{num_players}p"

    # Discover and enrich models
    models = discover_models_for_config(models_dir, board_type, num_players)
    models = get_models_with_elo(models, db)

    total = len(models)

    # Check if culling needed
    if total <= cull_threshold:
        return CullResult(
            config_key=config_key,
            total_models=total,
            kept=total,
            archived=0,
            archived_paths=[],
            threshold=cull_threshold,
            keep_fraction=keep_fraction,
        )

    # Sort by Elo (highest first)
    sorted_models = sorted(models, key=lambda m: m.elo, reverse=True)

    # Determine keep count
    keep_count = max(MIN_KEEP_COUNT, int(total * keep_fraction))

    # Split into keep and archive
    sorted_models[:keep_count]
    to_archive = sorted_models[keep_count:]

    archived_paths = []

    if not dry_run:
        for model in to_archive:
            if archive_model(model, archive_dir):
                archived_paths.append(str(model.path))

                # Mark as archived in database
                try:
                    conn = db._get_connection()
                    conn.execute("""
                        UPDATE elo_ratings
                        SET archived_at = ?, archive_reason = ?
                        WHERE participant_id = ? AND board_type = ? AND num_players = ?
                    """, (
                        time.time(),
                        "low_elo_cull",
                        model.participant_id,
                        model.board_type,
                        model.num_players,
                    ))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Could not mark {model.participant_id} as archived: {e}")
    else:
        archived_paths = [str(m.path) for m in to_archive]

    return CullResult(
        config_key=config_key,
        total_models=total,
        kept=keep_count,
        archived=len(to_archive),
        archived_paths=archived_paths,
        threshold=cull_threshold,
        keep_fraction=keep_fraction,
    )


def cull_all_configs(
    models_dir: Path = MODELS_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    db_path: Path = ELO_DB_PATH,
    cull_threshold: int = DEFAULT_CULL_THRESHOLD,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    config_filter: str | None = None,
    dry_run: bool = False,
) -> list[CullResult]:
    """Cull models across all configs (or specific config)."""
    db = EloDatabase(db_path)
    results = []

    for board_type, num_players in CONFIG_KEYS:
        config_key = f"{board_type}_{num_players}p"

        # Apply filter if specified
        if config_filter and config_key != config_filter:
            continue

        result = cull_config(
            models_dir=models_dir,
            archive_dir=archive_dir,
            db=db,
            board_type=board_type,
            num_players=num_players,
            cull_threshold=cull_threshold,
            keep_fraction=keep_fraction,
            dry_run=dry_run,
        )

        results.append(result)

    return results


def print_status(
    models_dir: Path = MODELS_DIR,
    db_path: Path = ELO_DB_PATH,
    cull_threshold: int = DEFAULT_CULL_THRESHOLD,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
):
    """Print status of all configs."""
    db = EloDatabase(db_path)

    print()
    print("=" * 70)
    print("MODEL CULLING STATUS")
    print("=" * 70)
    print(f"Cull threshold: {cull_threshold} models")
    print(f"Keep fraction: {keep_fraction:.0%}")
    print(f"Minimum keep: {MIN_KEEP_COUNT}")
    print()

    total_models = 0
    total_would_cull = 0

    for board_type, num_players in CONFIG_KEYS:
        status = check_config_status(
            models_dir, db, board_type, num_players,
            cull_threshold, keep_fraction
        )

        total_models += status["total_models"]
        total_would_cull += status["would_archive"]

        flag = " [NEEDS CULLING]" if status["needs_culling"] else ""
        print(f"{status['config_key']:<15} {status['total_models']:>4} models{flag}")

        if status["top_5"]:
            print(f"  Top:    {status['top_5'][0]['name'][:30]} (Elo: {status['top_5'][0]['elo']:.0f})")
        if status["bottom_5"]:
            print(f"  Bottom: {status['bottom_5'][-1]['name'][:30]} (Elo: {status['bottom_5'][-1]['elo']:.0f})")

        if status["needs_culling"]:
            print(f"  -> Would archive {status['would_archive']} models, keep {status['keep_count']}")
        print()

    print("-" * 70)
    print(f"TOTAL: {total_models} models, {total_would_cull} would be archived")


def main():
    parser = argparse.ArgumentParser(description="Per-config model culling")
    parser.add_argument("--status", action="store_true", help="Show culling status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be archived")
    parser.add_argument("--cull", action="store_true", help="Execute culling")
    parser.add_argument("--config", type=str, help="Filter by config (e.g., square8_2p)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_CULL_THRESHOLD,
                        help=f"Cull threshold (default: {DEFAULT_CULL_THRESHOLD})")
    parser.add_argument("--keep-fraction", type=float, default=DEFAULT_KEEP_FRACTION,
                        help=f"Fraction to keep (default: {DEFAULT_KEEP_FRACTION})")
    parser.add_argument("--models-dir", type=str, help="Models directory")
    parser.add_argument("--db", type=str, help="Elo database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    db_path = Path(args.db) if args.db else ELO_DB_PATH

    if args.status:
        print_status(models_dir, db_path, args.threshold, args.keep_fraction)
        return

    if args.cull or args.dry_run:
        results = cull_all_configs(
            models_dir=models_dir,
            db_path=db_path,
            cull_threshold=args.threshold,
            keep_fraction=args.keep_fraction,
            config_filter=args.config,
            dry_run=args.dry_run,
        )

        print()
        print("=" * 70)
        print(f"CULLING RESULTS {'(DRY RUN)' if args.dry_run else ''}")
        print("=" * 70)

        total_archived = 0
        for r in results:
            if r.archived > 0:
                print(f"{r.config_key}: Archived {r.archived}/{r.total_models} models (kept {r.kept})")
                total_archived += r.archived
            else:
                print(f"{r.config_key}: No culling needed ({r.total_models} models)")

        print()
        print(f"Total archived: {total_archived}")
        if args.dry_run:
            print("(No files were moved - this was a dry run)")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
