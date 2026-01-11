#!/usr/bin/env python3
"""Audit stale Elo entries and generate reports.

January 2026: Created to identify Elo entries lacking harness metadata,
entries with no games, and entries for missing models.

Usage:
    # Show statistics only
    python scripts/audit_stale_elo.py --stats

    # Full report to stdout
    python scripts/audit_stale_elo.py --verbose

    # Output to JSON for analysis
    python scripts/audit_stale_elo.py --output-json stale_audit.json

    # Check specific config
    python scripts/audit_stale_elo.py --config square8_2p
"""

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EloEntry:
    """Single Elo rating entry."""
    participant_id: str
    board_type: str
    num_players: int
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    last_update: float

    # Derived fields
    has_harness_info: bool = False
    harness_type: str | None = None
    simulation_count: int | None = None
    age_days: float = 0.0
    model_exists: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def composite_id(self) -> bool:
        """Check if this is a composite ID with harness info."""
        return ":" in self.participant_id


@dataclass
class AuditReport:
    """Audit report summary."""
    timestamp: str
    database_path: str
    total_entries: int = 0
    entries_with_games: int = 0
    entries_no_games: int = 0
    entries_with_harness: int = 0
    entries_without_harness: int = 0
    entries_older_than_14d: int = 0
    entries_older_than_30d: int = 0
    phantom_entries: int = 0

    # By category
    by_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    stale_entries: list[dict] = field(default_factory=list)
    top_models: list[dict] = field(default_factory=list)


def parse_composite_id(participant_id: str) -> tuple[str, str | None, int | None]:
    """Parse composite participant ID into components.

    Format: model_name:harness_type:bN or model_name:harness_type
    Examples:
        canonical_hex8_2p:gumbel_mcts:b800 -> (canonical_hex8_2p, gumbel_mcts, 800)
        canonical_hex8_2p:gumbel_mcts -> (canonical_hex8_2p, gumbel_mcts, None)
        canonical_hex8_2p -> (canonical_hex8_2p, None, None)
    """
    parts = participant_id.split(":")
    model_name = parts[0]
    harness_type = None
    simulation_count = None

    if len(parts) >= 2:
        harness_type = parts[1]

    if len(parts) >= 3 and parts[2].startswith("b"):
        try:
            simulation_count = int(parts[2][1:])
        except ValueError:
            pass

    return model_name, harness_type, simulation_count


def find_model_file(model_name: str, search_dirs: list[Path]) -> Path | None:
    """Search for model file in common directories."""
    # Common model file patterns
    patterns = [
        f"{model_name}.pth",
        f"{model_name}.pt",
        f"canonical_{model_name}.pth",
        f"ringrift_best_{model_name}.pth",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            path = search_dir / pattern
            if path.exists():
                return path

    return None


def audit_elo_database(
    db_path: Path,
    model_dirs: list[Path] | None = None,
    config_filter: str | None = None,
    max_age_days: int = 14,
) -> AuditReport:
    """Audit Elo database for stale and problematic entries.

    Args:
        db_path: Path to unified_elo.db
        model_dirs: Directories to search for model files
        config_filter: Only audit specific config (e.g., "square8_2p")
        max_age_days: Threshold for "stale" entries

    Returns:
        AuditReport with findings
    """
    if model_dirs is None:
        model_dirs = [
            Path("models"),
            Path("models_essential"),
            Path.home() / "ringrift" / "ai-service" / "models",
        ]

    now = datetime.now()
    cutoff_14d = (now - timedelta(days=14)).timestamp()
    cutoff_30d = (now - timedelta(days=30)).timestamp()

    report = AuditReport(
        timestamp=now.isoformat(),
        database_path=str(db_path),
    )

    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return report

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Detect schema - some DBs use model_id, others use participant_id
    cur = conn.execute("PRAGMA table_info(elo_ratings)")
    columns = {row["name"] for row in cur.fetchall()}
    id_column = "model_id" if "model_id" in columns else "participant_id"

    # Build query
    query = f"""
        SELECT
            {id_column} as participant_id, board_type, num_players,
            rating, games_played, wins, losses, draws,
            last_update
        FROM elo_ratings
    """
    params = []

    if config_filter:
        parts = config_filter.split("_")
        if len(parts) >= 2:
            board = "_".join(parts[:-1])
            players = int(parts[-1].replace("p", ""))
            query += " WHERE board_type = ? AND num_players = ?"
            params = [board, players]

    query += " ORDER BY rating DESC"

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    for row in rows:
        entry = EloEntry(
            participant_id=row["participant_id"],
            board_type=row["board_type"],
            num_players=row["num_players"],
            rating=row["rating"],
            games_played=row["games_played"],
            wins=row["wins"],
            losses=row["losses"],
            draws=row["draws"],
            last_update=row["last_update"] or 0,
        )

        # Parse composite ID
        model_name, harness_type, sim_count = parse_composite_id(entry.participant_id)
        entry.harness_type = harness_type
        entry.simulation_count = sim_count
        entry.has_harness_info = harness_type is not None

        # Calculate age
        if entry.last_update > 0:
            entry.age_days = (now.timestamp() - entry.last_update) / 86400

        # Check for issues
        if entry.games_played == 0:
            entry.issues.append("no_games")

        if not entry.has_harness_info and entry.games_played > 0:
            entry.issues.append("missing_harness")

        if entry.last_update < cutoff_14d and entry.last_update > 0:
            entry.issues.append("stale_14d")

        if entry.last_update < cutoff_30d and entry.last_update > 0:
            entry.issues.append("stale_30d")

        # Check if model file exists (skip baselines)
        if not model_name.startswith(("random", "heuristic", "baseline")):
            model_path = find_model_file(model_name, model_dirs)
            if model_path is None:
                entry.model_exists = False
                entry.issues.append("phantom")

        # Update report totals
        report.total_entries += 1

        if entry.games_played > 0:
            report.entries_with_games += 1
        else:
            report.entries_no_games += 1

        if entry.has_harness_info:
            report.entries_with_harness += 1
        else:
            report.entries_without_harness += 1

        if "stale_14d" in entry.issues:
            report.entries_older_than_14d += 1

        if "stale_30d" in entry.issues:
            report.entries_older_than_30d += 1

        if "phantom" in entry.issues:
            report.phantom_entries += 1

        # Track by config
        config_key = f"{entry.board_type}_{entry.num_players}p"
        if config_key not in report.by_config:
            report.by_config[config_key] = {
                "total": 0,
                "with_games": 0,
                "with_harness": 0,
                "top_rating": 0,
                "top_model": None,
            }

        cfg = report.by_config[config_key]
        cfg["total"] += 1
        if entry.games_played > 0:
            cfg["with_games"] += 1
        if entry.has_harness_info:
            cfg["with_harness"] += 1
        if entry.rating > cfg["top_rating"]:
            cfg["top_rating"] = entry.rating
            cfg["top_model"] = entry.participant_id

        # Add to stale entries if has issues
        if entry.issues:
            report.stale_entries.append({
                "participant_id": entry.participant_id,
                "config": config_key,
                "rating": entry.rating,
                "games": entry.games_played,
                "age_days": round(entry.age_days, 1),
                "has_harness": entry.has_harness_info,
                "issues": entry.issues,
            })

        # Track top models per config (top 3)
        if entry.games_played >= 10:
            report.top_models.append({
                "participant_id": entry.participant_id,
                "config": config_key,
                "rating": entry.rating,
                "games": entry.games_played,
                "harness": entry.harness_type,
                "simulations": entry.simulation_count,
                "age_days": round(entry.age_days, 1),
            })

    conn.close()

    # Sort top models by rating
    report.top_models.sort(key=lambda x: x["rating"], reverse=True)
    report.top_models = report.top_models[:20]  # Keep top 20

    return report


def print_report(report: AuditReport, verbose: bool = False) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 70)
    print("ELO DATABASE AUDIT REPORT")
    print("=" * 70)
    print(f"\nDatabase: {report.database_path}")
    print(f"Timestamp: {report.timestamp}")

    print("\n--- SUMMARY ---")
    print(f"Total entries:           {report.total_entries}")
    print(f"  With games > 0:        {report.entries_with_games}")
    print(f"  With games = 0:        {report.entries_no_games}")
    print(f"  With harness info:     {report.entries_with_harness}")
    print(f"  Missing harness info:  {report.entries_without_harness}")
    print(f"  Older than 14 days:    {report.entries_older_than_14d}")
    print(f"  Older than 30 days:    {report.entries_older_than_30d}")
    print(f"  Phantom (no file):     {report.phantom_entries}")

    print("\n--- BY CONFIG ---")
    for config_key, cfg in sorted(report.by_config.items()):
        status = "OK" if cfg["with_games"] > 0 else "NO GAMES"
        harness_status = "OK" if cfg["with_harness"] > 0 else "NO HARNESS"
        print(f"  {config_key:20} total={cfg['total']:3d}  "
              f"games={cfg['with_games']:3d}  harness={cfg['with_harness']:3d}  "
              f"top={cfg['top_rating']:.0f}")

    print("\n--- TOP 10 MODELS ---")
    for i, model in enumerate(report.top_models[:10], 1):
        harness = model.get("harness") or "unknown"
        sims = model.get("simulations")
        sims_str = f"b{sims}" if sims else ""
        print(f"  {i:2d}. {model['participant_id'][:40]:40} "
              f"{model['rating']:7.1f} Elo  "
              f"({model['games']:4d} games, {model['age_days']:.0f}d old)  "
              f"[{harness}{sims_str}]")

    if verbose and report.stale_entries:
        print("\n--- STALE/PROBLEMATIC ENTRIES ---")
        for entry in report.stale_entries[:30]:
            issues = ", ".join(entry["issues"])
            print(f"  {entry['participant_id'][:50]:50} "
                  f"rating={entry['rating']:.0f}  "
                  f"games={entry['games']:4d}  "
                  f"issues: {issues}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit Elo database for stale entries")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/unified_elo.db"),
        help="Path to Elo database (default: data/unified_elo.db)",
    )
    parser.add_argument(
        "--config",
        help="Audit specific config only (e.g., square8_2p)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only (minimal output)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed entry list",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output report to JSON file",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=14,
        help="Threshold for stale entries (default: 14)",
    )

    args = parser.parse_args()

    # Check for archived database if main one is empty/missing
    db_paths = [
        args.db,
        Path("archive/deprecated_elo_dbs_20251229/elo_leaderboard.db"),
        Path("archive/deprecated_elo_dbs_20251229/unified_elo.db"),
    ]

    for db_path in db_paths:
        if db_path.exists() and db_path.stat().st_size > 1000:
            logger.info(f"Using database: {db_path}")
            args.db = db_path
            break

    report = audit_elo_database(
        args.db,
        config_filter=args.config,
        max_age_days=args.max_age_days,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Report written to {args.output_json}")

    if args.stats:
        print(f"Total: {report.total_entries}, "
              f"With games: {report.entries_with_games}, "
              f"With harness: {report.entries_with_harness}, "
              f"Stale (>14d): {report.entries_older_than_14d}")
    else:
        print_report(report, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
