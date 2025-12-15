#!/usr/bin/env python3
"""League-based training system for RingRift AI models.

Implements a tiered league system where models compete and get promoted/demoted
based on Elo performance. This creates a curriculum of increasingly difficult
opponents, similar to OpenAI Five's league training.

Leagues (from lowest to highest):
- Bronze: New models, random/heuristic opponents
- Silver: Proven models, compete against each other
- Gold: Top performers, elite competition
- Champion: Best models, used for production

Features:
1. Automatic promotion/demotion based on Elo thresholds
2. Match scheduling prioritizes competitive games
3. Models train against opponents from their league + adjacent leagues
4. Prevents overfitting to specific opponents

Usage:
    python scripts/league_training.py --run-season
    python scripts/league_training.py --promote-demote
    python scripts/league_training.py --status
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
LEAGUE_DB_PATH = AI_SERVICE_ROOT / "data" / "league_standings.db"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"


# League configuration
@dataclass
class LeagueConfig:
    name: str
    min_elo: float
    max_elo: float
    promotion_threshold: float  # Elo needed to promote
    demotion_threshold: float   # Elo to trigger demotion
    games_for_evaluation: int   # Games needed before promotion/demotion


LEAGUES = {
    "bronze": LeagueConfig(
        name="Bronze",
        min_elo=800,
        max_elo=1200,
        promotion_threshold=1150,
        demotion_threshold=0,  # Can't demote from Bronze
        games_for_evaluation=20,
    ),
    "silver": LeagueConfig(
        name="Silver",
        min_elo=1100,
        max_elo=1400,
        promotion_threshold=1350,
        demotion_threshold=1100,
        games_for_evaluation=30,
    ),
    "gold": LeagueConfig(
        name="Gold",
        min_elo=1300,
        max_elo=1600,
        promotion_threshold=1550,
        demotion_threshold=1300,
        games_for_evaluation=40,
    ),
    "champion": LeagueConfig(
        name="Champion",
        min_elo=1500,
        max_elo=2500,
        promotion_threshold=float("inf"),  # Can't promote from Champion
        demotion_threshold=1500,
        games_for_evaluation=50,
    ),
}

LEAGUE_ORDER = ["bronze", "silver", "gold", "champion"]


def init_league_db() -> None:
    """Initialize the league database."""
    LEAGUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LEAGUE_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS league_standings (
            model_id TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            league TEXT NOT NULL,
            elo REAL NOT NULL,
            games_in_league INTEGER DEFAULT 0,
            promoted_at TEXT,
            demoted_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (model_id, board_type, num_players)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS league_matches (
            match_id TEXT PRIMARY KEY,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            model_a TEXT NOT NULL,
            model_b TEXT NOT NULL,
            league TEXT NOT NULL,
            winner TEXT,
            played_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_standings_league
        ON league_standings(board_type, num_players, league)
    """)
    conn.commit()
    conn.close()


def get_model_league(
    model_id: str,
    board_type: str,
    num_players: int,
) -> Optional[str]:
    """Get the current league of a model."""
    if not LEAGUE_DB_PATH.exists():
        return None

    conn = sqlite3.connect(str(LEAGUE_DB_PATH))
    cursor = conn.execute(
        "SELECT league FROM league_standings WHERE model_id = ? AND board_type = ? AND num_players = ?",
        (model_id, board_type, num_players)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def assign_league_by_elo(elo: float) -> str:
    """Determine appropriate league based on Elo rating."""
    for league_id in reversed(LEAGUE_ORDER):
        config = LEAGUES[league_id]
        if elo >= config.min_elo:
            return league_id
    return "bronze"


def add_model_to_league(
    model_id: str,
    board_type: str,
    num_players: int,
    elo: float,
    league: Optional[str] = None,
) -> str:
    """Add a model to the league system."""
    init_league_db()

    if league is None:
        league = assign_league_by_elo(elo)

    now = datetime.utcnow().isoformat() + "Z"

    conn = sqlite3.connect(str(LEAGUE_DB_PATH))
    conn.execute("""
        INSERT OR REPLACE INTO league_standings
        (model_id, board_type, num_players, league, elo, games_in_league, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, ?, ?)
    """, (model_id, board_type, num_players, league, elo, now, now))
    conn.commit()
    conn.close()

    return league


def get_league_standings(
    board_type: str,
    num_players: int,
    league: Optional[str] = None,
) -> List[Dict]:
    """Get standings for a league or all leagues."""
    if not LEAGUE_DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(LEAGUE_DB_PATH))
    conn.row_factory = sqlite3.Row

    if league:
        cursor = conn.execute("""
            SELECT * FROM league_standings
            WHERE board_type = ? AND num_players = ? AND league = ?
            ORDER BY elo DESC
        """, (board_type, num_players, league))
    else:
        cursor = conn.execute("""
            SELECT * FROM league_standings
            WHERE board_type = ? AND num_players = ?
            ORDER BY league, elo DESC
        """, (board_type, num_players))

    results = [dict(row) for row in cursor]
    conn.close()
    return results


def evaluate_promotions_demotions(
    board_type: str,
    num_players: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Evaluate which models should be promoted or demoted.

    Returns:
        Tuple of (promotions, demotions) where each is a list of dicts with
        model_id, from_league, to_league
    """
    standings = get_league_standings(board_type, num_players)
    promotions = []
    demotions = []

    for standing in standings:
        model_id = standing["model_id"]
        current_league = standing["league"]
        elo = standing["elo"]
        games = standing["games_in_league"]
        config = LEAGUES[current_league]

        # Check if enough games for evaluation
        if games < config.games_for_evaluation:
            continue

        # Check for promotion
        league_idx = LEAGUE_ORDER.index(current_league)
        if elo >= config.promotion_threshold and league_idx < len(LEAGUE_ORDER) - 1:
            new_league = LEAGUE_ORDER[league_idx + 1]
            promotions.append({
                "model_id": model_id,
                "from_league": current_league,
                "to_league": new_league,
                "elo": elo,
            })

        # Check for demotion
        elif elo < config.demotion_threshold and league_idx > 0:
            new_league = LEAGUE_ORDER[league_idx - 1]
            demotions.append({
                "model_id": model_id,
                "from_league": current_league,
                "to_league": new_league,
                "elo": elo,
            })

    return promotions, demotions


def apply_promotions_demotions(
    board_type: str,
    num_players: int,
    dry_run: bool = False,
) -> Dict:
    """Apply promotions and demotions for a board/player config."""
    promotions, demotions = evaluate_promotions_demotions(board_type, num_players)

    if dry_run:
        return {
            "promotions": promotions,
            "demotions": demotions,
            "applied": False,
        }

    now = datetime.utcnow().isoformat() + "Z"
    conn = sqlite3.connect(str(LEAGUE_DB_PATH))

    for p in promotions:
        conn.execute("""
            UPDATE league_standings
            SET league = ?, games_in_league = 0, promoted_at = ?, updated_at = ?
            WHERE model_id = ? AND board_type = ? AND num_players = ?
        """, (p["to_league"], now, now, p["model_id"], board_type, num_players))

    for d in demotions:
        conn.execute("""
            UPDATE league_standings
            SET league = ?, games_in_league = 0, demoted_at = ?, updated_at = ?
            WHERE model_id = ? AND board_type = ? AND num_players = ?
        """, (d["to_league"], now, now, d["model_id"], board_type, num_players))

    conn.commit()
    conn.close()

    return {
        "promotions": promotions,
        "demotions": demotions,
        "applied": True,
    }


def schedule_league_matches(
    board_type: str,
    num_players: int,
    league: str,
    num_matches: int = 10,
) -> List[Tuple[str, str]]:
    """Schedule matches for models in a league.

    Prioritizes matches between models with similar Elo (competitive games).
    Also includes some matches against adjacent leagues.
    """
    standings = get_league_standings(board_type, num_players, league)

    if len(standings) < 2:
        return []

    matches = []
    models = [s["model_id"] for s in standings]

    # Schedule competitive matches within league
    for _ in range(num_matches):
        if len(models) >= 2:
            # Pick two models with similar Elo
            idx1 = random.randint(0, len(models) - 1)
            # Prefer nearby models for competitive games
            offset = random.choice([-2, -1, 1, 2])
            idx2 = max(0, min(len(models) - 1, idx1 + offset))
            if idx2 == idx1:
                idx2 = (idx1 + 1) % len(models)

            matches.append((models[idx1], models[idx2]))

    return matches


def sync_from_elo_db(board_type: str, num_players: int) -> int:
    """Sync models from Elo database to league system.

    Returns number of models synced.
    """
    if not ELO_DB_PATH.exists():
        return 0

    conn = sqlite3.connect(str(ELO_DB_PATH))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT model_id, rating, games_played
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ? AND games_played >= 10
        ORDER BY rating DESC
    """, (board_type, num_players))

    count = 0
    for row in cursor:
        model_id = row["model_id"]
        elo = row["rating"]

        # Only add if not already in league
        if get_model_league(model_id, board_type, num_players) is None:
            add_model_to_league(model_id, board_type, num_players, elo)
            count += 1

    conn.close()
    return count


def print_league_status(board_type: str, num_players: int) -> None:
    """Print current league status."""
    standings = get_league_standings(board_type, num_players)

    print(f"\n{'='*60}")
    print(f"LEAGUE STATUS: {board_type} {num_players}p")
    print(f"{'='*60}\n")

    current_league = None
    for s in standings:
        if s["league"] != current_league:
            current_league = s["league"]
            config = LEAGUES[current_league]
            print(f"\n{config.name.upper()} LEAGUE (Elo {config.min_elo}-{config.max_elo})")
            print("-" * 40)

        print(f"  {s['model_id'][:30]:<30} Elo: {s['elo']:.0f}  Games: {s['games_in_league']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="League training system")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--sync", action="store_true", help="Sync models from Elo database")
    parser.add_argument("--status", action="store_true", help="Show league status")
    parser.add_argument("--promote-demote", action="store_true", help="Apply promotions/demotions")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply changes")
    parser.add_argument("--schedule", type=int, metavar="N", help="Schedule N matches per league")

    args = parser.parse_args()
    board_type = args.board
    num_players = args.players

    init_league_db()

    if args.sync:
        count = sync_from_elo_db(board_type, num_players)
        print(f"Synced {count} models to league system")

    if args.status:
        print_league_status(board_type, num_players)

    if args.promote_demote:
        result = apply_promotions_demotions(board_type, num_players, dry_run=args.dry_run)

        if result["promotions"]:
            print("\nPromotions:")
            for p in result["promotions"]:
                print(f"  {p['model_id']}: {p['from_league']} -> {p['to_league']} (Elo: {p['elo']:.0f})")

        if result["demotions"]:
            print("\nDemotions:")
            for d in result["demotions"]:
                print(f"  {d['model_id']}: {d['from_league']} -> {d['to_league']} (Elo: {d['elo']:.0f})")

        if not result["promotions"] and not result["demotions"]:
            print("No promotions or demotions needed")

        if args.dry_run:
            print("\n(Dry run - no changes applied)")

    if args.schedule:
        print(f"\nScheduling {args.schedule} matches per league:")
        for league_id in LEAGUE_ORDER:
            matches = schedule_league_matches(board_type, num_players, league_id, args.schedule)
            if matches:
                print(f"\n{LEAGUES[league_id].name}:")
                for m1, m2 in matches[:5]:  # Show first 5
                    print(f"  {m1[:20]} vs {m2[:20]}")
                if len(matches) > 5:
                    print(f"  ... and {len(matches) - 5} more")


if __name__ == "__main__":
    main()
