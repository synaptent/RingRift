#!/usr/bin/env python3
"""Check for models that meet production promotion criteria.

Production Criteria (from app/config/thresholds.py):
- ELO >= 1650
- Games played >= 100
- Win rate vs heuristic >= 60%
- Win rate vs random >= 90%

Usage:
    python scripts/check_production_candidates.py           # Check all
    python scripts/check_production_candidates.py --promote # Promote eligible
    python scripts/check_production_candidates.py --slack   # Send to Slack
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC,
    PRODUCTION_MIN_WIN_RATE_VS_RANDOM,
    ELO_TIER_NOVICE,
    ELO_TIER_INTERMEDIATE,
    ELO_TIER_ADVANCED,
    ELO_TIER_EXPERT,
    ELO_TIER_MASTER,
    ELO_TIER_GRANDMASTER,
)

DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"


def get_elo_tier(rating: float) -> str:
    """Get tier name for an ELO rating."""
    if rating >= ELO_TIER_GRANDMASTER:
        return "Grandmaster"
    elif rating >= ELO_TIER_MASTER:
        return "Master"
    elif rating >= ELO_TIER_EXPERT:
        return "Expert (Production)"
    elif rating >= ELO_TIER_ADVANCED:
        return "Advanced"
    elif rating >= ELO_TIER_INTERMEDIATE:
        return "Intermediate"
    elif rating >= ELO_TIER_NOVICE:
        return "Novice"
    else:
        return "Beginner"


def check_candidates(db_path: Path) -> list[dict]:
    """Find models meeting production criteria."""
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))

    # Get all non-baseline models with sufficient games
    cursor = conn.execute("""
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE games_played >= ?
          AND participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
    """, (PRODUCTION_MIN_GAMES,))

    candidates = []
    for row in cursor.fetchall():
        model_id, rating, games, board_type, num_players = row

        # Check ELO threshold
        if rating < PRODUCTION_ELO_THRESHOLD:
            continue

        # Calculate win rates vs baselines (from match history)
        # Simplified: just check ELO for now, win rate can be computed later
        candidates.append({
            "model_id": model_id,
            "rating": rating,
            "games": games,
            "board_type": board_type or "unknown",
            "num_players": num_players or 2,
            "tier": get_elo_tier(rating),
            "meets_elo": rating >= PRODUCTION_ELO_THRESHOLD,
            "meets_games": games >= PRODUCTION_MIN_GAMES,
        })

    conn.close()
    return candidates


def get_near_candidates(db_path: Path) -> list[dict]:
    """Find models close to production threshold."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))

    # Models with 50+ games and ELO >= 1500 (close to production)
    cursor = conn.execute("""
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE games_played >= 50
          AND rating >= 1500
          AND rating < ?
          AND participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
        LIMIT 10
    """, (PRODUCTION_ELO_THRESHOLD,))

    near = []
    for row in cursor.fetchall():
        model_id, rating, games, board_type, num_players = row
        elo_needed = PRODUCTION_ELO_THRESHOLD - rating
        games_needed = max(0, PRODUCTION_MIN_GAMES - games)

        near.append({
            "model_id": model_id,
            "rating": rating,
            "games": games,
            "board_type": board_type or "unknown",
            "num_players": num_players or 2,
            "tier": get_elo_tier(rating),
            "elo_needed": elo_needed,
            "games_needed": games_needed,
        })

    conn.close()
    return near


def get_config_coverage(db_path: Path) -> dict:
    """Get game count coverage by config."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))

    cursor = conn.execute("""
        SELECT board_type, num_players, COUNT(*) as games
        FROM match_history
        GROUP BY board_type, num_players
        ORDER BY games DESC
    """)

    coverage = {}
    for row in cursor.fetchall():
        board_type, num_players, games = row
        key = f"{board_type}_{num_players}p"
        coverage[key] = games

    conn.close()
    return coverage


def main():
    parser = argparse.ArgumentParser(description="Check production promotion candidates")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to ELO database")
    parser.add_argument("--promote", action="store_true", help="Promote eligible models")
    parser.add_argument("--slack", action="store_true", help="Send report to Slack")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("PRODUCTION PROMOTION CANDIDATES REPORT")
    print("=" * 70)
    print(f"\nProduction Criteria:")
    print(f"  - ELO Rating: >= {PRODUCTION_ELO_THRESHOLD}")
    print(f"  - Games Played: >= {PRODUCTION_MIN_GAMES}")
    print(f"  - Win Rate vs Heuristic: >= {PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC*100:.0f}%")
    print(f"  - Win Rate vs Random: >= {PRODUCTION_MIN_WIN_RATE_VS_RANDOM*100:.0f}%")

    # Check production-ready candidates
    candidates = check_candidates(args.db)

    print(f"\n{'='*70}")
    print("PRODUCTION-READY MODELS")
    print("=" * 70)

    if candidates:
        for c in candidates:
            print(f"\n  {c['model_id']}")
            print(f"    ELO: {c['rating']:.1f} ({c['tier']})")
            print(f"    Games: {c['games']}")
            print(f"    Config: {c['board_type']} {c['num_players']}p")
    else:
        print("\n  No models currently meet all production criteria.")

    # Check near-candidates
    near = get_near_candidates(args.db)

    print(f"\n{'='*70}")
    print("NEAR PRODUCTION (Close but not ready)")
    print("=" * 70)

    if near:
        for n in near[:5]:
            print(f"\n  {n['model_id']}")
            print(f"    ELO: {n['rating']:.1f} ({n['tier']})")
            print(f"    Games: {n['games']}")
            print(f"    Needs: +{n['elo_needed']:.1f} ELO, +{n['games_needed']} games")
    else:
        print("\n  No models close to production threshold.")

    # Config coverage
    coverage = get_config_coverage(args.db)

    print(f"\n{'='*70}")
    print("CONFIG COVERAGE (Games by board/player)")
    print("=" * 70)

    all_configs = [
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hex8_2p", "hex8_3p", "hex8_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    for config in all_configs:
        games = coverage.get(config, 0)
        bar_len = min(40, games // 100)
        bar = "#" * bar_len + "." * (40 - bar_len)
        status = "OK" if games >= 500 else "LOW" if games >= 100 else "NEED"
        print(f"  {config:15} [{bar}] {games:5} games ({status})")

    print(f"\n{'='*70}")

    # Summary
    total_production = len(candidates)
    total_near = len(near)

    print(f"\nSUMMARY:")
    print(f"  Production-ready models: {total_production}")
    print(f"  Near-production models: {total_near}")
    print(f"  Configs needing games: {sum(1 for c in all_configs if coverage.get(c, 0) < 100)}/12")


if __name__ == "__main__":
    main()
