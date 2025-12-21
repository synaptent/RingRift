#!/usr/bin/env python3
"""Elo Status Dashboard for RingRift AI.

Tracks progress toward 2000+ Elo across all 12 board/player combinations.

Usage:
    python scripts/elo_status_dashboard.py
    python scripts/elo_status_dashboard.py --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.thresholds import (
    ALL_CONFIG_KEYS,
    ELO_TARGET_ALL_CONFIGS,
    get_elo_gap,
    get_elo_target,
    get_priority_weight,
    is_target_met,
)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "unified_elo.db"


@dataclass
class ConfigStatus:
    config_key: str
    board_type: str
    num_players: int
    current_elo: float
    target_elo: float
    elo_gap: float
    target_met: bool
    games_played: int
    priority_weight: float


def get_config_status(db_path: Path) -> list[ConfigStatus]:
    statuses = []
    conn = None
    if db_path.exists():
        conn = sqlite3.connect(db_path)

    for config_key in ALL_CONFIG_KEYS:
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        current_elo = 1500.0
        games_played = 0

        if conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT rating, games_played FROM elo_ratings "
                    "WHERE board_type=? AND num_players=? ORDER BY rating DESC LIMIT 1",
                    (board_type, num_players),
                )
                row = cur.fetchone()
                if row:
                    current_elo, games_played = row
            except Exception:
                pass

        target = get_elo_target(config_key)
        statuses.append(ConfigStatus(
            config_key=config_key,
            board_type=board_type,
            num_players=num_players,
            current_elo=current_elo,
            target_elo=target,
            elo_gap=get_elo_gap(config_key, current_elo),
            target_met=is_target_met(config_key, current_elo),
            games_played=games_played,
            priority_weight=get_priority_weight(config_key, current_elo),
        ))

    if conn:
        conn.close()
    return sorted(statuses, key=lambda s: s.elo_gap, reverse=True)


def print_dashboard(statuses: list[ConfigStatus]) -> None:
    print("\n" + "=" * 70)
    print(f"{'RingRift Elo Status Dashboard':^70}")
    print(f"{'Target: 2000+ Elo for all 12 configurations':^70}")
    print("=" * 70)

    at_target = sum(1 for s in statuses if s.target_met)
    print(f"\nProgress: {at_target}/12 at target | Avg gap: {sum(s.elo_gap for s in statuses)/12:.0f}")
    print("-" * 70)
    print(f"{'Config':<14} {'Elo':>7} {'Target':>7} {'Gap':>7} {'Priority':>8} {'Status':<10}")
    print("-" * 70)

    for s in statuses:
        status = "✓ DONE" if s.target_met else f"{s.priority_weight:.1f}x"
        print(f"{s.config_key:<14} {s.current_elo:>7.0f} {s.target_elo:>7.0f} "
              f"{s.elo_gap:>7.0f} {s.priority_weight:>8.1f} {status:<10}")
    print("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    statuses = get_config_status(args.db)
    if args.json:
        print(json.dumps([asdict(s) for s in statuses], indent=2))
    else:
        print_dashboard(statuses)


if __name__ == "__main__":
    main()
