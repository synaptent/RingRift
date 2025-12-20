#!/usr/bin/env python3
"""Composite ELO System Dashboard - Quick status overview.

Usage:
    python scripts/composite_elo_dashboard.py
    python scripts/composite_elo_dashboard.py --board square19
    python scripts/composite_elo_dashboard.py --check-consistency
    python scripts/composite_elo_dashboard.py --check-culling
    python scripts/composite_elo_dashboard.py --full

Sprint 5 Integration: Provides a unified view of the Composite ELO System.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(title: str, char: str = "=") -> None:
    """Print a section header."""
    width = 70
    print(f"\n{char*width}")
    print(f" {title}")
    print(f"{char*width}")


def print_leaderboard(board_type: str, num_players: int, limit: int = 15) -> None:
    """Print the composite leaderboard."""
    from app.training.elo_service import get_elo_service

    elo_service = get_elo_service()
    leaderboard = elo_service.get_composite_leaderboard(
        board_type=board_type,
        num_players=num_players,
        min_games=5,
        limit=limit,
    )

    print_header(f"Top {limit} Composite Participants ({board_type}/{num_players}p)")

    if not leaderboard:
        print("  No participants with sufficient games")
        return

    print(f"{'Rank':<5} {'NN ID':<30} {'Algorithm':<15} {'Elo':>8} {'Games':>7}")
    print("-" * 70)

    for i, entry in enumerate(leaderboard, 1):
        nn_id = entry.get("nn_model_id", "unknown")[:28]
        if len(entry.get("nn_model_id", "")) > 28:
            nn_id += ".."
        algo = entry.get("ai_algorithm", "unknown")[:13]
        elo = entry.get("rating", 1500)
        games = entry.get("games_played", 0)
        print(f"{i:<5} {nn_id:<30} {algo:<15} {elo:>8.0f} {games:>7}")


def print_algorithm_rankings(board_type: str, num_players: int) -> None:
    """Print algorithm rankings aggregated across NNs."""
    from app.tournament.gauntlet_aggregation import aggregate_by_algorithm

    print_header(f"Algorithm Rankings ({board_type}/{num_players}p)", "-")

    summaries = aggregate_by_algorithm(board_type, num_players, min_games=5)

    if not summaries:
        print("  No algorithm data available")
        return

    print(f"{'Algorithm':<20} {'Avg Elo':>10} {'Best Elo':>10} {'NNs':>6} {'Games':>8}")
    print("-" * 60)

    for s in summaries:
        print(f"{s.ai_algorithm:<20} {s.avg_elo:>10.0f} {s.best_elo:>10.0f} "
              f"{s.nn_count:>6} {s.total_games:>8}")


def print_nn_summary(board_type: str, num_players: int, limit: int = 10) -> None:
    """Print NN summary aggregated across algorithms."""
    from app.tournament.gauntlet_aggregation import aggregate_by_nn

    print_header(f"Top {limit} NNs by Best Algorithm ({board_type}/{num_players}p)", "-")

    summaries = aggregate_by_nn(board_type, num_players, min_games=5)[:limit]

    if not summaries:
        print("  No NN data available")
        return

    print(f"{'NN ID':<35} {'Best Algo':<15} {'Best':>7} {'Avg':>7} {'Spread':>7}")
    print("-" * 75)

    for s in summaries:
        nn_short = s.nn_model_id[:33] + ".." if len(s.nn_model_id) > 33 else s.nn_model_id
        print(f"{nn_short:<35} {s.best_algorithm:<15} "
              f"{s.best_elo:>7.0f} {s.avg_elo:>7.0f} {s.elo_spread:>7.0f}")


def print_consistency_report(board_type: str, num_players: int) -> None:
    """Print consistency check report."""
    from app.tournament.consistency_monitor import run_consistency_checks

    print_header(f"Consistency Checks ({board_type}/{num_players}p)", "-")

    report = run_consistency_checks(board_type, num_players)

    status = "HEALTHY" if report.overall_healthy else "ISSUES DETECTED"
    print(f"  Overall Status: {status}")
    print(f"  Timestamp: {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for check in report.checks:
        icon = "✓" if check.passed else ("⚠" if check.severity == "warning" else "✗")
        print(f"  {icon} {check.name}")
        print(f"      {check.message}")

    if report.errors:
        print(f"\n  Errors: {len(report.errors)}")
    if report.warnings:
        print(f"  Warnings: {len(report.warnings)}")


def print_culling_preview(board_type: str, num_players: int) -> None:
    """Print culling preview (dry run)."""
    from app.tournament.composite_culling import run_hierarchical_culling

    print_header(f"Culling Preview ({board_type}/{num_players}p)", "-")

    report = run_hierarchical_culling(board_type, num_players, dry_run=True)

    print(f"  Total Participants: {report.total_participants}")
    print(f"  After Culling: {report.participants_kept}")
    print(f"  Would Cull: {report.total_culled}")
    print()
    print(f"  Level 1 (Weak NNs):       {report.level1_nns_culled} NNs")
    print(f"  Level 2 (Weak Combos):    {report.level2_combinations_culled} combinations")
    print(f"  Level 3 (Standard Elo):   {report.level3_participants_culled} participants")

    if report.level1_culled_nn_ids:
        print(f"\n  NNs marked for Level 1 culling:")
        for nn_id in report.level1_culled_nn_ids[:5]:
            nn_short = nn_id[:50] + "..." if len(nn_id) > 50 else nn_id
            print(f"    - {nn_short}")
        if len(report.level1_culled_nn_ids) > 5:
            print(f"    ... and {len(report.level1_culled_nn_ids) - 5} more")


def print_database_stats(board_type: str, num_players: int) -> None:
    """Print database statistics."""
    from app.training.elo_service import get_database_stats

    print_header(f"Database Statistics", "-")

    stats = get_database_stats()

    print(f"  Total Participants: {stats.get('total_participants', 0)}")
    print(f"  Total Matches: {stats.get('total_matches', 0)}")
    print(f"  Composite Participants: {stats.get('composite_participants', 0)}")
    print(f"  Database Size: {stats.get('db_size_mb', 0):.2f} MB")

    # Config-specific stats
    from app.training.elo_service import get_elo_service
    elo_service = get_elo_service()

    leaderboard = elo_service.get_composite_leaderboard(
        board_type=board_type,
        num_players=num_players,
        min_games=0,
        limit=10000,
    )

    total_games = sum(p.get("games_played", 0) for p in leaderboard)
    avg_games = total_games / len(leaderboard) if leaderboard else 0

    print(f"\n  For {board_type}/{num_players}p:")
    print(f"    Participants: {len(leaderboard)}")
    print(f"    Total Games: {total_games}")
    print(f"    Avg Games/Participant: {avg_games:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Composite ELO System Dashboard"
    )

    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of entries to show in leaderboard"
    )
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="Show consistency check results"
    )
    parser.add_argument(
        "--check-culling",
        action="store_true",
        help="Show culling preview"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full dashboard with all sections"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" COMPOSITE ELO SYSTEM DASHBOARD")
    print(f" Board: {args.board} | Players: {args.players}")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Always show basic stats
    print_database_stats(args.board, args.players)
    print_leaderboard(args.board, args.players, args.limit)

    if args.full or args.check_consistency or args.check_culling:
        print_algorithm_rankings(args.board, args.players)
        print_nn_summary(args.board, args.players, 10)

    if args.full or args.check_consistency:
        print_consistency_report(args.board, args.players)

    if args.full or args.check_culling:
        print_culling_preview(args.board, args.players)

    print("\n" + "=" * 70)
    print(" Dashboard complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
