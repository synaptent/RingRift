#!/usr/bin/env python3
"""Comprehensive game statistics analysis tool for RingRift selfplay data.

This script analyzes selfplay game data across all board types and player counts,
producing detailed statistics on:
- Victory type distribution
- Win rates by player position
- Game length statistics
- Recovery action usage
- Board type comparisons
- Performance metrics

Usage:
    python scripts/analyze_game_statistics.py --data-dir data/selfplay
    python scripts/analyze_game_statistics.py --data-dir data/selfplay --output report.json
    python scripts/analyze_game_statistics.py --data-dir data/selfplay --format markdown

Output formats:
    - json: Machine-readable JSON report
    - markdown: Human-readable markdown report (default)
    - both: Both formats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class GameStats:
    """Statistics for a single game configuration (board + player count)."""

    board_type: str
    num_players: int
    total_games: int = 0
    total_moves: int = 0
    total_time_seconds: float = 0.0
    wins_by_player: dict[str, int] = field(default_factory=dict)
    victory_types: dict[str, int] = field(default_factory=dict)
    draws: int = 0
    games_with_recovery: int = 0
    total_recovery_opportunities: int = 0
    games_with_fe: int = 0
    game_lengths: list[int] = field(default_factory=list)  # Individual game lengths

    @property
    def moves_per_game(self) -> float:
        return self.total_moves / self.total_games if self.total_games > 0 else 0.0

    @property
    def games_per_second(self) -> float:
        return self.total_games / self.total_time_seconds if self.total_time_seconds > 0 else 0.0

    @property
    def min_moves(self) -> int:
        return min(self.game_lengths) if self.game_lengths else 0

    @property
    def max_moves(self) -> int:
        return max(self.game_lengths) if self.game_lengths else 0

    @property
    def median_moves(self) -> float:
        if not self.game_lengths:
            return 0.0
        sorted_lengths = sorted(self.game_lengths)
        n = len(sorted_lengths)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
        return float(sorted_lengths[mid])

    @property
    def std_moves(self) -> float:
        if len(self.game_lengths) < 2:
            return 0.0
        avg = self.moves_per_game
        variance = sum((x - avg) ** 2 for x in self.game_lengths) / len(self.game_lengths)
        return variance ** 0.5

    def win_rate(self, player: int) -> float:
        wins = self.wins_by_player.get(str(player), 0)
        return wins / self.total_games if self.total_games > 0 else 0.0

    def victory_type_rate(self, vtype: str) -> float:
        count = self.victory_types.get(vtype, 0)
        return count / self.total_games if self.total_games > 0 else 0.0


@dataclass
class AnalysisReport:
    """Complete analysis report across all configurations."""

    stats_by_config: dict[tuple[str, int], GameStats] = field(default_factory=dict)
    recovery_analysis: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_sources: list[str] = field(default_factory=list)

    def total_games(self) -> int:
        return sum(s.total_games for s in self.stats_by_config.values())

    def total_moves(self) -> int:
        return sum(s.total_moves for s in self.stats_by_config.values())


def load_stats_json(path: Path) -> dict[str, Any] | None:
    """Load a stats.json file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return None


def load_recovery_analysis(path: Path) -> dict[str, Any] | None:
    """Load recovery analysis results if available."""
    candidates = [
        path / "recovery_analysis_results.json",
        path / "actual_recovery_opportunities.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def collect_stats(data_dir: Path) -> AnalysisReport:
    """Collect statistics from all subdirectories in data_dir."""
    report = AnalysisReport()

    # Find all stats.json files
    for stats_path in data_dir.rglob("stats.json"):
        data = load_stats_json(stats_path)
        if data is None:
            continue

        board_type = data.get("board_type", "unknown")
        num_players = data.get("num_players", 2)
        key = (board_type, num_players)

        if key not in report.stats_by_config:
            report.stats_by_config[key] = GameStats(board_type=board_type, num_players=num_players)

        stats = report.stats_by_config[key]
        stats.total_games += data.get("total_games", 0)
        stats.total_moves += data.get("total_moves", 0)
        stats.total_time_seconds += data.get("total_time_seconds", 0.0)
        stats.draws += data.get("draws", 0)
        stats.games_with_recovery += data.get("games_with_recovery_opportunities", 0)
        stats.total_recovery_opportunities += data.get("total_recovery_opportunities", 0)
        stats.games_with_fe += data.get("games_with_fe", 0)

        # Merge wins by player
        for player, wins in data.get("wins_by_player", {}).items():
            stats.wins_by_player[player] = stats.wins_by_player.get(player, 0) + wins

        # Merge victory types
        victory_types = data.get("victory_type_counts", data.get("victory_types", {}))
        for vtype, count in victory_types.items():
            stats.victory_types[vtype] = stats.victory_types.get(vtype, 0) + count

        # Collect individual game lengths if available
        game_lengths = data.get("game_lengths", [])
        if game_lengths:
            stats.game_lengths.extend(game_lengths)

        report.data_sources.append(str(stats_path))

    # Load recovery analysis if available
    recovery_data = load_recovery_analysis(data_dir)
    if recovery_data:
        report.recovery_analysis = recovery_data

    return report


def generate_markdown_report(report: AnalysisReport) -> str:
    """Generate a markdown report from the analysis."""
    lines = []
    lines.append("# RingRift Game Statistics Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Total Games Analyzed:** {report.total_games():,}")
    lines.append(f"**Total Moves:** {report.total_moves():,}")
    lines.append(f"**Data Sources:** {len(report.data_sources)}")
    lines.append("")

    # Victory Type Distribution
    lines.append("## 1. Victory Type Distribution")
    lines.append("")
    lines.append("| Board/Players | Games | Territory | LPS | Stalemate | Ring Elim | Draw |")
    lines.append("|---------------|-------|-----------|-----|-----------|-----------|------|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        territory = stats.victory_types.get("territory", 0)
        lps = stats.victory_types.get("lps", 0)
        stalemate = stats.victory_types.get("stalemate", 0)
        ring_elim = stats.victory_types.get("ring_elimination", 0)

        territory_pct = f"{100 * territory / stats.total_games:.1f}%" if territory else "0%"
        lps_pct = f"{100 * lps / stats.total_games:.1f}%" if lps else "0%"
        stalemate_pct = f"{100 * stalemate / stats.total_games:.1f}%" if stalemate else "0%"
        ring_elim_pct = f"{100 * ring_elim / stats.total_games:.1f}%" if ring_elim else "0%"
        draw_pct = f"{100 * stats.draws / stats.total_games:.1f}%" if stats.draws else "0%"

        lines.append(
            f"| {board_type} {num_players}p | {stats.total_games} | "
            f"{territory_pct} ({territory}) | {lps_pct} ({lps}) | "
            f"{stalemate_pct} ({stalemate}) | {ring_elim_pct} ({ring_elim}) | {draw_pct} |"
        )

    lines.append("")

    # Win Distribution by Player Position
    lines.append("## 2. Win Distribution by Player Position")
    lines.append("")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        lines.append(f"### {board_type.title()} {num_players}-player ({stats.total_games} games)")
        lines.append("")
        lines.append("| Player | Wins | Win Rate |")
        lines.append("|--------|------|----------|")

        for p in range(1, num_players + 1):
            wins = stats.wins_by_player.get(str(p), 0)
            rate = 100 * stats.win_rate(p)
            lines.append(f"| Player {p} | {wins} | {rate:.1f}% |")

        # Calculate expected win rate and deviation
        expected = 100 / num_players
        lines.append("")
        lines.append(f"*Expected win rate (uniform): {expected:.1f}%*")
        lines.append("")

    # Game Length Statistics
    lines.append("## 3. Game Length Statistics")
    lines.append("")
    lines.append("| Configuration | Avg Moves/Game | Games/Second | Total Time |")
    lines.append("|--------------|----------------|--------------|------------|")

    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games == 0:
            continue
        avg_moves = stats.moves_per_game
        gps = stats.games_per_second
        total_time = stats.total_time_seconds

        time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time / 60:.1f}m"
        lines.append(f"| {board_type} {num_players}p | {avg_moves:.1f} | {gps:.3f} | {time_str} |")

    lines.append("")

    # Recovery Action Analysis
    if report.recovery_analysis:
        lines.append("## 4. Recovery Action Analysis")
        lines.append("")
        ra = report.recovery_analysis

        total = ra.get("total_games", 0)
        with_fe = ra.get("games_with_fe", 0)
        with_recovery = ra.get("games_with_recovery", 0)
        states = ra.get("states_analyzed", 0)

        lines.append(f"- **Games Analyzed:** {total}")
        lines.append(f"- **Games with Forced Elimination:** {with_fe} ({100 * with_fe / total:.1f}%)" if total else "")
        lines.append(f"- **Games with Recovery Used:** {with_recovery}")
        lines.append(f"- **States Analyzed:** {states:,}")
        lines.append("")

        # Condition frequencies
        cond_freq = ra.get("condition_frequencies", {})
        if cond_freq:
            lines.append("### Recovery Condition Frequencies")
            lines.append("")
            lines.append("| Condition | Frequency | % of States |")
            lines.append("|-----------|-----------|-------------|")
            for cond, count in sorted(cond_freq.items(), key=lambda x: -x[1]):
                pct = 100 * count / states if states else 0
                lines.append(f"| {cond} | {count:,} | {pct:.1f}% |")
            lines.append("")

        # Conditions met distribution
        cond_dist = ra.get("conditions_met_distribution", {})
        if cond_dist:
            lines.append("### Conditions Met Distribution")
            lines.append("")
            lines.append("| # Conditions | States | % |")
            lines.append("|--------------|--------|---|")
            for key, count in sorted(cond_dist.items()):
                pct = 100 * count / states if states else 0
                lines.append(f"| {key} | {count:,} | {pct:.1f}% |")
            lines.append("")

    # Key Findings
    lines.append("## 5. Key Findings")
    lines.append("")

    # Position advantage analysis
    position_advantages = []
    for (board_type, num_players), stats in report.stats_by_config.items():
        if stats.total_games < 10:
            continue
        expected = 1.0 / num_players
        max_player = None
        max_rate = 0.0
        for p in range(1, num_players + 1):
            rate = stats.win_rate(p)
            if rate > max_rate:
                max_rate = rate
                max_player = p
        if max_player and max_rate > expected + 0.05:
            position_advantages.append((board_type, num_players, max_player, max_rate))

    if position_advantages:
        lines.append("### Position Advantages")
        for board, players, player, rate in position_advantages:
            lines.append(f"- **{board} {players}p:** Player {player} has advantage ({rate * 100:.1f}% win rate)")
        lines.append("")

    # Victory type patterns
    lines.append("### Victory Type Patterns")
    for (board_type, num_players), stats in sorted(report.stats_by_config.items()):
        if stats.total_games < 10:
            continue
        dominant = max(stats.victory_types.items(), key=lambda x: x[1], default=(None, 0))
        if dominant[0] and dominant[1] > 0:
            rate = 100 * dominant[1] / stats.total_games
            lines.append(f"- **{board_type} {num_players}p:** {dominant[0]} dominant ({rate:.0f}%)")
    lines.append("")

    # Recovery status
    if report.recovery_analysis:
        with_recovery = report.recovery_analysis.get("games_with_recovery", 0)
        conditions_met = report.recovery_analysis.get("conditions_met_distribution", {}).get("4_conditions", 0)
        lines.append("### Recovery Mechanic Status")
        if with_recovery == 0 and conditions_met > 0:
            lines.append(
                f"- **Recovery is NOT being used** despite {conditions_met} states meeting all 4 conditions"
            )
            lines.append("- This suggests turn-skipping prevents recovery-eligible players from taking turns")
        elif with_recovery > 0:
            lines.append(f"- Recovery was used in {with_recovery} games")
        lines.append("")

    return "\n".join(lines)


def generate_json_report(report: AnalysisReport) -> dict[str, Any]:
    """Generate a JSON-serializable report."""
    result: dict[str, Any] = {
        "timestamp": report.timestamp,
        "summary": {
            "total_games": report.total_games(),
            "total_moves": report.total_moves(),
            "data_sources": len(report.data_sources),
        },
        "configurations": {},
        "recovery_analysis": report.recovery_analysis,
    }

    for (board_type, num_players), stats in report.stats_by_config.items():
        key = f"{board_type}_{num_players}p"
        result["configurations"][key] = {
            "board_type": board_type,
            "num_players": num_players,
            "total_games": stats.total_games,
            "total_moves": stats.total_moves,
            "moves_per_game": stats.moves_per_game,
            "total_time_seconds": stats.total_time_seconds,
            "games_per_second": stats.games_per_second,
            "wins_by_player": stats.wins_by_player,
            "win_rates": {str(p): stats.win_rate(p) for p in range(1, num_players + 1)},
            "victory_types": stats.victory_types,
            "victory_type_rates": {
                vtype: stats.victory_type_rate(vtype) for vtype in stats.victory_types
            },
            "draws": stats.draws,
            "games_with_recovery": stats.games_with_recovery,
            "games_with_fe": stats.games_with_fe,
        }

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze RingRift selfplay game statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/selfplay"),
        help="Directory containing selfplay data (default: data/selfplay)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: stdout for markdown, auto-named for json)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Analyzing data in {args.data_dir}...", file=sys.stderr)

    report = collect_stats(args.data_dir)

    if report.total_games() == 0:
        print("Error: No game data found", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Found {report.total_games()} games across {len(report.stats_by_config)} configurations", file=sys.stderr)

    # Generate outputs
    if args.format in ("markdown", "both"):
        md_report = generate_markdown_report(report)
        if args.output and args.format == "markdown":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_report)
            if not args.quiet:
                print(f"Markdown report written to {args.output}", file=sys.stderr)
        elif args.format == "both" and args.output:
            md_path = args.output.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_report)
            if not args.quiet:
                print(f"Markdown report written to {md_path}", file=sys.stderr)
        else:
            print(md_report)

    if args.format in ("json", "both"):
        json_report = generate_json_report(report)
        if args.output:
            json_path = args.output if args.format == "json" else args.output.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_report, f, indent=2)
            if not args.quiet:
                print(f"JSON report written to {json_path}", file=sys.stderr)
        elif args.format == "json":
            print(json.dumps(json_report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
