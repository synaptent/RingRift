#!/usr/bin/env python3
"""Elo Progress Report - Demonstrate iterative NN strength improvement.

January 2026 - Created to provide evidence of training loop effectiveness.

This script generates a comprehensive report showing:
- Starting vs current Elo for each config
- Elo improvement (delta) over time
- Training iterations (generations) completed
- Progress towards 2000 Elo target
- Win rates vs baselines (random, heuristic)

Usage:
    python scripts/elo_progress_report.py
    python scripts/elo_progress_report.py --config hex8_2p
    python scripts/elo_progress_report.py --format json
    python scripts/elo_progress_report.py --days 7
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Target Elo for all configs
TARGET_ELO = 2000.0

# Starting Elo (baseline before any training)
BASELINE_ELO = 1000.0

# All 12 canonical configurations
ALL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


@dataclass
class HistoricalDataPoint:
    """A single historical data point for charting."""
    timestamp: str  # ISO format date string
    elo: float


@dataclass
class ConfigProgress:
    """Progress data for a single configuration."""
    config_key: str
    starting_elo: float
    current_elo: float
    delta: float
    iterations: int  # Training generations
    target_elo: float
    progress_percent: float
    is_improving: bool
    vs_random_rate: float | None
    vs_heuristic_rate: float | None
    games_played: int
    last_updated: str | None
    # January 21, 2026: Added real historical data points for charting
    # Previously the dashboard fabricated fake data which was misleading
    history: list[HistoricalDataPoint] | None = None
    # Feb 24, 2026: Trend indicator for recent snapshots (last 7 days)
    # Distinguishes "was good but plateaued" from "steadily improving"
    trend: str | None = None  # "improving", "stable", "declining"


@dataclass
class OverallProgress:
    """Summary progress across all configs."""
    total_iterations: int
    configs_improving: int
    configs_regressing: int
    configs_at_target: int
    avg_elo: float
    avg_delta: float
    avg_elo_gain_per_iteration: float | None


@dataclass
class ProgressReport:
    """Complete progress report."""
    configs: dict[str, ConfigProgress]
    overall: OverallProgress
    generated_at: str


def get_config_progress(config_key: str, days: float = 30.0) -> ConfigProgress:
    """Get progress data for a single config."""
    from app.coordination.elo_progress_tracker import get_elo_progress_tracker
    from app.coordination.generation_tracker import get_generation_tracker

    # Parse config key
    parts = config_key.replace("_", " ").split()
    if len(parts) < 2:
        board_type = config_key
        num_players = 2
    else:
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

    # Get Elo progress
    elo_tracker = get_elo_progress_tracker()
    elo_report = elo_tracker.get_progress_report(config_key, days=days)

    # Get generation count
    gen_tracker = get_generation_tracker()
    generations = gen_tracker.get_all_generations(board_type=board_type)
    config_generations = [g for g in generations if g.num_players == num_players]
    iterations = len(config_generations)

    # Get latest snapshot for win rates
    latest = elo_tracker.get_latest_snapshot(config_key)

    # Determine starting and current Elo
    if elo_report.start_elo is not None:
        starting_elo = elo_report.start_elo
    elif config_generations:
        # Try to get from first generation
        starting_elo = BASELINE_ELO
    else:
        starting_elo = BASELINE_ELO

    # Feb 24, 2026: Use max Elo in the window instead of last snapshot.
    # The last snapshot may have lower Elo due to sampling noise or different
    # evaluation conditions, causing phantom "REGRESSED" reports even when the
    # best model quality is steadily improving.
    if elo_report.snapshots:
        current_elo = max(s.best_elo for s in elo_report.snapshots)
    elif elo_report.end_elo is not None:
        current_elo = elo_report.end_elo
    elif latest:
        current_elo = latest.best_elo
    else:
        current_elo = BASELINE_ELO

    delta = current_elo - starting_elo
    progress_percent = ((current_elo - BASELINE_ELO) / (TARGET_ELO - BASELINE_ELO)) * 100
    progress_percent = max(0, min(100, progress_percent))

    last_updated = None
    if elo_report.end_time:
        last_updated = elo_report.end_time.strftime("%Y-%m-%d %H:%M")
    elif latest:
        last_updated = datetime.fromtimestamp(latest.timestamp).strftime("%Y-%m-%d %H:%M")

    # January 21, 2026: Extract real historical data points from snapshots
    # This replaces the previous fabricated/synthesized data in the dashboard
    history: list[HistoricalDataPoint] = []
    for snapshot in elo_report.snapshots:
        history.append(HistoricalDataPoint(
            timestamp=datetime.fromtimestamp(snapshot.timestamp).strftime("%Y-%m-%d"),
            elo=snapshot.best_elo,
        ))

    # Feb 24, 2026: Compute trend from recent snapshots (last 7 days)
    trend: str | None = None
    if elo_report.snapshots:
        import time as _time
        seven_days_ago = _time.time() - 7 * 86400
        recent = [s for s in elo_report.snapshots if s.timestamp >= seven_days_ago]
        if len(recent) >= 2:
            first_half = recent[: len(recent) // 2]
            second_half = recent[len(recent) // 2 :]
            avg_first = sum(s.best_elo for s in first_half) / len(first_half)
            avg_second = sum(s.best_elo for s in second_half) / len(second_half)
            elo_diff = avg_second - avg_first
            if elo_diff > 10:
                trend = "improving"
            elif elo_diff < -10:
                trend = "declining"
            else:
                trend = "stable"

    return ConfigProgress(
        config_key=config_key,
        starting_elo=starting_elo,
        current_elo=current_elo,
        delta=delta,
        iterations=iterations,
        target_elo=TARGET_ELO,
        progress_percent=progress_percent,
        is_improving=delta > 0,
        vs_random_rate=latest.vs_random_win_rate if latest else None,
        vs_heuristic_rate=latest.vs_heuristic_win_rate if latest else None,
        games_played=latest.games_played if latest else 0,
        last_updated=last_updated,
        history=history if history else None,
        trend=trend,
    )


def get_full_report(days: float = 30.0, config_filter: str | None = None) -> ProgressReport:
    """Generate the full progress report."""
    configs: dict[str, ConfigProgress] = {}

    for board_type, num_players in ALL_CONFIGS:
        config_key = f"{board_type}_{num_players}p"
        if config_filter and config_key != config_filter:
            continue
        configs[config_key] = get_config_progress(config_key, days=days)

    # Calculate overall stats
    total_iterations = sum(c.iterations for c in configs.values())
    configs_improving = sum(1 for c in configs.values() if c.is_improving)
    configs_regressing = sum(1 for c in configs.values() if c.delta < 0)
    configs_at_target = sum(1 for c in configs.values() if c.current_elo >= TARGET_ELO)

    elos = [c.current_elo for c in configs.values()]
    deltas = [c.delta for c in configs.values()]
    avg_elo = sum(elos) / len(elos) if elos else 0
    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    # Average Elo gain per iteration (across configs with iterations)
    gains_per_iter = []
    for c in configs.values():
        if c.iterations > 0 and c.delta > 0:
            gains_per_iter.append(c.delta / c.iterations)
    avg_gain_per_iter = sum(gains_per_iter) / len(gains_per_iter) if gains_per_iter else None

    overall = OverallProgress(
        total_iterations=total_iterations,
        configs_improving=configs_improving,
        configs_regressing=configs_regressing,
        configs_at_target=configs_at_target,
        avg_elo=avg_elo,
        avg_delta=avg_delta,
        avg_elo_gain_per_iteration=avg_gain_per_iter,
    )

    return ProgressReport(
        configs=configs,
        overall=overall,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def print_table(report: ProgressReport) -> None:
    """Print report as formatted table."""
    print()
    print("=" * 105)
    print("ELO PROGRESS REPORT - Demonstrating Iterative NN Strength Improvement")
    print("=" * 105)
    print(f"Generated: {report.generated_at} | Target: {TARGET_ELO} Elo")
    print()

    # Table header
    print(f"{'Config':<15} {'Start':>8} {'Current':>8} {'Delta':>8} {'Iters':>6} {'Progress':>10} {'Status':<12} {'Trend':<10}")
    print("-" * 105)

    # Sort by current Elo descending
    sorted_configs = sorted(report.configs.items(), key=lambda x: x[1].current_elo, reverse=True)

    for config_key, cfg in sorted_configs:
        # Determine status emoji
        if cfg.current_elo >= TARGET_ELO:
            status = "DONE"
        elif cfg.delta > 100:
            status = "STRONG"
        elif cfg.delta > 0:
            status = "IMPROVING"
        elif cfg.delta == 0:
            status = "STALLED"
        else:
            status = "REGRESSED"

        delta_str = f"+{cfg.delta:.0f}" if cfg.delta >= 0 else f"{cfg.delta:.0f}"
        progress_str = f"{cfg.progress_percent:.1f}%"

        trend_str = cfg.trend or "-"

        print(
            f"{config_key:<15} "
            f"{cfg.starting_elo:>8.0f} "
            f"{cfg.current_elo:>8.0f} "
            f"{delta_str:>8} "
            f"{cfg.iterations:>6} "
            f"{progress_str:>10} "
            f"{status:<12} "
            f"{trend_str:<10}"
        )

    # Summary
    print("-" * 105)
    overall = report.overall
    print(f"{'SUMMARY':<15} {'':>8} {overall.avg_elo:>8.0f} {overall.avg_delta:>+8.0f} {overall.total_iterations:>6}")
    print()
    print(f"Configs improving: {overall.configs_improving}/12")
    print(f"Configs at target: {overall.configs_at_target}/12")
    if overall.avg_elo_gain_per_iteration:
        print(f"Avg Elo gain per iteration: +{overall.avg_elo_gain_per_iteration:.1f}")
    print("=" * 105)
    print()


def print_json(report: ProgressReport) -> None:
    """Print report as JSON."""
    # Convert to dict for JSON serialization
    data = {
        "configs": {k: asdict(v) for k, v in report.configs.items()},
        "overall": asdict(report.overall),
        "generated_at": report.generated_at,
    }
    print(json.dumps(data, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Elo progress report demonstrating iterative NN improvement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0].strip(),
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Filter to specific config (e.g., hex8_2p)",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=30.0,
        help="Look back period in days (default: 30)",
    )

    args = parser.parse_args()

    try:
        report = get_full_report(days=args.days, config_filter=args.config)

        if args.format == "json":
            print_json(report)
        else:
            print_table(report)

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
