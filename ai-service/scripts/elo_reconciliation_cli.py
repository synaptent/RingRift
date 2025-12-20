#!/usr/bin/env python3
"""CLI for Elo Rating Reconciliation.

Provides command-line tools for managing Elo rating consistency across
distributed P2P nodes. Complements the unified_ai_loop.py by providing
manual reconciliation capabilities.

Usage:
    # Check for Elo drift
    python scripts/elo_reconciliation_cli.py check-drift

    # Sync from a specific remote host
    python scripts/elo_reconciliation_cli.py sync --host 192.168.1.100

    # Run full reconciliation across all configured hosts
    python scripts/elo_reconciliation_cli.py reconcile-all

    # Show detailed drift report with JSON output
    python scripts/elo_reconciliation_cli.py check-drift --json --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.elo_reconciliation import (
    EloDrift,
    EloReconciler,
    SyncResult,
)

try:
    from app.training.promotion_controller import (
        RollbackCriteria,
        RollbackMonitor,
    )
    HAS_ROLLBACK = True
except ImportError:
    HAS_ROLLBACK = False
    RollbackCriteria = None
    RollbackMonitor = None


def format_drift_report(drift: EloDrift, verbose: bool = False) -> str:
    """Format drift report for console output."""
    lines = [
        "=== Elo Drift Report ===",
        f"Source: {drift.source}",
        f"Target: {drift.target}",
        f"Checked at: {drift.checked_at}",
        "",
        f"Participants in source: {drift.participants_in_source}",
        f"Participants in target: {drift.participants_in_target}",
        f"Participants in both: {drift.participants_in_both}",
        "",
        f"Max rating diff: {drift.max_rating_diff:.1f}",
        f"Avg rating diff: {drift.avg_rating_diff:.1f}",
        f"Significant drift: {'YES' if drift.is_significant else 'No'}",
    ]

    if drift.is_significant:
        lines.append("")
        lines.append("WARNING: Significant drift detected!")
        lines.append("Consider running a full reconciliation.")

    if verbose and drift.rating_diffs:
        lines.append("")
        lines.append("=== Rating Differences ===")
        sorted_diffs = sorted(
            drift.rating_diffs.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for participant, diff in sorted_diffs[:20]:  # Top 20
            sign = "+" if diff > 0 else ""
            lines.append(f"  {participant}: {sign}{diff:.1f}")
        if len(drift.rating_diffs) > 20:
            lines.append(f"  ... and {len(drift.rating_diffs) - 20} more")

    return "\n".join(lines)


def format_sync_result(result: SyncResult) -> str:
    """Format sync result for console output."""
    lines = [
        f"=== Sync Result from {result.remote_host} ===",
        f"Synced at: {result.synced_at}",
        f"Matches added: {result.matches_added}",
        f"Matches skipped (duplicates): {result.matches_skipped}",
        f"Conflicts detected: {result.matches_conflict}",
        f"Participants added: {result.participants_added}",
    ]

    if result.error:
        lines.append("")
        lines.append(f"ERROR: {result.error}")

    return "\n".join(lines)


def cmd_check_drift(args: argparse.Namespace) -> int:
    """Check for Elo drift."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
    )

    remote_db = Path(args.remote_db) if args.remote_db else None

    drift = reconciler.check_drift(
        remote_db_path=remote_db,
        board_type=args.board_type,
        num_players=args.num_players,
    )

    if args.json:
        print(json.dumps(drift.to_dict(), indent=2))
    else:
        print(format_drift_report(drift, verbose=args.verbose))

    # Return non-zero if significant drift
    return 1 if drift.is_significant else 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync from a remote host."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
        ssh_timeout=args.timeout,
    )

    result = reconciler.sync_from_remote(
        remote_host=args.host,
        remote_db_path=args.remote_path,
        ssh_user=args.user,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_sync_result(result))

    return 0 if result.error is None else 1


def cmd_reconcile_all(args: argparse.Namespace) -> int:
    """Run full reconciliation across all hosts."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
        ssh_timeout=args.timeout,
    )

    hosts = args.hosts.split(",") if args.hosts else None

    report = reconciler.reconcile_all(hosts=hosts)

    if args.json:
        output = {
            "started_at": report.started_at,
            "completed_at": report.completed_at,
            "nodes_synced": report.nodes_synced,
            "nodes_failed": report.nodes_failed,
            "total_matches_added": report.total_matches_added,
            "total_matches_skipped": report.total_matches_skipped,
            "total_conflicts": report.total_conflicts,
            "drift_detected": report.drift_detected,
            "max_drift": report.max_drift,
            "sync_results": [r.to_dict() for r in report.sync_results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(report.summary())

        if args.verbose:
            print("\n=== Per-Host Results ===")
            for result in report.sync_results:
                print()
                print(format_sync_result(result))

    return 1 if report.nodes_failed else 0


def cmd_drift_history(args: argparse.Namespace) -> int:
    """Show drift history with trend analysis."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
        track_history=True,
        persist_history=True,
    )

    histories = reconciler.get_all_drift_histories()

    if not histories:
        print("No drift history found. Run check-drift to start tracking.")
        return 0

    if args.json:
        output = {key: hist.to_dict() for key, hist in histories.items()}
        print(json.dumps(output, indent=2))
    else:
        print("=== Drift History Report ===\n")
        for config_key, history in sorted(histories.items()):
            print(f"Configuration: {config_key}")
            print(f"  Snapshots: {len(history.snapshots)}")
            print(f"  Trend: {history.trend}")
            print(f"  Persistent drift: {'YES' if history.persistent_drift else 'No'}")
            print(f"  Avg drift (last hour): {history.avg_drift_last_hour:.1f}")

            if args.verbose and history.snapshots:
                print("  Recent snapshots:")
                for snap in history.snapshots[-5:]:
                    sig = "*" if snap["is_significant"] else " "
                    print(f"    {sig} {snap['checked_at'][:19]}: max={snap['max_rating_diff']:.1f}")
            print()

    return 0


def cmd_check_regression(args: argparse.Namespace) -> int:
    """Check if a model is showing regression."""
    if not HAS_ROLLBACK:
        print("ERROR: Rollback monitoring not available. Install promotion_controller.")
        return 1

    monitor = RollbackMonitor(
        criteria=RollbackCriteria(
            elo_regression_threshold=args.threshold,
            min_games_for_regression=args.min_games,
        )
    )

    should_rollback, event = monitor.check_for_regression(
        model_id=args.model_id,
        board_type=args.board_type,
        num_players=args.num_players,
        previous_model_id=args.previous_model_id,
        baseline_elo=args.baseline_elo,
    )

    status = monitor.get_regression_status(args.model_id)

    if args.json:
        output = {
            "model_id": args.model_id,
            "should_rollback": should_rollback,
            "event": event.to_dict() if event else None,
            "status": status,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"=== Regression Check: {args.model_id} ===\n")
        print(f"Board type: {args.board_type}")
        print(f"Players: {args.num_players}")
        print(f"Previous model: {args.previous_model_id or 'N/A'}")
        print(f"Baseline Elo: {args.baseline_elo or 'N/A'}")
        print()
        print(f"Checks performed: {status.get('checks', 0)}")
        print(f"Consecutive regressions: {status.get('consecutive_regressions', 0)}")
        print(f"Average regression: {status.get('avg_regression', 0):.1f}")
        print(f"At risk: {'YES' if status.get('at_risk') else 'No'}")
        print()

        if should_rollback:
            print("RESULT: ROLLBACK RECOMMENDED")
            if event:
                print(f"  Reason: {event.reason}")
                print(f"  Rollback to: {event.rollback_model_id}")
        else:
            print("RESULT: Model is healthy")

    return 1 if should_rollback else 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Manually trigger a rollback."""
    if not HAS_ROLLBACK:
        print("ERROR: Rollback monitoring not available. Install promotion_controller.")
        return 1

    from app.training.promotion_controller import RollbackEvent

    monitor = RollbackMonitor()

    # Create rollback event
    from datetime import datetime
    event = RollbackEvent(
        triggered_at=datetime.now().isoformat(),
        current_model_id=args.from_model,
        rollback_model_id=args.to_model,
        reason=args.reason or "Manual rollback via CLI",
        auto_triggered=False,
    )

    if args.dry_run:
        print(f"[DRY RUN] Would rollback {args.from_model} -> {args.to_model}")
        print(f"  Reason: {event.reason}")
        return 0

    if not args.force:
        confirm = input(f"Rollback {args.from_model} -> {args.to_model}? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return 0

    success = monitor.execute_rollback(event)

    if args.json:
        print(json.dumps({"success": success, "event": event.to_dict()}, indent=2))
    else:
        if success:
            print(f"SUCCESS: Rolled back {args.from_model} -> {args.to_model}")
        else:
            print(f"FAILED: Rollback {args.from_model} -> {args.to_model}")

    return 0 if success else 1


def cmd_rollback_status(args: argparse.Namespace) -> int:
    """Show rollback history and at-risk models."""
    if not HAS_ROLLBACK:
        print("ERROR: Rollback monitoring not available. Install promotion_controller.")
        return 1

    monitor = RollbackMonitor()

    history = monitor.get_rollback_history()

    if args.json:
        output = {
            "rollback_history": [e.to_dict() for e in history],
            "total_rollbacks": len(history),
        }
        print(json.dumps(output, indent=2))
    else:
        print("=== Rollback Status ===\n")
        print(f"Total rollbacks: {len(history)}")

        if history:
            print("\nRecent rollbacks:")
            for event in history[-10:]:
                print(f"  {event.triggered_at[:19]}: {event.current_model_id} -> {event.rollback_model_id}")
                print(f"    Reason: {event.reason}")
                if event.elo_regression:
                    print(f"    Elo regression: {event.elo_regression:.1f}")
        else:
            print("\nNo rollbacks recorded.")

    return 0


def cmd_compare_baselines(args: argparse.Namespace) -> int:
    """Compare a model against multiple baseline models."""
    if not HAS_ROLLBACK:
        print("ERROR: Rollback monitoring not available. Install promotion_controller.")
        return 1

    monitor = RollbackMonitor(
        criteria=RollbackCriteria(elo_regression_threshold=args.threshold)
    )

    # Parse baseline models if provided
    baseline_ids = None
    if args.baselines:
        baseline_ids = [b.strip() for b in args.baselines.split(",")]

    result = monitor.check_against_baselines(
        model_id=args.model_id,
        board_type=args.board_type,
        num_players=args.num_players,
        baseline_model_ids=baseline_ids,
        num_baselines=args.num_baselines,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return 1

        print(f"=== Multi-Baseline Comparison: {args.model_id} ===\n")
        print(f"Current Elo: {result.get('current_elo', 'N/A')}")
        print(f"Games played: {result.get('games_played', 'N/A')}")
        print()

        baselines = result.get("baselines", [])
        if baselines:
            print("Baseline comparisons:")
            for comp in baselines:
                if "error" in comp:
                    print(f"  {comp['baseline_id']}: {comp['error']}")
                else:
                    status = "REGRESSION" if comp.get("is_regression") else "OK"
                    diff = comp.get("elo_diff", 0)
                    sign = "+" if diff >= 0 else ""
                    print(f"  {comp['baseline_id']}: {sign}{diff:.1f} ({status})")
            print()

        print(f"Summary: {result.get('summary', 'unknown')}")
        print(f"  Regressions: {result.get('regressions', 0)}/{result.get('total_baselines', 0)}")
        print(f"  Avg diff: {result.get('avg_diff', 0):.1f}")
        print(f"  Min diff: {result.get('min_diff', 0):.1f}")
        print(f"  Max diff: {result.get('max_diff', 0):.1f}")

    return 0 if result.get("summary") != "regression_against_all" else 1


def cmd_backfill_history(args: argparse.Namespace) -> int:
    """Backfill drift history from existing match data.

    Reads historical match data and populates the drift history
    file by simulating reconciliation across time intervals.
    """

    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
        track_history=True,
        persist_history=True,
    )

    print(f"Backfilling drift history from {reconciler.local_db_path}...")

    if not reconciler.local_db_path.exists():
        print("Error: Local database does not exist")
        return 1

    # Get unique board_type/num_players combinations
    import sqlite3
    conn = sqlite3.connect(str(reconciler.local_db_path), timeout=10)
    try:
        cur = conn.cursor()

        # Get all configurations from elo_ratings table
        cur.execute("""
            SELECT DISTINCT board_type, num_players
            FROM elo_ratings
            WHERE board_type IS NOT NULL
        """)
        configs = cur.fetchall()

        if not configs:
            print("No configurations found in database")
            return 0

        print(f"Found {len(configs)} configurations to process")

        # Get match history with timestamps
        backfill_count = 0
        for board_type, num_players in configs:
            config_key = f"{board_type}_{num_players}p"
            print(f"\nProcessing {config_key}...")

            # Get matches sorted by timestamp (using rating_history for daily averages)
            cur.execute("""
                SELECT DATE(timestamp, 'unixepoch') as match_date, AVG(rating) as avg_rating
                FROM rating_history
                WHERE board_type = ? AND num_players = ?
                GROUP BY DATE(timestamp, 'unixepoch')
                ORDER BY timestamp
            """, (board_type, num_players))

            daily_ratings = cur.fetchall()

            if len(daily_ratings) < 2:
                print(f"  Insufficient data for {config_key}")
                continue

            # Simulate drift over time (comparing to previous day)
            prev_avg = None
            for _timestamp, avg_rating in daily_ratings:
                if prev_avg is not None:
                    drift = avg_rating - prev_avg
                    # Record this as drift
                    reconciler._record_drift(board_type, num_players, {
                        "avg_drift": drift,
                        "max_drift": drift,
                        "min_drift": drift,
                        "significant_drift": abs(drift) > 50,
                        "total_models": 1,
                        "drift_count": 1,
                    })
                    backfill_count += 1
                prev_avg = avg_rating

            print(f"  Added {len(daily_ratings) - 1} drift records for {config_key}")

        # Save the history
        reconciler.save_drift_history()
        print(f"\nBackfill complete: {backfill_count} drift records created")

        # Show summary
        if args.json:
            print(json.dumps({
                "status": "success",
                "records_created": backfill_count,
                "configurations": len(configs),
            }, indent=2))
        else:
            print(f"\nDrift history saved to: {reconciler._drift_history_path}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 1
    finally:
        conn.close()

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show current status and configuration."""
    reconciler = EloReconciler(
        local_db_path=Path(args.local_db) if args.local_db else None,
    )

    status = {
        "local_db_path": str(reconciler.local_db_path),
        "local_db_exists": reconciler.local_db_path.exists(),
        "remote_hosts_config": str(reconciler.remote_hosts_config),
        "config_exists": reconciler.remote_hosts_config.exists(),
        "ssh_timeout": reconciler.ssh_timeout,
    }

    # Check local DB stats
    if reconciler.local_db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(reconciler.local_db_path), timeout=10)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM participants")
            status["total_participants"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM match_history")
            status["total_matches"] = cur.fetchone()[0]
        except sqlite3.OperationalError:
            status["total_participants"] = "N/A (table not found)"
            status["total_matches"] = "N/A (table not found)"
        finally:
            conn.close()

    # Load configured hosts
    hosts = reconciler._load_p2p_hosts()
    status["configured_hosts"] = hosts
    status["num_configured_hosts"] = len(hosts)

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print("=== Elo Reconciliation Status ===")
        print(f"Local DB: {status['local_db_path']}")
        print(f"  Exists: {status['local_db_exists']}")
        if status['local_db_exists']:
            print(f"  Participants: {status.get('total_participants', 'N/A')}")
            print(f"  Matches: {status.get('total_matches', 'N/A')}")
        print()
        print(f"Config: {status['remote_hosts_config']}")
        print(f"  Exists: {status['config_exists']}")
        print(f"  Configured hosts: {status['num_configured_hosts']}")
        if hosts:
            for host in hosts:
                print(f"    - {host}")
        print()
        print(f"SSH timeout: {status['ssh_timeout']}s")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Elo Rating Reconciliation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--local-db",
        type=str,
        help="Path to local Elo database (default: data/unified_elo.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check-drift command
    drift_parser = subparsers.add_parser(
        "check-drift",
        help="Check for Elo drift between local and remote databases",
    )
    drift_parser.add_argument(
        "--remote-db",
        type=str,
        help="Path to remote database (local copy) to compare against",
    )
    drift_parser.add_argument(
        "--board-type",
        type=str,
        help="Filter to specific board type",
    )
    drift_parser.add_argument(
        "--num-players",
        type=int,
        help="Filter to specific number of players",
    )

    # sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync match history from a remote host",
    )
    sync_parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Remote host IP or hostname",
    )
    sync_parser.add_argument(
        "--remote-path",
        type=str,
        default="~/ringrift/ai-service/data/unified_elo.db",
        help="Path to Elo database on remote host",
    )
    sync_parser.add_argument(
        "--user",
        type=str,
        default="ubuntu",
        help="SSH username (default: ubuntu)",
    )
    sync_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SSH timeout in seconds (default: 30)",
    )

    # reconcile-all command
    reconcile_parser = subparsers.add_parser(
        "reconcile-all",
        help="Run full reconciliation across all configured hosts",
    )
    reconcile_parser.add_argument(
        "--hosts",
        type=str,
        help="Comma-separated list of hosts to sync (overrides config)",
    )
    reconcile_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SSH timeout in seconds (default: 30)",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show current status and configuration",
    )

    # drift-history command
    history_parser = subparsers.add_parser(
        "drift-history",
        help="Show drift history with trend analysis",
    )

    # check-regression command
    regression_parser = subparsers.add_parser(
        "check-regression",
        help="Check if a model is showing regression",
    )
    regression_parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID to check",
    )
    regression_parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        help="Board type (default: square8)",
    )
    regression_parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    regression_parser.add_argument(
        "--previous-model-id",
        type=str,
        help="Previous model ID for rollback target",
    )
    regression_parser.add_argument(
        "--baseline-elo",
        type=float,
        help="Baseline Elo at promotion time",
    )
    regression_parser.add_argument(
        "--threshold",
        type=float,
        default=-30.0,
        help="Elo regression threshold (default: -30.0)",
    )
    regression_parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Minimum games for regression check (default: 20)",
    )

    # rollback command
    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Manually trigger a rollback",
    )
    rollback_parser.add_argument(
        "--from-model",
        type=str,
        required=True,
        help="Model to rollback from",
    )
    rollback_parser.add_argument(
        "--to-model",
        type=str,
        required=True,
        help="Model to rollback to",
    )
    rollback_parser.add_argument(
        "--reason",
        type=str,
        help="Reason for rollback",
    )
    rollback_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would happen",
    )
    rollback_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # rollback-status command
    rollback_status_parser = subparsers.add_parser(
        "rollback-status",
        help="Show rollback history and at-risk models",
    )

    # compare-baselines command
    compare_parser = subparsers.add_parser(
        "compare-baselines",
        help="Compare a model against multiple baseline models",
    )
    compare_parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID to compare",
    )
    compare_parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        help="Board type (default: square8)",
    )
    compare_parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    compare_parser.add_argument(
        "--baselines",
        type=str,
        help="Comma-separated list of baseline model IDs (optional, auto-detect if not provided)",
    )
    compare_parser.add_argument(
        "--num-baselines",
        type=int,
        default=3,
        help="Number of recent models to compare against (default: 3)",
    )
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=-30.0,
        help="Elo regression threshold (default: -30.0)",
    )

    # backfill-history command
    backfill_parser = subparsers.add_parser(
        "backfill-history",
        help="Backfill drift history from existing match data",
    )

    # Add common args to all subparsers for convenience
    all_parsers = [
        drift_parser, sync_parser, reconcile_parser, status_parser,
        history_parser, regression_parser, rollback_parser, rollback_status_parser,
        compare_parser, backfill_parser
    ]
    for subparser in all_parsers:
        subparser.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format",
            dest="subcommand_json",
        )
        subparser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output",
            dest="subcommand_verbose",
        )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Merge global and subcommand flags
    args.json = getattr(args, "json", False) or getattr(args, "subcommand_json", False)
    args.verbose = getattr(args, "verbose", False) or getattr(args, "subcommand_verbose", False)

    if args.command == "check-drift":
        return cmd_check_drift(args)
    elif args.command == "sync":
        return cmd_sync(args)
    elif args.command == "reconcile-all":
        return cmd_reconcile_all(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "drift-history":
        return cmd_drift_history(args)
    elif args.command == "check-regression":
        return cmd_check_regression(args)
    elif args.command == "rollback":
        return cmd_rollback(args)
    elif args.command == "rollback-status":
        return cmd_rollback_status(args)
    elif args.command == "compare-baselines":
        return cmd_compare_baselines(args)
    elif args.command == "backfill-history":
        return cmd_backfill_history(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
