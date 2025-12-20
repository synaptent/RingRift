#!/usr/bin/env python3
"""
RingRift Training CLI

Unified command-line interface for training operations.

Usage:
    # Show available commands
    python scripts/cli.py --help

    # Harvest training data
    python scripts/cli.py harvest --board-type square8 --num-players 2

    # Monitor Elo ratings
    python scripts/cli.py monitor

    # Compare models
    python scripts/cli.py compare --model-a new.pt --model-b best.pt

    # Check cluster health
    python scripts/cli.py health

    # Run periodic tasks
    python scripts/cli.py periodic --type harvest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_harvest(args: argparse.Namespace) -> int:
    """Run data harvesting."""
    from scripts.harvest_local_training_data import harvest_local_data
    from scripts.lib.logging_config import setup_script_logging, get_logger

    setup_script_logging(
        script_name="harvest",
        level="DEBUG" if args.verbose else "INFO",
        json_logs=args.json_logs,
    )
    logger = get_logger(__name__)

    logger.info(f"Starting harvest: {args.board_type}_{args.num_players}p")

    try:
        games, stats = harvest_local_data(
            data_dir=Path(args.data_dir),
            board_type=args.board_type,
            num_players=args.num_players,
            min_quality=args.min_quality,
            max_games=args.max_games,
            output_file=Path(args.output),
            sample_rate=args.sample_rate,
            shuffle_files=not args.no_shuffle,
        )

        if games == 0:
            logger.warning("No games harvested")
            return 1

        logger.info(f"Harvested {games:,} games (avg quality: {stats.avg_quality:.3f})")
        return 0

    except Exception as e:
        logger.exception(f"Harvest failed: {e}")
        return 1


def cmd_monitor(args: argparse.Namespace) -> int:
    """Run Elo monitoring."""
    from scripts.elo_monitor import EloMonitor, StatusReporter, AlertSeverity
    from scripts.lib.logging_config import setup_script_logging, get_logger

    setup_script_logging(
        script_name="monitor",
        level="DEBUG" if args.verbose else "INFO",
        json_logs=args.json_logs,
    )
    logger = get_logger(__name__)

    monitor = EloMonitor(
        history_file=Path(args.history_file),
        alert_hours=args.alert_hours,
        regression_threshold=args.regression_threshold,
    )

    monitor.load_history()
    ratings = monitor.get_current_ratings()
    activity = monitor.check_training_activity()
    alerts = monitor.analyze(ratings)
    monitor.save_history()

    reporter = StatusReporter(use_colors=not args.json)

    if args.json:
        print(reporter.to_json(ratings, activity, alerts))
    else:
        reporter.print_report(ratings, activity, alerts)

    high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
    return 1 if high_alerts else 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run model comparison."""
    from scripts.compare_models_elo import ModelComparator, ComparisonReporter
    from scripts.lib.logging_config import setup_script_logging, get_logger

    setup_script_logging(
        script_name="compare",
        level="DEBUG" if args.verbose else "INFO",
        json_logs=args.json_logs,
    )
    logger = get_logger(__name__)

    model_a = Path(args.model_a)
    model_b = Path(args.model_b)

    if not model_a.exists():
        logger.error(f"Model A not found: {model_a}")
        return 1
    if not model_b.exists():
        logger.error(f"Model B not found: {model_b}")
        return 1

    mcts_sims = args.mcts_sims
    num_games = args.games

    if args.quick:
        mcts_sims = 50
        num_games = min(num_games, 20)

    comparator = ModelComparator(
        model_a_path=str(model_a),
        model_b_path=str(model_b),
        board_type=args.board_type,
        num_players=args.num_players,
        mcts_simulations=mcts_sims,
        parallel_games=args.parallel,
    )

    try:
        result = comparator.run(num_games)
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        return 1

    reporter = ComparisonReporter()
    reporter.print_report(result)

    if args.output:
        reporter.save_result(result, Path(args.output))

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check cluster health."""
    from scripts.lib.cluster import ClusterManager
    from scripts.lib.logging_config import setup_script_logging, get_logger

    setup_script_logging(
        script_name="health",
        level="DEBUG" if args.verbose else "INFO",
        json_logs=args.json_logs,
    )
    logger = get_logger(__name__)

    manager = ClusterManager()
    healthy_nodes = manager.get_healthy_nodes(force_check=True)

    print(f"\n{'=' * 60}")
    print(f"  CLUSTER HEALTH CHECK")
    print(f"{'=' * 60}\n")

    total_nodes = len(manager.nodes)
    healthy_count = len(healthy_nodes)

    print(f"Nodes: {healthy_count}/{total_nodes} healthy\n")

    for node in manager.nodes:
        health = node.check_health(force=True)
        if health.is_healthy:
            status = "OK"
            details = f"GPU: {health.gpu_utilization}%, Mem: {health.memory_used_gb:.1f}GB"
        else:
            status = "FAIL"
            details = health.error_message or "Unknown error"

        print(f"  {node.name:20s} [{status:4s}] {details}")

    print(f"\n{'=' * 60}\n")

    if args.json:
        import json
        metrics = manager.collect_metrics()
        print(json.dumps(metrics, indent=2, default=str))

    return 0 if healthy_count > 0 else 1


def cmd_periodic(args: argparse.Namespace) -> int:
    """Run periodic tasks."""
    from scripts.periodic_harvest import run_periodic_harvest
    from scripts.lib.logging_config import setup_script_logging, get_logger

    setup_script_logging(
        script_name=f"periodic_{args.task}",
        level="DEBUG" if args.verbose else "INFO",
        json_logs=args.json_logs,
    )
    logger = get_logger(__name__)

    if args.task == "harvest":
        return run_periodic_harvest(
            board_type=args.board_type,
            num_players=args.num_players,
            min_quality=args.min_quality,
            max_games=args.max_games,
            min_new_games_for_training=args.min_new_for_training,
            trigger_training=not args.no_training,
        )
    else:
        logger.error(f"Unknown periodic task: {args.task}")
        return 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON format for log files",
    )


# =============================================================================
# NEW CONSOLIDATED COMMANDS (December 2025)
# =============================================================================

def cmd_sync(args: argparse.Namespace) -> int:
    """Data sync operations."""
    import asyncio

    sync_type = args.sync_type

    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        async def do_sync():
            coordinator = SyncCoordinator.get_instance()
            if sync_type == "data":
                return await coordinator.sync_games()
            elif sync_type == "models":
                return await coordinator.sync_models()
            elif sync_type == "all":
                return await coordinator.full_cluster_sync()
            elif sync_type == "status":
                return coordinator.get_status()
            else:
                print(f"Unknown sync type: {sync_type}")
                return None

        result = asyncio.run(do_sync())

        if sync_type == "status":
            import json
            print(json.dumps(result, indent=2, default=str))
            return 0
        elif result:
            print(f"Sync complete: {result.files_synced} files")
            return 0
        return 1

    except Exception as e:
        print(f"Sync error: {e}")
        return 1


def cmd_daemon(args: argparse.Namespace) -> int:
    """Daemon management."""
    import asyncio

    action = args.daemon_action

    try:
        from app.coordination.daemon_manager import get_daemon_manager, DaemonType

        async def do_action():
            manager = get_daemon_manager()
            if action == "start":
                if args.name:
                    dtype = DaemonType(args.name)
                    return await manager.start(dtype)
                return await manager.start_all()
            elif action == "stop":
                if args.name:
                    dtype = DaemonType(args.name)
                    return await manager.stop(dtype)
                return await manager.stop_all()
            elif action == "status":
                return manager.get_status()

        result = asyncio.run(do_action())

        if action == "status":
            import json
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                summary = result.get("summary", {})
                print(f"\nDaemons: {summary.get('running', 0)}/{summary.get('total', 0)} running")
                for name, info in result.get("daemons", {}).items():
                    print(f"  {name}: {info.get('state', 'unknown')}")
            return 0
        else:
            print(f"Daemon {action} complete")
            return 0

    except Exception as e:
        print(f"Daemon error: {e}")
        return 1


def cmd_cluster(args: argparse.Namespace) -> int:
    """Cluster monitoring with unified_cluster_monitor."""
    try:
        from scripts.unified_cluster_monitor import ClusterConfig, UnifiedClusterMonitor

        config = ClusterConfig()
        if not config.nodes:
            print("No cluster nodes found")
            return 1

        monitor = UnifiedClusterMonitor(
            config=config,
            webhook_url=getattr(args, 'webhook', None),
            check_interval=getattr(args, 'interval', 60),
            deep_checks=getattr(args, 'deep', False),
        )

        if getattr(args, 'continuous', False):
            monitor.run_continuous(output_json=getattr(args, 'json', False))
        else:
            cluster = monitor.run_once(output_json=getattr(args, 'json', False))
            if cluster.critical_alerts:
                return 2
            elif cluster.alerts:
                return 1
        return 0

    except Exception as e:
        print(f"Cluster monitor error: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show unified cluster status."""
    import json as json_module

    try:
        from app.coordination import get_all_coordinator_status

        status = get_all_coordinator_status()

        if getattr(args, 'json', False):
            print(json_module.dumps(status, indent=2, default=str))
        else:
            print("\n" + "=" * 50)
            print("  RINGRIFT CLUSTER STATUS")
            print("=" * 50)
            for name, ok in status.items():
                if name.startswith("_"):
                    continue
                state = "OK" if ok else "DOWN"
                print(f"  {name}: {state}")
            print("=" * 50)
        return 0

    except Exception as e:
        print(f"Status error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RingRift Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  harvest   Harvest high-quality training data
  monitor   Monitor Elo ratings and training activity
  compare   Compare two models head-to-head
  health    Check cluster health (legacy)
  periodic  Run periodic automated tasks
  sync      Data synchronization (data, models, all, status)
  daemon    Daemon management (start, stop, status)
  cluster   Unified cluster monitoring (--deep, --continuous)
  status    Show coordinator status

Examples:
  %(prog)s harvest --output data/harvested/local.jsonl
  %(prog)s monitor --json
  %(prog)s compare --model-a new.pt --model-b best.pt --quick
  %(prog)s sync data
  %(prog)s daemon status
  %(prog)s cluster --continuous --interval 30
  %(prog)s status --json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Harvest command
    harvest_parser = subparsers.add_parser("harvest", help="Harvest training data")
    harvest_parser.add_argument(
        "--board-type", default="square8",
        help="Board type (default: square8)",
    )
    harvest_parser.add_argument(
        "--num-players", type=int, default=2,
        help="Number of players (default: 2)",
    )
    harvest_parser.add_argument(
        "--min-quality", type=float, default=0.7,
        help="Minimum quality score (default: 0.7)",
    )
    harvest_parser.add_argument(
        "--max-games", type=int, default=50000,
        help="Maximum games to harvest (default: 50000)",
    )
    harvest_parser.add_argument(
        "--data-dir", default="data/selfplay",
        help="Source data directory",
    )
    harvest_parser.add_argument(
        "--output", required=True,
        help="Output JSONL file",
    )
    harvest_parser.add_argument(
        "--sample-rate", type=float, default=1.0,
        help="Fraction of games to sample",
    )
    harvest_parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Don't shuffle file processing order",
    )
    add_common_args(harvest_parser)

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor Elo ratings")
    monitor_parser.add_argument(
        "--alert-hours", type=float, default=6,
        help="Hours without update before alerting",
    )
    monitor_parser.add_argument(
        "--regression-threshold", type=float, default=30,
        help="Elo drop threshold for alert",
    )
    monitor_parser.add_argument(
        "--history-file", default="data/elo_monitor_history.json",
        help="History file path",
    )
    monitor_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    add_common_args(monitor_parser)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument(
        "--model-a", required=True,
        help="Path to model A",
    )
    compare_parser.add_argument(
        "--model-b", required=True,
        help="Path to model B",
    )
    compare_parser.add_argument(
        "--board-type", default="square8",
        help="Board type",
    )
    compare_parser.add_argument(
        "--num-players", type=int, default=2,
        help="Number of players",
    )
    compare_parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games",
    )
    compare_parser.add_argument(
        "--mcts-sims", type=int, default=100,
        help="MCTS simulations per move",
    )
    compare_parser.add_argument(
        "--parallel", type=int, default=4,
        help="Parallel games",
    )
    compare_parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode",
    )
    compare_parser.add_argument(
        "--output", help="Output JSON file",
    )
    add_common_args(compare_parser)

    # Health command
    health_parser = subparsers.add_parser("health", help="Check cluster health")
    health_parser.add_argument(
        "--json", action="store_true",
        help="Output metrics as JSON",
    )
    add_common_args(health_parser)

    # Periodic command
    periodic_parser = subparsers.add_parser("periodic", help="Run periodic tasks")
    periodic_parser.add_argument(
        "--task", choices=["harvest"], default="harvest",
        help="Task to run",
    )
    periodic_parser.add_argument(
        "--board-type", default="square8",
        help="Board type",
    )
    periodic_parser.add_argument(
        "--num-players", type=int, default=2,
        help="Number of players",
    )
    periodic_parser.add_argument(
        "--min-quality", type=float, default=0.7,
        help="Minimum quality score",
    )
    periodic_parser.add_argument(
        "--max-games", type=int, default=25000,
        help="Max games per harvest",
    )
    periodic_parser.add_argument(
        "--min-new-for-training", type=int, default=10000,
        help="Minimum new games to trigger training",
    )
    periodic_parser.add_argument(
        "--no-training", action="store_true",
        help="Don't trigger training",
    )
    add_common_args(periodic_parser)

    # Sync command (December 2025)
    sync_parser = subparsers.add_parser("sync", help="Data synchronization")
    sync_parser.add_argument(
        "sync_type", choices=["data", "models", "all", "status"],
        help="Type of sync operation",
    )
    sync_parser.add_argument("--json", action="store_true", help="JSON output")

    # Daemon command (December 2025)
    daemon_parser = subparsers.add_parser("daemon", help="Daemon management")
    daemon_parser.add_argument(
        "daemon_action", choices=["start", "stop", "status"],
        help="Daemon action",
    )
    daemon_parser.add_argument("--name", help="Specific daemon name")
    daemon_parser.add_argument("--json", action="store_true", help="JSON output")

    # Cluster command (December 2025)
    cluster_parser = subparsers.add_parser("cluster", help="Cluster monitoring")
    cluster_parser.add_argument("--deep", "-d", action="store_true", help="Deep SSH checks")
    cluster_parser.add_argument("--continuous", "-c", action="store_true", help="Continuous monitoring")
    cluster_parser.add_argument("--interval", "-i", type=int, default=60, help="Check interval")
    cluster_parser.add_argument("--webhook", "-w", help="Alert webhook URL")
    cluster_parser.add_argument("--json", action="store_true", help="JSON output")

    # Status command (December 2025)
    status_parser = subparsers.add_parser("status", help="Coordinator status")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "harvest": cmd_harvest,
        "monitor": cmd_monitor,
        "compare": cmd_compare,
        "health": cmd_health,
        "periodic": cmd_periodic,
        # New consolidated commands (December 2025)
        "sync": cmd_sync,
        "daemon": cmd_daemon,
        "cluster": cmd_cluster,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
