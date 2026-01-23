#!/usr/bin/env python3
"""Master Daemon Launcher - Unified interface for managing all daemons.

This script provides a centralized CLI for starting, stopping, and monitoring
all daemons through the DaemonManager.

Usage:
    # Start all daemons
    python scripts/launch_daemons.py --all

    # Start specific daemon categories
    python scripts/launch_daemons.py --sync        # Sync daemons only
    python scripts/launch_daemons.py --training    # Training daemons only
    python scripts/launch_daemons.py --monitoring  # Monitoring daemons only

    # Start daemon profile (canonical from daemon_manager)
    python scripts/launch_daemons.py --profile coordinator

    # Start specific daemons
    python scripts/launch_daemons.py --daemon sync_coordinator --daemon event_router

    # Show daemon status
    python scripts/launch_daemons.py --status

    # Stop all daemons
    python scripts/launch_daemons.py --stop-all

    # Watch mode (show live status)
    python scripts/launch_daemons.py --watch --interval 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.daemon_manager import (
    DAEMON_PROFILES,
    DaemonManager,
    DaemonManagerConfig,
    DaemonType,
    get_daemon_manager,
    setup_signal_handlers,
)
from app.coordination.daemon_adapters import register_all_adapters_with_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Daemon categories for grouped launching
DAEMON_CATEGORIES = {
    "sync": [
        DaemonType.AUTO_SYNC,  # Primary P2P data sync with gossip replication
        DaemonType.HIGH_QUALITY_SYNC,
        DaemonType.ELO_SYNC,
        DaemonType.MODEL_SYNC,
        DaemonType.CLUSTER_DATA_SYNC,
    ],
    "training": [
        DaemonType.DATA_PIPELINE,
        DaemonType.TRAINING_NODE_WATCHER,
        DaemonType.SELFPLAY_COORDINATOR,
        DaemonType.DISTILLATION,
        DaemonType.UNIFIED_PROMOTION,
        DaemonType.CONTINUOUS_TRAINING_LOOP,
        DaemonType.TOURNAMENT_DAEMON,  # Auto-schedule tournaments on model events
        DaemonType.FEEDBACK_LOOP,  # Central feedback loop controller (Dec 2025)
        DaemonType.EVALUATION,  # Auto-evaluate models after training completes (Dec 2025)
    ],
    "monitoring": [
        DaemonType.HEALTH_SERVER,
        DaemonType.CLUSTER_MONITOR,
        DaemonType.QUEUE_MONITOR,
        DaemonType.REPLICATION_MONITOR,  # Monitor data replication health
        DaemonType.REPLICATION_REPAIR,   # Actively repair under-replicated data
        DaemonType.ORPHAN_DETECTION,     # Detect unregistered game databases
        DaemonType.NODE_HEALTH_MONITOR,  # Unified cluster health
    ],
    "events": [
        DaemonType.EVENT_ROUTER,
        DaemonType.CROSS_PROCESS_POLLER,
    ],
    "p2p": [
        DaemonType.P2P_BACKEND,
        DaemonType.GOSSIP_SYNC,
        DaemonType.DATA_SERVER,
    ],
    "external": [
        DaemonType.EXTERNAL_DRIVE_SYNC,
        DaemonType.VAST_CPU_PIPELINE,
    ],
}

PROFILE_CHOICES = sorted(DAEMON_PROFILES.keys())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Master Daemon Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  --sync        Start sync daemons (data, models, elo)
  --training    Start training pipeline daemons
  --monitoring  Start monitoring daemons
  --events      Start event processing daemons
  --p2p         Start P2P service daemons
  --external    Start external integration daemons

Examples:
  python scripts/launch_daemons.py --all
  python scripts/launch_daemons.py --sync --training
  python scripts/launch_daemons.py --status --watch
        """,
    )

    # Daemon selection
    group = parser.add_argument_group("Daemon Selection")
    group.add_argument(
        "--all", action="store_true",
        help="Start all available daemons",
    )
    group.add_argument(
        "--daemon", action="append", dest="daemons",
        help="Start specific daemon(s) by name (can repeat)",
    )
    group.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        help="Start a daemon profile from daemon_manager (overrides categories)",
    )

    # Category shortcuts
    cat_group = parser.add_argument_group("Category Shortcuts")
    cat_group.add_argument("--sync", action="store_true", help="Start sync daemons")
    cat_group.add_argument("--training", action="store_true", help="Start training daemons")
    cat_group.add_argument("--monitoring", action="store_true", help="Start monitoring daemons")
    cat_group.add_argument("--events", action="store_true", help="Start event daemons")
    cat_group.add_argument("--p2p", action="store_true", help="Start P2P daemons")
    cat_group.add_argument("--external", action="store_true", help="Start external daemons")
    cat_group.add_argument("--continuous", action="store_true",
                            help="Start continuous training loop daemon only")
    cat_group.add_argument("--sync-cluster", action="store_true",
                            help="Start cluster-wide data sync daemon only")

    # Actions
    action_group = parser.add_argument_group("Actions")
    action_group.add_argument("--status", action="store_true", help="Show daemon status")
    action_group.add_argument("--stop-all", action="store_true", help="Stop all daemons")
    action_group.add_argument("--stop", action="append", dest="stop_daemons",
                               help="Stop specific daemon(s)")
    action_group.add_argument("--diagnostics", action="store_true",
                               help="Run startup diagnostics without starting daemons")
    action_group.add_argument("--skip-diagnostics", action="store_true",
                               help="Skip startup diagnostics (for faster startup)")

    # Display options
    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument("--watch", action="store_true", help="Watch mode (live updates)")
    display_group.add_argument("--interval", type=float, default=5.0,
                                help="Watch interval in seconds")
    display_group.add_argument("--json", action="store_true", help="Output in JSON format")
    display_group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--health-interval", type=float, default=30.0,
                               help="Health check interval in seconds")
    config_group.add_argument("--no-auto-restart", action="store_true",
                               help="Disable auto-restart on failure")

    return parser.parse_args()


def get_daemons_to_start(args: argparse.Namespace) -> list[DaemonType]:
    """Determine which daemons to start based on arguments.

    December 2025: Coordinator-only mode
    When running on a coordinator node (role: coordinator in distributed_hosts.yaml),
    intensive daemons (selfplay, training, gauntlet, export) are automatically filtered out.
    """
    daemons: set[DaemonType] = set()

    if args.profile:
        daemons.update(DAEMON_PROFILES.get(args.profile, []))

        if args.all or args.sync or args.training or args.monitoring or args.events or args.p2p or args.external:
            logger.info("--profile set; ignoring category flags")

        if args.daemons:
            for name in args.daemons:
                try:
                    daemon_type = DaemonType(name)
                    daemons.add(daemon_type)
                except ValueError:
                    logger.warning(f"Unknown daemon type: {name}")

        return _filter_for_coordinator(list(daemons))

    # Handle --all
    if args.all:
        return _filter_for_coordinator(list(DaemonType))

    # Handle categories
    if args.sync:
        daemons.update(DAEMON_CATEGORIES["sync"])
    if args.training:
        daemons.update(DAEMON_CATEGORIES["training"])
    if args.monitoring:
        daemons.update(DAEMON_CATEGORIES["monitoring"])
    if args.events:
        daemons.update(DAEMON_CATEGORIES["events"])
    if args.p2p:
        daemons.update(DAEMON_CATEGORIES["p2p"])
    if args.external:
        daemons.update(DAEMON_CATEGORIES["external"])
    if args.continuous:
        daemons.add(DaemonType.CONTINUOUS_TRAINING_LOOP)
    if args.sync_cluster:
        daemons.add(DaemonType.CLUSTER_DATA_SYNC)

    # Handle specific daemons
    if args.daemons:
        for name in args.daemons:
            try:
                daemon_type = DaemonType(name)
                daemons.add(daemon_type)
            except ValueError:
                logger.warning(f"Unknown daemon type: {name}")

    return _filter_for_coordinator(list(daemons))


def _filter_for_coordinator(daemons: list[DaemonType]) -> list[DaemonType]:
    """Filter out intensive daemons when running on a coordinator node.

    December 2025: Coordinator-only mode
    Coordinators should NOT run CPU/GPU intensive daemons.
    """
    from app.config.env import env

    if not env.is_coordinator:
        return daemons

    # Daemons that run CPU/GPU intensive processes
    # These should NEVER run on coordinator nodes
    intensive_daemons = {
        DaemonType.IDLE_RESOURCE,           # spawns selfplay
        DaemonType.TRAINING_NODE_WATCHER,   # monitors training
        DaemonType.AUTO_EXPORT,             # exports training data (CPU-bound)
        DaemonType.TOURNAMENT_DAEMON,       # runs tournaments
        DaemonType.EVALUATION,              # runs gauntlets
        DaemonType.AUTO_PROMOTION,          # triggers promotion (can spawn gauntlet)
        DaemonType.QUEUE_POPULATOR,         # can spawn selfplay
        DaemonType.UTILIZATION_OPTIMIZER,   # spawns processes on idle GPUs
        DaemonType.CONTINUOUS_TRAINING_LOOP,  # full training loop
        DaemonType.VAST_CPU_PIPELINE,       # Vast.ai CPU selfplay
    }

    filtered = [d for d in daemons if d not in intensive_daemons]
    removed_count = len(daemons) - len(filtered)

    if removed_count > 0:
        logger.info(
            f"Coordinator-only mode: filtered out {removed_count} intensive daemons "
            f"(node: {env.node_id})"
        )

    return filtered


def format_status(status: dict, use_json: bool = False) -> str:
    """Format daemon status for display."""
    if use_json:
        return json.dumps(status, indent=2)

    lines = []
    lines.append("=" * 60)
    lines.append("DAEMON MANAGER STATUS")
    lines.append("=" * 60)

    summary = status.get("summary", {})
    lines.append(f"Running: {status.get('running', False)}")
    lines.append(f"Total: {summary.get('total', 0)} | "
                 f"Running: {summary.get('running', 0)} | "
                 f"Failed: {summary.get('failed', 0)} | "
                 f"Stopped: {summary.get('stopped', 0)}")
    lines.append("")

    daemons = status.get("daemons", {})
    if daemons:
        lines.append("Daemons:")
        lines.append("-" * 40)
        for name, info in sorted(daemons.items()):
            state = info.get("state", "unknown")
            uptime = info.get("uptime_seconds", 0)
            restarts = info.get("restart_count", 0)

            state_icon = {
                "running": "✓",
                "stopped": "○",
                "failed": "✗",
                "starting": "→",
                "stopping": "←",
                "restarting": "↻",
            }.get(state, "?")

            uptime_str = ""
            if uptime > 0:
                if uptime > 3600:
                    uptime_str = f" ({uptime / 3600:.1f}h)"
                elif uptime > 60:
                    uptime_str = f" ({uptime / 60:.1f}m)"
                else:
                    uptime_str = f" ({uptime:.0f}s)"

            restart_str = f" [restarts: {restarts}]" if restarts > 0 else ""

            lines.append(f"  {state_icon} {name}: {state}{uptime_str}{restart_str}")

            if info.get("last_error"):
                lines.append(f"      Error: {info['last_error'][:50]}...")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


async def show_status(manager: DaemonManager, args: argparse.Namespace) -> None:
    """Show daemon status."""
    status = manager.get_status()
    print(format_status(status, use_json=args.json))


async def watch_status(manager: DaemonManager, args: argparse.Namespace) -> None:
    """Watch daemon status with live updates."""
    import os

    try:
        while True:
            # Clear screen
            os.system("clear" if os.name == "posix" else "cls")

            status = manager.get_status()
            print(format_status(status, use_json=args.json))
            print(f"\nRefreshing every {args.interval}s... (Ctrl+C to exit)")

            await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


async def run_startup_diagnostics(daemons: list[DaemonType], verbose: bool = False) -> dict:
    """Run pre-flight diagnostics before starting daemons.

    Checks:
    - Python dependencies
    - GPU availability
    - Event bus connectivity
    - Required config files
    - Node identity

    Args:
        daemons: List of daemons to be started
        verbose: Show detailed diagnostic output

    Returns:
        Dictionary of diagnostic results
    """
    results = {
        "passed": True,
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    print("=" * 60)
    print("STARTUP DIAGNOSTICS")
    print("=" * 60)

    # 1. Check Python environment
    import sys
    results["checks"]["python_version"] = {
        "value": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "ok": sys.version_info >= (3, 10),
    }
    print(f"✓ Python: {results['checks']['python_version']['value']}")

    # 2. Check GPU availability
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if has_cuda else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "None"
        results["checks"]["gpu"] = {
            "available": has_cuda,
            "count": gpu_count,
            "name": gpu_name,
            "ok": True,  # GPU is optional
        }
        if has_cuda:
            print(f"✓ GPU: {gpu_count}x {gpu_name}")
        else:
            print("○ GPU: Not available (CPU mode)")
            results["warnings"].append("No GPU available - some daemons may run slower")
    except ImportError:
        results["checks"]["gpu"] = {"available": False, "ok": True}
        print("○ GPU: PyTorch not installed")

    # 3. Check event bus
    try:
        from app.coordination.event_router import get_event_bus
        bus = get_event_bus()
        if bus:
            results["checks"]["event_bus"] = {"available": True, "ok": True}
            print("✓ Event bus: Available")
        else:
            results["checks"]["event_bus"] = {"available": False, "ok": False}
            print("✗ Event bus: Not initialized")
            results["warnings"].append("Event bus not available - events may not be routed")
    except Exception as e:
        results["checks"]["event_bus"] = {"available": False, "error": str(e), "ok": False}
        print(f"✗ Event bus: Error - {e}")
        results["warnings"].append(f"Event bus error: {e}")

    # 4. Check node identity
    try:
        import socket
        hostname = socket.gethostname()
        results["checks"]["hostname"] = {"value": hostname, "ok": True}
        print(f"✓ Hostname: {hostname}")
    except OSError:
        results["checks"]["hostname"] = {"value": "unknown", "ok": True}

    # 5. Check config files
    config_dir = Path(__file__).parent.parent / "config"
    critical_configs = [
        "distributed_hosts.yaml",
    ]
    optional_configs = [
        "training_config.yaml",
        "selfplay_config.yaml",
    ]

    for config_name in critical_configs:
        config_path = config_dir / config_name
        if config_path.exists():
            results["checks"][f"config_{config_name}"] = {"exists": True, "ok": True}
            print(f"✓ Config: {config_name}")
        else:
            results["checks"][f"config_{config_name}"] = {"exists": False, "ok": False}
            print(f"✗ Config: {config_name} MISSING")
            results["errors"].append(f"Missing critical config: {config_name}")
            results["passed"] = False

    for config_name in optional_configs:
        config_path = config_dir / config_name
        if config_path.exists():
            if verbose:
                print(f"✓ Config: {config_name} (optional)")
        else:
            if verbose:
                print(f"○ Config: {config_name} not found (optional)")

    # 6. Check daemon dependencies
    deps_ok = True
    daemon_deps = {
        DaemonType.AUTO_SYNC: ["app.coordination.auto_sync_daemon"],
        DaemonType.P2P_BACKEND: ["aiohttp"],
        DaemonType.QUEUE_POPULATOR: ["app.coordination.queue_populator"],
        DaemonType.IDLE_RESOURCE: ["app.coordination.idle_resource_daemon"],
    }

    for daemon in daemons:
        if daemon in daemon_deps:
            for dep in daemon_deps[daemon]:
                try:
                    __import__(dep.split(".")[0])
                except ImportError as e:
                    deps_ok = False
                    results["warnings"].append(f"{daemon.value} requires {dep}: {e}")
                    if verbose:
                        print(f"○ {daemon.value}: Missing dependency {dep}")

    if deps_ok:
        print(f"✓ Dependencies: All required modules available")
    else:
        print(f"○ Dependencies: Some optional modules missing (see warnings)")

    # 7. Summary
    print("")
    print("-" * 60)
    print(f"Daemons to start: {len(daemons)}")
    for daemon in sorted(daemons, key=lambda d: d.value):
        print(f"  • {daemon.value}")
    print("-" * 60)

    if results["warnings"]:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for w in results["warnings"]:
            print(f"  ⚠ {w}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"  ✗ {e}")

    print("=" * 60)

    return results


async def start_daemons(
    manager: DaemonManager,
    daemons: list[DaemonType],
    args: argparse.Namespace,
) -> None:
    """Start specified daemons."""
    if not daemons:
        logger.info("No daemons specified to start")
        return

    # Run startup diagnostics unless skipped
    if not getattr(args, "skip_diagnostics", False):
        diag = await run_startup_diagnostics(daemons, verbose=args.verbose)

        if not diag["passed"]:
            logger.error("Startup diagnostics failed - cannot start daemons")
            for err in diag["errors"]:
                logger.error(f"  {err}")
            return
    else:
        logger.info("Skipping startup diagnostics (--skip-diagnostics)")

    logger.info(f"Starting {len(daemons)} daemon(s)...")

    # Register adapter-based daemons
    register_all_adapters_with_manager()

    # Start daemons
    results = await manager.start_all(daemons)

    # Report results
    success = sum(1 for v in results.values() if v)
    failed = len(results) - success

    if args.verbose:
        for daemon_type, started in results.items():
            status = "started" if started else "FAILED"
            logger.info(f"  {daemon_type.value}: {status}")

    logger.info(f"Started {success}/{len(results)} daemons ({failed} failed)")


async def stop_daemons(
    manager: DaemonManager,
    daemons: list[str] | None,
    args: argparse.Namespace,
) -> None:
    """Stop specified or all daemons."""
    if daemons:
        for name in daemons:
            try:
                daemon_type = DaemonType(name)
                await manager.stop(daemon_type)
                logger.info(f"Stopped {name}")
            except ValueError:
                logger.warning(f"Unknown daemon: {name}")
    else:
        logger.info("Stopping all daemons...")
        await manager.stop_all()
        logger.info("All daemons stopped")


async def run_foreground(manager: DaemonManager) -> None:
    """Run in foreground, waiting for shutdown signal."""
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info("Running in foreground. Press Ctrl+C to stop.")
    await shutdown_event.wait()

    logger.info("Shutting down...")
    await manager.shutdown()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Configure manager
    config = DaemonManagerConfig(
        health_check_interval=args.health_interval,
        auto_restart_failed=not args.no_auto_restart,
    )

    manager = get_daemon_manager(config)

    # Set up signal handlers
    setup_signal_handlers()

    try:
        # Handle actions
        if args.status:
            if args.watch:
                await watch_status(manager, args)
            else:
                await show_status(manager, args)
            return 0

        if args.stop_all:
            await stop_daemons(manager, None, args)
            return 0

        if args.stop_daemons:
            await stop_daemons(manager, args.stop_daemons, args)
            return 0

        # Handle diagnostics-only mode
        if args.diagnostics:
            daemons = get_daemons_to_start(args)
            if not daemons:
                # Default to all daemons for diagnostics
                daemons = list(DaemonType)
            diag = await run_startup_diagnostics(daemons, verbose=args.verbose)
            return 0 if diag["passed"] else 1

        # Determine daemons to start
        daemons = get_daemons_to_start(args)

        if not daemons and not args.status:
            print("No daemons specified. Use --help for options.")
            return 1

        # Start daemons
        await start_daemons(manager, daemons, args)

        # If daemons were started, run in foreground
        if daemons:
            await run_foreground(manager)

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted")
        await manager.shutdown()
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
