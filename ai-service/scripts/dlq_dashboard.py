#!/usr/bin/env python3
"""Dead Letter Queue (DLQ) Monitoring Dashboard.

December 29, 2025: Provides visibility into failed events and retry status.

Usage:
    # Show DLQ statistics
    python scripts/dlq_dashboard.py

    # Show pending events
    python scripts/dlq_dashboard.py --pending

    # Show all failed events (including abandoned)
    python scripts/dlq_dashboard.py --failed --include-abandoned

    # Retry pending events
    python scripts/dlq_dashboard.py --retry --max-events 10

    # Purge old events
    python scripts/dlq_dashboard.py --purge --older-than-days 7

    # Watch mode (refresh every 30 seconds)
    python scripts/dlq_dashboard.py --watch --interval 30

    # Show events by type
    python scripts/dlq_dashboard.py --event-type TRAINING_COMPLETED
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.dead_letter_queue import (
    DeadLetterQueue,
    DLQRetryDaemon,
    FailedEvent,
    get_dead_letter_queue,
)


def format_timestamp(ts: str | None) -> str:
    """Format ISO timestamp for display."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return ts[:19] if ts else "N/A"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_stats(dlq: DeadLetterQueue) -> None:
    """Print DLQ statistics."""
    stats = dlq.get_stats()

    print_header("DLQ Statistics")
    print(f"  Total events:      {stats.get('total', 0):,}")
    print(f"  Pending:           {stats.get('pending', 0):,}")
    print(f"  Recovered:         {stats.get('recovered', 0):,}")
    print(f"  Abandoned:         {stats.get('abandoned', 0):,}")

    # Recovery rate
    total = stats.get('total', 0)
    recovered = stats.get('recovered', 0)
    if total > 0:
        recovery_rate = (recovered / total) * 100
        print(f"  Recovery rate:     {recovery_rate:.1f}%")

    # Events by type
    by_type = stats.get('by_event_type', {})
    if by_type:
        print(f"\n  Events by type:")
        for event_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {event_type}: {count:,}")


def print_health(dlq: DeadLetterQueue) -> None:
    """Print DLQ health status."""
    health = dlq.health_check()

    print_header("DLQ Health")
    status = "HEALTHY" if health.healthy else "UNHEALTHY"
    print(f"  Status:    {status}")
    print(f"  Message:   {health.message}")

    if health.details:
        print(f"  Details:")
        for key, value in health.details.items():
            print(f"    {key}: {value}")


def print_events(events: list[FailedEvent], title: str = "Events") -> None:
    """Print list of failed events."""
    print_header(f"{title} ({len(events)} events)")

    if not events:
        print("  No events found.")
        return

    for i, event in enumerate(events, 1):
        print(f"\n  [{i}] Event ID: {event.event_id}")
        print(f"      Type:        {event.event_type}")
        print(f"      Handler:     {event.handler_name}")
        print(f"      Retry count: {event.retry_count}")
        print(f"      Created:     {format_timestamp(event.created_at)}")
        print(f"      Last retry:  {format_timestamp(event.last_retry_at)}")
        print(f"      Source:      {event.source or 'unknown'}")

        # Truncate error message
        error = event.error or ""
        if len(error) > 100:
            error = error[:97] + "..."
        print(f"      Error:       {error}")

        # Show payload summary
        if event.payload:
            payload_str = json.dumps(event.payload)
            if len(payload_str) > 80:
                payload_str = payload_str[:77] + "..."
            print(f"      Payload:     {payload_str}")


def print_daemon_status(daemon: DLQRetryDaemon | None) -> None:
    """Print retry daemon status."""
    print_header("Retry Daemon Status")

    if daemon is None:
        print("  Daemon not initialized.")
        return

    health = daemon.health_check()

    status = "RUNNING" if health.healthy else "STOPPED"
    print(f"  Status:          {status}")
    print(f"  Message:         {health.message}")

    details = health.details or {}
    print(f"  Cycles:          {details.get('cycles', 0):,}")
    print(f"  Recovered:       {details.get('recovered', 0):,}")
    print(f"  Failed retries:  {details.get('failed', 0):,}")
    print(f"  Abandoned:       {details.get('abandoned', 0):,}")


async def retry_events(dlq: DeadLetterQueue, max_events: int = 10) -> None:
    """Retry pending events."""
    print_header(f"Retrying Events (max {max_events})")

    pending = dlq.get_pending_events(limit=max_events)
    if not pending:
        print("  No pending events to retry.")
        return

    print(f"  Found {len(pending)} pending events.")

    results = await dlq.retry_failed_events(max_events=max_events)

    print(f"\n  Retry Results:")
    print(f"    Attempted:  {results.get('attempted', 0)}")
    print(f"    Succeeded:  {results.get('succeeded', 0)}")
    print(f"    Failed:     {results.get('failed', 0)}")
    print(f"    Skipped:    {results.get('skipped', 0)}")


def purge_old_events(dlq: DeadLetterQueue, older_than_days: int = 7) -> None:
    """Purge old events from DLQ."""
    print_header(f"Purging Events Older Than {older_than_days} Days")

    purged = dlq.purge_old_events(older_than_days=older_than_days)

    print(f"  Purged {purged:,} old events.")


def run_dashboard(
    dlq: DeadLetterQueue,
    show_stats: bool = True,
    show_health: bool = True,
    show_pending: bool = False,
    show_failed: bool = False,
    include_abandoned: bool = False,
    event_type: str | None = None,
    limit: int = 20,
) -> None:
    """Run dashboard display."""
    if show_stats:
        print_stats(dlq)

    if show_health:
        print_health(dlq)

    if show_pending:
        events = dlq.get_pending_events(limit=limit, event_type=event_type)
        print_events(events, "Pending Events")

    if show_failed:
        events = dlq.get_failed_events(
            limit=limit,
            event_type=event_type,
            include_abandoned=include_abandoned,
        )
        print_events(events, "Failed Events")

    print()  # Final newline


async def watch_mode(
    dlq: DeadLetterQueue,
    interval: int = 30,
    event_type: str | None = None,
) -> None:
    """Run in watch mode with periodic refresh."""
    print(f"Watching DLQ (refresh every {interval}s, Ctrl+C to stop)...")

    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")

            # Show timestamp
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"DLQ Dashboard - {now}")

            run_dashboard(
                dlq,
                show_stats=True,
                show_health=True,
                show_pending=True,
                show_failed=False,
                event_type=event_type,
                limit=10,
            )

            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dead Letter Queue (DLQ) Monitoring Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Display options
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        default=True,
        help="Show DLQ statistics (default: True)",
    )
    parser.add_argument(
        "--health", "-H",
        action="store_true",
        help="Show health status",
    )
    parser.add_argument(
        "--pending", "-p",
        action="store_true",
        help="Show pending events",
    )
    parser.add_argument(
        "--failed", "-f",
        action="store_true",
        help="Show failed events",
    )
    parser.add_argument(
        "--include-abandoned",
        action="store_true",
        help="Include abandoned events in failed list",
    )
    parser.add_argument(
        "--daemon-status",
        action="store_true",
        help="Show retry daemon status",
    )

    # Filter options
    parser.add_argument(
        "--event-type", "-t",
        type=str,
        help="Filter by event type",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum events to show (default: 20)",
    )

    # Actions
    parser.add_argument(
        "--retry", "-r",
        action="store_true",
        help="Retry pending events",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10,
        help="Maximum events to retry (default: 10)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge old events",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
        help="Purge events older than N days (default: 7)",
    )

    # Watch mode
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch mode with periodic refresh",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)",
    )

    # Database path
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to DLQ database (default: auto-detect)",
    )

    args = parser.parse_args()

    # Initialize DLQ
    if args.db_path:
        dlq = DeadLetterQueue(db_path=args.db_path)
    else:
        dlq = get_dead_letter_queue()

    # Handle actions
    if args.retry:
        asyncio.run(retry_events(dlq, max_events=args.max_events))
        return

    if args.purge:
        purge_old_events(dlq, older_than_days=args.older_than_days)
        return

    if args.watch:
        asyncio.run(watch_mode(dlq, interval=args.interval, event_type=args.event_type))
        return

    # Default: show dashboard
    show_health = args.health or not (args.pending or args.failed)

    run_dashboard(
        dlq,
        show_stats=args.stats,
        show_health=show_health,
        show_pending=args.pending,
        show_failed=args.failed,
        include_abandoned=args.include_abandoned,
        event_type=args.event_type,
        limit=args.limit,
    )

    if args.daemon_status:
        # Try to get daemon instance if it exists
        try:
            from app.coordination.dead_letter_queue import _dlq_retry_daemon
            print_daemon_status(_dlq_retry_daemon)
        except (ImportError, AttributeError):
            print_daemon_status(None)


if __name__ == "__main__":
    main()
