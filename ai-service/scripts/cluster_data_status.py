#!/usr/bin/env python3
"""Cluster Data Status Tool - Show unified game data across all sources.

Jan 2026: Created as part of unified cluster data visibility implementation.

Displays game counts from:
- Local node (local databases)
- Cluster nodes (P2P manifest)
- OWC external drive (mac-studio)
- AWS S3 bucket (archived data)

Usage:
    python scripts/cluster_data_status.py
    python scripts/cluster_data_status.py --refresh
    python scripts/cluster_data_status.py --format table
    python scripts/cluster_data_status.py --format json
    python scripts/cluster_data_status.py --config hex8_2p

Examples:
    # Default table format (auto-refreshes if manifest stale)
    python scripts/cluster_data_status.py

    # Force refresh from P2P leader
    python scripts/cluster_data_status.py --refresh

    # JSON output (for scripting)
    python scripts/cluster_data_status.py --format json

    # Filter to specific config
    python scripts/cluster_data_status.py --config square8_4p
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def format_number(num: int) -> str:
    """Format large numbers with K/M suffix."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1000:
        return f"{num / 1000:.1f}K"
    return str(num)


def get_cluster_data_status(config_filter: str | None = None) -> dict:
    """Get unified data status from registry.

    Args:
        config_filter: Optional config key to filter (e.g., "hex8_2p")

    Returns:
        Dict with status, configs, totals, and metadata
    """
    from app.distributed.data_catalog import get_data_registry

    registry = get_data_registry()
    status = registry.get_cluster_status()

    # Filter if requested
    if config_filter:
        status = {k: v for k, v in status.items() if k == config_filter}

    # Get manifest metadata
    manifest = registry.get_cluster_manifest()
    manifest_age = registry.get_manifest_age_seconds()

    # Compute overall totals
    totals = {
        "local": sum(c["local"] for c in status.values()),
        "cluster": sum(c["cluster"] for c in status.values()),
        "owc": sum(c["owc"] for c in status.values()),
        "s3": sum(c["s3"] for c in status.values()),
        "total": sum(c["total"] for c in status.values()),
    }

    # Get external storage info
    external_info = {}
    if manifest:
        external = getattr(manifest, "external_storage", None)
        if external:
            external_info = {
                "owc_available": getattr(external, "owc_available", False),
                "owc_total_games": getattr(external, "owc_total_games", 0),
                "owc_total_size": getattr(external, "owc_total_size_bytes", 0),
                "owc_scan_error": getattr(external, "owc_scan_error", ""),
                "s3_available": getattr(external, "s3_available", False),
                "s3_bucket": getattr(external, "s3_bucket", ""),
                "s3_total_games": getattr(external, "s3_total_games", 0),
                "s3_total_size": getattr(external, "s3_total_size_bytes", 0),
                "s3_scan_error": getattr(external, "s3_scan_error", ""),
            }

    return {
        "configs": status,
        "totals": totals,
        "manifest_age_seconds": manifest_age,
        "manifest_nodes": getattr(manifest, "total_nodes", 0) if manifest else 0,
        "external_storage": external_info,
    }


def print_table(data: dict) -> None:
    """Print status as formatted table."""
    configs = data["configs"]
    totals = data["totals"]
    manifest_age = data["manifest_age_seconds"]
    manifest_nodes = data["manifest_nodes"]
    external = data["external_storage"]

    # Header
    print("\n" + "=" * 75)
    print("CLUSTER DATA STATUS")
    print("=" * 75)

    if manifest_age >= 0:
        age_str = f"{manifest_age:.0f}s ago" if manifest_age < 300 else f"{manifest_age / 60:.1f}min ago"
        print(f"Manifest: {manifest_nodes} nodes, updated {age_str}")
    else:
        print("Manifest: Not yet received from leader")

    # External storage status
    if external:
        if external.get("owc_available"):
            owc_size = format_size(external.get("owc_total_size", 0))
            print(f"OWC Drive: {format_number(external.get('owc_total_games', 0))} games ({owc_size})")
        elif external.get("owc_scan_error"):
            print(f"OWC Drive: SCAN ERROR - {external.get('owc_scan_error')}")
        else:
            print("OWC Drive: Not scanned (no data)")

        if external.get("s3_available"):
            s3_size = format_size(external.get("s3_total_size", 0))
            print(f"S3 Bucket: {external.get('s3_bucket', 'N/A')} - {format_number(external.get('s3_total_games', 0))} games ({s3_size})")
        elif external.get("s3_scan_error"):
            print(f"S3 Bucket: SCAN ERROR - {external.get('s3_scan_error')}")
        else:
            print("S3 Bucket: Not scanned (no data)")

    print()

    # Config table
    print(f"{'Config':<15} {'Local':>10} {'Cluster':>10} {'OWC':>10} {'S3':>10} {'Total':>12}")
    print("-" * 75)

    # Sort configs by total games (descending)
    sorted_configs = sorted(configs.items(), key=lambda x: x[1]["total"], reverse=True)

    for config_key, counts in sorted_configs:
        local = format_number(counts["local"])
        cluster = format_number(counts["cluster"])
        owc = format_number(counts["owc"])
        s3 = format_number(counts["s3"])
        total = format_number(counts["total"])
        print(f"{config_key:<15} {local:>10} {cluster:>10} {owc:>10} {s3:>10} {total:>12}")

    # Totals
    print("-" * 75)
    print(
        f"{'TOTAL':<15} "
        f"{format_number(totals['local']):>10} "
        f"{format_number(totals['cluster']):>10} "
        f"{format_number(totals['owc']):>10} "
        f"{format_number(totals['s3']):>10} "
        f"{format_number(totals['total']):>12}"
    )
    print("=" * 75 + "\n")


def print_json(data: dict) -> None:
    """Print status as JSON."""
    print(json.dumps(data, indent=2))


async def refresh_manifest(verbose: bool = True) -> bool:
    """Refresh cluster manifest from P2P leader.

    Args:
        verbose: If True, print progress messages.

    Returns:
        True if refresh succeeded, False otherwise.
    """
    from app.distributed.data_catalog import get_data_registry

    registry = get_data_registry()

    if verbose:
        print("Refreshing cluster data from leader...", end=" ", flush=True)

    try:
        success = await registry.refresh_cluster_manifest()
        if verbose:
            if success:
                print("done")
            else:
                print("failed (leader may not have manifest)")
        return success
    except Exception as e:
        if verbose:
            print(f"error: {e}")
        return False


def should_auto_refresh(registry) -> bool:
    """Check if manifest should be auto-refreshed.

    Returns True if:
    - No manifest received yet (age == -1)
    - Manifest is stale (age > 600 seconds)
    """
    age = registry.get_manifest_age_seconds()
    return age < 0 or age > 600


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Show unified cluster data status across all sources.",
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
        "--refresh",
        action="store_true",
        help="Force refresh cluster data from P2P leader",
    )
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Disable automatic refresh when manifest is stale",
    )

    args = parser.parse_args()

    try:
        from app.distributed.data_catalog import get_data_registry

        registry = get_data_registry()

        # Determine if we need to refresh
        needs_refresh = args.refresh or (
            not args.no_auto_refresh and should_auto_refresh(registry)
        )

        if needs_refresh:
            # Run async refresh
            asyncio.run(refresh_manifest(verbose=args.format == "table"))

        data = get_cluster_data_status(config_filter=args.config)

        if args.format == "json":
            print_json(data)
        else:
            print_table(data)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
