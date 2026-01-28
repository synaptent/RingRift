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
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

# Add ai-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# S3 configuration
S3_BUCKET = os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
S3_GAMES_PREFIX = os.getenv("RINGRIFT_S3_GAMES_PREFIX", "consolidated/games/")


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


def scan_s3_games(verbose: bool = False) -> dict[str, int]:
    """Scan S3 bucket for canonical game databases and count games.

    Downloads each canonical database header to count games without
    downloading the entire file.

    Returns:
        Dict mapping config_key -> game count
    """
    result = {}

    # List canonical databases in S3
    try:
        cmd = ["aws", "s3", "ls", f"s3://{S3_BUCKET}/{S3_GAMES_PREFIX}"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            if verbose:
                print(f"S3 list failed: {proc.stderr}")
            return result

        # Parse S3 listing for canonical databases
        canonical_dbs = []
        for line in proc.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                filename = parts[3]
                # Match canonical_<board>_<n>p.db pattern
                if filename.startswith("canonical_") and filename.endswith(".db"):
                    # Extract config key from filename
                    # canonical_hex8_2p.db -> hex8_2p
                    config_key = filename.replace("canonical_", "").replace(".db", "")
                    # Validate it's a proper config (board_Np format)
                    if config_key.endswith(("_2p", "_3p", "_4p")):
                        canonical_dbs.append((filename, config_key))

        if verbose:
            print(f"Found {len(canonical_dbs)} canonical databases in S3")

        # For each canonical DB, download just enough to count games
        for filename, config_key in canonical_dbs:
            try:
                s3_path = f"s3://{S3_BUCKET}/{S3_GAMES_PREFIX}{filename}"

                # Use aws s3 cp with range to get just the header
                # SQLite databases have the table info near the start
                # Download first 1MB to query game count
                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                    tmp_path = tmp.name

                # Download full file (needed for SQLite query)
                # For efficiency, we could use sqlite3 remote access or
                # just estimate based on file size, but for accuracy let's
                # download the smaller DBs and estimate larger ones

                # Get file size first
                size_cmd = ["aws", "s3", "ls", s3_path]
                size_proc = subprocess.run(size_cmd, capture_output=True, text=True, timeout=10)
                if size_proc.returncode != 0:
                    continue

                size_parts = size_proc.stdout.strip().split()
                if len(size_parts) >= 3:
                    file_size = int(size_parts[2])
                else:
                    continue

                # For files > 100MB, estimate based on size (~4KB per game average)
                if file_size > 100_000_000:
                    # Estimation: average game record is ~4KB
                    game_estimate = file_size // 4096
                    result[config_key] = game_estimate
                    if verbose:
                        print(f"  {config_key}: ~{game_estimate:,} games (estimated from {file_size:,} bytes)")
                else:
                    # Download and count for smaller files
                    dl_cmd = ["aws", "s3", "cp", s3_path, tmp_path, "--quiet"]
                    dl_proc = subprocess.run(dl_cmd, capture_output=True, timeout=120)

                    if dl_proc.returncode == 0:
                        try:
                            conn = sqlite3.connect(tmp_path)
                            # Try different schema variants (game_status vs status)
                            for query in [
                                "SELECT COUNT(*) FROM games WHERE game_status='completed'",
                                "SELECT COUNT(*) FROM games WHERE status='completed'",
                                "SELECT COUNT(*) FROM games",  # Fallback: count all
                            ]:
                                try:
                                    cursor = conn.execute(query)
                                    count = cursor.fetchone()[0]
                                    break
                                except sqlite3.OperationalError:
                                    continue
                            else:
                                count = 0
                            conn.close()
                            if count > 0:
                                result[config_key] = count
                                if verbose:
                                    print(f"  {config_key}: {count:,} games (actual)")
                        except sqlite3.Error:
                            pass
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass

            except (subprocess.TimeoutExpired, OSError, ValueError):
                continue

    except subprocess.TimeoutExpired:
        if verbose:
            print("S3 scan timed out")
    except Exception as e:
        if verbose:
            print(f"S3 scan error: {e}")

    return result


def get_cluster_data_status(
    config_filter: str | None = None,
    scan_s3: bool = False,
    verbose: bool = False,
) -> dict:
    """Get unified data status from registry.

    Args:
        config_filter: Optional config key to filter (e.g., "hex8_2p")
        scan_s3: If True, directly scan S3 bucket for game counts
        verbose: If True, print progress during S3 scan

    Returns:
        Dict with status, configs, totals, and metadata
    """
    from app.distributed.data_catalog import get_data_registry

    registry = get_data_registry()
    status = registry.get_cluster_status()

    # Optionally scan S3 directly for game counts
    if scan_s3:
        if verbose:
            print("Scanning S3 bucket for game counts...")
        s3_counts = scan_s3_games(verbose=verbose)
        for config_key, count in s3_counts.items():
            if config_key not in status:
                status[config_key] = {"local": 0, "cluster": 0, "owc": 0, "s3": 0, "total": 0}
            status[config_key]["s3"] = count
            status[config_key]["total"] = (
                status[config_key]["local"]
                + status[config_key]["cluster"]
                + status[config_key]["owc"]
                + status[config_key]["s3"]
            )

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
    parser.add_argument(
        "--scan-s3",
        action="store_true",
        help="Directly scan S3 bucket for game counts (slower but accurate)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output during S3 scan",
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

        data = get_cluster_data_status(
            config_filter=args.config,
            scan_s3=args.scan_s3,
            verbose=args.verbose or (args.scan_s3 and args.format == "table"),
        )

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
