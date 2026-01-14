#!/usr/bin/env python3
"""Unified Data Status - Show game counts across all storage locations.

This script provides a comprehensive view of game data availability across:
- LOCAL: Local canonical databases
- CLUSTER: Games on Lambda GH200, Vast.ai, RunPod, Nebius nodes
- S3: AWS S3 bucket (ringrift-models-20251214)
- OWC: External drive on mac-studio (/Volumes/RingRift-Data)

Usage:
    # Human-readable table output
    python scripts/data_status.py

    # JSON output for automation
    python scripts/data_status.py --json

    # Specific config only
    python scripts/data_status.py --config hex8_2p

    # Force refresh (bypass cache)
    python scripts/data_status.py --refresh

    # Local only (skip remote sources)
    python scripts/data_status.py --local-only

January 2026: Created as part of unified data discovery infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.unified_game_aggregator import (
    AggregatedGameCount,
    GameSourceConfig,
    UnifiedGameAggregator,
    get_unified_game_aggregator,
)

logger = logging.getLogger(__name__)

# Constants
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Source display names and paths
SOURCE_INFO = {
    "local": {
        "name": "LOCAL",
        "path": lambda: str(Path(os.getcwd()) / "data" / "games"),
    },
    "cluster": {
        "name": "CLUSTER",
        "path": lambda: "lambda-gh200-*, nebius-*, vast-*, runpod-*",
    },
    "s3": {
        "name": "S3",
        "path": lambda: "s3://ringrift-models-20251214/games/",
    },
    "owc": {
        "name": "OWC",
        "path": lambda: "mac-studio:/Volumes/RingRift-Data/",
    },
}


def format_number(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"


def format_table(
    counts: dict[str, AggregatedGameCount],
    show_sources: bool = True,
) -> str:
    """Format counts as a table.

    Args:
        counts: Dict mapping config_key to AggregatedGameCount
        show_sources: Whether to show per-source breakdown

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"RingRift Data Status ({now})")
    lines.append("=" * 70)
    lines.append("")

    if show_sources:
        # Table with per-source breakdown
        header = f"{'Config':<15} {'LOCAL':>10} {'CLUSTER':>10} {'S3':>10} {'OWC':>10} {'TOTAL':>10}"
        lines.append(header)
        lines.append("-" * 70)

        grand_total = {"local": 0, "cluster": 0, "s3": 0, "owc": 0, "total": 0}

        for config_key in ALL_CONFIGS:
            if config_key not in counts:
                continue

            result = counts[config_key]
            local = result.sources.get("local", 0)
            cluster = result.sources.get("cluster", 0)
            s3 = result.sources.get("s3", 0)
            owc = result.sources.get("owc", 0)
            total = result.total_games

            grand_total["local"] += local
            grand_total["cluster"] += cluster
            grand_total["s3"] += s3
            grand_total["owc"] += owc
            grand_total["total"] += total

            line = (
                f"{config_key:<15} "
                f"{format_number(local):>10} "
                f"{format_number(cluster):>10} "
                f"{format_number(s3):>10} "
                f"{format_number(owc):>10} "
                f"{format_number(total):>10}"
            )
            lines.append(line)

        # Grand total
        lines.append("-" * 70)
        total_line = (
            f"{'TOTAL':<15} "
            f"{format_number(grand_total['local']):>10} "
            f"{format_number(grand_total['cluster']):>10} "
            f"{format_number(grand_total['s3']):>10} "
            f"{format_number(grand_total['owc']):>10} "
            f"{format_number(grand_total['total']):>10}"
        )
        lines.append(total_line)
    else:
        # Simple table with just totals
        header = f"{'Config':<15} {'Total Games':>15}"
        lines.append(header)
        lines.append("-" * 35)

        grand_total = 0
        for config_key in ALL_CONFIGS:
            if config_key not in counts:
                continue
            result = counts[config_key]
            grand_total += result.total_games
            lines.append(f"{config_key:<15} {format_number(result.total_games):>15}")

        lines.append("-" * 35)
        lines.append(f"{'TOTAL':<15} {format_number(grand_total):>15}")

    # Source information
    lines.append("")
    lines.append("Sources:")
    for source_key, info in SOURCE_INFO.items():
        lines.append(f"  - {info['name']}: {info['path']()}")

    # Errors if any
    all_errors = []
    for config_key, result in counts.items():
        for error in result.errors:
            all_errors.append(f"  {config_key}: {error}")

    if all_errors:
        lines.append("")
        lines.append("Warnings:")
        for error in all_errors[:10]:  # Limit to 10 errors
            lines.append(error)
        if len(all_errors) > 10:
            lines.append(f"  ... and {len(all_errors) - 10} more")

    return "\n".join(lines)


def format_json(counts: dict[str, AggregatedGameCount]) -> str:
    """Format counts as JSON.

    Args:
        counts: Dict mapping config_key to AggregatedGameCount

    Returns:
        JSON string
    """
    output = {
        "timestamp": datetime.now().isoformat(),
        "sources": {
            "local": {},
            "cluster": {},
            "s3": {},
            "owc": {},
        },
        "totals": {},
        "grand_total": {
            "local": 0,
            "cluster": 0,
            "s3": 0,
            "owc": 0,
            "total": 0,
        },
        "errors": [],
    }

    for config_key in ALL_CONFIGS:
        if config_key not in counts:
            continue

        result = counts[config_key]

        # Per-source
        for source in ["local", "cluster", "s3", "owc"]:
            count = result.sources.get(source, 0)
            output["sources"][source][config_key] = count
            output["grand_total"][source] += count

        # Total
        output["totals"][config_key] = result.total_games
        output["grand_total"]["total"] += result.total_games

        # Errors
        for error in result.errors:
            output["errors"].append({"config": config_key, "error": error})

    return json.dumps(output, indent=2)


async def get_data_status(
    config: str | None = None,
    local_only: bool = False,
    refresh: bool = False,
) -> dict[str, AggregatedGameCount]:
    """Get data status from all sources.

    Args:
        config: Specific config to query, or None for all
        local_only: If True, skip remote sources
        refresh: If True, bypass cache

    Returns:
        Dict mapping config_key to AggregatedGameCount
    """
    # Create aggregator with appropriate config
    source_config = (
        GameSourceConfig.local_only()
        if local_only
        else GameSourceConfig.all_sources()
    )
    aggregator = UnifiedGameAggregator(source_config)

    # Clear cache if refresh requested
    if refresh:
        aggregator.clear_cache()

    if config:
        # Single config
        parts = config.replace("-", "_").split("_")
        if len(parts) < 2 or not parts[-1].endswith("p"):
            raise ValueError(f"Invalid config format: {config}. Expected format: board_Np (e.g., hex8_2p)")

        board_type = "_".join(parts[:-1])
        num_players = int(parts[-1].rstrip("p"))

        result = await aggregator.get_total_games(board_type, num_players)
        return {result.config_key: result}
    else:
        # All configs
        return await aggregator.get_all_configs_counts()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Show game data status across all storage locations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data_status.py                # Show all configs
  python scripts/data_status.py --json         # JSON output
  python scripts/data_status.py --config hex8_2p  # Single config
  python scripts/data_status.py --local-only   # Skip remote sources
  python scripts/data_status.py --refresh      # Force cache refresh
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Specific config to query (e.g., hex8_2p)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--local-only", "-l",
        action="store_true",
        help="Query local sources only (skip cluster, S3, OWC)",
    )
    parser.add_argument(
        "--refresh", "-r",
        action="store_true",
        help="Force refresh (bypass cache)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide per-source breakdown (totals only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    try:
        # Get data status
        counts = await get_data_status(
            config=args.config,
            local_only=args.local_only,
            refresh=args.refresh,
        )

        # Output
        if args.json:
            print(format_json(counts))
        else:
            print(format_table(counts, show_sources=not args.no_sources))

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
