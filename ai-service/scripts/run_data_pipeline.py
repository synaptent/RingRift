#!/usr/bin/env python3
"""Automated data pipeline for distributed selfplay aggregation and training.

This script automates the complete data pipeline:
1. Sync JSONL files from all distributed compute sources
2. Convert JSONL files to SQLite using chunked parallel processing
3. Optionally aggregate with legacy method
4. Optionally trigger CMA-ES optimization runs
5. Generate statistics and reports

Designed to run as a cron job or continuous daemon for hands-off operation.

Usage:
    # Run complete pipeline once (sync + convert)
    python scripts/run_data_pipeline.py

    # Sync only (no conversion)
    python scripts/run_data_pipeline.py --sync-only

    # Convert only (skip sync, use existing files)
    python scripts/run_data_pipeline.py --convert-only

    # Convert with more workers
    python scripts/run_data_pipeline.py --convert-only --workers 4

    # Skip conversion, use legacy aggregate
    python scripts/run_data_pipeline.py --no-convert --aggregate-only

    # Run continuous daemon (sync + convert every 5 minutes)
    python scripts/run_data_pipeline.py --daemon --interval 300

    # Include CMA-ES optimization after conversion
    python scripts/run_data_pipeline.py --with-cmaes

    # Filter by board type
    python scripts/run_data_pipeline.py --board-type square8 --num-players 2

    # Dry run
    python scripts/run_data_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SyncSource:
    """Configuration for a remote selfplay data source."""

    name: str
    host: str  # user@hostname format
    remote_path: str  # Remote comprehensive selfplay directory
    ssh_key: Optional[str] = None
    ssh_port: int = 22


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    sync_started: Optional[str] = None
    sync_completed: Optional[str] = None
    sources_synced: Dict[str, bool] = field(default_factory=dict)
    total_jsonl_lines: int = 0
    games_imported: int = 0
    games_skipped_duplicate: int = 0
    aggregation_completed: Optional[str] = None
    cmaes_triggered: bool = False
    errors: List[str] = field(default_factory=list)


# Default sync sources - loaded from config/distributed_hosts.yaml
def load_sync_sources_from_config() -> List[SyncSource]:
    """Load sync sources from distributed_hosts.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print(f"Warning: {config_path} not found. Using empty sync sources.")
        return []

    with open(config_path) as f:
        config = yaml.safe_load(f)

    sources = []
    for name, host_config in config.get("hosts", {}).items():
        if host_config.get("status") != "ready":
            continue
        sources.append(SyncSource(
            name=name.replace("-", "_"),
            host=f"{host_config.get('ssh_user', 'ubuntu')}@{host_config.get('ssh_host', '')}",
            remote_path=f"{host_config.get('ringrift_path', '~/ringrift')}/ai-service/data/selfplay/comprehensive",
            ssh_key=host_config.get("ssh_key"),
            ssh_port=host_config.get("ssh_port", 22),
        ))
    return sources

DEFAULT_SYNC_SOURCES = load_sync_sources_from_config()


class DataPipeline:
    """Automated data pipeline for distributed selfplay aggregation."""

    def __init__(
        self,
        aggregated_dir: str = "ai-service/data/selfplay/aggregated",
        output_db_dir: str = "ai-service/data/games",
        comprehensive_dir: str = "ai-service/data/selfplay/comprehensive",
        sync_sources: Optional[List[SyncSource]] = None,
        dry_run: bool = False,
    ):
        self.aggregated_dir = Path(aggregated_dir)
        self.output_db_dir = Path(output_db_dir)
        self.comprehensive_dir = Path(comprehensive_dir)
        self.sync_sources = sync_sources or DEFAULT_SYNC_SOURCES
        self.dry_run = dry_run
        self.stats = PipelineStats()

        # Ensure directories exist
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)
        self.output_db_dir.mkdir(parents=True, exist_ok=True)

    def sync_from_source(self, source: SyncSource) -> bool:
        """Sync JSONL files from a single source.

        Returns True if sync succeeded.
        """
        local_dest = self.aggregated_dir / source.name
        local_dest.mkdir(parents=True, exist_ok=True)

        # Build rsync command
        ssh_opts = f"-o ConnectTimeout=30 -o ServerAliveInterval=10"
        if source.ssh_key:
            ssh_key_expanded = os.path.expanduser(source.ssh_key)
            ssh_opts += f" -i {ssh_key_expanded}"
        if source.ssh_port != 22:
            ssh_opts += f" -p {source.ssh_port}"

        remote_path = f"{source.host}:{source.remote_path}/"
        cmd = [
            "rsync",
            "-az",
            "--timeout=60",
            "-e", f"ssh {ssh_opts}",
            remote_path,
            str(local_dest) + "/",
        ]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would sync from {source.name}: {' '.join(cmd)}")
            return True

        logger.info(f"Syncing from {source.name}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"  {source.name}: OK")
                return True
            else:
                logger.warning(f"  {source.name}: FAIL (exit {result.returncode})")
                if result.stderr:
                    logger.debug(f"  stderr: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"  {source.name}: TIMEOUT")
            return False
        except Exception as e:
            logger.warning(f"  {source.name}: ERROR - {e}")
            return False

    def sync_local_comprehensive(self) -> bool:
        """Copy local comprehensive data to aggregated directory."""
        local_dest = self.aggregated_dir / "local"
        local_dest.mkdir(parents=True, exist_ok=True)

        if not self.comprehensive_dir.exists():
            logger.warning(f"Local comprehensive dir not found: {self.comprehensive_dir}")
            return False

        if self.dry_run:
            logger.info(f"[DRY RUN] Would copy local comprehensive to {local_dest}")
            return True

        cmd = [
            "rsync",
            "-az",
            str(self.comprehensive_dir) + "/",
            str(local_dest) + "/",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info("  Local: OK")
                return True
            else:
                logger.warning(f"  Local: FAIL (exit {result.returncode})")
                return False
        except Exception as e:
            logger.warning(f"  Local: ERROR - {e}")
            return False

    def sync_all_sources(self) -> Dict[str, bool]:
        """Sync from all configured sources.

        Returns dict mapping source name to success status.
        """
        self.stats.sync_started = datetime.now().isoformat()
        results = {}

        logger.info("=" * 60)
        logger.info("SYNCING FROM DISTRIBUTED SOURCES")
        logger.info("=" * 60)

        # Sync from remote sources
        for source in self.sync_sources:
            results[source.name] = self.sync_from_source(source)

        # Sync local comprehensive
        results["local"] = self.sync_local_comprehensive()

        self.stats.sources_synced = results
        self.stats.sync_completed = datetime.now().isoformat()

        # Summary
        success_count = sum(1 for v in results.values() if v)
        logger.info("")
        logger.info(f"Sync complete: {success_count}/{len(results)} sources succeeded")

        return results

    def count_jsonl_lines(self) -> Dict[str, int]:
        """Count JSONL lines per source for reporting."""
        counts = {}

        for source_dir in sorted(self.aggregated_dir.iterdir()):
            if not source_dir.is_dir():
                continue

            total_lines = 0
            for jsonl_file in source_dir.rglob("games.jsonl"):
                try:
                    with open(jsonl_file, "r") as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    pass

            counts[source_dir.name] = total_lines

        return counts

    def run_chunked_conversion(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        workers: int = 2,
    ) -> Dict[str, Any]:
        """Run chunked JSONL to SQLite conversion.

        This uses the chunked_jsonl_converter.py for robust handling of
        large files with parallel processing.

        Returns conversion statistics.
        """
        if input_dir is None:
            input_dir = self.aggregated_dir
        if output_dir is None:
            output_dir = self.output_db_dir

        logger.info("")
        logger.info("=" * 60)
        logger.info("CHUNKED JSONL CONVERSION")
        logger.info("=" * 60)
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Workers: {workers}")

        converter_script = Path(__file__).parent / "chunked_jsonl_converter.py"
        if not converter_script.exists():
            logger.error(f"Chunked converter not found: {converter_script}")
            return {"success": False, "error": "converter not found"}

        # Build command
        cmd = [
            sys.executable,
            str(converter_script),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--workers", str(workers),
            "--chunk-size", "500",
        ]

        if board_type:
            cmd.extend(["--board-type", board_type])
        if num_players:
            cmd.extend(["--num-players", str(num_players)])
        if self.dry_run:
            cmd.append("--dry-run")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout for large backlogs
            )

            stats = {"success": result.returncode == 0}

            # Parse output for stats
            for line in result.stdout.split("\n"):
                if "Games:" in line and "added" in line:
                    try:
                        # Extract games added from summary line
                        import re
                        match = re.search(r"(\d+,?\d*)\s+added", line)
                        if match:
                            stats["games_converted"] = int(match.group(1).replace(",", ""))
                    except Exception:
                        pass

            if result.returncode == 0:
                logger.info("Chunked conversion completed successfully")
                logger.info(f"Games converted: {stats.get('games_converted', 'unknown')}")
            else:
                logger.error(f"Chunked conversion failed: {result.stderr[:500]}")
                self.stats.errors.append(f"Chunked conversion failed: {result.stderr[:200]}")

            return stats

        except subprocess.TimeoutExpired:
            logger.error("Chunked conversion timed out")
            self.stats.errors.append("Chunked conversion timed out")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Chunked conversion error: {e}")
            self.stats.errors.append(f"Chunked conversion error: {e}")
            return {"success": False, "error": str(e)}

    def aggregate_to_database(
        self,
        output_db: Optional[str] = None,
        sources: Optional[List[str]] = None,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run JSONL to SQLite aggregation.

        Returns aggregation statistics.
        """
        if output_db is None:
            # Generate default output path based on filters
            suffix_parts = []
            if board_type:
                suffix_parts.append(board_type)
            if num_players:
                suffix_parts.append(f"{num_players}p")
            if suffix_parts:
                suffix = "_".join(suffix_parts)
                output_db = str(self.output_db_dir / f"training_{suffix}.db")
            else:
                output_db = str(self.output_db_dir / "training_aggregated.db")

        logger.info("")
        logger.info("=" * 60)
        logger.info("AGGREGATING TO DATABASE")
        logger.info("=" * 60)
        logger.info(f"Input: {self.aggregated_dir}")
        logger.info(f"Output: {output_db}")

        # Build command
        cmd = [
            sys.executable,
            "ai-service/scripts/aggregate_jsonl_to_db.py",
            "--input-dir", str(self.aggregated_dir),
            "--output-db", output_db,
        ]

        if sources:
            cmd.extend(["--sources"] + sources)
        if board_type:
            cmd.extend(["--board-type", board_type])
        if num_players:
            cmd.extend(["--num-players", str(num_players)])
        if self.dry_run:
            cmd.append("--dry-run")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            # Parse output for stats
            stats = {"output_db": output_db, "success": result.returncode == 0}

            # Extract stats from output
            for line in result.stdout.split("\n"):
                if "Successfully imported:" in line:
                    try:
                        stats["imported"] = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Skipped (duplicate):" in line:
                    try:
                        stats["skipped_duplicate"] = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

            if result.returncode == 0:
                logger.info("Aggregation completed successfully")
                self.stats.aggregation_completed = datetime.now().isoformat()
                self.stats.games_imported = stats.get("imported", 0)
                self.stats.games_skipped_duplicate = stats.get("skipped_duplicate", 0)
            else:
                logger.error(f"Aggregation failed: {result.stderr[:500]}")
                self.stats.errors.append(f"Aggregation failed: {result.stderr[:200]}")

            return stats

        except subprocess.TimeoutExpired:
            logger.error("Aggregation timed out")
            self.stats.errors.append("Aggregation timed out")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            self.stats.errors.append(f"Aggregation error: {e}")
            return {"success": False, "error": str(e)}

    def trigger_cmaes(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        generations: int = 50,
        population: int = 16,
    ) -> bool:
        """Trigger a CMA-ES optimization run.

        Returns True if CMA-ES started successfully.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRIGGERING CMA-ES OPTIMIZATION")
        logger.info("=" * 60)

        cmd = [
            sys.executable,
            "ai-service/scripts/run_cmaes_optimization.py",
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--generations", str(generations),
            "--population", str(population),
            "--selfplay-data-dir", str(self.aggregated_dir),
        ]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
            return True

        logger.info(f"Command: {' '.join(cmd[:6])}...")

        try:
            # Run CMA-ES in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            logger.info(f"CMA-ES started with PID {process.pid}")
            self.stats.cmaes_triggered = True
            return True

        except Exception as e:
            logger.error(f"Failed to start CMA-ES: {e}")
            self.stats.errors.append(f"CMA-ES start failed: {e}")
            return False

    def run_pipeline(
        self,
        sync: bool = True,
        convert: bool = True,
        aggregate: bool = True,
        cmaes: bool = False,
        sources: Optional[List[str]] = None,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        workers: int = 2,
    ) -> PipelineStats:
        """Run the complete data pipeline.

        Args:
            sync: Whether to sync from remote sources
            convert: Whether to run chunked JSONL conversion (recommended)
            aggregate: Whether to aggregate JSONL to SQLite (legacy method)
            cmaes: Whether to trigger CMA-ES after aggregation
            sources: Filter to specific sources
            board_type: Filter to specific board type
            num_players: Filter to specific player count
            workers: Number of parallel workers for conversion

        Returns:
            PipelineStats with run statistics
        """
        logger.info("=" * 60)
        logger.info("DATA PIPELINE RUN")
        logger.info(f"Started: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        logger.info("")

        # Phase 1: Sync
        if sync:
            self.sync_all_sources()

        # Report line counts
        line_counts = self.count_jsonl_lines()
        self.stats.total_jsonl_lines = sum(line_counts.values())
        logger.info("")
        logger.info("JSONL line counts by source:")
        for source, count in sorted(line_counts.items()):
            logger.info(f"  {source}: {count:,}")
        logger.info(f"  TOTAL: {self.stats.total_jsonl_lines:,}")

        # Phase 2: Chunked Conversion (new, recommended)
        if convert:
            self.run_chunked_conversion(
                board_type=board_type,
                num_players=num_players,
                workers=workers,
            )

        # Phase 3: Aggregate (legacy method, can run in addition to convert)
        if aggregate and not convert:
            # Only run legacy aggregate if convert is disabled
            self.aggregate_to_database(
                sources=sources,
                board_type=board_type,
                num_players=num_players,
            )

        # Phase 4: CMA-ES (optional)
        if cmaes and not self.dry_run:
            self.trigger_cmaes(
                board_type=board_type or "square8",
                num_players=num_players or 2,
            )

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE RUN COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total JSONL lines: {self.stats.total_jsonl_lines:,}")
        logger.info(f"Games imported: {self.stats.games_imported:,}")
        logger.info(f"Games skipped (duplicate): {self.stats.games_skipped_duplicate:,}")
        if self.stats.errors:
            logger.warning(f"Errors: {len(self.stats.errors)}")
            for err in self.stats.errors:
                logger.warning(f"  - {err}")

        return self.stats


def run_daemon(
    pipeline: DataPipeline,
    interval: int = 300,
    **pipeline_kwargs,
):
    """Run pipeline as a continuous daemon.

    Args:
        pipeline: DataPipeline instance
        interval: Seconds between runs
        **pipeline_kwargs: Arguments passed to run_pipeline()
    """
    logger.info(f"Starting daemon mode (interval: {interval}s)")

    # Handle graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received shutdown signal, finishing current run...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    iteration = 0
    while running:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"DAEMON ITERATION {iteration}")
        logger.info(f"{'='*60}")

        try:
            pipeline.run_pipeline(**pipeline_kwargs)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")

        if running:
            logger.info(f"\nSleeping for {interval}s until next run...")
            # Sleep in small increments to allow graceful shutdown
            for _ in range(interval):
                if not running:
                    break
                time.sleep(1)

    logger.info("Daemon shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Automated data pipeline for distributed selfplay aggregation"
    )

    # Mode selection
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync from remote sources (no conversion/aggregation)",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only run chunked conversion (no sync)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only run legacy aggregation (no sync, no chunked conversion)",
    )
    parser.add_argument(
        "--with-cmaes",
        action="store_true",
        help="Trigger CMA-ES optimization after conversion",
    )

    # Daemon mode
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as continuous daemon",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between daemon runs (default: 300)",
    )

    # Filters
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        help="Filter to specific sources",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal", "hex8"],
        help="Filter to specific board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        help="Filter to specific player count",
    )

    # Paths
    parser.add_argument(
        "--aggregated-dir",
        type=str,
        default="ai-service/data/selfplay/aggregated",
        help="Directory for aggregated JSONL files",
    )
    parser.add_argument(
        "--output-db-dir",
        type=str,
        default="ai-service/data/games",
        help="Directory for output SQLite databases",
    )
    parser.add_argument(
        "--comprehensive-dir",
        type=str,
        default="ai-service/data/selfplay/comprehensive",
        help="Local comprehensive selfplay directory",
    )

    # Conversion options
    parser.add_argument(
        "--convert",
        action="store_true",
        default=True,
        help="Run chunked JSONL conversion (default: enabled)",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip chunked JSONL conversion",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=min(8, max(2, os.cpu_count() // 2 if os.cpu_count() else 2)),
        help="Number of parallel workers for conversion (default: half of CPU cores, min 2, max 8)",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create pipeline
    pipeline = DataPipeline(
        aggregated_dir=args.aggregated_dir,
        output_db_dir=args.output_db_dir,
        comprehensive_dir=args.comprehensive_dir,
        dry_run=args.dry_run,
    )

    # Determine what to run
    if args.sync_only:
        sync, convert, aggregate = True, False, False
    elif args.convert_only:
        sync, convert, aggregate = False, True, False
    elif args.aggregate_only:
        sync, convert, aggregate = False, False, True
    else:
        # Default: sync + convert (modern pipeline)
        sync = True
        convert = args.convert and not args.no_convert
        aggregate = args.no_convert  # Only use legacy if convert is disabled

    pipeline_kwargs = {
        "sync": sync,
        "convert": convert,
        "aggregate": aggregate,
        "cmaes": args.with_cmaes,
        "sources": args.sources,
        "board_type": args.board_type,
        "num_players": args.num_players,
        "workers": args.workers,
    }

    if args.daemon:
        run_daemon(pipeline, interval=args.interval, **pipeline_kwargs)
    else:
        stats = pipeline.run_pipeline(**pipeline_kwargs)

        # Save stats to file
        stats_file = Path("logs/pipeline/data_pipeline_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump({
                "sync_started": stats.sync_started,
                "sync_completed": stats.sync_completed,
                "sources_synced": stats.sources_synced,
                "total_jsonl_lines": stats.total_jsonl_lines,
                "games_imported": stats.games_imported,
                "games_skipped_duplicate": stats.games_skipped_duplicate,
                "aggregation_completed": stats.aggregation_completed,
                "cmaes_triggered": stats.cmaes_triggered,
                "errors": stats.errors,
            }, f, indent=2)
        logger.info(f"\nStats saved to: {stats_file}")


if __name__ == "__main__":
    main()
