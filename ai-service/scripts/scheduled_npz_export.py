#!/usr/bin/env python3
"""Scheduled NPZ Export Daemon for OWC Data Repository.

Runs on mac-studio to periodically export training data from OWC databases
to NPZ files, which are then served via HTTP for distribution to training nodes.

Features:
- Exports all board configurations automatically
- Runs on configurable schedule (default: every 2 hours)
- Skips recent exports (configurable freshness window)
- Integrates with dynamic_data_distribution.py
- Logs progress and errors

Usage:
    # Run as daemon (every 2 hours)
    python scripts/scheduled_npz_export.py --daemon

    # Run once for all configs
    python scripts/scheduled_npz_export.py --once

    # Run for specific config
    python scripts/scheduled_npz_export.py --once --config hex8_2p

    # Custom interval
    python scripts/scheduled_npz_export.py --daemon --interval 3600

Environment:
    OWC_DATA_PATH: Path to OWC external drive (default: /Volumes/RingRift-Data)
    EXPORT_OUTPUT_DIR: Output directory for NPZ files (default: {OWC}/canonical_data)

December 2025: Created for automated training data freshness.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Configuration
OWC_DATA_PATH = Path(os.getenv("OWC_DATA_PATH", "/Volumes/RingRift-Data"))
DEFAULT_OUTPUT_DIR = OWC_DATA_PATH / "canonical_data"
DEFAULT_INTERVAL = 7200  # 2 hours
FRESHNESS_WINDOW = 3600  # 1 hour - skip if exported within this window
LOG_FILE = Path("/tmp/scheduled_npz_export.log")
# Explicit Python path for subprocess (needed on mac-studio)
# Default to venv Python if it exists, otherwise use PYTHON_EXECUTABLE env var or sys.executable
_venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
PYTHON_EXECUTABLE = os.getenv(
    "PYTHON_EXECUTABLE",
    str(_venv_python) if _venv_python.exists() else sys.executable
)

# Board configurations to export
EXPORT_CONFIGS = [
    ("hex8", 2),
    ("hex8", 3),
    ("hex8", 4),
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("scheduled_export")


@dataclass
class ExportResult:
    """Result of an export operation."""
    config: str
    success: bool
    samples: int = 0
    duration: float = 0.0
    error: str = ""
    output_path: str = ""


class ScheduledExportDaemon:
    """Daemon for scheduled NPZ exports from OWC databases."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        interval: int = DEFAULT_INTERVAL,
        freshness_window: int = FRESHNESS_WINDOW,
    ):
        self.output_dir = output_dir
        self.interval = interval
        self.freshness_window = freshness_window
        self._running = False
        self._stats = {
            "cycles_completed": 0,
            "total_exports": 0,
            "total_samples": 0,
            "last_cycle_time": 0.0,
        }

    def _find_databases(self, board_type: str, num_players: int) -> list[Path]:
        """Find all databases for a configuration on OWC."""
        import sqlite3

        databases = []
        patterns = [
            f"*{board_type}*{num_players}p*.db",
            f"*{board_type}_{num_players}p*.db",
            f"*{board_type}{num_players}*.db",
        ]

        search_dirs = [
            OWC_DATA_PATH / "canonical_games",
            OWC_DATA_PATH / "cluster_games",
            OWC_DATA_PATH / "selfplay_repository",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for pattern in patterns:
                databases.extend(search_dir.glob(pattern))

        # Also search for jsonl_aggregated.db files which contain all configs
        jsonl_dbs = list((OWC_DATA_PATH / "selfplay_repository" / "raw").glob("**/jsonl_aggregated.db"))
        for db_path in jsonl_dbs[:20]:  # Limit to 20 to avoid timeout
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM games
                    WHERE board_type = ? AND num_players = ?
                    AND game_status = 'completed' AND total_moves >= 5
                """, (board_type, num_players))
                count = cursor.fetchone()[0]
                conn.close()
                if count > 10:  # Only include if it has meaningful data
                    databases.append(db_path)
                    logger.info(f"  Found {count} {board_type}_{num_players}p games in {db_path.parent.name}")
            except sqlite3.Error as e:
                logger.debug(f"Could not query {db_path}: {e}")
            except OSError as e:
                logger.debug(f"Could not access {db_path}: {e}")

        return list(set(databases))

    def _should_export(self, config: str) -> tuple[bool, str]:
        """Check if config should be exported based on freshness."""
        output_path = self.output_dir / f"{config}.npz"

        if not output_path.exists():
            return True, "No existing export"

        mtime = output_path.stat().st_mtime
        age = time.time() - mtime

        if age < self.freshness_window:
            return False, f"Recent export ({age/60:.0f}m ago)"

        return True, f"Export is {age/3600:.1f}h old"

    async def _run_export(
        self,
        board_type: str,
        num_players: int,
    ) -> ExportResult:
        """Run export for a single configuration."""
        config = f"{board_type}_{num_players}p"
        output_path = self.output_dir / f"{config}.npz"

        # Find databases
        databases = self._find_databases(board_type, num_players)
        if not databases:
            return ExportResult(
                config=config,
                success=False,
                error="No databases found",
            )

        # Build export command
        # Use the export script from ai-service
        script_dir = Path(__file__).parent
        export_script = script_dir / "export_replay_dataset.py"

        if not export_script.exists():
            return ExportResult(
                config=config,
                success=False,
                error=f"Export script not found: {export_script}",
            )

        # Build command with explicit database paths
        cmd = [
            PYTHON_EXECUTABLE,
            str(export_script),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--output", str(output_path),
            "--require-completed",
            "--min-moves", "5",
            "--allow-noncanonical",  # Allow non-registry databases
        ]

        # Add database paths
        for db in databases[:10]:  # Limit to 10 databases
            cmd.extend(["--db", str(db)])

        logger.info(f"Exporting {config}: {len(databases)} databases -> {output_path}")

        # Feb 2026: Cross-process export coordination
        try:
            from app.coordination.export_coordinator import get_export_coordinator
            _coord = get_export_coordinator()
            if not _coord.try_acquire(config):
                logger.info(f"  {config}: SKIPPED - cross-process export slot unavailable")
                return ExportResult(config=config, success=False, error="Export slot unavailable")
            _release_slot = True
        except Exception:
            _release_slot = False

        start_time = time.time()
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(script_dir.parent),
                env={**os.environ, "PYTHONPATH": str(script_dir.parent)},
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=1800,  # 30 minute timeout
            )

            duration = time.time() - start_time

            if process.returncode == 0:
                # Parse sample count from output
                samples = 0
                output_text = stdout.decode()
                for line in output_text.split("\n"):
                    if "samples" in line.lower():
                        import re
                        match = re.search(r"(\d+)\s*samples", line, re.IGNORECASE)
                        if match:
                            samples = int(match.group(1))
                            break

                logger.info(f"  {config}: OK - {samples} samples in {duration:.1f}s")
                return ExportResult(
                    config=config,
                    success=True,
                    samples=samples,
                    duration=duration,
                    output_path=str(output_path),
                )
            else:
                error = stderr.decode()[:200]
                logger.error(f"  {config}: FAILED - {error}")
                return ExportResult(
                    config=config,
                    success=False,
                    duration=duration,
                    error=error,
                )

        except asyncio.TimeoutError:
            return ExportResult(
                config=config,
                success=False,
                duration=time.time() - start_time,
                error="Export timed out",
            )
        except Exception as e:
            return ExportResult(
                config=config,
                success=False,
                duration=time.time() - start_time,
                error=str(e),
            )
        finally:
            # Feb 2026: Release cross-process export slot
            if _release_slot:
                try:
                    _coord.release(config)
                except Exception:
                    pass

    async def run_export_cycle(self, specific_config: str | None = None) -> list[ExportResult]:
        """Run export cycle for all (or specific) configurations."""
        results = []
        cycle_start = time.time()

        logger.info("=" * 60)
        logger.info("Starting NPZ export cycle")
        logger.info(f"Output directory: {self.output_dir}")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Filter configs if specific one requested
        configs = EXPORT_CONFIGS
        if specific_config:
            parts = specific_config.split("_")
            if len(parts) == 2 and parts[1].endswith("p"):
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))
                configs = [(board_type, num_players)]

        # Export each configuration
        for board_type, num_players in configs:
            config = f"{board_type}_{num_players}p"

            # Check freshness
            should_export, reason = self._should_export(config)
            if not should_export:
                logger.info(f"Skipping {config}: {reason}")
                results.append(ExportResult(
                    config=config,
                    success=True,
                    error=f"Skipped: {reason}",
                ))
                continue

            result = await self._run_export(board_type, num_players)
            results.append(result)

            if result.success:
                self._stats["total_exports"] += 1
                self._stats["total_samples"] += result.samples

        # Summary
        successful = sum(1 for r in results if r.success and not r.error.startswith("Skipped"))
        skipped = sum(1 for r in results if r.error.startswith("Skipped"))
        failed = sum(1 for r in results if not r.success)
        total_samples = sum(r.samples for r in results)

        self._stats["cycles_completed"] += 1
        self._stats["last_cycle_time"] = time.time()

        logger.info(f"\nCycle complete: {successful} exported, {skipped} skipped, {failed} failed")
        logger.info(f"Total samples this cycle: {total_samples}")
        logger.info(f"Duration: {time.time() - cycle_start:.1f}s")

        return results

    async def run_daemon(self) -> None:
        """Run as a daemon, exporting on schedule."""
        self._running = True

        logger.info("=" * 60)
        logger.info("Starting Scheduled NPZ Export Daemon")
        logger.info(f"  Interval: {self.interval}s ({self.interval/3600:.1f}h)")
        logger.info(f"  Freshness window: {self.freshness_window}s ({self.freshness_window/60:.0f}m)")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 60)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown())

        while self._running:
            try:
                await self.run_export_cycle()
            except Exception as e:
                logger.error(f"Export cycle failed: {e}", exc_info=True)

            if self._running:
                logger.info(f"Next cycle in {self.interval/60:.0f} minutes...")
                await asyncio.sleep(self.interval)

        logger.info("Daemon stopped")

    def _shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        return {
            **self._stats,
            "running": self._running,
            "output_dir": str(self.output_dir),
            "interval_seconds": self.interval,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Scheduled NPZ export daemon for OWC data repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--once", action="store_true", help="Run one export cycle")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                       help=f"Export interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--freshness", type=int, default=FRESHNESS_WINDOW,
                       help=f"Skip if exported within this window (default: {FRESHNESS_WINDOW}s)")
    parser.add_argument("--config", type=str, help="Export specific config (e.g., hex8_2p)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help=f"Output directory for NPZ files (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    daemon = ScheduledExportDaemon(
        output_dir=Path(args.output_dir),
        interval=args.interval,
        freshness_window=args.freshness,
    )

    if args.daemon:
        asyncio.run(daemon.run_daemon())
    elif args.once:
        results = asyncio.run(daemon.run_export_cycle(args.config))
        # Print summary
        for r in results:
            status = "OK" if r.success else "FAILED"
            print(f"{r.config}: {status} - {r.samples} samples")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
