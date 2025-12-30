#!/usr/bin/env python3
"""CPU-intensive data pipeline daemon for Vast.ai nodes.

This daemon runs continuously on Vast.ai nodes, handling:
1. Data aggregation from Lambda selfplay nodes
2. SQLite → NPZ export
3. Data quality validation
4. Parity testing
5. NPZ merging and preprocessing

Uses ramdisk (/dev/shm) for temporary storage to maximize throughput.

Usage:
    PYTHONPATH=. python3 scripts/vast_cpu_pipeline_daemon.py \
        --role export \
        --lambda-nodes <node-ip-1>,<node-ip-2> \
        --use-ramdisk
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vast-pipeline")


class RamdiskManager:
    """Manage ramdisk for temporary storage."""

    def __init__(self, base_path: str = "/dev/shm/ringrift"):
        self.base_path = Path(base_path)
        self.enabled = False

    def setup(self) -> bool:
        """Set up ramdisk directories."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            (self.base_path / "tmp").mkdir(exist_ok=True)
            (self.base_path / "export").mkdir(exist_ok=True)
            (self.base_path / "merge").mkdir(exist_ok=True)
            self.enabled = True
            logger.info(f"Ramdisk enabled at {self.base_path}")
            return True
        except Exception as e:
            logger.warning(f"Ramdisk setup failed: {e}, using disk")
            return False

    def get_tmp_path(self) -> Path:
        """Get temporary storage path."""
        if self.enabled:
            return self.base_path / "tmp"
        return Path("/tmp/ringrift")

    def cleanup(self):
        """Clean up ramdisk."""
        if self.enabled and self.base_path.exists():
            shutil.rmtree(self.base_path, ignore_errors=True)


class DataAggregator:
    """Aggregate selfplay data from Lambda nodes."""

    def __init__(self, lambda_nodes: list[str], ramdisk: RamdiskManager):
        self.lambda_nodes = lambda_nodes
        self.ramdisk = ramdisk
        self.ssh_key = os.path.expanduser("~/.ssh/id_cluster")

    def sync_from_lambda(self, output_dir: Path) -> dict:
        """Sync databases from Lambda nodes."""
        stats = {"nodes": 0, "files": 0, "bytes": 0}

        for node in self.lambda_nodes:
            logger.info(f"Syncing from {node}...")
            node_dir = output_dir / node.replace(".", "_")
            node_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "rsync", "-avz", "--compress",
                "--include=*.db", "--exclude=*",
                "-e", f"ssh -i {self.ssh_key} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                f"ubuntu@{node}:~/ringrift/ai-service/data/selfplay/",
                str(node_dir) + "/"
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    stats["nodes"] += 1
                    # Count synced files
                    db_files = list(node_dir.rglob("*.db"))
                    stats["files"] += len(db_files)
                    stats["bytes"] += sum(f.stat().st_size for f in db_files)
                    logger.info(f"  Synced {len(db_files)} databases from {node}")
                else:
                    logger.warning(f"  Sync failed for {node}: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                logger.warning(f"  Timeout syncing from {node}")
            except Exception as e:
                logger.warning(f"  Error syncing from {node}: {e}")

        return stats


class DataExporter:
    """Export SQLite games to NPZ training files."""

    def __init__(self, ramdisk: RamdiskManager):
        self.ramdisk = ramdisk

    def export_database(self, db_path: Path, board_type: str, num_players: int, output_dir: Path) -> bool:
        """Export a single database to NPZ."""
        output_file = output_dir / f"{board_type}_{num_players}p_{db_path.stem}.npz"

        if output_file.exists():
            logger.debug(f"Skipping existing: {output_file}")
            return True

        cmd = [
            sys.executable, "scripts/export_replay_dataset.py",
            "--db", str(db_path),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--output", str(output_file),
        ]

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            if result.returncode == 0:
                logger.info(f"Exported: {output_file.name}")
                return True
            else:
                logger.warning(f"Export failed for {db_path}: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"Export timeout for {db_path}")
            return False
        except Exception as e:
            logger.warning(f"Export error for {db_path}: {e}")
            return False

    def export_all(self, db_dir: Path, output_dir: Path) -> dict:
        """Export all databases in directory."""
        stats = {"exported": 0, "failed": 0, "skipped": 0}

        for db_file in db_dir.rglob("*.db"):
            # Infer config from filename
            name = db_file.stem.lower()

            board_type = None
            num_players = 2

            if "square8" in name or "sq8" in name:
                board_type = "square8"
            elif "square19" in name or "sq19" in name:
                board_type = "square19"
            elif "hex8" in name:
                board_type = "hex8"
            elif "hexagonal" in name:
                board_type = "hexagonal"

            if "_2p" in name or "2p" in name:
                num_players = 2
            elif "_3p" in name or "3p" in name:
                num_players = 3
            elif "_4p" in name or "4p" in name:
                num_players = 4

            if board_type:
                if self.export_database(db_file, board_type, num_players, output_dir):
                    stats["exported"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["skipped"] += 1
                logger.debug(f"Skipping unknown config: {db_file}")

        return stats


class DataValidator:
    """Validate data quality."""

    def validate_database(self, db_path: Path) -> dict:
        """Validate a database."""
        import sqlite3

        result = {"path": str(db_path), "valid": False, "games": 0, "issues": []}

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check game count
            cursor.execute("SELECT COUNT(*) FROM games")
            result["games"] = cursor.fetchone()[0]

            # Check for required columns
            cursor.execute("PRAGMA table_info(games)")
            columns = {row[1] for row in cursor.fetchall()}
            required = {"game_id", "board_type", "num_players", "winner"}
            missing = required - columns
            if missing:
                result["issues"].append(f"Missing columns: {missing}")

            # Check for empty games
            if result["games"] == 0:
                result["issues"].append("No games in database")

            conn.close()
            result["valid"] = len(result["issues"]) == 0

        except Exception as e:
            result["issues"].append(str(e))

        return result


class PipelineDaemon:
    """Main daemon orchestrating the CPU pipeline."""

    def __init__(self, args):
        self.args = args
        self.running = True
        self.ramdisk = RamdiskManager()
        self.stats = {
            "cycles": 0,
            "last_sync": None,
            "total_exported": 0,
            "total_validated": 0,
        }

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def run(self):
        """Run the daemon loop."""
        logger.info("Starting CPU pipeline daemon")
        logger.info(f"Role: {self.args.role}")
        logger.info(f"Lambda nodes: {self.args.lambda_nodes}")
        logger.info(f"Ramdisk: {'enabled' if self.args.use_ramdisk else 'disabled'}")

        if self.args.use_ramdisk:
            self.ramdisk.setup()

        # Ensure output directories exist
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        while self.running:
            try:
                self._run_cycle()
                self.stats["cycles"] += 1

                # Log status
                if self.stats["cycles"] % 10 == 0:
                    logger.info(f"Pipeline status: cycles={self.stats['cycles']}, "
                              f"exported={self.stats['total_exported']}, "
                              f"validated={self.stats['total_validated']}")

                # Sleep between cycles
                time.sleep(self.args.interval)

            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(60)  # Back off on error

        logger.info("Daemon stopped")
        self.ramdisk.cleanup()

    def _run_cycle(self):
        """Run one cycle of the pipeline."""
        if self.args.role == "aggregate":
            self._run_aggregation()
        elif self.args.role == "export":
            self._run_export()
        elif self.args.role == "validate":
            self._run_validation()
        elif self.args.role == "all":
            self._run_aggregation()
            self._run_export()
            self._run_validation()

    def _run_aggregation(self):
        """Run data aggregation from Lambda nodes."""
        if not self.args.lambda_nodes:
            return

        aggregator = DataAggregator(
            self.args.lambda_nodes.split(","),
            self.ramdisk
        )

        output_dir = Path(self.args.output_dir) / "aggregated"
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = aggregator.sync_from_lambda(output_dir)
        self.stats["last_sync"] = datetime.now().isoformat()
        logger.info(f"Aggregation complete: {stats}")

    def _run_export(self):
        """Run data export."""
        exporter = DataExporter(self.ramdisk)

        db_dir = Path(self.args.output_dir) / "aggregated"
        if not db_dir.exists():
            db_dir = Path("data/selfplay")

        export_dir = Path(self.args.output_dir) / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)

        stats = exporter.export_all(db_dir, export_dir)
        self.stats["total_exported"] += stats["exported"]
        logger.info(f"Export complete: {stats}")

    def _run_validation(self):
        """Run data validation."""
        validator = DataValidator()

        db_dir = Path(self.args.output_dir) / "aggregated"
        if not db_dir.exists():
            db_dir = Path("data/selfplay")

        valid = 0
        invalid = 0

        for db_file in list(db_dir.rglob("*.db"))[:50]:  # Limit per cycle
            result = validator.validate_database(db_file)
            if result["valid"]:
                valid += 1
            else:
                invalid += 1
                if result["issues"]:
                    logger.warning(f"Validation issues in {db_file}: {result['issues']}")

        self.stats["total_validated"] += valid + invalid
        logger.info(f"Validation complete: valid={valid}, invalid={invalid}")

    def health_check(self) -> "HealthCheckResult":
        """Return health check result for daemon protocol.

        Returns:
            HealthCheckResult with status and metrics
        """
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        try:
            details = {
                "running": self.running,
                "cycles_completed": self.stats.get("cycles", 0),
                "total_exported": self.stats.get("total_exported", 0),
                "total_validated": self.stats.get("total_validated", 0),
                "last_sync": self.stats.get("last_sync"),
                "role": self.args.role,
                "ramdisk_enabled": self.ramdisk.enabled if hasattr(self, "ramdisk") else False,
            }

            # Check if running
            if not self.running:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.STOPPED,
                    message="VastCpuPipelineDaemon is not running",
                    details=details,
                )

            # Check if making progress (cycles should increase)
            cycles = self.stats.get("cycles", 0)
            if cycles == 0:
                # Just started, give it grace period
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.RUNNING,
                    message="VastCpuPipelineDaemon starting up",
                    details=details,
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="VastCpuPipelineDaemon healthy",
                details=details,
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )


def main():
    parser = argparse.ArgumentParser(description="Vast.ai CPU Pipeline Daemon")
    parser.add_argument("--role", choices=["aggregate", "export", "validate", "all"],
                       default="all", help="Pipeline role")
    parser.add_argument("--lambda-nodes", type=str, default="",
                       help="Comma-separated Lambda node IPs")
    parser.add_argument("--output-dir", type=str, default="data/pipeline",
                       help="Output directory")
    parser.add_argument("--use-ramdisk", action="store_true",
                       help="Use ramdisk for temporary storage")
    parser.add_argument("--interval", type=int, default=300,
                       help="Seconds between cycles")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit")

    args = parser.parse_args()

    daemon = PipelineDaemon(args)

    if args.once:
        daemon._run_cycle()
    else:
        daemon.run()


if __name__ == "__main__":
    main()
