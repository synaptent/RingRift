#!/usr/bin/env python3
"""External Drive Sync Daemon - Continuous sync of selfplay data to external storage.

This daemon syncs selfplay databases (.db), JSONL files, and NN/NNUE models from all
cluster hosts to the Mac Studio's external drive with deduplication and data quality.

Features:
- Continuous polling (configurable interval, default 5 minutes)
- Tailscale-aware connections (uses Tailscale IPs when available)
- Deduplication via game_id tracking across all sources
- Incremental sync with rsync
- Merges all data into a single deduplicated repository
- Data quality analysis with quarantine for bad/timeout games
- NN and NNUE model collection from cluster
- Detailed analysis reports after each sync cycle
- Integration with unified_ai_loop.py via event bus

Target directory structure on external drive:
  /Volumes/RingRift-Data/selfplay_repository/
    raw/                  # Raw synced files by host
      GH200-e/
      GH200-f/
      Lambda-a/
      ...
    merged/               # Deduplicated merged databases
      selfplay_merged.db  # Main merged database
      jsonl/              # Merged JSONL files
    quarantine/           # Bad/timeout data quarantined here
      malformed/
      timeout/
      unknown_board/
    models/               # Synced NN and NNUE models
      nnue/
      checkpoints/
    reports/              # Analysis reports
    manifest.db           # Sync manifest for deduplication

Usage:
    # Run as daemon
    python scripts/external_drive_sync_daemon.py --start

    # Run in foreground
    python scripts/external_drive_sync_daemon.py --foreground

    # Single sync cycle
    python scripts/external_drive_sync_daemon.py --once

    # Dry run
    python scripts/external_drive_sync_daemon.py --dry-run

    # Custom target
    python scripts/external_drive_sync_daemon.py --target /path/to/drive

    # Skip model sync
    python scripts/external_drive_sync_daemon.py --once --no-models

    # Skip data quality analysis
    python scripts/external_drive_sync_daemon.py --once --no-analysis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import coordination helpers for sync, bandwidth, and orchestration
from app.coordination.helpers import (
    # Sync lock
    has_sync_lock,
    get_sync_lock_context,
    acquire_sync_lock_safe,
    release_sync_lock_safe,
    # Bandwidth
    has_bandwidth_manager,
    get_transfer_priorities,
    request_bandwidth_safe,
    release_bandwidth_safe,
    # Orchestrator
    has_coordination,
    get_orchestrator_roles,
    get_registry_safe,
    acquire_role_safe,
    release_role_safe,
)

HAS_SYNC_LOCK = has_sync_lock()
sync_lock = get_sync_lock_context()

HAS_BANDWIDTH_MANAGER = has_bandwidth_manager()
TransferPriority = get_transfer_priorities()

HAS_ORCHESTRATOR_REGISTRY = has_coordination()
OrchestratorRole = get_orchestrator_roles()
get_registry = get_registry_safe

# Wrapper functions for backwards compatibility
def request_bandwidth(host: str, mbps: float = 100.0, priority=None):
    return request_bandwidth_safe(host, mbps, priority)

def release_bandwidth(host: str) -> None:
    release_bandwidth_safe(host)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Default external drive path on Mac Studio
DEFAULT_EXTERNAL_DRIVE = "/Volumes/RingRift-Data/selfplay_repository"

# Default poll interval (5 minutes)
DEFAULT_POLL_INTERVAL = 300

# Disk capacity limits (consistent with orchestrator MAX_DISK_USAGE_PERCENT)
MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", "70"))


def check_disk_has_capacity(path: Path, threshold: float = None) -> Tuple[bool, float]:
    """Check if disk has capacity for more data.

    Args:
        path: Path to check disk usage for
        threshold: Max disk usage percentage (defaults to MAX_DISK_USAGE_PERCENT)

    Returns:
        Tuple of (has_capacity, current_usage_percent)
    """
    threshold = threshold if threshold is not None else MAX_DISK_USAGE_PERCENT
    try:
        usage = shutil.disk_usage(path)
        percent = (usage.used / usage.total) * 100
        return percent < threshold, percent
    except Exception:
        return True, 0.0  # Allow sync to continue if we can't check


@dataclass
class HostConfig:
    """Configuration for a remote host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    tailscale_ip: Optional[str] = None
    remote_path: str = "~/ringrift/ai-service"
    enabled: bool = True


@dataclass
class SyncManifest:
    """Tracks synced game IDs for deduplication."""
    db_path: Path

    def __post_init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize the manifest database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS synced_games (
                game_id TEXT PRIMARY KEY,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                synced_at REAL NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                merged_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_synced_games_host
            ON synced_games(source_host);

            CREATE INDEX IF NOT EXISTS idx_synced_games_merged
            ON synced_games(merged_at);

            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_name TEXT NOT NULL,
                sync_time REAL NOT NULL,
                files_synced INTEGER NOT NULL,
                games_synced INTEGER NOT NULL,
                duration_seconds REAL,
                success INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS merge_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merge_time REAL NOT NULL,
                source_dbs INTEGER NOT NULL,
                games_before INTEGER NOT NULL,
                games_after INTEGER NOT NULL,
                games_added INTEGER NOT NULL,
                duplicates_skipped INTEGER NOT NULL,
                duration_seconds REAL
            );
        """)
        conn.commit()
        conn.close()

    def get_synced_game_ids(self) -> Set[str]:
        """Get all synced game IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT game_id FROM synced_games")
        game_ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return game_ids

    def mark_games_synced(
        self,
        game_ids: List[str],
        source_host: str,
        source_db: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ):
        """Mark games as synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for game_id in game_ids:
            cursor.execute("""
                INSERT OR IGNORE INTO synced_games
                (game_id, source_host, source_db, synced_at, board_type, num_players)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (game_id, source_host, source_db, now, board_type, num_players))

        conn.commit()
        conn.close()

    def mark_games_merged(self, game_ids: List[str]):
        """Mark games as merged into the main database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for game_id in game_ids:
            cursor.execute(
                "UPDATE synced_games SET merged_at = ? WHERE game_id = ?",
                (now, game_id)
            )

        conn.commit()
        conn.close()

    def record_sync(
        self,
        host_name: str,
        files_synced: int,
        games_synced: int,
        duration: float,
        success: bool
    ):
        """Record a sync operation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sync_history
            (host_name, sync_time, files_synced, games_synced, duration_seconds, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (host_name, time.time(), files_synced, games_synced, duration, int(success)))
        conn.commit()
        conn.close()

    def record_merge(
        self,
        source_dbs: int,
        games_before: int,
        games_after: int,
        games_added: int,
        duplicates_skipped: int,
        duration: float,
    ):
        """Record a merge operation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO merge_history
            (merge_time, source_dbs, games_before, games_after, games_added,
             duplicates_skipped, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (time.time(), source_dbs, games_before, games_after, games_added,
              duplicates_skipped, duration))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM synced_games")
        total_games = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM synced_games WHERE merged_at IS NOT NULL")
        merged_games = cursor.fetchone()[0]

        cursor.execute("""
            SELECT source_host, COUNT(*) as count
            FROM synced_games
            GROUP BY source_host
        """)
        games_by_host = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT COUNT(*), SUM(games_synced)
            FROM sync_history
            WHERE success = 1
        """)
        row = cursor.fetchone()
        successful_syncs = row[0] or 0
        total_synced = row[1] or 0

        conn.close()

        return {
            "total_games": total_games,
            "merged_games": merged_games,
            "games_by_host": games_by_host,
            "successful_syncs": successful_syncs,
            "total_synced": total_synced,
        }


class ExternalDriveSyncDaemon:
    """Daemon for syncing selfplay data to external drive."""

    def __init__(
        self,
        target_dir: Path,
        hosts: List[HostConfig],
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        dry_run: bool = False,
        verbose: bool = False,
        sync_models: bool = True,
        run_analysis: bool = True,
    ):
        self.target_dir = target_dir
        self.hosts = {h.name: h for h in hosts}
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.verbose = verbose
        self.sync_models = sync_models
        self.run_analysis = run_analysis

        # Directories
        self.raw_dir = target_dir / "raw"
        self.merged_dir = target_dir / "merged"
        self.quarantine_dir = target_dir / "quarantine"
        self.models_dir = target_dir / "models"
        self.reports_dir = target_dir / "reports"
        self.manifest_path = target_dir / "manifest.db"

        # Initialize
        self.manifest = SyncManifest(self.manifest_path)
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Stats
        self._last_sync_time: Dict[str, float] = {}
        self._sync_errors: Dict[str, int] = {}
        self._last_analysis_report: Optional[Dict[str, Any]] = None

    def _log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

    async def _test_ssh_connection(self, host: HostConfig) -> Tuple[bool, str]:
        """Test SSH connection to a host. Returns (success, effective_host)."""
        # Try primary host first
        ssh_opts = f"-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"
        if host.ssh_port != 22:
            ssh_opts += f" -p {host.ssh_port}"
        if host.ssh_key:
            ssh_opts += f" -i {host.ssh_key}"

        cmd = f'ssh {ssh_opts} {host.ssh_user}@{host.ssh_host} "echo ok"'

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15)

            if process.returncode == 0:
                return True, host.ssh_host

        except (asyncio.TimeoutError, Exception):
            pass

        # Try Tailscale IP if available
        if host.tailscale_ip:
            ssh_opts = f"-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"
            if host.ssh_key:
                ssh_opts += f" -i {host.ssh_key}"

            cmd = f'ssh {ssh_opts} {host.ssh_user}@{host.tailscale_ip} "echo ok"'

            try:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15)

                if process.returncode == 0:
                    return True, host.tailscale_ip

            except (asyncio.TimeoutError, Exception):
                pass

        return False, ""

    async def _sync_host(self, host: HostConfig) -> Tuple[int, int]:
        """Sync data from a single host. Returns (files_synced, games_synced)."""
        if not host.enabled:
            return 0, 0

        start_time = time.time()

        # Test connection
        connected, effective_host = await self._test_ssh_connection(host)
        if not connected:
            self._log(f"{host.name}: unreachable", "WARN")
            self._sync_errors[host.name] = self._sync_errors.get(host.name, 0) + 1
            self.manifest.record_sync(host.name, 0, 0, time.time() - start_time, False)
            return 0, 0

        if self.verbose:
            self._log(f"{host.name}: connected via {effective_host}")

        # Create host directory
        host_dir = self.raw_dir / host.name
        host_dir.mkdir(parents=True, exist_ok=True)

        # Check disk capacity before syncing
        has_capacity, disk_percent = check_disk_has_capacity(self.raw_dir)
        if not has_capacity:
            self._log(f"{host.name}: SKIPPING - disk usage {disk_percent:.1f}% >= {MAX_DISK_USAGE_PERCENT}%", "WARN")
            return 0, 0

        if self.dry_run:
            self._log(f"{host.name}: [DRY RUN] would sync to {host_dir}")
            return 0, 0

        # Build rsync command
        ssh_opts = "-o ConnectTimeout=15 -o StrictHostKeyChecking=no"
        if host.ssh_key:
            ssh_opts += f" -i {host.ssh_key}"

        # Acquire sync_lock to prevent concurrent rsync to same host
        sync_lock_acquired = False
        if HAS_SYNC_LOCK and sync_lock is not None:
            try:
                from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
                sync_lock_acquired = acquire_sync_lock(host.name, "rsync-inbound", wait=True, wait_timeout=60.0)
                if not sync_lock_acquired:
                    self._log(f"{host.name}: could not acquire sync lock, skipping", "WARN")
                    return 0, 0
                if self.verbose:
                    self._log(f"{host.name}: acquired sync lock")
            except Exception as e:
                self._log(f"{host.name}: sync lock error: {e}", "WARN")
                # Continue without lock

        try:
            # Sync .db files
            db_cmd = f'rsync -avz --progress -e "ssh {ssh_opts}" {host.ssh_user}@{effective_host}:{host.remote_path}/data/games/*.db {host_dir}/ 2>/dev/null'

            files_synced = 0
            try:
                process = await asyncio.create_subprocess_shell(
                    db_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=300)

                if process.returncode == 0:
                    # Count synced files
                    files_synced = len(list(host_dir.glob("*.db")))

            except Exception as e:
                self._log(f"{host.name}: rsync error: {e}", "ERROR")

            # Sync .jsonl files from all data directories (recursive)
            jsonl_dir = host_dir / "jsonl"
            jsonl_dir.mkdir(exist_ok=True)

            # Primary: Recursive sync of ALL .jsonl files from the entire data/ directory
            # Excludes: quarantine, logs, statistics/metrics files, toxic archives
            jsonl_cmd = (
                f'rsync -avz --progress '
                f'--include="*/" '
                f'--include="*.jsonl" '
                f'--exclude="quarantine*" '
                f'--exclude="*quarantine*" '
                f'--exclude="toxic*" '
                f'--exclude="*_stats.jsonl" '
                f'--exclude="*_metrics.jsonl" '
                f'--exclude="*_analysis.jsonl" '
                f'--exclude="statistics*" '
                f'--exclude="eval_pools*" '
                f'--exclude="critical_positions*" '
                f'--exclude="holdouts*" '
                f'--exclude="*" '
                f'-e "ssh {ssh_opts}" '
                f'{host.ssh_user}@{effective_host}:{host.remote_path}/data/ '
                f'{jsonl_dir}/ 2>/dev/null'
            )

            # Request bandwidth allocation for large JSONL transfer
            bandwidth_allocated = False
            if HAS_BANDWIDTH_MANAGER:
                try:
                    # Request 50 Mbps for JSONL sync (lower priority than model updates)
                    bandwidth_allocated = request_bandwidth(
                        host.name, 50, TransferPriority.NORMAL, timeout=30.0
                    )
                    if not bandwidth_allocated:
                        self._log(f"{host.name}: bandwidth unavailable, proceeding anyway", "WARN")
                except Exception as e:
                    self._log(f"{host.name}: bandwidth request error: {e}", "WARN")

            try:
                process = await asyncio.create_subprocess_shell(
                    jsonl_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=900)  # 15 min timeout for full recursive sync
            except Exception as e:
                if self.verbose:
                    self._log(f"{host.name}: JSONL sync error: {e}", "WARN")
            finally:
                # Release bandwidth allocation
                if bandwidth_allocated and HAS_BANDWIDTH_MANAGER:
                    try:
                        release_bandwidth(host.name)
                    except Exception as e:
                        self._log(f"{host.name}: bandwidth release error: {e}", "WARN")

        finally:
            # Release sync_lock
            if sync_lock_acquired and HAS_SYNC_LOCK:
                try:
                    from app.coordination.sync_mutex import release_sync_lock
                    release_sync_lock(host.name)
                    if self.verbose:
                        self._log(f"{host.name}: released sync lock")
                except Exception as e:
                    self._log(f"{host.name}: sync lock release error: {e}", "WARN")

        # Count all synced JSONL files recursively
        jsonl_count = len(list(jsonl_dir.glob("**/*.jsonl")))
        files_synced += jsonl_count

        if self.verbose and jsonl_count > 0:
            self._log(f"{host.name}: synced {jsonl_count} JSONL files")

        # Count games in synced DBs
        games_synced = 0
        for db_file in host_dir.glob("*.db"):
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM games")
                games_synced += cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass

        # Count games in synced JSONL files (one game per line)
        for jsonl_file in jsonl_dir.glob("**/*.jsonl"):
            try:
                with open(jsonl_file, 'r') as f:
                    games_synced += sum(1 for line in f if line.strip())
            except Exception:
                pass

        duration = time.time() - start_time
        self.manifest.record_sync(host.name, files_synced, games_synced, duration, True)
        self._sync_errors[host.name] = 0
        self._last_sync_time[host.name] = time.time()

        if files_synced > 0:
            self._log(f"{host.name}: synced {files_synced} files, {games_synced} games")

        return files_synced, games_synced

    async def _merge_databases(self) -> Dict[str, int]:
        """Merge all synced databases with deduplication."""
        start_time = time.time()

        # Find all source databases
        source_dbs = list(self.raw_dir.glob("**/*.db"))
        if not source_dbs:
            return {"merged": 0, "duplicates": 0}

        # Get existing game IDs for deduplication
        existing_ids = self.manifest.get_synced_game_ids()

        # Output database
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        merged_db_path = self.merged_dir / "selfplay_merged.db"

        # Count games before merge
        games_before = 0
        if merged_db_path.exists():
            try:
                conn = sqlite3.connect(merged_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM games")
                games_before = cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass

        if self.dry_run:
            self._log(f"[DRY RUN] would merge {len(source_dbs)} databases")
            return {"merged": 0, "duplicates": 0}

        # Build merge command using merge_game_dbs.py
        merge_script = AI_SERVICE_ROOT / "scripts" / "merge_game_dbs.py"

        cmd = [
            sys.executable,
            str(merge_script),
            "--output", str(merged_db_path),
            "--dedupe-by-game-id",
            "--compress-states",
        ]

        for db in source_dbs:
            cmd.extend(["--db", str(db)])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)

            if process.returncode != 0:
                self._log(f"Merge failed: {stderr.decode()}", "ERROR")
                return {"merged": 0, "duplicates": 0}

            # Parse output for stats
            output = stdout.decode()
            games_added = 0
            duplicates = 0

            for line in output.split("\n"):
                if "Games merged:" in line:
                    games_added = int(line.split(":")[1].strip())
                elif "Conflicts skipped:" in line or "Duplicates skipped:" in line:
                    duplicates = int(line.split(":")[1].strip())

            # Count games after
            games_after = games_before
            try:
                conn = sqlite3.connect(merged_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM games")
                games_after = cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass

            duration = time.time() - start_time
            self.manifest.record_merge(
                len(source_dbs), games_before, games_after, games_added, duplicates, duration
            )

            if games_added > 0:
                self._log(f"Merged {games_added} new games ({duplicates} duplicates skipped)")

            return {"merged": games_added, "duplicates": duplicates}

        except asyncio.TimeoutError:
            self._log("Merge timed out", "ERROR")
            return {"merged": 0, "duplicates": 0}
        except Exception as e:
            self._log(f"Merge error: {e}", "ERROR")
            return {"merged": 0, "duplicates": 0}

    async def _run_data_quality_analysis(self) -> Dict[str, Any]:
        """Run data quality analysis with quarantine and normalization."""
        if not self.run_analysis:
            return {}

        start_time = time.time()
        self._log("Running data quality analysis...")

        # Create directories
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            self._log("[DRY RUN] would run data quality analysis")
            return {}

        # Run analyze_game_statistics.py with quarantine and fix-in-place
        analysis_script = AI_SERVICE_ROOT / "scripts" / "analyze_game_statistics.py"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"analysis_{timestamp}.json"

        cmd = [
            sys.executable,
            str(analysis_script),
            "--jsonl-dir", str(self.raw_dir),
            "--recursive",
            "--quarantine-dir", str(self.quarantine_dir),
            "--fix-in-place",
            "--output", str(report_path),
            "--format", "both",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            output = stdout.decode()
            duration = time.time() - start_time

            # Parse results
            result = {
                "duration_seconds": duration,
                "report_path": str(report_path),
                "success": process.returncode == 0,
            }

            # Load report if exists
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        report_data = json.load(f)
                    result["total_games"] = report_data.get("summary", {}).get("total_games", 0)
                    result["quarantined"] = report_data.get("summary", {}).get("quarantined", 0)
                    result["fixed"] = report_data.get("summary", {}).get("fixed", 0)

                    # Store for printing
                    self._last_analysis_report = report_data
                except Exception:
                    pass

            self._log(f"Analysis complete in {duration:.1f}s")
            return result

        except asyncio.TimeoutError:
            self._log("Analysis timed out", "ERROR")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            self._log(f"Analysis error: {e}", "ERROR")
            return {"success": False, "error": str(e)}

    async def _sync_models(self) -> Dict[str, Any]:
        """Sync NN and NNUE models from cluster to external drive."""
        if not self.sync_models:
            return {}

        start_time = time.time()
        self._log("Syncing NN/NNUE models...")

        # Create model directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        nnue_dir = self.models_dir / "nnue"
        nnue_dir.mkdir(exist_ok=True)
        checkpoints_dir = self.models_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        if self.dry_run:
            self._log("[DRY RUN] would sync models")
            return {}

        # Use sync_models.py to collect from cluster
        sync_script = AI_SERVICE_ROOT / "scripts" / "sync_models.py"

        # First collect to local, then copy to external drive
        cmd = [
            sys.executable,
            str(sync_script),
            "--collect",
            "--config", str(AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            # Now copy collected models to external drive
            local_models = AI_SERVICE_ROOT / "models"
            local_nnue = local_models / "nnue"

            models_copied = 0

            # Copy NNUE models
            if local_nnue.exists():
                for model_file in local_nnue.glob("*.nnue"):
                    dest = nnue_dir / model_file.name
                    if not dest.exists() or model_file.stat().st_mtime > dest.stat().st_mtime:
                        shutil.copy2(model_file, dest)
                        models_copied += 1

            # Copy checkpoints (.pt, .pth files)
            for checkpoint in local_models.glob("**/*.pt"):
                dest = checkpoints_dir / checkpoint.name
                if not dest.exists() or checkpoint.stat().st_mtime > dest.stat().st_mtime:
                    shutil.copy2(checkpoint, dest)
                    models_copied += 1

            for checkpoint in local_models.glob("**/*.pth"):
                dest = checkpoints_dir / checkpoint.name
                if not dest.exists() or checkpoint.stat().st_mtime > dest.stat().st_mtime:
                    shutil.copy2(checkpoint, dest)
                    models_copied += 1

            duration = time.time() - start_time

            result = {
                "models_copied": models_copied,
                "duration_seconds": duration,
                "success": process.returncode == 0,
            }

            if models_copied > 0:
                self._log(f"Copied {models_copied} models to external drive")

            return result

        except asyncio.TimeoutError:
            self._log("Model sync timed out", "ERROR")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            self._log(f"Model sync error: {e}", "ERROR")
            return {"success": False, "error": str(e)}

    def _print_analysis_report(self):
        """Print formatted analysis report."""
        if not self._last_analysis_report:
            return

        report = self._last_analysis_report
        summary = report.get("summary", {})

        print("\n" + "=" * 60)
        print("  DATA QUALITY ANALYSIS REPORT")
        print("=" * 60)

        print(f"\n  Total Games Analyzed: {summary.get('total_games', 0):,}")
        print(f"  Valid Games: {summary.get('valid_games', 0):,}")
        print(f"  Quarantined: {summary.get('quarantined', 0):,}")
        print(f"  Fixed/Normalized: {summary.get('fixed', 0):,}")

        # Victory distribution
        if "victory_distribution" in report:
            print("\n  Victory Type Distribution:")
            for vtype, count in report["victory_distribution"].items():
                pct = (count / max(summary.get('total_games', 1), 1)) * 100
                print(f"    {vtype}: {count:,} ({pct:.1f}%)")

        # Board type distribution
        if "board_distribution" in report:
            print("\n  Board Type Distribution:")
            for btype, count in report["board_distribution"].items():
                pct = (count / max(summary.get('total_games', 1), 1)) * 100
                print(f"    {btype}: {count:,} ({pct:.1f}%)")

        # Data quality issues
        if "issues" in report and report["issues"]:
            print("\n  Data Quality Issues:")
            for issue in report["issues"][:5]:  # Top 5 issues
                print(f"    - {issue}")

        print("\n" + "=" * 60 + "\n")

    async def run_sync_cycle(self) -> Dict[str, Any]:
        """Run a complete sync cycle."""
        self._log("Starting sync cycle...")
        cycle_start = time.time()

        # Sync from all hosts in parallel
        tasks = [self._sync_host(host) for host in self.hosts.values() if host.enabled]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_files = 0
        total_games = 0
        hosts_synced = 0

        for result in results:
            if isinstance(result, tuple):
                files, games = result
                total_files += files
                total_games += games
                if files > 0:
                    hosts_synced += 1

        # Merge databases
        merge_result = await self._merge_databases()

        # Run data quality analysis
        analysis_result = await self._run_data_quality_analysis()

        # Sync models
        model_result = await self._sync_models()

        cycle_duration = time.time() - cycle_start

        summary = {
            "hosts_synced": hosts_synced,
            "files_synced": total_files,
            "games_synced": total_games,
            "games_merged": merge_result.get("merged", 0),
            "duplicates_skipped": merge_result.get("duplicates", 0),
            "analysis": analysis_result,
            "models": model_result,
            "duration_seconds": cycle_duration,
        }

        self._log(
            f"Cycle complete: {hosts_synced} hosts, {total_files} files, "
            f"{merge_result.get('merged', 0)} new games in {cycle_duration:.1f}s"
        )

        # Print analysis report
        self._print_analysis_report()

        return summary

    async def run(self):
        """Main daemon loop."""
        self._running = True

        self._log(f"Starting daemon with {len(self.hosts)} hosts")
        self._log(f"Target: {self.target_dir}")
        self._log(f"Poll interval: {self.poll_interval}s")

        if self.dry_run:
            self._log("DRY RUN MODE - no actual operations")

        # Acquire DATA_SYNC role to prevent multiple daemons
        role_acquired = False
        if HAS_ORCHESTRATOR_REGISTRY:
            try:
                registry = get_registry()
                role_acquired = registry.acquire_role(OrchestratorRole.DATA_SYNC)
                if role_acquired:
                    self._log("Acquired DATA_SYNC orchestrator role")
                else:
                    self._log("Could not acquire DATA_SYNC role - another daemon is running", "WARN")
                    self._log("Running in secondary mode (read-only monitoring)")
            except Exception as e:
                self._log(f"OrchestratorRegistry error: {e}", "WARN")

        try:
            while self._running:
                try:
                    # Only run sync cycles if we hold the role (or registry unavailable)
                    if role_acquired or not HAS_ORCHESTRATOR_REGISTRY:
                        await self.run_sync_cycle()
                    else:
                        # Secondary mode: just log status
                        if self.verbose:
                            self._log("Secondary mode: skipping sync cycle")

                    # Send heartbeat to keep role
                    if role_acquired and HAS_ORCHESTRATOR_REGISTRY:
                        try:
                            registry = get_registry()
                            registry.heartbeat(OrchestratorRole.DATA_SYNC)
                        except Exception:
                            pass

                except Exception as e:
                    self._log(f"Cycle error: {e}", "ERROR")

                # Wait for next cycle
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.poll_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Continue loop
        finally:
            # Release role on shutdown
            if role_acquired and HAS_ORCHESTRATOR_REGISTRY:
                try:
                    registry = get_registry()
                    registry.release_role(OrchestratorRole.DATA_SYNC)
                    self._log("Released DATA_SYNC orchestrator role")
                except Exception as e:
                    self._log(f"Failed to release role: {e}", "WARN")

        self._log("Daemon stopped")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()

    async def distribute_to_high_capacity_hosts(
        self,
        min_storage_gb: int = 2000,
        target_hosts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Distribute collected training data to high-capacity hosts.

        Args:
            min_storage_gb: Minimum storage threshold (default 2TB)
            target_hosts: Specific hosts to distribute to, or None for auto-selection
        """
        self._log("Distributing data to high-capacity training hosts...")

        # High-capacity hosts (Lambda GPU instances have large NVMe drives)
        HIGH_CAPACITY_HOSTS = [
            "lambda-h100",
            "lambda-2xh100",
            "lambda-a10",
            "lambda-gh200-a",
            "lambda-gh200-b",
            "lambda-gh200-c",
            "lambda-gh200-d",
            "lambda-gh200-e",
            "lambda-gh200-f",
            "lambda-gh200-g",
            "lambda-gh200-h",
        ]

        if target_hosts:
            hosts_to_sync = [h for h in self.hosts.values() if h.name in target_hosts]
        else:
            hosts_to_sync = [h for h in self.hosts.values() if h.name in HIGH_CAPACITY_HOSTS]

        if not hosts_to_sync:
            self._log("No high-capacity hosts available for distribution", "WARN")
            return {"hosts_synced": 0}

        results = {}

        for host in hosts_to_sync:
            # Test connection
            connected, effective_host = await self._test_ssh_connection(host)
            if not connected:
                self._log(f"{host.name}: unreachable for distribution", "WARN")
                results[host.name] = {"success": False, "error": "unreachable"}
                continue

            self._log(f"Distributing to {host.name}...")

            if self.dry_run:
                self._log(f"  [DRY RUN] would sync to {host.name}")
                results[host.name] = {"success": True, "dry_run": True}
                continue

            # Build SSH options
            ssh_opts = "-o ConnectTimeout=15 -o StrictHostKeyChecking=no"
            if host.ssh_key:
                ssh_opts += f" -i {host.ssh_key}"

            # Acquire sync_lock to prevent concurrent rsync to same host
            sync_lock_acquired = False
            if HAS_SYNC_LOCK and sync_lock is not None:
                try:
                    from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
                    sync_lock_acquired = acquire_sync_lock(host.name, "rsync-outbound", wait=True, wait_timeout=60.0)
                    if not sync_lock_acquired:
                        self._log(f"  {host.name}: could not acquire sync lock, skipping", "WARN")
                        results[host.name] = {"success": False, "error": "sync_lock_unavailable"}
                        continue
                except Exception as e:
                    self._log(f"  {host.name}: sync lock error: {e}", "WARN")
                    # Continue without lock

            try:
                # Sync merged database to host's training_pool directory
                merged_db = self.merged_dir / "selfplay_merged.db"
                if merged_db.exists():
                    db_cmd = (
                        f'rsync -avz --progress -e "ssh {ssh_opts}" '
                        f'{merged_db} '
                        f'{host.ssh_user}@{effective_host}:{host.remote_path}/data/games/training_pool/ '
                        f'2>/dev/null'
                    )

                    try:
                        process = await asyncio.create_subprocess_shell(
                            db_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await asyncio.wait_for(process.communicate(), timeout=600)

                        if process.returncode == 0:
                            self._log(f"  {host.name}: synced merged DB")

                    except Exception as e:
                        self._log(f"  {host.name}: DB sync error: {e}", "WARN")

                # Sync JSONL files to host's training_pool/jsonl directory
                jsonl_source = self.raw_dir
                jsonl_cmd = (
                    f'rsync -avz --progress '
                    f'--include="*/" --include="*.jsonl" --exclude="*" '
                    f'-e "ssh {ssh_opts}" '
                    f'{jsonl_source}/ '
                    f'{host.ssh_user}@{effective_host}:{host.remote_path}/data/games/training_pool/synced_jsonl/ '
                    f'2>/dev/null'
                )

                try:
                    process = await asyncio.create_subprocess_shell(
                        jsonl_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(process.communicate(), timeout=1800)  # 30 min for large JSONL sync

                    if process.returncode == 0:
                        self._log(f"  {host.name}: synced JSONL files")
                        results[host.name] = {"success": True}
                    else:
                        results[host.name] = {"success": False, "error": "rsync failed"}

                except asyncio.TimeoutError:
                    self._log(f"  {host.name}: JSONL sync timeout", "WARN")
                    results[host.name] = {"success": False, "error": "timeout"}
                except Exception as e:
                    self._log(f"  {host.name}: JSONL sync error: {e}", "WARN")
                    results[host.name] = {"success": False, "error": str(e)}

            finally:
                # Release sync_lock
                if sync_lock_acquired and HAS_SYNC_LOCK:
                    try:
                        from app.coordination.sync_mutex import release_sync_lock
                        release_sync_lock(host.name)
                    except Exception as e:
                        self._log(f"  {host.name}: sync lock release error: {e}", "WARN")

        successful = sum(1 for r in results.values() if r.get("success"))
        self._log(f"Distribution complete: {successful}/{len(results)} hosts synced")

        return {
            "hosts_synced": successful,
            "total_hosts": len(results),
            "results": results,
        }


def load_hosts_from_yaml(config_path: Path) -> List[HostConfig]:
    """Load host configurations from YAML."""
    if not config_path.exists():
        return []

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    hosts = []

    for name, host_data in data.get("hosts", {}).items():
        status = host_data.get("status", "ready")
        if status not in ("ready", "setup"):
            continue

        hosts.append(HostConfig(
            name=name,
            ssh_host=host_data.get("ssh_host", ""),
            ssh_user=host_data.get("ssh_user", "ubuntu"),
            ssh_port=host_data.get("ssh_port", 22),
            ssh_key=host_data.get("ssh_key"),
            tailscale_ip=host_data.get("tailscale_ip"),
            remote_path=host_data.get("ringrift_path", "~/ringrift/ai-service"),
            enabled=host_data.get("enabled", True),
        ))

    return hosts


def main():
    parser = argparse.ArgumentParser(description="External Drive Sync Daemon")
    parser.add_argument("--start", action="store_true", help="Start daemon in background")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    parser.add_argument("--once", action="store_true", help="Run single sync cycle")
    parser.add_argument("--stop", action="store_true", help="Stop daemon")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_EXTERNAL_DRIVE,
        help=f"Target directory (default: {DEFAULT_EXTERNAL_DRIVE})"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/distributed_hosts.yaml",
        help="Hosts config file"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-models", action="store_true", help="Skip model sync")
    parser.add_argument("--no-analysis", action="store_true", help="Skip data quality analysis")
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute collected data to high-capacity training hosts (>2TB storage)"
    )
    parser.add_argument(
        "--distribute-to",
        type=str,
        nargs="+",
        help="Specific hosts to distribute to (e.g., lambda-h100 lambda-2xh100)"
    )

    args = parser.parse_args()

    target_dir = Path(args.target)
    config_path = AI_SERVICE_ROOT / args.config

    # Status command
    if args.status:
        manifest_path = target_dir / "manifest.db"
        if manifest_path.exists():
            manifest = SyncManifest(manifest_path)
            stats = manifest.get_stats()
            print("External Drive Sync Status:")
            print(f"  Total games synced: {stats['total_games']}")
            print(f"  Merged games: {stats['merged_games']}")
            print(f"  Successful syncs: {stats['successful_syncs']}")
            print("\n  Games by host:")
            for host, count in stats['games_by_host'].items():
                print(f"    {host}: {count}")
        else:
            print("No sync manifest found")
        return

    # Stop command
    if args.stop:
        pid_path = target_dir / "daemon.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                print(f"Process {pid} not found")
            pid_path.unlink()
        else:
            print("No PID file found")
        return

    # Load hosts
    hosts = load_hosts_from_yaml(config_path)
    if not hosts:
        print(f"No hosts found in {config_path}")
        print("Create config/distributed_hosts.yaml with host definitions")
        return

    print(f"Loaded {len(hosts)} hosts from {config_path}")

    # Verify target is accessible
    if not target_dir.parent.exists():
        print(f"Error: Target parent directory does not exist: {target_dir.parent}")
        print("Ensure external drive is mounted")
        return

    # Create daemon
    daemon = ExternalDriveSyncDaemon(
        target_dir=target_dir,
        hosts=hosts,
        poll_interval=args.interval,
        dry_run=args.dry_run,
        verbose=args.verbose,
        sync_models=not args.no_models,
        run_analysis=not args.no_analysis,
    )

    # Handle signals
    def signal_handler(sig, frame):
        print("\nShutdown requested")
        daemon.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.distribute or args.distribute_to:
        # Distribution mode - push data to high-capacity hosts
        target_hosts = args.distribute_to if args.distribute_to else None
        asyncio.run(daemon.distribute_to_high_capacity_hosts(target_hosts=target_hosts))
    elif args.once:
        asyncio.run(daemon.run_sync_cycle())
    elif args.start or args.foreground:
        target_dir.mkdir(parents=True, exist_ok=True)

        if args.start and not args.foreground:
            # Daemonize
            pid = os.fork()
            if pid > 0:
                # Parent
                pid_path = target_dir / "daemon.pid"
                pid_path.write_text(str(pid))
                print(f"Started daemon with PID {pid}")
                return

        asyncio.run(daemon.run())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
