#!/usr/bin/env python3
"""Elo rating reconciliation for distributed P2P nodes.

This module handles distributed consistency for Elo ratings across the cluster.
It complements `app.training.elo_service` (the canonical Elo service interface)
by providing reconciliation utilities for multi-node deployments.

In a distributed P2P setup, multiple nodes may:
1. Record match results to their local DB copies
2. Have network partitions causing temporary inconsistencies
3. Need to sync their Elo states with the central DB

This module provides reconciliation utilities to:
- Detect Elo drift between local and central DBs
- Sync match histories from remote nodes
- Merge conflicting match records
- Alert on significant Elo discrepancies

Note: For core Elo operations (rating queries, match recording), use
`app.training.elo_service`. This module is specifically for distributed
consistency concerns.

Usage:
    from app.training.elo_reconciliation import (
        EloReconciler,
        sync_elo_from_remote,
        check_elo_drift,
    )

    reconciler = EloReconciler()

    # Check if local DB is in sync with central
    drift = reconciler.check_drift()
    if drift.max_rating_diff > 50:
        print(f"WARNING: Elo drift detected: {drift}")

    # Sync matches from a remote node
    added, skipped = reconciler.sync_from_remote("192.168.1.100")
    print(f"Synced {added} new matches, {skipped} duplicates")

    # Full reconciliation across all known nodes
    report = reconciler.reconcile_all()
    print(report.summary())
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConflictResolution(Enum):
    """Strategy for resolving conflicting match records.

    When the same match_id exists with different data (e.g., different winner),
    this determines how to handle the conflict.
    """
    SKIP = "skip"  # Keep existing, count as conflict
    LAST_WRITE_WINS = "last_write_wins"  # Accept more recent timestamp
    FIRST_WRITE_WINS = "first_write_wins"  # Keep existing record
    RAISE = "raise"  # Raise an error

# Path setup
from app.utils.paths import AI_SERVICE_ROOT


@dataclass
class EloDrift:
    """Represents Elo drift between two databases."""
    source: str
    target: str
    checked_at: str
    participants_in_source: int
    participants_in_target: int
    participants_in_both: int
    rating_diffs: Dict[str, float] = field(default_factory=dict)
    board_type: Optional[str] = None
    num_players: Optional[int] = None

    @property
    def max_rating_diff(self) -> float:
        """Maximum absolute rating difference."""
        if not self.rating_diffs:
            return 0.0
        return max(abs(d) for d in self.rating_diffs.values())

    @property
    def avg_rating_diff(self) -> float:
        """Average absolute rating difference."""
        if not self.rating_diffs:
            return 0.0
        return sum(abs(d) for d in self.rating_diffs.values()) / len(self.rating_diffs)

    @property
    def is_significant(self) -> bool:
        """Whether drift is significant enough to warrant action."""
        return self.max_rating_diff > 50 or self.avg_rating_diff > 25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "checked_at": self.checked_at,
            "participants_in_source": self.participants_in_source,
            "participants_in_target": self.participants_in_target,
            "participants_in_both": self.participants_in_both,
            "max_rating_diff": self.max_rating_diff,
            "avg_rating_diff": self.avg_rating_diff,
            "is_significant": self.is_significant,
            "rating_diffs": self.rating_diffs,
            "board_type": self.board_type,
            "num_players": self.num_players,
        }


@dataclass
class DriftHistory:
    """Historical drift tracking for trend analysis."""
    config_key: str  # e.g., "square8_2p"
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    max_snapshots: int = 100  # Keep last 100 snapshots

    def add_snapshot(self, drift: EloDrift) -> None:
        """Add a drift snapshot to history."""
        snapshot = {
            "checked_at": drift.checked_at,
            "max_rating_diff": drift.max_rating_diff,
            "avg_rating_diff": drift.avg_rating_diff,
            "is_significant": drift.is_significant,
            "participants": drift.participants_in_both,
        }
        self.snapshots.append(snapshot)
        # Keep only the most recent snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

    @property
    def trend(self) -> str:
        """Analyze drift trend over recent snapshots.

        Returns: 'improving', 'stable', 'worsening', or 'unknown'
        """
        if len(self.snapshots) < 3:
            return "unknown"

        recent = self.snapshots[-5:]  # Last 5 snapshots
        diffs = [s["max_rating_diff"] for s in recent]

        # Calculate trend
        if len(diffs) >= 2:
            first_half_avg = sum(diffs[:len(diffs)//2]) / (len(diffs)//2)
            second_half_avg = sum(diffs[len(diffs)//2:]) / (len(diffs) - len(diffs)//2)

            if second_half_avg < first_half_avg * 0.8:
                return "improving"
            elif second_half_avg > first_half_avg * 1.2:
                return "worsening"

        return "stable"

    @property
    def persistent_drift(self) -> bool:
        """Check if drift has been significant for multiple consecutive checks."""
        if len(self.snapshots) < 3:
            return False
        return all(s["is_significant"] for s in self.snapshots[-3:])

    @property
    def avg_drift_last_hour(self) -> float:
        """Average max drift over approximately the last hour (assumes 30-min intervals)."""
        recent = self.snapshots[-2:] if len(self.snapshots) >= 2 else self.snapshots
        if not recent:
            return 0.0
        return sum(s["max_rating_diff"] for s in recent) / len(recent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_key": self.config_key,
            "trend": self.trend,
            "persistent_drift": self.persistent_drift,
            "avg_drift_last_hour": self.avg_drift_last_hour,
            "snapshot_count": len(self.snapshots),
            "recent_snapshots": self.snapshots[-5:] if self.snapshots else [],
        }


@dataclass
class SyncResult:
    """Result of syncing from a remote node."""
    remote_host: str
    synced_at: str
    matches_added: int
    matches_skipped: int  # Duplicates
    matches_conflict: int  # Same ID but different data (unresolved)
    matches_resolved: int = 0  # Conflicts resolved via conflict resolution strategy
    participants_added: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "remote_host": self.remote_host,
            "synced_at": self.synced_at,
            "matches_added": self.matches_added,
            "matches_skipped": self.matches_skipped,
            "matches_conflict": self.matches_conflict,
            "matches_resolved": self.matches_resolved,
            "participants_added": self.participants_added,
            "error": self.error,
        }


@dataclass
class ReconciliationReport:
    """Full reconciliation report across all nodes."""
    started_at: str
    completed_at: str
    nodes_synced: List[str]
    nodes_failed: List[str]
    total_matches_added: int
    total_matches_skipped: int
    total_conflicts: int
    total_resolved: int
    drift_detected: bool
    max_drift: float
    sync_results: List[SyncResult] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Elo Reconciliation Report ===",
            f"Started: {self.started_at}",
            f"Completed: {self.completed_at}",
            f"Nodes synced: {len(self.nodes_synced)}",
            f"Nodes failed: {len(self.nodes_failed)}",
            f"Matches added: {self.total_matches_added}",
            f"Matches skipped: {self.total_matches_skipped}",
            f"Conflicts: {self.total_conflicts}",
            f"Resolved: {self.total_resolved}",
            f"Max drift: {self.max_drift:.1f}",
        ]
        if self.drift_detected:
            lines.append("WARNING: SIGNIFICANT DRIFT DETECTED")
        if self.nodes_failed:
            lines.append(f"Failed nodes: {', '.join(self.nodes_failed)}")
        return "\n".join(lines)


class EloReconciler:
    """Reconciles Elo ratings across distributed P2P nodes.

    This class manages synchronization of Elo data between:
    - Local unified_elo.db
    - Remote node databases
    - Central (if different from local)

    Supports configurable conflict resolution strategies:
    - SKIP: Keep existing record, count as conflict (default)
    - LAST_WRITE_WINS: Accept match with more recent timestamp
    - FIRST_WRITE_WINS: Keep existing record, skip incoming
    - RAISE: Raise an exception on conflict

    Also tracks drift history for trend analysis and persistent drift detection.
    """

    def __init__(
        self,
        local_db_path: Optional[Path] = None,
        remote_hosts_config: Optional[Path] = None,
        ssh_timeout: int = 30,
        conflict_resolution: ConflictResolution = ConflictResolution.SKIP,
        track_history: bool = True,
        persist_history: bool = True,
    ):
        self.local_db_path = local_db_path or (AI_SERVICE_ROOT / "data" / "unified_elo.db")
        self.remote_hosts_config = remote_hosts_config or (AI_SERVICE_ROOT / "config" / "remote_hosts.yaml")
        self.ssh_timeout = ssh_timeout
        self.conflict_resolution = conflict_resolution
        self.track_history = track_history
        self.persist_history = persist_history
        self._drift_history: Dict[str, DriftHistory] = {}  # config_key -> DriftHistory

        # Load persisted history if available
        if self.track_history and self.persist_history:
            self.load_drift_history()

    @property
    def _drift_history_path(self) -> Path:
        """Path to persisted drift history file."""
        return self.local_db_path.parent / "elo_drift_history.json"

    def save_drift_history(self) -> None:
        """Persist drift history to disk."""
        if not self.persist_history or not self._drift_history:
            return

        try:
            data = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "histories": {
                    key: {
                        "config_key": hist.config_key,
                        "max_snapshots": hist.max_snapshots,
                        "snapshots": hist.snapshots,
                    }
                    for key, hist in self._drift_history.items()
                }
            }
            self._drift_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._drift_history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Non-fatal - history is a nice-to-have
            print(f"[EloReconciler] Warning: Failed to save drift history: {e}")

    def load_drift_history(self) -> None:
        """Load persisted drift history from disk."""
        if not self._drift_history_path.exists():
            return

        try:
            with open(self._drift_history_path) as f:
                data = json.load(f)

            histories = data.get("histories", {})
            for key, hist_data in histories.items():
                self._drift_history[key] = DriftHistory(
                    config_key=hist_data.get("config_key", key),
                    snapshots=hist_data.get("snapshots", []),
                    max_snapshots=hist_data.get("max_snapshots", 100),
                )
        except Exception as e:
            # Non-fatal - start fresh if history is corrupted
            print(f"[EloReconciler] Warning: Failed to load drift history: {e}")

    def get_drift_history(self, config_key: str) -> Optional[DriftHistory]:
        """Get drift history for a specific configuration."""
        return self._drift_history.get(config_key)

    def get_all_drift_histories(self) -> Dict[str, DriftHistory]:
        """Get all drift histories."""
        return self._drift_history.copy()

    def _record_drift(self, drift: EloDrift, board_type: Optional[str], num_players: Optional[int]) -> None:
        """Record drift in history."""
        if not self.track_history:
            return

        config_key = f"{board_type or 'all'}_{num_players or 'all'}"
        if config_key not in self._drift_history:
            self._drift_history[config_key] = DriftHistory(config_key=config_key)

        self._drift_history[config_key].add_snapshot(drift)

        # Persist to disk
        self.save_drift_history()

    def check_drift(
        self,
        remote_db_path: Optional[Path] = None,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> EloDrift:
        """Check Elo drift between local and another database.

        Args:
            remote_db_path: Path to compare against (local copy of remote DB)
            board_type: Filter to specific board type
            num_players: Filter to specific player count

        Returns:
            EloDrift object with comparison details
        """
        if not self.local_db_path.exists():
            drift = EloDrift(
                source=str(self.local_db_path),
                target=str(remote_db_path) if remote_db_path else "N/A",
                checked_at=datetime.now(timezone.utc).isoformat(),
                participants_in_source=0,
                participants_in_target=0,
                participants_in_both=0,
                board_type=board_type,
                num_players=num_players,
            )
            self._record_drift(drift, board_type, num_players)
            return drift

        if remote_db_path and not remote_db_path.exists():
            drift = EloDrift(
                source=str(self.local_db_path),
                target=str(remote_db_path),
                checked_at=datetime.now(timezone.utc).isoformat(),
                participants_in_source=self._count_participants(self.local_db_path, board_type, num_players),
                participants_in_target=0,
                participants_in_both=0,
                board_type=board_type,
                num_players=num_players,
            )
            self._record_drift(drift, board_type, num_players)
            return drift

        # Get ratings from local DB
        local_ratings = self._get_ratings(self.local_db_path, board_type, num_players)

        # Get ratings from remote DB (or use local if not specified)
        target_path = remote_db_path or self.local_db_path
        remote_ratings = self._get_ratings(target_path, board_type, num_players) if remote_db_path else {}

        # Calculate diffs
        local_set = set(local_ratings.keys())
        remote_set = set(remote_ratings.keys())
        common = local_set & remote_set

        rating_diffs = {}
        for participant in common:
            diff = local_ratings[participant] - remote_ratings[participant]
            if abs(diff) > 1:  # Only track meaningful diffs
                rating_diffs[participant] = diff

        drift = EloDrift(
            source=str(self.local_db_path),
            target=str(target_path),
            checked_at=datetime.now(timezone.utc).isoformat(),
            participants_in_source=len(local_ratings),
            participants_in_target=len(remote_ratings),
            participants_in_both=len(common),
            rating_diffs=rating_diffs,
            board_type=board_type,
            num_players=num_players,
        )

        # Record drift history
        self._record_drift(drift, board_type, num_players)

        # Emit drift metrics
        self._emit_drift_metrics(drift, board_type, num_players)

        return drift

    def _emit_drift_metrics(
        self,
        drift: EloDrift,
        board_type: Optional[str],
        num_players: Optional[int],
    ) -> None:
        """Emit Prometheus metrics for drift detection."""
        try:
            from app.metrics import record_elo_drift
            record_elo_drift(
                board_type=board_type or "all",
                num_players=num_players or 0,
                max_drift=drift.max_rating_diff,
                avg_drift=drift.avg_rating_diff,
                is_significant=drift.is_significant,
            )
        except ImportError:
            pass

    def sync_from_remote(
        self,
        remote_host: str,
        remote_db_path: str = "~/ringrift/ai-service/data/unified_elo.db",
        ssh_user: str = "ubuntu",
    ) -> SyncResult:
        """Sync match history from a remote node.

        Pulls matches from remote that don't exist locally and adds them.

        Args:
            remote_host: SSH host (IP or hostname)
            remote_db_path: Path to Elo DB on remote
            ssh_user: SSH username

        Returns:
            SyncResult with sync statistics
        """
        synced_at = datetime.now(timezone.utc).isoformat()

        # Create temp file for remote DB export
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_export = f.name

        try:
            # Export matches from remote
            export_cmd = f"""
            ssh -o ConnectTimeout={self.ssh_timeout} -o StrictHostKeyChecking=no {ssh_user}@{remote_host} \\
            'python3 -c "
import sqlite3
import json
from pathlib import Path
db_path = Path({repr(remote_db_path)}).expanduser()
if not db_path.exists():
    print(json.dumps(dict(error=\"DB not found\")))
else:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # Export matches from last 7 days
    cur.execute(\"\"\"
        SELECT match_id, player1_id, player2_id, winner_id,
               player1_rating_before, player2_rating_before,
               player1_rating_after, player2_rating_after,
               board_type, num_players, game_length, timestamp, source
        FROM match_history
        WHERE timestamp > datetime(\"now\", \"-7 days\")
        ORDER BY timestamp DESC
        LIMIT 10000
    \"\"\")
    matches = [dict(row) for row in cur.fetchall()]
    conn.close()
    print(json.dumps(dict(matches=matches)))
"'
            """
            result = subprocess.run(
                export_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.ssh_timeout + 10,
            )

            if result.returncode != 0:
                return SyncResult(
                    remote_host=remote_host,
                    synced_at=synced_at,
                    matches_added=0,
                    matches_skipped=0,
                    matches_conflict=0,
                    participants_added=0,
                    error=f"SSH failed: {result.stderr[:200]}",
                )

            # Parse response
            try:
                data = json.loads(result.stdout.strip())
                if "error" in data:
                    return SyncResult(
                        remote_host=remote_host,
                        synced_at=synced_at,
                        matches_added=0,
                        matches_skipped=0,
                        matches_conflict=0,
                        participants_added=0,
                        error=data["error"],
                    )
                matches = data.get("matches", [])
            except json.JSONDecodeError as e:
                return SyncResult(
                    remote_host=remote_host,
                    synced_at=synced_at,
                    matches_added=0,
                    matches_skipped=0,
                    matches_conflict=0,
                    participants_added=0,
                    error=f"JSON parse error: {e}",
                )

            # Import matches to local DB
            result = self._import_matches(remote_host, synced_at, matches)
            self._emit_sync_metrics(result)
            return result

        except subprocess.TimeoutExpired:
            result = SyncResult(
                remote_host=remote_host,
                synced_at=synced_at,
                matches_added=0,
                matches_skipped=0,
                matches_conflict=0,
                participants_added=0,
                error="SSH timeout",
            )
            self._emit_sync_metrics(result)
            return result
        except Exception as e:
            result = SyncResult(
                remote_host=remote_host,
                synced_at=synced_at,
                matches_added=0,
                matches_skipped=0,
                matches_conflict=0,
                participants_added=0,
                error=str(e),
            )
            self._emit_sync_metrics(result)
            return result
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_export)
            except FileNotFoundError:
                pass

    def _emit_sync_metrics(self, result: SyncResult) -> None:
        """Emit Prometheus metrics for a sync result."""
        try:
            from app.metrics import record_elo_sync
            record_elo_sync(
                remote_host=result.remote_host,
                success=result.error is None,
                matches_added=result.matches_added,
                conflicts=result.matches_conflict,
            )
        except ImportError:
            pass

    def _import_matches(
        self,
        remote_host: str,
        synced_at: str,
        matches: List[Dict[str, Any]],
    ) -> SyncResult:
        """Import matches into local DB.

        Handles conflict resolution based on self.conflict_resolution strategy:
        - SKIP: Keep existing, count as unresolved conflict
        - LAST_WRITE_WINS: Accept match with more recent timestamp
        - FIRST_WRITE_WINS: Keep existing, count as resolved
        - RAISE: Raise ConflictError on first conflict
        """
        if not matches:
            return SyncResult(
                remote_host=remote_host,
                synced_at=synced_at,
                matches_added=0,
                matches_skipped=0,
                matches_conflict=0,
                matches_resolved=0,
                participants_added=0,
            )

        added = 0
        skipped = 0
        conflicts = 0
        resolved = 0
        participants_added = set()

        # Ensure local DB exists
        self.local_db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.local_db_path), timeout=30)
        try:
            cur = conn.cursor()

            # Create tables if needed
            cur.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    match_id TEXT PRIMARY KEY,
                    player1_id TEXT NOT NULL,
                    player2_id TEXT NOT NULL,
                    winner_id TEXT,
                    player1_rating_before REAL,
                    player2_rating_before REAL,
                    player1_rating_after REAL,
                    player2_rating_after REAL,
                    board_type TEXT,
                    num_players INTEGER,
                    game_length INTEGER,
                    timestamp TEXT,
                    source TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    rating REAL DEFAULT 1500.0,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    last_update TEXT,
                    board_type TEXT,
                    num_players INTEGER
                )
            """)

            # Get existing match IDs
            cur.execute("SELECT match_id FROM match_history")
            existing_ids = {row[0] for row in cur.fetchall()}

            for match in matches:
                match_id = match.get("match_id")
                if not match_id:
                    continue

                if match_id in existing_ids:
                    # Check for conflict (same ID, different data)
                    cur.execute(
                        "SELECT winner_id, timestamp FROM match_history WHERE match_id = ?",
                        (match_id,),
                    )
                    row = cur.fetchone()
                    existing_winner = row[0] if row else None
                    existing_timestamp = row[1] if row else None
                    incoming_winner = match.get("winner_id")
                    incoming_timestamp = match.get("timestamp")

                    if existing_winner != incoming_winner:
                        # Conflict detected - handle based on resolution strategy
                        if self.conflict_resolution == ConflictResolution.RAISE:
                            raise ValueError(
                                f"Match conflict: {match_id} has winner={existing_winner} "
                                f"locally but winner={incoming_winner} from {remote_host}"
                            )
                        elif self.conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
                            # Keep existing, mark as resolved
                            resolved += 1
                        elif self.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
                            # Compare timestamps, update if incoming is newer
                            if self._is_newer_timestamp(incoming_timestamp, existing_timestamp):
                                self._update_match(cur, match, remote_host)
                                resolved += 1
                            else:
                                # Existing is newer, keep it
                                resolved += 1
                        else:  # SKIP
                            conflicts += 1
                    else:
                        skipped += 1
                    continue

                # Insert new match
                cur.execute(
                    """
                    INSERT INTO match_history (
                        match_id, player1_id, player2_id, winner_id,
                        player1_rating_before, player2_rating_before,
                        player1_rating_after, player2_rating_after,
                        board_type, num_players, game_length, timestamp, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        match.get("player1_id"),
                        match.get("player2_id"),
                        match.get("winner_id"),
                        match.get("player1_rating_before"),
                        match.get("player2_rating_before"),
                        match.get("player1_rating_after"),
                        match.get("player2_rating_after"),
                        match.get("board_type"),
                        match.get("num_players"),
                        match.get("game_length"),
                        match.get("timestamp"),
                        match.get("source", f"sync:{remote_host}"),
                    ),
                )
                added += 1

                # Track new participants
                for pid in [match.get("player1_id"), match.get("player2_id")]:
                    if pid:
                        participants_added.add(pid)

            # Ensure participants exist
            for pid in participants_added:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO participants (participant_id, rating)
                    VALUES (?, 1500.0)
                    """,
                    (pid,),
                )

            conn.commit()

        finally:
            conn.close()

        return SyncResult(
            remote_host=remote_host,
            synced_at=synced_at,
            matches_added=added,
            matches_skipped=skipped,
            matches_conflict=conflicts,
            matches_resolved=resolved,
            participants_added=len(participants_added),
        )

    def _is_newer_timestamp(
        self,
        incoming: Optional[str],
        existing: Optional[str],
    ) -> bool:
        """Compare timestamps to determine if incoming is newer.

        Returns True if incoming timestamp is more recent than existing.
        If either is None or unparseable, defaults to accepting incoming.
        """
        if not incoming:
            return False
        if not existing:
            return True

        try:
            # Try ISO format first
            incoming_dt = datetime.fromisoformat(incoming.replace("Z", "+00:00"))
            existing_dt = datetime.fromisoformat(existing.replace("Z", "+00:00"))
            return incoming_dt > existing_dt
        except (ValueError, AttributeError):
            # Fall back to string comparison (works for ISO timestamps)
            return incoming > existing

    def _update_match(
        self,
        cursor: sqlite3.Cursor,
        match: Dict[str, Any],
        remote_host: str,
    ) -> None:
        """Update an existing match with new data (for conflict resolution)."""
        cursor.execute(
            """
            UPDATE match_history SET
                player1_id = ?,
                player2_id = ?,
                winner_id = ?,
                player1_rating_before = ?,
                player2_rating_before = ?,
                player1_rating_after = ?,
                player2_rating_after = ?,
                board_type = ?,
                num_players = ?,
                game_length = ?,
                timestamp = ?,
                source = ?
            WHERE match_id = ?
            """,
            (
                match.get("player1_id"),
                match.get("player2_id"),
                match.get("winner_id"),
                match.get("player1_rating_before"),
                match.get("player2_rating_before"),
                match.get("player1_rating_after"),
                match.get("player2_rating_after"),
                match.get("board_type"),
                match.get("num_players"),
                match.get("game_length"),
                match.get("timestamp"),
                match.get("source", f"sync:{remote_host}"),
                match.get("match_id"),
            ),
        )

    def reconcile_all(
        self,
        hosts: Optional[List[str]] = None,
    ) -> ReconciliationReport:
        """Run full reconciliation across all known P2P nodes.

        Args:
            hosts: List of hosts to sync from. If None, reads from config.

        Returns:
            ReconciliationReport with full details
        """
        started_at = datetime.now(timezone.utc).isoformat()

        if hosts is None:
            hosts = self._load_p2p_hosts()

        nodes_synced = []
        nodes_failed = []
        sync_results = []
        total_added = 0
        total_skipped = 0
        total_conflicts = 0
        total_resolved = 0

        for host in hosts:
            result = self.sync_from_remote(host)
            sync_results.append(result)

            if result.error:
                nodes_failed.append(host)
            else:
                nodes_synced.append(host)
                total_added += result.matches_added
                total_skipped += result.matches_skipped
                total_conflicts += result.matches_conflict
                total_resolved += result.matches_resolved

        # Check for drift after sync
        drift = self.check_drift()

        return ReconciliationReport(
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            nodes_synced=nodes_synced,
            nodes_failed=nodes_failed,
            total_matches_added=total_added,
            total_matches_skipped=total_skipped,
            total_conflicts=total_conflicts,
            total_resolved=total_resolved,
            drift_detected=drift.is_significant,
            max_drift=drift.max_rating_diff,
            sync_results=sync_results,
        )

    def _get_ratings(
        self,
        db_path: Path,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> Dict[str, float]:
        """Get ratings from a database.

        Handles multiple schema variations:
        1. elo_ratings table (production schema): participant_id, board_type, num_players, rating
        2. participants table with rating column (test schema): participant_id, rating, board_type, num_players
        """
        if not db_path.exists():
            return {}

        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            cur = conn.cursor()

            # Check which table exists and its schema
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}

            # Prefer elo_ratings table (production schema)
            if "elo_ratings" in tables:
                query = "SELECT participant_id, rating FROM elo_ratings WHERE 1=1"
                params = []
                if board_type:
                    query += " AND board_type = ?"
                    params.append(board_type)
                if num_players:
                    query += " AND num_players = ?"
                    params.append(num_players)

                cur.execute(query, params)
                return {row[0]: row[1] for row in cur.fetchall() if row[1] is not None}

            elif "participants" in tables:
                # Check if participants table has a rating column
                cur.execute("PRAGMA table_info(participants)")
                columns = {row[1] for row in cur.fetchall()}

                if "rating" in columns:
                    query = "SELECT participant_id, rating FROM participants WHERE rating IS NOT NULL"
                    params = []
                    if board_type and "board_type" in columns:
                        query += " AND board_type = ?"
                        params.append(board_type)
                    if num_players and "num_players" in columns:
                        query += " AND num_players = ?"
                        params.append(num_players)

                    cur.execute(query, params)
                    return {row[0]: row[1] for row in cur.fetchall()}

            return {}

        finally:
            conn.close()

    def _count_participants(
        self,
        db_path: Path,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> int:
        """Count participants in a database."""
        ratings = self._get_ratings(db_path, board_type, num_players)
        return len(ratings)

    def _load_p2p_hosts(self) -> List[str]:
        """Load P2P hosts from config."""
        if not self.remote_hosts_config.exists():
            return []

        try:
            import yaml
            with open(self.remote_hosts_config) as f:
                config = yaml.safe_load(f)

            hosts = []

            # GH200 and other standard hosts
            for name, info in config.get("standard_hosts", {}).items():
                if "gh200" in name.lower() or info.get("role", "") == "selfplay":
                    host = info.get("ssh_host")
                    if host:
                        hosts.append(host)

            return hosts

        except Exception:
            return []


def sync_elo_from_remote(
    remote_host: str,
    remote_db_path: str = "~/ringrift/ai-service/data/unified_elo.db",
) -> SyncResult:
    """Convenience function to sync Elo from a remote host."""
    reconciler = EloReconciler()
    return reconciler.sync_from_remote(remote_host, remote_db_path)


def check_elo_drift(
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
) -> EloDrift:
    """Convenience function to check local Elo drift."""
    reconciler = EloReconciler()
    return reconciler.check_drift(board_type=board_type, num_players=num_players)


if __name__ == "__main__":
    # Demo/CLI
    import argparse

    parser = argparse.ArgumentParser(description="Elo Reconciliation Tool")
    parser.add_argument("--check-drift", action="store_true", help="Check Elo drift")
    parser.add_argument("--sync-from", type=str, help="Sync from remote host")
    parser.add_argument("--reconcile-all", action="store_true", help="Full reconciliation")

    args = parser.parse_args()

    reconciler = EloReconciler()

    if args.check_drift:
        drift = reconciler.check_drift()
        print(f"Participants: {drift.participants_in_source}")
        print(f"Max drift: {drift.max_rating_diff:.1f}")
        print(f"Avg drift: {drift.avg_rating_diff:.1f}")
        print(f"Significant: {drift.is_significant}")

    elif args.sync_from:
        result = reconciler.sync_from_remote(args.sync_from)
        print(f"Added: {result.matches_added}")
        print(f"Skipped: {result.matches_skipped}")
        print(f"Conflicts: {result.matches_conflict}")
        if result.error:
            print(f"Error: {result.error}")

    elif args.reconcile_all:
        report = reconciler.reconcile_all()
        print(report.summary())

    else:
        parser.print_help()
